import ast
import shlex
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

VARIANT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VARIANT_ROOT))

from options import parser
from temperature_schedule import (
    epoch_temperatures,
    temperature_config,
    validate_checkpoint_temperature,
    validate_temperature_config,
)
from topk_pooling import (
    smooth_temporal_scores, temporal_segment_pool, topk_pool, video_topk_size,
)


def scheduled_args(*extra):
    return parser.parse_args([
        '--topk-pooling', 'soft', '--a-topk-temperature-schedule', 'linear',
        *extra,
    ])


class TemperatureScheduleTests(unittest.TestCase):
    def test_documented_kaggle_arguments_parse_and_enable_schedule(self):
        design = (VARIANT_ROOT / 'SOFT_TOPK_TEMPERATURE_DESIGN.md').read_text(
            encoding='utf-8')
        command = design.split('python experiments/topk_variants/train_ucf.py', 1)[1]
        command = command.split('```', 1)[0].replace('\\\n', ' ')
        args = parser.parse_args(shlex.split(command))
        validate_temperature_config(args)
        self.assertEqual((args.seed, args.batch_size, args.max_epoch), (234, 64, 10))
        self.assertEqual(epoch_temperatures(args, 1), (1.0, 2.0))
        self.assertEqual(epoch_temperatures(args, 4), (1.0, 1.0))

    def test_defaults_preserve_mean_and_fixed_temperature(self):
        args = parser.parse_args([])
        self.assertEqual(args.topk_pooling, 'mean')
        self.assertEqual(args.a_topk_temperature_schedule, 'constant')
        self.assertEqual(epoch_temperatures(args, 3), (1.0, 1.0))

    def test_fixed_soft_respects_existing_temperatures(self):
        args = parser.parse_args([
            '--topk-pooling', 'soft', '--a-topk-temperature', '0.7',
            '--c-topk-temperature', '1.3',
        ])
        for epoch in (1, 4, 10):
            self.assertEqual(epoch_temperatures(args, epoch), (1.3, 0.7))

    def test_linear_schedule_and_final_plateau(self):
        args = scheduled_args()
        for epoch, expected in enumerate([2.0, 5 / 3, 4 / 3, 1.0, 1.0], 1):
            c, a = epoch_temperatures(args, epoch)
            self.assertEqual(c, 1.0)
            self.assertAlmostEqual(a, expected)

    def test_custom_endpoints_and_delayed_start(self):
        args = scheduled_args(
            '--a-topk-temperature-start', '3', '--a-topk-temperature', '0.5',
            '--a-topk-temperature-start-epoch', '3',
            '--a-topk-temperature-end-epoch', '5',
        )
        self.assertEqual(epoch_temperatures(args, 1), (1.0, 3.0))
        self.assertEqual(epoch_temperatures(args, 3), (1.0, 3.0))
        self.assertEqual(epoch_temperatures(args, 4), (1.0, 1.75))
        self.assertEqual(epoch_temperatures(args, 9), (1.0, 0.5))

    def test_resume_uses_absolute_epoch(self):
        args = scheduled_args()
        checkpoint = {'topk_temperature_config': temperature_config(args)}
        validate_checkpoint_temperature(args, checkpoint)
        self.assertAlmostEqual(epoch_temperatures(args, 3)[1], 4 / 3)
        self.assertEqual(epoch_temperatures(args, 7)[1], 1.0)

    def test_changed_or_missing_resume_schedule_is_rejected(self):
        args = scheduled_args()
        checkpoint = {'topk_temperature_config': temperature_config(args)}
        args.a_topk_temperature_start = 4.0
        for saved in (checkpoint, {}):
            with self.subTest(saved=saved), self.assertRaises(ValueError):
                validate_checkpoint_temperature(args, saved)

    def test_legacy_constant_resume_is_accepted(self):
        validate_checkpoint_temperature(parser.parse_args([]), {})

    def test_inactive_flags_do_not_change_checkpoint_identity(self):
        args = parser.parse_args([])
        saved = temperature_config(args)
        args.a_topk_temperature_start = 5
        validate_checkpoint_temperature(args, {'topk_temperature_config': saved})

    def test_invalid_temperatures_are_rejected(self):
        for field in ('a_topk_temperature', 'c_topk_temperature',
                      'a_topk_temperature_start'):
            for value in (0.0, -1.0, float('nan'), float('inf')):
                args = scheduled_args()
                setattr(args, field, value)
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        validate_temperature_config(args)

    def test_invalid_epochs_are_rejected(self):
        for start, end in ((0, 4), (4, 4), (5, 4)):
            args = scheduled_args()
            args.a_topk_temperature_start_epoch = start
            args.a_topk_temperature_end_epoch = end
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                validate_temperature_config(args)
        with self.assertRaises(ValueError):
            epoch_temperatures(scheduled_args(), 0)

    def test_non_soft_and_conflicting_selectors_are_rejected(self):
        for pooling in ('mean', 'multi_k'):
            with self.subTest(pooling=pooling), self.assertRaises(ValueError):
                validate_temperature_config(scheduled_args('--topk-pooling', pooling))
        for flag in ('--dual-k', '--adaptive-instance-selection',
                     '--temporal-segment-topk', '--c-temporal-segment-topk'):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                validate_temperature_config(scheduled_args(flag))


class TemperatureLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the actual loss definitions without importing CLIP or datasets.
        tree = ast.parse((VARIANT_ROOT / 'train_ucf.py').read_text(encoding='utf-8'))
        names = {'resolve_pool_k', 'CLASM', 'CLAS2'}
        module = ast.Module(body=[n for n in tree.body
                                 if isinstance(n, ast.FunctionDef) and n.name in names],
                            type_ignores=[])
        cls.losses = dict(torch=torch, F=F, topk_pool=topk_pool,
                          video_topk_size=video_topk_size,
                          smooth_temporal_scores=smooth_temporal_scores,
                          temporal_segment_pool=temporal_segment_pool)
        exec(compile(module, str(VARIANT_ROOT / 'train_ucf.py'), 'exec'), cls.losses)

    def test_schedule_changes_a_loss_not_c_loss_and_keeps_gradients(self):
        args = scheduled_args()
        labels = torch.tensor([[0.0, 1.0]])
        lengths = torch.tensor([16])  # K=2; padding must never be selected.
        base_a = torch.zeros(1, 18, 2)
        base_a[0, 0] = torch.tensor([1.0, 1.0])
        base_a[0, 1] = torch.tensor([2.0, 4.0])
        base_a[0, 16:] = 100.0
        base_c = torch.linspace(-2.0, 2.0, 18).reshape(1, 18, 1)
        losses_a, losses_c, gradients_a = [], [], []
        for epoch in (1, 4):
            c_temp, a_temp = epoch_temperatures(args, epoch)
            a_logits = base_a.clone().requires_grad_()
            c_logits = base_c.clone().requires_grad_()
            a_loss = self.losses['CLASM'](
                a_logits, labels, lengths, 'cpu', a_temp, 'soft')
            c_loss = self.losses['CLAS2'](
                c_logits, labels, lengths, 'cpu', c_temp, 'soft')
            (a_loss + c_loss).backward()
            losses_a.append(a_loss.detach())
            losses_c.append(c_loss.detach())
            gradients_a.append(a_logits.grad)
            for grad in (a_logits.grad, c_logits.grad):
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad[:, :16].abs().sum().item(), 0)
                self.assertEqual(grad[:, 16:].abs().sum().item(), 0)
        self.assertFalse(torch.allclose(*losses_a))
        self.assertFalse(torch.allclose(*gradients_a))
        torch.testing.assert_close(*losses_c, rtol=0, atol=0)

    def test_final_schedule_matches_fixed_loss_and_gradient_exactly(self):
        args = scheduled_args()
        labels = torch.tensor([[0.0, 1.0]])
        lengths = torch.tensor([16])
        base = torch.arange(32, dtype=torch.float32).reshape(1, 16, 2) / 10
        results = []
        for temperature in (epoch_temperatures(args, 4)[1], 1.0):
            logits = base.clone().requires_grad_()
            loss = self.losses['CLASM'](logits, labels, lengths, 'cpu', temperature)
            loss.backward()
            results.append((loss.detach(), logits.grad))
        for scheduled, fixed in zip(*results):
            torch.testing.assert_close(scheduled, fixed, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
