import ast
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from options import parser
from shift_augmentation import TailShiftAugmentation, tail_shift_batch
import test_snippet_pca_loss as snippet_tests


def reference_args(*extra):
    return parser.parse_args([
        '--temporal-segment-topk', '--temporal-segment-start-epoch', '1',
        '--temporal-smoothing-kernel', '3', '--temporal-smoothness-branch', 'a',
        '--temporal-smoothness-start-epoch', '1',
        '--a-temporal-smoothness-weight', '0.05', *extra,
    ])


class TailGeometryTests(unittest.TestCase):
    def test_fixed_tail_matches_source_negative_offset_without_wrap(self):
        x = torch.arange(1., 7.).reshape(1, 6, 1)
        moved, lengths = tail_shift_batch(x, torch.tensor([6]), 2)
        torch.testing.assert_close(moved.flatten(), torch.tensor([0., 0., 1., 2., 3., 4.]))
        self.assertEqual(lengths.tolist(), [6])
        self.assertEqual(x.flatten().tolist(), [1., 2., 3., 4., 5., 6.])

    def test_short_video_keeps_all_content_and_zeros_original_padding(self):
        x = torch.tensor([1., 2., 3., float('nan'), 99., 99.]).reshape(1, 6, 1)
        moved, lengths = tail_shift_batch(x, torch.tensor([3]), 2)
        torch.testing.assert_close(moved.flatten(), torch.tensor([0., 0., 1., 2., 3., 0.]))
        self.assertEqual(lengths.tolist(), [5])

    def test_26_steps_preserve_short_video_and_clip_long_video(self):
        x = torch.arange(1., 513.).reshape(2, 256, 1)
        moved, lengths = tail_shift_batch(x, torch.tensor([100, 256]))
        self.assertEqual(lengths.tolist(), [126, 256])
        self.assertEqual(torch.count_nonzero(moved[:, :26]).item(), 0)
        torch.testing.assert_close(moved[0, 26:126], x[0, :100])
        self.assertEqual(torch.count_nonzero(moved[0, 126:]).item(), 0)
        torch.testing.assert_close(moved[1, 26:], x[1, :230])

    def test_gradient_only_reaches_retained_real_content(self):
        x = torch.ones(2, 6, 2, requires_grad=True)
        moved, _ = tail_shift_batch(x, torch.tensor([3, 6]), 2)
        moved.sum().backward()
        torch.testing.assert_close(x.grad[0, :3], torch.ones(3, 2))
        self.assertEqual(x.grad[0, 3:].sum().item(), 0)
        self.assertEqual(x.grad[1, 4:].sum().item(), 0)

    def test_invalid_offset_shape_and_length_rejected(self):
        x = torch.ones(2, 32, 4)
        for offset in (0, -26, 32, 2.5, True):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                tail_shift_batch(x, torch.tensor([5, 32]), offset)
        for lengths in (torch.tensor([0, 32]), torch.tensor([5, 33]),
                        torch.tensor([5., 32.]), torch.tensor([True, True]),
                        torch.tensor([5])):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                tail_shift_batch(x, lengths, 26)

    def test_no_random_sampling(self):
        before = torch.get_rng_state().clone()
        x = torch.ones(2, 32, 4)
        tail_shift_batch(x, torch.tensor([5, 32]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_small_integer_dtype_does_not_overflow_shifted_length(self):
        _, lengths = tail_shift_batch(torch.ones(1, 256, 4),
                                      torch.tensor([250], dtype=torch.uint8))
        self.assertEqual(lengths.tolist(), [256])


class TailConfigurationTests(unittest.TestCase):
    def test_disabled_defaults_and_legacy_resume(self):
        augmentation = TailShiftAugmentation(parser.parse_args([]))
        self.assertEqual(augmentation.config, {'enabled': False})
        augmentation.validate_checkpoint({})

    def test_only_temporal_a_and_no_auxiliary_losses(self):
        with self.assertRaises(ValueError):
            TailShiftAugmentation(parser.parse_args(['--tail-shift-augmentation']))
        for extra in (['--c-pca-loss'], ['--snippet-ranking-loss'], ['--dual-k'],
                      ['--adaptive-instance-selection'], ['--c-ranking-loss'],
                      ['--c-temporal-smoothing-kernel', '3'],
                      ['--a-temporal-smoothness-weight', '0.01'],
                      ['--tail-shift-offset', '256']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                TailShiftAugmentation(reference_args('--tail-shift-augmentation', *extra))

    def test_resume_cannot_add_remove_or_change_shift(self):
        active = TailShiftAugmentation(reference_args('--tail-shift-augmentation'))
        saved = {'tail_shift_config': active.config}
        active.validate_checkpoint(saved)
        for candidate in (TailShiftAugmentation(reference_args()),
                          TailShiftAugmentation(reference_args(
                              '--tail-shift-augmentation', '--tail-shift-offset', '20'))):
            with self.assertRaises(ValueError):
                candidate.validate_checkpoint(saved)
        with self.assertRaises(ValueError):
            active.validate_checkpoint({})

    def test_kaggle_commands_use_paired_reference_and_unique_outputs(self):
        commands = json.loads((ROOT / 'tests/fixtures/training_commands.json').read_text(
            encoding='utf-8'))['tail_shift']
        paths = set()
        for name, command in commands.items():
            args = parser.parse_args(command)
            augmentation = TailShiftAugmentation(args)
            self.assertEqual(augmentation.enabled, name == 'RA_tail26')
            self.assertEqual(augmentation.offset, 26)
            self.assertFalse(args.amp or args.skip_eval or args.use_checkpoint)
            self.assertEqual((args.seed, args.batch_size, args.max_epoch), (234, 64, 10))
            for path in (args.model_path, args.checkpoint_path, args.log_path):
                self.assertNotIn(path, paths)
                paths.add(path)


class GridDataset(Dataset):
    def __init__(self, normal, count=2):
        self.normal, self.count = normal, count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index + (0 if self.normal else 1000))
        x = torch.randn(32, 4, generator=generator)
        length = 32 if index % 2 == 0 else 5
        x[length:] = 0
        return x, 'Normal' if self.normal else 'Abnormal', length


class RecordingModel(snippet_tests.ToyModel):
    def __init__(self):
        super().__init__()
        self.forward_batches = []

    def forward(self, visual, padding_mask, text, lengths, **kwargs):
        self.forward_batches.append((visual.shape[0], lengths.detach().tolist()))
        return super().forward(visual, padding_mask, text, lengths, **kwargs)


class TailTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        tree = ast.parse((ROOT / 'train_ucf.py').read_text(encoding='utf-8'))
        cls.env = dict(vars(snippet_tests),
                       get_prompt_text=lambda labels: list(labels.values()),
                       get_batch_label=lambda labels, *_: torch.tensor([
                           [1., 0.] if y == 'Normal' else [0., 1.] for y in labels]))
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), 'train_ucf.py', 'exec'), cls.env)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def args(self, root, enabled=True, skip_eval=True):
        for name in ('gt', 'segments', 'labels'):
            np.save(root / (name + '.npy'), np.zeros(1))
        flags = ['--tail-shift-augmentation'] if enabled else []
        if skip_eval:
            flags += ['--skip-eval']
        return reference_args(
            *flags, '--visual-length', '32', '--batch-size', '2',
            '--max-epoch', '1', '--log-interval', '1',
            '--gt-path', str(root / 'gt.npy'),
            '--gt-segment-path', str(root / 'segments.npy'),
            '--gt-label-path', str(root / 'labels.npy'),
            '--model-path', str(root / 'model.pth'),
            '--checkpoint-path', str(root / 'checkpoint.pth'),
            '--log-path', str(root / 'train.log'))

    def run_train(self, model, normal, anomaly, args):
        with contextlib.redirect_stdout(io.StringIO()):
            self.env['train'](model, normal, anomaly, [], args,
                              {'Normal': 'normal', 'Abnormal': 'abnormal'}, 'cpu')

    def test_optimizer_matches_two_mil_views_and_original_tv(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = self.args(root, enabled)
                torch.manual_seed(234)
                model = RecordingModel()
                expected = copy.deepcopy(model)
                normal = DataLoader(GridDataset(True), batch_size=2)
                anomaly = DataLoader(GridDataset(False), batch_size=2)
                n, _, nl = next(iter(normal)); a, _, al = next(iter(anomaly))
                x, lengths = torch.cat([n, a]), torch.cat([nl, al])
                labels = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
                text, c, alignment = expected(x, None, ['normal', 'abnormal'], lengths)
                c_loss = self.env['CLAS2'](c, labels, lengths, 'cpu')
                a_loss = self.env['CLASM'](alignment, labels, lengths, 'cpu',
                                         temporal_segment=True, temporal_smoothing_kernel=3)
                if enabled:
                    # Independent reference construction, not the augmentation helper.
                    moved = torch.cat([torch.zeros(4, 26, 4), x[:, :6]], dim=1)
                    moved_lengths = (lengths + 26).clamp(max=32)
                    _, moved_c, moved_a = expected(moved, None, ['normal', 'abnormal'], moved_lengths)
                    c_loss = .5 * (c_loss + self.env['CLAS2'](moved_c, labels, moved_lengths, 'cpu'))
                    a_loss = .5 * (a_loss + self.env['CLASM'](
                        moved_a, labels, moved_lengths, 'cpu',
                        temporal_segment=True, temporal_smoothing_kernel=3))
                normalized = torch.nn.functional.normalize(text, dim=-1)
                text_loss = (normalized[0] @ normalized[1]).abs() * .1
                tv = self.env['a_branch_temporal_smoothness'](alignment, lengths)
                loss = c_loss + a_loss + text_loss + .05 * tv
                optimizer = torch.optim.AdamW(expected.parameters(), lr=args.lr)
                loss.backward(); optimizer.step()
                self.run_train(model, normal, anomaly, args)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, expected.state_dict()[name], rtol=1e-5, atol=1e-7)
                self.assertEqual(model.forward_batches[0][0], 8 if enabled else 4)
                log = (root / 'train.log').read_text(encoding='utf-8')
                self.assertEqual('loss1_tail=' in log, enabled)
                if enabled:
                    self.assertIn('shift_consistency_active=False', log)
                    self.assertIn(f'loss_smooth_a={tv.item():.6f}', log)
                saved = torch.load(args.checkpoint_path, weights_only=False)
                self.assertEqual(saved['tail_shift_config'], TailShiftAugmentation(args).config)

    def test_evaluation_uses_original_single_view_and_saves_shift_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root, skip_eval=False)
            args.batch_size = 64
            normal = DataLoader(GridDataset(True, 704), batch_size=64)
            anomaly = DataLoader(GridDataset(False, 704), batch_size=64)
            model = RecordingModel()

            def raw_evaluation(model, *unused, **kwargs):
                model.eval()
                with torch.no_grad():
                    model(torch.ones(1, 32, 4), None, ['normal', 'abnormal'], torch.tensor([32]))
                model.train()
                return .6, .3

            old_test = self.env.get('test')
            self.env['test'] = raw_evaluation
            try:
                self.run_train(model, normal, anomaly, args)
            finally:
                if old_test is None:
                    self.env.pop('test', None)
                else:
                    self.env['test'] = old_test
            self.assertEqual(model.forward_batches[-1], (1, [32]))
            self.assertTrue(all(batch == 256 for batch, _ in model.forward_batches[:-1]))
            saved = torch.load(args.checkpoint_path, weights_only=False)
            self.assertEqual(saved['tail_shift_config']['consistency_weight'], 0.0)

    def test_trainer_rejects_changed_resume_and_overwriting_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root)
            normal = DataLoader(GridDataset(True), batch_size=2)
            anomaly = DataLoader(GridDataset(False), batch_size=2)
            self.run_train(RecordingModel(), normal, anomaly, args)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                self.run_train(RecordingModel(), normal, anomaly, args)
            args.use_checkpoint = True
            args.tail_shift_offset = 20
            with self.assertRaisesRegex(ValueError, 'tail shift configuration'):
                self.run_train(RecordingModel(), normal, anomaly, args)

    def test_matching_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_root, resumed_root = root / 'full', root / 'resumed'
            full_root.mkdir(); resumed_root.mkdir()
            normal = DataLoader(GridDataset(True), batch_size=2)
            anomaly = DataLoader(GridDataset(False), batch_size=2)
            full_args = self.args(full_root)
            full_args.max_epoch = 2
            torch.manual_seed(234)
            full_model = RecordingModel()
            self.run_train(full_model, normal, anomaly, full_args)
            args = self.args(resumed_root)
            torch.manual_seed(234)
            self.run_train(RecordingModel(), normal, anomaly, args)
            args.use_checkpoint, args.max_epoch = True, 2
            resumed_model = RecordingModel()
            self.run_train(resumed_model, normal, anomaly, args)
            for name, value in full_model.state_dict().items():
                torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)
            saved = torch.load(args.checkpoint_path, weights_only=False)
            self.assertEqual(saved['next_epoch'], 2)


if __name__ == '__main__':
    unittest.main()
