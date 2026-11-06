"""Source formula/gradient parity and actual Temporal-A trainer integration on CPU."""
import ast
import contextlib
import copy
import io
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from options import parser
from finetune_margin_adapter import FinetuneMarginLoss
from finetune_margin_loss import ranking_margin_loss, abnormal_background_suppression_loss
import test_snippet_pca_loss as fixture


REFERENCE = ['--topk-pooling', 'mean', '--temporal-segment-topk',
             '--temporal-segment-start-epoch', '1', '--temporal-smoothing-kernel', '3',
             '--temporal-smoothness-branch', 'a', '--temporal-smoothness-start-epoch', '1',
             '--a-temporal-smoothness-weight', '.05']


def arguments(mode='ranking', *extra):
    return parser.parse_args([*REFERENCE, '--finetune-margin', mode,
                              '--lambda-abnormal-bg', '.05', *extra])


def inputs(c_values=None, a_values=None):
    c_values = c_values if c_values is not None else [[.3, .2, .1, .05], [.5, .4, .3, .2]]
    p = torch.tensor(c_values)
    q = torch.tensor(a_values) if a_values is not None else p
    c = p.logit().unsqueeze(-1).requires_grad_()
    a = torch.stack((torch.zeros_like(q), q.logit()), -1).requires_grad_()
    labels = torch.tensor([[1., 0.], [0., 1.]])
    return c, a, labels, torch.tensor([4, 4])


class FormulaTests(unittest.TestCase):
    def test_ranking_is_hinge_of_group_means_not_mean_pairwise_hinges(self):
        scores = torch.tensor([.6, .01, .7, .5])
        c = scores.logit().reshape(4, 1, 1).requires_grad_()
        a = torch.zeros(4, 1, 2, requires_grad=True)
        labels = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        loss = ranking_margin_loss(c, a, labels, [1]*4, .15, .1, 'classifier', 'cpu')
        self.assertEqual(loss.item(), 0.)
        pairwise = torch.relu(.1 + scores[:2][None, :] - scores[2:, None]).mean()
        self.assertAlmostEqual(pairwise.item(), .05, places=6)

    def test_background_uses_boundaries_only_on_abnormal_videos(self):
        c, a, labels, lengths = inputs()
        loss = abnormal_background_suppression_loss(c, a, labels, lengths, .5, .2, 'classifier', 'cpu')
        self.assertAlmostEqual(loss.item(), .1, places=6)  # .3 - .4 + .2
        loss.backward()
        self.assertEqual(float(c.grad[0].abs().sum()), 0.)
        self.assertGreater(float(c.grad[1].abs().sum()), 0.)
        self.assertIsNone(a.grad)

    def test_both_branches_average_losses_not_probabilities(self):
        c, a, labels, lengths = inputs(a_values=[[.3, .2, .1, .05], [.8, .7, .6, .5]])
        for function in (ranking_margin_loss, abnormal_background_suppression_loss):
            values = [function(c, a, labels, lengths, .5, .4, branch, 'cpu')
                      for branch in ('classifier', 'alignment', 'both')]
            torch.testing.assert_close(values[2], (values[0] + values[1]) / 2)
            self.assertFalse(torch.equal(values[0], values[1]))

    def test_imported_functions_match_actual_finetune_values_and_gradients(self):
        source = ROOT.parents[2] / 'Finetune-VadCLIP/VadCLIP/src/ucf_train.py'
        if not source.is_file():
            self.skipTest('Original Finetune checkout is optional; analytic tests remain portable')
        names = ('get_anomaly_score_branches', 'ranking_margin_loss', 'abnormal_background_suppression_loss')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        env = {'torch': torch, 'F': F}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), env)
        imported = ast.parse((ROOT / 'finetune_margin_loss.py').read_text(encoding='utf-8'))
        for definition in definitions:
            actual = next(n for n in imported.body if isinstance(n, ast.FunctionDef) and n.name == definition.name)
            self.assertEqual(ast.dump(definition), ast.dump(actual))
        for mode, name in [('ranking', 'ranking_margin_loss'), ('background', 'abnormal_background_suppression_loss')]:
            for branch in ('classifier', 'alignment', 'both'):
                for ratio in (.15, .5):
                    with self.subTest(mode=mode, branch=branch, ratio=ratio):
                        torch.manual_seed(42)
                        c = torch.randn(4, 17, 1, requires_grad=True)
                        a = torch.randn(4, 17, 3, requires_grad=True)
                        expected_c, expected_a = [v.detach().clone().requires_grad_() for v in (c, a)]
                        labels = F.one_hot(torch.tensor([0, 1, 0, 2]), 3).float()
                        lengths = torch.tensor([17, 12, 8, 6])
                        args = arguments(mode, '--regularize-score-branch', branch,
                                         '--regularize-top-k-ratio', str(ratio),
                                         '--ranking-margin', '.7', '--abnormal-bg-margin', '.7')
                        weighted, _ = FinetuneMarginLoss(args)(c, a, labels, lengths, 2)
                        expected = env[name](expected_c, expected_a, labels, lengths, ratio, .7, branch, 'cpu')
                        torch.testing.assert_close(weighted, .05 * expected, rtol=0, atol=0)
                        weighted.backward()
                        (.05 * expected).backward()
                        for actual, reference in ((c.grad, expected_c.grad), (a.grad, expected_a.grad)):
                            if reference is None:
                                self.assertIsNone(actual)
                            else:
                                torch.testing.assert_close(actual, reference, rtol=0, atol=0)


class AdapterTests(unittest.TestCase):
    def test_warmup_counts_completed_initial_epochs_and_reports_weighted_loss(self):
        module = FinetuneMarginLoss(arguments('ranking', '--ranking-warmup-epochs', '2', '--ranking-margin', '.5'))
        for epoch in (1, 2):
            weighted, stats = module(*inputs(), epoch)
            self.assertEqual(weighted.item(), 0.)
            self.assertEqual(stats['finetune_margin_active'], 0.)
        weighted, stats = module(*inputs(), 3)
        self.assertGreater(weighted.item(), 0.)
        self.assertAlmostEqual(weighted.item(), .05 * stats['loss_finetune_margin'], places=7)
        self.assertEqual(stats['finetune_margin_active'], 1.)
        self.assertEqual(stats['finetune_margin_weight'], .05)

    def test_padding_nan_empty_sequences_and_missing_classes_have_safe_gradients(self):
        for mode in ('ranking', 'background'):
            for branch in ('classifier', 'alignment', 'both'):
                c, a, labels, _ = inputs()
                c = torch.cat((c, c[:1].detach())).detach().requires_grad_()
                a = torch.cat((a, a[:1].detach())).detach().requires_grad_()
                with torch.no_grad():
                    c[1, 2:] = a[1, 2:] = float('nan')
                    c[2] = a[2] = float('nan')
                labels = torch.cat((labels, labels[:1]))
                lengths = torch.tensor([4, 2, 0])
                module = FinetuneMarginLoss(arguments(mode, '--regularize-score-branch', branch))
                actual, _ = module(c, a, labels, lengths, 2)
                clean, _ = module(c.nan_to_num(), a.nan_to_num(), labels, lengths, 2)
                torch.testing.assert_close(actual, clean)
                actual.backward()
                for value in (c, a):
                    if value.grad is not None:
                        self.assertTrue(torch.isfinite(value.grad).all())
                        self.assertEqual(float(value.grad[1, 2:].abs().sum()), 0.)
                        self.assertEqual(float(value.grad[2].abs().sum()), 0.)
        for mode in ('ranking', 'background'):
            c, a, labels, lengths = inputs()
            labels[:] = labels[0].clone()
            zero, _ = FinetuneMarginLoss(arguments(mode))(c, a, labels, lengths, 2)
            self.assertEqual(zero.item(), 0.)
            zero.backward()
            self.assertTrue(torch.isfinite(c.grad).all())

    def test_config_is_opt_in_and_requires_the_exact_temporal_a_reference(self):
        baseline = FinetuneMarginLoss(parser.parse_args([]))
        self.assertFalse(baseline.enabled)
        self.assertEqual(baseline.checkpoint_fields(), {})
        baseline.validate_checkpoint({})
        for flags in (['--topk-pooling', 'soft'], ['--c-temporal-segment-topk'],
                      ['--temporal-smoothing-kernel', '1'], ['--a-temporal-smoothness-weight', '.01'],
                      ['--snippet-ranking-loss'], ['--c-ranking-loss'], ['--pca-loss'],
                      ['--tail-shift-augmentation'], ['--dual-k'], ['--adaptive-instance-selection'],
                      ['--regularize-top-k-ratio', 'nan'], ['--ranking-margin', 'inf'],
                      ['--ranking-warmup-epochs', '-1'], ['--lambda-ranking-margin', '0']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                FinetuneMarginLoss(arguments('ranking', *flags))
        with self.assertRaises(ValueError):
            FinetuneMarginLoss(arguments('background', '--regularize-top-k-ratio', '.6'))

    def test_checkpoint_guard_distinguishes_modes_and_each_active_parameter(self):
        original = FinetuneMarginLoss(arguments())
        saved = original.checkpoint_fields()
        original.validate_checkpoint(saved)
        for flags in (['--regularize-score-branch', 'classifier'], ['--regularize-top-k-ratio', '.2'],
                      ['--ranking-margin', '.3'], ['--lambda-ranking-margin', '.1'],
                      ['--ranking-warmup-epochs', '2']):
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                FinetuneMarginLoss(arguments('ranking', *flags)).validate_checkpoint(saved)
        for module, checkpoint in ((FinetuneMarginLoss(arguments('background')), saved),
                                   (FinetuneMarginLoss(parser.parse_args([])), saved), (original, {})):
            with self.assertRaises(ValueError):
                module.validate_checkpoint(checkpoint)

    def test_invalid_lengths_and_valid_nan_are_rejected(self):
        module = FinetuneMarginLoss(arguments())
        c, a, labels, lengths = inputs()
        for lengths in ([4], [-1, 3], [5, 4], [4., 3.]):
            with self.assertRaises(ValueError): module(c, a, labels, lengths, 2)
        with torch.no_grad(): c[1, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'): module(c, a, labels, [4, 4], 2)


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.TrainerIntegrationTests.setUpClass()
        cls.train = staticmethod(fixture.TrainerIntegrationTests.train)

    @classmethod
    def tearDownClass(cls):
        fixture.TrainerIntegrationTests.tearDownClass()

    def run_arm(self, root, mode='none', epochs=3, resume=False):
        root.mkdir(exist_ok=True)
        np.save(root / 'gt.npy', np.zeros(1))
        args = arguments(mode, '--skip-eval', '--batch-size', '2', '--visual-width', '4',
                         '--max-epoch', str(epochs), '--ranking-margin', '.8', '--abnormal-bg-margin', '.8',
                         '--abnormal-bg-warmup-epochs', '1', '--model-path', str(root / 'model.pth'),
                         '--checkpoint-path', str(root / 'last.pth'), '--log-path', str(root / 'train.log'))
        args.gt_path = args.gt_segment_path = args.gt_label_path = str(root / 'gt.npy')
        args.use_checkpoint = resume
        torch.manual_seed(234)
        model = fixture.ToyModel()
        loaders = [DataLoader(fixture.ToyDataset(n), batch_size=2, shuffle=True) for n in (True, False)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.train(model, *loaders, [], args, {'Normal': 'normal', 'Abnormal': 'abnormal'}, 'cpu')
        return torch.load(root / 'last.pth', weights_only=False)

    def test_both_ports_train_after_warmup_resume_and_keep_temporal_a(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.run_arm(root / 'baseline')
            for mode in ('ranking', 'background'):
                full = self.run_arm(root / mode, mode)
                self.run_arm(root / (mode + '_resume'), mode, epochs=2)
                resumed = self.run_arm(root / (mode + '_resume'), mode, resume=True)
                for name, value in full['model_state_dict'].items():
                    torch.testing.assert_close(value, resumed['model_state_dict'][name], rtol=0, atol=0)
                self.assertFalse(torch.equal(full['model_state_dict']['encoder.weight'], baseline['model_state_dict']['encoder.weight']))
                self.assertEqual(full['finetune_margin_config']['mode'], mode)
                log = (root / mode / 'train.log').read_text()
                self.assertIn('finetune_margin_active=False', log)
                self.assertIn('finetune_margin_active=True', log)
                self.assertIn('a_branch_pooling=temporal_segment', log)
                self.assertIn('c_branch_pooling=original_hard_topk', log)
                self.assertIn('avg_weighted_finetune_margin=', log)
                self.assertIn('loss_smooth_a=', log)
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    self.run_arm(root / mode, mode)
            self.assertNotIn('finetune_margin_config', baseline)

    def test_warmup_preserves_the_real_trainer_updates_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # max_epoch stays 2 so the configured loss has an active epoch;
            # capture model checkpoints as the first (warm-up) epoch is saved.
            states = {}
            original_save = torch.save
            def capture(payload, path, *args, **kwargs):
                if isinstance(payload, dict) and payload.get('next_epoch') == 1:
                    states[Path(path).parent.name] = copy.deepcopy(payload['model_state_dict'])
                return original_save(payload, path, *args, **kwargs)
            from unittest.mock import patch
            with patch('torch.save', side_effect=capture):
                for mode in ('none', 'ranking', 'background'):
                    self.run_arm(root / mode, mode, epochs=2)
            self.assertEqual(len(states), 3)
            for mode in ('ranking', 'background'):
                for name, tensor in states['none'].items():
                    torch.testing.assert_close(tensor, states[mode][name], rtol=0, atol=0)

    def test_actual_optimizer_matches_reference_r_plus_imported_loss(self):
        for mode, function in (('ranking', ranking_margin_loss),
                               ('background', abnormal_background_suppression_loss)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                np.save(root / 'gt.npy', np.zeros(1))
                args = arguments(mode, '--skip-eval', '--max-epoch', '1', '--batch-size', '2',
                                 '--visual-width', '4', '--ranking-margin', '.8', '--abnormal-bg-margin', '.8',
                                 '--ranking-warmup-epochs', '0', '--abnormal-bg-warmup-epochs', '0',
                                 '--model-path', str(root / 'model.pth'), '--checkpoint-path', str(root / 'last.pth'))
                args.gt_path = args.gt_segment_path = args.gt_label_path = str(root / 'gt.npy')
                torch.manual_seed(234)
                actual = fixture.ToyModel()
                expected = copy.deepcopy(actual)
                loaders = [DataLoader(fixture.ToyDataset(n, count=2), batch_size=2) for n in (True, False)]
                batches = [next(iter(loader)) for loader in loaders]
                visual = torch.cat([batch[0] for batch in batches])
                lengths = torch.cat([batch[2] for batch in batches])
                labels = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
                text, c, a = expected(visual, None, ['normal', 'abnormal'], lengths)
                env = fixture.TrainerIntegrationTests.train_env
                loss = env['CLAS2'](c, labels, lengths, 'cpu', pooling='mean')
                loss += env['CLASM'](a, labels, lengths, 'cpu', pooling='mean',
                                    temporal_segment=True, temporal_smoothing_kernel=3)
                normalized = F.normalize(text, dim=-1)
                loss += (normalized[0] @ normalized[1]).abs() * .1
                loss += .05 * env['a_branch_temporal_smoothness'](a, lengths)
                loss += .05 * function(c, a, labels, lengths, .15, .8, 'both', 'cpu')
                optimizer = torch.optim.AdamW(expected.parameters(), lr=args.lr)
                loss.backward()
                optimizer.step()
                with contextlib.redirect_stdout(io.StringIO()):
                    self.train(actual, *loaders, [], args, {'Normal': 'normal', 'Abnormal': 'abnormal'}, 'cpu')
                for name, tensor in actual.state_dict().items():
                    torch.testing.assert_close(tensor, expected.state_dict()[name], rtol=1e-6, atol=1e-8)

    def test_evaluation_and_best_reload_keep_the_margin_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / 'gt.npy', np.zeros(1))
            args = arguments('ranking', '--max-epoch', '1', '--batch-size', '64', '--visual-width', '4',
                             '--ranking-warmup-epochs', '0', '--model-path', str(root / 'model.pth'),
                             '--checkpoint-path', str(root / 'best.pth'), '--log-path', str(root / 'train.log'))
            args.gt_path = args.gt_segment_path = args.gt_label_path = str(root / 'gt.npy')
            loaders = [DataLoader(fixture.ToyDataset(n, count=704), batch_size=64) for n in (True, False)]
            env = fixture.TrainerIntegrationTests.train_env
            old_test = env['test']
            seen = []
            def evaluate(model, *unused, **kwargs):
                seen.append(True)
                # Evaluator retains the ordinary three-output model interface.
                model.eval()
                with torch.no_grad():
                    output = model(torch.randn(1, 17, 4), None, [], torch.tensor([17]))
                self.assertEqual(len(output), 3)
                model.train()
                return .6, .4
            env['test'] = evaluate
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.train(fixture.ToyModel(), *loaders, [], args,
                               {'Normal': 'normal', 'Abnormal': 'abnormal'}, 'cpu')
            finally:
                env['test'] = old_test
            self.assertEqual(len(seen), 1)
            checkpoint = torch.load(args.checkpoint_path, weights_only=False)
            self.assertEqual(checkpoint['finetune_margin_config'], FinetuneMarginLoss(args).config)
            self.assertTrue(Path(args.model_path).is_file())


class CommandTests(unittest.TestCase):
    def test_documented_training_cells_parse_and_select_only_one_loss(self):
        text = (ROOT / 'FINETUNE_MARGINS.md').read_text(encoding='utf-8')
        modes, outputs = [], set()
        for cell in re.findall(r'```python\n(.*?)\n```', text, re.S):
            self.assertEqual(cell.count('!'), 1)
            command = cell.replace('\\\n', ' ').split('python -u experiments/topk_variants/train_ucf.py ', 1)[1]
            args = parser.parse_args(shlex.split(command))
            module = FinetuneMarginLoss(args)
            modes.append(module.mode)
            self.assertNotIn(args.model_path, outputs)
            outputs.add(args.model_path)
            self.assertEqual(args.batch_size, 64)
            self.assertEqual(args.lr, 2e-5)
            self.assertTrue(args.temporal_segment_topk)
            if module.enabled:
                self.assertEqual(module.branch, 'both')
                self.assertEqual(module.ratio, .15)
                self.assertEqual(module.warmup, 1)
        self.assertEqual(modes, ['none', 'ranking', 'background'])


if __name__ == '__main__':
    unittest.main()
