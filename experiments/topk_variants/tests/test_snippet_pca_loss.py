import ast
import contextlib
import io
from pathlib import Path
import random
import shlex
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from options import parser
from snippet_pca_loss import (NormalPCASubspace, SnippetAuxiliaryLoss,
                              calibrate_normal_pca, pca_auxiliary_loss,
                              snippet_score_ranking, validate_snippet_config)
from topk_pooling import (topk_pool, video_topk_size, smooth_temporal_scores,
                         temporal_segment_pool)
from temporal_smoothness import a_branch_temporal_smoothness, c_branch_temporal_smoothness
from ranking_loss import (validate_ranking_config, ranking_config, ranking_active,
                          validate_checkpoint_ranking)
from temperature_schedule import (validate_temperature_config, temperature_config,
                                  epoch_temperatures, validate_checkpoint_temperature)
from training_log import TrainingLogger


def args_for(*flags):
    return parser.parse_args(['--snippet-ranking-loss', *flags])


class SnippetRankingTests(unittest.TestCase):
    def test_individual_violations_survive_when_pooled_ranking_is_zero(self):
        scores = torch.full((2, 32), .1)
        scores[0, 0] = .6
        scores[1, :3] = torch.tensor([.9, .8, .5])
        logits = scores.logit().requires_grad_()
        loss, stats, indices = snippet_score_ranking(logits, torch.tensor([32, 32]), torch.tensor([0, 1]))
        self.assertAlmostEqual(loss.item(), .2 / 3, places=6)
        self.assertEqual(stats['pairs'], 1)
        self.assertEqual(indices[1].tolist(), [0, 1, 2])
        self.assertEqual(max(0, .1 + .6 - (.9+.8+.5)/3), 0)
        loss.backward()
        self.assertGreater(logits.grad[0, 0].item(), 0)
        self.assertLess(logits.grad[1, 2].item(), 0)
        self.assertEqual(logits.grad[1, 0].item(), 0)

    def test_padding_permutation_and_batch_duplication(self):
        logits = torch.zeros(4, 20, requires_grad=True)
        lengths, labels = torch.tensor([16, 17, 8, 1]), torch.tensor([0, 1, 1, 0])
        with torch.no_grad():
            for i, n in enumerate(lengths):
                logits[i, n:] = float('nan')
        loss, stats, _ = snippet_score_ranking(logits, lengths, labels)
        self.assertAlmostEqual(loss.item(), .1, places=6)
        self.assertEqual(stats['pairs'], 4)
        permutation = torch.tensor([3, 1, 0, 2])
        shuffled, _, _ = snippet_score_ranking(logits[permutation], lengths[permutation], labels[permutation])
        doubled, _, _ = snippet_score_ranking(logits.repeat(2, 1), lengths.repeat(2), labels.repeat(2))
        torch.testing.assert_close(loss, shuffled)
        torch.testing.assert_close(loss, doubled)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        for i, n in enumerate(lengths):
            self.assertEqual(logits.grad[i, n:].abs().sum().item(), 0)

    def test_different_k_has_equal_video_weight(self):
        probabilities = torch.full((3, 32), .1)
        probabilities[0, 0] = .6
        probabilities[1, 0] = .2  # K=1, loss=.5
        probabilities[2, :3] = .5  # K=3, each loss=.2
        loss, _, _ = snippet_score_ranking(probabilities.logit(), torch.tensor([1, 1, 32]), torch.tensor([0, 1, 1]))
        self.assertAlmostEqual(loss.item(), .35, places=6)

    def test_softplus_small_temperature_and_no_pair(self):
        logits = torch.tensor([[.6], [.5]]).logit().requires_grad_()
        lengths, labels = torch.ones(2, dtype=torch.long), torch.tensor([0, 1])
        hinge, _, _ = snippet_score_ranking(logits, lengths, labels)
        soft, _, _ = snippet_score_ranking(logits, lengths, labels, mode='softplus', temperature=1e-5)
        torch.testing.assert_close(hinge, soft)
        for same_label in (0, 1):
            loss, stats, _ = snippet_score_ranking(logits, lengths, torch.full((2,), same_label))
            loss.backward()
            self.assertEqual(loss.item(), 0)
            self.assertEqual(stats['pairs'], 0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_valid_length_label_and_nonfinite(self):
        for lengths, labels, value in [([0, 1], [0, 1], 0), ([3, 1], [0, 1], 0),
                                      ([1.5, 1], [0, 1], 0), ([1, 1], [.5, 1], 0),
                                      ([1, 1], [0, 1], float('inf'))]:
            with self.subTest(lengths=lengths, labels=labels, value=value), self.assertRaises(ValueError):
                snippet_score_ranking(torch.full((2, 2), float(value)), torch.tensor(lengths), torch.tensor(labels))


class PCATests(unittest.TestCase):
    def axis_pca(self):
        pca = NormalPCASubspace()
        pca.fit(torch.tensor([[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]]), 1)
        return pca

    def test_projection_scale_invariance_and_gradient(self):
        pca = self.axis_pca()
        features = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], requires_grad=True)
        residual = pca.residuals(features)
        torch.testing.assert_close(residual, torch.tensor([0., 1., .5]))
        torch.testing.assert_close(residual, pca.residuals(7 * features))
        residual.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad[-1].abs().sum().item(), 0)
        self.assertFalse(pca.basis.requires_grad)

    def test_pca_loss_uses_score_indices_masks_padding_and_keeps_gradients(self):
        pca = self.axis_pca()
        h = torch.tensor([[[1., 1.], [float('nan'), float('nan')]],
                          [[2., 1.], [0., 1.]]], requires_grad=True)
        # Only index 0 selected by score; residual-only selection would pick 1.
        loss, stats = pca_auxiliary_loss(h, torch.tensor([1, 2]), torch.tensor([0, 1]), {1: torch.tensor([0])}, pca)
        self.assertAlmostEqual(stats['normal'], .5, places=6)
        self.assertAlmostEqual(stats['residual'], .4, places=6)
        self.assertAlmostEqual(loss.item(), .9, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertEqual(h.grad[0, 1].abs().sum().item(), 0)
        self.assertEqual(h.grad[1, 1].abs().sum().item(), 0)
        self.assertGreater(h.grad[0, 0, 1].item(), 0)
        self.assertLess(h.grad[1, 0, 1].item(), 0)

    def test_state_roundtrip_and_degenerate_fit(self):
        pca = self.axis_pca()
        clone = NormalPCASubspace()
        clone.load_state_dict(pca.state_dict(), 2, 1)
        x = torch.tensor([[2., 3.]])
        torch.testing.assert_close(pca.residuals(x), clone.residuals(x))
        for sample in (torch.zeros(8, 2), torch.ones(8, 2), torch.full((8, 2), float('nan'))):
            with self.assertRaises(ValueError):
                NormalPCASubspace().fit(sample, 1)
        with self.assertRaises(ValueError):
            clone.load_state_dict(pca.state_dict(), 3, 1)

    def test_calibration_preserves_rng_modes_and_rejects_test(self):
        model = ToyModel()
        model.train()
        model.classifier.eval()
        args = args_for('--pca-loss', '--visual-width', '4', '--pca-rank', '2', '--pca-fit-videos', '4')
        dataset = ToyDataset(True, 4, random_access=True)
        torch.manual_seed(789); np.random.seed(789); random.seed(789)
        expected = (torch.rand(3), np.random.rand(), random.random())
        torch.manual_seed(789); np.random.seed(789); random.seed(789)
        pca = calibrate_normal_pca(model, dataset, 'cpu', args, 3)
        actual = (torch.rand(3), np.random.rand(), random.random())
        torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
        self.assertEqual(expected[1:], actual[1:])
        self.assertTrue(model.training)
        self.assertFalse(model.classifier.training)
        self.assertEqual(pca.metadata['videos'], 4)
        dataset.test_mode = True
        with self.assertRaises(ValueError):
            calibrate_normal_pca(model, dataset, 'cpu', args, 3)


class ConfigTests(unittest.TestCase):
    def test_disabled_warmup_conflicts_and_resume(self):
        disabled = SnippetAuxiliaryLoss(parser.parse_args([]))
        self.assertFalse(disabled.active(100))
        disabled.restore({})
        args = args_for('--pca-loss')
        auxiliary = SnippetAuxiliaryLoss(args)
        self.assertFalse(auxiliary.active(2))
        self.assertTrue(auxiliary.active(3))
        auxiliary.restore({'epoch': 1, 'snippet_auxiliary': auxiliary.state_dict()})
        with self.assertRaisesRegex(ValueError, 'missing'):
            auxiliary.restore({'epoch': 2, 'snippet_auxiliary': auxiliary.state_dict()})
        with self.assertRaises(ValueError): auxiliary.restore({})
        with self.assertRaises(ValueError): disabled.restore({'snippet_auxiliary': auxiliary.state_dict()})
        for flags in [('--pca-loss', '--pca-rank', '512'), ('--snippet-ranking-margin', 'nan'),
                      ('--snippet-ranking-weight', '-1'), ('--snippet-ranking-start-epoch', '0'),
                      ('--dual-k',), ('--adaptive-instance-selection',), ('--topk-pooling', 'soft'),
                      ('--c-temporal-segment-topk',), ('--temporal-smoothness-branch', 'both'),
                      ('--pca-loss', '--pca-start-epoch', '2')]:
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate_snippet_config(args_for(*flags))
        with self.assertRaises(ValueError):
            validate_snippet_config(parser.parse_args(['--pca-loss']))
        validate_snippet_config(args_for('--temporal-segment-topk', '--temporal-smoothing-kernel', '3',
                                         '--temporal-smoothness-branch', 'a'))

    def test_command_catalog(self):
        doc = (ROOT.parents[1] / 'reports/topk_variants/LENH_KAGGLE_SNIPPET_RANKING_PCA.md').read_text(encoding='utf-8')
        commands = doc.split('python -u experiments/topk_variants/train_ucf.py')[1:]
        self.assertEqual(len(commands), 3)
        paths = set()
        for body in commands:
            command = body.split('```')[0].replace('\\\n', ' ')
            for seed in (234, 3407, 2026):
                args = parser.parse_args(shlex.split(command.replace('234', str(seed))))
                validate_snippet_config(args)
                validate_ranking_config(args)
                validate_temperature_config(args)
                self.assertFalse(args.amp or args.skip_eval or args.use_checkpoint)
                self.assertEqual((args.seed, args.batch_size, args.max_epoch), (seed, 64, 10))
                self.assertNotIn(args.checkpoint_path, paths)
                paths.add(args.checkpoint_path)


# Exercise the real model forward and trainer without importing/downloading CLIP.
model_ast = ast.parse((ROOT.parents[1] / 'src/model.py').read_text(encoding='utf-8'))
forward_ast = next(n for c in model_ast.body if isinstance(c, ast.ClassDef) and c.name == 'CLIPVAD'
                   for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
forward_env = {}
exec(compile(ast.Module(body=[forward_ast], type_ignores=[]), 'model.py', 'exec'), forward_env)


class ToyModel(torch.nn.Module):
    forward = forward_env['forward']

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.classifier = torch.nn.Linear(4, 1)
        self.mlp1, self.mlp2 = torch.nn.Identity(), torch.nn.Identity()
        self.texts = torch.nn.Parameter(torch.randn(2, 4))

    def encode_video(self, visual, padding_mask, lengths):
        return self.encoder(visual)

    def encode_textprompt(self, text):
        return self.texts


class ToyDataset(Dataset):
    test_mode = False

    def __init__(self, normal, count=4, random_access=False):
        self.normal, self.count, self.random_access = normal, count, random_access

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index + (0 if self.normal else 1000))
        x = torch.randn(17, 4, generator=generator) + (0 if self.normal else 1)
        if self.random_access:
            x = x + torch.rand(()) + np.random.rand() + random.random()
        return x, 'Normal' if self.normal else 'Abnormal', 17


class TrainerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        tree = ast.parse((ROOT / 'train_ucf.py').read_text(encoding='utf-8'))
        env = dict(globals(), get_prompt_text=lambda labels: list(labels.values()),
                   get_batch_label=lambda labels, *_: torch.tensor([[1., 0.] if y == 'Normal' else [0., 1.] for y in labels]),
                   test=lambda *a, **kw: (0.5, 0.4))
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), 'train_ucf.py', 'exec'), env)
        cls.train = staticmethod(env['train'])
        cls.train_env = env

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_optional_feature_return_preserves_outputs_and_gradient(self):
        torch.manual_seed(12)
        model = ToyModel()
        x, lengths = torch.randn(2, 17, 4), torch.tensor([17, 17])
        base = model(x, None, [], lengths)
        extended = model(x, None, [], lengths, return_visual_features=True)
        self.assertEqual(len(base), 3)
        for before, after in zip(base, extended[:3]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertTrue(extended[3].requires_grad)
        for output in (base, extended):
            model.zero_grad()
            (output[1].sum() + output[2].sum()).backward()
            gradient = model.encoder.weight.grad.clone()
            if output is base: original = gradient
            else: torch.testing.assert_close(original, gradient, rtol=0, atol=0)

    def test_train_all_arms_and_pca_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.save(root / 'gt.npy', np.zeros(1))
            checkpoints = []
            for name, flags in [('r', []), ('s', ['--snippet-ranking-loss']),
                                ('sp', ['--snippet-ranking-loss', '--pca-loss'])]:
                args = parser.parse_args([
                    '--skip-eval', '--batch-size', '2', '--max-epoch', '3', '--visual-width', '4', '--pca-rank', '2',
                    '--temporal-segment-topk', '--temporal-segment-start-epoch', '1',
                    '--temporal-smoothing-kernel', '3', '--temporal-smoothness-branch', 'a',
                    '--model-path', str(root / (name+'.pth')), '--checkpoint-path', str(root / (name+'_ckpt.pth')),
                    '--log-path', str(root / (name+'.log')), *flags])
                args.gt_path = args.gt_segment_path = args.gt_label_path = str(root / 'gt.npy')
                normal = DataLoader(ToyDataset(True), batch_size=2)
                anomaly = DataLoader(ToyDataset(False), batch_size=2)
                torch.manual_seed(44)
                model = ToyModel()
                with contextlib.redirect_stdout(io.StringIO()):
                    self.train(model, normal, anomaly, [], args, {'Normal': 'normal', 'Abnormal': 'abnormal'}, 'cpu')
                ckpt = torch.load(args.checkpoint_path, weights_only=False)
                checkpoints.append(ckpt)
                self.assertTrue(all(torch.isfinite(t).all() for t in ckpt['model_state_dict'].values()))
                if name == 'sp':
                    log = Path(args.log_path).read_text(encoding='utf-8')
                    self.assertEqual(log.count('pca_fit='), 1)
                    self.assertIn('avg_weighted_pca=', log)
                    args.use_checkpoint, args.max_epoch = True, 4
                    before = ckpt['snippet_auxiliary']['pca']['basis']
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.train(ToyModel(), normal, anomaly, [], args, {'Normal': 'normal', 'Abnormal': 'abnormal'}, 'cpu')
                    resumed = torch.load(args.checkpoint_path, weights_only=False)
                    torch.testing.assert_close(before, resumed['snippet_auxiliary']['pca']['basis'], rtol=0, atol=0)
                    self.assertEqual(Path(args.log_path).read_text(encoding='utf-8').count('pca_fit='), 1)
            self.assertIsNone(checkpoints[1]['snippet_auxiliary']['pca'])
            self.assertIsNotNone(checkpoints[2]['snippet_auxiliary']['pca'])
            # Additional losses must actually reach the learned encoder.
            self.assertFalse(torch.equal(checkpoints[0]['model_state_dict']['encoder.weight'], checkpoints[1]['model_state_dict']['encoder.weight']))
            self.assertFalse(torch.equal(checkpoints[1]['model_state_dict']['encoder.weight'], checkpoints[2]['model_state_dict']['encoder.weight']))

    def test_evaluation_checkpoint_keeps_fixed_pca_across_best_model_reload(self):
        # Three improving epochs followed by a decline exercise both best saves
        # and the legacy epoch-end model-only reload. Calibration must happen once.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.save(root / 'gt.npy', np.zeros(1))
            args = args_for('--pca-loss', '--batch-size', '64', '--max-epoch', '4',
                            '--visual-width', '4', '--pca-rank', '2', '--pca-fit-videos', '4',
                            '--model-path', str(root / 'model.pth'),
                            '--checkpoint-path', str(root / 'checkpoint.pth'),
                            '--log-path', str(root / 'train.log'))
            args.gt_path = args.gt_segment_path = args.gt_label_path = str(root / 'gt.npy')
            aucs = iter([.5, .6, .7, .65])
            old_test = self.train_env['test']
            self.train_env['test'] = lambda *a, **kw: (next(aucs), .1)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.train(ToyModel(), DataLoader(ToyDataset(True, 704), batch_size=64),
                               DataLoader(ToyDataset(False, 704), batch_size=64), [], args,
                               {'Normal': 'normal', 'Abnormal': 'abnormal'}, 'cpu')
            finally:
                self.train_env['test'] = old_test
            checkpoint = torch.load(args.checkpoint_path, weights_only=False)
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertEqual(checkpoint['snippet_auxiliary']['pca']['metadata']['fit_epoch'], 3)
            self.assertEqual(Path(args.log_path).read_text(encoding='utf-8').count('pca_fit='), 1)
            exported = torch.load(args.model_path, weights_only=True)
            for name, tensor in exported.items():
                torch.testing.assert_close(tensor, checkpoint['model_state_dict'][name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
