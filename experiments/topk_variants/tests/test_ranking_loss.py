import ast
from pathlib import Path
import shlex
import sys
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from options import parser
from ranking_loss import (c_pairwise_ranking, ranking_active, ranking_config,
                          validate_checkpoint_ranking, validate_ranking_config)
from topk_pooling import (topk_pool, video_topk_size, smooth_temporal_scores,
                         temporal_segment_pool)


def args_for(*extra):
    return parser.parse_args(['--topk-pooling', 'soft', '--c-ranking-loss', *extra])


class RankingTests(unittest.TestCase):
    def test_numeric_examples(self):
        loss, stats = c_pairwise_ranking(torch.tensor([.55, .6]), torch.tensor([0, 1]))
        self.assertAlmostEqual(loss.item(), .15, places=6)
        self.assertEqual(stats['pairs'], 1)
        self.assertEqual(stats['violation_fraction'], 1)
        loss, stats = c_pairwise_ranking(torch.tensor([.3, .8]), torch.tensor([0, 1]))
        self.assertEqual(loss.item(), 0)
        self.assertEqual(stats['violation_fraction'], 0)

    def test_all_pairs_and_order_invariance(self):
        scores = torch.tensor([.5, .7, .2, .4])
        labels = torch.tensor([1, 1, 0, 0])
        loss, stats = c_pairwise_ranking(scores, labels)
        # Only positive=.5 vs negative=.4 violates: .1 / 4 pairs.
        self.assertAlmostEqual(loss.item(), .025, places=6)
        self.assertEqual(stats['pairs'], 4)
        self.assertEqual(stats['violation_fraction'], .25)
        permutation = torch.tensor([3, 0, 2, 1])
        shuffled, _ = c_pairwise_ranking(scores[permutation], labels[permutation])
        torch.testing.assert_close(loss, shuffled)

    def test_mean_is_invariant_to_batch_duplication(self):
        scores, labels = torch.tensor([.6, .55]), torch.tensor([1, 0])
        first, _ = c_pairwise_ranking(scores, labels)
        doubled, stats = c_pairwise_ranking(scores.repeat(2), labels.repeat(2))
        torch.testing.assert_close(first, doubled)
        self.assertEqual(stats['pairs'], 4)

    def test_gradient_pushes_positive_up_negative_down(self):
        scores = torch.tensor([.55, .6], requires_grad=True)
        loss, _ = c_pairwise_ranking(scores, torch.tensor([0, 1]))
        loss.backward()
        torch.testing.assert_close(scores.grad, torch.tensor([1., -1.]))

    def test_no_pairs_returns_differentiable_zero(self):
        for labels in (torch.zeros(3), torch.ones(3), torch.empty(0)):
            scores = torch.full(labels.shape, .5, requires_grad=True)
            loss, stats = c_pairwise_ranking(scores, labels)
            loss.backward()
            self.assertEqual(loss.item(), 0)
            self.assertEqual(stats['pairs'], 0)
            self.assertIsNotNone(scores.grad)

    def test_half_precision_is_promoted(self):
        scores = torch.tensor([.5, .6], dtype=torch.float16, requires_grad=True)
        loss, _ = c_pairwise_ranking(scores, torch.tensor([0, 1]))
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_invalid_inputs(self):
        for scores, labels in [(torch.ones(2, 1), torch.ones(2)),
                               (torch.tensor([float('nan')]), torch.zeros(1)),
                               (torch.tensor([1.1]), torch.zeros(1)),
                               (torch.tensor([.5]), torch.tensor([.5]))]:
            with self.assertRaises(ValueError): c_pairwise_ranking(scores, labels)

    def test_default_off_and_absolute_start_epoch(self):
        args = parser.parse_args([])
        validate_ranking_config(args)
        self.assertFalse(ranking_active(args, 10))
        args = args_for('--c-ranking-start-epoch', '4')
        self.assertFalse(ranking_active(args, 3))
        self.assertTrue(ranking_active(args, 4))
        self.assertTrue(ranking_active(args, 8))

    def test_bad_config_and_conflicts(self):
        for extra in [('--c-ranking-margin', '0'), ('--c-ranking-margin', '1'),
                      ('--c-ranking-margin', 'nan'), ('--c-ranking-weight', '-1'),
                      ('--c-ranking-weight', 'inf'), ('--c-ranking-start-epoch', '0'),
                      ('--topk-pooling', 'mean'), ('--topk-pooling', 'multi_k'),
                      ('--a-topk-temperature-schedule', 'linear'),
                      ('--temporal-smoothness-branch', 'a'),
                      ('--c-temporal-smoothing-kernel', '3'),
                      ('--temporal-smoothing-kernel', '3'),
                      ('--dual-k',), ('--adaptive-instance-selection',),
                      ('--temporal-segment-topk',), ('--c-temporal-segment-topk',)]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_ranking_config(args_for(*extra))
        validate_ranking_config(args_for('--c-ranking-weight', '0'))

    def test_resume_config(self):
        args = args_for()
        ckpt = {'c_ranking_config': ranking_config(args)}
        validate_checkpoint_ranking(args, ckpt)
        validate_checkpoint_ranking(parser.parse_args([]), {})
        with self.assertRaises(ValueError): validate_checkpoint_ranking(args, {})
        args.c_ranking_margin = .3
        with self.assertRaises(ValueError): validate_checkpoint_ranking(args, ckpt)
        with self.assertRaises(ValueError):
            validate_checkpoint_ranking(parser.parse_args([]), ckpt)

    def test_documented_command_parses(self):
        doc = (ROOT / 'SOFT_TOPK_RANKING_DESIGN.md').read_text(encoding='utf-8')
        command = doc.split('python experiments/topk_variants/train_ucf.py', 1)[1]
        command = command.split('```', 1)[0].replace('\\\n', ' ')
        args = parser.parse_args(shlex.split(command))
        validate_ranking_config(args)
        self.assertTrue(args.c_ranking_loss)
        self.assertEqual((args.seed, args.batch_size, args.max_epoch), (234, 64, 10))


class RankingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / 'train_ucf.py').read_text(encoding='utf-8'))
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name in ('CLAS2', 'resolve_pool_k')]
        env = dict(torch=torch, F=F, topk_pool=topk_pool, video_topk_size=video_topk_size,
                   smooth_temporal_scores=smooth_temporal_scores,
                   temporal_segment_pool=temporal_segment_pool)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), 'train_ucf.py', 'exec'), env)
        cls.clas2 = staticmethod(env['CLAS2'])

    def test_pooled_scores_match_bce_and_padding_has_no_gradient(self):
        logits = torch.zeros(2, 19, 1, requires_grad=True)
        labels = torch.tensor([[1., 0.], [0., 1.]])
        lengths = torch.tensor([16, 17])
        bce, scores = self.clas2(logits, labels, lengths, 'cpu', return_video_scores=True)
        expected = torch.stack([topk_pool(logits[b, :n, 0].sigmoid(), video_topk_size(n),
                                          mode='soft', temperature=1.)
                                for b, n in enumerate((16, 17))])
        torch.testing.assert_close(scores, expected)
        torch.testing.assert_close(bce, F.binary_cross_entropy(expected, 1-labels[:, 0]))
        rank, stats = c_pairwise_ranking(scores, 1-labels[:, 0])
        (bce + .1 * rank).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[0, :16].sum().item(), 0)
        self.assertLess(logits.grad[1, :17].sum().item(), 0)
        self.assertEqual(logits.grad[0, 16:].abs().sum().item(), 0)
        self.assertEqual(logits.grad[1, 17:].abs().sum().item(), 0)

    def test_zero_weight_matches_old_bce_loss_and_gradient(self):
        labels = torch.tensor([[1., 0.], [0., 1.]])
        lengths = torch.tensor([16, 16])
        base = torch.linspace(-1, 1, 32).reshape(2, 16, 1)
        results = []
        for enabled in (False, True):
            logits = base.clone().requires_grad_()
            result = self.clas2(logits, labels, lengths, 'cpu', return_video_scores=enabled)
            if enabled:
                bce, scores = result
                rank, _ = c_pairwise_ranking(scores, 1-labels[:, 0])
                result = bce + 0.0 * rank
            result.backward()
            results.append((result.detach(), logits.grad))
        for before, after in zip(*results):
            torch.testing.assert_close(before, after, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
