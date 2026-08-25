import sys
import unittest
from pathlib import Path

import torch


VARIANT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VARIANT_ROOT))

from topk_pooling import (
    multi_k_pooled_scores,
    multi_k_scale_sizes,
    smooth_temporal_scores,
    soft_topk_pool,
    temporal_segment_pool,
    topk_pool,
)


class MultiKPercentageTests(unittest.TestCase):
    def test_default_scales_are_one_five_ten_twenty_percent(self):
        self.assertEqual(multi_k_scale_sizes(256), [3, 13, 26, 52])
        self.assertEqual(multi_k_scale_sizes(139), [2, 7, 14, 28])
        self.assertEqual(multi_k_scale_sizes(50), [1, 3, 5, 10])

    def test_multi_k_returns_four_percentage_scales(self):
        scores = torch.linspace(0.0, 1.0, 256)
        pooled, sizes = multi_k_pooled_scores(scores)
        self.assertEqual(sizes, [3, 13, 26, 52])
        self.assertEqual(tuple(pooled.shape), (4,))

    def test_soft_topk_returns_one_score_per_a_branch_class(self):
        scores = torch.rand(256, 14)
        pooled = soft_topk_pool(scores, k=17, temperature=1.0)

        self.assertEqual(tuple(pooled.shape), (14,))


class TemporalSegmentPoolingTests(unittest.TestCase):
    def test_selects_one_contiguous_segment(self):
        scores = torch.tensor([0.0, 9.0, 0.0, 8.0, 8.0, 0.0])

        pooled, start = temporal_segment_pool(
            scores, k=2, return_start_indices=True
        )

        self.assertEqual(start.item(), 3)
        self.assertAlmostEqual(pooled.item(), 8.0)

    def test_selection_is_class_specific(self):
        scores = torch.tensor([
            [5.0, 0.0],
            [5.0, 0.0],
            [0.0, 0.0],
            [0.0, 7.0],
            [0.0, 7.0],
        ])

        pooled, starts = temporal_segment_pool(
            scores, k=2, return_start_indices=True
        )

        self.assertTrue(torch.equal(starts, torch.tensor([0, 3])))
        self.assertTrue(torch.allclose(pooled, torch.tensor([5.0, 7.0])))

    def test_smoothing_uses_replicate_padding_and_preserves_shape(self):
        scores = torch.tensor([3.0, 0.0, 0.0])

        smoothed = smooth_temporal_scores(scores, kernel_size=3)

        self.assertEqual(tuple(smoothed.shape), (3,))
        self.assertTrue(torch.allclose(
            smoothed, torch.tensor([2.0, 1.0, 0.0])
        ))

    def test_kernel_one_is_exact_identity(self):
        scores = torch.rand(7, 4)

        smoothed = smooth_temporal_scores(scores, kernel_size=1)

        self.assertIs(smoothed, scores)

    def test_smoothing_keeps_class_channels_independent(self):
        scores = torch.tensor([
            [3.0, 0.0],
            [0.0, 0.0],
            [0.0, 6.0],
        ])

        smoothed = smooth_temporal_scores(scores, kernel_size=3)

        self.assertTrue(torch.allclose(
            smoothed,
            torch.tensor([
                [2.0, 0.0],
                [1.0, 2.0],
                [0.0, 4.0],
            ]),
        ))

    def test_c_style_smoothing_happens_before_hard_topk(self):
        probabilities = torch.tensor([0.1, 0.9, 0.1])

        smoothed = smooth_temporal_scores(probabilities, kernel_size=3)
        pooled = topk_pool(smoothed, k=1, mode="mean")

        torch.testing.assert_close(pooled, torch.tensor(1.1 / 3.0))

    def test_temporal_segment_pool_backpropagates(self):
        scores = torch.tensor(
            [[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]],
            requires_grad=True,
        )

        temporal_segment_pool(scores, k=2).sum().backward()

        self.assertIsNotNone(scores.grad)
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_smoothing_kernel_must_be_positive_and_odd(self):
        with self.assertRaises(ValueError):
            smooth_temporal_scores(torch.ones(4), kernel_size=2)


if __name__ == "__main__":
    unittest.main()
