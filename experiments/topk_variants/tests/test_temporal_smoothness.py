import sys
import unittest
from pathlib import Path

import torch


VARIANT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VARIANT_ROOT))

from temporal_smoothness import (
    a_branch_temporal_smoothness,
    c_branch_temporal_smoothness,
    temporal_total_variation,
)


class TemporalTotalVariationTests(unittest.TestCase):
    def test_constant_scores_have_zero_loss(self):
        scores = torch.full((2, 5), 0.4)
        lengths = torch.tensor([5, 3])
        loss = temporal_total_variation(scores, lengths)
        self.assertEqual(loss.item(), 0.0)

    def test_padding_is_ignored(self):
        scores = torch.tensor([[0.0, 1.0, 3.0, 100.0]])
        loss = temporal_total_variation(scores, torch.tensor([3]))
        self.assertAlmostEqual(loss.item(), 1.5)

    def test_each_video_is_averaged_before_batch_mean(self):
        scores = torch.tensor([
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 2.0, 0.0, 0.0],
        ])
        loss = temporal_total_variation(scores, torch.tensor([4, 2]))
        expected = ((1.0 / 3.0) + 2.0) / 2.0
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_single_snippet_batch_returns_differentiable_zero(self):
        scores = torch.tensor([[0.7]], requires_grad=True)
        loss = temporal_total_variation(scores, torch.tensor([1]))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(scores.grad)


class BranchSmoothnessTests(unittest.TestCase):
    def test_c_branch_uses_sigmoid_probabilities(self):
        logits1 = torch.tensor([[[0.0], [0.0], [0.0]]])
        loss = c_branch_temporal_smoothness(
            logits1, torch.tensor([3])
        )
        self.assertEqual(loss.item(), 0.0)

    def test_a_branch_uses_one_minus_normal_probability(self):
        logits2 = torch.tensor([[
            [4.0, 0.0],
            [0.0, 4.0],
            [0.0, 4.0],
        ]])
        loss = a_branch_temporal_smoothness(
            logits2, torch.tensor([3])
        )
        expected_jump = torch.softmax(
            torch.tensor([0.0, 4.0]), dim=0
        )[1] - torch.softmax(torch.tensor([4.0, 0.0]), dim=0)[1]
        self.assertAlmostEqual(
            loss.item(), expected_jump.item() / 2.0, places=6
        )

    def test_a_branch_loss_backpropagates(self):
        logits2 = torch.randn(2, 5, 4, requires_grad=True)
        loss = a_branch_temporal_smoothness(
            logits2, torch.tensor([5, 3])
        )
        loss.backward()
        self.assertIsNotNone(logits2.grad)
        self.assertTrue(torch.isfinite(logits2.grad).all())

    def test_invalid_c_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            c_branch_temporal_smoothness(
                torch.ones(2, 5), torch.tensor([5, 5])
            )


if __name__ == "__main__":
    unittest.main()
