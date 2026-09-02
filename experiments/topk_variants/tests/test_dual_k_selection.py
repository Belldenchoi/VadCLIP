import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dual_k_selection import (  # noqa: E402
    dual_constraint_loss,
    selection_statistics,
    select_a_branch,
    select_c_branch,
    target_class_mean,
    update_dual_lambda,
)


class DualKSelectionTests(unittest.TestCase):
    def test_c_selector_masks_padding_and_standardizes_valid_evidence(self):
        logits = torch.tensor([
            [[-2.0], [-1.0], [1.0], [2.0]],
            [[-1.0], [1.0], [100.0], [100.0]],
        ])
        selection = select_c_branch(
            logits, torch.tensor([4, 2]), dual_lambda=0.0
        )

        self.assertEqual(selection.weights.shape, (2, 4))
        self.assertTrue(torch.all(selection.weights[1, 2:] == 0))
        self.assertTrue(torch.all(selection.normalized_evidence[1, 2:] == 0))
        self.assertTrue(torch.allclose(
            selection.effective_k, selection.weights.sum(dim=1)
        ))
        for index, length in enumerate((4, 2)):
            normalized = selection.normalized_evidence[index, :length]
            self.assertAlmostEqual(normalized.mean().item(), 0.0, places=5)
            self.assertAlmostEqual(
                normalized.std(unbiased=False).item(), 1.0, places=4
            )

    def test_c_uncertainty_is_normalized_binary_entropy(self):
        selection = select_c_branch(
            torch.tensor([[-20.0, 0.0, 20.0]]),
            torch.tensor([3]), dual_lambda=0.0,
        )

        self.assertLess(selection.uncertainty[0, 0].item(), 0.001)
        self.assertAlmostEqual(selection.uncertainty[0, 1].item(), 1.0, places=5)
        self.assertLess(selection.uncertainty[0, 2].item(), 0.001)

    def test_a_selector_is_classwise_and_masks_padding(self):
        logits = torch.tensor([[
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ]])
        selection = select_a_branch(
            logits, torch.tensor([2]), dual_lambda=0.0
        )

        self.assertEqual(selection.weights.shape, logits.shape)
        self.assertEqual(selection.effective_k.shape, (1, 3))
        self.assertEqual(selection.uncertainty_ratio.shape, (1, 3))
        self.assertTrue(torch.all(selection.weights[0, 2] == 0))
        self.assertTrue(torch.all(selection.normalized_evidence[0, 2] == 0))

    def test_a_evidence_normalization_removes_temporal_affine_scale(self):
        logits = torch.tensor([[
            [1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [100.0, -100.0],
        ]])
        transformed = logits * 7.0 + torch.tensor([[[11.0, -3.0]]])
        lengths = torch.tensor([3])

        original = select_a_branch(logits, lengths, dual_lambda=0.0)
        affine = select_a_branch(transformed, lengths, dual_lambda=0.0)

        self.assertTrue(torch.allclose(
            original.normalized_evidence,
            affine.normalized_evidence,
            atol=1e-5,
        ))
        self.assertTrue(torch.allclose(original.weights, affine.weights, atol=1e-5))

    def test_legacy_evidence_path_remains_available_for_ablation(self):
        c_logits = torch.tensor([[-2.0, 0.0, 2.0]])
        a_logits = torch.tensor([[
            [2.0, 0.0], [0.0, 2.0], [1.0, 1.0],
        ]])
        lengths = torch.tensor([3])

        c_selection = select_c_branch(
            c_logits, lengths, dual_lambda=0.0,
            threshold=0.5, temperature=0.2,
            normalize_evidence=False,
        )
        a_selection = select_a_branch(
            a_logits, lengths, dual_lambda=0.0,
            threshold=0.25, temperature=0.2,
            normalize_evidence=False,
        )
        expected_c = torch.sigmoid(
            (torch.sigmoid(c_logits) - 0.5) / 0.2
        )
        expected_a = torch.sigmoid(
            (torch.softmax(a_logits, dim=-1) - 0.25) / 0.2
        )

        self.assertTrue(torch.allclose(c_selection.weights, expected_c))
        self.assertTrue(torch.allclose(a_selection.weights, expected_a))

    def test_a_margin_uncertainty_uses_full_zero_to_one_range(self):
        selection = select_a_branch(
            torch.tensor([[[0.0, 0.0], [8.0, 0.0]]]),
            torch.tensor([2]), dual_lambda=0.0,
        )

        self.assertTrue(torch.allclose(
            selection.uncertainty[0, 0], torch.ones(2)
        ))
        self.assertLess(selection.uncertainty[0, 1, 0].item(), 0.001)
        self.assertEqual(selection.uncertainty[0, 1, 1].item(), 1.0)

    def test_a_uncertainty_ratio_is_weighted_per_class(self):
        logits = torch.tensor([[
            [4.0, 0.0], [1.0, 3.0], [0.0, 2.0],
        ]])
        selection = select_a_branch(
            logits, torch.tensor([3]), dual_lambda=0.4
        )
        expected = (
            (selection.weights * selection.uncertainty).sum(dim=1)
            / (selection.weights.sum(dim=1) + 1e-6)
        )

        self.assertTrue(torch.allclose(selection.uncertainty_ratio, expected))

    def test_support_ratio_uses_each_valid_length(self):
        selection = select_c_branch(
            torch.zeros(2, 4), torch.tensor([4, 2]), dual_lambda=0.0
        )
        expected = selection.effective_k / torch.tensor([4.0, 2.0])
        self.assertTrue(torch.allclose(selection.support_ratio, expected))

    def test_larger_dual_variables_reduce_effective_support(self):
        c_logits = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])
        a_logits = torch.tensor([[
            [3.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 3.0],
        ]])
        lengths = torch.tensor([4])

        c_low = select_c_branch(c_logits, lengths, dual_lambda=0.0)
        c_high = select_c_branch(c_logits, lengths, dual_lambda=1.0)
        a_low = select_a_branch(a_logits, lengths, dual_lambda=0.0)
        a_high = select_a_branch(a_logits, lengths, dual_lambda=1.0)

        self.assertTrue(torch.all(c_high.effective_k < c_low.effective_k))
        self.assertTrue(torch.all(a_high.effective_k < a_low.effective_k))

    def test_a_constraint_can_be_restricted_to_target_classes(self):
        logits = torch.tensor([[
            [5.0, 0.0], [4.0, 1.0], [3.0, 2.0],
        ]])
        selection = select_a_branch(
            logits, torch.tensor([3]), dual_lambda=0.2
        )
        labels = torch.tensor([[1.0, 0.0]])
        expected = selection.uncertainty_ratio[0, 0] - 0.35

        actual = dual_constraint_loss(selection, 0.35, labels)
        statistics = selection_statistics(selection, labels)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(
            statistics["risk"], selection.uncertainty_ratio[0, 0]
        ))
        self.assertTrue(torch.allclose(
            statistics["k_mean"], selection.effective_k[0, 0]
        ))
        self.assertTrue(torch.allclose(
            target_class_mean(selection.effective_k, labels),
            selection.effective_k[0, 0],
        ))

    def test_selectors_backpropagate_through_weights_and_risk(self):
        c_logits = torch.tensor(
            [[-1.0, 0.0, 1.0]], requires_grad=True
        )
        a_logits = torch.tensor([[
            [2.0, 0.0], [0.0, 2.0], [1.0, 1.0],
        ]], requires_grad=True)
        lengths = torch.tensor([3])
        labels = torch.tensor([[1.0, 0.0]])

        c_selection = select_c_branch(c_logits, lengths, dual_lambda=0.2)
        a_selection = select_a_branch(a_logits, lengths, dual_lambda=0.2)
        loss = (
            c_selection.weights.sum()
            + a_selection.weights.sum()
            + dual_constraint_loss(c_selection, 0.35)
            + dual_constraint_loss(a_selection, 0.35, labels)
        )
        loss.backward()

        self.assertIsNotNone(c_logits.grad)
        self.assertIsNotNone(a_logits.grad)
        self.assertTrue(torch.isfinite(c_logits.grad).all())
        self.assertTrue(torch.isfinite(a_logits.grad).all())

    def test_invalid_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 1"):
            select_c_branch(
                torch.zeros(1, 3), torch.tensor([0]), dual_lambda=0.0
            )
        with self.assertRaisesRegex(ValueError, "between 1"):
            select_c_branch(
                torch.zeros(1, 3), torch.tensor([4]), dual_lambda=0.0
            )

    def test_dual_update_is_projected(self):
        self.assertEqual(
            update_dual_lambda(0.1, torch.tensor(-2.0), 1.0), 0.0
        )
        self.assertAlmostEqual(
            update_dual_lambda(0.1, torch.tensor(2.0), 1.0), 2.1
        )


if __name__ == "__main__":
    unittest.main()
