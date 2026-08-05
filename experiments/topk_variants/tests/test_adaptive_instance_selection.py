import sys
import unittest
from pathlib import Path

import torch


VARIANT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VARIANT_ROOT))

from adaptive_instance_selection import select_adaptive_instance_k


class AdaptiveInstanceSelectionTests(unittest.TestCase):
    def test_smooth_confident_pair_selects_adaptive_k(self):
        negative = torch.full((1, 4), 0.1)
        positive = torch.full((1, 4), 0.95)
        lengths = torch.tensor([4])

        result = select_adaptive_instance_k(
            negative, positive, lengths, lengths
        )

        # omega=0.9, confident_count=4, floor(3.6)=3
        self.assertEqual(result.pair_k.tolist(), [3])
        self.assertEqual(result.batch_k.tolist(), [3, 3])
        torch.testing.assert_close(
            result.confidence, torch.tensor([0.9])
        )

    def test_k_is_capped_by_shorter_valid_length(self):
        negative = torch.zeros(1, 4)
        positive = torch.ones(1, 4)

        result = select_adaptive_instance_k(
            negative,
            positive,
            negative_lengths=torch.tensor([3]),
            positive_lengths=torch.tensor([4]),
        )

        self.assertEqual(result.pair_k.tolist(), [3])

    def test_no_confident_positive_falls_back_to_one(self):
        negative = torch.zeros(1, 4)
        positive = torch.full((1, 4), 0.5)
        lengths = torch.tensor([4])

        result = select_adaptive_instance_k(
            negative, positive, lengths, lengths
        )

        self.assertEqual(result.confident_positive_count.tolist(), [0])
        self.assertEqual(result.batch_k.tolist(), [1, 1])

    def test_omega_uses_temporal_continuity_and_ignores_padding(self):
        negative = torch.tensor([[0.0, 0.2, 0.0, 1.0]])
        positive = torch.tensor([[1.0, 0.5, 1.0, 0.0]])
        lengths = torch.tensor([3])

        result = select_adaptive_instance_k(
            negative, positive, lengths, lengths
        )

        # mean(Sn)=1/15, mean(|dSn|)=0.2, mean(|dSp|)=0.5.
        expected_omega = 1.0 - (1.0 / 15.0) - 0.5 * (0.2 + 0.5)
        torch.testing.assert_close(
            result.confidence, torch.tensor([expected_omega])
        )

    def test_batch_order_matches_normal_then_anomaly_concatenation(self):
        negative = torch.tensor([
            [0.1, 0.1, 0.1, 0.1],
            [0.8, 0.8, 0.8, 0.8],
        ])
        positive = torch.full((2, 4), 0.95)
        lengths = torch.tensor([4, 4])

        result = select_adaptive_instance_k(
            negative, positive, lengths, lengths
        )

        self.assertEqual(result.pair_k.tolist(), [3, 1])
        self.assertEqual(result.batch_k.tolist(), [3, 1, 3, 1])


if __name__ == "__main__":
    unittest.main()
