import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.soft_topk import (
    multi_k_pooled_scores,
    multi_k_scale_sizes,
    soft_topk_pool,
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


if __name__ == "__main__":
    unittest.main()
