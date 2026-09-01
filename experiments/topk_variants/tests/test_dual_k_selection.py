import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dual_k_selection import (  # noqa: E402
    select_a_branch,
    select_c_branch,
    update_dual_lambda,
)


def test_c_selector_masks_padding_and_reports_effective_support():
    selection = select_c_branch(
        torch.zeros(2, 4, 1), torch.tensor([4, 2]), dual_lambda=0.0
    )
    assert selection.weights.shape == (2, 4)
    assert torch.all(selection.weights[1, 2:] == 0)
    assert torch.allclose(selection.effective_k, selection.weights.sum(dim=1))


def test_a_selector_is_classwise_and_does_not_need_ground_truth():
    logits = torch.tensor([
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    ])
    selection = select_a_branch(
        logits, torch.tensor([2]), dual_lambda=0.0
    )
    assert selection.weights.shape == logits.shape
    assert selection.effective_k.shape == (1, 3)
    assert torch.all(selection.weights[0, 2] == 0)


def test_dual_update_is_projected():
    assert update_dual_lambda(0.1, torch.tensor(-2.0), 1.0) == 0.0
    assert update_dual_lambda(0.1, torch.tensor(2.0), 1.0) == 2.1

