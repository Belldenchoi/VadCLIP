"""Branch-specific soft instance selection for the Dual-K experiment.

The selectors are deliberately kept outside the model.  They operate on the
current batch scores and return continuous weights, effective support sizes,
and uncertainty-constraint diagnostics.  Padding is masked using ``lengths``.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DualKSelection:
    """Selection weights and diagnostics for one branch."""

    weights: Tensor
    effective_k: Tensor
    uncertainty: Tensor
    uncertainty_ratio: Tensor


def _valid_mask(lengths: Tensor, max_length: int) -> Tensor:
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.to(device=positions.device).unsqueeze(1)


def _binary_entropy(probabilities: Tensor, eps: float = 1e-6) -> Tensor:
    probabilities = probabilities.clamp(eps, 1.0 - eps)
    entropy = -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    )
    return entropy / torch.log(probabilities.new_tensor(2.0))


def _safe_ratio(weights: Tensor, uncertainty: Tensor, mask: Tensor,
                eps: float = 1e-6) -> Tensor:
    masked_weights = weights * mask.to(dtype=weights.dtype)
    masked_uncertainty = uncertainty * mask.to(dtype=uncertainty.dtype)
    return (
        (masked_weights * masked_uncertainty).sum(dim=1)
        / (masked_weights.sum(dim=1) + eps)
    )


def select_c_branch(logits: Tensor, lengths: Tensor, dual_lambda: float,
                    threshold: float = 0.5, temperature: float = 0.15,
                    eps: float = 1e-6) -> DualKSelection:
    """Select coarse anomaly evidence with a binary-entropy penalty."""
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    if logits.ndim != 2:
        raise ValueError("C-branch logits must have shape [B, T] or [B, T, 1]")
    if lengths.ndim != 1 or lengths.numel() != logits.shape[0]:
        raise ValueError("lengths must have shape [B]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    probabilities = torch.sigmoid(logits.float())
    uncertainty = _binary_entropy(probabilities, eps)
    mask = _valid_mask(lengths, logits.shape[1])
    values = (
        (probabilities - float(dual_lambda) * uncertainty - threshold)
        / temperature
    )
    weights = torch.sigmoid(values) * mask.to(dtype=values.dtype)
    effective_k = weights.sum(dim=1)
    ratio = _safe_ratio(weights, uncertainty, mask, eps)
    return DualKSelection(weights, effective_k, uncertainty, ratio)


def select_a_branch(logits: Tensor, lengths: Tensor, dual_lambda: float,
                    threshold: float = 0.25, temperature: float = 0.15,
                    eps: float = 1e-6) -> DualKSelection:
    """Select semantic evidence independently for every class.

    Evidence is the class posterior rather than the raw ``logits2`` value.
    VadCLIP divides these logits by 0.07, so using the raw scale makes a
    sigmoid selector almost binary and poorly calibrated.  Uncertainty is a
    class-wise margin uncertainty and does not require the ground-truth class,
    which keeps the same selector valid at inference time.
    """
    if logits.ndim != 3:
        raise ValueError("A-branch logits must have shape [B, T, C]")
    if lengths.ndim != 1 or lengths.numel() != logits.shape[0]:
        raise ValueError("lengths must have shape [B]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if logits.shape[-1] < 2:
        raise ValueError("A-branch requires at least two classes")

    probabilities = F.softmax(logits.float(), dim=-1)
    top_two = probabilities.topk(k=2, dim=-1).values
    strongest_other = top_two[..., 0:1].expand_as(probabilities)
    # For the winning class, use the second-highest class as its competitor;
    # for all other classes, the top class is the strongest competitor.
    winner = probabilities.argmax(dim=-1, keepdim=True)
    winner_competitor = top_two[..., 1:2]
    strongest_other = torch.where(
        torch.arange(probabilities.shape[-1], device=logits.device)
        .view(1, 1, -1).eq(winner),
        winner_competitor.expand_as(probabilities),
        strongest_other,
    )
    margin = probabilities - strongest_other
    uncertainty = 1.0 - ((margin + 1.0) * 0.5).clamp(0.0, 1.0)
    mask = _valid_mask(lengths, logits.shape[1])
    values = (
        (probabilities - float(dual_lambda) * uncertainty - threshold)
        / temperature
    )
    weights = torch.sigmoid(values) * mask.unsqueeze(-1).to(dtype=values.dtype)
    effective_k = weights.sum(dim=1)
    ratio = _safe_ratio(
        weights.mean(dim=-1), uncertainty.mean(dim=-1), mask, eps
    )
    return DualKSelection(weights, effective_k, uncertainty, ratio)


def dual_constraint_loss(selection: DualKSelection, budget: float) -> Tensor:
    """Return the differentiable uncertainty constraint term for a branch."""
    return (selection.uncertainty_ratio - float(budget)).mean()


def update_dual_lambda(dual_lambda: float, violation: Tensor,
                       learning_rate: float, maximum: float = 100.0) -> float:
    """Projected ascent update for a scalar dual variable."""
    updated = dual_lambda + learning_rate * float(violation.detach().item())
    return max(0.0, min(float(maximum), updated))

