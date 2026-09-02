"""Branch-specific soft instance selection for the Dual-K experiment.

The selectors are deliberately kept outside the model. They operate on the
current batch scores and return continuous weights, effective support sizes,
and uncertainty-constraint diagnostics. Evidence is standardized within each
video and padding is excluded from every statistic.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DualKSelection:
    """Selection weights and diagnostics for one branch.

    C-branch tensors have shape ``[B, T]`` and per-video diagnostics have
    shape ``[B]``. A-branch tensors have shape ``[B, T, C]`` and per-class
    diagnostics have shape ``[B, C]``.
    """

    weights: Tensor
    effective_k: Tensor
    uncertainty: Tensor
    uncertainty_ratio: Tensor
    evidence: Tensor
    normalized_evidence: Tensor
    support_ratio: Tensor
    valid_mask: Tensor


def _valid_mask(lengths: Tensor, max_length: int) -> Tensor:
    if lengths.ndim != 1:
        raise ValueError("lengths must have shape [B]")
    if torch.any(lengths < 1) or torch.any(lengths > max_length):
        raise ValueError("lengths must be between 1 and the padded sequence length")
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.to(device=positions.device).unsqueeze(1)


def _binary_entropy(probabilities: Tensor, eps: float = 1e-6) -> Tensor:
    probabilities = probabilities.clamp(eps, 1.0 - eps)
    entropy = -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    )
    return entropy / torch.log(probabilities.new_tensor(2.0))


def _masked_standardize(evidence: Tensor, mask: Tensor,
                        eps: float = 1e-6) -> Tensor:
    """Z-score temporal evidence per video (and per class for A)."""
    expanded_mask = mask
    while expanded_mask.ndim < evidence.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.to(dtype=evidence.dtype)

    count = expanded_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (evidence * expanded_mask).sum(dim=1, keepdim=True) / count
    centered = (evidence - mean) * expanded_mask
    variance = centered.square().sum(dim=1, keepdim=True) / count
    standardized = centered / (variance.sqrt() + eps)
    return standardized * expanded_mask


def _weighted_uncertainty_ratio(weights: Tensor, uncertainty: Tensor,
                                eps: float = 1e-6) -> Tensor:
    return (
        (weights * uncertainty).sum(dim=1)
        / (weights.sum(dim=1) + eps)
    )


def _support_ratio(effective_k: Tensor, lengths: Tensor) -> Tensor:
    denominator = lengths.to(
        device=effective_k.device, dtype=effective_k.dtype
    )
    while denominator.ndim < effective_k.ndim:
        denominator = denominator.unsqueeze(-1)
    return effective_k / denominator


def _validate_selector_inputs(logits: Tensor, lengths: Tensor,
                              temperature: float, dual_lambda: float,
                              eps: float) -> Tensor:
    if lengths.ndim != 1 or lengths.numel() != logits.shape[0]:
        raise ValueError("lengths must have shape [B]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if dual_lambda < 0:
        raise ValueError("dual_lambda must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    return _valid_mask(lengths, logits.shape[1])


def select_c_branch(logits: Tensor, lengths: Tensor, dual_lambda: float,
                    threshold: float = 0.5, temperature: float = 0.15,
                    eps: float = 1e-6,
                    normalize_evidence: bool = True) -> DualKSelection:
    """Select normalized coarse evidence with normalized entropy risk."""
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    if logits.ndim != 2:
        raise ValueError("C-branch logits must have shape [B, T] or [B, T, 1]")
    mask = _validate_selector_inputs(
        logits, lengths, temperature, dual_lambda, eps
    )

    evidence = torch.sigmoid(logits.float())
    normalized_evidence = _masked_standardize(evidence, mask, eps)
    uncertainty = _binary_entropy(evidence, eps)
    selector_evidence = normalized_evidence if normalize_evidence else evidence
    values = (
        selector_evidence
        - float(dual_lambda) * uncertainty
        - float(threshold)
    ) / float(temperature)
    weights = torch.sigmoid(values) * mask.to(dtype=values.dtype)
    effective_k = weights.sum(dim=1)
    uncertainty_ratio = _weighted_uncertainty_ratio(
        weights, uncertainty, eps
    )
    return DualKSelection(
        weights=weights,
        effective_k=effective_k,
        uncertainty=uncertainty,
        uncertainty_ratio=uncertainty_ratio,
        evidence=evidence,
        normalized_evidence=normalized_evidence,
        support_ratio=_support_ratio(effective_k, lengths),
        valid_mask=mask,
    )


def select_a_branch(logits: Tensor, lengths: Tensor, dual_lambda: float,
                    threshold: float = 0.25, temperature: float = 0.15,
                    eps: float = 1e-6,
                    normalize_evidence: bool = True) -> DualKSelection:
    """Select standardized semantic evidence independently for every class.

    The gate uses each class's raw ``logits2`` evidence standardized over the
    valid snippets of the same video. Its uncertainty is
    ``clamp(1 - (p_c - max_{j != c} p_j), 0, 1)``. This generalizes the
    ground-truth-class formulation while keeping weighted MIL class-wise.
    """
    if logits.ndim != 3:
        raise ValueError("A-branch logits must have shape [B, T, C]")
    if logits.shape[-1] < 2:
        raise ValueError("A-branch requires at least two classes")
    mask = _validate_selector_inputs(
        logits, lengths, temperature, dual_lambda, eps
    )

    evidence = logits.float()
    normalized_evidence = _masked_standardize(evidence, mask, eps)
    probabilities = F.softmax(evidence, dim=-1)
    top_two = probabilities.topk(k=2, dim=-1).values
    winner = probabilities.argmax(dim=-1, keepdim=True)
    strongest_other = top_two[..., 0:1].expand_as(probabilities)
    strongest_other = torch.where(
        torch.arange(probabilities.shape[-1], device=logits.device)
        .view(1, 1, -1).eq(winner),
        top_two[..., 1:2].expand_as(probabilities),
        strongest_other,
    )
    margin = probabilities - strongest_other
    uncertainty = (1.0 - margin).clamp(0.0, 1.0)
    # The unnormalized path intentionally reproduces the original
    # implementation's posterior gate for the scale-diagnostic ablation.
    selector_evidence = (
        normalized_evidence if normalize_evidence else probabilities
    )

    values = (
        selector_evidence
        - float(dual_lambda) * uncertainty
        - float(threshold)
    ) / float(temperature)
    expanded_mask = mask.unsqueeze(-1).to(dtype=values.dtype)
    weights = torch.sigmoid(values) * expanded_mask
    effective_k = weights.sum(dim=1)
    uncertainty_ratio = _weighted_uncertainty_ratio(
        weights, uncertainty, eps
    )
    return DualKSelection(
        weights=weights,
        effective_k=effective_k,
        uncertainty=uncertainty,
        uncertainty_ratio=uncertainty_ratio,
        evidence=evidence,
        normalized_evidence=normalized_evidence,
        support_ratio=_support_ratio(effective_k, lengths),
        valid_mask=mask,
    )


def _normalized_class_weights(class_weights: Tensor,
                              expected_shape: torch.Size,
                              eps: float = 1e-6) -> Tensor:
    if class_weights.shape != expected_shape:
        raise ValueError(
            f"class_weights must have shape {tuple(expected_shape)}"
        )
    class_weights = class_weights.float().clamp_min(0.0)
    denominator = class_weights.sum(dim=-1, keepdim=True)
    if torch.any(denominator <= 0):
        raise ValueError("each class_weights row must have positive mass")
    return class_weights / denominator.clamp_min(eps)


def target_class_mean(values: Tensor, class_weights: Tensor) -> Tensor:
    """Average ``[B, C]`` diagnostics over target classes and the batch."""
    if values.ndim != 2:
        raise ValueError("target-class diagnostics must have shape [B, C]")
    normalized = _normalized_class_weights(
        class_weights.to(device=values.device), values.shape
    )
    return (values * normalized.to(dtype=values.dtype)).sum(dim=-1).mean()


def selection_statistics(selection: DualKSelection,
                         class_weights: Tensor | None = None) -> dict[str, Tensor]:
    """Summarize valid evidence, uncertainty, risk, and support.

    Passing video-level ``class_weights`` reports A-branch statistics only for
    the target class(es). Without them, all returned values cover the complete
    selection tensor.
    """
    evidence = selection.evidence
    normalized_evidence = selection.normalized_evidence
    uncertainty = selection.uncertainty
    effective_k = selection.effective_k
    support_ratio = selection.support_ratio
    uncertainty_ratio = selection.uncertainty_ratio

    if class_weights is not None:
        if evidence.ndim != 3:
            raise ValueError("class_weights are only valid for A-branch selection")
        normalized = _normalized_class_weights(
            class_weights.to(device=evidence.device), effective_k.shape
        ).to(dtype=evidence.dtype)
        temporal_weights = normalized.unsqueeze(1)
        evidence = (evidence * temporal_weights).sum(dim=-1)
        normalized_evidence = (
            normalized_evidence * temporal_weights
        ).sum(dim=-1)
        uncertainty = (uncertainty * temporal_weights).sum(dim=-1)
        effective_k = (effective_k * normalized).sum(dim=-1)
        support_ratio = (support_ratio * normalized).sum(dim=-1)
        uncertainty_ratio = (
            uncertainty_ratio * normalized
        ).sum(dim=-1)

    mask = selection.valid_mask
    valid_evidence = evidence[mask]
    valid_normalized_evidence = normalized_evidence[mask]
    valid_uncertainty = uncertainty[mask]
    risk_values = uncertainty_ratio.reshape(-1)
    return {
        "e_mean": valid_evidence.mean(),
        "e_std": valid_evidence.std(unbiased=False),
        "e_min": valid_evidence.min(),
        "e_max": valid_evidence.max(),
        "e_norm_mean": valid_normalized_evidence.mean(),
        "e_norm_std": valid_normalized_evidence.std(unbiased=False),
        "u_mean": valid_uncertainty.mean(),
        "u_std": valid_uncertainty.std(unbiased=False),
        "k_mean": effective_k.mean(),
        "support_ratio": support_ratio.mean(),
        "risk": uncertainty_ratio.mean(),
        "risk_q40": torch.quantile(risk_values, 0.4),
        "risk_q50": torch.quantile(risk_values, 0.5),
        "risk_q60": torch.quantile(risk_values, 0.6),
    }


def dual_constraint_loss(selection: DualKSelection, budget: float,
                         class_weights: Tensor | None = None) -> Tensor:
    """Return the differentiable mean uncertainty-constraint violation.

    For A-branch class-wise ratios, ``class_weights`` restricts the scalar
    dual update to the video-level target class(es), matching the formulation
    in the design document. C-branch ratios are averaged directly.
    """
    violation = selection.uncertainty_ratio - float(budget)
    if class_weights is None:
        return violation.mean()
    return target_class_mean(violation, class_weights)


def update_dual_lambda(dual_lambda: float, violation: Tensor,
                       learning_rate: float, maximum: float = 100.0) -> float:
    """Projected ascent update for a scalar dual variable."""
    if learning_rate < 0:
        raise ValueError("learning_rate must be non-negative")
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    updated = dual_lambda + learning_rate * float(violation.detach().item())
    return max(0.0, min(float(maximum), updated))
