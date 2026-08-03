"""Adaptive Instance Selection (AIS) for paired UCF-Crime bags.

AIS is intentionally kept separate from ``soft_topk.py``. It uses C-branch
anomaly probabilities to choose one integer K for each Normal/anomaly pair.
That K can then be shared by the C- and A-branch MIL pooling losses.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AISSelection:
    """Adaptive K values and diagnostics for one paired batch."""

    pair_k: Tensor
    batch_k: Tensor
    confidence: Tensor
    confident_positive_count: Tensor


def _mean_absolute_delta(scores: Tensor) -> Tensor:
    if scores.numel() <= 1:
        return scores.new_zeros(())
    return torch.diff(scores).abs().mean()


@torch.no_grad()
def select_adaptive_instance_k(
        negative_scores: Tensor,
        positive_scores: Tensor,
        negative_lengths: Tensor,
        positive_lengths: Tensor,
        score_threshold: float = 0.9,
        min_k: int = 1) -> AISSelection:
    """Compute AIS K from paired C-branch anomaly probabilities.

    The paper assumes equal bag length T. VadCLIP retains a valid length for
    each padded video, so the continuity terms are normalized independently:

        omega = 1 - mean(Sn)
                  - 0.5 * (mean(|delta Sn|) + mean(|delta Sp|))
        K = floor(omega * count(Sp >= score_threshold))

    K is clamped to ``[min_k, min(Tn, Tp)]`` and copied to both members of the
    pair. Inputs must be ordered as Normal bags and their paired anomaly bags.
    """
    if negative_scores.ndim != 2 or positive_scores.ndim != 2:
        raise ValueError("AIS scores must have shape [B, T]")
    if negative_scores.shape[0] != positive_scores.shape[0]:
        raise ValueError("AIS requires the same number of negative/positive bags")
    if negative_lengths.ndim != 1 or positive_lengths.ndim != 1:
        raise ValueError("AIS lengths must have shape [B]")
    batch_size = negative_scores.shape[0]
    if (negative_lengths.numel() != batch_size or
            positive_lengths.numel() != batch_size):
        raise ValueError("AIS lengths must match the paired batch size")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("AIS score threshold must be in [0, 1]")
    if min_k < 1:
        raise ValueError("AIS min_k must be at least 1")

    negative_scores = negative_scores.detach().float()
    positive_scores = positive_scores.detach().float()
    pair_k = []
    confidence = []
    confident_counts = []

    for pair_index in range(batch_size):
        negative_length = int(negative_lengths[pair_index].item())
        positive_length = int(positive_lengths[pair_index].item())
        if negative_length <= 0 or positive_length <= 0:
            raise ValueError("AIS valid lengths must be positive")
        if (negative_length > negative_scores.shape[1] or
                positive_length > positive_scores.shape[1]):
            raise ValueError("AIS valid length exceeds the score sequence")

        negative = negative_scores[pair_index, :negative_length]
        positive = positive_scores[pair_index, :positive_length]

        omega = (
            1.0
            - negative.mean()
            - 0.5 * (
                _mean_absolute_delta(negative)
                + _mean_absolute_delta(positive)
            )
        ).clamp(0.0, 1.0)
        confident_count = (positive >= score_threshold).sum()
        raw_k = torch.floor(omega * confident_count.to(omega.dtype))

        maximum_k = min(negative_length, positive_length)
        selected_k = min(
            maximum_k,
            max(min_k, int(raw_k.item())),
        )
        pair_k.append(torch.tensor(
            selected_k, device=negative_scores.device, dtype=torch.long
        ))
        confidence.append(omega)
        confident_counts.append(confident_count.to(dtype=torch.long))

    pair_k = torch.stack(pair_k)
    return AISSelection(
        pair_k=pair_k,
        batch_k=torch.cat((pair_k, pair_k), dim=0),
        confidence=torch.stack(confidence),
        confident_positive_count=torch.stack(confident_counts),
    )
