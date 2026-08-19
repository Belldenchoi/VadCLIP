"""Temporal smoothness regularizers for VadCLIP C/A anomaly scores."""

import torch
from torch import Tensor


def temporal_total_variation(probabilities: Tensor, lengths: Tensor) -> Tensor:
    """Return mean L1 difference between adjacent valid temporal scores.

    ``probabilities`` must have shape ``[B, T]``. Each sample is averaged over
    its own valid adjacent pairs before averaging across the batch so long
    videos do not dominate short videos. Padding never contributes.
    """
    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape [B, T], "
            f"got {tuple(probabilities.shape)}"
        )
    if lengths.ndim != 1 or lengths.shape[0] != probabilities.shape[0]:
        raise ValueError(
            "lengths must have shape [B] matching probabilities"
        )

    sample_losses = []
    temporal_size = probabilities.shape[1]
    for sample_index in range(probabilities.shape[0]):
        length = int(lengths[sample_index].item())
        if length < 1 or length > temporal_size:
            raise ValueError(
                f"invalid valid length {length} for temporal size {temporal_size}"
            )
        if length == 1:
            continue
        valid_scores = probabilities[sample_index, :length]
        sample_losses.append(
            torch.diff(valid_scores, dim=0).abs().mean()
        )

    if not sample_losses:
        return probabilities.sum() * 0.0
    return torch.stack(sample_losses).mean()


def c_branch_temporal_smoothness(logits1: Tensor,
                                 lengths: Tensor) -> Tensor:
    """Smoothness of sigmoid C-branch anomaly probabilities."""
    if logits1.ndim != 3 or logits1.shape[-1] != 1:
        raise ValueError(
            "C-branch logits must have shape [B, T, 1], "
            f"got {tuple(logits1.shape)}"
        )
    probabilities = torch.sigmoid(logits1.float()).squeeze(-1)
    return temporal_total_variation(probabilities, lengths)


def a_branch_temporal_smoothness(logits2: Tensor, lengths: Tensor,
                                 normal_class_index: int = 0) -> Tensor:
    """Smoothness of A-branch anomaly probability ``1 - P(normal)``."""
    if logits2.ndim != 3:
        raise ValueError(
            "A-branch logits must have shape [B, T, C], "
            f"got {tuple(logits2.shape)}"
        )
    if not 0 <= normal_class_index < logits2.shape[-1]:
        raise ValueError("normal class index is outside A-branch classes")
    class_probabilities = torch.softmax(logits2.float(), dim=-1)
    anomaly_probabilities = (
        1.0 - class_probabilities[..., normal_class_index]
    )
    return temporal_total_variation(anomaly_probabilities, lengths)
