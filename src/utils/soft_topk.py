"""Differentiable aggregation for the frame scores selected by Top-K."""

import math

import torch
from torch import Tensor


def multi_k_scale_sizes(length: int,
                        percentages=(1.0, 5.0, 10.0, 20.0)):
    """Return percentage-based K values while preserving scale identity."""
    if length <= 0:
        raise ValueError(f"video length must be positive, got {length}")
    if any(percentage <= 0 or percentage > 100
           for percentage in percentages):
        raise ValueError("multi-K percentages must be in the range (0, 100]")
    sizes = [
        min(length, max(1, math.ceil(length * percentage / 100.0)))
        for percentage in percentages
    ]
    return sizes


def multi_k_sizes(length: int, percentages=(1.0, 5.0, 10.0, 20.0)):
    """Return unique percentage-based K values for a valid video length."""
    return list(dict.fromkeys(multi_k_scale_sizes(length, percentages)))


def multi_k_pooled_scores(scores: Tensor,
                          percentages=(1.0, 5.0, 10.0, 20.0),
                          preserve_scales: bool = False):
    """Return per-scale arithmetic means and their K values."""
    if scores.ndim not in (1, 2):
        raise ValueError(
            f"scores must have shape [T] or [T, C], got {scores.shape}"
        )
    if scores.shape[0] == 0:
        raise ValueError("cannot pool an empty sequence")

    size_fn = multi_k_scale_sizes if preserve_scales else multi_k_sizes
    sizes = size_fn(scores.shape[0], percentages)
    pooled_scales = []
    for scale_k in sizes:
        scale_scores = torch.topk(
            scores, k=scale_k, dim=0, largest=True
        ).values
        pooled_scales.append(scale_scores.mean(dim=0))
    return torch.stack(pooled_scales, dim=0), sizes


def soft_topk_pool(scores: Tensor, k: int, temperature: float = 1.0) -> Tensor:
    """Pool the K highest temporal scores with score-derived soft weights.

    ``scores`` can have shape ``[T]`` (C-branch) or ``[T, C]``
    (A-branch). For the latter, Top-K selection and weighting are performed
    independently for every class. A lower temperature concentrates more
    weight on the highest-scoring frames.
    """
    if scores.ndim not in (1, 2):
        raise ValueError(f"scores must have shape [T] or [T, C], got {scores.shape}")
    if scores.shape[0] == 0:
        raise ValueError("cannot pool an empty sequence")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    k = max(1, min(int(k), scores.shape[0]))
    topk_scores = torch.topk(scores, k=k, dim=0, largest=True).values
    frame_weights = torch.softmax(topk_scores / temperature, dim=0)
    return torch.sum(frame_weights * topk_scores, dim=0)


def topk_pool(scores: Tensor, k: int, mode: str = "soft",
              temperature: float = 1.0,
              multi_k_percentages=(1.0, 5.0, 10.0, 20.0)) -> Tensor:
    """Apply either VadCLIP's original mean or score-weighted Top-K pooling."""
    if mode == "soft":
        return soft_topk_pool(scores, k, temperature)
    if mode == "mean":
        k = max(1, min(int(k), scores.shape[0]))
        return torch.topk(scores, k=k, dim=0, largest=True).values.mean(dim=0)
    if mode == "multi_k":
        pooled_scales, _ = multi_k_pooled_scores(
            scores, multi_k_percentages
        )
        return pooled_scales.mean(dim=0)
    raise ValueError(f"unsupported Top-K pooling mode: {mode}")


def video_topk_size(length: int, divisor: int = 16) -> int:
    """Return VadCLIP's original length-dependent number of selected frames."""
    if length <= 0:
        raise ValueError(f"video length must be positive, got {length}")
    return min(length, length // divisor + 1)
