"""Validity masks for cached temporal feature batches."""

import torch


def valid_steps(features, lengths):
    if features.ndim != 3:
        raise ValueError("Expected features [B,T,D]")
    lengths = torch.as_tensor(lengths, device=features.device, dtype=torch.long)
    if lengths.shape != (features.shape[0],) or bool(
        ((lengths < 0) | (lengths > features.shape[1])).any()
    ):
        raise ValueError("Lengths must match the batch and stay within [0,T]")
    return torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]
