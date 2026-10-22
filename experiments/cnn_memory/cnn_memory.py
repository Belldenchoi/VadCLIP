"""Temporal CNN and learnable normal/abnormal visual prototypes."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class CNNVisualMemory(nn.Module):
    def __init__(self, input_dim=512, feature_dim=256, num_classes=14,
                 slots=8, kernel=3, temperature=0.1):
        super().__init__()
        if min(input_dim, feature_dim, num_classes, slots) < 1 or kernel not in (1, 3):
            raise ValueError("Positive dimensions and kernel 1 or 3 are required")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Memory temperature must be finite and positive")
        self.temperature = temperature
        self.conv1 = nn.Conv1d(input_dim, feature_dim, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(feature_dim, feature_dim, 1)
        # Bank 0 is normal; bank 1 is abnormal. Neither contains test samples.
        self.prototypes = nn.Parameter(torch.randn(2, slots, feature_dim) * 0.02)
        self.c_head = nn.Linear(feature_dim + 2 * slots, 1)
        self.a_head = nn.Linear(feature_dim + 2 * slots, num_classes)
        # The first active forward is identical to the unmodified backbone.
        for head in (self.c_head, self.a_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features, lengths):
        if features.ndim != 3:
            raise ValueError("Expected [batch, time, channels] features")
        lengths = torch.as_tensor(lengths, device=features.device, dtype=torch.long)
        if lengths.shape != (features.shape[0],) or bool(
            ((lengths < 0) | (lengths > features.shape[1])).any()
        ):
            raise ValueError("Lengths must match the batch and stay within [0, T]")
        valid = torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]
        # masked_fill, rather than multiplication, also removes NaNs in padding.
        features = features.float().masked_fill(~valid[..., None], 0)
        hidden = F.gelu(self.conv1(features.transpose(1, 2)))
        hidden = hidden.masked_fill(~valid[:, None, :], 0)
        z = self.conv2(hidden).transpose(1, 2)
        z = F.normalize(z, dim=-1, eps=1e-6).masked_fill(~valid[..., None], 0)
        prototypes = F.normalize(self.prototypes, dim=-1, eps=1e-6)
        similarities = torch.einsum("btd,nsd->btns", z, prototypes)
        # Smooth maximum over slots; equal slots count makes the banks comparable.
        bank_logits = torch.logsumexp(similarities / self.temperature, dim=-1)
        memory_logits = (bank_logits[..., 1] - bank_logits[..., 0]).unsqueeze(-1)
        matching = torch.cat((z, similarities.flatten(-2)), dim=-1)
        return {
            "delta_c": self.c_head(matching).masked_fill(~valid[..., None], 0),
            "delta_a": self.a_head(matching).masked_fill(~valid[..., None], 0),
            "memory_logits": memory_logits.masked_fill(~valid[..., None], 0),
            "features": z,
        }
