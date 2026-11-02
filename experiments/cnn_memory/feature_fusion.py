"""One temporal Conv1D and gated feature residual for the original C head."""

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .future_features import valid_steps
except ImportError:
    from future_features import valid_steps


class CNNFeatureEnhancement(nn.Module):
    """H' = H + sigmoid(Linear([H, P(GELU(Conv(H)))])) * P(...)."""

    def __init__(self, dim=512, cnn_channels=128):
        super().__init__()
        if min(dim, cnn_channels) < 1:
            raise ValueError("Feature dimension and CNN channels must be positive")
        self.conv = nn.Conv1d(dim, cnn_channels, 3, stride=1, padding=1, dilation=1)
        self.proj = nn.Linear(cnn_channels, dim)
        self.gate = nn.Linear(2 * dim, dim)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, features, lengths):
        valid = valid_steps(features, lengths)
        # Removing padding BEFORE convolution protects valid boundary snippets.
        original = features.float().masked_fill(~valid[..., None], 0)
        hidden = F.gelu(self.conv(original.transpose(1, 2))).transpose(1, 2)
        projected = self.proj(hidden).masked_fill(~valid[..., None], 0)
        gate = torch.sigmoid(self.gate(torch.cat((original, projected), dim=-1)))
        return (original + gate * projected).masked_fill(~valid[..., None], 0)
