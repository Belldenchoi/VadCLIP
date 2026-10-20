"""Pairwise video-label features and class-independent temporal filtering."""

import torch
from torch import nn
import torch.nn.functional as F


class SemanticResidualBranch(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim=64, kernel_size=3,
                 branch='both'):
        super().__init__()
        if branch not in ('a', 'c', 'both'):
            raise ValueError('branch must be a, c or both')
        if min(feature_dim, hidden_dim) < 1 or num_classes < 2:
            raise ValueError('Positive dimensions and at least two classes are required')
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be positive and odd')
        self.num_classes, self.hidden_dim = num_classes, hidden_dim
        self.visual_proj = nn.Linear(feature_dim, hidden_dim)
        self.text_proj = nn.Linear(feature_dim, hidden_dim)
        self.interaction = nn.Linear(3 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.temporal = (nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                                   padding=kernel_size // 2, groups=hidden_dim)
                         if kernel_size > 1 else None)
        # Zero output starts at exactly the original logits, while the small
        # fixed scales allow the output head to learn on the first update.
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.score.weight)
        self.c_readout = nn.Linear(num_classes, 1) if branch in ('c', 'both') else None
        if self.c_readout is not None:
            with torch.no_grad():
                self.c_readout.weight.fill_(1.0 / (num_classes - 1))
                self.c_readout.weight[0, 0] = -1.0
                self.c_readout.bias.zero_()

    def forward(self, visual, labels, lengths=None):
        if visual.ndim != 3 or labels.ndim != 2:
            raise ValueError('Expected visual [B,T,D] and labels [K,D]')
        batch, steps, width = visual.shape
        if steps < 1 or labels.shape != (self.num_classes, width):
            raise ValueError('Video/text feature shapes or class count do not match')
        if lengths is None:
            lengths = torch.full((batch,), steps, device=visual.device, dtype=torch.long)
        lengths = torch.as_tensor(lengths, device=visual.device)
        # Zero-length chunks are emitted by the original evaluator for exact
        # multiples of visual_length. Their residual must be entirely zero.
        if (lengths.shape != (batch,) or lengths.is_floating_point() or
                lengths.dtype == torch.bool or
                bool(((lengths < 0) | (lengths > steps)).any())):
            raise ValueError('lengths must contain B integers between 0 and T')
        valid = torch.arange(steps, device=visual.device)[None, :] < lengths[:, None]
        clean = visual.masked_fill(~valid[..., None], 0)
        v = self.visual_proj(F.normalize(clean.float(), dim=-1))
        text = self.text_proj(F.normalize(labels.float(), dim=-1))
        # Equivalent to Linear(cat(v, text, v*text)), without materializing
        # the large [B,T,K,3H] concatenation or repeating its first two terms.
        wv, wt, wi = self.interaction.weight.split(self.hidden_dim, dim=1)
        pairs = (F.linear(v, wv)[:, :, None, :] +
                 F.linear(text, wt)[None, None, :, :] +
                 F.linear(v[:, :, None, :] * text[None, None, :, :],
                          wi, self.interaction.bias))
        pairs = self.norm(F.gelu(pairs))
        pairs = pairs.masked_fill(~valid[:, :, None, None], 0)
        if self.temporal is not None:
            # Classes form independent sequences; the SAME filter is shared
            # across classes. There is no convolution over the label order.
            sequences = pairs.permute(0, 2, 3, 1).reshape(
                batch * self.num_classes, self.hidden_dim, steps)
            filtered = self.temporal(sequences).reshape(
                batch, self.num_classes, self.hidden_dim, steps).permute(0, 3, 1, 2)
            pairs = (pairs + F.gelu(filtered)).masked_fill(
                ~valid[:, :, None, None], 0)
        class_scores = self.score(pairs).squeeze(-1).masked_fill(~valid[..., None], 0)
        c_scores = (self.c_readout(class_scores).masked_fill(~valid[..., None], 0)
                    if self.c_readout is not None else None)
        return c_scores, class_scores
