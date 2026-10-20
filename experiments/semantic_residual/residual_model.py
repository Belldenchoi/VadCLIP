"""Add residual logits after the unchanged original CLIPVAD forward."""

import copy
import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[2] / 'src'
sys.path.insert(0, str(SRC_ROOT))
from model import CLIPVAD

from .residual_branch import SemanticResidualBranch


class SemanticResidualCLIPVAD(CLIPVAD):
    def __init__(self, *args, residual_config, class_names, **kwargs):
        super().__init__(*args, **kwargs)
        if len(class_names) != self.num_class:
            raise ValueError('class_names must match the model class count')
        self.semantic_residual_config = copy.deepcopy(residual_config)
        self.semantic_residual_config['class_names'] = list(class_names)
        self.residual_branch = None
        if residual_config['branch'] != 'none':
            # Preserve the global RNG after base initialization so adding R
            # does not change initial base weights or data-loader shuffling.
            with torch.random.fork_rng(devices=[]):
                self.residual_branch = SemanticResidualBranch(
                    self.visual_width, self.num_class,
                    residual_config['hidden_dim'], residual_config['kernel_size'],
                    residual_config['branch'])
        if residual_config['freeze_lgt']:
            for name in ('temporal', 'gc1', 'gc2', 'gc3', 'gc4', 'disAdj',
                         'linear', 'frame_position_embeddings'):
                getattr(self, name).requires_grad_(False)
        self._last_residual = None

    def get_extra_state(self):
        # Included in both trainer checkpoints and exported model state dicts.
        return copy.deepcopy(self.semantic_residual_config)

    def set_extra_state(self, state):
        if state != self.semantic_residual_config:
            raise ValueError('Semantic residual configuration differs from checkpoint: '
                             f'saved={state}, requested={self.semantic_residual_config}')

    def forward(self, visual, padding_mask, text, lengths,
                return_visual_features=False, return_c_features=False):
        if list(text) != self.semantic_residual_config['class_names']:
            raise ValueError('Prompt label order differs from the saved residual class order')
        if self.residual_branch is None:
            return super().forward(visual, padding_mask, text, lengths,
                                   return_visual_features=return_visual_features,
                                   return_c_features=return_c_features)
        if return_c_features:
            raise ValueError('C PCA features are outside the isolated residual experiment')
        # This computes C0, the original C0 visual prompt, and A0 exactly once.
        labels, c0, a0, shared = super().forward(
            visual, padding_mask, text, lengths, return_visual_features=True)
        dc, da = self.residual_branch(shared, labels, lengths)
        cfg = self.semantic_residual_config
        c_final, a_final = c0, a0
        if cfg['branch'] in ('c', 'both'):
            c_final = c0 + cfg['c_scale'] * dc.to(c0.dtype)
        if cfg['branch'] in ('a', 'both'):
            a_final = a0 + cfg['a_scale'] * da.to(a0.dtype)
        self._last_residual = (
            None if dc is None else dc.detach(), da.detach(), lengths)
        result = (labels, c_final, a_final)
        return (*result, shared) if return_visual_features else result

    def residual_log_text(self):
        if self._last_residual is None:
            return 'residual_active=False '
        dc, da, lengths = self._last_residual
        if lengths is None:
            valid = torch.ones(da.shape[:2], dtype=torch.bool, device=da.device)
        else:
            valid = torch.arange(da.shape[1], device=da.device)[None, :] < lengths.to(da.device)[:, None]
        cfg = self.semantic_residual_config
        parts = ['residual_active=True']
        for name, delta, scale in (('a', da, cfg['a_scale']), ('c', dc, cfg['c_scale'])):
            if name in cfg['branch'] or cfg['branch'] == 'both':
                selected = delta[valid].float().abs()
                mean = selected.mean().item() if selected.numel() else 0.0
                parts.append(f'residual_{name}_weighted_abs_mean={scale * mean:.6f}')
        return ' '.join(parts) + ' '
