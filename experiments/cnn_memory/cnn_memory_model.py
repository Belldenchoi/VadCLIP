"""Single-Conv gated residual feature fusion for C; original A path."""

import torch
import torch.nn.functional as F

try:
    from . import bootstrap
    from .feature_fusion import CNNFeatureEnhancement
    from .feature_utils import valid_steps
except ImportError:
    import bootstrap
    from feature_fusion import CNNFeatureEnhancement
    from feature_utils import valid_steps

from model import CLIPVAD


class CNNMemoryCLIPVAD(CLIPVAD):
    LGT_MODULES = ("temporal", "gc1", "gc2", "gc3", "gc4", "disAdj", "linear",
                   "frame_position_embeddings")

    def __init__(self, *args, aux_branch="c", cnn_kernel=3,
                 aux_mode="gated_residual", input_features="post_lgt", cnn_channels=128, **kwargs):
        super().__init__(*args, **kwargs)
        if input_features not in ("clip", "post_lgt"):
            raise ValueError("Unknown input feature kind")
        self.input_features = input_features
        if input_features == "post_lgt":
            # Retain state-dict compatibility, but never optimize the unused LGT.
            for name in self.LGT_MODULES:
                getattr(self, name).requires_grad_(False)
        if aux_mode != "gated_residual":
            raise ValueError("Only gated_residual is supported; previous CNN methods have been removed")
        if aux_branch not in ("c", "none") or cnn_kernel != 3:
            raise ValueError("gated_residual baseline v1 supports C-only fusion with kernel 3")
        self.aux_branch = aux_branch
        self.register_buffer("enhancement_active", torch.tensor(False))
        self.cnn_enhancement = None
        if aux_branch == "c":
            # Match backbone initialization and subsequent loader RNG to E0.
            with torch.random.fork_rng(devices=[]):
                self.cnn_enhancement = CNNFeatureEnhancement(self.visual_width, cnn_channels)

    def set_enhancement_active(self, active):
        self.enhancement_active.fill_(bool(active) and self.cnn_enhancement is not None)

    def forward(self, visual, padding_mask, text, lengths, return_aux=False,
                return_shared_features=False):
        if self.input_features == "post_lgt":
            if visual.ndim != 3 or visual.shape[-1] != self.visual_width:
                raise ValueError(f"Expected cached LGT features [B,T,{self.visual_width}]")
            valid = valid_steps(visual, lengths)
            features = visual.float().masked_fill(~valid[..., None], 0)
        else:
            # Original online LGT for unencoded CLIP inputs.
            features = self.encode_video(visual, padding_mask, lengths)
        logits_c = self.classifier(features + self.mlp2(features))
        # A keeps its original C-derived aggregation weights and original H.
        # Reuse the SAME classifier/MLP for H'; there is no auxiliary head.
        visual_prompt_logits = logits_c
        if self.cnn_enhancement is not None and bool(self.enhancement_active):
            enhanced = self.cnn_enhancement(features, lengths)
            logits_c = self.classifier(enhanced + self.mlp2(enhanced))
        text_original = self.encode_textprompt(text)
        visual_attention = visual_prompt_logits.permute(0, 2, 1) @ features
        if self.input_features == "post_lgt":
            # Cached padding and all-padding test chunks can be exactly zero.
            visual_attention = F.normalize(visual_attention, dim=-1, eps=1e-6)
        else:
            visual_attention = visual_attention / visual_attention.norm(dim=-1, keepdim=True)
        text_features = text_original.unsqueeze(0) + visual_attention.expand(
            visual_attention.shape[0], text_original.shape[0], visual_attention.shape[2]
        )
        text_features = text_features + self.mlp1(text_features)
        if self.input_features == "post_lgt":
            normalized_video = F.normalize(features, dim=-1, eps=1e-6)
            normalized_text = F.normalize(text_features, dim=-1, eps=1e-6)
        else:
            normalized_video = features / features.norm(dim=-1, keepdim=True)
            normalized_text = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_a = normalized_video @ normalized_text.permute(0, 2, 1).type(normalized_video.dtype) / 0.07

        outputs = (text_original, logits_c, logits_a)
        if return_shared_features:
            return (*outputs, None, features)
        return (*outputs, None) if return_aux else outputs


def build_model(config, device):
    return CNNMemoryCLIPVAD(
        14, 512, config["visual_length"], 512, 1, config["visual_layers"],
        config["attn_window"], config["prompt_prefix"], config["prompt_postfix"], device,
        aux_branch=config["aux_branch"], cnn_kernel=config["cnn_kernel"],
        aux_mode=config["aux_mode"], input_features=config["input_features"],
        cnn_channels=config["cnn_channels"],
    )
