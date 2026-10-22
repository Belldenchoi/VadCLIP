"""Add logit residuals without changing the original LGT or Visual Prompt."""

import torch
import torch.nn.functional as F

try:
    from . import bootstrap
    from .cnn_memory import CNNVisualMemory
except ImportError:
    import bootstrap
    from cnn_memory import CNNVisualMemory

from model import CLIPVAD


class CNNMemoryCLIPVAD(CLIPVAD):
    def __init__(self, *args, aux_branch="both", cnn_kernel=3, feature_dim=256,
                 memory_slots=8, memory_temperature=0.1, residual_scale=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if aux_branch not in ("none", "c", "a", "both"):
            raise ValueError("Unknown auxiliary branch")
        self.aux_branch = aux_branch
        self.residual_scale = residual_scale
        self.register_buffer("memory_active", torch.tensor(False))
        self.memory = None
        if aux_branch != "none":
            # Do not change the subsequent loader RNG relative to --aux-branch none.
            with torch.random.fork_rng(devices=[]):
                self.memory = CNNVisualMemory(
                    self.visual_width, feature_dim, self.num_class,
                    memory_slots, cnn_kernel, memory_temperature,
                )

    def set_memory_active(self, active):
        self.memory_active.fill_(bool(active) and self.memory is not None)

    def forward(self, visual, padding_mask, text, lengths, return_aux=False):
        # Same ordinary forward as CLIPVAD; no auxiliary interfaces in src required.
        features = self.encode_video(visual, padding_mask, lengths)
        logits_c = self.classifier(features + self.mlp2(features))
        text_original = self.encode_textprompt(text)
        visual_attention = logits_c.permute(0, 2, 1) @ features
        visual_attention = visual_attention / visual_attention.norm(dim=-1, keepdim=True)
        text_features = text_original.unsqueeze(0) + visual_attention.expand(
            visual_attention.shape[0], text_original.shape[0], visual_attention.shape[2]
        )
        text_features = text_features + self.mlp1(text_features)
        normalized_video = features / features.norm(dim=-1, keepdim=True)
        normalized_text = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_a = normalized_video @ normalized_text.permute(0, 2, 1).type(normalized_video.dtype) / 0.07

        auxiliary = None
        if self.memory is not None and bool(self.memory_active):
            auxiliary = self.memory(features, lengths)
            # Visual Prompt above always consumes the original C logits.
            if self.aux_branch in ("c", "both"):
                logits_c = logits_c + self.residual_scale * auxiliary["delta_c"]
            if self.aux_branch in ("a", "both"):
                logits_a = logits_a + self.residual_scale * auxiliary["delta_a"]
        outputs = (text_original, logits_c, logits_a)
        return (*outputs, auxiliary) if return_aux else outputs


def build_model(config, device):
    return CNNMemoryCLIPVAD(
        14, 512, config["visual_length"], 512, 1, config["visual_layers"],
        config["attn_window"], config["prompt_prefix"], config["prompt_postfix"], device,
        **{key: config[key] for key in (
            "aux_branch", "cnn_kernel", "feature_dim", "memory_slots",
            "memory_temperature", "residual_scale",
        )},
    )
