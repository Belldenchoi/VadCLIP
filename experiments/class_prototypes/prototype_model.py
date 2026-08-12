"""VadCLIP variant using direct cosine with fixed class prototypes."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_ROOT))

from model import CLIPVAD


class PrototypeCLIPVAD(CLIPVAD):
    uses_fixed_prototypes = True

    def __init__(self, *args, prototype_path, expected_class_names,
                 logit_temperature=0.07, **kwargs):
        super().__init__(*args, **kwargs)
        cache = torch.load(
            prototype_path, map_location="cpu", weights_only=True
        )
        class_names = cache.get("class_names")
        if class_names != list(expected_class_names):
            raise ValueError(
                "prototype class order mismatch: "
                f"cache={class_names}, expected={list(expected_class_names)}"
            )
        prototypes = cache.get("prototypes")
        if prototypes.ndim != 2 or prototypes.shape[0] != len(class_names):
            raise ValueError("invalid prototype tensor shape")
        if prototypes.shape[1] != self.visual_width:
            raise ValueError(
                f"prototype dim {prototypes.shape[1]} does not match "
                f"visual width {self.visual_width}"
            )
        if logit_temperature <= 0:
            raise ValueError("logit temperature must be positive")
        self.logit_temperature = float(logit_temperature)
        self.register_buffer(
            "class_prototypes", F.normalize(prototypes.float(), dim=-1)
        )
        # Text is already encoded offline. Removing this frozen copy releases
        # VRAM; learned prompt embeddings also cannot enter the forward path.
        del self.clipmodel
        del self.text_prompt_embeddings

    def forward(self, visual, padding_mask, text, lengths):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(
            visual_features + self.mlp2(visual_features)
        )
        visual_features_norm = F.normalize(visual_features, dim=-1)
        prototype_norm = F.normalize(self.class_prototypes, dim=-1)
        logits2 = (
            visual_features_norm
            @ prototype_norm.T.to(visual_features_norm.dtype)
            / self.logit_temperature
        )
        return self.class_prototypes, logits1, logits2
