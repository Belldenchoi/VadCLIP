"""C-head feature fusion or legacy logit residuals, with optional cached LGT inputs."""

import torch
import torch.nn.functional as F

try:
    from . import bootstrap
    from .cnn_memory import CNNVisualMemory
    from .feature_fusion import CNNFeatureEnhancement
    from .future_features import FutureFeatureAuxiliary, JointFutureFeatureAuxiliary, valid_steps
except ImportError:
    import bootstrap
    from cnn_memory import CNNVisualMemory
    from feature_fusion import CNNFeatureEnhancement
    from future_features import FutureFeatureAuxiliary, JointFutureFeatureAuxiliary, valid_steps

from model import CLIPVAD


class CNNMemoryCLIPVAD(CLIPVAD):
    LGT_MODULES = ("temporal", "gc1", "gc2", "gc3", "gc4", "disAdj", "linear",
                   "frame_position_embeddings")

    def __init__(self, *args, aux_branch="both", cnn_kernel=3, feature_dim=256,
                 memory_slots=8, memory_temperature=0.1, residual_scale=0.1,
                 aux_mode="prototype", input_features="clip", cnn_channels=128, **kwargs):
        super().__init__(*args, **kwargs)
        if input_features not in ("clip", "post_lgt"):
            raise ValueError("Unknown input feature kind")
        self.input_features = input_features
        if input_features == "post_lgt":
            # Retain state-dict compatibility, but never optimize the unused LGT.
            for name in self.LGT_MODULES:
                getattr(self, name).requires_grad_(False)
        if aux_branch not in ("none", "c", "a", "both"):
            raise ValueError("Unknown auxiliary branch")
        self.aux_branch = aux_branch
        if aux_mode not in ("prototype", "future_mse", "future_mse_joint", "gated_residual"):
            raise ValueError("Unknown auxiliary mode")
        if aux_mode == "gated_residual" and (aux_branch not in ("c", "none") or cnn_kernel != 3):
            raise ValueError("gated_residual baseline v1 supports C-only fusion with kernel 3")
        self.aux_mode = aux_mode
        self.pretraining_info = None
        self.residual_scale = residual_scale
        self.register_buffer("memory_active", torch.tensor(False))
        self.memory = None
        self.cnn_enhancement = None
        if aux_branch != "none":
            # Do not change the subsequent loader RNG relative to --aux-branch none.
            with torch.random.fork_rng(devices=[]):
                if aux_mode == "gated_residual":
                    self.cnn_enhancement = CNNFeatureEnhancement(self.visual_width, cnn_channels)
                elif aux_mode == "future_mse_joint":
                    self.memory = JointFutureFeatureAuxiliary(
                        self.visual_width, feature_dim, self.num_class, cnn_kernel,
                    )
                elif aux_mode == "future_mse":
                    self.memory = FutureFeatureAuxiliary(
                        self.visual_width, feature_dim, self.num_class, cnn_kernel,
                    )
                else:
                    self.memory = CNNVisualMemory(
                        self.visual_width, feature_dim, self.num_class,
                        memory_slots, cnn_kernel, memory_temperature,
                    )

    def set_memory_active(self, active):
        self.memory_active.fill_(bool(active) and (self.memory is not None or self.cnn_enhancement is not None))

    def forward(self, visual, padding_mask, text, lengths, return_aux=False,
                return_shared_features=False):
        if self.input_features == "post_lgt":
            if visual.ndim != 3 or visual.shape[-1] != self.visual_width:
                raise ValueError(f"Expected cached LGT features [B,T,{self.visual_width}]")
            valid = valid_steps(visual, lengths)
            features = visual.float().masked_fill(~valid[..., None], 0)
        else:
            # Preserve the original forward for CLIP inputs and old checkpoints.
            features = self.encode_video(visual, padding_mask, lengths)
        logits_c = self.classifier(features + self.mlp2(features))
        # A keeps its original C-derived aggregation weights and original H.
        # Reuse the SAME classifier/MLP for H'; there is no auxiliary head.
        visual_prompt_logits = logits_c
        if self.cnn_enhancement is not None and bool(self.memory_active):
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

        auxiliary = None
        if self.memory is not None and bool(self.memory_active):
            memory_input = features if self.input_features == "post_lgt" or self.aux_mode == "prototype" else visual
            if self.aux_mode == "future_mse_joint":
                auxiliary = self.memory(memory_input, lengths, return_prediction=return_aux)
            else:
                auxiliary = self.memory(memory_input, lengths)
            # Visual Prompt above always consumes the original C logits.
            if self.aux_branch in ("c", "both"):
                logits_c = logits_c + self.residual_scale * auxiliary["delta_c"]
            if self.aux_branch in ("a", "both"):
                logits_a = logits_a + self.residual_scale * auxiliary["delta_a"]
        outputs = (text_original, logits_c, logits_a)
        if return_shared_features:
            return (*outputs, auxiliary, features)
        return (*outputs, auxiliary) if return_aux else outputs


def build_model(config, device):
    return CNNMemoryCLIPVAD(
        14, 512, config["visual_length"], 512, 1, config["visual_layers"],
        config["attn_window"], config["prompt_prefix"], config["prompt_postfix"], device,
        **{key: config[key] for key in (
            "aux_branch", "cnn_kernel", "feature_dim", "memory_slots",
            "memory_temperature", "residual_scale",
        )},
        aux_mode=config.get("aux_mode", "prototype"),
        input_features=config.get("input_features", "clip"),
        cnn_channels=config.get("cnn_channels", 128),
    )
