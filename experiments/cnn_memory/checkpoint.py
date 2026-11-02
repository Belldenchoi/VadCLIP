"""Versioned checkpoints including the active phase and reproducible resume state."""

import random

import numpy as np
import torch

FORMAT_VERSION = 2


def capture_rng():
    state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    version = checkpoint.get("format_version")
    if version not in (1, FORMAT_VERSION) or checkpoint.get("experiment") != "cnn_memory":
        raise ValueError("Expected a versioned cnn_memory checkpoint, not an original VadCLIP .pth")
    if checkpoint.get("model_config", {}).get("aux_mode") != "gated_residual":
        raise ValueError("Only gated_residual checkpoints are supported; prototype/MSE methods were removed")
    if version == 1:
        # Migrate metadata/phase names for the already published GRF only.
        # No legacy model or loss is constructed.
        try:
            from .config import MODEL_KEYS
        except ImportError:
            from config import MODEL_KEYS
        config = checkpoint["model_config"]
        config.setdefault("input_features", "clip")
        config.setdefault("cnn_channels", 128)
        checkpoint["model_config"] = {key: config[key] for key in MODEL_KEYS}
        train_args = checkpoint["train_args"]
        train_args["cnn_start_epoch"] = train_args.pop("memory_start_epoch", 1)
        train_args.setdefault("input_features", config["input_features"])
        train_args.setdefault("cnn_channels", config["cnn_channels"])
        for key in ("memory_lr", "memory_loss_weight", "mse_loss_weight", "feature_dim",
                    "memory_slots", "memory_temperature", "residual_scale", "future_horizon", "pretrained_cnn"):
            train_args.pop(key, None)
        state = checkpoint["model_state_dict"]
        state["enhancement_active"] = state.pop("memory_active")
        checkpoint.pop("pretraining_info", None)
        checkpoint["format_version"] = FORMAT_VERSION
    return checkpoint


def checkpoint_payload(model, config, args, epoch, global_step, best_metrics, kind):
    return {
        "format_version": FORMAT_VERSION, "experiment": "cnn_memory", "kind": kind,
        "model_config": config, "train_args": vars(args).copy(),
        "model_state_dict": model.state_dict(), "epoch": epoch,
        "global_step": global_step, "best_metrics": best_metrics,
    }
