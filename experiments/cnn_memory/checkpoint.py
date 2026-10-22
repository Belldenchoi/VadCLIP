"""Versioned checkpoints including the active phase and reproducible resume state."""

import random

import numpy as np
import torch

FORMAT_VERSION = 1


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
    if checkpoint.get("format_version") != FORMAT_VERSION or checkpoint.get("experiment") != "cnn_memory":
        raise ValueError("Expected a versioned cnn_memory checkpoint, not an original VadCLIP .pth")
    return checkpoint


def checkpoint_payload(model, config, args, epoch, global_step, best_metrics, kind):
    return {
        "format_version": FORMAT_VERSION, "experiment": "cnn_memory", "kind": kind,
        "model_config": config, "train_args": vars(args).copy(),
        "model_state_dict": model.state_dict(), "epoch": epoch,
        "global_step": global_step, "best_metrics": best_metrics,
    }
