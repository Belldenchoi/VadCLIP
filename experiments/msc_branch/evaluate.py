"""Evaluate best.pt/last.pt using their saved architecture and active phase."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.cnn_memory.config import LABEL_MAP, add_data_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--log-path")
    add_data_options(parser)
    args = parser.parse_args()
    from experiments.cnn_memory import bootstrap
    from experiments.msc_branch.checkpoint import load_checkpoint
    from experiments.msc_branch.msc_model import build_model
    from experiments.cnn_memory.data_check import validate_data
    from experiments.cnn_memory.evaluation import evaluate
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from training_log import TrainingLogger
    from utils.dataset import UCFDataset
    from utils.tools import get_prompt_text

    validate_data(args, training=False)
    checkpoint = load_checkpoint(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("The original LGT DistanceAdj uses CUDA. Enable a Kaggle GPU.")
    model = build_model(checkpoint["model_config"], device).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = UCFDataset(checkpoint["model_config"]["visual_length"], args.test_list, True, LABEL_MAP)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    logger = TrainingLogger(args.log_path)
    logger.log(f"evaluation_model={args.checkpoint} clips={len(dataset)} actions=all "
               f"device={device} epoch={checkpoint['epoch']} "
               f"memory_active={bool(model.memory_active)} config={json.dumps(checkpoint['model_config'])}")
    evaluate(model, loader, checkpoint["model_config"]["visual_length"],
             get_prompt_text(LABEL_MAP), np.load(args.gt_path),
             np.load(args.gt_segment_path, allow_pickle=True),
             np.load(args.gt_label_path, allow_pickle=True), device, logger.log)


if __name__ == "__main__":
    main()
