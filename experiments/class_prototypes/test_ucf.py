"""Evaluate a direct-prototype VadCLIP checkpoint."""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import options as ucf_option

EXPERIMENT_ROOT = Path(__file__).resolve().parent
TOPK_ROOT = EXPERIMENT_ROOT.parent / "topk_variants"
SRC_ROOT = EXPERIMENT_ROOT.parents[1] / "src"
sys.path.insert(0, str(TOPK_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from dataset_variants import UCFDataset
from prototype_model import PrototypeCLIPVAD
from training_log import TrainingLogger
from ucf_labels import LABEL_MAP
from utils.tools import get_prompt_text

# Import after TOPK_ROOT is first so evaluation remains identical to the
# isolated Top-K experiment evaluator.
from test_ucf import test


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    prompt_text = get_prompt_text(LABEL_MAP)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    dataset = UCFDataset(
        args.visual_length, args.test_list, True, LABEL_MAP
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = PrototypeCLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length,
        args.visual_width, args.visual_head, args.visual_layers,
        args.attn_window, args.prompt_prefix, args.prompt_postfix, device,
        prototype_path=args.prototype_path,
        expected_class_names=list(LABEL_MAP.values()),
        logit_temperature=args.prototype_logit_temperature,
    )
    state_dict = torch.load(
        args.model_path, map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    logger = TrainingLogger(args.log_path)
    logger.log(
        f"evaluation_model={args.model_path} prototypes={args.prototype_path} "
        f"clips={len(dataset)}"
    )
    test(
        model, loader, args.visual_length, prompt_text,
        gt, gtsegments, gtlabels, device, logger=logger
    )


if __name__ == "__main__":
    main()
