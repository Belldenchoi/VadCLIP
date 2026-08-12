"""Train direct class prototypes with original VadCLIP MIL-Align Top-K."""

import sys
from pathlib import Path

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
from train_ucf import setup_seed, train
from ucf_labels import LABEL_MAP


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    if args.topk_pooling != "mean":
        raise ValueError(
            "First prototype experiment keeps original hard Top-K; "
            "use --topk-pooling mean"
        )
    if args.adaptive_instance_selection or args.temporal_segment_topk:
        raise ValueError(
            "First prototype experiment cannot combine AIS or temporal segment Top-K"
        )
    setup_seed(args.seed)

    normal_dataset = UCFDataset(
        args.visual_length, args.train_list, False, LABEL_MAP, True,
        max_samples_per_label=args.max_samples_per_label,
    )
    anomaly_dataset = UCFDataset(
        args.visual_length, args.train_list, False, LABEL_MAP, False,
        actions=args.train_actions,
        max_samples_per_label=args.max_samples_per_label,
    )
    normal_loader = DataLoader(
        normal_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True
    )
    anomaly_loader = DataLoader(
        anomaly_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True
    )
    if len(normal_loader) == 0 or len(anomaly_loader) == 0:
        raise ValueError(
            "selected training subset is smaller than --batch-size or empty"
        )
    test_dataset = UCFDataset(
        args.visual_length, args.test_list, True, LABEL_MAP
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = PrototypeCLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length,
        args.visual_width, args.visual_head, args.visual_layers,
        args.attn_window, args.prompt_prefix, args.prompt_postfix, device,
        prototype_path=args.prototype_path,
        expected_class_names=list(LABEL_MAP.values()),
        logit_temperature=args.prototype_logit_temperature,
    )
    print(
        f"Training clips: {len(normal_dataset)} normal, "
        f"{len(anomaly_dataset)} anomaly | prototypes: {args.prototype_path} | "
        "C/A pooling: original hard Top-K | loss3: disabled"
    )
    train(
        model, normal_loader, anomaly_loader, test_loader,
        args, LABEL_MAP, device
    )


if __name__ == "__main__":
    main()
