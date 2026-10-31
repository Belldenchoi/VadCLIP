"""Explicit options for this experiment, without unrelated ablation flags."""

import argparse
import math

LABEL_MAP = dict(zip(
    ("Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
     "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
     "Shoplifting", "Stealing", "Vandalism"),
    ("normal", "abuse", "arrest", "arson", "assault", "burglary",
     "explosion", "fighting", "roadAccidents", "robbery", "shooting",
     "shoplifting", "stealing", "vandalism"),
))
MODEL_KEYS = (
    "visual_length", "visual_layers", "attn_window", "prompt_prefix",
    "prompt_postfix", "aux_branch", "cnn_kernel", "feature_dim",
    "memory_slots", "memory_temperature", "residual_scale", "aux_mode",
)
RESUME_KEYS = MODEL_KEYS + (
    "seed", "batch_size", "lr", "memory_lr", "scheduler_milestones",
    "scheduler_rate", "memory_loss_weight", "memory_start_epoch",
    "eval_steps", "train_list", "test_list", "gt_path",
    "gt_segment_path", "gt_label_path",
)


def add_data_options(parser, training=False):
    if training:
        parser.add_argument("--train-list", default="list/ucf_CLIP_rgb.csv")
    parser.add_argument("--test-list", default="list/ucf_CLIP_rgbtest.csv")
    parser.add_argument("--gt-path", default="list/gt_ucf.npy")
    parser.add_argument("--gt-segment-path", default="list/gt_segment_ucf.npy")
    parser.add_argument("--gt-label-path", default="list/gt_label_ucf.npy")


def training_parser():
    parser = argparse.ArgumentParser(description="VadCLIP + CNN visual memory (UCF-Crime)")
    add_data_options(parser, training=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Clips per normal/anomaly loader; combined batch is twice this")
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--memory-lr", type=float, default=2e-4)
    parser.add_argument("--scheduler-milestones", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--scheduler-rate", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=10,
                        help="Evaluate every N optimizer updates, and at every epoch end")
    parser.add_argument("--visual-length", type=int, default=256)
    parser.add_argument("--visual-layers", type=int, default=2)
    parser.add_argument("--attn-window", type=int, default=8)
    parser.add_argument("--prompt-prefix", type=int, default=10)
    parser.add_argument("--prompt-postfix", type=int, default=10)
    parser.add_argument("--aux-branch", choices=("none", "c", "a", "both"), default="both")
    parser.add_argument("--aux-mode", choices=("prototype", "future_mse", "future_mse_joint"), default="prototype",
                        help="Prototype memory, frozen pretrained CNN, or jointly trained CNN + future-feature MSE")
    parser.add_argument("--input-features", choices=("clip", "post_lgt"), default="clip",
                        help="clip runs the original LGT; post_lgt uses cached LGT outputs directly for C/A and CNN")
    parser.add_argument("--pretrained-cnn", help="Stage-1 encoder.pt; required for a fresh future_mse run")
    parser.add_argument("--future-horizon", type=int, default=1,
                        help="Future target offset on the feature grid in future_mse_joint mode")
    parser.add_argument("--mse-loss-weight", type=float, default=0.1,
                        help="Joint MSE coefficient; 0 disables only MSE for a matched ablation")
    parser.add_argument("--cnn-kernel", type=int, choices=(1, 3), default=3)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--memory-slots", type=int, default=8, help="Slots in each of two banks")
    parser.add_argument("--memory-temperature", type=float, default=0.1)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--memory-loss-weight", type=float, default=0.1)
    parser.add_argument("--memory-start-epoch", type=int, default=3,
                        help="1-based epoch enabling both residuals and memory MIL loss")
    parser.add_argument("--resume", help="Resume the last.pt checkpoint at the next epoch")
    parser.add_argument("--preflight", action="store_true",
                        help="Validate input files/GT without loading CLIP or training")
    return parser


def validate_args(args):
    if args.aux_mode == "future_mse" and args.aux_branch != "none" and not (args.pretrained_cnn or args.resume):
        raise ValueError("future_mse requires --pretrained-cnn from pretrain.py (or --resume)")
    if args.aux_mode != "future_mse" and args.pretrained_cnn:
        raise ValueError("--pretrained-cnn only applies to --aux-mode future_mse")
    if args.aux_mode == "future_mse_joint" and not 1 <= args.future_horizon < args.visual_length:
        raise ValueError("Joint training requires 1 <= --future-horizon < --visual-length")
    if not math.isfinite(args.mse_loss_weight) or args.mse_loss_weight < 0:
        raise ValueError("--mse-loss-weight must be finite and non-negative")
    for key in ("batch_size", "max_epoch", "eval_steps", "visual_length",
                "visual_layers", "attn_window", "feature_dim", "memory_slots",
                "memory_start_epoch"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    for key in ("lr", "memory_lr", "scheduler_rate", "memory_temperature", "residual_scale"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.memory_loss_weight) or args.memory_loss_weight < 0:
        raise ValueError("--memory-loss-weight must be finite and non-negative")
    if args.visual_length % args.attn_window:
        raise ValueError("--visual-length must be divisible by --attn-window")
    if min(args.prompt_prefix, args.prompt_postfix) < 0 or args.prompt_prefix + args.prompt_postfix > 60:
        raise ValueError("Prompt lengths must be non-negative with total <= 60")
    if any(epoch < 1 for epoch in args.scheduler_milestones):
        raise ValueError("Scheduler milestones must be positive")
    if args.memory_start_epoch > args.max_epoch:
        raise ValueError("--memory-start-epoch must not exceed --max-epoch")


def model_config(args):
    config = {key: getattr(args, key) for key in MODEL_KEYS}
    # Keep the keys shared with older experiments unchanged.
    config["input_features"] = args.input_features
    if args.aux_mode == "future_mse_joint":
        config["future_horizon"] = args.future_horizon
    return config
