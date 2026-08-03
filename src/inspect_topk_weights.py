"""Inspect frame-level Top-K weights for VadCLIP C- and A-branches."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from model import CLIPVAD
from utils.soft_topk import multi_k_sizes, video_topk_size
from utils.tools import process_feat
from utils.training_log import TrainingLogger


UCF_LABELS = [
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism",
]
UCF_PROMPTS = [
    "normal", "abuse", "arrest", "arson", "assault", "burglary",
    "explosion", "fighting", "roadAccidents", "robbery", "shooting",
    "shoplifting", "stealing", "vandalism",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect VadCLIP Top-K weights")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pooling",
                        choices=["mean", "soft", "multi_k"],
                        required=True)
    parser.add_argument(
        "--sample", action="append", required=True,
        help="Repeat LABEL=FEATURE_PATH for every sample"
    )
    parser.add_argument("--c-temperature", default=1.0, type=float)
    parser.add_argument("--a-temperature", default=1.0, type=float)
    parser.add_argument("--multi-k-percentages", nargs="+", type=float,
                        default=[1.0, 5.0, 10.0, 20.0])
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--log-path", required=True)
    return parser.parse_args()


def parse_sample(value):
    label, separator, feature_path = value.partition("=")
    if not separator or label not in UCF_LABELS:
        raise ValueError(f"Invalid --sample value: {value}")
    return label, Path(feature_path)


def get_weights(topk_scores, pooling, temperature):
    if pooling == "mean":
        return torch.full_like(topk_scores, 1.0 / len(topk_scores))
    return torch.softmax(topk_scores / temperature, dim=0)


def get_multi_k_weights(scores, percentages):
    sizes = multi_k_sizes(len(scores), percentages)
    max_k = max(sizes)
    topk_scores, topk_indices = torch.topk(scores, k=max_k, dim=0)
    weights = torch.zeros_like(topk_scores)
    for scale_k in sizes:
        weights[:scale_k] += 1.0 / (len(sizes) * scale_k)
    return topk_scores, topk_indices, weights, sizes


def infer(model, feature, device):
    processed, length = process_feat(feature, 256)
    visual = torch.from_numpy(processed).unsqueeze(0).to(device)
    lengths = torch.tensor([length], device=device)
    with torch.no_grad():
        _, logits1, logits2 = model(
            visual, None, UCF_PROMPTS, lengths
        )
    c_scores = torch.sigmoid(logits1[0, :length, 0]).float().cpu()
    a_logits = logits2[0, :length].float().cpu()
    return c_scores, a_logits, int(length)


def main():
    args = parse_args()
    if args.c_temperature <= 0 or args.a_temperature <= 0:
        raise ValueError("Top-K temperatures must be positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPVAD(
        14, 512, 256, 512, 1, 2, 8, 10, 10, device
    )
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    logger = TrainingLogger(args.log_path)
    logger.log(
        f"model={args.model_path} pooling={args.pooling} "
        f"c_temperature={args.c_temperature} "
        f"a_temperature={args.a_temperature} "
        f"multi_k_percentages={args.multi_k_percentages}"
    )
    rows = []

    for sample_value in args.sample:
        label, feature_path = parse_sample(sample_value)
        feature = np.load(feature_path)
        c_scores, a_logits, length = infer(model, feature, device)
        class_index = UCF_LABELS.index(label)
        default_k = video_topk_size(length)

        for branch, scores, temperature in (
            ("C_branch", c_scores, args.c_temperature),
            (f"A_branch_{label}", a_logits[:, class_index],
             args.a_temperature),
        ):
            scale_weight_display = ""
            if args.pooling == "multi_k":
                topk_scores, topk_indices, weights, k_values = \
                    get_multi_k_weights(scores, args.multi_k_percentages)
                k_display = "|".join(str(value) for value in k_values)
            else:
                topk_scores, topk_indices = torch.topk(
                    scores, k=default_k, dim=0
                )
                weights = get_weights(
                    topk_scores, args.pooling, temperature
                )
                k_display = str(default_k)
            contributions = weights * topk_scores
            pooled_score = contributions.sum().item()
            arithmetic_mean = topk_scores.mean().item()
            logger.log(
                f"video={feature_path.stem} label={label} branch={branch} "
                f"length={length} K={k_display} "
                f"weight_sum={weights.sum().item():.6f} "
                f"topk_mean={arithmetic_mean:.6f} "
                f"pooled_score={pooled_score:.6f} "
                f"scale_weights={scale_weight_display or 'n/a'}"
            )

            for rank, (index, score, weight, contribution) in enumerate(
                zip(topk_indices, topk_scores, weights, contributions),
                start=1,
            ):
                row = {
                    "pooling": args.pooling,
                    "video": feature_path.stem,
                    "label": label,
                    "branch": branch,
                    "length": length,
                    "K": k_display,
                    "rank": rank,
                    "frame_index": int(index),
                    "frame_start": int(index) * 16,
                    "frame_end": (int(index) + 1) * 16 - 1,
                    "score": float(score),
                    "weight": float(weight),
                    "contribution": float(contribution),
                    "topk_mean": arithmetic_mean,
                    "pooled_score": pooled_score,
                    "scale_weights": scale_weight_display,
                }
                rows.append(row)
                logger.log(
                    f"  rank={rank:02d} frame_index={int(index):03d} "
                    f"score={float(score):.6f} "
                    f"weight={float(weight):.6f} "
                    f"contribution={float(contribution):.6f}"
                )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.log(f"csv_saved={output_path}")


if __name__ == "__main__":
    main()
