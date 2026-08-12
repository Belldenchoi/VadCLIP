"""Encode, score, filter, and cache UCF-Crime class prototypes."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from clip import clip
from prototype_utils import (
    aggregate_prototype,
    encode_texts,
    load_description_file,
    score_embeddings,
    select_diverse_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build frozen CLIP class prototypes"
    )
    parser.add_argument(
        "--descriptions",
        default=str(EXPERIMENT_ROOT / "descriptions" /
                    "ucf_crime_descriptions.json"),
    )
    parser.add_argument(
        "--output", default="outputs/class_prototypes/ucf_crime.pt"
    )
    parser.add_argument(
        "--audit-output",
        default="outputs/class_prototypes/ucf_crime_selection.json",
    )
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--diversity-threshold", default=0.90, type=float)
    parser.add_argument(
        "--aggregation", choices=["mean", "weighted_mean"],
        default="weighted_mean"
    )
    parser.add_argument("--weight-temperature", default=0.10, type=float)
    parser.add_argument("--alpha", default=0.4, type=float)
    parser.add_argument("--beta", default=0.3, type=float)
    parser.add_argument("--gamma", default=0.3, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if not -1.0 <= args.diversity_threshold <= 1.0:
        raise ValueError("--diversity-threshold must be in [-1, 1]")

    payload, description_hash = load_description_file(args.descriptions)
    class_names = list(payload["classes"])
    descriptions_by_class = [
        payload["classes"][name] for name in class_names
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load(args.clip_model, device)
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    flat_descriptions = [
        text for descriptions in descriptions_by_class for text in descriptions
    ]
    flat_embeddings = encode_texts(
        clip_model, clip, flat_descriptions, device, args.batch_size
    )
    class_embeddings = []
    offset = 0
    for descriptions in descriptions_by_class:
        end = offset + len(descriptions)
        class_embeddings.append(flat_embeddings[offset:end])
        offset = end
    # Class names themselves are anchors; no article-bearing prompt template.
    anchor_texts = [
        "road accidents" if name == "roadAccidents" else name
        for name in class_names
    ]
    anchors = encode_texts(
        clip_model, clip, anchor_texts, device, args.batch_size
    )
    scores_by_class = score_embeddings(
        class_embeddings, anchors, args.alpha, args.beta, args.gamma
    )

    prototypes = []
    audit_classes = {}
    for class_index, class_name in enumerate(class_names):
        embeddings = class_embeddings[class_index]
        score_data = scores_by_class[class_index]
        selected, rejected = select_diverse_embeddings(
            embeddings, score_data["total"], args.top_k,
            args.diversity_threshold
        )
        if not selected:
            raise RuntimeError(f"no prototype contributor for {class_name}")
        prototype, weights = aggregate_prototype(
            embeddings[selected], score_data["total"][selected],
            args.aggregation, args.weight_temperature
        )
        prototypes.append(prototype)
        selected_rows = []
        for position, embedding_index in enumerate(selected):
            selected_rows.append({
                "index": embedding_index,
                "description": descriptions_by_class[class_index][embedding_index],
                "score": float(score_data["total"][embedding_index]),
                "name_score": float(score_data["name"][embedding_index]),
                "intra_score": float(score_data["intra"][embedding_index]),
                "inter_score": float(score_data["inter"][embedding_index]),
                "hard_inter_class": class_names[int(
                    score_data["hard_inter_class"][embedding_index]
                )],
                "weight": float(weights[position]),
            })
        audit_classes[class_name] = {
            "candidate_count": len(descriptions_by_class[class_index]),
            "selected_count": len(selected),
            "selected": selected_rows,
            "diversity_rejections": rejected,
        }

    prototype_tensor = torch.stack(prototypes).float()
    config = {
        "clip_model": args.clip_model,
        "description_sha256": description_hash,
        "top_k": args.top_k,
        "diversity_threshold": args.diversity_threshold,
        "aggregation": args.aggregation,
        "weight_temperature": args.weight_temperature,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
    }
    cache = {
        "prototypes": prototype_tensor,
        "class_names": class_names,
        "config": config,
    }
    output_path = Path(args.output)
    audit_path = Path(args.audit_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description_file": str(Path(args.descriptions).resolve()),
        "class_order": class_names,
        "prototype_shape": list(prototype_tensor.shape),
        "prototype_norms": prototype_tensor.norm(dim=-1).tolist(),
        "config": config,
        "classes": audit_classes,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved prototypes: {output_path}")
    print(f"saved audit: {audit_path}")
    print(f"shape: {tuple(prototype_tensor.shape)} device_used: {device}")


if __name__ == "__main__":
    main()
