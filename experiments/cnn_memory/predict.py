"""Optional video gate and temporal intervals; never changes the AUC evaluator."""

import argparse
import csv
import json
import math
from pathlib import Path


def prediction_record(scores, video_threshold=.5, frame_threshold=.5):
    import numpy as np
    for threshold in (video_threshold, frame_threshold):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be finite probabilities in [0,1]")
    scores = np.asarray(scores).reshape(-1)
    if not len(scores) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected nonempty, finite temporal probabilities in [0,1]")
    # Same probability-form hard Top-K as the original CLAS2 loss.
    k = len(scores) // 16 + 1
    video_score = float(np.sort(scores)[-k:].mean())
    abnormal = video_score >= video_threshold
    segments = []
    if abnormal:
        selected = scores >= frame_threshold
        starts = np.flatnonzero(np.diff(np.r_[False, selected].astype(np.int8)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[selected, False].astype(np.int8)) == -1) + 1
        for start, end in zip(starts, ends):
            segments.append(dict(start_feature=int(start), end_feature_exclusive=int(end),
                                 start_frame=int(start) * 16, end_frame_exclusive=int(end) * 16,
                                 max_score=float(scores[start:end].max())))
    return dict(predicted_label="Abnormal" if abnormal else "Normal", video_score=video_score,
                video_threshold=video_threshold, frame_threshold=frame_threshold,
                feature_steps=len(scores), frame_repeat=16, segments=segments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-list", required=True, help="CSV requires path only; any labels are ignored")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--video-threshold", type=float, default=.5)
    parser.add_argument("--frame-threshold", type=float, default=.5)
    parser.add_argument("--score-branch", choices=("c", "a"), default="c")
    args = parser.parse_args()
    # Validate thresholds before allocating/downloading a model.
    prediction_record([0.], args.video_threshold, args.frame_threshold)
    output = Path(args.output_path)
    if output.exists():
        raise FileExistsError(f"Choose a new prediction output: {output}")
    with Path(args.video_list).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "path" not in (reader.fieldnames or []):
            raise ValueError("Video CSV requires a path column")
        rows = list(reader)
    if not rows:
        raise ValueError("Video CSV is empty")
    try:
        from . import bootstrap
        from .checkpoint import load_checkpoint
        from .cnn_memory_model import build_model
        from .config import LABEL_MAP
    except ImportError:
        import bootstrap
        from checkpoint import load_checkpoint
        from cnn_memory_model import build_model
        from config import LABEL_MAP
    import numpy as np
    import torch
    from utils.tools import get_batch_mask, get_prompt_text

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["model_config"]
    if not torch.cuda.is_available() and config.get("input_features", "clip") == "clip":
        raise RuntimeError("The original LGT DistanceAdj requires CUDA")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config, device).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    maxlen = config["visual_length"]
    records = []
    with torch.no_grad():
        for row in rows:
            feature = np.load(row["path"], allow_pickle=False)
            if feature.ndim != 2 or feature.shape[1] != 512 or not len(feature) or not np.isfinite(feature).all():
                raise ValueError(f"Invalid 512-D video features: {row['path']}")
            count = (len(feature) + maxlen - 1) // maxlen
            chunks = np.zeros((count, maxlen, 512), dtype=np.float32)
            chunks.reshape(-1, 512)[:len(feature)] = feature
            lengths = torch.tensor([min(maxlen, len(feature) - i * maxlen) for i in range(count)])
            mask = get_batch_mask(lengths, maxlen).to(device)
            _, c, a = model(torch.from_numpy(chunks).to(device), mask, get_prompt_text(LABEL_MAP), lengths.to(device))
            if args.score_branch == "c":
                scores = c.reshape(-1)[:len(feature)].sigmoid()
            else:
                scores = 1 - a.reshape(-1, len(LABEL_MAP))[:len(feature)].softmax(-1)[:, 0]
            record = prediction_record(scores.cpu().numpy(), args.video_threshold, args.frame_threshold)
            records.append(dict(path=row["path"], **record))
            print(f"video={Path(row['path']).name} prediction={record['predicted_label']} "
                  f"video_score={record['video_score']:.6f} segments={len(record['segments'])}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(checkpoint=args.checkpoint, score_branch=args.score_branch,
                                     ground_truth_used=False, videos=records), indent=2) + "\n", encoding="utf-8")
    print("Saved:", output, flush=True)


if __name__ == "__main__":
    main()
