"""Check CSV paths, feature dimensions and frame-GT alignment before CLIP loads."""

import csv
from pathlib import Path

import numpy as np

try:
    from .config import LABEL_MAP
except ImportError:
    from config import LABEL_MAP


def inspect_csv(csv_path, all_shapes=False):
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires path,label columns: {csv_path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    counts, checked_labels, frame_lengths = {}, set(), []
    for row in rows:
        label, path = row["label"], Path(row["path"])
        if label not in LABEL_MAP:
            raise ValueError(f"Unknown UCF-Crime label: {label}")
        if not path.is_file():
            raise FileNotFoundError(f"Feature not found: {path}. Remap the CSV to /kaggle/input first.")
        counts[label] = counts.get(label, 0) + 1
        if all_shapes or label not in checked_labels:
            feat = np.load(path, mmap_mode="r", allow_pickle=False)
            if feat.ndim != 2 or feat.shape[1] != 512 or feat.shape[0] < 1:
                raise ValueError(f"Expected nonempty M x 512 feature: {path}: {feat.shape}")
            if not np.issubdtype(feat.dtype, np.floating):
                raise ValueError(f"Feature must have floating dtype: {path}")
            frame_lengths.append(int(feat.shape[0]) * 16)
            checked_labels.add(label)
    return counts, frame_lengths


def validate_data(args, training=True):
    summary = {}
    if training:
        counts, _ = inspect_csv(args.train_list)
        normal = counts.get("Normal", 0)
        abnormal = sum(counts.values()) - normal
        if min(normal, abnormal) < args.batch_size:
            raise ValueError("Normal and abnormal training sets must each contain at least --batch-size clips")
        summary.update(train_normal=normal, train_abnormal=abnormal)
    test_counts, frames = inspect_csv(args.test_list, all_shapes=True)
    gt = np.load(args.gt_path, allow_pickle=False)
    if gt.ndim != 1 or len(gt) != sum(frames) or not np.isin(gt, [0, 1]).all() or len(np.unique(gt)) != 2:
        raise ValueError("Frame GT must be binary, include both classes and match sum(test feature lengths)*16")
    for path in (args.gt_segment_path, args.gt_label_path):
        if len(np.load(path, allow_pickle=True)) != sum(test_counts.values()):
            raise ValueError(f"GT video count does not match test CSV order/count: {path}")
    summary.update(test_clips=sum(test_counts.values()), test_frames=len(gt))
    return summary
