"""Remap the original CSVs to extracted Kaggle features, preserving row order/crops."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def remap_csv(source, destination, index):
    with Path(source).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if not {"path", "label"}.issubset(fields or []) or not rows:
        raise ValueError(f"Invalid source CSV: {source}")
    mapped = []
    for row in rows:
        name = row["path"].replace("\\", "/").rsplit("/", 1)[-1]
        candidates = index.get(name, [])
        if len(candidates) != 1:
            raise ValueError(f"Expected one match for {name}, found {len(candidates)}. "
                             "Use --feature-root pointing to one fully extracted UCF feature dataset.")
        mapped.append(dict(row, path=str(candidates[0])))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(mapped)
    print(f"Mapped {len(mapped)} rows: {source} -> {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", default="/kaggle/input")
    parser.add_argument("--output-dir", default="outputs/cnn_memory_data")
    parser.add_argument("--train-source", default="list/ucf_CLIP_rgb.csv")
    parser.add_argument("--test-source", default="list/ucf_CLIP_rgbtest.csv")
    args = parser.parse_args()
    root = Path(args.feature_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Feature root is not a directory: {root}")
    index = defaultdict(list)
    for path in root.rglob("*.npy"):
        index[path.name].append(path.resolve())
    remap_csv(args.train_source, Path(args.output_dir) / "train.csv", index)
    remap_csv(args.test_source, Path(args.output_dir) / "test.csv", index)


if __name__ == "__main__":
    main()
