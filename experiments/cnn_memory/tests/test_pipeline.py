"""CPU fixture checks; these do not establish Kaggle training accuracy."""

import copy
import contextlib
import csv
import importlib
import io
import json
import random
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.cnn_memory.checkpoint import capture_rng, load_checkpoint, restore_rng
from experiments.cnn_memory.config import training_parser, validate_args
from experiments.cnn_memory.data_check import validate_data
from experiments.cnn_memory.prepare_csv import remap_csv

torch.set_num_threads(1)


class DataAndEvaluationTests(unittest.TestCase):
    def test_remap_preserves_crop_names_labels_and_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "original.csv", root / "remapped.csv"
            source.write_text("path,label\n/old/B__5.npy,Normal\n/old/A__2.npy,Abuse\n", encoding="utf-8")
            index = {name: [root / name] for name in ("A__2.npy", "B__5.npy")}
            remap_csv(source, target, index)
            with target.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([Path(row["path"]).name for row in rows], ["B__5.npy", "A__2.npy"])
            self.assertEqual([row["label"] for row in rows], ["Normal", "Abuse"])
            with self.assertRaises(ValueError):
                remap_csv(source, target, {"B__5.npy": [root / "one", root / "two"]})

    def test_preflight_rejects_misaligned_gt_and_runs_without_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = root / "feature.npy"
            np.save(feature, np.zeros((3, 512), dtype=np.float16))
            csv_path = root / "features.csv"
            csv_path.write_text(f"path,label\n{feature},Normal\n{feature},Abuse\n", encoding="utf-8")
            gt_path, segments, labels = root / "gt.npy", root / "segments.npy", root / "labels.npy"
            np.save(gt_path, np.repeat([0, 0, 0, 1, 1, 1], 16))
            np.save(segments, np.zeros((2, 1, 2)))
            np.save(labels, np.array(["Normal", "Abuse"]))
            command = [str(REPO / "experiments/cnn_memory/train.py"), "--output-dir", str(root / "run"),
                       "--batch-size", "1", "--preflight", "--train-list", str(csv_path),
                       "--test-list", str(csv_path), "--gt-path", str(gt_path),
                       "--gt-segment-path", str(segments), "--gt-label-path", str(labels)]
            result = subprocess.run([sys.executable, "-B", *command], cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Preflight passed", result.stdout)
            self.assertFalse((root / "run").exists())
            np.save(gt_path, np.array([0, 1]))
            args = training_parser().parse_args(command[1:])
            with self.assertRaises(ValueError):
                validate_data(args)

    def test_evaluation_keeps_original_chunks_and_uses_raw_repeat16_scores(self):
        from experiments.cnn_memory.evaluation import evaluate

        class Detector(nn.Module):
            def forward(self, features, mask, text, lengths):
                self.batch_count = len(features)
                self.lengths = lengths
                c = features[..., :1]
                a = torch.cat((-c, c), dim=-1)
                return torch.zeros(2, 1), c, a

        features = torch.zeros(1, 3, 4, 512)
        features[0, 0, :, 0] = -2
        features[0, 1, :, 0] = 2
        detector = Detector()
        gt = np.repeat([0] * 4 + [1] * 4, 16)
        messages = []
        loader = [(features, ["Abuse"], torch.tensor([8]))]
        with patch("experiments.topk_variants.test_ucf.dmAP", return_value=([10., 8., 6., 4., 2.], [.1, .2, .3, .4, .5])):
            result = evaluate(detector, loader, 4, [], gt, [], [], "cpu", log=messages.append)
        self.assertEqual(detector.batch_count, 3)
        self.assertEqual(detector.lengths.tolist(), [4, 4, 4])
        self.assertEqual(result["AUC1"], 1.)
        self.assertEqual(result["AUC2"], 1.)
        self.assertEqual(result["mAP"], 6.)  # Already percent, not multiplied by 100.
        self.assertIn("temporal_eval_active=False evaluation_mode=original", messages)
        self.assertIn("AUC1=1.000000 AP1=1.000000", messages)
        self.assertIn("AUC2=1.000000 AP2=1.000000", messages)
        self.assertEqual([line for line in messages if line.startswith("mAP@")], [
            "mAP@0.1=10.00%", "mAP@0.2=8.00%", "mAP@0.3=6.00%",
            "mAP@0.4=4.00%", "mAP@0.5=2.00%",
        ])
        self.assertEqual(messages[-1], "average_mAP=6.00%")
        with self.assertRaises(ValueError):
            evaluate(detector, loader, 4, [], np.zeros(2), [], [], "cpu", log=lambda _: None)

    def test_cli_help_entrypoints_work_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("train.py", "evaluate.py", "prepare_csv.py"):
                result = subprocess.run([sys.executable, "-B", str(REPO / "experiments/cnn_memory" / name), "--help"],
                                        cwd=directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_configuration_rejects_unreachable_activation(self):
        args = training_parser().parse_args(["--output-dir", "unused", "--max-epoch", "1", "--cnn-start-epoch", "2"])
        with self.assertRaises(ValueError):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
