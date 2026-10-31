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
from experiments.cnn_memory.cnn_memory import CNNVisualMemory
from experiments.cnn_memory.checkpoint import capture_rng, load_checkpoint, restore_rng
from experiments.cnn_memory.config import training_parser, validate_args
from experiments.cnn_memory.data_check import validate_data
from experiments.cnn_memory.prepare_csv import remap_csv

torch.set_num_threads(1)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(234)
        self.memory = CNNVisualMemory(8, 6, 3, slots=2)

    def test_padding_nan_cannot_change_valid_features_or_gradients(self):
        clean = torch.randn(2, 7, 8)
        contaminated = clean.clone()
        contaminated[0, 4:] = float("nan")
        contaminated[1] = float("nan")
        clean[0, 4:] = 0
        clean[1] = 0
        actual = self.memory(contaminated.requires_grad_(), [4, 0])
        expected = self.memory(clean, [4, 0])
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key])
            self.assertTrue(bool(torch.isfinite(actual[key]).all()))
            self.assertEqual(int(torch.count_nonzero(actual[key][1])), 0)
        actual["memory_logits"].sum().backward()
        self.assertTrue(bool(torch.isfinite(contaminated.grad).all()))
        self.assertEqual(int(torch.count_nonzero(contaminated.grad[0, 4:])), 0)

    def test_first_residual_is_zero_and_mil_trains_cnn_and_banks(self):
        from experiments.cnn_memory import bootstrap
        # Original probability Top-K MIL, without a CLIP model import.
        logits = self.memory(torch.randn(2, 20, 8), [20, 15])
        self.assertEqual(int(torch.count_nonzero(logits["delta_c"])), 0)
        self.assertEqual(int(torch.count_nonzero(logits["delta_a"])), 0)
        pooled = torch.stack([
            logits["memory_logits"][i, :length, 0].sigmoid().topk(length // 16 + 1).values.mean()
            for i, length in enumerate((20, 15))
        ])
        loss = nn.functional.binary_cross_entropy(pooled, torch.tensor([0., 1.]))
        loss.backward()
        for parameter in (self.memory.conv1.weight, self.memory.conv2.weight, self.memory.prototypes):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_kernel_one_has_no_added_neighbor_dependence_kernel_three_does(self):
        features = torch.zeros(1, 5, 1)
        features[0, 1] = 2
        for kernel in (1, 3):
            memory = CNNVisualMemory(1, 2, 2, slots=1, kernel=kernel)
            with torch.no_grad():
                memory.conv1.weight.fill_(1)
                memory.conv1.bias.zero_()
            hidden = memory.conv1(features.transpose(1, 2))
            self.assertEqual(hidden.shape[-1], 5)
            self.assertEqual(float(hidden[0, 0, 2].detach()), 2. if kernel == 3 else 0.)

    def test_checkpoint_rng_optimizer_roundtrip_reproduces_next_update(self):
        optimizer = torch.optim.AdamW(self.memory.parameters(), lr=1e-3)
        features = torch.randn(2, 8, 8)

        def update():
            optimizer.zero_grad()
            output = self.memory(features + torch.randn_like(features) * .01, [8, 6])
            loss = output["memory_logits"].square().mean() + output["delta_a"].square().mean()
            loss.backward()
            optimizer.step()

        update()
        payload = copy.deepcopy(dict(format_version=1, experiment="cnn_memory",
                                     model_state_dict=self.memory.state_dict(),
                                     optimizer=optimizer.state_dict(), rng=capture_rng()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.pt"
            torch.save(payload, path)
            checkpoint = load_checkpoint(path)
        update()
        expected = copy.deepcopy(self.memory.state_dict())
        self.memory.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"])
        update()
        for key, value in self.memory.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_invalid_lengths_and_kernel_are_rejected(self):
        for lengths in ([8], [-1, 3], [3, 10]):
            with self.assertRaises(ValueError):
                self.memory(torch.zeros(2, 7, 8), lengths)
        with self.assertRaises(ValueError):
            CNNVisualMemory(kernel=2)


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 8)
        self.dtype = torch.float32

    def encode_token(self, tokens):
        return self.embedding(tokens)

    def encode_text(self, embeddings, tokens):
        return embeddings.sum(dim=1)


def tokenize(text):
    tokens = torch.zeros(len(text), 77, dtype=torch.long)
    tokens[:, 0] = 1
    tokens[:, 1] = torch.arange(len(text)) + 2
    tokens[:, 2] = 99
    return tokens


class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_package = types.ModuleType("clip")
        fake_package.clip = types.SimpleNamespace(load=lambda *a: (FakeCLIP(), None), tokenize=tokenize)
        cls.clip_patch = patch.dict(sys.modules, {"clip": fake_package})
        cls.clip_patch.start()
        cls.module = importlib.import_module("experiments.cnn_memory.cnn_memory_model")
        cls.base_class = importlib.import_module("model").CLIPVAD

    @classmethod
    def tearDownClass(cls):
        cls.clip_patch.stop()

    def make_model(self, branch="both"):
        torch.manual_seed(234)
        model = self.module.CNNMemoryCLIPVAD(
            3, 8, 8, 8, 1, 1, 4, 1, 1, "cpu", aux_branch=branch,
            feature_dim=6, memory_slots=2,
        )
        # Original DistanceAdj hardcodes CUDA; only this fixture uses identity.
        model.disAdj.forward = lambda batch, length: torch.eye(length).expand(batch, -1, -1)
        return model.eval()

    def inputs(self):
        return torch.randn(2, 8, 8), None, ["normal", "abuse", "arrest"], torch.tensor([8, 5])

    def test_inactive_and_zero_initialized_active_match_original_forward(self):
        model = self.make_model()
        inputs = self.inputs()
        expected = self.base_class.forward(model, *inputs)
        for active in (False, True):
            model.set_memory_active(active)
            actual = model(*inputs)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_c_only_does_not_feed_corrected_logits_back_into_visual_prompt(self):
        model = self.make_model("c")
        inputs = self.inputs()
        expected = model(*inputs)
        model.set_memory_active(True)
        with torch.no_grad():
            model.memory.c_head.bias.fill_(2)
            model.memory.a_head.bias.fill_(5)
        actual = model(*inputs)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(actual[1][0], expected[1][0] + .2)

    def test_a_only_keeps_c_logits_and_learns_class_specific_residuals(self):
        model = self.make_model("a")
        inputs = self.inputs()
        expected = model(*inputs)
        model.set_memory_active(True)
        with torch.no_grad():
            model.memory.a_head.bias.copy_(torch.tensor([0., 1., 2.]))
        actual = model(*inputs)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        torch.testing.assert_close(actual[2][0] - expected[2][0], torch.tensor([0., .1, .2]).expand(8, -1))

    def test_auxiliary_initialization_preserves_backbone_and_loader_rng(self):
        control = self.make_model("none")
        control_rng = torch.get_rng_state()
        augmented = self.make_model("both")
        torch.testing.assert_close(torch.get_rng_state(), control_rng, rtol=0, atol=0)
        for key, value in control.state_dict().items():
            torch.testing.assert_close(augmented.state_dict()[key], value, rtol=0, atol=0)

    def test_active_flag_and_outputs_survive_checkpoint_roundtrip(self):
        model = self.make_model()
        model.set_memory_active(True)
        with torch.no_grad():
            model.memory.c_head.bias.fill_(1)
        inputs = self.inputs()
        expected = model(*inputs)
        restored = self.make_model()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertTrue(bool(restored.memory_active))
        for left, right in zip(restored(*inputs), expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_original_mil_plus_memory_loss_backpropagates(self):
        from ucf_train import CLASM, CLAS2
        model = self.make_model().train()
        model.set_memory_active(True)
        inputs = self.inputs()
        _, c, a, auxiliary = model(*inputs, return_aux=True)
        labels = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        loss = CLAS2(c, labels, inputs[3], "cpu") + CLASM(a, labels, inputs[3], "cpu")
        loss = loss + .1 * CLAS2(auxiliary["memory_logits"], labels, inputs[3], "cpu")
        loss.backward()
        for parameter in (model.memory.conv1.weight, model.memory.prototypes,
                          model.memory.c_head.weight, model.memory.a_head.weight,
                          model.classifier.weight, model.linear.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_training_entrypoint_checkpoint_resume_matches_continuous_fixture(self):
        from experiments.cnn_memory import train as entrypoint
        from experiments.cnn_memory import cnn_memory_model

        class TinyDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(512, 8)
                self.classifier = nn.Linear(8, 1)
                self.text = nn.Parameter(torch.randn(14, 8))
                self.memory = CNNVisualMemory(8, 6, 14, slots=2)
                self.register_buffer("memory_active", torch.tensor(False))

            def set_memory_active(self, active):
                self.memory_active.fill_(active)

            def forward(self, visual, mask, text, lengths, return_aux=False):
                features = self.projection(visual.float())
                c = self.classifier(features)
                a = features @ self.text.T
                auxiliary = self.memory(features, lengths) if bool(self.memory_active) else None
                if auxiliary is not None:
                    c = c + .1 * auxiliary["delta_c"]
                    a = a + .1 * auxiliary["delta_a"]
                outputs = (self.text, c, a)
                return (*outputs, auxiliary) if return_aux else outputs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal, abnormal = root / "normal.npy", root / "abnormal.npy"
            np.save(normal, np.full((5, 512), -.5, dtype=np.float32))
            np.save(abnormal, np.full((9, 512), .5, dtype=np.float32))
            csv_path = root / "data.csv"
            csv_path.write_text(f"path,label\n{normal},Normal\n{abnormal},Abuse\n", encoding="utf-8")
            gt, segments, labels = root / "gt.npy", root / "segments.npy", root / "labels.npy"
            np.save(gt, np.repeat([0] * 5 + [1] * 9, 16))
            np.save(segments, np.zeros((2, 1, 2)))
            np.save(labels, np.array(["Normal", "Abuse"]))
            common = ["--train-list", str(csv_path), "--test-list", str(csv_path),
                      "--gt-path", str(gt), "--gt-segment-path", str(segments),
                      "--gt-label-path", str(labels), "--batch-size", "1",
                      "--visual-length", "8", "--attn-window", "4", "--eval-steps", "1",
                      "--memory-start-epoch", "1", "--feature-dim", "6", "--memory-slots", "2"]

            def run(output, epochs, resume=False):
                args = ["train.py", *common, "--output-dir", str(output), "--max-epoch", str(epochs)]
                if resume:
                    args += ["--resume", str(output / "last.pt")]
                with patch.object(sys, "argv", args), patch.object(entrypoint, "training_device", return_value="cpu"), \
                        patch.object(cnn_memory_model, "build_model", side_effect=lambda *a: TinyDetector()), \
                        patch("experiments.topk_variants.test_ucf.dmAP", return_value=([0.] * 5, [.1, .2, .3, .4, .5])), \
                        contextlib.redirect_stdout(io.StringIO()):
                    entrypoint.main()

            uninterrupted, resumed = root / "continuous", root / "resumed"
            run(uninterrupted, 2)
            run(resumed, 1)
            # Older checkpoints have no feature-domain field and must stay CLIP-input runs.
            legacy = load_checkpoint(resumed / "last.pt")
            legacy["train_args"].pop("input_features")
            legacy["model_config"].pop("input_features")
            torch.save(legacy, resumed / "last.pt")
            run(resumed, 2, resume=True)
            expected = load_checkpoint(uninterrupted / "last.pt")
            actual = load_checkpoint(resumed / "last.pt")
            self.assertEqual(actual["global_step"], expected["global_step"])
            self.assertEqual(actual["epoch"], 2)
            self.assertTrue(bool(actual["model_state_dict"]["memory_active"]))
            self.assertEqual(actual["best_metrics"], expected["best_metrics"])
            for key, value in actual["model_state_dict"].items():
                torch.testing.assert_close(value, expected["model_state_dict"][key], rtol=0, atol=0)

            # Same progress/AUC patterns used by tools/report_helpers/export_tables.py.
            # Every evaluation must retain its epoch/batch context, including the
            # final batch, while the precise JSON metrics stay unchanged.
            log_lines = (uninterrupted / "train.log").read_text(encoding="utf-8").splitlines()
            history = [json.loads(line) for line in (uninterrupted / "metrics.jsonl").read_text().splitlines()]
            epoch = batch = None
            legacy_records = []
            for line in log_lines:
                progress = re.search(r"epoch=(\d+)/\d+ batch=(\d+)/\d+", line)
                if progress:
                    epoch, batch = map(int, progress.groups())
                    losses = {key: float(value) for key, value in re.findall(
                        r"\b(loss|loss1|loss2|loss3|weighted_memory)=([\d.]+)", line
                    )}
                    self.assertAlmostEqual(losses["loss"], sum(losses[k] for k in (
                        "loss1", "loss2", "loss3", "weighted_memory"
                    )), delta=0.0003)
                    self.assertIn("lr=", line)
                    self.assertIn("peak_vram=", line)
                metric = re.search(r"AUC1[=:]\s*([\d.]+)", line)
                if metric:
                    legacy_records.append((epoch, batch, float(metric[1])))
            self.assertEqual(len(legacy_records), len(history))
            self.assertEqual([(e, b) for e, b, _ in legacy_records], [(1, 1), (2, 1)])
            for (_, _, auc), record in zip(legacy_records, history):
                self.assertAlmostEqual(auc, record["AUC1"], places=6)
            self.assertEqual(sum("epoch_summary=" in line for line in log_lines), 2)
            self.assertIn("device=cpu amp=False epochs=2 seed=234 batch_size=1", log_lines[0])
            self.assertTrue(any("model_saved=" in line for line in log_lines))
            self.assertIn("checkpoint_resumed=", (resumed / "train.log").read_text())


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
        args = training_parser().parse_args(["--output-dir", "unused", "--max-epoch", "1"])
        with self.assertRaises(ValueError):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
