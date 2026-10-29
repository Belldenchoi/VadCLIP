"""CPU fixtures for MSE learning, causal input, original C/A flow and resume."""

import contextlib
import argparse
import ast
import copy
import importlib
import io
import json
import re
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.cnn_memory.future_features import (
    CausalFeatureEncoder, FutureFeatureAuxiliary, FutureFeatureDataset,
    FutureFeaturePredictor, future_feature_mse, load_pretrained_encoder,
)
from experiments.cnn_memory.config import add_data_options, training_parser, validate_args
from experiments.cnn_memory.checkpoint import load_checkpoint
from experiments.cnn_memory.tests.test_cnn_memory import FakeCLIP, tokenize

torch.set_num_threads(1)


def training_fixture(root):
    rng = np.random.default_rng(234)
    paths = [root / "normal.npy", root / "abnormal.npy", root / "short.npy"]
    for path, length, offset in zip(paths, (5, 9, 1), (-.4, .4, 0)):
        np.save(path, rng.normal(offset, .1, (length, 512)).astype(np.float32))
    csv = root / "train.csv"
    csv.write_text("path,label\n" + "".join(f"{p},{label}\n" for p, label in
                   zip(paths, ("Normal", "Abuse", "Normal"))), encoding="utf-8")
    return csv


def run_pretrain(csv, output, epochs=1, resume=False):
    from experiments.cnn_memory import pretrain
    args = ["pretrain.py", "--train-list", str(csv), "--output-dir", str(output),
            "--visual-length", "8", "--feature-dim", "6", "--batch-size", "1",
            "--max-epoch", str(epochs), "--device", "cpu"]
    if resume:
        args += ["--resume", str(output / "last.pt")]
    with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
        pretrain.main()


class FutureLossTests(unittest.TestCase):
    def test_causal_encoder_cannot_see_future_and_masks_padding_nan(self):
        torch.manual_seed(234)
        encoder = CausalFeatureEncoder(4, 6, 3)
        features = torch.randn(2, 8, 4)
        changed = features.clone()
        changed[:, 4:] += 100
        expected = encoder(features, [8, 8])
        actual = encoder(changed, [8, 8])
        torch.testing.assert_close(actual[:, :4], expected[:, :4], rtol=0, atol=0)
        clean = features.clone()
        clean[0, 5:] = 0
        clean[1] = 0
        dirty = clean.clone()
        dirty[0, 5:] = float("nan")
        dirty[1] = float("nan")
        dirty.requires_grad_()
        torch.testing.assert_close(encoder(dirty, [5, 0]), encoder(clean, [5, 0]))
        encoder(dirty, [5, 0]).square().sum().backward()
        self.assertTrue(bool(torch.isfinite(dirty.grad).all()))
        self.assertEqual(int(torch.count_nonzero(dirty.grad[0, 5:])), 0)
        self.assertEqual(int(torch.count_nonzero(dirty.grad[1])), 0)

    def test_mse_matches_manual_future_targets_and_detaches_them(self):
        features = torch.arange(20, dtype=torch.float32).reshape(2, 5, 2).requires_grad_()
        predictions = torch.zeros_like(features, requires_grad=True)
        loss, pairs = future_feature_mse(predictions, features, [5, 3], horizon=2)
        expected = torch.cat((features[0, 2:5].detach(), features[1, 2:3].detach())).square().mean()
        torch.testing.assert_close(loss, expected)
        self.assertEqual(pairs, 4)
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertEqual(int(torch.count_nonzero(predictions.grad[0, 3:])), 0)
        self.assertEqual(int(torch.count_nonzero(predictions.grad[1, 1:])), 0)

    def test_mse_ignores_nan_padding_and_rejects_invalid_pairs(self):
        features = torch.randn(2, 6, 4)
        features[0, 4:] = float("nan")
        features[1] = float("nan")
        model = FutureFeaturePredictor(4, 6, 3)
        loss, pairs = future_feature_mse(model(features, [4, 0]), features, [4, 0], 2)
        self.assertEqual(pairs, 2)
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        self.assertGreater(float(model.encoder.conv1.weight.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            future_feature_mse(torch.zeros_like(features), features, [1, 0], 2)
        with self.assertRaises(ValueError):
            future_feature_mse(torch.zeros_like(features), features, [4, 0], 0)

    def test_repeated_updates_learn_a_synthetic_future_target(self):
        torch.manual_seed(234)
        model = FutureFeaturePredictor(4, 8, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        features = torch.tensor([[[1., -.5, .3, -.1]] * 8,
                                 [[-.3, .2, -.8, .5]] * 8])
        initial = future_feature_mse(model(features, [8, 8]), features, [8, 8])[0].item()
        for _ in range(35):
            optimizer.zero_grad()
            loss, _ = future_feature_mse(model(features, [8, 8]), features, [8, 8])
            loss.backward()
            optimizer.step()
        final = future_feature_mse(model(features, [8, 8]), features, [8, 8])[0].item()
        self.assertLess(final, initial * .2)  # Learning behavior, not UCF accuracy.


class PretrainingTests(unittest.TestCase):
    def test_markdown_commands_match_cli_and_share_run_configuration(self):
        from experiments.cnn_memory.pretrain import pretraining_parser
        markdown = (REPO / "experiments/cnn_memory/FUTURE_MSE.md").read_text(encoding="utf-8")

        def parser_for(script):
            if script.endswith("/train.py"):
                return training_parser()
            if script.endswith("/pretrain.py"):
                return pretraining_parser()
            module = ast.parse((REPO / script).read_text(encoding="utf-8"))
            main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            prefix = []
            for node in main.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets):
                    break
                prefix.append(node)
            namespace = dict(argparse=argparse, add_data_options=add_data_options, __doc__=ast.get_docstring(module))
            exec(compile(ast.Module(body=prefix, type_ignores=[]), script, "exec"), namespace)
            return namespace["parser"]

        commands = []
        for block in re.findall(r"```python\n(.*?)\n```", markdown, re.S):
            if not block.startswith("!cd"):
                ast.parse(block)
                continue
            lines = block.splitlines()
            self.assertTrue(lines[0].endswith("&& \\"))
            self.assertTrue(lines[1].startswith("python "))
            self.assertTrue(all(line and not line.startswith("!") for line in lines[1:]))
            tokens = shlex.split(" ".join(line.removesuffix("\\") for line in lines)[1:])
            self.assertEqual(tokens[:4], ["cd", "/kaggle/working/VadCLIP", "&&", "python"])
            args = parser_for(tokens[4]).parse_args(tokens[5:])
            commands.append((Path(tokens[4]).name, args))
        self.assertEqual([name for name, _ in commands], ["pretrain.py", "train.py", "evaluate.py", "train.py", "predict.py"])
        pretrain, detection, evaluation, control, prediction = [args for _, args in commands]
        validate_args(detection)
        validate_args(control)
        self.assertEqual(detection.pretrained_cnn, pretrain.output_dir + "/encoder.pt")
        self.assertEqual(evaluation.checkpoint, detection.output_dir + "/best.pt")
        self.assertEqual(prediction.checkpoint, evaluation.checkpoint)
        self.assertEqual(detection.aux_mode, "future_mse")
        self.assertEqual(control.aux_branch, "none")
        self.assertEqual(detection.seed, pretrain.seed)
        self.assertEqual(detection.cnn_kernel, pretrain.cnn_kernel)
        self.assertEqual(detection.feature_dim, pretrain.feature_dim)
        self.assertEqual(detection.train_list, pretrain.train_list)
        self.assertEqual(control.train_list, detection.train_list)
        self.assertEqual(control.test_list, detection.test_list)
        self.assertEqual(control.memory_start_epoch, detection.memory_start_epoch)

    def test_video_gate_ignores_high_frame_peaks_when_video_is_normal(self):
        from experiments.cnn_memory.predict import prediction_record
        scores = np.zeros(32)
        scores[:3] = [.95, .2, .2]
        normal = prediction_record(scores, video_threshold=.7, frame_threshold=.9)
        self.assertEqual(normal["predicted_label"], "Normal")
        self.assertEqual(normal["segments"], [])
        abnormal = prediction_record([.9, .8, .1, .95], .5, .7)
        self.assertEqual(abnormal["predicted_label"], "Abnormal")
        self.assertEqual([(s["start_frame"], s["end_frame_exclusive"]) for s in abnormal["segments"]],
                         [(0, 32), (48, 64)])
        self.assertEqual(abnormal["frame_repeat"], 16)

    def test_dataset_preserves_detector_grid_and_skips_unusable_clips(self):
        from experiments.cnn_memory import bootstrap
        from utils.tools import process_feat
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = training_fixture(root)
            dataset = FutureFeatureDataset(csv, visual_length=8, horizon=1)
            self.assertEqual((len(dataset), dataset.skipped), (2, 1))
            for i, name in enumerate(("normal.npy", "abnormal.npy")):
                expected, length = process_feat(np.load(root / name), 8)
                actual, actual_length = dataset[i]
                np.testing.assert_array_equal(actual.numpy(), expected)
                self.assertEqual(actual_length, length)

    def test_pretrain_resume_and_encoder_transfer_are_self_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = training_fixture(root)
            continuous, resumed = root / "continuous", root / "resumed"
            run_pretrain(csv, continuous, 2)
            run_pretrain(csv, resumed)
            run_pretrain(csv, resumed, 2, resume=True)
            expected = torch.load(continuous / "last.pt", weights_only=True)
            actual = torch.load(resumed / "last.pt", weights_only=True)
            self.assertEqual(expected["global_step"], actual["global_step"])
            for section in ("encoder_state_dict", "predictor_state_dict"):
                for key, tensor in actual[section].items():
                    torch.testing.assert_close(tensor, expected[section][key], rtol=0, atol=0)
            self.assertEqual((continuous / "metrics.jsonl").read_text(), (resumed / "metrics.jsonl").read_text())
            memory = FutureFeatureAuxiliary(512, 6, 14, 3)
            info = load_pretrained_encoder(memory, resumed / "encoder.pt", 8)
            self.assertEqual(info["epoch"], 2)
            self.assertEqual(info["horizon"], 1)
            self.assertEqual(len(info["sha256"]), 64)
            self.assertTrue(all(not p.requires_grad for p in memory.encoder.parameters()))
            artifact = torch.load(resumed / "encoder.pt", weights_only=True)
            self.assertNotIn("predictor_state_dict", artifact)
            with self.assertRaises(ValueError):
                load_pretrained_encoder(FutureFeatureAuxiliary(512, 7, 14, 3), resumed / "encoder.pt", 8)
            with self.assertRaises(ValueError):
                load_pretrained_encoder(memory, resumed / "encoder.pt", 16)
            csv.write_text(csv.read_text() + "\n")
            with self.assertRaises(ValueError):
                run_pretrain(csv, resumed, 3, resume=True)

    def test_future_mode_requires_pretraining_but_old_options_still_work(self):
        base = ["--output-dir", "unused"]
        validate_args(training_parser().parse_args(base))
        with self.assertRaises(ValueError):
            validate_args(training_parser().parse_args(base + ["--aux-mode", "future_mse"]))
        validate_args(training_parser().parse_args(base + ["--aux-mode", "future_mse", "--aux-branch", "none"]))
        validate_args(training_parser().parse_args(base + ["--aux-mode", "future_mse", "--resume", "last.pt"]))

    def test_pretrain_help_outside_repo_does_not_import_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", str(REPO / "experiments/cnn_memory/pretrain.py"), "--help"],
                                    cwd=directory, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--horizon", result.stdout)


class FutureModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake = types.ModuleType("clip")
        fake.clip = types.SimpleNamespace(load=lambda *a: (FakeCLIP(), None), tokenize=tokenize)
        cls.clip_patch = patch.dict(sys.modules, {"clip": fake})
        cls.clip_patch.start()
        cls.module = importlib.import_module("experiments.cnn_memory.cnn_memory_model")
        cls.base_class = importlib.import_module("model").CLIPVAD

    @classmethod
    def tearDownClass(cls):
        cls.clip_patch.stop()

    def make_model(self, branch="both", mode="future_mse"):
        torch.manual_seed(234)
        model = self.module.CNNMemoryCLIPVAD(3, 8, 8, 8, 1, 1, 4, 1, 1, "cpu",
                                           aux_branch=branch, aux_mode=mode, feature_dim=6)
        model.disAdj.forward = lambda batch, length: torch.eye(length).expand(batch, -1, -1)
        return model.eval()

    def test_initial_identity_raw_input_and_visual_prompt_isolation(self):
        model = self.make_model("c")
        inputs = (torch.randn(2, 8, 8), None, ["normal", "abuse", "arrest"], torch.tensor([8, 5]))
        expected = self.base_class.forward(model, *inputs)
        model.set_memory_active(True)
        for a, b in zip(model(*inputs), expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        seen = []
        handle = model.memory.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        with torch.no_grad():
            model.memory.score_head.bias.fill_(2)
        actual = model(*inputs)
        handle.remove()
        torch.testing.assert_close(seen[0], inputs[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(actual[1][0], expected[1][0] + .2)
        self.assertFalse(hasattr(model.memory, "prototypes"))

    def test_both_residuals_keep_normal_logit_and_abnormal_class_differences(self):
        model = self.make_model()
        inputs = (torch.randn(2, 8, 8), None, ["normal", "abuse", "arrest"], torch.tensor([8, 5]))
        original = model(*inputs)
        model.set_memory_active(True)
        with torch.no_grad():
            model.memory.score_head.bias.fill_(2)
        _, c, a, auxiliary = model(*inputs, return_aux=True)
        torch.testing.assert_close(c, original[1] + .1 * auxiliary["memory_logits"])
        torch.testing.assert_close(a[..., 0], original[2][..., 0], rtol=0, atol=0)
        torch.testing.assert_close(a[..., 1:], original[2][..., 1:] + .1 * auxiliary["memory_logits"])
        self.assertTrue(bool((a.softmax(-1)[0, :, 0] < original[2].softmax(-1)[0, :, 0]).all()))
        torch.testing.assert_close(a[..., 1] - a[..., 2], original[2][..., 1] - original[2][..., 2])
        self.assertEqual(int(torch.count_nonzero(auxiliary["memory_logits"][1, 5:])), 0)
        restored = self.make_model()
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        for a, b in zip(restored(*inputs), model(*inputs)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_mil_updates_scorer_and_original_classifier_but_not_frozen_encoder(self):
        from experiments.cnn_memory import bootstrap
        from ucf_train import CLAS2, CLASM
        model = self.make_model().train()
        model.set_memory_active(True)
        before = copy.deepcopy(model.memory.encoder.state_dict())
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
        inputs = (torch.randn(2, 8, 8), None, ["normal", "abuse", "arrest"], torch.tensor([8, 5]))
        _, c, a, aux = model(*inputs, return_aux=True)
        labels = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        loss = CLAS2(c, labels, inputs[3], "cpu") + CLASM(a, labels, inputs[3], "cpu")
        loss = loss + .1 * CLAS2(aux["memory_logits"], labels, inputs[3], "cpu")
        loss.backward()
        self.assertGreater(float(model.memory.score_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.classifier.weight.grad.abs().sum()), 0)
        optimizer.step()
        for name, parameter in model.memory.encoder.named_parameters():
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_two_stage_detection_entrypoint_resumes_without_pretraining_file(self):
        from experiments.cnn_memory import train as entrypoint
        from experiments.cnn_memory import cnn_memory_model

        class TinyFutureDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(512, 8)
                self.classifier = nn.Linear(8, 1)
                self.text = nn.Parameter(torch.randn(14, 8))
                self.memory = FutureFeatureAuxiliary(512, 6, 14, 3)
                self.pretraining_info = None
                self.register_buffer("memory_active", torch.tensor(False))

            def set_memory_active(self, active):
                self.memory_active.fill_(active)

            def forward(self, visual, mask, text, lengths, return_aux=False):
                hidden = self.projection(visual.float())
                c, a = self.classifier(hidden), hidden @ self.text.T
                aux = self.memory(visual, lengths) if bool(self.memory_active) else None
                if aux is not None:
                    c, a = c + .1 * aux["delta_c"], a + .1 * aux["delta_a"]
                result = self.text, c, a
                return (*result, aux) if return_aux else result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = training_fixture(root)
            # Use only the first two rows so detection GT follows exactly two videos.
            csv.write_text("\n".join(csv.read_text().splitlines()[:3]) + "\n")
            run_pretrain(csv, root / "pretrain")
            artifact = root / "pretrain/encoder.pt"
            expected_encoder = torch.load(artifact, weights_only=True)["encoder_state_dict"]
            gt, segments, labels = root / "gt.npy", root / "segments.npy", root / "labels.npy"
            np.save(gt, np.repeat([0] * 5 + [1] * 9, 16))
            np.save(segments, np.zeros((2, 1, 2)))
            np.save(labels, np.array(["Normal", "Abuse"]))
            common = ["--train-list", str(csv), "--test-list", str(csv), "--gt-path", str(gt),
                      "--gt-segment-path", str(segments), "--gt-label-path", str(labels),
                      "--aux-mode", "future_mse", "--batch-size", "1", "--visual-length", "8",
                      "--attn-window", "4", "--feature-dim", "6", "--eval-steps", "1", "--memory-start-epoch", "1"]

            def run(output, epochs, resume=False):
                args = ["train.py", *common, "--output-dir", str(output), "--max-epoch", str(epochs)]
                args += ["--resume", str(output / "last.pt")] if resume else ["--pretrained-cnn", str(artifact)]
                with patch.object(sys, "argv", args), patch.object(entrypoint, "training_device", return_value="cpu"), \
                        patch.object(cnn_memory_model, "build_model", side_effect=lambda *a: TinyFutureDetector()), \
                        patch("experiments.topk_variants.test_ucf.dmAP", return_value=([0.] * 5, [.1, .2, .3, .4, .5])), \
                        contextlib.redirect_stdout(io.StringIO()):
                    entrypoint.main()

            run(root / "continuous", 2)
            run(root / "resumed", 1)
            artifact.unlink()  # Resume must use its own embedded encoder, not this file.
            run(root / "resumed", 2, resume=True)
            expected = load_checkpoint(root / "continuous/last.pt")
            actual = load_checkpoint(root / "resumed/last.pt")
            self.assertEqual(expected["best_metrics"], actual["best_metrics"])
            self.assertEqual(expected["pretraining_info"], actual["pretraining_info"])
            self.assertIsNotNone(actual["pretraining_info"])
            for key, tensor in actual["model_state_dict"].items():
                torch.testing.assert_close(tensor, expected["model_state_dict"][key], rtol=0, atol=0)
            for key, tensor in expected_encoder.items():
                torch.testing.assert_close(tensor, actual["model_state_dict"]["memory.encoder." + key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
