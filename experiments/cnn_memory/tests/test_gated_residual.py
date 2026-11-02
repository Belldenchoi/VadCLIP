"""The agreed single-Conv feature fusion, original A path and original loss."""

import contextlib
import copy
import io
import json
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.cnn_memory.checkpoint import checkpoint_payload, load_checkpoint
from experiments.cnn_memory.config import model_config, training_parser, validate_args
from experiments.cnn_memory.feature_fusion import CNNFeatureEnhancement
from experiments.cnn_memory.tests import fixtures as fixture


class FusionModuleTests(unittest.TestCase):
    def test_single_conv_projection_and_feature_wise_gate_contract(self):
        module = CNNFeatureEnhancement()
        self.assertEqual(sum(isinstance(layer, nn.Conv1d) for layer in module.modules()), 1)
        self.assertEqual((module.conv.in_channels, module.conv.out_channels), (512, 128))
        self.assertEqual((module.conv.kernel_size, module.conv.stride, module.conv.dilation,
                          module.conv.padding), ((3,), (1,), (1,), (1,)))
        self.assertEqual((module.proj.in_features, module.proj.out_features), (128, 512))
        self.assertEqual((module.gate.in_features, module.gate.out_features), (1024, 512))
        torch.testing.assert_close(module.gate.bias, torch.full((512,), -2.))
        features = torch.randn(2, 6, 512)
        self.assertEqual(module(features, [6, 4]).shape, features.shape)
        self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in module.modules()), 2)

    def test_gate_opens_and_closes_the_feature_residual(self):
        module = CNNFeatureEnhancement(8, 4)
        features = torch.randn(2, 6, 8)
        with torch.no_grad():
            module.gate.weight.zero_()
            module.gate.bias.fill_(-100)
        torch.testing.assert_close(module(features, [6, 6]), features, rtol=0, atol=0)
        projected = module.proj(nn.functional.gelu(module.conv(features.transpose(1, 2))).transpose(1, 2))
        with torch.no_grad():
            module.gate.bias.fill_(100)
        torch.testing.assert_close(module(features, [6, 6]), features + projected, rtol=0, atol=0)

    def test_masking_matches_convolution_on_valid_segments_and_handles_empty_chunks(self):
        module = CNNFeatureEnhancement(8, 4)
        features = torch.randn(3, 9, 8)
        features[1, 5:] = float("nan")
        features[2] = float("nan")
        features.requires_grad_()
        actual = module(features, [9, 5, 0])
        single = module(features[1:2, :5], [5])
        torch.testing.assert_close(actual[1, :5], single[0])
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertEqual(int(torch.count_nonzero(actual[1, 5:])), 0)
        self.assertEqual(int(torch.count_nonzero(actual[2])), 0)
        actual.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(features.grad).all()))
        self.assertEqual(int(torch.count_nonzero(features.grad[1, 5:])), 0)
        self.assertEqual(int(torch.count_nonzero(features.grad[2])), 0)


class FusionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.ModelIntegrationTests.setUpClass()
        cls.module = fixture.ModelIntegrationTests.module

    @classmethod
    def tearDownClass(cls):
        fixture.ModelIntegrationTests.tearDownClass()

    def make_model(self, branch="c", kind="post_lgt", channels=4):
        torch.manual_seed(234)
        model = self.module.CNNMemoryCLIPVAD(
            3, 8, 8, 8, 1, 1, 4, 1, 1, "cpu", aux_branch=branch,
            aux_mode="gated_residual", input_features=kind, cnn_channels=channels,
        ).eval()
        model.disAdj.forward = lambda batch, length: torch.eye(length).expand(batch, -1, -1)
        return model

    def test_only_original_c_classifier_is_used_and_a_is_unchanged(self):
        for kind in ("post_lgt", "clip"):
            with self.subTest(kind=kind):
                model = self.make_model(kind=kind)
                visual, lengths = torch.randn(2, 8, 8), torch.tensor([8, 5])
                prompt = ["normal", "abuse", "arrest"]
                baseline = model(visual, None, prompt, lengths, return_shared_features=True)
                model.set_enhancement_active(True)
                seen = []
                hook = model.classifier.register_forward_pre_hook(lambda layer, args: seen.append(args[0].clone()))
                with patch.object(model, "encode_video", wraps=model.encode_video) as encode:
                    actual = model(visual, None, prompt, lengths, return_shared_features=True)
                hook.remove()
                self.assertEqual(encode.call_count, int(kind == "clip"))
                self.assertEqual(len(seen), 2)  # Same head for original aggregation weights and enhanced C.
                torch.testing.assert_close(actual[2], baseline[2], rtol=0, atol=0)
                torch.testing.assert_close(actual[-1], baseline[-1], rtol=0, atol=0)
                self.assertGreater(float((actual[1] - baseline[1]).detach().abs().max()), 1e-8)
                self.assertIsNone(actual[3])
                self.assertFalse(hasattr(model, "memory"))
                self.assertFalse(hasattr(model.cnn_enhancement, "score_head"))
                self.assertFalse(hasattr(model.cnn_enhancement, "predictor"))
                self.assertEqual([name for name, _ in model.named_modules() if name == "classifier"], ["classifier"])

    def test_original_c_loss_trains_conv_projection_gate_and_online_lgt(self):
        from ucf_train import CLAS2
        for kind in ("post_lgt", "clip"):
            with self.subTest(kind=kind):
                model = self.make_model(kind=kind).train()
                model.set_enhancement_active(True)
                visual, lengths = torch.randn(2, 8, 8), torch.tensor([8, 5])
                _, c, _ = model(visual, None, ["normal", "abuse", "arrest"], lengths)
                labels = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
                CLAS2(c, labels, lengths, "cpu").backward()
                for parameter in (model.cnn_enhancement.conv.weight, model.cnn_enhancement.proj.weight,
                                  model.cnn_enhancement.gate.weight, model.classifier.weight, model.mlp2.c_fc.weight):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                    self.assertGreater(float(parameter.grad.abs().sum()), 0)
                if kind == "clip":
                    self.assertGreater(float(model.linear.weight.grad.abs().sum()), 0)
                else:
                    self.assertFalse(model.linear.weight.requires_grad)
                    self.assertIsNone(model.linear.weight.grad)

    def test_a_loss_has_no_direct_path_to_cnn_fusion(self):
        from ucf_train import CLASM
        model = self.make_model().train()
        model.set_enhancement_active(True)
        visual, lengths = torch.randn(2, 8, 8), torch.tensor([8, 5])
        _, _, a = model(visual, None, ["normal", "abuse", "arrest"], lengths)
        CLASM(a, torch.tensor([[1., 0., 0.], [0., 1., 0.]]), lengths, "cpu").backward()
        self.assertTrue(all(p.grad is None for p in model.cnn_enhancement.parameters()))
        self.assertIsNotNone(model.mlp1.c_fc.weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)

    def test_control_keeps_backbone_and_loader_rng_and_checkpoint_restores_active_fusion(self):
        control = self.make_model("none")
        rng = torch.get_rng_state()
        model = self.make_model()
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in control.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        model.set_enhancement_active(True)
        inputs = torch.randn(2, 8, 8), None, ["normal", "abuse", "arrest"], torch.tensor([8, 8])
        expected = model(*inputs)
        restored = self.make_model()
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        self.assertTrue(bool(restored.enhancement_active))
        for a, b in zip(restored(*inputs), expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_raw_score_evaluator_preserves_a_metrics_with_short_exact_and_partial_chunks(self):
        from experiments.cnn_memory.evaluation import evaluate
        from utils.tools import process_split
        model = self.make_model()
        loader, truth = [], []
        for index, length in enumerate((5, 8, 11)):
            array = np.random.default_rng(index).normal(size=(length, 8)).astype(np.float32)
            split, count = process_split(array, 8)
            loader.append((torch.from_numpy(split).unsqueeze(0), ["Normal" if index == 0 else "Abuse"], torch.tensor([count])))
            truth.extend([int(index != 0)] * (length * 16))
        values = loader, 8, ["normal", "abuse", "arrest"], np.array(truth), np.zeros((3, 1, 2)), np.array(["Normal", "Abuse", "Abuse"]), "cpu"
        with patch.object(model, "encode_video", side_effect=AssertionError("LGT ran on cached features")), \
                patch("experiments.topk_variants.test_ucf.dmAP", return_value=([0.] * 5, [.1, .2, .3, .4, .5])):
            baseline = evaluate(model, *values, log=lambda s: None)
            model.set_enhancement_active(True)
            actual = evaluate(model, *values, log=lambda s: None)
        self.assertTrue(all(np.isfinite(v) for v in actual.values()))
        self.assertEqual((actual["AUC2"], actual["AP2"]), (baseline["AUC2"], baseline["AP2"]))


class FusionEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.ModelIntegrationTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixture.ModelIntegrationTests.tearDownClass()

    def test_configuration_is_c_only_single_kernel_and_serializes_cnn_channels(self):
        base = ["--output-dir", "unused", "--aux-mode", "gated_residual", "--aux-branch", "c"]
        args = training_parser().parse_args(base)
        validate_args(args)
        self.assertEqual(model_config(args)["cnn_channels"], 128)
        for flags in (["--cnn-kernel", "1"], ["--cnn-channels", "0"]):
            with self.assertRaises(ValueError):
                validate_args(training_parser().parse_args(base + flags))
        validate_args(training_parser().parse_args(base + ["--aux-branch", "none"]))
        defaults = training_parser().parse_args(["--output-dir", "unused"])
        validate_args(defaults)
        self.assertEqual((defaults.aux_mode, defaults.aux_branch, defaults.input_features,
                          defaults.cnn_start_epoch), ("gated_residual", "c", "clip", 1))
        for flags in (["--aux-mode", "prototype"], ["--aux-mode", "future_mse"],
                      ["--aux-mode", "future_mse_joint"], ["--aux-branch", "both"],
                      ["--memory-lr", ".01"], ["--mse-loss-weight", "0"],
                      ["--pretrained-cnn", "old.pt"], ["--memory-start-epoch", "1"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training_parser().parse_args(["--output-dir", "unused", *flags])
        module = fixture.ModelIntegrationTests.module
        with patch.object(module, "CNNMemoryCLIPVAD") as constructor:
            module.build_model(model_config(args), "cpu")
        self.assertEqual(constructor.call_args.kwargs["aux_mode"], "gated_residual")
        self.assertEqual(constructor.call_args.kwargs["cnn_channels"], 128)

    def test_kaggle_cells_run_the_agreed_grf_and_matched_baseline(self):
        text = (REPO / "experiments/cnn_memory/GATED_RESIDUAL.md").read_text(encoding="utf-8")
        runs, evaluation = [], None
        for block in re.findall(r"```python\n(.*?)\n```", text, re.S):
            self.assertEqual(block.count("!"), 1)
            parts = re.split(r"\s+&&\s+", " ".join(line.removesuffix("\\") for line in block.splitlines())[1:])
            self.assertEqual(shlex.split(parts.pop(0)), ["cd", "/kaggle/working/VadCLIP"])
            if parts[0].startswith("git "):
                self.assertEqual(shlex.split(parts.pop(0)), ["git", "pull", "--ff-only", "origin", "main"])
            self.assertEqual(len(parts), 1)
            tokens = shlex.split(parts[0])
            self.assertEqual(tokens[:2], ["python", "-u"])
            if tokens[2].endswith("/train.py"):
                args = training_parser().parse_args(tokens[3:])
                validate_args(args)
                self.assertEqual(args.aux_mode, "gated_residual")
                self.assertEqual(args.input_features, "clip")
                self.assertEqual((args.cnn_channels, args.cnn_kernel), (128, 3))
                self.assertEqual(args.cnn_start_epoch, 1)
                self.assertEqual(args.lr, 2e-5)
                runs.append(args)
            else:
                self.assertTrue(tokens[2].endswith("/evaluate.py"))
                evaluation = dict(zip(tokens[3::2], tokens[4::2]))
        self.assertEqual([args.aux_branch for args in runs], ["c", "none"])
        self.assertEqual(evaluation["--checkpoint"], runs[0].output_dir + "/best.pt")
        for key in ("train_list", "test_list", "seed", "lr", "batch_size", "max_epoch", "input_features", "cnn_start_epoch"):
            self.assertEqual(getattr(runs[0], key), getattr(runs[1], key))

    def test_training_uses_only_original_losses_updates_fusion_and_resumes_exactly(self):
        from experiments.cnn_memory import train as entrypoint
        from experiments.cnn_memory import cnn_memory_model
        from ucf_train import CLAS2

        class TinyFusionDetector(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.projection = nn.Linear(512, 8)
                self.classifier = nn.Linear(8, 1)
                self.text = nn.Parameter(torch.randn(14, 8))
                self.cnn_enhancement = CNNFeatureEnhancement(8, config["cnn_channels"])
                self.register_buffer("enhancement_active", torch.tensor(False))

            def set_enhancement_active(self, active):
                self.enhancement_active.fill_(active)

            def forward(self, visual, mask, text, lengths, return_aux=False):
                h = self.projection(visual.float())
                enhanced = self.cnn_enhancement(h, lengths) if bool(self.enhancement_active) else h
                outputs = self.text, self.classifier(enhanced), h @ self.text.T
                return (*outputs, None) if return_aux else outputs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = root / "normal.npy", root / "abnormal.npy"
            rng = np.random.default_rng(234)
            for path, length in zip(paths, (5, 9)):
                np.save(path, rng.normal(size=(length, 512)).astype(np.float32))
            csv = root / "data.csv"
            csv.write_text(f"path,label\n{paths[0]},Normal\n{paths[1]},Abuse\n", encoding="utf-8")
            gt, segments, labels = root / "gt.npy", root / "segments.npy", root / "labels.npy"
            np.save(gt, np.repeat([0] * 5 + [1] * 9, 16))
            np.save(segments, np.zeros((2, 1, 2)))
            np.save(labels, np.array(["Normal", "Abuse"]))
            common = ["--train-list", str(csv), "--test-list", str(csv), "--gt-path", str(gt),
                      "--gt-segment-path", str(segments), "--gt-label-path", str(labels),
                      "--aux-mode", "gated_residual", "--aux-branch", "c", "--input-features", "post_lgt",
                      "--cnn-channels", "4", "--batch-size", "1", "--visual-length", "8", "--attn-window", "4",
                      "--eval-steps", "1", "--cnn-start-epoch", "1"]
            initial = {}

            def build(config, device):
                model = TinyFusionDetector(config)
                initial.update(copy.deepcopy(model.cnn_enhancement.state_dict()))
                return model

            def run(output, epochs, resume=False, extra=()):
                args = ["train.py", *common, "--output-dir", str(output), "--max-epoch", str(epochs), *extra]
                if resume:
                    args += ["--resume", str(output / "last.pt")]
                with patch.object(sys, "argv", args), patch.object(entrypoint, "training_device", return_value="cpu"), \
                        patch.object(cnn_memory_model, "build_model", side_effect=build), \
                        patch("ucf_train.CLAS2", wraps=CLAS2) as binary_loss, \
                        patch("experiments.topk_variants.test_ucf.dmAP", return_value=([0.] * 5, [.1, .2, .3, .4, .5])), \
                        contextlib.redirect_stdout(io.StringIO()):
                    entrypoint.main()
                self.assertEqual(binary_loss.call_count, epochs - int(resume))

            run(root / "continuous", 2)
            run(root / "resumed", 1)
            run(root / "resumed", 2, resume=True)
            # A published GRF v1 checkpoint must resume the same trajectory too.
            run(root / "v1", 1)
            legacy = load_checkpoint(root / "v1/last.pt")
            legacy["format_version"] = 1
            legacy["model_state_dict"]["memory_active"] = legacy["model_state_dict"].pop("enhancement_active")
            legacy["train_args"]["memory_start_epoch"] = legacy["train_args"].pop("cnn_start_epoch")
            for key, value in dict(feature_dim=256, memory_slots=8, memory_temperature=.1,
                                   residual_scale=.1).items():
                legacy["model_config"][key] = legacy["train_args"][key] = value
            legacy["train_args"].update(memory_lr=2e-4, memory_loss_weight=0., mse_loss_weight=0.,
                                        future_horizon=1, pretrained_cnn=None)
            legacy["pretraining_info"] = None
            torch.save(legacy, root / "v1/last.pt")
            run(root / "v1", 2, resume=True)
            expected, actual = [load_checkpoint(root / name / "last.pt") for name in ("continuous", "resumed")]
            migrated = load_checkpoint(root / "v1/last.pt")
            self.assertEqual(migrated["format_version"], 2)
            self.assertEqual(migrated["best_metrics"], expected["best_metrics"])
            self.assertNotIn("memory_lr", migrated["train_args"])
            self.assertNotIn("pretraining_info", migrated)
            for key, tensor in migrated["model_state_dict"].items():
                torch.testing.assert_close(tensor, expected["model_state_dict"][key], rtol=0, atol=0)
            self.assertEqual(actual["model_config"]["cnn_channels"], 4)
            self.assertEqual(actual["model_config"]["aux_mode"], "gated_residual")
            self.assertEqual(len(actual["optimizer_state_dict"]["param_groups"]), 1)
            self.assertEqual(actual["optimizer_state_dict"]["param_groups"][0]["lr"], 2e-5)
            for key, tensor in actual["model_state_dict"].items():
                torch.testing.assert_close(tensor, expected["model_state_dict"][key], rtol=0, atol=0)
            for key in ("conv.weight", "proj.weight", "gate.weight"):
                self.assertFalse(torch.equal(initial[key], actual["model_state_dict"]["cnn_enhancement." + key]))
            logs = (root / "continuous/train.log").read_text()
            self.assertIn("objective=vadclip_original", logs)
            self.assertIn("auxiliary_classifier=False mse=False", logs)
            for line in logs.splitlines():
                if re.search(r"epoch=\d+/\d+ batch=", line):
                    values = dict((k, float(v)) for k, v in re.findall(r"\b(loss|loss1|loss2|loss3)=([\d.]+)", line))
                    self.assertNotIn("memory", line)
                    self.assertNotIn("mse", line)
                    self.assertAlmostEqual(values["loss"], sum(values[k] for k in ("loss1", "loss2", "loss3")), delta=.0003)
            with self.assertRaisesRegex(ValueError, "cnn_channels"):
                run(root / "resumed", 3, resume=True, extra=("--cnn-channels", "8"))


if __name__ == "__main__":
    unittest.main()
