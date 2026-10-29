"""Joint CNN/C/A learning, inference isolation and resumable training fixtures."""

import argparse
import ast
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
from experiments.cnn_memory.config import add_data_options, model_config, training_parser, validate_args
from experiments.cnn_memory.checkpoint import load_checkpoint
from experiments.cnn_memory.future_features import JointFutureFeatureAuxiliary, future_feature_mse
from experiments.cnn_memory.tests import test_future_features as future_fixture

torch.set_num_threads(1)


class JointLossTests(unittest.TestCase):
    def test_no_future_pairs_skips_only_mse_without_reading_padding_nan(self):
        predicted = torch.full((2, 4, 8), float("nan"), requires_grad=True)
        loss, pairs = future_feature_mse(predicted, torch.zeros_like(predicted), [1, 0], 1, allow_empty=True)
        self.assertEqual(pairs, 0)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(predicted.grad)), 0)

    def test_joint_cnn_is_causal_and_predictor_is_not_used_in_eval(self):
        module = JointFutureFeatureAuxiliary(8, 6, 3, 3).train()
        features = torch.randn(2, 8, 8)
        expected = module(features, [8, 5], return_prediction=True)
        changed = features.clone()
        changed[:, 4:] += 100
        actual = module(changed, [8, 5], return_prediction=True)
        torch.testing.assert_close(expected["future_predictions"][:, :4], actual["future_predictions"][:, :4], rtol=0, atol=0)
        module.eval()
        with patch.object(module.predictor, "forward", side_effect=AssertionError("Prediction head ran in eval")):
            result = module(features, [8, 5], return_prediction=True)
        self.assertNotIn("future_predictions", result)
        self.assertTrue(all(p.requires_grad for p in module.encoder.parameters()))

    def test_joint_configuration_from_scratch_and_invalid_options(self):
        base = ["--output-dir", "unused", "--aux-mode", "future_mse_joint", "--memory-start-epoch", "1"]
        args = training_parser().parse_args(base)
        validate_args(args)
        self.assertIsNone(args.pretrained_cnn)
        self.assertEqual(model_config(args)["future_horizon"], 1)
        for flags in (["--future-horizon", "0"], ["--future-horizon", "256"],
                      ["--mse-loss-weight", "nan"], ["--mse-loss-weight", "-1"],
                      ["--pretrained-cnn", "encoder.pt"]):
            with self.assertRaises(ValueError):
                validate_args(training_parser().parse_args(base + flags))
        validate_args(training_parser().parse_args(base + ["--mse-loss-weight", "0"]))


class JointModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        future_fixture.FutureModelTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        future_fixture.FutureModelTests.tearDownClass()

    def model(self, branch="both"):
        fixture = future_fixture.FutureModelTests()
        return fixture.make_model(branch, mode="future_mse_joint")

    def test_original_forward_identity_visual_prompt_and_checkpoint(self):
        model = self.model("c")
        inputs = (torch.randn(2, 8, 8), None, ["normal", "abuse", "arrest"], torch.tensor([8, 5]))
        expected = future_fixture.FutureModelTests.base_class.forward(model, *inputs)
        model.set_memory_active(True)
        for a, b in zip(model(*inputs), expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with torch.no_grad():
            model.memory.score_head.bias.fill_(2)
        actual = model(*inputs)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        restored = self.model("c")
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        with patch.object(restored.memory.predictor, "forward", side_effect=AssertionError("MSE in inference")):
            for a, b in zip(restored(*inputs), actual):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_shared_encoder_receives_mil_and_mse_gradients(self):
        from ucf_train import CLAS2, CLASM
        model = self.model().train()
        model.set_memory_active(True)
        # The real initialization is zero (baseline parity); after its first
        # scoring update the task losses must also reach the shared encoder.
        with torch.no_grad():
            model.memory.score_head.weight.normal_(0, .1)
        features, lengths = torch.randn(2, 8, 8), torch.tensor([8, 5])
        _, c, a, aux = model(features, None, ["normal", "abuse", "arrest"], lengths, return_aux=True)
        labels = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        mil = CLAS2(c, labels, lengths, "cpu") + CLASM(a, labels, lengths, "cpu")
        mse, pairs = future_feature_mse(aux["future_predictions"], features, lengths)
        self.assertEqual(pairs, 11)
        weight = model.memory.encoder.conv1.weight
        task_grad = torch.autograd.grad(mil, weight, retain_graph=True)[0]
        mse_grad = torch.autograd.grad(mse, weight, retain_graph=True)[0]
        self.assertGreater(float(task_grad.abs().sum()), 0)
        self.assertGreater(float(mse_grad.abs().sum()), 0)
        self.assertIsNone(torch.autograd.grad(mse, model.classifier.weight, allow_unused=True, retain_graph=True)[0])
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
        before = weight.detach().clone()
        (mil + .1 * mse).backward()
        self.assertGreater(float(model.memory.predictor.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.classifier.weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(weight, before))

    def test_control_and_joint_preserve_original_initialization_rng(self):
        control = self.model("none")
        rng = torch.get_rng_state()
        joint = self.model()
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in control.state_dict().items():
            torch.testing.assert_close(joint.state_dict()[key], value, rtol=0, atol=0)


class JointEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        future_fixture.FutureModelTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        future_fixture.FutureModelTests.tearDownClass()

    def test_joint_train_updates_cnn_logs_mse_and_resumes_exactly(self):
        from experiments.cnn_memory import train as entrypoint
        from experiments.cnn_memory import cnn_memory_model

        class TinyJointDetector(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.projection = nn.Linear(512, 8)
                self.classifier = nn.Linear(8, 1)
                self.text = nn.Parameter(torch.randn(14, 8))
                self.memory = JointFutureFeatureAuxiliary(512, 6, 14, 3)
                self.pretraining_info = None
                self.register_buffer("memory_active", torch.tensor(False))

            def set_memory_active(self, active):
                self.memory_active.fill_(active)

            def forward(self, visual, mask, text, lengths, return_aux=False):
                features = self.projection(visual.float())
                c, a = self.classifier(features), features @ self.text.T
                auxiliary = self.memory(visual, lengths, return_prediction=return_aux) if bool(self.memory_active) else None
                if auxiliary is not None:
                    c, a = c + .1 * auxiliary["delta_c"], a + .1 * auxiliary["delta_a"]
                result = self.text, c, a
                return (*result, auxiliary) if return_aux else result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = future_fixture.training_fixture(root)
            csv.write_text("\n".join(csv.read_text().splitlines()[:3]) + "\n")
            gt, segments, labels = root / "gt.npy", root / "segments.npy", root / "labels.npy"
            np.save(gt, np.repeat([0] * 5 + [1] * 9, 16))
            np.save(segments, np.zeros((2, 1, 2)))
            np.save(labels, np.array(["Normal", "Abuse"]))
            common = ["--train-list", str(csv), "--test-list", str(csv), "--gt-path", str(gt),
                      "--gt-segment-path", str(segments), "--gt-label-path", str(labels),
                      "--aux-mode", "future_mse_joint", "--batch-size", "1", "--visual-length", "8",
                      "--attn-window", "4", "--feature-dim", "6", "--eval-steps", "1", "--memory-start-epoch", "1"]
            initial = {}

            def build(config, device):
                model = TinyJointDetector(config)
                initial.update(copy.deepcopy(model.memory.encoder.state_dict()))
                return model

            def run(output, epochs, resume=False, extra=()):
                args = ["train.py", *common, "--output-dir", str(output), "--max-epoch", str(epochs), *extra]
                if resume:
                    args += ["--resume", str(output / "last.pt")]
                with patch.object(sys, "argv", args), patch.object(entrypoint, "training_device", return_value="cpu"), \
                        patch.object(cnn_memory_model, "build_model", side_effect=build), \
                        patch("experiments.topk_variants.test_ucf.dmAP", return_value=([0.] * 5, [.1, .2, .3, .4, .5])), \
                        contextlib.redirect_stdout(io.StringIO()):
                    entrypoint.main()

            run(root / "continuous", 2)
            run(root / "resumed", 1)
            run(root / "resumed", 2, resume=True)
            expected = load_checkpoint(root / "continuous/last.pt")
            actual = load_checkpoint(root / "resumed/last.pt")
            self.assertIsNone(actual["pretraining_info"])
            self.assertEqual(actual["best_metrics"], expected["best_metrics"])
            for key, tensor in actual["model_state_dict"].items():
                torch.testing.assert_close(tensor, expected["model_state_dict"][key], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(value, actual["model_state_dict"]["memory.encoder." + key])
                                for key, value in initial.items()))
            logs = (root / "continuous/train.log").read_text()
            self.assertIn("encoder_frozen=False", logs)
            self.assertEqual(logs.count("mse_pairs=11"), 4)  # Batch and summary for both epochs.
            self.assertIn("avg_loss_mse=", logs)
            for line in logs.splitlines():
                if re.search(r"epoch=\d+/\d+ batch=", line):
                    values = dict((key, float(value)) for key, value in re.findall(
                        r"\b(loss|loss1|loss2|loss3|weighted_memory|weighted_mse)=([\d.]+)", line))
                    self.assertAlmostEqual(values['loss'], sum(values[key] for key in
                                           ('loss1', 'loss2', 'loss3', 'weighted_memory', 'weighted_mse')), delta=.0004)
            with self.assertRaises(ValueError):
                run(root / "resumed", 3, resume=True, extra=("--future-horizon", "2"))

    def test_joint_markdown_commands_match_cli_and_use_no_pretrain(self):
        markdown = (REPO / "experiments/cnn_memory/JOINT_MSE.md").read_text(encoding="utf-8")
        commands = []
        values = dict(TRAIN_LIST='/kaggle/train.csv', TEST_LIST='/kaggle/test.csv',
                      GT_DIR='/kaggle/gt', SEED=234)
        for block in re.findall(r"```python\n(.*?)\n```", markdown, re.S):
            if not block.startswith('!cd'):
                continue
            text = ' '.join(line.removesuffix('\\') for line in block.format_map(values).splitlines())[1:]
            parts = re.split(r'\s+&&\s+', text)
            self.assertEqual(shlex.split(parts[0]), ['cd', '/kaggle/working/VadCLIP'])
            for part in parts[1:]:
                tokens = shlex.split(part)
                self.assertEqual(tokens[:2], ['python', '-u'])
                if tokens[2].endswith('/train.py'):
                    args = training_parser().parse_args(tokens[3:])
                    validate_args(args)
                    self.assertEqual(args.aux_mode, 'future_mse_joint')
                    self.assertIsNone(args.pretrained_cnn)
                    self.assertEqual(args.memory_start_epoch, 1)
                else:
                    self.assertTrue(tokens[2].endswith('/evaluate.py'))
                    namespace = dict(argparse=argparse, add_data_options=add_data_options, __doc__='evaluate')
                    module = ast.parse((REPO / tokens[2]).read_text())
                    main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
                    prefix = []
                    for node in main.body:
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in node.targets):
                            break
                        prefix.append(node)
                    exec(compile(ast.Module(body=prefix, type_ignores=[]), tokens[2], 'exec'), namespace)
                    args = namespace['parser'].parse_args(tokens[3:])
                commands.append((tokens[2], args))
        self.assertEqual(len(commands), 3)  # Joint train/evaluate + matched MSE-off control.
        self.assertEqual(commands[1][1].checkpoint, commands[0][1].output_dir + '/best.pt')
        self.assertEqual(commands[2][1].mse_loss_weight, 0.)


if __name__ == "__main__":
    unittest.main()
