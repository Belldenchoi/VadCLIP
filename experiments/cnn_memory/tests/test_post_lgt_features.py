"""Cached LGT inputs must bypass the trunk in training and inference."""

import contextlib
import io
import json
from pathlib import Path
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
from experiments.cnn_memory.future_features import future_feature_mse
from experiments.cnn_memory.tests import test_cnn_memory as fixture


class PostLGTFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.ModelIntegrationTests.setUpClass()
        cls.module = fixture.ModelIntegrationTests.module
        cls.base_class = fixture.ModelIntegrationTests.base_class

    @classmethod
    def tearDownClass(cls):
        fixture.ModelIntegrationTests.tearDownClass()

    def make_model(self, kind="post_lgt", mode="future_mse_joint", branch="both"):
        torch.manual_seed(234)
        return self.module.CNNMemoryCLIPVAD(
            3, 8, 8, 8, 1, 1, 4, 1, 1, "cpu", aux_branch=branch,
            aux_mode=mode, input_features=kind, feature_dim=6,
        ).eval()

    def test_cached_features_feed_original_heads_and_cnn_without_any_lgt_call(self):
        model = self.make_model()
        inputs = torch.randn(2, 8, 8), None, ["normal", "abuse", "arrest"], torch.tensor([8, 8])
        # Reference: original heads, supplied the same already-encoded features.
        with patch.object(model, "encode_video", return_value=inputs[0]):
            expected = self.base_class.forward(model, *inputs)
        model.set_memory_active(True)
        seen = []
        hook = model.memory.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        with patch.object(model, "encode_video", side_effect=AssertionError("LGT ran twice")), \
                patch.object(model.disAdj, "forward", side_effect=AssertionError("Online graph ran")), \
                patch.object(model.memory.predictor, "forward", side_effect=AssertionError("Predictor ran in eval")):
            actual = model(*inputs, return_shared_features=True)
        hook.remove()
        torch.testing.assert_close(actual[-1], inputs[0], rtol=0, atol=0)
        torch.testing.assert_close(seen[0], actual[-1], rtol=0, atol=0)
        for a, b in zip(actual[:3], expected):
            torch.testing.assert_close(a, b)

    def test_padding_nan_and_empty_chunks_are_safe_for_heads_mil_and_mse(self):
        from ucf_train import CLAS2, CLASM
        model = self.make_model().train()
        model.set_memory_active(True)
        with torch.no_grad():
            model.memory.score_head.weight.normal_(0, .1)
        lengths = torch.tensor([8, 5, 0])
        clean = torch.randn(3, 8, 8)
        clean[1, 5:] = 0
        clean[2] = 0
        dirty = clean.clone()
        dirty[1, 5:] = float("nan")
        dirty[2] = float("nan")
        dirty.requires_grad_()
        prompt = ["normal", "abuse", "arrest"]
        expected = model(clean, None, prompt, lengths, return_aux=True)
        with patch.object(model, "encode_video", side_effect=AssertionError("LGT ran twice")):
            actual = model(dirty, None, prompt, lengths, return_aux=True)
        for a, b in zip(actual[:3], expected[:3]):
            self.assertTrue(bool(torch.isfinite(a).all()))
            torch.testing.assert_close(a, b)
        labels = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        loss = CLAS2(actual[1][:2], labels, lengths[:2], "cpu")
        loss += CLASM(actual[2][:2], labels, lengths[:2], "cpu")
        loss += .1 * CLAS2(actual[3]["memory_logits"][:2], labels, lengths[:2], "cpu")
        mse, pairs = future_feature_mse(actual[3]["future_predictions"], dirty, lengths)
        self.assertEqual(pairs, 11)
        (loss + .1 * mse).backward()
        self.assertTrue(bool(torch.isfinite(dirty.grad).all()))
        self.assertEqual(int(torch.count_nonzero(dirty.grad[1, 5:])), 0)
        self.assertEqual(int(torch.count_nonzero(dirty.grad[2])), 0)
        for parameter in (model.classifier.weight, model.mlp1.c_fc.weight,
                          model.memory.encoder.conv1.weight, model.memory.predictor.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_lgt_is_frozen_and_excluded_from_optimizer_while_heads_train(self):
        model = self.make_model().train()
        optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        for name in model.LGT_MODULES:
            for parameter in getattr(model, name).parameters():
                self.assertFalse(parameter.requires_grad)
                self.assertNotIn(id(parameter), optimized)
        for module in (model.classifier, model.mlp1, model.mlp2,
                       model.text_prompt_embeddings, model.memory):
            self.assertTrue(all(p.requires_grad for p in module.parameters()))
            self.assertTrue(all(id(p) in optimized for p in module.parameters()))
        legacy = self.make_model("clip")
        self.assertTrue(legacy.temporal.resblocks[0].attn.in_proj_weight.requires_grad)

    def test_post_lgt_residual_preserves_original_visual_prompt(self):
        model = self.make_model()
        visual = torch.randn(2, 8, 8)
        prompt, lengths = ["normal", "abuse", "arrest"], torch.tensor([8, 8])
        baseline = model(visual, None, prompt, lengths)
        model.set_memory_active(True)
        with torch.no_grad():
            model.memory.score_head.bias.fill_(2)
        actual = model(visual, None, prompt, lengths)
        torch.testing.assert_close(actual[1], baseline[1] + .2)
        torch.testing.assert_close(actual[2][..., 0], baseline[2][..., 0], rtol=0, atol=0)
        torch.testing.assert_close(actual[2][..., 1:], baseline[2][..., 1:] + .2)

    def test_checkpoint_reconstruction_saves_input_kind_and_legacy_defaults_to_clip(self):
        args = training_parser().parse_args([
            "--output-dir", "unused", "--aux-mode", "future_mse_joint",
            "--input-features", "post_lgt", "--memory-start-epoch", "1",
        ])
        validate_args(args)
        config = model_config(args)
        self.assertEqual(config["input_features"], "post_lgt")
        model = self.make_model()
        model.set_memory_active(True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save(checkpoint_payload(model, config, args, 1, 1, {}, "best"), path)
            saved = load_checkpoint(path)
        with patch.object(self.module, "CNNMemoryCLIPVAD",
                          side_effect=lambda *a, **kw: model if kw["input_features"] == "post_lgt" else None) as build:
            restored = self.module.build_model(saved["model_config"], "cpu")
            self.assertIs(restored, model)
            restored.load_state_dict(saved["model_state_dict"], strict=True)
            legacy_config = dict(config)
            legacy_config.pop("input_features")
            self.module.build_model(legacy_config, "cpu")
            self.assertEqual(build.call_args.kwargs["input_features"], "clip")

    def test_evaluator_handles_short_exact_and_partial_cached_chunks_without_lgt(self):
        from experiments.cnn_memory.evaluation import evaluate
        from utils.tools import process_split
        model = self.make_model()
        model.set_memory_active(True)
        loader, truth = [], []
        for index, length in enumerate((5, 8, 11)):
            array = np.random.default_rng(index).normal(size=(length, 8)).astype(np.float32)
            split, count = process_split(array, 8)
            loader.append((torch.from_numpy(split).unsqueeze(0), ["Normal" if index == 0 else "Abuse"],
                           torch.tensor([count])))
            truth.extend([int(index != 0)] * (length * 16))
        with patch.object(model, "encode_video", side_effect=AssertionError("LGT ran in evaluator")), \
                patch.object(model.memory.predictor, "forward", side_effect=AssertionError("MSE ran in evaluator")), \
                patch("experiments.topk_variants.test_ucf.dmAP", return_value=([0.] * 5, [.1, .2, .3, .4, .5])):
            metrics = evaluate(model, loader, 8, ["normal", "abuse", "arrest"], np.array(truth),
                               np.zeros((3, 1, 2)), np.array(["Normal", "Abuse", "Abuse"]), "cpu", lambda s: None)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_evaluate_and_predict_entrypoints_use_checkpoint_input_kind_on_cpu(self):
        from experiments.cnn_memory import evaluate, predict

        class CachedDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("memory_active", torch.tensor(True))

            def forward(self, visual, mask, text, lengths):
                return torch.zeros(14, 8), visual[..., :1], visual[..., :14]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = root / "normal.npy", root / "abnormal.npy"
            for path, length, value in zip(paths, (5, 9), (-.5, .5)):
                np.save(path, np.full((length, 512), value, dtype=np.float32))
            csv = root / "features.csv"
            csv.write_text(f"path,label\n{paths[0]},Normal\n{paths[1]},Abuse\n", encoding="utf-8")
            gt, segments, labels = root / "gt.npy", root / "segments.npy", root / "labels.npy"
            np.save(gt, np.repeat([0] * 5 + [1] * 9, 16))
            np.save(segments, np.zeros((2, 1, 2)))
            np.save(labels, np.array(["Normal", "Abuse"]))
            args = training_parser().parse_args([
                "--output-dir", str(root), "--input-features", "post_lgt",
                "--aux-mode", "future_mse_joint", "--visual-length", "8", "--attn-window", "4",
            ])
            checkpoint = root / "best.pt"
            torch.save(checkpoint_payload(CachedDetector(), model_config(args), args, 1, 1, {}, "best"), checkpoint)
            calls = []

            def build(config, device):
                calls.append((config["input_features"], device))
                return CachedDetector()

            eval_args = ["evaluate.py", "--checkpoint", str(checkpoint), "--test-list", str(csv),
                         "--gt-path", str(gt), "--gt-segment-path", str(segments), "--gt-label-path", str(labels)]
            pred_args = ["predict.py", "--checkpoint", str(checkpoint), "--video-list", str(csv),
                         "--output-path", str(root / "prediction.json")]
            with patch.object(self.module, "build_model", side_effect=build), \
                    patch("torch.cuda.is_available", return_value=False), \
                    patch("experiments.topk_variants.test_ucf.dmAP", return_value=([0.] * 5, [.1, .2, .3, .4, .5])), \
                    contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", eval_args):
                    evaluate.main()
                with patch.object(sys, "argv", pred_args):
                    predict.main()
            self.assertEqual(calls, [("post_lgt", "cpu"), ("post_lgt", "cpu")])
            records = json.loads((root / "prediction.json").read_text())
            self.assertFalse(records["ground_truth_used"])
            self.assertEqual([r["predicted_label"] for r in records["videos"]], ["Normal", "Abnormal"])

    def test_invalid_cached_dimensions_are_rejected(self):
        model = self.make_model()
        for visual in (torch.zeros(8, 8), torch.zeros(2, 8, 7)):
            with self.assertRaises(ValueError):
                model(visual, None, ["normal", "abuse", "arrest"], torch.tensor([8, 8]))


if __name__ == "__main__":
    unittest.main()
