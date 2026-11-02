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


from experiments.msc_branch.prototype_memory import CNNVisualMemory
from experiments.cnn_memory.tests.fixtures import mocked_clip

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
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
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
        cls.clip_patch = mocked_clip()
        cls.clip_patch.__enter__()
        cls.module = importlib.import_module("experiments.msc_branch.prototype_control")
        cls.base_class = importlib.import_module("model").CLIPVAD

    @classmethod
    def tearDownClass(cls):
        cls.clip_patch.__exit__(None, None, None)

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
