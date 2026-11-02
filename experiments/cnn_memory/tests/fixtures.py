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


@contextlib.contextmanager
def mocked_clip():
    """Replace only CLIP, preserving lazy PyTorch imports across fixtures."""
    package = types.ModuleType("clip")
    package.clip = types.SimpleNamespace(load=lambda *a: (FakeCLIP(), None), tokenize=tokenize)
    missing = object()
    previous = sys.modules.get("clip", missing)
    sys.modules["clip"] = package
    try:
        yield
    finally:
        if previous is missing:
            sys.modules.pop("clip", None)
        else:
            sys.modules["clip"] = previous


class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip_patch = mocked_clip()
        cls.clip_patch.__enter__()
        cls.module = importlib.import_module("experiments.cnn_memory.cnn_memory_model")
        cls.base_class = importlib.import_module("model").CLIPVAD

    @classmethod
    def tearDownClass(cls):
        cls.clip_patch.__exit__(None, None, None)
