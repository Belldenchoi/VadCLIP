"""Causal visual-feature pretraining and a prototype-free auxiliary branch."""

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


def valid_steps(features, lengths):
    if features.ndim != 3:
        raise ValueError("Expected features [B,T,D]")
    lengths = torch.as_tensor(lengths, device=features.device, dtype=torch.long)
    if lengths.shape != (features.shape[0],) or bool(
        ((lengths < 0) | (lengths > features.shape[1])).any()
    ):
        raise ValueError("Lengths must match the batch and stay within [0,T]")
    return torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]


class CausalFeatureEncoder(nn.Module):
    """At step t, only f[t-kernel+1:t+1] can influence the representation."""

    def __init__(self, input_dim=512, feature_dim=256, kernel=3):
        super().__init__()
        if min(input_dim, feature_dim) < 1 or kernel not in (1, 3):
            raise ValueError("Positive dimensions and kernel 1 or 3 are required")
        self.config = dict(input_dim=input_dim, feature_dim=feature_dim, kernel=kernel)
        self.conv1 = nn.Conv1d(input_dim, feature_dim, kernel)
        self.conv2 = nn.Conv1d(feature_dim, feature_dim, 1)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features, lengths):
        valid = valid_steps(features, lengths)
        # Remove padding NaNs before convolution, including the zero-length chunks
        # that the original UCF evaluator can pass for divisible video lengths.
        features = features.float().masked_fill(~valid[..., None], 0)
        hidden = F.gelu(self.conv1(F.pad(features.transpose(1, 2), (self.config["kernel"] - 1, 0))))
        hidden = self.conv2(hidden).transpose(1, 2)
        return F.gelu(self.norm(hidden)).masked_fill(~valid[..., None], 0)


class FutureFeaturePredictor(nn.Module):
    def __init__(self, input_dim=512, feature_dim=256, kernel=3):
        super().__init__()
        self.encoder = CausalFeatureEncoder(input_dim, feature_dim, kernel)
        self.predictor = nn.Linear(feature_dim, input_dim)

    def forward(self, features, lengths):
        return self.predictor(self.encoder(features, lengths))


def future_feature_mse(predictions, features, lengths, horizon=1, allow_empty=False):
    """Mean over valid future pairs AND channels, with a fixed detached target.

    Horizon counts steps on the detector's process_feat grid, not raw frames.
    Standalone pretraining rejects batches without pairs. Joint training can
    explicitly skip MSE for such a batch, while still learning its video labels.
    """
    if horizon < 1 or predictions.shape != features.shape:
        raise ValueError("Positive horizon and matching [B,T,D] predictions/targets required")
    valid = valid_steps(features, lengths)
    pairs = valid[:, horizon:]
    if not bool(pairs.any()):
        if allow_empty:
            # An empty sum keeps the gradient graph and never reads padding NaNs.
            return predictions.reshape(-1)[:0].sum(), 0
        raise ValueError("No valid future-feature pairs; use longer clips or a smaller horizon")
    predicted = predictions[:, :-horizon][pairs]
    target = features.detach()[:, horizon:][pairs].to(predicted.dtype)
    if not bool(torch.isfinite(predicted).all() and torch.isfinite(target).all()):
        raise FloatingPointError("Non-finite prediction/target in a valid future pair")
    return F.mse_loss(predicted, target), int(pairs.sum())


class FutureFeatureAuxiliary(nn.Module):
    """Frozen MSE-pretrained encoder, followed by a learned binary MIL scorer.

    C gets a scalar residual. A gets the same scalar on abnormal classes only;
    class zero (Normal) stays unchanged. No text, prototypes or cosine bank.
    """

    def __init__(self, input_dim=512, feature_dim=256, num_classes=14, kernel=3):
        super().__init__()
        if num_classes < 2:
            raise ValueError("At least Normal and one abnormal class are required")
        self.encoder = CausalFeatureEncoder(input_dim, feature_dim, kernel)
        self.encoder.requires_grad_(False)
        self.score_head = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)
        self.register_buffer("abnormal_classes", torch.cat((torch.zeros(1), torch.ones(num_classes - 1))))

    def forward(self, features, lengths):
        valid = valid_steps(features, lengths)
        # The backbone and encoder must not be trained through the frozen branch.
        with torch.no_grad():
            encoded = self.encoder(features, lengths)
        score = self.score_head(encoded).masked_fill(~valid[..., None], 0)
        return dict(delta_c=score, delta_a=score * self.abnormal_classes,
                    memory_logits=score, features=encoded)


class JointFutureFeatureAuxiliary(FutureFeatureAuxiliary):
    """One trainable encoder shared by the anomaly scorer and MSE predictor."""

    def __init__(self, input_dim=512, feature_dim=256, num_classes=14, kernel=3):
        super().__init__(input_dim, feature_dim, num_classes, kernel)
        self.encoder.requires_grad_(True)
        self.predictor = nn.Linear(feature_dim, input_dim)

    def forward(self, features, lengths, return_prediction=False):
        valid = valid_steps(features, lengths)
        encoded = self.encoder(features, lengths)
        score = self.score_head(encoded).masked_fill(~valid[..., None], 0)
        output = dict(delta_c=score, delta_a=score * self.abnormal_classes,
                      memory_logits=score, features=encoded)
        if self.training and return_prediction:
            output["future_predictions"] = self.predictor(encoded)
        return output


class FutureFeatureDataset(Dataset):
    """Training CSV only; identical deterministic resampling to detector training."""

    def __init__(self, csv_path, visual_length=256, horizon=1):
        try:
            from .data_check import inspect_csv
            from . import bootstrap
        except ImportError:
            from data_check import inspect_csv
            import bootstrap
        from utils.tools import process_feat

        if horizon < 1 or visual_length <= horizon:
            raise ValueError("Require 1 <= horizon < visual_length")
        self.process_feat = process_feat
        self.visual_length = visual_length
        self.counts, frame_lengths = inspect_csv(csv_path, all_shapes=True)
        with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [row for row, frames in zip(rows, frame_lengths)
                     if min(frames // 16, visual_length) > horizon]
        self.skipped = len(rows) - len(self.rows)
        if not self.rows:
            raise ValueError("Training CSV has no clip with valid future-feature targets")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        array = np.load(self.rows[index]["path"], allow_pickle=False)
        array, length = self.process_feat(array, self.visual_length)
        if not np.isfinite(array[:length]).all():
            raise FloatingPointError(f"Non-finite training features: {self.rows[index]['path']}")
        return torch.from_numpy(np.asarray(array, dtype=np.float32)), length


def load_pretrained_encoder(auxiliary, path, visual_length):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("experiment") != "cnn_memory_future_pretrain"
            or checkpoint.get("format_version") != 1 or checkpoint.get("epoch", 0) < 1):
        raise ValueError("Expected a completed cnn_memory future-feature pretraining checkpoint")
    if checkpoint["encoder_config"] != auxiliary.encoder.config:
        raise ValueError("Pretrained CNN dimensions/kernel differ from the detection model")
    if checkpoint["train_args"]["visual_length"] != visual_length:
        raise ValueError("Pretrain and detector must use the same visual_length feature grid")
    auxiliary.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                epoch=checkpoint["epoch"], horizon=checkpoint["train_args"]["horizon"],
                train_csv_sha256=checkpoint["train_csv_sha256"],
                encoder_config=checkpoint["encoder_config"], encoder_frozen=True)
