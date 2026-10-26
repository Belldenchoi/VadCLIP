"""Compare real metrics and chunk inputs against both existing evaluators."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.cnn_memory.config import LABEL_MAP
from experiments.cnn_memory.evaluation import evaluate
from experiments.topk_variants import test_ucf as topk
from utils.tools import process_split

torch.set_num_threads(1)


class EvaluationParityTests(unittest.TestCase):
    def test_original_topk_and_auxiliary_evaluators_match_with_real_map(self):
        # CLIPVAD is only constructed by the original script's CLI. The fixture
        # supplies logits directly, without importing CLIP or changing src/.
        spec = importlib.util.spec_from_file_location("original_ucf_eval_fixture", REPO / "src/ucf_test.py")
        original = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"model": SimpleNamespace(CLIPVAD=nn.Module)}):
            spec.loader.exec_module(original)

        class Detector(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, visual, mask, text, lengths):
                self.calls.append((visual.clone(), mask.clone(), lengths.clone()))
                self.assert_no_grad = not torch.is_grad_enabled()
                return torch.zeros(14, 1), visual[..., :1], visual[..., 1:15]

        loader, truth, raw, segments, labels = [], [], [], [], []
        rng = np.random.default_rng(234)
        maxlen = 8
        # Short clips, exact multiples (including their padded chunk), and
        # partial final chunks all go through the actual process_split.
        for index, label in enumerate(LABEL_MAP):
            length = (5, 8, 9, 16, 17)[index % 5]
            start, end = length // 2 - 1, length // 2 + 1
            target = np.zeros(length, dtype=np.int64)
            if index:
                target[start:end] = 1
            features = np.zeros((length, 512), dtype=np.float32)
            features[:, 0] = rng.normal(-0.5, 1.0, length) + 1.5 * target
            features[:, 1:15] = -4.0
            features[:, 1] = 2.0
            features[start:end, index + 1] = 6.0
            split, valid_length = process_split(features, maxlen)
            loader.append((torch.from_numpy(split).unsqueeze(0), [label], torch.tensor([valid_length])))
            raw.append(torch.from_numpy(features))
            truth.extend(target.tolist())
            segments.append([[start * 16, end * 16]])
            labels.append([label])
        gt = np.repeat(truth, 16)
        prompt = list(LABEL_MAP)
        baseline_model, adapter_model, topk_model = Detector(), Detector(), Detector()
        original_predictions, adapter_predictions, original_dmap = [], [], []
        original_map, topk_map = original.dmAP, topk.dmAP

        def record_original(predictions, *args, **kwargs):
            original_predictions.extend(p.copy() for p in predictions)
            values, thresholds = original_map(predictions, *args, **kwargs)
            original_dmap.extend(values)
            return values, thresholds

        def record_adapter(predictions, *args, **kwargs):
            adapter_predictions.extend(p.copy() for p in predictions)
            return topk_map(predictions, *args, **kwargs)

        with patch.object(original, "dmAP", side_effect=record_original), contextlib.redirect_stdout(io.StringIO()):
            baseline = original.test(baseline_model, loader, maxlen, prompt, gt, segments, labels, "cpu")
        messages = []
        with patch.object(topk, "dmAP", side_effect=record_adapter):
            actual = evaluate(adapter_model, loader, maxlen, prompt, gt, segments, labels, "cpu", messages.append)
        topk_messages = []
        topk_default = topk.test(topk_model, loader, maxlen, prompt, gt, segments, labels,
                                "cpu", logger=SimpleNamespace(log=topk_messages.append))

        self.assertEqual(baseline, topk_default)
        self.assertEqual(baseline, (actual["AUC1"], actual["AP1"]))
        raw = torch.cat(raw)
        c = np.repeat(raw[:, 0].sigmoid().numpy(), 16)
        a = np.repeat((1 - raw[:, 1:15].softmax(-1)[:, 0]).numpy(), 16)
        expected = dict(AUC1=float(roc_auc_score(gt, c)), AP1=float(average_precision_score(gt, c)),
                        AUC2=float(roc_auc_score(gt, a)), AP2=float(average_precision_score(gt, a)),
                        mAP=float(sum(original_dmap) / len(original_dmap)))
        self.assertEqual(actual, expected)
        self.assertGreater(actual["mAP"], 0.0)
        self.assertLess(actual["AUC1"], 1.0)
        self.assertEqual(messages[1:], topk_messages)
        for left, right in zip(original_predictions, adapter_predictions):
            np.testing.assert_array_equal(left, right)
        for model in (adapter_model, topk_model):
            self.assertFalse(model.training)
            self.assertTrue(model.assert_no_grad)
            self.assertEqual(len(model.calls), len(baseline_model.calls))
            for expected_call, actual_call in zip(baseline_model.calls, model.calls):
                for left, right in zip(expected_call, actual_call):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
