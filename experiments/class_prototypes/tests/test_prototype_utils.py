import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prototype_utils import (
    aggregate_prototype,
    load_description_file,
    score_embeddings,
    select_diverse_embeddings,
)
from ucf_labels import LABEL_MAP


class PrototypeUtilityTests(unittest.TestCase):
    def test_repository_descriptions_have_no_articles(self):
        payload, digest = load_description_file(
            ROOT / "descriptions" / "ucf_crime_descriptions.json"
        )
        self.assertEqual(len(payload["classes"]), 14)
        self.assertTrue(all(
            len(descriptions) == 20
            for descriptions in payload["classes"].values()
        ))
        self.assertEqual(
            list(payload["classes"]), list(LABEL_MAP.values())
        )
        self.assertEqual(len(digest), 64)

    def test_article_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({
                "classes": {"fighting": ["a person fights"]}
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_description_file(path)

    def test_diversity_filter_rejects_near_duplicate(self):
        embeddings = torch.tensor([
            [1.0, 0.0], [0.999, 0.001], [0.0, 1.0]
        ])
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        selected, rejected = select_diverse_embeddings(
            embeddings, torch.tensor([3.0, 2.0, 1.0]), 2, 0.90
        )
        self.assertEqual(selected, [0, 2])
        self.assertEqual(rejected[0]["duplicate_of"], 0)

    def test_weighted_prototype_is_normalized(self):
        embeddings = torch.eye(3)
        prototype, weights = aggregate_prototype(
            embeddings, torch.tensor([3.0, 2.0, 1.0]),
            mode="weighted_mean", temperature=0.5
        )
        self.assertAlmostEqual(prototype.norm().item(), 1.0, places=6)
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)
        self.assertGreater(weights[0].item(), weights[1].item())

    def test_scoring_returns_one_score_per_embedding(self):
        class_embeddings = [
            torch.nn.functional.normalize(torch.rand(4, 8), dim=-1),
            torch.nn.functional.normalize(torch.rand(4, 8), dim=-1),
        ]
        anchors = torch.nn.functional.normalize(torch.rand(2, 8), dim=-1)
        scored = score_embeddings(class_embeddings, anchors)
        self.assertEqual(tuple(scored[0]["total"].shape), (4,))

    def test_build_cli_exposes_diversity_disable_flag(self):
        source = (ROOT / "build_prototypes.py").read_text(encoding="utf-8")
        self.assertIn("--disable-diversity-filter", source)
        self.assertIn('"diversity_filter_enabled"', source)


if __name__ == "__main__":
    unittest.main()
