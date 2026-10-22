"""Resolve repository imports before importing the original dataset/model."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
TOPK_ROOT = REPO_ROOT / "experiments" / "topk_variants"
for root in (REPO_ROOT, TOPK_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
