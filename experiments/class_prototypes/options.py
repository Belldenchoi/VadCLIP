"""CLI shared with Top-K experiments plus prototype-specific arguments."""

import importlib.util
from pathlib import Path

TOPK_OPTIONS = Path(__file__).resolve().parents[1] / "topk_variants" / "options.py"
spec = importlib.util.spec_from_file_location("topk_variant_options", TOPK_OPTIONS)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parser = module.parser

parser.add_argument(
    "--prototype-path", required=True,
    help="Cache produced by build_prototypes.py"
)
parser.add_argument(
    "--prototype-logit-temperature", default=0.07, type=float
)
