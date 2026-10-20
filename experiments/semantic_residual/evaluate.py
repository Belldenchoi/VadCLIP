"""Evaluate residual logits with the unchanged full UCF scoring protocol."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'topk_variants'))

from experiments.semantic_residual.options import build_parser, validate_config


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_config(args)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from dataset_variants import UCFDataset
    from test_ucf import test
    from training_log import TrainingLogger
    from utils.tools import get_prompt_text
    from experiments.semantic_residual.runtime import LABEL_MAP, build_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Exported model.pth includes R weights AND the exact experiment config.
    state = torch.load(args.model_path, map_location='cpu', weights_only=True)
    model = build_model(args, device)
    model.load_state_dict(state)
    if args.log_path and Path(args.log_path).resolve() in {
            Path(args.model_path).resolve(), Path(args.checkpoint_path).resolve()}:
        raise ValueError('Evaluation log must differ from model/checkpoint paths')
    # Append so an explicit evaluation cannot erase a training log.
    logger = TrainingLogger(args.log_path, append=True)
    logger.log(f'evaluation_model={args.model_path} '
               f'semantic_residual_config={model.semantic_residual_config}')
    data = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    test(model, DataLoader(data, batch_size=1, shuffle=False), args.visual_length,
         get_prompt_text(LABEL_MAP), np.load(args.gt_path),
         np.load(args.gt_segment_path, allow_pickle=True),
         np.load(args.gt_label_path, allow_pickle=True), device, logger=logger)


if __name__ == '__main__':
    main()
