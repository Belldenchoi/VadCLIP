"""Run the original UCF training loop with optional video-label residual logits."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'topk_variants'))
sys.path.insert(0, str(REPO_ROOT / 'src'))

from experiments.semantic_residual.options import build_parser, validate_config, validate_output_paths


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_config(args)
    validate_output_paths(args)

    import torch
    from torch.utils.data import DataLoader
    from dataset_variants import UCFDataset
    from train_ucf import setup_seed, train
    from experiments.semantic_residual.runtime import LABEL_MAP, build_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setup_seed(args.seed)
    normal = UCFDataset(args.visual_length, args.train_list, False, LABEL_MAP, True,
                        max_samples_per_label=args.max_samples_per_label)
    anomaly = UCFDataset(args.visual_length, args.train_list, False, LABEL_MAP, False,
                         actions=args.train_actions,
                         max_samples_per_label=args.max_samples_per_label)
    normal_loader = DataLoader(normal, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_loader = DataLoader(anomaly, batch_size=args.batch_size, shuffle=True, drop_last=True)
    if not len(normal_loader) or not len(anomaly_loader):
        raise ValueError('Both Normal and Anomaly subsets must have at least --batch-size clips')
    test_data = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    model = build_model(args, device)
    print(f'Training clips: {len(normal)} normal, {len(anomaly)} anomaly | '
          f'residual={args.residual_branch} kernel={args.residual_kernel_size} | '
          f'LGT weights frozen={args.freeze_lgt} | '
          f'new parameters={sum(p.numel() for p in model.residual_branch.parameters()) if model.residual_branch else 0} | '
          f'torch={torch.__version__} GPU={torch.cuda.get_device_name(0) if device == "cuda" else "none"}',
          flush=True)
    train(model, normal_loader, anomaly_loader, test_loader, args, LABEL_MAP, device)


if __name__ == '__main__':
    main()
