"""Reuse the existing UCF trainer options, with isolated residual switches."""

import importlib.util
import math
from pathlib import Path


def build_parser():
    source = Path(__file__).resolve().parents[1] / 'topk_variants' / 'options.py'
    spec = importlib.util.spec_from_file_location('_semantic_residual_options', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.parser
    parser.description = 'VadCLIP video-label semantic residual ablation (UCF-Crime)'
    # Scope CLI type fixes to this entry point; leave older parsers unchanged.
    for option in ('--lr', '--scheduler-rate'):
        parser._option_string_actions[option].type = float
    milestones = parser._option_string_actions['--scheduler-milestones']
    milestones.type, milestones.nargs = int, '+'
    parser.add_argument('--residual-branch', choices=['none', 'a', 'c', 'both'],
                        default='none', help='Apply residual logits to A, C, both, or neither')
    parser.add_argument('--residual-hidden-dim', type=int, default=64)
    parser.add_argument('--residual-kernel-size', type=int, default=3,
                        help='Learned temporal feature Conv1D; 1 uses only the pair MLP')
    parser.add_argument('--residual-a-scale', type=float, default=0.1)
    parser.add_argument('--residual-c-scale', type=float, default=0.1)
    parser.add_argument('--freeze-lgt', action='store_true',
                        help='Also freeze LGT weights; pair with the same flag in the control')
    # The inherited trainer uses truthiness, but type=bool misreads "False".
    parser.add_argument('--resume', action='store_true', dest='use_checkpoint')
    parser.set_defaults(model_path='outputs/semantic_residual/control/model.pth',
                        checkpoint_path='outputs/semantic_residual/control/checkpoint.pth',
                        log_path='outputs/semantic_residual/control/train.log')
    return parser


def validate_config(args):
    """Keep the first experiment separate from all older ablations."""
    if args.residual_hidden_dim < 1:
        raise ValueError('--residual-hidden-dim must be positive')
    if args.residual_kernel_size < 1 or args.residual_kernel_size % 2 == 0:
        raise ValueError('--residual-kernel-size must be a positive odd integer')
    for name in ('residual_a_scale', 'residual_c_scale'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be finite and positive')
    if args.classes_num != 14:
        raise ValueError('This UCF-Crime entry point requires --classes-num 14')
    if args.visual_width != args.embed_dim:
        raise ValueError('The original A branch requires matching visual/text dimensions')
    if args.topk_pooling != 'mean':
        raise ValueError('Residual ablation keeps hard Top-K: use --topk-pooling mean')
    for name in ('tail_shift_augmentation', 'snippet_ranking_loss', 'pca_loss',
                 'c_pca_loss', 'c_ranking_loss', 'adaptive_instance_selection',
                 'dual_k', 'dual_k_diagnostics_only', 'temporal_segment_topk',
                 'c_temporal_segment_topk'):
        if getattr(args, name):
            raise ValueError(f'Residual ablation cannot combine --{name.replace("_", "-")}')
    if (args.temporal_smoothness_branch != 'none' or
            args.temporal_smoothing_kernel != 1 or
            args.c_temporal_smoothing_kernel != 1 or
            args.a_topk_temperature_schedule != 'constant'):
        raise ValueError('Residual ablation requires original pooling and no score smoothing/TV')
    if args.eval_actions:
        raise ValueError('Keep the full test set in this experiment; omit --eval-actions')
    return {
        'version': 1,
        'branch': args.residual_branch,
        'hidden_dim': args.residual_hidden_dim,
        'kernel_size': args.residual_kernel_size,
        'a_scale': args.residual_a_scale,
        'c_scale': args.residual_c_scale,
        'freeze_lgt': args.freeze_lgt,
        'start_epoch': 1,
        'text_source': 'text_features_ori',
        'prompt_source': 'original_C_logits',
    }


def validate_output_paths(args):
    paths = [Path(p).resolve() for p in
             (args.model_path, args.checkpoint_path, args.log_path) if p]
    current = Path(args.model_path).with_name(
        Path(args.model_path).stem + '_cur' + Path(args.model_path).suffix).resolve()
    paths.append(current)
    if len(set(paths)) != len(paths):
        raise ValueError('Model, checkpoint, log and generated model_cur paths must be distinct')
    if not args.use_checkpoint:
        for path in paths:
            if path.exists():
                raise ValueError(f'New residual run would overwrite {path}; choose a new run directory')
    elif not Path(args.checkpoint_path).is_file():
        raise ValueError('--resume requires an existing --checkpoint-path')
