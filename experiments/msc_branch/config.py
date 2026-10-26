"""CNN-memory controls plus a training-only contrastive representation branch."""
import math

from experiments.cnn_memory.config import (
    LABEL_MAP, MODEL_KEYS as CNN_MODEL_KEYS, RESUME_KEYS as CNN_RESUME_KEYS,
    training_parser as cnn_parser, validate_args as validate_cnn_args,
)

MODEL_KEYS = CNN_MODEL_KEYS + ('representation_hidden', 'representation_dim')
RESUME_KEYS = CNN_RESUME_KEYS + (
    'representation_hidden', 'representation_dim', 'representation_lr',
    'contrastive_mode', 'contrastive_weight', 'contrastive_temperature',
    'contrastive_start_epoch', 'center_videos', 'center_snippets', 'center_seed',
)


def training_parser():
    parser = cnn_parser()
    parser.description = 'VadCLIP + CNN memory + auxiliary MSC representation branch'
    parser.add_argument('--contrastive-mode', choices=('none', 'standard', 'mean_shifted'), default='none')
    parser.add_argument('--representation-hidden', type=int, default=256)
    parser.add_argument('--representation-dim', type=int, default=128)
    parser.add_argument('--representation-lr', type=float, default=2e-4)
    parser.add_argument('--contrastive-weight', type=float, default=.01)
    parser.add_argument('--contrastive-temperature', type=float, default=.1)
    parser.add_argument('--contrastive-start-epoch', type=int, default=3)
    parser.add_argument('--center-videos', type=int, default=128)
    parser.add_argument('--center-snippets', type=int, default=32)
    parser.add_argument('--center-seed', type=int, default=0)
    return parser


def validate_args(args):
    validate_cnn_args(args)
    for name in ('representation_hidden', 'representation_dim', 'contrastive_start_epoch',
                 'center_videos', 'center_snippets'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    for name in ('representation_lr', 'contrastive_weight', 'contrastive_temperature'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive; use mode none to disable the loss')
    if args.center_seed < 0:
        raise ValueError('center_seed must be non-negative')
    if args.contrastive_mode != 'none' and args.contrastive_start_epoch > args.max_epoch:
        raise ValueError('contrastive_start_epoch exceeds max_epoch')


def model_config(args):
    return {key: getattr(args, key) for key in MODEL_KEYS}
