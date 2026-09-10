"""All-pairs video-level hinge ranking for C-branch Soft Top-K scores."""
import math
import torch


def ranking_config(args):
    if not args.c_ranking_loss:
        return {'enabled': False}
    return {'enabled': True, 'margin': args.c_ranking_margin,
            'weight': args.c_ranking_weight,
            'start_epoch': args.c_ranking_start_epoch}


def validate_ranking_config(args):
    if not args.c_ranking_loss:
        return
    if not math.isfinite(args.c_ranking_margin) or not 0 < args.c_ranking_margin < 1:
        raise ValueError('--c-ranking-margin must be finite and in (0, 1)')
    if not math.isfinite(args.c_ranking_weight) or args.c_ranking_weight < 0:
        raise ValueError('--c-ranking-weight must be finite and non-negative')
    if args.c_ranking_start_epoch < 1:
        raise ValueError('--c-ranking-start-epoch must be at least 1')
    if args.topk_pooling != 'soft':
        raise ValueError('--c-ranking-loss requires --topk-pooling soft')
    if args.a_topk_temperature_schedule != 'constant':
        raise ValueError('Ranking ablation requires a constant temperature schedule')
    if any(getattr(args, flag) for flag in (
            'dual_k', 'adaptive_instance_selection',
            'temporal_segment_topk', 'c_temporal_segment_topk')):
        raise ValueError('Ranking ablation cannot be combined with other selectors')
    if (args.c_temporal_smoothing_kernel != 1 or
            args.temporal_smoothing_kernel != 1 or
            args.temporal_smoothness_branch != 'none'):
        raise ValueError('Ranking ablation requires kernels=1 and smoothness=none')


def ranking_active(args, epoch):
    return args.c_ranking_loss and epoch >= args.c_ranking_start_epoch


def validate_checkpoint_ranking(args, checkpoint):
    saved = checkpoint.get('c_ranking_config', {'enabled': False})
    if saved != ranking_config(args):
        raise ValueError('Checkpoint ranking configuration differs from CLI. '
                         'Use matching flags or start a new run with separate outputs.')


def c_pairwise_ranking(scores, binary_labels, margin=0.2):
    """scores: [B] pooled probabilities; labels: [B], 0 normal / 1 anomaly.

    Returns differentiable mean hinge and detached scalar diagnostics.
    """
    if not math.isfinite(margin) or not 0 < margin < 1:
        raise ValueError('margin must be finite and in (0, 1)')
    if scores.ndim != 1 or binary_labels.shape != scores.shape:
        raise ValueError('scores and binary_labels must have matching [B] shapes')
    with torch.autocast(device_type=scores.device.type, enabled=False):
        scores = scores.float()
        labels = binary_labels.to(scores.device)
        if not torch.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError('scores must be finite probabilities in [0, 1]')
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError('ranking requires binary video labels')
        positive, negative = scores[labels == 1], scores[labels == 0]
        count = positive.numel() * negative.numel()
        stats = {'pairs': count,
                 'positive_mean': positive.detach().mean().item() if positive.numel() else 0.0,
                 'negative_mean': negative.detach().mean().item() if negative.numel() else 0.0,
                 'gap_mean': 0.0, 'violation_fraction': 0.0}
        if not count:
            return scores.sum() * 0.0, stats
        gaps = positive[:, None] - negative[None, :]
        hinge = torch.relu(margin - gaps)
        stats['gap_mean'] = gaps.detach().mean().item()
        stats['violation_fraction'] = (gaps.detach() < margin).float().mean().item()
        return hinge.mean(), stats
