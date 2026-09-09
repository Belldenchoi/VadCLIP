"""Opt-in A-branch pooling temperature schedule (1-based absolute epochs)."""

import math


def temperature_config(args):
    """Canonical, serializable configuration; ignore inactive schedule flags."""
    mode = args.a_topk_temperature_schedule
    config = {
        'pooling': args.topk_pooling,
        'c_temperature': args.c_topk_temperature,
        'a_temperature': args.a_topk_temperature,
        'a_schedule': mode,
    }
    if mode == 'linear':
        config.update(
            a_start=args.a_topk_temperature_start,
            start_epoch=args.a_topk_temperature_start_epoch,
            end_epoch=args.a_topk_temperature_end_epoch,
        )
    return config


def validate_temperature_config(args):
    mode = args.a_topk_temperature_schedule
    if mode not in ('constant', 'linear'):
        raise ValueError('Unknown A Top-K temperature schedule')
    if args.topk_pooling == 'soft' or mode != 'constant':
        values = [args.c_topk_temperature, args.a_topk_temperature]
        if mode == 'linear':
            values.append(args.a_topk_temperature_start)
        if any(not math.isfinite(t) or t <= 0 for t in values):
            raise ValueError('Soft Top-K temperatures must be finite and positive')
    if mode == 'constant':
        return
    if args.topk_pooling != 'soft':
        raise ValueError('A temperature schedule requires --topk-pooling soft')
    for name in ('dual_k', 'adaptive_instance_selection',
                 'temporal_segment_topk', 'c_temporal_segment_topk'):
        if getattr(args, name, False):
            raise ValueError('A temperature schedule cannot be used with ' + name)
    if args.a_topk_temperature_start_epoch < 1:
        raise ValueError('--a-topk-temperature-start-epoch must be at least 1')
    if (args.a_topk_temperature_end_epoch <=
            args.a_topk_temperature_start_epoch):
        raise ValueError('A temperature end epoch must be greater than start epoch')


def epoch_temperatures(args, epoch):
    """Return (C, A); no mutable state, so resume uses the absolute epoch."""
    validate_temperature_config(args)
    if epoch < 1:
        raise ValueError('epoch must be 1-based')
    a_temperature = args.a_topk_temperature
    if args.a_topk_temperature_schedule == 'linear':
        start = args.a_topk_temperature_start_epoch
        end = args.a_topk_temperature_end_epoch
        progress = min(1.0, max(0.0, (epoch - start) / (end - start)))
        a_temperature = (
            (1.0 - progress) * args.a_topk_temperature_start
            + progress * args.a_topk_temperature
        )
    return args.c_topk_temperature, a_temperature


def validate_checkpoint_temperature(args, checkpoint):
    """Do not silently resume with a different pooling temperature policy."""
    saved = checkpoint.get('topk_temperature_config')
    if saved is None:
        if args.a_topk_temperature_schedule != 'constant':
            raise ValueError(
                'Checkpoint has no temperature configuration. Start a new '
                'scheduled experiment with separate output paths.'
            )
        return  # Preserve legacy fixed-temperature resume behavior.
    if saved != temperature_config(args):
        raise ValueError(
            'Checkpoint temperature configuration differs from CLI. '
            'Use the original temperature flags or start a new experiment.'
        )
