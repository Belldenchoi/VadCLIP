"""Fixed tail feature augmentation, ported from Finetune-VadCLIP's shift_view.

Source: vngclinh/Finetune-VadCLIP, commit e8aa527, dataset_augment.py.
Here a positive offset denotes the magnitude of a tail shift (the source uses
a negative signed offset). No cyclic wrap, extra loss, or random sampling.
"""
import torch


def tail_shift_batch(features, lengths, offset=26):
    """Prepend zeros to [B,T,D]; return the shifted view and prefix lengths.

    Content at p moves to p + offset. For a short video the leading zeros are
    inside the model's valid prefix, matching the source implementation. Only
    real content pushed beyond the grid is lost; original padding stays zero.
    """
    if features.ndim != 3 or lengths.shape != (features.shape[0],):
        raise ValueError('Tail shift requires [B,T,D] features and [B] lengths')
    grid = features.shape[1]
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 < offset < grid:
        raise ValueError('Tail shift offset must be an integer in (0, grid length)')
    lengths = lengths.to(features.device)
    if (lengths.is_floating_point() or lengths.is_complex()
            or lengths.dtype == torch.bool):
        raise ValueError('Tail shift requires integer valid lengths in [1,T]')
    lengths = lengths.long()
    if ((lengths < 1) | (lengths > grid)).any():
        raise ValueError('Tail shift requires integer valid lengths in [1,T]')
    source_positions = torch.arange(grid - offset, device=features.device)
    content = torch.where(
        (source_positions[None, :] < lengths[:, None]).unsqueeze(-1),
        features[:, :grid - offset],
        torch.zeros_like(features[:, :grid - offset]),
    )
    shifted = torch.cat([
        features.new_zeros(features.shape[0], offset, features.shape[2]), content,
    ], dim=1)
    return shifted, (lengths + offset).clamp(max=grid)


class TailShiftAugmentation:
    """Configuration and resume guard for the isolated Temporal-A experiment."""
    def __init__(self, args):
        self.enabled = bool(getattr(args, 'tail_shift_augmentation', False))
        self.offset = getattr(args, 'tail_shift_offset', 26)
        self.config = {'enabled': False}
        if not self.enabled:
            return
        if not 0 < self.offset < args.visual_length:
            raise ValueError('--tail-shift-offset must be in (0, --visual-length)')
        if not (
                args.topk_pooling == 'mean'
                and args.temporal_segment_topk
                and args.temporal_segment_start_epoch == 1
                and args.temporal_smoothing_kernel == 3
                and args.temporal_smoothness_branch == 'a'
                and args.temporal_smoothness_start_epoch == 1
                and args.a_temporal_smoothness_weight == .05
                and args.c_temporal_smoothing_kernel == 1
                and not args.c_temporal_segment_topk):
            raise ValueError('Tail shift requires the Temporal-A reference: original '
                             'C Hard Top-K; A segment, kernel 3 and TV A=0.05 '
                             'from epoch 1')
        if any(getattr(args, flag, False) for flag in (
                'dual_k', 'adaptive_instance_selection', 'c_ranking_loss',
                'snippet_ranking_loss', 'pca_loss', 'c_pca_loss')):
            raise ValueError('Tail shift is a separate Temporal-A ablation; disable '
                             'Dual-K, AIS, ranking and PCA')
        self.config = {
            'enabled': True, 'direction': 'tail', 'offset': self.offset,
            'seed': args.seed, 'visual_length': args.visual_length,
            'task_loss_branches': 'both', 'task_view_reduction': 'mean',
            'tv_scope': 'original', 'consistency_weight': 0.0,
            'a_pooling': 'segment_conv3', 'a_tv_weight': .05,
            'c_pooling': 'original_hard_topk', 'start_epoch': 1,
        }

    def make_view(self, features, lengths):
        if not self.enabled:
            raise ValueError('Tail shift augmentation is disabled')
        return tail_shift_batch(features, lengths, self.offset)

    def validate_checkpoint(self, checkpoint):
        saved = checkpoint.get('tail_shift_config', {'enabled': False})
        if saved != self.config:
            raise ValueError('Checkpoint tail shift configuration differs from CLI. '
                             'Use matching flags or start a new run with separate outputs.')
