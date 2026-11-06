"""Run the two imported Finetune losses on the preserved Temporal-A reference.

The original loss functions live in finetune_margin_loss.py. This adapter adds
input/padding validation, warm-up, diagnostics and checkpoint configuration;
it does not change the original selectors, reductions or ratio-based K.
"""
import math

import torch

from finetune_margin_loss import (
    abnormal_background_suppression_loss, get_anomaly_score_branches,
    ranking_margin_loss,
)


class FinetuneMarginLoss:
    def __init__(self, args):
        self.mode = getattr(args, 'finetune_margin', 'none')
        self.enabled = self.mode != 'none'
        self.config = {'enabled': False}
        if not self.enabled:
            return
        if self.mode not in ('ranking', 'background'):
            raise ValueError('Unknown --finetune-margin mode')
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
            raise ValueError('Finetune margins require the Temporal-A reference: '
                             'original C Hard Top-K; A segment/k3/TV=0.05 from epoch 1')
        if any(getattr(args, name, False) for name in (
                'c_ranking_loss', 'snippet_ranking_loss', 'pca_loss', 'c_pca_loss',
                'dual_k', 'adaptive_instance_selection', 'tail_shift_augmentation')):
            raise ValueError('Finetune margin is an isolated Temporal-A ablation; '
                             'disable old ranking, PCA, Dual-K, AIS and tail shift')
        self.branch = args.regularize_score_branch
        self.ratio = args.regularize_top_k_ratio
        if self.branch not in ('classifier', 'alignment', 'both'):
            raise ValueError('Unknown --regularize-score-branch')
        if not math.isfinite(self.ratio) or not 0 < self.ratio <= 1:
            raise ValueError('--regularize-top-k-ratio must be finite and in (0,1]')
        if self.mode == 'ranking':
            self.margin, self.weight, self.warmup = (
                args.ranking_margin, args.lambda_ranking_margin, args.ranking_warmup_epochs)
        else:
            self.margin, self.weight, self.warmup = (
                args.abnormal_bg_margin, args.lambda_abnormal_bg, args.abnormal_bg_warmup_epochs)
            if self.ratio > .5:
                raise ValueError('Background separation requires ratio <= 0.5 for disjoint groups')
        if not math.isfinite(self.margin) or not 0 <= self.margin < 1:
            raise ValueError('Finetune margin must be finite and in [0,1)')
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError('Enabled Finetune margin needs a finite positive lambda')
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int) or self.warmup < 0:
            raise ValueError('Finetune warmup epochs must be a non-negative integer')
        if self.warmup >= args.max_epoch:
            raise ValueError('Finetune warmup must leave at least one active training epoch')
        self.config = {
            'enabled': True, 'mode': self.mode, 'score_branch': self.branch,
            'top_k_ratio': self.ratio, 'margin': self.margin, 'weight': self.weight,
            'warmup_epochs': self.warmup, 'first_active_epoch': self.warmup + 1,
            'reference': 'temporal_a_segment_k3_tv005',
            'reduction': 'batch_mean_hinge' if self.mode == 'ranking' else 'intra_video_boundary_hinge',
        }

    def active(self, epoch):
        return self.enabled and epoch > self.warmup

    def checkpoint_fields(self):
        return {'finetune_margin_config': self.config.copy()} if self.enabled else {}

    def validate_checkpoint(self, checkpoint):
        if checkpoint.get('finetune_margin_config', {'enabled': False}) != self.config:
            raise ValueError('Checkpoint Finetune margin configuration differs; '
                             'use matching flags or start a new run')

    def _prepare(self, logits1, logits2, labels, lengths):
        if (logits1.ndim != 3 or logits1.shape[-1] != 1 or logits2.ndim != 3
                or logits2.shape[:2] != logits1.shape[:2] or logits2.shape[-1] < 2
                or labels.shape != (logits1.shape[0], logits2.shape[-1])):
            raise ValueError('Expected C [B,T,1], A [B,T,classes] and video labels [B,classes]')
        if logits1.device != logits2.device or labels.device != logits1.device:
            raise ValueError('Finetune logits and labels must share a device')
        lengths = torch.as_tensor(lengths, device=logits1.device)
        if (lengths.shape != (logits1.shape[0],) or lengths.is_floating_point()
                or lengths.is_complex() or lengths.dtype == torch.bool
                or ((lengths < 0) | (lengths > logits1.shape[1])).any()):
            raise ValueError('Finetune valid lengths must be integers in [0,T]')
        if not torch.isfinite(labels).all() or (labels < 0).any() or (labels.sum(-1) <= 0).any():
            raise ValueError('Expected finite non-negative video class labels')
        valid = torch.arange(logits1.shape[1], device=logits1.device)[None, :] < lengths[:, None]
        # Mask BEFORE sigmoid/softmax: invalid NaNs must not contaminate gradients.
        c = logits1.float().masked_fill(~valid[..., None], 0)
        a = logits2.float().masked_fill(~valid[..., None], 0)
        if not torch.isfinite(c).all() or not torch.isfinite(a).all():
            raise ValueError('Non-finite logits in valid Finetune margin positions')
        return c, a, labels, lengths.long()

    def _diagnostics(self, c, a, labels, lengths):
        gaps, violations, counts = [], [], []
        with torch.no_grad():
            binary = labels.argmax(-1) != 0
            for scores in get_anomaly_score_branches(c.detach(), a.detach(), self.branch):
                if self.mode == 'ranking':
                    positive, negative = [], []
                    for index, length in enumerate(lengths.tolist()):
                        if length <= 0:
                            continue
                        k = max(1, int(length * self.ratio))
                        value = scores[index, :length].topk(k).values.mean()
                        (positive if bool(binary[index]) else negative).append(value)
                    if positive and negative:
                        gap = torch.stack(positive).mean() - torch.stack(negative).mean()
                        gaps.append(gap)
                        violations.append((gap < self.margin).float())
                        counts.append(len(positive) + len(negative))
                else:
                    branch_gaps = []
                    for index, length in enumerate(lengths.tolist()):
                        if not bool(binary[index]) or length < 2:
                            continue
                        k = max(1, int(length * self.ratio))
                        if 2 * k > length:
                            continue
                        values = scores[index, :length]
                        branch_gaps.append(values.topk(k).values.min() - values.topk(k, largest=False).values.max())
                    if branch_gaps:
                        values = torch.stack(branch_gaps)
                        gaps.append(values.mean())
                        violations.append((values < self.margin).float().mean())
                        counts.append(len(branch_gaps))
        return {
            'finetune_margin_gap': float(torch.stack(gaps).mean()) if gaps else 0.,
            'finetune_margin_violation': float(torch.stack(violations).mean()) if violations else 0.,
            'finetune_margin_videos': sum(counts) / len(counts) if counts else 0.,
        }

    def __call__(self, logits1, logits2, labels, lengths, epoch):
        if not self.enabled:
            raise ValueError('Finetune margin is disabled')
        stats = {'loss_finetune_margin': 0., 'weighted_finetune_margin': 0.,
                 'finetune_margin_weight': 0., 'finetune_margin_active': 0.,
                 'finetune_margin_gap': 0., 'finetune_margin_violation': 0.,
                 'finetune_margin_videos': 0.}
        if not self.active(epoch):
            return logits1.new_zeros((), dtype=torch.float32), stats
        c, a, labels, lengths = self._prepare(logits1, logits2, labels, lengths)
        function = ranking_margin_loss if self.mode == 'ranking' else abnormal_background_suppression_loss
        with torch.autocast(device_type=c.device.type, enabled=False):
            loss = function(c, a, labels, lengths, self.ratio, self.margin, self.branch, c.device)
            if not loss.requires_grad:
                # Source functions return a constant zero for empty selections.
                # Keep that value while allowing a safe standalone backward.
                loss = loss + sum(scores.sum() * 0 for scores in get_anomaly_score_branches(c, a, self.branch))
            weighted = self.weight * loss
        stats.update(self._diagnostics(c, a, labels, lengths))
        stats.update(loss_finetune_margin=float(loss.detach()),
                     weighted_finetune_margin=float(weighted.detach()),
                     finetune_margin_weight=self.weight, finetune_margin_active=1.)
        return weighted, stats
