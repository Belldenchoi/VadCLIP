"""Opt-in snippet ranking and a fixed, normal-train PCA auxiliary loss."""
import hashlib
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from topk_pooling import video_topk_size


def validate_snippet_config(args):
    if args.pca_loss and not args.snippet_ranking_loss:
        raise ValueError('--pca-loss requires --snippet-ranking-loss')
    if not args.snippet_ranking_loss:
        return
    if (args.c_ranking_loss or args.dual_k or args.adaptive_instance_selection
            or args.topk_pooling != 'mean' or args.c_temporal_segment_topk
            or args.c_temporal_smoothing_kernel != 1
            or args.temporal_smoothness_branch in ('c', 'both')):
        raise ValueError('Snippet ablation requires original C Hard Top-K; '
                         'disable old ranking, Dual-K, AIS and C smoothing/segment')
    for name in ('snippet_ranking_weight', 'snippet_ranking_margin',
                 'snippet_ranking_temperature'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be finite and positive')
    if args.snippet_ranking_margin >= 1 or args.snippet_ranking_start_epoch < 1:
        raise ValueError('Score margin must be < 1 and start epoch >= 1')
    if args.pca_loss:
        for name in ('pca_weight', 'pca_margin'):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 1 <= args.pca_rank < args.visual_width:
            raise ValueError('--pca-rank must be in [1, visual-width)')
        if args.pca_start_epoch < args.snippet_ranking_start_epoch:
            raise ValueError('PCA must start no earlier than snippet ranking')
        if min(args.pca_fit_videos, args.pca_fit_snippets) < 1 or args.pca_fit_seed < 0:
            raise ValueError('PCA sample limits must be positive and fit seed >= 0')


def snippet_config(args):
    if not args.snippet_ranking_loss:
        return {'enabled': False}
    names = ['seed', 'topk_pooling', 'temporal_segment_topk',
             'temporal_segment_start_epoch', 'temporal_smoothing_kernel',
             'temporal_smoothness_branch', 'temporal_smoothness_start_epoch',
             'a_temporal_smoothness_weight', 'visual_width', 'visual_length',
             'train_list', 'train_actions', 'max_samples_per_label',
             'snippet_ranking_mode', 'snippet_ranking_margin',
             'snippet_ranking_weight', 'snippet_ranking_temperature',
             'snippet_ranking_start_epoch', 'pca_loss']
    if args.pca_loss:
        names += ['pca_weight', 'pca_margin', 'pca_rank', 'pca_start_epoch',
                  'pca_fit_videos', 'pca_fit_snippets', 'pca_fit_seed']
    return {'enabled': True, 'version': 1, **{n: getattr(args, n) for n in names}}


def valid_videos(values, lengths, labels):
    """Validate masks without inspecting padding values."""
    if values.ndim != 2 or lengths.shape != (values.shape[0],) or labels.shape != lengths.shape:
        raise ValueError('Expected values [B,T], lengths [B], labels [B]')
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError('Expected binary video labels (0 Normal, 1 abnormal)')
    lens = lengths.detach().cpu().tolist()
    if any(not math.isfinite(n) or n != int(n) or not 1 <= n <= values.shape[1] for n in lens):
        raise ValueError('Every valid length must be an integer in [1,T]')
    videos = [values[i, :int(n)].float() for i, n in enumerate(lens)]
    if any(not torch.isfinite(v).all() for v in videos):
        raise ValueError('Non-finite value in valid snippets')
    positive = [i for i, y in enumerate(labels.detach().cpu().tolist()) if y == 1]
    negative = [i for i in range(len(videos)) if i not in positive]
    return videos, positive, negative


def selected_ranking(positive, negative_max, zero, margin, mode='hinge', temperature=.05):
    """Equal weight per video and per pair, even when K differs by length."""
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError('margin must be finite and positive')
    if mode not in ('hinge', 'softplus') or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Expected hinge/softplus and positive finite temperature')
    count = len(positive) * len(negative_max)
    if not count:
        return zero, {'pairs': 0, 'violation': 0.0}
    hardest = torch.stack(negative_max)
    penalties, violations = [], []
    for values in positive:
        violation = margin + hardest[None, :] - values[:, None]
        penalty = (F.relu(violation) if mode == 'hinge'
                   else temperature * F.softplus(violation / temperature))
        penalties.append(penalty.mean())
        violations.append((violation.detach() > 0).float().mean())
    return torch.stack(penalties).mean(), {
        'pairs': count, 'violation': torch.stack(violations).mean().item(),
        'positive_mean': torch.stack([v.detach().mean() for v in positive]).mean().item(),
        'negative_max_mean': hardest.detach().mean().item(),
    }


def snippet_score_ranking(logits, lengths, labels, margin=.1, mode='hinge', temperature=.05):
    with torch.autocast(device_type=logits.device.type, enabled=False):
        scores = logits.float().squeeze(-1) if logits.ndim == 3 and logits.shape[-1] == 1 else logits.float()
        # Validate logits first: sigmoid would hide +/-inf.
        videos, positive, negative = valid_videos(scores, lengths, labels)
        videos = [v.sigmoid() for v in videos]
        indices = {i: v.detach().topk(video_topk_size(len(v))).indices
                   for i, v in enumerate(videos) if i in positive}
        zero = sum((v.sum() * 0 for v in videos), scores.new_zeros(()))
        loss, stats = selected_ranking(
            [videos[i][indices[i]] for i in positive],
            [videos[i].max() for i in negative], zero, margin, mode, temperature)
        return loss, stats, indices


class NormalPCASubspace:
    """Detached PCA parameters; residuals remain differentiable wrt features."""
    def __init__(self):
        self.mean = None
        self.basis = None
        self.metadata = {}

    @property
    def fitted(self):
        return self.mean is not None

    def fit(self, features, rank, metadata=None):
        samples = features.detach().to(device='cpu', dtype=torch.float64)
        if samples.ndim != 2 or not 1 <= rank < min(samples.shape):
            raise ValueError('PCA needs more samples and dimensions than rank')
        if not torch.isfinite(samples).all() or (samples.norm(dim=-1) <= 1e-12).any():
            raise ValueError('PCA calibration features must be finite and nonzero')
        samples = F.normalize(samples, dim=-1)
        mean = samples.mean(dim=0)
        _, singular, vh = torch.linalg.svd(samples - mean, full_matrices=False)
        tolerance = singular[0] * max(samples.shape) * torch.finfo(samples.dtype).eps
        effective_rank = int((singular > tolerance).sum())
        if effective_rank < rank:
            raise ValueError(f'PCA calibration effective rank {effective_rank} < requested {rank}')
        self.mean = mean.float()
        self.basis = vh[:rank].T.contiguous().float()
        self.metadata = dict(metadata or {}, samples=len(samples), rank=rank,
                             effective_rank=effective_rank,
                             explained_variance=(singular[:rank].square().sum() /
                                                 singular.square().sum()).item())

    def residuals(self, features):
        if not self.fitted:
            raise ValueError('PCA basis has not been fitted')
        with torch.autocast(device_type=features.device.type, enabled=False):
            h = F.normalize(features.float(), dim=-1, eps=1e-12)
            mean, basis = self.mean.to(h.device), self.basis.to(h.device)
            centered = h - mean
            return (centered - (centered @ basis) @ basis.T).square().sum(dim=-1)

    def state_dict(self):
        if not self.fitted:
            return None
        return {'mean': self.mean, 'basis': self.basis, 'metadata': self.metadata}

    def load_state_dict(self, state, width, rank):
        mean, basis = state['mean'].detach().cpu().float(), state['basis'].detach().cpu().float()
        if (mean.shape != (width,) or basis.shape != (width, rank)
                or not torch.isfinite(mean).all() or not torch.isfinite(basis).all()
                or not torch.allclose(basis.T @ basis, torch.eye(rank), atol=1e-4, rtol=1e-4)):
            raise ValueError('Invalid PCA checkpoint mean/basis')
        self.mean, self.basis, self.metadata = mean, basis, state['metadata']


def calibrate_normal_pca(model, dataset, device, args, epoch):
    """Read only Normal train samples, preserving RNG and all module modes."""
    if getattr(dataset, 'test_mode', True) or not getattr(dataset, 'normal', False):
        raise ValueError('PCA calibration requires the Normal training dataset')
    indices = np.random.default_rng(args.pca_fit_seed).permutation(len(dataset))[:args.pca_fit_videos]
    states = [(module, module.training) for module in model.modules()]
    numpy_state, python_state = np.random.get_state(), random.getstate()
    # fork all visible CUDA generators as well as CPU; no effect on loader RNG.
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    samples = []
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            model.eval()
            for index in indices:
                features, label, length = dataset[int(index)]
                if label != 'Normal' or not 1 <= int(length) <= len(features):
                    raise ValueError('Invalid Normal train calibration sample')
                encoded = model.encode_video(features.unsqueeze(0).to(device), None,
                                             torch.tensor([length], device=device))[0, :int(length)]
                positions = torch.linspace(0, int(length) - 1,
                                           min(args.pca_fit_snippets, int(length))).long().to(encoded.device)
                samples.append(encoded[positions].float().cpu())
            if not samples:
                raise ValueError('No Normal training samples for PCA')
            subspace = NormalPCASubspace()
            subspace.fit(torch.cat(samples), args.pca_rank, {
                'fit_epoch': epoch, 'videos': len(indices), 'fit_seed': args.pca_fit_seed,
                'sample_indices_sha256': hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
            })
            return subspace
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        for module, training in states:
            module.training = training


def pca_auxiliary_loss(features, lengths, labels, indices, subspace, margin=.1):
    # Mask before normalization/projection: padding (including NaN) has no gradient.
    if features.ndim != 3:
        raise ValueError('Expected learned features [B,T,D]')
    valid = torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]
    features = features.float().masked_fill(~valid[..., None], 0)
    if not torch.isfinite(features).all() or (features.norm(dim=-1)[valid] <= 1e-12).any():
        raise ValueError('Valid PCA loss features must be finite and nonzero')
    residuals = subspace.residuals(features)
    videos, positive, negative = valid_videos(residuals, lengths, labels)
    zero = residuals.sum() * 0
    normal_loss = torch.stack([videos[i].mean() for i in negative]).mean() if negative else zero
    residual_loss, stats = selected_ranking([videos[i][indices[i]] for i in positive],
                                            [videos[i].max() for i in negative], zero, margin)
    stats.update(normal=normal_loss.detach().item(), residual=residual_loss.detach().item())
    return normal_loss + residual_loss, stats


class SnippetAuxiliaryLoss:
    def __init__(self, args):
        validate_snippet_config(args)
        self.args = args
        self.config = snippet_config(args)
        self.pca = NormalPCASubspace()

    def active(self, epoch):
        return self.args.snippet_ranking_loss and epoch >= self.args.snippet_ranking_start_epoch

    def pca_active(self, epoch):
        return self.args.pca_loss and epoch >= self.args.pca_start_epoch

    def begin_epoch(self, epoch, model, normal_dataset, device, logger):
        if self.pca_active(epoch) and not self.pca.fitted:
            self.pca = calibrate_normal_pca(model, normal_dataset, device, self.args, epoch)
            logger.log(f'pca_fit={self.pca.metadata}')
        if self.args.snippet_ranking_loss:
            logger.log(f'epoch={epoch} snippet_active={self.active(epoch)} pca_active={self.pca_active(epoch)}')

    def __call__(self, logits, features, lengths, labels, epoch):
        if not self.active(epoch):
            raise ValueError('Call auxiliary loss only after its start epoch')
        score_loss, stats, indices = snippet_score_ranking(
            logits, lengths, labels, self.args.snippet_ranking_margin,
            self.args.snippet_ranking_mode, self.args.snippet_ranking_temperature)
        total = self.args.snippet_ranking_weight * score_loss
        metrics = {f'snippet_{k}': v for k, v in stats.items()}
        metrics.update(loss_snippet=score_loss.detach().item(), weighted_snippet=total.detach().item())
        if self.pca_active(epoch):
            pca_loss, pca_stats = pca_auxiliary_loss(features, lengths, labels, indices,
                                                   self.pca, self.args.pca_margin)
            weighted = self.args.pca_weight * pca_loss
            total = total + weighted
            metrics.update({f'pca_{k}': v for k, v in pca_stats.items()})
            metrics.update(loss_pca=pca_loss.detach().item(), weighted_pca=weighted.detach().item())
        return total, metrics

    def state_dict(self):
        return {'config': self.config, 'pca': self.pca.state_dict()}

    def restore(self, checkpoint):
        saved = checkpoint.get('snippet_auxiliary', {'config': {'enabled': False}, 'pca': None})
        if saved['config'] != self.config:
            raise ValueError('Checkpoint snippet/PCA configuration differs; use matching flags or a new run')
        state = saved.get('pca')
        if state is not None:
            if not self.args.pca_loss:
                raise ValueError('Unexpected PCA basis in checkpoint')
            self.pca.load_state_dict(state, self.args.visual_width, self.args.pca_rank)
        elif self.pca_active(checkpoint.get('epoch', -1) + 1):
            raise ValueError('Active PCA checkpoint is missing its fitted basis')
