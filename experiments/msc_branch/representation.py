"""Training-only MLP, Normal calibration and video-balanced contrastive loss."""
import hashlib
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments.cnn_memory.checkpoint import capture_rng, restore_rng

EPS = 1e-6


def valid_mask(features, lengths):
    lengths = torch.as_tensor(lengths, device=features.device)
    if features.ndim != 3 or lengths.shape != (features.shape[0],):
        raise ValueError('Expected features [B,T,D] and lengths [B]')
    if not torch.isfinite(lengths).all() or ((lengths < 1) | (lengths > features.shape[1])
                                            | (lengths != lengths.long())).any():
        raise ValueError('Valid lengths must be integers in [1,T]')
    return torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]


class RepresentationBranch(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) < 1:
            raise ValueError('Representation dimensions must be positive')
        with torch.random.fork_rng(devices=[]):
            self.projector = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                           nn.Linear(hidden_dim, output_dim))
        self.register_buffer('center', torch.zeros(output_dim))
        self.register_buffer('center_ready', torch.tensor(False))
        self.register_buffer('center_fit_epoch', torch.tensor(0))
        self.register_buffer('center_sample_count', torch.tensor(0))
        self.register_buffer('center_video_count', torch.tensor(0))

    def forward(self, features, lengths):
        mask = valid_mask(features, lengths)
        with torch.autocast(device_type=features.device.type, enabled=False):
            clean = features.float().masked_fill(~mask[..., None], 0)
            z = F.normalize(self.projector(clean), dim=-1, eps=EPS)
            return z.masked_fill(~mask[..., None], 0)


def contrastive_loss(z, logits_c, lengths, class_ids, source_ids, mode='standard',
                     center=None, temperature=.1):
    """Same-event positives across sources; only Normal keys are negatives.

    Weighted partition: each source contributes total key weight 1. Average
    positive logits with the same weights, then average anchors per source.
    """
    if mode not in ('standard', 'mean_shifted'):
        raise ValueError('Contrastive loss requires standard or mean_shifted mode')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    mask = valid_mask(z, lengths)
    classes = torch.as_tensor(class_ids, device=z.device)
    if classes.shape != (len(z),) or len(source_ids) != len(z):
        raise ValueError('Labels and source IDs must match batch size')
    if not torch.isfinite(classes).all() or ((classes < 0) | (classes != classes.long())).any():
        raise ValueError('Class IDs must be nonnegative integers; Normal=0')
    source_map, source_classes, source_codes = {}, {}, []
    for source, label in zip(source_ids, classes.detach().cpu().tolist()):
        if source in source_classes and source_classes[source] != label:
            raise ValueError('One video source cannot have conflicting event labels')
        source_classes[source] = label
        source_codes.append(source_map.setdefault(source, len(source_map)))
    scores = logits_c.squeeze(-1) if logits_c.ndim == 3 and logits_c.shape[-1] == 1 else logits_c
    if scores.shape != mask.shape or not torch.isfinite(scores[mask]).all():
        raise ValueError('Expected finite valid C logits [B,T] or [B,T,1]')
    with torch.autocast(device_type=z.device.type, enabled=False):
        clean = z.float().masked_fill(~mask[..., None], 0)
        if not torch.isfinite(clean).all():
            raise ValueError('Non-finite representation in valid snippets')
        zero = clean.sum() * 0
        vectors, labels, sources = [], [], []
        for i, length in enumerate(torch.as_tensor(lengths).detach().cpu().tolist()):
            length = int(length)
            indices = scores[i, :length].detach().float().sigmoid().topk(min(length, length // 16 + 1)).indices
            vectors.append(clean[i, indices])
            labels.append(classes[i].expand(len(indices)))
            sources.append(torch.full((len(indices),), source_codes[i], device=z.device, dtype=torch.long))
        selected = torch.cat(vectors)
        labels, sources = torch.cat(labels), torch.cat(sources)
        usable = selected.norm(dim=-1) >= EPS
        selected = F.normalize(selected, dim=-1, eps=EPS)
        if mode == 'mean_shifted':
            if center is None or center.shape != (z.shape[-1],) or not torch.isfinite(center).all():
                raise ValueError('Mean-shifted loss needs a finite calibrated center in the same feature space')
            selected = selected - center.detach().to(device=z.device, dtype=torch.float32)
            usable &= selected.norm(dim=-1) >= EPS
        stats = {'anchors_used': 0, 'anchors_without_partner': 0, 'anchor_sources': 0,
                 'selected_snippets': len(selected), 'near_center_skipped': int((~usable).sum()),
                 'normal_keys': 0, 'positive_cosine': 0., 'negative_cosine': 0.,
                 'comparison_norm': selected.detach().norm(dim=-1).mean().item()}
        q = F.normalize(selected[usable], dim=-1, eps=EPS)
        labels, sources = labels[usable], sources[usable]
        normal = labels == 0
        positive = ((labels[:, None] == labels[None, :]) & (labels[:, None] > 0)
                    & (sources[:, None] != sources[None, :]))
        stats['normal_keys'] = int(normal.sum())
        stats['anchors_without_partner'] = int(((labels > 0) & ~positive.any(dim=1)).sum())
        anchors = (labels > 0) & positive.any(dim=1) & normal.any()
        if not anchors.any():
            return zero, stats
        counts = torch.bincount(sources, minlength=len(source_map)).clamp_min(1).float()
        key_weights = counts[sources].reciprocal()
        anchor_sources = sources[anchors]
        similarities = q[anchors] @ q.T
        logits = similarities / temperature
        positive = positive[anchors]
        allowed = positive | normal[None, :]
        positive_weights = positive.float() * key_weights[None, :]
        positive_weights = positive_weights / positive_weights.sum(dim=1, keepdim=True)
        partition = torch.logsumexp((logits + key_weights.log()[None, :]).masked_fill(~allowed, -torch.inf), dim=1)
        losses = partition - (positive_weights * logits).sum(dim=1)
        sums = losses.new_zeros(len(source_map)).index_add(0, anchor_sources, losses)
        anchor_counts = torch.bincount(anchor_sources, minlength=len(source_map))
        present = anchor_counts > 0
        loss = (sums[present] / anchor_counts[present]).mean()
        normal_weights = normal.float() * key_weights
        normal_weights = normal_weights / normal_weights.sum()
        stats.update(anchors_used=int(anchors.sum()), anchor_sources=int(present.sum()),
                     positive_cosine=(similarities.detach() * positive_weights).sum(dim=1).mean().item(),
                     negative_cosine=(similarities.detach() * normal_weights).sum(dim=1).mean().item())
        return loss, stats


def calibrate_center(model, dataset, args, device, epoch):
    """Fixed mean from distinct Normal train sources, independent of loader RNG."""
    if getattr(dataset, 'test_mode', True) or not getattr(dataset, 'normal', False):
        raise ValueError('Calibration must use the Normal training dataset')
    first = {}
    for index, identity in enumerate(dataset.source_ids):
        first.setdefault(identity, index)
    identities = list(first)
    order = np.random.default_rng(args.center_seed).permutation(len(identities))[:args.center_videos]
    modes = [(module, module.training) for module in model.modules()]
    rng = capture_rng()
    means, sample_count = [], 0
    try:
        model.eval()
        with torch.no_grad():
            for position in order:
                features, label, length, _ = dataset[first[identities[int(position)]]]
                if label != 'Normal':
                    raise ValueError('Calibration encountered a non-Normal sample')
                lengths = torch.tensor([length], device=device)
                encoded = model.encode_video(features.unsqueeze(0).to(device), None, lengths)
                z = model.representation(encoded, lengths)[0, :int(length)]
                positions = torch.linspace(0, int(length) - 1, min(args.center_snippets, int(length))).long().to(z.device)
                samples = z[positions].double().cpu()
                if not torch.isfinite(samples).all() or (samples.norm(dim=-1) < EPS).any():
                    raise ValueError('Degenerate/nonfinite calibration representation')
                means.append(samples.mean(dim=0))
                sample_count += len(samples)
        if not means:
            raise ValueError('No Normal train source available for center calibration')
        center = torch.stack(means).mean(dim=0).float().to(device)
        branch = model.representation
        branch.center.copy_(center)
        branch.center_ready.fill_(True)
        branch.center_fit_epoch.fill_(epoch)
        branch.center_sample_count.fill_(sample_count)
        branch.center_video_count.fill_(len(means))
        digest = hashlib.sha256('\n'.join(identities[int(i)] for i in order).encode('utf-8')).hexdigest()
        return dict(epoch=epoch, videos=len(means), samples=sample_count,
                    center_norm=center.norm().item(), source_ids_sha256=digest)
    finally:
        restore_rng(rng)
        for module, training in modes:
            module.training = training
