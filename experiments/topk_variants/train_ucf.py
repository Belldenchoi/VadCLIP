import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import sys
from pathlib import Path

ORIGINAL_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(ORIGINAL_SRC))

from model import CLIPVAD
from test_ucf import test
from dataset_variants import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
from adaptive_instance_selection import select_adaptive_instance_k
from dual_k_selection import (
    dual_constraint_loss,
    selection_statistics,
    select_a_branch,
    select_c_branch,
    update_dual_lambda,
)
from temporal_smoothness import (
    a_branch_temporal_smoothness,
    c_branch_temporal_smoothness,
)
from topk_pooling import (
    smooth_temporal_scores,
    temporal_segment_pool,
    topk_pool,
    video_topk_size,
)
from training_log import TrainingLogger
from temperature_schedule import (
    epoch_temperatures,
    temperature_config,
    validate_checkpoint_temperature,
    validate_temperature_config,
)
import options as ucf_option


def resolve_pool_k(length, sample_index, instance_k=None):
    if instance_k is None:
        return video_topk_size(length)
    return max(1, min(length, int(instance_k[sample_index].item())))


def CLASM(logits, labels, lengths, device, temperature=1.0,
          pooling='soft',
          multi_k_percentages=(1.0, 5.0, 10.0, 20.0),
          instance_k=None, temporal_segment=False,
          temporal_smoothing_kernel=1, selection_weights=None):
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    # Pooling and log-softmax can overflow in FP16 after several epochs even
    # when the model parameters remain finite. Keep this loss path in FP32.
    with torch.autocast(device_type=device, enabled=False):
        logits = logits.float()
        instance_logits = []
        for i in range(logits.shape[0]):
            length = int(lengths[i].item())
            pool_k = resolve_pool_k(length, i, instance_k)
            if selection_weights is not None:
                weights = selection_weights[i, :length].float()
                pooled = (
                    weights * logits[i, :length].float()
                ).sum(dim=0) / (weights.sum(dim=0) + 1e-6)
            elif temporal_segment:
                pooled = temporal_segment_pool(
                    logits[i, :length], pool_k,
                    temporal_smoothing_kernel
                )
            else:
                pooled = topk_pool(
                    logits[i, :length], pool_k, pooling,
                    temperature, multi_k_percentages
                )
            instance_logits.append(pooled)
        instance_logits = torch.stack(instance_logits)
        milloss = -torch.mean(torch.sum(
            labels.float() * F.log_softmax(instance_logits, dim=1), dim=1
        ), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device, temperature=1.0,
          pooling='soft',
          multi_k_percentages=(1.0, 5.0, 10.0, 20.0),
          instance_k=None, c_temporal_smoothing_kernel=1,
          selection_weights=None, temporal_segment=False):
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    instance_logits = []
    for i in range(logits.shape[0]):
        length = int(lengths[i].item())
        raw_scores = logits[i, :length]
        if selection_weights is not None:
            valid_scores = smooth_temporal_scores(
                raw_scores, c_temporal_smoothing_kernel
            )
            weights = selection_weights[i, :length].float()
            pooled = (weights * valid_scores).sum() / (weights.sum() + 1e-6)
        elif temporal_segment:
            pooled = temporal_segment_pool(
                raw_scores,
                resolve_pool_k(length, i, instance_k),
                c_temporal_smoothing_kernel,
            )
        else:
            valid_scores = smooth_temporal_scores(
                raw_scores, c_temporal_smoothing_kernel
            )
            pooled = topk_pool(
                valid_scores,
                resolve_pool_k(length, i, instance_k), pooling, temperature,
                multi_k_percentages
            )
        instance_logits.append(pooled)
    instance_logits = torch.stack(instance_logits)

    # BCE on probabilities is intentionally kept in FP32 because PyTorch
    # disallows this operation inside CUDA autocast. Clamping avoids the very
    # large BCE gradients at exactly 0/1 from destabilizing training.
    with torch.autocast(device_type=device, enabled=False):
        clsloss = F.binary_cross_entropy(
            instance_logits.float().clamp(1e-6, 1.0 - 1e-6),
            labels.float()
        )
    return clsloss

def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    validate_temperature_config(args)
    model.to(device)
    if args.dual_k_diagnostics_only and not args.dual_k:
        raise ValueError("--dual-k-diagnostics-only requires --dual-k")
    if args.dual_k and args.adaptive_instance_selection:
        raise ValueError("Dual-K cannot be combined with AIS")
    if args.dual_k and args.temporal_segment_topk:
        raise ValueError("Dual-K cannot be combined with temporal segment Top-K")
    if args.dual_k and args.c_temporal_segment_topk:
        raise ValueError(
            "Dual-K cannot be combined with C-branch temporal segment Top-K"
        )
    if args.dual_k and args.topk_pooling != 'mean':
        raise ValueError("Use --topk-pooling mean with --dual-k")
    if args.dual_k and not (0.0 <= args.dual_k_c_budget <= 1.0 and
                            0.0 <= args.dual_k_a_budget <= 1.0):
        raise ValueError("Dual-K uncertainty budgets must be in [0, 1]")
    if (args.dual_k and
            (args.dual_k_c_temperature <= 0 or
             args.dual_k_a_temperature <= 0)):
        raise ValueError("Dual-K temperatures must be positive")
    if (args.dual_k and
            (args.dual_k_c_lambda < 0 or args.dual_k_a_lambda < 0)):
        raise ValueError("Initial Dual-K variables must be non-negative")
    if args.dual_k and args.dual_k_dual_lr < 0:
        raise ValueError("--dual-k-dual-lr must be non-negative")
    if args.dual_k and args.dual_k_loss_weight < 0:
        raise ValueError("--dual-k-loss-weight must be non-negative")
    if args.adaptive_instance_selection and args.topk_pooling != 'mean':
        raise ValueError(
            "Adaptive Instance Selection is a separate mean Top-K method. "
            "Use --topk-pooling mean with --adaptive-instance-selection."
        )
    if not 0.0 <= args.ais_score_threshold <= 1.0:
        raise ValueError("--ais-score-threshold must be in [0, 1]")
    if args.ais_min_k < 1:
        raise ValueError("--ais-min-k must be at least 1")
    if args.temporal_segment_topk and args.topk_pooling != 'mean':
        raise ValueError(
            "Temporal segment Top-K is an isolated hard Top-K experiment. "
            "Use --topk-pooling mean."
        )
    if args.temporal_segment_topk and args.adaptive_instance_selection:
        raise ValueError(
            "Temporal segment Top-K cannot be combined with Adaptive "
            "Instance Selection in the first experiment."
        )
    if (args.temporal_segment_topk and
            args.temporal_segment_start_epoch < 1):
        raise ValueError("--temporal-segment-start-epoch must be at least 1")
    if (args.temporal_segment_topk and
            (args.temporal_smoothing_kernel < 1 or
             args.temporal_smoothing_kernel % 2 == 0)):
        raise ValueError(
            "--temporal-smoothing-kernel must be a positive odd integer"
        )
    if args.c_temporal_segment_topk and args.topk_pooling != 'mean':
        raise ValueError(
            "C-branch temporal segment Top-K is an isolated hard Top-K "
            "experiment. Use --topk-pooling mean."
        )
    if args.c_temporal_segment_topk and args.adaptive_instance_selection:
        raise ValueError(
            "C-branch temporal segment Top-K cannot be combined with "
            "Adaptive Instance Selection."
        )
    if (args.c_temporal_segment_topk and
            args.c_temporal_segment_start_epoch < 1):
        raise ValueError(
            "--c-temporal-segment-start-epoch must be at least 1"
        )
    if (args.c_temporal_smoothing_kernel < 1 or
            args.c_temporal_smoothing_kernel % 2 == 0):
        raise ValueError(
            "--c-temporal-smoothing-kernel must be a positive odd integer"
        )
    smoothness_enabled = args.temporal_smoothness_branch != 'none'
    if (smoothness_enabled and
            args.temporal_smoothness_start_epoch < 1):
        raise ValueError(
            "--temporal-smoothness-start-epoch must be at least 1"
        )
    if (args.temporal_smoothness_branch in ('c', 'both') and
            args.c_temporal_smoothness_weight <= 0):
        raise ValueError(
            "--c-temporal-smoothness-weight must be positive for C smoothness"
        )
    if (args.temporal_smoothness_branch in ('a', 'both') and
            args.a_temporal_smoothness_weight <= 0):
        raise ValueError(
            "--a-temporal-smoothness-weight must be positive for A smoothness"
        )
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    use_amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accumulation_steps = max(1, args.gradient_accumulation_steps)
    log_interval = max(1, args.log_interval)
    logger = TrainingLogger(args.log_path, append=args.use_checkpoint)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    logger.log(
        f"device={device} amp={use_amp} epochs={args.max_epoch} "
        f"batch_size={args.batch_size} accumulation={accumulation_steps} "
        f"actions={args.train_actions or 'all'} "
        f"topk_pooling={args.topk_pooling} "
        f"topk_temperature_config={temperature_config(args)} "
        f"multi_k_percentages={args.multi_k_percentages} "
        f"adaptive_instance_selection={args.adaptive_instance_selection} "
        f"dual_k={args.dual_k} "
        f"dual_k_diagnostics_only={args.dual_k_diagnostics_only} "
        f"dual_k_evidence_normalization="
        f"{args.dual_k_evidence_normalization} "
        f"dual_k_a_risk_scope={args.dual_k_a_risk_scope} "
        f"dual_k_c_threshold={args.dual_k_c_threshold} "
        f"dual_k_a_threshold={args.dual_k_a_threshold} "
        f"dual_k_c_temperature={args.dual_k_c_temperature} "
        f"dual_k_a_temperature={args.dual_k_a_temperature} "
        f"dual_k_c_budget={args.dual_k_c_budget} "
        f"dual_k_a_budget={args.dual_k_a_budget} "
        f"dual_k_c_initial_lambda={args.dual_k_c_lambda} "
        f"dual_k_a_initial_lambda={args.dual_k_a_lambda} "
        f"dual_k_dual_lr={args.dual_k_dual_lr} "
        f"dual_k_loss_weight={args.dual_k_loss_weight} "
        f"dual_k_c_uncertainty=normalized_binary_entropy "
        f"dual_k_a_uncertainty=clamped_class_margin "
        f"ais_score_threshold={args.ais_score_threshold} "
        f"ais_min_k={args.ais_min_k} "
        f"temporal_segment_topk={args.temporal_segment_topk} "
        f"temporal_segment_start_epoch="
        f"{args.temporal_segment_start_epoch} "
        f"c_temporal_segment_topk={args.c_temporal_segment_topk} "
        f"c_temporal_segment_start_epoch="
        f"{args.c_temporal_segment_start_epoch} "
        f"c_temporal_smoothing_kernel="
        f"{args.c_temporal_smoothing_kernel} "
        f"temporal_smoothing_kernel={args.temporal_smoothing_kernel} "
        f"temporal_smoothness_branch={args.temporal_smoothness_branch} "
        f"temporal_smoothness_start_epoch="
        f"{args.temporal_smoothness_start_epoch} "
        f"c_temporal_smoothness_weight="
        f"{args.c_temporal_smoothness_weight} "
        f"a_temporal_smoothness_weight="
        f"{args.a_temporal_smoothness_weight}"
    )
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    start_epoch = 0
    dual_lambda_c = float(args.dual_k_c_lambda)
    dual_lambda_a = float(args.dual_k_a_lambda)

    if args.use_checkpoint == True:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=device, weights_only=False
        )
        validate_checkpoint_temperature(args, checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        if (device == "cuda" and
                checkpoint.get('cuda_rng_state_all') is not None):
            torch.cuda.set_rng_state_all([
                state.cpu() for state in checkpoint['cuda_rng_state_all']
            ])
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])
        if args.dual_k:
            dual_lambda_c = float(
                checkpoint.get('dual_lambda_c', dual_lambda_c)
            )
            dual_lambda_a = float(
                checkpoint.get('dual_lambda_a', dual_lambda_a)
            )
        start_epoch = checkpoint.get(
            'next_epoch', checkpoint.get('epoch', -1) + 1
        )
        ap_best = checkpoint.get('ap', 0)
        print("checkpoint info:")
        print("resume_epoch:", start_epoch + 1, "ap:", ap_best)
        logger.log(
            f"checkpoint_resumed={args.checkpoint_path} "
            f"next_epoch={start_epoch + 1}"
        )
        del checkpoint
        if device == "cuda":
            torch.cuda.empty_cache()

    for e in range(start_epoch, args.max_epoch):
        c_temperature, a_temperature = epoch_temperatures(args, e + 1)
        logger.log(
            f"epoch={e + 1} "
            f"a_topk_temperature_schedule={args.a_topk_temperature_schedule} "
            f"c_topk_temperature={c_temperature:.6f} "
            f"a_topk_temperature={a_temperature:.6f}"
        )
        model.train()
        temporal_segment_active = (
            args.temporal_segment_topk and
            e + 1 >= args.temporal_segment_start_epoch
        )
        c_temporal_segment_active = (
            args.c_temporal_segment_topk and
            e + 1 >= args.c_temporal_segment_start_epoch
        )
        temporal_smoothness_active = (
            smoothness_enabled and
            e + 1 >= args.temporal_smoothness_start_epoch
        )
        if args.temporal_segment_topk or args.c_temporal_segment_topk:
            c_branch_pooling = (
                'contiguous_conv1d_topk'
                if c_temporal_segment_active
                else ('fixed_conv1d_hard_topk'
                      if args.c_temporal_smoothing_kernel > 1
                      else 'original_hard_topk')
            )
            logger.log(
                f"epoch={e + 1} temporal_segment_active="
                f"{temporal_segment_active} "
                f"c_temporal_segment_active={c_temporal_segment_active} "
                f"c_branch_pooling={c_branch_pooling} "
                f"a_branch_pooling="
                f"{'temporal_segment' if temporal_segment_active else 'original_hard_topk'}"
            )
        if smoothness_enabled:
            logger.log(
                f"epoch={e + 1} temporal_smoothness_active="
                f"{temporal_smoothness_active} "
                f"temporal_smoothness_branch="
                f"{args.temporal_smoothness_branch}"
            )
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        loss_total_smooth_c = 0
        loss_total_smooth_a = 0
        loss_total_weighted_smooth_c = 0
        loss_total_weighted_smooth_a = 0
        ais_pair_count = 0
        ais_k_total = 0.0
        ais_confidence_total = 0.0
        ais_confident_count_total = 0.0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        num_batches = min(len(normal_loader), len(anomaly_loader))
        optimizer.zero_grad(set_to_none=True)
        for i in range(num_batches):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            with torch.autocast(device_type=device, dtype=torch.float16,
                                enabled=use_amp):
                text_features, logits1, logits2 = model(
                    visual_features, None, prompt_text, feat_lengths
                )
                ais_selection = None
                dual_c_selection = None
                dual_a_selection = None
                if args.adaptive_instance_selection:
                    pair_count = normal_features.shape[0]
                    c_probabilities = torch.sigmoid(
                        logits1.detach()
                    ).squeeze(-1)
                    ais_selection = select_adaptive_instance_k(
                        negative_scores=c_probabilities[:pair_count],
                        positive_scores=c_probabilities[pair_count:],
                        negative_lengths=feat_lengths[:pair_count],
                        positive_lengths=feat_lengths[pair_count:],
                        score_threshold=args.ais_score_threshold,
                        min_k=args.ais_min_k,
                    )
                if args.dual_k:
                    normalize_dual_evidence = (
                        args.dual_k_evidence_normalization == 'per_video'
                    )
                    dual_c_selection = select_c_branch(
                        logits1, feat_lengths, dual_lambda_c,
                        threshold=args.dual_k_c_threshold,
                        temperature=args.dual_k_c_temperature,
                        normalize_evidence=normalize_dual_evidence,
                    )
                    dual_a_selection = select_a_branch(
                        logits2, feat_lengths, dual_lambda_a,
                        threshold=args.dual_k_a_threshold,
                        temperature=args.dual_k_a_temperature,
                        normalize_evidence=normalize_dual_evidence,
                    )
                loss1 = CLAS2(logits1, text_labels, feat_lengths, device,
                              c_temperature, args.topk_pooling,
                              args.multi_k_percentages,
                              None if ais_selection is None
                              else ais_selection.batch_k,
                              args.c_temporal_smoothing_kernel,
                              None if (dual_c_selection is None or
                                       args.dual_k_diagnostics_only)
                              else dual_c_selection.weights,
                              temporal_segment=c_temporal_segment_active)
                loss2 = CLASM(logits2, text_labels, feat_lengths, device,
                              a_temperature, args.topk_pooling,
                              args.multi_k_percentages,
                              None if ais_selection is None
                              else ais_selection.batch_k,
                              temporal_segment_active,
                              args.temporal_smoothing_kernel,
                              None if (dual_a_selection is None or
                                       args.dual_k_diagnostics_only)
                              else dual_a_selection.weights)
                loss3 = torch.zeros(1, device=device)
                if not getattr(model, 'uses_fixed_prototypes', False):
                    text_feature_normal = (
                        text_features[0] / text_features[0].norm(
                            dim=-1, keepdim=True
                        )
                    )
                    for j in range(1, text_features.shape[0]):
                        text_feature_abr = (
                            text_features[j] / text_features[j].norm(
                                dim=-1, keepdim=True
                            )
                        )
                        loss3 += torch.abs(
                            text_feature_normal @ text_feature_abr
                        )
                    loss3 = loss3 / (text_features.shape[0] - 1) * 1e-1
                loss_smooth_c = torch.zeros(
                    (), device=device, dtype=torch.float32
                )
                loss_smooth_a = torch.zeros(
                    (), device=device, dtype=torch.float32
                )
                if temporal_smoothness_active:
                    if args.temporal_smoothness_branch in ('c', 'both'):
                        loss_smooth_c = c_branch_temporal_smoothness(
                            logits1, feat_lengths
                        )
                    if args.temporal_smoothness_branch in ('a', 'both'):
                        loss_smooth_a = a_branch_temporal_smoothness(
                            logits2, feat_lengths
                        )
                weighted_smooth_c = (
                    args.c_temporal_smoothness_weight * loss_smooth_c
                )
                weighted_smooth_a = (
                    args.a_temporal_smoothness_weight * loss_smooth_a
                )
                loss = (
                    loss1 + loss2 + loss3
                    + weighted_smooth_c + weighted_smooth_a
                )
                dual_loss_c = torch.zeros((), device=device)
                dual_loss_a = torch.zeros((), device=device)
                if args.dual_k:
                    a_constraint_weights = (
                        text_labels
                        if args.dual_k_a_risk_scope == 'target_class'
                        else None
                    )
                    dual_loss_c = dual_constraint_loss(
                        dual_c_selection, args.dual_k_c_budget
                    )
                    dual_loss_a = dual_constraint_loss(
                        dual_a_selection, args.dual_k_a_budget,
                        a_constraint_weights,
                    )
                    if not args.dual_k_diagnostics_only:
                        loss = loss + args.dual_k_loss_weight * (
                            dual_lambda_c * dual_loss_c
                            + dual_lambda_a * dual_loss_a
                        )

            if args.dual_k and not args.dual_k_diagnostics_only:
                dual_lambda_c = update_dual_lambda(
                    dual_lambda_c, dual_loss_c, args.dual_k_dual_lr
                )
                dual_lambda_a = update_dual_lambda(
                    dual_lambda_a, dual_loss_a, args.dual_k_dual_lr
                )

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()
            loss_total3 += loss3.item()
            loss_total_smooth_c += loss_smooth_c.item()
            loss_total_smooth_a += loss_smooth_a.item()
            loss_total_weighted_smooth_c += weighted_smooth_c.item()
            loss_total_weighted_smooth_a += weighted_smooth_a.item()
            if ais_selection is not None:
                ais_pair_count += ais_selection.pair_k.numel()
                ais_k_total += ais_selection.pair_k.float().sum().item()
                ais_confidence_total += (
                    ais_selection.confidence.sum().item()
                )
                ais_confident_count_total += (
                    ais_selection.confident_positive_count.float().sum().item()
                )
            scaler.scale(loss / accumulation_steps).backward()
            should_step = (i + 1) % accumulation_steps == 0 or i + 1 == num_batches
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if (i + 1) % log_interval == 0 or i + 1 == num_batches:
                peak_vram = (torch.cuda.max_memory_allocated() / 1024 ** 3
                             if device == "cuda" else 0.0)
                ais_text = ""
                if ais_selection is not None:
                    ais_text = (
                        f"ais_k_mean="
                        f"{ais_selection.pair_k.float().mean().item():.3f} "
                        f"ais_k_min={int(ais_selection.pair_k.min().item())} "
                        f"ais_k_max={int(ais_selection.pair_k.max().item())} "
                        f"ais_omega_mean="
                        f"{ais_selection.confidence.mean().item():.4f} "
                        f"ais_confident_mean="
                        f"{ais_selection.confident_positive_count.float().mean().item():.3f} "
                    )
                dual_text = ""
                if args.dual_k:
                    dual_c_stats = selection_statistics(
                        dual_c_selection
                    )
                    dual_a_stats = selection_statistics(
                        dual_a_selection, text_labels
                    )
                    dual_a_all_class_k = (
                        dual_a_selection.effective_k.mean()
                    )
                    dual_a_all_class_risk = (
                        dual_a_selection.uncertainty_ratio.mean()
                    )
                    dual_text = (
                        f"dual_k_c_mean="
                        f"{dual_c_stats['k_mean'].item():.3f} "
                        f"dual_k_a_target_mean="
                        f"{dual_a_stats['k_mean'].item():.3f} "
                        f"dual_k_a_all_class_mean="
                        f"{dual_a_all_class_k.item():.3f} "
                        f"dual_support_ratio_c="
                        f"{dual_c_stats['support_ratio'].item():.4f} "
                        f"dual_support_ratio_a_target="
                        f"{dual_a_stats['support_ratio'].item():.4f} "
                        f"dual_r_c={dual_c_stats['risk'].item():.4f} "
                        f"dual_r_c_q40={dual_c_stats['risk_q40'].item():.4f} "
                        f"dual_r_c_q50={dual_c_stats['risk_q50'].item():.4f} "
                        f"dual_r_c_q60={dual_c_stats['risk_q60'].item():.4f} "
                        f"dual_r_a_target="
                        f"{dual_a_stats['risk'].item():.4f} "
                        f"dual_r_a_target_q40="
                        f"{dual_a_stats['risk_q40'].item():.4f} "
                        f"dual_r_a_target_q50="
                        f"{dual_a_stats['risk_q50'].item():.4f} "
                        f"dual_r_a_target_q60="
                        f"{dual_a_stats['risk_q60'].item():.4f} "
                        f"dual_r_a_all_class="
                        f"{dual_a_all_class_risk.item():.4f} "
                        f"dual_r_a_constraint="
                        f"{(dual_loss_a + args.dual_k_a_budget).item():.4f} "
                        f"dual_violation_c={dual_loss_c.item():.4f} "
                        f"dual_violation_a={dual_loss_a.item():.4f} "
                        f"dual_lambda_c={dual_lambda_c:.4f} "
                        f"dual_lambda_a={dual_lambda_a:.4f} "
                        f"dual_e_c_mean={dual_c_stats['e_mean'].item():.4f} "
                        f"dual_e_c_std={dual_c_stats['e_std'].item():.4f} "
                        f"dual_e_c_min={dual_c_stats['e_min'].item():.4f} "
                        f"dual_e_c_max={dual_c_stats['e_max'].item():.4f} "
                        f"dual_e_c_norm_mean="
                        f"{dual_c_stats['e_norm_mean'].item():.4f} "
                        f"dual_e_c_norm_std="
                        f"{dual_c_stats['e_norm_std'].item():.4f} "
                        f"dual_e_a_target_mean="
                        f"{dual_a_stats['e_mean'].item():.4f} "
                        f"dual_e_a_target_std="
                        f"{dual_a_stats['e_std'].item():.4f} "
                        f"dual_e_a_target_min="
                        f"{dual_a_stats['e_min'].item():.4f} "
                        f"dual_e_a_target_max="
                        f"{dual_a_stats['e_max'].item():.4f} "
                        f"dual_e_a_norm_target_mean="
                        f"{dual_a_stats['e_norm_mean'].item():.4f} "
                        f"dual_e_a_norm_target_std="
                        f"{dual_a_stats['e_norm_std'].item():.4f} "
                        f"dual_u_c_mean={dual_c_stats['u_mean'].item():.4f} "
                        f"dual_u_c_std={dual_c_stats['u_std'].item():.4f} "
                        f"dual_u_a_target_mean="
                        f"{dual_a_stats['u_mean'].item():.4f} "
                        f"dual_u_a_target_std="
                        f"{dual_a_stats['u_std'].item():.4f} "
                    )
                logger.log(
                    f"epoch={e + 1}/{args.max_epoch} batch={i + 1}/{num_batches} "
                    f"loss={loss.item():.4f} loss1={loss1.item():.4f} "
                    f"loss2={loss2.item():.4f} loss3={loss3.item():.4f} "
                    f"loss_smooth_c={loss_smooth_c.item():.6f} "
                    f"weighted_smooth_c={weighted_smooth_c.item():.6f} "
                    f"loss_smooth_a={loss_smooth_a.item():.6f} "
                    f"weighted_smooth_a={weighted_smooth_a.item():.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"c_topk_temperature={c_temperature:.6f} "
                    f"a_topk_temperature={a_temperature:.6f} "
                    f"{ais_text}"
                    f"{dual_text}"
                    f"peak_vram={peak_vram:.2f}GB"
                )
            step += i * normal_loader.batch_size * 2
            if not args.skip_eval and step % 1280 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                AUC, AP = test(model, testloader, args.visual_length,
                               prompt_text, gt, gtsegments, gtlabels, device,
                               logger=logger)
                AP = AUC

                if AP > ap_best:
                    ap_best = AP 
                    checkpoint = {
                        'epoch': e,
                        'topk_temperature_config': temperature_config(args),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'dual_lambda_c': dual_lambda_c,
                        'dual_lambda_a': dual_lambda_a,
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                
        peak_vram = (torch.cuda.max_memory_allocated() / 1024 ** 3
                     if device == "cuda" else 0.0)
        ais_epoch_text = ""
        if ais_pair_count:
            ais_epoch_text = (
                f"ais_k_mean={ais_k_total / ais_pair_count:.3f} "
                f"ais_omega_mean="
                f"{ais_confidence_total / ais_pair_count:.4f} "
                f"ais_confident_mean="
                f"{ais_confident_count_total / ais_pair_count:.3f} "
            )
        logger.log(
            f"epoch_summary={e + 1}/{args.max_epoch} batches={num_batches} "
            f"avg_loss1={loss_total1 / num_batches:.4f} "
            f"avg_loss2={loss_total2 / num_batches:.4f} "
            f"avg_loss3={loss_total3 / num_batches:.4f} "
            f"avg_loss_smooth_c="
            f"{loss_total_smooth_c / num_batches:.6f} "
            f"avg_weighted_smooth_c="
            f"{loss_total_weighted_smooth_c / num_batches:.6f} "
            f"avg_loss_smooth_a="
            f"{loss_total_smooth_a / num_batches:.6f} "
            f"avg_weighted_smooth_a="
            f"{loss_total_weighted_smooth_a / num_batches:.6f} "
            f"{ais_epoch_text}"
            f"peak_vram={peak_vram:.2f}GB"
        )
        scheduler.step()

        if args.skip_eval:
            checkpoint = {
                'epoch': e,
                'next_epoch': e + 1,
                'topk_temperature_config': temperature_config(args),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'dual_lambda_c': dual_lambda_c,
                'dual_lambda_a': dual_lambda_a,
                'ap': ap_best,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state_all': (
                    torch.cuda.get_rng_state_all()
                    if device == "cuda" else None
                ),
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(),
            }
            torch.save(checkpoint, args.checkpoint_path)
            logger.log(
                f"checkpoint_saved={args.checkpoint_path} "
                f"next_epoch={e + 2}"
            )
        
        if not args.skip_eval:
            model_path = Path(args.model_path)
            current_model_path = model_path.with_name(
                f"{model_path.stem}_cur{model_path.suffix}"
            )
            torch.save(model.state_dict(), current_model_path)
            if Path(args.checkpoint_path).exists():
                checkpoint = torch.load(
                    args.checkpoint_path, weights_only=False
                )
                model.load_state_dict(checkpoint['model_state_dict'])

    if args.skip_eval:
        torch.save(model.state_dict(), args.model_path)
    else:
        if not Path(args.checkpoint_path).exists():
            raise RuntimeError(
                "No checkpoint was produced. Use --skip-eval for a short "
                "smoke run or increase the number of training batches."
            )
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        torch.save(checkpoint['model_state_dict'], args.model_path)
    logger.log(f"model_saved={args.model_path}")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    validate_temperature_config(args)
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    normal_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, True,
        max_samples_per_label=args.max_samples_per_label
    )
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, False,
        actions=args.train_actions,
        max_samples_per_label=args.max_samples_per_label
    )
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if len(normal_loader) == 0 or len(anomaly_loader) == 0:
        raise ValueError(
            "The selected training subset is smaller than --batch-size or "
            "contains no matching action."
        )

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Training clips: {len(normal_dataset)} normal, "
          f"{len(anomaly_dataset)} anomaly | AMP: {args.amp} | "
          f"batch: {args.batch_size} | accumulation: "
          f"{args.gradient_accumulation_steps}")

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length,
        args.visual_width, args.visual_head, args.visual_layers,
        args.attn_window, args.prompt_prefix, args.prompt_postfix, device
    )

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
