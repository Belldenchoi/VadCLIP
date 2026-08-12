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
from topk_pooling import temporal_segment_pool, topk_pool, video_topk_size
from training_log import TrainingLogger
import options as ucf_option


def resolve_pool_k(length, sample_index, instance_k=None):
    if instance_k is None:
        return video_topk_size(length)
    return max(1, min(length, int(instance_k[sample_index].item())))


def CLASM(logits, labels, lengths, device, temperature=1.0,
          pooling='soft',
          multi_k_percentages=(1.0, 5.0, 10.0, 20.0),
          instance_k=None, temporal_segment=False,
          temporal_smoothing_kernel=1):
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
            if temporal_segment:
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
          instance_k=None):
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    instance_logits = []
    for i in range(logits.shape[0]):
        length = int(lengths[i].item())
        pooled = topk_pool(
            logits[i, :length],
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
    model.to(device)
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
        f"multi_k_percentages={args.multi_k_percentages} "
        f"adaptive_instance_selection={args.adaptive_instance_selection} "
        f"ais_score_threshold={args.ais_score_threshold} "
        f"ais_min_k={args.ais_min_k} "
        f"temporal_segment_topk={args.temporal_segment_topk} "
        f"temporal_segment_start_epoch="
        f"{args.temporal_segment_start_epoch} "
        f"temporal_smoothing_kernel={args.temporal_smoothing_kernel}"
    )
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    start_epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=device, weights_only=False
        )
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
        model.train()
        temporal_segment_active = (
            args.temporal_segment_topk and
            e + 1 >= args.temporal_segment_start_epoch
        )
        if args.temporal_segment_topk:
            logger.log(
                f"epoch={e + 1} temporal_segment_active="
                f"{temporal_segment_active} "
                "c_branch_pooling=original_hard_topk "
                f"a_branch_pooling="
                f"{'temporal_segment' if temporal_segment_active else 'original_hard_topk'}"
            )
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
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
                loss1 = CLAS2(logits1, text_labels, feat_lengths, device,
                              args.c_topk_temperature, args.topk_pooling,
                              args.multi_k_percentages,
                              None if ais_selection is None
                              else ais_selection.batch_k)
                loss2 = CLASM(logits2, text_labels, feat_lengths, device,
                              args.a_topk_temperature, args.topk_pooling,
                              args.multi_k_percentages,
                              None if ais_selection is None
                              else ais_selection.batch_k,
                              temporal_segment_active,
                              args.temporal_smoothing_kernel)
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
                loss = loss1 + loss2 + loss3

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()
            loss_total3 += loss3.item()
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
                logger.log(
                    f"epoch={e + 1}/{args.max_epoch} batch={i + 1}/{num_batches} "
                    f"loss={loss.item():.4f} loss1={loss1.item():.4f} "
                    f"loss2={loss2.item():.4f} loss3={loss3.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"{ais_text}"
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
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
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
            f"{ais_epoch_text}"
            f"peak_vram={peak_vram:.2f}GB"
        )
        scheduler.step()

        if args.skip_eval:
            checkpoint = {
                'epoch': e,
                'next_epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
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
