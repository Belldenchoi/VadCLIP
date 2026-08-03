import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
from pathlib import Path

from model import CLIPVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
from utils.soft_topk import topk_pool, video_topk_size
from utils.training_log import TrainingLogger
import xd_option

def CLASM(logits, labels, lengths, device, temperature=1.0,
          pooling='soft',
          multi_k_percentages=(1.0, 5.0, 10.0, 20.0)):
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    # Pooling and log-softmax can overflow in FP16 after several epochs even
    # when the model parameters remain finite. Keep this loss path in FP32.
    with torch.autocast(device_type=device, enabled=False):
        logits = logits.float()
        instance_logits = []
        for i in range(logits.shape[0]):
            length = int(lengths[i].item())
            pooled = topk_pool(
                logits[i, :length], video_topk_size(length), pooling,
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
          multi_k_percentages=(1.0, 5.0, 10.0, 20.0)):
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    instance_logits = []
    for i in range(logits.shape[0]):
        length = int(lengths[i].item())
        pooled = topk_pool(
            logits[i, :length], video_topk_size(length), pooling, temperature,
            multi_k_percentages
        )
        instance_logits.append(pooled)
    instance_logits = torch.stack(instance_logits)

    # BCE on probabilities is intentionally kept in FP32 because PyTorch
    # disallows this operation inside CUDA autocast.
    with torch.autocast(device_type=device, enabled=False):
        clsloss = F.binary_cross_entropy(
            instance_logits.float(), labels.float()
        )
    return clsloss

def train(model, train_loader, test_loader, args, label_map: dict, device):
    model.to(device)

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
        f"multi_k_percentages={args.multi_k_percentages}"
    )
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        optimizer.zero_grad(set_to_none=True)
        for i, item in enumerate(train_loader):
            step = 0
            visual_feat, text_labels, feat_lengths = item
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            with torch.autocast(device_type=device, dtype=torch.float16,
                                enabled=use_amp):
                text_features, logits1, logits2 = model(
                    visual_feat, None, prompt_text, feat_lengths
                )
                loss1 = CLAS2(logits1, text_labels, feat_lengths, device,
                              args.c_topk_temperature, args.topk_pooling,
                              args.multi_k_percentages)
                loss2 = CLASM(logits2, text_labels, feat_lengths, device,
                              args.a_topk_temperature, args.topk_pooling,
                              args.multi_k_percentages)
                loss3 = torch.zeros(1, device=device)
                text_feature_normal = text_features[0] / text_features[0].norm(
                    dim=-1, keepdim=True
                )
                for j in range(1, text_features.shape[0]):
                    text_feature_abr = text_features[j] / text_features[j].norm(
                        dim=-1, keepdim=True
                    )
                    loss3 += torch.abs(text_feature_normal @ text_feature_abr)
                loss3 = loss3 / 6
                loss = loss1 + loss2 + loss3 * 1e-4

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()
            loss_total3 += loss3.item()
            scaler.scale(loss / accumulation_steps).backward()
            should_step = ((i + 1) % accumulation_steps == 0 or
                           i + 1 == len(train_loader))
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if ((i + 1) % log_interval == 0 or
                    i + 1 == len(train_loader)):
                peak_vram = (torch.cuda.max_memory_allocated() / 1024 ** 3
                             if device == "cuda" else 0.0)
                logger.log(
                    f"epoch={e + 1}/{args.max_epoch} "
                    f"batch={i + 1}/{len(train_loader)} "
                    f"loss={loss.item():.4f} loss1={loss1.item():.4f} "
                    f"loss2={loss2.item():.4f} loss3={loss3.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"peak_vram={peak_vram:.2f}GB"
                )
            step += i * train_loader.batch_size
            if step % 4800 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                
        peak_vram = (torch.cuda.max_memory_allocated() / 1024 ** 3
                     if device == "cuda" else 0.0)
        logger.log(
            f"epoch_summary={e + 1}/{args.max_epoch} "
            f"batches={len(train_loader)} "
            f"avg_loss1={loss_total1 / len(train_loader):.4f} "
            f"avg_loss2={loss_total2 / len(train_loader):.4f} "
            f"avg_loss3={loss_total3 / len(train_loader):.4f} "
            f"peak_vram={peak_vram:.2f}GB"
        )
        scheduler.step()
        if args.skip_eval:
            continue

        AUC, AP, mAP = test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)

        if AP > ap_best:
            ap_best = AP 
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best}
            torch.save(checkpoint, args.checkpoint_path)

        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    if args.skip_eval:
        torch.save(model.state_dict(), args.model_path)
    else:
        checkpoint = torch.load(args.checkpoint_path)
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
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    train_dataset = XDDataset(
        args.visual_length, args.train_list, False, label_map,
        actions=args.train_actions,
        max_samples_per_label=args.max_samples_per_label
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if len(train_loader) == 0:
        raise ValueError("No clips match the selected training actions.")

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Training clips: {len(train_dataset)} | AMP: {args.amp} | "
          f"batch: {args.batch_size} | accumulation: "
          f"{args.gradient_accumulation_steps}")

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length,
        args.visual_width, args.visual_head, args.visual_layers,
        args.attn_window, args.prompt_prefix, args.prompt_postfix, device
    )
    train(model, train_loader, test_loader, args, label_map, device)
