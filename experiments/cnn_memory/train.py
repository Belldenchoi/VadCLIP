"""Train the isolated CNN-memory experiment; no changes to src are needed."""

import json
import hashlib
from pathlib import Path

try:
    from .config import LABEL_MAP, RESUME_KEYS, model_config, training_parser, validate_args
except ImportError:
    from config import LABEL_MAP, RESUME_KEYS, model_config, training_parser, validate_args


def training_device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The original LGT DistanceAdj uses CUDA. Enable a Kaggle GPU before training.")
    return "cuda"


def main():
    args = training_parser().parse_args()
    validate_args(args)
    # Keep --help and preflight independent of CLIP/torchvision downloads.
    try:
        from .data_check import validate_data
    except ImportError:
        from data_check import validate_data
    summary = validate_data(args)
    print("Data check:", json.dumps(summary), flush=True)
    if args.preflight:
        print("Preflight passed. CLIP was not loaded; no training was executed.")
        return

    if (args.aux_mode == "future_mse" and args.aux_branch != "none" and not args.resume
            and not Path(args.pretrained_cnn).is_file()):
        raise FileNotFoundError(f"Run pretrain.py first; missing CNN encoder: {args.pretrained_cnn}")

    try:
        from . import bootstrap
        from .checkpoint import capture_rng, checkpoint_payload, load_checkpoint, restore_rng
        from .cnn_memory_model import build_model
        from .future_features import future_feature_mse
        from .evaluation import evaluate
    except ImportError:
        import bootstrap
        from checkpoint import capture_rng, checkpoint_payload, load_checkpoint, restore_rng
        from cnn_memory_model import build_model
        from future_features import future_feature_mse
        from evaluation import evaluate

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import MultiStepLR
    from training_log import TrainingLogger
    from utils.dataset import UCFDataset
    from utils.tools import get_batch_label, get_prompt_text
    from ucf_train import CLASM, CLAS2, setup_seed

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError(f"Output directory is not empty: {output}. Choose a new --output-dir.")
    saved = load_checkpoint(args.resume) if args.resume else None
    if saved:
        if saved["kind"] != "last":
            raise ValueError("Resume requires last.pt, which includes optimizer, scheduler and RNG state")
        changed = [key for key in RESUME_KEYS
                   if saved["train_args"].get(key, "prototype" if key == "aux_mode" else None) != getattr(args, key)]
        if args.aux_mode == "future_mse_joint":
            changed.extend(key for key in ("future_horizon", "mse_loss_weight")
                           if saved["train_args"].get(key) != getattr(args, key))
        if changed:
            raise ValueError(f"Resume configuration differs for: {', '.join(changed)}")
        if saved["epoch"] >= args.max_epoch:
            raise ValueError("--max-epoch must exceed the completed checkpoint epoch")
        if output.resolve() != Path(args.resume).resolve().parent:
            raise ValueError("Resume must use the directory containing last.pt as --output-dir")
        if not (output / "best.pt").is_file() and saved["best_metrics"]:
            raise FileNotFoundError("Keep best.pt beside last.pt when resuming")
    output.mkdir(parents=True, exist_ok=True)
    device = training_device()
    logger = TrainingLogger(output / "train.log", append=bool(saved))
    gpu_name = torch.cuda.get_device_name(device) if device == "cuda" else "none"
    logger.log(f"device={device} amp=False epochs={args.max_epoch} seed={args.seed} "
               f"batch_size={args.batch_size} accumulation=1 actions=all topk_pooling=mean "
               f"experiment=cnn_memory aux_mode={args.aux_mode} gpu_name={gpu_name!r}")
    logger.log("experiment=cnn_memory data=" + json.dumps(summary))
    logger.log("args=" + json.dumps(vars(args), sort_keys=True))
    logger.log("pooling=original_hard_topk loss3=original_text_separation "
               "selection_metric=AUC1 selection_split=test "
               f"selection_start_epoch={args.memory_start_epoch} "
               "epoch_policy=continuous_training_no_best_reload")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    setup_seed(args.seed)
    normal = UCFDataset(args.visual_length, args.train_list, False, LABEL_MAP, True)
    abnormal = UCFDataset(args.visual_length, args.train_list, False, LABEL_MAP, False)
    test_data = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    normal_loader = DataLoader(normal, batch_size=args.batch_size, shuffle=True, drop_last=True)
    abnormal_loader = DataLoader(abnormal, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    num_batches = min(len(normal_loader), len(abnormal_loader))
    config = model_config(args)
    model = build_model(config, device).to(device)
    if args.aux_mode == "future_mse_joint":
        logger.log(f"joint_training=True encoder_frozen=False future_horizon={args.future_horizon} "
                   f"mse_loss_weight={args.mse_loss_weight} predictor_evaluation=False")
    if args.aux_mode == "future_mse" and model.memory is not None and not saved:
        try:
            from .future_features import load_pretrained_encoder
        except ImportError:
            from future_features import load_pretrained_encoder
        model.pretraining_info = load_pretrained_encoder(model.memory, args.pretrained_cnn, args.visual_length)
        logger.log("future_pretraining=" + json.dumps(model.pretraining_info))
    if args.aux_mode == "future_mse" and model.memory is not None:
        info = saved.get("pretraining_info") if saved else model.pretraining_info
        csv_hash = hashlib.sha256(Path(args.train_list).read_bytes()).hexdigest()
        if not info or info["train_csv_sha256"] != csv_hash:
            raise ValueError("MSE pretraining and detection must use the same training CSV contents")
    memory_params = [p for p in model.memory.parameters() if p.requires_grad] if model.memory is not None else []
    memory_ids = {id(param) for param in memory_params}
    base_params = [param for param in model.parameters() if param.requires_grad and id(param) not in memory_ids]
    groups = [{"params": base_params, "lr": args.lr}]
    if memory_params:
        groups.append({"params": memory_params, "lr": args.memory_lr})
    optimizer = torch.optim.AdamW(groups)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, gamma=args.scheduler_rate)
    prompt_text = get_prompt_text(LABEL_MAP)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    best_metrics, global_step, first_epoch = {}, 0, 1
    if saved:
        model.load_state_dict(saved["model_state_dict"], strict=True)
        model.pretraining_info = saved.get("pretraining_info")
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        best_metrics = saved["best_metrics"]
        global_step = saved["global_step"]
        first_epoch = saved["epoch"] + 1
        restore_rng(saved["rng_state"])
        logger.log(f"checkpoint_resumed={args.resume} next_epoch={first_epoch} global_step={global_step}")
    del saved

    def peak_vram():
        return torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device == "cuda" else 0.0

    def evaluate_and_save(epoch):
        nonlocal best_metrics
        rng = capture_rng()
        try:
            metrics = evaluate(model, test_loader, args.visual_length, prompt_text,
                               gt, gtsegments, gtlabels, device, logger.log)
        finally:
            restore_rng(rng)
            model.train()
        eligible = epoch >= args.memory_start_epoch
        logger.log(f"epoch={epoch} global_step={global_step} best_checkpoint_eligible={eligible}")
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(epoch=epoch, global_step=global_step,
                                         memory_active=bool(model.memory_active), **metrics)) + "\n")
        if eligible and metrics["AUC1"] > best_metrics.get("AUC1", -1):
            best_metrics = dict(metrics, epoch=epoch, global_step=global_step)
            payload = checkpoint_payload(model, config, args, epoch, global_step, best_metrics, "best")
            torch.save(payload, output / "best.pt")
            logger.log("saved_best=" + json.dumps(best_metrics))

    for epoch in range(first_epoch, args.max_epoch + 1):
        model.train()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.set_memory_active(epoch >= args.memory_start_epoch)
        logger.log(f"epoch={epoch} memory_active={bool(model.memory_active)} "
                   f"aux_branch={args.aux_branch} aux_mode={args.aux_mode} cnn_kernel={args.cnn_kernel} "
                   f"learning_rates={[group['lr'] for group in optimizer.param_groups]}")
        totals = {key: 0.0 for key in ("loss_c", "loss_a", "loss_text", "loss_memory", "loss_mse")}
        epoch_mse_pairs = 0
        last_eval_step = -1
        for batch_index, (normal_item, abnormal_item) in enumerate(zip(normal_loader, abnormal_loader), 1):
            features = torch.cat((normal_item[0], abnormal_item[0])).to(device)
            lengths = torch.cat((normal_item[2], abnormal_item[2])).to(device)
            labels = get_batch_label(list(normal_item[1]) + list(abnormal_item[1]), prompt_text, LABEL_MAP).to(device)
            text_features, logits_c, logits_a, auxiliary = model(features, None, prompt_text, lengths, return_aux=True)
            loss_c = CLAS2(logits_c, labels, lengths, device)
            loss_a = CLASM(logits_a, labels, lengths, device)
            normalized_text = text_features / text_features.norm(dim=-1, keepdim=True)
            loss_text = (normalized_text[1:] @ normalized_text[0]).abs().mean() * 0.1
            loss_memory = features.new_zeros(())
            if auxiliary is not None and args.memory_loss_weight:
                loss_memory = CLAS2(auxiliary["memory_logits"], labels, lengths, device)
            loss_mse, mse_pairs = features.new_zeros(()), 0
            if auxiliary is not None and args.aux_mode == "future_mse_joint" and args.mse_loss_weight:
                loss_mse, mse_pairs = future_feature_mse(
                    auxiliary["future_predictions"], features, lengths,
                    args.future_horizon, allow_empty=True,
                )
            loss = (loss_c + loss_a + loss_text + args.memory_loss_weight * loss_memory
                    + args.mse_loss_weight * loss_mse)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={batch_index}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            for key, value in zip(totals, (loss_c, loss_a, loss_text, loss_memory, loss_mse)):
                totals[key] += float(value.detach())
            epoch_mse_pairs += mse_pairs
            if batch_index % args.eval_steps == 0 or batch_index == num_batches:
                # Match the existing trainer: current-batch losses here, full
                # epoch means below. Formatting never changes the backward loss.
                memory_lr = optimizer.param_groups[1]["lr"] if memory_params else 0.0
                logger.log(
                    f"epoch={epoch}/{args.max_epoch} batch={batch_index}/{num_batches} "
                    f"loss={loss.item():.4f} loss1={loss_c.item():.4f} "
                    f"loss2={loss_a.item():.4f} loss3={loss_text.item():.4f} "
                    f"loss_memory={loss_memory.item():.6f} "
                    f"weighted_memory={args.memory_loss_weight * loss_memory.item():.6f} "
                    f"loss_mse={loss_mse.item():.6f} weighted_mse={args.mse_loss_weight * loss_mse.item():.6f} "
                    f"mse_pairs={mse_pairs} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} memory_lr={memory_lr:.2e} "
                    f"global_step={global_step} peak_vram={peak_vram():.2f}GB"
                )
            if batch_index % args.eval_steps == 0:
                evaluate_and_save(epoch)
                last_eval_step = global_step
        if global_step != last_eval_step:
            evaluate_and_save(epoch)
        average_total = (totals["loss_c"] + totals["loss_a"] + totals["loss_text"] +
                         args.memory_loss_weight * totals["loss_memory"] +
                         args.mse_loss_weight * totals["loss_mse"]) / num_batches
        logger.log(
            f"epoch_summary={epoch}/{args.max_epoch} batches={num_batches} "
            f"avg_loss1={totals['loss_c'] / num_batches:.4f} "
            f"avg_loss2={totals['loss_a'] / num_batches:.4f} "
            f"avg_loss3={totals['loss_text'] / num_batches:.4f} "
            f"avg_loss_memory={totals['loss_memory'] / num_batches:.6f} "
            f"avg_weighted_memory={args.memory_loss_weight * totals['loss_memory'] / num_batches:.6f} "
            f"avg_loss_mse={totals['loss_mse'] / num_batches:.6f} "
            f"avg_weighted_mse={args.mse_loss_weight * totals['loss_mse'] / num_batches:.6f} "
            f"mse_pairs={epoch_mse_pairs} "
            f"avg_loss={average_total:.4f} peak_vram={peak_vram():.2f}GB"
        )
        scheduler.step()
        payload = checkpoint_payload(model, config, args, epoch, global_step, best_metrics, "last")
        payload.update(optimizer_state_dict=optimizer.state_dict(),
                       scheduler_state_dict=scheduler.state_dict(), rng_state=capture_rng())
        torch.save(payload, output / "last.pt")
        logger.log(f"completed_epoch={epoch} saved_last={output / 'last.pt'}")
    logger.log(f"model_saved={output / 'best.pt'}")
    logger.log("training_complete best_test_metrics=" + json.dumps(best_metrics))


if __name__ == "__main__":
    main()
