"""Stage 1: predict future CLIP features using the TRAIN CSV only (no CLIP/LGT)."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def pretraining_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", default="list/ucf_CLIP_rgb.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visual-length", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=1, help="Future offset on the process_feat feature grid")
    parser.add_argument("--cnn-kernel", type=int, choices=(1, 3), default=3)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", help="Resume this stage's last.pt at the next epoch")
    parser.add_argument("--preflight", action="store_true")
    return parser


def main():
    args = pretraining_parser().parse_args()
    for key in ("batch_size", "max_epoch", "visual_length", "horizon", "feature_dim"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.horizon >= args.visual_length or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Require horizon < visual_length and a finite positive learning rate")
    try:
        from . import bootstrap
        from .checkpoint import capture_rng, restore_rng
        from .future_features import FutureFeatureDataset, FutureFeaturePredictor, future_feature_mse
    except ImportError:
        import bootstrap
        from checkpoint import capture_rng, restore_rng
        from future_features import FutureFeatureDataset, FutureFeaturePredictor, future_feature_mse
    import random
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from training_log import TrainingLogger

    dataset = FutureFeatureDataset(args.train_list, args.visual_length, args.horizon)
    summary = dict(clips=len(dataset), skipped_short_clips=dataset.skipped,
                   labels=dataset.counts, grid="process_feat", horizon=args.horizon,
                   input_dim=512, test_data_used=False)
    print("Future-feature data check:", json.dumps(summary), flush=True)
    if args.preflight:
        print("Preflight passed. No CLIP, LGT, test data or pretraining was used.")
        return
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output directory is not empty; choose a new --output-dir")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable a GPU or use --device cpu")
    csv_hash = hashlib.sha256(Path(args.train_list).read_bytes()).hexdigest()
    saved = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if saved:
        if (saved.get("experiment") != "cnn_memory_future_pretrain" or saved.get("format_version") != 1
                or saved.get("kind") != "last"):
            raise ValueError("Resume requires the future-feature pretraining last.pt")
        keys = ("train_list", "visual_length", "horizon", "cnn_kernel", "feature_dim", "batch_size", "lr", "seed")
        if any(saved["train_args"][key] != getattr(args, key) for key in keys) or saved["train_csv_sha256"] != csv_hash:
            raise ValueError("Pretraining resume configuration/training CSV differs")
        if output.resolve() != Path(args.resume).resolve().parent or args.max_epoch <= saved["epoch"]:
            raise ValueError("Resume in the same directory and increase --max-epoch")
    output.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(output / "pretrain.log", append=bool(saved))
    logger.log(f"experiment=cnn_memory_future_pretrain device={device} seed={args.seed} "
               f"epochs={args.max_epoch} target=future_clip_feature_512 test_data_used=False")
    logger.log("args=" + json.dumps(vars(args), sort_keys=True))
    (output / "config.json").write_text(json.dumps(dict(vars(args), data=summary), indent=2) + "\n", encoding="utf-8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = FutureFeaturePredictor(512, args.feature_dim, args.cnn_kernel).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    first_epoch, global_step = 1, 0
    if saved:
        model.encoder.load_state_dict(saved["encoder_state_dict"], strict=True)
        model.predictor.load_state_dict(saved["predictor_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        first_epoch, global_step = saved["epoch"] + 1, saved["global_step"]
        restore_rng(saved["rng_state"])
        logger.log(f"checkpoint_resumed={args.resume} next_epoch={first_epoch} global_step={global_step}")
    del saved
    for epoch in range(first_epoch, args.max_epoch + 1):
        model.train()
        squared_error, pair_count = 0.0, 0
        for batch, (features, lengths) in enumerate(loader, 1):
            features, lengths = features.to(device), lengths.to(device)
            predicted = model(features, lengths)
            loss, pairs = future_feature_mse(predicted, features, lengths, args.horizon)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            squared_error += float(loss.detach()) * pairs
            pair_count += pairs
            if batch % 10 == 0 or batch == len(loader):
                logger.log(f"epoch={epoch}/{args.max_epoch} batch={batch}/{len(loader)} "
                           f"loss_mse={loss.item():.6f} valid_pairs={pairs} lr={args.lr:.2e} global_step={global_step}")
        mean_mse = squared_error / pair_count
        logger.log(f"epoch_summary={epoch}/{args.max_epoch} avg_loss_mse={mean_mse:.6f} valid_pairs={pair_count}")
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(epoch=epoch, global_step=global_step, MSE=mean_mse, valid_pairs=pair_count)) + "\n")
        payload = dict(format_version=1, experiment="cnn_memory_future_pretrain", kind="encoder",
                       encoder_config=model.encoder.config, encoder_state_dict=model.encoder.state_dict(),
                       epoch=epoch, global_step=global_step, train_args=vars(args).copy(),
                       train_csv_sha256=csv_hash, metrics=dict(MSE=mean_mse, valid_pairs=pair_count))
        torch.save(payload, output / "encoder.pt")
        payload.update(kind="last", predictor_state_dict=model.predictor.state_dict(),
                       optimizer_state_dict=optimizer.state_dict(), rng_state=capture_rng())
        torch.save(payload, output / "last.pt")
    logger.log(f"model_saved={output / 'encoder.pt'}")


if __name__ == "__main__":
    main()
