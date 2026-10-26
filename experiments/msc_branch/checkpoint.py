"""Keep new checkpoints separate from the older CNN-memory experiment."""
import torch

from experiments.cnn_memory.checkpoint import capture_rng, restore_rng


def checkpoint_payload(model, config, args, epoch, global_step, best_metrics, kind):
    return dict(format_version=1, experiment='msc_branch', kind=kind,
                model_config=config, train_args=vars(args).copy(), model_state_dict=model.state_dict(),
                epoch=epoch, global_step=global_step, best_metrics=best_metrics)


def load_checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format_version') != 1 or saved.get('experiment') != 'msc_branch':
        raise ValueError('Expected an msc_branch checkpoint; start a separate run for the new architecture')
    phase = saved['train_args']
    if (phase['contrastive_mode'] == 'mean_shifted'
            and saved['epoch'] >= phase['contrastive_start_epoch']
            and not bool(saved['model_state_dict'].get('representation.center_ready', False))):
        raise ValueError('Active mean-shifted checkpoint is missing a calibrated center')
    return saved
