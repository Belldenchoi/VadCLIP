"""Use the Top-K evaluator, which retains the original UCF evaluation flow."""

from types import SimpleNamespace

try:
    from . import bootstrap
except ImportError:
    import bootstrap

from experiments.topk_variants.test_ucf import test as test_ucf


def evaluate(model, loader, maxlen, prompt_text, gt, gtsegments, gtlabels, device, log=print):
    log(f"cnn_memory_eval_active={bool(getattr(model, 'memory_active', False))} "
        "evaluation=raw_scores repeat=16 evaluation_backend=topk_variants.test_ucf")
    return test_ucf(
        model, loader, maxlen, prompt_text, gt, gtsegments, gtlabels, device,
        logger=SimpleNamespace(log=log), return_metrics=True,
    )
