"""Raw-score frame AUC/AP and the existing experiment detection mAP."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from . import bootstrap
except ImportError:
    import bootstrap

from detection_map import getDetectionMAP
from utils.tools import get_batch_mask


def frame_metrics(c_scores, a_scores, gt):
    c_scores = np.repeat(np.asarray(c_scores), 16)
    a_scores = np.repeat(np.asarray(a_scores), 16)
    if len(c_scores) != len(gt) or len(a_scores) != len(gt):
        raise ValueError("Feature-score count * 16 does not match frame ground truth")
    return dict(
        AUC1=float(roc_auc_score(gt, c_scores)),
        AP1=float(average_precision_score(gt, c_scores)),
        AUC2=float(roc_auc_score(gt, a_scores)),
        AP2=float(average_precision_score(gt, a_scores)),
    )


@torch.no_grad()
def evaluate(model, loader, prompt_text, gt, gtsegments, gtlabels, device, log=print):
    model.eval()
    c_scores, a_scores, predictions = [], [], []
    for visual, _, lengths in loader:
        length = int(lengths.item())
        visual = visual.squeeze(0)
        if visual.ndim == 2:
            visual = visual.unsqueeze(0)
        chunk_length = visual.shape[1]
        count = (length + chunk_length - 1) // chunk_length
        # process_split may add an entirely padded chunk for exact multiples.
        visual = visual[:count].to(device)
        valid_lengths = (length - torch.arange(count, device=device) * chunk_length).clamp(max=chunk_length)
        mask = get_batch_mask(valid_lengths.cpu(), chunk_length).to(device)
        _, logits_c, logits_a = model(visual, mask, prompt_text, valid_lengths)
        logits_c = logits_c.reshape(-1)[:length]
        probabilities = logits_a.reshape(-1, logits_a.shape[-1])[:length].softmax(dim=-1)
        c_scores.extend(logits_c.sigmoid().cpu().tolist())
        a_scores.extend((1 - probabilities[:, 0]).cpu().tolist())
        predictions.append(np.repeat(probabilities.cpu().numpy(), 16, axis=0))
    metrics = frame_metrics(c_scores, a_scores, gt)
    dmap, iou = getDetectionMAP(predictions, gtsegments, gtlabels, excludeNormal=False)
    metrics["mAP"] = float(np.mean(dmap))
    log("evaluation=raw_scores repeat=16 " + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
    log(" ".join(f"mAP@{threshold:.1f}={value:.2f}%" for threshold, value in zip(iou, dmap)))
    return metrics
