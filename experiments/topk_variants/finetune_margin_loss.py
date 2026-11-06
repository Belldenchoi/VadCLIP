"""Original Finetune-VadCLIP losses, copied without changing their formulas.

Source: VanLinh2801/Finetune-VadCLIP, local revision 330f3661edde133f09318ae1ed79e8af74894ca6,
VadCLIP/src/ucf_train.py (get_anomaly_score_branches and the two margin losses).

Temporal-A training integration and padding/resume guards are in
finetune_margin_adapter.py. Normal Top-K suppression and the source trainer's
epoch-only evaluation policy are intentionally outside this port's scope.
"""
import torch
import torch.nn.functional as F


def get_anomaly_score_branches(logits1, logits2, score_branch):
    branches = []
    if score_branch in ('classifier', 'both'):
        branches.append(torch.sigmoid(logits1[..., 0].float()))
    if score_branch in ('alignment', 'both'):
        branches.append(1.0 - F.softmax(logits2.float(), dim=-1)[..., 0])
    if not branches:
        raise ValueError(f'Unknown score branch: {score_branch}')
    return branches


def abnormal_background_suppression_loss(
    logits1, logits2, labels, lengths, top_k_ratio, margin, score_branch, device
):
    losses = []
    margin = float(margin)
    branches = get_anomaly_score_branches(logits1, logits2, score_branch)
    for scores in branches:
        branch_losses = []
        for i, length in enumerate(lengths):
            class_index = int(labels[i].argmax().item())
            if class_index == 0:
                continue
            valid_length = max(1, min(int(length), scores.shape[1]))
            top_k = max(1, int(valid_length * top_k_ratio))
            top_k = min(top_k, valid_length)
            if 2 * top_k > valid_length:
                continue
            valid_scores = scores[i, :valid_length]
            high_indices = torch.topk(valid_scores.detach(), k=top_k, largest=True).indices
            low_indices = torch.topk(valid_scores.detach(), k=top_k, largest=False).indices
            high_vals = valid_scores[high_indices]
            low_vals = valid_scores[low_indices]
            branch_losses.append(F.relu(low_vals.max() - high_vals.min() + margin))
        if branch_losses:
            losses.append(torch.stack(branch_losses).mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def ranking_margin_loss(logits1, logits2, labels, lengths, top_k_ratio, margin, score_branch, device):
    losses = []
    for scores in get_anomaly_score_branches(logits1, logits2, score_branch):
        normal_scores = []
        abnormal_scores = []
        for i, length in enumerate(lengths):
            valid_length = min(int(length), scores.shape[1])
            if valid_length <= 0:
                continue
            top_k = max(1, int(valid_length * top_k_ratio))
            topk_mean = scores[i, :valid_length].topk(top_k).values.mean()
            if int(labels[i].argmax().item()) == 0:
                normal_scores.append(topk_mean)
            else:
                abnormal_scores.append(topk_mean)
        if normal_scores and abnormal_scores:
            normal_mean = torch.stack(normal_scores).mean()
            abnormal_mean = torch.stack(abnormal_scores).mean()
            losses.append(F.relu(normal_mean + margin - abnormal_mean))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()
