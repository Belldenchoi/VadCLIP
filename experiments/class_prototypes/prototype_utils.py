"""Build auditable class prototypes from frozen CLIP text embeddings."""

import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F


ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def load_description_file(path):
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    classes = payload.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("description file must contain a non-empty classes map")
    validate_descriptions(classes)
    return payload, hashlib.sha256(raw).hexdigest()


def validate_descriptions(classes):
    for class_name, descriptions in classes.items():
        if not descriptions:
            raise ValueError(f"class {class_name!r} has no descriptions")
        normalized = [text.strip().lower() for text in descriptions]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"class {class_name!r} contains duplicate descriptions")
        for text in descriptions:
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"class {class_name!r} contains invalid text")
            if ARTICLES.search(text):
                raise ValueError(
                    f"article found in description for {class_name!r}: {text!r}"
                )


def encode_texts(clip_model, clip_module, texts, device, batch_size=64):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = clip_module.tokenize(batch).to(device)
            token_embeddings = clip_model.encode_token(tokens)
            embeddings = clip_model.encode_text(token_embeddings, tokens)
            outputs.append(F.normalize(embeddings.float(), dim=-1).cpu())
    return torch.cat(outputs, dim=0)


def score_embeddings(class_embeddings, class_anchors,
                     alpha=0.4, beta=0.3, gamma=0.3):
    centroids = torch.stack([
        F.normalize(embeddings.mean(dim=0), dim=0)
        for embeddings in class_embeddings
    ])
    scored = []
    for class_index, embeddings in enumerate(class_embeddings):
        name_scores = embeddings @ class_anchors[class_index]
        if embeddings.shape[0] > 1:
            leave_one_out = (
                embeddings.sum(dim=0, keepdim=True) - embeddings
            ) / (embeddings.shape[0] - 1)
            leave_one_out = F.normalize(leave_one_out, dim=-1)
            intra_scores = (embeddings * leave_one_out).sum(dim=-1)
        else:
            intra_scores = torch.zeros_like(name_scores)
        other_centroids = torch.cat([
            centroids[:class_index], centroids[class_index + 1:]
        ])
        inter_scores = embeddings @ other_centroids.T
        hard_inter_scores, hard_inter_indices = inter_scores.max(dim=1)
        hard_inter_indices = hard_inter_indices + (
            hard_inter_indices >= class_index
        ).long()
        total_scores = (
            alpha * name_scores + beta * intra_scores
            - gamma * hard_inter_scores
        )
        scored.append({
            "name": name_scores,
            "intra": intra_scores,
            "inter": hard_inter_scores,
            "hard_inter_class": hard_inter_indices,
            "total": total_scores,
        })
    return scored


def select_diverse_embeddings(embeddings, scores, top_k,
                              diversity_threshold):
    ranked = torch.argsort(scores, descending=True).tolist()
    selected = []
    rejected = []
    for index in ranked:
        duplicate_index = None
        duplicate_similarity = None
        if selected:
            similarities = embeddings[index] @ embeddings[selected].T
            maximum, position = similarities.max(dim=0)
            if maximum.item() >= diversity_threshold:
                duplicate_index = selected[position.item()]
                duplicate_similarity = maximum.item()
        if duplicate_index is None:
            selected.append(index)
            if len(selected) == top_k:
                break
        else:
            rejected.append({
                "index": index,
                "duplicate_of": duplicate_index,
                "cosine": duplicate_similarity,
            })
    return selected, rejected


def aggregate_prototype(embeddings, scores, mode="weighted_mean",
                        temperature=0.1):
    if mode == "mean":
        prototype = embeddings.mean(dim=0)
        weights = torch.full(
            (embeddings.shape[0],), 1.0 / embeddings.shape[0]
        )
    elif mode == "weighted_mean":
        if temperature <= 0:
            raise ValueError("weight temperature must be positive")
        weights = torch.softmax(scores / temperature, dim=0)
        prototype = (weights[:, None] * embeddings).sum(dim=0)
    else:
        raise ValueError(f"unsupported aggregation mode: {mode}")
    return F.normalize(prototype, dim=0), weights
