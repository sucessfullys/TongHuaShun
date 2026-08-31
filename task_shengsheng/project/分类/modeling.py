"""CLIP feature extraction, MLP head, and binary metrics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        dropout: float,
        num_classes: int = 2,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@torch.inference_mode()
def extract_clip_features(clip_model, loader, device: torch.device):
    clip_model.eval()
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    paths: list[str] = []
    groups: list[str] = []
    for images, labels, batch_paths, batch_groups in loader:
        images = images.to(device, non_blocking=True)
        encoded = clip_model.encode_image(images)
        encoded = F.normalize(encoded.float(), dim=-1)
        feature_batches.append(encoded.cpu())
        label_batches.append(labels.long().cpu())
        paths.extend(batch_paths)
        groups.extend(batch_groups)
    return torch.cat(feature_batches), torch.cat(label_batches), paths, groups


def binary_metrics(labels: torch.Tensor, predictions: torch.Tensor) -> dict:
    labels = labels.long().cpu()
    predictions = predictions.long().cpu()
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision_bad": precision,
        "recall_bad": recall,
        "specificity_good": specificity,
        "f1_bad": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


@torch.inference_mode()
def evaluate_head(head, features, labels, device):
    head.eval()
    logits = head(features.to(device)).cpu()
    probabilities = logits.softmax(dim=-1)
    metrics = binary_metrics(labels, probabilities.argmax(dim=-1))
    metrics["loss"] = float(F.cross_entropy(logits, labels).item())
    return metrics, probabilities
