"""Shared components for frozen-CLIP hand anomaly classification."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_TO_IDX = {"good": 0, "bad": 1}
IDX_TO_CLASS = {value: key for key, value in CLASS_TO_IDX.items()}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    group: str

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "label": self.label,
            "class_name": IDX_TO_CLASS[self.label],
            "group": self.group,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return resolved


def discover_samples(data_root: str | Path, require_pairs: bool = True) -> list[Sample]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    samples: list[Sample] = []
    group_labels: dict[str, set[int]] = {}
    for class_name, label in CLASS_TO_IDX.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        files = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise RuntimeError(f"No images found in {class_dir}")
        for path in files:
            sample = Sample(path.resolve(), label, path.stem)
            samples.append(sample)
            group_labels.setdefault(sample.group, set()).add(label)
    keys = [(sample.group, sample.label) for sample in samples]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate class/group pairs were found")
    if require_pairs:
        expected = set(CLASS_TO_IDX.values())
        unpaired = sorted(group for group, labels in group_labels.items() if labels != expected)
        if unpaired:
            raise RuntimeError(f"Every UUID must have good and bad images; unpaired: {unpaired}")
    return sorted(samples, key=lambda sample: (sample.group, sample.label))


def split_samples_by_group(
    samples: Sequence[Sample], val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[Sample]]:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1")
    groups = sorted({sample.group for sample in samples})
    if len(groups) < 6:
        raise ValueError("At least 6 paired groups are required")
    random.Random(seed).shuffle(groups)
    n_test = max(1, round(len(groups) * test_ratio))
    n_val = max(1, round(len(groups) * val_ratio))
    if len(groups) - n_val - n_test < 2:
        raise ValueError("Split leaves fewer than 2 training groups")
    selected = {
        "test": set(groups[:n_test]),
        "val": set(groups[n_test:n_test + n_val]),
        "train": set(groups[n_test + n_val:]),
    }
    output = {
        name: sorted(
            [sample for sample in samples if sample.group in group_set],
            key=lambda sample: (sample.group, sample.label),
        )
        for name, group_set in selected.items()
    }
    if selected["train"] & selected["val"] or selected["train"] & selected["test"] or selected["val"] & selected["test"]:
        raise AssertionError("Group leakage detected")
    for name, split in output.items():
        labels = [sample.label for sample in split]
        if labels.count(0) != labels.count(1):
            raise AssertionError(f"Split {name} is not balanced")
    return output


def save_split_manifest(path: str | Path, splits: dict[str, Sequence[Sample]], seed: int) -> None:
    payload = {
        "seed": seed,
        "class_to_idx": CLASS_TO_IDX,
        "splits": {
            name: {
                "groups": sorted({sample.group for sample in samples}),
                "samples": [sample.to_dict() for sample in samples],
            }
            for name, samples in splits.items()
        },
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class HandImageDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform, repeats: int = 1):
        self.samples = list(samples)
        self.transform = transform
        self.repeats = repeats
        if not self.samples or repeats < 1:
            raise ValueError("Dataset must be non-empty and repeats >= 1")

    def __len__(self) -> int:
        return len(self.samples) * self.repeats

    def __getitem__(self, index: int):
        sample = self.samples[index % len(self.samples)]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, sample.label, str(sample.path), sample.group


def make_train_transform(clip_preprocess, augment: bool):
    if not augment:
        return clip_preprocess
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01),
        clip_preprocess,
    ])


def dataloader_worker_init(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
