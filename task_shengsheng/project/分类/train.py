#!/usr/bin/env python3
"""Train a frozen OpenAI CLIP ViT-L/14 plus MLP hand anomaly classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from hand_clip import (
    CLASS_TO_IDX,
    HandImageDataset,
    dataloader_worker_init,
    discover_samples,
    make_train_transform,
    resolve_device,
    save_split_manifest,
    seed_everything,
    split_samples_by_group,
)
from modeling import MLPHead, evaluate_head, extract_clip_features


DEFAULT_DATA = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/手部异常数据集/hand"
DEFAULT_CLIP = "/mnt/image-edit/datasets/duanyufa/RAR/checkpoints/RAR_modelzoo/CLIP/ViT-L-14.pt"
DEFAULT_OUTPUT = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/outputs/clip_vitl14_mlp"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--clip-checkpoint", default=DEFAULT_CLIP)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--train-views", type=int, default=4, help="Augmented CLIP views per train image")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4, help="CLIP feature extraction batch size")
    parser.add_argument("--mlp-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--reuse-feature-cache", action="store_true")
    return parser.parse_args()


def make_image_loader(samples, transform, repeats, args, shuffle=False):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        HandImageDataset(samples, transform, repeats),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=dataloader_worker_init,
        generator=generator,
    )


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_feature_cache(args, splits, device, cache_path):
    checkpoint = Path(args.clip_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint}")
    print(f"Loading frozen CLIP from {checkpoint}")
    clip_model, preprocess = clip.load(str(checkpoint), device=device, jit=False)
    clip_model.eval().requires_grad_(False)

    train_transform = make_train_transform(preprocess, not args.no_augment)
    loaders = {
        "train": make_image_loader(splits["train"], train_transform, args.train_views, args),
        "val": make_image_loader(splits["val"], preprocess, 1, args),
        "test": make_image_loader(splits["test"], preprocess, 1, args),
    }
    cache = {
        "clip_checkpoint": str(checkpoint),
        "seed": args.seed,
        "train_views": args.train_views,
        "augment": not args.no_augment,
        "splits": {},
    }
    for name, loader in loaders.items():
        print(f"Extracting {name} CLIP features ({len(loader.dataset)} views) ...")
        features, labels, paths, groups = extract_clip_features(clip_model, loader, device)
        cache["splits"][name] = {
            "features": features,
            "labels": labels,
            "paths": paths,
            "groups": groups,
        }
    cache["feature_dim"] = int(cache["splits"]["train"]["features"].shape[1])
    torch.save(cache, cache_path)
    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache


def validate_cache(cache, args):
    expected = str(Path(args.clip_checkpoint).expanduser().resolve())
    if cache.get("clip_checkpoint") != expected or cache.get("seed") != args.seed:
        raise ValueError("Feature cache does not match --clip-checkpoint/--seed; remove it or omit --reuse-feature-cache")


def save_epoch_predictions(path, epoch, split_predictions):
    payload = {"epoch": epoch, "splits": {}}
    for split_name, split, probabilities in split_predictions:
        aggregated = {}
        for image_path, group, label, probability in zip(
            split["paths"], split["groups"], split["labels"], probabilities
        ):
            row = aggregated.setdefault(
                image_path,
                {
                    "path": image_path,
                    "group": group,
                    "label": int(label),
                    "prob_good_sum": 0.0,
                    "prob_bad_sum": 0.0,
                    "num_views": 0,
                },
            )
            row["prob_good_sum"] += float(probability[0])
            row["prob_bad_sum"] += float(probability[1])
            row["num_views"] += 1

        rows = []
        for row in aggregated.values():
            prob_good = row.pop("prob_good_sum") / row["num_views"]
            prob_bad = row.pop("prob_bad_sum") / row["num_views"]
            prediction = int(prob_bad >= prob_good)
            row.update(
                {
                    "prob_good": prob_good,
                    "prob_bad": prob_bad,
                    "prediction": prediction,
                    "correct": prediction == row["label"],
                }
            )
            rows.append(row)
        payload["splits"][split_name] = rows

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_head(args, cache, output_dir, device):
    train = cache["splits"]["train"]
    val = cache["splits"]["val"]
    test = cache["splits"]["test"]
    head = MLPHead(
        cache["feature_dim"], args.hidden_dim, args.bottleneck_dim, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        TensorDataset(train["features"], train["labels"]),
        batch_size=args.mlp_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    checkpoint_path = output_dir / "best_model.pt"
    best_score = -1.0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        running_loss = 0.0
        seen = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.numel()
            seen += labels.numel()

        train_metrics, train_probs = evaluate_head(head, train["features"], train["labels"], device)
        val_metrics, val_probs = evaluate_head(head, val["features"], val["labels"], device)
        save_epoch_predictions(
            output_dir / "epoch_predictions" / f"epoch_{epoch:03d}.json",
            epoch,
            [("train", train, train_probs), ("val", val, val_probs)],
        )
        train_metrics["optimization_loss"] = running_loss / max(1, seen)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        score = float(val_metrics["balanced_accuracy"])
        print(
            f"epoch={epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_metrics['optimization_loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"train_acc={train_metrics['accuracy']:.3f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_bal_acc={score:.3f} val_f1_bad={val_metrics['f1_bad']:.3f}"
        )
        if score > best_score + 1e-8:
            best_score = score
            stale_epochs = 0
            torch.save(
                {
                    "head_state_dict": head.state_dict(),
                    "feature_dim": cache["feature_dim"],
                    "hidden_dim": args.hidden_dim,
                    "bottleneck_dim": args.bottleneck_dim,
                    "dropout": args.dropout,
                    "class_to_idx": CLASS_TO_IDX,
                    "clip_checkpoint": cache["clip_checkpoint"],
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    best = safe_torch_load(checkpoint_path, device)
    head.load_state_dict(best["head_state_dict"])
    val_metrics, val_probs = evaluate_head(head, val["features"], val["labels"], device)
    test_metrics, test_probs = evaluate_head(head, test["features"], test["labels"], device)
    metrics = {
        "best_epoch": best["epoch"],
        "validation": val_metrics,
        "test": test_metrics,
        "warning": "Only 26 paired UUID groups are available; metrics have high uncertainty.",
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, split, probs in [("val", val, val_probs), ("test", test, test_probs)]:
        rows = []
        for path, group, label, prob in zip(split["paths"], split["groups"], split["labels"], probs):
            rows.append({
                "path": path,
                "group": group,
                "label": int(label),
                "prob_good": float(prob[0]),
                "prob_bad": float(prob[1]),
                "prediction": int(prob.argmax()),
            })
        (output_dir / f"{name}_predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Best checkpoint:", checkpoint_path)
    print("Validation metrics:", json.dumps(val_metrics, ensure_ascii=False))
    print("Test metrics:", json.dumps(test_metrics, ensure_ascii=False))


def main():
    args = parse_args()
    if args.train_views < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("train_views, epochs, and patience must be >= 1")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(args.data_root, require_pairs=True)
    splits = split_samples_by_group(samples, args.val_ratio, args.test_ratio, args.seed)
    save_split_manifest(output_dir / "split.json", splits, args.seed)
    print("Device:", device)
    print("Split groups:", {name: len({s.group for s in split}) for name, split in splits.items()})
    print("Split images:", {name: len(split) for name, split in splits.items()})

    cache_path = output_dir / "feature_cache.pt"
    if args.reuse_feature_cache and cache_path.is_file():
        cache = safe_torch_load(cache_path)
        validate_cache(cache, args)
        print("Reusing feature cache:", cache_path)
    else:
        cache = build_feature_cache(args, splits, device, cache_path)
    train_head(args, cache, output_dir, device)


if __name__ == "__main__":
    main()
