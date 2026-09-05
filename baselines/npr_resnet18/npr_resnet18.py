"""Shared NPR + ResNet-18 training utilities.

NPR means Neighboring Pixel Relationships. The model receives an RGB image,
builds a local residual map, then trains a randomly initialized ResNet-18
from scratch for real/fake classification.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class NPRLayer(nn.Module):
    """Compute neighboring pixel relationship residuals."""

    def __init__(self, factor: float = 0.5, scale: float = 2.0 / 3.0):
        super().__init__()
        self.factor = factor
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if height % 2 == 1:
            x = x[:, :, :-1, :]
        if width % 2 == 1:
            x = x[:, :, :, :-1]

        down = F.interpolate(
            x,
            scale_factor=self.factor,
            mode="nearest",
            recompute_scale_factor=True,
        )
        up = F.interpolate(
            down,
            size=x.shape[-2:],
            mode="nearest",
        )
        return (x - up) * self.scale


class NPRResNet18(nn.Module):
    """NPR front-end followed by randomly initialized ResNet-18.

    The default ImageNet stem uses a 7x7 stride-2 convolution and max pooling.
    For NPR residuals, we keep early spatial detail with a 3x3 stride-1 stem.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        from torchvision.models import resnet18

        self.npr = NPRLayer()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.npr(x))


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_output_root(project_root: Path, name: str) -> tuple[Path, str]:
    output_root = project_root / "outputs" / name
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root, time.strftime("%Y%m%d_%H%M%S")


def make_model(device: str) -> nn.Module:
    return NPRResNet18(num_classes=2).to(device)


def stratified_train_val_split(df: pd.DataFrame, val_fraction: float, seed: int):
    if len(df) < 4 or val_fraction <= 0:
        return df.reset_index(drop=True), df.reset_index(drop=True)

    stratify = df["label"].astype(str) + "_" + df["generator"].astype(str)
    if stratify.value_counts().min() < 2:
        stratify = df["label"]
    if pd.Series(stratify).value_counts().min() < 2:
        stratify = None

    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_kaggle_loaders(
    splits: dict[str, Any],
    dataset_class,
    collate_fn,
    transform_train,
    transform_eval,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    val_fraction: float,
    seed: int,
):
    train_inner, val_inner = stratified_train_val_split(splits["train_df"], val_fraction, seed)
    splits["train_inner_df"] = train_inner
    splits["val_inner_df"] = val_inner

    train_dataset = dataset_class(train_inner, eval_case=splits["eval_case"], transform=transform_train)
    val_dataset = dataset_class(val_inner, eval_case=splits["eval_case"], transform=transform_eval)
    test_dataset = dataset_class(splits["eval_df"], eval_case=splits["eval_case"], transform=transform_eval)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader, splits


def maybe_limit_dataset(dataset, max_samples: int | None, seed: int):
    if max_samples is None:
        return dataset
    if hasattr(dataset, "select") and hasattr(dataset, "shuffle"):
        n = min(len(dataset), max_samples)
        return dataset.shuffle(seed=seed).select(range(n))
    if hasattr(dataset, "take"):
        return dataset.take(max_samples)
    return dataset


def build_hf_loaders(
    splits: dict[str, Any],
    dataset_class,
    streaming: bool,
    collate_fn,
    transform_train,
    transform_eval,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    max_train_samples: int | None,
    max_test_samples: int | None,
    val_fraction: float,
    seed: int,
):
    train_source = maybe_limit_dataset(splits["train"], max_train_samples, seed)
    test_source = maybe_limit_dataset(splits["eval"], max_test_samples, seed)
    val_source = test_source

    if not streaming and hasattr(train_source, "train_test_split"):
        train_val = train_source.train_test_split(test_size=val_fraction, seed=seed)
        train_source = train_val["train"]
        val_source = train_val["test"]
    else:
        print("WARNING: HF streaming mode uses eval split for early stopping and final test.")

    train_dataset = dataset_class(
        train_source,
        split_name=splits["train_split_name"],
        eval_case=splits["eval_case"],
        transform=transform_train,
        task_type="classification",
    )
    val_dataset = dataset_class(
        val_source,
        split_name=splits["train_split_name"] + "_inner_val" if val_source is not test_source else splits["eval_split_name"],
        eval_case=splits["eval_case"],
        transform=transform_eval,
        task_type="classification",
    )
    test_dataset = dataset_class(
        test_source,
        split_name=splits["eval_split_name"],
        eval_case=splits["eval_case"],
        transform=transform_eval,
        task_type="classification",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False if streaming else True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader, splits


def train_one_epoch(model, loader, optimizer, criterion, scaler, device: str, use_amp: bool) -> float:
    model.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict(model, loader, device: str, use_amp: bool):
    model.eval()
    labels, probs, rows = [], [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
        prob_fake = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        probs.extend(prob_fake.tolist())
        labels.extend(batch["label"].numpy().tolist())
        rows.extend(batch["metadata"])
    return np.array(labels, dtype=int), np.array(probs, dtype=float), pd.DataFrame(rows)


def compute_metrics(y_true, y_prob) -> dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def evaluate_by_generator(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for generator, part in pred_df.groupby("generator"):
        metrics = compute_metrics(part["label"].to_numpy(), part["fake_probability"].to_numpy())
        metrics["generator"] = generator
        metrics["num_samples"] = int(len(part))
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("generator")


def train_eval_experiment(
    exp: dict[str, Any],
    run_dir: Path,
    loaders: tuple[Any, Any, Any],
    splits: dict[str, Any],
    config_summary: dict[str, Any],
    device: str,
    use_amp: bool,
    max_epochs: int,
    patience: int,
    min_delta: float,
    learning_rate: float,
    weight_decay: float,
    save_predictions: bool,
    model_factory=None,
) -> dict[str, Any]:
    train_loader, val_loader, test_loader = loaders
    for subdir in ["checkpoints", "predictions", "metrics"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    model = model_factory(device) if model_factory is not None else make_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    best_path = run_dir / "checkpoints" / "best_npr_resnet18.pt"

    for epoch in range(1, max_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        y_val, p_val, _ = predict(model, val_loader, device, use_amp)
        val_metrics = compute_metrics(y_val, p_val)
        current = val_metrics["balanced_accuracy"]
        history.append({"epoch": epoch, "train_loss": loss, **val_metrics})
        print(f"epoch {epoch}/{max_epochs} loss={loss:.4f} val_bal_acc={current:.4f}")

        if current > best_metric + min_delta:
            best_metric = current
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    y_test, p_test, meta_df = predict(model, test_loader, device, use_amp)
    metrics = compute_metrics(y_test, p_test)
    metrics.update(
        {
            "experiment_name": exp["name"],
            "eval_case": exp["eval_case"],
            "model_name": "npr_resnet18_from_scratch",
            "pretrained": False,
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_metric,
            **config_summary,
        }
    )

    pred_df = meta_df.copy()
    pred_df["label"] = y_test
    pred_df["predicted_label"] = (p_test >= 0.5).astype(int)
    pred_df["fake_probability"] = p_test
    pred_df["experiment_name"] = exp["name"]
    pred_df["model_name"] = "npr_resnet18_from_scratch"

    pd.DataFrame(history).to_csv(run_dir / "metrics" / "history.csv", index=False)
    evaluate_by_generator(pred_df).to_csv(run_dir / "metrics" / "generator_metrics.csv", index=False)
    if save_predictions:
        pred_df.to_csv(run_dir / "predictions" / "predictions.csv", index=False)
    with open(run_dir / "metrics" / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    del train_loader, val_loader, test_loader, model, optimizer, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics
