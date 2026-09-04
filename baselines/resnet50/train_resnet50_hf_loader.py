"""ResNet-50 last-layer baseline using the Hugging Face Tiny-GenImage loader.

Run on Kaggle/Colab:
    %run /path/to/HoangHa_Code/baselines/resnet50/train_resnet50_hf_loader.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def find_code_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents, Path.cwd(), Path.cwd().parent]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob("*"))
        candidates.extend(kaggle_input.glob("*/*"))
        for init_file in kaggle_input.rglob("__init__.py"):
            if init_file.parent.name == "data_loader":
                return init_file.parent.parent

    for candidate in candidates:
        if (candidate / "data_loader" / "__init__.py").exists():
            return candidate
    raise FileNotFoundError("Cannot find HoangHa_Code/data_loader.")


CODE_ROOT = find_code_root()
PROJECT_ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else CODE_ROOT
sys.path.insert(0, str(CODE_ROOT))

from data_loader import (  # noqa: E402
    TinyGenImageDataset,
    TinyGenImageIterableDataset,
    TinyGenImageSplitConfig,
    build_image_transform,
    build_tiny_genimage_splits,
    collate_unified_batch,
)


# ---------------------------
# Config
# ---------------------------

RUN_ALL_CASES = False
SELECTED_EXPERIMENT = "combined"
EXPERIMENT_CONFIGS = [
    {"name": "combined", "eval_case": "combined"},
    {"name": "in_domain_biggan", "eval_case": "in_domain", "generator": "BigGAN"},
    {"name": "cross_generator_glide", "eval_case": "cross_generator", "heldout_generator": "GLIDE"},
    {"name": "train_one_generator_biggan", "eval_case": "train_one_generator", "base_generator": "BigGAN"},
]

STREAMING = True
CACHE_DIR = None
BALANCE_REAL = True
RANDOM_SEED = 42
STREAMING_SHUFFLE_BUFFER_SIZE = 512

MAX_TRAIN_SAMPLES = 500
MAX_TEST_SAMPLES = 300
VAL_FRACTION = 0.2

BATCH_SIZE = 16
NUM_WORKERS = 0
MAX_EPOCHS = 20
PATIENCE = 3
MIN_DELTA = 1e-3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SAVE_PREDICTIONS = True
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "resnet50_last_layer_hf"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ID = time.strftime("%Y%m%d_%H%M%S")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = torch.cuda.is_available()
USE_AMP = torch.cuda.is_available()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_limit_dataset(dataset, max_samples: int | None, seed: int):
    if max_samples is None:
        return dataset
    if hasattr(dataset, "select") and hasattr(dataset, "shuffle"):
        n = min(len(dataset), max_samples)
        return dataset.shuffle(seed=seed).select(range(n))
    if hasattr(dataset, "take"):
        return dataset.take(max_samples)
    return dataset


def split_train_val(dataset, val_fraction: float, seed: int):
    if STREAMING or not hasattr(dataset, "train_test_split"):
        print("WARNING: STREAMING=True, using a slice of eval split for early stopping.")
        return dataset, None
    split = dataset.train_test_split(test_size=val_fraction, seed=seed)
    return split["train"], split["test"]


def make_model() -> nn.Module:
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model.to(DEVICE)


def build_loaders(splits: dict):
    clean_train_transform = build_image_transform(image_size=224, train=True)
    clean_eval_transform = build_image_transform(image_size=224, train=False)

    train_source = maybe_limit_dataset(splits["train"], MAX_TRAIN_SAMPLES, RANDOM_SEED)
    test_source = maybe_limit_dataset(splits["eval"], MAX_TEST_SAMPLES, RANDOM_SEED)
    train_inner, val_inner = split_train_val(train_source, VAL_FRACTION, RANDOM_SEED)
    val_source = val_inner if val_inner is not None else test_source

    DatasetClass = TinyGenImageIterableDataset if STREAMING else TinyGenImageDataset
    train_dataset = DatasetClass(
        train_inner,
        split_name=splits["train_split_name"],
        eval_case=splits["eval_case"],
        transform=clean_train_transform,
        task_type="classification",
    )
    val_dataset = DatasetClass(
        val_source,
        split_name=splits["train_split_name"] + "_inner_val" if val_inner is not None else splits["eval_split_name"],
        eval_case=splits["eval_case"],
        transform=clean_eval_transform,
        task_type="classification",
    )
    test_dataset = DatasetClass(
        test_source,
        split_name=splits["eval_split_name"],
        eval_case=splits["eval_case"],
        transform=clean_eval_transform,
        task_type="classification",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False if STREAMING else True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_unified_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_unified_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_unified_batch,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, scaler) -> float:
    model.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict(model, loader):
    model.eval()
    labels, probs, rows = [], [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(images)
        prob_fake = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        probs.extend(prob_fake.tolist())
        labels.extend(batch["label"].numpy().tolist())
        rows.extend(batch["metadata"])
    return np.array(labels, dtype=int), np.array(probs, dtype=float), pd.DataFrame(rows)


def compute_metrics(y_true, y_prob) -> dict:
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


def run_experiment(exp: dict) -> dict:
    seed_everything(RANDOM_SEED)
    run_dir = OUTPUT_ROOT / exp["name"] / RUN_ID
    for subdir in ["checkpoints", "predictions", "metrics"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    split_config = TinyGenImageSplitConfig(
        eval_case=exp["eval_case"],
        generator=exp.get("generator"),
        heldout_generator=exp.get("heldout_generator"),
        base_generator=exp.get("base_generator"),
        cache_dir=CACHE_DIR,
        streaming=STREAMING,
        balance_real=BALANCE_REAL,
        seed=RANDOM_SEED,
        streaming_shuffle_buffer_size=STREAMING_SHUFFLE_BUFFER_SIZE,
    )
    splits = build_tiny_genimage_splits(split_config)
    print(f"\n=== {exp['name']} ===")
    print(splits["notes"])
    print("train real/fake:", splits["train_real_count"], splits["train_fake_count"])
    print("eval real/fake:", splits["eval_real_count"], splits["eval_fake_count"])

    train_loader, val_loader, test_loader = build_loaders(splits)
    model = make_model()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    best_path = run_dir / "checkpoints" / "best_resnet50_fc.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        y_val, p_val, _ = predict(model, val_loader)
        val_metrics = compute_metrics(y_val, p_val)
        current = val_metrics["balanced_accuracy"]
        history.append({"epoch": epoch, "train_loss": loss, **val_metrics})
        print(f"epoch {epoch}/{MAX_EPOCHS} loss={loss:.4f} val_bal_acc={current:.4f}")

        if current > best_metric + MIN_DELTA:
            best_metric = current
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    y_test, p_test, meta_df = predict(model, test_loader)
    metrics = compute_metrics(y_test, p_test)
    metrics.update(
        {
            "experiment_name": exp["name"],
            "eval_case": exp["eval_case"],
            "dataset": "TheKernel01/Tiny-GenImage",
            "streaming": STREAMING,
            "balance_real": BALANCE_REAL,
            "max_train_samples": MAX_TRAIN_SAMPLES,
            "max_test_samples": MAX_TEST_SAMPLES,
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_metric,
            "model_name": "resnet50_last_layer",
        }
    )

    pred_df = meta_df.copy()
    pred_df["label"] = y_test
    pred_df["predicted_label"] = (p_test >= 0.5).astype(int)
    pred_df["fake_probability"] = p_test
    gen_metrics = evaluate_by_generator(pred_df)

    pd.DataFrame(history).to_csv(run_dir / "metrics" / "history.csv", index=False)
    gen_metrics.to_csv(run_dir / "metrics" / "generator_metrics.csv", index=False)
    if SAVE_PREDICTIONS:
        pred_df.to_csv(run_dir / "predictions" / "predictions.csv", index=False)
    with open(run_dir / "metrics" / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    del train_loader, val_loader, test_loader, model, optimizer, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    print("CODE_ROOT =", CODE_ROOT)
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("DEVICE =", DEVICE)
    experiments = EXPERIMENT_CONFIGS if RUN_ALL_CASES else [
        exp for exp in EXPERIMENT_CONFIGS if exp["name"] == SELECTED_EXPERIMENT
    ]
    if not experiments:
        raise ValueError(f"Unknown SELECTED_EXPERIMENT={SELECTED_EXPERIMENT!r}")
    all_metrics = [run_experiment(exp) for exp in experiments]
    summary_df = pd.DataFrame(all_metrics)
    summary_path = OUTPUT_ROOT / f"summary_metrics_{RUN_ID}.csv"
    summary_df.to_csv(summary_path, index=False)
    print("Summary saved:", summary_path)
    print(summary_df[["experiment_name", "balanced_accuracy", "f1", "roc_auc", "best_epoch"]])


if __name__ == "__main__":
    main()
