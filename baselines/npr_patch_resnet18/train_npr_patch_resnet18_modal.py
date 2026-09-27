"""Train native sliding-patch NPR-ResNet18 experiments on Modal.

The suite supports all Tiny-GenImage regimes:

* combined
* in_domain (one run per generator)
* cross_generator (leave one generator out per run)
* train_one_generator (one source generator per run)

Only the best ``combined`` checkpoint is evaluated on a balanced,
multi-generator CommunityForensics-Eval sample.

Examples (run from the repository root):

    python -m modal run baselines/npr_patch_resnet18/train_npr_patch_resnet18_modal.py \
        --experiment-preset smoke

    python -m modal run baselines/npr_patch_resnet18/train_npr_patch_resnet18_modal.py \
        --experiment-preset all --full-train --max-epochs 10

    python -m modal run baselines/npr_patch_resnet18/train_npr_patch_resnet18_modal.py \
        --experiment-preset combined --suite-run-id 20260926_120000
"""

from __future__ import annotations

import time
from pathlib import Path

import modal


APP_NAME = "npr-patch-resnet18"
GPU_TYPE = "A100-40GB"

TINY_VOLUME_NAME = "tiny-genimage-data"
OUTPUT_VOLUME_NAME = "npr-patch-resnet18-outputs"
HF_CACHE_VOLUME_NAME = "hf-cache"

REMOTE_CODE_ROOT = "/root/HoangHa_Code"
REMOTE_TINY_ROOT = "/data/tiny-genimage"
REMOTE_OUTPUT_ROOT = "/outputs/npr_patch_resnet18"
REMOTE_HF_HOME = "/hf-cache"

def find_project_root() -> Path:
    """Resolve the repository both on the local launcher and in Modal."""

    file_path = Path(__file__).resolve()
    candidates = [Path.cwd(), Path(REMOTE_CODE_ROOT)]
    candidates.extend([file_path.parent, *file_path.parents])
    for candidate in candidates:
        if (candidate / "data_loader" / "__init__.py").exists():
            return candidate
    raise FileNotFoundError(
        "Cannot locate HoangHa_Code/data_loader from the local launcher or Modal container."
    )


LOCAL_PROJECT_ROOT = find_project_root()
LOCAL_DATA_LOADER_DIR = LOCAL_PROJECT_ROOT / "data_loader"
LOCAL_BASELINE_DIR = LOCAL_PROJECT_ROOT / "baselines" / "npr_patch_resnet18"

image = modal.Image.debian_slim(python_version="3.12").apt_install("git").pip_install(
    "torch",
    "torchvision",
    "datasets",
    "kaggle",
    "pandas<3.0",
    "scikit-learn<1.9",
    "pillow<12.0",
    "tqdm",
)
image = image.env(
    {
        "HF_HOME": REMOTE_HF_HOME,
        "HF_DATASETS_CACHE": f"{REMOTE_HF_HOME}/datasets",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)

function_mounts = []
if hasattr(image, "add_local_dir"):
    image = image.add_local_dir(str(LOCAL_DATA_LOADER_DIR), remote_path=f"{REMOTE_CODE_ROOT}/data_loader")
    image = image.add_local_dir(
        str(LOCAL_BASELINE_DIR),
        remote_path=f"{REMOTE_CODE_ROOT}/baselines/npr_patch_resnet18",
    )
elif hasattr(modal, "Mount"):
    function_mounts = [
        modal.Mount.from_local_dir(LOCAL_DATA_LOADER_DIR, remote_path=f"{REMOTE_CODE_ROOT}/data_loader"),
        modal.Mount.from_local_dir(
            LOCAL_BASELINE_DIR,
            remote_path=f"{REMOTE_CODE_ROOT}/baselines/npr_patch_resnet18",
        ),
    ]
else:
    raise RuntimeError("Modal SDK does not support add_local_dir or Mount.from_local_dir.")

app = modal.App(APP_NAME, image=image)
tiny_volume = modal.Volume.from_name(TINY_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

DEFAULT_CONFIG = {
    "tiny_dataset_root": REMOTE_TINY_ROOT,
    "output_root": REMOTE_OUTPUT_ROOT,
    "download_tiny_from_kaggle": False,
    "kaggle_dataset_slug": "yangsangtai/tiny-genimage",
    "balance_real": True,
    "random_seed": 42,
    "val_fraction": 0.2,
    "max_train_samples": None,
    "max_eval_samples": None,
    "max_val_samples": None,
    "max_epochs": 10,
    "patience": 3,
    "min_delta": 1e-3,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "gradient_accumulation_steps": 8,
    "max_grad_norm": 1.0,
    "checkpoint_every_optimizer_steps": 100,
    "resume": True,
    "patch_sizes": [256, 128, 64, 32],
    "stride_ratio": 0.5,
    "max_train_patches": 16,
    "max_eval_patches": 64,
    "patch_micro_batch_size": 16,
    "train_horizontal_flip_probability": 0.5,
    "num_workers": 0,
    "save_predictions": True,
    "commfor_dataset_name": "OwensLab/CommunityForensics-Eval",
    "commfor_split": "CompEval",
    "commfor_streaming": True,
    "commfor_shuffle_buffer_size": 1000,
    "commfor_discover_scan_limit": 50000,
    "commfor_fake_per_generator": 100,
    "commfor_real_reference_size": 100,
    "commfor_min_fake_per_generator": 100,
    "commfor_max_generators": 9,
    "commfor_target_generators": None,
}

common_function_options = {
    "gpu": GPU_TYPE,
    "timeout": 60 * 60 * 24,
    "memory": 65536,
    "volumes": {
        "/data": tiny_volume,
        "/outputs": output_volume,
        REMOTE_HF_HOME: hf_cache_volume,
    },
}
if function_mounts:
    common_function_options["mounts"] = function_mounts

train_function_options = dict(common_function_options)


def _experiment_slug(experiment: dict) -> str:
    return str(experiment["name"]).lower().replace(" ", "_")


@app.function(**train_function_options)
def train_npr_patch_experiment(suite_run_id: str, experiment: dict, config_overrides: dict | None = None) -> dict:
    import gc
    import json
    import os
    import random
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import ImageFile
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

    sys.path.insert(0, REMOTE_CODE_ROOT)
    from data_loader import (  # noqa: PLC0415
        TinyGenImageKaggleConfig,
        TinyGenImageKaggleDataset,
        build_kaggle_tiny_index,
        build_kaggle_tiny_splits,
        collate_unified_batch,
        find_tiny_genimage_root,
        summarize_index,
    )
    from baselines.npr_patch_resnet18.npr_patch_resnet18 import (  # noqa: PLC0415
        NPRPatchResNet18,
        extract_native_patches,
        pil_patch_to_normalized_tensor,
    )

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()
    pin_memory = torch.cuda.is_available()
    experiment_slug = _experiment_slug(experiment)
    run_dir = Path(config["output_root"]) / suite_run_id / experiment_slug
    for subdir in ["checkpoints/latest", "checkpoints/best", "dataset", "metrics", "predictions"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def save_json(path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def cleanup_cuda() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def looks_like_tiny_root(root: Path) -> bool:
        if not root.exists() or not root.is_dir():
            return False
        return any(
            child.is_dir() and (child / "train").exists()
            for child in root.iterdir()
        )

    def normalize_kaggle_secret() -> None:
        if "KAGGLE_USERNAME" not in os.environ and "username" in os.environ:
            os.environ["KAGGLE_USERNAME"] = os.environ["username"]
        if "KAGGLE_KEY" not in os.environ and "key" in os.environ:
            os.environ["KAGGLE_KEY"] = os.environ["key"]
        if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
            raise RuntimeError("Modal secret 'kaggle-secret' must contain KAGGLE_USERNAME and KAGGLE_KEY.")

    def find_downloaded_root(staging: Path) -> Path | None:
        for candidate in [staging / "tiny-genimage", staging / "tiny_genimage", staging]:
            if looks_like_tiny_root(candidate):
                return candidate
        for candidate in staging.rglob("*"):
            if candidate.is_dir() and looks_like_tiny_root(candidate):
                return candidate
        return None

    def ensure_tiny_dataset() -> Path:
        target = Path(config["tiny_dataset_root"])
        if looks_like_tiny_root(target):
            return find_tiny_genimage_root(str(target))
        if not config["download_tiny_from_kaggle"]:
            raise FileNotFoundError(
                f"Tiny-GenImage is not present at {target} in Modal Volume '{TINY_VOLUME_NAME}'. "
                "Populate that volume first. The existing Qwen Modal trainer supports "
                "--download-only after Modal secret 'kaggle-secret' is configured."
            )
        normalize_kaggle_secret()
        staging = Path("/data/_tiny_genimage_kaggle_download")
        staging.mkdir(parents=True, exist_ok=True)
        if find_downloaded_root(staging) is None:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", config["kaggle_dataset_slug"], "-p", str(staging), "--unzip", "-o"],
                check=True,
            )
        downloaded = find_downloaded_root(staging)
        if downloaded is None:
            raise FileNotFoundError("Kaggle download completed but Tiny-GenImage structure was not found.")
        target.parent.mkdir(parents=True, exist_ok=True)
        if downloaded.resolve() != target.resolve():
            if target.exists() and any(target.iterdir()):
                raise FileExistsError(f"Non-empty unrecognized Tiny target: {target}")
            if target.exists():
                target.rmdir()
            shutil.move(str(downloaded), str(target))
        tiny_volume.commit()
        return find_tiny_genimage_root(str(target))

    def maybe_limit(df: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
        if maximum is None or len(df) <= int(maximum):
            return df.reset_index(drop=True)
        return df.sample(n=int(maximum), random_state=seed, replace=False).reset_index(drop=True)

    def stratified_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        stratify = df["label"].astype(str) + "_" + df["generator"].astype(str)
        if stratify.value_counts().min() < 2:
            stratify = df["label"]
        train_df, val_df = train_test_split(
            df,
            test_size=config["val_fraction"],
            random_state=config["random_seed"],
            shuffle=True,
            stratify=stratify,
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    def split_config() -> TinyGenImageKaggleConfig:
        return TinyGenImageKaggleConfig(
            dataset_root=str(detected_root),
            eval_case=experiment["eval_case"],
            generator=experiment.get("generator"),
            heldout_generator=experiment.get("heldout_generator"),
            base_generator=experiment.get("base_generator"),
            balance_real=config["balance_real"],
            seed=config["random_seed"],
            max_train_samples=config["max_train_samples"],
            max_eval_samples=config["max_eval_samples"],
        )

    def make_loader(df: pd.DataFrame, eval_case: str) -> DataLoader:
        dataset = TinyGenImageKaggleDataset(df, eval_case=eval_case, transform=None)
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=pin_memory,
            collate_fn=collate_unified_batch,
        )

    def prepare_patch_bag(image, training: bool, sample_seed: int):
        rng = random.Random(sample_seed)
        patches, patch_info = extract_native_patches(
            image,
            patch_sizes=config["patch_sizes"],
            stride_ratio=config["stride_ratio"],
            max_patches=config["max_train_patches"] if training else config["max_eval_patches"],
            random_sample=training,
            rng=rng,
        )
        flip = training and rng.random() < float(config["train_horizontal_flip_probability"])
        tensors = [pil_patch_to_normalized_tensor(patch, horizontal_flip=flip) for patch in patches]
        return torch.stack(tensors), patch_info

    def compute_metrics(y_true, y_prob) -> dict[str, Any]:
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        metrics = {
            "num_samples": int(len(y_true)),
            "num_real": int((y_true == 0).sum()),
            "num_fake": int((y_true == 1).sum()),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "real_recall": float(tn / (tn + fp)) if tn + fp else None,
            "fake_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "fake_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "fake_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "confusion_matrix": matrix.tolist(),
        }
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        else:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None
        return metrics

    def generator_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for generator, part in pred_df.groupby("generator", dropna=False):
            row = compute_metrics(part["label"], part["fake_probability"])
            row["generator"] = generator
            row["mean_num_patches"] = float(part["num_patches"].mean())
            row["mean_fake_probability"] = float(part["fake_probability"].mean())
            rows.append(row)
        return pd.DataFrame(rows).sort_values("generator").reset_index(drop=True)

    @torch.no_grad()
    def predict(loader: DataLoader, description: str) -> pd.DataFrame:
        model.eval()
        rows = []
        for index, batch in enumerate(tqdm(loader, desc=description, leave=False)):
            patches, patch_info = prepare_patch_bag(batch["image"][0], training=False, sample_seed=config["random_seed"] + index)
            patches = patches.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                image_logits, patch_logits = model(patches, config["patch_micro_batch_size"])
            image_probability = torch.softmax(image_logits.float(), dim=-1)[0, 1].item()
            patch_probabilities = torch.softmax(patch_logits.float(), dim=-1)[:, 1].detach().cpu().numpy()
            row = dict(batch["metadata"][0])
            row.update(
                {
                    "label": int(batch["label"][0]),
                    "fake_probability": float(image_probability),
                    "predicted_label": int(image_probability >= 0.5),
                    "patch_size": patch_info["patch_size"],
                    "num_patches": patch_info["num_patches"],
                    "total_available_patches": patch_info["total_available_patches"],
                    "native_width": patch_info["native_width"],
                    "native_height": patch_info["native_height"],
                    "patch_fake_probability_mean": float(patch_probabilities.mean()),
                    "patch_fake_probability_max": float(patch_probabilities.max()),
                    "patch_fake_probability_std": float(patch_probabilities.std()),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def evaluate(loader: DataLoader, tag: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        pred_df = predict(loader, tag)
        metrics = compute_metrics(pred_df["label"], pred_df["fake_probability"])
        by_generator = generator_metrics(pred_df)
        save_json(run_dir / "metrics" / f"{tag}_overall_metrics.json", metrics)
        by_generator.to_csv(run_dir / "metrics" / f"{tag}_generator_metrics.csv", index=False)
        if config["save_predictions"]:
            pred_df.to_csv(run_dir / "predictions" / f"{tag}_predictions.csv", index=False)
        print(f"{tag} per-generator metrics:")
        print(by_generator.to_string(index=False))
        return metrics, by_generator, pred_df

    def save_checkpoint(
        checkpoint_dir: Path,
        epoch: int,
        next_batch_index: int,
        global_step: int,
        best_metric: float,
        best_epoch: int,
        bad_epochs: int,
    ) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "next_batch_index": next_batch_index,
                "global_step": global_step,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "experiment": experiment,
                "config": config,
            },
            checkpoint_dir / "training_state.pt",
        )
        save_json(
            checkpoint_dir / "checkpoint_info.json",
            {
                "epoch": epoch,
                "next_batch_index": next_batch_index,
                "global_step": global_step,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
            },
        )
        output_volume.commit()

    seed_everything(config["random_seed"])
    detected_root = ensure_tiny_dataset()
    print("Experiment:", experiment)
    print("Tiny root:", detected_root)
    print("Device:", device)
    save_json(run_dir / "config.json", {**config, "experiment": experiment, "suite_run_id": suite_run_id})

    full_index = build_kaggle_tiny_index(TinyGenImageKaggleConfig(dataset_root=str(detected_root)))
    summarize_index(full_index).to_csv(run_dir / "dataset" / "tiny_structure.csv", index=False)
    splits = build_kaggle_tiny_splits(split_config())
    train_df, val_df = stratified_train_val(splits["train_df"])
    val_df = maybe_limit(val_df, config["max_val_samples"], config["random_seed"])
    test_df = splits["eval_df"].reset_index(drop=True)
    train_df.to_csv(run_dir / "dataset" / "train_inner_split.csv", index=False)
    val_df.to_csv(run_dir / "dataset" / "val_inner_split.csv", index=False)
    test_df.to_csv(run_dir / "dataset" / "tiny_test_split.csv", index=False)

    val_loader = make_loader(val_df, experiment["eval_case"])
    test_loader = make_loader(test_df, experiment["eval_case"])
    model = NPRPatchResNet18(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    start_batch_index = 0
    global_step = 0
    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    latest_state = run_dir / "checkpoints" / "latest" / "training_state.pt"
    if config["resume"] and latest_state.exists():
        state = torch.load(latest_state, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"])
        start_batch_index = int(state["next_batch_index"])
        global_step = int(state["global_step"])
        best_metric = float(state["best_metric"])
        best_epoch = int(state["best_epoch"])
        bad_epochs = int(state["bad_epochs"])
        print("Resuming:", start_epoch, start_batch_index, global_step)

    history_path = run_dir / "metrics" / "history.csv"
    history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    optimizer.zero_grad(set_to_none=True)
    stop_training = False
    for epoch in range(start_epoch, int(config["max_epochs"]) + 1):
        epoch_df = train_df.sample(frac=1, random_state=config["random_seed"] + epoch).reset_index(drop=True)
        current_start = start_batch_index if epoch == start_epoch else 0
        epoch_dataset = TinyGenImageKaggleDataset(
            epoch_df.iloc[current_start:].reset_index(drop=True),
            eval_case=experiment["eval_case"],
            transform=None,
        )
        epoch_loader = DataLoader(
            epoch_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=pin_memory,
            collate_fn=collate_unified_batch,
        )
        model.train()
        losses = []
        accumulation = int(config["gradient_accumulation_steps"])
        progress = tqdm(epoch_loader, desc=f"{experiment_slug} epoch {epoch}", leave=False)
        for local_index, batch in enumerate(progress):
            absolute_index = current_start + local_index
            patches, _ = prepare_patch_bag(
                batch["image"][0],
                training=True,
                sample_seed=config["random_seed"] + epoch * 10_000_000 + absolute_index,
            )
            patches = patches.to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            accumulation_group_start = (absolute_index // accumulation) * accumulation
            accumulation_divisor = min(accumulation, len(epoch_df) - accumulation_group_start)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                image_logits, _ = model(patches, config["patch_micro_batch_size"])
                raw_loss = criterion(image_logits, labels)
                loss = raw_loss / accumulation_divisor
            scaler.scale(loss).backward()
            losses.append(float(raw_loss.detach().cpu()))
            should_step = (absolute_index + 1) % accumulation == 0 or absolute_index + 1 == len(epoch_df)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                every = int(config["checkpoint_every_optimizer_steps"])
                if every > 0 and global_step % every == 0:
                    save_checkpoint(
                        run_dir / "checkpoints" / "latest",
                        epoch,
                        absolute_index + 1,
                        global_step,
                        best_metric,
                        best_epoch,
                        bad_epochs,
                    )
            progress.set_postfix(loss=float(raw_loss.detach().cpu()), patches=len(patches))

        val_metrics, val_by_generator, _ = evaluate(val_loader, "tiny_val")
        macro_generator_bacc = float(val_by_generator["balanced_accuracy"].mean())
        train_loss = float(np.mean(losses)) if losses else None
        history_row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "macro_generator_balanced_accuracy": macro_generator_bacc,
            **val_metrics,
        }
        history.append(history_row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        if macro_generator_bacc > best_metric + float(config["min_delta"]):
            best_metric = macro_generator_bacc
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best" / "model.pt")
            save_json(
                run_dir / "checkpoints" / "best" / "checkpoint_info.json",
                {"epoch": epoch, "macro_generator_balanced_accuracy": best_metric},
            )
        else:
            bad_epochs += 1

        save_checkpoint(
            run_dir / "checkpoints" / "latest",
            epoch + 1,
            0,
            global_step,
            best_metric,
            best_epoch,
            bad_epochs,
        )
        start_batch_index = 0
        if bad_epochs >= int(config["patience"]):
            stop_training = True
        if stop_training:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    best_path = run_dir / "checkpoints" / "best" / "model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Best checkpoint was not created: {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    tiny_metrics, tiny_by_generator, _ = evaluate(test_loader, "tiny_test")
    summary = {
        "suite_run_id": suite_run_id,
        "experiment_name": experiment["name"],
        "eval_case": experiment["eval_case"],
        "best_epoch": best_epoch,
        "best_val_macro_generator_balanced_accuracy": best_metric,
        "tiny_test_balanced_accuracy": tiny_metrics["balanced_accuracy"],
        "tiny_test_roc_auc": tiny_metrics["roc_auc"],
        "tiny_test_macro_generator_balanced_accuracy": float(tiny_by_generator["balanced_accuracy"].mean()),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
    }
    summary.update({key: value for key, value in experiment.items() if key not in summary})
    save_json(run_dir / "metrics" / "summary.json", summary)
    pd.DataFrame([summary]).to_csv(run_dir / "metrics" / "summary.csv", index=False)
    output_volume.commit()
    cleanup_cuda()
    return summary


@app.function(**common_function_options)
def evaluate_combined_on_commfor(
    suite_run_id: str,
    combined_experiment_name: str,
    config_overrides: dict | None = None,
) -> dict:
    import gc
    import io
    import json
    import random
    import sys
    from collections import Counter
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_dataset
    from PIL import Image, ImageFile
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
    from tqdm.auto import tqdm

    sys.path.insert(0, REMOTE_CODE_ROOT)
    from baselines.npr_patch_resnet18.npr_patch_resnet18 import (  # noqa: PLC0415
        NPRPatchResNet18,
        extract_native_patches,
        pil_patch_to_normalized_tensor,
    )

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    if isinstance(config.get("commfor_target_generators"), str):
        config["commfor_target_generators"] = [
            item.strip() for item in config["commfor_target_generators"].split(",") if item.strip()
        ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()
    combined_slug = combined_experiment_name.lower().replace(" ", "_")
    combined_dir = Path(config["output_root"]) / suite_run_id / combined_slug
    best_path = combined_dir / "checkpoints" / "best" / "model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Combined best checkpoint not found: {best_path}")
    training_config_path = combined_dir / "config.json"
    if training_config_path.exists():
        with open(training_config_path, "r", encoding="utf-8") as handle:
            training_config = json.load(handle)
        # These settings define the learned input representation and must stay
        # aligned with training. Evaluation-only quotas remain overrideable.
        for key in ["patch_sizes", "stride_ratio"]:
            if key in training_config:
                config[key] = training_config[key]
    run_dir = Path(config["output_root"]) / suite_run_id / "commfor_combined"
    for subdir in ["dataset", "metrics", "predictions"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def save_json(path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def image_from_record(record: dict[str, Any]) -> Image.Image:
        image_data = record.get("image_data", record.get("image"))
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        if isinstance(image_data, (bytes, bytearray)):
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        if isinstance(image_data, dict):
            if image_data.get("bytes") is not None:
                return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
            if image_data.get("path") is not None:
                return Image.open(image_data["path"]).convert("RGB")
        if isinstance(image_data, list):
            return Image.open(io.BytesIO(bytes(image_data))).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image_data)}")

    def generator_from_record(record: dict[str, Any]) -> str:
        return str(record.get("model_name") or record.get("architecture") or "unknown")

    def raw_stream(seed_offset: int = 0):
        dataset = load_dataset(
            config["commfor_dataset_name"],
            split=config["commfor_split"],
            streaming=config["commfor_streaming"],
        )
        if config["commfor_streaming"]:
            return dataset.shuffle(
                seed=config["random_seed"] + seed_offset,
                buffer_size=config["commfor_shuffle_buffer_size"],
            )
        return dataset.shuffle(seed=config["random_seed"] + seed_offset)

    fake_counts: Counter[str] = Counter()
    real_source_counts: Counter[str] = Counter()
    architecture_counts: Counter[str] = Counter()
    for scanned, record in enumerate(raw_stream(seed_offset=0), start=1):
        label = int(record.get("label"))
        if label == 1:
            fake_counts[generator_from_record(record)] += 1
        else:
            real_source_counts[str(record.get("real_source") or "unknown")] += 1
        architecture_counts[str(record.get("architecture") or "unknown")] += 1
        if scanned >= int(config["commfor_discover_scan_limit"]):
            break

    discovered_df = pd.DataFrame(
        [{"generator": key, "fake_seen": value} for key, value in fake_counts.items()]
    ).sort_values(["fake_seen", "generator"], ascending=[False, True])
    discovered_df.to_csv(run_dir / "dataset" / "discovered_generator_counts.csv", index=False)
    pd.DataFrame(
        [{"real_source": key, "real_seen": value} for key, value in real_source_counts.items()]
    ).sort_values("real_seen", ascending=False).to_csv(run_dir / "dataset" / "discovered_real_source_counts.csv", index=False)
    pd.DataFrame(
        [{"architecture": key, "num_seen": value} for key, value in architecture_counts.items()]
    ).sort_values("num_seen", ascending=False).to_csv(run_dir / "dataset" / "discovered_architecture_counts.csv", index=False)

    if config["commfor_target_generators"]:
        target_generators = list(config["commfor_target_generators"])
    else:
        target_generators = discovered_df.loc[
            discovered_df["fake_seen"] >= int(config["commfor_min_fake_per_generator"]), "generator"
        ].tolist()
    if config["commfor_max_generators"] is not None:
        target_generators = target_generators[: int(config["commfor_max_generators"])]
    if not target_generators:
        raise RuntimeError("No CommFor generators satisfy the configured minimum fake quota.")

    fake_quota = int(config["commfor_fake_per_generator"])
    real_quota = int(config["commfor_real_reference_size"])
    selected_fake: dict[str, list[dict]] = {generator: [] for generator in target_generators}
    selected_real: list[dict] = []
    for record in raw_stream(seed_offset=1):
        label = int(record.get("label"))
        if label == 0 and len(selected_real) < real_quota:
            selected_real.append(record)
        elif label == 1:
            generator = generator_from_record(record)
            if generator in selected_fake and len(selected_fake[generator]) < fake_quota:
                selected_fake[generator].append(record)
        if len(selected_real) >= real_quota and all(len(rows) >= fake_quota for rows in selected_fake.values()):
            break

    missing = {generator: fake_quota - len(rows) for generator, rows in selected_fake.items() if len(rows) < fake_quota}
    if len(selected_real) < real_quota or missing:
        raise RuntimeError(f"CommFor stream ended before quotas were filled: real={len(selected_real)}/{real_quota}, fake_missing={missing}")

    selected_rows = []
    for index, record in enumerate(selected_real):
        selected_rows.append(
            {
                "selection_group": "shared_real_reference",
                "sample_index": index,
                "label": 0,
                "generator": "shared_real",
                "image_name": record.get("image_name"),
                "real_source": record.get("real_source"),
                "architecture": record.get("architecture"),
            }
        )
    for generator, records in selected_fake.items():
        for index, record in enumerate(records):
            selected_rows.append(
                {
                    "selection_group": "generator_fake",
                    "sample_index": index,
                    "label": 1,
                    "generator": generator,
                    "image_name": record.get("image_name"),
                    "real_source": record.get("real_source"),
                    "architecture": record.get("architecture"),
                }
            )
    pd.DataFrame(selected_rows).to_csv(run_dir / "dataset" / "selected_samples.csv", index=False)
    selected_counts_df = (
        pd.DataFrame(selected_rows)
        .groupby(["selection_group", "generator", "label"])
        .size()
        .reset_index(name="num_samples")
    )
    selected_counts_df.to_csv(run_dir / "dataset" / "selected_generator_label_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "generator": generator,
                "metric_cohort": "generator_fake_plus_shared_real",
                "num_real": real_quota,
                "num_fake": len(selected_fake[generator]),
            }
            for generator in target_generators
        ]
    ).to_csv(run_dir / "dataset" / "evaluation_cohorts.csv", index=False)

    model = NPRPatchResNet18(num_classes=2).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    @torch.no_grad()
    def predict_record(record: dict, label: int, generator: str, sample_index: int) -> dict:
        image = image_from_record(record)
        patches, patch_info = extract_native_patches(
            image,
            patch_sizes=config["patch_sizes"],
            stride_ratio=config["stride_ratio"],
            max_patches=config["max_eval_patches"],
            random_sample=False,
            rng=random.Random(config["random_seed"] + sample_index),
        )
        patch_tensor = torch.stack([pil_patch_to_normalized_tensor(patch) for patch in patches]).to(device)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            image_logits, patch_logits = model(patch_tensor, config["patch_micro_batch_size"])
        fake_probability = torch.softmax(image_logits.float(), dim=-1)[0, 1].item()
        patch_probabilities = torch.softmax(patch_logits.float(), dim=-1)[:, 1].cpu().numpy()
        return {
            "sample_id": str(record.get("image_name") or f"commfor:{label}:{generator}:{sample_index}"),
            "label": label,
            "label_name": "fake" if label == 1 else "real",
            "generator": generator,
            "architecture": record.get("architecture"),
            "real_source": record.get("real_source"),
            "subset": record.get("subset"),
            "image_name": record.get("image_name"),
            "fake_probability": float(fake_probability),
            "predicted_label": int(fake_probability >= 0.5),
            "patch_size": patch_info["patch_size"],
            "num_patches": patch_info["num_patches"],
            "total_available_patches": patch_info["total_available_patches"],
            "native_width": patch_info["native_width"],
            "native_height": patch_info["native_height"],
            "patch_fake_probability_mean": float(patch_probabilities.mean()),
            "patch_fake_probability_max": float(patch_probabilities.max()),
            "patch_fake_probability_std": float(patch_probabilities.std()),
        }

    predictions = []
    for index, record in enumerate(tqdm(selected_real, desc="CommFor shared real")):
        predictions.append(predict_record(record, 0, "shared_real", index))
    for generator in target_generators:
        for index, record in enumerate(tqdm(selected_fake[generator], desc=f"CommFor {generator}")):
            predictions.append(predict_record(record, 1, generator, index))
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(run_dir / "predictions" / "commfor_predictions.csv", index=False)

    def compute_metrics(frame: pd.DataFrame) -> dict[str, Any]:
        y_true = frame["label"].to_numpy(dtype=int)
        y_prob = frame["fake_probability"].to_numpy(dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "num_samples": int(len(frame)),
            "num_real": int((y_true == 0).sum()),
            "num_fake": int((y_true == 1).sum()),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "real_recall": float(tn / (tn + fp)) if tn + fp else None,
            "fake_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "fake_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "fake_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "mean_num_patches": float(frame["num_patches"].mean()),
            "mean_fake_probability": float(frame["fake_probability"].mean()),
        }

    real_predictions = pred_df[pred_df["label"] == 0]
    metric_rows = []
    for generator in target_generators:
        fake_predictions = pred_df[(pred_df["label"] == 1) & (pred_df["generator"] == generator)]
        cohort = pd.concat([real_predictions, fake_predictions], ignore_index=True)
        metrics = compute_metrics(cohort)
        metrics.update(
            {
                "generator": generator,
                "metric_cohort": "generator_fake_plus_shared_real",
            }
        )
        metric_rows.append(metrics)
    generator_metrics_df = pd.DataFrame(metric_rows).sort_values("generator").reset_index(drop=True)
    generator_metrics_df.to_csv(run_dir / "metrics" / "commfor_generator_metrics.csv", index=False)
    print("CommFor per-generator metrics:")
    print(generator_metrics_df.to_string(index=False))

    macro_columns = [
        "accuracy", "balanced_accuracy", "real_recall", "fake_recall", "fake_precision",
        "fake_f1", "roc_auc", "average_precision",
    ]
    macro = {f"macro_generator_{column}": float(generator_metrics_df[column].mean()) for column in macro_columns}
    macro.update(
        {
            "worst_generator_balanced_accuracy": float(generator_metrics_df["balanced_accuracy"].min()),
            "best_generator_balanced_accuracy": float(generator_metrics_df["balanced_accuracy"].max()),
            "num_generators": int(len(generator_metrics_df)),
            "fake_per_generator": fake_quota,
            "shared_real_reference_size": real_quota,
            "target_generators": target_generators,
            "suite_run_id": suite_run_id,
            "combined_checkpoint": str(best_path),
            "metric_cohort": "generator_fake_plus_shared_real",
        }
    )
    save_json(run_dir / "metrics" / "macro_summary.json", macro)
    save_json(run_dir / "config.json", config)
    output_volume.commit()
    hf_cache_volume.commit()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return macro


def build_experiment_manifest(preset: str, generators: list[str]) -> list[dict]:
    combined = {"name": "combined", "eval_case": "combined"}
    if preset == "combined" or preset == "smoke":
        return [combined]
    if preset == "notebook":
        return [
            combined,
            {"name": "in_domain_biggan", "eval_case": "in_domain", "generator": "BigGAN"},
            {"name": "cross_generator_glide", "eval_case": "cross_generator", "heldout_generator": "GLIDE"},
            {"name": "cross_generator_wukong", "eval_case": "cross_generator", "heldout_generator": "Wukong"},
            {"name": "train_one_generator_biggan", "eval_case": "train_one_generator", "base_generator": "BigGAN"},
        ]
    if preset != "all":
        raise ValueError("experiment_preset must be one of: smoke, combined, notebook, all")
    experiments = [combined]
    experiments.extend(
        {"name": f"in_domain_{generator.lower()}", "eval_case": "in_domain", "generator": generator}
        for generator in generators
    )
    experiments.extend(
        {
            "name": f"cross_generator_{generator.lower()}",
            "eval_case": "cross_generator",
            "heldout_generator": generator,
        }
        for generator in generators
    )
    experiments.extend(
        {
            "name": f"train_one_generator_{generator.lower()}",
            "eval_case": "train_one_generator",
            "base_generator": generator,
        }
        for generator in generators
    )
    return experiments


summary_function_options = {
    "timeout": 60 * 30,
    "memory": 4096,
    "volumes": {"/outputs": output_volume},
}


@app.function(**summary_function_options)
def save_suite_summary(
    suite_run_id: str,
    experiments: list[dict],
    summaries: list[dict],
    config_overrides: dict,
    commfor_summary: dict | None,
) -> str:
    """Persist the suite manifest even though orchestration runs locally."""

    import json
    from pathlib import Path

    import pandas as pd

    suite_dir = Path(REMOTE_OUTPUT_ROOT) / suite_run_id
    summary_dir = suite_dir / "suite_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(experiments).to_csv(summary_dir / "experiment_manifest.csv", index=False)
    pd.DataFrame(summaries).to_csv(summary_dir / "all_experiments.csv", index=False)
    for eval_case, part in pd.DataFrame(summaries).groupby("eval_case"):
        part.to_csv(summary_dir / f"{eval_case}.csv", index=False)
    with open(summary_dir / "suite_config.json", "w", encoding="utf-8") as handle:
        json.dump(config_overrides, handle, ensure_ascii=False, indent=2)
    with open(summary_dir / "suite_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "suite_run_id": suite_run_id,
                "num_experiments": len(experiments),
                "training_summaries": summaries,
                "commfor_summary": commfor_summary,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    output_volume.commit()
    return str(summary_dir)


@app.local_entrypoint()
def main(
    experiment_preset: str = "all",
    generators: str = "ADM,BigGAN,GLIDE,Midjourney,SD15,VQDM,Wukong",
    suite_run_id: str | None = None,
    max_train_samples: int = 5000,
    full_train: bool = False,
    max_eval_samples: int = 1000,
    full_tiny_eval: bool = False,
    max_val_samples: int = 1000,
    max_epochs: int = 10,
    max_train_patches: int = 16,
    max_eval_patches: int = 64,
    gradient_accumulation_steps: int = 8,
    checkpoint_every_optimizer_steps: int = 100,
    no_resume: bool = False,
    skip_commfor: bool = False,
    commfor_only: bool = False,
    commfor_fake_per_generator: int = 100,
    commfor_real_reference_size: int = 100,
    commfor_min_fake_per_generator: int = 100,
    commfor_max_generators: int | None = 9,
    commfor_target_generators: str | None = None,
):
    """Run the selected experiment suite sequentially, then test combined on CommFor."""

    generator_names = [item.strip() for item in generators.split(",") if item.strip()]
    requested_suite_run_id = suite_run_id
    suite_run_id = suite_run_id or time.strftime("%Y%m%d_%H%M%S")
    experiments = build_experiment_manifest(experiment_preset, generator_names)
    if commfor_only and not requested_suite_run_id:
        raise ValueError("--commfor-only requires --suite-run-id containing a trained combined checkpoint.")
    if experiment_preset == "smoke":
        max_train_samples = min(max_train_samples, 200)
        max_eval_samples = min(max_eval_samples, 100)
        max_val_samples = min(max_val_samples, 100)
        max_epochs = 1
        commfor_fake_per_generator = min(commfor_fake_per_generator, 10)
        commfor_real_reference_size = min(commfor_real_reference_size, 10)
        commfor_min_fake_per_generator = min(commfor_min_fake_per_generator, 10)

    overrides = {
        "max_train_samples": None if full_train else max_train_samples,
        "max_eval_samples": None if full_tiny_eval else max_eval_samples,
        "max_val_samples": max_val_samples,
        "max_epochs": max_epochs,
        "max_train_patches": max_train_patches,
        "max_eval_patches": max_eval_patches,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "checkpoint_every_optimizer_steps": checkpoint_every_optimizer_steps,
        "resume": not no_resume,
        "commfor_fake_per_generator": commfor_fake_per_generator,
        "commfor_real_reference_size": commfor_real_reference_size,
        "commfor_min_fake_per_generator": commfor_min_fake_per_generator,
        "commfor_max_generators": commfor_max_generators,
        "commfor_target_generators": commfor_target_generators,
    }

    print("Suite run ID:", suite_run_id)
    print("Experiments:", [experiment["name"] for experiment in experiments])
    summaries = []
    if not commfor_only:
        for index, experiment in enumerate(experiments, start=1):
            print(f"Starting experiment {index}/{len(experiments)}: {experiment['name']}")
            summary = train_npr_patch_experiment.remote(suite_run_id, experiment, overrides)
            summaries.append(summary)
            print("Completed:", summary)

    commfor_summary = None
    combined_was_run = commfor_only or any(experiment["eval_case"] == "combined" for experiment in experiments)
    if combined_was_run and not skip_commfor:
        commfor_summary = evaluate_combined_on_commfor.remote(suite_run_id, "combined", overrides)
        print("CommFor summary:", commfor_summary)

    if not commfor_only:
        summary_dir = save_suite_summary.remote(
            suite_run_id,
            experiments,
            summaries,
            overrides,
            commfor_summary,
        )
        print("Suite summary saved:", summary_dir)

    print("Suite completed:", suite_run_id)
    print("Training summaries:", summaries)
    if commfor_summary is not None:
        print("Combined CommFor summary:", commfor_summary)
