"""Test the trained NPR-ResNet18 checkpoint on AIDE's exact CommFor cohort.

This is an inference-only Modal entrypoint.  It mounts:

* the existing NPR-ResNet18 checkpoint from run ``20260906_122613``;
* AIDE run ``20260927_001615``'s ``selected_samples.csv`` manifest.

Every CommunityForensics record is matched using the composite key
``(label, generator, image_name, real_source, architecture)``.  The job fails
if all 1,000 manifest rows cannot be recovered, which prevents an accidental
comparison on a different random sample.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal


APP_NAME = "npr-resnet18-commfor-aide-cohort"
GPU_TYPE = "T4"
OUTPUT_VOLUME_NAME = "npr-resnet18-commfor-aide-outputs"
HF_CACHE_VOLUME_NAME = "hf-cache"

REMOTE_CODE_ROOT = "/root/HoangHa_Code"
REMOTE_CHECKPOINT_PATH = "/root/checkpoints/npr_resnet18_from_scratch.pt"
REMOTE_AIDE_MANIFEST_PATH = "/root/manifests/selected_samples.csv"
REMOTE_OUTPUT_ROOT = "/outputs/npr_resnet18_commfor_aide_manifest"
REMOTE_HF_HOME = "/hf-cache"

CHECKPOINT_SOURCE_RUN_ID = "20260906_122613"
AIDE_SOURCE_RUN_ID = "20260927_001615"


def find_project_root() -> Path:
    file_path = Path(__file__).resolve()
    candidates = [Path.cwd(), Path(REMOTE_CODE_ROOT), file_path.parent, *file_path.parents]
    for candidate in candidates:
        if (candidate / "data_loader" / "__init__.py").exists():
            return candidate
    raise FileNotFoundError("Cannot locate HoangHa_Code/data_loader.")


LOCAL_PROJECT_ROOT = find_project_root()
LOCAL_DATA_LOADER_DIR = LOCAL_PROJECT_ROOT / "data_loader"
LOCAL_NPR_DIR = LOCAL_PROJECT_ROOT / "baselines" / "npr_resnet18"
LOCAL_CHECKPOINT_PATH = (
    LOCAL_PROJECT_ROOT
    / "baselines"
    / "npr_resnet18"
    / "artifacts"
    / "checkpoints"
    / "npr_resnet18_from_scratch.pt"
)
LOCAL_AIDE_MANIFEST_PATH = (
    LOCAL_PROJECT_ROOT
    / "baselines"
    / "aide_original_forensic_resnet50"
    / "artifacts"
    / "runs"
    / "modal_download"
    / AIDE_SOURCE_RUN_ID
    / "commfor_combined"
    / "dataset"
    / "selected_samples.csv"
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        "datasets",
        "pandas<3.0",
        "scikit-learn<1.9",
        "pillow<12.0",
        "tqdm",
    )
    .env(
        {
            "HF_HOME": REMOTE_HF_HOME,
            "HF_DATASETS_CACHE": f"{REMOTE_HF_HOME}/datasets",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

function_mounts = []
if hasattr(image, "add_local_dir"):
    image = image.add_local_dir(str(LOCAL_DATA_LOADER_DIR), remote_path=f"{REMOTE_CODE_ROOT}/data_loader")
    image = image.add_local_dir(
        str(LOCAL_NPR_DIR),
        remote_path=f"{REMOTE_CODE_ROOT}/baselines/npr_resnet18",
    )
    # On the local launcher, attach the Windows files.  During container
    # hydration these remote targets already exist, so do not re-resolve the
    # unavailable Windows/project-relative source paths.
    if not Path(REMOTE_CHECKPOINT_PATH).is_file():
        if not LOCAL_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"NPR checkpoint not found: {LOCAL_CHECKPOINT_PATH}")
        image = image.add_local_file(str(LOCAL_CHECKPOINT_PATH), remote_path=REMOTE_CHECKPOINT_PATH)
    if not Path(REMOTE_AIDE_MANIFEST_PATH).is_file():
        if not LOCAL_AIDE_MANIFEST_PATH.is_file():
            raise FileNotFoundError(f"AIDE CommFor manifest not found: {LOCAL_AIDE_MANIFEST_PATH}")
        image = image.add_local_file(str(LOCAL_AIDE_MANIFEST_PATH), remote_path=REMOTE_AIDE_MANIFEST_PATH)
else:
    # Compatibility path for older Modal SDKs.
    if not LOCAL_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"NPR checkpoint not found: {LOCAL_CHECKPOINT_PATH}")
    if not LOCAL_AIDE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"AIDE CommFor manifest not found: {LOCAL_AIDE_MANIFEST_PATH}")
    function_mounts = [
        modal.Mount.from_local_dir(LOCAL_DATA_LOADER_DIR, remote_path=f"{REMOTE_CODE_ROOT}/data_loader"),
        modal.Mount.from_local_dir(LOCAL_NPR_DIR, remote_path=f"{REMOTE_CODE_ROOT}/baselines/npr_resnet18"),
        modal.Mount.from_local_dir(LOCAL_CHECKPOINT_PATH.parent, remote_path="/root/checkpoints"),
        modal.Mount.from_local_dir(LOCAL_AIDE_MANIFEST_PATH.parent, remote_path="/root/manifests"),
    ]

app = modal.App(APP_NAME, image=image)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

FUNCTION_OPTIONS = {
    "gpu": GPU_TYPE,
    "timeout": 60 * 60 * 6,
    "memory": 32768,
    "volumes": {
        "/outputs": output_volume,
        REMOTE_HF_HOME: hf_cache_volume,
    },
}
if function_mounts:
    FUNCTION_OPTIONS["mounts"] = function_mounts

DEFAULT_CONFIG = {
    "output_root": REMOTE_OUTPUT_ROOT,
    "checkpoint_path": REMOTE_CHECKPOINT_PATH,
    "aide_manifest_path": REMOTE_AIDE_MANIFEST_PATH,
    "commfor_dataset_name": "OwensLab/CommunityForensics-Eval",
    "commfor_split": "CompEval",
    "commfor_streaming": True,
    "commfor_shuffle_seed": 43,
    "commfor_shuffle_buffer_size": 1000,
    "max_scan_records": 500000,
    "batch_size": 32,
    "image_size": 224,
    "random_seed": 42,
}


@app.function(**FUNCTION_OPTIONS)
def test_npr_on_aide_commfor(inference_id: str, config_overrides: dict | None = None) -> dict:
    import gc
    import hashlib
    import io
    import json
    import random
    import sys
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
    from baselines.npr_resnet18.npr_resnet18 import NPRResNet18  # noqa: PLC0415
    from data_loader import build_image_transform  # noqa: PLC0415

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)

    random.seed(int(config["random_seed"]))
    np.random.seed(int(config["random_seed"]))
    torch.manual_seed(int(config["random_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["random_seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    run_dir = Path(config["output_root"]) / inference_id
    for subdir in ["dataset", "metrics", "predictions"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def save_json(path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def clean(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return str(value)

    def manifest_key(row: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(int(row["label"])),
            clean(row["generator"]),
            clean(row["image_name"]),
            clean(row["real_source"]),
            clean(row["architecture"]),
        )

    def generator_from_record(record: dict[str, Any]) -> str:
        return str(record.get("model_name") or record.get("architecture") or "unknown")

    def record_key(record: dict[str, Any]) -> tuple[str, ...]:
        label = int(record.get("label"))
        generator = "shared_real" if label == 0 else generator_from_record(record)
        return (
            str(label),
            generator,
            clean(record.get("image_name")),
            clean(record.get("real_source")),
            clean(record.get("architecture")),
        )

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
        raise TypeError(f"Unsupported CommFor image type: {type(image_data)}")

    manifest = pd.read_csv(config["aide_manifest_path"])
    required_columns = {
        "selection_group",
        "sample_index",
        "label",
        "generator",
        "image_name",
        "real_source",
        "architecture",
    }
    missing_columns = sorted(required_columns.difference(manifest.columns))
    if missing_columns:
        raise ValueError(f"AIDE manifest is missing columns: {missing_columns}")
    manifest = manifest.copy()
    manifest["_manifest_order"] = np.arange(len(manifest), dtype=int)
    manifest_records = manifest.to_dict("records")
    manifest_by_key = {manifest_key(row): row for row in manifest_records}
    if len(manifest_by_key) != len(manifest):
        raise ValueError("AIDE manifest composite keys are not unique; exact matching is ambiguous.")
    expected_count = int(len(manifest))
    expected_real = int((manifest["label"] == 0).sum())
    expected_fake = int((manifest["label"] == 1).sum())
    expected_generators = manifest.loc[manifest["label"] == 1, "generator"].drop_duplicates().tolist()

    checkpoint_path = Path(config["checkpoint_path"])
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint_payload, torch.nn.Module):
        state_dict = checkpoint_payload.state_dict()
    elif isinstance(checkpoint_payload, dict):
        if "model_state_dict" in checkpoint_payload:
            state_dict = checkpoint_payload["model_state_dict"]
        elif "state_dict" in checkpoint_payload:
            state_dict = checkpoint_payload["state_dict"]
        elif "model" in checkpoint_payload and isinstance(checkpoint_payload["model"], dict):
            state_dict = checkpoint_payload["model"]
        else:
            state_dict = checkpoint_payload
    else:
        raise TypeError(f"Unsupported checkpoint payload: {type(checkpoint_payload)}")
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    model = NPRResNet18(num_classes=2)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model = model.to(device).eval()
    transform = build_image_transform(image_size=int(config["image_size"]), train=False)

    dataset = load_dataset(
        config["commfor_dataset_name"],
        split=config["commfor_split"],
        streaming=bool(config["commfor_streaming"]),
    )
    if bool(config["commfor_streaming"]):
        dataset = dataset.shuffle(
            seed=int(config["commfor_shuffle_seed"]),
            buffer_size=int(config["commfor_shuffle_buffer_size"]),
        )
    else:
        dataset = dataset.shuffle(seed=int(config["commfor_shuffle_seed"]))

    remaining_keys = set(manifest_by_key)
    prediction_rows: list[dict[str, Any]] = []
    batch_tensors: list[torch.Tensor] = []
    batch_metadata: list[dict[str, Any]] = []

    @torch.no_grad()
    def flush_batch() -> None:
        if not batch_tensors:
            return
        images = torch.stack(batch_tensors).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        for metadata, probability in zip(batch_metadata, probabilities):
            row = dict(metadata)
            row["fake_probability"] = float(probability)
            row["predicted_label"] = int(probability >= 0.5)
            prediction_rows.append(row)
        batch_tensors.clear()
        batch_metadata.clear()

    scanned = 0
    progress = tqdm(total=expected_count, desc="Matching exact AIDE CommFor cohort")
    for record in dataset:
        scanned += 1
        key = record_key(record)
        if key in remaining_keys:
            manifest_row = manifest_by_key[key]
            image = image_from_record(record)
            batch_tensors.append(transform(image))
            batch_metadata.append(
                {
                    "manifest_order": int(manifest_row["_manifest_order"]),
                    "selection_group": manifest_row["selection_group"],
                    "sample_index": int(manifest_row["sample_index"]),
                    "sample_id": clean(record.get("image_name")),
                    "label": int(manifest_row["label"]),
                    "label_name": "fake" if int(manifest_row["label"]) == 1 else "real",
                    "generator": clean(manifest_row["generator"]),
                    "image_name": clean(record.get("image_name")),
                    "model_name": clean(record.get("model_name")),
                    "architecture": clean(record.get("architecture")),
                    "real_source": clean(record.get("real_source")),
                    "subset": clean(record.get("subset")),
                    "native_width": int(image.width),
                    "native_height": int(image.height),
                }
            )
            remaining_keys.remove(key)
            progress.update(1)
            if len(batch_tensors) >= int(config["batch_size"]):
                flush_batch()
            if not remaining_keys:
                break
        if scanned >= int(config["max_scan_records"]):
            break
    flush_batch()
    progress.close()

    if remaining_keys:
        missing_rows = [manifest_by_key[key] for key in sorted(remaining_keys)]
        pd.DataFrame(missing_rows).to_csv(run_dir / "dataset" / "missing_manifest_samples.csv", index=False)
        output_volume.commit()
        raise RuntimeError(
            f"Only matched {expected_count - len(remaining_keys)}/{expected_count} AIDE samples after "
            f"scanning {scanned} CommFor records. Missing manifest written to output."
        )

    predictions = pd.DataFrame(prediction_rows).sort_values("manifest_order").reset_index(drop=True)
    if len(predictions) != expected_count:
        raise AssertionError(f"Expected {expected_count} predictions, got {len(predictions)}")
    predictions.to_csv(run_dir / "predictions" / "commfor_aide_cohort_predictions.csv", index=False)
    manifest.drop(columns=["_manifest_order"]).to_csv(
        run_dir / "dataset" / "aide_selected_samples_manifest.csv", index=False
    )

    def compute_metrics(frame: pd.DataFrame) -> dict[str, Any]:
        y_true = frame["label"].to_numpy(dtype=int)
        y_prob = frame["fake_probability"].to_numpy(dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        result = {
            "num_samples": int(len(frame)),
            "num_real": int((y_true == 0).sum()),
            "num_fake": int((y_true == 1).sum()),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "real_recall": float(tn / (tn + fp)) if tn + fp else None,
            "fake_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "fake_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "fake_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "confusion_matrix": matrix.tolist(),
            "mean_fake_probability": float(y_prob.mean()),
        }
        return result

    shared_real = predictions[predictions["label"] == 0]
    generator_rows = []
    for generator in expected_generators:
        generator_fake = predictions[
            (predictions["label"] == 1) & (predictions["generator"] == generator)
        ]
        cohort = pd.concat([shared_real, generator_fake], ignore_index=True)
        row = compute_metrics(cohort)
        row.update(
            {
                "generator": generator,
                "metric_cohort": "generator_fake_plus_shared_real",
            }
        )
        generator_rows.append(row)
    generator_metrics = pd.DataFrame(generator_rows)
    generator_metrics.to_csv(run_dir / "metrics" / "commfor_generator_metrics.csv", index=False)

    overall_unique = compute_metrics(predictions)
    save_json(run_dir / "metrics" / "overall_unique_sample_metrics.json", overall_unique)
    macro_columns = [
        "accuracy",
        "balanced_accuracy",
        "real_recall",
        "fake_recall",
        "fake_precision",
        "fake_f1",
        "roc_auc",
        "average_precision",
    ]
    macro = {
        f"macro_generator_{column}": float(generator_metrics[column].mean())
        for column in macro_columns
    }
    macro.update(
        {
            "worst_generator_balanced_accuracy": float(generator_metrics["balanced_accuracy"].min()),
            "best_generator_balanced_accuracy": float(generator_metrics["balanced_accuracy"].max()),
            "num_generators": int(len(generator_metrics)),
            "manifest_samples": expected_count,
            "manifest_real": expected_real,
            "manifest_fake": expected_fake,
            "matched_samples": int(len(predictions)),
            "commfor_records_scanned": int(scanned),
            "target_generators": expected_generators,
            "checkpoint_source_run_id": CHECKPOINT_SOURCE_RUN_ID,
            "checkpoint_sha256": checkpoint_sha256,
            "aide_source_run_id": AIDE_SOURCE_RUN_ID,
            "metric_cohort": "generator_fake_plus_shared_real",
            "inference_id": inference_id,
        }
    )
    save_json(run_dir / "metrics" / "macro_summary.json", macro)
    save_json(run_dir / "config.json", config)
    print("NPR-ResNet18 metrics on the exact AIDE CommFor cohort:")
    print(generator_metrics.to_string(index=False))
    print(json.dumps(macro, ensure_ascii=False, indent=2))

    output_volume.commit()
    hf_cache_volume.commit()
    del model, checkpoint_payload, state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return macro


@app.local_entrypoint()
def main(
    inference_id: str | None = None,
    batch_size: int = 32,
    max_scan_records: int = 500000,
):
    inference_id = inference_id or time.strftime("%Y%m%d_%H%M%S")
    overrides = {
        "batch_size": batch_size,
        "max_scan_records": max_scan_records,
    }
    print("Inference ID:", inference_id)
    print("Checkpoint source run:", CHECKPOINT_SOURCE_RUN_ID)
    print("Exact AIDE cohort source run:", AIDE_SOURCE_RUN_ID)
    summary = test_npr_on_aide_commfor.remote(inference_id, overrides)
    print("Inference completed:", summary)
    print("Modal output root:", f"{REMOTE_OUTPUT_ROOT}/{inference_id}")
