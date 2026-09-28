"""Inference-only Full AIDE evaluation on the established CommFor cohort.

This runner loads the best Full AIDE checkpoint trained on Tiny-GenImage and
evaluates the exact 1,000-image CommunityForensics cohort previously used by
the AIDE-forensic and NPR-ResNet18 experiments:

* 100 shared real images;
* 100 fake images from each of nine generators.

The checkpoint and manifest are both protected by SHA-256 checks.  CommFor is
strictly test-only: this file performs no training, threshold tuning, or model
selection.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image, ImageFile
from tqdm.auto import tqdm

import train_aide_full_tiny_commfor_kaggle as aide


ImageFile.LOAD_TRUNCATED_IMAGES = True

EXPECTED_CHECKPOINT_SHA256 = (
    "62848895b44255d6a0567754c503a05807ed0a29b8046f3c6956a049844e47c3"
)
EXPECTED_MANIFEST_SHA256 = (
    "20b04a56d2709d091d51fe585621686b2cd792a373a20ec5068fc6a5be28b9cd"
)
AIDE_FORENSIC_COHORT_SOURCE_RUN_ID = "20260927_001615"
EXPECTED_GENERATORS = (
    "Hourglass",
    "MidjourneyV5_2",
    "Firefly_Image3",
    "Firefly_Image2",
    "MidjourneyV6_1",
    "DFGAN",
    "GALIP",
    "kandinsky_2_2",
    "kvikontent_midjourney_v6",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "input_root": "/kaggle/input",
    "output_root": "/kaggle/working/aide_full_commfor_exact_eval",
    "checkpoint_path": None,
    "manifest_path": None,
    "semantic_checkpoint": None,
    "semantic_model_name": "convnext_xxlarge",
    "semantic_pretrained_tag": "laion2b_s34b_b82k_augreg_soup",
    "semantic_half_precision": True,
    "commfor_dataset_name": "OwensLab/CommunityForensics-Eval",
    "commfor_split": "CompEval",
    "commfor_streaming": True,
    "commfor_shuffle_seed": 43,
    "commfor_shuffle_buffer_size": 1000,
    "max_scan_records": 500000,
    "random_seed": 42,
    "threshold": 0.5,
    "dct_window_size": 32,
    "dct_stride": 16,
    "dct_grade_bands": 6,
    "image_size": 256,
    "resume": True,
    "save_every": 25,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def clean_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def discover_exact_file(
    explicit: str | None,
    input_root: str,
    filename: str,
    expected_sha256: str,
) -> Path:
    if explicit:
        candidates = [Path(explicit)]
    else:
        candidates = sorted(Path(input_root).rglob(filename))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find {filename}. Attach it as a Kaggle Dataset or set its explicit path."
        )
    matches: list[Path] = []
    observed: list[tuple[str, str]] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        digest = sha256_file(candidate)
        observed.append((str(candidate), digest))
        if digest.lower() == expected_sha256.lower():
            matches.append(candidate)
    if not matches:
        raise RuntimeError(
            f"No {filename} matches required SHA-256 {expected_sha256}. Observed: {observed}"
        )
    selected = matches[0]
    if len(matches) > 1:
        print(f"Multiple byte-identical {filename} files found; using {selected}")
    return selected


def manifest_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(int(row["label"])),
        clean_value(row["generator"]),
        clean_value(row["image_name"]),
        clean_value(row["real_source"]),
        clean_value(row["architecture"]),
    )


def generator_from_record(record: dict[str, Any]) -> str:
    return str(record.get("model_name") or record.get("architecture") or "unknown")


def record_key(record: dict[str, Any]) -> tuple[str, ...]:
    label = int(record.get("label"))
    generator = "shared_real" if label == 0 else generator_from_record(record)
    return (
        str(label),
        generator,
        clean_value(record.get("image_name")),
        clean_value(record.get("real_source")),
        clean_value(record.get("architecture")),
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


def validate_manifest(manifest: pd.DataFrame) -> list[str]:
    required = {
        "selection_group",
        "sample_index",
        "label",
        "generator",
        "image_name",
        "real_source",
        "architecture",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"CommFor manifest is missing columns: {missing}")
    if len(manifest) != 1000:
        raise ValueError(f"Expected 1,000 manifest rows, got {len(manifest)}")
    if int((manifest["label"] == 0).sum()) != 100:
        raise ValueError("Expected exactly 100 shared real samples")
    if int((manifest["label"] == 1).sum()) != 900:
        raise ValueError("Expected exactly 900 fake samples")
    generators = (
        manifest.loc[manifest["label"] == 1, "generator"].drop_duplicates().tolist()
    )
    if generators != list(EXPECTED_GENERATORS):
        raise ValueError(
            f"Generator order/content differs from the established cohort: {generators}"
        )
    counts = manifest.loc[manifest["label"] == 1].groupby("generator").size()
    bad_counts = {name: int(counts.get(name, 0)) for name in generators if counts.get(name, 0) != 100}
    if bad_counts:
        raise ValueError(f"Every generator must contain 100 fake samples: {bad_counts}")
    keys = [manifest_key(row) for row in manifest.to_dict("records")]
    if len(set(keys)) != len(keys):
        raise ValueError("Manifest composite keys are not unique")
    return generators


def prepare_aide_input(
    image: Image.Image,
    selector: aide.AIDEDCTPatchSelector,
    image_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    patches, selection = selector(tensor)
    patches = F.interpolate(
        patches,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    raw = F.interpolate(
        tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    patches = (patches - aide.IMAGENET_MEAN) / aide.IMAGENET_STD
    raw = (raw - aide.IMAGENET_MEAN) / aide.IMAGENET_STD
    return torch.cat([patches, raw], dim=0), selection


def scalar(value: Any, *, integer: bool = False) -> int | float:
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    return int(value) if integer else float(value)


def compute_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    result = aide.compute_metrics(frame, threshold=threshold)
    result["mean_fake_probability"] = float(frame["fake_probability"].mean())
    return result


def load_partial_predictions(
    partial_path: Path,
    manifest: pd.DataFrame,
    resume: bool,
) -> pd.DataFrame:
    if not resume or not partial_path.is_file():
        return pd.DataFrame()
    partial = pd.read_csv(partial_path)
    if partial.empty:
        return partial
    required = {"manifest_order", "label", "generator", "image_name", "fake_probability"}
    missing = sorted(required.difference(partial.columns))
    if missing:
        raise ValueError(f"Partial predictions are missing columns: {missing}")
    if partial["manifest_order"].duplicated().any():
        raise ValueError("Partial predictions contain duplicate manifest_order values")
    manifest_by_order = manifest.set_index("manifest_order")
    for row in partial.to_dict("records"):
        order = int(row["manifest_order"])
        if order not in manifest_by_order.index:
            raise ValueError(f"Partial prediction has invalid manifest_order={order}")
        expected = manifest_by_order.loc[order]
        for column in ("label", "generator", "image_name"):
            if clean_value(row[column]) != clean_value(expected[column]):
                raise ValueError(
                    f"Partial prediction mismatch at order={order}, column={column}"
                )
    print(f"Resuming with {len(partial)} already completed predictions")
    return partial.sort_values("manifest_order").reset_index(drop=True)


def run_inference(config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Enable a Kaggle GPU accelerator.")

    aide.seed_everything(int(config["random_seed"]))
    device = torch.device("cuda")
    run_dir = Path(config["output_root"])
    for subdirectory in ("dataset", "metrics", "predictions", "provenance"):
        (run_dir / subdirectory).mkdir(parents=True, exist_ok=True)

    checkpoint_path = discover_exact_file(
        config.get("checkpoint_path"),
        str(config["input_root"]),
        "model_trainable.pt",
        EXPECTED_CHECKPOINT_SHA256,
    )
    manifest_path = discover_exact_file(
        config.get("manifest_path"),
        str(config["input_root"]),
        "selected_samples.csv",
        EXPECTED_MANIFEST_SHA256,
    )
    manifest = pd.read_csv(manifest_path).copy()
    manifest.insert(0, "manifest_order", np.arange(len(manifest), dtype=int))
    generators = validate_manifest(manifest)
    manifest.to_csv(run_dir / "dataset" / "commfor_manifest_used.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_trainable_state_dict" not in checkpoint:
        raise TypeError("Expected a Full AIDE trainable-state checkpoint dictionary")
    if checkpoint.get("architecture") != (
        "AIDE full hybrid: OpenCLIP ConvNeXt-XXLarge + DCT/SRM dual ResNet50"
    ):
        raise ValueError(f"Unexpected checkpoint architecture: {checkpoint.get('architecture')}")
    checkpoint_config = checkpoint.get("config", {})
    if int(checkpoint_config.get("image_size", 256)) != int(config["image_size"]):
        raise ValueError("Inference image_size does not match the training checkpoint")

    model_config = dict(aide.DEFAULT_CONFIG)
    model_config.update(
        {
            "input_root": config["input_root"],
            "random_seed": config["random_seed"],
            "dct_window_size": config["dct_window_size"],
            "dct_stride": config["dct_stride"],
            "dct_grade_bands": config["dct_grade_bands"],
            "image_size": config["image_size"],
            "semantic_model_name": config["semantic_model_name"],
            "semantic_pretrained_tag": config["semantic_pretrained_tag"],
            "semantic_checkpoint": config.get("semantic_checkpoint"),
            "semantic_half_precision": config["semantic_half_precision"],
            # The trained checkpoint replaces every ResNet parameter. Avoid an
            # unnecessary torchvision ImageNet-weight download at inference.
            "imagenet_resnet_init": False,
            "resnet_checkpoint": None,
        }
    )
    print("Building Full AIDE and restoring the frozen OpenCLIP trunk...")
    model, semantic_provenance = aide.build_model(model_config, device)
    aide.load_trainable_model_state(model, checkpoint["model_trainable_state_dict"])
    model.eval()
    checkpoint_metadata = {
        "path": str(checkpoint_path),
        "sha256": EXPECTED_CHECKPOINT_SHA256,
        "epoch": int(checkpoint.get("epoch", -1)),
        "optimizer_step": int(checkpoint.get("optimizer_step", -1)),
        "best_val_balanced_accuracy": float(
            checkpoint.get("best_val_balanced_accuracy", float("nan"))
        ),
        "frozen_semantic_backbone_included": bool(
            checkpoint.get("frozen_semantic_backbone_included", False)
        ),
        "semantic_backbone": semantic_provenance,
    }
    save_json(run_dir / "provenance" / "checkpoint.json", checkpoint_metadata)
    save_json(
        run_dir / "provenance" / "cohort.json",
        {
            "manifest_path": str(manifest_path),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "source_run_id": AIDE_FORENSIC_COHORT_SOURCE_RUN_ID,
            "num_samples": 1000,
            "num_real": 100,
            "num_fake": 900,
            "generators": generators,
            "metric_cohort": "generator_fake_plus_shared_real",
        },
    )
    del checkpoint
    gc.collect()

    selector = aide.AIDEDCTPatchSelector(
        window_size=int(config["dct_window_size"]),
        stride=int(config["dct_stride"]),
        grade_bands=int(config["dct_grade_bands"]),
    )
    manifest_records = manifest.to_dict("records")
    manifest_by_key = {manifest_key(row): row for row in manifest_records}
    partial_path = run_dir / "predictions" / "commfor_predictions.partial.csv"
    partial = load_partial_predictions(
        partial_path,
        manifest,
        resume=bool(config["resume"]),
    )
    completed_orders = (
        set(partial["manifest_order"].astype(int).tolist()) if not partial.empty else set()
    )
    remaining_keys = {
        manifest_key(row)
        for row in manifest_records
        if int(row["manifest_order"]) not in completed_orders
    }
    prediction_rows = partial.to_dict("records") if not partial.empty else []

    scanned = 0
    if remaining_keys:
        dataset = load_dataset(
            str(config["commfor_dataset_name"]),
            split=str(config["commfor_split"]),
            streaming=bool(config["commfor_streaming"]),
        )
        dataset = dataset.shuffle(
            seed=int(config["commfor_shuffle_seed"]),
            buffer_size=int(config["commfor_shuffle_buffer_size"]),
        )
        progress = tqdm(
            total=len(manifest),
            initial=len(prediction_rows),
            desc="Full AIDE on exact CommFor cohort",
        )
        with torch.inference_mode():
            for record in dataset:
                scanned += 1
                key = record_key(record)
                if key not in remaining_keys:
                    if scanned >= int(config["max_scan_records"]):
                        break
                    continue
                manifest_row = manifest_by_key[key]
                image = image_from_record(record)
                model_input, selection = prepare_aide_input(
                    image,
                    selector,
                    image_size=int(config["image_size"]),
                )
                model_input = model_input.unsqueeze(0).to(device, non_blocking=True)
                amp = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with amp:
                    logits = model(model_input)
                probability = float(torch.softmax(logits.float(), dim=-1)[0, 1].cpu())
                order = int(manifest_row["manifest_order"])
                prediction_rows.append(
                    {
                        "manifest_order": order,
                        "selection_group": clean_value(manifest_row["selection_group"]),
                        "sample_index": int(manifest_row["sample_index"]),
                        "sample_id": f"commfor:{order:06d}",
                        "label": int(manifest_row["label"]),
                        "label_name": "fake" if int(manifest_row["label"]) else "real",
                        "generator": clean_value(manifest_row["generator"]),
                        "image_name": clean_value(record.get("image_name")),
                        "model_name": clean_value(record.get("model_name")),
                        "architecture": clean_value(record.get("architecture")),
                        "real_source": clean_value(record.get("real_source")),
                        "subset": clean_value(record.get("subset")),
                        "native_width": int(image.width),
                        "native_height": int(image.height),
                        "fake_probability": probability,
                        "predicted_label": int(probability >= float(config["threshold"])),
                        "num_candidates": scalar(selection["num_candidates"], integer=True),
                        "low_1_score": scalar(selection["low_1_score"]),
                        "high_1_score": scalar(selection["high_1_score"]),
                        "low_2_score": scalar(selection["low_2_score"]),
                        "high_2_score": scalar(selection["high_2_score"]),
                    }
                )
                remaining_keys.remove(key)
                progress.update(1)
                if len(prediction_rows) % int(config["save_every"]) == 0:
                    pd.DataFrame(prediction_rows).sort_values("manifest_order").to_csv(
                        partial_path, index=False
                    )
                del image, model_input, logits
                if not remaining_keys or scanned >= int(config["max_scan_records"]):
                    break
        progress.close()

    predictions = pd.DataFrame(prediction_rows).sort_values("manifest_order").reset_index(drop=True)
    predictions.to_csv(partial_path, index=False)
    if remaining_keys:
        missing = pd.DataFrame([manifest_by_key[key] for key in sorted(remaining_keys)])
        missing.to_csv(run_dir / "dataset" / "missing_manifest_samples.csv", index=False)
        raise RuntimeError(
            f"Matched {len(predictions)}/1000 exact CommFor samples after scanning {scanned} records. "
            "Partial predictions and missing rows were saved."
        )
    if len(predictions) != 1000 or predictions["manifest_order"].duplicated().any():
        raise AssertionError("Predictions do not form the required unique 1,000-sample cohort")

    final_predictions_path = run_dir / "predictions" / "commfor_predictions.csv"
    predictions.to_csv(final_predictions_path, index=False)
    if partial_path.is_file():
        partial_path.unlink()

    threshold = float(config["threshold"])
    overall = compute_metrics(predictions, threshold)
    shared_real = predictions[predictions["label"] == 0]
    generator_rows: list[dict[str, Any]] = []
    for generator in generators:
        generator_fake = predictions[
            (predictions["label"] == 1) & (predictions["generator"] == generator)
        ]
        if len(generator_fake) != 100 or len(shared_real) != 100:
            raise AssertionError(f"Invalid metric cohort for {generator}")
        cohort = pd.concat([shared_real, generator_fake], ignore_index=True)
        generator_rows.append(
            {
                "generator": generator,
                "metric_cohort": "generator_fake_plus_shared_real",
                **compute_metrics(cohort, threshold),
            }
        )
    generator_metrics = pd.DataFrame(generator_rows)
    generator_metrics.to_csv(
        run_dir / "metrics" / "commfor_generator_metrics.csv", index=False
    )
    macro_columns = (
        "accuracy",
        "balanced_accuracy",
        "real_recall",
        "fake_recall",
        "fake_precision",
        "fake_f1",
        "roc_auc",
        "average_precision",
    )
    macro = {
        f"macro_generator_{column}": float(generator_metrics[column].mean())
        for column in macro_columns
    }
    macro.update(
        {
            "worst_generator_balanced_accuracy": float(
                generator_metrics["balanced_accuracy"].min()
            ),
            "best_generator_balanced_accuracy": float(
                generator_metrics["balanced_accuracy"].max()
            ),
            "num_generators": len(generators),
            "manifest_samples": len(manifest),
            "manifest_real": int((manifest["label"] == 0).sum()),
            "manifest_fake": int((manifest["label"] == 1).sum()),
            "matched_samples": len(predictions),
            "commfor_records_scanned_this_run": scanned,
            "target_generators": generators,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "aide_forensic_cohort_source_run_id": AIDE_FORENSIC_COHORT_SOURCE_RUN_ID,
            "metric_cohort": "generator_fake_plus_shared_real",
        }
    )
    save_json(run_dir / "metrics" / "commfor_overall.json", overall)
    save_json(run_dir / "metrics" / "commfor_macro_summary.json", macro)
    save_json(run_dir / "config.json", config)
    output = {
        "experiment": {
            "architecture": "Full AIDE hybrid",
            "training_dataset": "Tiny-GenImage combined",
            "evaluation_dataset": "CommunityForensics-Eval CompEval",
            "test_only": True,
            "checkpoint": checkpoint_metadata,
            "cohort_source_run_id": AIDE_FORENSIC_COHORT_SOURCE_RUN_ID,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        },
        "commfor": {"overall": overall, "macro": macro},
    }
    save_json(run_dir / "output.json", output)
    archive_path = Path(
        shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    )
    output["archive"] = str(archive_path)
    save_json(run_dir / "output.json", output)

    print("\nCommFor overall (100 real + 900 fake unique samples):")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\nCommFor per generator (100 shared real + 100 generator fake):")
    print(generator_metrics.to_string(index=False))
    print("\nMacro summary:")
    print(json.dumps(macro, ensure_ascii=False, indent=2))
    print("\nResults ZIP:", archive_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output


if __name__ == "__main__":
    run_inference()
