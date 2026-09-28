"""Evaluate CLIP, NPR-ResNet18, and AIDE on one Tiny-GenImage folder cohort.

This Kaggle-only runner reuses the exact detector runtime implemented in
``test_three_models_commfor_unseen_kaggle.py`` and adds deterministic discovery,
indexing, per-generator balancing, evaluation, and output generation for a
folder-style Tiny-GenImage Kaggle Dataset.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "validation": ("val", "valid", "validation", "test"),
    "val": ("val", "valid", "validation", "test"),
    "test": ("test", "val", "valid", "validation"),
}
LABEL_DIR_TO_ID = {
    "nature": 0,
    "real": 0,
    "0_real": 0,
    "0-real": 0,
    "0": 0,
    "ai": 1,
    "fake": 1,
    "1_fake": 1,
    "1-fake": 1,
    "1": 1,
}
GENERATOR_HINTS = (
    "adm",
    "biggan",
    "glide",
    "midjourney",
    "sd14",
    "sd15",
    "sdv4",
    "sdv5",
    "stable",
    "vqdm",
    "wukong",
)


def load_common_runtime():
    runtime_path = Path(__file__).with_name("test_three_models_commfor_unseen_kaggle.py")
    if not runtime_path.is_file():
        candidates = sorted(
            Path("/kaggle/input").rglob("test_three_models_commfor_unseen_kaggle.py")
        )
        if len(candidates) != 1:
            raise FileNotFoundError(
                "Upload exactly one test_three_models_commfor_unseen_kaggle.py next to this runner."
            )
        runtime_path = candidates[0]
    spec = importlib.util.spec_from_file_location("three_detector_common_runtime", runtime_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMMON = load_common_runtime()


def normalize_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "").replace(".", "")


def canonical_generator(folder_name: str) -> str:
    normalized = normalize_name(folder_name)
    if "biggan" in normalized:
        return "BigGAN"
    if "vqdm" in normalized:
        return "VQDM"
    if "midjourney" in normalized:
        return "Midjourney"
    if "wukong" in normalized:
        return "Wukong"
    if "glide" in normalized:
        return "GLIDE"
    if "adm" in normalized or "guided" in normalized:
        return "ADM"
    if any(token in normalized for token in ("sd15", "sdv5", "stablediffusion15")):
        return "SD15"
    if any(token in normalized for token in ("sd14", "sdv4", "stablediffusion14")):
        return "SD14"
    return folder_name


def find_label_dirs(split_dir: Path) -> dict[int, Path]:
    if not split_dir.is_dir():
        return {}
    return {
        LABEL_DIR_TO_ID[child.name.lower()]: child
        for child in split_dir.iterdir()
        if child.is_dir() and child.name.lower() in LABEL_DIR_TO_ID
    }


def resolve_split_dir(generator_dir: Path, requested: str) -> Path | None:
    for alias in SPLIT_ALIASES.get(requested, (requested,)):
        candidate = generator_dir / alias
        if candidate.is_dir() and find_label_dirs(candidate):
            return candidate
    return None


def looks_like_generator_dir(path: Path, validation_split: str) -> bool:
    if not path.is_dir() or resolve_split_dir(path, validation_split) is None:
        return False
    normalized = normalize_name(path.name)
    return any(hint in normalized for hint in GENERATOR_HINTS) or bool(
        find_label_dirs(resolve_split_dir(path, validation_split))
    )


def find_tiny_root(
    input_root: Path, explicit_root: str | Path | None, validation_split: str
) -> Path:
    if explicit_root:
        root = Path(explicit_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Tiny-GenImage root does not exist: {root}")
        return root
    candidates = [input_root, *[path for path in input_root.rglob("*") if path.is_dir()]]
    scored: list[tuple[int, Path]] = []
    for candidate in candidates:
        try:
            children = [child for child in candidate.iterdir() if child.is_dir()]
        except OSError:
            continue
        count = sum(looks_like_generator_dir(child, validation_split) for child in children)
        if count:
            scored.append((count, candidate))
    if not scored:
        raise FileNotFoundError(
            "Cannot find Tiny-GenImage. Expected generator/val/{ai,nature} or "
            "generator/validation/{fake,real}."
        )
    scored.sort(key=lambda item: (-item[0], len(item[1].parts), str(item[1])))
    return scored[0][1]


def iter_images(directory: Path):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_validation_index(
    dataset_root: Path,
    validation_split: str,
    seed: int,
    balance_per_generator: bool,
    max_per_class_per_generator: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    generator_dirs = sorted(
        [
            child
            for child in dataset_root.iterdir()
            if child.is_dir() and looks_like_generator_dir(child, validation_split)
        ],
        key=lambda path: path.name.lower(),
    )
    if not generator_dirs:
        raise FileNotFoundError(f"No Tiny generator folders found under {dataset_root}")
    for generator_dir in generator_dirs:
        split_dir = resolve_split_dir(generator_dir, validation_split)
        generator = canonical_generator(generator_dir.name)
        for label, label_dir in find_label_dirs(split_dir).items():
            for image_path in iter_images(label_dir):
                relative = image_path.relative_to(dataset_root).as_posix()
                records.append(
                    {
                        "sample_id": f"Tiny-GenImage:{relative}",
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "cached_image_path": str(image_path),
                        "label": int(label),
                        "label_name": "fake" if label == 1 else "real",
                        "generator": generator,
                        "generator_family": generator,
                        "generator_source_folder": generator_dir.name,
                        "split": "validation",
                        "dataset_source": "Tiny-GenImage-Kaggle",
                    }
                )
    full_index = pd.DataFrame(records)
    if full_index.empty:
        raise FileNotFoundError(f"No validation images found below {dataset_root}")

    selected = []
    for generator_index, generator in enumerate(sorted(full_index["generator"].unique())):
        group = full_index[full_index["generator"] == generator]
        real = group[group["label"] == 0]
        fake = group[group["label"] == 1]
        if real.empty or fake.empty:
            raise ValueError(
                f"Generator {generator} must contain both real and fake validation images: "
                f"real={len(real)}, fake={len(fake)}"
            )
        if balance_per_generator:
            per_class = min(len(real), len(fake))
        else:
            per_class = max(len(real), len(fake))
        if max_per_class_per_generator is not None:
            per_class = min(per_class, int(max_per_class_per_generator))
        if balance_per_generator:
            real = real.sample(n=per_class, random_state=seed + generator_index, replace=False)
            fake = fake.sample(n=per_class, random_state=seed + 100 + generator_index, replace=False)
        elif max_per_class_per_generator is not None:
            real = real.sample(n=min(len(real), per_class), random_state=seed + generator_index)
            fake = fake.sample(n=min(len(fake), per_class), random_state=seed + 100 + generator_index)
        selected.append(pd.concat([real, fake], ignore_index=True))
    manifest = pd.concat(selected, ignore_index=True)
    manifest = manifest.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    manifest.insert(0, "manifest_order", np.arange(len(manifest), dtype=int))
    return manifest, full_index


def compute_model_metrics(
    model_name: str, predictions: pd.DataFrame, metrics_dir: Path
) -> dict[str, Any]:
    overall = {"model": model_name, **COMMON.compute_metrics(predictions)}
    COMMON.save_json(metrics_dir / f"{model_name}_overall.json", overall)
    generator_rows = []
    for generator, group in predictions.groupby("generator", sort=True):
        generator_rows.append(
            {
                "model": model_name,
                "generator": generator,
                "metric_cohort": "tiny_validation_generator_own_real_and_fake",
                **COMMON.compute_metrics(group),
            }
        )
    generator_metrics = pd.DataFrame(generator_rows)
    generator_metrics.to_csv(
        metrics_dir / f"{model_name}_generator_metrics.csv", index=False
    )
    metric_columns = [
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
        "model": model_name,
        **{
            f"macro_generator_{column}": float(generator_metrics[column].mean())
            for column in metric_columns
        },
        "worst_generator_balanced_accuracy": float(
            generator_metrics["balanced_accuracy"].min()
        ),
        "best_generator_balanced_accuracy": float(
            generator_metrics["balanced_accuracy"].max()
        ),
        "num_generators": int(len(generator_metrics)),
    }
    COMMON.save_json(metrics_dir / f"{model_name}_macro_summary.json", macro)
    return {"overall": overall, "macro": macro, "generator_metrics": generator_metrics}


def save_formatted_output(
    output_root: Path,
    manifest: pd.DataFrame,
    checkpoint_info: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "experiment": {
            "dataset": "Tiny-GenImage",
            "split": "validation",
            "manifest_samples": int(len(manifest)),
            "manifest_real": int((manifest["label"] == 0).sum()),
            "manifest_fake": int((manifest["label"] == 1).sum()),
            "generators": sorted(manifest["generator"].unique().tolist()),
            "manifest_sha256": COMMON.sha256_file(output_root / "dataset/test_manifest.csv"),
            "checkpoints": checkpoint_info,
        },
        "models": {},
    }
    for model_name, result in results.items():
        rows = json.loads(result["generator_metrics"].to_json(orient="records"))
        for row in rows:
            if isinstance(row.get("confusion_matrix"), str):
                row["confusion_matrix"] = json.loads(row["confusion_matrix"])
        payload["models"][model_name] = {
            "overall": result["overall"],
            "macro": result["macro"],
            "generators": rows,
        }
    COMMON.save_json(output_root / "output.json", payload)
    return payload


def run_evaluation(user_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "input_root": "/kaggle/input",
        "dataset_root": None,
        "output_root": "/kaggle/working/tiny_genimage_three_models",
        "validation_split": "validation",
        "selection_seed": 42,
        "balance_per_generator": True,
        "max_per_class_per_generator": None,
        "clip_checkpoint": None,
        "npr_checkpoint": None,
        "aide_checkpoint": None,
        "clip_backbone_path": None,
        "clip_batch_size": 32,
        "npr_batch_size": 32,
        "resume": True,
    }
    if user_config:
        config.update(user_config)
    random.seed(int(config["selection_seed"]))
    np.random.seed(int(config["selection_seed"]))
    torch.manual_seed(int(config["selection_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["selection_seed"]))

    input_root = Path(config["input_root"])
    output_root = Path(config["output_root"])
    dataset_dir = output_root / "dataset"
    metrics_dir = output_root / "metrics"
    predictions_dir = output_root / "predictions"
    provenance_dir = output_root / "provenance"
    for directory in (dataset_dir, metrics_dir, predictions_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths = COMMON.resolve_input_paths(config)
    dataset_root = find_tiny_root(
        input_root, config.get("dataset_root"), str(config["validation_split"])
    )
    manifest, full_index = build_validation_index(
        dataset_root=dataset_root,
        validation_split=str(config["validation_split"]),
        seed=int(config["selection_seed"]),
        balance_per_generator=bool(config["balance_per_generator"]),
        max_per_class_per_generator=config.get("max_per_class_per_generator"),
    )
    manifest.to_csv(dataset_dir / "test_manifest.csv", index=False)
    (
        full_index.groupby(["generator", "label_name"])
        .size()
        .reset_index(name="available_images")
        .to_csv(dataset_dir / "available_counts.csv", index=False)
    )
    (
        manifest.groupby(["generator", "label_name"])
        .size()
        .reset_index(name="selected_images")
        .to_csv(dataset_dir / "selected_counts.csv", index=False)
    )

    checkpoint_info = {
        "clip_linear_probe": {
            "path": str(paths["clip_checkpoint"]),
            "sha256": COMMON.sha256_file(paths["clip_checkpoint"]),
        },
        "npr_resnet18": {
            "path": str(paths["npr_checkpoint"]),
            "sha256": COMMON.sha256_file(paths["npr_checkpoint"]),
        },
        "aide_forensic_resnet50": {
            "path": str(paths["aide_checkpoint"]),
            "sha256": COMMON.sha256_file(paths["aide_checkpoint"]),
        },
    }
    COMMON.save_json(provenance_dir / "checkpoints.json", checkpoint_info)
    serializable_config = {
        key: str(value) if isinstance(value, Path) else value for key, value in config.items()
    }
    serializable_config["detected_dataset_root"] = str(dataset_root)
    COMMON.save_json(output_root / "config.json", serializable_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Tiny root: {dataset_root}")
    print(f"Device: {device}; test samples: {len(manifest)}")
    print(
        manifest.groupby(["generator", "label_name"]).size().unstack(fill_value=0)
    )
    clip_pretrained = (
        str(paths["clip_backbone_path"])
        if paths["clip_backbone_path"] is not None
        else "openai"
    )
    predictors = {
        "clip_linear_probe": lambda: COMMON.predict_clip(
            manifest,
            Path("/"),
            paths["clip_checkpoint"],
            device,
            int(config["clip_batch_size"]),
            clip_pretrained,
        ),
        "npr_resnet18": lambda: COMMON.predict_npr(
            manifest,
            Path("/"),
            paths["npr_checkpoint"],
            device,
            int(config["npr_batch_size"]),
        ),
        "aide_forensic_resnet50": lambda: COMMON.predict_aide(
            manifest, Path("/"), paths["aide_checkpoint"], device
        ),
    }

    predictions_by_model: dict[str, pd.DataFrame] = {}
    results: dict[str, dict[str, Any]] = {}
    expected_orders = manifest["manifest_order"].to_numpy(dtype=int)
    for model_name, predictor in predictors.items():
        prediction_path = predictions_dir / f"{model_name}_predictions.csv"
        predictions = None
        if bool(config["resume"]) and prediction_path.is_file():
            candidate = pd.read_csv(prediction_path).sort_values("manifest_order").reset_index(drop=True)
            if len(candidate) == len(manifest) and np.array_equal(
                candidate["manifest_order"].to_numpy(dtype=int), expected_orders
            ):
                predictions = candidate
                print(f"Resuming completed {model_name} predictions.")
        if predictions is None:
            predictions = predictor()
            predictions.to_csv(prediction_path, index=False)
        predictions_by_model[model_name] = predictions
        results[model_name] = compute_model_metrics(model_name, predictions, metrics_dir)

    long_predictions = pd.concat(predictions_by_model.values(), ignore_index=True)
    long_predictions.to_csv(predictions_dir / "all_models_predictions.csv", index=False)
    wide_predictions = manifest.copy()
    for model_name, predictions in predictions_by_model.items():
        wide_predictions[f"{model_name}_fake_probability"] = predictions[
            "fake_probability"
        ].to_numpy()
        wide_predictions[f"{model_name}_predicted_label"] = predictions[
            "predicted_label"
        ].to_numpy(dtype=int)
    wide_predictions.to_csv(
        predictions_dir / "all_models_predictions_wide.csv", index=False
    )

    all_generator = pd.concat(
        [result["generator_metrics"] for result in results.values()], ignore_index=True
    )
    all_generator.to_csv(metrics_dir / "all_models_generator_metrics.csv", index=False)
    comparison = pd.DataFrame(
        [
            {
                "model": model_name,
                **{
                    f"overall_{key}": value
                    for key, value in result["overall"].items()
                    if not isinstance(value, list)
                },
                **result["macro"],
            }
            for model_name, result in results.items()
        ]
    )
    comparison.to_csv(metrics_dir / "all_models_summary.csv", index=False)
    payload = save_formatted_output(output_root, manifest, checkpoint_info, results)
    print(
        comparison[
            [
                "model",
                "overall_balanced_accuracy",
                "macro_generator_balanced_accuracy",
                "macro_generator_roc_auc",
                "worst_generator_balanced_accuracy",
            ]
        ].to_string(index=False)
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload

