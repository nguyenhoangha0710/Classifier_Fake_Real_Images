"""Evaluate three Tiny-GenImage checkpoints under controlled image-codec shifts.

Models
------
* CLIP ViT-B/32 with the trained linear probe.
* NPR-ResNet18 trained from scratch.
* Original Full AIDE (OpenCLIP semantic branch + DCT/SRM forensic branch).

The runtime audits the native Tiny-GenImage validation files, creates four
requested transformations in memory, adds raw reference cohorts, evaluates all
models with their original preprocessing, and writes resumable predictions,
metrics, plots, and provenance to ``/kaggle/working``.

No transformed image overwrites the source dataset. JPEG/PNG round-trips happen
before each model's original resize/crop/normalization pipeline.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_DIRECTORIES = (("nature", 0, "real"), ("ai", 1, "fake"))
REQUESTED_CASES = (
    "png_roundtrip",
    "fake_jpeg96",
    "all_jpeg96",
    "controlled_jpeg96",
)
REFERENCE_CASES = ("raw", "raw_on_controlled_cohort")
ALL_CASES = REFERENCE_CASES + REQUESTED_CASES
EXPECTED_ARCHITECTURE = (
    "AIDE full hybrid: OpenCLIP ConvNeXt-XXLarge + DCT/SRM dual ResNet50"
)

EXPECTED_CHECKPOINT_SHA256 = {
    "clip_linear_probe": "788461421cb585bbf489aabc76f3537058cd7753eccf9432d3a792622f738057",
    "npr_resnet18": "328f9f431d378f9528c86966796e9f3ac604a5ce2bf3d307028062706c1537f0",
    "aide_original_full": "62848895b44255d6a0567754c503a05807ed0a29b8046f3c6956a049844e47c3",
}
CHECKPOINT_FILENAMES = {
    "clip_linear_probe": "clip_linear_head.pt",
    "npr_resnet18": "npr_resnet18_from_scratch.pt",
    "aide_original_full": "aide_original_full_trainable.pt",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "input_root": "/kaggle/input",
    "dataset_root": None,
    "output_root": "/kaggle/working/tiny_jpeg_bias_three_models_results",
    "validation_split": "val",
    "selection_seed": 42,
    "smoke": False,
    "max_per_class_per_generator": None,
    "models": ["clip_linear_probe", "npr_resnet18", "aide_original_full"],
    "cases": list(ALL_CASES),
    "threshold": 0.5,
    "jpeg_quality": 96,
    "jpeg_subsampling": 0,
    "jpeg_optimize": False,
    "jpeg_progressive": False,
    "controlled_real_quality_min": 94,
    "controlled_real_quality_max": 98,
    "controlled_balance_across_generators": True,
    "clip_batch_size": 32,
    "npr_batch_size": 32,
    "aide_batch_size": 4,
    "num_workers": 2,
    "save_every_batches": 25,
    "resume": True,
    "verify_checkpoint_hashes": True,
    "clip_checkpoint": None,
    "npr_checkpoint": None,
    "aide_checkpoint": None,
    "clip_pretrained": "openai",
    "aide_semantic_checkpoint": None,
    "common_runtime_path": None,
    "aide_runtime_path": None,
}


STANDARD_LUMINANCE_QTABLE = np.asarray(
    [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ],
    dtype=np.float64,
)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_dataframe_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_membership_sha256(frame: pd.DataFrame) -> str:
    columns = [
        column
        for column in ("sample_id", "label", "generator", "relative_path")
        if column in frame.columns
    ]
    canonical = frame[columns].sort_values("sample_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def import_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def repository_root_from_script() -> Path | None:
    path = Path(__file__).resolve()
    for candidate in (path.parent, *path.parents):
        if (candidate / "baselines").is_dir() and (candidate / "benchmarks").is_dir():
            return candidate
    return None


def resolve_runtime_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    explicit_common = config.get("common_runtime_path")
    explicit_aide = config.get("aide_runtime_path")
    if explicit_common and explicit_aide:
        common_path = Path(explicit_common)
        aide_path = Path(explicit_aide)
    else:
        root = repository_root_from_script()
        if root is None:
            raise FileNotFoundError(
                "Cannot locate embedded runtimes. Set common_runtime_path and aide_runtime_path."
            )
        common_path = (
            root
            / "benchmarks"
            / "commfor_unseen_three_models"
            / "code"
            / "test_three_models_commfor_unseen_kaggle.py"
        )
        aide_path = (
            root
            / "baselines"
            / "aide_original_full"
            / "train_aide_original_full_tiny_commfor_kaggle.py"
        )
    for path in (common_path, aide_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return common_path, aide_path


def resolve_unique_file(input_root: Path, explicit: Any, filename: str) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    matches = sorted(path for path in input_root.rglob(filename) if path.is_file())
    if not matches:
        raise FileNotFoundError(
            f"Cannot find {filename} below {input_root}. Attach the checkpoint Kaggle Dataset."
        )
    if len(matches) > 1:
        exact_checkpoint_matches = [path for path in matches if path.parent.name == "checkpoints"]
        if len(exact_checkpoint_matches) == 1:
            return exact_checkpoint_matches[0]
        raise RuntimeError(f"Multiple files named {filename}: {matches}")
    return matches[0]


def resolve_checkpoint_paths(config: dict[str, Any]) -> dict[str, Path]:
    input_root = Path(config["input_root"])
    explicit_keys = {
        "clip_linear_probe": "clip_checkpoint",
        "npr_resnet18": "npr_checkpoint",
        "aide_original_full": "aide_checkpoint",
    }
    paths = {
        model: resolve_unique_file(
            input_root,
            config.get(explicit_keys[model]),
            CHECKPOINT_FILENAMES[model],
        )
        for model in CHECKPOINT_FILENAMES
    }
    if bool(config["verify_checkpoint_hashes"]):
        for model, path in paths.items():
            actual = sha256_file(path)
            expected = EXPECTED_CHECKPOINT_SHA256[model]
            if actual != expected:
                raise ValueError(
                    f"Wrong checkpoint for {model}: expected {expected}, got {actual} at {path}"
                )
    return paths


def has_tiny_layout(path: Path, split_name: str) -> bool:
    if not path.is_dir():
        return False
    generator_dirs = [child for child in path.iterdir() if child.is_dir()]
    return any(
        (generator / split_name / "nature").is_dir()
        and (generator / split_name / "ai").is_dir()
        for generator in generator_dirs
    )


def find_tiny_root(input_root: Path, explicit: Any, split_name: str) -> Path:
    split_aliases = tuple(dict.fromkeys((split_name, "val", "validation", "test")))
    if explicit:
        candidate = Path(explicit)
        for alias in split_aliases:
            if has_tiny_layout(candidate, alias):
                return candidate
        raise FileNotFoundError(f"Tiny-GenImage layout not found at {candidate}")

    preferred = (
        input_root / "datasets" / "yangsangtai" / "tiny-genimage",
        input_root / "tiny-genimage",
    )
    for candidate in preferred:
        for alias in split_aliases:
            if has_tiny_layout(candidate, alias):
                return candidate
    for candidate in input_root.rglob("*"):
        if not candidate.is_dir():
            continue
        for alias in split_aliases:
            try:
                if has_tiny_layout(candidate, alias):
                    return candidate
            except PermissionError:
                continue
    raise FileNotFoundError(
        "Cannot find Tiny-GenImage. Attach yangsangtai/tiny-genimage to the notebook."
    )


def detect_split_name(dataset_root: Path, requested: str) -> str:
    for alias in tuple(dict.fromkeys((requested, "val", "validation", "test"))):
        if has_tiny_layout(dataset_root, alias):
            return alias
    raise FileNotFoundError(f"No validation split below {dataset_root}")


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_test_manifest(
    dataset_root: Path,
    split_name: str,
    seed: int,
    max_per_class_per_generator: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    generators = sorted(
        child
        for child in dataset_root.iterdir()
        if child.is_dir()
        and (child / split_name / "nature").is_dir()
        and (child / split_name / "ai").is_dir()
    )
    if not generators:
        raise RuntimeError(f"No Tiny-GenImage generators found below {dataset_root}")

    for generator_dir in generators:
        class_paths: dict[int, list[Path]] = {}
        for directory_name, label, _ in LABEL_DIRECTORIES:
            class_paths[label] = list_images(generator_dir / split_name / directory_name)
        balanced_count = min(len(class_paths[0]), len(class_paths[1]))
        if max_per_class_per_generator is not None:
            balanced_count = min(balanced_count, int(max_per_class_per_generator))
        if balanced_count <= 0:
            continue

        for directory_name, label, label_name in LABEL_DIRECTORIES:
            paths = class_paths[label]
            selected = paths if len(paths) == balanced_count else rng.sample(paths, balanced_count)
            for path in sorted(selected):
                relative = path.relative_to(dataset_root).as_posix()
                rows.append(
                    {
                        "sample_id": f"tiny:{generator_dir.name}:{label_name}:{relative}",
                        "image_path": str(path),
                        "relative_path": relative,
                        "image_name": path.name,
                        "generator": generator_dir.name,
                        "label": int(label),
                        "label_name": label_name,
                        "original_extension": path.suffix.lower(),
                    }
                )
    frame = pd.DataFrame(rows).sort_values(
        ["generator", "label", "relative_path"]
    ).reset_index(drop=True)
    frame.insert(0, "manifest_order", np.arange(len(frame), dtype=int))
    if frame.empty:
        raise RuntimeError("Tiny-GenImage manifest is empty")
    return frame


def scaled_standard_luminance_table(quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 1, 100))
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    table = np.floor((STANDARD_LUMINANCE_QTABLE * scale + 50.0) / 100.0)
    return np.clip(table, 1, 255)


def estimate_jpeg_quality(image: Image.Image) -> tuple[float, float]:
    quantization = getattr(image, "quantization", None)
    if not quantization:
        return float("nan"), float("nan")
    luminance = quantization.get(0)
    if luminance is None:
        luminance = next(iter(quantization.values()), None)
    if luminance is None or len(luminance) != 64:
        return float("nan"), float("nan")
    observed = np.asarray(luminance, dtype=np.float64)
    candidates = []
    for quality in range(1, 101):
        reference = scaled_standard_luminance_table(quality)
        mse = float(np.mean((observed - reference) ** 2))
        candidates.append((mse, quality))
    mse, quality = min(candidates)
    return float(quality), float(mse)


def size_bin(short_side: int) -> str:
    if short_side < 128:
        return "lt_128"
    if short_side < 256:
        return "128_255"
    if short_side < 512:
        return "256_511"
    if short_side < 1024:
        return "512_1023"
    return "ge_1024"


def megapixel_bin(megapixels: float) -> str:
    if megapixels < 0.1:
        return "lt_0.1"
    if megapixels < 0.25:
        return "0.1_0.25"
    if megapixels < 1.0:
        return "0.25_1"
    if megapixels <= 4.0:
        return "1_4"
    return "gt_4"


def aspect_group(ratio: float) -> str:
    if ratio < 0.9:
        return "portrait"
    if ratio <= 1.1:
        return "nearly_square"
    return "landscape"


def jpeg_quality_bin(quality: float) -> str:
    if not np.isfinite(quality):
        return "not_jpeg_or_unknown"
    if quality <= 69:
        return "50_69_or_lower"
    if quality <= 84:
        return "70_84"
    if quality <= 93:
        return "85_93"
    if quality <= 98:
        return "94_98_near_q96"
    return "99_100"


def audit_manifest(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for row in tqdm(manifest.to_dict("records"), desc="Auditing Tiny test images"):
        path = Path(row["image_path"])
        try:
            # Pillow requires verify() immediately after open; metadata access can
            # advance/decode the stream for formats such as PNG.
            with Image.open(path) as verification_image:
                verification_image.verify()
            with Image.open(path) as image:
                detected_format = str(image.format or "unknown").upper()
                width, height = image.size
                mode = str(image.mode)
                quality, quality_mse = estimate_jpeg_quality(image)
                try:
                    orientation = image.getexif().get(274)
                except Exception:
                    orientation = None
                has_alpha = "A" in image.getbands() or "transparency" in image.info
            short = min(width, height)
            long_side = max(width, height)
            ratio = float(width / max(height, 1))
            megapixels = float(width * height / 1_000_000.0)
            file_bytes = int(path.stat().st_size)
            rows.append(
                {
                    **row,
                    "pil_format": detected_format,
                    "color_mode": mode,
                    "has_alpha": bool(has_alpha),
                    "exif_orientation": orientation,
                    "native_width": int(width),
                    "native_height": int(height),
                    "native_size": f"{width}x{height}",
                    "short_side": int(short),
                    "long_side": int(long_side),
                    "aspect_ratio": ratio,
                    "aspect_group": aspect_group(ratio),
                    "megapixels": megapixels,
                    "size_bin": size_bin(short),
                    "megapixel_bin": megapixel_bin(megapixels),
                    "file_bytes": file_bytes,
                    "file_kib": float(file_bytes / 1024.0),
                    "bytes_per_pixel": float(file_bytes / max(width * height, 1)),
                    "jpeg_quality_estimate": quality,
                    "jpeg_quality_mse": quality_mse,
                    "jpeg_quality_bin": jpeg_quality_bin(quality),
                }
            )
        except Exception as exc:
            invalid.append(
                {
                    **row,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    audited = pd.DataFrame(rows).sort_values("manifest_order").reset_index(drop=True)
    invalid_frame = pd.DataFrame(invalid)
    if audited.empty:
        raise RuntimeError("All Tiny test images failed validation")
    return audited, invalid_frame


def grouped_count(frame: pd.DataFrame, columns: list[str], name: str = "count") -> pd.DataFrame:
    return frame.groupby(columns, dropna=False).size().reset_index(name=name)


def build_controlled_manifest(
    audited: pd.DataFrame,
    quality_min: int,
    quality_max: int,
    seed: int,
    balance_across_generators: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    real_candidates = audited[
        (audited["label"] == 0)
        & (audited["pil_format"] == "JPEG")
        & audited["jpeg_quality_estimate"].between(quality_min, quality_max, inclusive="both")
    ]
    fake_candidates = audited[
        (audited["label"] == 1) & (audited["pil_format"] != "JPEG")
    ]
    generators = sorted(audited["generator"].unique())
    availability: list[dict[str, Any]] = []
    per_generator_counts: dict[str, int] = {}
    for generator in generators:
        num_real = int((real_candidates["generator"] == generator).sum())
        num_fake = int((fake_candidates["generator"] == generator).sum())
        usable = min(num_real, num_fake)
        availability.append(
            {
                "generator": generator,
                "real_q96_candidates": num_real,
                "lossless_fake_candidates": num_fake,
                "usable_per_class": usable,
            }
        )
        per_generator_counts[generator] = usable
    if not per_generator_counts or min(per_generator_counts.values()) <= 0:
        raise RuntimeError(
            "Cannot build controlled JPEG96 cohort. Widen controlled_real_quality_min/max."
        )
    common_count = min(per_generator_counts.values()) if balance_across_generators else None
    rng = random.Random(seed + 9600)
    selected: list[pd.DataFrame] = []
    for generator in generators:
        count = int(common_count if common_count is not None else per_generator_counts[generator])
        for candidates in (real_candidates, fake_candidates):
            part = candidates[candidates["generator"] == generator]
            indices = list(part.index)
            chosen = indices if len(indices) == count else rng.sample(indices, count)
            selected.append(audited.loc[sorted(chosen)])
    controlled = pd.concat(selected, ignore_index=True).sort_values(
        ["generator", "label", "sample_id"]
    ).reset_index(drop=True)
    controlled["controlled_manifest_order"] = np.arange(len(controlled), dtype=int)
    metadata = {
        "quality_min": int(quality_min),
        "quality_max": int(quality_max),
        "balance_across_generators": bool(balance_across_generators),
        "common_per_class_per_generator": common_count,
        "num_samples": int(len(controlled)),
        "num_real": int((controlled["label"] == 0).sum()),
        "num_fake": int((controlled["label"] == 1).sum()),
        "availability": availability,
        "membership_sha256": dataframe_membership_sha256(controlled),
    }
    return controlled, metadata


def create_dataset_statistics(
    audited: pd.DataFrame,
    invalid: pd.DataFrame,
    controlled: pd.DataFrame,
    controlled_metadata: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    dataset_dir = output_root / "dataset"
    plot_dir = output_root / "plots"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "generator_label_counts.csv": grouped_count(audited, ["generator", "label_name"]),
        "format_distribution.csv": grouped_count(audited, ["label_name", "pil_format"]),
        "generator_format_distribution.csv": grouped_count(
            audited, ["generator", "label_name", "pil_format"]
        ),
        "exact_size_distribution.csv": grouped_count(
            audited, ["native_width", "native_height", "native_size", "label_name"]
        ),
        "size_bin_distribution.csv": grouped_count(audited, ["label_name", "size_bin"]),
        "megapixel_distribution.csv": grouped_count(
            audited, ["label_name", "megapixel_bin"]
        ),
        "aspect_ratio_distribution.csv": grouped_count(
            audited, ["label_name", "aspect_group"]
        ),
        "jpeg_quality_distribution.csv": grouped_count(
            audited, ["label_name", "pil_format", "jpeg_quality_bin"]
        ),
        "color_mode_distribution.csv": grouped_count(
            audited, ["label_name", "color_mode", "has_alpha"]
        ),
    }
    file_size_summary = (
        audited.groupby(["label_name", "pil_format"], dropna=False)
        .agg(
            count=("sample_id", "size"),
            mean_kib=("file_kib", "mean"),
            median_kib=("file_kib", "median"),
            min_kib=("file_kib", "min"),
            max_kib=("file_kib", "max"),
            mean_bytes_per_pixel=("bytes_per_pixel", "mean"),
        )
        .reset_index()
    )
    tables["file_size_distribution.csv"] = file_size_summary
    for filename, table in tables.items():
        save_dataframe_atomic(table, dataset_dir / filename)

    overall = {
        "num_valid_images": int(len(audited)),
        "num_invalid_images": int(len(invalid)),
        "num_real": int((audited["label"] == 0).sum()),
        "num_fake": int((audited["label"] == 1).sum()),
        "num_generators": int(audited["generator"].nunique()),
        "num_unique_exact_sizes": int(audited[["native_width", "native_height"]].drop_duplicates().shape[0]),
        "num_formats": int(audited["pil_format"].nunique()),
        "num_square": int((audited["native_width"] == audited["native_height"]).sum()),
        "num_non_square": int((audited["native_width"] != audited["native_height"]).sum()),
        "min_width": int(audited["native_width"].min()),
        "max_width": int(audited["native_width"].max()),
        "median_width": float(audited["native_width"].median()),
        "min_height": int(audited["native_height"].min()),
        "max_height": int(audited["native_height"].max()),
        "median_height": float(audited["native_height"].median()),
        "full_membership_sha256": dataframe_membership_sha256(audited),
        "controlled_jpeg96": controlled_metadata,
    }
    save_json(dataset_dir / "overall_statistics.json", overall)

    def plot_stacked(table: pd.DataFrame, index: str, columns: str, values: str, title: str, path: Path):
        pivot = table.pivot_table(index=index, columns=columns, values=values, aggfunc="sum", fill_value=0)
        axis = pivot.plot(kind="bar", stacked=True, figsize=(10, 5))
        axis.set_title(title)
        axis.set_ylabel("Number of images")
        axis.figure.tight_layout()
        axis.figure.savefig(path, dpi=160)
        plt.close(axis.figure)

    plot_stacked(
        tables["format_distribution.csv"],
        "pil_format",
        "label_name",
        "count",
        "Native file format by label",
        plot_dir / "format_by_label.png",
    )
    plot_stacked(
        tables["size_bin_distribution.csv"],
        "size_bin",
        "label_name",
        "count",
        "Native short-side bins by label",
        plot_dir / "size_bins_by_label.png",
    )
    plot_stacked(
        tables["aspect_ratio_distribution.csv"],
        "aspect_group",
        "label_name",
        "count",
        "Aspect-ratio groups by label",
        plot_dir / "aspect_ratio_by_label.png",
    )
    jpeg_table = tables["jpeg_quality_distribution.csv"]
    jpeg_table = jpeg_table[jpeg_table["pil_format"] == "JPEG"]
    if not jpeg_table.empty:
        plot_stacked(
            jpeg_table,
            "jpeg_quality_bin",
            "label_name",
            "count",
            "Estimated native JPEG quality by label",
            plot_dir / "jpeg_quality_by_label.png",
        )
    top_sizes = (
        audited.groupby("native_size").size().sort_values(ascending=False).head(20).sort_values()
    )
    axis = top_sizes.plot(kind="barh", figsize=(10, 7), title="Top 20 native image sizes")
    axis.set_xlabel("Number of images")
    axis.figure.tight_layout()
    axis.figure.savefig(plot_dir / "top_native_sizes.png", dpi=160)
    plt.close(axis.figure)

    print("\nDataset audit summary:")
    print(json.dumps(overall, ensure_ascii=False, indent=2, default=json_default))
    print("\nGenerator/label counts:")
    print(tables["generator_label_counts.csv"].to_string(index=False))
    print("\nNative format distribution:")
    print(tables["format_distribution.csv"].to_string(index=False))
    print("\nTop native sizes:")
    print(top_sizes.sort_values(ascending=False).to_string())
    return overall


def jpeg_roundtrip(image: Image.Image, config: dict[str, Any]) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=int(config["jpeg_quality"]),
        subsampling=int(config["jpeg_subsampling"]),
        optimize=bool(config["jpeg_optimize"]),
        progressive=bool(config["jpeg_progressive"]),
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def png_roundtrip(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def load_case_image(row: dict[str, Any], case_name: str, config: dict[str, Any]) -> Image.Image:
    with Image.open(row["image_path"]) as source:
        image = source.convert("RGB").copy()
    if case_name in ("raw", "raw_on_controlled_cohort"):
        return image
    if case_name == "png_roundtrip":
        return png_roundtrip(image)
    if case_name == "fake_jpeg96":
        return jpeg_roundtrip(image, config) if int(row["label"]) == 1 else image
    if case_name == "all_jpeg96":
        return jpeg_roundtrip(image, config)
    if case_name == "controlled_jpeg96":
        return jpeg_roundtrip(image, config) if int(row["label"]) == 1 else image
    raise ValueError(f"Unknown case: {case_name}")


def case_manifest(case_name: str, full: pd.DataFrame, controlled: pd.DataFrame) -> pd.DataFrame:
    if case_name in ("controlled_jpeg96", "raw_on_controlled_cohort"):
        return controlled.copy()
    return full.copy()


def prediction_base(
    row: dict[str, Any],
    model_name: str,
    case_name: str,
    probability: float,
    checkpoint_sha256: str,
    case_sha256: str,
    threshold: float,
) -> dict[str, Any]:
    return {
        "manifest_order": int(row["manifest_order"]),
        "sample_id": str(row["sample_id"]),
        "model": model_name,
        "case": case_name,
        "label": int(row["label"]),
        "label_name": str(row["label_name"]),
        "generator": str(row["generator"]),
        "image_name": str(row["image_name"]),
        "relative_path": str(row["relative_path"]),
        "original_extension": str(row["original_extension"]),
        "pil_format": str(row["pil_format"]),
        "native_width": int(row["native_width"]),
        "native_height": int(row["native_height"]),
        "jpeg_quality_estimate": row["jpeg_quality_estimate"],
        "fake_probability": float(probability),
        "predicted_label": int(float(probability) >= float(threshold)),
        "checkpoint_sha256": checkpoint_sha256,
        "case_manifest_sha256": case_sha256,
    }


class VariantTensorDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        case_name: str,
        config: dict[str, Any],
        preprocess: Callable[[Image.Image], torch.Tensor],
    ):
        self.records = frame.to_dict("records")
        self.case_name = case_name
        self.config = config
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        image = load_case_image(row, self.case_name, self.config)
        return {"input": self.preprocess(image), "row_index": int(index)}


class AIDEVariantDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        case_name: str,
        config: dict[str, Any],
        aide_module: Any,
        model_config: dict[str, Any],
    ):
        self.records = frame.to_dict("records")
        self.case_name = case_name
        self.config = config
        self.aide = aide_module
        self.model_config = model_config
        self.selector = aide_module.AIDEDCTPatchSelector(
            window_size=int(model_config["dct_window_size"]),
            stride=int(model_config["dct_stride"]),
            grade_bands=int(model_config["dct_grade_bands"]),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        image = load_case_image(row, self.case_name, self.config)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        patches, selection = self.selector(tensor)
        size = int(self.model_config["image_size"])
        patches = F.interpolate(
            patches,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        raw = F.interpolate(
            tensor.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        patches = (patches - self.aide.IMAGENET_MEAN) / self.aide.IMAGENET_STD
        raw = (raw - self.aide.IMAGENET_MEAN) / self.aide.IMAGENET_STD
        output = {
            "input": torch.cat([patches, raw], dim=0),
            "row_index": int(index),
        }
        output.update(selection)
        return output


def make_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if int(num_workers) > 0:
        options.update({"persistent_workers": False, "prefetch_factor": 2})
    return DataLoader(**options)


def completed_prediction_or_partial(
    final_path: Path,
    partial_path: Path,
    expected_ids: set[str],
    resume: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    if resume and final_path.is_file():
        final = pd.read_csv(final_path)
        if set(final["sample_id"].astype(str)) == expected_ids:
            return final.sort_values("manifest_order").reset_index(drop=True), final.iloc[0:0].copy()
    if resume and partial_path.is_file():
        partial = pd.read_csv(partial_path)
        partial = partial[partial["sample_id"].astype(str).isin(expected_ids)]
        partial = partial.drop_duplicates("sample_id", keep="last")
        return None, partial
    return None, pd.DataFrame()


def infer_tensor_case(
    model: nn.Module,
    model_name: str,
    checkpoint_sha256: str,
    preprocess: Callable[[Image.Image], torch.Tensor],
    frame: pd.DataFrame,
    case_name: str,
    config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    output_root: Path,
) -> pd.DataFrame:
    prediction_dir = output_root / "predictions" / model_name
    final_path = prediction_dir / f"{case_name}.csv"
    partial_path = prediction_dir / f"{case_name}.partial.csv"
    expected_ids = set(frame["sample_id"].astype(str))
    final, existing = completed_prediction_or_partial(
        final_path, partial_path, expected_ids, bool(config["resume"])
    )
    if final is not None:
        print(f"Resume: completed {model_name}/{case_name}")
        return final
    done = set(existing["sample_id"].astype(str)) if not existing.empty else set()
    remaining = frame[~frame["sample_id"].astype(str).isin(done)].reset_index(drop=True)
    records = remaining.to_dict("records")
    case_sha = dataframe_membership_sha256(frame)
    dataset = VariantTensorDataset(remaining, case_name, config, preprocess)
    loader = make_loader(dataset, batch_size, int(config["num_workers"]))
    new_rows: list[dict[str, Any]] = []
    model.eval()
    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"{model_name} / {case_name}")
    ):
        inputs = batch["input"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(inputs)
        probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        for row_index, probability in zip(batch["row_index"].tolist(), probabilities):
            row = records[int(row_index)]
            new_rows.append(
                prediction_base(
                    row,
                    model_name,
                    case_name,
                    float(probability),
                    checkpoint_sha256,
                    case_sha,
                    float(config["threshold"]),
                )
            )
        if (batch_index + 1) % int(config["save_every_batches"]) == 0:
            combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            save_dataframe_atomic(combined.sort_values("manifest_order"), partial_path)
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.drop_duplicates("sample_id", keep="last")
    if set(combined["sample_id"].astype(str)) != expected_ids:
        raise RuntimeError(f"Incomplete predictions for {model_name}/{case_name}")
    combined = combined.sort_values("manifest_order").reset_index(drop=True)
    save_dataframe_atomic(combined, final_path)
    if partial_path.exists():
        partial_path.unlink()
    return combined


def infer_aide_case(
    model: nn.Module,
    checkpoint_sha256: str,
    frame: pd.DataFrame,
    case_name: str,
    config: dict[str, Any],
    aide_module: Any,
    model_config: dict[str, Any],
    device: torch.device,
    output_root: Path,
) -> pd.DataFrame:
    model_name = "aide_original_full"
    prediction_dir = output_root / "predictions" / model_name
    final_path = prediction_dir / f"{case_name}.csv"
    partial_path = prediction_dir / f"{case_name}.partial.csv"
    expected_ids = set(frame["sample_id"].astype(str))
    final, existing = completed_prediction_or_partial(
        final_path, partial_path, expected_ids, bool(config["resume"])
    )
    if final is not None:
        print(f"Resume: completed {model_name}/{case_name}")
        return final
    done = set(existing["sample_id"].astype(str)) if not existing.empty else set()
    remaining = frame[~frame["sample_id"].astype(str).isin(done)].reset_index(drop=True)
    records = remaining.to_dict("records")
    case_sha = dataframe_membership_sha256(frame)
    dataset = AIDEVariantDataset(remaining, case_name, config, aide_module, model_config)
    loader = make_loader(
        dataset, int(config["aide_batch_size"]), int(config["num_workers"])
    )
    new_rows: list[dict[str, Any]] = []
    model.eval()
    selection_keys = (
        "num_candidates",
        "low_1_score",
        "high_1_score",
        "low_2_score",
        "high_2_score",
    )
    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"{model_name} / {case_name}")
    ):
        inputs = batch["input"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits, features = model(inputs, return_features=True)
        probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        semantic_norm = features["semantic"].float().norm(dim=1).cpu().numpy()
        forensic_norm = features["forensic"].float().norm(dim=1).cpu().numpy()
        for output_index, (row_index, probability) in enumerate(
            zip(batch["row_index"].tolist(), probabilities)
        ):
            row = records[int(row_index)]
            prediction = prediction_base(
                row,
                model_name,
                case_name,
                float(probability),
                checkpoint_sha256,
                case_sha,
                float(config["threshold"]),
            )
            prediction["semantic_feature_l2"] = float(semantic_norm[output_index])
            prediction["forensic_feature_l2"] = float(forensic_norm[output_index])
            for key in selection_keys:
                value = batch[key][output_index]
                prediction[key] = int(value) if key == "num_candidates" else float(value)
            new_rows.append(prediction)
        if (batch_index + 1) % int(config["save_every_batches"]) == 0:
            combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            save_dataframe_atomic(combined.sort_values("manifest_order"), partial_path)
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.drop_duplicates("sample_id", keep="last")
    if set(combined["sample_id"].astype(str)) != expected_ids:
        raise RuntimeError(f"Incomplete predictions for {model_name}/{case_name}")
    combined = combined.sort_values("manifest_order").reset_index(drop=True)
    save_dataframe_atomic(combined, final_path)
    if partial_path.exists():
        partial_path.unlink()
    return combined


def compute_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    y_true = frame["label"].to_numpy(dtype=int)
    y_prob = frame["fake_probability"].to_numpy(dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    output: dict[str, Any] = {
        "num_samples": int(len(frame)),
        "num_real": int((y_true == 0).sum()),
        "num_fake": int((y_true == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "real_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "fake_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fake_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fake_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mean_real_fake_probability": float(y_prob[y_true == 0].mean()),
        "mean_fake_fake_probability": float(y_prob[y_true == 1].mean()),
        "threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    output["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    output["average_precision"] = (
        float(average_precision_score(y_true, y_prob)) if (y_true == 1).any() else float("nan")
    )
    return output


def compile_results(
    predictions: dict[tuple[str, str], pd.DataFrame],
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    threshold = float(config["threshold"])
    metric_dir = output_root / "metrics"
    plot_dir = output_root / "plots"
    metric_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    overall_rows: list[dict[str, Any]] = []
    generator_rows: list[dict[str, Any]] = []
    confusion_payload: dict[str, Any] = {}
    long_predictions: list[pd.DataFrame] = []
    for (model_name, case_name), frame in predictions.items():
        metrics = compute_metrics(frame, threshold)
        overall_rows.append(
            {
                "model": model_name,
                "case": case_name,
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            }
        )
        confusion_payload[f"{model_name}/{case_name}"] = metrics["confusion_matrix"]
        for generator, part in frame.groupby("generator", sort=True):
            generator_metrics = compute_metrics(part, threshold)
            generator_rows.append(
                {
                    "model": model_name,
                    "case": case_name,
                    "generator": str(generator),
                    **{
                        key: value
                        for key, value in generator_metrics.items()
                        if key != "confusion_matrix"
                    },
                }
            )
        long_predictions.append(frame)
    overall = pd.DataFrame(overall_rows).sort_values(["model", "case"])
    by_generator = pd.DataFrame(generator_rows).sort_values(
        ["model", "case", "generator"]
    )
    save_dataframe_atomic(overall, metric_dir / "overall_by_model_case.csv")
    save_dataframe_atomic(by_generator, metric_dir / "generator_by_model_case.csv")
    save_json(metric_dir / "confusion_matrices.json", confusion_payload)
    all_predictions = pd.concat(long_predictions, ignore_index=True)
    save_dataframe_atomic(all_predictions, output_root / "predictions" / "all_predictions_long.csv")

    paired_rows: list[pd.DataFrame] = []
    delta_metric_rows: list[dict[str, Any]] = []
    metric_columns = (
        "accuracy",
        "balanced_accuracy",
        "real_recall",
        "fake_recall",
        "fake_f1",
        "roc_auc",
        "average_precision",
    )
    for model_name in sorted({key[0] for key in predictions}):
        for case_name in sorted({key[1] for key in predictions if key[0] == model_name}):
            if case_name in ("raw", "raw_on_controlled_cohort"):
                continue
            base_case = (
                "raw_on_controlled_cohort"
                if case_name == "controlled_jpeg96"
                else "raw"
            )
            base_key = (model_name, base_case)
            case_key = (model_name, case_name)
            if base_key not in predictions or case_key not in predictions:
                continue
            base = predictions[base_key][
                ["sample_id", "label", "generator", "fake_probability", "predicted_label"]
            ].rename(
                columns={
                    "fake_probability": "base_fake_probability",
                    "predicted_label": "base_predicted_label",
                }
            )
            changed = predictions[case_key][
                ["sample_id", "fake_probability", "predicted_label"]
            ].rename(
                columns={
                    "fake_probability": "case_fake_probability",
                    "predicted_label": "case_predicted_label",
                }
            )
            paired = base.merge(changed, on="sample_id", how="inner")
            paired.insert(0, "model", model_name)
            paired.insert(1, "base_case", base_case)
            paired.insert(2, "case", case_name)
            paired["fake_probability_delta"] = (
                paired["case_fake_probability"] - paired["base_fake_probability"]
            )
            paired["prediction_changed"] = (
                paired["case_predicted_label"] != paired["base_predicted_label"]
            )
            paired_rows.append(paired)

            base_metrics = compute_metrics(predictions[base_key], threshold)
            case_metrics = compute_metrics(predictions[case_key], threshold)
            delta_metric_rows.append(
                {
                    "model": model_name,
                    "base_case": base_case,
                    "case": case_name,
                    **{
                        f"delta_{column}": float(case_metrics[column] - base_metrics[column])
                        for column in metric_columns
                    },
                    "mean_probability_delta": float(paired["fake_probability_delta"].mean()),
                    "mean_real_probability_delta": float(
                        paired.loc[paired["label"] == 0, "fake_probability_delta"].mean()
                    ),
                    "mean_fake_probability_delta": float(
                        paired.loc[paired["label"] == 1, "fake_probability_delta"].mean()
                    ),
                    "prediction_flip_rate": float(paired["prediction_changed"].mean()),
                }
            )
    paired_all = pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()
    delta_metrics = pd.DataFrame(delta_metric_rows)
    if not paired_all.empty:
        save_dataframe_atomic(paired_all, metric_dir / "paired_probability_shift.csv")
    if not delta_metrics.empty:
        save_dataframe_atomic(delta_metrics, metric_dir / "delta_from_matching_raw.csv")

    for metric, filename, title in (
        ("accuracy", "accuracy_by_case.png", "Accuracy by model and codec case"),
        ("fake_recall", "fake_recall_by_case.png", "Fake recall by model and codec case"),
        ("roc_auc", "roc_auc_by_case.png", "ROC-AUC by model and codec case"),
    ):
        pivot = overall.pivot(index="case", columns="model", values=metric)
        axis = pivot.plot(kind="bar", figsize=(12, 6), ylim=(0.0, 1.0), title=title)
        axis.set_ylabel(metric)
        axis.figure.tight_layout()
        axis.figure.savefig(plot_dir / filename, dpi=160)
        plt.close(axis.figure)

    if not delta_metrics.empty:
        pivot = delta_metrics.pivot(index="case", columns="model", values="delta_fake_recall")
        axis = pivot.plot(
            kind="bar",
            figsize=(12, 6),
            title="Change in fake recall versus matching raw cohort",
        )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_ylabel("Delta fake recall")
        axis.figure.tight_layout()
        axis.figure.savefig(plot_dir / "fake_recall_delta.png", dpi=160)
        plt.close(axis.figure)

    summary = {
        "status": "complete",
        "models": sorted(overall["model"].unique()),
        "cases": sorted(overall["case"].unique()),
        "overall": overall.to_dict("records"),
        "delta_from_matching_raw": delta_metrics.to_dict("records"),
    }
    save_json(metric_dir / "summary.json", summary)
    print("\nOverall metrics:")
    print(
        overall[
            [
                "model",
                "case",
                "num_samples",
                "accuracy",
                "balanced_accuracy",
                "real_recall",
                "fake_recall",
                "fake_f1",
                "roc_auc",
            ]
        ].to_string(index=False)
    )
    if not delta_metrics.empty:
        print("\nDelta versus matching raw cohort:")
        print(delta_metrics.to_string(index=False))
    return summary


def software_provenance() -> dict[str, Any]:
    packages: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": PIL.__version__,
    }
    try:
        import torchvision

        packages["torchvision"] = torchvision.__version__
    except Exception:
        pass
    try:
        import open_clip

        packages["open_clip"] = getattr(open_clip, "__version__", "unknown")
    except Exception:
        pass
    if torch.cuda.is_available():
        packages["cuda"] = torch.version.cuda
        packages["gpu"] = torch.cuda.get_device_name(0)
    return packages


def resolve_aide_semantic_checkpoint(config: dict[str, Any]) -> str | None:
    explicit = config.get("aide_semantic_checkpoint")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    candidates = sorted(
        path
        for path in Path(config["input_root"]).rglob("open_clip_pytorch_model.bin")
        if path.is_file()
    )
    if len(candidates) == 1:
        return str(candidates[0])
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple open_clip_pytorch_model.bin files found; set aide_semantic_checkpoint."
        )
    return None


def run_evaluation(user_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if user_config:
        config.update(user_config)
    if bool(config["smoke"]) and config.get("max_per_class_per_generator") is None:
        config["max_per_class_per_generator"] = 100
    requested_models = list(dict.fromkeys(config["models"]))
    unknown_models = set(requested_models) - set(CHECKPOINT_FILENAMES)
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")
    requested_cases = list(dict.fromkeys(config["cases"]))
    for required in ("raw", "raw_on_controlled_cohort"):
        if required not in requested_cases:
            requested_cases.insert(0, required)
    unknown_cases = set(requested_cases) - set(ALL_CASES)
    if unknown_cases:
        raise ValueError(f"Unknown cases: {sorted(unknown_cases)}")
    config["models"] = requested_models
    config["cases"] = requested_cases

    seed = int(config["selection_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_root = Path(config["output_root"])
    for relative in ("dataset", "metrics", "plots", "predictions", "provenance"):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    serializable_config = {
        key: str(value) if isinstance(value, Path) else value for key, value in config.items()
    }
    save_json(output_root / "config.json", serializable_config)
    save_json(
        output_root / "status.json",
        {"status": "running", "started_at_unix": time.time()},
    )

    try:
        input_root = Path(config["input_root"])
        checkpoint_paths = resolve_checkpoint_paths(config)
        checkpoint_provenance = {
            model: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for model, path in checkpoint_paths.items()
        }
        save_json(output_root / "provenance" / "checkpoints.json", checkpoint_provenance)
        save_json(output_root / "provenance" / "software_versions.json", software_provenance())
        save_json(
            output_root / "provenance" / "codec_transform.json",
            {
                "jpeg_encoder": "Pillow/libjpeg",
                "quality": int(config["jpeg_quality"]),
                "subsampling": int(config["jpeg_subsampling"]),
                "optimize": bool(config["jpeg_optimize"]),
                "progressive": bool(config["jpeg_progressive"]),
                "application_order": "native image -> codec roundtrip -> original model preprocessing",
                "png_roundtrip_note": "Lossless pixel roundtrip; pre-existing JPEG artifacts remain.",
                "all_jpeg96_note": "Native JPEG real images may be double-compressed.",
            },
        )

        dataset_root = find_tiny_root(
            input_root, config.get("dataset_root"), str(config["validation_split"])
        )
        split_name = detect_split_name(dataset_root, str(config["validation_split"]))
        raw_manifest = build_test_manifest(
            dataset_root,
            split_name,
            seed,
            config.get("max_per_class_per_generator"),
        )
        audited, invalid = audit_manifest(raw_manifest)
        save_dataframe_atomic(audited, output_root / "dataset" / "image_audit.csv")
        save_dataframe_atomic(audited, output_root / "dataset" / "full_test_manifest.csv")
        save_dataframe_atomic(invalid, output_root / "dataset" / "invalid_images.csv")
        controlled, controlled_metadata = build_controlled_manifest(
            audited,
            int(config["controlled_real_quality_min"]),
            int(config["controlled_real_quality_max"]),
            seed,
            bool(config["controlled_balance_across_generators"]),
        )
        save_dataframe_atomic(
            controlled, output_root / "dataset" / "controlled_jpeg96_manifest.csv"
        )
        save_json(
            output_root / "dataset" / "controlled_jpeg96_summary.json",
            controlled_metadata,
        )
        statistics = create_dataset_statistics(
            audited, invalid, controlled, controlled_metadata, output_root
        )
        case_counts = []
        for name in requested_cases:
            frame = case_manifest(name, audited, controlled)
            case_counts.append(
                {
                    "case": name,
                    "num_samples": int(len(frame)),
                    "num_real": int((frame["label"] == 0).sum()),
                    "num_fake": int((frame["label"] == 1).sum()),
                    "num_generators": int(frame["generator"].nunique()),
                    "membership_sha256": dataframe_membership_sha256(frame),
                }
            )
        case_counts_frame = pd.DataFrame(case_counts)
        save_dataframe_atomic(case_counts_frame, output_root / "dataset" / "case_sample_counts.csv")
        print("\nCase sample counts:")
        print(case_counts_frame.to_string(index=False))

        common_path, aide_path = resolve_runtime_paths(config)
        common = import_file("jpeg_bias_common_runtime", common_path)
        aide = import_file("aide_original_full_runtime", aide_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            print("WARNING: GPU is strongly recommended, especially for Full AIDE.")
        predictions: dict[tuple[str, str], pd.DataFrame] = {}

        if "clip_linear_probe" in requested_models:
            import open_clip

            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained=str(config["clip_pretrained"]),
                device=device,
            )
            clip_head = nn.Linear(512, 2)
            clip_payload = torch.load(
                checkpoint_paths["clip_linear_probe"],
                map_location="cpu",
                weights_only=False,
            )
            clip_head.load_state_dict(common.extract_state_dict(clip_payload, "head"), strict=True)
            clip_head = clip_head.to(device).eval()
            clip_model.eval()

            class CLIPWithHead(nn.Module):
                def __init__(self, backbone: nn.Module, head: nn.Module):
                    super().__init__()
                    self.backbone = backbone
                    self.head = head

                def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                    features = self.backbone.encode_image(inputs)
                    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    return self.head(features)

            clip_combined = CLIPWithHead(clip_model, clip_head).to(device).eval()
            for name in requested_cases:
                frame = case_manifest(name, audited, controlled)
                predictions[("clip_linear_probe", name)] = infer_tensor_case(
                    clip_combined,
                    "clip_linear_probe",
                    checkpoint_provenance["clip_linear_probe"]["sha256"],
                    clip_preprocess,
                    frame,
                    name,
                    config,
                    device,
                    int(config["clip_batch_size"]),
                    output_root,
                )
            del clip_combined, clip_model, clip_head, clip_payload
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if "npr_resnet18" in requested_models:
            npr_model = common.NPRResNet18(2)
            npr_payload = torch.load(
                checkpoint_paths["npr_resnet18"], map_location="cpu", weights_only=False
            )
            npr_model.load_state_dict(common.extract_state_dict(npr_payload), strict=True)
            npr_model = npr_model.to(device).eval()
            npr_preprocess = transforms.Compose(
                [
                    transforms.Resize(int(224 * 1.15)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
            for name in requested_cases:
                frame = case_manifest(name, audited, controlled)
                predictions[("npr_resnet18", name)] = infer_tensor_case(
                    npr_model,
                    "npr_resnet18",
                    checkpoint_provenance["npr_resnet18"]["sha256"],
                    npr_preprocess,
                    frame,
                    name,
                    config,
                    device,
                    int(config["npr_batch_size"]),
                    output_root,
                )
            del npr_model, npr_payload
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if "aide_original_full" in requested_models:
            aide_checkpoint = torch.load(
                checkpoint_paths["aide_original_full"],
                map_location="cpu",
                weights_only=False,
            )
            if aide_checkpoint.get("architecture") != EXPECTED_ARCHITECTURE:
                raise ValueError(
                    f"Unexpected AIDE architecture: {aide_checkpoint.get('architecture')}"
                )
            model_config = dict(aide.DEFAULT_CONFIG)
            model_config.update(aide_checkpoint.get("config", {}))
            model_config.update(
                {
                    "semantic_checkpoint": resolve_aide_semantic_checkpoint(config),
                    "imagenet_resnet_init": False,
                    "resnet_checkpoint": None,
                    "batch_size": int(config["aide_batch_size"]),
                    "num_workers": int(config["num_workers"]),
                }
            )
            aide_model, semantic_provenance = aide.build_model(model_config, device)
            aide.load_trainable_model_state(
                aide_model, aide_checkpoint["model_trainable_state_dict"]
            )
            aide_model.eval()
            save_json(
                output_root / "provenance" / "aide_semantic_backbone.json",
                semantic_provenance,
            )
            for name in requested_cases:
                frame = case_manifest(name, audited, controlled)
                predictions[("aide_original_full", name)] = infer_aide_case(
                    aide_model,
                    checkpoint_provenance["aide_original_full"]["sha256"],
                    frame,
                    name,
                    config,
                    aide,
                    model_config,
                    device,
                    output_root,
                )
            del aide_model, aide_checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary = compile_results(predictions, config, output_root)
        completed = {
            "status": "complete",
            "completed_at_unix": time.time(),
            "output_root": str(output_root),
            "dataset_root": str(dataset_root),
            "split_name": split_name,
            "statistics": statistics,
            "models": requested_models,
            "cases": requested_cases,
            "metrics_summary": str(output_root / "metrics" / "summary.json"),
        }
        save_json(output_root / "status.json", completed)
        return {**completed, "summary": summary}
    except Exception as exc:
        save_json(
            output_root / "status.json",
            {
                "status": "failed",
                "failed_at_unix": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    result = run_evaluation()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
