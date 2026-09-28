r"""Run four trained detectors on a fixed, balanced 2,000-image CommFor benchmark.

The benchmark is deliberately derived from the exact legacy 1,000-image cohort:

* all 100 legacy real images are retained;
* all 900 legacy fake images (nine generators) are retained;
* 900 new real images are added (180 per real source);
* 100 GALIP images are added as the tenth fake generator.

Consequently, the final cohort contains 1,000 real and 1,000 fake images.  The
lossless image cache is stored in a dedicated Modal volume, separate from the
small result files.  Every model writes resumable partial predictions.

Run from the repository root (PowerShell)::

    .\.venv12\Scripts\python.exe -m modal run `
      benchmarks/commfor_balanced_four_models/code/test_four_models_commfor_balanced_modal.py

Resume a named run by passing the same inference id::

    .\.venv12\Scripts\python.exe -m modal run `
      benchmarks/commfor_balanced_four_models/code/test_four_models_commfor_balanced_modal.py `
      --inference-id balanced_2000_v1
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "commfor-balanced-four-models"
GPU_TYPE = "A100-40GB"
OUTPUT_VOLUME_NAME = "commfor-balanced-four-models-outputs"
BENCHMARK_VOLUME_NAME = "commfor-balanced-2000-cache"
LEGACY_CACHE_VOLUME_NAME = "commfor-unseen-three-models-outputs"
HF_CACHE_VOLUME_NAME = "hf-cache"

REMOTE_CODE_ROOT = "/root/HoangHa_Code"
REMOTE_OUTPUT_ROOT = "/outputs/commfor_balanced_four_models"
REMOTE_BENCHMARK_ROOT = "/benchmark-cache/commfor_balanced_2000/v1"
REMOTE_LEGACY_ROOT = (
    "/legacy-cache/commfor_unseen_three_models/commfor_unseen_cache_20260927"
)
REMOTE_HF_HOME = "/hf-cache"
REMOTE_CLIP_CHECKPOINT = "/root/checkpoints/clip/clip_linear_head.pt"
REMOTE_NPR_CHECKPOINT = "/root/checkpoints/npr/npr_resnet18_from_scratch.pt"
REMOTE_FORENSIC_CHECKPOINT = "/root/checkpoints/aide_original_forensic/model.pt"
REMOTE_FULL_AIDE_CHECKPOINT = "/root/checkpoints/aide_original_full/model_trainable.pt"
REMOTE_FULL_AIDE_RUNTIME = (
    f"{REMOTE_CODE_ROOT}/baselines/aide_original_full/"
    "train_aide_original_full_tiny_commfor_kaggle.py"
)
REMOTE_LEGACY_RUNTIME = "/root/runtime/test_three_models_commfor_unseen_modal.py"

EXPECTED_CHECKPOINT_SHA256 = {
    "clip_linear_probe": "788461421cb585bbf489aabc76f3537058cd7753eccf9432d3a792622f738057",
    "npr_resnet18": "328f9f431d378f9528c86966796e9f3ac604a5ce2bf3d307028062706c1537f0",
    "aide_original_forensic_resnet50": "57024d5c0256869f8855a081abf27e6e882f5bb6de61659c5ef6a28afe8d5b39",
    "aide_original_full": "62848895b44255d6a0567754c503a05807ed0a29b8046f3c6956a049844e47c3",
}
EXPECTED_LEGACY_MANIFEST_SHA256 = (
    "50ffc7d91fa8092e4282184cb823439b8784362264b3f2bc2593a608f97dd22f"
)
EXPECTED_FULL_AIDE_ARCHITECTURE = (
    "AIDE full hybrid: OpenCLIP ConvNeXt-XXLarge + DCT/SRM dual ResNet50"
)

LEGACY_GENERATORS = (
    "Firefly_Image2",
    "Firefly_Image3",
    "FLUX-dev",
    "FLUX-schnell",
    "Dalle2",
    "Dalle3",
    "IdeogramV1",
    "IdeogramV2",
    "Imagen3",
)
TARGET_GENERATORS = (*LEGACY_GENERATORS, "GALIP")
GENERATOR_FAMILIES = {
    "Firefly_Image2": "Firefly",
    "Firefly_Image3": "Firefly",
    "FLUX-dev": "FLUX",
    "FLUX-schnell": "FLUX",
    "Dalle2": "DALL-E",
    "Dalle3": "DALL-E",
    "IdeogramV1": "Ideogram",
    "IdeogramV2": "Ideogram",
    "Imagen3": "Imagen",
    "GALIP": "GALIP",
}
TARGET_REAL_SOURCES = ("LAION", "RAISE", "imagenet", "ffhq", "coco")


def find_project_root() -> Path:
    file_path = Path(__file__).resolve()
    for candidate in (Path.cwd(), Path(REMOTE_CODE_ROOT), file_path.parent, *file_path.parents):
        if (candidate / "data_loader" / "__init__.py").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate the HoangHa_Code repository root")


REMOTE_ASSETS_READY = all(
    Path(path).is_file()
    for path in (
        REMOTE_CLIP_CHECKPOINT,
        REMOTE_NPR_CHECKPOINT,
        REMOTE_FORENSIC_CHECKPOINT,
        REMOTE_FULL_AIDE_CHECKPOINT,
        REMOTE_FULL_AIDE_RUNTIME,
        REMOTE_LEGACY_RUNTIME,
    )
)
if REMOTE_ASSETS_READY:
    LOCAL_PROJECT_ROOT = Path(REMOTE_CODE_ROOT)
else:
    LOCAL_PROJECT_ROOT = find_project_root()

LOCAL_DATA_LOADER = LOCAL_PROJECT_ROOT / "data_loader"
LOCAL_NPR_RUNTIME = LOCAL_PROJECT_ROOT / "baselines" / "npr_resnet18"
LOCAL_FULL_AIDE_RUNTIME = (
    LOCAL_PROJECT_ROOT
    / "baselines"
    / "aide_original_full"
    / "train_aide_original_full_tiny_commfor_kaggle.py"
)
LOCAL_LEGACY_RUNTIME = (
    LOCAL_PROJECT_ROOT
    / "benchmarks"
    / "commfor_unseen_three_models"
    / "code"
    / "test_three_models_commfor_unseen_modal.py"
)
LOCAL_CHECKPOINTS = {
    "clip_linear_probe": LOCAL_PROJECT_ROOT
    / "baselines"
    / "clip_linear_probe"
    / "artifacts"
    / "checkpoints"
    / "clip_linear_head.pt",
    "npr_resnet18": LOCAL_PROJECT_ROOT
    / "baselines"
    / "npr_resnet18"
    / "artifacts"
    / "checkpoints"
    / "npr_resnet18_from_scratch.pt",
    "aide_original_forensic_resnet50": LOCAL_PROJECT_ROOT
    / "baselines"
    / "aide_original_forensic_resnet50"
    / "artifacts"
    / "checkpoints"
    / "model.pt",
    "aide_original_full": LOCAL_PROJECT_ROOT
    / "baselines"
    / "aide_original_full"
    / "artifacts"
    / "checkpoints"
    / "model_trainable.pt",
}
REMOTE_CHECKPOINTS = {
    "clip_linear_probe": REMOTE_CLIP_CHECKPOINT,
    "npr_resnet18": REMOTE_NPR_CHECKPOINT,
    "aide_original_forensic_resnet50": REMOTE_FORENSIC_CHECKPOINT,
    "aide_original_full": REMOTE_FULL_AIDE_CHECKPOINT,
}

if not REMOTE_ASSETS_READY:
    required = [LOCAL_DATA_LOADER, LOCAL_NPR_RUNTIME, LOCAL_FULL_AIDE_RUNTIME, LOCAL_LEGACY_RUNTIME]
    required.extend(LOCAL_CHECKPOINTS.values())
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required local assets are missing: {missing}")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "open_clip_torch==2.26.1",
        "datasets",
        "huggingface_hub",
        "pandas<3.0",
        "scikit-learn<1.9",
        "pillow<12.0",
        "tqdm",
    )
    .env(
        {
            "HF_HOME": REMOTE_HF_HOME,
            "HF_DATASETS_CACHE": f"{REMOTE_HF_HOME}/datasets",
            "HUGGINGFACE_HUB_CACHE": f"{REMOTE_HF_HOME}/hub",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

function_mounts: list[Any] = []
if hasattr(image, "add_local_file"):
    if not REMOTE_ASSETS_READY:
        image = image.add_local_dir(str(LOCAL_DATA_LOADER), remote_path=f"{REMOTE_CODE_ROOT}/data_loader")
        image = image.add_local_dir(
            str(LOCAL_NPR_RUNTIME), remote_path=f"{REMOTE_CODE_ROOT}/baselines/npr_resnet18"
        )
        image = image.add_local_file(str(LOCAL_FULL_AIDE_RUNTIME), remote_path=REMOTE_FULL_AIDE_RUNTIME)
        image = image.add_local_file(str(LOCAL_LEGACY_RUNTIME), remote_path=REMOTE_LEGACY_RUNTIME)
        for name, local_path in LOCAL_CHECKPOINTS.items():
            image = image.add_local_file(str(local_path), remote_path=REMOTE_CHECKPOINTS[name])
else:
    if not REMOTE_ASSETS_READY:
        function_mounts = [
            modal.Mount.from_local_dir(LOCAL_DATA_LOADER, remote_path=f"{REMOTE_CODE_ROOT}/data_loader"),
            modal.Mount.from_local_dir(
                LOCAL_NPR_RUNTIME, remote_path=f"{REMOTE_CODE_ROOT}/baselines/npr_resnet18"
            ),
            modal.Mount.from_local_dir(
                LOCAL_FULL_AIDE_RUNTIME.parent,
                remote_path=f"{REMOTE_CODE_ROOT}/baselines/aide_original_full",
            ),
            modal.Mount.from_local_dir(LOCAL_LEGACY_RUNTIME.parent, remote_path="/root/runtime"),
        ]
        for name, local_path in LOCAL_CHECKPOINTS.items():
            function_mounts.append(
                modal.Mount.from_local_dir(local_path.parent, remote_path=str(Path(REMOTE_CHECKPOINTS[name]).parent))
            )

app = modal.App(APP_NAME, image=image)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
benchmark_volume = modal.Volume.from_name(BENCHMARK_VOLUME_NAME, create_if_missing=True)
legacy_cache_volume = modal.Volume.from_name(LEGACY_CACHE_VOLUME_NAME, create_if_missing=False)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

FUNCTION_OPTIONS: dict[str, Any] = {
    "gpu": GPU_TYPE,
    "timeout": 60 * 60 * 24,
    "memory": 65536,
    "secrets": [modal.Secret.from_name("huggingface-secret")],
    "volumes": {
        "/outputs": output_volume,
        "/benchmark-cache": benchmark_volume,
        "/legacy-cache": legacy_cache_volume,
        REMOTE_HF_HOME: hf_cache_volume,
    },
}
if function_mounts:
    FUNCTION_OPTIONS["mounts"] = function_mounts

DEFAULT_CONFIG: dict[str, Any] = {
    "output_root": REMOTE_OUTPUT_ROOT,
    "benchmark_root": REMOTE_BENCHMARK_ROOT,
    "legacy_root": REMOTE_LEGACY_ROOT,
    "dataset_name": "OwensLab/CommunityForensics-Eval",
    "dataset_split": "CompEval",
    "selection_seed": 44,
    "shuffle_buffer_size": 10000,
    "max_scan_records": 500000,
    "threshold": 0.5,
    "resume": True,
    "cache_commit_interval": 25,
    "save_every": 50,
    "clip_batch_size": 32,
    "npr_batch_size": 32,
    "forensic_batch_size": 4,
    "full_aide_batch_size": 4,
    "full_aide_num_workers": 4,
    "npr_image_size": 224,
    "dct_window_size": 32,
    "dct_stride": 16,
    "dct_grade_bands": 6,
    "image_size": 256,
    "semantic_model_name": "convnext_xxlarge",
    "semantic_pretrained_tag": "laion2b_s34b_b82k_augreg_soup",
    "semantic_half_precision": True,
    "verify_image_hashes": False,
}


@app.function(**FUNCTION_OPTIONS)
def evaluate_four_models(inference_id: str, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    import contextlib
    import gc
    import hashlib
    import importlib.util
    import io
    import os
    import random
    import shutil
    import sys

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

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    sys.path.insert(0, REMOTE_CODE_ROOT)
    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    if not inference_id or any(character in inference_id for character in "/\\"):
        raise ValueError("inference_id must be a non-empty path-safe name")
    random.seed(int(config["selection_seed"]))
    np.random.seed(int(config["selection_seed"]))
    torch.manual_seed(int(config["selection_seed"]))
    torch.cuda.manual_seed_all(int(config["selection_seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    print(f"Hugging Face authentication: {'HF_TOKEN is active' if hf_token else 'HF_TOKEN is missing'}")
    device = torch.device("cuda")
    threshold = float(config["threshold"])

    run_dir = Path(config["output_root"]) / inference_id
    benchmark_dir = Path(config["benchmark_root"])
    legacy_dir = Path(config["legacy_root"])
    for subdir in ("dataset", "metrics", "predictions", "provenance"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "image_cache").mkdir(parents=True, exist_ok=True)

    def save_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary.replace(path)

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def import_file(module_name: str, path: str):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_forensic_components():
        """Load only the proven forensic factory, without importing its Modal app."""
        import ast

        source_path = Path(REMOTE_LEGACY_RUNTIME)
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        factory_node = next(
            (
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "_aide_runtime_components"
            ),
            None,
        )
        if factory_node is None:
            raise ImportError(f"Forensic runtime factory not found in {source_path}")
        isolated = ast.Module(body=[factory_node], type_ignores=[])
        ast.fix_missing_locations(isolated)
        namespace: dict[str, Any] = {}
        exec(compile(isolated, str(source_path), "exec"), namespace)
        return namespace["_aide_runtime_components"]()

    def clean(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return str(value)

    def generator_from_record(record: dict[str, Any]) -> str:
        return str(record.get("model_name") or record.get("architecture") or "unknown")

    def record_key(record: dict[str, Any]) -> tuple[str, ...]:
        label = int(record.get("label"))
        return (
            str(label),
            "shared_real" if label == 0 else generator_from_record(record),
            clean(record.get("image_name")),
            clean(record.get("real_source")),
            clean(record.get("architecture")),
            clean(record.get("subset")),
        )

    def row_key(row: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(int(row["label"])),
            clean(row["generator"]),
            clean(row.get("image_name")),
            clean(row.get("real_source")),
            clean(row.get("architecture")),
            clean(row.get("subset")),
        )

    def image_from_record(record: dict[str, Any]) -> Image.Image:
        value = record.get("image_data", record.get("image"))
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, (bytes, bytearray)):
            return Image.open(io.BytesIO(value)).convert("RGB")
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            if value.get("path") is not None:
                return Image.open(value["path"]).convert("RGB")
        if isinstance(value, list):
            return Image.open(io.BytesIO(bytes(value))).convert("RGB")
        raise TypeError(f"Unsupported CommFor image type: {type(value)}")

    def dataset_stream():
        dataset = load_dataset(config["dataset_name"], split=config["dataset_split"], streaming=True)
        return dataset.shuffle(
            seed=int(config["selection_seed"]),
            buffer_size=int(config["shuffle_buffer_size"]),
        )

    legacy_manifest_path = legacy_dir / "dataset" / "commfor_unseen_manifest.csv"
    legacy_index_path = legacy_dir / "dataset" / "image_cache_manifest.csv"
    if not legacy_manifest_path.is_file() or not legacy_index_path.is_file():
        raise FileNotFoundError(
            f"Legacy exact cohort is missing under Modal volume {LEGACY_CACHE_VOLUME_NAME}: {legacy_dir}"
        )
    if sha256_file(legacy_manifest_path) != EXPECTED_LEGACY_MANIFEST_SHA256:
        raise ValueError("Legacy manifest SHA256 mismatch")
    legacy = pd.read_csv(legacy_manifest_path).sort_values("manifest_order").reset_index(drop=True)
    legacy_index = pd.read_csv(legacy_index_path).sort_values("manifest_order").reset_index(drop=True)
    if len(legacy) != 1000 or legacy["label"].value_counts().to_dict() != {1: 900, 0: 100}:
        raise ValueError("Legacy cohort is not the expected 100-real/900-fake cohort")
    if legacy["manifest_order"].astype(int).tolist() != list(range(1000)):
        raise ValueError("Legacy manifest_order must contain 0..999 exactly once")
    if (
        len(legacy_index) != 1000
        or legacy_index["manifest_order"].astype(int).tolist() != list(range(1000))
        or legacy_index["cached_image_path"].astype(str).tolist()
        != legacy["cached_image_path"].astype(str).tolist()
    ):
        raise ValueError("Legacy image-cache index does not match the exact legacy manifest")

    manifest_path = benchmark_dir / "commfor_balanced_2000_manifest.csv"
    cache_index_path = benchmark_dir / "image_cache_manifest.csv"
    complete_path = benchmark_dir / "image_cache_complete.json"
    scan_records = 0
    source_records: dict[int, dict[str, Any]] = {}

    def cache_relative(order: int) -> str:
        return f"image_cache/{order:06d}.png"

    if not manifest_path.is_file() and (cache_index_path.is_file() or complete_path.is_file()):
        raise RuntimeError(
            "Benchmark cache metadata exists but its manifest is missing; use a new "
            "benchmark volume/version instead of silently resampling images"
        )
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path).sort_values("manifest_order").reset_index(drop=True)
        print(f"Loaded fixed balanced benchmark manifest with {len(manifest)} rows")
    else:
        legacy_keys = {row_key(row) for row in legacy.to_dict("records")}
        added_real: dict[str, list[dict[str, Any]]] = {source: [] for source in TARGET_REAL_SOURCES}
        added_fake: list[dict[str, Any]] = []
        selected_keys: set[tuple[str, ...]] = set()
        for scan_records, record in enumerate(dataset_stream(), start=1):
            key = record_key(record)
            if key in legacy_keys or key in selected_keys:
                continue
            label = int(record.get("label"))
            if label == 0:
                source = str(record.get("real_source") or "unknown")
                if source in added_real and len(added_real[source]) < 180:
                    added_real[source].append(record)
                    selected_keys.add(key)
            elif generator_from_record(record) == "GALIP" and len(added_fake) < 100:
                added_fake.append(record)
                selected_keys.add(key)
            if all(len(rows) == 180 for rows in added_real.values()) and len(added_fake) == 100:
                break
            if scan_records >= int(config["max_scan_records"]):
                break
        missing_real = {key: 180 - len(rows) for key, rows in added_real.items() if len(rows) < 180}
        if missing_real or len(added_fake) < 100:
            raise RuntimeError(
                f"Could not fill balanced benchmark after {scan_records} rows: "
                f"missing_real={missing_real}, missing_GALIP={100-len(added_fake)}"
            )

        rows: list[dict[str, Any]] = []
        order = 0
        for old in legacy[legacy["label"] == 0].to_dict("records"):
            row = dict(old)
            row.update(
                {
                    "manifest_order": order,
                    "cohort_origin": "legacy_commfor_unseen_1000",
                    "legacy_manifest_order": int(old["manifest_order"]),
                    "cached_image_path": cache_relative(order),
                }
            )
            rows.append(row)
            order += 1
        for source in TARGET_REAL_SOURCES:
            for source_index, record in enumerate(added_real[source], start=20):
                rows.append(
                    {
                        "manifest_order": order,
                        "selection_group": "balanced_real_addition",
                        "sample_index": source_index,
                        "label": 0,
                        "label_name": "real",
                        "generator": "shared_real",
                        "generator_family": "real",
                        "seen_status": "real_reference",
                        "image_name": record.get("image_name"),
                        "model_name": record.get("model_name"),
                        "architecture": record.get("architecture"),
                        "real_source": record.get("real_source"),
                        "subset": record.get("subset"),
                        "cohort_origin": "balanced_2000_addition",
                        "legacy_manifest_order": -1,
                        "cached_image_path": cache_relative(order),
                    }
                )
                source_records[order] = record
                order += 1
        for old in legacy[legacy["label"] == 1].to_dict("records"):
            row = dict(old)
            row.update(
                {
                    "manifest_order": order,
                    "cohort_origin": "legacy_commfor_unseen_1000",
                    "legacy_manifest_order": int(old["manifest_order"]),
                    "cached_image_path": cache_relative(order),
                }
            )
            rows.append(row)
            order += 1
        for sample_index, record in enumerate(added_fake):
            rows.append(
                {
                    "manifest_order": order,
                    "selection_group": "generator_fake",
                    "sample_index": sample_index,
                    "label": 1,
                    "label_name": "fake",
                    "generator": "GALIP",
                    "generator_family": "GALIP",
                    "seen_status": "unseen_generator_family",
                    "image_name": record.get("image_name"),
                    "model_name": record.get("model_name"),
                    "architecture": record.get("architecture"),
                    "real_source": record.get("real_source"),
                    "subset": record.get("subset"),
                    "cohort_origin": "balanced_2000_addition",
                    "legacy_manifest_order": -1,
                    "cached_image_path": cache_relative(order),
                }
            )
            source_records[order] = record
            order += 1
        manifest = pd.DataFrame(rows)
        manifest.to_csv(manifest_path, index=False)
        benchmark_volume.commit()
        print(f"Created balanced manifest after scanning {scan_records} CommFor rows")

    manifest = manifest.sort_values("manifest_order").reset_index(drop=True)
    if len(manifest) != 2000 or manifest["manifest_order"].astype(int).tolist() != list(range(2000)):
        raise ValueError("Balanced manifest must contain manifest_order 0..1999")
    if manifest["label"].astype(int).value_counts().to_dict() != {0: 1000, 1: 1000}:
        raise ValueError("Balanced manifest must contain exactly 1,000 real and 1,000 fake")
    fake_counts = manifest[manifest["label"] == 1]["generator"].value_counts().to_dict()
    if fake_counts != {generator: 100 for generator in TARGET_GENERATORS}:
        raise ValueError(f"Generator quotas mismatch: {fake_counts}")
    real_counts = manifest[manifest["label"] == 0]["real_source"].value_counts().to_dict()
    if real_counts != {source: 200 for source in TARGET_REAL_SOURCES}:
        raise ValueError(f"Real-source quotas mismatch: {real_counts}")
    if len({row_key(row) for row in manifest.to_dict("records")}) != 2000:
        raise ValueError("Balanced manifest has duplicate composite sample keys")

    manifest_rows = manifest.to_dict("records")
    missing_added = [
        row
        for row in manifest_rows
        if row["cohort_origin"] == "balanced_2000_addition"
        and not (benchmark_dir / row["cached_image_path"]).is_file()
        and int(row["manifest_order"]) not in source_records
    ]
    if missing_added:
        wanted = {row_key(row): row for row in missing_added}
        for scan_records, record in enumerate(dataset_stream(), start=1):
            key = record_key(record)
            if key in wanted:
                order = int(wanted.pop(key)["manifest_order"])
                source_records[order] = record
            if not wanted or scan_records >= int(config["max_scan_records"]):
                break
        if wanted:
            raise RuntimeError(f"Could not recover {len(wanted)} uncached added samples")

    cached_this_run = 0
    for row in tqdm(manifest_rows, desc="Building balanced CommFor cache"):
        destination = benchmark_dir / row["cached_image_path"]
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".png.partial")
        if row["cohort_origin"] == "legacy_commfor_unseen_1000":
            legacy_order = int(row["legacy_manifest_order"])
            old = legacy.iloc[legacy_order]
            source = legacy_dir / str(old["cached_image_path"])
            if not source.is_file():
                raise FileNotFoundError(f"Legacy cached image missing: {source}")
            shutil.copyfile(source, temporary)
        else:
            order = int(row["manifest_order"])
            image_from_record(source_records[order]).save(temporary, format="PNG")
        temporary.replace(destination)
        cached_this_run += 1
        if cached_this_run % int(config["cache_commit_interval"]) == 0:
            benchmark_volume.commit()
    benchmark_volume.commit()

    manifest_sha256 = sha256_file(manifest_path)
    rebuild_index = not cache_index_path.is_file() or not complete_path.is_file()
    if rebuild_index:
        index_rows = []
        for row in tqdm(manifest_rows, desc="Indexing balanced CommFor cache"):
            path = benchmark_dir / row["cached_image_path"]
            with Image.open(path) as cached:
                cached.load()
                width, height = cached.size
                mode = cached.mode
            image_sha256 = sha256_file(path)
            if row["cohort_origin"] == "legacy_commfor_unseen_1000":
                legacy_order = int(row["legacy_manifest_order"])
                expected_legacy_sha256 = str(legacy_index.iloc[legacy_order]["sha256"])
                if image_sha256 != expected_legacy_sha256:
                    raise ValueError(
                        f"Copied legacy image hash mismatch for legacy order {legacy_order}"
                    )
            index_rows.append(
                {
                    "manifest_order": int(row["manifest_order"]),
                    "cached_image_path": row["cached_image_path"],
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "file_size_bytes": path.stat().st_size,
                    "sha256": image_sha256,
                }
            )
        cache_index = pd.DataFrame(index_rows)
        cache_index.to_csv(cache_index_path, index=False)
        save_json(
            complete_path,
            {
                "complete": True,
                "image_count": 2000,
                "manifest_sha256": manifest_sha256,
                "format": "lossless RGB PNG",
                "total_bytes": int(cache_index["file_size_bytes"].sum()),
            },
        )
        benchmark_volume.commit()
    else:
        cache_index = pd.read_csv(cache_index_path).sort_values("manifest_order").reset_index(drop=True)
        with complete_path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
        if marker.get("manifest_sha256") != manifest_sha256 or len(cache_index) != 2000:
            raise ValueError("Existing benchmark cache marker/index belongs to another manifest")

    for index_row, row in tqdm(
        zip(cache_index.to_dict("records"), manifest_rows),
        total=2000,
        desc="Verifying balanced CommFor cache",
    ):
        path = benchmark_dir / row["cached_image_path"]
        if not path.is_file() or path.stat().st_size != int(index_row["file_size_bytes"]):
            raise ValueError(f"Cached image missing or size mismatch: {path}")
        if bool(config["verify_image_hashes"]) and sha256_file(path) != index_row["sha256"]:
            raise ValueError(f"Cached image hash mismatch: {path}")

    manifest["image_path"] = manifest["cached_image_path"].map(lambda value: str(benchmark_dir / value))
    manifest["sample_id"] = manifest["manifest_order"].astype(int).astype(str)
    manifest.to_csv(run_dir / "dataset" / "commfor_balanced_2000_manifest_used.csv", index=False)
    save_json(
        run_dir / "provenance" / "benchmark.json",
        {
            "benchmark_volume": BENCHMARK_VOLUME_NAME,
            "benchmark_root": str(benchmark_dir),
            "manifest_sha256": manifest_sha256,
            "num_samples": 2000,
            "num_real": 1000,
            "num_fake": 1000,
            "real_sources": {source: 200 for source in TARGET_REAL_SOURCES},
            "generators": {generator: 100 for generator in TARGET_GENERATORS},
            "legacy_manifest_sha256": EXPECTED_LEGACY_MANIFEST_SHA256,
            "legacy_samples_retained": 1000,
            "added_samples": 1000,
            "added_generator": "GALIP",
            "records_scanned_this_run": scan_records,
        },
    )

    checkpoint_paths = {name: Path(path) for name, path in REMOTE_CHECKPOINTS.items()}
    checkpoint_info = {}
    for name, path in checkpoint_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != EXPECTED_CHECKPOINT_SHA256[name]:
            raise ValueError(f"{name} checkpoint SHA256 mismatch: {actual_hash}")
        checkpoint_info[name] = {"path": str(path), "sha256": actual_hash}
    save_json(run_dir / "provenance" / "checkpoints.json", checkpoint_info)
    save_json(run_dir / "provenance" / "config.json", config)

    manifest_records = manifest.to_dict("records")
    manifest_by_order = {int(row["manifest_order"]): row for row in manifest_records}

    def load_image(order: int) -> Image.Image:
        with Image.open(manifest_by_order[order]["image_path"]) as image_file:
            return image_file.convert("RGB")

    def extract_state_dict(payload: Any, preferred_key: str | None = None):
        if preferred_key and isinstance(payload, dict) and preferred_key in payload:
            state = payload[preferred_key]
        elif isinstance(payload, torch.nn.Module):
            state = payload.state_dict()
        elif isinstance(payload, dict) and "model_state_dict" in payload:
            state = payload["model_state_dict"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            state = payload["state_dict"]
        elif isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            state = payload["model"]
        elif isinstance(payload, dict):
            state = payload
        else:
            raise TypeError(f"Unsupported checkpoint payload: {type(payload)}")
        if any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        return state

    def base_prediction(order: int, probability: float, model_name: str) -> dict[str, Any]:
        source = manifest_by_order[order]
        return {
            "manifest_order": order,
            "sample_id": f"commfor:{order:06d}",
            "model": model_name,
            "checkpoint_sha256": checkpoint_info[model_name]["sha256"],
            "manifest_sha256": manifest_sha256,
            "label": int(source["label"]),
            "label_name": source["label_name"],
            "generator": source["generator"],
            "generator_family": source["generator_family"],
            "real_source": source.get("real_source", ""),
            "image_name": source.get("image_name", ""),
            "cohort_origin": source["cohort_origin"],
            "fake_probability": probability,
            "predicted_label": int(probability >= threshold),
        }

    def prediction_state(model_name: str) -> tuple[list[dict[str, Any]], Path, Path]:
        final_path = run_dir / "predictions" / f"{model_name}_predictions.csv"
        partial_path = run_dir / "predictions" / f"{model_name}_predictions.partial.csv"
        path = final_path if final_path.is_file() else partial_path
        rows: list[dict[str, Any]] = []
        if bool(config["resume"]) and path.is_file():
            frame = pd.read_csv(path)
            if (
                set(frame["checkpoint_sha256"].astype(str))
                != {checkpoint_info[model_name]["sha256"]}
                or set(frame["manifest_sha256"].astype(str)) != {manifest_sha256}
            ):
                raise ValueError(f"Stale partial predictions found for {model_name}")
            if frame["manifest_order"].duplicated().any():
                raise ValueError(f"Duplicate partial prediction orders for {model_name}")
            rows = frame.sort_values("manifest_order").to_dict("records")
            print(f"Resuming {model_name} from {len(rows)}/2000")
        return rows, partial_path, final_path

    def commit_predictions(rows: list[dict[str, Any]], path: Path) -> None:
        pd.DataFrame(rows).sort_values("manifest_order").to_csv(path, index=False)
        output_volume.commit()

    @torch.inference_mode()
    def predict_clip(existing: list[dict[str, Any]], partial_path: Path) -> list[dict[str, Any]]:
        import open_clip

        done = {int(row["manifest_order"]) for row in existing}
        remaining = [order for order in range(2000) if order not in done]
        if not remaining:
            return existing
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        model.eval()
        head = torch.nn.Linear(512, 2)
        payload = torch.load(checkpoint_paths["clip_linear_probe"], map_location="cpu", weights_only=False)
        head.load_state_dict(extract_state_dict(payload, "head"), strict=True)
        head = head.to(device).eval()
        batch_size = int(config["clip_batch_size"])
        for start in tqdm(range(0, len(remaining), batch_size), desc="CLIP linear probe"):
            orders = remaining[start : start + batch_size]
            inputs = torch.stack([preprocess(load_image(order)) for order in orders]).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                features = model.encode_image(inputs)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                logits = head(features)
            probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            existing.extend(base_prediction(order, float(probability), "clip_linear_probe") for order, probability in zip(orders, probabilities))
            if len(existing) % int(config["save_every"]) < len(orders):
                commit_predictions(existing, partial_path)
        del model, head, payload
        gc.collect()
        torch.cuda.empty_cache()
        return existing

    @torch.inference_mode()
    def predict_npr(existing: list[dict[str, Any]], partial_path: Path) -> list[dict[str, Any]]:
        from baselines.npr_resnet18.npr_resnet18 import NPRResNet18
        from data_loader import build_image_transform

        done = {int(row["manifest_order"]) for row in existing}
        remaining = [order for order in range(2000) if order not in done]
        if not remaining:
            return existing
        payload = torch.load(checkpoint_paths["npr_resnet18"], map_location="cpu", weights_only=False)
        model = NPRResNet18(num_classes=2)
        model.load_state_dict(extract_state_dict(payload), strict=True)
        model = model.to(device).eval()
        transform = build_image_transform(image_size=int(config["npr_image_size"]), train=False)
        batch_size = int(config["npr_batch_size"])
        for start in tqdm(range(0, len(remaining), batch_size), desc="NPR-ResNet18"):
            orders = remaining[start : start + batch_size]
            inputs = torch.stack([transform(load_image(order)) for order in orders]).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(inputs)
            probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            existing.extend(base_prediction(order, float(probability), "npr_resnet18") for order, probability in zip(orders, probabilities))
            if len(existing) % int(config["save_every"]) < len(orders):
                commit_predictions(existing, partial_path)
        del model, payload
        gc.collect()
        torch.cuda.empty_cache()
        return existing

    @torch.inference_mode()
    def predict_forensic(existing: list[dict[str, Any]], partial_path: Path) -> list[dict[str, Any]]:
        Selector, ForensicModel = load_forensic_components()
        done = {int(row["manifest_order"]) for row in existing}
        remaining = [order for order in range(2000) if order not in done]
        if not remaining:
            return existing
        selector = Selector(
            window_size=int(config["dct_window_size"]),
            stride=int(config["dct_stride"]),
            grade_bands=int(config["dct_grade_bands"]),
            output_size=int(config["image_size"]),
        )
        model = ForensicModel(imagenet_init=False)
        payload = torch.load(
            checkpoint_paths["aide_original_forensic_resnet50"],
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(extract_state_dict(payload), strict=True)
        model = model.to(device).eval()
        batch_size = int(config["forensic_batch_size"])
        for start in tqdm(range(0, len(remaining), batch_size), desc="AIDE forensic ResNet50"):
            orders = remaining[start : start + batch_size]
            patches = []
            selections = []
            for order in orders:
                array = np.asarray(load_image(order), dtype=np.float32).copy() / 255.0
                selected, selection = selector(torch.from_numpy(array).permute(2, 0, 1).contiguous())
                patches.append(selected)
                selections.append(selection)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(torch.stack(patches).to(device))
            probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            for order, probability, selection in zip(orders, probabilities, selections):
                row = base_prediction(
                    order,
                    float(probability),
                    "aide_original_forensic_resnet50",
                )
                row.update(selection)
                existing.append(row)
            if len(existing) % int(config["save_every"]) < len(orders):
                commit_predictions(existing, partial_path)
        del model, selector, payload
        gc.collect()
        torch.cuda.empty_cache()
        return existing

    @torch.inference_mode()
    def predict_aide_original_full(
        existing: list[dict[str, Any]], partial_path: Path
    ) -> list[dict[str, Any]]:
        aide = import_file("aide_original_full_runtime", REMOTE_FULL_AIDE_RUNTIME)
        done = {int(row["manifest_order"]) for row in existing}
        remaining = manifest[~manifest["manifest_order"].isin(done)].copy()
        if remaining.empty:
            return existing
        checkpoint = torch.load(
            checkpoint_paths["aide_original_full"],
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("architecture") != EXPECTED_FULL_AIDE_ARCHITECTURE:
            raise ValueError(f"Unexpected Full AIDE architecture: {checkpoint.get('architecture')}")
        checkpoint_config = checkpoint.get("config", {})
        expected = {
            "dct_window_size": int(config["dct_window_size"]),
            "dct_stride": int(config["dct_stride"]),
            "dct_grade_bands": int(config["dct_grade_bands"]),
            "image_size": int(config["image_size"]),
            "semantic_model_name": config["semantic_model_name"],
            "semantic_pretrained_tag": config["semantic_pretrained_tag"],
        }
        for key, value in expected.items():
            if checkpoint_config.get(key) != value:
                raise ValueError(f"Full AIDE checkpoint config mismatch for {key}")
        model_config = dict(aide.DEFAULT_CONFIG)
        model_config.update(
            {
                **expected,
                "batch_size": int(config["full_aide_batch_size"]),
                "num_workers": int(config["full_aide_num_workers"]),
                "semantic_checkpoint": None,
                "semantic_half_precision": bool(config["semantic_half_precision"]),
                "imagenet_resnet_init": False,
                "resnet_checkpoint": None,
            }
        )
        model, semantic_provenance = aide.build_model(model_config, device)
        aide.load_trainable_model_state(model, checkpoint["model_trainable_state_dict"])
        model.eval()
        save_json(
            run_dir / "provenance" / "aide_original_full_semantic_backbone.json",
            semantic_provenance,
        )
        del checkpoint
        hf_cache_volume.commit()
        loader = aide.make_loader(remaining, config=model_config, training=False)
        last_commit = len(existing)
        for batch in tqdm(loader, desc="Full AIDE"):
            inputs = batch["inputs"].to(device, non_blocking=True)
            amp = torch.autocast("cuda", dtype=torch.float16) if bool(config["semantic_half_precision"]) else contextlib.nullcontext()
            with amp:
                logits = model(inputs)
            probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            for index, probability in enumerate(probabilities):
                order = int(batch["sample_id"][index])
                row = base_prediction(order, float(probability), "aide_original_full")
                for key in ("num_candidates", "low_1_score", "high_1_score", "low_2_score", "high_2_score"):
                    row[key] = float(batch[key][index]) if key != "num_candidates" else int(batch[key][index])
                existing.append(row)
            if len(existing) - last_commit >= int(config["save_every"]):
                commit_predictions(existing, partial_path)
                last_commit = len(existing)
        del loader, model, aide
        gc.collect()
        torch.cuda.empty_cache()
        return existing

    predictors = {
        "clip_linear_probe": predict_clip,
        "npr_resnet18": predict_npr,
        "aide_original_forensic_resnet50": predict_forensic,
        "aide_original_full": predict_aide_original_full,
    }

    def compute_metrics(frame: pd.DataFrame) -> dict[str, Any]:
        y_true = frame["label"].to_numpy(dtype=int)
        y_prob = frame["fake_probability"].to_numpy(dtype=float)
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "num_samples": int(len(frame)),
            "num_real": int((y_true == 0).sum()),
            "num_fake": int((y_true == 1).sum()),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "real_recall": float(tn / (tn + fp)),
            "fake_recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "fake_precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "fake_f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "mean_real_fake_probability": float(y_prob[y_true == 0].mean()),
            "mean_fake_fake_probability": float(y_prob[y_true == 1].mean()),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "threshold": threshold,
        }

    metric_columns = (
        "accuracy", "balanced_accuracy", "real_recall", "fake_recall",
        "fake_precision", "fake_f1", "roc_auc", "average_precision",
    )

    def evaluate_model(model_name: str, predictions: pd.DataFrame) -> dict[str, Any]:
        predictions = predictions.sort_values("manifest_order").reset_index(drop=True)
        if len(predictions) != 2000 or predictions["manifest_order"].astype(int).tolist() != list(range(2000)):
            raise ValueError(f"{model_name} predictions do not exactly cover the benchmark")
        overall = compute_metrics(predictions)
        save_json(run_dir / "metrics" / f"{model_name}_overall.json", overall)
        fake = predictions[predictions["label"] == 1]
        all_real = predictions[predictions["label"] == 0]
        legacy_real = all_real[all_real["cohort_origin"] == "legacy_commfor_unseen_1000"]
        if len(legacy_real) != 100:
            raise ValueError("Expected exactly 100 legacy shared-real reference images")
        generator_rows = []
        all_real_rows = []
        for generator in TARGET_GENERATORS:
            generator_fake = fake[fake["generator"] == generator]
            primary = compute_metrics(pd.concat([legacy_real, generator_fake], ignore_index=True))
            primary.update(
                {
                    "model": model_name,
                    "generator": generator,
                    "generator_family": GENERATOR_FAMILIES[generator],
                    "metric_cohort": "100_legacy_real_plus_100_generator_fake",
                }
            )
            generator_rows.append(primary)
            secondary = compute_metrics(pd.concat([all_real, generator_fake], ignore_index=True))
            secondary.update(
                {
                    "model": model_name,
                    "generator": generator,
                    "generator_family": GENERATOR_FAMILIES[generator],
                    "metric_cohort": "1000_real_plus_100_generator_fake",
                }
            )
            all_real_rows.append(secondary)
        generator_metrics = pd.DataFrame(generator_rows)
        all_real_metrics = pd.DataFrame(all_real_rows)
        generator_metrics.to_csv(run_dir / "metrics" / f"{model_name}_generator_metrics.csv", index=False)
        all_real_metrics.to_csv(run_dir / "metrics" / f"{model_name}_generator_metrics_all_real.csv", index=False)
        macro = {
            "model": model_name,
            **{f"macro_generator_{column}": float(generator_metrics[column].mean()) for column in metric_columns},
            "worst_generator_balanced_accuracy": float(generator_metrics["balanced_accuracy"].min()),
            "best_generator_balanced_accuracy": float(generator_metrics["balanced_accuracy"].max()),
            "num_generators": 10,
            "primary_generator_metric_cohort": "100_legacy_real_plus_100_generator_fake",
            "manifest_sha256": manifest_sha256,
            "checkpoint_sha256": checkpoint_info[model_name]["sha256"],
        }
        save_json(run_dir / "metrics" / f"{model_name}_macro_summary.json", macro)
        return {"overall": overall, "macro": macro, "generator_metrics": generator_metrics, "all_real_metrics": all_real_metrics}

    predictions_by_model: dict[str, pd.DataFrame] = {}
    results: dict[str, dict[str, Any]] = {}
    completed_models = []
    for model_name, predictor in predictors.items():
        rows, partial_path, final_path = prediction_state(model_name)
        if len(rows) != 2000:
            rows = predictor(rows, partial_path)
        frame = pd.DataFrame(rows).sort_values("manifest_order").reset_index(drop=True)
        if len(frame) != 2000:
            commit_predictions(rows, partial_path)
            raise RuntimeError(f"{model_name} produced {len(frame)}/2000 predictions")
        frame.to_csv(final_path, index=False)
        frame.to_csv(partial_path, index=False)
        predictions_by_model[model_name] = frame
        results[model_name] = evaluate_model(model_name, frame)
        completed_models.append(model_name)
        save_json(
            run_dir / "status.json",
            {
                "inference_id": inference_id,
                "manifest_sha256": manifest_sha256,
                "completed_models": completed_models,
                "total_models": 4,
                "complete": len(completed_models) == 4,
            },
        )
        output_volume.commit()
        print(f"Completed and committed {model_name}")

    all_generator = pd.concat([value["generator_metrics"] for value in results.values()], ignore_index=True)
    all_generator.to_csv(run_dir / "metrics" / "all_models_generator_metrics.csv", index=False)
    all_generator_all_real = pd.concat([value["all_real_metrics"] for value in results.values()], ignore_index=True)
    all_generator_all_real.to_csv(run_dir / "metrics" / "all_models_generator_metrics_all_real.csv", index=False)
    comparison = pd.DataFrame(
        [
            {
                "model": model_name,
                **{f"overall_{key}": value for key, value in result["overall"].items() if not isinstance(value, list)},
                **result["macro"],
            }
            for model_name, result in results.items()
        ]
    )
    comparison.to_csv(run_dir / "metrics" / "all_models_summary.csv", index=False)
    wide = manifest.drop(columns=["image_path"], errors="ignore").copy()
    for model_name, frame in predictions_by_model.items():
        wide[f"{model_name}_fake_probability"] = frame["fake_probability"].to_numpy()
        wide[f"{model_name}_predicted_label"] = frame["predicted_label"].to_numpy()
    wide.to_csv(run_dir / "predictions" / "all_models_predictions_wide.csv", index=False)

    summary = {
        "inference_id": inference_id,
        "output_root": str(run_dir),
        "benchmark_root": str(benchmark_dir),
        "manifest_sha256": manifest_sha256,
        "manifest_samples": 2000,
        "manifest_real": 1000,
        "manifest_fake": 1000,
        "target_generators": list(TARGET_GENERATORS),
        "target_real_sources": list(TARGET_REAL_SOURCES),
        "primary_generator_metric_cohort": "100_legacy_real_plus_100_generator_fake",
        "models": {
            model_name: {"overall": result["overall"], "macro": result["macro"]}
            for model_name, result in results.items()
        },
    }
    save_json(run_dir / "metrics" / "comparison_summary.json", summary)
    output_volume.commit()
    hf_cache_volume.commit()
    print("\nFour-model balanced benchmark summary:")
    print(
        comparison[
            [
                "model",
                "overall_balanced_accuracy",
                "overall_fake_f1",
                "overall_roc_auc",
                "macro_generator_balanced_accuracy",
                "worst_generator_balanced_accuracy",
            ]
        ].to_string(index=False)
    )
    print(f"Modal output root: {run_dir}")
    return summary


@app.local_entrypoint()
def main(
    inference_id: str = "",
    selection_seed: int = 44,
    max_scan_records: int = 500000,
    clip_batch_size: int = 32,
    npr_batch_size: int = 32,
    forensic_batch_size: int = 4,
    full_aide_batch_size: int = 4,
    full_aide_num_workers: int = 4,
    save_every: int = 50,
    verify_image_hashes: bool = False,
    resume: bool = True,
):
    run_id = inference_id.strip() or time.strftime("%Y%m%d_%H%M%S")
    overrides = {
        "selection_seed": int(selection_seed),
        "max_scan_records": int(max_scan_records),
        "clip_batch_size": int(clip_batch_size),
        "npr_batch_size": int(npr_batch_size),
        "forensic_batch_size": int(forensic_batch_size),
        "full_aide_batch_size": int(full_aide_batch_size),
        "full_aide_num_workers": int(full_aide_num_workers),
        "save_every": int(save_every),
        "verify_image_hashes": bool(verify_image_hashes),
        "resume": bool(resume),
    }
    result = evaluate_four_models.remote(run_id, overrides)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Modal output root: {REMOTE_OUTPUT_ROOT}/{run_id}")
