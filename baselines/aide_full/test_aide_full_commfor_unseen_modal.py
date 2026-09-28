r"""Evaluate the trained Full AIDE checkpoint on the cached CommFor unseen cohort.

This Modal job deliberately reuses the exact 1,000-image lossless cache created by
``test_three_models_commfor_unseen_modal.py``.  It never streams or resamples
CommunityForensics, so its metrics are directly comparable with the existing CLIP,
NPR-ResNet18, and forensic-only AIDE results for the same manifest.

Run from the repository root::

    .\.venv12\Scripts\python.exe -m modal run `
      baselines/aide_full/test_aide_full_commfor_unseen_modal.py

Resume a named run::

    .\.venv12\Scripts\python.exe -m modal run `
      baselines/aide_full/test_aide_full_commfor_unseen_modal.py `
      --inference-id 20260928_full_aide --batch-size 4

The first run may download the frozen OpenCLIP ConvNeXt-XXLarge weights into the
shared ``hf-cache`` volume.  Subsequent runs reuse that cache.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import modal


APP_NAME = "aide-full-commfor-unseen"
GPU_TYPE = "A100-40GB"

OUTPUT_VOLUME_NAME = "aide-full-commfor-unseen-outputs"
COMMFOR_CACHE_VOLUME_NAME = "commfor-unseen-three-models-outputs"
HF_CACHE_VOLUME_NAME = "hf-cache"

REMOTE_CODE_ROOT = "/root/HoangHa_Code"
REMOTE_TRAIN_MODULE = f"{REMOTE_CODE_ROOT}/baselines/aide_full/train_aide_full_tiny_commfor_kaggle.py"
REMOTE_CHECKPOINT = "/root/checkpoints/aide_full/model_trainable.pt"
REMOTE_OUTPUT_ROOT = "/outputs/aide_full_commfor_unseen"
REMOTE_HF_HOME = "/hf-cache"
REMOTE_COMMFOR_RUN_ROOT = (
    "/commfor-cache/commfor_unseen_three_models/commfor_unseen_cache_20260927"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "62848895b44255d6a0567754c503a05807ed0a29b8046f3c6956a049844e47c3"
)
EXPECTED_MANIFEST_SHA256 = (
    "50ffc7d91fa8092e4282184cb823439b8784362264b3f2bc2593a608f97dd22f"
)
EXPECTED_ARCHITECTURE = (
    "AIDE full hybrid: OpenCLIP ConvNeXt-XXLarge + DCT/SRM dual ResNet50"
)

TARGET_GENERATORS = (
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
}

TARGET_REAL_SOURCES = ("LAION", "RAISE", "imagenet", "ffhq", "coco")


def find_project_root() -> Path:
    file_path = Path(__file__).resolve()
    for candidate in (
        Path.cwd(),
        Path(REMOTE_CODE_ROOT),
        file_path.parent,
        *file_path.parents,
    ):
        if (candidate / "baselines" / "aide_full" / "train_aide_full_tiny_commfor_kaggle.py").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate the HoangHa_Code repository root.")


REMOTE_ASSETS_READY = Path(REMOTE_TRAIN_MODULE).is_file() and Path(REMOTE_CHECKPOINT).is_file()
if REMOTE_ASSETS_READY:
    # Modal imports this entrypoint as /root/<filename>.py.  At that point the
    # files added to the image already live at their explicit remote paths, so
    # local repository discovery must not run again inside the container.
    LOCAL_PROJECT_ROOT = Path(REMOTE_CODE_ROOT)
    LOCAL_TRAIN_MODULE = Path(REMOTE_TRAIN_MODULE)
    LOCAL_CHECKPOINT = Path(REMOTE_CHECKPOINT)
else:
    LOCAL_PROJECT_ROOT = find_project_root()
    LOCAL_TRAIN_MODULE = (
        LOCAL_PROJECT_ROOT
        / "baselines"
        / "aide_full"
        / "train_aide_full_tiny_commfor_kaggle.py"
    )
    LOCAL_CHECKPOINT = (
        LOCAL_PROJECT_ROOT
        / "baselines"
        / "aide_full"
        / "check_point"
        / "model_trainable.pt"
    )
    if not LOCAL_TRAIN_MODULE.is_file():
        raise FileNotFoundError(f"Full AIDE runtime not found: {LOCAL_TRAIN_MODULE}")
    if not LOCAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Full AIDE checkpoint not found: {LOCAL_CHECKPOINT}")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "open_clip_torch==2.26.1",
        "huggingface_hub",
        "pandas<3.0",
        "scikit-learn<1.9",
        "pillow<12.0",
        "tqdm",
    )
    .env(
        {
            "HF_HOME": REMOTE_HF_HOME,
            "HUGGINGFACE_HUB_CACHE": f"{REMOTE_HF_HOME}/hub",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

function_mounts: list[Any] = []
if hasattr(image, "add_local_file"):
    if not REMOTE_ASSETS_READY:
        image = image.add_local_file(str(LOCAL_TRAIN_MODULE), remote_path=REMOTE_TRAIN_MODULE)
        image = image.add_local_file(str(LOCAL_CHECKPOINT), remote_path=REMOTE_CHECKPOINT)
else:
    # Compatibility fallback for older Modal SDKs.
    if not REMOTE_ASSETS_READY:
        function_mounts = [
            modal.Mount.from_local_dir(
                LOCAL_TRAIN_MODULE.parent,
                remote_path=f"{REMOTE_CODE_ROOT}/baselines/aide_full",
            ),
            modal.Mount.from_local_dir(
                LOCAL_CHECKPOINT.parent,
                remote_path="/root/checkpoints/aide_full",
            ),
        ]


app = modal.App(APP_NAME, image=image)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
commfor_cache_volume = modal.Volume.from_name(
    COMMFOR_CACHE_VOLUME_NAME, create_if_missing=False
)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

FUNCTION_OPTIONS: dict[str, Any] = {
    "gpu": GPU_TYPE,
    "timeout": 60 * 60 * 24,
    "memory": 65536,
    "volumes": {
        "/outputs": output_volume,
        "/commfor-cache": commfor_cache_volume,
        REMOTE_HF_HOME: hf_cache_volume,
    },
}
if function_mounts:
    FUNCTION_OPTIONS["mounts"] = function_mounts


DEFAULT_CONFIG: dict[str, Any] = {
    "output_root": REMOTE_OUTPUT_ROOT,
    "checkpoint_path": REMOTE_CHECKPOINT,
    "commfor_run_root": REMOTE_COMMFOR_RUN_ROOT,
    "manifest_path": f"{REMOTE_COMMFOR_RUN_ROOT}/dataset/commfor_unseen_manifest.csv",
    "cache_index_path": f"{REMOTE_COMMFOR_RUN_ROOT}/dataset/image_cache_manifest.csv",
    "cache_complete_path": f"{REMOTE_COMMFOR_RUN_ROOT}/dataset/image_cache_complete.json",
    "batch_size": 4,
    "num_workers": 4,
    "save_every": 25,
    "resume": True,
    "threshold": 0.5,
    "random_seed": 42,
    "dct_window_size": 32,
    "dct_stride": 16,
    "dct_grade_bands": 6,
    "image_size": 256,
    "semantic_model_name": "convnext_xxlarge",
    "semantic_pretrained_tag": "laion2b_s34b_b82k_augreg_soup",
    "semantic_half_precision": True,
    "verify_image_hashes": True,
}


@app.function(**FUNCTION_OPTIONS)
def infer_full_aide_commfor_unseen(
    inference_id: str,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import contextlib
    import gc
    import hashlib
    import importlib.util
    import os
    import shutil
    import time

    import numpy as np
    import pandas as pd
    import torch
    from tqdm.auto import tqdm

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

    def load_aide_runtime():
        module_path = Path(REMOTE_TRAIN_MODULE)
        if not module_path.is_file():
            raise FileNotFoundError(f"Full AIDE runtime not mounted: {module_path}")
        spec = importlib.util.spec_from_file_location("aide_full_runtime", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import Full AIDE runtime: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def metrics_with_probability(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
        result = aide.compute_metrics(frame, threshold=threshold)
        real = frame[frame["label"] == 0]["fake_probability"]
        fake = frame[frame["label"] == 1]["fake_probability"]
        result["mean_real_fake_probability"] = float(real.mean()) if len(real) else None
        result["mean_fake_fake_probability"] = float(fake.mean()) if len(fake) else None
        return result

    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    if not inference_id or any(character in inference_id for character in "/\\"):
        raise ValueError("inference_id must be a non-empty path-safe name")
    if int(config["batch_size"]) < 1:
        raise ValueError("batch_size must be >= 1")
    if int(config["num_workers"]) < 0:
        raise ValueError("num_workers must be >= 0")
    if int(config["save_every"]) < 1:
        raise ValueError("save_every must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    aide = load_aide_runtime()
    aide.seed_everything(int(config["random_seed"]))
    device = torch.device("cuda")
    run_dir = Path(str(config["output_root"])) / inference_id
    for subdirectory in ("dataset", "metrics", "predictions", "provenance"):
        (run_dir / subdirectory).mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(str(config["checkpoint_path"]))
    manifest_path = Path(str(config["manifest_path"]))
    cache_complete_path = Path(str(config["cache_complete_path"]))
    cache_index_path = Path(str(config["cache_index_path"]))
    commfor_run_root = Path(str(config["commfor_run_root"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Full AIDE checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Cached CommFor manifest not found: {manifest_path}. "
            f"Expected Modal volume '{COMMFOR_CACHE_VOLUME_NAME}'."
        )
    if not cache_complete_path.is_file():
        raise FileNotFoundError(f"CommFor cache completion marker not found: {cache_complete_path}")
    if not cache_index_path.is_file():
        raise FileNotFoundError(f"CommFor image-cache index not found: {cache_index_path}")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"Checkpoint SHA256 mismatch: {checkpoint_sha256}; "
            f"expected {EXPECTED_CHECKPOINT_SHA256}"
        )
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"Manifest SHA256 mismatch: {manifest_sha256}; "
            f"expected {EXPECTED_MANIFEST_SHA256}"
        )

    with cache_complete_path.open("r", encoding="utf-8") as handle:
        cache_complete = json.load(handle)
    if not bool(cache_complete.get("complete")):
        raise RuntimeError("CommFor image cache is not marked complete")
    if int(cache_complete.get("image_count", -1)) != 1000:
        raise RuntimeError(f"Expected 1,000 cached images, got {cache_complete.get('image_count')}")
    if cache_complete.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Cache completion marker belongs to a different manifest")

    manifest = pd.read_csv(manifest_path).sort_values("manifest_order").reset_index(drop=True)
    required_columns = {
        "manifest_order",
        "label",
        "label_name",
        "generator",
        "generator_family",
        "real_source",
        "image_name",
        "cached_image_path",
    }
    missing_columns = sorted(required_columns.difference(manifest.columns))
    if missing_columns:
        raise ValueError(f"CommFor manifest is missing columns: {missing_columns}")
    if len(manifest) != 1000:
        raise ValueError(f"Expected 1,000 manifest rows, got {len(manifest)}")
    if manifest["manifest_order"].astype(int).tolist() != list(range(1000)):
        raise ValueError("manifest_order must contain each integer from 0 through 999 exactly once")
    label_counts = manifest["label"].astype(int).value_counts().to_dict()
    if label_counts != {1: 900, 0: 100}:
        raise ValueError(f"Expected 100 real and 900 fake rows, got {label_counts}")
    fake_counts = (
        manifest[manifest["label"].astype(int) == 1]["generator"].value_counts().to_dict()
    )
    expected_fake_counts = {generator: 100 for generator in TARGET_GENERATORS}
    if fake_counts != expected_fake_counts:
        raise ValueError(
            f"CommFor generator cohort mismatch: {fake_counts}; expected {expected_fake_counts}"
        )
    real_counts = (
        manifest[manifest["label"].astype(int) == 0]["real_source"].value_counts().to_dict()
    )
    expected_real_counts = {source: 20 for source in TARGET_REAL_SOURCES}
    if real_counts != expected_real_counts:
        raise ValueError(
            f"CommFor real-source cohort mismatch: {real_counts}; expected {expected_real_counts}"
        )

    manifest["image_path"] = manifest["cached_image_path"].map(
        lambda relative: str(commfor_run_root / str(relative))
    )
    missing_images = [path for path in manifest["image_path"] if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"CommFor cache is missing {len(missing_images)} images; first={missing_images[0]}"
        )

    cache_index = pd.read_csv(cache_index_path).sort_values("manifest_order").reset_index(drop=True)
    required_cache_columns = {
        "manifest_order",
        "cached_image_path",
        "file_size_bytes",
        "sha256",
    }
    missing_cache_columns = sorted(required_cache_columns.difference(cache_index.columns))
    if missing_cache_columns:
        raise ValueError(f"Image-cache index is missing columns: {missing_cache_columns}")
    if len(cache_index) != 1000:
        raise ValueError(f"Expected 1,000 image-cache index rows, got {len(cache_index)}")
    if cache_index["manifest_order"].astype(int).tolist() != list(range(1000)):
        raise ValueError("Image-cache index orders do not match the exact 1,000-image manifest")
    if cache_index["cached_image_path"].astype(str).tolist() != manifest[
        "cached_image_path"
    ].astype(str).tolist():
        raise ValueError("Image-cache index paths do not match the CommFor manifest")

    verify_hashes = bool(config["verify_image_hashes"])
    for index_row, image_path in tqdm(
        zip(cache_index.to_dict("records"), manifest["image_path"].tolist()),
        total=1000,
        desc="Verifying exact CommFor image cache",
    ):
        path = Path(image_path)
        expected_size = int(index_row["file_size_bytes"])
        if path.stat().st_size != expected_size:
            raise ValueError(
                f"Cached image size mismatch for {path}: "
                f"actual={path.stat().st_size}, expected={expected_size}"
            )
        if verify_hashes:
            actual_image_sha256 = sha256_file(path)
            if actual_image_sha256 != str(index_row["sha256"]):
                raise ValueError(
                    f"Cached image SHA256 mismatch for manifest_order="
                    f"{int(index_row['manifest_order'])}"
                )
    manifest["sample_id"] = manifest["manifest_order"].astype(int).astype(str)
    manifest.to_csv(run_dir / "dataset" / "commfor_unseen_manifest_used.csv", index=False)

    save_json(
        run_dir / "provenance" / "cohort.json",
        {
            "source_volume": COMMFOR_CACHE_VOLUME_NAME,
            "source_run_root": str(commfor_run_root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "image_cache_manifest_sha256": sha256_file(cache_index_path),
            "image_hashes_verified": verify_hashes,
            "num_samples": 1000,
            "num_real": 100,
            "num_fake": 900,
            "generators": list(TARGET_GENERATORS),
            "real_sources": list(TARGET_REAL_SOURCES),
            "metric_cohort": "generator_fake_plus_shared_real",
        },
    )
    save_json(run_dir / "provenance" / "config.json", config)

    partial_path = run_dir / "predictions" / "commfor_predictions.partial.csv"
    prediction_rows: list[dict[str, Any]] = []
    if bool(config["resume"]) and partial_path.is_file():
        partial = pd.read_csv(partial_path)
        required_partial_columns = {
            "manifest_order",
            "checkpoint_sha256",
            "manifest_sha256",
        }
        missing_partial_columns = sorted(required_partial_columns.difference(partial.columns))
        if missing_partial_columns:
            raise ValueError(
                f"Existing partial prediction file is missing provenance columns: "
                f"{missing_partial_columns}"
            )
        if set(partial["checkpoint_sha256"].astype(str)) != {checkpoint_sha256}:
            raise ValueError("Existing partial predictions were produced by another checkpoint")
        if set(partial["manifest_sha256"].astype(str)) != {manifest_sha256}:
            raise ValueError("Existing partial predictions belong to another CommFor manifest")
        partial["manifest_order"] = partial["manifest_order"].astype(int)
        if partial["manifest_order"].duplicated().any():
            raise ValueError("Existing partial prediction file contains duplicate manifest_order")
        valid_orders = set(manifest["manifest_order"].astype(int).tolist())
        invalid_orders = set(partial["manifest_order"].tolist()).difference(valid_orders)
        if invalid_orders:
            raise ValueError(f"Partial prediction file has invalid orders: {sorted(invalid_orders)}")
        prediction_rows = partial.to_dict("records")
        print(f"Resuming {inference_id} from {len(prediction_rows)}/1000 completed images.")
    elif partial_path.is_file():
        partial_path.unlink()

    completed_orders = {int(row["manifest_order"]) for row in prediction_rows}
    remaining = manifest[~manifest["manifest_order"].isin(completed_orders)].copy()
    started_at = time.time()

    if not remaining.empty:
        print(
            f"Loading Full AIDE checkpoint and frozen OpenCLIP trunk; "
            f"remaining={len(remaining)}, batch_size={config['batch_size']}"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_trainable_state_dict" not in checkpoint:
            raise TypeError("Expected a Full AIDE trainable-state checkpoint dictionary")
        if checkpoint.get("architecture") != EXPECTED_ARCHITECTURE:
            raise ValueError(f"Unexpected checkpoint architecture: {checkpoint.get('architecture')}")

        checkpoint_config = checkpoint.get("config", {})
        expected_checkpoint_config = {
            "dct_window_size": int(config["dct_window_size"]),
            "dct_stride": int(config["dct_stride"]),
            "dct_grade_bands": int(config["dct_grade_bands"]),
            "image_size": int(config["image_size"]),
            "semantic_model_name": str(config["semantic_model_name"]),
            "semantic_pretrained_tag": str(config["semantic_pretrained_tag"]),
        }
        for key, expected in expected_checkpoint_config.items():
            actual = checkpoint_config.get(key)
            if actual != expected:
                raise ValueError(
                    f"Checkpoint config mismatch for {key}: actual={actual!r}, expected={expected!r}"
                )

        model_config = dict(aide.DEFAULT_CONFIG)
        model_config.update(
            {
                "random_seed": int(config["random_seed"]),
                "batch_size": int(config["batch_size"]),
                "num_workers": int(config["num_workers"]),
                "dct_window_size": int(config["dct_window_size"]),
                "dct_stride": int(config["dct_stride"]),
                "dct_grade_bands": int(config["dct_grade_bands"]),
                "image_size": int(config["image_size"]),
                "semantic_model_name": str(config["semantic_model_name"]),
                "semantic_pretrained_tag": str(config["semantic_pretrained_tag"]),
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
            run_dir / "provenance" / "checkpoint.json",
            {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
                "architecture": checkpoint.get("architecture"),
                "epoch": int(checkpoint.get("epoch", -1)),
                "optimizer_step": int(checkpoint.get("optimizer_step", -1)),
                "best_val_balanced_accuracy": float(
                    checkpoint.get("best_val_balanced_accuracy", float("nan"))
                ),
                "frozen_semantic_backbone_included": bool(
                    checkpoint.get("frozen_semantic_backbone_included", False)
                ),
                "semantic_backbone": semantic_provenance,
            },
        )
        del checkpoint
        gc.collect()
        hf_cache_volume.commit()

        loader = aide.make_loader(remaining, config=model_config, training=False)
        manifest_by_order = {
            int(row["manifest_order"]): row for row in manifest.to_dict("records")
        }
        last_saved_count = len(prediction_rows)
        with torch.inference_mode():
            for batch in tqdm(loader, desc="Full AIDE on cached CommFor unseen"):
                inputs = batch["inputs"].to(device, non_blocking=True)
                amp = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with amp:
                    logits = model(inputs)
                probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
                for index, probability in enumerate(probabilities):
                    order = int(batch["sample_id"][index])
                    source_row = manifest_by_order[order]
                    probability_float = float(probability)
                    prediction_rows.append(
                        {
                            "manifest_order": order,
                            "sample_id": f"commfor:{order:06d}",
                            "checkpoint_sha256": checkpoint_sha256,
                            "manifest_sha256": manifest_sha256,
                            "selection_group": source_row.get("selection_group", ""),
                            "sample_index": int(source_row.get("sample_index", 0)),
                            "label": int(source_row["label"]),
                            "label_name": str(source_row["label_name"]),
                            "generator": str(source_row["generator"]),
                            "generator_family": str(source_row["generator_family"]),
                            "image_name": str(source_row.get("image_name", "")),
                            "model_name": str(source_row.get("model_name", "")),
                            "architecture": str(source_row.get("architecture", "")),
                            "real_source": str(source_row.get("real_source", "")),
                            "subset": str(source_row.get("subset", "")),
                            "fake_probability": probability_float,
                            "predicted_label": int(
                                probability_float >= float(config["threshold"])
                            ),
                            "num_candidates": int(batch["num_candidates"][index]),
                            "low_1_score": float(batch["low_1_score"][index]),
                            "high_1_score": float(batch["high_1_score"][index]),
                            "low_2_score": float(batch["low_2_score"][index]),
                            "high_2_score": float(batch["high_2_score"][index]),
                        }
                    )
                if len(prediction_rows) - last_saved_count >= int(config["save_every"]):
                    pd.DataFrame(prediction_rows).sort_values("manifest_order").to_csv(
                        partial_path, index=False
                    )
                    save_json(
                        run_dir / "status.json",
                        {
                            "inference_id": inference_id,
                            "status": "running",
                            "completed_samples": len(prediction_rows),
                            "total_samples": 1000,
                            "checkpoint_sha256": checkpoint_sha256,
                            "manifest_sha256": manifest_sha256,
                        },
                    )
                    output_volume.commit()
                    last_saved_count = len(prediction_rows)
                del inputs, logits

        del loader, model
        gc.collect()
        torch.cuda.empty_cache()

    predictions = pd.DataFrame(prediction_rows)
    if len(predictions) != 1000:
        predictions.sort_values("manifest_order").to_csv(partial_path, index=False)
        output_volume.commit()
        raise RuntimeError(f"Expected 1,000 predictions, got {len(predictions)}")
    predictions["manifest_order"] = predictions["manifest_order"].astype(int)
    if predictions["manifest_order"].duplicated().any():
        raise ValueError("Predictions contain duplicate manifest_order values")
    predictions = predictions.sort_values("manifest_order").reset_index(drop=True)
    if predictions["manifest_order"].tolist() != list(range(1000)):
        raise ValueError("Predictions do not cover every manifest order from 0 through 999")

    final_predictions_path = run_dir / "predictions" / "commfor_predictions.csv"
    predictions.to_csv(final_predictions_path, index=False)
    predictions.to_csv(partial_path, index=False)

    threshold = float(config["threshold"])
    overall = metrics_with_probability(predictions, threshold)
    shared_real = predictions[predictions["label"].astype(int) == 0]
    fake = predictions[predictions["label"].astype(int) == 1]

    generator_rows: list[dict[str, Any]] = []
    for generator in TARGET_GENERATORS:
        generator_fake = fake[fake["generator"] == generator]
        if len(shared_real) != 100 or len(generator_fake) != 100:
            raise AssertionError(
                f"Invalid metric cohort for {generator}: "
                f"real={len(shared_real)}, fake={len(generator_fake)}"
            )
        cohort = pd.concat([shared_real, generator_fake], ignore_index=True)
        generator_rows.append(
            {
                "generator": generator,
                "generator_family": GENERATOR_FAMILIES[generator],
                "metric_cohort": "generator_fake_plus_shared_real",
                **metrics_with_probability(cohort, threshold),
            }
        )
    generator_metrics = pd.DataFrame(generator_rows)

    family_rows: list[dict[str, Any]] = []
    for family in dict.fromkeys(GENERATOR_FAMILIES.values()):
        family_fake = fake[fake["generator_family"] == family]
        cohort = pd.concat([shared_real, family_fake], ignore_index=True)
        family_rows.append(
            {
                "generator_family": family,
                "num_generator_versions": int(
                    sum(value == family for value in GENERATOR_FAMILIES.values())
                ),
                "metric_cohort": "family_fake_plus_shared_real",
                **metrics_with_probability(cohort, threshold),
            }
        )
    family_metrics = pd.DataFrame(family_rows)

    macro = aide.macro_summary(generator_metrics)
    for column in (
        "accuracy",
        "balanced_accuracy",
        "real_recall",
        "fake_recall",
        "fake_precision",
        "fake_f1",
        "roc_auc",
        "average_precision",
    ):
        macro[f"macro_family_{column}"] = float(family_metrics[column].mean())
    macro.update(
        {
            "num_generator_families": int(len(family_metrics)),
            "manifest_sha256": manifest_sha256,
            "checkpoint_sha256": checkpoint_sha256,
        }
    )

    elapsed_seconds = float(time.time() - started_at)
    summary = {
        "inference_id": inference_id,
        "model": "aide_full",
        "architecture": EXPECTED_ARCHITECTURE,
        "gpu": GPU_TYPE,
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "threshold": threshold,
        "elapsed_seconds_this_invocation": elapsed_seconds,
        "manifest_samples": 1000,
        "manifest_real": 100,
        "manifest_fake": 900,
        "target_generators": list(TARGET_GENERATORS),
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "metric_cohort": "generator_fake_plus_shared_real",
        "overall": overall,
        "macro": macro,
    }

    save_json(run_dir / "metrics" / "overall_metrics.json", overall)
    generator_metrics.to_csv(run_dir / "metrics" / "generator_metrics.csv", index=False)
    family_metrics.to_csv(run_dir / "metrics" / "family_metrics.csv", index=False)
    save_json(run_dir / "metrics" / "macro_summary.json", macro)
    save_json(run_dir / "summary.json", summary)
    save_json(
        run_dir / "status.json",
        {
            "inference_id": inference_id,
            "status": "complete",
            "completed_samples": 1000,
            "total_samples": 1000,
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_sha256": manifest_sha256,
        },
    )
    output_volume.commit()
    hf_cache_volume.commit()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


@app.local_entrypoint()
def main(
    inference_id: str = "",
    batch_size: int = 4,
    num_workers: int = 4,
    save_every: int = 25,
    threshold: float = 0.5,
    resume: bool = True,
    verify_image_hashes: bool = True,
):
    resolved_inference_id = inference_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    overrides = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "save_every": int(save_every),
        "threshold": float(threshold),
        "resume": bool(resume),
        "verify_image_hashes": bool(verify_image_hashes),
    }
    result = infer_full_aide_commfor_unseen.remote(resolved_inference_id, overrides)
    print(f"Modal output root: {REMOTE_OUTPUT_ROOT}/{resolved_inference_id}")
    print(
        json.dumps(
            {
                "inference_id": result["inference_id"],
                "overall": result["overall"],
                "macro": result["macro"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
