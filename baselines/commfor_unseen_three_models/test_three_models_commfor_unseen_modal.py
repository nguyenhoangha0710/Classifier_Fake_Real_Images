"""Evaluate three trained detectors on one fixed unseen-family CommFor cohort.

The script performs inference only. It creates one deterministic manifest with:

* 100 real images: 20 each from LAION, RAISE, ImageNet, FFHQ, and COCO;
* 900 fake images: 100 from each of nine generator versions;
* no Midjourney or Stable-Diffusion/LCM derivative used by Tiny-GenImage train;
* five unseen product families: Firefly, FLUX, DALL-E, Ideogram, and Imagen.

The exact same 1,000 records are evaluated by:

1. frozen OpenAI CLIP ViT-B/32 + the trained 512->2 linear head;
2. the trained NPR-ResNet18 full checkpoint;
3. the trained AIDE forensic-only dual-ResNet50 checkpoint.

Outputs include a lossless PNG cache of the selected images, the manifest,
per-image predictions, unique-sample metrics, per-generator metrics, per-family
metrics, macro summaries, and a three-model comparison. Once the image cache is
complete, resume never has to read CommFor from Hugging Face again. The manifest,
cache batches, and each completed model are committed separately to the Modal
volume so an interrupted run can resume with the same inference ID.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal


APP_NAME = "commfor-unseen-three-models"
GPU_TYPE = "A100-40GB"
OUTPUT_VOLUME_NAME = "commfor-unseen-three-models-outputs"
HF_CACHE_VOLUME_NAME = "hf-cache"

REMOTE_CODE_ROOT = "/root/HoangHa_Code"
REMOTE_OUTPUT_ROOT = "/outputs/commfor_unseen_three_models"
REMOTE_HF_HOME = "/hf-cache"
REMOTE_CLIP_CHECKPOINT = "/root/checkpoints/clip/clip_linear_head.pt"
REMOTE_NPR_CHECKPOINT = "/root/checkpoints/npr/npr_resnet18_from_scratch.pt"
REMOTE_AIDE_CHECKPOINT = "/root/checkpoints/aide/model.pt"

CLIP_SOURCE_RUN_ID = "20260906_100207"
NPR_SOURCE_RUN_ID = "20260906_122613"
AIDE_SOURCE_RUN_ID = "20260927_001615"

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

# These are the seven fake-image lineages present in Combined Tiny-GenImage.
TRAINED_GENERATOR_FAMILIES = (
    "BigGAN",
    "VQDM",
    "Stable Diffusion v1.5",
    "Wukong",
    "ADM",
    "GLIDE",
    "Midjourney",
)


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
LOCAL_CLIP_CHECKPOINT = (
    LOCAL_PROJECT_ROOT
    / "modal_results"
    / "tiny_combined_to_commfor_eval"
    / "clip_linear_head"
    / CLIP_SOURCE_RUN_ID
    / "checkpoints"
    / "clip_linear_head.pt"
)
LOCAL_NPR_CHECKPOINT = (
    LOCAL_PROJECT_ROOT
    / "modal_results"
    / "tiny_combined_to_commfor_eval"
    / "npr_resnet18_from_scratch"
    / NPR_SOURCE_RUN_ID
    / "checkpoints"
    / "npr_resnet18_from_scratch.pt"
)
LOCAL_AIDE_CHECKPOINT = (
    LOCAL_PROJECT_ROOT
    / "modal_results"
    / "aide_forensic_resnet50_download"
    / AIDE_SOURCE_RUN_ID
    / "combined"
    / "checkpoints"
    / "best"
    / "model.pt"
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "open_clip_torch",
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
    image = image.add_local_dir(
        str(LOCAL_DATA_LOADER_DIR), remote_path=f"{REMOTE_CODE_ROOT}/data_loader"
    )
    image = image.add_local_dir(
        str(LOCAL_NPR_DIR), remote_path=f"{REMOTE_CODE_ROOT}/baselines/npr_resnet18"
    )
    checkpoint_pairs = (
        (LOCAL_CLIP_CHECKPOINT, REMOTE_CLIP_CHECKPOINT, "CLIP linear-head"),
        (LOCAL_NPR_CHECKPOINT, REMOTE_NPR_CHECKPOINT, "NPR-ResNet18"),
        (LOCAL_AIDE_CHECKPOINT, REMOTE_AIDE_CHECKPOINT, "AIDE forensic"),
    )
    for local_path, remote_path, label in checkpoint_pairs:
        if not Path(remote_path).is_file():
            if not local_path.is_file():
                raise FileNotFoundError(f"{label} checkpoint not found: {local_path}")
            image = image.add_local_file(str(local_path), remote_path=remote_path)
else:
    for local_path, label in (
        (LOCAL_CLIP_CHECKPOINT, "CLIP linear-head"),
        (LOCAL_NPR_CHECKPOINT, "NPR-ResNet18"),
        (LOCAL_AIDE_CHECKPOINT, "AIDE forensic"),
    ):
        if not local_path.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {local_path}")
    function_mounts = [
        modal.Mount.from_local_dir(
            LOCAL_DATA_LOADER_DIR, remote_path=f"{REMOTE_CODE_ROOT}/data_loader"
        ),
        modal.Mount.from_local_dir(
            LOCAL_NPR_DIR, remote_path=f"{REMOTE_CODE_ROOT}/baselines/npr_resnet18"
        ),
        modal.Mount.from_local_dir(LOCAL_CLIP_CHECKPOINT.parent, remote_path="/root/checkpoints/clip"),
        modal.Mount.from_local_dir(LOCAL_NPR_CHECKPOINT.parent, remote_path="/root/checkpoints/npr"),
        modal.Mount.from_local_dir(LOCAL_AIDE_CHECKPOINT.parent, remote_path="/root/checkpoints/aide"),
    ]

app = modal.App(APP_NAME, image=image)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

FUNCTION_OPTIONS = {
    "gpu": GPU_TYPE,
    "timeout": 60 * 60 * 24,
    "memory": 65536,
    "volumes": {
        "/outputs": output_volume,
        REMOTE_HF_HOME: hf_cache_volume,
    },
}
if function_mounts:
    FUNCTION_OPTIONS["mounts"] = function_mounts

DEFAULT_CONFIG = {
    "output_root": REMOTE_OUTPUT_ROOT,
    "clip_checkpoint_path": REMOTE_CLIP_CHECKPOINT,
    "npr_checkpoint_path": REMOTE_NPR_CHECKPOINT,
    "aide_checkpoint_path": REMOTE_AIDE_CHECKPOINT,
    "commfor_dataset_name": "OwensLab/CommunityForensics-Eval",
    "commfor_split": "CompEval",
    "streaming": True,
    "selection_seed": 43,
    "shuffle_buffer_size": 1000,
    "max_scan_records": 500000,
    "fake_per_generator": 100,
    "real_per_source": 20,
    "target_generators": list(TARGET_GENERATORS),
    "target_real_sources": list(TARGET_REAL_SOURCES),
    "generator_families": dict(GENERATOR_FAMILIES),
    "clip_model_name": "ViT-B-32",
    "clip_pretrained": "openai",
    "clip_batch_size": 32,
    "npr_batch_size": 32,
    "npr_image_size": 224,
    "dct_window_size": 32,
    "dct_stride": 16,
    "dct_grade_bands": 6,
    "aide_patch_size": 256,
    "cache_commit_interval": 25,
    "threshold": 0.5,
    "resume": True,
}


def _aide_runtime_components():
    """Build the exact inference components used by the trained AIDE checkpoint."""

    import math
    from typing import Any

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import ResNet50_Weights, resnet50

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def dct_matrix(size: int) -> torch.Tensor:
        matrix = []
        for i in range(size):
            scale = math.sqrt(1.0 / size) if i == 0 else math.sqrt(2.0 / size)
            matrix.append(
                [scale * math.cos((j + 0.5) * math.pi * i / size) for j in range(size)]
            )
        return torch.tensor(matrix, dtype=torch.float32)

    class AIDEDCTPatchSelector(nn.Module):
        def __init__(
            self,
            window_size: int = 32,
            stride: int = 16,
            grade_bands: int = 6,
            output_size: int = 256,
        ):
            super().__init__()
            self.window_size = int(window_size)
            self.stride = int(stride)
            self.grade_bands = int(grade_bands)
            self.output_size = int(output_size)
            self.register_buffer("dct", dct_matrix(self.window_size), persistent=False)

            coordinates = torch.arange(self.window_size)
            diagonal_index = coordinates[:, None] + coordinates[None, :]
            masks = []
            counts = []
            for band in range(self.grade_bands):
                start = self.window_size * 2.0 / self.grade_bands * band
                end = self.window_size * 2.0 / self.grade_bands * (band + 1)
                mask = ((diagonal_index >= start) & (diagonal_index <= end)).float()
                masks.append(mask)
                counts.append(mask.sum())
            self.register_buffer("band_masks", torch.stack(masks), persistent=False)
            self.register_buffer("band_counts", torch.stack(counts), persistent=False)
            self.register_buffer(
                "band_weights",
                torch.tensor([2.0**band for band in range(self.grade_bands)]),
                persistent=False,
            )

        def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"Expected RGB tensor [3,H,W], got {tuple(image.shape)}")
            if min(image.shape[-2:]) < self.window_size:
                raise ValueError(
                    f"Image {tuple(image.shape[-2:])} is smaller than patch size {self.window_size}."
                )

            columns = F.unfold(
                image.unsqueeze(0), kernel_size=self.window_size, stride=self.stride
            ).squeeze(0).transpose(0, 1)
            patches = columns.reshape(-1, 3, self.window_size, self.window_size)
            dct_coefficients = self.dct @ patches @ self.dct.transpose(0, 1)
            log_magnitude = torch.log(torch.abs(dct_coefficients) + 1.0)

            band_values = []
            for band in range(self.grade_bands):
                value = (
                    log_magnitude
                    * self.band_masks[band].view(1, 1, self.window_size, self.window_size)
                ).sum(dim=(1, 2, 3)) / self.band_counts[band]
                band_values.append(value)
            band_values_tensor = torch.stack(band_values, dim=1)
            scores = (band_values_tensor * self.band_weights.view(1, -1)).sum(dim=1)
            sorted_indices = torch.argsort(scores)
            if len(sorted_indices) < 2:
                raise ValueError("AIDE selection requires at least two candidate patches.")

            selected_indices = torch.stack(
                [sorted_indices[0], sorted_indices[-1], sorted_indices[1], sorted_indices[-2]]
            )
            selected_dct = dct_coefficients.index_select(0, selected_indices)
            selected = self.dct.transpose(0, 1) @ selected_dct @ self.dct
            selected = F.interpolate(
                selected,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            selected = (selected - imagenet_mean.to(selected)) / imagenet_std.to(selected)
            selected_scores = scores.index_select(0, selected_indices)
            return selected.contiguous(), {
                "num_candidates": int(len(scores)),
                "low_1_score": float(selected_scores[0]),
                "high_1_score": float(selected_scores[1]),
                "low_2_score": float(selected_scores[2]),
                "high_2_score": float(selected_scores[3]),
            }

    def normalized_srm_kernels() -> torch.Tensor:
        f1 = [
            [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
            [[0, 1, 0], [0, -1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
            [[0, 0, 0], [1, -1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
            [[0, 0, 0], [0, -1, 0], [1, 0, 0]],
            [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
            [[0, 0, 0], [0, -1, 0], [0, 0, 1]],
        ]
        f2 = [
            [[1, 0, 0], [0, -2, 0], [0, 0, 1]],
            [[0, 1, 0], [0, -2, 0], [0, 1, 0]],
            [[0, 0, 1], [0, -2, 0], [1, 0, 0]],
            [[0, 0, 0], [1, -2, 1], [0, 0, 0]],
        ]
        f3 = [
            [[-1, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, -3, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]],
            [[0, 0, -1, 0, 0], [0, 0, 3, 0, 0], [0, 0, -3, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, -1], [0, 0, 0, 3, 0], [0, 0, -3, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -3, 3, -1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -3, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, -1]],
            [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -3, 0, 0], [0, 0, 3, 0, 0], [0, 0, -1, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -3, 0, 0], [0, 3, 0, 0, 0], [-1, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 3, -3, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ]
        edge3 = [
            [[-1, 2, -1], [2, -4, 2], [0, 0, 0]],
            [[0, 2, -1], [0, -4, 2], [0, 2, -1]],
            [[0, 0, 0], [2, -4, 2], [-1, 2, -1]],
            [[-1, 2, 0], [2, -4, 0], [-1, 2, 0]],
        ]
        edge5 = [
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, -2, 2, -1], [0, 0, 8, -6, 2], [0, 0, -12, 8, -2], [0, 0, 8, -6, 2], [0, 0, -2, 2, -1]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            [[-1, 2, -2, 0, 0], [2, -6, 8, 0, 0], [-2, 8, -12, 0, 0], [2, -6, 8, 0, 0], [-1, 2, -2, 0, 0]],
        ]
        square3 = [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]
        square5 = [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ]

        kernels: list[torch.Tensor] = []
        for group, divisor in (
            (f1, 1.0),
            (f2, 2.0),
            (f3, 3.0),
            (edge3, 4.0),
            (edge5, 12.0),
            ([square3], 4.0),
            ([square5], 12.0),
        ):
            for values in group:
                kernel = torch.tensor(values, dtype=torch.float32) / divisor
                if kernel.shape == (3, 3):
                    kernel = F.pad(kernel, (1, 1, 1, 1))
                kernels.append(kernel)
        output = torch.stack(kernels)
        if output.shape != (30, 5, 5):
            raise AssertionError(f"Expected 30 SRM kernels, got {tuple(output.shape)}")
        return output

    class AIDEHPF(nn.Module):
        def __init__(self):
            super().__init__()
            kernels = normalized_srm_kernels().view(30, 1, 5, 5).repeat(1, 3, 1, 1)
            self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
            self.hpf.weight = nn.Parameter(kernels, requires_grad=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.hpf(inputs)

    def make_resnet50_encoder(imagenet_init: bool) -> nn.Module:
        weights = ResNet50_Weights.IMAGENET1K_V1 if imagenet_init else None
        backbone = resnet50(weights=weights, zero_init_residual=True)
        backbone.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        backbone.fc = nn.Identity()
        return backbone

    class AIDEForensicResNet50(nn.Module):
        def __init__(self, imagenet_init: bool = False):
            super().__init__()
            self.hpf = AIDEHPF()
            self.model_min = make_resnet50_encoder(imagenet_init)
            self.model_max = make_resnet50_encoder(imagenet_init)
            self.classifier = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, 2),
            )

        def forward(self, patches: torch.Tensor) -> torch.Tensor:
            if patches.ndim != 5 or patches.shape[1] != 4 or patches.shape[2] != 3:
                raise ValueError(f"Expected [B,4,3,H,W], got {tuple(patches.shape)}")
            low_1 = self.model_min(self.hpf(patches[:, 0]))
            high_1 = self.model_max(self.hpf(patches[:, 1]))
            low_2 = self.model_min(self.hpf(patches[:, 2]))
            high_2 = self.model_max(self.hpf(patches[:, 3]))
            forensic = (low_1 + high_1 + low_2 + high_2) / 4.0
            return self.classifier(forensic)

    return AIDEDCTPatchSelector, AIDEForensicResNet50


@app.function(**FUNCTION_OPTIONS)
def evaluate_three_models(inference_id: str, config_overrides: dict | None = None) -> dict:
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
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    config["target_generators"] = list(config["target_generators"])
    config["target_real_sources"] = list(config["target_real_sources"])
    config["generator_families"] = dict(config["generator_families"])

    random.seed(int(config["selection_seed"]))
    np.random.seed(int(config["selection_seed"]))
    torch.manual_seed(int(config["selection_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["selection_seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    threshold = float(config["threshold"])
    run_dir = Path(config["output_root"]) / inference_id
    for subdir in ["dataset", "metrics", "predictions", "provenance"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    final_summary_path = run_dir / "metrics" / "comparison_summary.json"
    prediction_paths = {
        "clip_linear_probe": run_dir / "predictions" / "clip_linear_probe_predictions.csv",
        "npr_resnet18": run_dir / "predictions" / "npr_resnet18_predictions.csv",
        "aide_forensic_resnet50": run_dir / "predictions" / "aide_forensic_resnet50_predictions.csv",
    }
    final_outputs_ready = bool(config["resume"]) and final_summary_path.is_file() and all(
        path.is_file() for path in prediction_paths.values()
    )

    def save_json(path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def clean(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return str(value)

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
            clean(record.get("subset")),
        )

    def manifest_key(row: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(int(row["label"])),
            clean(row["generator"]),
            clean(row["image_name"]),
            clean(row["real_source"]),
            clean(row["architecture"]),
            clean(row["subset"]),
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

    def dataset_stream():
        dataset = load_dataset(
            config["commfor_dataset_name"],
            split=config["commfor_split"],
            streaming=bool(config["streaming"]),
        )
        if bool(config["streaming"]):
            return dataset.shuffle(
                seed=int(config["selection_seed"]),
                buffer_size=int(config["shuffle_buffer_size"]),
            )
        return dataset.shuffle(seed=int(config["selection_seed"]))

    expected_total = (
        len(config["target_generators"]) * int(config["fake_per_generator"])
        + len(config["target_real_sources"]) * int(config["real_per_source"])
    )
    manifest_path = run_dir / "dataset" / "commfor_unseen_manifest.csv"
    cache_dir = run_dir / "dataset" / "image_cache"
    cache_index_path = run_dir / "dataset" / "image_cache_manifest.csv"
    cache_complete_path = run_dir / "dataset" / "image_cache_complete.json"
    selection_policy_path = run_dir / "dataset" / "selection_policy.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def relative_cache_path(manifest_order: int) -> str:
        return f"dataset/image_cache/{manifest_order:06d}.png"

    def absolute_cache_path(row: dict[str, Any]) -> Path:
        return run_dir / str(row["cached_image_path"])

    def cache_file_is_ready(row: dict[str, Any]) -> bool:
        path = absolute_cache_path(row)
        return path.is_file() and path.stat().st_size > 0

    previous_selection_policy: dict[str, Any] = {}
    if selection_policy_path.is_file():
        with open(selection_policy_path, "r", encoding="utf-8") as handle:
            previous_selection_policy = json.load(handle)

    records_by_order: dict[int, dict[str, Any]] = {}
    records_scanned_this_run = 0
    selection_records_scanned = int(previous_selection_policy.get("records_scanned", 0))

    if manifest_path.is_file() and bool(config["resume"]):
        manifest = pd.read_csv(manifest_path).sort_values("manifest_order").reset_index(drop=True)
        if len(manifest) != expected_total:
            raise RuntimeError(
                f"Existing manifest has {len(manifest)} rows, expected {expected_total}. "
                "Use a new inference ID for a different cohort configuration."
            )
        expected_cache_paths = [
            relative_cache_path(int(order)) for order in manifest["manifest_order"]
        ]
        if "cached_image_path" not in manifest.columns or manifest[
            "cached_image_path"
        ].astype(str).tolist() != expected_cache_paths:
            manifest["cached_image_path"] = expected_cache_paths
            manifest.to_csv(manifest_path, index=False)

        manifest_rows = manifest.to_dict("records")
        all_wanted = {manifest_key(row): row for row in manifest_rows}
        if len(all_wanted) != len(manifest):
            raise ValueError("Existing manifest has non-unique composite keys.")
        missing_cache_rows = [row for row in manifest_rows if not cache_file_is_ready(row)]
        if missing_cache_rows:
            wanted_missing = {manifest_key(row): row for row in missing_cache_rows}
            found_records: dict[tuple[str, ...], dict[str, Any]] = {}
            for records_scanned_this_run, record in enumerate(dataset_stream(), start=1):
                key = record_key(record)
                if key in wanted_missing and key not in found_records:
                    found_records[key] = record
                if len(found_records) == len(wanted_missing):
                    break
                if records_scanned_this_run >= int(config["max_scan_records"]):
                    break
            missing_keys = set(wanted_missing).difference(found_records)
            if missing_keys:
                raise RuntimeError(
                    f"Could not recover {len(missing_keys)} uncached manifest records after "
                    f"scanning {records_scanned_this_run} rows."
                )
            for key, record in found_records.items():
                order = int(wanted_missing[key]["manifest_order"])
                records_by_order[order] = record
            print(
                f"Recovered {len(found_records)} uncached records from the existing manifest "
                f"after scanning {records_scanned_this_run} CommFor rows."
            )
        else:
            print(
                f"Loaded complete {len(manifest)}-image cache; Hugging Face dataset scan skipped."
            )
    else:
        selected_real: dict[str, list[dict[str, Any]]] = {
            source: [] for source in config["target_real_sources"]
        }
        selected_fake: dict[str, list[dict[str, Any]]] = {
            generator: [] for generator in config["target_generators"]
        }
        for records_scanned_this_run, record in enumerate(dataset_stream(), start=1):
            label = int(record.get("label"))
            if label == 0:
                source = str(record.get("real_source") or "unknown")
                if source in selected_real and len(selected_real[source]) < int(
                    config["real_per_source"]
                ):
                    selected_real[source].append(record)
            else:
                generator = generator_from_record(record)
                if generator in selected_fake and len(selected_fake[generator]) < int(
                    config["fake_per_generator"]
                ):
                    selected_fake[generator].append(record)

            real_complete = all(
                len(records) >= int(config["real_per_source"])
                for records in selected_real.values()
            )
            fake_complete = all(
                len(records) >= int(config["fake_per_generator"])
                for records in selected_fake.values()
            )
            if real_complete and fake_complete:
                break
            if records_scanned_this_run >= int(config["max_scan_records"]):
                break

        real_missing = {
            source: int(config["real_per_source"]) - len(records)
            for source, records in selected_real.items()
            if len(records) < int(config["real_per_source"])
        }
        fake_missing = {
            generator: int(config["fake_per_generator"]) - len(records)
            for generator, records in selected_fake.items()
            if len(records) < int(config["fake_per_generator"])
        }
        if real_missing or fake_missing:
            raise RuntimeError(
                f"CommFor quotas not filled after {records_scanned_this_run} rows: "
                f"real_missing={real_missing}, fake_missing={fake_missing}"
            )

        manifest_rows: list[dict[str, Any]] = []
        manifest_order = 0
        for source in config["target_real_sources"]:
            for source_index, record in enumerate(selected_real[source]):
                manifest_rows.append(
                    {
                        "manifest_order": manifest_order,
                        "selection_group": "shared_real_reference",
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
                        "cached_image_path": relative_cache_path(manifest_order),
                    }
                )
                records_by_order[manifest_order] = record
                manifest_order += 1
        for generator in config["target_generators"]:
            for generator_index, record in enumerate(selected_fake[generator]):
                manifest_rows.append(
                    {
                        "manifest_order": manifest_order,
                        "selection_group": "generator_fake",
                        "sample_index": generator_index,
                        "label": 1,
                        "label_name": "fake",
                        "generator": generator,
                        "generator_family": config["generator_families"][generator],
                        "seen_status": "unseen_generator_family",
                        "image_name": record.get("image_name"),
                        "model_name": record.get("model_name"),
                        "architecture": record.get("architecture"),
                        "real_source": record.get("real_source"),
                        "subset": record.get("subset"),
                        "cached_image_path": relative_cache_path(manifest_order),
                    }
                )
                records_by_order[manifest_order] = record
                manifest_order += 1

        manifest = pd.DataFrame(manifest_rows)
        if len(manifest) != expected_total:
            raise AssertionError(f"Built {len(manifest)} rows, expected {expected_total}.")
        keys = [manifest_key(row) for row in manifest.to_dict("records")]
        if len(set(keys)) != len(keys):
            raise ValueError("Selected cohort has non-unique composite keys.")
        manifest.to_csv(manifest_path, index=False)
        selection_records_scanned = records_scanned_this_run
        print(
            f"Created {len(manifest)}-row unseen-family manifest after scanning "
            f"{records_scanned_this_run} rows."
        )

    # Persist the manifest before materializing images. If image caching is
    # interrupted, the next run recovers only the cache entries still missing.
    output_volume.commit()

    manifest = manifest.sort_values("manifest_order").reset_index(drop=True)
    manifest_rows = manifest.to_dict("records")
    missing_before_cache = sum(not cache_file_is_ready(row) for row in manifest_rows)
    cached_this_run = 0
    commit_interval = max(1, int(config["cache_commit_interval"]))
    for row in tqdm(manifest_rows, desc="Caching selected CommFor images"):
        if cache_file_is_ready(row):
            continue
        order = int(row["manifest_order"])
        if order not in records_by_order:
            raise RuntimeError(f"Missing source record for uncached manifest order {order}.")
        image = image_from_record(records_by_order[order])
        cache_path = absolute_cache_path(row)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".png.partial")
        image.save(temporary_path, format="PNG")
        temporary_path.replace(cache_path)
        cached_this_run += 1
        if cached_this_run % commit_interval == 0:
            save_json(
                run_dir / "status.json",
                {
                    "inference_id": inference_id,
                    "manifest_ready": True,
                    "image_cache_ready": False,
                    "cached_images": int(
                        sum(cache_file_is_ready(item) for item in manifest_rows)
                    ),
                    "expected_cached_images": len(manifest_rows),
                    "completed_models": [],
                },
            )
            output_volume.commit()

    incomplete_cache_rows = [row for row in manifest_rows if not cache_file_is_ready(row)]
    if incomplete_cache_rows:
        raise RuntimeError(
            f"Image cache is incomplete: {len(incomplete_cache_rows)} of "
            f"{len(manifest_rows)} images are missing."
        )

    cache_index_rows = []
    for row in tqdm(manifest_rows, desc="Indexing cached CommFor images"):
        cache_path = absolute_cache_path(row)
        with Image.open(cache_path) as cached_image:
            cached_image.load()
            width, height = cached_image.size
            mode = cached_image.mode
        cache_index_rows.append(
            {
                "manifest_order": int(row["manifest_order"]),
                "cached_image_path": row["cached_image_path"],
                "width": int(width),
                "height": int(height),
                "mode": mode,
                "file_size_bytes": int(cache_path.stat().st_size),
                "sha256": sha256_file(cache_path),
            }
        )
    pd.DataFrame(cache_index_rows).to_csv(cache_index_path, index=False)

    manifest_sha256 = sha256_file(manifest_path)
    cache_total_bytes = int(sum(row["file_size_bytes"] for row in cache_index_rows))
    save_json(
        cache_complete_path,
        {
            "complete": True,
            "image_count": len(cache_index_rows),
            "format": "lossless RGB PNG",
            "total_bytes": cache_total_bytes,
            "manifest_sha256": manifest_sha256,
            "cached_this_run": cached_this_run,
            "missing_before_cache": missing_before_cache,
        },
    )
    save_json(
        run_dir / "status.json",
        {
            "inference_id": inference_id,
            "manifest_ready": True,
            "manifest_sha256": manifest_sha256,
            "image_cache_ready": True,
            "cached_images": len(cache_index_rows),
            "expected_cached_images": len(manifest_rows),
            "completed_models": [],
        },
    )
    output_volume.commit()
    records_by_order.clear()

    def load_cached_image(index: int) -> Image.Image:
        path = absolute_cache_path(manifest_rows[index])
        with Image.open(path) as cached_image:
            return cached_image.convert("RGB")

    manifest_records = manifest.to_dict("records")

    cohort_rows = []
    for generator in config["target_generators"]:
        cohort_rows.append(
            {
                "cohort_type": "generator",
                "cohort": generator,
                "generator_family": config["generator_families"][generator],
                "num_real_reference": len(config["target_real_sources"])
                * int(config["real_per_source"]),
                "num_fake": int(config["fake_per_generator"]),
            }
        )
    for family in dict.fromkeys(config["generator_families"].values()):
        family_generators = [
            generator
            for generator in config["target_generators"]
            if config["generator_families"][generator] == family
        ]
        cohort_rows.append(
            {
                "cohort_type": "family",
                "cohort": family,
                "generator_family": family,
                "num_real_reference": len(config["target_real_sources"])
                * int(config["real_per_source"]),
                "num_fake": len(family_generators) * int(config["fake_per_generator"]),
            }
        )
    pd.DataFrame(cohort_rows).to_csv(run_dir / "dataset" / "cohort_summary.csv", index=False)

    selection_policy = {
        "definition": "unseen product/model lineage relative to Combined Tiny-GenImage",
        "trained_generator_families": list(TRAINED_GENERATOR_FAMILIES),
        "target_generators": config["target_generators"],
        "generator_families": config["generator_families"],
        "target_real_sources": config["target_real_sources"],
        "fake_per_generator": int(config["fake_per_generator"]),
        "real_per_source": int(config["real_per_source"]),
        "selection_seed": int(config["selection_seed"]),
        "manifest_sha256": manifest_sha256,
        "records_scanned": int(selection_records_scanned),
        "records_scanned_this_run": int(records_scanned_this_run),
        "image_cache": {
            "format": "lossless RGB PNG",
            "image_count": len(cache_index_rows),
            "total_bytes": cache_total_bytes,
            "path": "dataset/image_cache",
            "index": "dataset/image_cache_manifest.csv",
        },
        "excluded_lineages": [
            "all Midjourney versions and derivatives",
            "Stable Diffusion and direct LCM/LoRA/DeciDiffusion derivatives",
        ],
        "note": (
            "Two versions per unseen family are retained where available; family-level metrics are "
            "reported so those duplicated versions do not dominate interpretation."
        ),
    }
    save_json(run_dir / "dataset" / "selection_policy.json", selection_policy)
    save_json(run_dir / "config.json", config)

    checkpoint_info = {
        "clip_linear_probe": {
            "source_run_id": CLIP_SOURCE_RUN_ID,
            "path": config["clip_checkpoint_path"],
            "sha256": sha256_file(Path(config["clip_checkpoint_path"])),
            "checkpoint_type": "clip_linear_head_only",
            "encoder": f"{config['clip_model_name']}:{config['clip_pretrained']}",
        },
        "npr_resnet18": {
            "source_run_id": NPR_SOURCE_RUN_ID,
            "path": config["npr_checkpoint_path"],
            "sha256": sha256_file(Path(config["npr_checkpoint_path"])),
            "checkpoint_type": "full_model",
        },
        "aide_forensic_resnet50": {
            "source_run_id": AIDE_SOURCE_RUN_ID,
            "path": config["aide_checkpoint_path"],
            "sha256": sha256_file(Path(config["aide_checkpoint_path"])),
            "checkpoint_type": "full_model_state_dict",
        },
    }
    save_json(run_dir / "provenance" / "checkpoints.json", checkpoint_info)
    if final_outputs_ready:
        with open(final_summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["image_cache"] = {
            "ready": True,
            "image_count": len(cache_index_rows),
            "total_bytes": cache_total_bytes,
            "format": "lossless RGB PNG",
        }
        save_json(final_summary_path, summary)
        save_json(
            run_dir / "status.json",
            {
                "inference_id": inference_id,
                "manifest_ready": True,
                "manifest_sha256": manifest_sha256,
                "image_cache_ready": True,
                "cached_images": len(cache_index_rows),
                "completed_models": list(prediction_paths),
                "complete": True,
            },
        )
        output_volume.commit()
        print(
            f"Run {inference_id} predictions were already complete; image cache is ready, "
            "returning saved summary."
        )
        return summary
    save_json(
        run_dir / "status.json",
        {
            "inference_id": inference_id,
            "manifest_ready": True,
            "manifest_sha256": manifest_sha256,
            "image_cache_ready": True,
            "cached_images": len(cache_index_rows),
            "completed_models": [],
        },
    )
    output_volume.commit()

    def extract_state_dict(payload: Any, preferred_key: str | None = None):
        if preferred_key and isinstance(payload, dict) and preferred_key in payload:
            state_dict = payload[preferred_key]
        elif isinstance(payload, torch.nn.Module):
            state_dict = payload.state_dict()
        elif isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        elif isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            state_dict = payload["model"]
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise TypeError(f"Unsupported checkpoint payload: {type(payload)}")
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        return state_dict

    @torch.no_grad()
    def predict_clip() -> pd.DataFrame:
        import open_clip

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            config["clip_model_name"],
            pretrained=config["clip_pretrained"],
            device=device,
        )
        clip_model.eval()
        for parameter in clip_model.parameters():
            parameter.requires_grad = False

        head = torch.nn.Linear(512, 2)
        payload = torch.load(config["clip_checkpoint_path"], map_location="cpu", weights_only=False)
        head.load_state_dict(extract_state_dict(payload, preferred_key="head"), strict=True)
        head = head.to(device).eval()

        rows: list[dict[str, Any]] = []
        batch_tensors: list[torch.Tensor] = []
        batch_metadata: list[tuple[int, int, int]] = []

        def flush() -> None:
            if not batch_tensors:
                return
            images = torch.stack(batch_tensors).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                features = clip_model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                logits = head(features)
            probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            for (index, width, height), probability in zip(batch_metadata, probabilities):
                row = dict(manifest_records[index])
                row.update(
                    {
                        "sample_id": str(row.get("image_name") or f"commfor:{index}"),
                        "native_width": width,
                        "native_height": height,
                        "fake_probability": float(probability),
                        "predicted_label": int(float(probability) >= threshold),
                        "model": "clip_linear_probe",
                    }
                )
                rows.append(row)
            batch_tensors.clear()
            batch_metadata.clear()

        for index in tqdm(range(len(manifest_records)), desc="CLIP linear probe"):
            pil_image = load_cached_image(index)
            batch_tensors.append(clip_preprocess(pil_image))
            batch_metadata.append((index, int(pil_image.width), int(pil_image.height)))
            if len(batch_tensors) >= int(config["clip_batch_size"]):
                flush()
        flush()
        del head, clip_model, payload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return pd.DataFrame(rows).sort_values("manifest_order").reset_index(drop=True)

    @torch.no_grad()
    def predict_npr() -> pd.DataFrame:
        from baselines.npr_resnet18.npr_resnet18 import NPRResNet18
        from data_loader import build_image_transform

        payload = torch.load(config["npr_checkpoint_path"], map_location="cpu", weights_only=False)
        model = NPRResNet18(num_classes=2)
        model.load_state_dict(extract_state_dict(payload), strict=True)
        model = model.to(device).eval()
        transform = build_image_transform(image_size=int(config["npr_image_size"]), train=False)

        rows: list[dict[str, Any]] = []
        batch_tensors: list[torch.Tensor] = []
        batch_metadata: list[tuple[int, int, int]] = []

        def flush() -> None:
            if not batch_tensors:
                return
            images = torch.stack(batch_tensors).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
            probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            for (index, width, height), probability in zip(batch_metadata, probabilities):
                row = dict(manifest_records[index])
                row.update(
                    {
                        "sample_id": str(row.get("image_name") or f"commfor:{index}"),
                        "native_width": width,
                        "native_height": height,
                        "fake_probability": float(probability),
                        "predicted_label": int(float(probability) >= threshold),
                        "model": "npr_resnet18",
                    }
                )
                rows.append(row)
            batch_tensors.clear()
            batch_metadata.clear()

        for index in tqdm(range(len(manifest_records)), desc="NPR-ResNet18"):
            pil_image = load_cached_image(index)
            batch_tensors.append(transform(pil_image))
            batch_metadata.append((index, int(pil_image.width), int(pil_image.height)))
            if len(batch_tensors) >= int(config["npr_batch_size"]):
                flush()
        flush()
        del model, payload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return pd.DataFrame(rows).sort_values("manifest_order").reset_index(drop=True)

    @torch.no_grad()
    def predict_aide() -> pd.DataFrame:
        AIDEDCTPatchSelector, AIDEForensicResNet50 = _aide_runtime_components()
        selector = AIDEDCTPatchSelector(
            window_size=int(config["dct_window_size"]),
            stride=int(config["dct_stride"]),
            grade_bands=int(config["dct_grade_bands"]),
            output_size=int(config["aide_patch_size"]),
        )
        model = AIDEForensicResNet50(imagenet_init=False)
        payload = torch.load(config["aide_checkpoint_path"], map_location="cpu", weights_only=True)
        model.load_state_dict(extract_state_dict(payload), strict=True)
        model = model.to(device).eval()

        rows = []
        for index in tqdm(range(len(manifest_records)), desc="AIDE forensic ResNet50"):
            pil_image = load_cached_image(index)
            array = np.asarray(pil_image, dtype=np.float32).copy() / 255.0
            image_tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
            patches, selection = selector(image_tensor)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(patches.unsqueeze(0).to(device, non_blocking=True))
            probability = float(torch.softmax(logits.float(), dim=-1)[0, 1].cpu())
            row = dict(manifest_records[index])
            row.update(
                {
                    "sample_id": str(row.get("image_name") or f"commfor:{index}"),
                    "native_width": int(pil_image.width),
                    "native_height": int(pil_image.height),
                    "fake_probability": probability,
                    "predicted_label": int(probability >= threshold),
                    "model": "aide_forensic_resnet50",
                    **selection,
                }
            )
            rows.append(row)
        del model, selector, payload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return pd.DataFrame(rows).sort_values("manifest_order").reset_index(drop=True)

    def compute_metrics(frame: pd.DataFrame) -> dict[str, Any]:
        y_true = frame["label"].to_numpy(dtype=int)
        y_prob = frame["fake_probability"].to_numpy(dtype=float)
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
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
            "fake_precision": float(
                precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            ),
            "fake_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "mean_real_fake_probability": float(y_prob[y_true == 0].mean()),
            "mean_fake_fake_probability": float(y_prob[y_true == 1].mean()),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "threshold": threshold,
        }

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

    def evaluate_predictions(model_name: str, predictions: pd.DataFrame) -> dict[str, Any]:
        if len(predictions) != len(manifest):
            raise AssertionError(
                f"{model_name} produced {len(predictions)} rows; expected {len(manifest)}."
            )
        expected_orders = manifest["manifest_order"].to_numpy(dtype=int)
        actual_orders = predictions["manifest_order"].to_numpy(dtype=int)
        if not np.array_equal(expected_orders, actual_orders):
            raise AssertionError(f"{model_name} prediction order does not match manifest.")

        overall = compute_metrics(predictions)
        save_json(run_dir / "metrics" / f"{model_name}_overall.json", overall)

        shared_real = predictions[predictions["label"] == 0]
        generator_rows = []
        for generator in config["target_generators"]:
            generator_fake = predictions[
                (predictions["label"] == 1) & (predictions["generator"] == generator)
            ]
            cohort = pd.concat([shared_real, generator_fake], ignore_index=True)
            row = compute_metrics(cohort)
            row.update(
                {
                    "model": model_name,
                    "generator": generator,
                    "generator_family": config["generator_families"][generator],
                    "metric_cohort": "generator_fake_plus_shared_real",
                }
            )
            generator_rows.append(row)
        generator_metrics = pd.DataFrame(generator_rows)
        generator_metrics.to_csv(
            run_dir / "metrics" / f"{model_name}_generator_metrics.csv", index=False
        )

        family_rows = []
        for family in dict.fromkeys(config["generator_families"].values()):
            family_fake = predictions[
                (predictions["label"] == 1)
                & (predictions["generator_family"] == family)
            ]
            cohort = pd.concat([shared_real, family_fake], ignore_index=True)
            row = compute_metrics(cohort)
            row.update(
                {
                    "model": model_name,
                    "generator_family": family,
                    "num_generator_versions": int(family_fake["generator"].nunique()),
                    "metric_cohort": "family_fake_plus_shared_real",
                }
            )
            family_rows.append(row)
        family_metrics = pd.DataFrame(family_rows)
        family_metrics.to_csv(
            run_dir / "metrics" / f"{model_name}_family_metrics.csv", index=False
        )

        macro = {
            "model": model_name,
            **{
                f"macro_generator_{column}": float(generator_metrics[column].mean())
                for column in metric_columns
            },
            **{
                f"macro_family_{column}": float(family_metrics[column].mean())
                for column in metric_columns
            },
            "worst_generator_balanced_accuracy": float(
                generator_metrics["balanced_accuracy"].min()
            ),
            "best_generator_balanced_accuracy": float(
                generator_metrics["balanced_accuracy"].max()
            ),
            "num_generators": int(len(generator_metrics)),
            "num_generator_families": int(len(family_metrics)),
            "manifest_sha256": manifest_sha256,
            "checkpoint_sha256": checkpoint_info[model_name]["sha256"],
        }
        save_json(run_dir / "metrics" / f"{model_name}_macro_summary.json", macro)
        return {
            "overall": overall,
            "macro": macro,
            "generator_metrics": generator_metrics,
            "family_metrics": family_metrics,
        }

    predictors = {
        "clip_linear_probe": predict_clip,
        "npr_resnet18": predict_npr,
        "aide_forensic_resnet50": predict_aide,
    }
    predictions_by_model: dict[str, pd.DataFrame] = {}
    results_by_model: dict[str, dict[str, Any]] = {}
    completed_models = []
    for model_name, predictor in predictors.items():
        path = prediction_paths[model_name]
        predictions = None
        if bool(config["resume"]) and path.is_file():
            candidate = pd.read_csv(path).sort_values("manifest_order").reset_index(drop=True)
            candidate_orders = candidate.get("manifest_order", pd.Series(dtype=int)).to_numpy(
                dtype=int
            )
            expected_orders = manifest["manifest_order"].to_numpy(dtype=int)
            if len(candidate) == len(manifest) and np.array_equal(
                candidate_orders, expected_orders
            ):
                predictions = candidate
                print(f"Loaded completed {model_name} predictions from {path}.")
            else:
                print(
                    f"Ignoring incomplete/stale {model_name} predictions at {path}; "
                    "the model will be rerun."
                )
        if predictions is None:
            predictions = predictor()
            predictions.to_csv(path, index=False)
        predictions_by_model[model_name] = predictions
        results_by_model[model_name] = evaluate_predictions(model_name, predictions)
        completed_models.append(model_name)
        save_json(
            run_dir / "status.json",
            {
                "inference_id": inference_id,
                "manifest_ready": True,
                "manifest_sha256": manifest_sha256,
                "image_cache_ready": True,
                "cached_images": len(cache_index_rows),
                "completed_models": completed_models,
            },
        )
        output_volume.commit()
        print(f"Completed and committed {model_name}.")

    long_predictions = pd.concat(predictions_by_model.values(), ignore_index=True)
    long_predictions.to_csv(run_dir / "predictions" / "all_models_predictions.csv", index=False)

    wide_predictions = manifest.copy()
    for model_name, predictions in predictions_by_model.items():
        wide_predictions[f"{model_name}_fake_probability"] = predictions[
            "fake_probability"
        ].to_numpy()
        wide_predictions[f"{model_name}_predicted_label"] = predictions[
            "predicted_label"
        ].to_numpy(dtype=int)
    wide_predictions.to_csv(
        run_dir / "predictions" / "all_models_predictions_wide.csv", index=False
    )

    all_generator_metrics = pd.concat(
        [results["generator_metrics"] for results in results_by_model.values()],
        ignore_index=True,
    )
    all_generator_metrics.to_csv(
        run_dir / "metrics" / "all_models_generator_metrics.csv", index=False
    )
    all_family_metrics = pd.concat(
        [results["family_metrics"] for results in results_by_model.values()],
        ignore_index=True,
    )
    all_family_metrics.to_csv(
        run_dir / "metrics" / "all_models_family_metrics.csv", index=False
    )

    comparison_rows = []
    for model_name, results in results_by_model.items():
        row = {
            "model": model_name,
            **{f"overall_{key}": value for key, value in results["overall"].items() if not isinstance(value, list)},
            **results["macro"],
        }
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(run_dir / "metrics" / "all_models_summary.csv", index=False)

    summary = {
        "inference_id": inference_id,
        "output_root": str(run_dir),
        "manifest_samples": int(len(manifest)),
        "manifest_real": int((manifest["label"] == 0).sum()),
        "manifest_fake": int((manifest["label"] == 1).sum()),
        "manifest_sha256": manifest_sha256,
        "records_scanned": int(selection_records_scanned),
        "records_scanned_this_run": int(records_scanned_this_run),
        "image_cache": {
            "ready": True,
            "image_count": len(cache_index_rows),
            "total_bytes": cache_total_bytes,
            "format": "lossless RGB PNG",
            "path": "dataset/image_cache",
            "index": "dataset/image_cache_manifest.csv",
        },
        "target_generators": config["target_generators"],
        "generator_families": config["generator_families"],
        "target_real_sources": config["target_real_sources"],
        "models": {
            model_name: {
                "overall": results["overall"],
                "macro": results["macro"],
            }
            for model_name, results in results_by_model.items()
        },
    }
    save_json(final_summary_path, summary)
    save_json(
        run_dir / "status.json",
        {
            "inference_id": inference_id,
            "manifest_ready": True,
            "manifest_sha256": manifest_sha256,
            "image_cache_ready": True,
            "cached_images": len(cache_index_rows),
            "completed_models": completed_models,
            "complete": True,
        },
    )
    output_volume.commit()

    print("\nThree-model summary:")
    display_columns = [
        "model",
        "macro_generator_balanced_accuracy",
        "macro_generator_fake_recall",
        "macro_generator_fake_f1",
        "macro_generator_roc_auc",
        "macro_family_balanced_accuracy",
    ]
    print(comparison_df[display_columns].to_string(index=False))
    print(f"Modal output root: {run_dir}")
    return summary


@app.local_entrypoint()
def main(
    inference_id: str | None = None,
    fake_per_generator: int = 100,
    real_per_source: int = 20,
    max_scan_records: int = 500000,
    selection_seed: int = 43,
    smoke: bool = False,
    resume: bool = True,
):
    run_id = inference_id or time.strftime("%Y%m%d_%H%M%S")
    if smoke:
        fake_per_generator = min(fake_per_generator, 2)
        real_per_source = min(real_per_source, 2)
    overrides = {
        "fake_per_generator": fake_per_generator,
        "real_per_source": real_per_source,
        "max_scan_records": max_scan_records,
        "selection_seed": selection_seed,
        "resume": resume,
    }
    summary = evaluate_three_models.remote(run_id, overrides)
    print("Inference completed:", summary)
    print(f"Modal output root: {REMOTE_OUTPUT_ROOT}/{run_id}")
