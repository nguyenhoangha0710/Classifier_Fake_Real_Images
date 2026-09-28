"""Train original Full AIDE on Tiny-GenImage and evaluate Tiny + CommFor on Kaggle.

The architecture follows the official AIDE implementation:

* four native 32x32 patches selected by the six-band DCT score;
* 30 fixed SRM filters and separate low/high ResNet-50 encoders;
* a frozen OpenCLIP ConvNeXt-XXLarge semantic image trunk;
* 3072 -> 256 semantic projection;
* concatenation of 2048 forensic and 256 semantic features;
* GELU MLP classifier (2304 -> 1024 -> 2).

This runtime is designed to be embedded in a self-contained Kaggle notebook.
CommunityForensics is test-only and is never used for training, model selection,
threshold tuning, or early stopping.

Official source adapted under the upstream MIT license:
https://github.com/shilinyan99/AIDE
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm.auto import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "validation": ("val", "valid", "validation", "test"),
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

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)


DEFAULT_CONFIG: dict[str, Any] = {
    "input_root": "/kaggle/input",
    "tiny_dataset_root": None,
    "output_root": "/kaggle/working/aide_original_full_tiny_commfor",
    "random_seed": 42,
    "val_fraction": 0.10,
    "balance_real": True,
    "max_train_samples": None,
    "max_val_samples": None,
    "max_tiny_test_samples": None,
    "epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_workers": 2,
    "base_learning_rate": 1e-4,
    "learning_rate": None,
    "min_learning_rate": 1e-6,
    "weight_decay": 0.0,
    "label_smoothing": 0.1,
    "max_grad_norm": 1.0,
    "checkpoint_every_optimizer_steps": 250,
    "resume": True,
    "dct_window_size": 32,
    "dct_stride": 16,
    "dct_grade_bands": 6,
    "image_size": 256,
    "train_gaussian_blur_probability": 0.1,
    "train_jpeg_probability": 0.1,
    "semantic_model_name": "convnext_xxlarge",
    "semantic_pretrained_tag": "laion2b_s34b_b82k_augreg_soup",
    "semantic_checkpoint": None,
    "semantic_half_precision": True,
    "resnet_checkpoint": None,
    "imagenet_resnet_init": True,
    "commfor_manifest_path": None,
    "commfor_allow_hf_fallback": True,
    "commfor_dataset_name": "OwensLab/CommunityForensics-Eval",
    "commfor_split": "CompEval",
    "commfor_fake_per_generator": 100,
    "commfor_real_per_source": 20,
    "commfor_shuffle_buffer_size": 10_000,
    "commfor_max_scan_records": 250_000,
    "threshold": 0.5,
    "package_commfor_images": False,
    "smoke_test": False,
}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "").replace(".", "")


def find_label_dirs(split_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    if not split_dir.is_dir():
        return result
    for child in split_dir.iterdir():
        if child.is_dir() and child.name.lower() in LABEL_DIR_TO_ID:
            result[LABEL_DIR_TO_ID[child.name.lower()]] = child
    return result


def resolve_split_dir(generator_dir: Path, canonical_split: str) -> Path | None:
    for alias in SPLIT_ALIASES[canonical_split]:
        candidate = generator_dir / alias
        if candidate.is_dir() and find_label_dirs(candidate):
            return candidate
    return None


def looks_like_generator_dir(path: Path) -> bool:
    return path.is_dir() and any(resolve_split_dir(path, split) is not None for split in SPLIT_ALIASES)


def find_tiny_genimage_root(input_root: str | Path, explicit_root: str | Path | None = None) -> Path:
    if explicit_root:
        root = Path(explicit_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Tiny-GenImage root not found: {root}")
        return root
    input_path = Path(input_root)
    candidates = [input_path, *[p for p in input_path.rglob("*") if p.is_dir()]]
    best: tuple[int, Path] | None = None
    for candidate in candidates:
        try:
            count = sum(1 for child in candidate.iterdir() if looks_like_generator_dir(child))
        except OSError:
            continue
        if count and (best is None or count > best[0]):
            best = (count, candidate)
    if best is None:
        raise FileNotFoundError(
            "Cannot find Tiny-GenImage. Expected generator/train/{ai,nature} and generator/val/{ai,nature}."
        )
    return best[1]


def iter_images(path: Path) -> Iterable[Path]:
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            yield item


def build_tiny_index(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    generators = sorted(
        [path for path in root.iterdir() if looks_like_generator_dir(path)],
        key=lambda path: path.name.lower(),
    )
    for generator_dir in generators:
        for split in ("train", "validation"):
            split_dir = resolve_split_dir(generator_dir, split)
            if split_dir is None:
                continue
            for label, label_dir in find_label_dirs(split_dir).items():
                for image_path in iter_images(label_dir):
                    relative_image = image_path.relative_to(label_dir).as_posix()
                    rows.append(
                        {
                            "sample_id": f"{generator_dir.name}:{split}:{label}:{relative_image}",
                            "image_path": str(image_path),
                            "image_name": image_path.name,
                            "label": int(label),
                            "label_name": "fake" if label else "real",
                            "generator": generator_dir.name,
                            "split": split,
                            "dataset_source": "Tiny-GenImage",
                        }
                    )
    if not rows:
        raise FileNotFoundError(f"No Tiny-GenImage images found below {root}")
    return pd.DataFrame(rows)


def stratified_limit(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or len(frame) <= int(maximum):
        return frame.reset_index(drop=True)
    strata = frame["generator"].astype(str) + "::" + frame["label"].astype(str)
    pieces: list[pd.DataFrame] = []
    remaining = int(maximum)
    groups = list(strata.value_counts().sort_index().items())
    for index, (key, count) in enumerate(groups):
        part = frame[strata == key]
        if index == len(groups) - 1:
            take = min(len(part), remaining)
        else:
            take = min(len(part), max(1, round(int(maximum) * int(count) / len(frame))))
        if take:
            pieces.append(part.sample(n=take, random_state=seed + index, replace=False))
            remaining -= take
    result = pd.concat(pieces, ignore_index=True).drop_duplicates("sample_id")
    if len(result) > int(maximum):
        result = result.sample(n=int(maximum), random_state=seed)
    elif len(result) < int(maximum):
        available = frame[~frame["sample_id"].isin(result["sample_id"])]
        result = pd.concat(
            [result, available.sample(n=min(int(maximum) - len(result), len(available)), random_state=seed + 999)],
            ignore_index=True,
        )
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def balance_binary(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    real = frame[frame["label"] == 0]
    fake = frame[frame["label"] == 1]
    count = min(len(real), len(fake))
    return pd.concat(
        [
            real.sample(n=count, random_state=seed, replace=False),
            fake.sample(n=count, random_state=seed + 1, replace=False),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def build_tiny_splits(config: dict[str, Any], run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = find_tiny_genimage_root(config["input_root"], config.get("tiny_dataset_root"))
    full = build_tiny_index(root)
    combined_train = full[full["split"] == "train"].reset_index(drop=True)
    tiny_test = full[full["split"] == "validation"].reset_index(drop=True)
    if config["balance_real"]:
        combined_train = balance_binary(combined_train, int(config["random_seed"]))
        tiny_test = balance_binary(tiny_test, int(config["random_seed"]) + 10)
    strata = combined_train["generator"].astype(str) + "::" + combined_train["label"].astype(str)
    train, val = train_test_split(
        combined_train,
        test_size=float(config["val_fraction"]),
        random_state=int(config["random_seed"]),
        shuffle=True,
        stratify=strata,
    )
    train = stratified_limit(train, config.get("max_train_samples"), int(config["random_seed"]))
    val = stratified_limit(val, config.get("max_val_samples"), int(config["random_seed"]) + 1)
    tiny_test = stratified_limit(
        tiny_test, config.get("max_tiny_test_samples"), int(config["random_seed"]) + 2
    )
    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(dataset_dir / "tiny_train.csv", index=False)
    val.to_csv(dataset_dir / "tiny_val.csv", index=False)
    tiny_test.to_csv(dataset_dir / "tiny_test.csv", index=False)
    (
        full.groupby(["split", "generator", "label_name"])
        .size()
        .reset_index(name="num_images")
        .to_csv(dataset_dir / "tiny_full_counts.csv", index=False)
    )
    save_json(
        dataset_dir / "tiny_split_summary.json",
        {
            "detected_root": str(root),
            "train_samples": len(train),
            "val_samples": len(val),
            "tiny_test_samples": len(tiny_test),
            "tiny_test_real": int((tiny_test["label"] == 0).sum()),
            "tiny_test_fake": int((tiny_test["label"] == 1).sum()),
            "generators": sorted(full["generator"].unique().tolist()),
        },
    )
    config["detected_tiny_root"] = str(root)
    return train, val, tiny_test


def dct_matrix(size: int) -> torch.Tensor:
    matrix = []
    for frequency in range(size):
        scale = math.sqrt(1.0 / size) if frequency == 0 else math.sqrt(2.0 / size)
        matrix.append(
            [
                scale * math.cos((position + 0.5) * math.pi * frequency / size)
                for position in range(size)
            ]
        )
    return torch.tensor(matrix, dtype=torch.float32)


class AIDEDCTPatchSelector(nn.Module):
    """Official six-band DCT grade and min1/max1/min2/max2 selection."""

    def __init__(self, window_size: int = 32, stride: int = 16, grade_bands: int = 6):
        super().__init__()
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.grade_bands = int(grade_bands)
        self.register_buffer("dct", dct_matrix(self.window_size), persistent=False)
        coordinates = torch.arange(self.window_size)
        diagonal_index = coordinates[:, None] + coordinates[None, :]
        masks, counts = [], []
        for band in range(self.grade_bands):
            start = self.window_size * 2.0 / self.grade_bands * band
            end = self.window_size * 2.0 / self.grade_bands * (band + 1)
            # Inclusive upper bound matches the official generate_filter implementation.
            mask = ((diagonal_index >= start) & (diagonal_index <= end)).float()
            masks.append(mask)
            counts.append(mask.sum())
        self.register_buffer("band_masks", torch.stack(masks), persistent=False)
        self.register_buffer("band_counts", torch.stack(counts), persistent=False)
        self.register_buffer(
            "band_weights",
            torch.tensor([2.0**band for band in range(self.grade_bands)], dtype=torch.float32),
            persistent=False,
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected RGB tensor [3,H,W], got {tuple(image.shape)}")
        if min(image.shape[-2:]) < self.window_size:
            scale = self.window_size / min(image.shape[-2:])
            image = F.interpolate(
                image.unsqueeze(0),
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        columns = F.unfold(
            image.unsqueeze(0), kernel_size=self.window_size, stride=self.stride
        ).squeeze(0).transpose(0, 1)
        patches = columns.reshape(-1, 3, self.window_size, self.window_size)
        if len(patches) < 2:
            raise ValueError("AIDE requires at least two candidate patches")
        coefficients = self.dct @ patches @ self.dct.transpose(0, 1)
        log_magnitude = torch.log(torch.abs(coefficients) + 1.0)
        band_values = []
        for band in range(self.grade_bands):
            value = (
                log_magnitude
                * self.band_masks[band].view(1, 1, self.window_size, self.window_size)
            ).sum(dim=(1, 2, 3)) / self.band_counts[band]
            band_values.append(value)
        scores = (
            torch.stack(band_values, dim=1) * self.band_weights.view(1, -1)
        ).sum(dim=1)
        sorted_indices = torch.argsort(scores)
        selected_indices = torch.stack(
            [sorted_indices[0], sorted_indices[-1], sorted_indices[1], sorted_indices[-2]]
        )
        selected_coefficients = coefficients.index_select(0, selected_indices)
        selected = self.dct.transpose(0, 1) @ selected_coefficients @ self.dct
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
        weight = normalized_srm_kernels().view(30, 1, 5, 5).repeat(1, 3, 1, 1)
        self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.hpf(inputs)


def load_resnet_source_state(config: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    explicit = config.get("resnet_checkpoint")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"resnet_checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            payload = payload["model"]
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported ResNet checkpoint payload: {type(payload)}")
        return {str(k).removeprefix("module."): value for k, value in payload.items()}
    if not config.get("imagenet_resnet_init", True):
        return None
    source = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    state = source.state_dict()
    del source
    return state


def make_resnet50_encoder(source_state: dict[str, torch.Tensor] | None) -> nn.Module:
    backbone = resnet50(weights=None, zero_init_residual=True)
    if source_state is not None:
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in backbone.state_dict() and backbone.state_dict()[key].shape == value.shape
        }
        backbone.load_state_dict(compatible, strict=False)
    backbone.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
    backbone.fc = nn.Identity()
    return backbone


def discover_semantic_checkpoint(config: dict[str, Any]) -> Path | None:
    explicit = config.get("semantic_checkpoint")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"semantic_checkpoint not found: {path}")
        return path
    candidates = sorted(
        path
        for path in Path(config["input_root"]).rglob("open_clip_pytorch_model.bin")
        if path.is_file()
    )
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Multiple open_clip_pytorch_model.bin files found: {candidates}. Set CONFIG['semantic_checkpoint']."
        )
    return candidates[0] if candidates else None


def build_openclip_convnext_trunk(config: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    import open_clip

    local_path = discover_semantic_checkpoint(config)
    pretrained = str(local_path) if local_path else str(config["semantic_pretrained_tag"])
    print(f"Loading OpenCLIP {config['semantic_model_name']} pretrained={pretrained}")
    full_model, _, _ = open_clip.create_model_and_transforms(
        str(config["semantic_model_name"]), pretrained=pretrained
    )
    trunk = full_model.visual.trunk
    trunk.head.global_pool = nn.Identity()
    trunk.head.flatten = nn.Identity()
    del full_model
    gc.collect()
    for parameter in trunk.parameters():
        parameter.requires_grad = False
    trunk.eval()
    provenance = {
        "model_name": config["semantic_model_name"],
        "pretrained": pretrained,
        "checkpoint_path": str(local_path) if local_path else None,
        "checkpoint_sha256": sha256_file(local_path) if local_path else None,
    }
    return trunk, provenance


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = 1024, out_features: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(inputs)))


class AIDEFullModel(nn.Module):
    """Full AIDE: frozen ConvNeXt-XXL semantics + DCT/SRM dual-ResNet50 forensics."""

    def __init__(self, semantic_trunk: nn.Module, resnet_source_state: dict[str, torch.Tensor] | None):
        super().__init__()
        self.hpf = AIDEHPF()
        self.model_min = make_resnet50_encoder(resnet_source_state)
        self.model_max = make_resnet50_encoder(resnet_source_state)
        self.semantic_trunk = semantic_trunk
        self.semantic_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.semantic_projection = nn.Linear(3072, 256)
        self.classifier = MLP(2048 + 256, 1024, 2)

    def train(self, mode: bool = True):
        super().train(mode)
        self.semantic_trunk.eval()
        return self

    def forward(
        self, inputs: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if inputs.ndim != 5 or inputs.shape[1:3] != (5, 3):
            raise ValueError(f"Expected [B,5,3,H,W], got {tuple(inputs.shape)}")
        # Keep the four passes separate, as in the official implementation.
        # Combining low_1/low_2 into one larger batch would change BatchNorm
        # statistics during training even though it is equivalent at inference.
        low_1 = self.model_min(self.hpf(inputs[:, 0]))
        high_1 = self.model_max(self.hpf(inputs[:, 1]))
        low_2 = self.model_min(self.hpf(inputs[:, 2]))
        high_2 = self.model_max(self.hpf(inputs[:, 3]))
        forensic = (low_1 + high_1 + low_2 + high_2) / 4.0

        raw_imagenet = inputs[:, 4]
        clip_inputs = (
            raw_imagenet * (IMAGENET_STD.to(raw_imagenet) / CLIP_STD.to(raw_imagenet))
            + (IMAGENET_MEAN.to(raw_imagenet) - CLIP_MEAN.to(raw_imagenet))
            / CLIP_STD.to(raw_imagenet)
        )
        with torch.no_grad():
            semantic_map = self.semantic_trunk(clip_inputs)
        if isinstance(semantic_map, (tuple, list)):
            semantic_map = semantic_map[-1]
        if isinstance(semantic_map, dict):
            semantic_map = (
                semantic_map["x"]
                if "x" in semantic_map
                else list(semantic_map.values())[-1]
            )
        if semantic_map.ndim == 4:
            semantic_backbone = self.semantic_pool(semantic_map).flatten(1)
        elif semantic_map.ndim == 2:
            semantic_backbone = semantic_map
        else:
            raise ValueError(f"Unexpected semantic feature shape: {tuple(semantic_map.shape)}")
        if semantic_backbone.shape[1] != 3072:
            raise ValueError(
                f"Expected ConvNeXt-XXLarge 3072-D feature, got {tuple(semantic_backbone.shape)}"
            )
        # The backbone is frozen, while the 3072->256 adapter is intentionally trainable.
        semantic = self.semantic_projection(semantic_backbone.float())
        fused = torch.cat([semantic, forensic], dim=1)
        logits = self.classifier(fused)
        if return_features:
            return logits, {
                "semantic": semantic,
                "forensic": forensic,
                "low_mean": (low_1 + low_2) / 2.0,
                "high_mean": (high_1 + high_2) / 2.0,
                "fused": fused,
            }
        return logits


class AIDEImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, config: dict[str, Any], training: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.training = bool(training)
        self.selector = AIDEDCTPatchSelector(
            window_size=int(config["dct_window_size"]),
            stride=int(config["dct_stride"]),
            grade_bands=int(config["dct_grade_bands"]),
        )
        self.perturbations = None
        if self.training:
            import kornia.augmentation as K

            self.perturbations = K.container.ImageSequential(
                K.RandomGaussianBlur(
                    kernel_size=(3, 3),
                    sigma=(0.1, 3.0),
                    p=float(config["train_gaussian_blur_probability"]),
                ),
                K.RandomJPEG(
                    jpeg_quality=(30.0, 100.0),
                    p=float(config["train_jpeg_probability"]),
                ),
            )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if self.perturbations is not None:
            tensor = self.perturbations(tensor.unsqueeze(0)).squeeze(0)
        patches, selection = self.selector(tensor)
        size = int(self.config["image_size"])
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
        patches = (patches - IMAGENET_MEAN) / IMAGENET_STD
        raw = (raw - IMAGENET_MEAN) / IMAGENET_STD
        model_input = torch.cat([patches, raw], dim=0)
        return {
            "inputs": model_input,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_id": str(row.get("sample_id", index)),
            "image_name": str(row.get("image_name", Path(row["image_path"]).name)),
            "image_path": str(row["image_path"]),
            "generator": str(row.get("generator", "unknown")),
            "generator_family": str(row.get("generator_family", "unknown")),
            "label_name": str(row.get("label_name", "fake" if int(row["label"]) else "real")),
            **selection,
        }


def make_loader(
    frame: pd.DataFrame,
    config: dict[str, Any],
    training: bool,
    epoch: int = 0,
) -> DataLoader:
    dataset = AIDEImageDataset(frame, config=config, training=training)
    generator = torch.Generator()
    generator.manual_seed(int(config["random_seed"]) + int(epoch))
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(config["batch_size"]),
        "shuffle": bool(training),
        "num_workers": int(config["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": bool(training),
        "generator": generator,
    }
    if int(config["num_workers"]) > 0:
        options.update({"persistent_workers": False, "prefetch_factor": 2})
    return DataLoader(**options)


def compute_metrics(frame: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
    y_true = frame["label"].to_numpy(dtype=int)
    y_prob = frame["fake_probability"].to_numpy(dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result: dict[str, Any] = {
        "threshold": float(threshold),
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
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        result["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
    threshold: float,
    include_selection: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description):
            inputs = batch["inputs"].to(device, non_blocking=True)
            with (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else contextlib.nullcontext()
            ):
                logits = model(inputs)
                probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            batch_size = len(probabilities)
            for index in range(batch_size):
                row = {
                    "sample_id": str(batch["sample_id"][index]),
                    "image_name": str(batch["image_name"][index]),
                    "image_path": str(batch["image_path"][index]),
                    "label": int(batch["label"][index]),
                    "label_name": str(batch["label_name"][index]),
                    "generator": str(batch["generator"][index]),
                    "generator_family": str(batch["generator_family"][index]),
                    "fake_probability": float(probabilities[index]),
                    "predicted_label": int(probabilities[index] >= threshold),
                }
                if include_selection:
                    for key in (
                        "num_candidates",
                        "low_1_score",
                        "high_1_score",
                        "low_2_score",
                        "high_2_score",
                    ):
                        value = batch[key][index]
                        row[key] = float(value) if key != "num_candidates" else int(value)
                rows.append(row)
    predictions = pd.DataFrame(rows)
    return compute_metrics(predictions, threshold=threshold), predictions


def tiny_generator_metrics(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for generator, part in predictions.groupby("generator", sort=True):
        rows.append(
            {
                "generator": str(generator),
                "metric_cohort": "tiny_generator_own_real_and_fake",
                **compute_metrics(part, threshold=threshold),
            }
        )
    return pd.DataFrame(rows)


def commfor_group_metrics(
    predictions: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real = predictions[predictions["label"] == 0]
    fake = predictions[predictions["label"] == 1]
    generator_rows = []
    for generator in TARGET_GENERATORS:
        cohort = pd.concat([real, fake[fake["generator"] == generator]], ignore_index=True)
        generator_rows.append(
            {
                "generator": generator,
                "generator_family": GENERATOR_FAMILIES[generator],
                "metric_cohort": "generator_fake_plus_shared_real",
                **compute_metrics(cohort, threshold=threshold),
            }
        )
    family_rows = []
    for family in dict.fromkeys(GENERATOR_FAMILIES.values()):
        cohort = pd.concat(
            [real, fake[fake["generator_family"] == family]], ignore_index=True
        )
        family_rows.append(
            {
                "generator_family": family,
                "metric_cohort": "family_fake_plus_shared_real",
                **compute_metrics(cohort, threshold=threshold),
            }
        )
    return pd.DataFrame(generator_rows), pd.DataFrame(family_rows)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Exclude the frozen 3.4 GB semantic trunk from AIDE checkpoints."""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("semantic_trunk.")
    }


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    next_batch: int,
    optimizer_step: int,
    best_val_ba: float,
    config: dict[str, Any],
    semantic_provenance: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "architecture": "AIDE full hybrid: OpenCLIP ConvNeXt-XXLarge + DCT/SRM dual ResNet50",
            "model_trainable_state_dict": trainable_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": int(epoch),
            "next_batch": int(next_batch),
            "optimizer_step": int(optimizer_step),
            "best_val_balanced_accuracy": float(best_val_ba),
            "config": config,
            "semantic_provenance": semantic_provenance,
            "history": history,
            "rng_state": capture_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def save_best_model_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    optimizer_step: int,
    best_val_ba: float,
    config: dict[str, Any],
    semantic_provenance: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    """Save the trainable AIDE weights without optimizer or frozen ConvNeXt weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "architecture": "AIDE full hybrid: OpenCLIP ConvNeXt-XXLarge + DCT/SRM dual ResNet50",
            "model_trainable_state_dict": trainable_state_dict(model),
            "epoch": int(epoch),
            "optimizer_step": int(optimizer_step),
            "best_val_balanced_accuracy": float(best_val_ba),
            "config": config,
            "semantic_provenance": semantic_provenance,
            "history": history,
            "frozen_semantic_backbone_included": False,
        },
        temporary,
    )
    temporary.replace(path)


def load_trainable_model_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_nonsemantic = [key for key in missing if not key.startswith("semantic_trunk.")]
    if missing_nonsemantic or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing_nonsemantic={missing_nonsemantic}, unexpected={unexpected}"
        )


def build_model(
    config: dict[str, Any], device: torch.device
) -> tuple[AIDEFullModel, dict[str, Any]]:
    semantic_trunk, semantic_provenance = build_openclip_convnext_trunk(config)
    resnet_state = load_resnet_source_state(config)
    model = AIDEFullModel(semantic_trunk, resnet_state)
    del resnet_state
    model.to(device)
    if bool(config["semantic_half_precision"]) and device.type == "cuda":
        model.semantic_trunk.half()
    model.semantic_trunk.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, semantic_provenance


def train_full_aide(
    model: AIDEFullModel,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    semantic_provenance: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]]]:
    accumulation = int(config["gradient_accumulation_steps"])
    effective_batch = int(config["batch_size"]) * accumulation
    learning_rate = config.get("learning_rate")
    if learning_rate is None:
        learning_rate = float(config["base_learning_rate"]) * effective_batch / 256.0
    config["resolved_learning_rate"] = float(learning_rate)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(config["weight_decay"]),
    )
    steps_per_epoch = math.ceil(
        math.ceil(len(train_frame) / int(config["batch_size"])) / accumulation
    )
    total_steps = max(1, int(config["epochs"]) * steps_per_epoch)

    def lr_factor(step: int) -> float:
        if total_steps <= 1:
            return 1.0
        minimum_factor = float(config["min_learning_rate"]) / float(learning_rate)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    latest_path = run_dir / "checkpoints" / "latest" / "model_trainable.pt"
    best_path = run_dir / "checkpoints" / "best" / "model_trainable.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    start_batch = 0
    optimizer_step = 0
    best_val_ba = -1.0

    if bool(config["resume"]) and latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        load_trainable_model_state(model, checkpoint["model_trainable_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint.get("next_batch", 0))
        optimizer_step = int(checkpoint.get("optimizer_step", 0))
        best_val_ba = float(checkpoint.get("best_val_balanced_accuracy", -1.0))
        history = list(checkpoint.get("history", []))
        restore_rng_state(checkpoint.get("rng_state"))
        print(
            f"Resumed latest checkpoint: epoch={start_epoch}, batch={start_batch}, "
            f"optimizer_step={optimizer_step}, best_val_ba={best_val_ba:.4f}"
        )

    for epoch in range(start_epoch, int(config["epochs"])):
        loader = make_loader(train_frame, config=config, training=True, epoch=epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        observed = 0
        progress = tqdm(loader, desc=f"AIDE full epoch {epoch + 1}/{config['epochs']}")
        for batch_index, batch in enumerate(progress):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            inputs = batch["inputs"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else contextlib.nullcontext()
            ):
                logits = model(inputs)
                unscaled_loss = criterion(logits, labels)
                loss = unscaled_loss / accumulation
            scaler.scale(loss).backward()
            running_loss += float(unscaled_loss.detach()) * len(labels)
            observed += len(labels)
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if should_step:
                if float(config["max_grad_norm"]) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, float(config["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1
                interval = int(config["checkpoint_every_optimizer_steps"])
                if interval > 0 and optimizer_step % interval == 0:
                    save_training_checkpoint(
                        latest_path,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        batch_index + 1,
                        optimizer_step,
                        best_val_ba,
                        config,
                        semantic_provenance,
                        history,
                    )
            progress.set_postfix(
                loss=f"{running_loss / max(observed, 1):.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                step=optimizer_step,
            )

        start_batch = 0
        val_loader = make_loader(val_frame, config=config, training=False, epoch=epoch)
        val_metrics, _ = evaluate_loader(
            model,
            val_loader,
            device,
            description=f"Validation epoch {epoch + 1}",
            threshold=float(config["threshold"]),
            include_selection=False,
        )
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(observed, 1),
            "optimizer_step": optimizer_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_record)
        pd.DataFrame(history).to_csv(run_dir / "metrics" / "training_history.csv", index=False)
        improved = val_metrics["balanced_accuracy"] > best_val_ba
        if improved:
            best_val_ba = float(val_metrics["balanced_accuracy"])
            save_best_model_checkpoint(
                best_path,
                model,
                epoch + 1,
                optimizer_step,
                best_val_ba,
                config,
                semantic_provenance,
                history,
            )
            print(f"New best checkpoint: val BA={best_val_ba:.4f}")
        save_training_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch + 1,
            0,
            optimizer_step,
            best_val_ba,
            config,
            semantic_provenance,
            history,
        )
        print(json.dumps(epoch_record, ensure_ascii=False, indent=2))

    if not best_path.is_file():
        raise RuntimeError("Training completed without a best checkpoint")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    load_trainable_model_state(model, checkpoint["model_trainable_state_dict"])
    return best_path, history


def clean_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def generator_from_record(record: dict[str, Any]) -> str:
    return str(record.get("model_name") or record.get("architecture") or "unknown")


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


def discover_uploaded_commfor_manifest(config: dict[str, Any]) -> Path | None:
    explicit = config.get("commfor_manifest_path")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"commfor_manifest_path not found: {path}")
        return path
    candidates = sorted(
        path
        for path in Path(config["input_root"]).rglob("commfor_unseen_manifest.csv")
        if path.is_file()
    )
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Multiple CommFor manifests found: {candidates}. Set CONFIG['commfor_manifest_path']."
        )
    return candidates[0] if candidates else None


def validate_commfor_manifest(manifest: pd.DataFrame, config: dict[str, Any]) -> None:
    expected = len(TARGET_GENERATORS) * int(config["commfor_fake_per_generator"]) + len(
        TARGET_REAL_SOURCES
    ) * int(config["commfor_real_per_source"])
    if len(manifest) != expected:
        raise ValueError(f"CommFor manifest has {len(manifest)} rows, expected {expected}")
    if manifest["manifest_order"].duplicated().any():
        raise ValueError("CommFor manifest_order is not unique")
    if int((manifest["label"] == 0).sum()) != len(TARGET_REAL_SOURCES) * int(
        config["commfor_real_per_source"]
    ):
        raise ValueError("CommFor real count does not match configuration")
    for generator in TARGET_GENERATORS:
        count = int(((manifest["label"] == 1) & (manifest["generator"] == generator)).sum())
        if count != int(config["commfor_fake_per_generator"]):
            raise ValueError(f"CommFor {generator} has {count} fake rows")


def resolve_uploaded_commfor_cache(
    manifest_path: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, Path]:
    manifest = pd.read_csv(manifest_path).sort_values("manifest_order").reset_index(drop=True)
    # Older cached manifests used image_name as an implicit identifier.  Use a
    # stable unique ID so duplicate filenames across sources cannot break joins.
    manifest["sample_id"] = [
        f"commfor:{int(order):06d}" for order in manifest["manifest_order"]
    ]
    validate_commfor_manifest(manifest, config)
    if "cached_image_path" not in manifest.columns:
        manifest["cached_image_path"] = [
            f"dataset/image_cache/{int(order):06d}.png" for order in manifest["manifest_order"]
        ]
    candidate_roots = [manifest_path.parent]
    candidate_roots.extend(manifest_path.parents[index] for index in range(1, min(4, len(manifest_path.parents))))
    candidate_roots = list(dict.fromkeys(candidate_roots))
    for root in candidate_roots:
        if all(
            (root / str(row["cached_image_path"])).is_file()
            for row in manifest.to_dict("records")
        ):
            print(f"Using uploaded CommFor cache: {root}")
            return manifest, root
        alternate = root / "image_cache"
        if all(
            (alternate / f"{int(row['manifest_order']):06d}.png").is_file()
            for row in manifest.to_dict("records")
        ):
            manifest["cached_image_path"] = [
                f"image_cache/{int(order):06d}.png" for order in manifest["manifest_order"]
            ]
            return manifest, root
    raise FileNotFoundError(
        f"Manifest found at {manifest_path}, but its 1,000 cached images were not found"
    )


def build_commfor_cache_from_hf(
    config: dict[str, Any], run_dir: Path
) -> tuple[pd.DataFrame, Path]:
    from datasets import load_dataset

    dataset_dir = run_dir / "dataset" / "commfor"
    image_cache = dataset_dir / "image_cache"
    image_cache.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(
        config["commfor_dataset_name"],
        split=config["commfor_split"],
        streaming=True,
    ).shuffle(
        seed=int(config["random_seed"]),
        buffer_size=int(config["commfor_shuffle_buffer_size"]),
    )
    selected_real = {source: [] for source in TARGET_REAL_SOURCES}
    selected_fake = {generator: [] for generator in TARGET_GENERATORS}
    scanned = 0
    for scanned, record in enumerate(stream, start=1):
        label = int(record.get("label"))
        if label == 0:
            source = str(record.get("real_source") or "unknown")
            if source in selected_real and len(selected_real[source]) < int(
                config["commfor_real_per_source"]
            ):
                selected_real[source].append(record)
        else:
            generator = generator_from_record(record)
            if generator in selected_fake and len(selected_fake[generator]) < int(
                config["commfor_fake_per_generator"]
            ):
                selected_fake[generator].append(record)
        real_ready = all(
            len(records) >= int(config["commfor_real_per_source"])
            for records in selected_real.values()
        )
        fake_ready = all(
            len(records) >= int(config["commfor_fake_per_generator"])
            for records in selected_fake.values()
        )
        if real_ready and fake_ready:
            break
        if scanned >= int(config["commfor_max_scan_records"]):
            break
    missing_real = {
        key: int(config["commfor_real_per_source"]) - len(value)
        for key, value in selected_real.items()
        if len(value) < int(config["commfor_real_per_source"])
    }
    missing_fake = {
        key: int(config["commfor_fake_per_generator"]) - len(value)
        for key, value in selected_fake.items()
        if len(value) < int(config["commfor_fake_per_generator"])
    }
    if missing_real or missing_fake:
        raise RuntimeError(
            f"CommFor quotas not filled after {scanned} records: "
            f"real_missing={missing_real}, fake_missing={missing_fake}"
        )

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    order = 0
    for source in TARGET_REAL_SOURCES:
        for source_index, record in enumerate(selected_real[source]):
            rows.append(
                {
                    "manifest_order": order,
                    "selection_group": "shared_real_reference",
                    "sample_index": source_index,
                    "sample_id": f"commfor:{order:06d}",
                    "label": 0,
                    "label_name": "real",
                    "generator": "shared_real",
                    "generator_family": "real",
                    "real_source": source,
                    "image_name": clean_value(record.get("image_name")),
                    "model_name": clean_value(record.get("model_name")),
                    "architecture": clean_value(record.get("architecture")),
                    "subset": clean_value(record.get("subset")),
                    "cached_image_path": f"dataset/commfor/image_cache/{order:06d}.png",
                }
            )
            records.append(record)
            order += 1
    for generator in TARGET_GENERATORS:
        for generator_index, record in enumerate(selected_fake[generator]):
            rows.append(
                {
                    "manifest_order": order,
                    "selection_group": "generator_fake",
                    "sample_index": generator_index,
                    "sample_id": f"commfor:{order:06d}",
                    "label": 1,
                    "label_name": "fake",
                    "generator": generator,
                    "generator_family": GENERATOR_FAMILIES[generator],
                    "real_source": "",
                    "image_name": clean_value(record.get("image_name")),
                    "model_name": clean_value(record.get("model_name")),
                    "architecture": clean_value(record.get("architecture")),
                    "subset": clean_value(record.get("subset")),
                    "cached_image_path": f"dataset/commfor/image_cache/{order:06d}.png",
                }
            )
            records.append(record)
            order += 1
    manifest = pd.DataFrame(rows)
    validate_commfor_manifest(manifest, config)
    for row, record in tqdm(
        zip(rows, records), total=len(rows), desc="Caching fixed CommFor cohort"
    ):
        destination = run_dir / row["cached_image_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        image_from_record(record).save(destination, format="PNG")
    manifest_path = dataset_dir / "commfor_unseen_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    save_json(
        dataset_dir / "image_cache_complete.json",
        {
            "complete": True,
            "image_count": len(manifest),
            "records_scanned": scanned,
            "manifest_sha256": sha256_file(manifest_path),
            "format": "lossless RGB PNG",
        },
    )
    return manifest, run_dir


def resolve_commfor_cohort(
    config: dict[str, Any], run_dir: Path
) -> tuple[pd.DataFrame, Path]:
    local_manifest = run_dir / "dataset" / "commfor" / "commfor_unseen_manifest.csv"
    if local_manifest.is_file():
        try:
            return resolve_uploaded_commfor_cache(local_manifest, config)
        except (FileNotFoundError, ValueError):
            pass
    uploaded = discover_uploaded_commfor_manifest(config)
    if uploaded is not None:
        return resolve_uploaded_commfor_cache(uploaded, config)
    if not bool(config["commfor_allow_hf_fallback"]):
        raise FileNotFoundError(
            "No uploaded CommFor manifest/cache found and Hugging Face fallback is disabled"
        )
    return build_commfor_cache_from_hf(config, run_dir)


def commfor_frame_from_cache(manifest: pd.DataFrame, cache_root: Path) -> pd.DataFrame:
    frame = manifest.copy()
    frame["image_path"] = [
        str(cache_root / str(path)) for path in frame["cached_image_path"]
    ]
    missing = [path for path in frame["image_path"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} cached CommFor images; first={missing[0]}")
    return frame


def macro_summary(generator_metrics: pd.DataFrame, prefix: str = "macro_generator") -> dict[str, Any]:
    columns = (
        "accuracy",
        "balanced_accuracy",
        "real_recall",
        "fake_recall",
        "fake_precision",
        "fake_f1",
        "roc_auc",
        "average_precision",
    )
    result = {
        f"{prefix}_{column}": float(generator_metrics[column].mean()) for column in columns
    }
    result["worst_generator_balanced_accuracy"] = float(
        generator_metrics["balanced_accuracy"].min()
    )
    result["best_generator_balanced_accuracy"] = float(
        generator_metrics["balanced_accuracy"].max()
    )
    result["num_generators"] = int(len(generator_metrics))
    return result


def parameter_summary(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "frozen_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
        ),
    }


def package_results(run_dir: Path, include_commfor_images: bool = False) -> Path:
    """Create a practical ZIP with best weights and reports, excluding the large resume optimizer state."""
    staging = run_dir.parent / f"{run_dir.name}_package"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in ("config.json", "output.json"):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, staging / name)
    for directory in ("dataset", "metrics", "predictions", "provenance"):
        source = run_dir / directory
        if source.is_dir():
            ignore = None
            if directory == "dataset" and not include_commfor_images:
                ignore = shutil.ignore_patterns("image_cache")
            shutil.copytree(source, staging / directory, ignore=ignore)
    best_source = run_dir / "checkpoints" / "best"
    if best_source.is_dir():
        shutil.copytree(best_source, staging / "checkpoints" / "best")
    zip_base = run_dir.parent / f"{run_dir.name}_results"
    archive = Path(shutil.make_archive(str(zip_base), "zip", root_dir=staging))
    shutil.rmtree(staging)
    return archive


def run_experiment(config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    if bool(config.get("smoke_test")):
        config.update(
            {
                "max_train_samples": min(int(config.get("max_train_samples") or 64), 64),
                "max_val_samples": min(int(config.get("max_val_samples") or 32), 32),
                "max_tiny_test_samples": min(int(config.get("max_tiny_test_samples") or 32), 32),
                "epochs": 1,
                "commfor_fake_per_generator": 2,
                "commfor_real_per_source": 2,
                "checkpoint_every_optimizer_steps": 5,
            }
        )

    seed_everything(int(config["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Full AIDE with ConvNeXt-XXLarge requires a Kaggle GPU accelerator")
    run_dir = Path(config["output_root"])
    for directory in (
        "checkpoints/latest",
        "checkpoints/best",
        "dataset",
        "metrics",
        "predictions",
        "provenance",
    ):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Full AIDE Kaggle experiment")
    print("Train: Tiny-GenImage combined")
    print("Test 1: Tiny-GenImage validation")
    print("Test 2: fixed 1,000-image CommFor unseen-family cohort")
    print("Device:", device, torch.cuda.get_device_name(0))
    print("Output:", run_dir)
    print("=" * 80)

    train_frame, val_frame, tiny_test_frame = build_tiny_splits(config, run_dir)
    save_json(run_dir / "config.json", config)
    print(
        f"Tiny splits: train={len(train_frame)}, val={len(val_frame)}, "
        f"test={len(tiny_test_frame)}"
    )

    model, semantic_provenance = build_model(config, device)
    provenance = {
        "official_aide_repository": "https://github.com/shilinyan99/AIDE",
        "semantic_backbone": semantic_provenance,
        "architecture": {
            "patch_selector": "DCT 32x32 stride 16, six inclusive diagonal bands, min1/max1/min2/max2",
            "forensic_branch": "30 fixed SRM filters + low/high ResNet50 + four-feature mean",
            "semantic_branch": "frozen OpenCLIP ConvNeXt-XXLarge + trainable 3072->256 projection",
            "fusion": "concat 2048 forensic + 256 semantic",
            "classifier": "MLP 2304->1024->2 with GELU",
        },
        "implementation_note": (
            "The official architecture is preserved. The semantic ConvNeXt trunk is frozen; "
            "the 3072->256 projection is trainable so the randomly initialized adapter can learn."
        ),
        **parameter_summary(model),
    }
    save_json(run_dir / "provenance" / "architecture.json", provenance)
    print(json.dumps(parameter_summary(model), indent=2))

    best_checkpoint, history = train_full_aide(
        model,
        train_frame,
        val_frame,
        config,
        run_dir,
        device,
        semantic_provenance,
    )
    save_json(run_dir / "config.json", config)
    save_json(
        run_dir / "provenance" / "best_checkpoint.json",
        {
            "path": str(best_checkpoint),
            "sha256": sha256_file(best_checkpoint),
            "frozen_semantic_backbone_included": False,
            "semantic_backbone": semantic_provenance,
        },
    )

    tiny_loader = make_loader(tiny_test_frame, config=config, training=False)
    tiny_overall, tiny_predictions = evaluate_loader(
        model,
        tiny_loader,
        device,
        description="Tiny-GenImage full AIDE test",
        threshold=float(config["threshold"]),
    )
    tiny_predictions.to_csv(run_dir / "predictions" / "tiny_predictions.csv", index=False)
    tiny_by_generator = tiny_generator_metrics(
        tiny_predictions, threshold=float(config["threshold"])
    )
    tiny_by_generator.to_csv(
        run_dir / "metrics" / "tiny_generator_metrics.csv", index=False
    )
    tiny_macro = macro_summary(tiny_by_generator)
    save_json(run_dir / "metrics" / "tiny_overall.json", tiny_overall)
    save_json(run_dir / "metrics" / "tiny_macro_summary.json", tiny_macro)
    print("Tiny overall:")
    print(json.dumps(tiny_overall, ensure_ascii=False, indent=2))

    commfor_manifest, cache_root = resolve_commfor_cohort(config, run_dir)
    commfor_manifest.to_csv(
        run_dir / "dataset" / "commfor_manifest_used.csv", index=False
    )
    commfor_frame = commfor_frame_from_cache(commfor_manifest, cache_root)
    commfor_loader = make_loader(commfor_frame, config=config, training=False)
    commfor_overall, commfor_predictions = evaluate_loader(
        model,
        commfor_loader,
        device,
        description="CommFor 1,000 full AIDE test",
        threshold=float(config["threshold"]),
    )
    commfor_predictions = commfor_predictions.merge(
        commfor_manifest[
            [
                "sample_id",
                "manifest_order",
                "selection_group",
                "real_source",
                "model_name",
                "architecture",
                "subset",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    ).sort_values("manifest_order")
    commfor_predictions.to_csv(
        run_dir / "predictions" / "commfor_predictions.csv", index=False
    )
    commfor_generator, commfor_family = commfor_group_metrics(
        commfor_predictions, threshold=float(config["threshold"])
    )
    commfor_generator.to_csv(
        run_dir / "metrics" / "commfor_generator_metrics.csv", index=False
    )
    commfor_family.to_csv(
        run_dir / "metrics" / "commfor_family_metrics.csv", index=False
    )
    commfor_macro = {
        **macro_summary(commfor_generator),
        **{
            f"macro_family_{column}": float(commfor_family[column].mean())
            for column in (
                "accuracy",
                "balanced_accuracy",
                "real_recall",
                "fake_recall",
                "fake_precision",
                "fake_f1",
                "roc_auc",
                "average_precision",
            )
        },
        "num_generator_families": int(len(commfor_family)),
    }
    save_json(run_dir / "metrics" / "commfor_overall.json", commfor_overall)
    save_json(run_dir / "metrics" / "commfor_macro_summary.json", commfor_macro)
    print("CommFor overall:")
    print(json.dumps(commfor_overall, ensure_ascii=False, indent=2))

    output = {
        "experiment": {
            "architecture": "Full AIDE hybrid",
            "training_dataset": "Tiny-GenImage combined",
            "tiny_test_samples": len(tiny_test_frame),
            "commfor_test_samples": len(commfor_manifest),
            "commfor_real": int((commfor_manifest["label"] == 0).sum()),
            "commfor_fake": int((commfor_manifest["label"] == 1).sum()),
            "commfor_generators": list(TARGET_GENERATORS),
            "commfor_generator_families": list(dict.fromkeys(GENERATOR_FAMILIES.values())),
            "threshold": float(config["threshold"]),
            "best_checkpoint": str(best_checkpoint),
            "best_checkpoint_sha256": sha256_file(best_checkpoint),
        },
        "tiny": {"overall": tiny_overall, "macro": tiny_macro},
        "commfor": {"overall": commfor_overall, "macro": commfor_macro},
        "training_history": history,
    }
    save_json(run_dir / "output.json", output)
    archive = package_results(
        run_dir,
        include_commfor_images=bool(config["package_commfor_images"]),
    )
    output["archive"] = str(archive)
    save_json(run_dir / "output.json", output)
    print("Results ZIP:", archive)
    return output


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps(result["experiment"], ensure_ascii=False, indent=2))
