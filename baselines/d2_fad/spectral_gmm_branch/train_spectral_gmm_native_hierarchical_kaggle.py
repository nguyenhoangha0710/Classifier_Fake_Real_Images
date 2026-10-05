"""Native-resolution hierarchical Spectral-GMM training for Kaggle.

This implementation follows ``plan.md``.  It intentionally does not reuse the
older resized-crop MFM pipeline so the two experiments cannot be confused.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

import timm


ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
FEATURE_NAMES = ["jpeg_low", "jpeg_high"]
SPLITS = ("real_train", "real_validation", "real_calibration")


DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": "spectral_gmm_native_hierarchical_jpeg70_100_v1",
    "run_phases": ["stage1", "cache", "stage2", "gmm"],
    "output_parent": "/kaggle/working/spectral_gmm_native_hierarchical",
    "imagenet_train_root": None,
    "resume_run_roots": [],
    "seed": 20261004,
    # Deterministic real-only cohort.
    "images_per_synset": 100,
    "train_per_synset": 80,
    "validation_per_synset": 10,
    "calibration_per_synset": 10,
    "expected_synsets": 1000,
    "strict_synset_count": True,
    "valid_extensions": [".jpeg", ".jpg", ".png", ".webp", ".bmp"],
    # Native tiling.  Images >= 224 on both sides are never resized.
    "tile_size": 224,
    "tile_stride": 224,
    "small_image_resize": "preserve_aspect_ratio",
    "padding_mode": "reflect",
    # Mandatory whole-image codec normalization. Every source format is decoded
    # to RGB, encoded as JPEG Q70-100, then decoded before native tiling.
    "jpeg_quality_min": 70,
    "jpeg_quality_max": 100,
    "jpeg_subsampling_values": [0, 1, 2],
    # SPAI/MFM spectral corruption.
    "spectral_mask_implementation": "spai",
    "spectral_mask_radius": 16,
    "low_probability": 0.50,
    # Stage 1: all tiles, image-balanced loss.
    "stage1_epochs": 20,
    "stage1_warmup_epochs": 2,
    "stage1_patience": 3,
    "stage1_images_per_batch": 2,
    "stage1_tile_microbatch": 32,
    "stage1_num_workers": 2,
    "stage1_learning_rate": 3.0e-4,
    "stage1_min_learning_rate": 2.5e-6,
    "stage1_weight_decay": 0.05,
    "stage1_betas": [0.9, 0.95],
    "stage1_grad_clip": 3.0,
    "stage1_log_every_images": 10,
    "stage1_resume": True,
    # Frozen Local-CLS cache. One seeded JPEG realization per image; no raw view.
    "cache_images_per_shard": 250,
    "cache_local_microbatch": 64,
    "cache_shard_start": None,
    "cache_shard_stop": None,
    "cache_log_every_images": 10,
    # Stage 2: low/high fusion and Image-CLS bottleneck.
    "local_embedding_dim": 768,
    "global_dim": 768,
    "global_depth": 4,
    "global_heads": 12,
    "global_mlp_ratio": 4.0,
    "global_dropout": 0.10,
    "global_decoder_depth": 2,
    "global_mask_ratio": 0.40,
    "stage2_epochs": 10,
    "stage2_patience": 3,
    "stage2_max_images_per_batch": 16,
    "stage2_max_tokens_per_batch": 768,
    "stage2_learning_rate": 1.0e-4,
    "stage2_min_learning_rate": 1.0e-6,
    "stage2_weight_decay": 0.05,
    "stage2_grad_clip": 1.0,
    "stage2_log_every_images": 50,
    "stage2_resume": True,
    # Runtime and GMM.
    "amp": True,
    "gmm_components": [1, 2, 4, 8, 16],
    "gmm_covariance_type": "diag",
    "gmm_reg_covar": 1.0e-6,
    "gmm_max_iter": 100,
    "gmm_n_init": 1,
    "gmm_min_component_occupancy": 0.001,
}


# ---------------------------------------------------------------------------
# Reproducibility, files and manifests
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(global_seed: int, *parts: Any) -> int:
    text = "::".join([str(global_seed), *map(str, parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % (2**31)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def find_imagenet_train_root(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        configured = Path(explicit)
        candidates.extend(
            [
                configured,
                Path("/kaggle/input/competitions") / configured.as_posix().lstrip("/kaggle/input/"),
            ]
        )
    candidates.extend(
        [
            Path("/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"),
            Path("/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"),
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            resolved = candidate.resolve()
            print(f"Detected ImageNet train root: {resolved}", flush=True)
            return resolved
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for candidate in kaggle_input.glob("**/ILSVRC/Data/CLS-LOC/train"):
            if candidate.is_dir():
                resolved = candidate.resolve()
                print(f"Detected ImageNet train root: {resolved}", flush=True)
                return resolved
    raise FileNotFoundError("Cannot locate ILSVRC/Data/CLS-LOC/train under /kaggle/input.")


def discover_resume_roots(config: dict[str, Any], run_name: str) -> list[Path]:
    roots: list[Path] = []
    for value in config.get("resume_run_roots", []):
        candidate = Path(value)
        if candidate.is_dir():
            roots.append(candidate.resolve())
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for dataset_root in kaggle_input.iterdir():
            if not dataset_root.is_dir():
                continue
            candidates = (
                dataset_root / run_name,
                dataset_root / "spectral_gmm_native_hierarchical" / run_name,
                dataset_root,
            )
            for candidate in candidates:
                if candidate.is_dir() and (candidate / "config.json").is_file():
                    roots.append(candidate.resolve())
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def find_existing(relative: Path, output_root: Path, resume_roots: Sequence[Path]) -> Path | None:
    for candidate in (output_root / relative, *[root / relative for root in resume_roots]):
        if candidate.is_file():
            return candidate
    return None


def update_status(output_root: Path, phase: str, state: str, **extra: Any) -> None:
    path = output_root / "status.json"
    payload = read_json(path) if path.is_file() else {}
    payload.update({"phase": phase, "state": state, "updated_at": time.time(), **extra})
    write_json(path, payload)


def build_manifest(config: dict[str, Any], root: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    destination = output_root / "manifests" / "imagenet_real_100k.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    synsets = sorted(path for path in root.iterdir() if path.is_dir())
    if config["strict_synset_count"] and len(synsets) != int(config["expected_synsets"]):
        raise AssertionError(
            f"Expected {config['expected_synsets']} synsets, found {len(synsets)}"
        )
    n = int(config["images_per_synset"])
    counts = (
        int(config["train_per_synset"]),
        int(config["validation_per_synset"]),
        int(config["calibration_per_synset"]),
    )
    if sum(counts) != n:
        raise ValueError("Per-synset split counts must sum to images_per_synset.")
    extensions = {str(value).lower() for value in config["valid_extensions"]}
    rows: list[dict[str, Any]] = []
    print("Building manifest from filenames only; no image audit or decode.", flush=True)
    for synset in tqdm(synsets, desc="Build native hierarchy manifest", unit="synset"):
        candidates = sorted(path for path in synset.iterdir() if path.suffix.lower() in extensions)
        if len(candidates) < n:
            raise RuntimeError(f"{synset.name} has {len(candidates)} candidates; need {n}.")
        rng = random.Random(stable_seed(config["seed"], "sample", synset.name))
        rng.shuffle(candidates)
        selected = candidates[:n]
        split_rng = random.Random(stable_seed(config["seed"], "split", synset.name))
        split_rng.shuffle(selected)
        for index, path in enumerate(selected):
            if index < counts[0]:
                split = "real_train"
            elif index < counts[0] + counts[1]:
                split = "real_validation"
            else:
                split = "real_calibration"
            rows.append(
                {
                    "sample_id": f"{synset.name}/{path.stem}",
                    "relative_path": path.relative_to(root).as_posix(),
                    "synset": synset.name,
                    "split": split,
                    "sampling_seed": stable_seed(config["seed"], synset.name, path.name),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["split", "synset", "relative_path"], ignore_index=True)
    expected = len(synsets) * n
    if len(frame) != expected:
        raise AssertionError(f"Expected {expected} manifest rows, found {len(frame)}")
    frame.to_csv(destination, index=False)
    summary = {
        "num_synsets": len(synsets),
        "num_images": len(frame),
        "split_counts": frame["split"].value_counts().to_dict(),
        "manifest_sha256": sha256_file(destination),
        "selection": "deterministic_filename_only",
    }
    write_json(destination.parent / "manifest_summary.json", summary)
    return destination, summary


def ensure_manifest(
    config: dict[str, Any], root: Path, output_root: Path, resume_roots: Sequence[Path]
) -> tuple[Path, dict[str, Any]]:
    relative = Path("manifests/imagenet_real_100k.csv")
    existing = find_existing(relative, output_root, resume_roots)
    if existing is None:
        return build_manifest(config, root, output_root)
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if existing.resolve() != destination.resolve():
        shutil.copy2(existing, destination)
    summary = {
        "num_images": len(pd.read_csv(destination)),
        "manifest_sha256": sha256_file(destination),
    }
    write_json(destination.parent / "manifest_summary.json", summary)
    return destination, summary


# ---------------------------------------------------------------------------
# Native image preprocessing and spectral operations
# ---------------------------------------------------------------------------


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return TF.pil_to_tensor(image.convert("RGB")).float().div_(255.0)


def jpeg_recompress_pil(
    image: Image.Image, quality: int, subsampling: int
) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=int(quality),
        subsampling=int(subsampling),
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        decoded.load()
        return decoded.convert("RGB").copy()


def resize_small_preserve_aspect(image: Image.Image, tile_size: int) -> Image.Image:
    width, height = image.size
    if min(width, height) >= tile_size:
        return image
    scale = tile_size / float(min(width, height))
    new_width = max(tile_size, int(round(width * scale)))
    new_height = max(tile_size, int(round(height * scale)))
    return image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)


def deterministic_jpeg_parameters(config: dict[str, Any], sample_id: str) -> tuple[int, int]:
    rng = random.Random(stable_seed(config["seed"], sample_id, "cached_jpeg"))
    quality = rng.randint(int(config["jpeg_quality_min"]), int(config["jpeg_quality_max"]))
    subsampling = rng.choice(list(config["jpeg_subsampling_values"]))
    return quality, int(subsampling)


def prepare_working_image(
    path: Path,
    sample_id: str,
    config: dict[str, Any],
    mode: str,
    epoch: int = 0,
) -> tuple[Image.Image, dict[str, Any]]:
    with Image.open(path) as opened:
        opened.load()
        image = opened.convert("RGB").copy()
    if mode == "jpeg_random_epoch":
        rng = random.Random(stable_seed(config["seed"], "jpeg", epoch, sample_id))
        quality = rng.randint(int(config["jpeg_quality_min"]), int(config["jpeg_quality_max"]))
        subsampling = int(rng.choice(list(config["jpeg_subsampling_values"])))
        image = jpeg_recompress_pil(image, quality, subsampling)
        use_jpeg = True
    elif mode == "jpeg":
        quality, subsampling = deterministic_jpeg_parameters(config, sample_id)
        image = jpeg_recompress_pil(image, quality, subsampling)
        use_jpeg = True
    else:
        raise ValueError(f"Unknown image mode {mode!r}")
    image = resize_small_preserve_aspect(image, int(config["tile_size"]))
    return image, {
        "jpeg_applied": use_jpeg,
        "jpeg_quality": quality,
        "jpeg_subsampling": subsampling,
    }


def tile_native_tensor(
    image: torch.Tensor, tile_size: int, stride: int, padding_mode: str = "reflect"
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(image.shape)}")
    _, height, width = image.shape

    def layout(length: int) -> tuple[int, int]:
        count = max(1, math.ceil(max(length - tile_size, 0) / stride) + 1)
        target = (count - 1) * stride + tile_size
        return count, target

    rows, target_h = layout(height)
    columns, target_w = layout(width)
    pad_bottom, pad_right = target_h - height, target_w - width
    if pad_bottom or pad_right:
        mode = padding_mode
        if mode == "reflect" and (pad_bottom >= height or pad_right >= width):
            mode = "replicate"
        image = F.pad(image, (0, pad_right, 0, pad_bottom), mode=mode)
    tiles = (
        image.unfold(1, tile_size, stride)
        .unfold(2, tile_size, stride)
        .permute(1, 2, 0, 3, 4)
        .reshape(rows * columns, 3, tile_size, tile_size)
        .contiguous()
    )
    row_centers = (torch.arange(rows, dtype=torch.float32) * stride + tile_size / 2) / target_h
    col_centers = (torch.arange(columns, dtype=torch.float32) * stride + tile_size / 2) / target_w
    yy, xx = torch.meshgrid(row_centers, col_centers, indexing="ij")
    positions = torch.stack([yy * 2 - 1, xx * 2 - 1], dim=-1).reshape(-1, 2)
    return tiles, positions, {
        "original_height": height,
        "original_width": width,
        "padded_height": target_h,
        "padded_width": target_w,
        "rows": rows,
        "columns": columns,
    }


def load_native_tiles(
    path: Path,
    sample_id: str,
    config: dict[str, Any],
    mode: str,
    epoch: int = 0,
) -> dict[str, Any]:
    image, codec = prepare_working_image(path, sample_id, config, mode, epoch)
    tiles, positions, layout = tile_native_tensor(
        pil_to_tensor(image),
        int(config["tile_size"]),
        int(config["tile_stride"]),
        str(config["padding_mode"]),
    )
    return {
        "sample_id": sample_id,
        "tiles": tiles,
        "positions": positions,
        "layout": layout,
        **codec,
    }


def make_low_mask(size: int, radius: int, device: torch.device, dtype: torch.dtype, implementation: str) -> torch.Tensor:
    if implementation == "spai":
        half = torch.arange(size // 2, device=device, dtype=dtype)
        coordinates = torch.cat([half.flip(0), half], dim=0)
    elif implementation == "mfm":
        coordinates = torch.arange(size, device=device, dtype=dtype) - size // 2
    else:
        raise ValueError(f"Unknown mask implementation {implementation!r}")
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return ((xx.square() + yy.square()) < float(radius) ** 2).to(dtype)


def frequency_corrupt(images: torch.Tensor, keep_masks: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fftshift(torch.fft.fft2(images.float(), dim=(-2, -1)), dim=(-2, -1))
    reconstructed = torch.fft.ifft2(
        torch.fft.ifftshift(spectrum * keep_masks, dim=(-2, -1)), dim=(-2, -1)
    ).real
    return reconstructed.clamp(0.0, 1.0)


def per_tile_frequency_loss(
    predictions: torch.Tensor, targets: torch.Tensor, keep_masks: torch.Tensor
) -> torch.Tensor:
    prediction_spectrum = torch.fft.fftshift(
        torch.fft.fft2(predictions.float(), norm="ortho", dim=(-2, -1)), dim=(-2, -1)
    )
    target_spectrum = torch.fft.fftshift(
        torch.fft.fft2(targets.float(), norm="ortho", dim=(-2, -1)), dim=(-2, -1)
    )
    difference = prediction_spectrum - target_spectrum
    distance = torch.sqrt(difference.real.square() + difference.imag.square() + 1e-12)
    missing = 1.0 - keep_masks
    numerator = (distance * missing).sum(dim=(1, 2, 3))
    denominator = (missing.sum(dim=(1, 2, 3)) * targets.shape[1]).clamp_min(1.0)
    return numerator / denominator


# ---------------------------------------------------------------------------
# Stage 1: local token attention and image-balanced reconstruction
# ---------------------------------------------------------------------------


class NativeImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        root: Path,
        config: dict[str, Any],
        mode: str,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.config = config
        self.mode = mode
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return load_native_tiles(
            self.root / row.relative_path,
            str(row.sample_id),
            self.config,
            self.mode,
            self.epoch,
        )


def list_collate(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(items)


class LocalMFMModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="token"
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 16 * 16 * 3, kernel_size=1),
            nn.PixelShuffle(16),
        )

    def forward(self, normalized_spectral_tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encoder.forward_features(normalized_spectral_tiles)
        if tokens.ndim != 3 or tokens.shape[1:] != (197, 768):
            raise RuntimeError(f"Expected [B,197,768], got {tuple(tokens.shape)}")
        spatial = tokens[:, 1:].transpose(1, 2).reshape(-1, 768, 14, 14)
        reconstruction = self.decoder(spatial)
        return reconstruction, tokens[:, 0]

    def encode_cls(self, normalized_spectral_tiles: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_features(normalized_spectral_tiles)[:, 0]


def cosine_lr(update: int, total: int, warmup: int, base: float, minimum: float) -> float:
    if update < warmup:
        return base * (update + 1) / max(warmup, 1)
    progress = min(max((update - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def make_stage1_loader(
    dataset: NativeImageDataset, config: dict[str, Any], epoch: int, training: bool
) -> DataLoader:
    generator = torch.Generator().manual_seed(
        stable_seed(config["seed"], "stage1_loader", epoch, training)
    )
    workers = int(config["stage1_num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["stage1_images_per_batch"]),
        shuffle=training,
        num_workers=workers,
        collate_fn=list_collate,
        persistent_workers=workers > 0,
        pin_memory=False,
        generator=generator,
    )


def stage1_train_epoch(
    model: LocalMFMModel,
    dataset: NativeImageDataset,
    config: dict[str, Any],
    device: torch.device,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_update: int,
    total_updates: int,
    warmup_updates: int,
) -> tuple[dict[str, float], int]:
    dataset.set_epoch(epoch)
    loader = make_stage1_loader(dataset, config, epoch, True)
    model.train()
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    microbatch = int(config["stage1_tile_microbatch"])
    low_template = make_low_mask(
        int(config["tile_size"]),
        int(config["spectral_mask_radius"]),
        device,
        torch.float32,
        str(config["spectral_mask_implementation"]),
    ).view(1, 1, int(config["tile_size"]), int(config["tile_size"]))
    totals = {"image_loss": 0.0, "low_loss": 0.0, "high_loss": 0.0}
    counts = {"images": 0, "tiles": 0, "low": 0, "high": 0}
    started = time.time()
    print(
        f"[stage1][train] epoch={epoch + 1}/{config['stage1_epochs']} START "
        f"images={len(dataset):,}; all native tiles; image-balanced loss",
        flush=True,
    )
    for image_batch_index, image_batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        images_in_batch = len(image_batch)
        for item in image_batch:
            tiles = item["tiles"]
            tile_count = len(tiles)
            image_loss_value = 0.0
            for start in range(0, tile_count, microbatch):
                target = tiles[start : start + microbatch].to(device, non_blocking=True)
                choose_low = (
                    torch.rand(len(target), 1, 1, 1, device=device)
                    < float(config["low_probability"])
                )
                low = low_template.expand(len(target), -1, -1, -1)
                keep = torch.where(choose_low, low, 1.0 - low)
                corrupted = frequency_corrupt(target, keep)
                with torch.autocast(device_type=device.type, enabled=bool(config["amp"] and device.type == "cuda")):
                    reconstruction, _ = model((corrupted - mean) / std)
                tile_losses = per_tile_frequency_loss(reconstruction, target, keep)
                weighted = tile_losses.sum() / tile_count / images_in_batch
                scaler.scale(weighted).backward()
                detached = tile_losses.detach()
                image_loss_value += float(detached.sum().cpu()) / tile_count
                flags = choose_low.flatten()
                low_values = detached[flags]
                high_values = detached[~flags]
                if len(low_values):
                    totals["low_loss"] += float(low_values.sum().cpu())
                    counts["low"] += len(low_values)
                if len(high_values):
                    totals["high_loss"] += float(high_values.sum().cpu())
                    counts["high"] += len(high_values)
            totals["image_loss"] += image_loss_value
            counts["images"] += 1
            counts["tiles"] += tile_count

        lr = cosine_lr(
            global_update,
            total_updates,
            warmup_updates,
            float(config["stage1_learning_rate"]),
            float(config["stage1_min_learning_rate"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), float(config["stage1_grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        global_update += 1

        log_every = int(config["stage1_log_every_images"])
        if counts["images"] % log_every < images_in_batch or counts["images"] == len(dataset):
            elapsed = max(time.time() - started, 1e-6)
            rate = counts["images"] / elapsed
            eta = (len(dataset) - counts["images"]) / max(rate, 1e-6) / 60
            print(
                f"[stage1][train] epoch={epoch + 1} "
                f"images={counts['images']:,}/{len(dataset):,} tiles={counts['tiles']:,} "
                f"image_loss={totals['image_loss']/counts['images']:.6f} "
                f"low_loss={totals['low_loss']/max(counts['low'],1):.6f} "
                f"high_loss={totals['high_loss']/max(counts['high'],1):.6f} "
                f"lr={lr:.8f} speed={rate:.2f}_img/s eta={eta:.1f}_min",
                flush=True,
            )
    return {
        "image_mean_loss": totals["image_loss"] / counts["images"],
        "low_loss": totals["low_loss"] / max(counts["low"], 1),
        "high_loss": totals["high_loss"] / max(counts["high"], 1),
        "num_images": counts["images"],
        "num_tiles": counts["tiles"],
    }, global_update


@torch.inference_mode()
def stage1_validate_mode(
    model: LocalMFMModel,
    dataset: NativeImageDataset,
    config: dict[str, Any],
    device: torch.device,
    mode_name: str,
) -> dict[str, float]:
    loader = make_stage1_loader(dataset, config, 0, False)
    model.eval()
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    low_template = make_low_mask(
        int(config["tile_size"]),
        int(config["spectral_mask_radius"]),
        device,
        torch.float32,
        str(config["spectral_mask_implementation"]),
    ).view(1, 1, int(config["tile_size"]), int(config["tile_size"]))
    microbatch = int(config["stage1_tile_microbatch"])
    total_low = total_high = 0.0
    num_images = num_tiles = 0
    print(f"[stage1][validation_{mode_name}] START images={len(dataset):,}", flush=True)
    for image_batch in loader:
        for item in image_batch:
            tiles = item["tiles"]
            image_low = image_high = 0.0
            for start in range(0, len(tiles), microbatch):
                target = tiles[start : start + microbatch].to(device)
                low = low_template.expand(len(target), -1, -1, -1)
                for band, keep in (("low", low), ("high", 1.0 - low)):
                    corrupted = frequency_corrupt(target, keep)
                    with torch.autocast(device_type=device.type, enabled=bool(config["amp"] and device.type == "cuda")):
                        reconstruction, _ = model((corrupted - mean) / std)
                    values = per_tile_frequency_loss(reconstruction, target, keep)
                    if band == "low":
                        image_low += float(values.sum().cpu()) / len(tiles)
                    else:
                        image_high += float(values.sum().cpu()) / len(tiles)
            total_low += image_low
            total_high += image_high
            num_images += 1
            num_tiles += len(tiles)
    low_loss = total_low / num_images
    high_loss = total_high / num_images
    result = {
        "low_loss": low_loss,
        "high_loss": high_loss,
        "mean_loss": 0.5 * (low_loss + high_loss),
        "num_images": num_images,
        "num_tiles": num_tiles,
    }
    print(f"[stage1][validation_{mode_name}] COMPLETE {json.dumps(result)}", flush=True)
    return result


def save_local_encoder(model: LocalMFMModel, path: Path, metadata: dict[str, Any]) -> None:
    atomic_torch_save({"encoder": model.encoder.state_dict(), **metadata}, path)


def train_stage1(
    config: dict[str, Any], manifest_path: Path, root: Path, output_root: Path, resume_roots: Sequence[Path]
) -> tuple[Path, list[dict[str, Any]]]:
    manifest = pd.read_csv(manifest_path)
    train_frame = manifest[manifest.split == "real_train"].reset_index(drop=True)
    val_frame = manifest[manifest.split == "real_validation"].reset_index(drop=True)
    # Every training image is JPEG-normalized. A fresh deterministic Q70-100 /
    # subsampling realization is selected for each image and epoch.
    train_dataset = NativeImageDataset(train_frame, root, config, "jpeg_random_epoch")
    val_jpeg = NativeImageDataset(val_frame, root, config, "jpeg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalMFMModel().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["stage1_learning_rate"]),
        betas=tuple(config["stage1_betas"]),
        weight_decay=float(config["stage1_weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"] and device.type == "cuda"))
    updates_per_epoch = math.ceil(len(train_dataset) / int(config["stage1_images_per_batch"]))
    total_updates = updates_per_epoch * int(config["stage1_epochs"])
    warmup_updates = updates_per_epoch * int(config["stage1_warmup_epochs"])
    checkpoint_dir = output_root / "checkpoints"
    metrics_dir = output_root / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "stage1_local_last.pt"
    best_path = checkpoint_dir / "stage1_local_best.pt"
    encoder_path = checkpoint_dir / "stage1_local_encoder_best.pt"
    start_epoch = global_update = stale = 0
    best = float("inf")
    history: list[dict[str, Any]] = []
    resume = find_existing(Path("checkpoints/stage1_local_last.pt"), output_root, resume_roots)
    if config["stage1_resume"] and resume is not None:
        state = torch_load(resume)
        if state["manifest_sha256"] != sha256_file(manifest_path):
            raise RuntimeError("Stage 1 resume manifest mismatch.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        global_update = int(state["global_update"])
        best = float(state["best_validation"])
        stale = int(state["stale_epochs"])
        history = list(state["history"])

    for epoch in range(start_epoch, int(config["stage1_epochs"])):
        train_metrics, global_update = stage1_train_epoch(
            model, train_dataset, config, device, optimizer, scaler, epoch,
            global_update, total_updates, warmup_updates,
        )
        jpeg_metrics = stage1_validate_mode(model, val_jpeg, config, device, "jpeg")
        primary = jpeg_metrics["mean_loss"]
        improved = primary < best
        if improved:
            best, stale = primary, 0
        else:
            stale += 1
        row = {
            "epoch": epoch + 1,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_jpeg_{key}": value for key, value in jpeg_metrics.items()},
            "validation_primary_loss": primary,
            "global_update": global_update,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(metrics_dir / "stage1_training_history.csv", index=False)
        metadata = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "global_update": global_update,
            "best_validation": best,
            "stale_epochs": stale,
            "history": history,
            "config": config,
            "manifest_sha256": sha256_file(manifest_path),
            "architecture": "native_tiles_local_token_attention",
        }
        atomic_torch_save({"model": model.state_dict(), **metadata}, last_path)
        if improved:
            atomic_torch_save({"model": model.state_dict(), **metadata}, best_path)
            save_local_encoder(model, encoder_path, metadata)
        print(json.dumps(row, indent=2, default=_json_default), flush=True)
        if stale >= int(config["stage1_patience"]):
            print(f"Stage 1 early stopping after {stale} stale epochs.", flush=True)
            break
    if not encoder_path.is_file():
        existing_encoder = find_existing(
            Path("checkpoints/stage1_local_encoder_best.pt"), output_root, resume_roots
        )
        if existing_encoder is None:
            raise FileNotFoundError("Stage 1 did not produce stage1_local_encoder_best.pt")
        shutil.copy2(existing_encoder, encoder_path)
    return encoder_path, history


# ---------------------------------------------------------------------------
# Frozen Local-CLS cache for Stage 2
# ---------------------------------------------------------------------------


def load_frozen_local_encoder(path: Path, device: torch.device) -> LocalMFMModel:
    state = torch_load(path)
    model = LocalMFMModel()
    model.encoder.load_state_dict(state["encoder"], strict=True)
    model.decoder = nn.Identity()
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_band_cls(
    tiles: torch.Tensor,
    band: str,
    encoder: LocalMFMModel,
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    low_template = make_low_mask(
        int(config["tile_size"]), int(config["spectral_mask_radius"]), device,
        torch.float32, str(config["spectral_mask_implementation"]),
    ).view(1, 1, int(config["tile_size"]), int(config["tile_size"]))
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    results: list[torch.Tensor] = []
    microbatch = int(config["cache_local_microbatch"])
    for start in range(0, len(tiles), microbatch):
        batch = tiles[start : start + microbatch].to(device)
        low = low_template.expand(len(batch), -1, -1, -1)
        keep = low if band == "low" else 1.0 - low
        spectral = frequency_corrupt(batch, keep)
        with torch.autocast(device_type=device.type, enabled=bool(config["amp"] and device.type == "cuda")):
            cls = encoder.encode_cls((spectral - mean) / std)
        results.append(cls.float().cpu())
    return torch.cat(results, dim=0)


def cache_path(output_root: Path, split: str, index: int) -> Path:
    return output_root / "feature_cache" / f"{split}_{index:05d}.pt"


def cache_stage2_features(
    config: dict[str, Any], manifest_path: Path, root: Path, encoder_path: Path,
    output_root: Path, resume_roots: Sequence[Path],
) -> dict[str, Any]:
    manifest = pd.read_csv(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_frozen_local_encoder(encoder_path, device)
    encoder_sha = sha256_file(encoder_path)
    shard_size = int(config["cache_images_per_shard"])
    shard_start = config.get("cache_shard_start")
    shard_stop = config.get("cache_shard_stop")
    index_rows: list[dict[str, Any]] = []
    global_index = 0
    for split in SPLITS:
        split_frame = manifest[manifest.split == split].reset_index(drop=True)
        for start in range(0, len(split_frame), shard_size):
            shard_frame = split_frame.iloc[start : start + shard_size]
            destination = cache_path(output_root, split, start // shard_size)
            relative = destination.relative_to(output_root)
            existing = find_existing(relative, output_root, resume_roots)
            selected = (
                (shard_start is None or global_index >= int(shard_start))
                and (shard_stop is None or global_index < int(shard_stop))
            )
            if existing is None and selected:
                flattened: list[torch.Tensor] = []
                positions: list[torch.Tensor] = []
                offsets = [0]
                sample_ids: list[str] = []
                jpeg_qualities: list[int] = []
                started = time.time()
                for local_index, row in enumerate(shard_frame.itertuples(index=False), start=1):
                    path = root / row.relative_path
                    jpeg = load_native_tiles(path, row.sample_id, config, "jpeg")
                    jpeg_low = encode_band_cls(jpeg["tiles"], "low", encoder, config, device)
                    jpeg_high = encode_band_cls(jpeg["tiles"], "high", encoder, config, device)
                    features = torch.stack([jpeg_low, jpeg_high], dim=1)
                    flattened.append(features.to(torch.float16))
                    positions.append(jpeg["positions"].to(torch.float16))
                    offsets.append(offsets[-1] + len(features))
                    sample_ids.append(row.sample_id)
                    jpeg_qualities.append(int(jpeg["jpeg_quality"]))
                    if local_index % int(config["cache_log_every_images"]) == 0 or local_index == len(shard_frame):
                        elapsed = max(time.time() - started, 1e-6)
                        print(
                            f"[cache] shard={global_index} images={local_index}/{len(shard_frame)} "
                            f"tiles={offsets[-1]} speed={local_index/elapsed:.2f}_img/s",
                            flush=True,
                        )
                payload = {
                    "schema_version": 2,
                    "split": split,
                    "feature_names": FEATURE_NAMES,
                    "features": torch.cat(flattened, dim=0),
                    "positions": torch.cat(positions, dim=0),
                    "offsets": torch.tensor(offsets, dtype=torch.int64),
                    "sample_ids": sample_ids,
                    "jpeg_qualities": jpeg_qualities,
                    "manifest_sha256": sha256_file(manifest_path),
                    "local_encoder_sha256": encoder_sha,
                    "tile_size": int(config["tile_size"]),
                    "tile_stride": int(config["tile_stride"]),
                }
                atomic_torch_save(payload, destination)
                existing = destination
            available = destination if destination.is_file() else existing
            index_rows.append(
                {
                    "global_shard_index": global_index,
                    "split": split,
                    "relative_path": relative.as_posix(),
                    "num_images": len(shard_frame),
                    "complete": available is not None and available.is_file(),
                }
            )
            global_index += 1
    index = {
        "schema_version": 2,
        "manifest_sha256": sha256_file(manifest_path),
        "local_encoder_sha256": encoder_sha,
        "feature_names": FEATURE_NAMES,
        "jpeg_policy": "all_images_decode_rgb_then_seeded_jpeg_q70_100",
        "shards": index_rows,
        "num_complete_shards": sum(row["complete"] for row in index_rows),
        "num_total_shards": len(index_rows),
    }
    write_json(output_root / "feature_cache" / "index.json", index)
    return index


# ---------------------------------------------------------------------------
# Stage 2: low/high fusion, within-image attention and Image-CLS bottleneck
# ---------------------------------------------------------------------------


def positional_encoding_2d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError("global_dim must be divisible by four")
    quarter = dim // 4
    frequencies = torch.exp(
        torch.arange(quarter, device=positions.device, dtype=positions.dtype)
        * (-math.log(10000.0) / max(quarter - 1, 1))
    )
    y = positions[..., 0:1] * frequencies
    x = positions[..., 1:2] * frequencies
    return torch.cat([torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)], dim=-1)


class GlobalImageCLSBottleneck(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        dim = int(config["global_dim"])
        ff = int(dim * float(config["global_mlp_ratio"]))
        dropout = float(config["global_dropout"])
        self.dim = dim
        self.fusion = nn.Sequential(
            nn.Linear(2 * int(config["local_embedding_dim"]), dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.image_cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=int(config["global_heads"]), dim_feedforward=ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(config["global_depth"]), norm=nn.LayerNorm(dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=int(config["global_heads"]), dim_feedforward=ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=int(config["global_decoder_depth"]), norm=nn.LayerNorm(dim)
        )
        self.decoder_query = nn.Parameter(torch.zeros(1, 1, dim))
        self.low_head = nn.Linear(dim, int(config["local_embedding_dim"]))
        self.high_head = nn.Linear(dim, int(config["local_embedding_dim"]))
        nn.init.trunc_normal_(self.image_cls, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_query, std=0.02)

    def fuse(self, low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low_target = F.normalize(low.float(), dim=-1)
        high_target = F.normalize(high.float(), dim=-1)
        token = self.fusion(torch.cat([low_target, high_target], dim=-1))
        return token, low_target, high_target

    def encode(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
        masked: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token, low_target, high_target = self.fuse(low, high)
        if masked is not None:
            token = torch.where(masked.unsqueeze(-1), self.mask_token.expand_as(token), token)
        token = token + positional_encoding_2d(positions.float(), self.dim)
        cls = self.image_cls.expand(len(token), -1, -1)
        sequence = torch.cat([cls, token], dim=1)
        padding = torch.cat(
            [torch.zeros(len(token), 1, dtype=torch.bool, device=valid.device), ~valid], dim=1
        )
        output = self.encoder(sequence, src_key_padding_mask=padding)
        image_cls = output[:, 0]
        contextual = output[:, 1:]
        return image_cls, contextual, low_target, high_target

    def reconstruct_from_image_cls(
        self, image_cls: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.decoder_query.expand(len(image_cls), positions.shape[1], -1)
        queries = queries + positional_encoding_2d(positions.float(), self.dim)
        decoded = self.decoder(
            tgt=queries,
            memory=image_cls.unsqueeze(1),
            tgt_key_padding_mask=~valid,
        )
        return self.low_head(decoded), self.high_head(decoded)

    def forward(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
        masked: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_cls, contextual, low_target, high_target = self.encode(
            low, high, positions, valid, masked
        )
        low_prediction, high_prediction = self.reconstruct_from_image_cls(
            image_cls, positions, valid
        )
        return {
            "image_cls": image_cls,
            "contextual": contextual,
            "low_target": low_target.detach(),
            "high_target": high_target.detach(),
            "low_prediction": low_prediction,
            "high_prediction": high_prediction,
        }


def available_cache_shards(
    output_root: Path, resume_roots: Sequence[Path], split: str
) -> list[Path]:
    index_path = find_existing(Path("feature_cache/index.json"), output_root, resume_roots)
    if index_path is None:
        raise FileNotFoundError("Missing feature_cache/index.json; run cache phase first.")
    index = read_json(index_path)
    paths: list[Path] = []
    for row in index["shards"]:
        if row["split"] != split or not row["complete"]:
            continue
        relative = Path(row["relative_path"])
        existing = find_existing(relative, output_root, resume_roots)
        if existing is None:
            raise FileNotFoundError(f"Missing completed cache shard {relative}")
        paths.append(existing)
    if not paths:
        raise RuntimeError(f"No cache shards found for {split}")
    return paths


def samples_from_cache_shard(path: Path, order: Sequence[int], view: str, rng: random.Random) -> Iterator[dict[str, Any]]:
    payload = torch_load(path)
    if payload["feature_names"] != FEATURE_NAMES:
        raise RuntimeError(f"Unexpected feature order in {path}")
    offsets = payload["offsets"]
    for index in order:
        start, stop = int(offsets[index]), int(offsets[index + 1])
        features = payload["features"][start:stop].float()
        if view != "jpeg":
            raise ValueError(f"JPEG-only cache does not support view={view!r}")
        low, high = features[:, 0], features[:, 1]
        yield {
            "sample_id": payload["sample_ids"][index],
            "low": low,
            "high": high,
            "positions": payload["positions"][start:stop].float(),
        }


def collate_global(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    batch = len(samples)
    max_tiles = max(len(sample["low"]) for sample in samples)
    dim = samples[0]["low"].shape[-1]
    low = torch.zeros(batch, max_tiles, dim)
    high = torch.zeros_like(low)
    positions = torch.zeros(batch, max_tiles, 2)
    valid = torch.zeros(batch, max_tiles, dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = len(sample["low"])
        low[index, :count] = sample["low"]
        high[index, :count] = sample["high"]
        positions[index, :count] = sample["positions"]
        valid[index, :count] = True
    return {
        "sample_ids": [sample["sample_id"] for sample in samples],
        "low": low,
        "high": high,
        "positions": positions,
        "valid": valid,
    }


def iter_global_batches(
    shard_paths: Sequence[Path], config: dict[str, Any], epoch: int, view: str, shuffle: bool
) -> Iterator[dict[str, Any]]:
    rng = random.Random(stable_seed(config["seed"], "global_batches", epoch, view))
    paths = list(shard_paths)
    if shuffle:
        rng.shuffle(paths)
    pending: list[dict[str, Any]] = []
    pending_tokens = 0
    max_images = int(config["stage2_max_images_per_batch"])
    max_tokens = int(config["stage2_max_tokens_per_batch"])
    for path in paths:
        payload = torch_load(path)
        indices = list(range(len(payload["sample_ids"])))
        del payload
        if shuffle:
            rng.shuffle(indices)
        for sample in samples_from_cache_shard(path, indices, view, rng):
            count = len(sample["low"])
            if pending and (len(pending) >= max_images or pending_tokens + count > max_tokens):
                yield collate_global(pending)
                pending, pending_tokens = [], 0
            pending.append(sample)
            pending_tokens += count
    if pending:
        yield collate_global(pending)


def make_tile_mask(valid: torch.Tensor, ratio: float, seed: int | None = None) -> torch.Tensor:
    masked = torch.zeros_like(valid)
    rng = random.Random(seed) if seed is not None else None
    for row in range(len(valid)):
        indices = torch.where(valid[row])[0].tolist()
        count = max(1, int(round(len(indices) * ratio)))
        if rng is None:
            selected = random.sample(indices, min(count, len(indices)))
        else:
            selected = rng.sample(indices, min(count, len(indices)))
        masked[row, selected] = True
    return masked


def global_image_balanced_loss(
    output: dict[str, torch.Tensor], masked: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    low_per_token = (output["low_prediction"] - output["low_target"]).square().mean(dim=-1)
    high_per_token = (output["high_prediction"] - output["high_target"]).square().mean(dim=-1)
    counts = masked.sum(dim=1).clamp_min(1)
    low_per_image = (low_per_token * masked).sum(dim=1) / counts
    high_per_image = (high_per_token * masked).sum(dim=1) / counts
    low_loss = low_per_image.mean()
    high_loss = high_per_image.mean()
    loss = 0.5 * (low_loss + high_loss)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "low_loss": float(low_loss.detach().cpu()),
        "high_loss": float(high_loss.detach().cpu()),
    }


def run_stage2_epoch(
    model: GlobalImageCLSBottleneck,
    shard_paths: Sequence[Path],
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
    view: str,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "low_loss": 0.0, "high_loss": 0.0}
    num_images = num_tiles = num_masked = 0
    started = time.time()
    phase = "train" if training else f"validation_{view}"
    print(f"[stage2][{phase}] epoch={epoch + 1} START", flush=True)
    context = torch.enable_grad if training else torch.no_grad
    for batch_index, batch in enumerate(
        iter_global_batches(shard_paths, config, epoch, view, shuffle=training)
    ):
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        positions = batch["positions"].to(device)
        valid = batch["valid"].to(device)
        mask_seed = None if training else stable_seed(config["seed"], "val_mask", view, *batch["sample_ids"])
        masked = make_tile_mask(valid.cpu(), float(config["global_mask_ratio"]), mask_seed).to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with context():
            with torch.autocast(device_type=device.type, enabled=bool(config["amp"] and device.type == "cuda")):
                output = model(low, high, positions, valid, masked)
                loss, metrics = global_image_balanced_loss(output, masked)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(config["stage2_grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
        batch_images = len(low)
        for key in totals:
            totals[key] += metrics[key] * batch_images
        num_images += batch_images
        num_tiles += int(valid.sum().item())
        num_masked += int(masked.sum().item())
        if num_images % int(config["stage2_log_every_images"]) < batch_images:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[stage2][{phase}] images={num_images:,} tiles={num_tiles:,} "
                f"masked={num_masked:,} loss={totals['loss']/num_images:.6f} "
                f"low={totals['low_loss']/num_images:.6f} "
                f"high={totals['high_loss']/num_images:.6f} "
                f"speed={num_images/elapsed:.2f}_img/s",
                flush=True,
            )
    return {
        "loss": totals["loss"] / num_images,
        "low_loss": totals["low_loss"] / num_images,
        "high_loss": totals["high_loss"] / num_images,
        "num_images": num_images,
        "num_tiles": num_tiles,
        "num_masked_tiles": num_masked,
    }


def train_stage2(
    config: dict[str, Any], output_root: Path, manifest_sha: str, resume_roots: Sequence[Path]
) -> tuple[Path, list[dict[str, Any]]]:
    train_shards = available_cache_shards(output_root, resume_roots, "real_train")
    val_shards = available_cache_shards(output_root, resume_roots, "real_validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlobalImageCLSBottleneck(config).to(device)
    optimizer = AdamW(
        model.parameters(), lr=float(config["stage2_learning_rate"]),
        weight_decay=float(config["stage2_weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(config["amp"] and device.type == "cuda")
    )
    checkpoint_dir = output_root / "checkpoints"
    metrics_dir = output_root / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "stage2_global_last.pt"
    best_path = checkpoint_dir / "stage2_global_best.pt"
    start_epoch = stale = 0
    best = float("inf")
    history: list[dict[str, Any]] = []
    resume = find_existing(Path("checkpoints/stage2_global_last.pt"), output_root, resume_roots)
    if config["stage2_resume"] and resume is not None:
        state = torch_load(resume)
        if state["manifest_sha256"] != manifest_sha:
            raise RuntimeError("Stage 2 resume manifest mismatch.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        stale = int(state["stale_epochs"])
        best = float(state["best_validation"])
        history = list(state["history"])
    total_epochs = int(config["stage2_epochs"])
    for epoch in range(start_epoch, total_epochs):
        lr = float(config["stage2_min_learning_rate"]) + 0.5 * (
            float(config["stage2_learning_rate"]) - float(config["stage2_min_learning_rate"])
        ) * (1 + math.cos(math.pi * epoch / max(total_epochs - 1, 1)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_metrics = run_stage2_epoch(
            model, train_shards, config, device, epoch, "jpeg", optimizer, scaler
        )
        jpeg = run_stage2_epoch(
            model, val_shards, config, device, epoch, "jpeg", None, None
        )
        primary = jpeg["loss"]
        improved = primary < best
        if improved:
            best, stale = primary, 0
        else:
            stale += 1
        row = {
            "epoch": epoch + 1,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_jpeg_{key}": value for key, value in jpeg.items()},
            "validation_primary_loss": primary,
            "learning_rate": lr,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(metrics_dir / "stage2_training_history.csv", index=False)
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch, "stale_epochs": stale, "best_validation": best,
            "history": history, "config": config, "manifest_sha256": manifest_sha,
            "architecture": "low_high_concat_image_cls_bottleneck",
        }
        atomic_torch_save(state, last_path)
        if improved:
            atomic_torch_save(state, best_path)
        print(json.dumps(row, indent=2, default=_json_default), flush=True)
        if stale >= int(config["stage2_patience"]):
            print(f"Stage 2 early stopping after {stale} stale epochs.", flush=True)
            break
    if not best_path.is_file():
        existing_best = find_existing(
            Path("checkpoints/stage2_global_best.pt"), output_root, resume_roots
        )
        if existing_best is None:
            raise FileNotFoundError("Stage 2 did not produce stage2_global_best.pt")
        shutil.copy2(existing_best, best_path)
    return best_path, history


# ---------------------------------------------------------------------------
# Stage 3: one Image CLS per image -> StandardScaler -> real-only GMM
# ---------------------------------------------------------------------------


@torch.inference_mode()
def extract_image_cls(
    model: GlobalImageCLSBottleneck,
    shard_paths: Sequence[Path],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    model.eval()
    vectors: list[np.ndarray] = []
    sample_ids: list[str] = []
    # The exact seeded JPEG Q70-100 view used by Stage 2; no raw/original view.
    for batch in iter_global_batches(shard_paths, config, 0, "jpeg", shuffle=False):
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        positions = batch["positions"].to(device)
        valid = batch["valid"].to(device)
        with torch.autocast(device_type=device.type, enabled=bool(config["amp"] and device.type == "cuda")):
            image_cls, _, _, _ = model.encode(low, high, positions, valid, masked=None)
        vectors.append(image_cls.float().cpu().numpy())
        sample_ids.extend(batch["sample_ids"])
    return np.concatenate(vectors, axis=0), sample_ids


def fit_stage3_gmm(
    config: dict[str, Any], output_root: Path, global_checkpoint: Path,
    resume_roots: Sequence[Path],
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch_load(global_checkpoint)
    model = GlobalImageCLSBottleneck(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    embeddings: dict[str, np.ndarray] = {}
    identifiers: dict[str, list[str]] = {}
    embedding_dir = output_root / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        shards = available_cache_shards(output_root, resume_roots, split)
        matrix, ids = extract_image_cls(model, shards, config, device)
        embeddings[split] = matrix
        identifiers[split] = ids
        np.save(embedding_dir / f"{split}_image_cls.npy", matrix)
        pd.DataFrame({"sample_id": ids}).to_csv(
            embedding_dir / f"{split}_sample_ids.csv", index=False
        )
    train = embeddings["real_train"]
    validation = embeddings["real_validation"]
    calibration = embeddings["real_calibration"]
    if train.shape[1] != 768:
        raise AssertionError(f"GMM requires one 768-D vector per image, got {train.shape}")
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    validation_scaled = scaler.transform(validation)
    calibration_scaled = scaler.transform(calibration)
    candidates: list[tuple[float, GaussianMixture, dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    for components in config["gmm_components"]:
        gmm = GaussianMixture(
            n_components=int(components),
            covariance_type=str(config["gmm_covariance_type"]),
            reg_covar=float(config["gmm_reg_covar"]),
            max_iter=int(config["gmm_max_iter"]),
            n_init=int(config["gmm_n_init"]),
            random_state=int(config["seed"]),
        ).fit(train_scaled)
        occupancy = gmm.predict_proba(train_scaled).mean(axis=0)
        row = {
            "n_components": int(components),
            "converged": bool(gmm.converged_),
            "n_iter": int(gmm.n_iter_),
            "train_log_likelihood": float(gmm.score(train_scaled)),
            "validation_log_likelihood": float(gmm.score(validation_scaled)),
            "train_bic": float(gmm.bic(train_scaled)),
            "min_soft_occupancy": float(occupancy.min()),
            "valid_occupancy": bool(
                occupancy.min() >= float(config["gmm_min_component_occupancy"])
            ),
        }
        rows.append(row)
        if row["converged"] and row["valid_occupancy"]:
            candidates.append((row["validation_log_likelihood"], gmm, row))
    if not candidates:
        raise RuntimeError("No GMM candidate passed convergence and occupancy checks.")
    _, selected, selected_row = max(candidates, key=lambda item: item[0])
    model_dir = output_root / "gmm"
    metrics_dir = output_root / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "stage3_real_feature_scaler.joblib")
    joblib.dump(selected, model_dir / "stage3_real_distribution_gmm.joblib")
    pd.DataFrame(rows).to_csv(metrics_dir / "stage3_gmm_selection.csv", index=False)
    calibration_nll = -selected.score_samples(calibration_scaled)
    thresholds = {
        "nll_q90": float(np.quantile(calibration_nll, 0.90)),
        "nll_q95": float(np.quantile(calibration_nll, 0.95)),
        "nll_q99": float(np.quantile(calibration_nll, 0.99)),
        "primary": "nll_q95",
    }
    write_json(model_dir / "stage3_real_only_thresholds.json", thresholds)
    component_std = np.sqrt(selected.covariances_)
    np.savez_compressed(
        model_dir / "stage3_gmm_component_statistics.npz",
        weights=selected.weights_, means=selected.means_, std=component_std,
    )
    summary = {
        "gmm_input": "one final-layernorm Image CLS 768-D per image",
        "image_cls_shape_train": list(train.shape),
        "selected": selected_row,
        "thresholds": thresholds,
        "global_checkpoint_sha256": sha256_file(global_checkpoint),
        "feature_augmentation": "all images decoded RGB then seeded JPEG Q70-100",
    }
    write_json(metrics_dir / "stage3_gmm_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(user_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if user_config:
        config.update(user_config)
    seed_everything(int(config["seed"]))
    phases = [str(value).lower() for value in config["run_phases"]]
    allowed = {"manifest", "stage1", "cache", "stage2", "gmm"}
    unknown = set(phases) - allowed
    if unknown:
        raise ValueError(f"Unknown phases: {sorted(unknown)}")
    if int(config["tile_stride"]) <= 0 or int(config["tile_stride"]) > int(config["tile_size"]):
        raise ValueError("tile_stride must be in [1, tile_size].")
    if int(config["jpeg_quality_min"]) != 70 or int(config["jpeg_quality_max"]) != 100:
        raise ValueError("This experiment requires JPEG quality range [70, 100].")
    output_root = Path(config["output_parent"]) / str(config["run_name"])
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "config.json", config)
    root = find_imagenet_train_root(config.get("imagenet_train_root"))
    resume_roots = discover_resume_roots(config, str(config["run_name"]))
    update_status(output_root, "manifest", "running")
    manifest_path, manifest_summary = ensure_manifest(
        config, root, output_root, resume_roots
    )
    update_status(output_root, "manifest", "complete", **manifest_summary)
    result: dict[str, Any] = {
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_summary["manifest_sha256"],
    }
    try:
        encoder_path = find_existing(
            Path("checkpoints/stage1_local_encoder_best.pt"), output_root, resume_roots
        )
        if "stage1" in phases:
            update_status(output_root, "stage1", "running")
            encoder_path, history = train_stage1(
                config, manifest_path, root, output_root, resume_roots
            )
            result["stage1_epochs"] = len(history)
            update_status(output_root, "stage1", "complete", checkpoint=str(encoder_path))

        if "cache" in phases:
            if encoder_path is None:
                raise FileNotFoundError("Cache requires stage1_local_encoder_best.pt")
            update_status(output_root, "cache", "running")
            cache_index = cache_stage2_features(
                config, manifest_path, root, encoder_path, output_root, resume_roots
            )
            result["cache"] = cache_index
            cache_state = (
                "complete"
                if cache_index["num_complete_shards"] == cache_index["num_total_shards"]
                else "partial"
            )
            update_status(output_root, "cache", cache_state)

        global_path = find_existing(
            Path("checkpoints/stage2_global_best.pt"), output_root, resume_roots
        )
        if "stage2" in phases:
            update_status(output_root, "stage2", "running")
            global_path, history = train_stage2(
                config, output_root, manifest_summary["manifest_sha256"], resume_roots
            )
            result["stage2_epochs"] = len(history)
            update_status(output_root, "stage2", "complete", checkpoint=str(global_path))

        if "gmm" in phases:
            if global_path is None:
                raise FileNotFoundError("GMM requires stage2_global_best.pt")
            update_status(output_root, "gmm", "running")
            result["gmm"] = fit_stage3_gmm(
                config, output_root, global_path, resume_roots
            )
            update_status(output_root, "gmm", "complete")
        write_json(output_root / "run_summary.json", result)
        update_status(output_root, "pipeline", "complete")
        print(f"Pipeline completed: {output_root}", flush=True)
        return result
    except Exception as exc:
        update_status(
            output_root, "pipeline", "failed",
            error_type=type(exc).__name__, error_message=str(exc),
        )
        raise


if __name__ == "__main__":
    run_pipeline()
