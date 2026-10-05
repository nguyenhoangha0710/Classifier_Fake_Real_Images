"""Train the real-only D2-FAD Spectral-GMM branch on Kaggle ImageNet.

Pipeline
--------
1. Build a deterministic 100-images-per-synset manifest from ImageNet filenames.
2. Cache frozen official-MFM local CLS vectors for raw/JPEG and low/high views.
3. Train a global masked-feature Transformer with an Image-CLS bottleneck.
4. Fit a real-only diagonal GMM and calibrate real-only anomaly thresholds.

The code intentionally refuses to use a random or supervised ViT fallback when
the official MFM checkpoint is absent or incompatible.
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
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

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
from torchvision import transforms as vision_transforms
from torchvision.transforms import InterpolationMode

try:
    import timm
except ImportError as exc:  # pragma: no cover - Kaggle normally includes timm
    raise ImportError("This notebook requires timm. Install it before running.") from exc


ImageFile.LOAD_TRUNCATED_IMAGES = False


DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": "spectral_gmm_imagenet100k_v1",
    "run_phases": ["cache", "train", "gmm"],
    "output_parent": "/kaggle/working/spectral_gmm_branch",
    "imagenet_train_root": None,
    "mfm_checkpoint_path": None,
    "mfm_source": "official_checkpoint",
    "resume_run_roots": [],
    "seed": 20261003,
    "images_per_synset": 100,
    "train_per_synset": 80,
    "validation_per_synset": 10,
    "calibration_per_synset": 10,
    "expected_synsets": 1000,
    "strict_synset_count": True,
    "valid_extensions": [".jpeg", ".jpg", ".png"],
    "tile_size": 224,
    "mfm_mask_radius": 16,
    "spectral_mask_implementation": "spai",
    "local_embedding_dim": 768,
    "local_mfm_epochs": 20,
    "local_mfm_warmup_epochs": 2,
    "local_mfm_batch_size": 32,
    "local_mfm_accumulation_steps": 4,
    "local_mfm_learning_rate": 3.0e-4,
    "local_mfm_min_learning_rate": 2.5e-6,
    "local_mfm_weight_decay": 0.05,
    "local_mfm_betas": [0.9, 0.95],
    "local_mfm_grad_clip": 3.0,
    "local_mfm_patience": 3,
    "local_mfm_num_workers": 4,
    "local_mfm_log_every_batches": 1,
    "local_mfm_train_crop_scale": [0.2, 1.0],
    "local_mfm_low_pass_probability": 0.5,
    "local_mfm_jpeg_probability": 0.5,
    "local_mfm_validation_limit": None,
    "local_mfm_resume": True,
    "local_feature_batch_size": 64,
    "cache_images_per_shard": 500,
    "cache_shard_start": None,
    "cache_shard_stop": None,
    "jpeg_quality_min": 70,
    "jpeg_quality_max": 100,
    "jpeg_subsampling_values": [0, 1, 2],
    "global_dim": 768,
    "global_depth": 4,
    "global_heads": 12,
    "global_mlp_ratio": 4.0,
    "global_dropout": 0.1,
    "global_decoder_depth": 2,
    "global_mask_ratio": 0.40,
    "jpeg_input_probability": 0.50,
    "global_batch_size": 64,
    "global_max_padded_tokens_per_batch": 768,
    "global_epochs": 10,
    "global_learning_rate": 1.0e-4,
    "global_weight_decay": 0.05,
    "global_min_learning_rate": 1.0e-6,
    "global_grad_clip": 1.0,
    "global_mse_weight": 1.0,
    "global_consistency_weight": 0.10,
    "global_patience": 3,
    "amp": True,
    "resume_training": True,
    "gmm_components": [1, 2, 4, 8, 16],
    "gmm_covariance_type": "diag",
    "gmm_reg_covar": 1.0e-6,
    "gmm_max_iter": 100,
    "gmm_n_init": 1,
    "gmm_min_component_occupancy": 0.001,
}


SPLIT_TO_PREFIX = {
    "real_train": "train",
    "real_validation": "validation",
    "real_calibration": "calibration",
}
FEATURE_NAMES = ["raw_low", "raw_high", "jpeg_low", "jpeg_high"]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
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
    text = ":".join([str(global_seed), *map(str, parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def find_imagenet_train_root(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            Path("/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"),
            Path("/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/"),
            Path("/kaggle/input/imagenet-object-localization-challenge/Data/CLS-LOC/train"),
        ]
    )

    # Kaggle occasionally changes the mounted dataset slug or adds one wrapper
    # directory. Search only bounded structural patterns; never recursively walk
    # through the 1.2M ImageNet files.
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        dataset_roots = sorted(path for path in input_root.iterdir() if path.is_dir())
        structural_suffixes = (
            Path("ILSVRC/Data/CLS-LOC/train"),
            Path("Data/CLS-LOC/train"),
            Path("CLS-LOC/train"),
            Path("train"),
        )
        for dataset_root in dataset_roots:
            for suffix in structural_suffixes:
                candidates.append(dataset_root / suffix)
            # Support a single extra wrapper such as dataset/version/ILSVRC/...
            try:
                wrappers = [path for path in dataset_root.iterdir() if path.is_dir()]
            except PermissionError:
                wrappers = []
            for wrapper in wrappers:
                for suffix in structural_suffixes[:-1]:
                    candidates.append(wrapper / suffix)

    checked: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        checked.append(normalized)
        if candidate.is_dir():
            synset_preview = [
                child
                for child in candidate.iterdir()
                if child.is_dir() and child.name.startswith("n")
            ]
            if synset_preview:
                resolved = candidate.resolve()
                print(f"Detected ImageNet train root: {resolved}")
                return resolved

    mounted = []
    if input_root.is_dir():
        mounted = [path.name for path in input_root.iterdir()]
    raise FileNotFoundError(
        "Cannot find ImageNet train. Add the Kaggle competition input and ensure "
        "ILSVRC/Data/CLS-LOC/train exists. If you just added the input, restart "
        "the Kaggle session.\n"
        f"Mounted /kaggle/input entries: {mounted}\n"
        f"Checked candidate paths (first 20): {checked[:20]}"
    )


def find_mfm_checkpoint(explicit: str | None, imagenet_root: Path) -> Path:
    if explicit:
        checkpoint = Path(explicit)
        if checkpoint.is_file():
            return checkpoint.resolve()
        raise FileNotFoundError(f"Configured MFM checkpoint does not exist: {checkpoint}")

    input_root = Path("/kaggle/input")
    matches: list[Path] = []
    if input_root.is_dir():
        for dataset_root in sorted(input_root.iterdir()):
            if not dataset_root.is_dir() or dataset_root in imagenet_root.parents:
                continue
            for pattern in ("mfm_pretrain_vit_base.pth", "*mfm*vit*base*.pth"):
                matches.extend(dataset_root.rglob(pattern))
    unique = sorted({path.resolve() for path in matches if path.is_file()})
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError(
            "Multiple MFM checkpoints found. Set CONFIG['mfm_checkpoint_path'] explicitly:\n"
            + "\n".join(map(str, unique))
        )
    raise FileNotFoundError(
        "Missing mfm_pretrain_vit_base.pth. Add it as a Kaggle Dataset input. "
        "The pipeline intentionally has no random or supervised-ViT fallback."
    )


def discover_resume_roots(config: dict[str, Any], run_name: str) -> list[Path]:
    roots: list[Path] = []
    for raw in config.get("resume_run_roots", []):
        path = Path(raw)
        if path.is_dir():
            roots.append(path.resolve())

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for dataset_root in sorted(kaggle_input.iterdir()):
            if not dataset_root.is_dir():
                continue
            candidates = [
                dataset_root / run_name,
                dataset_root / "spectral_gmm_branch" / run_name,
            ]
            for candidate in candidates:
                if candidate.is_dir():
                    roots.append(candidate.resolve())
    deduplicated: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            seen.add(root)
            deduplicated.append(root)
    return deduplicated


def find_existing_relative(
    relative: str | Path,
    output_root: Path,
    resume_roots: Sequence[Path],
) -> Path | None:
    relative = Path(relative)
    candidates = [output_root / relative, *[root / relative for root in resume_roots]]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def update_status(output_root: Path, phase: str, state: str, **extra: Any) -> None:
    path = output_root / "status.json"
    payload = read_json(path) if path.is_file() else {}
    payload.update(
        {
            "phase": phase,
            "state": state,
            "updated_at_unix": time.time(),
            **extra,
        }
    )
    write_json(path, payload)


def build_manifest(
    config: dict[str, Any],
    imagenet_root: Path,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = output_root / "manifests" / "imagenet_real_100k.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    synset_dirs = sorted(path for path in imagenet_root.iterdir() if path.is_dir())
    expected = int(config["expected_synsets"])
    if config.get("strict_synset_count", True) and len(synset_dirs) != expected:
        raise AssertionError(f"Expected {expected} synsets, found {len(synset_dirs)}")

    images_per_synset = int(config["images_per_synset"])
    train_count = int(config["train_per_synset"])
    validation_count = int(config["validation_per_synset"])
    calibration_count = int(config["calibration_per_synset"])
    if train_count + validation_count + calibration_count != images_per_synset:
        raise ValueError("Per-synset split counts must sum to images_per_synset")

    valid_extensions = {suffix.lower() for suffix in config["valid_extensions"]}
    rows: list[dict[str, Any]] = []

    print(
        "Building deterministic manifest from filenames only; images are not "
        "opened or decoded.",
        flush=True,
    )
    for synset_dir in tqdm(
        synset_dirs,
        desc="Build ImageNet manifest",
        unit="synset",
        dynamic_ncols=True,
    ):
        # Suffix filtering avoids costly PIL.open(), stat() and verify() calls.
        candidates = sorted(
            path
            for path in synset_dir.iterdir()
            if path.suffix.lower() in valid_extensions
        )
        if len(candidates) < images_per_synset:
            raise RuntimeError(
                f"Synset {synset_dir.name} contains {len(candidates)} candidate "
                f"images; required {images_per_synset}."
            )
        rng = random.Random(stable_seed(config["seed"], "sample", synset_dir.name))
        rng.shuffle(candidates)
        selected = candidates[:images_per_synset]

        split_rng = random.Random(stable_seed(config["seed"], "split", synset_dir.name))
        split_rng.shuffle(selected)
        boundaries = (train_count, train_count + validation_count)
        for index, candidate in enumerate(selected):
            if index < boundaries[0]:
                split = "real_train"
            elif index < boundaries[1]:
                split = "real_validation"
            else:
                split = "real_calibration"
            rows.append(
                {
                    "sample_id": f"{synset_dir.name}/{candidate.stem}",
                    "relative_path": candidate.relative_to(imagenet_root).as_posix(),
                    "synset": synset_dir.name,
                    "extension": candidate.suffix.lower(),
                    "split": split,
                    "sampling_seed": stable_seed(
                        config["seed"], synset_dir.name, candidate.name
                    ),
                }
            )

    manifest = pd.DataFrame(rows).sort_values(
        ["split", "synset", "relative_path"], ignore_index=True
    )
    expected_total = len(synset_dirs) * images_per_synset
    if len(manifest) != expected_total:
        raise AssertionError(f"Expected {expected_total} rows, found {len(manifest)}")

    per_synset_split = manifest.groupby(["synset", "split"]).size().unstack(fill_value=0)
    required = {
        "real_train": train_count,
        "real_validation": validation_count,
        "real_calibration": calibration_count,
    }
    for split, count in required.items():
        if not (per_synset_split[split] == count).all():
            raise AssertionError(f"Per-synset count mismatch for {split}")

    manifest.to_csv(manifest_path, index=False)
    manifest_sha = sha256_file(manifest_path)
    summary = {
        "imagenet_train_root": str(imagenet_root),
        "num_synsets": len(synset_dirs),
        "num_selected_images": len(manifest),
        "split_counts": manifest["split"].value_counts().sort_index().to_dict(),
        "images_per_synset": images_per_synset,
        "manifest_sha256": manifest_sha,
        "extensions": manifest["extension"].value_counts().to_dict(),
        "selection_mode": "filename_only_no_image_audit",
        "config_hash": config_hash(config),
    }
    write_json(output_root / "manifests" / "manifest_summary.json", summary)
    return manifest_path, summary


def ensure_manifest(
    config: dict[str, Any],
    imagenet_root: Path,
    output_root: Path,
    resume_roots: Sequence[Path],
) -> tuple[Path, dict[str, Any]]:
    relative_manifest = Path("manifests/imagenet_real_100k.csv")
    existing = find_existing_relative(relative_manifest, output_root, resume_roots)
    if existing is None:
        return build_manifest(config, imagenet_root, output_root)

    destination = output_root / relative_manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    if existing.resolve() != destination.resolve():
        shutil.copy2(existing, destination)
        for relative in ("manifests/manifest_summary.json",):
            source = find_existing_relative(relative, output_root, resume_roots)
            if source:
                target = output_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
    summary_path = output_root / "manifests" / "manifest_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {
        "manifest_sha256": sha256_file(destination),
        "num_selected_images": len(pd.read_csv(destination)),
    }
    return destination, summary


# ---------------------------------------------------------------------------
# Stage 1A: MFM reconstruction pre-training on real ImageNet images
# ---------------------------------------------------------------------------


class MFMRealDataset(Dataset):
    """Real-only ImageNet subset with the official MFM spatial augmentations."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        imagenet_root: Path,
        config: dict[str, Any],
        training: bool,
    ) -> None:
        self.rows = manifest.reset_index(drop=True)
        self.imagenet_root = imagenet_root
        self.config = config
        self.training = training
        crop_scale = tuple(float(value) for value in config["local_mfm_train_crop_scale"])
        if training:
            self.spatial_transform = vision_transforms.Compose(
                [
                    vision_transforms.RandomResizedCrop(
                        224,
                        scale=crop_scale,
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    vision_transforms.RandomHorizontalFlip(),
                    vision_transforms.ToTensor(),
                ]
            )
        else:
            self.spatial_transform = vision_transforms.Compose(
                [
                    vision_transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                    vision_transforms.CenterCrop(224),
                    vision_transforms.ToTensor(),
                ]
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.rows.iloc[index]
        path = self.imagenet_root / row.relative_path
        with Image.open(path) as image:
            image.load()
            tensor = self.spatial_transform(image.convert("RGB"))

        # Project extension for compression robustness. Set probability=0 to
        # reproduce the original MFM spatial preprocessing exactly.
        jpeg_probability = float(self.config["local_mfm_jpeg_probability"])
        if self.training and torch.rand(()) < jpeg_probability:
            quality = int(
                torch.randint(
                    int(self.config["jpeg_quality_min"]),
                    int(self.config["jpeg_quality_max"]) + 1,
                    (1,),
                ).item()
            )
            subsampling_values = list(self.config["jpeg_subsampling_values"])
            subsampling = int(
                subsampling_values[
                    int(torch.randint(0, len(subsampling_values), (1,)).item())
                ]
            )
            tensor = jpeg_recompress(tensor, quality, subsampling)
        return tensor, str(row.sample_id)


def make_mfm_low_pass_mask(
    size: int,
    radius: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Exact circular low-pass convention from the official MFM data loader."""
    coordinates = torch.arange(size, device=device, dtype=dtype) - size // 2
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return ((xx.square() + yy.square()) < float(radius) ** 2).to(dtype)


def sample_mfm_keep_masks(
    batch_size: int,
    size: int,
    radius: int,
    low_pass_probability: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    low = make_mfm_low_pass_mask(size, radius, device, dtype)
    low = low.view(1, 1, size, size).expand(batch_size, -1, -1, -1)
    choose_low = (
        torch.rand(batch_size, 1, 1, 1, device=device) < low_pass_probability
    )
    return torch.where(choose_low, low, 1.0 - low)


def mfm_frequency_corrupt(
    images: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(images.float(), dim=(-2, -1)), dim=(-2, -1)
    )
    corrupted = torch.fft.ifft2(
        torch.fft.ifftshift(spectrum * keep_mask, dim=(-2, -1)),
        dim=(-2, -1),
    ).real
    return corrupted.clamp(0.0, 1.0)


def masked_frequency_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Official MFM objective: complex spectral distance on missing frequencies."""
    prediction_spectrum = torch.fft.fftshift(
        torch.fft.fft2(predictions.float(), norm="ortho", dim=(-2, -1)),
        dim=(-2, -1),
    )
    target_spectrum = torch.fft.fftshift(
        torch.fft.fft2(targets.float(), norm="ortho", dim=(-2, -1)),
        dim=(-2, -1),
    )
    difference = prediction_spectrum - target_spectrum
    distance = torch.sqrt(difference.real.square() + difference.imag.square() + 1e-12)
    missing_mask = 1.0 - keep_mask
    denominator = missing_mask.sum().clamp_min(1.0) * targets.shape[1]
    return (distance * missing_mask).sum() / denominator


class MFMReconstructionModel(nn.Module):
    """ViT-B/16 plus the depth-0 pixel decoder used by official MFM."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="token",
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 16 * 16 * 3, kernel_size=1),
            nn.PixelShuffle(16),
        )

    def forward(self, corrupted_normalized: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_features(corrupted_normalized)
        if tokens.ndim != 3 or tokens.shape[1] != 197:
            raise RuntimeError(
                f"Expected ViT tokens [B,197,768], got {tuple(tokens.shape)}"
            )
        patch_tokens = tokens[:, 1:].transpose(1, 2).reshape(-1, 768, 14, 14)
        return self.decoder(patch_tokens)


def build_mfm_loader(
    dataset: MFMRealDataset,
    config: dict[str, Any],
    epoch: int,
    training: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(stable_seed(config["seed"], "local_mfm_loader", epoch, training))
    workers = int(config["local_mfm_num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["local_mfm_batch_size"]),
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        persistent_workers=workers > 0,
        generator=generator,
    )


def local_mfm_learning_rate(
    update: int,
    total_updates: int,
    warmup_updates: int,
    base_lr: float,
    min_lr: float,
) -> float:
    if update < warmup_updates:
        return base_lr * float(update + 1) / max(warmup_updates, 1)
    progress = (update - warmup_updates) / max(total_updates - warmup_updates, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def run_local_mfm_epoch(
    model: MFMReconstructionModel,
    dataset: MFMRealDataset,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
    optimizer: AdamW | None,
    scaler: torch.cuda.amp.GradScaler,
    global_update: int,
    total_updates: int,
    warmup_updates: int,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    loader = build_mfm_loader(dataset, config, epoch, training)
    accumulation = int(config["local_mfm_accumulation_steps"]) if training else 1
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    total_loss, total_images = 0.0, 0
    if training:
        optimizer.zero_grad(set_to_none=True)

    phase_name = "train" if training else "validation"
    log_every_batches = max(
        1, int(config.get("local_mfm_log_every_batches", 1))
    )
    expected_images = (
        len(loader) * int(loader.batch_size)
        if training and loader.drop_last
        else len(dataset)
    )
    epoch_started_at = time.time()
    print(
        f"[local_mfm][{phase_name}] epoch={epoch + 1}/"
        f"{int(config['local_mfm_epochs'])} START "
        f"batches={len(loader):,} images={expected_images:,}; "
        "waiting for first batch...",
        flush=True,
    )
    progress = tqdm(
        loader,
        desc=f"local MFM {phase_name} {epoch + 1}",
        dynamic_ncols=True,
    )
    grad_context = torch.enable_grad if training else torch.no_grad
    for step, (images, _) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        masks_to_evaluate: list[torch.Tensor]
        if training:
            masks_to_evaluate = [
                sample_mfm_keep_masks(
                    len(images),
                    224,
                    int(config["mfm_mask_radius"]),
                    float(config["local_mfm_low_pass_probability"]),
                    device,
                    images.dtype,
                )
            ]
        else:
            low = make_mfm_low_pass_mask(
                224, int(config["mfm_mask_radius"]), device, images.dtype
            ).view(1, 1, 224, 224).expand(len(images), -1, -1, -1)
            masks_to_evaluate = [low, 1.0 - low]

        batch_loss = torch.zeros((), device=device)
        with grad_context():
            for keep_mask in masks_to_evaluate:
                corrupted = mfm_frequency_corrupt(images, keep_mask)
                normalized = (corrupted - mean) / std
                amp_enabled = bool(config["amp"] and device.type == "cuda")
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    reconstruction = model(normalized)
                reconstruction_loss = masked_frequency_reconstruction_loss(
                    reconstruction, images, keep_mask
                )
                batch_loss = batch_loss + reconstruction_loss
            batch_loss = batch_loss / len(masks_to_evaluate)

        if training:
            scaled_loss = batch_loss / accumulation
            scaler.scale(scaled_loss).backward()
            should_update = (step + 1) % accumulation == 0 or (step + 1) == len(loader)
            if should_update:
                lr = local_mfm_learning_rate(
                    global_update,
                    total_updates,
                    warmup_updates,
                    float(config["local_mfm_learning_rate"]),
                    float(config["local_mfm_min_learning_rate"]),
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["local_mfm_grad_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

        batch_loss_value = float(batch_loss.detach().cpu())
        total_loss += batch_loss_value * len(images)
        total_images += len(images)
        running_loss = total_loss / max(total_images, 1)
        progress.set_postfix(loss=f"{running_loss:.5f}")

        completed_batches = step + 1
        if completed_batches % log_every_batches == 0 or completed_batches == len(loader):
            elapsed_seconds = max(time.time() - epoch_started_at, 1e-6)
            images_per_second = total_images / elapsed_seconds
            remaining_images = max(expected_images - total_images, 0)
            eta_seconds = remaining_images / max(images_per_second, 1e-6)
            learning_rate = (
                float(optimizer.param_groups[0]["lr"]) if training else 0.0
            )
            print(
                f"[local_mfm][{phase_name}] epoch={epoch + 1}/"
                f"{int(config['local_mfm_epochs'])} "
                f"batch={completed_batches:,}/{len(loader):,} "
                f"images={total_images:,}/{expected_images:,} "
                f"batch_loss={batch_loss_value:.6f} "
                f"running_loss={running_loss:.6f} "
                f"lr={learning_rate:.8f} "
                f"speed={images_per_second:.2f}_img/s "
                f"eta={eta_seconds / 60.0:.1f}_min",
                flush=True,
            )

    if total_images == 0:
        raise RuntimeError("Local MFM epoch received no images.")
    print(
        f"[local_mfm][{phase_name}] epoch={epoch + 1} COMPLETE "
        f"images={total_images:,} loss={total_loss / total_images:.6f} "
        f"elapsed={(time.time() - epoch_started_at) / 60.0:.1f}_min",
        flush=True,
    )
    return {"loss": total_loss / total_images, "num_images": total_images}, global_update


def save_local_encoder_checkpoint(
    model: MFMReconstructionModel,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    encoder_state = {
        f"encoder.{key}": value.detach().cpu()
        for key, value in model.encoder.state_dict().items()
    }
    atomic_torch_save({"model": encoder_state, **metadata}, path)


def train_local_mfm_reconstructor(
    config: dict[str, Any],
    manifest_path: Path,
    imagenet_root: Path,
    output_root: Path,
    resume_roots: Sequence[Path],
) -> tuple[Path, list[dict[str, Any]]]:
    manifest = pd.read_csv(manifest_path)
    train_frame = manifest[manifest["split"] == "real_train"].reset_index(drop=True)
    validation_frame = manifest[
        manifest["split"] == "real_validation"
    ].reset_index(drop=True)
    validation_limit = config.get("local_mfm_validation_limit")
    if validation_limit is not None:
        validation_frame = validation_frame.iloc[: int(validation_limit)].copy()

    train_dataset = MFMRealDataset(train_frame, imagenet_root, config, training=True)
    validation_dataset = MFMRealDataset(
        validation_frame, imagenet_root, config, training=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MFMReconstructionModel().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["local_mfm_learning_rate"]),
        betas=tuple(float(value) for value in config["local_mfm_betas"]),
        weight_decay=float(config["local_mfm_weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config["amp"] and device.type == "cuda")
    )

    accumulation = int(config["local_mfm_accumulation_steps"])
    updates_per_epoch = math.ceil(
        math.floor(len(train_dataset) / int(config["local_mfm_batch_size"]))
        / accumulation
    )
    total_updates = max(updates_per_epoch * int(config["local_mfm_epochs"]), 1)
    warmup_updates = updates_per_epoch * int(config["local_mfm_warmup_epochs"])
    checkpoint_dir = output_root / "checkpoints"
    metrics_dir = output_root / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "local_mfm_last.pt"
    best_path = checkpoint_dir / "local_mfm_best.pt"
    encoder_best_path = checkpoint_dir / "local_encoder_best.pt"

    start_epoch, global_update = 0, 0
    best_validation, stale_epochs = float("inf"), 0
    history: list[dict[str, Any]] = []
    resume_checkpoint = find_existing_relative(
        Path("checkpoints/local_mfm_last.pt"), output_root, resume_roots
    )
    if bool(config["local_mfm_resume"]) and resume_checkpoint is not None:
        state = _torch_load(resume_checkpoint)
        if state.get("manifest_sha256") != sha256_file(manifest_path):
            raise RuntimeError("Local MFM resume manifest hash mismatch.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        global_update = int(state.get("global_update", 0))
        best_validation = float(state.get("best_validation_loss", float("inf")))
        stale_epochs = int(state.get("stale_epochs", 0))
        history = list(state.get("history", []))

    for epoch in range(start_epoch, int(config["local_mfm_epochs"])):
        train_metrics, global_update = run_local_mfm_epoch(
            model,
            train_dataset,
            config,
            device,
            epoch,
            optimizer,
            scaler,
            global_update,
            total_updates,
            warmup_updates,
        )
        validation_metrics, _ = run_local_mfm_epoch(
            model,
            validation_dataset,
            config,
            device,
            epoch,
            None,
            scaler,
            global_update,
            total_updates,
            warmup_updates,
        )
        improved = validation_metrics["loss"] < best_validation
        if improved:
            best_validation = validation_metrics["loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch + 1,
            "train_frequency_loss": train_metrics["loss"],
            "validation_frequency_loss": validation_metrics["loss"],
            "train_images": train_metrics["num_images"],
            "validation_images": validation_metrics["num_images"],
            "global_update": global_update,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            metrics_dir / "local_mfm_training_history.csv", index=False
        )
        metadata = {
            "schema_version": 1,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "global_update": global_update,
            "best_validation_loss": best_validation,
            "stale_epochs": stale_epochs,
            "history": history,
            "config": config,
            "manifest_sha256": sha256_file(manifest_path),
            "objective": "official_mfm_masked_frequency_reconstruction",
        }
        atomic_torch_save({"model": model.state_dict(), **metadata}, last_path)
        if improved:
            atomic_torch_save({"model": model.state_dict(), **metadata}, best_path)
            save_local_encoder_checkpoint(
                model,
                encoder_best_path,
                {
                    "schema_version": 1,
                    "epoch": epoch,
                    "best_validation_loss": best_validation,
                    "config": config,
                    "manifest_sha256": sha256_file(manifest_path),
                    "objective": "official_mfm_masked_frequency_reconstruction",
                },
            )
        print(json.dumps(row, indent=2, default=_json_default))

        patience = config.get("local_mfm_patience")
        if patience is not None and stale_epochs >= int(patience):
            print(f"Local MFM early stopping after {stale_epochs} stale epochs.")
            break

    if not encoder_best_path.is_file():
        existing = find_existing_relative(
            Path("checkpoints/local_encoder_best.pt"), output_root, resume_roots
        )
        if existing is None:
            raise FileNotFoundError("No local_encoder_best.pt was produced.")
        shutil.copy2(existing, encoder_best_path)
    for relative, destination in (
        (Path("checkpoints/local_mfm_best.pt"), best_path),
        (Path("checkpoints/local_mfm_last.pt"), last_path),
    ):
        if not destination.is_file():
            existing = find_existing_relative(relative, output_root, resume_roots)
            if existing is not None:
                shutil.copy2(existing, destination)
    return encoder_best_path, history


# ---------------------------------------------------------------------------
# Frozen local MFM encoder and exact spectral views
# ---------------------------------------------------------------------------


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    """Load checkpoints created by older PyTorch releases as well."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=map_location)


class FrozenMFMLocalEncoder(nn.Module):
    """ViT-B/16 local encoder used by the official MFM pre-training code."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="token",
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_features(images)
        if tokens.ndim == 3:
            return tokens[:, 0]
        if tokens.ndim == 2:
            return tokens
        raise RuntimeError(f"Unexpected MFM encoder output shape: {tuple(tokens.shape)}")


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "teacher"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict) and candidate:
                checkpoint = candidate
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("MFM checkpoint does not contain a state dictionary.")
    return {str(key): value for key, value in checkpoint.items() if torch.is_tensor(value)}


def load_official_mfm_encoder(
    checkpoint_path: Path,
    output_report_path: Path | None = None,
) -> tuple[FrozenMFMLocalEncoder, dict[str, Any]]:
    """Load official MFM encoder weights into timm ViT-B/16.

    Official MFM checkpoints store attention q/v biases separately while timm
    stores a single qkv bias. The conversion below is deterministic and the
    parameter-coverage assertion prevents accidental use of unrelated weights.
    """
    model = FrozenMFMLocalEncoder()
    target = model.encoder.state_dict()
    raw = _unwrap_state_dict(_torch_load(checkpoint_path))

    cleaned: dict[str, torch.Tensor] = {}
    for original_key, value in raw.items():
        key = original_key
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key.startswith("encoder."):
            key = key[len("encoder.") :]
        elif any(token in key for token in ("decoder", "criterion", "loss")):
            continue
        if key.startswith("head.") or key.startswith("fc_norm."):
            continue
        cleaned[key] = value

    # MFM/BEiT style: attn.q_bias + attn.v_bias -> timm attn.qkv.bias.
    for block_index in range(12):
        prefix = f"blocks.{block_index}.attn."
        q_key, v_key = prefix + "q_bias", prefix + "v_bias"
        qkv_key = prefix + "qkv.bias"
        if q_key in cleaned and v_key in cleaned and qkv_key not in cleaned:
            q_bias = cleaned[q_key]
            cleaned[qkv_key] = torch.cat(
                [q_bias, torch.zeros_like(q_bias), cleaned[v_key]], dim=0
            )

    compatible: dict[str, torch.Tensor] = {}
    shape_mismatches: dict[str, dict[str, list[int]]] = {}
    for key, value in cleaned.items():
        if key not in target:
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            shape_mismatches[key] = {
                "checkpoint": list(value.shape),
                "expected": list(target[key].shape),
            }
            continue
        compatible[key] = value

    total_numel = sum(value.numel() for value in target.values())
    loaded_numel = sum(target[key].numel() for key in compatible)
    coverage = loaded_numel / max(total_numel, 1)
    critical_prefixes = ("patch_embed.", "cls_token", "pos_embed", "blocks.", "norm.")
    critical_missing = [
        key
        for key in target
        if key.startswith(critical_prefixes) and key not in compatible
    ]
    if coverage < 0.95 or critical_missing:
        raise RuntimeError(
            "The supplied checkpoint is not a compatible official MFM ViT-B/16 "
            f"encoder: coverage={coverage:.2%}, critical_missing={critical_missing[:12]}, "
            f"shape_mismatches={list(shape_mismatches)[:8]}."
        )

    incompatible = model.encoder.load_state_dict(compatible, strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "target_parameter_tensors": len(target),
        "loaded_parameter_tensors": len(compatible),
        "parameter_numel_coverage": coverage,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "shape_mismatches": shape_mismatches,
        "architecture": "vit_base_patch16_224",
        "local_embedding_dim": 768,
    }
    if output_report_path is not None:
        write_json(output_report_path, report)
    return model, report


def pil_to_float_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def float_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def jpeg_recompress(
    image: torch.Tensor,
    quality: int,
    subsampling: int,
) -> torch.Tensor:
    buffer = io.BytesIO()
    float_tensor_to_pil(image).save(
        buffer,
        format="JPEG",
        quality=int(quality),
        subsampling=int(subsampling),
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        decoded.load()
        return pil_to_float_tensor(decoded)


def make_frequency_mask(
    height: int,
    width: int,
    radius: float,
    device: torch.device,
    dtype: torch.dtype,
    implementation: str = "spai",
) -> torch.Tensor:
    if implementation == "spai":
        if height % 2 or width % 2:
            raise ValueError("SPAI centered mask requires even spatial dimensions.")
        y_half = torch.arange(height // 2, device=device, dtype=dtype)
        x_half = torch.arange(width // 2, device=device, dtype=dtype)
        y = torch.cat([y_half.flip(0), y_half], dim=0)
        x = torch.cat([x_half.flip(0), x_half], dim=0)
    elif implementation == "mfm":
        y = torch.arange(height, device=device, dtype=dtype) - height // 2
        x = torch.arange(width, device=device, dtype=dtype) - width // 2
    else:
        raise ValueError(f"Unknown spectral mask implementation: {implementation}")
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return ((xx.square() + yy.square()) < float(radius) ** 2).to(dtype)


def spectral_view(
    image: torch.Tensor,
    radius: int,
    keep: str,
    implementation: str = "spai",
) -> torch.Tensor:
    """MFM frequency corruption: FFT -> centered circular mask -> inverse FFT."""
    if keep not in {"low", "high"}:
        raise ValueError(f"keep must be low/high, got {keep!r}")
    _, height, width = image.shape
    low_mask = make_frequency_mask(
        height, width, radius, image.device, image.dtype, implementation
    )
    mask = low_mask if keep == "low" else (1.0 - low_mask)
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1)
    )
    reconstructed = torch.fft.ifft2(
        torch.fft.ifftshift(spectrum * mask[None], dim=(-2, -1)),
        dim=(-2, -1),
    ).real
    return reconstructed.clamp(0.0, 1.0)


def tile_native_image(
    image: torch.Tensor,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reflect-pad only the right/bottom borders, then create non-overlap tiles."""
    _, height, width = image.shape
    padded_height = math.ceil(height / tile_size) * tile_size
    padded_width = math.ceil(width / tile_size) * tile_size
    pad_bottom, pad_right = padded_height - height, padded_width - width
    if pad_bottom or pad_right:
        mode = "reflect" if height > 1 and width > 1 else "replicate"
        # PyTorch reflect padding must be smaller than the input dimension.
        if pad_bottom >= height or pad_right >= width:
            mode = "replicate"
        image = F.pad(image, (0, pad_right, 0, pad_bottom), mode=mode)

    rows, columns = padded_height // tile_size, padded_width // tile_size
    tiles = (
        image.unfold(1, tile_size, tile_size)
        .unfold(2, tile_size, tile_size)
        .permute(1, 2, 0, 3, 4)
        .reshape(rows * columns, 3, tile_size, tile_size)
    )
    row_centers = (torch.arange(rows, dtype=torch.float32) + 0.5) / rows
    column_centers = (torch.arange(columns, dtype=torch.float32) + 0.5) / columns
    yy, xx = torch.meshgrid(row_centers, column_centers, indexing="ij")
    positions = torch.stack([yy * 2 - 1, xx * 2 - 1], dim=-1).reshape(-1, 2)
    return tiles.contiguous(), positions


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.inference_mode()
def encode_local_views(
    image_path: Path,
    sample_id: str,
    model: FrozenMFMLocalEncoder,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    with Image.open(image_path) as image:
        image.load()
        raw = pil_to_float_tensor(image)

    jpeg_rng = random.Random(stable_seed(config["seed"], sample_id, "jpeg"))
    jpeg_quality = jpeg_rng.randint(
        int(config["jpeg_quality_min"]), int(config["jpeg_quality_max"])
    )
    jpeg_subsampling = jpeg_rng.choice(list(config["jpeg_subsampling_values"]))
    jpeg = jpeg_recompress(raw, jpeg_quality, jpeg_subsampling)

    raw_tiles, positions = tile_native_image(raw, int(config["tile_size"]))
    jpeg_tiles, jpeg_positions = tile_native_image(jpeg, int(config["tile_size"]))
    if raw_tiles.shape != jpeg_tiles.shape or not torch.equal(positions, jpeg_positions):
        raise RuntimeError("Raw and JPEG views produced inconsistent native tiles.")

    ordered_views = []
    for source in (raw_tiles, jpeg_tiles):
        ordered_views.extend(
            [
                torch.stack(
                    [
                        spectral_view(
                            tile,
                            int(config["mfm_mask_radius"]),
                            "low",
                            str(config["spectral_mask_implementation"]),
                        )
                        for tile in source
                    ]
                ),
                torch.stack(
                    [
                        spectral_view(
                            tile,
                            int(config["mfm_mask_radius"]),
                            "high",
                            str(config["spectral_mask_implementation"]),
                        )
                        for tile in source
                    ]
                ),
            ]
        )

    features: list[torch.Tensor] = []
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    local_batch_size = int(config["local_feature_batch_size"])
    amp_enabled = bool(config["amp"] and device.type == "cuda")
    for view in ordered_views:
        view_features = []
        for start in range(0, len(view), local_batch_size):
            batch = view[start : start + local_batch_size].to(device, non_blocking=True)
            batch = (batch - mean) / std
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                embedding = model(batch)
            view_features.append(embedding.float().cpu())
        features.append(torch.cat(view_features, dim=0))

    # [K, raw_low/raw_high/jpeg_low/jpeg_high, 768]
    stacked = torch.stack(features, dim=1).to(torch.float16)
    if not torch.isfinite(stacked).all():
        raise FloatingPointError(f"Non-finite MFM feature detected for {image_path}")
    metadata = {
        "jpeg_quality": jpeg_quality,
        "jpeg_subsampling": jpeg_subsampling,
        "num_patches": len(positions),
    }
    return stacked, positions.to(torch.float16), metadata


def cache_shard_path(output_root: Path, split: str, shard_index: int) -> Path:
    prefix = SPLIT_TO_PREFIX[split]
    return output_root / "feature_cache" / f"{prefix}_{shard_index:05d}.pt"


def cache_local_features(
    config: dict[str, Any],
    manifest_path: Path,
    imagenet_root: Path,
    mfm_checkpoint_path: Path,
    output_root: Path,
    resume_roots: Sequence[Path],
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mfm_report = load_official_mfm_encoder(
        mfm_checkpoint_path,
        output_root / "provenance" / "mfm_checkpoint_load_report.json",
    )
    model.to(device)
    manifest = pd.read_csv(manifest_path)
    shard_size = int(config["cache_images_per_shard"])
    shard_start = config.get("cache_shard_start")
    shard_stop = config.get("cache_shard_stop")
    index_rows: list[dict[str, Any]] = []
    global_shard_index = 0

    cache_dir = output_root / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_TO_PREFIX:
        split_frame = manifest[manifest["split"] == split].reset_index(drop=True)
        for local_start in range(0, len(split_frame), shard_size):
            shard_frame = split_frame.iloc[local_start : local_start + shard_size]
            destination = cache_shard_path(
                output_root, split, local_start // shard_size
            )
            relative = destination.relative_to(output_root)
            existing = find_existing_relative(relative, output_root, resume_roots)

            selected_by_range = (
                (shard_start is None or global_shard_index >= int(shard_start))
                and (shard_stop is None or global_shard_index < int(shard_stop))
            )
            if existing is None and selected_by_range:
                all_embeddings: list[torch.Tensor] = []
                all_positions: list[torch.Tensor] = []
                offsets = [0]
                sample_ids: list[str] = []
                relative_paths: list[str] = []
                jpeg_qualities: list[int] = []
                jpeg_subsamplings: list[int] = []
                iterator = tqdm(
                    shard_frame.itertuples(index=False),
                    total=len(shard_frame),
                    desc=f"MFM cache {destination.stem}",
                )
                for row in iterator:
                    embeddings, positions, metadata = encode_local_views(
                        imagenet_root / row.relative_path,
                        row.sample_id,
                        model,
                        device,
                        config,
                    )
                    all_embeddings.append(embeddings)
                    all_positions.append(positions)
                    offsets.append(offsets[-1] + len(positions))
                    sample_ids.append(row.sample_id)
                    relative_paths.append(row.relative_path)
                    jpeg_qualities.append(metadata["jpeg_quality"])
                    jpeg_subsamplings.append(metadata["jpeg_subsampling"])

                payload = {
                    "schema_version": 1,
                    "split": split,
                    "feature_names": FEATURE_NAMES,
                    "embeddings": torch.cat(all_embeddings, dim=0),
                    "positions": torch.cat(all_positions, dim=0),
                    "offsets": torch.tensor(offsets, dtype=torch.int64),
                    "sample_ids": sample_ids,
                    "relative_paths": relative_paths,
                    "jpeg_qualities": jpeg_qualities,
                    "jpeg_subsamplings": jpeg_subsamplings,
                    "manifest_sha256": sha256_file(manifest_path),
                    "mfm_checkpoint_sha256": mfm_report["checkpoint_sha256"],
                    "spectral_radius": int(config["mfm_mask_radius"]),
                    "tile_size": int(config["tile_size"]),
                }
                atomic_torch_save(payload, destination)

            available_path = destination if destination.is_file() else existing
            row_payload = {
                "global_shard_index": global_shard_index,
                "split": split,
                "shard_index": local_start // shard_size,
                "num_images": len(shard_frame),
                "relative_path": str(relative).replace("\\", "/"),
                "complete": available_path is not None and available_path.is_file(),
            }
            if available_path is not None and available_path.is_file():
                row_payload["sha256"] = sha256_file(available_path)
                row_payload["size_bytes"] = available_path.stat().st_size
            index_rows.append(row_payload)
            global_shard_index += 1

    index = {
        "schema_version": 1,
        "manifest_sha256": sha256_file(manifest_path),
        "mfm_checkpoint_sha256": mfm_report["checkpoint_sha256"],
        "feature_names": FEATURE_NAMES,
        "shards": index_rows,
        "num_complete_shards": sum(row["complete"] for row in index_rows),
        "num_total_shards": len(index_rows),
    }
    write_json(cache_dir / "index.json", index)
    if index["num_complete_shards"] != index["num_total_shards"]:
        print(
            "Cache range completed, but the full cache is not ready: "
            f"{index['num_complete_shards']}/{index['num_total_shards']} shards."
        )
    return index


# ---------------------------------------------------------------------------
# Global masked-feature model
# ---------------------------------------------------------------------------


def positional_encoding_2d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Fixed continuous 2-D sine/cosine encoding for arbitrary tile grids."""
    if dim % 4 != 0:
        raise ValueError("global_dim must be divisible by four")
    quarter = dim // 4
    frequencies = torch.exp(
        torch.arange(quarter, device=positions.device, dtype=positions.dtype)
        * (-math.log(10000.0) / max(quarter - 1, 1))
    )
    y = positions[..., 0:1] * frequencies
    x = positions[..., 1:2] * frequencies
    return torch.cat([torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)], dim=-1)


class GlobalMaskedImageEncoder(nn.Module):
    """Variable-resolution global encoder with a single Image-CLS bottleneck."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        dim = int(config["global_dim"])
        heads = int(config["global_heads"])
        feedforward_dim = int(dim * float(config["global_mlp_ratio"]))
        dropout = float(config["global_dropout"])
        self.dim = dim
        self.input_norm = nn.LayerNorm(dim)
        self.image_cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.decoder_query = nn.Parameter(torch.zeros(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(config["global_depth"]),
            norm=nn.LayerNorm(dim),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=int(config["global_decoder_depth"]),
            norm=nn.LayerNorm(dim),
        )
        self.prediction_head = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.image_cls, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_query, std=0.02)

    def encode_image(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
        masked_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.input_norm(features)
        if masked_tokens is not None:
            features = torch.where(
                masked_tokens.unsqueeze(-1),
                self.mask_token.expand_as(features),
                features,
            )
        features = features + positional_encoding_2d(positions, self.dim)
        cls = self.image_cls.expand(features.shape[0], -1, -1)
        sequence = torch.cat([cls, features], dim=1)
        cls_padding = torch.zeros(
            (padding_mask.shape[0], 1), dtype=torch.bool, device=padding_mask.device
        )
        encoded = self.encoder(
            sequence,
            src_key_padding_mask=torch.cat([cls_padding, padding_mask], dim=1),
        )
        return encoded[:, 0]

    def reconstruct_tokens(
        self,
        image_cls: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.decoder_query.expand(
            positions.shape[0], positions.shape[1], -1
        ) + positional_encoding_2d(positions, self.dim)
        decoded = self.decoder(
            tgt=queries,
            memory=image_cls[:, None, :],
            tgt_key_padding_mask=padding_mask,
        )
        return self.prediction_head(decoded)

    def forward_training(
        self,
        student_features: torch.Tensor,
        target_features: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
        masked_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        augmented_cls = self.encode_image(
            student_features, positions, padding_mask, masked_tokens
        )
        with torch.no_grad():
            reference_cls = self.encode_image(
                target_features, positions, padding_mask, masked_tokens=None
            )
        predictions = self.reconstruct_tokens(augmented_cls, positions, padding_mask)
        return predictions, augmented_cls, reference_cls


def make_masked_tokens(
    padding_mask: torch.Tensor,
    ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    random_values = torch.rand(
        padding_mask.shape,
        device=padding_mask.device,
        generator=generator,
    )
    mask = (random_values < ratio) & ~padding_mask
    valid_counts = (~padding_mask).sum(dim=1)
    for row in range(mask.shape[0]):
        valid = torch.nonzero(~padding_mask[row], as_tuple=False).flatten()
        if len(valid) and mask[row].sum() == 0:
            mask[row, valid[0]] = True
        if len(valid) > 1 and mask[row].sum() >= valid_counts[row]:
            mask[row, valid[-1]] = False
    return mask


def per_image_masked_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masked_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    predictions_norm = F.layer_norm(predictions.float(), (predictions.shape[-1],))
    targets_norm = F.layer_norm(targets.float(), (targets.shape[-1],))
    cosine = 1.0 - F.cosine_similarity(predictions_norm, targets_norm, dim=-1)
    mse = (predictions_norm - targets_norm).square().mean(dim=-1)
    counts = masked_tokens.sum(dim=1).clamp_min(1)
    cosine_per_image = (cosine * masked_tokens).sum(dim=1) / counts
    mse_per_image = (mse * masked_tokens).sum(dim=1) / counts
    cosine_loss = cosine_per_image.mean()
    mse_loss = mse_per_image.mean()
    return cosine_loss, mse_loss, {
        "cosine": float(cosine_loss.detach().cpu()),
        "mse": float(mse_loss.detach().cpu()),
    }


def list_cache_shards(
    output_root: Path,
    split: str,
    resume_roots: Sequence[Path] = (),
) -> list[Path]:
    index_path = find_existing_relative(
        Path("feature_cache/index.json"), output_root, resume_roots
    )
    if index_path is None:
        raise FileNotFoundError(f"Missing feature-cache index: {index_path}")
    index = read_json(index_path)
    rows = [row for row in index["shards"] if row["split"] == split]
    missing = [row["relative_path"] for row in rows if not row.get("complete")]
    if missing:
        raise RuntimeError(
            f"Feature cache for {split} is incomplete ({len(missing)} missing); "
            "finish the cache phase before training."
        )
    paths: list[Path] = []
    for row in rows:
        path = find_existing_relative(row["relative_path"], output_root, resume_roots)
        if path is None:
            raise FileNotFoundError(f"Missing cache shard: {row['relative_path']}")
        paths.append(path)
    return paths


def _sample_from_cache_payload(
    payload: dict[str, Any],
    index: int,
    training: bool,
    rng: random.Random,
    jpeg_probability: float,
) -> dict[str, Any]:
    start = int(payload["offsets"][index])
    stop = int(payload["offsets"][index + 1])
    features = payload["embeddings"][start:stop].float()
    positions = payload["positions"][start:stop].float()
    target = (features[:, 0] + features[:, 1]) * 0.5
    if training:
        source_offset = 2 if rng.random() < jpeg_probability else 0
        frequency_offset = rng.randrange(2)
        student = features[:, source_offset + frequency_offset]
    else:
        # The validation corruption is deterministic and deliberately harder:
        # JPEG low/high average must reconstruct the raw low/high target.
        student = (features[:, 2] + features[:, 3]) * 0.5
    return {
        "student": student,
        "target": target,
        "positions": positions,
        "sample_id": payload["sample_ids"][index],
    }


def collate_variable_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(samples)
    max_tokens = max(len(sample["positions"]) for sample in samples)
    dim = samples[0]["student"].shape[-1]
    student = torch.zeros(batch_size, max_tokens, dim, dtype=torch.float32)
    target = torch.zeros_like(student)
    positions = torch.zeros(batch_size, max_tokens, 2, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, max_tokens, dtype=torch.bool)
    for row, sample in enumerate(samples):
        length = len(sample["positions"])
        student[row, :length] = sample["student"]
        target[row, :length] = sample["target"]
        positions[row, :length] = sample["positions"]
        padding_mask[row, :length] = False
    return {
        "student": student,
        "target": target,
        "positions": positions,
        "padding_mask": padding_mask,
        "sample_ids": [sample["sample_id"] for sample in samples],
    }


def iter_cache_batches(
    shard_paths: Sequence[Path],
    config: dict[str, Any],
    training: bool,
    epoch: int,
) -> Iterator[dict[str, Any]]:
    rng = random.Random(stable_seed(config["seed"], "batches", epoch, training))
    paths = list(shard_paths)
    if training:
        rng.shuffle(paths)
    max_batch = int(config["global_batch_size"])
    token_budget = int(config["global_max_padded_tokens_per_batch"])

    for path in paths:
        payload = _torch_load(path)
        order = list(range(len(payload["sample_ids"])))
        if training:
            rng.shuffle(order)
        pending: list[dict[str, Any]] = []
        pending_max_tokens = 0
        for index in order:
            sample = _sample_from_cache_payload(
                payload,
                index,
                training,
                rng,
                float(config["jpeg_input_probability"]),
            )
            proposed_max = max(pending_max_tokens, len(sample["positions"]))
            proposed_batch = len(pending) + 1
            exceeds = (
                pending
                and (
                    proposed_batch > max_batch
                    or proposed_max * proposed_batch > token_budget
                )
            )
            if exceeds:
                yield collate_variable_samples(pending)
                pending = []
                pending_max_tokens = 0
            pending.append(sample)
            pending_max_tokens = max(pending_max_tokens, len(sample["positions"]))
        if pending:
            yield collate_variable_samples(pending)
        del payload


def run_global_epoch(
    model: GlobalMaskedImageEncoder,
    shard_paths: Sequence[Path],
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
    optimizer: AdamW | None,
    scaler: torch.cuda.amp.GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = Counter()
    progress = tqdm(
        iter_cache_batches(shard_paths, config, training=training, epoch=epoch),
        desc=f"global {'train' if training else 'validation'} epoch {epoch + 1}",
    )
    grad_context = torch.enable_grad if training else torch.no_grad
    for batch in progress:
        student = batch["student"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        masked_tokens = make_masked_tokens(
            padding_mask, float(config["global_mask_ratio"])
        )
        amp_enabled = bool(config["amp"] and device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with grad_context():
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predictions, augmented_cls, reference_cls = model.forward_training(
                    student, target, positions, padding_mask, masked_tokens
                )
                cosine_loss, mse_loss, components = per_image_masked_loss(
                    predictions, target, masked_tokens
                )
                consistency = (
                    1.0
                    - F.cosine_similarity(
                        augmented_cls.float(), reference_cls.float(), dim=-1
                    )
                ).mean()
                loss = (
                    cosine_loss
                    + float(config["global_mse_weight"]) * mse_loss
                    + float(config["global_consistency_weight"]) * consistency
                )
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config["global_grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

        count = len(batch["sample_ids"])
        totals["images"] += count
        totals["batches"] += 1
        totals["loss"] += float(loss.detach().cpu()) * count
        totals["cosine"] += components["cosine"] * count
        totals["mse"] += components["mse"] * count
        totals["consistency"] += float(consistency.detach().cpu()) * count
        progress.set_postfix(loss=f"{totals['loss'] / totals['images']:.4f}")

    if totals["images"] == 0:
        raise RuntimeError("No samples were available for the global epoch.")
    return {
        key: totals[key] / totals["images"]
        for key in ("loss", "cosine", "mse", "consistency")
    } | {"num_images": int(totals["images"]), "num_batches": int(totals["batches"])}


def train_global_encoder(
    config: dict[str, Any],
    output_root: Path,
    manifest_sha256: str,
    resume_roots: Sequence[Path],
) -> tuple[Path, list[dict[str, Any]]]:
    train_shards = list_cache_shards(output_root, "real_train", resume_roots)
    validation_shards = list_cache_shards(output_root, "real_validation", resume_roots)
    cache_index_path = find_existing_relative(
        Path("feature_cache/index.json"), output_root, resume_roots
    )
    assert cache_index_path is not None
    cache_index_sha = sha256_file(cache_index_path)
    cache_metadata = read_json(cache_index_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlobalMaskedImageEncoder(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["global_learning_rate"]),
        weight_decay=float(config["global_weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(config["global_epochs"]), 1),
        eta_min=float(config["global_min_learning_rate"]),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config["amp"] and device.type == "cuda")
    )
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "global_encoder_last.pt"
    best_path = checkpoint_dir / "global_encoder_best.pt"
    history_path = output_root / "metrics" / "global_training_history.csv"

    start_epoch, best_validation, stale_epochs = 0, float("inf"), 0
    history: list[dict[str, Any]] = []
    resume_checkpoint = find_existing_relative(
        Path("checkpoints/global_encoder_last.pt"), output_root, resume_roots
    )
    if bool(config["resume_training"]) and resume_checkpoint is not None:
        state = _torch_load(resume_checkpoint)
        if state.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError("Resume checkpoint manifest hash does not match this run.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
        if "numpy_rng_state" in state:
            np.random.set_state(state["numpy_rng_state"])
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        start_epoch = int(state["epoch"]) + 1
        best_validation = float(state.get("best_validation_loss", float("inf")))
        stale_epochs = int(state.get("stale_epochs", 0))
        history = list(state.get("history", []))

    for epoch in range(start_epoch, int(config["global_epochs"])):
        train_metrics = run_global_epoch(
            model, train_shards, config, device, epoch, optimizer, scaler
        )
        validation_metrics = run_global_epoch(
            model, validation_shards, config, device, epoch, None, scaler
        )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)

        improved = validation_metrics["loss"] < best_validation
        if improved:
            best_validation = validation_metrics["loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint = {
            "schema_version": 1,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_validation_loss": best_validation,
            "stale_epochs": stale_epochs,
            "history": history,
            "config": config,
            "manifest_sha256": manifest_sha256,
            "cache_index_sha256": cache_index_sha,
            "mfm_checkpoint_sha256": cache_metadata["mfm_checkpoint_sha256"],
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        atomic_torch_save(checkpoint, last_path)
        if improved:
            atomic_torch_save(checkpoint, best_path)
        print(json.dumps(row, indent=2, default=_json_default))
        if stale_epochs >= int(config["global_patience"]):
            print(f"Early stopping after {stale_epochs} non-improving epochs.")
            break

    if not best_path.is_file():
        existing_best = find_existing_relative(
            Path("checkpoints/global_encoder_best.pt"), output_root, resume_roots
        )
        if existing_best is None:
            raise FileNotFoundError("No best global-encoder checkpoint was produced.")
        shutil.copy2(existing_best, best_path)
    return best_path, history


# ---------------------------------------------------------------------------
# Image representations, real-only GMM, and calibration
# ---------------------------------------------------------------------------


@torch.inference_mode()
def extract_image_representations(
    model: GlobalMaskedImageEncoder,
    shard_paths: Sequence[Path],
    config: dict[str, Any],
    device: torch.device,
    split: str,
    output_root: Path,
) -> tuple[np.ndarray, list[str]]:
    model.eval()
    representations: list[np.ndarray] = []
    sample_ids: list[str] = []
    batches = iter_cache_batches(shard_paths, config, training=False, epoch=0)
    for batch in tqdm(batches, desc=f"Image-CLS {split}"):
        target = batch["target"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        amp_enabled = bool(config["amp"] and device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            image_cls = model.encode_image(target, positions, padding_mask)
        representations.append(image_cls.float().cpu().numpy())
        sample_ids.extend(batch["sample_ids"])

    if not representations:
        raise RuntimeError(f"No representations extracted for {split}.")
    matrix = np.concatenate(representations, axis=0).astype(np.float32, copy=False)
    embedding_dir = output_root / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    np.save(embedding_dir / f"{split}_image_cls.npy", matrix)
    pd.DataFrame(
        {"row_index": np.arange(len(sample_ids)), "sample_id": sample_ids}
    ).to_csv(embedding_dir / f"{split}_sample_ids.csv", index=False)
    return matrix, sample_ids


def fit_real_only_gmm(
    config: dict[str, Any],
    output_root: Path,
    global_checkpoint_path: Path,
    resume_roots: Sequence[Path],
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _torch_load(global_checkpoint_path)
    model = GlobalMaskedImageEncoder(checkpoint.get("config", config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    matrices: dict[str, np.ndarray] = {}
    ids: dict[str, list[str]] = {}
    for split in SPLIT_TO_PREFIX:
        shards = list_cache_shards(output_root, split, resume_roots)
        matrices[split], ids[split] = extract_image_representations(
            model, shards, config, device, split, output_root
        )

    train = matrices["real_train"]
    validation = matrices["real_validation"]
    calibration = matrices["real_calibration"]
    scaler = StandardScaler(copy=True)
    train_scaled = scaler.fit_transform(train).astype(np.float32, copy=False)
    validation_scaled = scaler.transform(validation).astype(np.float32, copy=False)
    calibration_scaled = scaler.transform(calibration).astype(np.float32, copy=False)

    candidate_rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, GaussianMixture, dict[str, Any]]] = []
    for components in list(config["gmm_components"]):
        components = int(components)
        print(f"Fitting real-only diagonal GMM with K={components} ...")
        gmm = GaussianMixture(
            n_components=components,
            covariance_type=str(config["gmm_covariance_type"]),
            reg_covar=float(config["gmm_reg_covar"]),
            max_iter=int(config["gmm_max_iter"]),
            n_init=int(config["gmm_n_init"]),
            random_state=int(config["seed"]),
            verbose=1,
        )
        gmm.fit(train_scaled)
        responsibilities = gmm.predict_proba(train_scaled)
        occupancy = responsibilities.mean(axis=0)
        validation_log_likelihood = float(gmm.score(validation_scaled))
        row = {
            "n_components": components,
            "converged": bool(gmm.converged_),
            "n_iter": int(gmm.n_iter_),
            "train_log_likelihood": float(gmm.score(train_scaled)),
            "validation_log_likelihood": validation_log_likelihood,
            "train_bic": float(gmm.bic(train_scaled)),
            "min_soft_occupancy": float(occupancy.min()),
            "max_soft_occupancy": float(occupancy.max()),
            "valid_occupancy": bool(
                occupancy.min() >= float(config["gmm_min_component_occupancy"])
            ),
        }
        candidate_rows.append(row)
        if row["converged"] and row["valid_occupancy"]:
            candidates.append((validation_log_likelihood, gmm, row))

    if not candidates:
        warnings.warn(
            "No GMM candidate passed convergence/occupancy checks; selecting the "
            "highest validation likelihood among all fitted candidates."
        )
        # Refit the best K according to the recorded score to avoid retaining all models.
        best_row = max(candidate_rows, key=lambda row: row["validation_log_likelihood"])
        selected_gmm = GaussianMixture(
            n_components=int(best_row["n_components"]),
            covariance_type=str(config["gmm_covariance_type"]),
            reg_covar=float(config["gmm_reg_covar"]),
            max_iter=int(config["gmm_max_iter"]),
            n_init=int(config["gmm_n_init"]),
            random_state=int(config["seed"]),
        ).fit(train_scaled)
    else:
        _, selected_gmm, best_row = max(candidates, key=lambda item: item[0])

    model_dir = output_root / "gmm"
    metrics_dir = output_root / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "real_feature_scaler.joblib")
    joblib.dump(selected_gmm, model_dir / "real_distribution_gmm.joblib")
    pd.DataFrame(candidate_rows).to_csv(metrics_dir / "gmm_model_selection.csv", index=False)

    calibration_nll = -selected_gmm.score_samples(calibration_scaled)
    validation_nll = -selected_gmm.score_samples(validation_scaled)
    thresholds = {
        "nll_real_calibration_q90": float(np.quantile(calibration_nll, 0.90)),
        "nll_real_calibration_q95": float(np.quantile(calibration_nll, 0.95)),
        "nll_real_calibration_q99": float(np.quantile(calibration_nll, 0.99)),
        "primary_threshold": "nll_real_calibration_q95",
        "interpretation": "Predict anomaly/fake when NLL is above the threshold.",
    }
    write_json(model_dir / "real_only_thresholds.json", thresholds)

    component_std_scaled = np.sqrt(selected_gmm.covariances_)
    component_means_original = scaler.inverse_transform(selected_gmm.means_)
    component_std_original = component_std_scaled * scaler.scale_[None, :]
    np.savez_compressed(
        model_dir / "gmm_component_statistics.npz",
        weights=selected_gmm.weights_,
        means_standardized=selected_gmm.means_,
        std_standardized=component_std_scaled,
        means_original=component_means_original,
        std_original=component_std_original,
    )
    component_rows = []
    assignments = selected_gmm.predict(train_scaled)
    for component in range(selected_gmm.n_components):
        component_rows.append(
            {
                "component": component,
                "mixture_weight": float(selected_gmm.weights_[component]),
                "hard_assignment_count": int((assignments == component).sum()),
                "hard_assignment_fraction": float((assignments == component).mean()),
            }
        )
    pd.DataFrame(component_rows).to_csv(
        metrics_dir / "gmm_component_occupancy.csv", index=False
    )

    summary = {
        "selected_n_components": int(selected_gmm.n_components),
        "selected_validation_log_likelihood": float(selected_gmm.score(validation_scaled)),
        "train_num_images": len(train),
        "validation_num_images": len(validation),
        "calibration_num_images": len(calibration),
        "embedding_dim": int(train.shape[1]),
        "validation_nll_mean": float(validation_nll.mean()),
        "validation_nll_std": float(validation_nll.std()),
        "calibration_nll_mean": float(calibration_nll.mean()),
        "calibration_nll_std": float(calibration_nll.std()),
        "thresholds": thresholds,
        "global_checkpoint_sha256": sha256_file(global_checkpoint_path),
        "selection_rule": (
            "Maximum mean validation log-likelihood among converged candidates "
            "whose minimum soft component occupancy is above the configured floor."
        ),
        "gmm_training_data": "real_train only",
        "threshold_data": "real_calibration only",
    }
    write_json(metrics_dir / "gmm_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------


def run_pipeline(user_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if user_config is not None:
        config.update(user_config)
    seed_everything(int(config["seed"]))
    phases = [str(phase).lower() for phase in config["run_phases"]]
    unknown = sorted(
        set(phases) - {"local_mfm", "cache", "train", "gmm"}
    )
    if unknown:
        raise ValueError(f"Unknown run phases: {unknown}")

    run_name = str(config["run_name"])
    output_root = Path(config["output_parent"]) / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "config.json", config)
    resume_roots = discover_resume_roots(config, run_name)
    imagenet_root = find_imagenet_train_root(config.get("imagenet_train_root"))
    update_status(
        output_root,
        phase="initialization",
        state="running",
        output_directory=str(output_root),
        resume_roots=list(map(str, resume_roots)),
    )

    try:
        update_status(output_root, phase="manifest", state="running")
        manifest_path, manifest_summary = ensure_manifest(
            config, imagenet_root, output_root, resume_roots
        )
        update_status(
            output_root,
            phase="manifest",
            state="complete",
            manifest_sha256=manifest_summary["manifest_sha256"],
        )

        mfm_source = str(config["mfm_source"])
        if mfm_source not in {"official_checkpoint", "train_from_scratch"}:
            raise ValueError(
                "mfm_source must be official_checkpoint or train_from_scratch"
            )

        local_encoder_checkpoint: Path | None = None
        local_mfm_history: list[dict[str, Any]] = []
        if "local_mfm" in phases:
            if mfm_source != "train_from_scratch":
                raise ValueError(
                    "The local_mfm phase is only valid with mfm_source=train_from_scratch."
                )
            update_status(output_root, phase="local_mfm", state="running")
            local_encoder_checkpoint, local_mfm_history = train_local_mfm_reconstructor(
                config,
                manifest_path,
                imagenet_root,
                output_root,
                resume_roots,
            )
            update_status(
                output_root,
                phase="local_mfm",
                state="complete",
                local_encoder_checkpoint=str(local_encoder_checkpoint),
                local_mfm_completed_epochs=len(local_mfm_history),
            )

        cache_index = None
        if "cache" in phases:
            update_status(output_root, phase="cache", state="running")
            if mfm_source == "train_from_scratch":
                if local_encoder_checkpoint is None:
                    local_encoder_checkpoint = find_existing_relative(
                        Path("checkpoints/local_encoder_best.pt"),
                        output_root,
                        resume_roots,
                    )
                if local_encoder_checkpoint is None:
                    raise FileNotFoundError(
                        "Cache phase requires checkpoints/local_encoder_best.pt. "
                        "Run local_mfm first or attach its previous output."
                    )
                mfm_checkpoint = local_encoder_checkpoint
            else:
                mfm_checkpoint = find_mfm_checkpoint(
                    config.get("mfm_checkpoint_path"), imagenet_root
                )
            cache_index = cache_local_features(
                config,
                manifest_path,
                imagenet_root,
                mfm_checkpoint,
                output_root,
                resume_roots,
            )
            update_status(
                output_root,
                phase="cache",
                state=(
                    "complete"
                    if cache_index["num_complete_shards"]
                    == cache_index["num_total_shards"]
                    else "partial"
                ),
                cache_complete_shards=cache_index["num_complete_shards"],
                cache_total_shards=cache_index["num_total_shards"],
            )

        best_checkpoint: Path | None = None
        history: list[dict[str, Any]] = []
        if "train" in phases:
            update_status(output_root, phase="train", state="running")
            best_checkpoint, history = train_global_encoder(
                config,
                output_root,
                manifest_summary["manifest_sha256"],
                resume_roots,
            )
            update_status(
                output_root,
                phase="train",
                state="complete",
                best_global_checkpoint=str(best_checkpoint),
                completed_epochs=len(history),
            )

        gmm_summary = None
        if "gmm" in phases:
            update_status(output_root, phase="gmm", state="running")
            if best_checkpoint is None:
                best_checkpoint = find_existing_relative(
                    Path("checkpoints/global_encoder_best.pt"),
                    output_root,
                    resume_roots,
                )
            if best_checkpoint is None:
                raise FileNotFoundError(
                    "GMM phase requires checkpoints/global_encoder_best.pt."
                )
            gmm_summary = fit_real_only_gmm(
                config, output_root, best_checkpoint, resume_roots
            )
            update_status(output_root, phase="gmm", state="complete")

        result = {
            "output_root": str(output_root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_summary["manifest_sha256"],
            "cache": cache_index,
            "mfm_source": mfm_source,
            "local_encoder_checkpoint": (
                str(local_encoder_checkpoint) if local_encoder_checkpoint else None
            ),
            "local_mfm_completed_epochs": len(local_mfm_history),
            "best_global_checkpoint": str(best_checkpoint) if best_checkpoint else None,
            "gmm": gmm_summary,
        }
        write_json(output_root / "run_summary.json", result)
        update_status(output_root, phase="pipeline", state="complete")
        print(f"Pipeline completed. Output: {output_root}")
        return result
    except Exception as exc:
        update_status(
            output_root,
            phase="pipeline",
            state="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


if __name__ == "__main__":
    run_pipeline()
