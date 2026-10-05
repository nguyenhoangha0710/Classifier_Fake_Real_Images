"""Test-only evaluation of the trained Spectral-GMM model on Tiny-GenImage.

The file is intentionally standalone so it can be embedded in a Kaggle
notebook.  It restores the trained ``spectral_gmm_mfm_scratch`` model and
evaluates it with a controlled codec-normalization policy:

decode -> deterministic JPEG Q70--100 -> native 224 tiles -> SPAI low/high views -> frozen Local ViT CLS
-> mean(low, high) -> frozen Global Image-CLS -> StandardScaler -> real-only GMM.

The model is run once on Tiny-GenImage.  The cached NLL scores are then tested
against a fixed, explicitly reported threshold grid without rerunning either
encoder.
"""

from __future__ import annotations

import copy
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
from typing import Any, Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    roc_curve,
    precision_recall_curve,
)
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


ImageFile.LOAD_TRUNCATED_IMAGES = False

DEFAULT_CONFIG: dict[str, Any] = {
    "input_root": "/kaggle/input",
    "checkpoint_dataset_root": None,
    "tiny_dataset_root": None,
    "output_root": "/kaggle/working/spectral_gmm_tinygenimage_fixed_grid_test",
    "seed": 42,
    # Tiny-GenImage uses validation as its official held-out test cohort in
    # several mirrors.  Prefer a literal test directory when one exists.
    "test_split_aliases": ["test", "validation", "val", "valid"],
    "real_label_dirs": ["nature", "real", "0_real", "0-real", "0"],
    "fake_label_dirs": ["ai", "fake", "1_fake", "1-fake", "1"],
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
    # Codec normalization is applied to the whole decoded image before native
    # tiling.  The default recompresses every format, including an existing
    # JPEG, so real/fake receive the same controlled operation.  Other allowed
    # policies are ``non_jpeg_to_deterministic`` and ``as_is``.
    "jpeg_policy": "all_to_deterministic_jpeg",
    "jpeg_quality_min": 70,
    "jpeg_quality_max": 100,
    "jpeg_subsampling_values": [0, 1, 2],
    # The previous Tiny-GenImage comparisons used a balanced cohort inside
    # every generator.  None means use every available pair.
    "balance_per_generator": True,
    "max_per_class_per_generator": None,
    # Fixed test-time operating points requested for the NLL score.  Inference
    # is executed once; only the scalar comparison is repeated.
    "threshold_grid": [
        -334.892, -241.577, -149.749, -57.235, 27.958,
        133.346, 335.641, 614.874, 924.637, 1311.118,
    ],
    # Runtime.  Tiles are buffered across images before Local-ViT inference.
    "max_images_per_batch": 16,
    "max_tiles_per_batch": 128,
    "local_view_batch_size": 64,
    "amp": True,
    "save_every_images": 100,
    "resume": True,
    # Locked provenance from the uploaded completed run.
    "expected_global_checkpoint_sha256": (
        "2beb1df9020830f463a7b6ec942a5aedd87bcd98cc9117c991e6930719a20a23"
    ),
}

REQUIRED_ARTIFACTS = (
    "config.json",
    "checkpoints/local_encoder_best.pt",
    "checkpoints/global_encoder_best.pt",
    "gmm/real_feature_scaler.joblib",
    "gmm/real_distribution_gmm.joblib",
    "gmm/real_only_thresholds.json",
    "metrics/gmm_summary.json",
)

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

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Files, seeds and artifact discovery
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def stable_seed(seed: int, *parts: Any) -> int:
    text = "|".join([str(seed), *map(str, parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**31)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def is_artifact_root(path: Path) -> bool:
    return all((path / relative).is_file() for relative in REQUIRED_ARTIFACTS)


def discover_artifact_root(config: dict[str, Any]) -> Path:
    explicit = config.get("checkpoint_dataset_root")
    if explicit:
        root = Path(explicit)
        if not is_artifact_root(root):
            missing = [item for item in REQUIRED_ARTIFACTS if not (root / item).is_file()]
            raise FileNotFoundError(f"Checkpoint root is incomplete: {root}; missing={missing}")
        return root.resolve()

    input_root = Path(config["input_root"])
    candidates: list[Path] = []
    for config_path in input_root.rglob("config.json"):
        parent = config_path.parent
        if is_artifact_root(parent):
            candidates.append(parent.resolve())
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one complete Spectral-GMM artifact root under "
            f"{input_root}, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def audit_artifacts(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    training_config = read_json(root / "config.json")
    gmm_summary = read_json(root / "metrics/gmm_summary.json")
    thresholds = read_json(root / "gmm/real_only_thresholds.json")
    global_sha = sha256_file(root / "checkpoints/global_encoder_best.pt")
    expected = str(config["expected_global_checkpoint_sha256"])
    if global_sha != expected:
        raise RuntimeError(f"Global checkpoint SHA mismatch: expected={expected}, actual={global_sha}")
    if gmm_summary.get("global_checkpoint_sha256") != global_sha:
        raise RuntimeError("GMM summary and global checkpoint do not have the same SHA-256.")
    if training_config.get("tile_size") != 224:
        raise RuntimeError(f"Unexpected training tile size: {training_config.get('tile_size')}")
    if training_config.get("spectral_mask_implementation") != "spai":
        raise RuntimeError("This evaluator is locked to the trained SPAI spectral mask.")
    return {
        "artifact_root": str(root),
        "global_checkpoint_sha256": global_sha,
        "local_checkpoint_sha256": sha256_file(root / "checkpoints/local_encoder_best.pt"),
        "selected_gmm_components": int(gmm_summary["selected_n_components"]),
        "embedding_dim": int(gmm_summary["embedding_dim"]),
        "thresholds": thresholds,
        "training_config": training_config,
    }


# ---------------------------------------------------------------------------
# Tiny-GenImage exact folder cohort
# ---------------------------------------------------------------------------


def find_split_dir(generator_dir: Path, aliases: Sequence[str]) -> Path | None:
    children = {child.name.lower(): child for child in generator_dir.iterdir() if child.is_dir()}
    for alias in aliases:
        candidate = children.get(str(alias).lower())
        if candidate is not None and any(
            child.is_dir() and child.name.lower() in LABEL_DIR_TO_ID
            for child in candidate.iterdir()
        ):
            return candidate
    return None


def generator_dirs_at(root: Path, aliases: Sequence[str]) -> list[Path]:
    try:
        children = [child for child in root.iterdir() if child.is_dir()]
    except OSError:
        return []
    return sorted(
        [child for child in children if find_split_dir(child, aliases) is not None],
        key=lambda path: path.name.lower(),
    )


def discover_tiny_root(config: dict[str, Any]) -> tuple[Path, list[Path]]:
    aliases = list(config["test_split_aliases"])
    explicit = config.get("tiny_dataset_root")
    if explicit:
        root = Path(explicit)
        generators = generator_dirs_at(root, aliases)
        if not generators:
            raise FileNotFoundError(f"No Tiny-GenImage generators under explicit root {root}")
        return root.resolve(), generators

    input_root = Path(config["input_root"])
    best_root: Path | None = None
    best_generators: list[Path] = []
    # Walk directories without opening any images.  Stop descending once a
    # parent containing generator folders is found.
    for current, dirnames, _ in os.walk(input_root):
        root = Path(current)
        generators = generator_dirs_at(root, aliases)
        if len(generators) > len(best_generators):
            best_root, best_generators = root, generators
        if generators:
            dirnames[:] = []
    if best_root is None or not best_generators:
        raise FileNotFoundError(
            "Cannot locate Tiny-GenImage. Attach yangsangtai/tiny-genimage and "
            "expect generator/{test|validation}/{nature|ai}."
        )
    return best_root.resolve(), best_generators


def iter_image_files(path: Path, extensions: set[str]) -> Iterable[Path]:
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in extensions:
            yield item


def deterministic_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, replace=False, random_state=seed)


def build_tiny_test_manifest(config: dict[str, Any], output_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    tiny_root, generators = discover_tiny_root(config)
    aliases = list(config["test_split_aliases"])
    extensions = {str(value).lower() for value in config["image_extensions"]}
    records: list[dict[str, Any]] = []
    actual_splits: dict[str, str] = {}
    for generator_dir in generators:
        split_dir = find_split_dir(generator_dir, aliases)
        assert split_dir is not None
        actual_splits[generator_dir.name] = split_dir.name
        for label_dir in sorted((item for item in split_dir.iterdir() if item.is_dir())):
            label = LABEL_DIR_TO_ID.get(label_dir.name.lower())
            if label is None:
                continue
            for image_path in iter_image_files(label_dir, extensions):
                relative = image_path.relative_to(tiny_root).as_posix()
                records.append(
                    {
                        "sample_id": relative,
                        "image_path": str(image_path),
                        "relative_path": relative,
                        "generator": generator_dir.name,
                        "actual_split": split_dir.name,
                        "label": int(label),
                        "label_name": "fake" if label == 1 else "real",
                        "extension": image_path.suffix.lower(),
                    }
                )
    full = pd.DataFrame(records)
    if full.empty:
        raise RuntimeError(f"No test images found under {tiny_root}")
    if full["sample_id"].duplicated().any():
        raise RuntimeError("Tiny-GenImage manifest has duplicate sample_id values.")

    selected_parts: list[pd.DataFrame] = []
    for generator, group in full.groupby("generator", sort=True):
        real = group[group.label == 0]
        fake = group[group.label == 1]
        if len(real) == 0 or len(fake) == 0:
            raise RuntimeError(f"Generator {generator} does not contain both real and fake images.")
        if config["balance_per_generator"]:
            per_class = min(len(real), len(fake))
            limit = config.get("max_per_class_per_generator")
            if limit is not None:
                per_class = min(per_class, int(limit))
            real = deterministic_sample(real, per_class, stable_seed(config["seed"], generator, "real"))
            fake = deterministic_sample(fake, per_class, stable_seed(config["seed"], generator, "fake"))
        selected_parts.extend([real, fake])
    selected = pd.concat(selected_parts, ignore_index=True).sort_values(
        ["generator", "label", "relative_path"], ignore_index=True
    )
    selected["evaluation_cohort"] = "test_grid"
    manifest_dir = output_root / "dataset"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(manifest_dir / "tiny_test_manifest_full.csv", index=False)
    selected.to_csv(manifest_dir / "tiny_test_manifest_used.csv", index=False)
    counts = (
        selected.groupby(["generator", "label_name"])
        .size().unstack(fill_value=0).reset_index()
    )
    counts.to_csv(manifest_dir / "tiny_test_counts_by_generator.csv", index=False)
    summary = {
        "tiny_dataset_root": str(tiny_root),
        "actual_split_by_generator": actual_splits,
        "num_generators": int(selected.generator.nunique()),
        "num_samples": int(len(selected)),
        "num_real": int((selected.label == 0).sum()),
        "num_fake": int((selected.label == 1).sum()),
        "evaluation_cohort": "test_grid",
        "balance_per_generator": bool(config["balance_per_generator"]),
        "jpeg_policy": str(config["jpeg_policy"]),
        "jpeg_quality_range": [int(config["jpeg_quality_min"]), int(config["jpeg_quality_max"])],
        "manifest_sha256": dataframe_sha256(selected),
    }
    write_json(manifest_dir / "tiny_test_manifest_summary.json", summary)
    return selected, summary


# ---------------------------------------------------------------------------
# Exact trained model architecture and preprocessing
# ---------------------------------------------------------------------------


class FrozenMFMLocalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="token"
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_features(images)
        if tokens.ndim == 3:
            return tokens[:, 0]
        if tokens.ndim == 2:
            return tokens
        raise RuntimeError(f"Unexpected Local ViT output shape: {tuple(tokens.shape)}")


def positional_encoding_2d(positions: torch.Tensor, dim: int) -> torch.Tensor:
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
            d_model=dim, nhead=heads, dim_feedforward=feedforward_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(config["global_depth"]),
            norm=nn.LayerNorm(dim), enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=feedforward_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=int(config["global_decoder_depth"]),
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
                masked_tokens.unsqueeze(-1), self.mask_token.expand_as(features), features
            )
        features = features + positional_encoding_2d(positions, self.dim)
        cls = self.image_cls.expand(features.shape[0], -1, -1)
        sequence = torch.cat([cls, features], dim=1)
        cls_padding = torch.zeros(
            (padding_mask.shape[0], 1), dtype=torch.bool, device=padding_mask.device
        )
        encoded = self.encoder(
            sequence, src_key_padding_mask=torch.cat([cls_padding, padding_mask], dim=1)
        )
        return encoded[:, 0]


def load_local_encoder(path: Path, device: torch.device) -> FrozenMFMLocalEncoder:
    checkpoint = torch_load(path)
    raw = checkpoint.get("model", checkpoint)
    state = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("encoder."):
            key = key[len("encoder.") :]
        state[key] = value
    model = FrozenMFMLocalEncoder()
    model.encoder.load_state_dict(state, strict=True)
    del checkpoint, raw, state
    gc.collect()
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_global_encoder(path: Path, device: torch.device) -> tuple[GlobalMaskedImageEncoder, dict[str, Any]]:
    checkpoint = torch_load(path)
    model_config = checkpoint["config"]
    model = GlobalMaskedImageEncoder(model_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    metadata = {
        "epoch": int(checkpoint["epoch"]) + 1,
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "manifest_sha256": checkpoint["manifest_sha256"],
        "cache_index_sha256": checkpoint["cache_index_sha256"],
    }
    del checkpoint
    gc.collect()
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metadata


def pil_to_float_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tile_native_image(image: torch.Tensor, tile_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    _, height, width = image.shape
    padded_height = math.ceil(height / tile_size) * tile_size
    padded_width = math.ceil(width / tile_size) * tile_size
    pad_bottom, pad_right = padded_height - height, padded_width - width
    if pad_bottom or pad_right:
        mode = "reflect" if height > 1 and width > 1 else "replicate"
        if pad_bottom >= height or pad_right >= width:
            mode = "replicate"
        image = F.pad(image, (0, pad_right, 0, pad_bottom), mode=mode)
    rows, columns = padded_height // tile_size, padded_width // tile_size
    tiles = (
        image.unfold(1, tile_size, tile_size)
        .unfold(2, tile_size, tile_size)
        .permute(1, 2, 0, 3, 4)
        .reshape(rows * columns, 3, tile_size, tile_size)
        .contiguous()
    )
    row_centers = (torch.arange(rows, dtype=torch.float32) + 0.5) / rows
    col_centers = (torch.arange(columns, dtype=torch.float32) + 0.5) / columns
    yy, xx = torch.meshgrid(row_centers, col_centers, indexing="ij")
    positions = torch.stack([yy * 2 - 1, xx * 2 - 1], dim=-1).reshape(-1, 2)
    return tiles, positions


def make_spai_low_mask(size: int, radius: int, device: torch.device) -> torch.Tensor:
    half = torch.arange(size // 2, device=device, dtype=torch.float32)
    coordinates = torch.cat([half.flip(0), half], dim=0)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return ((xx.square() + yy.square()) < float(radius) ** 2).float().view(1, 1, size, size)


def spectral_views(tiles: torch.Tensor, low_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    spectrum = torch.fft.fftshift(torch.fft.fft2(tiles.float(), dim=(-2, -1)), dim=(-2, -1))
    low = torch.fft.ifft2(
        torch.fft.ifftshift(spectrum * low_mask, dim=(-2, -1)), dim=(-2, -1)
    ).real.clamp(0.0, 1.0)
    high = torch.fft.ifft2(
        torch.fft.ifftshift(spectrum * (1.0 - low_mask), dim=(-2, -1)), dim=(-2, -1)
    ).real.clamp(0.0, 1.0)
    return low, high


def load_runtime(
    artifact_root: Path, artifact_report: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Enable a Kaggle GPU accelerator before inference.")
    local = load_local_encoder(artifact_root / "checkpoints/local_encoder_best.pt", device)
    global_model, global_metadata = load_global_encoder(
        artifact_root / "checkpoints/global_encoder_best.pt", device
    )
    scaler = joblib.load(artifact_root / "gmm/real_feature_scaler.joblib")
    gmm = joblib.load(artifact_root / "gmm/real_distribution_gmm.joblib")
    thresholds = read_json(artifact_root / "gmm/real_only_thresholds.json")
    if int(getattr(scaler, "n_features_in_", -1)) != 768:
        raise RuntimeError("Scaler is not the expected 768-D real-feature scaler.")
    if tuple(gmm.means_.shape) != (artifact_report["selected_gmm_components"], 768):
        raise RuntimeError(f"Unexpected GMM means shape: {gmm.means_.shape}")
    primary_name = thresholds["primary_threshold"]
    primary_value = float(thresholds[primary_name])
    return {
        "device": device,
        "local": local,
        "global": global_model,
        "scaler": scaler,
        "gmm": gmm,
        "thresholds": thresholds,
        "primary_threshold_name": primary_name,
        "primary_threshold": primary_value,
        "training_config": artifact_report["training_config"],
        "global_metadata": global_metadata,
    }


# ---------------------------------------------------------------------------
# Batched inference and fixed-threshold scoring
# ---------------------------------------------------------------------------


def jpeg_recompress_pil(image: Image.Image, quality: int, subsampling: int) -> Image.Image:
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


def deterministic_jpeg_parameters(sample_id: str, config: dict[str, Any]) -> tuple[int, int]:
    quality_min = int(config["jpeg_quality_min"])
    quality_max = int(config["jpeg_quality_max"])
    subsampling_values = [int(value) for value in config["jpeg_subsampling_values"]]
    if not 1 <= quality_min <= quality_max <= 100:
        raise ValueError("JPEG quality range must satisfy 1 <= min <= max <= 100")
    if not subsampling_values or any(value not in {0, 1, 2} for value in subsampling_values):
        raise ValueError("jpeg_subsampling_values must be a non-empty subset of [0, 1, 2]")
    rng = random.Random(stable_seed(config["seed"], sample_id, "tiny_test_jpeg"))
    quality = rng.randint(quality_min, quality_max)
    subsampling = rng.choice(subsampling_values)
    return quality, subsampling


def read_image_entry(row: Any, tile_size: int, config: dict[str, Any]) -> dict[str, Any]:
    path = Path(row.image_path)
    with Image.open(path) as opened:
        opened.load()
        width, height = opened.size
        image_format = opened.format or path.suffix.lstrip(".").upper()
        image = opened.convert("RGB").copy()
    policy = str(config["jpeg_policy"])
    is_jpeg = str(image_format).upper() in {"JPEG", "JPG"} or path.suffix.lower() in {".jpg", ".jpeg"}
    if policy == "all_to_deterministic_jpeg":
        jpeg_applied = True
    elif policy == "non_jpeg_to_deterministic":
        jpeg_applied = not is_jpeg
    elif policy == "as_is":
        jpeg_applied = False
    else:
        raise ValueError(
            "jpeg_policy must be all_to_deterministic_jpeg, "
            "non_jpeg_to_deterministic, or as_is"
        )
    if jpeg_applied:
        jpeg_quality, jpeg_subsampling = deterministic_jpeg_parameters(str(row.sample_id), config)
        image = jpeg_recompress_pil(image, jpeg_quality, jpeg_subsampling)
        processed_format = "JPEG"
    else:
        jpeg_quality, jpeg_subsampling = None, None
        processed_format = str(image_format)
    tiles, positions = tile_native_image(pil_to_float_tensor(image), tile_size)
    return {
        "row": row,
        "tiles": tiles,
        "positions": positions,
        "width": int(width),
        "height": int(height),
        "image_format": str(image_format),
        "processed_format": processed_format,
        "jpeg_applied": bool(jpeg_applied),
        "jpeg_quality": jpeg_quality,
        "jpeg_subsampling": jpeg_subsampling,
        "num_tiles": int(len(tiles)),
    }


@torch.inference_mode()
def infer_pending(entries: Sequence[dict[str, Any]], runtime: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    device: torch.device = runtime["device"]
    local: FrozenMFMLocalEncoder = runtime["local"]
    global_model: GlobalMaskedImageEncoder = runtime["global"]
    train_config = runtime["training_config"]
    tile_size = int(train_config["tile_size"])
    radius = int(train_config["mfm_mask_radius"])
    all_tiles = torch.cat([entry["tiles"] for entry in entries], dim=0)
    offsets = [0]
    for entry in entries:
        offsets.append(offsets[-1] + entry["num_tiles"])
    low_mask = make_spai_low_mask(tile_size, radius, device)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    tile_microbatch = max(1, int(config["local_view_batch_size"]) // 2)
    fused_chunks: list[torch.Tensor] = []
    amp_enabled = bool(config["amp"] and device.type == "cuda")
    for start in range(0, len(all_tiles), tile_microbatch):
        tiles = all_tiles[start : start + tile_microbatch].to(device, non_blocking=True)
        low, high = spectral_views(tiles, low_mask)
        views = torch.cat([low, high], dim=0)
        normalized = (views - mean) / std
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            encoded = local(normalized)
        count = len(tiles)
        fused_chunks.append(((encoded[:count].float() + encoded[count:].float()) * 0.5))
        del tiles, low, high, views, normalized, encoded
    fused = torch.cat(fused_chunks, dim=0)

    batch_size = len(entries)
    max_tiles = max(entry["num_tiles"] for entry in entries)
    features = torch.zeros(batch_size, max_tiles, 768, device=device)
    positions = torch.zeros(batch_size, max_tiles, 2, device=device)
    padding = torch.ones(batch_size, max_tiles, dtype=torch.bool, device=device)
    for index, entry in enumerate(entries):
        start, stop = offsets[index], offsets[index + 1]
        count = stop - start
        features[index, :count] = fused[start:stop]
        positions[index, :count] = entry["positions"].to(device)
        padding[index, :count] = False
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        image_cls = global_model.encode_image(features, positions, padding)
    embeddings = image_cls.float().cpu().numpy()
    standardized = runtime["scaler"].transform(embeddings).astype(np.float32, copy=False)
    nll = -runtime["gmm"].score_samples(standardized)
    components = runtime["gmm"].predict(standardized)
    responsibilities = runtime["gmm"].predict_proba(standardized)
    primary = float(runtime["primary_threshold"])
    results: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        row = entry["row"]
        score = float(nll[index])
        results.append(
            {
                "sample_id": str(row.sample_id),
                "relative_path": str(row.relative_path),
                "image_path": str(row.image_path),
                "generator": str(row.generator),
                "actual_split": str(row.actual_split),
                "evaluation_cohort": str(row.evaluation_cohort),
                "label": int(row.label),
                "label_name": str(row.label_name),
                "width": entry["width"],
                "height": entry["height"],
                "image_format": entry["image_format"],
                "processed_format": entry["processed_format"],
                "jpeg_applied": entry["jpeg_applied"],
                "jpeg_quality": entry["jpeg_quality"],
                "jpeg_subsampling": entry["jpeg_subsampling"],
                "extension": str(row.extension),
                "num_tiles": entry["num_tiles"],
                "nll_anomaly_score": score,
                "gmm_component": int(components[index]),
                "gmm_max_responsibility": float(responsibilities[index].max()),
                "primary_threshold_name": runtime["primary_threshold_name"],
                "primary_threshold": primary,
                "artifact_primary_prediction": int(score > primary),
            }
        )
    return results


def run_inference(
    manifest: pd.DataFrame,
    runtime: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_dir = output_root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    partial_path = prediction_dir / "tiny_test_predictions.partial.csv"
    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if config["resume"] and partial_path.is_file():
        previous = pd.read_csv(partial_path)
        required_resume_columns = {
            "sample_id", "evaluation_cohort", "jpeg_applied", "jpeg_quality",
            "nll_anomaly_score",
        }
        if required_resume_columns.issubset(previous.columns):
            rows = previous.to_dict("records")
            done = set(previous.sample_id.astype(str))
            print(f"Resuming {len(done):,} completed Tiny-GenImage images.", flush=True)
        else:
            print("Ignoring an incompatible partial prediction file.", flush=True)
    last_saved_count = len(rows)

    tile_size = int(runtime["training_config"]["tile_size"])
    pending: list[dict[str, Any]] = []
    pending_tiles = 0
    errors: list[dict[str, Any]] = []
    started = time.time()

    def flush() -> None:
        nonlocal pending, pending_tiles, rows
        if not pending:
            return
        rows.extend(infer_pending(pending, runtime, config))
        pending, pending_tiles = [], 0

    remaining = manifest[~manifest.sample_id.astype(str).isin(done)]
    progress = tqdm(remaining.itertuples(index=False), total=len(remaining), desc="Spectral-GMM Tiny test")
    for row in progress:
        try:
            entry = read_image_entry(row, tile_size, config)
        except Exception as exc:
            errors.append(
                {"sample_id": row.sample_id, "image_path": row.image_path,
                 "error_type": type(exc).__name__, "error_message": str(exc)}
            )
            continue
        proposed_tiles = pending_tiles + entry["num_tiles"]
        if pending and (
            len(pending) >= int(config["max_images_per_batch"])
            or proposed_tiles > int(config["max_tiles_per_batch"])
        ):
            flush()
        pending.append(entry)
        pending_tiles += entry["num_tiles"]
        if len(rows) - last_saved_count >= int(config["save_every_images"]):
            pd.DataFrame(rows).drop_duplicates("sample_id", keep="last").to_csv(partial_path, index=False)
            last_saved_count = len(rows)
            elapsed = max(time.time() - started, 1e-6)
            progress.set_postfix(done=len(rows), img_s=f"{len(rows)/elapsed:.2f}")
    flush()

    predictions = pd.DataFrame(rows).drop_duplicates("sample_id", keep="last")
    predictions = manifest[["sample_id"]].merge(predictions, on="sample_id", how="inner")
    predictions.to_csv(prediction_dir / "tiny_test_predictions.csv", index=False)
    predictions.to_csv(partial_path, index=False)
    error_frame = pd.DataFrame(errors)
    error_frame.to_csv(prediction_dir / "tiny_test_errors.csv", index=False)
    return predictions, error_frame


def safe_metric(function, *args, **kwargs) -> float:
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return float("nan")


def metrics_at_threshold(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    labels = frame.label.to_numpy(dtype=int)
    scores = frame.nll_anomaly_score.to_numpy(dtype=float)
    predictions = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "num_samples": int(len(frame)),
        "num_real": int((labels == 0).sum()),
        "num_fake": int((labels == 1).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": safe_metric(accuracy_score, labels, predictions),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, labels, predictions),
        "real_recall": float(tn / max(tn + fp, 1)),
        "fake_recall": safe_metric(recall_score, labels, predictions, zero_division=0),
        "fake_precision": safe_metric(precision_score, labels, predictions, zero_division=0),
        "fake_f1": safe_metric(f1_score, labels, predictions, zero_division=0),
        "real_fpr": float(fp / max(tn + fp, 1)),
        "roc_auc": safe_metric(roc_auc_score, labels, scores),
        "average_precision": safe_metric(average_precision_score, labels, scores),
        "mean_real_nll": float(scores[labels == 0].mean()) if (labels == 0).any() else float("nan"),
        "mean_fake_nll": float(scores[labels == 1].mean()) if (labels == 1).any() else float("nan"),
        "threshold": float(threshold),
    }


def evaluate_fixed_threshold_grid(
    predictions: pd.DataFrame,
    output_root: Path,
    config: dict[str, Any],
) -> tuple[float, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate exactly the requested NLL thresholds on cached test scores."""
    predictions = predictions.reset_index(drop=True)
    if predictions.empty or set(predictions.label.unique()) != {0, 1}:
        raise RuntimeError("Fixed threshold grid requires both real and fake test labels.")
    for generator, group in predictions.groupby("generator", sort=True):
        if set(group.label.unique()) != {0, 1}:
            raise RuntimeError(
                f"Test generator {generator} lacks real or fake samples."
            )
    thresholds = [float(value) for value in config["threshold_grid"]]
    if len(thresholds) != 10 or len(set(thresholds)) != 10:
        raise ValueError("threshold_grid must contain exactly 10 unique values.")
    if not np.isfinite(np.asarray(thresholds, dtype=float)).all():
        raise ValueError("threshold_grid contains a non-finite value.")

    grid_rows: list[dict[str, Any]] = []
    generator_rows: list[dict[str, Any]] = []
    for threshold_index, threshold in enumerate(thresholds):
        overall = metrics_at_threshold(predictions, threshold)
        per_generator: list[dict[str, Any]] = []
        for generator, group in predictions.groupby("generator", sort=True):
            row = {
                "threshold_index": int(threshold_index),
                "threshold": float(threshold),
                "generator": str(generator),
                **metrics_at_threshold(group, threshold),
            }
            per_generator.append(row)
            generator_rows.append(row)
        per_generator_frame = pd.DataFrame(per_generator)
        grid_rows.append(
            {
                "threshold_index": int(threshold_index),
                **overall,
                "macro_generator_balanced_accuracy": float(
                    per_generator_frame.balanced_accuracy.mean()
                ),
                "macro_generator_real_recall": float(
                    per_generator_frame.real_recall.mean()
                ),
                "macro_generator_fake_recall": float(
                    per_generator_frame.fake_recall.mean()
                ),
                "macro_generator_fake_f1": float(per_generator_frame.fake_f1.mean()),
                "worst_generator_balanced_accuracy": float(
                    per_generator_frame.balanced_accuracy.min()
                ),
                "best_generator_balanced_accuracy": float(
                    per_generator_frame.balanced_accuracy.max()
                ),
                "real_fake_recall_gap": abs(
                    float(overall["real_recall"]) - float(overall["fake_recall"])
                ),
            }
        )

    grid = pd.DataFrame(grid_rows)
    generator_grid = pd.DataFrame(generator_rows)
    ranked = grid.sort_values(
        [
            "macro_generator_balanced_accuracy",
            "worst_generator_balanced_accuracy",
            "real_fake_recall_gap",
            "threshold",
        ],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    selected_threshold = float(best.threshold)
    grid["selected"] = grid.threshold.eq(selected_threshold)
    generator_grid["selected"] = generator_grid.threshold.eq(selected_threshold)

    metric_dir = output_root / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    grid.sort_values("threshold_index").to_csv(
        metric_dir / "test_fixed_threshold_grid_metrics.csv", index=False
    )
    generator_grid.sort_values(["threshold_index", "generator"]).to_csv(
        metric_dir / "test_fixed_threshold_grid_generator_metrics.csv", index=False
    )
    selection = {
        "selection_cohort": "full_test_grid",
        "selection_objective": "macro_generator_balanced_accuracy",
        "tie_breakers": [
            "worst_generator_balanced_accuracy_desc",
            "real_fake_recall_gap_asc",
            "threshold_asc",
        ],
        "num_test_samples": int(len(predictions)),
        "num_real": int((predictions.label == 0).sum()),
        "num_fake": int((predictions.label == 1).sum()),
        "num_generators": int(predictions.generator.nunique()),
        "num_grid_candidates": int(len(grid)),
        "threshold_grid": thresholds,
        "selected_threshold": selected_threshold,
        "test_balanced_accuracy": float(best.balanced_accuracy),
        "test_macro_generator_balanced_accuracy": float(
            best.macro_generator_balanced_accuracy
        ),
        "test_worst_generator_balanced_accuracy": float(
            best.worst_generator_balanced_accuracy
        ),
        "test_real_recall": float(best.real_recall),
        "test_fake_recall": float(best.fake_recall),
        "test_real_fake_recall_gap": float(best.real_fake_recall_gap),
        "methodological_note": (
            "Threshold selected on the test labels; report as test-grid/oracle, "
            "not as an independently calibrated deployment estimate."
        ),
    }
    write_json(metric_dir / "best_test_grid_threshold.json", selection)
    return selected_threshold, grid, generator_grid, selection


def dataset_statistics(predictions: pd.DataFrame, output_root: Path) -> dict[str, Any]:
    dataset_dir = output_root / "dataset"
    size_counts = (
        predictions.groupby(["width", "height"]).size().reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    format_counts = (
        predictions.groupby(["image_format", "processed_format", "extension", "jpeg_applied"])
        .size().reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    tile_counts = predictions.groupby("num_tiles").size().reset_index(name="count")
    size_counts.to_csv(dataset_dir / "image_size_counts.csv", index=False)
    format_counts.to_csv(dataset_dir / "image_format_counts.csv", index=False)
    tile_counts.to_csv(dataset_dir / "tile_count_distribution.csv", index=False)
    summary = {
        "num_decoded_images": int(len(predictions)),
        "cohort_counts": {
            str(key): int(value)
            for key, value in predictions.evaluation_cohort.value_counts().items()
        },
        "num_unique_sizes": int(len(size_counts)),
        "width": {
            "min": int(predictions.width.min()), "median": float(predictions.width.median()),
            "max": int(predictions.width.max()),
        },
        "height": {
            "min": int(predictions.height.min()), "median": float(predictions.height.median()),
            "max": int(predictions.height.max()),
        },
        "num_tiles": {
            "min": int(predictions.num_tiles.min()), "median": float(predictions.num_tiles.median()),
            "mean": float(predictions.num_tiles.mean()), "max": int(predictions.num_tiles.max()),
        },
        "format_counts": format_counts.to_dict("records"),
        "jpeg_application_counts": {
            ("applied" if bool(key) else "not_applied"): int(value)
            for key, value in predictions.jpeg_applied.value_counts().items()
        },
        "jpeg_quality": {
            "min": int(predictions.jpeg_quality.dropna().min()) if predictions.jpeg_quality.notna().any() else None,
            "max": int(predictions.jpeg_quality.dropna().max()) if predictions.jpeg_quality.notna().any() else None,
            "num_unique": int(predictions.jpeg_quality.dropna().nunique()),
        },
        "top_image_sizes": size_counts.head(20).to_dict("records"),
    }
    write_json(dataset_dir / "decoded_image_statistics.json", summary)
    return summary


def save_plots(predictions: pd.DataFrame, generator_metrics: pd.DataFrame, primary: dict[str, Any], output_root: Path) -> None:
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = predictions.label.to_numpy(dtype=int)
    scores = predictions.nll_anomaly_score.to_numpy(dtype=float)
    pred = (scores > float(primary["threshold"])).astype(int)

    matrix = confusion_matrix(labels, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    ax.set_xticks([0, 1], ["real", "fake"])
    ax.set_yticks([0, 1], ["real", "fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Spectral-GMM — Tiny-GenImage")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(plot_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    if len(np.unique(labels)) == 2:
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(fpr, tpr, label=f"AUC={primary['roc_auc']:.4f}")
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC")
        axes[0].legend()
        axes[1].plot(recall, precision, label=f"AP={primary['average_precision']:.4f}")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "roc_pr_curves.png", dpi=180)
        plt.close(fig)

    clipped_low, clipped_high = np.quantile(scores, [0.01, 0.99])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, name, color in [(0, "real", "tab:blue"), (1, "fake", "tab:orange")]:
        values = np.clip(scores[labels == label], clipped_low, clipped_high)
        ax.hist(values, bins=60, density=True, alpha=0.45, label=name, color=color)
    ax.axvline(
        float(primary["threshold"]), color="red", linestyle="--",
        label="grid-selected threshold",
    )
    ax.set(xlabel="GMM NLL anomaly score (1–99% clipped for display)", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "nll_distribution.png", dpi=180)
    plt.close(fig)

    ordered = generator_metrics.sort_values("balanced_accuracy")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(ordered))))
    ax.barh(ordered.generator, ordered.balanced_accuracy)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced accuracy @ best fixed test-grid threshold")
    fig.tight_layout()
    fig.savefig(plot_dir / "generator_balanced_accuracy.png", dpi=180)
    plt.close(fig)


def evaluate_predictions(
    predictions: pd.DataFrame,
    runtime: dict[str, Any],
    artifact_report: dict[str, Any],
    manifest_summary: dict[str, Any],
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    metric_dir = output_root / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_root / "predictions"
    selected_threshold, _, generator_grid, threshold_selection = evaluate_fixed_threshold_grid(
        predictions, output_root, config
    )
    predictions = predictions.copy()
    predictions["prediction"] = (
        predictions.nll_anomaly_score.to_numpy(dtype=float) > selected_threshold
    ).astype(int)
    predictions.to_csv(prediction_dir / "tiny_test_predictions_best_grid.csv", index=False)
    primary = {
        "threshold_name": "best_of_fixed_test_grid",
        **metrics_at_threshold(predictions, selected_threshold),
    }
    write_json(metric_dir / "test_best_grid_overall_metrics.json", primary)

    # Keep original real-only thresholds only for comparison with the fixed grid.
    thresholds = runtime["thresholds"]
    threshold_rows: list[dict[str, Any]] = []
    for name in ("nll_real_calibration_q90", "nll_real_calibration_q95", "nll_real_calibration_q99"):
        row = {
            "threshold_name": name,
            **metrics_at_threshold(predictions, float(thresholds[name])),
        }
        threshold_rows.append(row)
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(
        metric_dir / "test_legacy_real_only_threshold_metrics.csv", index=False
    )

    generator_metrics = generator_grid[
        generator_grid.threshold.eq(selected_threshold)
    ].copy().reset_index(drop=True)
    generator_metrics.to_csv(metric_dir / "test_best_grid_generator_metrics.csv", index=False)
    numeric_macro = [
        "accuracy", "balanced_accuracy", "real_recall", "fake_recall", "fake_precision",
        "fake_f1", "real_fpr", "roc_auc", "average_precision",
    ]
    macro = {f"macro_generator_{key}": float(generator_metrics[key].mean()) for key in numeric_macro}
    macro.update(
        {
            "num_generators": int(len(generator_metrics)),
            "worst_generator_balanced_accuracy": float(generator_metrics.balanced_accuracy.min()),
            "best_generator_balanced_accuracy": float(generator_metrics.balanced_accuracy.max()),
        }
    )
    write_json(metric_dir / "macro_generator_summary.json", macro)
    statistics = dataset_statistics(predictions, output_root)
    summary = {
        "model": "spectral_gmm_mfm_scratch_imagenet100k_v1",
        "score_direction": "higher NLL means more anomalous/fake",
        "threshold_policy": (
            "evaluate the 10 user-specified NLL thresholds on the full Tiny test; "
            "select maximum macro generator balanced accuracy"
        ),
        "threshold_selection": threshold_selection,
        "best_test_grid": primary,
        "macro": macro,
        "manifest": manifest_summary,
        "dataset_statistics": statistics,
        "provenance": {
            "global_checkpoint_sha256": artifact_report["global_checkpoint_sha256"],
            "local_checkpoint_sha256": artifact_report["local_checkpoint_sha256"],
            "selected_gmm_components": artifact_report["selected_gmm_components"],
            "global_checkpoint_epoch": runtime["global_metadata"]["epoch"],
            "global_best_validation_loss": runtime["global_metadata"]["best_validation_loss"],
        },
    }
    write_json(metric_dir / "evaluation_summary.json", summary)
    save_plots(predictions, generator_metrics, primary, output_root)
    return summary


def make_output_archive(output_root: Path) -> Path:
    archive = Path(shutil.make_archive(str(output_root), "zip", root_dir=output_root.parent, base_dir=output_root.name))
    return archive


def run_full_evaluation(user_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if user_config:
        config.update(user_config)
    seed_everything(int(config["seed"]))
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "evaluation_config.json", config)
    artifact_root = discover_artifact_root(config)
    artifact_report = audit_artifacts(artifact_root, config)
    write_json(output_root / "artifact_report.json", artifact_report)
    manifest, manifest_summary = build_tiny_test_manifest(config, output_root)
    runtime = load_runtime(artifact_root, artifact_report, config)
    predictions, errors = run_inference(manifest, runtime, config, output_root)
    if len(predictions) != len(manifest):
        raise RuntimeError(
            f"Inference incomplete: matched={len(predictions)}, expected={len(manifest)}, errors={len(errors)}"
        )
    summary = evaluate_predictions(
        predictions, runtime, artifact_report, manifest_summary, output_root, config
    )
    archive = make_output_archive(output_root)
    summary["output_root"] = str(output_root)
    summary["archive"] = str(archive)
    write_json(output_root / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Output archive: {archive}", flush=True)
    return summary


if __name__ == "__main__":
    run_full_evaluation()
