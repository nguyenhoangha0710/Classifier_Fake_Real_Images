"""Fit the real-only full-covariance GMM and evaluate Tiny-GenImage.

The primary local metric is codec-matched: native real JPEG versus fake PNG
encoded exactly once with JPEG quantization/subsampling profiles copied from
the paired real cohort. Native JPEG-real/PNG-fake is retained as a diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, JpegImagePlugin, __version__ as PILLOW_VERSION
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.checkpointing import load_checkpoint_cpu, sha256_file
from src.core import ForensicResidualExtractor, ForensicViT
from src.data import (
    IMAGE_SUFFIXES,
    ImagePathDataset,
    _estimate_jpeg_quality,
    read_manifest,
    real_manifest_filename,
    sha256_json,
)


FRE_RELEASE_URL = (
    "https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/"
    "releases/download/v1.2.0/fre.pt"
)
REAL_DIR_NAMES = {"nature", "real", "0_real", "0-real", "0"}
FAKE_DIR_NAMES = {"ai", "fake", "1_fake", "1-fake", "1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "configs" / "kaggle_2xt4.json")
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/forensic_twins_original"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--fre-checkpoint", type=Path, default=None)
    parser.add_argument("--gmm", type=Path, default=None)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--tiny-root", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fit-only", action="store_true")
    mode.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-eval-per-class", type=int, default=None)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=None,
        help="Real-only holdout fraction used only to calibrate NLL thresholds.",
    )
    parser.add_argument(
        "--threshold-percentiles",
        default=None,
        help="Comma-separated real-calibration percentiles, for example 80,85,90,95,98.",
    )
    parser.add_argument(
        "--primary-threshold-percentile",
        type=int,
        default=None,
        help="Percentile used for confusion-matrix and prediction visualizations.",
    )
    parser.add_argument(
        "--visualization-max-per-generator-class",
        type=int,
        default=None,
        help="Maximum deterministic PCA points for each generator/label pair.",
    )
    parser.add_argument(
        "--keep-codec-matched-images",
        action="store_true",
        help="Keep derived JPEG fakes. Default removes them after scoring to keep Kaggle Output small.",
    )
    parser.add_argument(
        "--jpeg-qualities",
        default="95,80,65,50,35",
        help="Comma-separated JPEG qualities applied to both real and fake images in memory.",
    )
    parser.add_argument(
        "--skip-fake-codec-matched",
        action="store_true",
        help="Skip the fake-only codec-matched diagnostic protocol.",
    )
    return parser.parse_args()


def resolve_file(explicit: Optional[Path], candidates: Sequence[Path], description: str) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {description}. Checked: {candidates}")


def resolve_fre(explicit: Optional[Path], input_root: Path, output_root: Path) -> Path:
    if explicit is not None and explicit.exists():
        return explicit
    matches: List[Path] = []
    if input_root.exists():
        direct = input_root / "fre.pt"
        if direct.is_file():
            matches.append(direct)
        for dataset_root in input_root.iterdir():
            if not dataset_root.is_dir():
                continue
            candidate = dataset_root / "fre.pt"
            if candidate.is_file():
                matches.append(candidate)
            matches.extend(path for path in dataset_root.glob("*/fre.pt") if path.is_file())
    if matches:
        return sorted(set(matches))[0]
    cached = output_root / "cache" / "fre.pt"
    if cached.exists():
        return cached
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(".tmp")
    try:
        request = urllib.request.Request(FRE_RELEASE_URL, headers={"User-Agent": "forensic-twins"})
        with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(temporary, cached)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Add official FSD fre.pt to Kaggle Input or enable Internet.") from error
    return cached


class FrozenFeatureExtractor(torch.nn.Module):
    def __init__(self, fre: ForensicResidualExtractor, encoder: ForensicViT):
        super().__init__()
        self.fre = fre.freeze()
        self.encoder = encoder.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, gray_halo: torch.Tensor) -> torch.Tensor:
        residual = self.fre(gray_halo)
        radius = self.fre.radius
        residual = residual[..., radius:-radius, radius:-radius]
        return self.encoder(residual)


class ProtocolImageDataset(Dataset):
    """Inference dataset with optional deterministic JPEG re-encoding in memory."""

    def __init__(self, rows: Sequence[Dict[str, object]], crop_size: int = 256, fre_radius: int = 7):
        self.rows = list(rows)
        self.crop_size = int(crop_size)
        self.fre_radius = int(fre_radius)

    @staticmethod
    def _flatten_qtables(value: object) -> Optional[List[List[int]]]:
        if value in (None, ""):
            return None
        return [[int(item) for item in table] for table in value]

    def _decode(self, row: Dict[str, object]) -> Image.Image:
        path = Path(str(row["image_path"]))
        with Image.open(path) as source:
            source.load()
            image = source.convert("RGB")
        if not bool(row.get("jpeg_reencode", False)):
            return image

        options: Dict[str, object] = {
            "format": "JPEG",
            "subsampling": int(row.get("jpeg_subsampling", 2)),
            "optimize": False,
            "progressive": False,
        }
        qtables = self._flatten_qtables(row.get("jpeg_qtables"))
        if qtables:
            options["qtables"] = qtables
        else:
            options["quality"] = int(row["jpeg_quality"])
        buffer = io.BytesIO()
        image.save(buffer, **options)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            encoded.load()
            return encoded.convert("RGB")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        path = Path(str(row["image_path"]))
        image = self._decode(row)
        if min(image.size) < self.crop_size:
            raise ValueError(f"Image shorter side is below {self.crop_size}: {path}")
        left = (image.width - self.crop_size) // 2
        top = (image.height - self.crop_size) // 2
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size)).convert("L")
        gray = torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).unsqueeze(0) / 255.0
        gray = F.pad(gray, (self.fre_radius,) * 4, mode="constant", value=0.0)
        return {
            "gray_halo": gray,
            "sample_id": str(row.get("sample_id", path.name)),
            "image_path": str(path),
            "label": int(row.get("label", 0)),
            "generator": str(row.get("generator", "real")),
            "protocol": str(row.get("protocol", "native_raw")),
        }


def load_extractor(config: Dict[str, object], checkpoint_path: Path, fre_path: Path, device: torch.device):
    model_cfg = config["model"]
    encoder = ForensicViT(
        in_channels=8,
        patch_size=int(model_cfg["patch_size"]),
        embed_dim=int(model_cfg["embed_dim"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
    )
    payload = load_checkpoint_cpu(checkpoint_path)
    encoder.load_state_dict(payload["encoder"], strict=True)
    fre = ForensicResidualExtractor.from_pretrained(fre_path)
    return FrozenFeatureExtractor(fre, encoder).to(device).eval(), payload


def extract_embeddings(
    rows: Sequence[Dict[str, object]],
    extractor: FrozenFeatureExtractor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    crop_size: int,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    eligible = []
    for row in rows:
        try:
            with Image.open(str(row["image_path"])) as image:
                if min(image.size) >= crop_size:
                    eligible.append(row)
        except Exception as error:
            print(f"[embedding] skip {row.get('image_path')}: {error}", flush=True)
    dataset = ProtocolImageDataset(eligible, crop_size=crop_size, fre_radius=7)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    outputs: List[np.ndarray] = []
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            gray = batch["gray_halo"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
                embedding = extractor(gray)
            outputs.append(embedding.float().cpu().numpy())
            cursor += len(gray)
            if cursor % 5000 < len(gray):
                print(f"[embedding] {cursor}/{len(dataset)}", flush=True)
    if not outputs:
        raise RuntimeError("No eligible images for embedding extraction.")
    return np.concatenate(outputs, axis=0), eligible


def fit_single_gaussian(embeddings: np.ndarray, reg_covar: float) -> Dict[str, np.ndarray]:
    values = np.asarray(embeddings, dtype=np.float64)
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / values.shape[0]
    covariance.flat[:: covariance.shape[0] + 1] += reg_covar
    cholesky = np.linalg.cholesky(covariance)
    return {
        "mean": mean,
        "covariance": covariance,
        "cholesky": cholesky,
        "reg_covar": np.asarray(reg_covar, dtype=np.float64),
        "num_fit": np.asarray(values.shape[0], dtype=np.int64),
    }


def gaussian_nll(embeddings: np.ndarray, gmm: Dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    centered = values - gmm["mean"]
    solved = np.linalg.solve(gmm["cholesky"], centered.T)
    mahalanobis = np.square(solved).sum(axis=0)
    log_det = 2.0 * np.log(np.diag(gmm["cholesky"])).sum()
    dim = values.shape[1]
    return 0.5 * (dim * math.log(2.0 * math.pi) + log_det + mahalanobis)


def save_gmm(gmm: Dict[str, np.ndarray], path: Path, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **gmm)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_gmm(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def parse_percentiles(value: object) -> List[int]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise ValueError(f"Unsupported percentile specification: {value!r}")
    percentiles = sorted({int(item) for item in items})
    if not percentiles or any(item <= 0 or item >= 100 for item in percentiles):
        raise ValueError("Threshold percentiles must be integers strictly between 0 and 100.")
    return percentiles


def split_real_rows(
    rows: Sequence[Dict[str, object]], calibration_fraction: float, seed: int
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Deterministically stratify ImageNet by class and COCO by source.

    ImageNet contributes exactly five calibration images per 25-image class at
    the default 20% fraction. The split is performed before fitting the GMM so
    calibration NLL values are out-of-fit for the density estimator.
    """
    if not 0.0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be in (0, 0.5).")
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row.get("label", 0)) != 0:
            raise ValueError("GMM fit/calibration manifest must contain real images only.")
        source = str(row.get("source", "unknown"))
        class_id = str(row.get("class_id", "")) if source == "imagenet1k" else "__all__"
        groups[(source, class_id)].append(dict(row))

    fit_rows: List[Dict[str, object]] = []
    calibration_rows: List[Dict[str, object]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: str(row.get("sample_id", "")))
        random.Random(f"{seed}:{key[0]}:{key[1]}:gmm-split").shuffle(group)
        calibration_count = max(1, int(round(len(group) * calibration_fraction)))
        calibration_count = min(calibration_count, len(group) - 1)
        calibration_rows.extend(group[:calibration_count])
        fit_rows.extend(group[calibration_count:])
    fit_rows.sort(key=lambda row: str(row.get("sample_id", "")))
    calibration_rows.sort(key=lambda row: str(row.get("sample_id", "")))
    if not fit_rows or not calibration_rows:
        raise RuntimeError("The deterministic GMM fit/calibration split is empty.")
    return fit_rows, calibration_rows


def write_rows_csv(rows: Sequence[Dict[str, object]], destination: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calibrate_thresholds(
    real_nll: np.ndarray, percentiles: Sequence[int], primary_percentile: int
) -> Dict[str, object]:
    scores = np.asarray(real_nll, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration NLL must be a finite, non-empty vector.")
    if primary_percentile not in percentiles:
        raise ValueError("primary_threshold_percentile must be present in threshold_percentiles.")
    values = {
        str(percentile): float(np.quantile(scores, percentile / 100.0))
        for percentile in percentiles
    }
    return {
        "score": "negative_log_likelihood",
        "prediction_rule": "fake_if_nll_gt_threshold",
        "percentiles": values,
        "primary_percentile": int(primary_percentile),
        "num_real_calibration": int(len(scores)),
        "calibration_nll_min": float(scores.min()),
        "calibration_nll_median": float(np.median(scores)),
        "calibration_nll_max": float(scores.max()),
        "empirical_real_fpr": {
            key: float(np.mean(scores > threshold)) for key, threshold in values.items()
        },
    }


def looks_like_tiny_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    children = [child for child in path.iterdir() if child.is_dir()]
    return sum(find_validation_labels(child) is not None for child in children) >= 2


def find_validation_labels(generator_dir: Path) -> Optional[Tuple[Path, Path]]:
    for split_name in ("val", "validation", "test"):
        split = generator_dir / split_name
        if not split.is_dir():
            continue
        real = next((child for child in split.iterdir() if child.is_dir() and child.name.lower() in REAL_DIR_NAMES), None)
        fake = next((child for child in split.iterdir() if child.is_dir() and child.name.lower() in FAKE_DIR_NAMES), None)
        if real is not None and fake is not None:
            return real, fake
    return None


def discover_tiny_root(explicit: Optional[Path], input_root: Path) -> Path:
    if explicit is not None:
        if not looks_like_tiny_root(explicit):
            raise FileNotFoundError(f"Not a Tiny-GenImage root: {explicit}")
        return explicit
    candidates = [input_root / "tiny-genimage", input_root / "datasets" / "yangsangtai" / "tiny-genimage"]
    candidates.extend(path for path in input_root.iterdir() if path.is_dir())
    for candidate in candidates:
        if looks_like_tiny_root(candidate):
            return candidate
        children = candidate.iterdir() if candidate.is_dir() else []
        for child in children:
            if child.is_dir() and looks_like_tiny_root(child):
                return child
    raise FileNotFoundError("Could not discover Tiny-GenImage under /kaggle/input")


def image_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _eligible_image_paths(paths: Sequence[Path], crop_size: int) -> Tuple[List[Path], int]:
    eligible: List[Path] = []
    invalid = 0
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if min(image.size) >= crop_size:
                    eligible.append(path)
                else:
                    invalid += 1
        except Exception as error:
            invalid += 1
            print(f"[tiny-audit] skip {path}: {error}", flush=True)
    return eligible, invalid


def build_native_eval_rows(
    tiny_root: Path,
    max_per_class: Optional[int],
    seed: int,
    crop_size: int = 256,
    audit_path: Optional[Path] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    for generator_dir in sorted(path for path in tiny_root.iterdir() if path.is_dir()):
        labels = find_validation_labels(generator_dir)
        if labels is None:
            continue
        real_dir, fake_dir = labels
        all_real, all_fake = image_files(real_dir), image_files(fake_dir)
        real, invalid_real = _eligible_image_paths(all_real, crop_size)
        fake, invalid_fake = _eligible_image_paths(all_fake, crop_size)
        count = min(len(real), len(fake))
        if max_per_class is not None:
            count = min(count, max_per_class)
        status = "included" if count > 0 else "excluded_no_balanced_eligible_pair"
        audit_rows.append(
            {
                "generator": generator_dir.name,
                "status": status,
                "raw_real": len(all_real),
                "raw_fake": len(all_fake),
                "eligible_real": len(real),
                "eligible_fake": len(fake),
                "invalid_or_too_small_real": invalid_real,
                "invalid_or_too_small_fake": invalid_fake,
                "selected_per_class": count,
                "crop_size": crop_size,
            }
        )
        print(
            f"[tiny-audit] {generator_dir.name}: real={len(real):,}/{len(all_real):,}, "
            f"fake={len(fake):,}/{len(all_fake):,}, selected/class={count:,}, status={status}",
            flush=True,
        )
        if count == 0:
            continue
        local_rng = random.Random(f"{seed}:{generator_dir.name}")
        local_rng.shuffle(real)
        local_rng.shuffle(fake)
        for label, paths in ((0, real[:count]), (1, fake[:count])):
            for path in paths:
                rows.append(
                    {
                        "sample_id": f"{generator_dir.name}:{label}:{path.name}",
                        "image_path": str(path),
                        "source_image_path": str(path),
                        "label": label,
                        "generator": generator_dir.name,
                        "protocol": "native_raw",
                    }
                )
    if audit_path is not None:
        write_dict_rows(audit_rows, audit_path)
    if not rows:
        raise RuntimeError(f"No Tiny-GenImage evaluation rows found below {tiny_root}")
    return rows


def build_common_jpeg_rows(
    native_rows: Sequence[Dict[str, object]], quality: int, subsampling: int = 2
) -> List[Dict[str, object]]:
    if not 1 <= int(quality) <= 100:
        raise ValueError(f"JPEG quality must be in [1, 100], got {quality}")
    protocol = f"jpeg_q{int(quality)}"
    rows: List[Dict[str, object]] = []
    for source_row in native_rows:
        row = dict(source_row)
        row.update(
            {
                "protocol": protocol,
                "jpeg_reencode": True,
                "jpeg_quality": int(quality),
                "jpeg_subsampling": int(subsampling),
            }
        )
        rows.append(row)
    return rows


def jpeg_profile(path: Path) -> Dict[str, object]:
    with Image.open(path) as image:
        if str(image.format).upper() != "JPEG" or not getattr(image, "quantization", None):
            raise ValueError(f"Codec-matched real cohort must be native JPEG: {path}")
        qtables = [list(image.quantization[key]) for key in sorted(image.quantization)]
        sampling = int(JpegImagePlugin.get_sampling(image))
    return {
        "profile_id": sha256_json({"qtables": qtables, "subsampling": sampling}),
        "qtables": qtables,
        "subsampling": sampling,
        "source_real": str(path),
    }


def encode_fake_with_profile(source: Path, destination: Path, profile: Dict[str, object]) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        original_size = image.size
        if "A" in image.getbands():
            alpha = np.asarray(image.getchannel("A"))
            if np.any(alpha != 255):
                raise ValueError(f"Fake PNG has non-opaque alpha; refusing silent composite: {source}")
        rgb = image.convert("RGB")
        temporary = destination.with_suffix(".tmp.jpg")
        rgb.save(
            temporary,
            format="JPEG",
            qtables=profile["qtables"],
            subsampling=int(profile["subsampling"]),
            optimize=False,
            progressive=False,
        )
    with Image.open(temporary) as check:
        if check.size != original_size or str(check.format).upper() != "JPEG":
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Codec matching changed dimensions or failed to produce JPEG.")
        written_qtables = [list(check.quantization[key]) for key in sorted(check.quantization)]
        written_sampling = int(JpegImagePlugin.get_sampling(check))
        if written_qtables != profile["qtables"] or written_sampling != int(profile["subsampling"]):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "JPEG encoder did not preserve the requested quantization/subsampling profile."
            )
    os.replace(temporary, destination)
    return sha256_file(destination)


def build_codec_matched_rows(
    native_rows: Sequence[Dict[str, object]], output_root: Path, seed: int
) -> List[Dict[str, object]]:
    """Match only fake images to real JPEG profiles, without writing derived files.

    This remains a diagnostic protocol: real images retain their native bytes while
    fake images are JPEG encoded once in memory. Common-Q protocols below are the
    fairer robustness comparison because both classes receive the same operation.
    """
    matched: List[Dict[str, object]] = []
    generators = sorted({str(row["generator"]) for row in native_rows})
    manifest_rows: List[Dict[str, object]] = []
    for generator in generators:
        real_rows = [dict(row) for row in native_rows if row["generator"] == generator and int(row["label"]) == 0]
        fake_rows = [dict(row) for row in native_rows if row["generator"] == generator and int(row["label"]) == 1]
        if len(real_rows) != len(fake_rows):
            raise RuntimeError(f"Unbalanced native cohort for {generator}")
        profiles = []
        for row in real_rows:
            try:
                profiles.append(jpeg_profile(Path(str(row["image_path"]))))
            except ValueError:
                continue
        if not profiles:
            raise RuntimeError(f"No native JPEG donor profile found for {generator}")
        order = list(range(len(profiles)))
        random.Random(f"{seed}:{generator}:codec").shuffle(order)
        for row in real_rows:
            row["protocol"] = "fake_codec_matched"
            matched.append(row)
        for fake_index, row in enumerate(fake_rows):
            profile = profiles[order[fake_index % len(order)]]
            source = Path(str(row["image_path"]))
            derived = dict(row)
            derived["source_image_path"] = str(source)
            derived["protocol"] = "fake_codec_matched"
            derived["jpeg_reencode"] = True
            derived["jpeg_qtables"] = profile["qtables"]
            derived["jpeg_subsampling"] = profile["subsampling"]
            estimated_quality = _estimate_jpeg_quality(
                {index: table for index, table in enumerate(profile["qtables"])}
            )
            derived["jpeg_quality"] = int(estimated_quality or 95)
            matched.append(derived)
            manifest_rows.append(
                {
                    "generator": generator,
                    "source_fake": str(source),
                    "profile_id": profile["profile_id"],
                    "source_real_profile": profile["source_real"],
                    "execution": "in_memory",
                    "estimated_quality": int(estimated_quality or 95),
                    "subsampling": profile["subsampling"],
                    "pillow_version": PILLOW_VERSION,
                }
            )
    manifest_path = output_root / "manifests" / "eval_codec_matched.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_dict_rows(manifest_rows, manifest_path)
    return matched


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = average_ranks(np.asarray(scores, dtype=np.float64))
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ranking_curves(
    labels: np.ndarray, scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if not positives or not negatives:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty
    cumulative_tp = np.cumsum(ordered_labels == 1)
    cumulative_fp = np.cumsum(ordered_labels == 0)
    distinct = np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    tp = cumulative_tp[distinct].astype(np.float64)
    fp = cumulative_fp[distinct].astype(np.float64)
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]
    recall = np.r_[0.0, tp / positives]
    precision = np.r_[1.0, tp / np.maximum(tp + fp, 1.0)]
    return fpr, tpr, recall, precision


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    _, _, recall, precision = ranking_curves(labels, scores)
    if not len(recall):
        return float("nan")
    return float(np.sum(np.diff(recall) * precision[1:]))


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = (np.asarray(scores, dtype=np.float64) > threshold).astype(np.int64)
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tpr = safe_divide(tp, tp + fn)
    tnr = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    recall = tpr
    return {
        "num_images": int(len(labels)),
        "support_real": int(tn + fp),
        "support_fake": int(tp + fn),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": safe_divide(tp + tn, len(labels)),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "precision": precision,
        "recall_tpr": recall,
        "specificity_tnr": tnr,
        "f1": safe_divide(2.0 * precision * recall, precision + recall),
        "fpr": safe_divide(fp, fp + tn),
        "fnr": safe_divide(fn, fn + tp),
    }


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "unnamed"


def write_dict_rows(rows: Sequence[Dict[str, object]], destination: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty metrics: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def threshold_metric_rows(
    labels: np.ndarray,
    scores: np.ndarray,
    generators: np.ndarray,
    protocol: str,
    thresholds: Dict[str, object],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    scopes = [("pooled", np.ones(len(labels), dtype=bool))]
    scopes.extend((generator, generators == generator) for generator in sorted(set(generators)))
    for scope, mask in scopes:
        for percentile, threshold in thresholds["percentiles"].items():
            metrics = classification_metrics(labels[mask], scores[mask], float(threshold))
            rows.append(
                {
                    "protocol": protocol,
                    "scope": scope,
                    "threshold_percentile": int(percentile),
                    "threshold_nll": float(threshold),
                    **metrics,
                }
            )
    return rows


def save_confusion_figure(
    metrics: Dict[str, object], title: str, destination: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    matrix = np.asarray(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        dtype=np.int64,
    )
    figure, axis = plt.subplots(figsize=(4.4, 4.0))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    axis.set_xticks([0, 1], labels=["Pred real", "Pred fake"])
    axis.set_yticks([0, 1], labels=["True real", "True fake"])
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def deterministic_visualization_indices(
    labels: np.ndarray,
    generators: np.ndarray,
    maximum_per_generator_class: int,
    seed: int,
) -> np.ndarray:
    selected: List[int] = []
    for generator in sorted(set(generators)):
        for label in (0, 1):
            candidates = np.flatnonzero((generators == generator) & (labels == label))
            if len(candidates) > maximum_per_generator_class:
                digest = hashlib.sha256(f"{seed}:{generator}:{label}:pca".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                candidates = np.sort(
                    rng.choice(candidates, size=maximum_per_generator_class, replace=False)
                )
            selected.extend(int(index) for index in candidates)
    return np.asarray(sorted(selected), dtype=np.int64)


def whitened_pca_2d(embeddings: np.ndarray, gmm: Dict[str, np.ndarray]) -> np.ndarray:
    centered = np.asarray(embeddings, dtype=np.float64) - gmm["mean"]
    whitened = np.linalg.solve(gmm["cholesky"], centered.T).T
    whitened -= whitened.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(whitened, full_matrices=False)
    return whitened @ right[:2].T


def save_visualizations(
    embeddings: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    generators: np.ndarray,
    protocol: str,
    gmm: Dict[str, np.ndarray],
    thresholds: Dict[str, object],
    output_root: Path,
    maximum_per_generator_class: int,
    seed: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    primary = str(thresholds["primary_percentile"])
    threshold = float(thresholds["percentiles"][primary])
    figure_root = output_root / "artifacts" / "figures" / protocol
    confusion_root = figure_root / "confusion_matrices" / f"q{primary}"
    predictions = (scores > threshold).astype(np.int64)

    scopes = [("pooled", np.ones(len(labels), dtype=bool))]
    scopes.extend((generator, generators == generator) for generator in sorted(set(generators)))
    for scope, mask in scopes:
        metrics = classification_metrics(labels[mask], scores[mask], threshold)
        save_confusion_figure(
            metrics,
            f"{protocol} | {scope} | q{primary}",
            confusion_root / f"{safe_name(scope)}.png",
        )

    chosen = deterministic_visualization_indices(
        labels, generators, maximum_per_generator_class, seed
    )
    coordinates = whitened_pca_2d(embeddings[chosen], gmm)
    shown_labels = labels[chosen]
    shown_generators = generators[chosen]
    shown_predictions = predictions[chosen]
    generator_names = sorted(set(shown_generators))
    colors = plt.get_cmap("tab20", max(len(generator_names), 1))

    figure, axis = plt.subplots(figsize=(11, 8))
    for generator_index, generator in enumerate(generator_names):
        for label, marker in ((0, "o"), (1, "x")):
            mask = (shown_generators == generator) & (shown_labels == label)
            if not np.any(mask):
                continue
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=16,
                alpha=0.55,
                marker=marker,
                color=colors(generator_index),
                label=f"{generator} | {'real' if label == 0 else 'fake'}",
            )
    axis.set_title(f"Whitened forensic embeddings by generator ({protocol})")
    axis.set_xlabel("PCA 1")
    axis.set_ylabel("PCA 2")
    axis.legend(fontsize=7, ncol=2, loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure_root.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_root / "pca_all_generators_true_labels.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    outcome_styles = {
        "TN real→real": (0, 0, "#2ca02c"),
        "FP real→fake": (0, 1, "#ff7f0e"),
        "FN fake→real": (1, 0, "#d62728"),
        "TP fake→fake": (1, 1, "#1f77b4"),
    }
    cluster_root = figure_root / f"generator_clusters_q{primary}"
    for generator in generator_names:
        figure, axis = plt.subplots(figsize=(6.2, 5.2))
        generator_mask = shown_generators == generator
        for outcome, (label, prediction, color) in outcome_styles.items():
            mask = generator_mask & (shown_labels == label) & (shown_predictions == prediction)
            if np.any(mask):
                axis.scatter(
                    coordinates[mask, 0], coordinates[mask, 1],
                    s=22, alpha=0.65, color=color, label=outcome,
                )
        axis.set_title(f"{generator} | {protocol} | q{primary}")
        axis.set_xlabel("Global PCA 1")
        axis.set_ylabel("Global PCA 2")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        cluster_root.mkdir(parents=True, exist_ok=True)
        figure.savefig(cluster_root / f"{safe_name(generator)}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    columns = min(3, max(1, len(generator_names)))
    rows = int(math.ceil(len(generator_names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.7 * rows), squeeze=False)
    for axis, generator in zip(axes.flat, generator_names):
        mask = generators == generator
        axis.hist(scores[mask & (labels == 0)], bins=35, alpha=0.55, label="real", color="#2ca02c")
        axis.hist(scores[mask & (labels == 1)], bins=35, alpha=0.55, label="fake", color="#d62728")
        axis.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"q{primary}")
        axis.set_title(generator)
        axis.set_xlabel("NLL (higher = more fake-like)")
        axis.set_ylabel("count")
        axis.legend(fontsize=8)
    for axis in axes.flat[len(generator_names):]:
        axis.axis("off")
    figure.suptitle(f"NLL distributions by generator ({protocol})", y=1.01)
    figure.tight_layout()
    figure.savefig(figure_root / "nll_distributions_by_generator.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(12, 5))
    for scope, mask in scopes:
        fpr, tpr, recall, precision = ranking_curves(labels[mask], scores[mask])
        if not len(fpr):
            continue
        roc_axis.plot(fpr, tpr, label=f"{scope} ({auroc(labels[mask], scores[mask]):.3f})")
        pr_axis.plot(
            recall,
            precision,
            label=f"{scope} ({average_precision(labels[mask], scores[mask]):.3f})",
        )
    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    roc_axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    pr_axis.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curves")
    for axis in (roc_axis, pr_axis):
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(figure_root / "roc_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_protocol(
    rows: Sequence[Dict[str, object]],
    protocol: str,
    extractor: FrozenFeatureExtractor,
    gmm: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    crop_size: int,
    output_root: Path,
    thresholds: Dict[str, object],
    visualization_max_per_generator_class: int,
    seed: int,
) -> Dict[str, object]:
    embeddings, eligible = extract_embeddings(
        rows, extractor, device, batch_size, num_workers, crop_size
    )
    scores = gaussian_nll(embeddings, gmm)
    labels = np.asarray([int(row["label"]) for row in eligible])
    generators = np.asarray([str(row["generator"]) for row in eligible])
    per_generator = []
    for generator in sorted(set(generators)):
        mask = generators == generator
        generator_labels = labels[mask]
        per_generator.append(
            {
                "protocol": protocol,
                "generator": generator,
                "num_images": int(mask.sum()),
                "support_real": int(np.sum(generator_labels == 0)),
                "support_fake": int(np.sum(generator_labels == 1)),
                "auroc": auroc(generator_labels, scores[mask]),
                "average_precision": average_precision(generator_labels, scores[mask]),
            }
        )
    valid_aurocs = [float(row["auroc"]) for row in per_generator if np.isfinite(row["auroc"])]
    valid_aps = [
        float(row["average_precision"])
        for row in per_generator
        if np.isfinite(row["average_precision"])
    ]
    summary = {
        "protocol": protocol,
        "pooled_auroc": auroc(labels, scores),
        "pooled_average_precision": average_precision(labels, scores),
        "macro_generator_auroc": float(np.mean(valid_aurocs)) if valid_aurocs else float("nan"),
        "macro_generator_average_precision": float(np.mean(valid_aps)) if valid_aps else float("nan"),
        "num_images": len(eligible),
        "support_real": int(np.sum(labels == 0)),
        "support_fake": int(np.sum(labels == 1)),
        "num_generators": len(per_generator),
        "num_valid_generators": len(valid_aurocs),
        "per_generator": per_generator,
    }
    metrics_dir = output_root / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{protocol}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_dict_rows(per_generator, metrics_dir / f"{protocol}_per_generator_ranking.csv")
    threshold_rows = threshold_metric_rows(labels, scores, generators, protocol, thresholds)
    write_dict_rows(threshold_rows, metrics_dir / f"{protocol}_threshold_metrics.csv")
    per_generator_dir = metrics_dir / protocol / "per_generator"
    for generator in sorted(set(generators)):
        generator_summary = {
            "protocol": protocol,
            "generator": generator,
            "ranking": next(row for row in per_generator if row["generator"] == generator),
            "threshold_metrics": [row for row in threshold_rows if row["scope"] == generator],
        }
        per_generator_dir.mkdir(parents=True, exist_ok=True)
        (per_generator_dir / f"{safe_name(generator)}.json").write_text(
            json.dumps(generator_summary, indent=2), encoding="utf-8"
        )

    embedding_dir = output_root / "artifacts" / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    np.save(embedding_dir / f"{protocol}_cls192.npy", embeddings.astype(np.float32))
    predictions = metrics_dir / f"{protocol}_predictions.csv"
    prediction_rows: List[Dict[str, object]] = []
    for row, score in zip(eligible, scores):
        output_row = {
            "sample_id": row["sample_id"],
            "image_path": row["image_path"],
            "source_image_path": row.get("source_image_path", row["image_path"]),
            "label": row["label"],
            "generator": row["generator"],
            "protocol": protocol,
            "nll": float(score),
        }
        for percentile, threshold in thresholds["percentiles"].items():
            output_row[f"pred_q{percentile}"] = int(float(score) > float(threshold))
        prediction_rows.append(output_row)
    write_dict_rows(prediction_rows, predictions)
    for generator in sorted(set(generators)):
        write_dict_rows(
            [row for row in prediction_rows if row["generator"] == generator],
            per_generator_dir / f"{safe_name(generator)}_predictions.csv",
        )

    save_visualizations(
        embeddings,
        labels,
        scores,
        generators,
        protocol,
        gmm,
        thresholds,
        output_root,
        visualization_max_per_generator_class,
        seed,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for embedding extraction.")
    device = torch.device("cuda:0")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_config = config["data"]
    manifest_name = real_manifest_filename(
        int(data_config["imagenet_per_class"]),
        int(data_config["coco_total"]),
    )
    manifest = args.manifest or args.output_root / "manifests" / manifest_name
    checkpoint = resolve_file(
        args.checkpoint,
        [
            args.output_root / "artifacts" / "checkpoints" / "last.ckpt",
            args.output_root / "artifacts" / "checkpoints" / "epoch_0080.ckpt",
        ],
        "training checkpoint",
    )
    fre_path = resolve_fre(args.fre_checkpoint, args.input_root, args.output_root)
    extractor, checkpoint_payload = load_extractor(config, checkpoint, fre_path, device)
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        extractor = torch.nn.DataParallel(extractor, device_ids=list(range(gpu_count)))
    print(f"[evaluation] CUDA devices={gpu_count}; inference DataParallel={gpu_count > 1}", flush=True)
    current_manifest_hash = sha256_file(manifest)
    current_fre_hash = sha256_file(fre_path)
    if str(checkpoint_payload.get("manifest_hash")) != current_manifest_hash:
        raise RuntimeError("Training manifest hash does not match the selected checkpoint.")
    if str(checkpoint_payload.get("fre_hash")) != current_fre_hash:
        raise RuntimeError("FRE hash does not match the selected checkpoint.")
    batch_size = int(config["evaluation"]["embedding_batch_per_gpu"]) * gpu_count
    crop_size = int(config["evaluation"]["center_crop"])
    num_workers = int(config["training"]["num_workers_per_rank"])
    evaluation_config = config["evaluation"]
    calibration_fraction = float(
        args.calibration_fraction
        if args.calibration_fraction is not None
        else evaluation_config.get("gmm_calibration_fraction", 0.2)
    )
    percentiles = parse_percentiles(
        args.threshold_percentiles
        if args.threshold_percentiles is not None
        else evaluation_config.get("threshold_percentiles", list(range(80, 99)))
    )
    primary_percentile = int(
        args.primary_threshold_percentile
        if args.primary_threshold_percentile is not None
        else evaluation_config.get("primary_threshold_percentile", 95)
    )
    visualization_max = int(
        args.visualization_max_per_generator_class
        if args.visualization_max_per_generator_class is not None
        else evaluation_config.get("visualization_max_per_generator_class", 500)
    )
    gmm_path = args.gmm or args.output_root / "artifacts" / "gmm" / "real_k1_full.npz"
    thresholds_path = (
        args.thresholds or args.output_root / "artifacts" / "gmm" / "thresholds.json"
    )

    if not args.eval_only:
        train_rows: List[Dict[str, object]] = list(read_manifest(manifest))
        fit_rows, calibration_rows = split_real_rows(
            train_rows, calibration_fraction, int(config["seed"])
        )
        split_dir = args.output_root / "manifests"
        write_rows_csv(fit_rows, split_dir / "gmm_fit_real.csv")
        write_rows_csv(calibration_rows, split_dir / "gmm_calibration_real.csv")
        split_summary = {
            "strategy": "deterministic_source_and_imagenet_class_stratified",
            "seed": int(config["seed"]),
            "calibration_fraction": calibration_fraction,
            "num_fit_requested": len(fit_rows),
            "num_calibration_requested": len(calibration_rows),
            "fit_by_source": {
                source: sum(str(row.get("source")) == source for row in fit_rows)
                for source in sorted({str(row.get("source")) for row in train_rows})
            },
            "calibration_by_source": {
                source: sum(str(row.get("source")) == source for row in calibration_rows)
                for source in sorted({str(row.get("source")) for row in train_rows})
            },
        }
        (split_dir / "gmm_split_summary.json").write_text(
            json.dumps(split_summary, indent=2), encoding="utf-8"
        )

        print(
            f"[gmm] extracting fit embeddings: {len(fit_rows):,} real images; "
            f"calibration: {len(calibration_rows):,}",
            flush=True,
        )
        fit_embeddings, eligible_fit = extract_embeddings(
            fit_rows, extractor, device, batch_size, num_workers, crop_size
        )
        embedding_dir = args.output_root / "artifacts" / "embeddings"
        embedding_dir.mkdir(parents=True, exist_ok=True)
        np.save(embedding_dir / "gmm_fit_real_cls192.npy", fit_embeddings.astype(np.float32))
        gmm = fit_single_gaussian(
            fit_embeddings, float(config["evaluation"]["gmm_reg_covar"])
        )
        save_gmm(
            gmm,
            gmm_path,
            {
                "components": 1,
                "covariance_type": "full",
                "score": "negative_log_likelihood",
                "embedding_dim": fit_embeddings.shape[1],
                "num_fit": fit_embeddings.shape[0],
                "manifest": str(manifest),
                "manifest_sha256": current_manifest_hash,
                "checkpoint": str(checkpoint),
                "checkpoint_optimizer_step": checkpoint_payload["optimizer_step"],
                "fre_sha256": current_fre_hash,
                "reg_covar": float(config["evaluation"]["gmm_reg_covar"]),
                "center_crop": crop_size,
                "calibration_fraction": calibration_fraction,
            },
        )
        print(f"[gmm] saved {gmm_path}", flush=True)

        calibration_embeddings, eligible_calibration = extract_embeddings(
            calibration_rows, extractor, device, batch_size, num_workers, crop_size
        )
        np.save(
            embedding_dir / "gmm_calibration_real_cls192.npy",
            calibration_embeddings.astype(np.float32),
        )
        calibration_nll = gaussian_nll(calibration_embeddings, gmm)
        np.save(embedding_dir / "gmm_calibration_real_nll.npy", calibration_nll)
        thresholds = calibrate_thresholds(
            calibration_nll, percentiles, primary_percentile
        )
        thresholds.update(
            {
                "manifest": str(manifest),
                "manifest_sha256": current_manifest_hash,
                "checkpoint": str(checkpoint),
                "checkpoint_optimizer_step": int(checkpoint_payload["optimizer_step"]),
                "fre_sha256": current_fre_hash,
                "num_real_fit": len(eligible_fit),
                "num_real_calibration": len(eligible_calibration),
                "calibration_fraction": calibration_fraction,
                "seed": int(config["seed"]),
            }
        )
        thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        thresholds_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
        print(f"[thresholds] saved {thresholds_path}", flush=True)
    if args.fit_only:
        return

    if not gmm_path.is_file():
        raise FileNotFoundError(f"Missing fitted GMM: {gmm_path}")
    if not thresholds_path.is_file():
        raise FileNotFoundError(f"Missing calibrated thresholds: {thresholds_path}")
    gmm = load_gmm(gmm_path)
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    tiny_root = discover_tiny_root(args.tiny_root, args.input_root)
    native_rows = build_native_eval_rows(
        tiny_root,
        max_per_class=args.max_eval_per_class,
        seed=int(config["evaluation"]["codec_seed"]),
        crop_size=crop_size,
        audit_path=args.output_root / "artifacts" / "metrics" / "tiny_eligibility_audit.csv",
    )
    quality_values = sorted(
        {int(value.strip()) for value in str(args.jpeg_qualities).split(",") if value.strip()},
        reverse=True,
    )
    if any(quality < 1 or quality > 100 for quality in quality_values):
        raise ValueError(f"JPEG qualities must be in [1, 100], got {quality_values}")
    protocol_rows: List[Tuple[str, List[Dict[str, object]]]] = [
        ("native_raw", list(native_rows))
    ]
    if not args.skip_fake_codec_matched:
        protocol_rows.append(
            (
                "fake_codec_matched",
                build_codec_matched_rows(
                    native_rows,
                    args.output_root,
                    seed=int(config["evaluation"]["codec_seed"]),
                ),
            )
        )
    protocol_rows.extend(
        (f"jpeg_q{quality}", build_common_jpeg_rows(native_rows, quality, subsampling=2))
        for quality in quality_values
    )

    summaries: List[Dict[str, object]] = []
    for protocol, rows in protocol_rows:
        print(f"[protocol] {protocol}: {len(rows):,} images", flush=True)
        summaries.append(
            evaluate_protocol(
                rows,
                protocol,
                extractor,
                gmm,
                device,
                batch_size,
                num_workers,
                crop_size,
                args.output_root,
                thresholds,
                visualization_max,
                int(config["seed"]),
            )
        )

    comparison_rows = [
        {
            "protocol": summary["protocol"],
            "num_images": summary["num_images"],
            "support_real": summary["support_real"],
            "support_fake": summary["support_fake"],
            "num_generators": summary["num_generators"],
            "pooled_auroc": summary["pooled_auroc"],
            "pooled_average_precision": summary["pooled_average_precision"],
            "macro_generator_auroc": summary["macro_generator_auroc"],
            "macro_generator_average_precision": summary["macro_generator_average_precision"],
        }
        for summary in summaries
    ]
    metrics_root = args.output_root / "artifacts" / "metrics"
    write_dict_rows(comparison_rows, metrics_root / "protocol_comparison.csv")
    per_generator_comparison: List[Dict[str, object]] = []
    for summary in summaries:
        protocol = str(summary["protocol"])
        with (metrics_root / f"{protocol}_threshold_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            primary_rows = {
                row["scope"]: row
                for row in csv.DictReader(handle)
                if int(row["threshold_percentile"]) == primary_percentile
                and row["scope"] != "pooled"
            }
        for ranking in summary["per_generator"]:
            generator = str(ranking["generator"])
            classification = primary_rows[generator]
            per_generator_comparison.append(
                {
                    "protocol": protocol,
                    "generator": generator,
                    "num_images": ranking["num_images"],
                    "support_real": ranking["support_real"],
                    "support_fake": ranking["support_fake"],
                    "auroc": ranking["auroc"],
                    "average_precision": ranking["average_precision"],
                    "threshold_percentile": primary_percentile,
                    "threshold_nll": classification["threshold_nll"],
                    "accuracy": classification["accuracy"],
                    "balanced_accuracy": classification["balanced_accuracy"],
                    "precision": classification["precision"],
                    "recall_tpr": classification["recall_tpr"],
                    "specificity_tnr": classification["specificity_tnr"],
                    "f1": classification["f1"],
                    "tn": classification["tn"],
                    "fp": classification["fp"],
                    "fn": classification["fn"],
                    "tp": classification["tp"],
                }
            )
    write_dict_rows(
        per_generator_comparison,
        metrics_root / f"per_generator_protocol_comparison_q{primary_percentile}.csv",
    )
    comparison = {
        "protocols": comparison_rows,
        "notes": {
            "native_raw": "Original Tiny-GenImage files; decoded and center-cropped only.",
            "fake_codec_matched": "Real files stay native; fake files are JPEG re-encoded in memory using real JPEG profiles.",
            "jpeg_qXX": "Both real and fake are decoded then JPEG re-encoded in memory at the same quality and 4:2:0 subsampling.",
        },
    }
    (metrics_root / "protocol_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2), flush=True)
    completion = {
        "status": "complete",
        "gmm": str(gmm_path),
        "thresholds": str(thresholds_path),
        "primary_threshold_percentile": primary_percentile,
        "protocols": [protocol for protocol, _ in protocol_rows],
        "metrics": str(args.output_root / "artifacts" / "metrics"),
        "figures": str(args.output_root / "artifacts" / "figures"),
    }
    (args.output_root / "artifacts" / "evaluation_complete.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
