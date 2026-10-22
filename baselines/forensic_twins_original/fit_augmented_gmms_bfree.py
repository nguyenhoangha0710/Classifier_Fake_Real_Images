"""Fit preprocessing-aware real-only GMMs and evaluate them on clean B-Free.

Stages are deliberately separable so Kaggle can persist artifacts after each
long operation:

    prepare -> extract -> fit -> evaluate

The encoder and FRE remain frozen.  All GMM thresholds are calibrated only on
held-out real base images.  Variants of one base image never cross the
fit/calibration boundary.  Positive class during B-Free evaluation is fake.
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
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fit_gmm_and_evaluate import (  # noqa: E402
    average_precision,
    load_extractor,
    parse_percentiles,
    split_real_rows,
)
from src.checkpointing import load_checkpoint_cpu, sha256_file  # noqa: E402
from src.data import discover_coco_roots, discover_imagenet_train, read_manifest  # noqa: E402
from test_bfree_gmm_kaggle import (  # noqa: E402
    FAKE_VARIANTS,
    REAL_VARIANT,
    atomic_write_json,
    bootstrap_confidence_intervals,
    build_bfree_manifests,
    extract_embeddings_resumable,
    load_reusable_manifest,
    paired_score_arrays,
    parse_protocol,
    ranking_row,
    resolve_dataset_root,
    resolve_file,
    safe_name,
    save_confusion_figure,
    threshold_rows_for_pairs,
    write_csv,
)


@dataclass(frozen=True)
class RealVariantSpec:
    name: str
    resize_square: Optional[int] = None
    resize_scale: Optional[float] = None
    blur_sigma: Optional[float] = None
    jpeg_quality: Optional[int] = None
    jpeg_subsampling: int = 2


VARIANT_SPECS: Mapping[str, RealVariantSpec] = {
    "native": RealVariantSpec("native"),
    # PNG itself is lossless.  This variant reproduces the pixel-changing part
    # of the B-Free pipeline: full-image resampling to a 512x512 raster.
    "resize512_png": RealVariantSpec("resize512_png", resize_square=512),
    "jpeg_q95": RealVariantSpec("jpeg_q95", jpeg_quality=95),
    "jpeg_q80": RealVariantSpec("jpeg_q80", jpeg_quality=80),
    "jpeg_q65": RealVariantSpec("jpeg_q65", jpeg_quality=65),
    "resize075_blur05_jpeg80": RealVariantSpec(
        "resize075_blur05_jpeg80",
        resize_scale=0.75,
        blur_sigma=0.5,
        jpeg_quality=80,
    ),
}
DEFAULT_VARIANTS = tuple(VARIANT_SPECS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=BASE_DIR / "configs" / "kaggle_2xt4_200k.json"
    )
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/forensic_twins_augmented_gmm"),
    )
    parser.add_argument("--encoder-dataset-root", type=Path, default=None)
    parser.add_argument("--bfree-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--fre-checkpoint", type=Path, default=None)
    parser.add_argument("--training-manifest", type=Path, default=None)
    parser.add_argument(
        "--stage",
        choices=("prepare", "extract", "fit", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--max-fit-base-images", type=int, default=None)
    parser.add_argument("--max-calibration-base-images", type=int, default=None)
    parser.add_argument("--max-bfree-ids", type=int, default=None)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--mixed-components", default="1,2,4,8")
    parser.add_argument(
        "--mixed-covariance",
        choices=("diag", "full", "tied", "spherical"),
        default="diag",
        help="Covariance shared by the mixed K-comparison. diag is practical for K>1 in 192-D.",
    )
    parser.add_argument("--skip-mixed-full-k1", action="store_true")
    parser.add_argument(
        "--max-gmm-fit-embeddings",
        type=int,
        default=None,
        help="Optional deterministic cap for pilot runs; omit to fit every available embedding.",
    )
    parser.add_argument("--gmm-max-iter", type=int, default=100)
    parser.add_argument("--gmm-n-init", type=int, default=1)
    parser.add_argument("--reg-covar", type=float, default=None)
    parser.add_argument("--threshold-percentiles", default="80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98")
    parser.add_argument("--primary-threshold-percentile", type=int, default=95)
    parser.add_argument("--batch-per-gpu", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--manifest-workers", type=int, default=16)
    parser.add_argument("--embedding-checkpoint-every", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-bfree-image-audit", action="store_true")
    parser.add_argument("--single-gpu", action="store_true")
    return parser.parse_args()


def parse_variants(value: str) -> List[str]:
    variants = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = [item for item in variants if item not in VARIANT_SPECS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; available={list(VARIANT_SPECS)}")
    if not variants:
        raise ValueError("At least one real preprocessing variant is required.")
    return variants


def parse_components(value: str) -> List[int]:
    components = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not components or any(item <= 0 for item in components):
        raise ValueError("--mixed-components must contain positive integers.")
    return components


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_inputs(args: argparse.Namespace) -> Dict[str, Path]:
    encoder_dataset = resolve_dataset_root(
        args.encoder_dataset_root,
        args.input_root,
        "hoangha071005",
        "twins-gmm-200k-checkpoint",
        "encoder checkpoint dataset",
    )
    bfree_root = resolve_dataset_root(
        args.bfree_root,
        args.input_root,
        "baopv051",
        "b-free-training-data",
        "B-Free dataset",
    )
    artifact_root = encoder_dataset / "forensic_twins_v2_200k"
    return {
        "encoder_dataset": encoder_dataset,
        "bfree_root": bfree_root,
        "checkpoint": resolve_file(
            args.checkpoint,
            (artifact_root / "artifacts" / "checkpoints" / "last.ckpt",),
            "last.ckpt",
        ),
        "fre": resolve_file(
            args.fre_checkpoint,
            (artifact_root / "cache" / "fre.pt",),
            "fre.pt",
        ),
        "training_manifest": resolve_file(
            args.training_manifest,
            (artifact_root / "manifests" / "train_real_200k.csv",),
            "train_real_200k.csv",
        ),
    }


def rebase_real_rows(rows: Sequence[Mapping[str, object]], input_root: Path) -> List[Dict[str, object]]:
    imagenet_root = discover_imagenet_train(input_root)
    coco_roots = discover_coco_roots(input_root)
    output: List[Dict[str, object]] = []
    for source_row in rows:
        row: Dict[str, object] = dict(source_row)
        source = str(row.get("source", ""))
        old_path = Path(str(row.get("image_path", "")))
        if source == "imagenet1k":
            class_id = str(row.get("class_id") or old_path.parent.name)
            path = imagenet_root / class_id / old_path.name
        elif source == "coco2017":
            split = str(row.get("original_split") or "train2017")
            root = coco_roots.get(split) or coco_roots.get("train2017")
            if root is None:
                raise FileNotFoundError(f"No COCO root for split={split}")
            path = root / old_path.name
        else:
            raise ValueError(f"Unsupported real source: {source}")
        row["image_path"] = str(path)
        row["label"] = 0
        output.append(row)
    return output


def _round_robin_class_sample(rows: Sequence[Dict[str, object]], count: int, seed: int) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("class_id", "__all__"))].append(dict(row))
    rng = random.Random(seed)
    for key in groups:
        groups[key].sort(key=lambda row: str(row.get("sample_id", "")))
        rng.shuffle(groups[key])
    selected: List[Dict[str, object]] = []
    keys = sorted(groups)
    cursor = 0
    while len(selected) < count:
        progressed = False
        for key in keys:
            if cursor < len(groups[key]):
                selected.append(groups[key][cursor])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def stratified_limit(
    rows: Sequence[Dict[str, object]], maximum: Optional[int], seed: int
) -> List[Dict[str, object]]:
    values = [dict(row) for row in rows]
    if maximum is None or maximum >= len(values):
        return sorted(values, key=lambda row: str(row.get("sample_id", "")))
    if maximum <= 0:
        raise ValueError("Base-image limits must be positive.")
    by_source: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in values:
        by_source[str(row.get("source", "unknown"))].append(row)
    exact = {source: maximum * len(group) / len(values) for source, group in by_source.items()}
    targets = {source: int(math.floor(value)) for source, value in exact.items()}
    remainder = maximum - sum(targets.values())
    for source in sorted(by_source, key=lambda key: (exact[key] - targets[key], key), reverse=True)[:remainder]:
        targets[source] += 1
    selected: List[Dict[str, object]] = []
    for index, source in enumerate(sorted(by_source)):
        group = by_source[source]
        target = min(targets[source], len(group))
        if source == "imagenet1k":
            selected.extend(_round_robin_class_sample(group, target, seed + index))
        else:
            group.sort(key=lambda row: str(row.get("sample_id", "")))
            random.Random(seed + index).shuffle(group)
            selected.extend(group[:target])
    if len(selected) != maximum:
        selected_ids = {str(row.get("sample_id")) for row in selected}
        remaining = [row for row in values if str(row.get("sample_id")) not in selected_ids]
        remaining.sort(key=lambda row: str(row.get("sample_id", "")))
        selected.extend(remaining[: maximum - len(selected)])
    return sorted(selected, key=lambda row: str(row.get("sample_id", "")))


def manifest_identity(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("sample_id", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("image_path", "")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_base_manifest(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(rows, path, fieldnames)


def prepare(args: argparse.Namespace, paths: Mapping[str, Path], config: Mapping[str, object]) -> Dict[str, object]:
    rows = rebase_real_rows(read_manifest(paths["training_manifest"]), args.input_root)
    fit_rows, calibration_rows = split_real_rows(rows, args.calibration_fraction, args.seed)
    fit_rows = stratified_limit(fit_rows, args.max_fit_base_images, args.seed + 11)
    calibration_rows = stratified_limit(
        calibration_rows, args.max_calibration_base_images, args.seed + 29
    )
    manifest_root = args.output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    fit_manifest = manifest_root / "augmented_gmm_fit_base.csv"
    calibration_manifest = manifest_root / "augmented_gmm_calibration_base.csv"
    write_base_manifest(fit_rows, fit_manifest)
    write_base_manifest(calibration_rows, calibration_manifest)
    crop_size = int(config["evaluation"]["center_crop"])
    bfree_manifest, bfree_summary = build_bfree_manifests(
        paths["bfree_root"],
        paths["training_manifest"],
        args.output_root,
        "valid_clean",
        args.max_bfree_ids,
        args.seed,
        crop_size,
        args.manifest_workers,
        not args.skip_bfree_image_audit,
    )
    summary = {
        "training_manifest": str(paths["training_manifest"]),
        "training_manifest_sha256": sha256_file(paths["training_manifest"]),
        "fit_base_manifest": str(fit_manifest),
        "fit_base_manifest_sha256": sha256_file(fit_manifest),
        "calibration_base_manifest": str(calibration_manifest),
        "calibration_base_manifest_sha256": sha256_file(calibration_manifest),
        "fit_base_images": len(fit_rows),
        "calibration_base_images": len(calibration_rows),
        "fit_by_source": dict(sorted(_counts(fit_rows, "source").items())),
        "calibration_by_source": dict(sorted(_counts(calibration_rows, "source").items())),
        "variants": list(DEFAULT_VARIANTS),
        "variant_specs": {name: asdict(spec) for name, spec in VARIANT_SPECS.items()},
        "bfree_manifest": str(bfree_manifest),
        "bfree_summary": bfree_summary,
        "split_unit": "base_image_before_variant_generation",
        "seed": args.seed,
    }
    atomic_write_json(manifest_root / "augmented_gmm_prepare_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _counts(rows: Sequence[Mapping[str, object]], key: str) -> Dict[str, int]:
    output: Dict[str, int] = defaultdict(int)
    for row in rows:
        output[str(row.get(key, ""))] += 1
    return dict(output)


class RealVariantDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        spec: RealVariantSpec,
        crop_size: int,
        fre_radius: int = 7,
    ):
        self.rows = list(rows)
        self.spec = spec
        self.crop_size = int(crop_size)
        self.fre_radius = int(fre_radius)

    def __len__(self) -> int:
        return len(self.rows)

    def transform(self, image: Image.Image) -> Image.Image:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        if self.spec.resize_square is not None:
            side = int(self.spec.resize_square)
            image = image.resize((side, side), resampling)
        if self.spec.resize_scale is not None:
            original = image.size
            reduced = (
                max(1, int(round(original[0] * self.spec.resize_scale))),
                max(1, int(round(original[1] * self.spec.resize_scale))),
            )
            image = image.resize(reduced, resampling).resize(original, resampling)
        if self.spec.blur_sigma is not None:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.spec.blur_sigma))
        if self.spec.jpeg_quality is not None:
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=int(self.spec.jpeg_quality),
                subsampling=int(self.spec.jpeg_subsampling),
                optimize=False,
                progressive=False,
            )
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                encoded.load()
                image = encoded.convert("RGB")
        return image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = Path(str(self.rows[index]["image_path"]))
        with Image.open(path) as source:
            source.load()
            image = source.convert("RGB")
        image = self.transform(image)
        if min(image.size) < self.crop_size:
            raise ValueError(f"Image shorter side below crop size: {path}")
        left = (image.width - self.crop_size) // 2
        top = (image.height - self.crop_size) // 2
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size)).convert("L")
        gray = torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).unsqueeze(0) / 255.0
        gray = F.pad(gray, (self.fre_radius,) * 4, mode="constant", value=0.0)
        return gray, index


def load_base_rows(args: argparse.Namespace, split: str) -> List[Dict[str, object]]:
    path = args.output_root / "manifests" / f"augmented_gmm_{split}_base.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run --stage prepare first: {path}")
    return rebase_real_rows(read_csv_rows(path), args.input_root)


def embedding_paths(output_root: Path, split: str, variant: str) -> Tuple[Path, Path]:
    root = output_root / "artifacts" / "augmented_embeddings"
    return root / f"{split}_{variant}_cls192.npy", root / f"{split}_{variant}_progress.json"


def extract_variant_embeddings(
    rows: Sequence[Mapping[str, object]],
    split: str,
    spec: RealVariantSpec,
    extractor: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    crop_size: int,
    embed_dim: int,
    checkpoint_every: int,
    output_root: Path,
) -> Path:
    embedding_path, progress_path = embedding_paths(output_root, split, spec.name)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    expected = {
        "split": split,
        "variant": spec.name,
        "variant_spec": asdict(spec),
        "manifest_identity": manifest_identity(rows),
        "row_count": len(rows),
        "embedding_dim": embed_dim,
    }
    start = 0
    if embedding_path.is_file() and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if progress.get(key) != value:
                raise RuntimeError(f"Resume mismatch {split}/{spec.name}: {key}")
        start = int(progress.get("next_index", 0))
        if bool(progress.get("complete")) and start == len(rows):
            print(f"[embedding] reuse {split}/{spec.name}: {embedding_path}", flush=True)
            return embedding_path
        matrix = np.lib.format.open_memmap(embedding_path, mode="r+")
        if matrix.shape != (len(rows), embed_dim):
            raise RuntimeError(f"Embedding shape mismatch: {matrix.shape}")
        print(f"[embedding] resume {split}/{spec.name}: {start:,}/{len(rows):,}", flush=True)
    else:
        matrix = np.lib.format.open_memmap(
            embedding_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), embed_dim),
        )
        atomic_write_json(progress_path, {**expected, "next_index": 0, "complete": False})
    dataset = RealVariantDataset(rows[start:], spec, crop_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    cursor = start
    last_saved = start
    with torch.inference_mode():
        for gray, _ in loader:
            gray = gray.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                values = extractor(gray)
            batch = values.float().cpu().numpy()
            matrix[cursor : cursor + len(batch)] = batch
            cursor += len(batch)
            if cursor - last_saved >= checkpoint_every or cursor == len(rows):
                matrix.flush()
                atomic_write_json(
                    progress_path,
                    {**expected, "next_index": cursor, "complete": cursor == len(rows)},
                )
                print(f"[embedding] {split}/{spec.name}: {cursor:,}/{len(rows):,}", flush=True)
                last_saved = cursor
    matrix.flush()
    del matrix
    if cursor != len(rows):
        raise RuntimeError(f"Extraction stopped at {cursor}/{len(rows)}")
    return embedding_path


def load_frozen_extractor(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    config: Mapping[str, object],
) -> Tuple[torch.nn.Module, torch.device, int, Mapping[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for embedding extraction.")
    payload_cpu = load_checkpoint_cpu(paths["checkpoint"])
    manifest_hash = sha256_file(paths["training_manifest"])
    if str(payload_cpu.get("manifest_hash")) != manifest_hash:
        raise RuntimeError("Checkpoint and train_real_200k.csv hashes do not match.")
    if str(payload_cpu.get("fre_hash")) != sha256_file(paths["fre"]):
        raise RuntimeError("Checkpoint and fre.pt hashes do not match.")
    device = torch.device("cuda:0")
    extractor, payload = load_extractor(dict(config), paths["checkpoint"], paths["fre"], device)
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1 and not args.single_gpu:
        extractor = torch.nn.DataParallel(extractor, device_ids=list(range(gpu_count)))
    used_gpus = gpu_count if gpu_count > 1 and not args.single_gpu else 1
    print(f"[gpu] visible={gpu_count}; used={used_gpus}; DataParallel={used_gpus > 1}", flush=True)
    return extractor, device, used_gpus, payload


def run_extract(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    config: Mapping[str, object],
    variants: Sequence[str],
) -> None:
    fit_rows = load_base_rows(args, "fit")
    calibration_rows = load_base_rows(args, "calibration")
    extractor, device, used_gpus, _ = load_frozen_extractor(args, paths, config)
    per_gpu = args.batch_per_gpu or int(config["evaluation"]["embedding_batch_per_gpu"])
    batch_size = per_gpu * used_gpus
    crop_size = int(config["evaluation"]["center_crop"])
    embed_dim = int(config["model"]["embed_dim"])
    for variant in variants:
        spec = VARIANT_SPECS[variant]
        for split, rows in (("fit", fit_rows), ("calibration", calibration_rows)):
            extract_variant_embeddings(
                rows,
                split,
                spec,
                extractor,
                device,
                batch_size,
                args.num_workers,
                crop_size,
                embed_dim,
                args.embedding_checkpoint_every,
                args.output_root,
            )


def deterministic_embedding_subset(values: np.ndarray, maximum: Optional[int], seed: int) -> np.ndarray:
    if maximum is None or maximum >= len(values):
        return np.asarray(values)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(values), maximum, replace=False))
    return np.asarray(values[indices])


def fit_sklearn_gmm(
    embeddings: np.ndarray,
    components: int,
    covariance_type: str,
    reg_covar: float,
    max_iter: int,
    n_init: int,
    seed: int,
):
    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(
        n_components=components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        max_iter=max_iter,
        n_init=n_init,
        init_params="kmeans",
        random_state=seed,
        verbose=1,
        verbose_interval=10,
    )
    model.fit(np.asarray(embeddings, dtype=np.float64))
    return model


def save_mixture(model, path: Path, metadata: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        weights=model.weights_,
        means=model.means_,
        covariances=model.covariances_,
        precisions_cholesky=model.precisions_cholesky_,
        covariance_type=np.asarray(model.covariance_type),
        converged=np.asarray(model.converged_),
        n_iter=np.asarray(model.n_iter_),
        lower_bound=np.asarray(model.lower_bound_),
    )
    atomic_write_json(path.with_suffix(".json"), dict(metadata))


def load_mixture(path: Path) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as payload:
        output: Dict[str, object] = {key: payload[key] for key in payload.files}
    output["covariance_type"] = str(np.asarray(output["covariance_type"]).item())
    return output


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.sum(np.exp(values - maximum), axis=axis)
    )


def mixture_nll(
    embeddings: np.ndarray, model: Mapping[str, object], batch_size: int = 8192
) -> np.ndarray:
    means = np.asarray(model["means"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    covariances = np.asarray(model["covariances"], dtype=np.float64)
    covariance_type = str(model["covariance_type"])
    dimension = means.shape[1]
    constant = dimension * math.log(2.0 * math.pi)
    outputs: List[np.ndarray] = []
    for start in range(0, len(embeddings), batch_size):
        values = np.asarray(embeddings[start : start + batch_size], dtype=np.float64)
        component_logs: List[np.ndarray] = []
        for component, mean in enumerate(means):
            difference = values - mean
            if covariance_type == "full":
                covariance = covariances[component]
                cholesky = np.linalg.cholesky(covariance)
                solved = np.linalg.solve(cholesky, difference.T)
                mahalanobis = np.square(solved).sum(axis=0)
                log_det = 2.0 * np.log(np.diag(cholesky)).sum()
            elif covariance_type == "tied":
                cholesky = np.linalg.cholesky(covariances)
                solved = np.linalg.solve(cholesky, difference.T)
                mahalanobis = np.square(solved).sum(axis=0)
                log_det = 2.0 * np.log(np.diag(cholesky)).sum()
            elif covariance_type == "diag":
                variance = covariances[component]
                mahalanobis = np.square(difference).dot(1.0 / variance)
                log_det = np.log(variance).sum()
            elif covariance_type == "spherical":
                variance = float(covariances[component])
                mahalanobis = np.square(difference).sum(axis=1) / variance
                log_det = dimension * math.log(variance)
            else:
                raise ValueError(f"Unsupported covariance type: {covariance_type}")
            component_logs.append(
                math.log(max(float(weights[component]), 1e-300))
                - 0.5 * (constant + log_det + mahalanobis)
            )
        log_probability = _logsumexp(np.column_stack(component_logs), axis=1)
        outputs.append(-log_probability)
    return np.concatenate(outputs)


def calibrated_thresholds(
    scores: np.ndarray, percentiles: Sequence[int], primary: int
) -> Dict[str, object]:
    values = np.asarray(scores, dtype=np.float64)
    thresholds = {str(q): float(np.percentile(values, q)) for q in percentiles}
    return {
        "score": "negative_log_likelihood",
        "prediction_rule": "fake_if_nll_gt_threshold",
        "positive_class": "fake",
        "primary_percentile": int(primary),
        "percentiles": thresholds,
        "num_real_calibration": int(len(values)),
        "nll_mean": float(values.mean()),
        "nll_std": float(values.std()),
        "nll_min": float(values.min()),
        "nll_max": float(values.max()),
    }


def load_variant_array(args: argparse.Namespace, split: str, variant: str) -> np.ndarray:
    path, progress_path = embedding_paths(args.output_root, split, variant)
    if not path.is_file() or not progress_path.is_file():
        raise FileNotFoundError(f"Missing embedding; run extract for {split}/{variant}: {path}")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if not progress.get("complete"):
        raise RuntimeError(f"Incomplete embedding: {progress_path}")
    return np.load(path, mmap_mode="r")


def model_case_path(output_root: Path, model_id: str) -> Path:
    return output_root / "artifacts" / "augmented_gmms" / model_id / "gmm.npz"


def fit_one_case(
    args: argparse.Namespace,
    model_id: str,
    fit_values: np.ndarray,
    calibration_by_variant: Mapping[str, np.ndarray],
    components: int,
    covariance_type: str,
    reg_covar: float,
    variants: Sequence[str],
    percentiles: Sequence[int],
) -> Dict[str, object]:
    selected_fit = deterministic_embedding_subset(
        fit_values, args.max_gmm_fit_embeddings, args.seed + components * 101
    )
    print(
        f"[gmm] {model_id}: fit={len(selected_fit):,}, K={components}, covariance={covariance_type}",
        flush=True,
    )
    model = fit_sklearn_gmm(
        selected_fit,
        components,
        covariance_type,
        reg_covar,
        args.gmm_max_iter,
        args.gmm_n_init,
        args.seed,
    )
    calibration_scores = {
        variant: mixture_nll(values, {
            "weights": model.weights_,
            "means": model.means_,
            "covariances": model.covariances_,
            "covariance_type": model.covariance_type,
        })
        for variant, values in calibration_by_variant.items()
    }
    pooled_scores = np.concatenate([calibration_scores[variant] for variant in variants])
    threshold_payload = calibrated_thresholds(
        pooled_scores, percentiles, args.primary_threshold_percentile
    )
    threshold_payload.update(
        {
            "model_id": model_id,
            "components": components,
            "covariance_type": covariance_type,
            "calibration_variants": list(variants),
            "by_variant": {
                variant: calibrated_thresholds(
                    calibration_scores[variant], percentiles, args.primary_threshold_percentile
                )
                for variant in variants
            },
            "calibration_policy": "global_equal_count_per_variant",
        }
    )
    path = model_case_path(args.output_root, model_id)
    metadata = {
        "model_id": model_id,
        "components": components,
        "covariance_type": covariance_type,
        "variants": list(variants),
        "num_fit_available": int(len(fit_values)),
        "num_fit_used": int(len(selected_fit)),
        "embedding_dim": int(selected_fit.shape[1]),
        "reg_covar": reg_covar,
        "converged": bool(model.converged_),
        "n_iter": int(model.n_iter_),
        "lower_bound": float(model.lower_bound_),
        "bic_on_fit_subset": float(model.bic(np.asarray(selected_fit, dtype=np.float64))),
        "aic_on_fit_subset": float(model.aic(np.asarray(selected_fit, dtype=np.float64))),
        "mean_calibration_nll": float(pooled_scores.mean()),
        "threshold_policy_for_bfree": "global",
    }
    save_mixture(model, path, metadata)
    atomic_write_json(path.parent / "thresholds.json", threshold_payload)
    calibration_rows: List[Dict[str, object]] = []
    for variant, scores in calibration_scores.items():
        for percentile, threshold in threshold_payload["by_variant"][variant]["percentiles"].items():
            calibration_rows.append(
                {
                    "model_id": model_id,
                    "variant": variant,
                    "threshold_policy": "matched_variant",
                    "threshold_percentile": int(percentile),
                    "threshold_nll": threshold,
                    "empirical_fpr": float(np.mean(scores > float(threshold))),
                }
            )
        for percentile, threshold in threshold_payload["percentiles"].items():
            calibration_rows.append(
                {
                    "model_id": model_id,
                    "variant": variant,
                    "threshold_policy": "global_mixture",
                    "threshold_percentile": int(percentile),
                    "threshold_nll": threshold,
                    "empirical_fpr": float(np.mean(scores > float(threshold))),
                }
            )
    write_csv(calibration_rows, path.parent / "calibration_fpr.csv")
    print(json.dumps(metadata, indent=2), flush=True)
    return {**metadata, "gmm": str(path), "thresholds": str(path.parent / "thresholds.json")}


def run_fit(
    args: argparse.Namespace,
    config: Mapping[str, object],
    variants: Sequence[str],
) -> List[Dict[str, object]]:
    fit_arrays = {variant: load_variant_array(args, "fit", variant) for variant in variants}
    calibration_arrays = {
        variant: load_variant_array(args, "calibration", variant) for variant in variants
    }
    reg_covar = float(
        args.reg_covar
        if args.reg_covar is not None
        else config["evaluation"].get("gmm_reg_covar", 1e-6)
    )
    percentiles = parse_percentiles(args.threshold_percentiles)
    index: List[Dict[str, object]] = []
    # Protocol-specific K=1/full models keep each legitimate processing mode compact.
    for variant in variants:
        index.append(
            fit_one_case(
                args,
                f"protocol_{variant}_k1_full",
                np.asarray(fit_arrays[variant]),
                {variant: np.asarray(calibration_arrays[variant])},
                1,
                "full",
                reg_covar,
                [variant],
                percentiles,
            )
        )
    mixed_fit = np.concatenate([np.asarray(fit_arrays[variant]) for variant in variants], axis=0)
    mixed_calibration = {
        variant: np.asarray(calibration_arrays[variant]) for variant in variants
    }
    if not args.skip_mixed_full_k1:
        index.append(
            fit_one_case(
                args,
                "mixed_k1_full",
                mixed_fit,
                mixed_calibration,
                1,
                "full",
                reg_covar,
                variants,
                percentiles,
            )
        )
    for components in parse_components(args.mixed_components):
        if (
            components == 1
            and args.mixed_covariance == "full"
            and not args.skip_mixed_full_k1
        ):
            continue
        index.append(
            fit_one_case(
                args,
                f"mixed_k{components}_{args.mixed_covariance}",
                mixed_fit,
                mixed_calibration,
                components,
                args.mixed_covariance,
                reg_covar,
                variants,
                percentiles,
            )
        )
    index_root = args.output_root / "artifacts" / "augmented_gmms"
    atomic_write_json(index_root / "model_index.json", index)
    write_csv(index, index_root / "real_only_model_selection.csv")
    return index


def resolve_bfree_manifest(output_root: Path) -> Path:
    summary_path = output_root / "manifests" / "bfree_manifest_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Run --stage prepare before B-Free evaluation.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    original = Path(str(summary.get("eval_manifest", "")))
    local = output_root / "manifests" / original.name
    if local.is_file():
        return local
    candidates = sorted((output_root / "manifests").glob("bfree_*_eval.csv"))
    if not candidates:
        raise FileNotFoundError("No reusable B-Free eval manifest found.")
    return candidates[0]


def evaluate_one_model(
    args: argparse.Namespace,
    model_entry: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    embeddings: np.ndarray,
) -> Dict[str, object]:
    model_id = str(model_entry["model_id"])
    model_path = Path(str(model_entry["gmm"]))
    if not model_path.is_file():
        model_path = model_case_path(args.output_root, model_id)
    thresholds_path = model_path.parent / "thresholds.json"
    model = load_mixture(model_path)
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    scores = mixture_nll(embeddings, model)
    ids, real_scores, fake_scores = paired_score_arrays(rows, scores)
    labels = np.r_[np.zeros(len(ids), dtype=np.int64), np.ones(len(ids), dtype=np.int64)]
    ranking: List[Dict[str, object]] = []
    pair_scores: List[np.ndarray] = []
    for variant in FAKE_VARIANTS:
        values = np.r_[real_scores, fake_scores[variant]]
        pair_scores.append(values)
        ranking.append(ranking_row(model_id, variant, labels, values))
    pooled_scores = np.concatenate(pair_scores)
    pooled_labels = np.tile(labels, len(FAKE_VARIANTS))
    pooled = ranking_row(model_id, "pooled_balanced", pooled_labels, pooled_scores)
    output_root = args.output_root / "artifacts" / "augmented_bfree_metrics" / model_id
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv([pooled, *ranking], output_root / "ranking.csv")
    threshold_rows = threshold_rows_for_pairs(model_id, real_scores, fake_scores, thresholds)
    write_csv(threshold_rows, output_root / "threshold_metrics.csv")
    alternative_threshold_summaries: Dict[str, Dict[str, float]] = {}
    for calibration_variant, variant_thresholds in thresholds.get("by_variant", {}).items():
        alternative_rows = threshold_rows_for_pairs(
            model_id, real_scores, fake_scores, variant_thresholds
        )
        write_csv(
            alternative_rows,
            output_root
            / f"threshold_metrics_calibrated_{safe_name(str(calibration_variant))}.csv",
        )
        q95_alternative = next(
            row
            for row in alternative_rows
            if row["scope"] == "pooled_balanced"
            and int(row["threshold_percentile"]) == 95
        )
        alternative_threshold_summaries[str(calibration_variant)] = {
            "q95_balanced_accuracy": float(q95_alternative["balanced_accuracy"]),
            "q95_precision": float(q95_alternative["precision"]),
            "q95_recall": float(q95_alternative["recall_tpr"]),
            "q95_specificity": float(q95_alternative["specificity_tnr"]),
        }
    prediction_rows: List[Dict[str, object]] = []
    for row, score in zip(rows, scores):
        output = {
            "sample_id": row["sample_id"],
            "bfree_id": row["bfree_id"],
            "relative_path": row["relative_path"],
            "label": row["label"],
            "variant": row["variant"],
            "model_id": model_id,
            "nll": float(score),
        }
        for percentile, threshold in thresholds["percentiles"].items():
            output[f"pred_q{percentile}"] = int(float(score) > float(threshold))
        prediction_rows.append(output)
    write_csv(prediction_rows, output_root / "predictions.csv")
    confusion_root = args.output_root / "artifacts" / "augmented_bfree_figures" / model_id
    for percentile in (80, 86, 95):
        for row in threshold_rows:
            if int(row["threshold_percentile"]) == percentile:
                save_confusion_figure(
                    dict(row),
                    f"B-Free native | {model_id} | {row['scope']} | q{percentile}",
                    confusion_root / f"q{percentile}" / f"{safe_name(str(row['scope']))}.png",
                )
    bootstrap = bootstrap_confidence_intervals(
        model_id,
        real_scores,
        fake_scores,
        thresholds,
        args.bootstrap_samples,
        args.seed,
        (80, 86, 95),
    )
    atomic_write_json(output_root / "bootstrap_ci95.json", bootstrap)
    macro_auc = float(np.mean([float(row["auroc"]) for row in ranking]))
    macro_ap = float(np.mean([float(row["average_precision"]) for row in ranking]))
    q95 = next(
        row
        for row in threshold_rows
        if row["scope"] == "pooled_balanced" and int(row["threshold_percentile"]) == 95
    )
    summary: Dict[str, object] = {
        "model_id": model_id,
        "components": int(model_entry["components"]),
        "covariance_type": str(model_entry["covariance_type"]),
        "fit_variants": model_entry["variants"],
        "num_paired_ids": len(ids),
        "pooled_balanced_auroc": float(pooled["auroc"]),
        "pooled_balanced_average_precision": float(pooled["average_precision"]),
        "macro_variant_auroc": macro_auc,
        "macro_variant_average_precision": macro_ap,
        "minimum_variant_auroc": float(min(float(row["auroc"]) for row in ranking)),
        "q95_balanced_accuracy": float(q95["balanced_accuracy"]),
        "q95_precision": float(q95["precision"]),
        "q95_recall": float(q95["recall_tpr"]),
        "q95_specificity": float(q95["specificity_tnr"]),
        "threshold_policy": "model_global_real_calibration",
        "alternative_q95_by_calibration_variant": alternative_threshold_summaries,
    }
    atomic_write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_evaluate(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    config: Mapping[str, object],
) -> List[Dict[str, object]]:
    index_path = args.output_root / "artifacts" / "augmented_gmms" / "model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError("Run --stage fit before evaluation.")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    bfree_manifest = resolve_bfree_manifest(args.output_root)
    rows = load_reusable_manifest(bfree_manifest, paths["bfree_root"])
    extractor, device, used_gpus, checkpoint_payload = load_frozen_extractor(args, paths, config)
    per_gpu = args.batch_per_gpu or int(config["evaluation"]["embedding_batch_per_gpu"])
    embeddings = extract_embeddings_resumable(
        rows,
        # Native B-Free as stored: the primary, non-cherry-picked protocol.
        parse_protocol("native_png"),
        extractor,
        device,
        per_gpu * used_gpus,
        args.num_workers,
        int(config["evaluation"]["center_crop"]),
        int(config["model"]["embed_dim"]),
        args.embedding_checkpoint_every,
        args.output_root,
    )
    summaries = [evaluate_one_model(args, entry, rows, embeddings) for entry in index]
    metric_root = args.output_root / "artifacts" / "augmented_bfree_metrics"
    write_csv(summaries, metric_root / "model_comparison.csv")
    atomic_write_json(metric_root / "model_comparison.json", summaries)
    completion = {
        "complete": True,
        "evaluation_protocol": "bfree_valid_clean_native_png",
        "num_models": len(summaries),
        "num_paired_ids": len(rows) // 7,
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_optimizer_step": checkpoint_payload.get("optimizer_step"),
        "warning": "B-Free ranking is final evaluation, not the model-selection criterion.",
    }
    atomic_write_json(args.output_root / "artifacts" / "augmented_evaluation_complete.json", completion)
    print(json.dumps(completion, indent=2), flush=True)
    return summaries


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = resolve_inputs(args)
    variants = parse_variants(args.variants)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "variants": variants,
                "output_root": str(args.output_root),
                **{key: str(value) for key, value in paths.items()},
            },
            indent=2,
        ),
        flush=True,
    )
    if args.stage in ("prepare", "all"):
        prepare(args, paths, config)
    if args.stage in ("extract", "all"):
        run_extract(args, paths, config, variants)
    if args.stage in ("fit", "all"):
        run_fit(args, config, variants)
    if args.stage in ("evaluate", "all"):
        run_evaluate(args, paths, config)


if __name__ == "__main__":
    main()
