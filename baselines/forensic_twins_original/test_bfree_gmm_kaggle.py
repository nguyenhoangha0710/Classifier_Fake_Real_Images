"""Zero-shot Forensic Twins + frozen GMM evaluation on B-Free.

The evaluator deliberately keeps the encoder, FRE, GMM and real-calibrated
thresholds frozen.  It removes every B-Free COCO id that occurs in the 200K
real training manifest, saves reusable B-Free manifests, and evaluates six
paired B-Free generation variants under native and common-degradation
protocols.

Positive class: fake (label=1).  Prediction rule: fake iff NLL > threshold.
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
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
    auroc,
    average_precision,
    classification_metrics,
    gaussian_nll,
    load_extractor,
    load_gmm,
    ranking_curves,
    safe_name,
    save_confusion_figure,
    whitened_pca_2d,
)
from src.checkpointing import sha256_file  # noqa: E402


REAL_VARIANT = "real"
FAKE_VARIANTS: Tuple[str, ...] = (
    "SD2.1_selfconditioned",
    "SD2.1_selfconditioned_origBG",
    "SD2.1_inpainted_samecat",
    "SD2.1_inpainted_samecat_origBG",
    "SD2.1_inpainted_diffcat",
    "SD2.1_inpainted_diffcat_origBG",
)
VARIANT_DIRECTORIES: Mapping[str, str] = {
    REAL_VARIANT: "COCO_real_512",
    **{name: name for name in FAKE_VARIANTS},
}
DEFAULT_PROTOCOLS: Tuple[str, ...] = (
    "native_png",
    "jpeg_q95",
    "jpeg_q80",
    "jpeg_q65",
    "jpeg_q50",
    "resize_075",
    "resize_050",
    "blur_1",
    "blur_2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=BASE_DIR / "configs" / "kaggle_2xt4_200k.json"
    )
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("/kaggle/working/forensic_twins_bfree_eval")
    )
    parser.add_argument("--encoder-dataset-root", type=Path, default=None)
    parser.add_argument("--gmm-dataset-root", type=Path, default=None)
    parser.add_argument("--bfree-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--fre-checkpoint", type=Path, default=None)
    parser.add_argument("--training-manifest", type=Path, default=None)
    parser.add_argument("--gmm", type=Path, default=None)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument(
        "--cohort",
        choices=("valid_clean", "all_clean"),
        default="valid_clean",
        help="Primary held-out cohort. valid_clean is recommended for reporting.",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="Reuse a previously saved row-level B-Free manifest; paths are rebased to --bfree-root.",
    )
    parser.add_argument("--max-ids", type=int, default=None, help="Deterministic smoke-test subset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protocols",
        default=",".join(DEFAULT_PROTOCOLS),
        help="Comma-separated: native_png, jpeg_qXX, resize_075, resize_050, blur_1, blur_2.",
    )
    parser.add_argument("--batch-per-gpu", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--manifest-workers", type=int, default=16)
    parser.add_argument("--embedding-checkpoint-every", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--visualization-max-per-class", type=int, default=500)
    parser.add_argument("--confusion-percentiles", default="80,86,95")
    parser.add_argument("--skip-image-audit", action="store_true")
    parser.add_argument("--single-gpu", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-manifest-only", action="store_true")
    mode.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(rows: Sequence[Mapping[str, object]], path: Path, fieldnames: Optional[Sequence[str]] = None) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty CSV: {path}")
        fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_dataset_root(
    explicit: Optional[Path], input_root: Path, owner: str, slug: str, description: str
) -> Path:
    candidates = [
        explicit,
        input_root / slug,
        input_root / "datasets" / owner / slug,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            print(f"[input] {description}: {candidate}", flush=True)
            return candidate
    raise FileNotFoundError(f"Missing {description}; checked: {[str(x) for x in candidates if x]}")


def resolve_file(explicit: Optional[Path], candidates: Sequence[Path], description: str) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Missing {description}: {explicit}")
        return explicit
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {description}; checked: {[str(x) for x in candidates]}")


def resolve_bfree_subdirectory(root: Path, directory_name: str) -> Path:
    candidates = (root / directory_name / directory_name, root / directory_name)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Missing B-Free directory {directory_name}; checked {candidates}")


def find_bfree_list(root: Path, filename: str) -> Path:
    candidates = (root / filename, root / "lists" / filename)
    return resolve_file(None, candidates, filename)


def coco_ids_from_training_manifest(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("source") != "coco2017":
                continue
            image_id = Path(row["image_path"]).stem
            if image_id.isdigit():
                ids.add(image_id.zfill(12))
    if not ids:
        raise RuntimeError(f"No COCO ids found in training manifest: {path}")
    return ids


def deterministic_subset(values: Sequence[str], maximum: Optional[int], seed: int) -> List[str]:
    ordered = sorted(set(values))
    if maximum is None or maximum >= len(ordered):
        return ordered
    if maximum <= 0:
        raise ValueError("--max-ids must be positive.")
    rng = random.Random(seed)
    return sorted(rng.sample(ordered, maximum))


def inspect_image_header(path: Path, crop_size: int) -> Tuple[bool, str, str, int, int, str]:
    if not path.is_file():
        return False, "missing", "", 0, 0, ""
    try:
        with Image.open(path) as image:
            width, height = image.size
            image_format = str(image.format or "")
            mode = str(image.mode)
            if min(width, height) < crop_size:
                return False, "short_side_below_crop", image_format, width, height, mode
            image.verify()
        return True, "", image_format, width, height, mode
    except Exception as error:
        return False, f"decode_error:{type(error).__name__}", "", 0, 0, ""


def build_bfree_manifests(
    bfree_root: Path,
    training_manifest: Path,
    output_root: Path,
    cohort: str,
    max_ids: Optional[int],
    seed: int,
    crop_size: int,
    workers: int,
    audit_images: bool,
) -> Tuple[Path, Dict[str, object]]:
    """Create reusable id-level and row-level manifests without scanning B-Free recursively."""
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    train_list = read_csv(find_bfree_list(bfree_root, "train_list.csv"))
    valid_list = read_csv(find_bfree_list(bfree_root, "valid_list.csv"))
    if not train_list or not valid_list:
        raise RuntimeError("B-Free train_list.csv or valid_list.csv is empty.")
    train_by_id = {row["id"].zfill(12): row for row in train_list}
    valid_ids = {row["id"].zfill(12) for row in valid_list}
    unknown_valid = valid_ids - set(train_by_id)
    if unknown_valid:
        raise RuntimeError(f"valid_list contains {len(unknown_valid)} ids absent from train_list.")
    overlap_ids = coco_ids_from_training_manifest(training_manifest)

    id_audit: List[Dict[str, object]] = []
    for image_id in sorted(train_by_id):
        source = train_by_id[image_id]
        in_valid = image_id in valid_ids
        overlap = image_id in overlap_ids
        id_audit.append(
            {
                "bfree_id": image_id,
                "same_category": source.get("same_category", ""),
                "diff_category": source.get("diff_category", ""),
                "in_valid_list": int(in_valid),
                "overlap_train_real_200k": int(overlap),
                "eligible_all_clean": int(not overlap),
                "eligible_valid_clean": int(in_valid and not overlap),
            }
        )
    write_csv(id_audit, manifest_root / "bfree_id_audit.csv")

    all_clean_ids = [row["bfree_id"] for row in id_audit if row["eligible_all_clean"]]
    valid_clean_ids = [row["bfree_id"] for row in id_audit if row["eligible_valid_clean"]]
    valid_overlap_ids = [
        row["bfree_id"]
        for row in id_audit
        if row["in_valid_list"] and row["overlap_train_real_200k"]
    ]
    id_columns = ["bfree_id", "same_category", "diff_category"]

    def id_rows(ids: Iterable[str]) -> List[Dict[str, object]]:
        return [
            {
                "bfree_id": image_id,
                "same_category": train_by_id[image_id].get("same_category", ""),
                "diff_category": train_by_id[image_id].get("diff_category", ""),
            }
            for image_id in sorted(ids)
        ]

    write_csv(id_rows(all_clean_ids), manifest_root / "bfree_all_clean_ids.csv", id_columns)
    write_csv(id_rows(valid_clean_ids), manifest_root / "bfree_valid_clean_ids.csv", id_columns)
    write_csv(
        id_rows(valid_overlap_ids),
        manifest_root / "bfree_valid_overlap_excluded.csv",
        id_columns,
    )

    cohort_ids = valid_clean_ids if cohort == "valid_clean" else all_clean_ids
    selected_ids = deterministic_subset(cohort_ids, max_ids, seed)
    directories = {
        variant: resolve_bfree_subdirectory(bfree_root, directory)
        for variant, directory in VARIANT_DIRECTORIES.items()
    }

    def inspect_group(image_id: str) -> Tuple[str, List[Dict[str, object]], List[str]]:
        group_rows: List[Dict[str, object]] = []
        errors: List[str] = []
        for variant in (REAL_VARIANT, *FAKE_VARIANTS):
            path = directories[variant] / f"{image_id}.png"
            if audit_images:
                valid, reason, image_format, width, height, mode = inspect_image_header(path, crop_size)
            else:
                valid = path.is_file()
                reason = "" if valid else "missing"
                image_format, width, height, mode = "", 0, 0, ""
            if not valid:
                errors.append(f"{variant}:{reason}")
            relative_path = path.relative_to(bfree_root).as_posix()
            group_rows.append(
                {
                    "sample_id": f"bfree:{image_id}:{variant}",
                    "bfree_id": image_id,
                    "relative_path": relative_path,
                    "image_path": str(path),
                    "label": 0 if variant == REAL_VARIANT else 1,
                    "variant": variant,
                    "generator": variant,
                    "source": "bfree_coco" if variant == REAL_VARIANT else "bfree_sd21",
                    "cohort": cohort,
                    "format": image_format,
                    "width": width,
                    "height": height,
                    "mode": mode,
                }
            )
        return image_id, group_rows, errors

    results: List[Tuple[str, List[Dict[str, object]], List[str]]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, result in enumerate(executor.map(inspect_group, selected_ids), start=1):
            results.append(result)
            if index % 250 == 0 or index == len(selected_ids):
                print(f"[manifest] audited paired ids: {index:,}/{len(selected_ids):,}", flush=True)

    eval_rows: List[Dict[str, object]] = []
    invalid_rows: List[Dict[str, object]] = []
    valid_group_ids: List[str] = []
    for image_id, group_rows, errors in results:
        if errors:
            invalid_rows.append({"bfree_id": image_id, "reasons": "|".join(errors)})
            continue
        valid_group_ids.append(image_id)
        eval_rows.extend(group_rows)
    if not eval_rows:
        raise RuntimeError("No complete B-Free real+six-fake groups survived manifest audit.")

    suffix = f"_{len(selected_ids)}ids" if max_ids is not None else ""
    eval_manifest = manifest_root / f"bfree_{cohort}{suffix}_eval.csv"
    write_csv(eval_rows, eval_manifest)
    write_csv(
        invalid_rows,
        manifest_root / f"bfree_{cohort}{suffix}_invalid_groups.csv",
        ("bfree_id", "reasons"),
    )
    write_csv(
        id_rows(valid_group_ids),
        manifest_root / f"bfree_{cohort}{suffix}_evaluated_ids.csv",
        id_columns,
    )
    summary: Dict[str, object] = {
        "source": "baopv051/b-free-training-data",
        "cohort": cohort,
        "seed": seed,
        "max_ids": max_ids,
        "crop_size": crop_size,
        "image_audit": audit_images,
        "train_list_ids": len(train_by_id),
        "valid_list_ids": len(valid_ids),
        "training_manifest_coco_ids": len(overlap_ids),
        "all_overlap_ids": len(set(train_by_id) & overlap_ids),
        "all_clean_ids": len(all_clean_ids),
        "valid_overlap_ids": len(valid_overlap_ids),
        "valid_clean_ids": len(valid_clean_ids),
        "selected_ids_before_file_audit": len(selected_ids),
        "complete_evaluated_ids": len(valid_group_ids),
        "invalid_or_incomplete_ids": len(invalid_rows),
        "row_count": len(eval_rows),
        "real_rows": sum(int(row["label"]) == 0 for row in eval_rows),
        "fake_rows": sum(int(row["label"]) == 1 for row in eval_rows),
        "variants": list(FAKE_VARIANTS),
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": sha256_file(eval_manifest),
        "training_manifest": str(training_manifest),
        "training_manifest_sha256": sha256_file(training_manifest),
    }
    atomic_write_json(manifest_root / "bfree_manifest_summary.json", summary)
    atomic_write_json(manifest_root / "manifest_ready.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return eval_manifest, summary


def load_reusable_manifest(path: Path, bfree_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in read_csv(path):
        relative = row.get("relative_path", "")
        if not relative:
            raise ValueError(f"Reusable manifest lacks relative_path: {path}")
        rebased = bfree_root / Path(relative)
        parsed: Dict[str, object] = dict(row)
        parsed["image_path"] = str(rebased)
        parsed["label"] = int(row["label"])
        parsed["width"] = int(row.get("width") or 0)
        parsed["height"] = int(row.get("height") or 0)
        rows.append(parsed)
    validate_paired_rows(rows)
    return rows


def validate_paired_rows(rows: Sequence[Mapping[str, object]]) -> None:
    grouped: Dict[str, set[str]] = {}
    for row in rows:
        image_id = str(row["bfree_id"])
        grouped.setdefault(image_id, set()).add(str(row["variant"]))
    expected = {REAL_VARIANT, *FAKE_VARIANTS}
    broken = [image_id for image_id, variants in grouped.items() if variants != expected]
    if broken:
        raise ValueError(f"Manifest has {len(broken)} incomplete paired ids; first={broken[:5]}")
    if len(rows) != len(grouped) * len(expected):
        raise ValueError("Manifest has duplicated B-Free id/variant rows.")


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    jpeg_quality: Optional[int] = None
    resize_scale: Optional[float] = None
    blur_sigma: Optional[float] = None


def parse_protocol(name: str) -> ProtocolSpec:
    value = name.strip().lower()
    if value == "native_png":
        return ProtocolSpec(name=value)
    jpeg = re.fullmatch(r"jpeg_q(\d{1,3})", value)
    if jpeg:
        quality = int(jpeg.group(1))
        if not 1 <= quality <= 100:
            raise ValueError(f"Invalid JPEG quality in {name}")
        return ProtocolSpec(name=value, jpeg_quality=quality)
    resize = re.fullmatch(r"resize_(\d{2,3})", value)
    if resize:
        scale = int(resize.group(1)) / 100.0
        if not 0.1 <= scale < 1.0:
            raise ValueError(f"Invalid resize scale in {name}")
        return ProtocolSpec(name=value, resize_scale=scale)
    blur = re.fullmatch(r"blur_(\d+(?:[._]\d+)?)", value)
    if blur:
        sigma = float(blur.group(1).replace("_", "."))
        if sigma <= 0:
            raise ValueError(f"Invalid blur sigma in {name}")
        return ProtocolSpec(name=value, blur_sigma=sigma)
    raise ValueError(f"Unsupported protocol: {name}")


class BFreeProtocolDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, object]], spec: ProtocolSpec, crop_size: int):
        self.rows = list(rows)
        self.spec = spec
        self.crop_size = int(crop_size)

    def __len__(self) -> int:
        return len(self.rows)

    def transform(self, image: Image.Image) -> Image.Image:
        if self.spec.jpeg_quality is not None:
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=self.spec.jpeg_quality,
                subsampling=2,
                optimize=False,
                progressive=False,
            )
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                encoded.load()
                image = encoded.convert("RGB")
        if self.spec.resize_scale is not None:
            original = image.size
            reduced = (
                max(1, int(round(original[0] * self.spec.resize_scale))),
                max(1, int(round(original[1] * self.spec.resize_scale))),
            )
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image = image.resize(reduced, resampling).resize(original, resampling)
        if self.spec.blur_sigma is not None:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.spec.blur_sigma))
        return image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = Path(str(row["image_path"]))
        with Image.open(path) as source:
            source.load()
            image = source.convert("RGB")
        image = self.transform(image)
        if min(image.size) < self.crop_size:
            raise ValueError(f"Image shorter side below {self.crop_size}: {path}")
        left = (image.width - self.crop_size) // 2
        top = (image.height - self.crop_size) // 2
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size)).convert("L")
        gray = torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).unsqueeze(0) / 255.0
        gray = F.pad(gray, (7, 7, 7, 7), mode="constant", value=0.0)
        return gray, index


def manifest_identity(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["sample_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["relative_path"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def extract_embeddings_resumable(
    rows: Sequence[Mapping[str, object]],
    spec: ProtocolSpec,
    extractor: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    crop_size: int,
    embed_dim: int,
    checkpoint_every: int,
    output_root: Path,
) -> np.ndarray:
    embedding_root = output_root / "artifacts" / "embeddings"
    embedding_root.mkdir(parents=True, exist_ok=True)
    embedding_path = embedding_root / f"{spec.name}_cls{embed_dim}.npy"
    progress_path = embedding_root / f"{spec.name}_progress.json"
    identity = manifest_identity(rows)
    expected = {
        "protocol": spec.name,
        "manifest_identity": identity,
        "row_count": len(rows),
        "embedding_dim": embed_dim,
    }
    start = 0
    if progress_path.is_file() and embedding_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if progress.get(key) != value:
                raise RuntimeError(f"Embedding resume mismatch for {spec.name}: {key}")
        start = int(progress.get("next_index", 0))
        if bool(progress.get("complete")) and start == len(rows):
            print(f"[embedding] reuse complete {spec.name}: {embedding_path}", flush=True)
            return np.load(embedding_path, mmap_mode="r")
        matrix = np.lib.format.open_memmap(embedding_path, mode="r+")
        if matrix.shape != (len(rows), embed_dim):
            raise RuntimeError(f"Embedding shape mismatch: {matrix.shape}")
        print(f"[embedding] resume {spec.name} at {start:,}/{len(rows):,}", flush=True)
    else:
        matrix = np.lib.format.open_memmap(
            embedding_path, mode="w+", dtype=np.float32, shape=(len(rows), embed_dim)
        )
        atomic_write_json(progress_path, {**expected, "next_index": 0, "complete": False})

    if start < 0 or start > len(rows):
        raise RuntimeError(f"Invalid resume index {start} for {len(rows)} rows")
    remaining = list(rows[start:])
    dataset = BFreeProtocolDataset(remaining, spec, crop_size)
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
                embedding = extractor(gray)
            values = embedding.float().cpu().numpy()
            matrix[cursor : cursor + len(values)] = values
            cursor += len(values)
            if cursor - last_saved >= checkpoint_every or cursor == len(rows):
                matrix.flush()
                atomic_write_json(
                    progress_path,
                    {**expected, "next_index": cursor, "complete": cursor == len(rows)},
                )
                print(f"[embedding] {spec.name}: {cursor:,}/{len(rows):,}", flush=True)
                last_saved = cursor
    matrix.flush()
    del matrix
    if cursor != len(rows):
        raise RuntimeError(f"Embedding extraction stopped at {cursor}/{len(rows)}")
    return np.load(embedding_path, mmap_mode="r")


def paired_score_arrays(
    rows: Sequence[Mapping[str, object]], scores: np.ndarray
) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
    real: Dict[str, float] = {}
    fake: Dict[str, Dict[str, float]] = {variant: {} for variant in FAKE_VARIANTS}
    for row, score in zip(rows, scores):
        image_id = str(row["bfree_id"])
        variant = str(row["variant"])
        if variant == REAL_VARIANT:
            real[image_id] = float(score)
        else:
            fake[variant][image_id] = float(score)
    common = sorted(real)
    for variant in FAKE_VARIANTS:
        if set(fake[variant]) != set(common):
            raise RuntimeError(f"Pair mismatch for {variant}")
    real_scores = np.asarray([real[image_id] for image_id in common], dtype=np.float64)
    fake_scores = {
        variant: np.asarray([fake[variant][image_id] for image_id in common], dtype=np.float64)
        for variant in FAKE_VARIANTS
    }
    return common, real_scores, fake_scores


def ranking_row(protocol: str, scope: str, labels: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    return {
        "protocol": protocol,
        "scope": scope,
        "num_images": int(len(labels)),
        "support_real": int(np.sum(labels == 0)),
        "support_fake": int(np.sum(labels == 1)),
        "auroc": auroc(labels, scores),
        "average_precision": average_precision(labels, scores),
    }


def percentile_values(thresholds: Mapping[str, object]) -> Dict[str, float]:
    values = thresholds.get("percentiles")
    if not isinstance(values, Mapping):
        raise ValueError("thresholds.json lacks a percentiles mapping")
    parsed = {str(int(key)): float(value) for key, value in values.items()}
    if not parsed:
        raise ValueError("thresholds.json contains no thresholds")
    return parsed


def threshold_rows_for_pairs(
    protocol: str,
    real_scores: np.ndarray,
    fake_scores: Mapping[str, np.ndarray],
    thresholds: Mapping[str, object],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    pairs = []
    for variant in FAKE_VARIANTS:
        labels = np.r_[np.zeros(len(real_scores), dtype=np.int64), np.ones(len(real_scores), dtype=np.int64)]
        scores = np.r_[real_scores, fake_scores[variant]]
        pairs.append(scores)
        for percentile, threshold in percentile_values(thresholds).items():
            output.append(
                {
                    "protocol": protocol,
                    "scope": variant,
                    "threshold_percentile": int(percentile),
                    "threshold_nll": threshold,
                    **classification_metrics(labels, scores, threshold),
                }
            )
    pooled_scores = np.concatenate(pairs)
    pooled_labels = np.tile(
        np.r_[np.zeros(len(real_scores), dtype=np.int64), np.ones(len(real_scores), dtype=np.int64)],
        len(FAKE_VARIANTS),
    )
    # np.concatenate(pairs) is [real,fake] per variant, matching tiled labels.
    for percentile, threshold in percentile_values(thresholds).items():
        output.insert(
            0,
            {
                "protocol": protocol,
                "scope": "pooled_balanced",
                "threshold_percentile": int(percentile),
                "threshold_nll": threshold,
                **classification_metrics(pooled_labels, pooled_scores, threshold),
            },
        )
    return output


def bootstrap_confidence_intervals(
    protocol: str,
    real_scores: np.ndarray,
    fake_scores: Mapping[str, np.ndarray],
    thresholds: Mapping[str, object],
    samples: int,
    seed: int,
    selected_percentiles: Sequence[int],
) -> Dict[str, object]:
    if samples <= 0:
        return {"protocol": protocol, "bootstrap_samples": 0}
    n = len(real_scores)
    rng = np.random.default_rng(seed)
    tracked: Dict[str, List[float]] = {"macro_auroc": [], "macro_average_precision": []}
    threshold_map = percentile_values(thresholds)
    for percentile in selected_percentiles:
        if str(percentile) in threshold_map:
            tracked[f"balanced_accuracy_q{percentile}"] = []
            tracked[f"recall_q{percentile}"] = []
            tracked[f"specificity_q{percentile}"] = []
    base_labels = np.r_[np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)]
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        aucs: List[float] = []
        aps: List[float] = []
        pooled_parts: List[np.ndarray] = []
        for variant in FAKE_VARIANTS:
            pair_scores = np.r_[real_scores[indices], fake_scores[variant][indices]]
            pooled_parts.append(pair_scores)
            aucs.append(auroc(base_labels, pair_scores))
            aps.append(average_precision(base_labels, pair_scores))
        tracked["macro_auroc"].append(float(np.mean(aucs)))
        tracked["macro_average_precision"].append(float(np.mean(aps)))
        pooled_scores = np.concatenate(pooled_parts)
        pooled_labels = np.tile(base_labels, len(FAKE_VARIANTS))
        for percentile in selected_percentiles:
            threshold = threshold_map.get(str(percentile))
            if threshold is None:
                continue
            metrics = classification_metrics(pooled_labels, pooled_scores, threshold)
            tracked[f"balanced_accuracy_q{percentile}"].append(float(metrics["balanced_accuracy"]))
            tracked[f"recall_q{percentile}"].append(float(metrics["recall_tpr"]))
            tracked[f"specificity_q{percentile}"].append(float(metrics["specificity_tnr"]))
    result: Dict[str, object] = {
        "protocol": protocol,
        "bootstrap_samples": samples,
        "resampling_unit": "paired_bfree_id_with_all_six_fake_variants",
    }
    for name, values in tracked.items():
        array = np.asarray(values, dtype=np.float64)
        result[name] = {
            "mean": float(array.mean()),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
        }
    return result


def save_protocol_figures(
    protocol: str,
    ids: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    embeddings: np.ndarray,
    scores: np.ndarray,
    gmm: Mapping[str, np.ndarray],
    real_scores: np.ndarray,
    fake_scores: Mapping[str, np.ndarray],
    thresholds: Mapping[str, object],
    threshold_rows: Sequence[Mapping[str, object]],
    confusion_percentiles: Sequence[int],
    maximum_per_class: int,
    seed: int,
    output_root: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_root = output_root / "artifacts" / "figures" / protocol
    figure_root.mkdir(parents=True, exist_ok=True)
    threshold_map = percentile_values(thresholds)
    for percentile in confusion_percentiles:
        if str(percentile) not in threshold_map:
            continue
        for row in threshold_rows:
            if int(row["threshold_percentile"]) != percentile:
                continue
            save_confusion_figure(
                dict(row),
                f"B-Free | {protocol} | {row['scope']} | q{percentile}",
                figure_root / "confusion_matrices" / f"q{percentile}" / f"{safe_name(str(row['scope']))}.png",
            )

    columns = 3
    plot_rows = int(math.ceil(len(FAKE_VARIANTS) / columns))
    figure, axes = plt.subplots(plot_rows, columns, figsize=(5.2 * columns, 3.8 * plot_rows), squeeze=False)
    for axis, variant in zip(axes.flat, FAKE_VARIANTS):
        axis.hist(real_scores, bins=40, alpha=0.55, label="real", color="#2ca02c")
        axis.hist(fake_scores[variant], bins=40, alpha=0.55, label="fake", color="#d62728")
        for percentile, style in ((80, ":"), (95, "--")):
            if str(percentile) in threshold_map:
                axis.axvline(threshold_map[str(percentile)], color="black", linestyle=style, label=f"q{percentile}")
        axis.set_title(variant)
        axis.set_xlabel("GMM NLL")
        axis.set_ylabel("count")
        axis.legend(fontsize=7)
    figure.suptitle(f"B-Free paired NLL distributions ({protocol})", y=1.01)
    figure.tight_layout()
    figure.savefig(figure_root / "nll_distributions_by_variant.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    labels = np.r_[np.zeros(len(ids), dtype=np.int64), np.ones(len(ids), dtype=np.int64)]
    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(12, 5))
    for variant in FAKE_VARIANTS:
        pair_scores = np.r_[real_scores, fake_scores[variant]]
        fpr, tpr, recall, precision = ranking_curves(labels, pair_scores)
        roc_axis.plot(fpr, tpr, label=f"{variant} ({auroc(labels, pair_scores):.3f})")
        pr_axis.plot(recall, precision, label=f"{variant} ({average_precision(labels, pair_scores):.3f})")
    roc_axis.plot([0, 1], [0, 1], "--", color="gray")
    roc_axis.set(xlabel="FPR", ylabel="TPR", title="ROC by B-Free variant")
    pr_axis.set(xlabel="Recall", ylabel="Precision", title="PR by B-Free variant")
    for axis in (roc_axis, pr_axis):
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(figure_root / "roc_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    variant_indices: Dict[str, List[int]] = {variant: [] for variant in (REAL_VARIANT, *FAKE_VARIANTS)}
    for index, row in enumerate(rows):
        variant_indices[str(row["variant"])].append(index)
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for variant, indices in variant_indices.items():
        values = np.asarray(indices, dtype=np.int64)
        if len(values) > maximum_per_class:
            values = np.sort(rng.choice(values, maximum_per_class, replace=False))
        selected.extend(values.tolist())
    chosen = np.asarray(sorted(selected), dtype=np.int64)
    coordinates = whitened_pca_2d(np.asarray(embeddings[chosen]), dict(gmm))
    shown_variants = np.asarray([str(rows[index]["variant"]) for index in chosen])
    figure, axis = plt.subplots(figsize=(11, 8))
    colors = plt.get_cmap("tab10", len(variant_indices))
    for color_index, variant in enumerate((REAL_VARIANT, *FAKE_VARIANTS)):
        mask = shown_variants == variant
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=16,
            alpha=0.55,
            color=colors(color_index),
            label=variant,
        )
    axis.set_title(f"GMM-whitened B-Free embeddings ({protocol})")
    axis.set_xlabel("PCA 1")
    axis.set_ylabel("PCA 2")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_root / "pca_real_and_six_variants.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_protocol(
    spec: ProtocolSpec,
    rows: Sequence[Mapping[str, object]],
    embeddings: np.ndarray,
    gmm: Mapping[str, np.ndarray],
    thresholds: Mapping[str, object],
    output_root: Path,
    bootstrap_samples: int,
    confusion_percentiles: Sequence[int],
    visualization_max_per_class: int,
    seed: int,
) -> Dict[str, object]:
    scores = gaussian_nll(np.asarray(embeddings), dict(gmm))
    ids, real_scores, fake_scores = paired_score_arrays(rows, scores)
    base_labels = np.r_[np.zeros(len(ids), dtype=np.int64), np.ones(len(ids), dtype=np.int64)]
    ranking: List[Dict[str, object]] = []
    pair_score_parts: List[np.ndarray] = []
    for variant in FAKE_VARIANTS:
        pair_scores = np.r_[real_scores, fake_scores[variant]]
        pair_score_parts.append(pair_scores)
        ranking.append(ranking_row(spec.name, variant, base_labels, pair_scores))
    pooled_scores = np.concatenate(pair_score_parts)
    pooled_labels = np.tile(base_labels, len(FAKE_VARIANTS))
    pooled = ranking_row(spec.name, "pooled_balanced", pooled_labels, pooled_scores)
    macro_auc = float(np.mean([float(row["auroc"]) for row in ranking]))
    macro_ap = float(np.mean([float(row["average_precision"]) for row in ranking]))
    summary: Dict[str, object] = {
        "protocol": spec.name,
        "num_paired_ids": len(ids),
        "num_variants": len(FAKE_VARIANTS),
        "unique_images": len(rows),
        "balanced_evaluation_rows": int(len(pooled_labels)),
        "pooled_balanced_auroc": pooled["auroc"],
        "pooled_balanced_average_precision": pooled["average_precision"],
        "macro_variant_auroc": macro_auc,
        "macro_variant_average_precision": macro_ap,
        "minimum_variant_auroc": float(min(float(row["auroc"]) for row in ranking)),
        "maximum_variant_auroc": float(max(float(row["auroc"]) for row in ranking)),
        "per_variant": ranking,
    }
    metric_root = output_root / "artifacts" / "metrics" / spec.name
    metric_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(metric_root / "summary.json", summary)
    write_csv([pooled, *ranking], metric_root / "ranking.csv")
    threshold_rows = threshold_rows_for_pairs(spec.name, real_scores, fake_scores, thresholds)
    write_csv(threshold_rows, metric_root / "threshold_metrics.csv")

    prediction_rows: List[Dict[str, object]] = []
    threshold_map = percentile_values(thresholds)
    for row, score in zip(rows, scores):
        output = {
            "sample_id": row["sample_id"],
            "bfree_id": row["bfree_id"],
            "relative_path": row["relative_path"],
            "label": row["label"],
            "variant": row["variant"],
            "protocol": spec.name,
            "nll": float(score),
        }
        for percentile, threshold in threshold_map.items():
            output[f"pred_q{percentile}"] = int(float(score) > threshold)
        prediction_rows.append(output)
    write_csv(prediction_rows, metric_root / "predictions.csv")

    bootstrap = bootstrap_confidence_intervals(
        spec.name,
        real_scores,
        fake_scores,
        thresholds,
        bootstrap_samples,
        seed,
        confusion_percentiles,
    )
    atomic_write_json(metric_root / "bootstrap_ci95.json", bootstrap)
    save_protocol_figures(
        spec.name,
        ids,
        rows,
        embeddings,
        scores,
        gmm,
        real_scores,
        fake_scores,
        thresholds,
        threshold_rows,
        confusion_percentiles,
        visualization_max_per_class,
        seed,
        output_root,
    )
    atomic_write_json(
        output_root / "artifacts" / "protocol_markers" / f"{spec.name}.json",
        {"complete": True, **summary},
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def save_cross_protocol_summary(summaries: Sequence[Mapping[str, object]], output_root: Path) -> None:
    rows = [
        {
            "protocol": summary["protocol"],
            "num_paired_ids": summary["num_paired_ids"],
            "pooled_balanced_auroc": summary["pooled_balanced_auroc"],
            "pooled_balanced_average_precision": summary["pooled_balanced_average_precision"],
            "macro_variant_auroc": summary["macro_variant_auroc"],
            "macro_variant_average_precision": summary["macro_variant_average_precision"],
            "minimum_variant_auroc": summary["minimum_variant_auroc"],
            "maximum_variant_auroc": summary["maximum_variant_auroc"],
        }
        for summary in summaries
    ]
    metric_root = output_root / "artifacts" / "metrics"
    write_csv(rows, metric_root / "protocol_comparison.csv")
    atomic_write_json(metric_root / "protocol_comparison.json", rows)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(max(8, len(rows) * 1.2), 5))
        positions = np.arange(len(rows))
        axis.plot(positions, [float(row["macro_variant_auroc"]) for row in rows], "o-", label="Macro AUROC")
        axis.plot(
            positions,
            [float(row["minimum_variant_auroc"]) for row in rows],
            "s--",
            label="Worst-variant AUROC",
        )
        axis.set_xticks(positions, [str(row["protocol"]) for row in rows], rotation=35, ha="right")
        axis.set_ylim(0.0, 1.02)
        axis.set_ylabel("AUROC")
        axis.set_title("B-Free robustness across common degradations")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure_root = output_root / "artifacts" / "figures"
        figure_root.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_root / "protocol_auroc_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    except Exception as error:
        print(f"[warning] cross-protocol figure failed: {error}", flush=True)


def completed_protocol_summaries(
    output_root: Path, preferred_order: Sequence[str]
) -> List[Dict[str, object]]:
    metric_root = output_root / "artifacts" / "metrics"
    discovered: Dict[str, Dict[str, object]] = {}
    if metric_root.is_dir():
        for path in metric_root.glob("*/summary.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            protocol = str(payload.get("protocol", path.parent.name))
            discovered[protocol] = payload
    ordered_names = list(dict.fromkeys([*preferred_order, *sorted(discovered)]))
    return [discovered[name] for name in ordered_names if name in discovered]


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    encoder_dataset = resolve_dataset_root(
        args.encoder_dataset_root,
        args.input_root,
        "hoangha071005",
        "twins-gmm-200k-checkpoint",
        "encoder checkpoint dataset",
    )
    gmm_dataset = resolve_dataset_root(
        args.gmm_dataset_root,
        args.input_root,
        "hoangha071005",
        "fit-gmm-checkpoint-200k",
        "fitted GMM dataset",
    )
    bfree_root = resolve_dataset_root(
        args.bfree_root,
        args.input_root,
        "baopv051",
        "b-free-training-data",
        "B-Free dataset",
    )
    encoder_artifact_root = encoder_dataset / "forensic_twins_v2_200k"
    gmm_artifact_root = gmm_dataset / "forensic_twins_200k_gmm_eval"
    checkpoint = resolve_file(
        args.checkpoint,
        (encoder_artifact_root / "artifacts" / "checkpoints" / "last.ckpt",),
        "last.ckpt",
    )
    fre_checkpoint = resolve_file(
        args.fre_checkpoint,
        (encoder_artifact_root / "cache" / "fre.pt",),
        "fre.pt",
    )
    training_manifest = resolve_file(
        args.training_manifest,
        (encoder_artifact_root / "manifests" / "train_real_200k.csv",),
        "train_real_200k.csv",
    )
    gmm_path = resolve_file(
        args.gmm,
        (gmm_artifact_root / "artifacts" / "gmm" / "real_k1_full.npz",),
        "fitted GMM",
    )
    thresholds_path = resolve_file(
        args.thresholds,
        (gmm_artifact_root / "artifacts" / "gmm" / "thresholds.json",),
        "thresholds.json",
    )
    crop_size = int(config["evaluation"]["center_crop"])
    if args.eval_only:
        if args.eval_manifest is None:
            raise ValueError("--eval-only requires --eval-manifest")
        eval_manifest = args.eval_manifest
        manifest_summary: Dict[str, object] = {"reused_manifest": str(eval_manifest)}
    else:
        eval_manifest, manifest_summary = build_bfree_manifests(
            bfree_root,
            training_manifest,
            args.output_root,
            args.cohort,
            args.max_ids,
            args.seed,
            crop_size,
            args.manifest_workers,
            not args.skip_image_audit,
        )
    if args.build_manifest_only:
        print(f"[complete] manifest ready: {eval_manifest}", flush=True)
        return

    rows = load_reusable_manifest(eval_manifest, bfree_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B-Free embedding extraction.")
    device = torch.device("cuda:0")
    extractor, checkpoint_payload = load_extractor(config, checkpoint, fre_checkpoint, device)
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1 and not args.single_gpu:
        extractor = torch.nn.DataParallel(extractor, device_ids=list(range(gpu_count)))
        print(f"[gpu] DataParallel enabled on {gpu_count} GPUs", flush=True)
    else:
        print(f"[gpu] using {device}; visible GPUs={gpu_count}", flush=True)
    per_gpu = args.batch_per_gpu or int(config["evaluation"]["embedding_batch_per_gpu"])
    batch_size = per_gpu * (gpu_count if gpu_count > 1 and not args.single_gpu else 1)
    gmm = load_gmm(gmm_path)
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    embed_dim = int(config["model"]["embed_dim"])
    if np.asarray(gmm["mean"]).shape != (embed_dim,):
        raise RuntimeError(f"GMM dimension {np.asarray(gmm['mean']).shape} != {embed_dim}")
    confusion_percentiles = sorted(
        {int(value.strip()) for value in args.confusion_percentiles.split(",") if value.strip()}
    )
    protocol_specs = [
        parse_protocol(value) for value in args.protocols.split(",") if value.strip()
    ]
    if not protocol_specs:
        raise ValueError("No evaluation protocols selected.")
    provenance = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_optimizer_step": checkpoint_payload.get("optimizer_step"),
        "fre_checkpoint": str(fre_checkpoint),
        "fre_sha256": sha256_file(fre_checkpoint),
        "gmm": str(gmm_path),
        "gmm_sha256": sha256_file(gmm_path),
        "thresholds": str(thresholds_path),
        "thresholds_sha256": sha256_file(thresholds_path),
        "training_manifest": str(training_manifest),
        "training_manifest_sha256": sha256_file(training_manifest),
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": sha256_file(eval_manifest),
        "manifest_identity": manifest_identity(rows),
        "manifest_summary": manifest_summary,
        "protocols": [spec.name for spec in protocol_specs],
        "batch_size": batch_size,
        "visible_gpus": gpu_count,
        "gmm_components": 1,
        "gmm_covariance": "full",
        "embedding_dim": embed_dim,
        "crop_size": crop_size,
        "prediction_rule": "fake_if_nll_gt_threshold",
        "positive_class": "fake",
    }
    atomic_write_json(args.output_root / "artifacts" / "run_provenance.json", provenance)

    summaries: List[Dict[str, object]] = []
    for spec in protocol_specs:
        print(f"[protocol] {spec.name}: {len(rows):,} unique images", flush=True)
        embeddings = extract_embeddings_resumable(
            rows,
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
        summaries.append(
            evaluate_protocol(
                spec,
                rows,
                embeddings,
                gmm,
                thresholds,
                args.output_root,
                args.bootstrap_samples,
                confusion_percentiles,
                args.visualization_max_per_class,
                args.seed,
            )
        )
    all_summaries = completed_protocol_summaries(
        args.output_root, [spec.name for spec in protocol_specs]
    )
    save_cross_protocol_summary(all_summaries, args.output_root)
    completed_protocols = [str(summary["protocol"]) for summary in all_summaries]
    requested_protocols = [spec.name for spec in protocol_specs]
    complete = {
        "complete": all(protocol in completed_protocols for protocol in requested_protocols),
        "requested_protocols": requested_protocols,
        "completed_protocols": completed_protocols,
        "eval_manifest": str(eval_manifest),
        "num_unique_images": len(rows),
        "num_paired_ids": len(rows) // 7,
    }
    atomic_write_json(args.output_root / "artifacts" / "evaluation_complete.json", complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == "__main__":
    main()
