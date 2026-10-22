"""Dataset discovery, real-only manifests and forensic-preserving view creation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, JpegImagePlugin
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MANIFEST_FIELDS = [
    "sample_id",
    "image_path",
    "source",
    "original_split",
    "class_id",
    "label",
    "width",
    "height",
    "magic_format",
    "mode",
    "file_size",
    "sha256",
    "jpeg_qtable_hash",
    "jpeg_estimated_quality",
    "jpeg_subsampling",
]


def real_manifest_filename(imagenet_per_class: int, coco_total: int) -> str:
    """Return an unambiguous manifest name for the requested real cohort."""
    total = 1000 * int(imagenet_per_class) + int(coco_total)
    if total <= 0:
        raise ValueError(f"Manifest size must be positive, got {total}")
    size_label = f"{total // 1000}k" if total % 1000 == 0 else str(total)
    return f"train_real_{size_label}.csv"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_hash(path: Path) -> str:
    return sha256_file(path)


def _estimate_jpeg_quality(qtables: Optional[Dict[int, Sequence[int]]]) -> Optional[int]:
    """Coarse audit-only estimate; quantization hash remains the source of truth."""
    if not qtables:
        return None
    values = np.concatenate([np.asarray(table, dtype=np.float32) for table in qtables.values()])
    mean_q = float(values.mean())
    # Monotonic approximation, deliberately not presented as exact encoder quality.
    return int(np.clip(round(100.0 - 1.6 * math.sqrt(max(mean_q - 1.0, 0.0))), 1, 100))


def inspect_image(path: Path, source: str, split: str, class_id: str = "") -> Optional[Dict[str, object]]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            magic_format = str(image.format or "UNKNOWN").upper()
            width, height = image.size
            mode = image.mode
            qtables = getattr(image, "quantization", None)
            if qtables:
                normalized_qtables = {str(key): list(value) for key, value in sorted(qtables.items())}
                qtable_hash = sha256_json(normalized_qtables)
                try:
                    subsampling = int(JpegImagePlugin.get_sampling(image))
                except Exception:
                    subsampling = -1
                estimated_quality = _estimate_jpeg_quality(qtables)
            else:
                qtable_hash = ""
                subsampling = -1
                estimated_quality = None
        digest = sha256_file(path)
        return {
            "sample_id": f"{source}:{split}:{digest[:20]}",
            "image_path": str(path),
            "source": source,
            "original_split": split,
            "class_id": class_id,
            "label": 0,
            "width": width,
            "height": height,
            "magic_format": magic_format,
            "mode": mode,
            "file_size": path.stat().st_size,
            "sha256": digest,
            "jpeg_qtable_hash": qtable_hash,
            "jpeg_estimated_quality": "" if estimated_quality is None else estimated_quality,
            "jpeg_subsampling": subsampling,
        }
    except Exception as error:
        print(f"[manifest] skip unreadable image {path}: {error}", flush=True)
        return None


def _images_below(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def discover_imagenet_train(input_root: Path) -> Path:
    candidates = [
        input_root / "imagenet-object-localization-challenge" / "ILSVRC" / "Data" / "CLS-LOC" / "train",
        input_root / "competitions" / "imagenet-object-localization-challenge" / "ILSVRC" / "Data" / "CLS-LOC" / "train",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    matches = list(input_root.glob("**/ILSVRC/Data/CLS-LOC/train"))
    if not matches:
        raise FileNotFoundError("Could not find ILSVRC/Data/CLS-LOC/train under Kaggle input.")
    return matches[0]


def discover_imagenet_train_index(input_root: Path, imagenet_train: Path) -> Path:
    """Locate the official train index without recursively walking ImageNet."""
    candidates = [
        input_root / "imagenet-object-localization-challenge" / "LOC_train_solution.csv",
        input_root / "competitions" / "imagenet-object-localization-challenge" / "LOC_train_solution.csv",
    ]
    # .../ILSVRC/Data/CLS-LOC/train -> competition root is parents[3].
    if len(imagenet_train.parents) > 3:
        candidates.append(imagenet_train.parents[3] / "LOC_train_solution.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find LOC_train_solution.csv beside the ImageNet competition input."
    )


def discover_coco_roots(input_root: Path) -> Dict[str, Path]:
    bases = [
        input_root / "coco-2017-dataset" / "coco2017",
        input_root / "datasets" / "awsaf49" / "coco-2017-dataset" / "coco2017",
        input_root / "coco2017",
    ]
    for base in bases:
        train = base / "train2017"
        if train.is_dir():
            result = {"train2017": train}
            unlabeled = base / "unlabeled2017"
            if unlabeled.is_dir():
                result["unlabeled2017"] = unlabeled
            return result
    # Shallow fallback only: never recursively walk the mounted ImageNet tree.
    for dataset_root in input_root.iterdir():
        if not dataset_root.is_dir():
            continue
        for base in (dataset_root, dataset_root / "coco2017"):
            train = base / "train2017"
            if train.is_dir():
                result = {"train2017": train}
                unlabeled = base / "unlabeled2017"
                if unlabeled.is_dir():
                    result["unlabeled2017"] = unlabeled
                return result
    raise FileNotFoundError("Could not find coco2017/train2017 under Kaggle input.")


def discover_coco_train_index(coco_train: Path) -> Path:
    """Prefer the smaller captions index; both official files list all train images."""
    annotations = coco_train.parent / "annotations"
    candidates = [
        annotations / "captions_train2017.json",
        annotations / "instances_train2017.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find captions_train2017.json or instances_train2017.json under {annotations}"
    )


def _sample_imagenet_index(
    index_path: Path,
    imagenet_root: Path,
    class_ids: Sequence[str],
    candidates_per_class: int,
    rng: random.Random,
) -> Tuple[Dict[str, List[Path]], int]:
    """Uniform reservoir sample per class from the official CSV index.

    Reading one sequential CSV is substantially faster on Kaggle than issuing
    more than a million directory-entry/stat calls against the mounted input.
    """
    expected_classes = set(class_ids)
    reservoirs: Dict[str, List[Path]] = {class_id: [] for class_id in class_ids}
    seen = {class_id: 0 for class_id in class_ids}
    rows_read = 0
    print(f"[manifest] reading ImageNet index: {index_path}", flush=True)
    with index_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"ImageNet index has no header: {index_path}")
        image_id_key = next(
            (key for key in reader.fieldnames if key.lower().replace("_", "") == "imageid"),
            None,
        )
        prediction_key = next(
            (key for key in reader.fieldnames if key.lower().replace("_", "") == "predictionstring"),
            None,
        )
        if image_id_key is None:
            raise RuntimeError(
                f"ImageNet index lacks an ImageId column; columns={reader.fieldnames}"
            )
        for row in reader:
            rows_read += 1
            image_id = str(row.get(image_id_key, "")).strip()
            if not image_id:
                continue
            prefix = image_id.split("_", 1)[0]
            predicted = str(row.get(prediction_key, "")).split(maxsplit=1)[0] if prediction_key else ""
            class_id = prefix if prefix in expected_classes else predicted
            if class_id not in expected_classes:
                continue
            seen[class_id] += 1
            population_index = seen[class_id]
            candidate = imagenet_root / class_id / f"{image_id}.JPEG"
            reservoir = reservoirs[class_id]
            if len(reservoir) < candidates_per_class:
                reservoir.append(candidate)
            else:
                replacement = rng.randrange(population_index)
                if replacement < candidates_per_class:
                    reservoir[replacement] = candidate
            if rows_read % 250_000 == 0:
                print(f"[manifest] ImageNet index rows: {rows_read:,}", flush=True)

    missing = [class_id for class_id in class_ids if len(reservoirs[class_id]) < candidates_per_class]
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(
            f"ImageNet index has fewer than {candidates_per_class} candidates for "
            f"{len(missing)} classes (first: {preview})."
        )
    for reservoir in reservoirs.values():
        rng.shuffle(reservoir)
    print(
        f"[manifest] ImageNet index ready: rows={rows_read:,}, "
        f"classes={len(reservoirs):,}, retained_candidates={sum(map(len, reservoirs.values())):,}",
        flush=True,
    )
    return reservoirs, rows_read


def _extend_imagenet_candidates_from_index(
    index_path: Path,
    imagenet_root: Path,
    deficient_classes: Sequence[str],
    candidates_by_class: Dict[str, List[Path]],
    rng: random.Random,
) -> int:
    """Append every remaining indexed candidate only for classes needing refill."""
    targets = set(deficient_classes)
    existing = {
        class_id: {str(path) for path in candidates_by_class[class_id]}
        for class_id in targets
    }
    additions: Dict[str, List[Path]] = {class_id: [] for class_id in targets}
    print(
        f"[manifest] expanding ImageNet refill pool for {len(targets)} deficient classes",
        flush=True,
    )
    with index_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"ImageNet index has no header: {index_path}")
        image_id_key = next(
            (key for key in reader.fieldnames if key.lower().replace("_", "") == "imageid"),
            None,
        )
        prediction_key = next(
            (key for key in reader.fieldnames if key.lower().replace("_", "") == "predictionstring"),
            None,
        )
        if image_id_key is None:
            raise RuntimeError(
                f"ImageNet index lacks an ImageId column; columns={reader.fieldnames}"
            )
        for row in reader:
            image_id = str(row.get(image_id_key, "")).strip()
            if not image_id:
                continue
            prefix = image_id.split("_", 1)[0]
            predicted = str(row.get(prediction_key, "")).split(maxsplit=1)[0] if prediction_key else ""
            class_id = prefix if prefix in targets else predicted
            if class_id not in targets:
                continue
            candidate = imagenet_root / class_id / f"{image_id}.JPEG"
            key = str(candidate)
            if key not in existing[class_id]:
                existing[class_id].add(key)
                additions[class_id].append(candidate)

    added = 0
    for class_id, paths in additions.items():
        rng.shuffle(paths)
        candidates_by_class[class_id].extend(paths)
        added += len(paths)
    print(
        f"[manifest] ImageNet refill pool expanded: additional_candidates={added:,}",
        flush=True,
    )
    return added


def _load_coco_index(index_path: Path, coco_train: Path, rng: random.Random) -> List[Path]:
    """Load COCO filenames from its official annotation index, not the image directory."""
    print(f"[manifest] reading COCO index: {index_path}", flush=True)
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        raise RuntimeError(f"COCO index has no top-level images list: {index_path}")
    candidates = []
    for entry in images:
        if not isinstance(entry, dict) or not entry.get("file_name"):
            continue
        candidates.append(coco_train / str(entry["file_name"]))
    if not candidates:
        raise RuntimeError(f"COCO index contains no image filenames: {index_path}")
    rng.shuffle(candidates)
    print(f"[manifest] COCO index ready: images={len(candidates):,}", flush=True)
    return candidates


def _audit_many(
    items: Sequence[Tuple[Path, str, str, str]],
    workers: int,
) -> List[Dict[str, object]]:
    def inspect(item: Tuple[Path, str, str, str]):
        return inspect_image(*item)

    records: List[Dict[str, object]] = []
    total = len(items)
    label = f"{items[0][1]}/{items[0][2]}" if items else "empty"
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, record in enumerate(pool.map(inspect, items), start=1):
            if record is not None:
                records.append(record)
            if index % 1_000 == 0 or index == total:
                print(
                    f"[manifest] audit {label}: {index:,}/{total:,} "
                    f"files, valid={len(records):,}",
                    flush=True,
                )
    return records


def _write_manifest_atomic(records: Sequence[Dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in MANIFEST_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_real_50k_manifest(
    input_root: Path,
    destination: Path,
    seed: int = 42,
    imagenet_per_class: int = 25,
    coco_total: int = 25_000,
    min_side: int = 256,
    workers: int = 16,
) -> Dict[str, object]:
    """Build a deterministic real-only manifest of the requested size.

    COCO uses train2017 and, when available, unlabeled2017. The exact source
    counts and any missing-unlabeled deviation are recorded in the summary.
    """
    if destination.exists():
        rows = read_manifest(destination)
        expected = 1000 * imagenet_per_class + coco_total
        if len(rows) != expected or {row.get("label") for row in rows} != {"0"}:
            raise RuntimeError(
                f"Existing manifest is not the requested real-only {expected}-row cohort: {destination}"
            )
        summary_path = destination.with_suffix(".summary.json")
        if not summary_path.exists():
            raise RuntimeError(f"Existing manifest has no provenance summary: {summary_path}")
        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        requested = {
            "seed": seed,
            "min_side": min_side,
            "imagenet_per_class": imagenet_per_class,
            "coco_total": coco_total,
        }
        mismatches = {
            key: (saved_summary.get(key), value)
            for key, value in requested.items()
            if saved_summary.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Existing manifest provenance differs from requested sampling: {mismatches}. "
                "Use a different --manifest path."
            )
        return {
            "manifest": str(destination),
            "manifest_sha256": manifest_hash(destination),
            "rows": len(rows),
            "reused": True,
        }

    rng = random.Random(seed)
    imagenet_root = discover_imagenet_train(input_root)
    imagenet_index = discover_imagenet_train_index(input_root, imagenet_root)
    coco_roots = discover_coco_roots(input_root)
    records: List[Dict[str, object]] = []
    seen_hashes = set()

    class_dirs = sorted(path for path in imagenet_root.iterdir() if path.is_dir())
    if len(class_dirs) != 1000:
        raise RuntimeError(f"Expected 1000 ImageNet class directories, found {len(class_dirs)}")
    class_ids = [class_dir.name for class_dir in class_dirs]
    # Audit the requested cohort first. Requiring a fixed 4x reserve here is
    # incorrect for classes with fewer indexed rows and can reject a class
    # that still has enough images for the requested sample. If corrupt,
    # undersized or duplicate files create a deficit, the refill pass below
    # expands only those classes from the remaining index rows.
    candidates_per_class = max(100, imagenet_per_class)
    candidates_by_class, imagenet_index_rows = _sample_imagenet_index(
        imagenet_index,
        imagenet_root,
        class_ids,
        candidates_per_class,
        rng,
    )
    by_class: Dict[str, List[Dict[str, object]]] = {class_dir.name: [] for class_dir in class_dirs}
    cursors = {class_dir.name: 0 for class_dir in class_dirs}
    # First pass touches exactly the requested count; small refill rounds cover
    # corrupt, undersized, missing or duplicate files.
    first_pass = True
    while any(len(rows) < imagenet_per_class for rows in by_class.values()):
        imagenet_tasks: List[Tuple[Path, str, str, str]] = []
        for class_dir in class_dirs:
            class_id = class_dir.name
            if len(by_class[class_id]) >= imagenet_per_class:
                continue
            start = cursors[class_id]
            needed = imagenet_per_class - len(by_class[class_id])
            request = needed if first_pass else max(5, needed)
            stop = min(start + request, len(candidates_by_class[class_id]))
            imagenet_tasks.extend(
                (path, "imagenet1k", "train", class_id)
                for path in candidates_by_class[class_id][start:stop]
            )
            cursors[class_id] = stop
        if not imagenet_tasks:
            deficient = [
                class_id
                for class_id, rows in by_class.items()
                if len(rows) < imagenet_per_class
            ]
            added = _extend_imagenet_candidates_from_index(
                imagenet_index,
                imagenet_root,
                deficient,
                candidates_by_class,
                rng,
            )
            if added == 0:
                break
            continue
        for row in _audit_many(imagenet_tasks, workers=workers):
            digest = str(row["sha256"])
            if (
                min(int(row["width"]), int(row["height"])) >= min_side
                and digest not in seen_hashes
                and len(by_class[str(row["class_id"])]) < imagenet_per_class
            ):
                by_class[str(row["class_id"])].append(row)
                seen_hashes.add(digest)
        first_pass = False
    for class_dir in class_dirs:
        eligible = by_class[class_dir.name]
        if len(eligible) < imagenet_per_class:
            raise RuntimeError(
                f"ImageNet class {class_dir.name} has only {len(eligible)} eligible audited images."
            )
        records.extend(eligible[:imagenet_per_class])

    coco_plan: List[Tuple[str, int]]
    if "unlabeled2017" in coco_roots:
        coco_plan = [("train2017", 12_235), ("unlabeled2017", 12_765)]
        coco_mode = "paper_count_matched_train_plus_unlabeled"
    else:
        coco_plan = [("train2017", coco_total)]
        coco_mode = "train2017_only_fallback"

    for split, target in coco_plan:
        if split != "train2017":
            # The currently supported Kaggle COCO input has train2017 only.
            # Keep a safe fallback for a future input that adds unlabeled2017.
            candidates = _images_below(coco_roots[split])
            rng.shuffle(candidates)
            coco_index_path = None
        else:
            coco_index_path = discover_coco_train_index(coco_roots[split])
            candidates = _load_coco_index(coco_index_path, coco_roots[split], rng)
        cursor = 0
        eligible: List[Dict[str, object]] = []
        first_coco_pass = True
        while len(eligible) < target and cursor < len(candidates):
            needed = target - len(eligible)
            request = needed if first_coco_pass else max(1_000, needed)
            stop = min(cursor + request, len(candidates))
            audited = _audit_many(
                [(path, "coco2017", split, "") for path in candidates[cursor:stop]],
                workers=workers,
            )
            cursor = stop
            for row in audited:
                digest = str(row["sha256"])
                if (
                    min(int(row["width"]), int(row["height"])) >= min_side
                    and digest not in seen_hashes
                    and len(eligible) < target
                ):
                    eligible.append(row)
                    seen_hashes.add(digest)
            first_coco_pass = False
        if len(eligible) < target:
            raise RuntimeError(
                f"COCO {split} has only {len(eligible)} eligible unique audited images, need {target}"
            )
        records.extend(eligible)

    expected = 1000 * imagenet_per_class + coco_total
    if len(records) != expected:
        raise RuntimeError(f"Sampling produced {len(records)} rows; expected {expected}.")
    records.sort(key=lambda row: (str(row["source"]), str(row["class_id"]), str(row["sample_id"])))
    _write_manifest_atomic(records, destination)

    summary = {
        "manifest": str(destination),
        "manifest_sha256": manifest_hash(destination),
        "rows": len(records),
        "imagenet_rows": sum(row["source"] == "imagenet1k" for row in records),
        "coco_rows": sum(row["source"] == "coco2017" for row in records),
        "coco_mode": coco_mode,
        "seed": seed,
        "min_side": min_side,
        "imagenet_per_class": imagenet_per_class,
        "coco_total": coco_total,
        "sampling_strategy": "official_indexes_with_seeded_reservoir_and_refill_v2",
        "imagenet_index": str(imagenet_index),
        "imagenet_index_rows": imagenet_index_rows,
        "imagenet_candidates_retained_per_class": candidates_per_class,
        "coco_train_index": str(discover_coco_train_index(coco_roots["train2017"])),
        "reused": False,
    }
    summary_path = destination.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.rows = read_manifest(manifest_path)
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest_path}")
        labels = {int(row["label"]) for row in self.rows}
        if labels != {0}:
            raise ValueError(f"Forensic Twins pretrain manifest must be real-only, got labels {labels}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self.rows[index]


def _transpose(image: Image.Image, operation: str) -> Image.Image:
    transpose = getattr(Image, "Transpose", Image)
    mapping = {
        "rot90": transpose.ROTATE_90,
        "rot180": transpose.ROTATE_180,
        "rot270": transpose.ROTATE_270,
        "flip_h": transpose.FLIP_LEFT_RIGHT,
        "flip_v": transpose.FLIP_TOP_BOTTOM,
    }
    return image.transpose(mapping[operation])


def _rectangles_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return max(ax, bx) < min(ax + aw, bx + bw) and max(ay, by) < min(ay + ah, by + bh)


def _sample_disjoint_boxes(
    rng: random.Random,
    image_w: int,
    image_h: int,
    size1: Tuple[int, int],
    size2: Tuple[int, int],
    max_tries: int = 256,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    h1, w1 = size1
    h2, w2 = size2
    if image_w < max(w1, w2) or image_h < max(h1, h2):
        raise ValueError(f"Image {image_w}x{image_h} is too small for {size1}/{size2}")
    for _ in range(max_tries):
        a = (rng.randint(0, image_w - w1), rng.randint(0, image_h - h1), w1, h1)
        b = (rng.randint(0, image_w - w2), rng.randint(0, image_h - h2), w2, h2)
        if not _rectangles_overlap(a, b):
            return a, b
    # Deterministic fallback for min-side>=256 and max crop<=128.
    candidates = [
        ((0, 0, w1, h1), (image_w - w2, image_h - h2, w2, h2)),
        ((image_w - w1, 0, w1, h1), (0, image_h - h2, w2, h2)),
    ]
    for a, b in candidates:
        if not _rectangles_overlap(a, b):
            return a, b
    raise RuntimeError("Could not place two strictly disjoint crops.")


def _extract_halo(gray: torch.Tensor, box: Tuple[int, int, int, int], radius: int) -> torch.Tensor:
    """Extract crop plus zero halo, matching full-image FRE convolution padding."""
    x, y, width, height = box
    _, image_h, image_w = gray.shape
    left, top = x - radius, y - radius
    right, bottom = x + width + radius, y + height + radius
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(image_w, right), min(image_h, bottom)
    patch = gray[:, src_top:src_bottom, src_left:src_right]
    padding = (src_left - left, right - src_right, src_top - top, bottom - src_bottom)
    patch = F.pad(patch, padding, mode="constant", value=0.0)
    expected = (1, height + 2 * radius, width + 2 * radius)
    if tuple(patch.shape) != expected:
        raise AssertionError(f"Halo extraction produced {tuple(patch.shape)}, expected {expected}")
    return patch


@dataclass
class TwinViewCollator:
    crop_sizes: Sequence[int] = (64, 96, 128)
    fre_radius: int = 7
    include_identity_rotation: bool = True
    flip_probability: float = 0.5

    def __post_init__(self) -> None:
        sizes = tuple(int(size) for size in self.crop_sizes)
        if not sizes or any(size <= 0 or size % 8 for size in sizes):
            raise ValueError("crop_sizes must be positive multiples of patch size 8.")
        self.crop_sizes = sizes
        self._rng: Optional[random.Random] = None

    def _get_rng(self) -> random.Random:
        if self._rng is None:
            self._rng = random.Random(torch.initial_seed())
        return self._rng

    def __call__(self, rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
        rng = self._get_rng()
        size1 = int(rng.choice(self.crop_sizes))
        size2 = int(rng.choice(self.crop_sizes))
        view1: List[torch.Tensor] = []
        view2: List[torch.Tensor] = []
        boxes1, boxes2, sample_ids = [], [], []
        rotations = [0, 1, 2, 3] if self.include_identity_rotation else [1, 2, 3]

        for row in rows:
            with Image.open(row["image_path"]) as source:
                image = source.convert("RGB")
            rotation = int(rng.choice(rotations))
            if rotation:
                image = _transpose(image, f"rot{rotation * 90}")
            if rng.random() < self.flip_probability:
                image = _transpose(image, "flip_h")
            if rng.random() < self.flip_probability:
                image = _transpose(image, "flip_v")
            gray_image = image.convert("L")
            gray_array = np.asarray(gray_image, dtype=np.float32) / 255.0
            gray = torch.from_numpy(np.array(gray_array, copy=True)).unsqueeze(0)
            box1, box2 = _sample_disjoint_boxes(
                rng,
                image.width,
                image.height,
                (size1, size1),
                (size2, size2),
            )
            if _rectangles_overlap(box1, box2):
                raise AssertionError("View sampler returned overlapping boxes.")
            view1.append(_extract_halo(gray, box1, self.fre_radius))
            view2.append(_extract_halo(gray, box2, self.fre_radius))
            boxes1.append(box1)
            boxes2.append(box2)
            sample_ids.append(row["sample_id"])

        return {
            "view1_halo": torch.stack(view1),
            "view2_halo": torch.stack(view2),
            "view1_size": (size1, size1),
            "view2_size": (size2, size2),
            "boxes1": boxes1,
            "boxes2": boxes2,
            "sample_ids": sample_ids,
        }


class ImagePathDataset(Dataset):
    """Inference dataset yielding center-cropped grayscale tensors with FRE halo."""

    def __init__(self, rows: Sequence[Dict[str, object]], crop_size: int = 256, fre_radius: int = 7):
        self.rows = list(rows)
        self.crop_size = crop_size
        self.fre_radius = fre_radius

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        path = Path(str(row["image_path"]))
        with Image.open(path) as source:
            image = source.convert("RGB")
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
            "protocol": str(row.get("protocol", "native")),
        }
