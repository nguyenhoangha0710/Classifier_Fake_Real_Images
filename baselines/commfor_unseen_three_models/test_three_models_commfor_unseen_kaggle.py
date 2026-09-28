"""Kaggle runtime for the fixed unseen-family CommFor three-model evaluation.

The module is intentionally independent of Modal.  A notebook imports
``run_evaluation`` from this file, points it at ``/kaggle/input``, and writes
all generated files to ``/kaggle/working``.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import random
from pathlib import Path
from typing import Any

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
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from tqdm.auto import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

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


def extract_state_dict(payload: Any, preferred_key: str | None = None) -> dict[str, torch.Tensor]:
    if preferred_key and isinstance(payload, dict) and preferred_key in payload:
        state_dict = payload[preferred_key]
    elif isinstance(payload, nn.Module):
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


def discover_one(input_root: Path, filename: str, path_hint: str = "") -> Path:
    candidates = sorted(path for path in input_root.rglob(filename) if path.is_file())
    if path_hint:
        hinted = [path for path in candidates if path_hint.lower() in str(path).lower()]
        if len(hinted) == 1:
            return hinted[0]
        if hinted:
            candidates = hinted
    if len(candidates) != 1:
        rendered = "\n".join(f"  - {path}" for path in candidates) or "  (none)"
        raise FileNotFoundError(
            f"Expected exactly one {filename!r} below {input_root}; found {len(candidates)}:\n"
            f"{rendered}\nSet the path explicitly in CONFIG."
        )
    return candidates[0]


def resolve_input_paths(config: dict[str, Any]) -> dict[str, Path | None]:
    input_root = Path(config["input_root"])

    def explicit_or_find(key: str, filename: str, hint: str = "") -> Path:
        explicit = config.get(key)
        path = Path(explicit) if explicit else discover_one(input_root, filename, hint)
        if not path.is_file():
            raise FileNotFoundError(f"{key} not found: {path}")
        return path

    manifest_value = config.get("manifest_path")
    manifest_path: Path | None
    if manifest_value:
        manifest_path = Path(manifest_value)
    else:
        manifests = sorted(input_root.rglob("commfor_unseen_manifest.csv"))
        manifest_path = manifests[0] if len(manifests) == 1 else None

    clip_backbone_value = config.get("clip_backbone_path")
    clip_backbone_path = Path(clip_backbone_value) if clip_backbone_value else None
    if clip_backbone_path is not None and not clip_backbone_path.is_file():
        raise FileNotFoundError(f"clip_backbone_path not found: {clip_backbone_path}")

    return {
        "clip_checkpoint": explicit_or_find(
            "clip_checkpoint", "clip_linear_head.pt", "clip_linear_head"
        ),
        "npr_checkpoint": explicit_or_find(
            "npr_checkpoint", "npr_resnet18_from_scratch.pt", "npr_resnet18"
        ),
        "aide_checkpoint": explicit_or_find("aide_checkpoint", "model.pt", "aide"),
        "manifest_path": manifest_path,
        "clip_backbone_path": clip_backbone_path,
    }


def clean(value: Any) -> str:
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


def build_commfor_cache(config: dict[str, Any], output_root: Path) -> tuple[pd.DataFrame, Path]:
    from datasets import load_dataset

    cache_root = output_root / "dataset"
    image_cache = cache_root / "image_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    image_cache.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        config["commfor_dataset_name"],
        split=config["commfor_split"],
        streaming=True,
    ).shuffle(
        seed=int(config["selection_seed"]),
        buffer_size=int(config["shuffle_buffer_size"]),
    )
    selected_real = {source: [] for source in TARGET_REAL_SOURCES}
    selected_fake = {generator: [] for generator in TARGET_GENERATORS}
    scanned = 0
    for scanned, record in enumerate(dataset, start=1):
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
        if all(
            len(records) >= int(config["real_per_source"])
            for records in selected_real.values()
        ) and all(
            len(records) >= int(config["fake_per_generator"])
            for records in selected_fake.values()
        ):
            break
        if scanned >= int(config["max_scan_records"]):
            break

    missing_real = {
        source: int(config["real_per_source"]) - len(records)
        for source, records in selected_real.items()
        if len(records) < int(config["real_per_source"])
    }
    missing_fake = {
        generator: int(config["fake_per_generator"]) - len(records)
        for generator, records in selected_fake.items()
        if len(records) < int(config["fake_per_generator"])
    }
    if missing_real or missing_fake:
        raise RuntimeError(
            f"CommFor quotas not filled after {scanned} rows: "
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
                    "cached_image_path": f"dataset/image_cache/{order:06d}.png",
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
                    "label": 1,
                    "label_name": "fake",
                    "generator": generator,
                    "generator_family": GENERATOR_FAMILIES[generator],
                    "seen_status": "unseen_generator_family",
                    "image_name": record.get("image_name"),
                    "model_name": record.get("model_name"),
                    "architecture": record.get("architecture"),
                    "real_source": record.get("real_source"),
                    "subset": record.get("subset"),
                    "cached_image_path": f"dataset/image_cache/{order:06d}.png",
                }
            )
            records.append(record)
            order += 1

    manifest = pd.DataFrame(rows)
    for row, record in tqdm(zip(rows, records), total=len(rows), desc="Caching CommFor PNGs"):
        destination = output_root / row["cached_image_path"]
        image_from_record(record).save(destination, format="PNG")
    manifest_path = cache_root / "commfor_unseen_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    save_json(
        cache_root / "image_cache_complete.json",
        {
            "complete": True,
            "image_count": len(manifest),
            "format": "lossless RGB PNG",
            "records_scanned": scanned,
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    return manifest, output_root


def resolve_manifest_and_cache(
    config: dict[str, Any], paths: dict[str, Path | None], output_root: Path
) -> tuple[pd.DataFrame, Path]:
    manifest_path = paths["manifest_path"]
    if manifest_path is None:
        if not config.get("allow_hf_fallback", True):
            raise FileNotFoundError(
                "No commfor_unseen_manifest.csv found in /kaggle/input and HF fallback is disabled."
            )
        print("No uploaded image cache found; selecting and caching CommFor from Hugging Face.")
        return build_commfor_cache(config, output_root)

    manifest = pd.read_csv(manifest_path).sort_values("manifest_order").reset_index(drop=True)
    if "cached_image_path" not in manifest:
        manifest["cached_image_path"] = [
            f"dataset/image_cache/{int(order):06d}.png" for order in manifest["manifest_order"]
        ]
    run_root = manifest_path.parent.parent
    manifest_rows = manifest.to_dict("records")
    standard_ready = all(
        (run_root / str(row["cached_image_path"])).is_file() for row in manifest_rows
    )
    if not standard_ready:
        alternate_root = manifest_path.parent
        alternate_ready = all(
            (
                alternate_root
                / "image_cache"
                / f"{int(row['manifest_order']):06d}.png"
            ).is_file()
            for row in manifest_rows
        )
        if alternate_ready:
            manifest["cached_image_path"] = [
                f"image_cache/{int(order):06d}.png" for order in manifest["manifest_order"]
            ]
            run_root = alternate_root
    missing = [
        run_root / str(row["cached_image_path"])
        for row in manifest.to_dict("records")
        if not (run_root / str(row["cached_image_path"])).is_file()
    ]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        raise FileNotFoundError(
            f"Manifest was found, but {len(missing)} cached PNG files are missing:\n{preview}"
        )
    print(f"Using uploaded fixed image cache: {len(manifest)} images from {run_root}")
    return manifest, run_root


def load_cached_image(manifest_row: dict[str, Any], cache_run_root: Path) -> Image.Image:
    path = cache_run_root / str(manifest_row["cached_image_path"])
    if not path.is_file():
        path = (
            cache_run_root
            / "dataset"
            / "image_cache"
            / f"{int(manifest_row['manifest_order']):06d}.png"
        )
    with Image.open(path) as image:
        return image.convert("RGB")


class NPRLayer(nn.Module):
    def __init__(self, factor: float = 0.5, scale: float = 2.0 / 3.0):
        super().__init__()
        self.factor = factor
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 2 == 1:
            x = x[:, :, :-1, :]
        if x.shape[-1] % 2 == 1:
            x = x[:, :, :, :-1]
        down = F.interpolate(x, scale_factor=self.factor, mode="nearest", recompute_scale_factor=True)
        up = F.interpolate(down, size=x.shape[-2:], mode="nearest")
        return (x - up) * self.scale


class NPRResNet18(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.npr = NPRLayer()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.npr(x))


def dct_matrix(size: int) -> torch.Tensor:
    matrix = []
    for i in range(size):
        scale = math.sqrt(1.0 / size) if i == 0 else math.sqrt(2.0 / size)
        matrix.append([scale * math.cos((j + 0.5) * math.pi * i / size) for j in range(size)])
    return torch.tensor(matrix, dtype=torch.float32)


class AIDEDCTPatchSelector(nn.Module):
    def __init__(self, window_size: int = 32, stride: int = 16, grade_bands: int = 6, output_size: int = 256):
        super().__init__()
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.grade_bands = int(grade_bands)
        self.output_size = int(output_size)
        self.register_buffer("dct", dct_matrix(self.window_size), persistent=False)
        coordinates = torch.arange(self.window_size)
        diagonal_index = coordinates[:, None] + coordinates[None, :]
        masks, counts = [], []
        for band in range(self.grade_bands):
            start = self.window_size * 2.0 / self.grade_bands * band
            end = self.window_size * 2.0 / self.grade_bands * (band + 1)
            mask = ((diagonal_index >= start) & (diagonal_index <= end)).float()
            masks.append(mask)
            counts.append(mask.sum())
        self.register_buffer("band_masks", torch.stack(masks), persistent=False)
        self.register_buffer("band_counts", torch.stack(counts), persistent=False)
        self.register_buffer(
            "band_weights", torch.tensor([2.0**band for band in range(self.grade_bands)]), persistent=False
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        columns = F.unfold(
            image.unsqueeze(0), kernel_size=self.window_size, stride=self.stride
        ).squeeze(0).transpose(0, 1)
        patches = columns.reshape(-1, 3, self.window_size, self.window_size)
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
        selected_dct = coefficients.index_select(0, selected_indices)
        selected = self.dct.transpose(0, 1) @ selected_dct @ self.dct
        selected = F.interpolate(
            selected,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=selected.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=selected.dtype).view(1, 3, 1, 1)
        selected = (selected - mean) / std
        selected_scores = scores.index_select(0, selected_indices)
        return selected.contiguous(), {
            "num_candidates": int(len(scores)),
            "low_1_score": float(selected_scores[0]),
            "high_1_score": float(selected_scores[1]),
            "low_2_score": float(selected_scores[2]),
            "high_2_score": float(selected_scores[3]),
        }


class AIDEHPF(nn.Module):
    def __init__(self):
        super().__init__()
        self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.hpf(inputs)


def make_resnet50_encoder() -> nn.Module:
    backbone = resnet50(weights=None, zero_init_residual=True)
    backbone.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
    backbone.fc = nn.Identity()
    return backbone


class AIDEForensicResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.hpf = AIDEHPF()
        self.model_min = make_resnet50_encoder()
        self.model_max = make_resnet50_encoder()
        self.classifier = nn.Sequential(nn.Linear(2048, 1024), nn.GELU(), nn.Linear(1024, 2))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        low_1 = self.model_min(self.hpf(patches[:, 0]))
        high_1 = self.model_max(self.hpf(patches[:, 1]))
        low_2 = self.model_min(self.hpf(patches[:, 2]))
        high_2 = self.model_max(self.hpf(patches[:, 3]))
        forensic = (low_1 + high_1 + low_2 + high_2) / 4.0
        return self.classifier(forensic)


def prediction_base(row: dict[str, Any], model: str, probability: float, width: int, height: int) -> dict[str, Any]:
    output = dict(row)
    output.update(
        {
            "sample_id": str(row.get("image_name") or f"commfor:{row['manifest_order']}"),
            "native_width": int(width),
            "native_height": int(height),
            "fake_probability": float(probability),
            "predicted_label": int(probability >= 0.5),
            "model": model,
        }
    )
    return output


@torch.no_grad()
def predict_clip(
    manifest: pd.DataFrame,
    cache_root: Path,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    clip_pretrained: str,
) -> pd.DataFrame:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=clip_pretrained, device=device
    )
    model.eval()
    head = nn.Linear(512, 2)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(extract_state_dict(payload, "head"), strict=True)
    head = head.to(device).eval()
    records = manifest.to_dict("records")
    rows, tensors, metadata = [], [], []

    def flush() -> None:
        if not tensors:
            return
        images = torch.stack(tensors).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            logits = head(features)
        probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        for (index, width, height), probability in zip(metadata, probabilities):
            rows.append(prediction_base(records[index], "clip_linear_probe", probability, width, height))
        tensors.clear()
        metadata.clear()

    for index, row in enumerate(tqdm(records, desc="CLIP linear probe")):
        image = load_cached_image(row, cache_root)
        tensors.append(preprocess(image))
        metadata.append((index, image.width, image.height))
        if len(tensors) >= batch_size:
            flush()
    flush()
    del model, head, payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows).sort_values("manifest_order").reset_index(drop=True)


@torch.no_grad()
def predict_npr(
    manifest: pd.DataFrame,
    cache_root: Path,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    model = NPRResNet18(2)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(extract_state_dict(payload), strict=True)
    model = model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize(int(224 * 1.15)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    records = manifest.to_dict("records")
    rows, tensors, metadata = [], [], []

    def flush() -> None:
        if not tensors:
            return
        images = torch.stack(tensors).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
        probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        for (index, width, height), probability in zip(metadata, probabilities):
            rows.append(prediction_base(records[index], "npr_resnet18", probability, width, height))
        tensors.clear()
        metadata.clear()

    for index, row in enumerate(tqdm(records, desc="NPR-ResNet18")):
        image = load_cached_image(row, cache_root)
        tensors.append(transform(image))
        metadata.append((index, image.width, image.height))
        if len(tensors) >= batch_size:
            flush()
    flush()
    del model, payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows).sort_values("manifest_order").reset_index(drop=True)


@torch.no_grad()
def predict_aide(
    manifest: pd.DataFrame,
    cache_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> pd.DataFrame:
    selector = AIDEDCTPatchSelector()
    model = AIDEForensicResNet50()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(extract_state_dict(payload), strict=True)
    model = model.to(device).eval()
    rows = []
    for row in tqdm(manifest.to_dict("records"), desc="AIDE forensic ResNet50"):
        image = load_cached_image(row, cache_root)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        image_tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        patches, selection = selector(image_tensor)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(patches.unsqueeze(0).to(device))
        probability = float(torch.softmax(logits.float(), dim=-1)[0, 1].cpu())
        prediction = prediction_base(
            row, "aide_forensic_resnet50", probability, image.width, image.height
        )
        prediction.update(selection)
        rows.append(prediction)
    del model, selector, payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows).sort_values("manifest_order").reset_index(drop=True)


def compute_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["label"].to_numpy(dtype=int)
    y_prob = frame["fake_probability"].to_numpy(dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
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
        "real_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "fake_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fake_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fake_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def evaluate_model(model_name: str, predictions: pd.DataFrame, metrics_dir: Path) -> dict[str, Any]:
    real = predictions[predictions["label"] == 0]
    fake = predictions[predictions["label"] == 1]
    overall = {"model": model_name, **compute_metrics(predictions)}
    save_json(metrics_dir / f"{model_name}_overall.json", overall)

    generator_rows = []
    for generator in TARGET_GENERATORS:
        cohort = pd.concat([real, fake[fake["generator"] == generator]], ignore_index=True)
        generator_rows.append(
            {
                "model": model_name,
                "generator": generator,
                "generator_family": GENERATOR_FAMILIES[generator],
                "metric_cohort": "generator_fake_plus_shared_real",
                **compute_metrics(cohort),
            }
        )
    generator_metrics = pd.DataFrame(generator_rows)
    generator_metrics.to_csv(metrics_dir / f"{model_name}_generator_metrics.csv", index=False)

    family_rows = []
    for family in dict.fromkeys(GENERATOR_FAMILIES.values()):
        family_fake = fake[fake["generator_family"] == family]
        cohort = pd.concat([real, family_fake], ignore_index=True)
        family_rows.append(
            {
                "model": model_name,
                "generator_family": family,
                "metric_cohort": "family_fake_plus_shared_real",
                **compute_metrics(cohort),
            }
        )
    family_metrics = pd.DataFrame(family_rows)
    family_metrics.to_csv(metrics_dir / f"{model_name}_family_metrics.csv", index=False)

    columns = [
        "accuracy",
        "balanced_accuracy",
        "real_recall",
        "fake_recall",
        "fake_precision",
        "fake_f1",
        "roc_auc",
        "average_precision",
    ]
    macro = {
        "model": model_name,
        **{f"macro_generator_{column}": float(generator_metrics[column].mean()) for column in columns},
        **{f"macro_family_{column}": float(family_metrics[column].mean()) for column in columns},
        "worst_generator_balanced_accuracy": float(generator_metrics["balanced_accuracy"].min()),
        "best_generator_balanced_accuracy": float(generator_metrics["balanced_accuracy"].max()),
    }
    save_json(metrics_dir / f"{model_name}_macro_summary.json", macro)
    return {
        "overall": overall,
        "macro": macro,
        "generator_metrics": generator_metrics,
        "family_metrics": family_metrics,
    }


def run_evaluation(config: dict[str, Any] | None = None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "input_root": "/kaggle/input",
        "output_root": "/kaggle/working/commfor_unseen_three_models",
        "clip_checkpoint": None,
        "npr_checkpoint": None,
        "aide_checkpoint": None,
        "manifest_path": None,
        "clip_backbone_path": None,
        "allow_hf_fallback": True,
        "commfor_dataset_name": "OwensLab/CommunityForensics-Eval",
        "commfor_split": "CompEval",
        "selection_seed": 43,
        "shuffle_buffer_size": 1000,
        "max_scan_records": 500000,
        "fake_per_generator": 100,
        "real_per_source": 20,
        "clip_batch_size": 32,
        "npr_batch_size": 32,
    }
    if config:
        defaults.update(config)
    config = defaults
    random.seed(int(config["selection_seed"]))
    np.random.seed(int(config["selection_seed"]))
    torch.manual_seed(int(config["selection_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["selection_seed"]))

    output_root = Path(config["output_root"])
    predictions_dir = output_root / "predictions"
    metrics_dir = output_root / "metrics"
    provenance_dir = output_root / "provenance"
    for directory in (predictions_dir, metrics_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths = resolve_input_paths(config)
    manifest, cache_root = resolve_manifest_and_cache(config, paths, output_root)
    expected = len(TARGET_GENERATORS) * int(config["fake_per_generator"]) + len(
        TARGET_REAL_SOURCES
    ) * int(config["real_per_source"])
    if len(manifest) != expected:
        raise ValueError(f"Manifest has {len(manifest)} rows, expected {expected}.")
    if manifest["manifest_order"].duplicated().any():
        raise ValueError("manifest_order is not unique.")

    manifest.to_csv(output_root / "commfor_unseen_manifest_used.csv", index=False)
    checkpoint_info = {
        key: {"path": str(path), "sha256": sha256_file(path)}
        for key, path in paths.items()
        if key.endswith("checkpoint") and path is not None
    }
    save_json(provenance_dir / "checkpoints.json", checkpoint_info)
    save_json(output_root / "config.json", {key: str(value) if isinstance(value, Path) else value for key, value in config.items()})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; manifest samples: {len(manifest)}")
    clip_pretrained = (
        str(paths["clip_backbone_path"])
        if paths["clip_backbone_path"] is not None
        else "openai"
    )
    predictors = {
        "clip_linear_probe": lambda: predict_clip(
            manifest,
            cache_root,
            paths["clip_checkpoint"],
            device,
            int(config["clip_batch_size"]),
            clip_pretrained,
        ),
        "npr_resnet18": lambda: predict_npr(
            manifest,
            cache_root,
            paths["npr_checkpoint"],
            device,
            int(config["npr_batch_size"]),
        ),
        "aide_forensic_resnet50": lambda: predict_aide(
            manifest, cache_root, paths["aide_checkpoint"], device
        ),
    }

    predictions_by_model = {}
    results_by_model = {}
    for model_name, predictor in predictors.items():
        prediction_path = predictions_dir / f"{model_name}_predictions.csv"
        if prediction_path.is_file():
            candidate = pd.read_csv(prediction_path).sort_values("manifest_order").reset_index(drop=True)
            if len(candidate) == len(manifest) and np.array_equal(
                candidate["manifest_order"].to_numpy(), manifest["manifest_order"].to_numpy()
            ):
                predictions = candidate
                print(f"Resuming completed {model_name} predictions.")
            else:
                predictions = predictor()
                predictions.to_csv(prediction_path, index=False)
        else:
            predictions = predictor()
            predictions.to_csv(prediction_path, index=False)
        predictions_by_model[model_name] = predictions
        results_by_model[model_name] = evaluate_model(model_name, predictions, metrics_dir)

    pd.concat(predictions_by_model.values(), ignore_index=True).to_csv(
        predictions_dir / "all_models_predictions.csv", index=False
    )
    wide = manifest.copy()
    for model_name, predictions in predictions_by_model.items():
        wide[f"{model_name}_fake_probability"] = predictions["fake_probability"].to_numpy()
        wide[f"{model_name}_predicted_label"] = predictions["predicted_label"].to_numpy()
    wide.to_csv(predictions_dir / "all_models_predictions_wide.csv", index=False)

    all_generator = pd.concat(
        [result["generator_metrics"] for result in results_by_model.values()], ignore_index=True
    )
    all_family = pd.concat(
        [result["family_metrics"] for result in results_by_model.values()], ignore_index=True
    )
    all_generator.to_csv(metrics_dir / "all_models_generator_metrics.csv", index=False)
    all_family.to_csv(metrics_dir / "all_models_family_metrics.csv", index=False)
    comparison = pd.DataFrame(
        [
            {
                "model": model_name,
                **{
                    f"overall_{key}": value
                    for key, value in result["overall"].items()
                    if not isinstance(value, list)
                },
                **result["macro"],
            }
            for model_name, result in results_by_model.items()
        ]
    )
    comparison.to_csv(metrics_dir / "all_models_summary.csv", index=False)
    summary = {
        "output_root": str(output_root),
        "manifest_samples": len(manifest),
        "manifest_real": int((manifest["label"] == 0).sum()),
        "manifest_fake": int((manifest["label"] == 1).sum()),
        "target_generators": list(TARGET_GENERATORS),
        "models": {
            model_name: {"overall": result["overall"], "macro": result["macro"]}
            for model_name, result in results_by_model.items()
        },
    }
    save_json(metrics_dir / "comparison_summary.json", summary)
    print(comparison[["model", "macro_generator_balanced_accuracy", "macro_generator_roc_auc"]])
    return summary
