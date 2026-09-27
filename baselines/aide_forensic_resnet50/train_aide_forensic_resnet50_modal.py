"""Train the forensic-only branch of AIDE on Tiny-GenImage with Modal.

This file intentionally keeps the original AIDE forensic design:

* native 32x32 patches with stride 16;
* the six-band DCT score from AIDE;
* two lowest-score and two highest-score patches;
* resize each selected patch to 256x256;
* the fixed bank of 30 SRM high-pass filters;
* separate low- and high-frequency ResNet-50 encoders;
* mean of all four 2048-D patch embeddings;
* a 2048 -> 1024 -> 2 MLP classifier.

The OpenCLIP semantic branch is deliberately omitted.  Only the Combined
Tiny-GenImage experiment is trained.  The best checkpoint is evaluated on the
Tiny test split and on balanced, per-generator CommunityForensics cohorts.

The DCT selector, SRM kernels, and model layout are adapted from AIDE:
https://github.com/shilinyan99/AIDE (MIT License, Copyright 2024 Shilin Yan).
"""

from __future__ import annotations

import time
from pathlib import Path

import modal


APP_NAME = "aide-forensic-resnet50"
GPU_TYPE = "A100-40GB"

TINY_VOLUME_NAME = "tiny-genimage-data"
OUTPUT_VOLUME_NAME = "aide-forensic-resnet50-outputs"
HF_CACHE_VOLUME_NAME = "hf-cache"

REMOTE_CODE_ROOT = "/root/HoangHa_Code"
REMOTE_TINY_ROOT = "/data/tiny-genimage"
REMOTE_OUTPUT_ROOT = "/outputs/aide_forensic_resnet50"
REMOTE_HF_HOME = "/hf-cache"


def find_project_root() -> Path:
    file_path = Path(__file__).resolve()
    candidates = [Path.cwd(), Path(REMOTE_CODE_ROOT), file_path.parent, *file_path.parents]
    for candidate in candidates:
        if (candidate / "data_loader" / "__init__.py").exists():
            return candidate
    raise FileNotFoundError("Cannot locate HoangHa_Code/data_loader.")


LOCAL_PROJECT_ROOT = find_project_root()
LOCAL_DATA_LOADER_DIR = LOCAL_PROJECT_ROOT / "data_loader"
LOCAL_BASELINE_DIR = LOCAL_PROJECT_ROOT / "baselines" / "aide_forensic_resnet50"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "kornia",
        "datasets",
        "kaggle",
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
    image = image.add_local_dir(str(LOCAL_DATA_LOADER_DIR), remote_path=f"{REMOTE_CODE_ROOT}/data_loader")
    image = image.add_local_dir(
        str(LOCAL_BASELINE_DIR),
        remote_path=f"{REMOTE_CODE_ROOT}/baselines/aide_forensic_resnet50",
    )
elif hasattr(modal, "Mount"):
    function_mounts = [
        modal.Mount.from_local_dir(LOCAL_DATA_LOADER_DIR, remote_path=f"{REMOTE_CODE_ROOT}/data_loader"),
        modal.Mount.from_local_dir(
            LOCAL_BASELINE_DIR,
            remote_path=f"{REMOTE_CODE_ROOT}/baselines/aide_forensic_resnet50",
        ),
    ]
else:
    raise RuntimeError("Modal SDK does not support local directory mounts.")

app = modal.App(APP_NAME, image=image)
tiny_volume = modal.Volume.from_name(TINY_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

FUNCTION_OPTIONS = {
    "gpu": GPU_TYPE,
    "timeout": 60 * 60 * 24,
    "memory": 65536,
    "volumes": {
        "/data": tiny_volume,
        "/outputs": output_volume,
        REMOTE_HF_HOME: hf_cache_volume,
    },
}
if function_mounts:
    FUNCTION_OPTIONS["mounts"] = function_mounts

DEFAULT_CONFIG = {
    "tiny_dataset_root": REMOTE_TINY_ROOT,
    "output_root": REMOTE_OUTPUT_ROOT,
    "balance_real": True,
    "random_seed": 42,
    "val_fraction": 0.2,
    "max_train_samples": None,
    "max_val_samples": None,
    "max_tiny_test_samples": None,
    "max_epochs": 5,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "image_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "num_workers": 4,
    "max_grad_norm": 1.0,
    "checkpoint_every_optimizer_steps": 250,
    "resume": True,
    "dct_window_size": 32,
    "dct_stride": 16,
    "dct_grade_bands": 6,
    "selected_patch_size": 256,
    "train_gaussian_blur_probability": 0.1,
    "train_jpeg_probability": 0.1,
    "commfor_dataset_name": "OwensLab/CommunityForensics-Eval",
    "commfor_split": "CompEval",
    "commfor_streaming": True,
    "commfor_shuffle_buffer_size": 1000,
    "commfor_discover_scan_limit": 50000,
    "commfor_fake_per_generator": 100,
    "commfor_real_reference_size": 100,
    "commfor_min_fake_per_generator": 100,
    "commfor_max_generators": 9,
    "commfor_target_generators": None,
}


def _runtime_components():
    """Create model/data classes after Modal has installed runtime packages."""

    import copy
    import io
    import math
    import random
    from typing import Any

    import kornia.augmentation as K
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import Dataset
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
        """Exact six-band AIDE DCT grading with two min/two max patches."""

        def __init__(
            self,
            window_size: int = 32,
            stride: int = 16,
            grade_bands: int = 6,
            output_size: int = 256,
            training: bool = False,
            blur_probability: float = 0.1,
            jpeg_probability: float = 0.1,
        ):
            super().__init__()
            self.window_size = int(window_size)
            self.stride = int(stride)
            self.grade_bands = int(grade_bands)
            self.output_size = int(output_size)
            self.training_mode = bool(training)
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

            self.perturbations = K.container.ImageSequential(
                K.RandomGaussianBlur(
                    kernel_size=(3, 3),
                    sigma=(0.1, 3.0),
                    p=float(blur_probability),
                ),
                K.RandomJPEG(jpeg_quality=(30.0, 100.0), p=float(jpeg_probability)),
            )

        def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"Expected RGB tensor [3,H,W], got {tuple(image.shape)}")
            if min(image.shape[-2:]) < self.window_size:
                raise ValueError(
                    f"Image {tuple(image.shape[-2:])} is smaller than the AIDE patch size {self.window_size}."
                )

            if self.training_mode:
                # Same augmentation family, ranges, order, and probabilities as AIDE.
                image = self.perturbations(image)[0]

            columns = F.unfold(
                image.unsqueeze(0),
                kernel_size=self.window_size,
                stride=self.stride,
            ).squeeze(0).transpose(0, 1)
            patches = columns.reshape(-1, 3, self.window_size, self.window_size)
            dct_coefficients = self.dct @ patches @ self.dct.transpose(0, 1)
            log_magnitude = torch.log(torch.abs(dct_coefficients) + 1.0)

            band_values = []
            for band in range(self.grade_bands):
                value = (
                    log_magnitude * self.band_masks[band].view(1, 1, self.window_size, self.window_size)
                ).sum(dim=(1, 2, 3)) / self.band_counts[band]
                band_values.append(value)
            band_values_tensor = torch.stack(band_values, dim=1)
            scores = (band_values_tensor * self.band_weights.view(1, -1)).sum(dim=1)
            sorted_indices = torch.argsort(scores)
            if len(sorted_indices) < 2:
                raise ValueError("AIDE selection requires at least two candidate patches.")

            # Official AIDE order: min1, max1, min2, max2.
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
        normalized_groups = [
            (f1, 1.0),
            (f2, 2.0),
            (f3, 3.0),
            (edge3, 4.0),
            (edge5, 12.0),
            ([square3], 4.0),
            ([square5], 12.0),
        ]
        for group, divisor in normalized_groups:
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
        """AIDE patchwise forensic branch without the semantic OpenCLIP branch."""

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

        def forward(self, patches: torch.Tensor, return_features: bool = False):
            if patches.ndim != 5 or patches.shape[1] != 4 or patches.shape[2] != 3:
                raise ValueError(f"Expected [B,4,3,H,W], got {tuple(patches.shape)}")
            # Official selector order is low1, high1, low2, high2.
            low_1 = self.model_min(self.hpf(patches[:, 0]))
            high_1 = self.model_max(self.hpf(patches[:, 1]))
            low_2 = self.model_min(self.hpf(patches[:, 2]))
            high_2 = self.model_max(self.hpf(patches[:, 3]))
            low_mean = (low_1 + low_2) / 2.0
            high_mean = (high_1 + high_2) / 2.0
            forensic = (low_1 + high_1 + low_2 + high_2) / 4.0
            logits = self.classifier(forensic)
            if return_features:
                return logits, low_mean, high_mean, forensic
            return logits

    class AIDEForensicTinyDataset(Dataset):
        def __init__(self, dataframe, config: dict, training: bool):
            self.records = dataframe.to_dict("records")
            self.training = bool(training)
            self.selector = AIDEDCTPatchSelector(
                window_size=config["dct_window_size"],
                stride=config["dct_stride"],
                grade_bands=config["dct_grade_bands"],
                output_size=config["selected_patch_size"],
                training=training,
                blur_probability=config["train_gaussian_blur_probability"],
                jpeg_probability=config["train_jpeg_probability"],
            )

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = self.records[index]
            with Image.open(row["image_path"]) as handle:
                image = handle.convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
            patches, selection = self.selector(tensor)
            return {
                "patches": patches,
                "label": int(row["label"]),
                "sample_id": str(row["sample_id"]),
                "generator": str(row["generator"]),
                "image_path": str(row["image_path"]),
                "native_width": int(image.width),
                "native_height": int(image.height),
                **selection,
            }

    return {
        "AIDEDCTPatchSelector": AIDEDCTPatchSelector,
        "AIDEForensicResNet50": AIDEForensicResNet50,
        "AIDEForensicTinyDataset": AIDEForensicTinyDataset,
    }


@app.function(**FUNCTION_OPTIONS)
def train_combined(suite_run_id: str, config_overrides: dict | None = None) -> dict:
    import gc
    import json
    import random
    import sys
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import ImageFile
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
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    sys.path.insert(0, REMOTE_CODE_ROOT)
    from data_loader import (  # noqa: PLC0415
        TinyGenImageKaggleConfig,
        build_kaggle_tiny_splits,
        find_tiny_genimage_root,
        summarize_index,
    )

    components = _runtime_components()
    AIDEForensicResNet50 = components["AIDEForensicResNet50"]
    AIDEForensicTinyDataset = components["AIDEForensicTinyDataset"]

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    pin_memory = device.type == "cuda"

    run_dir = Path(config["output_root"]) / suite_run_id / "combined"
    for subdir in ["checkpoints/latest", "checkpoints/best", "dataset", "metrics", "predictions"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def save_json(path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def maybe_limit(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
        if maximum is None or len(frame) <= int(maximum):
            return frame.reset_index(drop=True)
        # Preserve label/generator representation when a smoke or limited run is requested.
        strata = frame["label"].astype(str) + "_" + frame["generator"].astype(str)
        pieces = []
        fractions = strata.value_counts(normalize=True)
        remaining = int(maximum)
        for key, fraction in fractions.items():
            part = frame[strata == key]
            count = min(len(part), max(1, int(round(float(fraction) * int(maximum)))))
            pieces.append(part.sample(n=count, random_state=seed + len(pieces), replace=False))
            remaining -= count
        result = pd.concat(pieces, ignore_index=True).drop_duplicates("sample_id")
        if len(result) > int(maximum):
            result = result.sample(n=int(maximum), random_state=seed, replace=False)
        elif len(result) < int(maximum):
            available = frame[~frame["sample_id"].isin(result["sample_id"])]
            extra = min(int(maximum) - len(result), len(available))
            if extra:
                result = pd.concat(
                    [result, available.sample(n=extra, random_state=seed + 997, replace=False)],
                    ignore_index=True,
                )
        return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    def make_loader(frame: pd.DataFrame, training: bool, epoch: int = 0) -> DataLoader:
        dataset = AIDEForensicTinyDataset(frame, config=config, training=training)
        generator = torch.Generator()
        generator.manual_seed(int(config["random_seed"]) + int(epoch))
        options = {
            "dataset": dataset,
            "batch_size": int(config["image_batch_size"]),
            "shuffle": bool(training),
            "num_workers": int(config["num_workers"]),
            "pin_memory": pin_memory,
            "drop_last": bool(training),
            "generator": generator,
        }
        if int(config["num_workers"]) > 0:
            options.update({"persistent_workers": False, "prefetch_factor": 2})
        return DataLoader(**options)

    def compute_metrics(y_true, y_prob) -> dict[str, Any]:
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        result = {
            "num_samples": int(len(y_true)),
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
            "fake_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "fake_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "confusion_matrix": matrix.tolist(),
        }
        if len(np.unique(y_true)) == 2:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            result["average_precision"] = float(average_precision_score(y_true, y_prob))
        else:
            result["roc_auc"] = None
            result["average_precision"] = None
        return result

    def per_generator_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for generator_name, part in predictions.groupby("generator", dropna=False):
            row = compute_metrics(part["label"], part["fake_probability"])
            row.update(
                {
                    "generator": str(generator_name),
                    "mean_fake_probability": float(part["fake_probability"].mean()),
                    "mean_num_candidates": float(part["num_candidates"].mean()),
                    "mean_low_dct_score": float(
                        pd.concat([part["low_1_score"], part["low_2_score"]]).mean()
                    ),
                    "mean_high_dct_score": float(
                        pd.concat([part["high_1_score"], part["high_2_score"]]).mean()
                    ),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows).sort_values("generator").reset_index(drop=True)

    seed_everything(int(config["random_seed"]))
    detected_root = find_tiny_genimage_root(config["tiny_dataset_root"])
    split_bundle = build_kaggle_tiny_splits(
        TinyGenImageKaggleConfig(
            dataset_root=str(detected_root),
            eval_case="combined",
            balance_real=bool(config["balance_real"]),
            seed=int(config["random_seed"]),
            max_train_samples=None,
            max_eval_samples=None,
        )
    )
    combined_train = split_bundle["train_df"].reset_index(drop=True)
    tiny_test_df = split_bundle["eval_df"].reset_index(drop=True)
    stratify = combined_train["label"].astype(str) + "_" + combined_train["generator"].astype(str)
    train_df, val_df = train_test_split(
        combined_train,
        test_size=float(config["val_fraction"]),
        random_state=int(config["random_seed"]),
        shuffle=True,
        stratify=stratify,
    )
    train_df = maybe_limit(train_df, config["max_train_samples"], int(config["random_seed"]))
    val_df = maybe_limit(val_df, config["max_val_samples"], int(config["random_seed"]) + 1)
    tiny_test_df = maybe_limit(
        tiny_test_df,
        config["max_tiny_test_samples"],
        int(config["random_seed"]) + 2,
    )

    train_df.to_csv(run_dir / "dataset" / "tiny_train.csv", index=False)
    val_df.to_csv(run_dir / "dataset" / "tiny_val.csv", index=False)
    tiny_test_df.to_csv(run_dir / "dataset" / "tiny_test.csv", index=False)
    summarize_index(split_bundle["full_index"]).to_csv(
        run_dir / "dataset" / "tiny_full_index_summary.csv", index=False
    )
    split_summary = {
        "dataset_root": str(detected_root),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "tiny_test_samples": int(len(tiny_test_df)),
        "generators": sorted(split_bundle["full_index"]["generator"].unique().tolist()),
    }
    save_json(run_dir / "dataset" / "split_summary.json", split_summary)
    save_json(run_dir / "config.json", config)
    output_volume.commit()

    model = AIDEForensicResNet50(imagenet_init=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    val_loader = make_loader(val_df, training=False)
    test_loader = make_loader(tiny_test_df, training=False)

    @torch.no_grad()
    def predict(loader: DataLoader, description: str) -> pd.DataFrame:
        model.eval()
        rows = []
        for batch in tqdm(loader, desc=description, leave=False):
            patches = batch["patches"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(patches)
            probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            labels = batch["label"].cpu().numpy()
            for index in range(len(labels)):
                rows.append(
                    {
                        "sample_id": batch["sample_id"][index],
                        "image_path": batch["image_path"][index],
                        "generator": batch["generator"][index],
                        "label": int(labels[index]),
                        "fake_probability": float(probabilities[index]),
                        "predicted_label": int(probabilities[index] >= 0.5),
                        "native_width": int(batch["native_width"][index]),
                        "native_height": int(batch["native_height"][index]),
                        "num_candidates": int(batch["num_candidates"][index]),
                        "low_1_score": float(batch["low_1_score"][index]),
                        "low_2_score": float(batch["low_2_score"][index]),
                        "high_1_score": float(batch["high_1_score"][index]),
                        "high_2_score": float(batch["high_2_score"][index]),
                    }
                )
        return pd.DataFrame(rows)

    def evaluate(loader: DataLoader, tag: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        predictions = predict(loader, tag)
        overall = compute_metrics(predictions["label"], predictions["fake_probability"])
        generator_table = per_generator_metrics(predictions)
        macro_columns = [
            "accuracy",
            "balanced_accuracy",
            "real_recall",
            "fake_recall",
            "fake_precision",
            "fake_f1",
            "roc_auc",
            "average_precision",
        ]
        for column in macro_columns:
            overall[f"macro_generator_{column}"] = float(generator_table[column].dropna().mean())
        save_json(run_dir / "metrics" / f"{tag}_overall.json", overall)
        generator_table.to_csv(run_dir / "metrics" / f"{tag}_generator_metrics.csv", index=False)
        predictions.to_csv(run_dir / "predictions" / f"{tag}_predictions.csv", index=False)
        print(f"{tag} per-generator metrics:")
        print(generator_table.to_string(index=False))
        return overall, generator_table, predictions

    def save_latest(epoch: int, next_batch: int, optimizer_step: int, history: list[dict]) -> None:
        state = {
            "epoch": int(epoch),
            "next_batch": int(next_batch),
            "optimizer_step": int(optimizer_step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(state, run_dir / "checkpoints" / "latest" / "training_state.pt")
        save_json(
            run_dir / "checkpoints" / "latest" / "checkpoint_info.json",
            {
                "epoch": int(epoch),
                "next_batch": int(next_batch),
                "optimizer_step": int(optimizer_step),
            },
        )
        output_volume.commit()

    start_epoch = 0
    resume_batch = 0
    optimizer_step = 0
    history: list[dict] = []
    best_score = float("-inf")
    latest_path = run_dir / "checkpoints" / "latest" / "training_state.pt"
    best_info_path = run_dir / "checkpoints" / "best" / "checkpoint_info.json"
    if best_info_path.exists():
        with open(best_info_path, "r", encoding="utf-8") as handle:
            best_score = float(json.load(handle)["selection_score"])
    if bool(config["resume"]) and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_epoch = int(state["epoch"])
        resume_batch = int(state["next_batch"])
        optimizer_step = int(state["optimizer_step"])
        history = list(state.get("history", []))
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        print(f"Resuming epoch={start_epoch + 1}, batch={resume_batch}, optimizer_step={optimizer_step}")

    accumulation = max(1, int(config["gradient_accumulation_steps"]))
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, int(config["max_epochs"])):
        train_loader = make_loader(train_df, training=True, epoch=epoch)
        model.train()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"combined epoch {epoch + 1}")
        for batch_index, batch in progress:
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            patches = batch["patches"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(patches)
                loss = criterion(logits, labels)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            epoch_loss_sum += float(loss.detach()) * len(labels)
            epoch_samples += int(len(labels))

            is_boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
            if is_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                every = int(config["checkpoint_every_optimizer_steps"])
                if every > 0 and optimizer_step % every == 0:
                    save_latest(epoch, batch_index + 1, optimizer_step, history)

            progress.set_postfix(loss=f"{float(loss.detach()):.5f}", step=optimizer_step)

        resume_batch = 0
        train_loss = epoch_loss_sum / max(1, epoch_samples)
        val_overall, _, val_predictions = evaluate(val_loader, f"tiny_val_epoch_{epoch + 1}")
        selection_score = float(val_overall["macro_generator_balanced_accuracy"])
        epoch_row = {
            "epoch": epoch + 1,
            "optimizer_step": optimizer_step,
            "train_loss": train_loss,
            "selection_score": selection_score,
            **{f"val_{key}": value for key, value in val_overall.items() if not isinstance(value, list)},
        }
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(run_dir / "metrics" / "history.csv", index=False)

        if selection_score > best_score:
            best_score = selection_score
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best" / "model.pt")
            save_json(
                best_info_path,
                {
                    "epoch": epoch + 1,
                    "optimizer_step": optimizer_step,
                    "selection_metric": "macro_generator_balanced_accuracy",
                    "selection_score": selection_score,
                    "validation_metrics": val_overall,
                },
            )
            val_predictions.to_csv(
                run_dir / "predictions" / "tiny_val_best_predictions.csv", index=False
            )
        save_latest(epoch + 1, 0, optimizer_step, history)

    best_path = run_dir / "checkpoints" / "best" / "model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Best checkpoint was not created: {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    tiny_test_overall, _, _ = evaluate(test_loader, "tiny_test")
    with open(best_info_path, "r", encoding="utf-8") as handle:
        best_info = json.load(handle)
    summary = {
        "suite_run_id": suite_run_id,
        "experiment": "combined",
        "architecture": "AIDE forensic-only dual ResNet50",
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "tiny_test_samples": int(len(tiny_test_df)),
        "best_epoch": int(best_info["epoch"]),
        "best_validation_score": float(best_info["selection_score"]),
        "best_checkpoint": str(best_path),
        "tiny_test_metrics": tiny_test_overall,
    }
    save_json(run_dir / "metrics" / "summary.json", summary)
    pd.DataFrame([{key: value for key, value in summary.items() if not isinstance(value, dict)}]).to_csv(
        run_dir / "metrics" / "summary.csv", index=False
    )
    output_volume.commit()
    del model, optimizer, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


@app.function(**FUNCTION_OPTIONS)
def evaluate_commfor(suite_run_id: str, config_overrides: dict | None = None) -> dict:
    import gc
    import io
    import json
    import sys
    from collections import Counter
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
    components = _runtime_components()
    AIDEDCTPatchSelector = components["AIDEDCTPatchSelector"]
    AIDEForensicResNet50 = components["AIDEForensicResNet50"]

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    if isinstance(config.get("commfor_target_generators"), str):
        config["commfor_target_generators"] = [
            item.strip() for item in config["commfor_target_generators"].split(",") if item.strip()
        ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    combined_dir = Path(config["output_root"]) / suite_run_id / "combined"
    best_path = combined_dir / "checkpoints" / "best" / "model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Combined best checkpoint not found: {best_path}")

    training_config_path = combined_dir / "config.json"
    if training_config_path.exists():
        with open(training_config_path, "r", encoding="utf-8") as handle:
            training_config = json.load(handle)
        for key in ["dct_window_size", "dct_stride", "dct_grade_bands", "selected_patch_size"]:
            if key in training_config:
                config[key] = training_config[key]

    run_dir = Path(config["output_root"]) / suite_run_id / "commfor_combined"
    for subdir in ["dataset", "metrics", "predictions"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def save_json(path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

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

    def generator_from_record(record: dict[str, Any]) -> str:
        return str(record.get("model_name") or record.get("architecture") or "unknown")

    def raw_stream(seed_offset: int = 0):
        dataset = load_dataset(
            config["commfor_dataset_name"],
            split=config["commfor_split"],
            streaming=bool(config["commfor_streaming"]),
        )
        if bool(config["commfor_streaming"]):
            return dataset.shuffle(
                seed=int(config["random_seed"]) + seed_offset,
                buffer_size=int(config["commfor_shuffle_buffer_size"]),
            )
        return dataset.shuffle(seed=int(config["random_seed"]) + seed_offset)

    fake_counts: Counter[str] = Counter()
    real_source_counts: Counter[str] = Counter()
    architecture_counts: Counter[str] = Counter()
    for scanned, record in enumerate(raw_stream(seed_offset=0), start=1):
        label = int(record.get("label"))
        if label == 1:
            fake_counts[generator_from_record(record)] += 1
        else:
            real_source_counts[str(record.get("real_source") or "unknown")] += 1
        architecture_counts[str(record.get("architecture") or "unknown")] += 1
        if scanned >= int(config["commfor_discover_scan_limit"]):
            break

    discovered_df = pd.DataFrame(
        [{"generator": key, "fake_seen": value} for key, value in fake_counts.items()]
    ).sort_values(["fake_seen", "generator"], ascending=[False, True])
    discovered_df.to_csv(run_dir / "dataset" / "discovered_generator_counts.csv", index=False)
    pd.DataFrame(
        [{"real_source": key, "real_seen": value} for key, value in real_source_counts.items()]
    ).sort_values("real_seen", ascending=False).to_csv(
        run_dir / "dataset" / "discovered_real_source_counts.csv", index=False
    )
    pd.DataFrame(
        [{"architecture": key, "num_seen": value} for key, value in architecture_counts.items()]
    ).sort_values("num_seen", ascending=False).to_csv(
        run_dir / "dataset" / "discovered_architecture_counts.csv", index=False
    )

    if config["commfor_target_generators"]:
        target_generators = list(config["commfor_target_generators"])
    else:
        target_generators = discovered_df.loc[
            discovered_df["fake_seen"] >= int(config["commfor_min_fake_per_generator"]), "generator"
        ].tolist()
    if config["commfor_max_generators"] is not None:
        target_generators = target_generators[: int(config["commfor_max_generators"])]
    if not target_generators:
        raise RuntimeError("No CommFor generators satisfy the configured minimum fake quota.")

    fake_quota = int(config["commfor_fake_per_generator"])
    real_quota = int(config["commfor_real_reference_size"])
    selected_fake: dict[str, list[dict]] = {generator: [] for generator in target_generators}
    selected_real: list[dict] = []
    for record in raw_stream(seed_offset=1):
        label = int(record.get("label"))
        if label == 0 and len(selected_real) < real_quota:
            selected_real.append(record)
        elif label == 1:
            generator = generator_from_record(record)
            if generator in selected_fake and len(selected_fake[generator]) < fake_quota:
                selected_fake[generator].append(record)
        if len(selected_real) >= real_quota and all(
            len(records) >= fake_quota for records in selected_fake.values()
        ):
            break

    missing = {
        generator: fake_quota - len(records)
        for generator, records in selected_fake.items()
        if len(records) < fake_quota
    }
    if len(selected_real) < real_quota or missing:
        raise RuntimeError(
            f"CommFor stream ended before quotas were filled: real={len(selected_real)}/{real_quota}, "
            f"fake_missing={missing}"
        )

    selected_rows = []
    for index, record in enumerate(selected_real):
        selected_rows.append(
            {
                "selection_group": "shared_real_reference",
                "sample_index": index,
                "label": 0,
                "generator": "shared_real",
                "image_name": record.get("image_name"),
                "real_source": record.get("real_source"),
                "architecture": record.get("architecture"),
            }
        )
    for generator, records in selected_fake.items():
        for index, record in enumerate(records):
            selected_rows.append(
                {
                    "selection_group": "generator_fake",
                    "sample_index": index,
                    "label": 1,
                    "generator": generator,
                    "image_name": record.get("image_name"),
                    "real_source": record.get("real_source"),
                    "architecture": record.get("architecture"),
                }
            )
    pd.DataFrame(selected_rows).to_csv(run_dir / "dataset" / "selected_samples.csv", index=False)
    pd.DataFrame(
        [
            {
                "generator": generator,
                "metric_cohort": "generator_fake_plus_shared_real",
                "num_real": real_quota,
                "num_fake": len(selected_fake[generator]),
            }
            for generator in target_generators
        ]
    ).to_csv(run_dir / "dataset" / "evaluation_cohorts.csv", index=False)

    selector = AIDEDCTPatchSelector(
        window_size=config["dct_window_size"],
        stride=config["dct_stride"],
        grade_bands=config["dct_grade_bands"],
        output_size=config["selected_patch_size"],
        training=False,
        blur_probability=0.0,
        jpeg_probability=0.0,
    )
    model = AIDEForensicResNet50(imagenet_init=False).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    @torch.no_grad()
    def predict_record(record: dict, label: int, generator: str, sample_index: int) -> dict:
        image = image_from_record(record)
        array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        patches, selection = selector(image_tensor)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(patches.unsqueeze(0).to(device, non_blocking=True))
        fake_probability = float(torch.softmax(logits.float(), dim=-1)[0, 1].cpu())
        return {
            "sample_id": str(record.get("image_name") or f"commfor:{label}:{generator}:{sample_index}"),
            "label": int(label),
            "label_name": "fake" if label == 1 else "real",
            "generator": generator,
            "architecture": record.get("architecture"),
            "real_source": record.get("real_source"),
            "subset": record.get("subset"),
            "image_name": record.get("image_name"),
            "native_width": int(image.width),
            "native_height": int(image.height),
            "fake_probability": fake_probability,
            "predicted_label": int(fake_probability >= 0.5),
            **selection,
        }

    predictions = []
    for index, record in enumerate(tqdm(selected_real, desc="CommFor shared real")):
        predictions.append(predict_record(record, 0, "shared_real", index))
    for generator in target_generators:
        for index, record in enumerate(tqdm(selected_fake[generator], desc=f"CommFor {generator}")):
            predictions.append(predict_record(record, 1, generator, index))
    prediction_df = pd.DataFrame(predictions)
    prediction_df.to_csv(run_dir / "predictions" / "commfor_predictions.csv", index=False)

    def compute_metrics(frame: pd.DataFrame) -> dict[str, Any]:
        y_true = frame["label"].to_numpy(dtype=int)
        y_prob = frame["fake_probability"].to_numpy(dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        result = {
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
            "fake_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "fake_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "confusion_matrix": [[int(value) for value in row] for row in confusion_matrix(y_true, y_pred, labels=[0, 1])],
            "mean_fake_probability": float(y_prob.mean()),
            "mean_num_candidates": float(frame["num_candidates"].mean()),
        }
        return result

    real_predictions = prediction_df[prediction_df["label"] == 0]
    metric_rows = []
    for generator in target_generators:
        fake_predictions = prediction_df[
            (prediction_df["label"] == 1) & (prediction_df["generator"] == generator)
        ]
        cohort = pd.concat([real_predictions, fake_predictions], ignore_index=True)
        row = compute_metrics(cohort)
        row.update(
            {
                "generator": generator,
                "metric_cohort": "generator_fake_plus_shared_real",
            }
        )
        metric_rows.append(row)
    generator_metrics = pd.DataFrame(metric_rows).sort_values("generator").reset_index(drop=True)
    generator_metrics.to_csv(run_dir / "metrics" / "commfor_generator_metrics.csv", index=False)
    print("CommFor per-generator metrics:")
    print(generator_metrics.to_string(index=False))

    macro_columns = [
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
        f"macro_generator_{column}": float(generator_metrics[column].mean())
        for column in macro_columns
    }
    macro.update(
        {
            "worst_generator_balanced_accuracy": float(generator_metrics["balanced_accuracy"].min()),
            "best_generator_balanced_accuracy": float(generator_metrics["balanced_accuracy"].max()),
            "num_generators": int(len(generator_metrics)),
            "fake_per_generator": fake_quota,
            "shared_real_reference_size": real_quota,
            "target_generators": target_generators,
            "suite_run_id": suite_run_id,
            "combined_checkpoint": str(best_path),
            "metric_cohort": "generator_fake_plus_shared_real",
        }
    )
    save_json(run_dir / "metrics" / "macro_summary.json", macro)
    save_json(run_dir / "config.json", config)
    output_volume.commit()
    hf_cache_volume.commit()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return macro


@app.local_entrypoint()
def main(
    suite_run_id: str | None = None,
    smoke: bool = False,
    commfor_only: bool = False,
    skip_commfor: bool = False,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
    max_tiny_test_samples: int = 0,
    max_epochs: int = 5,
    image_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    num_workers: int = 4,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    checkpoint_every_optimizer_steps: int = 250,
    no_resume: bool = False,
    commfor_fake_per_generator: int = 100,
    commfor_real_reference_size: int = 100,
    commfor_min_fake_per_generator: int = 100,
    commfor_max_generators: int | None = 9,
    commfor_target_generators: str | None = None,
):
    """Train Combined Tiny, test Tiny, then evaluate the best model on CommFor."""

    requested_run_id = suite_run_id
    suite_run_id = suite_run_id or time.strftime("%Y%m%d_%H%M%S")
    if commfor_only and not requested_run_id:
        raise ValueError("--commfor-only requires --suite-run-id with a trained combined checkpoint.")

    if smoke:
        max_train_samples = min(max_train_samples or 128, 128)
        max_val_samples = min(max_val_samples or 64, 64)
        max_tiny_test_samples = min(max_tiny_test_samples or 64, 64)
        max_epochs = 1
        commfor_fake_per_generator = min(commfor_fake_per_generator, 2)
        commfor_real_reference_size = min(commfor_real_reference_size, 2)
        commfor_min_fake_per_generator = min(commfor_min_fake_per_generator, 2)
        commfor_max_generators = min(commfor_max_generators or 2, 2)

    overrides = {
        "max_train_samples": None if max_train_samples <= 0 else max_train_samples,
        "max_val_samples": None if max_val_samples <= 0 else max_val_samples,
        "max_tiny_test_samples": None if max_tiny_test_samples <= 0 else max_tiny_test_samples,
        "max_epochs": max_epochs,
        "image_batch_size": image_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_workers": num_workers,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "checkpoint_every_optimizer_steps": checkpoint_every_optimizer_steps,
        "resume": not no_resume,
        "commfor_fake_per_generator": commfor_fake_per_generator,
        "commfor_real_reference_size": commfor_real_reference_size,
        "commfor_min_fake_per_generator": commfor_min_fake_per_generator,
        "commfor_max_generators": commfor_max_generators,
        "commfor_target_generators": commfor_target_generators,
    }

    print("Suite run ID:", suite_run_id)
    print("Experiment: combined only")
    print("Pipeline: DCT 2-low/2-high -> 30 SRM -> dual ResNet50 -> mean -> MLP")

    training_summary = None
    if not commfor_only:
        training_summary = train_combined.remote(suite_run_id, overrides)
        print("Combined training and Tiny test completed:", training_summary)

    commfor_summary = None
    if not skip_commfor:
        commfor_summary = evaluate_commfor.remote(suite_run_id, overrides)
        print("CommFor evaluation completed:", commfor_summary)

    print("Suite completed:", suite_run_id)
    print("Modal output root:", f"{REMOTE_OUTPUT_ROOT}/{suite_run_id}")
