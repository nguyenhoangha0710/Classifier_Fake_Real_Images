r"""Train and evaluate a dimension-balanced AIDE + NPR fusion model on Modal.

The experiment replaces AIDE's DCT/SRM dual-ResNet50 forensic branch with the
already-trained NPR-ResNet18 detector while balancing both fusion branches:

* frozen OpenCLIP ConvNeXt-XXLarge trunk: 3072-D;
* AIDE semantic projection: 3072 -> 256;
* NPR-ResNet18 penultimate feature: 512-D;
* AIDE semantic feature: 256-D -> LayerNorm;
* trainable NPR adapter: 512 -> 256 -> LayerNorm;
* concatenation: 256 + 256 = 512;
* classifier: 512 -> 256 -> 2.

Data comparability
------------------
Tiny-GenImage uses the exact combined-train/validation indexing, class
balancing, seed (42), and 90/10 train/inner-validation split used by the Full
AIDE experiment.  CommFor uses the immutable balanced 2,000-image manifest
previously used to compare CLIP, NPR-ResNet18, AIDE forensic, and Full AIDE.
CommunityForensics is test-only.

Training is staged. Stage 1 trains the new 512->256 adapter, both branch
normalizers, and the fusion head. Stage 2 additionally fine-tunes NPR layer4
and the AIDE semantic projection. Stage 3 (disabled by default) can fine-tune
all NPR layers while the very large OpenCLIP trunk remains frozen.

This experiment intentionally does not use branch dropout. Relative to the
original 256+2048 run, the fusion width is the only architectural intervention.

Run from the repository root (PowerShell)::

    .\.venv12\Scripts\python.exe -m modal run `
      baselines/aide_npr_fusion/train_aide_npr_fusion_256_256_modal.py

Resume a run after interruption by reusing its run id::

    .\.venv12\Scripts\python.exe -m modal run `
      baselines/aide_npr_fusion/train_aide_npr_fusion_256_256_modal.py `
      --run-id 20260928_120000

Quick smoke test::

    .\.venv12\Scripts\python.exe -m modal run `
      baselines/aide_npr_fusion/train_aide_npr_fusion_256_256_modal.py `
      --smoke-test
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "aide-npr-fusion-256-256"
GPU_TYPE = "A100-40GB"

TINY_VOLUME_NAME = "tiny-genimage-data"
BENCHMARK_VOLUME_NAME = "commfor-balanced-2000-cache"
OUTPUT_VOLUME_NAME = "aide-npr-fusion-outputs"
HF_CACHE_VOLUME_NAME = "hf-cache"

REMOTE_CODE_ROOT = "/root/HoangHa_Code"
REMOTE_TINY_ROOT = "/data/tiny-genimage"
REMOTE_BENCHMARK_ROOT = "/benchmark-cache/commfor_balanced_2000/v1"
REMOTE_OUTPUT_ROOT = "/outputs/aide_npr_fusion_256_256"
REMOTE_HF_HOME = "/hf-cache"
REMOTE_NPR_CHECKPOINT = "/root/checkpoints/npr/npr_resnet18_from_scratch.pt"
REMOTE_AIDE_CHECKPOINT = "/root/checkpoints/aide/model_trainable.pt"
REMOTE_AIDE_RUNTIME = (
    f"{REMOTE_CODE_ROOT}/baselines/aide_original_full/"
    "train_aide_original_full_tiny_commfor_kaggle.py"
)

EXPECTED_NPR_SHA256 = "328f9f431d378f9528c86966796e9f3ac604a5ce2bf3d307028062706c1537f0"
EXPECTED_AIDE_SHA256 = "62848895b44255d6a0567754c503a05807ed0a29b8046f3c6956a049844e47c3"
EXPECTED_COMMFOR_MANIFEST_SHA256 = (
    "c32a3fe854d0964c15aaac058e23bf50305ee9ea377a038e4787da6af649621a"
)
# Ordered sample-id hashes from the completed Full AIDE Tiny-GenImage run.
# Absolute Kaggle/Modal paths are intentionally excluded from these hashes.
EXPECTED_TINY_MEMBERSHIP_SHA256 = {
    "train": "139d55100d3b7fb1d58beeae76af110137f605ed58fb9780edaa0cde7cc57731",
    "val": "c5a37eac8c3792822c19361b932eed1c782d0630cd96d4d21719fd135421cc0d",
    "test": "1702afa0425690b0fb398a174d1b925051880215bcf2385120cb93282254cb78",
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
    "GALIP",
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
    "GALIP": "GALIP",
}


def find_project_root() -> Path:
    file_path = Path(__file__).resolve()
    for candidate in (Path.cwd(), Path(REMOTE_CODE_ROOT), file_path.parent, *file_path.parents):
        if (candidate / "data_loader" / "__init__.py").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate the HoangHa_Code repository root")


REMOTE_ASSETS_READY = all(
    Path(path).is_file()
    for path in (REMOTE_NPR_CHECKPOINT, REMOTE_AIDE_CHECKPOINT, REMOTE_AIDE_RUNTIME)
)
LOCAL_PROJECT_ROOT = Path(REMOTE_CODE_ROOT) if REMOTE_ASSETS_READY else find_project_root()
LOCAL_DATA_LOADER = LOCAL_PROJECT_ROOT / "data_loader"
LOCAL_NPR_RUNTIME = LOCAL_PROJECT_ROOT / "baselines" / "npr_resnet18"
LOCAL_AIDE_RUNTIME = (
    LOCAL_PROJECT_ROOT
    / "baselines"
    / "aide_original_full"
    / "train_aide_original_full_tiny_commfor_kaggle.py"
)
LOCAL_NPR_CHECKPOINT = (
    LOCAL_PROJECT_ROOT
    / "baselines"
    / "npr_resnet18"
    / "artifacts"
    / "checkpoints"
    / "npr_resnet18_from_scratch.pt"
)
LOCAL_AIDE_CHECKPOINT = (
    LOCAL_PROJECT_ROOT
    / "baselines"
    / "aide_original_full"
    / "artifacts"
    / "checkpoints"
    / "model_trainable.pt"
)

if not REMOTE_ASSETS_READY:
    required = [
        LOCAL_DATA_LOADER,
        LOCAL_NPR_RUNTIME,
        LOCAL_AIDE_RUNTIME,
        LOCAL_NPR_CHECKPOINT,
        LOCAL_AIDE_CHECKPOINT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required local assets are missing: {missing}")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "open_clip_torch==2.26.1",
        "kornia",
        "pandas<3.0",
        "scikit-learn<1.9",
        "pillow<12.0",
        "tqdm",
    )
    .env(
        {
            "HF_HOME": REMOTE_HF_HOME,
            "HUGGINGFACE_HUB_CACHE": f"{REMOTE_HF_HOME}/hub",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

function_mounts: list[Any] = []
if hasattr(image, "add_local_file"):
    if not REMOTE_ASSETS_READY:
        image = image.add_local_dir(
            str(LOCAL_DATA_LOADER), remote_path=f"{REMOTE_CODE_ROOT}/data_loader"
        )
        image = image.add_local_dir(
            str(LOCAL_NPR_RUNTIME),
            remote_path=f"{REMOTE_CODE_ROOT}/baselines/npr_resnet18",
        )
        image = image.add_local_file(str(LOCAL_AIDE_RUNTIME), remote_path=REMOTE_AIDE_RUNTIME)
        image = image.add_local_file(
            str(LOCAL_NPR_CHECKPOINT), remote_path=REMOTE_NPR_CHECKPOINT
        )
        image = image.add_local_file(
            str(LOCAL_AIDE_CHECKPOINT), remote_path=REMOTE_AIDE_CHECKPOINT
        )
elif not REMOTE_ASSETS_READY:
    function_mounts = [
        modal.Mount.from_local_dir(
            LOCAL_DATA_LOADER, remote_path=f"{REMOTE_CODE_ROOT}/data_loader"
        ),
        modal.Mount.from_local_dir(
            LOCAL_NPR_RUNTIME,
            remote_path=f"{REMOTE_CODE_ROOT}/baselines/npr_resnet18",
        ),
        modal.Mount.from_local_dir(
            LOCAL_AIDE_RUNTIME.parent,
            remote_path=f"{REMOTE_CODE_ROOT}/baselines/aide_original_full",
        ),
        modal.Mount.from_local_dir(
            LOCAL_NPR_CHECKPOINT.parent,
            remote_path=str(Path(REMOTE_NPR_CHECKPOINT).parent),
        ),
        modal.Mount.from_local_dir(
            LOCAL_AIDE_CHECKPOINT.parent,
            remote_path=str(Path(REMOTE_AIDE_CHECKPOINT).parent),
        ),
    ]


app = modal.App(APP_NAME, image=image)
tiny_volume = modal.Volume.from_name(TINY_VOLUME_NAME, create_if_missing=False)
benchmark_volume = modal.Volume.from_name(BENCHMARK_VOLUME_NAME, create_if_missing=False)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

FUNCTION_OPTIONS: dict[str, Any] = {
    "gpu": GPU_TYPE,
    "timeout": 60 * 60 * 24,
    "memory": 65536,
    "secrets": [modal.Secret.from_name("huggingface-secret")],
    "volumes": {
        "/data": tiny_volume,
        "/benchmark-cache": benchmark_volume,
        "/outputs": output_volume,
        REMOTE_HF_HOME: hf_cache_volume,
    },
}
if function_mounts:
    FUNCTION_OPTIONS["mounts"] = function_mounts


DEFAULT_CONFIG: dict[str, Any] = {
    "tiny_dataset_root": REMOTE_TINY_ROOT,
    "benchmark_root": REMOTE_BENCHMARK_ROOT,
    "output_root": REMOTE_OUTPUT_ROOT,
    "random_seed": 42,
    "val_fraction": 0.10,
    "balance_real": True,
    "max_train_samples": None,
    "max_val_samples": None,
    "max_tiny_test_samples": None,
    "fusion_epochs": 3,
    "partial_finetune_epochs": 2,
    "full_npr_finetune_epochs": 0,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_workers": 4,
    "fusion_learning_rate": 1e-3,
    "partial_learning_rate": 1e-4,
    "full_npr_learning_rate": 2e-5,
    "minimum_learning_rate": 1e-6,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "max_grad_norm": 1.0,
    "checkpoint_every_optimizer_steps": 100,
    "resume": True,
    "semantic_image_size": 256,
    "npr_image_size": 224,
    "train_gaussian_blur_probability": 0.1,
    "train_jpeg_probability": 0.1,
    "semantic_model_name": "convnext_xxlarge",
    "semantic_pretrained_tag": "laion2b_s34b_b82k_augreg_soup",
    "semantic_half_precision": True,
    "semantic_feature_dim": 256,
    "forensic_feature_dim": 256,
    "fusion_hidden_dim": 256,
    "threshold": 0.5,
}


@app.function(**FUNCTION_OPTIONS)
def train_and_evaluate(
    run_id: str,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import contextlib
    import gc
    import hashlib
    import importlib.util
    import json
    import math
    import random
    import sys
    from pathlib import Path

    import kornia.augmentation as K
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
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    sys.path.insert(0, REMOTE_CODE_ROOT)

    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    if not run_id or any(character in run_id for character in "/\\"):
        raise ValueError("run_id must be a non-empty path-safe name")

    run_dir = Path(config["output_root"]) / run_id
    for directory in (
        "checkpoints/latest",
        "checkpoints/best",
        "dataset",
        "metrics",
        "predictions",
        "provenance",
    ):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)

    def save_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary.replace(path)

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
        torch.cuda.manual_seed_all(seed)

    seed_everything(int(config["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for checkpoint_path, expected_hash, name in (
        (Path(REMOTE_NPR_CHECKPOINT), EXPECTED_NPR_SHA256, "NPR-ResNet18"),
        (Path(REMOTE_AIDE_CHECKPOINT), EXPECTED_AIDE_SHA256, "Full AIDE"),
    ):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"{name} checkpoint missing: {checkpoint_path}")
        actual_hash = sha256_file(checkpoint_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{name} checkpoint SHA256 mismatch: expected={expected_hash}, actual={actual_hash}"
            )

    spec = importlib.util.spec_from_file_location(
        "aide_original_full_runtime", REMOTE_AIDE_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Full AIDE runtime: {REMOTE_AIDE_RUNTIME}")
    aide = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aide)

    from baselines.npr_resnet18.npr_resnet18 import NPRResNet18
    from data_loader import build_image_transform

    split_config = dict(aide.DEFAULT_CONFIG)
    split_config.update(
        {
            "input_root": "/data",
            "tiny_dataset_root": config["tiny_dataset_root"],
            "output_root": str(run_dir),
            "random_seed": int(config["random_seed"]),
            "val_fraction": float(config["val_fraction"]),
            "balance_real": bool(config["balance_real"]),
            "max_train_samples": config.get("max_train_samples"),
            "max_val_samples": config.get("max_val_samples"),
            "max_tiny_test_samples": config.get("max_tiny_test_samples"),
        }
    )
    train_frame, val_frame, tiny_test_frame = aide.build_tiny_splits(split_config, run_dir)

    def membership_sha256(frame: pd.DataFrame) -> str:
        payload = "\n".join(frame["sample_id"].astype(str).tolist()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    tiny_membership_hashes = {
        "train": membership_sha256(train_frame),
        "val": membership_sha256(val_frame),
        "test": membership_sha256(tiny_test_frame),
    }
    full_tiny_cohort = all(
        config.get(key) is None
        for key in ("max_train_samples", "max_val_samples", "max_tiny_test_samples")
    )
    if full_tiny_cohort and tiny_membership_hashes != EXPECTED_TINY_MEMBERSHIP_SHA256:
        raise ValueError(
            "Tiny-GenImage membership does not match the completed Full AIDE experiment. "
            f"expected={EXPECTED_TINY_MEMBERSHIP_SHA256}, actual={tiny_membership_hashes}"
        )
    if not full_tiny_cohort:
        print("Tiny membership hash lock skipped because sample limits are active (smoke run).")
    print(
        f"Tiny synchronized splits: train={len(train_frame)}, val={len(val_frame)}, "
        f"test={len(tiny_test_frame)}"
    )

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    imagenet_mean_batch = imagenet_mean.view(1, 3, 1, 1)
    imagenet_std_batch = imagenet_std.view(1, 3, 1, 1)

    class FusionImageDataset(Dataset):
        """Produce branch-specific inputs without changing benchmark membership."""

        def __init__(self, frame: pd.DataFrame, training: bool):
            self.frame = frame.reset_index(drop=True)
            self.training = bool(training)
            # This is the exact transform builder used to train and evaluate the
            # source NPR checkpoint (RandomResizedCrop+flip for train;
            # Resize(1.15x)+CenterCrop for evaluation; ImageNet normalization).
            self.npr_transform = build_image_transform(
                image_size=int(config["npr_image_size"]),
                train=self.training,
            )
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
            with Image.open(row["image_path"]) as image_file:
                image = image_file.convert("RGB")
                array = np.asarray(image, dtype=np.float32).copy() / 255.0
                npr_input = self.npr_transform(image)
            source = torch.from_numpy(array).permute(2, 0, 1).contiguous()

            if self.training:
                source = self.perturbations(source.unsqueeze(0)).squeeze(0).clamp(0.0, 1.0)

            semantic = F.interpolate(
                source.unsqueeze(0),
                size=(int(config["semantic_image_size"]), int(config["semantic_image_size"])),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)

            semantic = (semantic - imagenet_mean) / imagenet_std
            label = int(row["label"])
            return {
                "semantic_input": semantic.contiguous(),
                "npr_input": npr_input.contiguous(),
                "label": torch.tensor(label, dtype=torch.long),
                "sample_id": str(row.get("sample_id", index)),
                "image_name": str(row.get("image_name", Path(row["image_path"]).name)),
                "image_path": str(row["image_path"]),
                "generator": str(row.get("generator", "unknown")),
                "generator_family": str(row.get("generator_family", "unknown")),
                "label_name": str(row.get("label_name", "fake" if label else "real")),
                "manifest_order": int(row.get("manifest_order", index)),
                "cohort_origin": str(row.get("cohort_origin", "Tiny-GenImage")),
            }

    def make_loader(
        frame: pd.DataFrame,
        training: bool,
        epoch_seed: int = 0,
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(int(config["random_seed"]) + int(epoch_seed))
        options: dict[str, Any] = {
            "dataset": FusionImageDataset(frame, training=training),
            "batch_size": int(config["batch_size"]),
            "shuffle": bool(training),
            "num_workers": int(config["num_workers"]),
            "pin_memory": device.type == "cuda",
            "drop_last": bool(training),
            "generator": generator,
        }
        if int(config["num_workers"]) > 0:
            options.update({"persistent_workers": False, "prefetch_factor": 2})
        return DataLoader(**options)

    def extract_state_dict(payload: Any, preferred_key: str | None = None) -> dict[str, Any]:
        if preferred_key and isinstance(payload, dict) and preferred_key in payload:
            payload = payload[preferred_key]
        elif isinstance(payload, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in payload and isinstance(payload[key], dict):
                    payload = payload[key]
                    break
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported checkpoint payload: {type(payload)}")
        return {str(key).removeprefix("module."): value for key, value in payload.items()}

    print("Loading frozen OpenCLIP ConvNeXt-XXLarge semantic trunk...")
    semantic_config = dict(aide.DEFAULT_CONFIG)
    semantic_config.update(
        {
            "input_root": "/root/empty-input-root",
            "semantic_model_name": config["semantic_model_name"],
            "semantic_pretrained_tag": config["semantic_pretrained_tag"],
            "semantic_checkpoint": None,
        }
    )
    semantic_trunk, semantic_provenance = aide.build_openclip_convnext_trunk(semantic_config)
    hf_cache_volume.commit()

    npr_classifier = NPRResNet18(num_classes=2)
    npr_payload = torch.load(REMOTE_NPR_CHECKPOINT, map_location="cpu", weights_only=False)
    npr_classifier.load_state_dict(extract_state_dict(npr_payload), strict=True)
    npr_classifier.backbone.fc = nn.Identity()

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            fused_dim = int(config["semantic_feature_dim"]) + int(
                config["forensic_feature_dim"]
            )
            self.fc1 = nn.Linear(fused_dim, int(config["fusion_hidden_dim"]))
            self.activation = nn.GELU()
            self.fc2 = nn.Linear(int(config["fusion_hidden_dim"]), 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.activation(self.fc1(inputs)))

    class AIDENPRFusion(nn.Module):
        def __init__(self, trunk: nn.Module, npr_encoder: nn.Module):
            super().__init__()
            self.semantic_trunk = trunk
            self.semantic_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.semantic_projection = nn.Linear(
                3072, int(config["semantic_feature_dim"])
            )
            self.semantic_norm = nn.LayerNorm(int(config["semantic_feature_dim"]))
            self.npr_encoder = npr_encoder
            self.npr_adapter = nn.Linear(512, int(config["forensic_feature_dim"]))
            self.forensic_norm = nn.LayerNorm(int(config["forensic_feature_dim"]))
            self.classifier = MLP()
            self.stage_name = "fusion"

        def forward(
            self,
            semantic_input: torch.Tensor,
            npr_input: torch.Tensor,
            return_features: bool = False,
        ):
            clip_input = (
                semantic_input
                * (imagenet_std_batch.to(semantic_input) / clip_std.to(semantic_input))
                + (imagenet_mean_batch.to(semantic_input) - clip_mean.to(semantic_input))
                / clip_std.to(semantic_input)
            )
            with torch.no_grad():
                semantic_map = self.semantic_trunk(clip_input)
            if isinstance(semantic_map, (tuple, list)):
                semantic_map = semantic_map[-1]
            if isinstance(semantic_map, dict):
                semantic_map = (
                    semantic_map["x"] if "x" in semantic_map else list(semantic_map.values())[-1]
                )
            semantic_backbone = (
                self.semantic_pool(semantic_map).flatten(1)
                if semantic_map.ndim == 4
                else semantic_map
            )
            if semantic_backbone.shape[1] != 3072:
                raise ValueError(f"Expected semantic feature [B,3072], got {semantic_backbone.shape}")
            semantic = self.semantic_norm(
                self.semantic_projection(semantic_backbone.float())
            )
            npr_native = self.npr_encoder(npr_input)
            if npr_native.shape[1] != 512:
                raise ValueError(f"Expected NPR feature [B,512], got {npr_native.shape}")
            forensic = self.forensic_norm(self.npr_adapter(npr_native))
            fused = torch.cat([semantic, forensic], dim=1)
            logits = self.classifier(fused)
            if return_features:
                return logits, {
                    "semantic": semantic,
                    "npr_native": npr_native,
                    "forensic": forensic,
                    "fused": fused,
                }
            return logits

        def configure_stage(self, stage_name: str) -> None:
            self.stage_name = stage_name
            for parameter in self.parameters():
                parameter.requires_grad = False
            for parameter in self.npr_adapter.parameters():
                parameter.requires_grad = True
            for parameter in self.semantic_norm.parameters():
                parameter.requires_grad = True
            for parameter in self.forensic_norm.parameters():
                parameter.requires_grad = True
            for parameter in self.classifier.parameters():
                parameter.requires_grad = True
            if stage_name in {"partial", "full_npr"}:
                for parameter in self.semantic_projection.parameters():
                    parameter.requires_grad = True
            if stage_name == "partial":
                for parameter in self.npr_encoder.backbone.layer4.parameters():
                    parameter.requires_grad = True
            elif stage_name == "full_npr":
                for parameter in self.npr_encoder.parameters():
                    parameter.requires_grad = True
            for parameter in self.semantic_trunk.parameters():
                parameter.requires_grad = False

        def enforce_train_modes(self) -> None:
            self.semantic_trunk.eval()
            if self.stage_name == "fusion":
                self.npr_encoder.eval()
            elif self.stage_name == "partial":
                self.npr_encoder.eval()
                self.npr_encoder.backbone.layer4.train()
            else:
                self.npr_encoder.train()

    model = AIDENPRFusion(semantic_trunk, npr_classifier)
    aide_payload = torch.load(REMOTE_AIDE_CHECKPOINT, map_location="cpu", weights_only=False)
    aide_state = extract_state_dict(aide_payload, "model_trainable_state_dict")
    semantic_state = {
        key.removeprefix("semantic_projection."): value
        for key, value in aide_state.items()
        if key.startswith("semantic_projection.")
    }
    if set(semantic_state) != {"weight", "bias"}:
        raise RuntimeError(
            f"Full AIDE checkpoint does not contain the expected semantic projection: {semantic_state.keys()}"
        )
    model.semantic_projection.load_state_dict(semantic_state, strict=True)
    del npr_payload, aide_payload, aide_state, semantic_state
    model.to(device)
    if bool(config["semantic_half_precision"]) and device.type == "cuda":
        model.semantic_trunk.half()
    model.semantic_trunk.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    architecture = {
        "name": "AIDE-NPR fusion 256+256",
        "semantic_branch": "frozen OpenCLIP ConvNeXt-XXLarge 3072 -> AIDE projection 256 -> LayerNorm",
        "forensic_branch": "NPR-ResNet18 penultimate 512 -> adapter 256 -> LayerNorm",
        "fusion": "concat 256 + 256 = 512",
        "classifier": "MLP 512 -> 256 -> 2 with GELU",
        "branch_dropout": False,
        "openclip_always_frozen": True,
        "training_stages": ["fusion", "partial", "full_npr(optional)"],
    }
    source_provenance = {
        "npr_checkpoint": REMOTE_NPR_CHECKPOINT,
        "npr_checkpoint_sha256": EXPECTED_NPR_SHA256,
        "aide_checkpoint": REMOTE_AIDE_CHECKPOINT,
        "aide_checkpoint_sha256": EXPECTED_AIDE_SHA256,
        "semantic_backbone": semantic_provenance,
    }
    save_json(run_dir / "provenance" / "architecture.json", architecture)
    save_json(run_dir / "provenance" / "source_checkpoints.json", source_provenance)
    save_json(run_dir / "provenance" / "training_config.json", config)
    # Persist split manifests and provenance before the first long training epoch.
    output_volume.commit()

    def model_state_without_openclip() -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if not key.startswith("semantic_trunk.")
        }

    def load_model_state(state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing_nonsemantic = [key for key in missing if not key.startswith("semantic_trunk.")]
        if missing_nonsemantic or unexpected:
            raise RuntimeError(
                f"Fusion checkpoint mismatch: missing={missing_nonsemantic}, unexpected={unexpected}"
            )

    latest_path = run_dir / "checkpoints" / "latest" / "model.pt"
    best_path = run_dir / "checkpoints" / "best" / "model.pt"
    history: list[dict[str, Any]] = []
    best_val_ba = -1.0
    best_val_auc = -1.0
    global_optimizer_step = 0
    resume_payload: dict[str, Any] | None = None
    if bool(config["resume"]) and latest_path.is_file():
        resume_payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        previous_config = resume_payload.get("config", {})
        resume_critical_keys = (
            "random_seed",
            "val_fraction",
            "balance_real",
            "max_train_samples",
            "max_val_samples",
            "max_tiny_test_samples",
            "fusion_epochs",
            "partial_finetune_epochs",
            "full_npr_finetune_epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "semantic_image_size",
            "npr_image_size",
            "semantic_feature_dim",
            "forensic_feature_dim",
            "fusion_hidden_dim",
        )
        mismatches = {
            key: {"checkpoint": previous_config.get(key), "requested": config.get(key)}
            for key in resume_critical_keys
            if previous_config.get(key) != config.get(key)
        }
        if mismatches:
            raise ValueError(
                "Refusing to resume the same run_id with incompatible settings: "
                f"{mismatches}. Use the original settings or a new run_id."
            )
        load_model_state(resume_payload["model_state_dict"])
        history = list(resume_payload.get("history", []))
        best_val_ba = float(resume_payload.get("best_val_balanced_accuracy", -1.0))
        best_val_auc = float(resume_payload.get("best_val_roc_auc", -1.0))
        global_optimizer_step = int(resume_payload.get("global_optimizer_step", 0))
        print(
            "Resuming training at "
            f"stage={resume_payload.get('stage_index')}, "
            f"epoch={resume_payload.get('stage_epoch')}, "
            f"batch={resume_payload.get('next_batch')}"
        )

    stages = [
        {
            "name": "fusion",
            "epochs": int(config["fusion_epochs"]),
            "learning_rate": float(config["fusion_learning_rate"]),
        },
        {
            "name": "partial",
            "epochs": int(config["partial_finetune_epochs"]),
            "learning_rate": float(config["partial_learning_rate"]),
        },
        {
            "name": "full_npr",
            "epochs": int(config["full_npr_finetune_epochs"]),
            "learning_rate": float(config["full_npr_learning_rate"]),
        },
    ]
    stages = [stage for stage in stages if stage["epochs"] > 0]
    if not stages:
        raise ValueError("At least one training stage must have a positive epoch count")

    def capture_rng_state() -> dict[str, Any]:
        state = {
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

    def save_checkpoint(
        path: Path,
        *,
        stage_index: int,
        stage_epoch: int,
        next_batch: int,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        scaler: torch.amp.GradScaler | None,
        include_training_state: bool,
    ) -> None:
        payload = {
            "architecture": architecture,
            "model_state_dict": model_state_without_openclip(),
            "stage_index": int(stage_index),
            "stage_epoch": int(stage_epoch),
            "next_batch": int(next_batch),
            "global_optimizer_step": int(global_optimizer_step),
            "best_val_balanced_accuracy": float(best_val_ba),
            "best_val_roc_auc": float(best_val_auc),
            "history": history,
            "config": config,
            "source_provenance": source_provenance,
            "frozen_semantic_backbone_included": False,
        }
        if include_training_state:
            payload.update(
                {
                    "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "scaler_state_dict": scaler.state_dict() if scaler else None,
                    "rng_state": capture_rng_state(),
                }
            )
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def compute_metrics(frame: pd.DataFrame) -> dict[str, Any]:
        threshold = float(config["threshold"])
        y_true = frame["label"].to_numpy(dtype=int)
        y_prob = frame["fake_probability"].to_numpy(dtype=float)
        y_pred = (y_prob >= threshold).astype(int)
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
            "real_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "fake_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "fake_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "fake_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "mean_real_fake_probability": (
                float(y_prob[y_true == 0].mean()) if bool((y_true == 0).any()) else None
            ),
            "mean_fake_fake_probability": (
                float(y_prob[y_true == 1].mean()) if bool((y_true == 1).any()) else None
            ),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "threshold": threshold,
        }
        if len(np.unique(y_true)) == 2:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            result["average_precision"] = float(average_precision_score(y_true, y_prob))
        else:
            result["roc_auc"] = None
            result["average_precision"] = None
        return result

    @torch.inference_mode()
    def evaluate(frame: pd.DataFrame, description: str) -> tuple[dict[str, Any], pd.DataFrame]:
        loader = make_loader(frame, training=False)
        model.eval()
        rows: list[dict[str, Any]] = []
        for batch in tqdm(loader, desc=description):
            semantic_input = batch["semantic_input"].to(device, non_blocking=True)
            npr_input = batch["npr_input"].to(device, non_blocking=True)
            amp = (
                torch.autocast("cuda", dtype=torch.float16)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with amp:
                logits = model(semantic_input, npr_input)
            probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            for index, probability in enumerate(probabilities):
                rows.append(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "manifest_order": int(batch["manifest_order"][index]),
                        "image_name": str(batch["image_name"][index]),
                        "image_path": str(batch["image_path"][index]),
                        "label": int(batch["label"][index]),
                        "label_name": str(batch["label_name"][index]),
                        "generator": str(batch["generator"][index]),
                        "generator_family": str(batch["generator_family"][index]),
                        "cohort_origin": str(batch["cohort_origin"][index]),
                        "fake_probability": float(probability),
                        "predicted_label": int(probability >= float(config["threshold"])),
                    }
                )
        predictions = pd.DataFrame(rows)
        return compute_metrics(predictions), predictions

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    accumulation = int(config["gradient_accumulation_steps"])
    resume_stage_index = int(resume_payload.get("stage_index", 0)) if resume_payload else 0

    for stage_index, stage in enumerate(stages):
        if stage_index < resume_stage_index:
            continue
        model.configure_stage(stage["name"])
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=stage["learning_rate"],
            weight_decay=float(config["weight_decay"]),
        )
        batches_per_epoch = math.ceil(len(train_frame) / int(config["batch_size"]))
        optimizer_steps = max(1, math.ceil(batches_per_epoch / accumulation) * stage["epochs"])

        def lr_factor(step: int) -> float:
            minimum = min(
                1.0, float(config["minimum_learning_rate"]) / stage["learning_rate"]
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(step, optimizer_steps) / optimizer_steps))
            return minimum + (1.0 - minimum) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        start_epoch = 0
        start_batch = 0
        if resume_payload and stage_index == resume_stage_index:
            start_epoch = int(resume_payload.get("stage_epoch", 0))
            start_batch = int(resume_payload.get("next_batch", 0))
            if resume_payload.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
                for optimizer_state in optimizer.state.values():
                    for key, value in optimizer_state.items():
                        if torch.is_tensor(value):
                            optimizer_state[key] = value.to(device)
                scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
                scaler.load_state_dict(resume_payload["scaler_state_dict"])
                restore_rng_state(resume_payload.get("rng_state"))

        for stage_epoch in range(start_epoch, stage["epochs"]):
            loader = make_loader(
                train_frame,
                training=True,
                epoch_seed=stage_index * 10_000 + stage_epoch,
            )
            model.train()
            model.enforce_train_modes()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            observed = 0
            progress = tqdm(
                loader,
                desc=f"{stage['name']} epoch {stage_epoch + 1}/{stage['epochs']}",
            )
            for batch_index, batch in enumerate(progress):
                if stage_epoch == start_epoch and batch_index < start_batch:
                    continue
                semantic_input = batch["semantic_input"].to(device, non_blocking=True)
                npr_input = batch["npr_input"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                amp = (
                    torch.autocast("cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with amp:
                    logits = model(semantic_input, npr_input)
                    raw_loss = criterion(logits, labels)
                    loss = raw_loss / accumulation
                scaler.scale(loss).backward()
                running_loss += float(raw_loss.detach()) * len(labels)
                observed += len(labels)
                should_step = (
                    (batch_index + 1) % accumulation == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
                    if float(config["max_grad_norm"]) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            trainable_parameters, float(config["max_grad_norm"])
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_optimizer_step += 1
                    interval = int(config["checkpoint_every_optimizer_steps"])
                    if interval > 0 and global_optimizer_step % interval == 0:
                        save_checkpoint(
                            latest_path,
                            stage_index=stage_index,
                            stage_epoch=stage_epoch,
                            next_batch=batch_index + 1,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            include_training_state=True,
                        )
                        output_volume.commit()
                progress.set_postfix(
                    loss=f"{running_loss / max(observed, 1):.5f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

            start_batch = 0
            val_metrics, _ = evaluate(
                val_frame,
                description=f"validation {stage['name']} epoch {stage_epoch + 1}",
            )
            epoch_record = {
                "stage": stage["name"],
                "stage_index": stage_index,
                "epoch": stage_epoch + 1,
                "train_loss": running_loss / max(observed, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **val_metrics,
            }
            history.append(epoch_record)
            pd.DataFrame(history).to_csv(run_dir / "metrics" / "history.csv", index=False)
            current_ba = float(val_metrics["balanced_accuracy"])
            current_auc = float(val_metrics.get("roc_auc") or -1.0)
            is_better = current_ba > best_val_ba + 1e-12 or (
                abs(current_ba - best_val_ba) <= 1e-12
                and current_auc > best_val_auc + 1e-12
            )
            if is_better:
                best_val_ba = current_ba
                best_val_auc = current_auc
                save_checkpoint(
                    best_path,
                    stage_index=stage_index,
                    stage_epoch=stage_epoch + 1,
                    next_batch=0,
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    include_training_state=False,
                )
                print(
                    "Saved new best checkpoint: "
                    f"val balanced accuracy={best_val_ba:.4f}, "
                    f"ROC-AUC={best_val_auc:.4f}"
                )
            save_checkpoint(
                latest_path,
                stage_index=stage_index,
                stage_epoch=stage_epoch + 1,
                next_batch=0,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                include_training_state=True,
            )
            output_volume.commit()

        save_checkpoint(
            latest_path,
            stage_index=stage_index + 1,
            stage_epoch=0,
            next_batch=0,
            optimizer=None,
            scheduler=None,
            scaler=None,
            include_training_state=False,
        )
        output_volume.commit()
        resume_payload = None

    if not best_path.is_file():
        raise FileNotFoundError(f"Best checkpoint was not created: {best_path}")
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    load_model_state(best_payload["model_state_dict"])
    model.to(device).eval()

    tiny_overall, tiny_predictions = evaluate(tiny_test_frame, "Tiny-GenImage test")
    tiny_predictions.to_csv(run_dir / "predictions" / "tiny_predictions.csv", index=False)
    tiny_generator_rows = []
    for generator, group in tiny_predictions.groupby("generator", sort=True):
        tiny_generator_rows.append(
            {
                "generator": str(generator),
                "metric_cohort": "tiny_generator_own_real_and_fake",
                **compute_metrics(group),
            }
        )
    tiny_generator_metrics = pd.DataFrame(tiny_generator_rows)
    tiny_generator_metrics.to_csv(
        run_dir / "metrics" / "tiny_generator_metrics.csv", index=False
    )
    metric_columns = (
        "accuracy",
        "balanced_accuracy",
        "real_recall",
        "fake_recall",
        "fake_precision",
        "fake_f1",
        "roc_auc",
        "average_precision",
    )
    tiny_macro = {
        f"macro_generator_{column}": float(tiny_generator_metrics[column].mean())
        for column in metric_columns
    }
    tiny_macro.update(
        {
            "worst_generator_balanced_accuracy": float(
                tiny_generator_metrics["balanced_accuracy"].min()
            ),
            "best_generator_balanced_accuracy": float(
                tiny_generator_metrics["balanced_accuracy"].max()
            ),
            "num_generators": int(len(tiny_generator_metrics)),
        }
    )
    save_json(run_dir / "metrics" / "tiny_overall.json", tiny_overall)
    save_json(run_dir / "metrics" / "tiny_macro_summary.json", tiny_macro)

    benchmark_root = Path(config["benchmark_root"])
    manifest_path = benchmark_root / "commfor_balanced_2000_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Fixed CommFor benchmark manifest missing from volume {BENCHMARK_VOLUME_NAME}: "
            f"{manifest_path}"
        )
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != EXPECTED_COMMFOR_MANIFEST_SHA256:
        raise ValueError(
            "CommFor manifest SHA256 mismatch; refusing to evaluate a different cohort: "
            f"expected={EXPECTED_COMMFOR_MANIFEST_SHA256}, actual={manifest_hash}"
        )
    commfor_frame = pd.read_csv(manifest_path).sort_values("manifest_order").reset_index(drop=True)
    if (
        len(commfor_frame) != 2000
        or commfor_frame["label"].value_counts().to_dict() != {0: 1000, 1: 1000}
        or commfor_frame["manifest_order"].astype(int).tolist() != list(range(2000))
    ):
        raise ValueError("CommFor manifest is not the expected balanced 2,000-image cohort")
    commfor_frame["image_path"] = commfor_frame["cached_image_path"].map(
        lambda relative: str(benchmark_root / str(relative))
    )
    missing_images = [path for path in commfor_frame["image_path"] if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"CommFor cache is incomplete: {len(missing_images)} files missing; first={missing_images[0]}"
        )
    commfor_frame["sample_id"] = commfor_frame["manifest_order"].astype(int).astype(str)
    commfor_frame.to_csv(
        run_dir / "dataset" / "commfor_balanced_2000_manifest_used.csv", index=False
    )

    commfor_overall, commfor_predictions = evaluate(
        commfor_frame, "CommFor balanced 2,000 test"
    )
    commfor_predictions = commfor_predictions.sort_values("manifest_order").reset_index(drop=True)
    commfor_predictions.to_csv(
        run_dir / "predictions" / "commfor_balanced_2000_predictions.csv", index=False
    )
    real = commfor_predictions[commfor_predictions["label"] == 0]
    fake = commfor_predictions[commfor_predictions["label"] == 1]
    legacy_real = real[real["cohort_origin"] == "legacy_commfor_unseen_1000"]
    if len(legacy_real) != 100:
        raise ValueError(f"Expected 100 legacy shared-real images, found {len(legacy_real)}")

    primary_rows = []
    all_real_rows = []
    for generator in TARGET_GENERATORS:
        generator_fake = fake[fake["generator"] == generator]
        if len(generator_fake) != 100:
            raise ValueError(f"Expected 100 fake images for {generator}, found {len(generator_fake)}")
        primary_rows.append(
            {
                "model": "aide_npr_fusion_256_256",
                "generator": generator,
                "generator_family": GENERATOR_FAMILIES[generator],
                "metric_cohort": "100_legacy_real_plus_100_generator_fake",
                **compute_metrics(pd.concat([legacy_real, generator_fake], ignore_index=True)),
            }
        )
        all_real_rows.append(
            {
                "model": "aide_npr_fusion_256_256",
                "generator": generator,
                "generator_family": GENERATOR_FAMILIES[generator],
                "metric_cohort": "1000_real_plus_100_generator_fake",
                **compute_metrics(pd.concat([real, generator_fake], ignore_index=True)),
            }
        )
    commfor_generator_metrics = pd.DataFrame(primary_rows)
    commfor_generator_all_real = pd.DataFrame(all_real_rows)
    commfor_generator_metrics.to_csv(
        run_dir / "metrics" / "commfor_generator_metrics.csv", index=False
    )
    commfor_generator_all_real.to_csv(
        run_dir / "metrics" / "commfor_generator_metrics_all_real.csv", index=False
    )
    commfor_macro = {
        "model": "aide_npr_fusion_256_256",
        **{
            f"macro_generator_{column}": float(commfor_generator_metrics[column].mean())
            for column in metric_columns
        },
        "worst_generator_balanced_accuracy": float(
            commfor_generator_metrics["balanced_accuracy"].min()
        ),
        "best_generator_balanced_accuracy": float(
            commfor_generator_metrics["balanced_accuracy"].max()
        ),
        "num_generators": len(TARGET_GENERATORS),
        "primary_generator_metric_cohort": "100_legacy_real_plus_100_generator_fake",
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": sha256_file(best_path),
    }
    save_json(run_dir / "metrics" / "commfor_overall.json", commfor_overall)
    save_json(run_dir / "metrics" / "commfor_macro_summary.json", commfor_macro)

    split_hashes = {
        name: sha256_file(run_dir / "dataset" / name)
        for name in ("tiny_train.csv", "tiny_val.csv", "tiny_test.csv")
    }
    summary = {
        "run_id": run_id,
        "architecture": architecture,
        "best_validation_balanced_accuracy": best_val_ba,
        "best_validation_roc_auc": best_val_auc,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "tiny_split_sha256": split_hashes,
        "tiny_membership_sha256": tiny_membership_hashes,
        "commfor_manifest_sha256": manifest_hash,
        "tiny": {"overall": tiny_overall, "macro": tiny_macro},
        "commfor": {"overall": commfor_overall, "macro": commfor_macro},
    }
    save_json(run_dir / "metrics" / "summary.json", summary)
    save_json(
        run_dir / "status.json",
        {
            "run_id": run_id,
            "complete": True,
            "best_checkpoint": str(best_path),
            "tiny_test_complete": True,
            "commfor_test_complete": True,
        },
    )
    output_volume.commit()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Modal output root: {run_dir}")
    return summary


@app.local_entrypoint()
def main(
    run_id: str = "",
    smoke_test: bool = False,
    fusion_epochs: int = 3,
    partial_finetune_epochs: int = 2,
    full_npr_finetune_epochs: int = 0,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
) -> None:
    resolved_run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    overrides: dict[str, Any] = {
        "fusion_epochs": int(fusion_epochs),
        "partial_finetune_epochs": int(partial_finetune_epochs),
        "full_npr_finetune_epochs": int(full_npr_finetune_epochs),
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
    }
    if smoke_test:
        overrides.update(
            {
                "max_train_samples": 64,
                "max_val_samples": 32,
                "max_tiny_test_samples": 32,
                "fusion_epochs": 1,
                "partial_finetune_epochs": 0,
                "full_npr_finetune_epochs": 0,
                "checkpoint_every_optimizer_steps": 2,
                "num_workers": 0,
            }
        )
    result = train_and_evaluate.remote(resolved_run_id, overrides)
    print(f"Completed run: {result['run_id']}")
    print(f"Modal output root: {REMOTE_OUTPUT_ROOT}/{resolved_run_id}")
