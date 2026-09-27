"""Core native-patch NPR model utilities.

The input image is never resized. It is divided into overlapping native-resolution
patches. NPR is computed independently on each patch, ResNet18 produces patch
logits, and the logits are averaged into one image-level prediction.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def select_patch_size(short_side: int, patch_sizes: Sequence[int]) -> int:
    """Choose the largest even patch size that fits the image short side."""

    for size in patch_sizes:
        size = int(size)
        if size <= 0 or size % 2:
            raise ValueError(f"Patch sizes must be positive even integers, got {size}.")
        if size <= short_side:
            return size
    raise ValueError(f"Image short side {short_side} is smaller than all patch sizes {tuple(patch_sizes)}.")


def _sliding_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        return []
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if last not in starts:
        starts.append(last)
    return starts


def make_sliding_boxes(
    width: int,
    height: int,
    patch_size: int,
    stride_ratio: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """Return full-coverage, even-aligned sliding boxes for an even-sized image."""

    if width % 2 or height % 2:
        raise ValueError("make_sliding_boxes expects even image dimensions.")
    if patch_size <= 0 or patch_size % 2:
        raise ValueError("patch_size must be a positive even integer.")
    if not 0 < stride_ratio <= 1:
        raise ValueError("stride_ratio must be in (0, 1].")
    stride = max(2, int(patch_size * stride_ratio))
    stride -= stride % 2
    xs = _sliding_starts(width, patch_size, stride)
    ys = _sliding_starts(height, patch_size, stride)
    if not xs or not ys:
        return []
    return [(left, top, patch_size, patch_size) for top in ys for left in xs]


def _uniform_subsample(items: list, count: int) -> list:
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[int(index)] for index in indices]


def extract_native_patches(
    image: Image.Image,
    patch_sizes: Sequence[int] = (256, 128, 64, 32),
    stride_ratio: float = 0.5,
    max_patches: int | None = None,
    random_sample: bool = False,
    rng: random.Random | None = None,
) -> tuple[list[Image.Image], dict]:
    """Crop an image into native-resolution patches without interpolation."""

    image = image.convert("RGB")
    even_width = image.width - image.width % 2
    even_height = image.height - image.height % 2
    if even_width <= 0 or even_height <= 0:
        raise ValueError(f"Invalid image size: {image.size}")
    if (even_width, even_height) != image.size:
        image = image.crop((0, 0, even_width, even_height))

    patch_size = select_patch_size(min(image.size), patch_sizes)
    boxes = make_sliding_boxes(image.width, image.height, patch_size, stride_ratio)
    total_patches = len(boxes)
    if max_patches is not None:
        max_patches = max(1, int(max_patches))
        if random_sample and len(boxes) > max_patches:
            sampler = rng or random.Random()
            boxes = sorted(sampler.sample(boxes, max_patches), key=lambda box: (box[1], box[0]))
        else:
            boxes = _uniform_subsample(boxes, max_patches)
    patches = [image.crop((left, top, left + width, top + height)) for left, top, width, height in boxes]
    return patches, {
        "native_width": image.width,
        "native_height": image.height,
        "patch_size": patch_size,
        "num_patches": len(patches),
        "total_available_patches": total_patches,
        "boxes": boxes,
    }


def pil_patch_to_normalized_tensor(patch: Image.Image, horizontal_flip: bool = False) -> torch.Tensor:
    """Convert one native patch to an ImageNet-normalized tensor without resize."""

    if horizontal_flip:
        patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    array = np.asarray(patch.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class NPRLayer(nn.Module):
    """Neighboring Pixel Relationship residual used by the existing baseline."""

    def __init__(self, factor: float = 0.5, scale: float = 2.0 / 3.0):
        super().__init__()
        self.factor = factor
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [N,C,H,W] patches, got shape {tuple(x.shape)}")
        if x.shape[-2] % 2:
            x = x[:, :, :-1, :]
        if x.shape[-1] % 2:
            x = x[:, :, :, :-1]
        down = F.interpolate(x, scale_factor=self.factor, mode="nearest", recompute_scale_factor=True)
        up = F.interpolate(down, size=x.shape[-2:], mode="nearest")
        return (x - up) * self.scale


class NPRPatchResNet18(nn.Module):
    """Classify a bag of native NPR patches and aggregate to image logits."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        from torchvision.models import resnet18

        self.npr = NPRLayer()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward_patches(self, patches: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.npr(patches))

    def forward(
        self,
        patches: torch.Tensor,
        patch_micro_batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(image_logits, patch_logits)` for one image's patch bag."""

        if patches.ndim != 4 or len(patches) == 0:
            raise ValueError("patches must have shape [num_patches, 3, H, W] and be non-empty.")
        micro = len(patches) if patch_micro_batch_size is None else max(1, int(patch_micro_batch_size))
        logits = [self.forward_patches(patches[start : start + micro]) for start in range(0, len(patches), micro)]
        patch_logits = torch.cat(logits, dim=0)
        image_logits = patch_logits.mean(dim=0, keepdim=True)
        return image_logits, patch_logits
