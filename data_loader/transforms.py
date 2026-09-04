"""Image transforms for clean and robustness evaluation."""

from __future__ import annotations

import io
from typing import Literal

from PIL import Image, ImageFilter


PerturbationName = Literal["none", "jpeg", "resize", "center_crop", "blur"]


class JpegCompression:
    def __init__(self, quality: int = 75):
        self.quality = int(quality)

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class ResizeRoundTrip:
    def __init__(self, scale: float = 0.5):
        if not 0 < scale <= 1:
            raise ValueError("scale must be in (0, 1].")
        self.scale = float(scale)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        small_size = (max(1, int(width * self.scale)), max(1, int(height * self.scale)))
        down = image.resize(small_size, Image.BICUBIC)
        return down.resize((width, height), Image.BICUBIC)


class CenterCropRatio:
    def __init__(self, ratio: float = 0.8):
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1].")
        self.ratio = float(ratio)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        crop_w = max(1, int(width * self.ratio))
        crop_h = max(1, int(height * self.ratio))
        left = (width - crop_w) // 2
        top = (height - crop_h) // 2
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((width, height), Image.BICUBIC)


class GaussianBlur:
    def __init__(self, radius: float = 1.0):
        self.radius = float(radius)

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))


def build_image_transform(
    image_size: int = 224,
    train: bool = False,
    normalize: bool = True,
    perturbation: PerturbationName = "none",
    jpeg_quality: int = 75,
    resize_scale: float = 0.5,
    crop_ratio: float = 0.8,
    blur_radius: float = 1.0,
):
    """Build a torchvision transform for classifier/alignment phases."""

    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("torchvision is required for build_image_transform().") from exc

    ops = []
    if perturbation == "jpeg":
        ops.append(JpegCompression(jpeg_quality))
    elif perturbation == "resize":
        ops.append(ResizeRoundTrip(resize_scale))
    elif perturbation == "center_crop":
        ops.append(CenterCropRatio(crop_ratio))
    elif perturbation == "blur":
        ops.append(GaussianBlur(blur_radius))
    elif perturbation != "none":
        raise ValueError(f"Unsupported perturbation: {perturbation}")

    if train:
        ops.extend(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        ops.extend(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.CenterCrop(image_size),
            ]
        )

    ops.append(transforms.ToTensor())
    if normalize:
        ops.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    return transforms.Compose(ops)
