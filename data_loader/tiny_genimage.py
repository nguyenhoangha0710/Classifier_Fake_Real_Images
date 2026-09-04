"""Tiny-GenImage adapter and PyTorch dataset wrapper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch
from PIL import Image

from .schemas import LABEL_ID_TO_NAME, UnifiedSample


DATASET_NAME = "TheKernel01/Tiny-GenImage"
DATASET_SOURCE = "Tiny-GenImage"

TaskType = Literal["classification", "alignment", "sft", "inference"]


def load_tiny_genimage(cache_dir: str | None = None, streaming: bool = False):
    """Load Tiny-GenImage from Hugging Face Datasets."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required. Install it with `pip install datasets`."
        ) from exc
    return load_dataset(DATASET_NAME, cache_dir=cache_dir, streaming=streaming)


def get_class_names(dataset, field: str) -> list[str] | None:
    feature = dataset.features.get(field)
    names = getattr(feature, "names", None)
    return list(names) if names is not None else None


def generator_name_from_record(record: dict[str, Any], generator_names: list[str] | None) -> str:
    value = record.get("generator")
    if isinstance(value, str):
        return value
    if generator_names is not None and isinstance(value, int) and 0 <= value < len(generator_names):
        return generator_names[value]
    return str(value)


def image_path_from_record(record: dict[str, Any]) -> str | None:
    image = record.get("image")
    if isinstance(image, dict):
        return image.get("path")
    return getattr(image, "filename", None)


def _record_to_unified_sample(
    record: dict[str, Any],
    index: int,
    split_name: str,
    eval_case: str,
    generator_names: list[str] | None,
    task_type: TaskType,
    prompt_template: str,
) -> tuple[Image.Image, dict[str, Any]]:
    image = record["image"]
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    label = int(record["label"])
    label_name = LABEL_ID_TO_NAME[label]
    generator_id = record.get("generator")
    if isinstance(generator_id, str):
        generator_id_value = None
    else:
        generator_id_value = int(generator_id)

    sample = UnifiedSample(
        sample_id=f"{DATASET_SOURCE}:{split_name}:{index}",
        label=label,
        label_name=label_name,
        dataset_source=DATASET_SOURCE,
        generator=generator_name_from_record(record, generator_names),
        split=split_name,
        eval_case=eval_case,
        generator_id=generator_id_value,
        image_path=image_path_from_record(record),
    ).as_dict()

    if task_type == "alignment":
        sample["target_text"] = label_name
        sample["label_token"] = label_name
    elif task_type in {"sft", "inference"}:
        sample["prompt"] = prompt_template
        sample["response"] = label_name if task_type == "sft" else None
    return image, sample


class TinyGenImageDataset(torch.utils.data.Dataset):
    """Map-style dataset that returns a unified sample dict.

    task_type controls the extra fields:
    - classification: image, label, metadata
    - alignment: adds target_text = "real" or "fake"
    - sft/inference: adds prompt and response placeholders for future reuse
    """

    def __init__(
        self,
        dataset,
        split_name: str,
        eval_case: str,
        transform: Callable[[Image.Image], Any] | None = None,
        task_type: TaskType = "classification",
        prompt_template: str = "Is this image real or AI-generated? Answer with one label: real or fake.",
    ):
        self.dataset = dataset
        self.split_name = split_name
        self.eval_case = eval_case
        self.transform = transform
        self.task_type = task_type
        self.prompt_template = prompt_template
        self.generator_names = get_class_names(dataset, "generator")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.dataset[index]
        image, sample = _record_to_unified_sample(
            record=record,
            index=index,
            split_name=self.split_name,
            eval_case=self.eval_case,
            generator_names=self.generator_names,
            task_type=self.task_type,
            prompt_template=self.prompt_template,
        )
        if self.transform is not None:
            image = self.transform(image)
        sample["image"] = image
        return sample


class TinyGenImageIterableDataset(torch.utils.data.IterableDataset):
    """Streaming wrapper for Tiny-GenImage.

    Use this when the Hugging Face split was loaded with streaming=True.
    It reads samples lazily from Hugging Face instead of downloading the whole
    dataset into the local datasets cache first.
    """

    def __init__(
        self,
        dataset,
        split_name: str,
        eval_case: str,
        transform: Callable[[Image.Image], Any] | None = None,
        task_type: TaskType = "classification",
        prompt_template: str = "Is this image real or AI-generated? Answer with one label: real or fake.",
    ):
        self.dataset = dataset
        self.split_name = split_name
        self.eval_case = eval_case
        self.transform = transform
        self.task_type = task_type
        self.prompt_template = prompt_template
        self.generator_names = get_class_names(dataset, "generator")

    def __iter__(self):
        for index, record in enumerate(self.dataset):
            image, sample = _record_to_unified_sample(
                record=record,
                index=index,
                split_name=self.split_name,
                eval_case=self.eval_case,
                generator_names=self.generator_names,
                task_type=self.task_type,
                prompt_template=self.prompt_template,
            )
            if self.transform is not None:
                image = self.transform(image)
            sample["image"] = image
            yield sample


def collate_unified_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate unified samples while keeping metadata readable."""

    images = [item["image"] for item in batch]
    can_stack_images = all(torch.is_tensor(image) for image in images)
    output: dict[str, Any] = {
        "image": torch.stack(images) if can_stack_images else images,
        "label": torch.tensor([int(item["label"]) for item in batch], dtype=torch.long),
        "label_name": [item["label_name"] for item in batch],
        "generator": [item["generator"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "metadata": [
            {key: value for key, value in item.items() if key not in {"image"}}
            for item in batch
        ],
    }
    for optional_key in ["target_text", "label_token", "prompt", "response"]:
        if optional_key in batch[0]:
            output[optional_key] = [item.get(optional_key) for item in batch]
    return output
