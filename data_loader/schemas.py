"""Common sample schema shared across project phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


LABEL_ID_TO_NAME = {0: "real", 1: "fake"}
LABEL_NAME_TO_ID = {name: idx for idx, name in LABEL_ID_TO_NAME.items()}


@dataclass(frozen=True)
class UnifiedSample:
    """Metadata contract used by Stage 1, Stage 2, and future SFT loaders."""

    sample_id: str
    label: int
    label_name: str
    dataset_source: str
    generator: str
    split: str
    eval_case: str
    generator_id: int | None = None
    image_path: str | None = None
    prompt: str | None = None
    response: str | None = None
    explanation: str | None = None
    forensic_attributes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "label": self.label,
            "label_name": self.label_name,
            "dataset_source": self.dataset_source,
            "generator": self.generator,
            "split": self.split,
            "eval_case": self.eval_case,
            "generator_id": self.generator_id,
            "image_path": self.image_path,
            "prompt": self.prompt,
            "response": self.response,
            "explanation": self.explanation,
            "forensic_attributes": list(self.forensic_attributes),
            "metadata": dict(self.metadata),
        }
