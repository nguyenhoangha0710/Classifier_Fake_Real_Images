"""Integration test for index-backed manifest construction."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.data import build_real_50k_manifest, read_manifest, real_manifest_filename


def _write_lossless_image(path: Path, value: int) -> None:
    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    pixels[:, :, 0] = value & 255
    pixels[:, :, 1] = (value >> 8) & 255
    pixels[::2, ::2, 2] = (value * 37) & 255
    Image.fromarray(pixels).save(path, format="PNG")


def main() -> None:
    assert real_manifest_filename(25, 25_000) == "train_real_50k.csv"
    assert real_manifest_filename(100, 100_000) == "train_real_200k.csv"
    with tempfile.TemporaryDirectory() as temporary_directory:
        input_root = Path(temporary_directory) / "input"
        competition = input_root / "imagenet-object-localization-challenge"
        imagenet_train = competition / "ILSVRC" / "Data" / "CLS-LOC" / "train"
        imagenet_train.mkdir(parents=True)
        index_path = competition / "LOC_train_solution.csv"

        with index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("ImageId", "PredictionString"))
            for index in range(1_000):
                class_id = f"n{index:08d}"
                class_dir = imagenet_train / class_id
                class_dir.mkdir()
                image_id = f"{class_id}_0"
                _write_lossless_image(class_dir / f"{image_id}.JPEG", index)
                # The production reservoir retains 100 candidates per class.
                for _ in range(100):
                    writer.writerow((image_id, f"{class_id} 0 0 15 15"))

        coco_root = input_root / "coco-2017-dataset" / "coco2017"
        coco_train = coco_root / "train2017"
        coco_train.mkdir(parents=True)
        coco_names = []
        for index in range(4):
            name = f"{index:012d}.jpg"
            _write_lossless_image(coco_train / name, 10_000 + index)
            coco_names.append(name)
        annotations = coco_root / "annotations"
        annotations.mkdir()
        (annotations / "captions_train2017.json").write_text(
            json.dumps({"images": [{"file_name": name} for name in coco_names]}),
            encoding="utf-8",
        )

        destination = Path(temporary_directory) / "output" / "train_real_50k.csv"
        summary = build_real_50k_manifest(
            input_root=input_root,
            destination=destination,
            seed=42,
            imagenet_per_class=1,
            coco_total=3,
            min_side=8,
            workers=4,
        )
        rows = read_manifest(destination)
        assert summary["rows"] == 1_003
        assert summary["imagenet_rows"] == 1_000
        assert summary["coco_rows"] == 3
        assert len(rows) == 1_003
        assert len({row["sha256"] for row in rows}) == 1_003
        assert summary["sampling_strategy"].endswith("_v2")

        reused = build_real_50k_manifest(
            input_root=input_root,
            destination=destination,
            seed=42,
            imagenet_per_class=1,
            coco_total=3,
            min_side=8,
            workers=4,
        )
        assert reused["reused"] is True
        print("manifest integration test passed")


if __name__ == "__main__":
    main()
