from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from test_bfree_gmm_kaggle import (  # noqa: E402
    BFreeProtocolDataset,
    FAKE_VARIANTS,
    REAL_VARIANT,
    VARIANT_DIRECTORIES,
    build_bfree_manifests,
    load_reusable_manifest,
    paired_score_arrays,
    parse_protocol,
    threshold_rows_for_pairs,
)


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_bfree(root: Path, ids) -> None:
    for directory_name in VARIANT_DIRECTORIES.values():
        directory = root / directory_name / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            Image.new("RGB", (512, 512), color=(17, 29, 41)).save(directory / f"{image_id}.png")
    list_rows = [
        {"id": image_id, "same_category": "cat", "diff_category": "bus"}
        for image_id in ids
    ]
    write_csv(root / "train_list.csv", ("id", "same_category", "diff_category"), list_rows)
    write_csv(root / "valid_list.csv", ("id", "same_category", "diff_category"), list_rows[:3])


def main() -> None:
    assert parse_protocol("native_png").jpeg_quality is None
    assert parse_protocol("jpeg_q95").jpeg_quality == 95
    assert parse_protocol("resize_075").resize_scale == 0.75
    assert parse_protocol("blur_2").blur_sigma == 2.0

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bfree = root / "bfree"
        output = root / "output"
        ids = ["000000000001", "000000000002", "000000000003", "000000000004"]
        create_bfree(bfree, ids)
        training_manifest = root / "train_real_200k.csv"
        write_csv(
            training_manifest,
            ("sample_id", "image_path", "source"),
            [
                {
                    "sample_id": "overlap",
                    "image_path": "/kaggle/input/coco/train2017/000000000001.jpg",
                    "source": "coco2017",
                },
                {
                    "sample_id": "imagenet",
                    "image_path": "/kaggle/input/imagenet/n000/abc.JPEG",
                    "source": "imagenet1k",
                },
            ],
        )
        manifest, summary = build_bfree_manifests(
            bfree_root=bfree,
            training_manifest=training_manifest,
            output_root=output,
            cohort="valid_clean",
            max_ids=None,
            seed=42,
            crop_size=256,
            workers=2,
            audit_images=True,
        )
        assert summary["train_list_ids"] == 4
        assert summary["valid_list_ids"] == 3
        assert summary["valid_overlap_ids"] == 1
        assert summary["valid_clean_ids"] == 2
        assert summary["complete_evaluated_ids"] == 2
        rows = load_reusable_manifest(manifest, bfree)
        assert len(rows) == 14
        assert {row["variant"] for row in rows} == {REAL_VARIANT, *FAKE_VARIANTS}
        assert all(Path(str(row["image_path"])).is_file() for row in rows)

        for protocol_name in ("native_png", "jpeg_q95", "resize_075", "blur_1"):
            dataset = BFreeProtocolDataset(rows[:1], parse_protocol(protocol_name), crop_size=256)
            tensor, index = dataset[0]
            assert tuple(tensor.shape) == (1, 270, 270)
            assert tensor.dtype.is_floating_point
            assert index == 0

        synthetic_scores = []
        for row in rows:
            synthetic_scores.append(-10.0 if row["variant"] == REAL_VARIANT else 10.0)
        paired_ids, real_scores, fake_scores = paired_score_arrays(
            rows, np.asarray(synthetic_scores, dtype=np.float64)
        )
        assert len(paired_ids) == 2
        assert np.all(real_scores == -10.0)
        assert all(np.all(values == 10.0) for values in fake_scores.values())
        threshold_rows = threshold_rows_for_pairs(
            "native_png",
            real_scores,
            fake_scores,
            {"percentiles": {"80": -5.0, "95": 0.0}},
        )
        pooled_q95 = next(
            row
            for row in threshold_rows
            if row["scope"] == "pooled_balanced" and row["threshold_percentile"] == 95
        )
        assert pooled_q95["tn"] == 12
        assert pooled_q95["tp"] == 12
        assert pooled_q95["fp"] == 0
        assert pooled_q95["fn"] == 0

    print("B-Free evaluation tests passed")


if __name__ == "__main__":
    main()
