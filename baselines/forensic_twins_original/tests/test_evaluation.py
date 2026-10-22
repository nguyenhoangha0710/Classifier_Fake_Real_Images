"""CPU tests for GMM scoring, AUROC and exact JPEG-profile matching."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from fit_gmm_and_evaluate import (
    ProtocolImageDataset,
    average_precision,
    auroc,
    build_common_jpeg_rows,
    build_native_eval_rows,
    calibrate_thresholds,
    classification_metrics,
    encode_fake_with_profile,
    fit_single_gaussian,
    gaussian_nll,
    jpeg_profile,
    parse_percentiles,
    save_visualizations,
    split_real_rows,
)


def main() -> None:
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.0, 1.0, 2.0, 3.0])) == 1.0
    assert auroc(np.array([0, 1]), np.array([1.0, 1.0])) == 0.5
    assert average_precision(np.array([0, 0, 1, 1]), np.array([0.0, 1.0, 2.0, 3.0])) == 1.0
    assert parse_percentiles("98,95,80,95") == [80, 95, 98]

    rows = []
    for class_id in ("a", "b"):
        for index in range(10):
            rows.append(
                {
                    "sample_id": f"imagenet1k:{class_id}:{index}",
                    "source": "imagenet1k",
                    "class_id": class_id,
                    "label": 0,
                }
            )
    for index in range(20):
        rows.append(
            {
                "sample_id": f"coco2017:{index}",
                "source": "coco2017",
                "class_id": "",
                "label": 0,
            }
        )
    fit_rows, calibration_rows = split_real_rows(rows, 0.2, 42)
    assert len(fit_rows) == 32 and len(calibration_rows) == 8
    assert {row["sample_id"] for row in fit_rows}.isdisjoint(
        {row["sample_id"] for row in calibration_rows}
    )

    calibration = np.arange(100, dtype=np.float64)
    thresholds = calibrate_thresholds(calibration, [80, 95, 98], 95)
    assert thresholds["percentiles"]["80"] < thresholds["percentiles"]["95"]
    metrics = classification_metrics(
        np.array([0, 0, 1, 1]), np.array([0.0, 2.0, 1.0, 3.0]), threshold=1.5
    )
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)

    rng = np.random.default_rng(42)
    train = rng.normal(size=(200, 8))
    gaussian = fit_single_gaussian(train, reg_covar=1e-6)
    near = gaussian_nll(np.zeros((4, 8)), gaussian).mean()
    far = gaussian_nll(np.full((4, 8), 8.0), gaussian).mean()
    assert np.isfinite(near) and far > near

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        real = root / "real.jpg"
        fake = root / "fake.png"
        matched = root / "matched.jpg"
        Image.fromarray(rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)).save(
            real, format="JPEG", quality=87, subsampling=2
        )
        Image.fromarray(rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)).save(fake)
        profile = jpeg_profile(real)
        digest = encode_fake_with_profile(fake, matched, profile)
        assert len(digest) == 64
        written = jpeg_profile(matched)
        assert written["qtables"] == profile["qtables"]
        assert written["subsampling"] == profile["subsampling"]
        with Image.open(matched) as image:
            assert image.size == (96, 80)

        tiny = root / "tiny"
        real_dir = tiny / "generator_a" / "val" / "nature"
        fake_dir = tiny / "generator_a" / "val" / "ai"
        real_dir.mkdir(parents=True)
        fake_dir.mkdir(parents=True)
        for label_dir, sizes in ((real_dir, (300, 220)), (fake_dir, (300, 220))):
            for index, size in enumerate(sizes):
                Image.new("RGB", (size, size), color=(index * 50, 20, 30)).save(
                    label_dir / f"{index}.png"
                )
        audit = root / "audit.csv"
        native_rows = build_native_eval_rows(tiny, None, 42, crop_size=256, audit_path=audit)
        assert len(native_rows) == 2 and audit.is_file()
        assert {int(row["label"]) for row in native_rows} == {0, 1}
        jpeg_rows = build_common_jpeg_rows(native_rows, 80)
        sample = ProtocolImageDataset(jpeg_rows, crop_size=256)[0]
        assert tuple(sample["gray_halo"].shape) == (1, 270, 270)

        visualization_embeddings = rng.normal(size=(40, 8))
        visualization_labels = np.tile(np.array([0, 1]), 20)
        visualization_generators = np.repeat(np.array(["generator_a", "generator_b"]), 20)
        visualization_scores = gaussian_nll(visualization_embeddings, gaussian)
        visualization_thresholds = calibrate_thresholds(
            gaussian_nll(rng.normal(size=(100, 8)), gaussian), [80, 95, 98], 95
        )
        save_visualizations(
            visualization_embeddings,
            visualization_labels,
            visualization_scores,
            visualization_generators,
            "synthetic",
            gaussian,
            visualization_thresholds,
            root,
            maximum_per_generator_class=10,
            seed=42,
        )
        figure_root = root / "artifacts" / "figures" / "synthetic"
        assert (figure_root / "pca_all_generators_true_labels.png").is_file()
        assert (figure_root / "confusion_matrices" / "q95" / "generator_a.png").is_file()
        assert (figure_root / "generator_clusters_q95" / "generator_b.png").is_file()
        assert (figure_root / "nll_distributions_by_generator.png").is_file()
        assert (figure_root / "roc_pr_curves.png").is_file()

    print("evaluation tests passed")


if __name__ == "__main__":
    main()
