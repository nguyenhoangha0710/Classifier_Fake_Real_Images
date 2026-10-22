"""CPU tests for augmented-real GMM fitting and calibration."""

from __future__ import annotations

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fit_augmented_gmms_bfree import (  # noqa: E402
    RealVariantDataset,
    VARIANT_SPECS,
    calibrated_thresholds,
    fit_one_case,
    load_mixture,
    mixture_nll,
    save_mixture,
    stratified_limit,
)


def test_mixture_scores() -> None:
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(7)
    values = np.r_[rng.normal(-2.0, 0.6, (160, 4)), rng.normal(2.0, 0.8, (160, 4))]
    probe = rng.normal(0.0, 2.0, (31, 4))
    for covariance_type in ("full", "diag", "tied", "spherical"):
        model = GaussianMixture(
            n_components=2,
            covariance_type=covariance_type,
            reg_covar=1e-6,
            random_state=11,
            n_init=1,
        ).fit(values)
        payload = {
            "weights": model.weights_,
            "means": model.means_,
            "covariances": model.covariances_,
            "covariance_type": model.covariance_type,
        }
        expected = -model.score_samples(probe)
        actual = mixture_nll(probe, payload, batch_size=7)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gmm.npz"
            save_mixture(model, path, {"covariance_type": covariance_type})
            loaded = load_mixture(path)
            np.testing.assert_allclose(
                mixture_nll(probe, loaded, batch_size=9), expected, rtol=1e-8, atol=1e-8
            )


def test_thresholds() -> None:
    scores = np.arange(1000, dtype=np.float64)
    thresholds = calibrated_thresholds(scores, (80, 95, 98), primary=95)
    assert thresholds["primary_percentile"] == 95
    for percentile in (80, 95, 98):
        threshold = float(thresholds["percentiles"][str(percentile)])
        empirical_fpr = float(np.mean(scores > threshold))
        assert abs(empirical_fpr - (1.0 - percentile / 100.0)) <= 0.002


def test_stratified_limit() -> None:
    rows = []
    for index in range(100):
        rows.append(
            {
                "sample_id": f"imagenet:{index}",
                "source": "imagenet1k",
                "class_id": f"n{index % 10:08d}",
            }
        )
        rows.append(
            {
                "sample_id": f"coco:{index}",
                "source": "coco2017",
                "class_id": "",
            }
        )
    selected = stratified_limit(rows, 40, seed=42)
    assert len(selected) == 40
    assert sum(row["source"] == "imagenet1k" for row in selected) == 20
    assert sum(row["source"] == "coco2017" for row in selected) == 20
    assert len({row["class_id"] for row in selected if row["source"] == "imagenet1k"}) == 10


def test_variant_dataset() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "gradient.png"
        yy, xx = np.mgrid[:600, :800]
        pixels = np.stack(
            ((xx % 256), (yy % 256), ((xx + yy) % 256)), axis=-1
        ).astype(np.uint8)
        Image.fromarray(pixels, mode="RGB").save(path)
        rows = [{"image_path": str(path), "sample_id": "one"}]
        tensors = {}
        for name, spec in VARIANT_SPECS.items():
            dataset = RealVariantDataset(rows, spec, crop_size=256)
            tensor, index = dataset[0]
            assert tuple(tensor.shape) == (1, 270, 270)
            assert index == 0
            tensors[name] = tensor.numpy()
        assert not np.array_equal(tensors["native"], tensors["resize512_png"])
        assert not np.array_equal(tensors["native"], tensors["jpeg_q65"])


def test_fit_one_case() -> None:
    rng = np.random.default_rng(17)
    fit_values = np.r_[
        rng.normal(-1.0, 0.5, (80, 4)),
        rng.normal(1.0, 0.5, (80, 4)),
    ].astype(np.float32)
    calibration = {
        "native": rng.normal(-0.5, 0.8, (41, 4)).astype(np.float32),
        "jpeg_q80": rng.normal(0.5, 0.8, (41, 4)).astype(np.float32),
    }
    with tempfile.TemporaryDirectory() as temporary:
        args = Namespace(
            output_root=Path(temporary),
            max_gmm_fit_embeddings=100,
            seed=23,
            gmm_max_iter=20,
            gmm_n_init=1,
            primary_threshold_percentile=95,
        )
        result = fit_one_case(
            args,
            "test_k2_diag",
            fit_values,
            calibration,
            components=2,
            covariance_type="diag",
            reg_covar=1e-6,
            variants=("native", "jpeg_q80"),
            percentiles=(80, 95, 98),
        )
        model_path = Path(result["gmm"])
        assert model_path.is_file()
        threshold_path = Path(result["thresholds"])
        assert threshold_path.is_file()
        assert (model_path.parent / "calibration_fpr.csv").is_file()
        loaded = load_mixture(model_path)
        scores = mixture_nll(calibration["native"], loaded)
        assert scores.shape == (41,)
        assert np.isfinite(scores).all()


def main() -> None:
    test_mixture_scores()
    test_thresholds()
    test_stratified_limit()
    test_variant_dataset()
    test_fit_one_case()
    print("Augmented GMM tests passed")


if __name__ == "__main__":
    main()
