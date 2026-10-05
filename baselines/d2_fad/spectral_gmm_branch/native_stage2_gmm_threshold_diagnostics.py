"""Tiny-GenImage separation diagnostics for the native hierarchical Spectral-GMM.

This module is embedded after ``train_spectral_gmm_native_hierarchical_kaggle.py``
inside the Kaggle notebook.  It never trains Stage 1.  It restores the frozen
best Local encoder, the best Stage-2 checkpoint, and the fitted real-only GMM.

Two threshold families are deliberately kept separate:

* real-only q90/q95/q99 thresholds from ImageNet calibration are legitimate
  operating points that do not inspect Tiny-GenImage labels;
* the exact best-balanced-accuracy threshold on Tiny-GenImage is an oracle
  diagnostic.  It measures separability but is not a deployment threshold.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
from tqdm.auto import tqdm


# When this file runs as a normal module, import the core implementation.  In
# the generated notebook those names already exist because the complete core
# runner is embedded in the preceding cell.
if "GlobalImageCLSBottleneck" not in globals():
    from train_spectral_gmm_native_hierarchical_kaggle import (  # type: ignore
        GlobalImageCLSBottleneck,
        encode_band_cls,
        load_frozen_local_encoder,
        load_native_tiles,
        read_json,
        sha256_file,
        stable_seed,
        torch_load,
        write_json,
    )


LABEL_DIR_TO_ID = {
    "nature": 0,
    "real": 0,
    "0_real": 0,
    "0-real": 0,
    "0": 0,
    "ai": 1,
    "fake": 1,
    "1_fake": 1,
    "1-fake": 1,
    "1": 1,
}

DIAGNOSTIC_DEFAULTS: dict[str, Any] = {
    "input_root": "/kaggle/input",
    "tiny_dataset_root": None,
    "test_split_aliases": ["test", "validation", "val", "valid"],
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
    "balance_per_generator": True,
    "max_per_class_per_generator": None,
    "diagnostic_save_every_images": 50,
    "diagnostic_resume": True,
}


def dataframe_sha256(frame: pd.DataFrame) -> str:
    import hashlib

    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def find_split_dir(generator_dir: Path, aliases: Sequence[str]) -> Path | None:
    children = {
        child.name.lower(): child
        for child in generator_dir.iterdir()
        if child.is_dir()
    }
    for alias in aliases:
        candidate = children.get(str(alias).lower())
        if candidate is None:
            continue
        if any(
            child.is_dir() and child.name.lower() in LABEL_DIR_TO_ID
            for child in candidate.iterdir()
        ):
            return candidate
    return None


def generator_dirs_at(root: Path, aliases: Sequence[str]) -> list[Path]:
    try:
        children = [child for child in root.iterdir() if child.is_dir()]
    except OSError:
        return []
    return sorted(
        [child for child in children if find_split_dir(child, aliases) is not None],
        key=lambda path: path.name.lower(),
    )


def discover_tiny_root(config: dict[str, Any]) -> tuple[Path, list[Path]]:
    aliases = list(config["test_split_aliases"])
    explicit = config.get("tiny_dataset_root")
    if explicit:
        root = Path(explicit)
        generators = generator_dirs_at(root, aliases)
        if not generators:
            raise FileNotFoundError(f"No Tiny-GenImage generators under {root}")
        return root.resolve(), generators

    input_root = Path(config["input_root"])
    best_root: Path | None = None
    best_generators: list[Path] = []
    for current, dirnames, _ in os.walk(input_root):
        root = Path(current)
        generators = generator_dirs_at(root, aliases)
        if len(generators) > len(best_generators):
            best_root, best_generators = root, generators
        if generators:
            dirnames[:] = []
    if best_root is None or not best_generators:
        raise FileNotFoundError(
            "Cannot locate Tiny-GenImage under /kaggle/input. Attach "
            "yangsangtai/tiny-genimage with generator/{val|test}/{nature|ai}."
        )
    return best_root.resolve(), best_generators


def iter_image_files(path: Path, extensions: set[str]) -> Iterable[Path]:
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in extensions:
            yield item


def deterministic_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, replace=False, random_state=seed)


def build_tiny_manifest(
    config: dict[str, Any], output_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tiny_root, generators = discover_tiny_root(config)
    aliases = list(config["test_split_aliases"])
    extensions = {str(value).lower() for value in config["image_extensions"]}
    records: list[dict[str, Any]] = []
    actual_splits: dict[str, str] = {}

    for generator_dir in generators:
        split_dir = find_split_dir(generator_dir, aliases)
        assert split_dir is not None
        actual_splits[generator_dir.name] = split_dir.name
        for label_dir in sorted(child for child in split_dir.iterdir() if child.is_dir()):
            label = LABEL_DIR_TO_ID.get(label_dir.name.lower())
            if label is None:
                continue
            for image_path in iter_image_files(label_dir, extensions):
                relative = image_path.relative_to(tiny_root).as_posix()
                records.append(
                    {
                        "sample_id": relative,
                        "image_path": str(image_path),
                        "relative_path": relative,
                        "generator": generator_dir.name,
                        "actual_split": split_dir.name,
                        "label": int(label),
                        "label_name": "fake" if label == 1 else "real",
                        "extension": image_path.suffix.lower(),
                    }
                )

    full = pd.DataFrame(records)
    if full.empty:
        raise RuntimeError(f"No Tiny-GenImage test images found under {tiny_root}")
    if full.sample_id.duplicated().any():
        raise RuntimeError("Tiny-GenImage manifest contains duplicate sample_id values.")

    selected_parts: list[pd.DataFrame] = []
    for generator, group in full.groupby("generator", sort=True):
        real = group[group.label == 0]
        fake = group[group.label == 1]
        if real.empty or fake.empty:
            raise RuntimeError(f"Generator {generator} lacks real or fake samples.")
        if config["balance_per_generator"]:
            count = min(len(real), len(fake))
            limit = config.get("max_per_class_per_generator")
            if limit is not None:
                count = min(count, int(limit))
            real = deterministic_sample(
                real, count, stable_seed(config["seed"], generator, "real")
            )
            fake = deterministic_sample(
                fake, count, stable_seed(config["seed"], generator, "fake")
            )
        selected_parts.extend([real, fake])

    selected = pd.concat(selected_parts, ignore_index=True).sort_values(
        ["generator", "label", "relative_path"], ignore_index=True
    )
    selected["evaluation_cohort"] = "tiny_balanced_per_generator"

    dataset_dir = output_root / "tiny_diagnostic" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(dataset_dir / "tiny_manifest_full.csv", index=False)
    selected.to_csv(dataset_dir / "tiny_manifest_used.csv", index=False)
    counts = (
        selected.groupby(["generator", "label_name"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    counts.to_csv(dataset_dir / "counts_by_generator.csv", index=False)
    summary = {
        "tiny_dataset_root": str(tiny_root),
        "actual_split_by_generator": actual_splits,
        "num_generators": int(selected.generator.nunique()),
        "num_samples": int(len(selected)),
        "num_real": int((selected.label == 0).sum()),
        "num_fake": int((selected.label == 1).sum()),
        "balance_per_generator": bool(config["balance_per_generator"]),
        "max_per_class_per_generator": config.get("max_per_class_per_generator"),
        "manifest_sha256": dataframe_sha256(selected),
    }
    write_json(dataset_dir / "manifest_summary.json", summary)
    return selected, summary


def load_native_runtime(
    config: dict[str, Any], output_root: Path, stage1_root: Path
) -> dict[str, Any]:
    local_path = stage1_root / "checkpoints" / "stage1_local_encoder_best.pt"
    global_path = output_root / "checkpoints" / "stage2_global_best.pt"
    scaler_path = output_root / "gmm" / "stage3_real_feature_scaler.joblib"
    gmm_path = output_root / "gmm" / "stage3_real_distribution_gmm.joblib"
    threshold_path = output_root / "gmm" / "stage3_real_only_thresholds.json"
    required = [local_path, global_path, scaler_path, gmm_path, threshold_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing trained artifacts: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model = load_frozen_local_encoder(local_path, device)
    global_state = torch_load(global_path)
    global_model = GlobalImageCLSBottleneck(config).to(device)
    global_model.load_state_dict(global_state["model"], strict=True)
    global_model.eval()
    scaler = joblib.load(scaler_path)
    gmm = joblib.load(gmm_path)
    thresholds = read_json(threshold_path)
    if int(scaler.n_features_in_) != 768 or int(gmm.means_.shape[1]) != 768:
        raise RuntimeError(
            f"Expected 768-D Stage-3 artifacts, got scaler={scaler.n_features_in_}, "
            f"gmm={gmm.means_.shape}"
        )
    return {
        "device": device,
        "local_model": local_model,
        "global_model": global_model,
        "scaler": scaler,
        "gmm": gmm,
        "thresholds": thresholds,
        "stage1_checkpoint": str(local_path),
        "stage2_checkpoint": str(global_path),
        "stage1_sha256": sha256_file(local_path),
        "stage2_sha256": sha256_file(global_path),
        "stage2_epoch": int(global_state["epoch"]) + 1,
        "stage2_best_validation": float(global_state["best_validation"]),
    }


@torch.inference_mode()
def score_tiny_manifest(
    manifest: pd.DataFrame,
    runtime: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
) -> pd.DataFrame:
    prediction_dir = output_root / "tiny_diagnostic" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    partial_path = prediction_dir / "tiny_predictions.partial.csv"
    final_path = prediction_dir / "tiny_predictions.csv"

    completed = pd.DataFrame()
    done: set[str] = set()
    if bool(config["diagnostic_resume"]) and partial_path.is_file():
        completed = pd.read_csv(partial_path)
        done = set(completed.sample_id.astype(str))
        print(f"Resuming {len(done):,} completed Tiny images.", flush=True)

    remaining = manifest[~manifest.sample_id.astype(str).isin(done)]
    rows: list[dict[str, Any]] = []
    device = runtime["device"]
    local_model = runtime["local_model"]
    global_model = runtime["global_model"]
    scaler = runtime["scaler"]
    gmm = runtime["gmm"]
    save_every = int(config["diagnostic_save_every_images"])

    def save_partial() -> None:
        frames = [frame for frame in (completed, pd.DataFrame(rows)) if not frame.empty]
        if frames:
            pd.concat(frames, ignore_index=True).drop_duplicates(
                "sample_id", keep="last"
            ).to_csv(partial_path, index=False)

    progress = tqdm(
        remaining.itertuples(index=False),
        total=len(remaining),
        desc="Native Stage2-GMM Tiny diagnostic",
        dynamic_ncols=True,
    )
    for index, row in enumerate(progress, start=1):
        loaded = load_native_tiles(
            Path(row.image_path), str(row.sample_id), config, "jpeg"
        )
        low = encode_band_cls(
            loaded["tiles"], "low", local_model, config, device
        )
        high = encode_band_cls(
            loaded["tiles"], "high", local_model, config, device
        )
        positions = loaded["positions"].unsqueeze(0).to(device)
        valid = torch.ones(1, len(low), dtype=torch.bool, device=device)
        with torch.autocast(
            device_type=device.type,
            enabled=bool(config["amp"] and device.type == "cuda"),
        ):
            image_cls, _, _, _ = global_model.encode(
                low.unsqueeze(0).to(device),
                high.unsqueeze(0).to(device),
                positions,
                valid,
                masked=None,
            )
        vector = image_cls.float().cpu().numpy()
        standardized = scaler.transform(vector)
        score = float(-gmm.score_samples(standardized)[0])
        component = int(gmm.predict(standardized)[0])
        responsibility = float(gmm.predict_proba(standardized)[0].max())
        layout = loaded["layout"]
        rows.append(
            {
                "sample_id": str(row.sample_id),
                "relative_path": str(row.relative_path),
                "image_path": str(row.image_path),
                "generator": str(row.generator),
                "actual_split": str(row.actual_split),
                "evaluation_cohort": str(row.evaluation_cohort),
                "label": int(row.label),
                "label_name": str(row.label_name),
                "extension": str(row.extension),
                "width": int(layout["original_width"]),
                "height": int(layout["original_height"]),
                "num_tiles": int(len(loaded["tiles"])),
                "jpeg_applied": bool(loaded["jpeg_applied"]),
                "jpeg_quality": int(loaded["jpeg_quality"]),
                "jpeg_subsampling": int(loaded["jpeg_subsampling"]),
                "nll_anomaly_score": score,
                "gmm_component": component,
                "gmm_max_responsibility": responsibility,
            }
        )
        if index % save_every == 0:
            save_partial()
        if index % 250 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_partial()
    result = pd.read_csv(partial_path).sort_values(
        ["generator", "label", "relative_path"], ignore_index=True
    )
    if len(result) != len(manifest):
        raise RuntimeError(
            f"Expected {len(manifest)} predictions, found {len(result)}"
        )
    result.to_csv(final_path, index=False)
    return result


def safe_metric(function: Any, *args: Any, **kwargs: Any) -> float:
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return float("nan")


def metrics_at_threshold(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    labels = frame.label.to_numpy(dtype=int)
    scores = frame.nll_anomaly_score.to_numpy(dtype=float)
    predictions = (scores >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "num_samples": int(len(frame)),
        "num_real": int((labels == 0).sum()),
        "num_fake": int((labels == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": safe_metric(accuracy_score, labels, predictions),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, labels, predictions),
        "real_recall": float(tn / max(tn + fp, 1)),
        "fake_recall": safe_metric(recall_score, labels, predictions, zero_division=0),
        "fake_precision": safe_metric(precision_score, labels, predictions, zero_division=0),
        "fake_f1": safe_metric(f1_score, labels, predictions, zero_division=0),
        "real_fpr": float(fp / max(tn + fp, 1)),
        "roc_auc": safe_metric(roc_auc_score, labels, scores),
        "average_precision": safe_metric(average_precision_score, labels, scores),
        "mean_real_nll": float(scores[labels == 0].mean()),
        "mean_fake_nll": float(scores[labels == 1].mean()),
        "threshold": float(threshold),
    }


def exact_oracle_threshold(frame: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    labels = frame.label.to_numpy(dtype=int)
    scores = frame.nll_anomaly_score.to_numpy(dtype=float)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    sweep = pd.DataFrame(
        {
            "threshold": thresholds[finite],
            "real_recall": 1.0 - fpr[finite],
            "fake_recall": tpr[finite],
        }
    )
    sweep["balanced_accuracy"] = 0.5 * (
        sweep.real_recall + sweep.fake_recall
    )
    sweep["recall_gap"] = (sweep.real_recall - sweep.fake_recall).abs()
    best = sweep.sort_values(
        ["balanced_accuracy", "recall_gap", "threshold"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    return float(best.threshold), sweep


def distribution_separation(frame: pd.DataFrame) -> dict[str, Any]:
    real = frame.loc[frame.label == 0, "nll_anomaly_score"].to_numpy(float)
    fake = frame.loc[frame.label == 1, "nll_anomaly_score"].to_numpy(float)
    pooled = np.sqrt(
        ((len(real) - 1) * real.var(ddof=1) + (len(fake) - 1) * fake.var(ddof=1))
        / max(len(real) + len(fake) - 2, 1)
    )
    cohens_d = float((fake.mean() - real.mean()) / max(pooled, 1e-12))
    lower, upper = np.quantile(np.concatenate([real, fake]), [0.001, 0.999])
    bins = np.linspace(lower, upper, 201)
    real_hist, _ = np.histogram(real, bins=bins, density=True)
    fake_hist, _ = np.histogram(fake, bins=bins, density=True)
    widths = np.diff(bins)
    overlap = float(np.sum(np.minimum(real_hist, fake_hist) * widths))
    ks = ks_2samp(real, fake, alternative="two-sided", method="auto")
    return {
        "mean_real_nll": float(real.mean()),
        "mean_fake_nll": float(fake.mean()),
        "fake_minus_real_mean": float(fake.mean() - real.mean()),
        "cohens_d_fake_minus_real": cohens_d,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "histogram_overlap_coefficient": overlap,
        "interpretation": "lower overlap and higher AUC/KS indicate stronger separation",
    }


def save_diagnostic_plots(
    predictions: pd.DataFrame,
    generator_metrics: pd.DataFrame,
    threshold_rows: pd.DataFrame,
    oracle_threshold: float,
    q95_threshold: float,
    sweep: pd.DataFrame,
    output_root: Path,
) -> None:
    plot_dir = output_root / "tiny_diagnostic" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = predictions.label.to_numpy(int)
    scores = predictions.nll_anomaly_score.to_numpy(float)

    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f"AUC={auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"AP={ap:.4f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "roc_pr.png", dpi=180)
    plt.close(fig)

    low, high = np.quantile(scores, [0.01, 0.99])
    bins = np.linspace(low, high, 61)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, name, color in [(0, "real", "tab:blue"), (1, "fake", "tab:orange")]:
        values = scores[labels == label]
        values = values[(values >= low) & (values <= high)]
        ax.hist(values, bins=bins, density=True, alpha=0.42, label=name, color=color)
    ax.axvline(q95_threshold, color="purple", linestyle=":", linewidth=2, label="real-only q95")
    ax.axvline(oracle_threshold, color="red", linestyle="--", linewidth=2, label="Tiny oracle")
    ax.set(xlabel="GMM NLL anomaly score (display 1-99%)", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "nll_real_fake.png", dpi=180)
    plt.close(fig)

    generators = sorted(predictions.generator.unique())
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    pooled_real = predictions.loc[predictions.label == 0, "nll_anomaly_score"].to_numpy(float)
    pooled_real = pooled_real[(pooled_real >= low) & (pooled_real <= high)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(generators)))
    for ax, generator, color in zip(axes, generators, colors):
        fake = predictions.loc[
            (predictions.generator == generator) & (predictions.label == 1),
            "nll_anomaly_score",
        ].to_numpy(float)
        fake = fake[(fake >= low) & (fake <= high)]
        ax.hist(pooled_real, bins=bins, density=True, alpha=0.20, color="tab:blue")
        ax.hist(fake, bins=bins, density=True, histtype="step", linewidth=1.8, color=color)
        ax.axvline(oracle_threshold, color="red", linestyle="--", linewidth=1.2)
        metric = generator_metrics[generator_metrics.generator == generator].iloc[0]
        name = str(generator).replace("imagenet_ai_", "").replace("imagenet_", "")
        ax.set_title(
            f"{name}\nBA={metric.balanced_accuracy:.3f} AUC={metric.roc_auc:.3f}",
            fontsize=10,
        )
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[len(generators):]:
        ax.axis("off")
    fig.supxlabel("GMM NLL anomaly score")
    fig.supylabel("Density")
    fig.tight_layout()
    fig.savefig(plot_dir / "nll_by_generator.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep.threshold, sweep.balanced_accuracy, label="Balanced accuracy")
    ax.plot(sweep.threshold, sweep.real_recall, label="Real recall", alpha=0.8)
    ax.plot(sweep.threshold, sweep.fake_recall, label="Fake recall", alpha=0.8)
    ax.axvline(oracle_threshold, color="red", linestyle="--", label="Tiny oracle")
    ax.axvline(q95_threshold, color="purple", linestyle=":", label="real-only q95")
    ax.set_xlim(low, high)
    ax.set(xlabel="Threshold", ylabel="Metric", ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "threshold_tradeoff.png", dpi=180)
    plt.close(fig)

    ordered = generator_metrics.sort_values("balanced_accuracy")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(ordered))))
    ax.barh(ordered.generator, ordered.balanced_accuracy)
    ax.set(xlim=(0, 1), xlabel="Balanced accuracy at global Tiny oracle threshold")
    fig.tight_layout()
    fig.savefig(plot_dir / "generator_balanced_accuracy.png", dpi=180)
    plt.close(fig)


def evaluate_tiny_separation(
    predictions: pd.DataFrame,
    runtime: dict[str, Any],
    manifest_summary: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    metric_dir = output_root / "tiny_diagnostic" / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    oracle_threshold, sweep = exact_oracle_threshold(predictions)
    sweep.to_csv(metric_dir / "oracle_threshold_sweep.csv", index=False)

    threshold_rows: list[dict[str, Any]] = []
    for name in ("nll_q90", "nll_q95", "nll_q99"):
        threshold_rows.append(
            {
                "threshold_name": f"real_calibration_{name}",
                "threshold_source": "ImageNet real calibration only",
                **metrics_at_threshold(predictions, float(runtime["thresholds"][name])),
            }
        )
    threshold_rows.append(
        {
            "threshold_name": "tiny_oracle_best_balanced_accuracy",
            "threshold_source": "Tiny labels; diagnostic only",
            **metrics_at_threshold(predictions, oracle_threshold),
        }
    )
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(metric_dir / "threshold_comparison.csv", index=False)

    generator_rows: list[dict[str, Any]] = []
    for generator, group in predictions.groupby("generator", sort=True):
        generator_rows.append(
            {
                "generator": str(generator),
                **metrics_at_threshold(group, oracle_threshold),
            }
        )
    generator_metrics = pd.DataFrame(generator_rows)
    generator_metrics.to_csv(metric_dir / "generator_metrics_oracle.csv", index=False)

    separation = distribution_separation(predictions)
    oracle = threshold_rows[-1]
    macro = {
        "macro_generator_balanced_accuracy": float(generator_metrics.balanced_accuracy.mean()),
        "macro_generator_fake_f1": float(generator_metrics.fake_f1.mean()),
        "macro_generator_roc_auc": float(generator_metrics.roc_auc.mean()),
        "worst_generator_balanced_accuracy": float(generator_metrics.balanced_accuracy.min()),
        "best_generator_balanced_accuracy": float(generator_metrics.balanced_accuracy.max()),
    }
    summary = {
        "purpose": "diagnose whether Stage1-epoch5 plus Stage2-GMM separates Tiny real/fake",
        "score_direction": "higher NLL means more anomalous/fake",
        "manifest": manifest_summary,
        "stage1_checkpoint_sha256": runtime["stage1_sha256"],
        "stage2_checkpoint_sha256": runtime["stage2_sha256"],
        "stage2_best_epoch": runtime["stage2_epoch"],
        "stage2_best_validation": runtime["stage2_best_validation"],
        "gmm_components": int(runtime["gmm"].n_components),
        "real_only_thresholds": runtime["thresholds"],
        "tiny_oracle": oracle,
        "separation": separation,
        "macro": macro,
        "methodological_warning": (
            "The Tiny oracle threshold uses Tiny test labels. Use it only to measure "
            "separability; do not report it as an independently calibrated deployment threshold."
        ),
    }
    write_json(metric_dir / "separation_summary.json", summary)
    save_diagnostic_plots(
        predictions,
        generator_metrics,
        threshold_frame,
        oracle_threshold,
        float(runtime["thresholds"]["nll_q95"]),
        sweep,
        output_root,
    )
    return summary


def run_tiny_threshold_diagnostic(
    training_config: dict[str, Any],
    stage1_root: str | Path,
    output_root: str | Path,
    diagnostic_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(DIAGNOSTIC_DEFAULTS)
    config.update(training_config)
    if diagnostic_overrides:
        config.update(diagnostic_overrides)
    output_root = Path(output_root)
    stage1_root = Path(stage1_root)
    manifest, manifest_summary = build_tiny_manifest(config, output_root)
    runtime = load_native_runtime(config, output_root, stage1_root)
    predictions = score_tiny_manifest(manifest, runtime, config, output_root)
    summary = evaluate_tiny_separation(
        predictions, runtime, manifest_summary, output_root
    )
    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary

