"""Independently verify the B-Free evaluation and summarize score behavior."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "source" / "gsd_bfree_samecat_origbg_best_bfree_valid_v1"
OLD = ROOT.parent / "gsd_bfree_valid_analysis" / "source" / "gsd_bfree_valid_v1" / "metrics.csv"


def metrics(frame: pd.DataFrame, threshold: float = 0.5) -> dict:
    y = frame.label.to_numpy(dtype=int)
    p = frame.prob_fake.to_numpy(dtype=float)
    pred = (p >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    rr = tn / (tn + fp)
    fr = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {
        "n": len(frame), "accuracy": (tn + tp) / len(frame),
        "balanced_accuracy": (rr + fr) / 2, "real_recall": rr,
        "fake_recall": fr, "fake_precision": precision, "fake_f1": f1,
        "auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def diagnostic_threshold(frame: pd.DataFrame) -> dict:
    y = frame.label.to_numpy(dtype=int)
    p = frame.prob_fake.to_numpy(dtype=float)
    fpr, tpr, thresholds = roc_curve(y, p)
    ba = (tpr + (1 - fpr)) / 2
    i = int(np.nanargmax(ba))
    return {"threshold": float(thresholds[i]), "balanced_accuracy": float(ba[i])}


def distribution(frame: pd.DataFrame) -> dict:
    result = {}
    for label, name in [(0, "real"), (1, "fake")]:
        x = frame.loc[frame.label == label, "prob_fake"].to_numpy(dtype=float)
        q = np.quantile(x, [0, .01, .05, .25, .5, .75, .95, .99, 1])
        result[name] = dict(zip(
            ["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"],
            map(float, q),
        ))
    return result


def paired_fake_delta(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> dict:
    a = left[left.label == 1][["coco_id", "prob_fake"]].rename(columns={"prob_fake": "left"})
    b = right[right.label == 1][["coco_id", "prob_fake"]].rename(columns={"prob_fake": "right"})
    pair = a.merge(b, on="coco_id", validate="one_to_one")
    delta = pair.right - pair.left
    return {
        "left": left_name, "right": right_name, "paired_ids": len(pair),
        "median_left": float(pair.left.median()), "median_right": float(pair.right.median()),
        "median_delta": float(delta.median()), "mean_delta": float(delta.mean()),
        "right_lower_fraction": float((delta < 0).mean()),
        "left_fake_recall_0.5": float((pair.left >= .5).mean()),
        "right_fake_recall_0.5": float((pair.right >= .5).mean()),
    }


def main() -> None:
    pred = pd.read_csv(RUN / "predictions.csv", dtype={"coco_id": str})
    partial = pd.read_csv(
        RUN / "predictions_SD2.1_inpainted_diffcat_origBG.partial.csv",
        dtype={"coco_id": str},
    )
    reported = pd.read_csv(RUN / "metrics.csv").set_index("branch")
    branches = list(reported.index)
    computed = {}
    distributions = {}
    max_delta = 0.0
    frames = {}
    for branch in branches:
        frame = pred[pred.generator == branch].copy()
        frames[branch] = frame
        computed[branch] = metrics(frame)
        computed[branch]["diagnostic_best_ba"] = diagnostic_threshold(frame)
        distributions[branch] = distribution(frame)
        for key in ["accuracy", "balanced_accuracy", "real_recall", "fake_recall",
                    "fake_precision", "fake_f1", "auc", "ap"]:
            max_delta = max(max_delta, abs(computed[branch][key] - float(reported.loc[branch, key])))

    partial_metrics = metrics(partial)
    partial_metrics["diagnostic_best_ba"] = diagnostic_threshold(partial)
    distributions["SD2.1_inpainted_diffcat_origBG_partial"] = distribution(partial)

    paired = [
        paired_fake_delta(frames["SD2.1_selfconditioned"],
                          frames["SD2.1_selfconditioned_origBG"],
                          "selfconditioned", "selfconditioned_origBG"),
        paired_fake_delta(frames["SD2.1_inpainted_samecat"],
                          frames["SD2.1_inpainted_samecat_origBG"],
                          "inpainted_samecat", "inpainted_samecat_origBG"),
        paired_fake_delta(frames["SD2.1_inpainted_diffcat"], partial,
                          "inpainted_diffcat", "inpainted_diffcat_origBG_partial"),
    ]

    old = pd.read_csv(OLD).set_index("branch") if OLD.is_file() else None
    comparison = []
    if old is not None:
        for branch in branches:
            comparison.append({
                "branch": branch,
                "old_tiny_trained_accuracy": float(old.loc[branch, "accuracy"]),
                "new_bfree_trained_accuracy": computed[branch]["accuracy"],
                "accuracy_change_pp": 100 * (computed[branch]["accuracy"] - float(old.loc[branch, "accuracy"])),
                "old_fake_recall": float(old.loc[branch, "fake_recall"]),
                "new_fake_recall": computed[branch]["fake_recall"],
                "fake_recall_change_pp": 100 * (computed[branch]["fake_recall"] - float(old.loc[branch, "fake_recall"])),
            })

    real_spread = pred[pred.label == 0].pivot(index="coco_id", columns="generator", values="prob_fake")
    result = {
        "integrity": {
            "combined_rows": len(pred), "partial_rows": len(partial),
            "combined_branches": branches,
            "per_branch_class_counts": pred.groupby(["generator", "label"]).size().unstack().to_dict("index"),
            "all_probabilities_finite": bool(np.isfinite(pred.prob_fake).all() and np.isfinite(partial.prob_fake).all()),
            "max_probability_sigmoid_margin_error": float(np.max(np.abs(
                pred.prob_fake.to_numpy() - 1 / (1 + np.exp(-pred.logit_margin.to_numpy()))
            ))),
            "max_reported_metric_delta": max_delta,
            "real_score_cross_branch_max_range": float((real_spread.max(axis=1) - real_spread.min(axis=1)).max()),
            "real_score_cross_branch_median_range": float((real_spread.max(axis=1) - real_spread.min(axis=1)).median()),
        },
        "computed_metrics": computed,
        "partial_metrics": partial_metrics,
        "distributions": distributions,
        "paired_fake_effects": paired,
        "comparison_to_tiny_trained_gsd": comparison,
    }
    (ROOT / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
