"""Build a self-contained Kaggle notebook for GMM fitting and Tiny-GenImage evaluation."""

from __future__ import annotations

import base64
import io
import json
import textwrap
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = BASE_DIR / "forensic_twins_200k_gmm_eval_kaggle.ipynb"
SOURCE_FILES = [
    "fit_gmm_and_evaluate.py",
    "configs/kaggle_2xt4_200k.json",
    "src/__init__.py",
    "src/core.py",
    "src/data.py",
    "src/checkpointing.py",
    "tests/test_evaluation.py",
]


def archive_source() -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in SOURCE_FILES:
            path = BASE_DIR / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.writestr(relative, path.read_bytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    encoded = archive_source()
    chunks = textwrap.wrap(encoded, width=100)
    archive_literal = "\n".join(f'    "{chunk}"' for chunk in chunks)

    cells = [
        markdown(
            """# Forensic Twins 200K — Fit GMM và đánh giá Tiny-GenImage

Notebook này **không train lại encoder**. Nó sử dụng checkpoint 80 epoch để:

1. đọc checkpoint gần nhất của run 200K và manifest 200K ảnh real từ ImageNet + COCO;
2. chia cố định 160K GMM-fit và 40K real-calibration;
3. fit Gaussian K=1 full covariance trên embedding 192-D;
4. tạo ngưỡng NLL q80–q98, với q95 là ngưỡng chính;
5. đánh giá Tiny-GenImage ở native/raw, fake-codec-matched và JPEG Q95/Q80/Q65/Q50/Q35;
6. lưu predictions, AUROC, AP, metrics mọi threshold, confusion matrix và PCA theo generator.

Kaggle Inputs cần gắn:

- `hoangha071005/twins-gmm-200k-checkpoint`;
- ImageNet Object Localization Challenge;
- `awsaf49/coco-2017-dataset`;
- `yangsangtai/tiny-genimage`.
"""
        ),
        code(
            """# ========================= USER CONFIG =========================
# Để trống để notebook tự tìm đúng Kaggle Input.
CHECKPOINT_DATASET_ROOT = ""
TINY_GENIMAGE_ROOT = ""

# None = đánh giá toàn bộ. Đặt 500 để chạy thử nhanh trước.
MAX_EVAL_PER_CLASS = None

# Cả real và fake đều được nén lại cùng quality cho các protocol này.
JPEG_QUALITIES = [95, 80, 65, 50, 35]
RUN_FAKE_CODEC_MATCHED = True

# q95 là kết quả chính; toàn bộ q80..q98 vẫn được lưu.
PRIMARY_THRESHOLD_PERCENTILE = 95
THRESHOLD_PERCENTILES = list(range(80, 99))

INPUT_ROOT = "/kaggle/input"
OUTPUT_ROOT = "/kaggle/working/forensic_twins_200k_gmm_eval"
CODE_DIR = "/kaggle/working/forensic_twins_200k_eval_code"
print("Configuration loaded.")
"""
        ),
        code(
            """import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
print("Python:", sys.version)
subprocess.run(["nvidia-smi"], check=False)
"""
        ),
        markdown("## 1. Bung source evaluation được nhúng trong notebook\n"),
        code(
            f"""import base64
import io
import zipfile

SOURCE_ARCHIVE_B64 = (
{archive_literal}
)

code_root = Path(CODE_DIR)
if code_root.exists():
    shutil.rmtree(code_root)
code_root.mkdir(parents=True, exist_ok=True)
archive_bytes = base64.b64decode(SOURCE_ARCHIVE_B64)
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
    archive.extractall(code_root)

CONFIG_PATH = code_root / "configs" / "kaggle_2xt4_200k.json"
EVAL_SCRIPT = code_root / "fit_gmm_and_evaluate.py"
assert CONFIG_PATH.is_file() and EVAL_SCRIPT.is_file()
print("Source extracted:", code_root)
print("Embedded archive bytes:", len(archive_bytes))
"""
        ),
        markdown("## 2. Xác minh bốn Kaggle Inputs và artifact checkpoint\n"),
        code(
            """input_root = Path(INPUT_ROOT)

def first_directory(candidates, description):
    for candidate in candidates:
        if candidate.is_dir():
            print(f"{description}: {candidate}")
            return candidate
    raise FileNotFoundError(f"Không tìm thấy {description}. Đã kiểm tra: {candidates}")

if CHECKPOINT_DATASET_ROOT:
    checkpoint_dataset = first_directory([Path(CHECKPOINT_DATASET_ROOT)], "checkpoint dataset")
else:
    checkpoint_dataset = first_directory(
        [
            input_root / "twins-gmm-200k-checkpoint",
            input_root / "datasets" / "hoangha071005" / "twins-gmm-200k-checkpoint",
        ],
        "checkpoint dataset",
    )

if TINY_GENIMAGE_ROOT:
    tiny_root = first_directory([Path(TINY_GENIMAGE_ROOT)], "Tiny-GenImage")
else:
    tiny_root = first_directory(
        [
            input_root / "tiny-genimage",
            input_root / "datasets" / "yangsangtai" / "tiny-genimage",
        ],
        "Tiny-GenImage",
    )

imagenet_train = (
    input_root / "competitions" / "imagenet-object-localization-challenge"
    / "ILSVRC" / "Data" / "CLS-LOC" / "train"
)
coco_train = (
    input_root / "datasets" / "awsaf49" / "coco-2017-dataset"
    / "coco2017" / "train2017"
)
assert imagenet_train.is_dir(), f"Thiếu ImageNet train: {imagenet_train}"
assert coco_train.is_dir(), f"Thiếu COCO train: {coco_train}"

checkpoint_candidates = sorted(checkpoint_dataset.rglob("last.ckpt"))
assert checkpoint_candidates, f"Không tìm thấy last.ckpt trong {checkpoint_dataset}"
CHECKPOINT_PATH = checkpoint_candidates[0]
artifact_root = CHECKPOINT_PATH.parents[2]
MANIFEST_PATH = artifact_root / "manifests" / "train_real_200k.csv"
FRE_PATH = artifact_root / "cache" / "fre.pt"
for required in (CHECKPOINT_PATH, MANIFEST_PATH, FRE_PATH):
    assert required.is_file(), f"Thiếu artifact: {required}"

output_root = Path(OUTPUT_ROOT)
output_root.mkdir(parents=True, exist_ok=True)
print("Checkpoint:", CHECKPOINT_PATH, f"({CHECKPOINT_PATH.stat().st_size / 1024**2:.2f} MiB)")
print("Manifest:", MANIFEST_PATH)
print("FRE:", FRE_PATH)
print("ImageNet:", imagenet_train)
print("COCO:", coco_train)
print("Tiny-GenImage:", tiny_root)
print("Output:", output_root)
"""
        ),
        code(
            """# Kiểm tra manifest mà không mở/giải mã toàn bộ ảnh.
import csv
from collections import Counter

source_counts = Counter()
path_samples = []
row_count = 0
with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        row_count += 1
        source_counts[row["source"]] += 1
        if len(path_samples) < 20:
            path_samples.append(row["image_path"])

assert row_count == 200_000, f"Manifest phải có 200.000 dòng, hiện có {row_count}"
missing_samples = [path for path in path_samples if not Path(path).is_file()]
assert not missing_samples, f"Các đường dẫn mẫu không tồn tại: {missing_samples}"
print("Manifest rows:", row_count)
print("Sources:", dict(source_counts))
print("Manifest hash sẽ được đối chiếu với checkpoint; chỉ kiểm tra nhanh 20 đường dẫn mẫu.")
"""
        ),
        markdown("## 3. Chạy CPU tests cho GMM, threshold, JPEG matching và figures\n"),
        code(
            """subprocess.run(
    [sys.executable, str(code_root / "tests" / "test_evaluation.py")],
    cwd=code_root,
    check=True,
)
"""
        ),
        markdown("## 4. Fit GMM và hiệu chỉnh threshold bằng real-only calibration\n"),
        code(
            """percentile_arg = ",".join(map(str, THRESHOLD_PERCENTILES))
fit_command = [
    sys.executable,
    str(EVAL_SCRIPT),
    "--config", str(CONFIG_PATH),
    "--input-root", INPUT_ROOT,
    "--output-root", OUTPUT_ROOT,
    "--manifest", str(MANIFEST_PATH),
    "--checkpoint", str(CHECKPOINT_PATH),
    "--fre-checkpoint", str(FRE_PATH),
    "--fit-only",
    "--threshold-percentiles", percentile_arg,
    "--primary-threshold-percentile", str(PRIMARY_THRESHOLD_PERCENTILE),
]
print("Running:", " ".join(fit_command))
subprocess.run(fit_command, cwd=code_root, check=True)

GMM_PATH = output_root / "artifacts" / "gmm" / "real_k1_full.npz"
THRESHOLDS_PATH = output_root / "artifacts" / "gmm" / "thresholds.json"
assert GMM_PATH.is_file() and THRESHOLDS_PATH.is_file()
print("GMM ready:", GMM_PATH)
print(THRESHOLDS_PATH.read_text(encoding="utf-8"))
"""
        ),
        markdown("## 5. Dự đoán và đánh giá Tiny-GenImage\n"),
        code(
            """eval_command = [
    sys.executable,
    str(EVAL_SCRIPT),
    "--config", str(CONFIG_PATH),
    "--input-root", INPUT_ROOT,
    "--output-root", OUTPUT_ROOT,
    "--manifest", str(MANIFEST_PATH),
    "--checkpoint", str(CHECKPOINT_PATH),
    "--fre-checkpoint", str(FRE_PATH),
    "--tiny-root", str(tiny_root),
    "--gmm", str(GMM_PATH),
    "--thresholds", str(THRESHOLDS_PATH),
    "--eval-only",
    "--visualization-max-per-generator-class", "500",
    "--jpeg-qualities", ",".join(map(str, JPEG_QUALITIES)),
]
if not RUN_FAKE_CODEC_MATCHED:
    eval_command.append("--skip-fake-codec-matched")
if MAX_EVAL_PER_CLASS is not None:
    eval_command.extend(["--max-eval-per-class", str(MAX_EVAL_PER_CLASS)])
print("Running:", " ".join(eval_command))
subprocess.run(eval_command, cwd=code_root, check=True)

complete = output_root / "artifacts" / "evaluation_complete.json"
assert complete.is_file(), "Evaluation chưa tạo marker hoàn tất."
print(complete.read_text(encoding="utf-8"))
"""
        ),
        markdown("## 6. Hiển thị metrics và figures chính\n"),
        code(
            """import json
import pandas as pd
from IPython.display import Image as DisplayImage, display

metrics_root = output_root / "artifacts" / "metrics"
figures_root = output_root / "artifacts" / "figures"

comparison = pd.read_csv(metrics_root / "protocol_comparison.csv")
display(comparison)
per_generator_comparison = pd.read_csv(
    metrics_root / f"per_generator_protocol_comparison_q{PRIMARY_THRESHOLD_PERCENTILE}.csv"
)
display(per_generator_comparison)
for protocol in comparison["protocol"].tolist():
    print("=" * 90)
    print(protocol)
    summary = json.loads((metrics_root / f"{protocol}_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))
    table = pd.read_csv(metrics_root / f"{protocol}_threshold_metrics.csv")
    display(table[table["threshold_percentile"] == PRIMARY_THRESHOLD_PERCENTILE])
    for figure_name in (
        "pca_all_generators_true_labels.png",
        "nll_distributions_by_generator.png",
        "roc_pr_curves.png",
    ):
        display(DisplayImage(filename=str(figures_root / protocol / figure_name)))
"""
        ),
        markdown("## 7. Liệt kê và đóng gói artifact để tải về\n"),
        code(
            """for path in sorted((output_root / "artifacts").rglob("*")):
    if path.is_file():
        print(f"{path.relative_to(output_root)} | {path.stat().st_size / 1024**2:.2f} MiB")

archive_base = Path("/kaggle/working/forensic_twins_200k_gmm_eval_artifacts")
archive_path = shutil.make_archive(
    str(archive_base),
    "zip",
    root_dir=output_root,
    base_dir=".",
)
print("Download bundle:", archive_path, f"({Path(archive_path).stat().st_size / 1024**2:.2f} MiB)")
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {"accelerator": "gpu"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Built {NOTEBOOK} with {len(cells)} cells and {len(encoded)} base64 chars")


if __name__ == "__main__":
    main()
