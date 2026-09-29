"""Build the self-contained Tiny-GenImage JPEG-bias Kaggle notebook."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
RUNNER = HERE / "test_tiny_jpeg_bias_three_models_kaggle.py"
COMMON_RUNTIME = (
    PROJECT_ROOT
    / "benchmarks"
    / "commfor_unseen_three_models"
    / "code"
    / "test_three_models_commfor_unseen_kaggle.py"
)
AIDE_RUNTIME = (
    PROJECT_ROOT
    / "baselines"
    / "aide_original_full"
    / "train_aide_original_full_tiny_commfor_kaggle.py"
)
CHECKPOINT_DIR = HERE / "checkpoints"
OUTPUT_NOTEBOOK = HERE / "test_tiny_jpeg_bias_three_models_kaggle.ipynb"

EXPECTED_CHECKPOINTS = {
    "clip_linear_head.pt": "788461421cb585bbf489aabc76f3537058cd7753eccf9432d3a792622f738057",
    "npr_resnet18_from_scratch.pt": "328f9f431d378f9528c86966796e9f3ac604a5ce2bf3d307028062706c1537f0",
    "aide_original_full_trainable.pt": "62848895b44255d6a0567754c503a05807ed0a29b8046f3c6956a049844e47c3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encoded(path: Path) -> str:
    return base64.b64encode(gzip.compress(path.read_bytes(), mtime=0)).decode("ascii")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def main() -> None:
    for path in (RUNNER, COMMON_RUNTIME, AIDE_RUNTIME):
        if not path.is_file():
            raise FileNotFoundError(path)
    for filename, expected in EXPECTED_CHECKPOINTS.items():
        path = CHECKPOINT_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checkpoint hash mismatch for {filename}: {actual}")

    runner_blob = encoded(RUNNER)
    common_blob = encoded(COMMON_RUNTIME)
    aide_blob = encoded(AIDE_RUNTIME)

    cells = [
        markdown(
            """# Tiny-GenImage JPEG-bias evaluation — CLIP, NPR-ResNet18, Full AIDE

Notebook này **không train lại**. Nó giữ nguyên ba checkpoint đã train trên Tiny-GenImage và kiểm tra ảnh test theo các trường hợp:

1. `raw` — baseline bắt buộc;
2. `png_roundtrip` — decode → PNG → decode;
3. `fake_jpeg96` — giữ real gốc, chỉ nén fake JPEG quality 96;
4. `all_jpeg96` — nén cả real và fake JPEG quality 96;
5. `controlled_jpeg96` — real gốc có estimated quality 94–98 và fake lossless được nén JPEG96;
6. `raw_on_controlled_cohort` — baseline cùng membership với controlled cohort.

Notebook audit số ảnh, format, kích thước gốc, aspect ratio, megapixel, dung lượng file và JPEG quality trước khi inference. Mọi biến đổi codec diễn ra **trước preprocessing gốc của từng model** và không ghi đè Tiny-GenImage.

## Kaggle cần chuẩn bị

- GPU accelerator;
- Add Input `yangsangtai/tiny-genimage`;
- Add Input là Kaggle Dataset tạo từ folder `checkpoints/` đi kèm notebook;
- bật Internet để tải CLIP ViT-B/32 và OpenCLIP ConvNeXt-XXLarge, hoặc attach `open_clip_pytorch_model.bin`;
- tùy chọn: Kaggle Secret `HF_TOKEN`.

Full AIDE là **AIDE original Full**, không phải forensic-only và không phải AIDE–NPR."""
        ),
        code(
            '%pip install -q "open_clip_torch==2.26.1" "pandas<3.0" '
            '"scikit-learn<1.9" "pillow<12.0" "tqdm"'
        ),
        code(
            """import os
from pathlib import Path

import torch

try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("HF_TOKEN loaded from Kaggle Secrets.")
except Exception:
    print("HF_TOKEN is not configured; public downloads will be anonymous.")

assert torch.cuda.is_available(), "Kaggle Settings → Accelerator → GPU trước khi chạy."
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)"""
        ),
        code(
            """CONFIG = {
    "input_root": "/kaggle/input",
    "dataset_root": None,  # None = tự tìm Tiny-GenImage.
    "output_root": "/kaggle/working/tiny_jpeg_bias_three_models_results",

    # Full run: 500 real + 500 fake mỗi generator. Đổi smoke=True để thử 100+100.
    "smoke": False,
    "max_per_class_per_generator": None,
    "selection_seed": 42,

    "models": ["clip_linear_probe", "npr_resnet18", "aide_original_full"],
    "cases": [
        "raw",
        "png_roundtrip",
        "fake_jpeg96",
        "all_jpeg96",
        "raw_on_controlled_cohort",
        "controlled_jpeg96",
    ],

    "jpeg_quality": 96,
    "jpeg_subsampling": 0,
    "jpeg_optimize": False,
    "jpeg_progressive": False,
    "controlled_real_quality_min": 94,
    "controlled_real_quality_max": 98,
    "controlled_balance_across_generators": True,

    "clip_batch_size": 32,
    "npr_batch_size": 32,
    "aide_batch_size": 4,
    "num_workers": 2,
    "threshold": 0.5,

    # Notebook lưu partial CSV. Chạy lại cell sẽ bỏ qua sample đã hoàn thành.
    "resume": True,
    "save_every_batches": 25,

    # None = tự tìm chính xác các filename trong /kaggle/input và xác minh SHA256.
    "clip_checkpoint": None,
    "npr_checkpoint": None,
    "aide_checkpoint": None,
    "verify_checkpoint_hashes": True,

    # None = tự tìm open_clip_pytorch_model.bin; nếu không có sẽ tải từ HF.
    "aide_semantic_checkpoint": None,
    "clip_pretrained": "openai",
}

CONFIG"""
        ),
        code(
            f'''import base64
import gzip
import importlib.util
import sys
from pathlib import Path

RUNNER_GZIP_BASE64 = """{runner_blob}"""
COMMON_RUNTIME_GZIP_BASE64 = """{common_blob}"""
AIDE_RUNTIME_GZIP_BASE64 = """{aide_blob}"""

runtime_dir = Path("/kaggle/working/tiny_jpeg_bias_runtime")
runtime_dir.mkdir(parents=True, exist_ok=True)
runner_path = runtime_dir / "test_tiny_jpeg_bias_three_models_kaggle.py"
common_path = runtime_dir / "test_three_models_commfor_unseen_kaggle.py"
aide_path = runtime_dir / "train_aide_original_full_tiny_commfor_kaggle.py"

runner_path.write_bytes(gzip.decompress(base64.b64decode(RUNNER_GZIP_BASE64)))
common_path.write_bytes(gzip.decompress(base64.b64decode(COMMON_RUNTIME_GZIP_BASE64)))
aide_path.write_bytes(gzip.decompress(base64.b64decode(AIDE_RUNTIME_GZIP_BASE64)))

sys.path.insert(0, str(runtime_dir))
spec = importlib.util.spec_from_file_location("tiny_jpeg_bias_runner", runner_path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

CONFIG["common_runtime_path"] = str(common_path)
CONFIG["aide_runtime_path"] = str(aide_path)
print("Runner:", runner_path)
print("Common runtime:", common_path)
print("Full AIDE runtime:", aide_path)'''
        ),
        code(
            """result = runner.run_evaluation(CONFIG)
result"""
        ),
        code(
            """import json
import pandas as pd
from IPython.display import Image as DisplayImage, display

run_dir = Path(CONFIG["output_root"])

print("Dataset summary")
display(pd.DataFrame([json.loads((run_dir / "dataset/overall_statistics.json").read_text())]))

print("Generator / label counts")
display(pd.read_csv(run_dir / "dataset/generator_label_counts.csv"))

print("Native formats")
display(pd.read_csv(run_dir / "dataset/format_distribution.csv"))

print("Case sample counts")
display(pd.read_csv(run_dir / "dataset/case_sample_counts.csv"))

print("Top native image sizes")
sizes = pd.read_csv(run_dir / "dataset/exact_size_distribution.csv")
display(sizes.groupby(["native_width", "native_height", "native_size"], as_index=False)["count"].sum().sort_values("count", ascending=False).head(30))

for plot_name in [
    "format_by_label.png",
    "top_native_sizes.png",
    "size_bins_by_label.png",
    "aspect_ratio_by_label.png",
    "jpeg_quality_by_label.png",
]:
    plot_path = run_dir / "plots" / plot_name
    if plot_path.is_file():
        print(plot_name)
        display(DisplayImage(filename=str(plot_path)))"""
        ),
        code(
            """print("Overall metrics (%)")
overall = pd.read_csv(run_dir / "metrics/overall_by_model_case.csv")
metric_columns = [
    "accuracy", "balanced_accuracy", "real_recall", "fake_recall",
    "fake_precision", "fake_f1", "roc_auc", "average_precision",
]
shown = overall.copy()
for column in metric_columns:
    shown[column] = (100 * shown[column]).round(2)
display(shown[["model", "case", "num_samples", *metric_columns]])

print("Delta versus matching raw cohort — percentage points")
delta = pd.read_csv(run_dir / "metrics/delta_from_matching_raw.csv")
delta_columns = [column for column in delta.columns if column.startswith("delta_")]
shown_delta = delta.copy()
for column in delta_columns:
    shown_delta[column] = (100 * shown_delta[column]).round(2)
display(shown_delta)

print("Per-generator metrics")
display(pd.read_csv(run_dir / "metrics/generator_by_model_case.csv"))"""
        ),
        code(
            """import shutil
from IPython.display import FileLink, display

archive_base = "/kaggle/working/tiny_jpeg_bias_three_models_results"
archive_path = shutil.make_archive(archive_base, "zip", root_dir=run_dir)
print("Download complete result package:")
display(FileLink(archive_path))"""
        ),
        markdown(
            """## Output chính

- `dataset/image_audit.csv`: audit từng ảnh test.
- `dataset/exact_size_distribution.csv`: toàn bộ kích thước `W×H`.
- `dataset/controlled_jpeg96_manifest.csv`: membership của controlled cohort.
- `predictions/<model>/<case>.csv`: prediction từng model/case.
- `metrics/overall_by_model_case.csv`: metric tổng quan.
- `metrics/generator_by_model_case.csv`: metric từng generator.
- `metrics/delta_from_matching_raw.csv`: mức thay đổi so với raw tương ứng.
- `metrics/paired_probability_shift.csv`: thay đổi xác suất trên cùng sample.
- `tiny_jpeg_bias_three_models_results.zip`: gói tải về.

`all_jpeg96` có thể tạo double-compression trên real JPEG; kết luận bias chính nên dựa vào `fake_jpeg96` và cặp `raw_on_controlled_cohort ↔ controlled_jpeg96`."""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"Wrote {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
