"""Build the self-contained Kaggle inference notebook for Full AIDE."""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
TRAIN_RUNTIME = HERE / "train_aide_original_full_tiny_commfor_kaggle.py"
TEST_RUNTIME = HERE / "test_aide_original_full_commfor_exact_kaggle.py"
CHECKPOINT = HERE / "artifacts" / "checkpoints" / "model_trainable.pt"
MANIFEST = (
    PROJECT_ROOT
    / "baselines"
    / "aide_original_forensic_resnet50"
    / "artifacts"
    / "runs"
    / "modal_download"
    / "20260927_001615"
    / "commfor_combined"
    / "dataset"
    / "selected_samples.csv"
)
OUTPUT_NOTEBOOK = HERE / "test_aide_original_full_commfor_exact_kaggle.ipynb"

EXPECTED_CHECKPOINT_SHA256 = (
    "62848895b44255d6a0567754c503a05807ed0a29b8046f3c6956a049844e47c3"
)
EXPECTED_MANIFEST_SHA256 = (
    "20b04a56d2709d091d51fe585621686b2cd792a373a20ec5068fc6a5be28b9cd"
)


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
    for path in (TRAIN_RUNTIME, TEST_RUNTIME, CHECKPOINT, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Local Full AIDE checkpoint hash mismatch")
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Established CommFor manifest hash mismatch")
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1000:
        raise RuntimeError(f"Expected 1,000 manifest rows, found {len(rows)}")

    train_blob = encoded(TRAIN_RUNTIME)
    test_blob = encoded(TEST_RUNTIME)
    manifest_blob = encoded(MANIFEST)

    cells = [
        markdown(
            """# Full AIDE — test-only trên đúng cohort CommFor cũ

Notebook này **không train lại**. Nó nạp checkpoint Full AIDE tốt nhất đã train trên Tiny-GenImage combined và dự đoán đúng cohort CommFor đã dùng để so sánh AIDE forensic với NPR-ResNet18:

- 100 ảnh real dùng chung;
- 9 generator × 100 ảnh fake = 900 ảnh fake;
- tổng cộng 1.000 ảnh;
- threshold cố định `0.5`, không tune trên CommFor.

Hai SHA-256 được khóa cứng để tránh test nhầm dữ liệu hoặc nhầm weight:

- checkpoint: `62848895b44255d6a0567754c503a05807ed0a29b8046f3c6956a049844e47c3`;
- manifest: `20b04a56d2709d091d51fe585621686b2cd792a373a20ec5068fc6a5be28b9cd`.

## Chuẩn bị trên Kaggle

1. Bật **GPU accelerator**.
2. Tạo Kaggle Dataset từ thư mục `baselines/aide_original_full/artifacts/checkpoints` và attach dataset đó vào notebook. Notebook tự tìm `model_trainable.pt` bằng checksum.
3. Bật Internet để tải `OwensLab/CommunityForensics-Eval` và OpenCLIP ConvNeXt-XXLarge. Nếu đã upload `open_clip_pytorch_model.bin` vào Kaggle Input thì notebook sẽ dùng file local.
4. Nếu dùng Hugging Face thường xuyên, nên tạo Kaggle Secret `HF_TOKEN`.

Kết quả cuối nằm tại `/kaggle/working/aide_original_full_commfor_exact_eval` và file ZIP tải về nằm tại `/kaggle/working/aide_original_full_commfor_exact_eval.zip`."""
        ),
        code(
            '%pip install -q "open_clip_torch==2.26.1" "datasets>=2.19,<4" '
            '"pandas<3.0" "scikit-learn<1.9" "pillow<12.0" "tqdm"'
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
    print("HF_TOKEN not configured; public downloads will be anonymous.")

assert torch.cuda.is_available(), "Kaggle Settings → Accelerator → GPU trước khi chạy."
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)"""
        ),
        code(
            """CONFIG = {
    "input_root": "/kaggle/input",
    "output_root": "/kaggle/working/aide_original_full_commfor_exact_eval",

    # None = tự tìm đúng model_trainable.pt bằng SHA-256 trong /kaggle/input.
    "checkpoint_path": None,

    # None = tự tìm open_clip_pytorch_model.bin; nếu không có sẽ tải từ HF.
    "semantic_checkpoint": None,

    "commfor_dataset_name": "OwensLab/CommunityForensics-Eval",
    "commfor_split": "CompEval",
    "commfor_streaming": True,
    "commfor_shuffle_seed": 43,
    "commfor_shuffle_buffer_size": 1000,
    "max_scan_records": 500000,

    "threshold": 0.5,
    "image_size": 256,
    "random_seed": 42,

    # Lưu predictions từng phần; chạy lại cell sẽ bỏ qua các ảnh đã hoàn thành.
    "resume": True,
    "save_every": 25,
}

CONFIG"""
        ),
        code(
            f'''import base64
import gzip
import importlib.util
import sys
from pathlib import Path

TRAIN_RUNTIME_GZIP_BASE64 = """{train_blob}"""
TEST_RUNTIME_GZIP_BASE64 = """{test_blob}"""
MANIFEST_GZIP_BASE64 = """{manifest_blob}"""

runtime_dir = Path("/kaggle/working/aide_original_full_commfor_runtime")
runtime_dir.mkdir(parents=True, exist_ok=True)
train_runtime_path = runtime_dir / "train_aide_original_full_tiny_commfor_kaggle.py"
test_runtime_path = runtime_dir / "test_aide_original_full_commfor_exact_kaggle.py"
manifest_path = runtime_dir / "selected_samples.csv"

train_runtime_path.write_bytes(gzip.decompress(base64.b64decode(TRAIN_RUNTIME_GZIP_BASE64)))
test_runtime_path.write_bytes(gzip.decompress(base64.b64decode(TEST_RUNTIME_GZIP_BASE64)))
manifest_path.write_bytes(gzip.decompress(base64.b64decode(MANIFEST_GZIP_BASE64)))

sys.path.insert(0, str(runtime_dir))
spec = importlib.util.spec_from_file_location(
    "aide_original_full_commfor_runner", test_runtime_path
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

CONFIG["manifest_path"] = str(manifest_path)
print("Runtime:", test_runtime_path)
print("Embedded exact manifest:", manifest_path)
print("Full AIDE CommFor inference runner imported successfully.")'''
        ),
        code(
            """result = runner.run_inference(CONFIG)
result["experiment"]"""
        ),
        code(
            """import json
import pandas as pd
from IPython.display import FileLink, display

run_dir = Path(CONFIG["output_root"])

print("CommFor overall: 100 real + 900 fake unique samples")
display(pd.DataFrame([json.loads((run_dir / "metrics/commfor_overall.json").read_text())]))

print("CommFor per generator: mỗi hàng = cùng 100 real + 100 fake của generator đó (%)")
per_generator = pd.read_csv(run_dir / "metrics/commfor_generator_metrics.csv")
metric_columns = [
    "accuracy", "balanced_accuracy", "real_recall", "fake_recall",
    "fake_precision", "fake_f1", "roc_auc", "average_precision",
]
shown = per_generator.copy()
for column in metric_columns:
    shown[column] = (100 * shown[column]).round(2)
display(shown)

print("Macro summary")
display(pd.DataFrame([json.loads((run_dir / "metrics/commfor_macro_summary.json").read_text())]))

print("Download:")
display(FileLink("/kaggle/working/aide_original_full_commfor_exact_eval.zip"))"""
        ),
        markdown(
            """## Output quan trọng

- `metrics/commfor_overall.json`: metric trên 1.000 ảnh unique (lưu ý tỷ lệ lớp 100 real / 900 fake).
- `metrics/commfor_macro_summary.json`: macro trung bình qua 9 cohort cân bằng theo generator.
- `metrics/commfor_generator_metrics.csv`: chi tiết từng generator, mỗi hàng gồm 100 shared real + 100 fake.
- `predictions/commfor_predictions.csv`: xác suất và dự đoán từng ảnh.
- `dataset/commfor_manifest_used.csv`: manifest chính xác đã sử dụng.
- `provenance/checkpoint.json`: checksum và metadata checkpoint.
- `aide_original_full_commfor_exact_eval.zip`: gói kết quả tải về."""
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
