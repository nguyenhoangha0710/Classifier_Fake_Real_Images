"""Build a self-contained Kaggle notebook for frozen-GMM B-Free evaluation."""

from __future__ import annotations

import base64
import io
import json
import textwrap
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = BASE_DIR / "forensic_twins_bfree_eval_kaggle.ipynb"
SOURCE_FILES = [
    "test_bfree_gmm_kaggle.py",
    "fit_gmm_and_evaluate.py",
    "configs/kaggle_2xt4_200k.json",
    "src/__init__.py",
    "src/core.py",
    "src/data.py",
    "src/checkpointing.py",
    "tests/test_bfree_evaluation.py",
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
            """# Forensic Twins 200K — Zero-shot B-Free evaluation

Notebook này không train lại encoder và không fit lại GMM. Nó:

1. đọc `last.ckpt`, `fre.pt`, fitted GMM và q80–q98 từ hai checkpoint Dataset;
2. đối chiếu COCO ID của B-Free với manifest real 200K;
3. chỉ giữ B-Free ID không overlap và lưu manifest để tái sử dụng;
4. đánh giá sáu fake variant theo từng cặp cân bằng với cùng real cohort;
5. chạy native PNG, common JPEG, resize và blur;
6. lưu AUROC/AP, q80/q86/q95, confusion matrix, bootstrap CI và PCA.

Kaggle Inputs bắt buộc:

- `baopv051/b-free-training-data`;
- `hoangha071005/twins-gmm-200k-checkpoint`;
- `hoangha071005/fit-gmm-checkpoint-200k`.

Không cần gắn ImageNet, COCO hoặc Tiny-GenImage.
"""
        ),
        code(
            """# ========================= USER CONFIG =========================
INPUT_ROOT = "/kaggle/input"
OUTPUT_ROOT = "/kaggle/working/forensic_twins_bfree_eval"
CODE_DIR = "/kaggle/working/forensic_twins_bfree_code"

# Để trống để tự tìm theo Kaggle slug.
ENCODER_DATASET_ROOT = ""
GMM_DATASET_ROOT = ""
BFREE_ROOT = ""

# valid_clean là cohort chính: valid_list và loại toàn bộ COCO ID đã xuất hiện trong manifest 200K.
COHORT = "valid_clean"  # valid_clean | all_clean
MAX_IDS = None          # 100 để smoke test; None để chạy toàn bộ cohort.
RUN_IMAGE_AUDIT = True

PROTOCOLS = [
    "native_png",
    "jpeg_q95",
    "jpeg_q80",
    "jpeg_q65",
    "jpeg_q50",
    "resize_075",
    "resize_050",
    "blur_1",
    "blur_2",
]
BOOTSTRAP_SAMPLES = 1000
NUM_WORKERS = 8
MANIFEST_WORKERS = 16
EMBEDDING_CHECKPOINT_EVERY = 1000
USE_BOTH_GPUS = True

# Nếu đã add output Dataset của lần trước, điền root để phục hồi partial embeddings/metrics.
RESUME_OUTPUT_DATASET_ROOT = ""

# Có thể trỏ trực tiếp tới bfree_*_eval.csv đã lưu ở một Kaggle Input để bỏ qua bước audit manifest.
REUSE_MANIFEST_PATH = ""

# Tùy chọn: điền slug để upload manifest ngay sau khi tạo và upload output sau từng protocol.
# Dataset chưa tồn tại sẽ được tạo private; nếu đã tồn tại sẽ tạo version mới.
MANIFEST_DATASET_SLUG = ""  # ví dụ: hoangha071005/forensic-twins-bfree-manifests
RESULT_DATASET_SLUG = ""    # ví dụ: hoangha071005/forensic-twins-bfree-eval

print("Configuration loaded.")
"""
        ),
        code(
            """import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
print("Python:", sys.version)
subprocess.run(["nvidia-smi"], check=False)
"""
        ),
        markdown("## 1. Bung source code được nhúng trong notebook\n"),
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
EVAL_SCRIPT = code_root / "test_bfree_gmm_kaggle.py"
assert CONFIG_PATH.is_file() and EVAL_SCRIPT.is_file()
print("Source extracted:", code_root)
print("Embedded archive bytes:", len(archive_bytes))
"""
        ),
        markdown("## 2. Xác minh ba Kaggle Inputs và phục hồi output cũ nếu có\n"),
        code(
            """input_root = Path(INPUT_ROOT)

def resolve_dataset(explicit, owner, slug):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([
        input_root / slug,
        input_root / "datasets" / owner / slug,
    ])
    for candidate in candidates:
        if candidate.is_dir():
            print(f"{owner}/{slug}: {candidate}")
            return candidate
    raise FileNotFoundError(f"Thiếu {owner}/{slug}; checked={candidates}")

encoder_dataset = resolve_dataset(
    ENCODER_DATASET_ROOT, "hoangha071005", "twins-gmm-200k-checkpoint"
)
gmm_dataset = resolve_dataset(
    GMM_DATASET_ROOT, "hoangha071005", "fit-gmm-checkpoint-200k"
)
bfree_root = resolve_dataset(BFREE_ROOT, "baopv051", "b-free-training-data")

output_root = Path(OUTPUT_ROOT)
if RESUME_OUTPUT_DATASET_ROOT:
    previous = Path(RESUME_OUTPUT_DATASET_ROOT)
    assert previous.is_dir(), f"Resume root không tồn tại: {previous}"
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(previous, output_root, dirs_exist_ok=True)
    print("Restored previous output:", previous)
else:
    output_root.mkdir(parents=True, exist_ok=True)

required = [
    encoder_dataset / "forensic_twins_v2_200k/artifacts/checkpoints/last.ckpt",
    encoder_dataset / "forensic_twins_v2_200k/cache/fre.pt",
    encoder_dataset / "forensic_twins_v2_200k/manifests/train_real_200k.csv",
    gmm_dataset / "forensic_twins_200k_gmm_eval/artifacts/gmm/real_k1_full.npz",
    gmm_dataset / "forensic_twins_200k_gmm_eval/artifacts/gmm/thresholds.json",
    bfree_root / "train_list.csv",
    bfree_root / "valid_list.csv",
]
for path in required:
    assert path.is_file(), f"Thiếu artifact: {path}"
    print(path)
"""
        ),
        markdown("## 3. Chạy CPU tests\n"),
        code(
            """subprocess.run(
    [sys.executable, str(code_root / "tests" / "test_bfree_evaluation.py")],
    cwd=code_root,
    check=True,
)
"""
        ),
        markdown("## 4. Tạo và lưu B-Free manifests trước khi chạy GPU\n"),
        code(
            """if REUSE_MANIFEST_PATH:
    source_manifest = Path(REUSE_MANIFEST_PATH)
    assert source_manifest.is_file(), f"Manifest tái sử dụng không tồn tại: {source_manifest}"
    reuse_dir = output_root / "manifests"
    reuse_dir.mkdir(parents=True, exist_ok=True)
    EVAL_MANIFEST = reuse_dir / source_manifest.name
    shutil.copy2(source_manifest, EVAL_MANIFEST)
    manifest_summary = {
        "reused": True,
        "source_manifest": str(source_manifest),
        "eval_manifest": str(EVAL_MANIFEST),
    }
    (reuse_dir / "bfree_manifest_summary.json").write_text(
        json.dumps(manifest_summary, indent=2), encoding="utf-8"
    )
    print("Reusing manifest:", source_manifest)
else:
    manifest_command = [
    sys.executable,
    str(EVAL_SCRIPT),
    "--config", str(CONFIG_PATH),
    "--input-root", INPUT_ROOT,
    "--output-root", OUTPUT_ROOT,
    "--encoder-dataset-root", str(encoder_dataset),
    "--gmm-dataset-root", str(gmm_dataset),
    "--bfree-root", str(bfree_root),
    "--cohort", COHORT,
    "--manifest-workers", str(MANIFEST_WORKERS),
    "--build-manifest-only",
    ]
    if MAX_IDS is not None:
        manifest_command.extend(["--max-ids", str(MAX_IDS)])
    if not RUN_IMAGE_AUDIT:
        manifest_command.append("--skip-image-audit")
    print("Running:", " ".join(manifest_command))
    subprocess.run(manifest_command, cwd=code_root, check=True)
    manifest_summary_path = output_root / "manifests" / "bfree_manifest_summary.json"
    manifest_summary = json.loads(manifest_summary_path.read_text(encoding="utf-8"))
    EVAL_MANIFEST = Path(manifest_summary["eval_manifest"])

assert EVAL_MANIFEST.is_file()
print(json.dumps(manifest_summary, indent=2, ensure_ascii=False))

manifest_zip = shutil.make_archive(
    "/kaggle/working/forensic_twins_bfree_manifests",
    "zip",
    root_dir=output_root,
    base_dir="manifests",
)
print("Reusable manifest bundle:", manifest_zip)
"""
        ),
        markdown("## 5. Tùy chọn upload manifest/output thành private Kaggle Dataset\n"),
        code(
            """def publish_kaggle_dataset(source_dir, slug, title, message, private_on_create=True):
    if not slug:
        return
    source_dir = Path(source_dir)
    metadata = {
        "title": title,
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (source_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    version = subprocess.run(
        ["kaggle", "datasets", "version", "-p", str(source_dir), "-m", message],
        check=False,
    )
    if version.returncode != 0:
        create_command = ["kaggle", "datasets", "create", "-p", str(source_dir)]
        if private_on_create:
            create_command.append("--private")
        subprocess.run(create_command, check=True)
    print("Published:", slug)

if MANIFEST_DATASET_SLUG:
    manifest_stage = Path("/kaggle/working/bfree_manifest_publish")
    if manifest_stage.exists():
        shutil.rmtree(manifest_stage)
    shutil.copytree(output_root / "manifests", manifest_stage)
    publish_kaggle_dataset(
        manifest_stage,
        MANIFEST_DATASET_SLUG,
        "Forensic Twins B-Free Clean Manifests",
        "Save B-Free non-overlap manifests",
    )
else:
    print("MANIFEST_DATASET_SLUG trống: manifests vẫn nằm trong Kaggle Output và file ZIP.")
"""
        ),
        markdown("## 6. Chạy từng protocol; embeddings có checkpoint và tự resume\n"),
        code(
            """import pandas as pd

common_eval_command = [
    sys.executable,
    str(EVAL_SCRIPT),
    "--config", str(CONFIG_PATH),
    "--input-root", INPUT_ROOT,
    "--output-root", OUTPUT_ROOT,
    "--encoder-dataset-root", str(encoder_dataset),
    "--gmm-dataset-root", str(gmm_dataset),
    "--bfree-root", str(bfree_root),
    "--eval-only",
    "--eval-manifest", str(EVAL_MANIFEST),
    "--num-workers", str(NUM_WORKERS),
    "--embedding-checkpoint-every", str(EMBEDDING_CHECKPOINT_EVERY),
    "--bootstrap-samples", str(BOOTSTRAP_SAMPLES),
]
if not USE_BOTH_GPUS:
    common_eval_command.append("--single-gpu")

for protocol in PROTOCOLS:
    command = [*common_eval_command, "--protocols", protocol]
    print("=" * 100)
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=code_root, check=True)
    if RESULT_DATASET_SLUG:
        publish_kaggle_dataset(
            output_root,
            RESULT_DATASET_SLUG,
            "Forensic Twins B-Free Evaluation",
            f"Completed protocol {protocol}",
        )

comparison_path = output_root / "artifacts" / "metrics" / "protocol_comparison.csv"
comparison = pd.read_csv(comparison_path)
completed_protocols = set(comparison["protocol"].astype(str))
missing_protocols = [protocol for protocol in PROTOCOLS if protocol not in completed_protocols]
if missing_protocols:
    raise RuntimeError(f"Missing completed protocols: {missing_protocols}")

completion_path = output_root / "artifacts" / "evaluation_complete.json"
completion = json.loads(completion_path.read_text(encoding="utf-8"))
completion["requested_protocols"] = list(PROTOCOLS)
completion["completed_protocols"] = sorted(completed_protocols)
completion["complete"] = True
completion_path.write_text(json.dumps(completion, indent=2), encoding="utf-8")
print("All requested protocols completed:", PROTOCOLS)
"""
        ),
        markdown("## 7. Tổng hợp kết quả\n"),
        code(
            """import pandas as pd
from IPython.display import Image as DisplayImage, display

comparison_path = output_root / "artifacts" / "metrics" / "protocol_comparison.csv"
comparison = pd.read_csv(comparison_path)
display(comparison)

for protocol in comparison["protocol"].tolist():
    print("=" * 100)
    print(protocol)
    protocol_root = output_root / "artifacts" / "metrics" / protocol
    display(pd.read_csv(protocol_root / "ranking.csv"))
    threshold_table = pd.read_csv(protocol_root / "threshold_metrics.csv")
    display(threshold_table[threshold_table["threshold_percentile"].isin([80, 86, 95])])

figure = output_root / "artifacts" / "figures" / "protocol_auroc_comparison.png"
if figure.is_file():
    display(DisplayImage(filename=str(figure)))

complete = output_root / "artifacts" / "evaluation_complete.json"
print(complete.read_text(encoding="utf-8"))
"""
        ),
        markdown("## 8. Đóng gói output để tải về hoặc lưu bằng Save Version\n"),
        code(
            """archive_path = shutil.make_archive(
    "/kaggle/working/forensic_twins_bfree_eval_artifacts",
    "zip",
    root_dir=output_root,
    base_dir=".",
)
print("Download bundle:", archive_path)
print("Manifest bundle:", manifest_zip)

for path in sorted((output_root / "manifests").glob("*")):
    if path.is_file():
        print(path.name, f"{path.stat().st_size / 1024**2:.2f} MiB")
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {"accelerator": "gpu"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Built {NOTEBOOK} with {len(cells)} cells and {len(encoded)} base64 chars")


if __name__ == "__main__":
    main()
