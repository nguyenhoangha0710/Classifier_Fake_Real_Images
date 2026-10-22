"""Build a self-contained Kaggle notebook for preprocessing-aware GMM experiments."""

from __future__ import annotations

import base64
import io
import json
import textwrap
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = BASE_DIR / "forensic_twins_augmented_gmm_bfree_kaggle.ipynb"
SOURCE_FILES = [
    "fit_augmented_gmms_bfree.py",
    "test_bfree_gmm_kaggle.py",
    "fit_gmm_and_evaluate.py",
    "configs/kaggle_2xt4_200k.json",
    "src/__init__.py",
    "src/core.py",
    "src/data.py",
    "src/checkpointing.py",
    "tests/test_augmented_gmm.py",
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
            """# Forensic Twins — preprocessing-aware GMMs and B-Free evaluation

Notebook này giữ frozen FRE/encoder, tạo sáu biến thể từ mỗi ảnh real, fit các
GMM protocol-specific K=1 và mixed K=1/2/4/8, tính q80–q98 chỉ bằng held-out
real calibration rồi đánh giá trên B-Free `valid_clean`.

Kaggle Inputs bắt buộc:

1. competition `imagenet-object-localization-challenge`;
2. dataset `awsaf49/coco-2017-dataset`;
3. dataset `hoangha071005/twins-gmm-200k-checkpoint`;
4. dataset `baopv051/b-free-training-data`.

Không cần `fit-gmm-checkpoint-200k`: notebook sẽ fit GMM mới.
"""
        ),
        code(
            """# ========================= USER CONFIG =========================
INPUT_ROOT = "/kaggle/input"
OUTPUT_ROOT = "/kaggle/working/forensic_twins_augmented_gmm_full"
CODE_DIR = "/kaggle/working/forensic_twins_augmented_gmm_code"

# Full 200K: khoảng 160K base images để fit GMM và 40K held-out real
# để hiệu chuẩn q80-q98. Toàn bộ 200K đều được sử dụng.
MAX_FIT_BASE_IMAGES = None
MAX_CALIBRATION_BASE_IMAGES = None
MAX_BFREE_IDS = None

VARIANTS = [
    "native",
    "resize512_png",
    "jpeg_q95",
    "jpeg_q80",
    "jpeg_q65",
    "resize075_blur05_jpeg80",
]
MIXED_COMPONENTS = [1, 2, 4, 8]
MIXED_COVARIANCE = "diag"
# Full run: protocol GMM dùng 160K/variant, mixed GMM dùng đủ khoảng 960K.
# Chỉ đặt một số nguyên (ví dụ 300_000) khi chủ động chạy pilot.
MAX_GMM_FIT_EMBEDDINGS = None
BOOTSTRAP_SAMPLES = 500
NUM_WORKERS = 8
MANIFEST_WORKERS = 16
EMBEDDING_CHECKPOINT_EVERY = 1000
USE_BOTH_GPUS = True

# Add một output Dataset cũ rồi điền root này để resume embeddings/models.
RESUME_OUTPUT_DATASET_ROOT = ""

# Khuyến nghị điền private Dataset slug. Output được version sau prepare,
# sau từng variant, sau fit và sau evaluate.
RESULT_DATASET_SLUG = "hoangha071005/forensic-twins-augmented-gmms"

print("Configuration loaded.", flush=True)
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
os.environ["PYTHONUNBUFFERED"] = "1"
print("Python:", sys.version, flush=True)
subprocess.run(["nvidia-smi"], check=False)
"""
        ),
        markdown("## 1. Bung source code nhúng trong notebook\n"),
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

SCRIPT = code_root / "fit_augmented_gmms_bfree.py"
CONFIG_PATH = code_root / "configs" / "kaggle_2xt4_200k.json"
assert SCRIPT.is_file() and CONFIG_PATH.is_file()
print("Source extracted:", code_root, flush=True)
print("Embedded source bytes:", len(archive_bytes), flush=True)
"""
        ),
        markdown("## 2. Xác minh bốn input và phục hồi output cũ\n"),
        code(
            """input_root = Path(INPUT_ROOT)

def resolve_dataset(owner, slug):
    candidates = [
        input_root / slug,
        input_root / "datasets" / owner / slug,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Thiếu {owner}/{slug}; checked={candidates}")

encoder_dataset = resolve_dataset("hoangha071005", "twins-gmm-200k-checkpoint")
bfree_dataset = resolve_dataset("baopv051", "b-free-training-data")
coco_dataset = resolve_dataset("awsaf49", "coco-2017-dataset")
imagenet_candidates = [
    input_root / "imagenet-object-localization-challenge",
    input_root / "competitions" / "imagenet-object-localization-challenge",
]
imagenet_dataset = next((path for path in imagenet_candidates if path.is_dir()), None)
if imagenet_dataset is None:
    raise FileNotFoundError(f"Thiếu ImageNet competition; checked={imagenet_candidates}")

required = {
    "checkpoint": encoder_dataset / "forensic_twins_v2_200k" / "artifacts" / "checkpoints" / "last.ckpt",
    "FRE": encoder_dataset / "forensic_twins_v2_200k" / "cache" / "fre.pt",
    "train manifest": encoder_dataset / "forensic_twins_v2_200k" / "manifests" / "train_real_200k.csv",
    "ImageNet train": imagenet_dataset / "ILSVRC" / "Data" / "CLS-LOC" / "train",
    "COCO train": coco_dataset / "coco2017" / "train2017",
    "B-Free train list": bfree_dataset / "train_list.csv",
}
for name, path in required.items():
    if not path.exists():
        raise FileNotFoundError(f"Thiếu {name}: {path}")
    print(f"{name}: {path}", flush=True)

output_root = Path(OUTPUT_ROOT)
if RESUME_OUTPUT_DATASET_ROOT:
    previous = Path(RESUME_OUTPUT_DATASET_ROOT)
    assert previous.is_dir(), f"Resume root không tồn tại: {previous}"
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(previous, output_root, dirs_exist_ok=True)
    # Kaggle CLI uploads subdirectories as manifests.zip/artifacts.zip when
    # --dir-mode zip is used. Restore both archive and direct-folder forms.
    for archive_name in ("manifests.zip", "artifacts.zip"):
        archive_path = output_root / archive_name
        if archive_path.is_file():
            destination = output_root / Path(archive_name).stem
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(destination)
            print("Restored archive:", archive_path, "->", destination, flush=True)
            archive_path.unlink()
    print("Restored previous output:", previous, flush=True)
else:
    output_root.mkdir(parents=True, exist_ok=True)
print("Output root:", output_root, flush=True)
"""
        ),
        markdown("## 3. CPU unit tests\n"),
        code(
            """subprocess.run(
    [sys.executable, str(code_root / "tests" / "test_augmented_gmm.py")],
    cwd=code_root,
    check=True,
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)
"""
        ),
        markdown("## 4. Hàm chạy stage và lưu output Dataset\n"),
        code(
            """def publish_output(message):
    if not RESULT_DATASET_SLUG:
        print("RESULT_DATASET_SLUG trống: output chỉ nằm trong /kaggle/working.", flush=True)
        return
    metadata = {
        "title": "Forensic Twins Augmented Real GMMs",
        "id": RESULT_DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (output_root / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    # New Kaggle datasets are private by default; the CLI has --public but no
    # --private flag. -r zip is required or nested artifact folders are skipped.
    # Publishing is best-effort so a temporary API/auth error cannot abort GPU work.
    version = subprocess.run(
        [
            "kaggle", "datasets", "version",
            "-p", str(output_root), "-m", message,
            "-r", "zip",
        ],
        check=False,
    )
    if version.returncode == 0:
        print("Persisted dataset version:", RESULT_DATASET_SLUG, message, flush=True)
        return True
    print("Dataset version failed; trying first-time private creation.", flush=True)
    create = subprocess.run(
        [
            "kaggle", "datasets", "create",
            "-p", str(output_root),
            "-r", "zip",
        ],
        check=False,
    )
    if create.returncode == 0:
        print("Created private output dataset:", RESULT_DATASET_SLUG, flush=True)
        return True
    print(
        "WARNING: Dataset publish failed (version rc=", version.returncode,
        ", create rc=", create.returncode,
        "). Pipeline will continue; Save Version preserves /kaggle/working.",
        sep="", flush=True,
    )
    return False

common = [
    sys.executable,
    str(SCRIPT),
    "--config", str(CONFIG_PATH),
    "--input-root", INPUT_ROOT,
    "--output-root", OUTPUT_ROOT,
    "--encoder-dataset-root", str(encoder_dataset),
    "--bfree-root", str(bfree_dataset),
    "--mixed-components", ",".join(map(str, MIXED_COMPONENTS)),
    "--mixed-covariance", MIXED_COVARIANCE,
    "--bootstrap-samples", str(BOOTSTRAP_SAMPLES),
    "--num-workers", str(NUM_WORKERS),
    "--manifest-workers", str(MANIFEST_WORKERS),
    "--embedding-checkpoint-every", str(EMBEDDING_CHECKPOINT_EVERY),
]
if not USE_BOTH_GPUS:
    common.append("--single-gpu")
if MAX_GMM_FIT_EMBEDDINGS is not None:
    common.extend(["--max-gmm-fit-embeddings", str(MAX_GMM_FIT_EMBEDDINGS)])

def run_stage(stage, extra=None, variants=None):
    command = [*common, "--stage", stage]
    selected_variants = VARIANTS if variants is None else variants
    command.extend(["--variants", ",".join(selected_variants)])
    if MAX_FIT_BASE_IMAGES is not None:
        command.extend(["--max-fit-base-images", str(MAX_FIT_BASE_IMAGES)])
    if MAX_CALIBRATION_BASE_IMAGES is not None:
        command.extend(["--max-calibration-base-images", str(MAX_CALIBRATION_BASE_IMAGES)])
    if MAX_BFREE_IDS is not None:
        command.extend(["--max-bfree-ids", str(MAX_BFREE_IDS)])
    if extra:
        command.extend(extra)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=code_root,
        check=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
"""
        ),
        markdown("## 5. Prepare: split theo base image và tạo B-Free clean manifest\n"),
        code(
            """run_stage("prepare")
publish_output("Prepared source split and B-Free clean manifests")
"""
        ),
        markdown("## 6. Trích embedding từng variant, có resume và persist sau mỗi variant\n"),
        code(
            """for variant in VARIANTS:
    print("=" * 100, flush=True)
    print("Extracting variant:", variant, flush=True)
    run_stage("extract", variants=[variant])
    publish_output(f"Completed real embeddings for {variant}")
"""
        ),
        markdown("## 7. Fit protocol-specific và mixed GMMs; calibrate q80–q98\n"),
        code(
            """run_stage("fit")
publish_output("Fitted augmented-real GMM candidates and thresholds")
"""
        ),
        markdown("## 8. Đánh giá frozen models trên B-Free native PNG\n"),
        code(
            """run_stage("evaluate")
publish_output("Completed clean B-Free evaluation")
"""
        ),
        markdown("## 9. Hiển thị bảng kết quả\n"),
        code(
            """import pandas as pd
from IPython.display import display

model_index = pd.read_csv(
    output_root / "artifacts" / "augmented_gmms" / "real_only_model_selection.csv"
)
display(model_index.sort_values(["covariance_type", "components"]))

comparison = pd.read_csv(
    output_root / "artifacts" / "augmented_bfree_metrics" / "model_comparison.csv"
)
display(comparison.sort_values("macro_variant_auroc", ascending=False))

complete = output_root / "artifacts" / "augmented_evaluation_complete.json"
print(complete.read_text(encoding="utf-8"), flush=True)
"""
        ),
        markdown("## 10. Đóng gói artifacts\n"),
        code(
            """archive_path = shutil.make_archive(
    "/kaggle/working/forensic_twins_augmented_gmm_artifacts",
    "zip",
    root_dir=output_root,
    base_dir=".",
)
print("Download bundle:", archive_path, flush=True)
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
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {NOTEBOOK} with {len(cells)} cells and {len(encoded)} base64 chars")


if __name__ == "__main__":
    main()
