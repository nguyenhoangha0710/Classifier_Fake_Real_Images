"""Embed the runnable source tree into the Kaggle notebook.

Run this after changing training source files so the notebook remains a
single-file artifact requiring only image datasets as Kaggle Inputs.
"""

from __future__ import annotations

import base64
import io
import json
import textwrap
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_NOTEBOOK = BASE_DIR / "forensic_twins_kaggle_2xt4.ipynb"
NOTEBOOK = BASE_DIR / "forensic_twins_200k_kaggle_2xt4.ipynb"
SOURCE_FILES = [
    "train_forensic_twins_kaggle.py",
    "fit_gmm_and_evaluate.py",
    "kaggle_launcher.py",
    "configs/kaggle_2xt4.json",
    "configs/kaggle_2xt4_200k.json",
    "src/__init__.py",
    "src/core.py",
    "src/data.py",
    "src/checkpointing.py",
    "tests/test_smoke.py",
    "tests/test_evaluation.py",
    "tests/test_manifest.py",
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


def source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True) or [""]


def main() -> None:
    notebook = json.loads(TEMPLATE_NOTEBOOK.read_text(encoding="utf-8"))
    encoded = archive_source()

    notebook["cells"][0]["source"] = source_lines(
        """# Forensic Twins — Kaggle 2×T4

Notebook tự chứa toàn bộ source code của baseline `genimage_lite_50k_provisional_v1`.

Kaggle Inputs duy nhất:

- `imagenet-object-localization-challenge`;
- `coco-2017-dataset`;
- `tiny-genimage` (chỉ cần khi evaluation).

`fre.pt` được tự động tải từ official FSD release. Hãy bật **Internet** cho notebook.

> Để không mất tiến độ khi session bị xóa, hãy tạo trước một private Kaggle Dataset cho checkpoint rồi điền `CHECKPOINT_DATASET_SLUG`. Checkpoint local được lưu mỗi 250 step và cuối mỗi epoch; checkpoint cuối epoch được version lên Dataset này.
"""
    )

    for cell in notebook["cells"]:
        joined = "".join(cell.get("source", []))
        if "USER CONFIG" in joined:
            cell["source"] = source_lines(
                """# ========================= USER CONFIG =========================
# Private Dataset phải được tạo trước, dạng: username/dataset-name
CHECKPOINT_DATASET_SLUG = ""  # Ví dụ: "myname/forensic-twins-checkpoints"

# Chỉ bật sau khi Tiny-GenImage đã có trong Kaggle Input.
RUN_EVALUATION = False
TINY_GENIMAGE_ROOT = ""  # Để trống để tự tìm
MAX_EVAL_PER_CLASS = None  # Có thể đặt 500 để chạy thử nhanh

INPUT_ROOT = "/kaggle/input"
OUTPUT_ROOT = "/kaggle/working/forensic_twins_original"
WORKING_CODE_DIR = "/kaggle/working/forensic_twins_code"
print("Configuration loaded.")
"""
            )
        elif joined.startswith("## 2. Tìm và chuẩn bị source code"):
            cell["source"] = source_lines(
                """## 2. Bung source code được nhúng trong notebook

Notebook mang sẵn training code, model, config và tests. Cell kế tiếp giải nén chúng vào `/kaggle/working`; không cần gắn thêm code Dataset và không cần clone GitHub.
"""
            )
        elif "imagenet_train = (" in joined and "coco_root = (" in joined:
            cell["source"] = source_lines(
                """# Exact mount paths derived from the three Kaggle source URLs.
input_root = Path(INPUT_ROOT)

imagenet_train = (
    input_root / "competitions" / "imagenet-object-localization-challenge"
    / "ILSVRC" / "Data" / "CLS-LOC" / "train"
)
imagenet_index = (
    input_root / "competitions" / "imagenet-object-localization-challenge"
    / "LOC_train_solution.csv"
)
coco_root = (
    input_root / "datasets" / "awsaf49" / "coco-2017-dataset" / "coco2017"
)
coco_train = coco_root / "train2017"
coco_unlabeled = coco_root / "unlabeled2017"
coco_index_candidates = [
    coco_root / "annotations" / "captions_train2017.json",
    coco_root / "annotations" / "instances_train2017.json",
]
tiny_genimage_root = input_root / "datasets" / "yangsangtai" / "tiny-genimage"

assert imagenet_train.is_dir(), f"Thiếu ImageNet train: {imagenet_train}"
assert coco_train.is_dir(), f"Thiếu COCO train2017: {coco_train}"
if RUN_EVALUATION:
    assert tiny_genimage_root.is_dir(), f"Thiếu Tiny-GenImage: {tiny_genimage_root}"

assert imagenet_index.is_file(), f"Missing ImageNet index: {imagenet_index}"
assert any(path.is_file() for path in coco_index_candidates), (
    f"Missing COCO train index under: {coco_root / 'annotations'}"
)

# Only list the 1,000 class folders; never recursively walk image files.
with os.scandir(imagenet_train) as entries:
    class_count = sum(1 for entry in entries if entry.is_dir())
assert class_count == 1000, f"Cần 1000 ImageNet class folders, tìm thấy {class_count}"

Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
print("ImageNet train:", imagenet_train)
print("ImageNet index:", imagenet_index)
print("ImageNet classes:", class_count)
print("COCO train2017:", coco_train)
print("COCO index:", next(path for path in coco_index_candidates if path.is_file()))
print("COCO unlabeled2017 exists:", coco_unlabeled.is_dir())
print("Tiny-GenImage:", tiny_genimage_root)
print("FRE: auto-download from official FSD release (Internet must be enabled)")
print("Output root:", OUTPUT_ROOT)
"""
            )
        elif "def discover_code_root" in joined or "SOURCE_ARCHIVE_B64" in joined:
            chunks = textwrap.wrap(encoded, width=100)
            lines = [
                "import base64\n",
                "import io\n",
                "import json\n",
                "import zipfile\n",
                "\n",
                "SOURCE_ARCHIVE_B64 = (\n",
            ]
            lines.extend(f'    "{chunk}"\n' for chunk in chunks)
            lines.extend(
                [
                    ")\n",
                    "input_root = Path(INPUT_ROOT)\n",
                    "working_code = Path(WORKING_CODE_DIR)\n",
                    "if working_code.exists():\n",
                    "    shutil.rmtree(working_code)\n",
                    "working_code.mkdir(parents=True, exist_ok=True)\n",
                    "archive_bytes = base64.b64decode(SOURCE_ARCHIVE_B64)\n",
                    "with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:\n",
                    "    archive.extractall(working_code)\n",
                    "CODE_DIR = working_code\n",
                    "CONFIG_PATH = CODE_DIR / \"configs\" / \"kaggle_2xt4_200k.json\"\n",
                    "if CHECKPOINT_DATASET_SLUG:\n",
                    "    runtime_config = json.loads(CONFIG_PATH.read_text(encoding=\"utf-8\"))\n",
                    "    runtime_config[\"checkpoint\"][\"kaggle_dataset_slug\"] = CHECKPOINT_DATASET_SLUG\n",
                    "    CONFIG_PATH.write_text(json.dumps(runtime_config, indent=2) + \"\\n\", encoding=\"utf-8\")\n",
                    "    print(\"Checkpoint slug written into runtime config:\", CHECKPOINT_DATASET_SLUG)\n",
                    "required = [\n",
                    "    CODE_DIR / \"kaggle_launcher.py\",\n",
                    "    CODE_DIR / \"train_forensic_twins_kaggle.py\",\n",
                    "    CODE_DIR / \"src\" / \"core.py\",\n",
                    "    CONFIG_PATH,\n",
                    "]\n",
                    "assert all(path.is_file() for path in required), required\n",
                    "print(\"Embedded source extracted to:\", CODE_DIR)\n",
                    "print(\"Embedded archive bytes:\", len(archive_bytes))\n",
                ]
            )
            cell["source"] = lines
        elif "def kaggle_dataset_status" in joined:
            cell["source"] = source_lines(
                """# Xác minh quyền truy cập private checkpoint Dataset trước khi train.
# Nếu cần, tạo Kaggle Secrets có tên KAGGLE_USERNAME và KAGGLE_KEY.
def kaggle_dataset_status(slug: str):
    return subprocess.run(
        ["kaggle", "datasets", "status", slug],
        capture_output=True, text=True, timeout=120,
    )

if CHECKPOINT_DATASET_SLUG:
    status = kaggle_dataset_status(CHECKPOINT_DATASET_SLUG)
    if status.returncode != 0:
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            os.environ.setdefault("KAGGLE_USERNAME", secrets.get_secret("KAGGLE_USERNAME"))
            os.environ.setdefault("KAGGLE_KEY", secrets.get_secret("KAGGLE_KEY"))
        except Exception as error:
            print("Không load được Kaggle API secrets:", error)
        status = kaggle_dataset_status(CHECKPOINT_DATASET_SLUG)
    if status.returncode != 0:
        raise RuntimeError(
            "Không truy cập được checkpoint Dataset. Hãy tạo private Dataset trước, "
            "kiểm tra slug và thêm KAGGLE_USERNAME/KAGGLE_KEY vào Kaggle Secrets.\\n"
            + (status.stderr or status.stdout)
        )
    print(status.stdout.strip() or f"Checkpoint Dataset OK: {CHECKPOINT_DATASET_SLUG}")

    # Tự tải checkpoint mới nhất bằng Kaggle API. Không cần Add Input checkpoint Dataset.
    checkpoint_dir = Path(OUTPUT_ROOT) / "artifacts" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not any(checkpoint_dir.glob("*.ckpt")):
        download = subprocess.run(
            [
                "kaggle", "datasets", "download", "-d", CHECKPOINT_DATASET_SLUG,
                "-p", str(checkpoint_dir), "--unzip", "--force",
            ],
            capture_output=True, text=True, timeout=1800,
        )
        if download.returncode == 0:
            print(download.stdout.strip())
            print("Downloaded checkpoints:", [p.name for p in checkpoint_dir.glob("*.ckpt")])
        else:
            print("Checkpoint Dataset chưa có checkpoint để tải (run đầu tiên là bình thường).")
            print(download.stderr or download.stdout)
else:
    print("WARNING: chưa cấu hình persistent checkpoint Dataset.")
"""
            )
        elif joined.startswith("## Khôi phục ở session tiếp theo"):
            cell["source"] = source_lines(
                """## Khôi phục ở session tiếp theo

1. Add lại đúng ba image Inputs: ImageNet, COCO và Tiny-GenImage.
2. Giữ nguyên `CHECKPOINT_DATASET_SLUG`, config và dữ liệu.
3. Notebook tự tải checkpoint mới nhất bằng Kaggle API; không cần Add Input checkpoint Dataset.
4. Chạy lại đến cell training với `--resume auto`.
5. Code xác minh checksum, config hash, manifest hash và FRE hash trước khi restore model, projector, optimizer, scheduler, AMP scaler và RNG state.
"""
            )

    for cell in notebook["cells"]:
        joined = "".join(cell.get("source", []))
        if "manifest_command = [" in joined:
            cell["source"] = source_lines(
                """manifest_command = [
    sys.executable, str(CODE_DIR / "kaggle_launcher.py"), "manifest",
    "--config", str(CONFIG_PATH),
    "--input-root", INPUT_ROOT,
    "--output-root", OUTPUT_ROOT,
]
if CHECKPOINT_DATASET_SLUG:
    manifest_command.extend(["--kaggle-dataset-slug", CHECKPOINT_DATASET_SLUG])
print("Running:", " ".join(manifest_command))
subprocess.run(manifest_command, cwd=CODE_DIR, check=True)
"""
            )
        elif "test_smoke.py" in joined and "test_evaluation.py" in joined:
            cell["source"] = source_lines(
                """subprocess.run([sys.executable, str(CODE_DIR / "tests" / "test_smoke.py")], cwd=CODE_DIR, check=True)
subprocess.run([sys.executable, str(CODE_DIR / "tests" / "test_evaluation.py")], cwd=CODE_DIR, check=True)
subprocess.run([sys.executable, str(CODE_DIR / "tests" / "test_manifest.py")], cwd=CODE_DIR, check=True)
"""
            )

    # Final v2-specific cells. Indexes are stable in the checked-in template;
    # assigning them explicitly prevents stale 50K labels or resume settings.
    notebook["cells"][0]["source"] = source_lines(
        """# Forensic Twins v2 — Kaggle 2×T4, 200K real images

This notebook trains from scratch on 100K ImageNet plus 100K COCO images.
It keeps the v1 architecture, global batch 128 and 80-epoch schedule so the
effect of additional real data can be measured independently.

Create a new private Kaggle Dataset for this run before starting. Do not reuse
the 50K checkpoint Dataset.
"""
    )
    notebook["cells"][1]["source"] = source_lines(
        """# ========================= USER CONFIG =========================
# Create this private Dataset before the first run, then fill its slug here.
# Recommended: "hoangha071005/forensic-twins-v2-200k-checkpoints"
CHECKPOINT_DATASET_SLUG = ""

# First run: "none" (guaranteed training from scratch).
# Later sessions after interruption: change only this value to "auto".
RESUME_MODE = "none"

RUN_EVALUATION = False
TINY_GENIMAGE_ROOT = ""
MAX_EVAL_PER_CLASS = None

INPUT_ROOT = "/kaggle/input"
OUTPUT_ROOT = "/kaggle/working/forensic_twins_v2_200k"
WORKING_CODE_DIR = "/kaggle/working/forensic_twins_v2_200k_code"
print("Configuration loaded; resume mode:", RESUME_MODE)
"""
    )
    notebook["cells"][11]["source"] = source_lines(
        """## 5. Build deterministic real-only 200K manifest

The cohort contains exactly 100,000 ImageNet images (100/class) and 100,000
COCO images. The resulting file is `train_real_200k.csv`.
"""
    )
    notebook["cells"][13]["source"] = source_lines(
        """import json

manifest_path = Path(OUTPUT_ROOT) / "manifests" / "train_real_200k.csv"
summary_path = manifest_path.with_suffix(".summary.json")
assert manifest_path.is_file(), manifest_path
assert summary_path.is_file(), summary_path
manifest_summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert manifest_summary["rows"] == 200000, manifest_summary
assert manifest_summary["imagenet_rows"] == 100000, manifest_summary
assert manifest_summary["coco_rows"] == 100000, manifest_summary
print(json.dumps(manifest_summary, indent=2))
print("Manifest verification passed:", manifest_path)
"""
    )
    notebook["cells"][14]["source"] = source_lines(
        """## 6. Train from scratch or resume the same 200K run on 2×T4

Use `RESUME_MODE="none"` only for the first run. After a Kaggle interruption,
use `RESUME_MODE="auto"` with the same config, inputs and checkpoint Dataset.
"""
    )
    notebook["cells"][15]["source"] = source_lines(
        """if RESUME_MODE not in {"none", "auto"}:
    raise ValueError('RESUME_MODE must be "none" or "auto"')
if not CHECKPOINT_DATASET_SLUG:
    print("WARNING: checkpoint Dataset is empty; progress will only live in /kaggle/working.")

existing_checkpoints = list((Path(OUTPUT_ROOT) / "artifacts" / "checkpoints").glob("*.ckpt"))
if RESUME_MODE == "none" and existing_checkpoints:
    raise RuntimeError(
        "RESUME_MODE='none' but checkpoints already exist. Set RESUME_MODE='auto' "
        "to continue the 200K run, or use a brand-new OUTPUT_ROOT/Dataset to restart."
    )

train_command = [
    sys.executable, str(CODE_DIR / "kaggle_launcher.py"), "train",
    "--config", str(CONFIG_PATH),
    "--input-root", INPUT_ROOT,
    "--output-root", OUTPUT_ROOT,
    "--resume", RESUME_MODE,
]
if CHECKPOINT_DATASET_SLUG:
    train_command.extend(["--kaggle-dataset-slug", CHECKPOINT_DATASET_SLUG])
print("Running:", " ".join(train_command))
subprocess.run(train_command, cwd=CODE_DIR, check=True)
"""
    )
    notebook["cells"][21]["source"] = source_lines(
        """## Resume after a Kaggle interruption

1. Keep the same ImageNet/COCO inputs, config and checkpoint Dataset slug.
2. Change `RESUME_MODE` from `"none"` to `"auto"`.
3. Run the notebook again. It downloads and validates the latest v2 checkpoint.
4. Model, projector, optimizer, scheduler, AMP scaler and RNG are restored.

Never switch back to `"none"` inside an existing v2 output directory.
"""
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Embedded {len(SOURCE_FILES)} files ({len(encoded)} base64 chars) into {NOTEBOOK}")


if __name__ == "__main__":
    main()
