"""Build the standalone Kaggle test notebook for Spectral-GMM."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "test_spectral_gmm_tinygenimage_kaggle.py"
OUTPUT = HERE / "test_spectral_gmm_tinygenimage_kaggle.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.rstrip() + "\n",
    }


runner = RUNNER.read_text(encoding="utf-8")
notebook_runner = runner.rsplit('\nif __name__ == "__main__":', 1)[0].rstrip()

cells = [
    markdown(
        """# Spectral-GMM — test-only trên Tiny-GenImage

Notebook này **không train lại encoder và không fit lại GMM**. Nó đọc model
hoàn chỉnh từ Kaggle Dataset
[`hoangha071005/chech-point-gmm-spectral`](https://www.kaggle.com/datasets/hoangha071005/chech-point-gmm-spectral),
rồi test trên held-out test/validation cohort của Tiny-GenImage.

## Kaggle Input bắt buộc

1. `hoangha071005/chech-point-gmm-spectral`;
2. `yangsangtai/tiny-genimage`;
3. bật GPU accelerator.

Inference dùng checkpoint đã train và chuẩn hóa codec test theo pipeline:

```text
raw image
→ decode RGB → JPEG quality 70–100 → decode RGB
→ native non-overlap tiles 224×224, chỉ reflect/replicate pad
→ SPAI low và high view
→ frozen Local ViT-B/16
→ mean(Low CLS, High CLS) cho từng tile
→ frozen Global Transformer, không mask
→ một Image CLS 768-D
→ StandardScaler
→ diagonal GMM K=16
→ NLL anomaly score
```

Model chạy đúng một lần trên toàn bộ Tiny-GenImage để tạo NLL score. Sau đó
notebook chạy vòng `for` qua đúng 10 threshold cố định, tính metric tổng và
metric từng generator, rồi đánh dấu threshold có Macro Generator Balanced
Accuracy cao nhất. Encoder không chạy lại cho từng threshold.

Mặc định **mọi ảnh** (kể cả PNG, WebP và JPEG gốc) đều chịu cùng phép biến đổi
decode → JPEG → decode trước khi chia tile. Quality và chroma subsampling được
sinh xác định từ `sample_id + seed`, nên chạy lại luôn dùng đúng tham số cho
từng ảnh. Prediction lưu format gốc, quality và subsampling để audit.
"""
    ),
    markdown(
        """## Cell 1 — Kiểm tra môi trường và Kaggle Input

Cell này chưa load checkpoint và chưa chạy inference.
"""
    ),
    code(
        """# CELL 1 — Environment only.
import sys
from pathlib import Path
import torch
import timm
import sklearn

print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('timm:', timm.__version__)
print('scikit-learn:', sklearn.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('Kaggle inputs:')
for item in sorted(Path('/kaggle/input').iterdir()):
    print(' -', item)
assert torch.cuda.is_available(), 'Hãy bật GPU accelerator trước khi chạy.'
"""
    ),
    markdown(
        """## Cell 2 — Runtime standalone

Toàn bộ kiến trúc, preprocessing, metric và kiểm tra provenance được nhúng trong
cell này. Chạy cell chỉ khai báo code, không tự chạy inference.
"""
    ),
    code(notebook_runner),
    markdown(
        """## Cell 3 — Cấu hình test

Mặc định dùng toàn bộ test cohort và cân bằng real/fake riêng trong từng
generator. Đặt
`max_per_class_per_generator=10` nếu muốn smoke test trước.
"""
    ),
    code(
        """# CELL 3 — Evaluation configuration.
CONFIG = copy.deepcopy(DEFAULT_CONFIG)
CONFIG.update({
    'input_root': '/kaggle/input',
    'checkpoint_dataset_root': None,  # Auto-detect chech-point-gmm-spectral.
    'tiny_dataset_root': None,        # Auto-detect tiny-genimage.
    'output_root': '/kaggle/working/spectral_gmm_tinygenimage_fixed_grid_test',
    'seed': 42,
    'jpeg_policy': 'all_to_deterministic_jpeg',
    'jpeg_quality_min': 70,
    'jpeg_quality_max': 100,
    'jpeg_subsampling_values': [0, 1, 2],
    'balance_per_generator': True,
    'threshold_grid': [
        -334.892, -241.577, -149.749, -57.235, 27.958,
        133.346, 335.641, 614.874, 924.637, 1311.118,
    ],
    'max_per_class_per_generator': None,  # 10 for a smoke test; None for full.
    'max_images_per_batch': 16,
    'max_tiles_per_batch': 128,
    'local_view_batch_size': 64,
    'amp': True,
    'resume': True,
})
print(json.dumps(CONFIG, indent=2))
"""
    ),
    markdown(
        """## Cell 4 — Kiểm tra checkpoint dataset

Notebook yêu cầu đủ Local Encoder, Global Encoder, scaler, GMM và threshold.
SHA-256 của Global checkpoint phải đúng với GMM đã fit; nếu sai notebook dừng
ngay để tránh test nhầm weight.
"""
    ),
    code(
        """# CELL 4 — Artifact discovery and provenance audit.
OUTPUT_ROOT = Path(CONFIG['output_root'])
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
ARTIFACT_ROOT = discover_artifact_root(CONFIG)
ARTIFACT_REPORT = audit_artifacts(ARTIFACT_ROOT, CONFIG)
write_json(OUTPUT_ROOT / 'artifact_report.json', ARTIFACT_REPORT)
print(json.dumps(ARTIFACT_REPORT, indent=2, ensure_ascii=False))
"""
    ),
    markdown(
        """## Cell 5 — Khóa test manifest Tiny-GenImage

Ưu tiên folder `test`; nếu mirror chỉ có `validation/val/valid`, notebook dùng
folder đó làm test. Không đọc train split. Manifest và số lượng real/fake theo
generator được lưu trước inference.
"""
    ),
    code(
        """# CELL 5 — Build the exact deterministic Tiny test cohort.
MANIFEST, MANIFEST_SUMMARY = build_tiny_test_manifest(CONFIG, OUTPUT_ROOT)
display(pd.crosstab(MANIFEST['generator'], MANIFEST['label_name']))
print(json.dumps(MANIFEST_SUMMARY, indent=2, ensure_ascii=False))
"""
    ),
    markdown(
        """## Cell 6 — Load model và chạy inference

Mỗi ảnh được decode, JPEG-recompress Q70–100 rồi decode lại trước khi chia tile
ở kích thước gốc. Tile từ nhiều ảnh được gom lại để FFT và chạy Local ViT theo
batch GPU. File prediction tạm được cập nhật định kỳ để có thể resume trong
cùng Kaggle working directory.
"""
    ),
    code(
        """# CELL 6 — Exact frozen-model inference.
RUNTIME = load_runtime(ARTIFACT_ROOT, ARTIFACT_REPORT, CONFIG)
print('Legacy artifact threshold (comparison only):', RUNTIME['primary_threshold_name'], '=', RUNTIME['primary_threshold'])
print('Global checkpoint metadata:', RUNTIME['global_metadata'])
PREDICTIONS, ERRORS = run_inference(MANIFEST, RUNTIME, CONFIG, OUTPUT_ROOT)
print('Predictions:', len(PREDICTIONS), '/', len(MANIFEST))
print('Decode/inference errors:', len(ERRORS))
if len(PREDICTIONS) != len(MANIFEST):
    display(ERRORS.head(20))
    raise RuntimeError('Inference incomplete; inspect tiny_test_errors.csv.')
display(PREDICTIONS.head())
"""
    ),
    markdown(
        """## Cell 7 — Chạy fixed threshold grid trên test score

Model đã chạy một lần ở Cell 6. Cell này chỉ lặp qua 10 con số threshold, không
chạy lại encoder. Mỗi threshold có metric tổng và từng generator; dòng tốt nhất
được chọn theo Macro Generator Balanced Accuracy. Đây là test-grid/oracle vì
nhãn test tham gia chọn threshold.
"""
    ),
    code(
        """# CELL 7 — Evaluate exactly the requested 10 thresholds on test scores.
SUMMARY = evaluate_predictions(
    PREDICTIONS, RUNTIME, ARTIFACT_REPORT, MANIFEST_SUMMARY, OUTPUT_ROOT, CONFIG
)
display(pd.DataFrame([SUMMARY['threshold_selection']]))
display(pd.DataFrame([SUMMARY['best_test_grid']]))
display(pd.read_csv(OUTPUT_ROOT / 'metrics/test_fixed_threshold_grid_metrics.csv'))
display(pd.read_csv(OUTPUT_ROOT / 'metrics/test_fixed_threshold_grid_generator_metrics.csv'))
print(json.dumps(SUMMARY['macro'], indent=2))
"""
    ),
    markdown(
        """## Cell 8 — Đóng gói output

ZIP chứa manifest, prediction, metric tổng, metric từng generator, thống kê dữ
liệu và biểu đồ. Sau khi cell hoàn tất, tải file ZIP trực tiếp từ Kaggle Output.
"""
    ),
    code(
        """# CELL 8 — Persist summary and make a downloadable archive.
SUMMARY['output_root'] = str(OUTPUT_ROOT)
write_json(OUTPUT_ROOT / 'run_summary.json', SUMMARY)
ARCHIVE = make_output_archive(OUTPUT_ROOT)
print('Output root:', OUTPUT_ROOT)
print('Download ZIP:', ARCHIVE)
print('\\nArtifacts:')
for artifact in sorted(OUTPUT_ROOT.rglob('*')):
    if artifact.is_file():
        print(f'{artifact.relative_to(OUTPUT_ROOT)}\t{artifact.stat().st_size / 1024**2:.2f} MB')
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {"accelerator": "gpu", "isGpuEnabled": True, "isInternetEnabled": False},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUTPUT}")
