"""Build the standalone Kaggle notebook for the native hierarchical model."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "train_spectral_gmm_native_hierarchical_kaggle.py"
OUTPUT = HERE / "train_spectral_gmm_native_hierarchical_kaggle.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.rstrip() + "\n",
    }


runner_source = RUNNER.read_text(encoding="utf-8")
# A notebook cell executes with ``__name__ == "__main__"``.  Keep the runner's
# CLI entry point in the standalone .py file, but do not let Cell 2 launch the
# full pipeline before the user has reviewed/overridden BASE_CONFIG in Cell 3.
notebook_runner_source = runner_source.rsplit('\nif __name__ == "__main__":', 1)[0].rstrip()

cells = [
    markdown(
        """# Spectral-GMM Native Hierarchical — ImageNet real-only

Notebook này triển khai đúng `plan.md`, tách biệt hoàn toàn với notebook MFM
resize/crop cũ.

Workflow:

1. **Stage 1:** mọi ảnh decode RGB → JPEG Q70–100 → native tiling `224×224` → random low/high
   → attention giữa 196 token trong từng tile → reconstruction; loss mean theo ảnh.
2. **Feature cache:** Frozen Local Encoder chỉ tạo `jpeg_low/jpeg_high` cho từng native tile.
3. **Stage 2:** normalize + concat low/high `1536-D` → projection `768-D` →
   Global Attention giữa toàn bộ tile cùng ảnh → `Image CLS bottleneck` dự đoán
   low/high target của tile bị mask.
4. **Stage 3:** đúng một `Image CLS 768-D/ảnh` → StandardScaler → real-only GMM.

Không dùng `RandomResizedCrop`, `CenterCrop` hoặc resize ảnh đủ lớn. Mọi định dạng nguồn
đều được chuyển qua JPEG Q70–100 trước khi chia tile. Ảnh nhỏ hơn
224 mới được resize đồng tỷ lệ. Notebook lưu checkpoint sau mỗi epoch và có thể
resume từ output version trước khi được Add Input trở lại Kaggle.
"""
    ),
    markdown(
        """## Cell 1 — Kiểm tra môi trường

GPU T4 đơn vẫn chạy được nhưng Stage 1 dùng toàn bộ native tile nên sẽ nặng hơn
pipeline crop cũ. Cell này chỉ in dependency và GPU, chưa đọc ảnh hoặc train.
"""
    ),
    code(
        """# CELL 1 — Environment check only.
import sys
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
"""
    ),
    markdown(
        """## Cell 2 — Implementation

Đây là source code hoàn chỉnh được nhúng để notebook chạy độc lập. Các invariant
quan trọng được giữ trong code:

- tile native `224×224`, stride cấu hình được;
- mọi tile được dùng nhưng loss mean theo ảnh;
- Stage 2 không attention giữa hai ảnh;
- contextual matrix không đi trực tiếp vào prediction head;
- decoder Stage 2 chỉ nhận `Image CLS + positional query`;
- GMM chỉ nhận tensor `[N,768]`, một hàng trên mỗi ảnh.
"""
    ),
    code(notebook_runner_source),
    markdown(
        """## Cell 3 — Cấu hình thí nghiệm

Giữ nguyên `run_name` và `seed` khi resume. `tile_stride=224` là cấu hình chính;
đặt `112` nếu muốn ablation overlap 50%. Toàn bộ ảnh luôn được decode RGB, nén JPEG
Q70–100 và decode lại. Stage 1 đổi quality theo ảnh + epoch; cache, Stage 2 và GMM dùng
một realization cố định theo `sample_id + seed` để tái lập kết quả.
"""
    ),
    code(
        """# CELL 3 — Main research configuration.
BASE_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
BASE_CONFIG.update({
    'run_name': 'spectral_gmm_native_hierarchical_jpeg70_100_v1',
    'output_parent': '/kaggle/working/spectral_gmm_native_hierarchical',
    'imagenet_train_root': '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train',
    'seed': 20261004,

    # 1000 synset × 100 real images; split inside every synset.
    'images_per_synset': 100,
    'train_per_synset': 80,
    'validation_per_synset': 10,
    'calibration_per_synset': 10,

    # Native tile configuration. No resize for images >=224 on both sides.
    'tile_size': 224,
    'tile_stride': 224,
    'padding_mode': 'reflect',

    # Mandatory whole-image JPEG normalization for every source image.
    'jpeg_quality_min': 70,
    'jpeg_quality_max': 100,

    # Local spectral pretraining.
    'spectral_mask_implementation': 'spai',
    'spectral_mask_radius': 16,
    'low_probability': 0.50,
    'stage1_epochs': 20,
    'stage1_warmup_epochs': 2,
    'stage1_patience': 3,
    'stage1_images_per_batch': 2,
    'stage1_tile_microbatch': 32,
    'stage1_num_workers': 2,
    'stage1_log_every_images': 10,

    # Cache can be split across Kaggle versions if required.
    'cache_images_per_shard': 250,
    'cache_local_microbatch': 64,
    'cache_shard_start': None,
    'cache_shard_stop': None,
    'cache_log_every_images': 10,

    # Global Image-CLS bottleneck.
    'global_mask_ratio': 0.40,
    'global_depth': 4,
    'global_heads': 12,
    'global_decoder_depth': 2,
    'stage2_epochs': 10,
    'stage2_patience': 3,
    'stage2_max_images_per_batch': 16,
    'stage2_max_tokens_per_batch': 768,
    'stage2_log_every_images': 50,
})

print(json.dumps(BASE_CONFIG, indent=2))
"""
    ),
    markdown(
        """## Cell 4 — Khóa manifest

Chỉ liệt kê tên file, random 100 đường dẫn trong từng synset và chia 80/10/10.
Không mở hoặc audit 100.000 ảnh ở bước này.
"""
    ),
    code(
        """# CELL 4 — Fast deterministic manifest only.
manifest_result = run_pipeline({**BASE_CONFIG, 'run_phases': ['manifest']})
manifest_result
"""
    ),
    markdown(
        """## Cell 5 — Stage 1: Local token attention

Cell nặng nhất. Với mỗi ảnh, model sử dụng toàn bộ native tile nhưng tính
`mean(tile_loss)` trong ảnh trước khi mean batch. Train và validation đều là JPEG
Q70–100; validation đồng thời chạy cả low/high. Checkpoint được lưu sau mỗi epoch.
"""
    ),
    code(
        """# CELL 5 — Train Local ViT + reconstruction decoder.
stage1_result = run_pipeline({**BASE_CONFIG, 'run_phases': ['stage1']})
stage1_result
"""
    ),
    markdown(
        """## Cell 6 — Cache Frozen Local CLS

Mỗi ảnh chỉ lưu hai feature `jpeg_low`, `jpeg_high`. JPEG quality và subsampling
được khóa bằng seed để cache tái lập được; không lưu hoặc dùng raw/original view.

Nếu cần chia nhiều phiên, đặt khoảng `[cache_shard_start, cache_shard_stop)` rồi
Save Version; Add Output của các phiên làm Input trước khi train Stage 2.
"""
    ),
    code(
        """# CELL 6 — Build resumable Stage-2 feature shards.
cache_result = run_pipeline({**BASE_CONFIG, 'run_phases': ['cache']})
cache_result
"""
    ),
    markdown(
        """## Cell 7 — Stage 2: Global attention và Image CLS bottleneck

Mỗi tile JPEG: normalized low/high `768+768` → concat `1536` → projection `768`.
Mask toàn bộ fused tile token. Decoder chỉ dùng `Image CLS + positional query`
để dự đoán frozen low/high target; contextual tile row không đi vào head.
"""
    ),
    code(
        """# CELL 7 — Train fusion projection + Global Transformer + bottleneck decoder.
stage2_result = run_pipeline({**BASE_CONFIG, 'run_phases': ['stage2']})
stage2_result
"""
    ),
    markdown(
        """## Cell 8 — Stage 3: Image CLS → StandardScaler → GMM

Không mask và dùng đúng JPEG realization đã cache. Mỗi ảnh tạo đúng một vector `768-D`; contextual
matrix `K×768` bị loại khỏi đường đi GMM.
"""
    ),
    code(
        """# CELL 8 — Extract one Image CLS per real image and fit the real-only GMM.
gmm_result = run_pipeline({**BASE_CONFIG, 'run_phases': ['gmm']})
gmm_result
"""
    ),
    markdown(
        """## Cell 9 — Kiểm tra artifact trước khi Save Version

In toàn bộ checkpoint, cache index, metric và GMM artifact đã tạo. Sau đó dùng
**Save Version** để lưu `/kaggle/working` thành output của notebook.
"""
    ),
    code(
        """# CELL 9 — Artifact inventory.
OUTPUT_ROOT = Path(BASE_CONFIG['output_parent']) / BASE_CONFIG['run_name']
print('Output root:', OUTPUT_ROOT)
for artifact in sorted(OUTPUT_ROOT.rglob('*')):
    if artifact.is_file():
        print(f'{artifact.relative_to(OUTPUT_ROOT)}\t{artifact.stat().st_size / 1024**2:.2f} MB')

status_path = OUTPUT_ROOT / 'status.json'
if status_path.is_file():
    print('\\nSTATUS:')
    print(status_path.read_text(encoding='utf-8'))
"""
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
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUTPUT}")
