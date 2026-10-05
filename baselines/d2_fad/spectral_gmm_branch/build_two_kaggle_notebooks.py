"""Generate the two annotated Kaggle notebooks requested for Spectral-GMM."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "train_spectral_gmm_kaggle.py"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


source = SOURCE_PATH.read_text(encoding="utf-8")
main_block = '\n\nif __name__ == "__main__":\n    run_pipeline()\n'
if main_block not in source:
    raise RuntimeError("Cannot find the source runner main block.")
definitions = source.replace(main_block, "\n")

environment_cell = """# CELL 1 — Kiểm tra môi trường trước khi sử dụng GPU.
import os
import platform
print('Python:', platform.python_version())
print('Kaggle input:', os.path.isdir('/kaggle/input'))
if os.path.isdir('/kaggle/input'):
    print('Mounted inputs:', sorted(os.listdir('/kaggle/input')))

import torch
import timm
import sklearn
print('PyTorch:', torch.__version__)
print('timm:', timm.__version__)
print('scikit-learn:', sklearn.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
if not torch.cuda.is_available():
    raise RuntimeError('Hãy bật GPU trong Kaggle Notebook settings trước khi train.')
"""

implementation_note = """## Cell 2 — Nạp toàn bộ implementation

Cell này định nghĩa nhưng **chưa chạy train**:

- manifest ImageNet xác định bằng tên file, không decode ảnh trước;
- MFM frequency mask, reconstructor và masked FrequencyLoss;
- local-feature cache theo shard;
- Global Attention và masked feature reconstruction;
- real-only GMM và calibration.

Không sửa cell này trực tiếp. Chỉ thay đổi các biến trong cell cấu hình phía sau.
"""

cache_note = """## Cell cache — Trích xuất Local CLS cho ảnh native-resolution

Mỗi ảnh được reflect-pad và chia thành các vùng không chồng lấp `224×224`.
Mỗi vùng tạo bốn view `raw_low/raw_high/jpeg_low/jpeg_high`, đi qua Local
Encoder và lưu `CLS 768-D`. Cache được chia shard, vì vậy có thể chạy nhiều
phiên bằng `CACHE_START/CACHE_STOP`, Save Version, rồi Add Input vào phiên sau.

Chỉ chạy Global Attention khi `feature_cache/index.json` báo đủ tất cả shard.
"""

global_note = """## Cell Global Attention — Học representation cấp ảnh

Local Encoder được freeze. Với một ảnh có `K` vùng, ta có `K×768` local CLS.
Một phần local CLS bị mask; Global Transformer tạo `Image CLS 768-D`, decoder
dùng duy nhất Image CLS để dự đoán các local CLS bị mask. Cell này lưu
`global_encoder_last.pt` và `global_encoder_best.pt`.
"""

gmm_note = """## Cell GMM — Fit phân phối real

Cell này bỏ decoder và trích đúng một `Image CLS 768-D` cho mỗi ảnh. StandardScaler
và diagonal GMM chỉ fit trên `real_train`; số component chọn bằng
`real_validation`; các ngưỡng NLL lấy từ `real_calibration`. Không dùng ảnh fake.
"""

inspect_cell = """# CELL CUỐI — Liệt kê artifact và trạng thái để kiểm tra trước khi Save Version.
OUTPUT_ROOT = Path(BASE_CONFIG['output_parent']) / BASE_CONFIG['run_name']
for artifact in sorted(OUTPUT_ROOT.rglob('*')):
    if artifact.is_file():
        print(f'{artifact.relative_to(OUTPUT_ROOT)}\t{artifact.stat().st_size / 1024**2:.2f} MB')

status_path = OUTPUT_ROOT / 'status.json'
if status_path.is_file():
    print('\\nSTATUS:')
    print(status_path.read_text(encoding='utf-8'))
"""


scratch_cells = [
    markdown(
        """# Spectral-GMM A — Tự pretrain MFM reconstructor trên ImageNet real

Notebook A chạy ba giai đoạn real-only:

1. **Local MFM reconstruction:** tự train ViT-B/16 + decoder bằng masked
   FrequencyLoss theo MFM mà SPAI sử dụng làm backbone.
2. **Global Attention:** lấy Local CLS của mỗi vùng để học một Image CLS cố định.
3. **GMM:** mô hình hóa phân phối Image CLS của ảnh real.

Input duy nhất là competition **ImageNet Object Localization Challenge**. Không
cần upload `mfm_pretrain_vit_base.pth`. Paper MFM dùng lịch 300 epoch, nhưng
notebook Kaggle này mặc định 20 epoch + early stopping để phù hợp giới hạn runtime.

JPEG recompression là extension chống shortcut của dự án. Đặt
`local_mfm_jpeg_probability=0.0` nếu muốn preprocessing MFM nguyên bản tuyệt đối.
"""
    ),
    markdown("## Cell 1 — Kiểm tra GPU và dependency\n\nCell chỉ kiểm tra, không train."),
    code(environment_cell),
    markdown(implementation_note),
    code(definitions),
    markdown(
        """## Cell 3 — Cấu hình notebook A

Không đổi `run_name` hoặc `seed` giữa các lần resume. `CACHE_START/STOP=None`
nghĩa là tạo toàn bộ cache. Mặc định là 20 epoch, warmup 2 epoch và early stop
sau 3 epoch validation không cải thiện. Chỉ đặt 300/20 nếu có hạ tầng đủ dài.
"""
    ),
    code(
        """# CELL 3 — Các tham số duy nhất cần chỉnh trước khi chạy.
CACHE_START = None
CACHE_STOP = None

BASE_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
BASE_CONFIG.update({
    'run_name': 'spectral_gmm_mfm_scratch_imagenet100k_v1',
    'output_parent': '/kaggle/working/spectral_gmm_branch',
    'imagenet_train_root': '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train',
    'seed': 20261003,
    'mfm_source': 'train_from_scratch',
    'mfm_checkpoint_path': None,
    'images_per_synset': 100,
    'train_per_synset': 80,
    'validation_per_synset': 10,
    'calibration_per_synset': 10,
    # Practical Kaggle schedule. Paper-scale reproduction would use 300/20.
    'local_mfm_epochs': 20,
    'local_mfm_warmup_epochs': 2,
    'local_mfm_patience': 3,
    'local_mfm_batch_size': 32,
    'local_mfm_accumulation_steps': 4,
    # Một dòng log cố định mỗi batch: ảnh đã xử lý, loss, tốc độ và ETA.
    'local_mfm_log_every_batches': 1,
    'local_mfm_learning_rate': 3e-4,
    'local_mfm_min_learning_rate': 2.5e-6,
    'local_mfm_low_pass_probability': 0.5,
    # Project extension against compression shortcut; use 0.0 for strict MFM preprocessing.
    'local_mfm_jpeg_probability': 0.5,
    'jpeg_quality_min': 70,
    'jpeg_quality_max': 100,
    'cache_shard_start': CACHE_START,
    'cache_shard_stop': CACHE_STOP,
    'spectral_mask_implementation': 'spai',
    'global_epochs': 10,
})
print(json.dumps(BASE_CONFIG, indent=2))
"""
    ),
    markdown(
        """## Cell 4 — Train Local MFM reconstructor từ đầu

Runner chỉ tạo manifest từ tên file, không mở hoặc audit 100.000 ảnh trước khi train.

Mỗi crop real `224×224` được FFT, ngẫu nhiên giữ low hoặc high, rồi IFFT tạo ảnh
corrupted. ViT-B/16 và pixel decoder tái tạo ảnh; loss so sánh FFT ảnh tái tạo
với FFT ảnh gốc **chỉ tại vùng tần số bị bỏ đi**.

Artifact:

- `local_mfm_last.pt`: encoder + decoder + optimizer để resume;
- `local_mfm_best.pt`: full model có validation FrequencyLoss tốt nhất;
- `local_encoder_best.pt`: chỉ encoder, dùng cho Global Attention.

Trong lúc chạy, mỗi batch 32 ảnh ghi một dòng log cố định gồm
`images`, `batch_loss`, `running_loss`, tốc độ và ETA; không phụ thuộc việc
Kaggle có render progress bar hay không.
"""
    ),
    code(
        """# CELL 4 — Đây là cell nặng nhất; có thể resume từ local_mfm_last.pt.
local_result = run_pipeline({**BASE_CONFIG, 'run_phases': ['local_mfm']})
local_result
"""
    ),
    markdown(cache_note),
    code(
        """# CELL 5 — Dùng local_encoder_best.pt vừa train để cache Local CLS.
cache_result = run_pipeline({**BASE_CONFIG, 'run_phases': ['cache']})
cache_result
"""
    ),
    markdown(global_note),
    code(
        """# CELL 6 — Chỉ chạy sau khi toàn bộ cache shard đã đầy đủ.
global_result = run_pipeline({**BASE_CONFIG, 'run_phases': ['train']})
global_result
"""
    ),
    markdown(gmm_note),
    code(
        """# CELL 7 — Trích Image CLS, chọn GMM và hiệu chỉnh threshold real-only.
gmm_result = run_pipeline({**BASE_CONFIG, 'run_phases': ['gmm']})
gmm_result
"""
    ),
    markdown("## Cell 8 — Kiểm tra và lưu output\n\nSau cell này, chọn **Save Version** trên Kaggle."),
    code(inspect_cell),
]


pretrained_cells = [
    markdown(
        """# Spectral-GMM B — Dùng official MFM checkpoint như SPAI

Notebook B không train lại Local MFM reconstructor. Nó nạp
`mfm_pretrain_vit_base.pth`, kiểm tra độ phủ tham số, freeze Local Encoder rồi
train Global Attention và fit GMM trên đúng cohort của notebook A.

Input bắt buộc:

1. Competition **ImageNet Object Localization Challenge**.
2. Kaggle Dataset chứa đúng file `mfm_pretrain_vit_base.pth`.

Notebook cố ý dừng nếu checkpoint thiếu hoặc không tương thích; không fallback
sang ImageNet-supervised hoặc random ViT.
"""
    ),
    markdown("## Cell 1 — Kiểm tra GPU và dependency\n\nCell chỉ kiểm tra, không train."),
    code(environment_cell),
    markdown(implementation_note),
    code(definitions),
    markdown(
        """## Cell 3 — Cấu hình notebook B

Để hai thí nghiệm công bằng, giữ nguyên seed và split 80/10/10 giống notebook A.
Nếu có nhiều file MFM trong `/kaggle/input`, đặt đường dẫn tuyệt đối tại
`mfm_checkpoint_path`; nếu chỉ có một file, runner tự tìm.
"""
    ),
    code(
        """# CELL 3 — Các tham số duy nhất cần chỉnh trước khi chạy.
CACHE_START = None
CACHE_STOP = None

BASE_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
BASE_CONFIG.update({
    'run_name': 'spectral_gmm_official_mfm_imagenet100k_v1',
    'output_parent': '/kaggle/working/spectral_gmm_branch',
    'imagenet_train_root': '/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train',
    'seed': 20261003,
    'mfm_source': 'official_checkpoint',
    'mfm_checkpoint_path': None,  # hoặc '/kaggle/input/<dataset>/mfm_pretrain_vit_base.pth'
    'images_per_synset': 100,
    'train_per_synset': 80,
    'validation_per_synset': 10,
    'calibration_per_synset': 10,
    'jpeg_quality_min': 70,
    'jpeg_quality_max': 100,
    'cache_shard_start': CACHE_START,
    'cache_shard_stop': CACHE_STOP,
    'spectral_mask_implementation': 'spai',
    'global_epochs': 10,
})
print(json.dumps(BASE_CONFIG, indent=2))
"""
    ),
    markdown(
        """## Cell 4 — Load official MFM và cache Local CLS

Runner chỉ tạo manifest từ tên file, không mở hoặc audit 100.000 ảnh trước khi cache.

Cell kiểm tra checkpoint ViT-B/16 có ít nhất 95% parameter coverage. Sau đó
backbone được freeze hoàn toàn; không có gradient và không có reconstructor.
Low/high decomposition ở bước inference dùng đúng circular mask của SPAI.
"""
    ),
    markdown(cache_note),
    code(
        """# CELL 4 — Có thể chia cache thành nhiều phiên bằng CACHE_START/CACHE_STOP.
cache_result = run_pipeline({**BASE_CONFIG, 'run_phases': ['cache']})
cache_result
"""
    ),
    markdown(global_note),
    code(
        """# CELL 5 — Chỉ train Global Attention; official MFM encoder vẫn frozen.
global_result = run_pipeline({**BASE_CONFIG, 'run_phases': ['train']})
global_result
"""
    ),
    markdown(gmm_note),
    code(
        """# CELL 6 — Fit GMM và calibrate NLL bằng ảnh real.
gmm_result = run_pipeline({**BASE_CONFIG, 'run_phases': ['gmm']})
gmm_result
"""
    ),
    markdown("## Cell 7 — Kiểm tra và lưu output\n\nSau cell này, chọn **Save Version** trên Kaggle."),
    code(inspect_cell),
]


outputs = {
    HERE / "train_spectral_gmm_mfm_from_scratch_kaggle.ipynb": notebook(scratch_cells),
    HERE / "train_spectral_gmm_official_mfm_kaggle.ipynb": notebook(pretrained_cells),
}
for path, payload in outputs.items():
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {path}")
