"""Build the Kaggle notebook that starts from a frozen Stage-1 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CORE = HERE / "train_spectral_gmm_native_hierarchical_kaggle.py"
DIAGNOSTICS = HERE / "native_stage2_gmm_threshold_diagnostics.py"
OUTPUT = HERE / "train_stage2_gmm_from_stage1_best_kaggle.ipynb"


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


core_source = CORE.read_text(encoding="utf-8")
core_source = core_source.rsplit('\nif __name__ == "__main__":', 1)[0].rstrip()
diagnostic_source = DIAGNOSTICS.read_text(encoding="utf-8")


cells = [
    markdown(
        """# Stage 2 + real-only GMM from the best native Stage-1 checkpoint

Notebook này **không train lại Stage 1**. Nó nhận checkpoint tốt nhất từ dataset
`hoangha071005/chech-point-gmm-hierarchical`, đóng băng Local ViT, rồi thực hiện:

1. kiểm tra checkpoint Stage 1 và lịch sử validation;
2. tạo cache `jpeg_low/jpeg_high` cho toàn bộ ImageNet real cohort 80k/10k/10k;
3. train Stage 2 Global Transformer với Image-CLS bottleneck;
4. trích đúng một Image CLS `768-D` cho mỗi ảnh và fit real-only GMM;
5. tạo threshold q90/q95/q99 chỉ từ ImageNet real calibration;
6. tùy chọn test cùng TinyGenImage cohort cân bằng như các thử nghiệm trước để đo
   AUC/AP, độ chồng lấn và threshold oracle chẩn đoán.

Threshold oracle sử dụng nhãn Tiny nên chỉ trả lời *representation có tách được hai lớp
hay không*. Không dùng threshold này như ngưỡng deployment độc lập.
"""
    ),
    markdown(
        """## Cell 1 — Kiểm tra môi trường

Input cần gắn vào notebook:

- Competition: **ImageNet Object Localization Challenge**;
- Dataset: **hoangha071005/chech-point-gmm-hierarchical**;
- Dataset: **yangsangtai/tiny-genimage** nếu chạy phần chẩn đoán real/fake.
"""
    ),
    code(
        """# CELL 1 — Environment only; chưa đọc ảnh hoặc train.
import sys
import torch
import timm
import sklearn
import scipy

print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('timm:', timm.__version__)
print('scikit-learn:', sklearn.__version__)
print('scipy:', scipy.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"""
    ),
    markdown(
        """## Cell 2 — Core native hierarchical implementation

Cell này nhúng nguyên implementation đã dùng để train Stage 1. Việc dùng cùng source bảo đảm
native tiling, JPEG Q70–100, spectral mask, Local CLS, Global Transformer và GMM không bị lệch
giữa hai phiên.
"""
    ),
    code(core_source),
    markdown(
        """## Cell 3 — TinyGenImage threshold diagnostics

Phần này chỉ định nghĩa hàm. Chưa chạy inference. Mỗi ảnh Tiny được xử lý bằng đúng pipeline:

`decode RGB → seeded JPEG Q70–100 → native tiles → low/high Local CLS → Stage 2 Image CLS → scaler → GMM NLL`.
"""
    ),
    code(diagnostic_source),
    markdown(
        """## Cell 4 — Tìm và xác minh Stage-1 input

Notebook yêu cầu đầy đủ `stage1_local_best.pt`, `stage1_local_last.pt`,
`stage1_local_encoder_best.pt`, manifest và training history. History quyết định epoch tốt nhất;
Stage 2 luôn dùng `stage1_local_encoder_best.pt`, không dùng checkpoint đang chạy dở.
"""
    ),
    code(
        """# CELL 4 — Locate and audit the uploaded Stage-1 dataset.
from pathlib import Path
import json
import pandas as pd

KAGGLE_INPUT = Path('/kaggle/input')

preferred = (
    KAGGLE_INPUT
    / 'chech-point-gmm-hierarchical'
    / 'spectral_gmm_native_hierarchical'
    / 'spectral_gmm_native_hierarchical_jpeg70_100_v1'
)

required_stage1 = [
    'config.json',
    'manifests/imagenet_real_100k.csv',
    'metrics/stage1_training_history.csv',
    'checkpoints/stage1_local_best.pt',
    'checkpoints/stage1_local_last.pt',
    'checkpoints/stage1_local_encoder_best.pt',
]

def complete_stage1_root(path):
    return path.is_dir() and all((path / item).is_file() for item in required_stage1)

if complete_stage1_root(preferred):
    STAGE1_ROOT = preferred.resolve()
else:
    candidates = []
    for checkpoint in KAGGLE_INPUT.rglob('stage1_local_encoder_best.pt'):
        root = checkpoint.parent.parent
        if complete_stage1_root(root):
            candidates.append(root.resolve())
    candidates = sorted(set(candidates), key=str)
    if len(candidates) != 1:
        raise RuntimeError(
            f'Expected exactly one complete Stage-1 root; found {len(candidates)}: {candidates}'
        )
    STAGE1_ROOT = candidates[0]

stage1_config = json.loads((STAGE1_ROOT / 'config.json').read_text(encoding='utf-8'))
stage1_history = pd.read_csv(STAGE1_ROOT / 'metrics/stage1_training_history.csv')
best_index = stage1_history['validation_primary_loss'].idxmin()
best_stage1 = stage1_history.loc[best_index]

print('Stage-1 root:', STAGE1_ROOT)
print('Completed Stage-1 epochs:', len(stage1_history))
print('Best completed epoch:', int(best_stage1['epoch']))
print('Best validation loss:', float(best_stage1['validation_primary_loss']))
display(stage1_history)

assert stage1_config['run_name'] == 'spectral_gmm_native_hierarchical_jpeg70_100_v1'
assert int(stage1_config['tile_size']) == 224
assert int(stage1_config['tile_stride']) == 224
assert int(stage1_config['jpeg_quality_min']) == 70
assert int(stage1_config['jpeg_quality_max']) == 100
"""
    ),
    markdown(
        """## Cell 5 — Cấu hình Stage 2 và GMM

`run_name` mới giúp tách artifact Stage 2/GMM khỏi output Stage 1. Stage 1 root được truyền rõ
qua `resume_run_roots`. Tần suất log đã giảm mạnh để tránh làm lag giao diện Kaggle.

Giữ `max_per_class_per_generator=None` để dùng toàn bộ cohort Tiny cân bằng trước đây
(thường là 500 real + 500 fake trên mỗi generator). Đặt `100` nếu chỉ muốn smoke diagnostic.
"""
    ),
    code(
        """# CELL 5 — Stage-2/GMM research configuration.
BASE_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
BASE_CONFIG.update({
    'run_name': 'spectral_gmm_native_stage1e5_stage2_gmm_v1',
    'output_parent': '/kaggle/working/spectral_gmm_native_stage2_gmm',
    'imagenet_train_root': None,  # auto-detect competition path
    'resume_run_roots': [str(STAGE1_ROOT)],
    'seed': int(stage1_config['seed']),

    # Explicitly exclude Stage 1.
    'run_phases': ['cache', 'stage2', 'gmm'],
    'stage1_resume': False,

    # Must match the frozen Stage-1 representation.
    'images_per_synset': int(stage1_config['images_per_synset']),
    'train_per_synset': int(stage1_config['train_per_synset']),
    'validation_per_synset': int(stage1_config['validation_per_synset']),
    'calibration_per_synset': int(stage1_config['calibration_per_synset']),
    'tile_size': int(stage1_config['tile_size']),
    'tile_stride': int(stage1_config['tile_stride']),
    'padding_mode': stage1_config['padding_mode'],
    'jpeg_quality_min': 70,
    'jpeg_quality_max': 100,
    'jpeg_subsampling_values': list(stage1_config['jpeg_subsampling_values']),
    'spectral_mask_implementation': stage1_config['spectral_mask_implementation'],
    'spectral_mask_radius': int(stage1_config['spectral_mask_radius']),

    # Frozen Local-CLS cache. Logs once per 250-image shard, not every 10 images.
    'cache_images_per_shard': 250,
    'cache_local_microbatch': 64,
    'cache_shard_start': None,
    'cache_shard_stop': None,
    'cache_log_every_images': 250,

    # Stage 2: low/high concat -> 768-D projection -> Global Attention -> Image CLS.
    'global_dim': 768,
    'global_depth': 4,
    'global_heads': 12,
    'global_mlp_ratio': 4.0,
    'global_dropout': 0.10,
    'global_decoder_depth': 2,
    'global_mask_ratio': 0.40,
    'stage2_epochs': 10,
    'stage2_patience': 3,
    'stage2_max_images_per_batch': 16,
    'stage2_max_tokens_per_batch': 768,
    'stage2_learning_rate': 1e-4,
    'stage2_min_learning_rate': 1e-6,
    'stage2_weight_decay': 0.05,
    'stage2_log_every_images': 1000,
    'stage2_resume': True,

    # Real-only GMM candidates.
    'gmm_components': [1, 2, 4, 8, 16],
    'gmm_covariance_type': 'diag',
    'gmm_reg_covar': 1e-6,
    'gmm_max_iter': 100,
    'gmm_n_init': 1,
    'gmm_min_component_occupancy': 0.001,
})

RUN_TINY_DIAGNOSTIC = True
TINY_MAX_PER_CLASS_PER_GENERATOR = None  # use 100 for a faster preliminary run

assert 'stage1' not in BASE_CONFIG['run_phases']
print(json.dumps(BASE_CONFIG, indent=2))
"""
    ),
    markdown(
        """## Cell 6 — Train Stage 2 và fit GMM

Cell này chạy tuần tự `cache → stage2 → gmm`. Checkpoint Stage 2 được lưu sau mỗi epoch,
gồm optimizer/scaler/history để resume. GMM dùng 80k Image CLS real để fit, 10k validation
để chọn số cụm và 10k calibration để tạo q90/q95/q99.

Nếu Kaggle timeout, đưa output version vừa tạo trở lại Input và chạy lại notebook. Pipeline sẽ
tự dùng cache shard/checkpoint Stage 2 đã hoàn thành.
"""
    ),
    code(
        """# CELL 6 — No Stage-1 training occurs here.
core_result = run_pipeline(BASE_CONFIG)
core_result
"""
    ),
    markdown(
        """## Cell 7 — Kiểm tra artifact Stage 2/GMM

Cell này xác nhận checkpoint tốt nhất, GMM, scaler và ngưỡng real-only đã tồn tại trước khi
chạy TinyGenImage.
"""
    ),
    code(
        """# CELL 7 — Core artifact audit.
OUTPUT_ROOT = Path(BASE_CONFIG['output_parent']) / BASE_CONFIG['run_name']

required_outputs = [
    'checkpoints/stage2_global_last.pt',
    'checkpoints/stage2_global_best.pt',
    'metrics/stage2_training_history.csv',
    'metrics/stage3_gmm_selection.csv',
    'metrics/stage3_gmm_summary.json',
    'gmm/stage3_real_feature_scaler.joblib',
    'gmm/stage3_real_distribution_gmm.joblib',
    'gmm/stage3_real_only_thresholds.json',
    'gmm/stage3_gmm_component_statistics.npz',
]

missing = [item for item in required_outputs if not (OUTPUT_ROOT / item).is_file()]
if missing:
    raise FileNotFoundError(f'Missing Stage-2/GMM outputs: {missing}')

print('Output root:', OUTPUT_ROOT)
print('\\nStage-2 history:')
display(pd.read_csv(OUTPUT_ROOT / 'metrics/stage2_training_history.csv'))
print('\\nGMM candidates:')
display(pd.read_csv(OUTPUT_ROOT / 'metrics/stage3_gmm_selection.csv'))
print('\\nGMM summary:')
print((OUTPUT_ROOT / 'metrics/stage3_gmm_summary.json').read_text(encoding='utf-8'))
"""
    ),
    markdown(
        """## Cell 8 — TinyGenImage separation và threshold diagnostic

Kết quả gồm:

- ROC-AUC và Average Precision không phụ thuộc threshold;
- overlap coefficient, KS statistic và Cohen's d;
- kết quả tại q90/q95/q99 lấy **chỉ từ real calibration**;
- threshold oracle tối đa balanced accuracy trên Tiny, được gắn nhãn diagnostic-only;
- metric và biểu đồ riêng từng generator;
- CSV score của từng ảnh để phân tích lại mà không chạy model lần nữa.
"""
    ),
    code(
        """# CELL 8 — Optional but enabled by default for the research question.
if RUN_TINY_DIAGNOSTIC:
    tiny_summary = run_tiny_threshold_diagnostic(
        training_config=BASE_CONFIG,
        stage1_root=STAGE1_ROOT,
        output_root=OUTPUT_ROOT,
        diagnostic_overrides={
            'tiny_dataset_root': None,  # auto-detect yangsangtai/tiny-genimage
            'balance_per_generator': True,
            'max_per_class_per_generator': TINY_MAX_PER_CLASS_PER_GENERATOR,
            'diagnostic_save_every_images': 50,
            'diagnostic_resume': True,
        },
    )
    print(json.dumps(tiny_summary, indent=2, default=_json_default))
else:
    print('Tiny diagnostic skipped by RUN_TINY_DIAGNOSTIC=False')
"""
    ),
    markdown(
        """## Cell 9 — Bảng kết quả chính và inventory trước khi Save Version

Đọc bảng này trước khi quyết định quay lại train Stage 1 lâu hơn. Nếu AUC tốt nhưng q95 thấp,
representation có khả năng xếp hạng nhưng real-only calibration chưa phù hợp. Nếu AUC cũng thấp,
Stage 1/Stage 2 representation chưa tách được hai phân phối.
"""
    ),
    code(
        """# CELL 9 — Concise final report.
if RUN_TINY_DIAGNOSTIC:
    metric_root = OUTPUT_ROOT / 'tiny_diagnostic' / 'metrics'
    threshold_table = pd.read_csv(metric_root / 'threshold_comparison.csv')
    generator_table = pd.read_csv(metric_root / 'generator_metrics_oracle.csv')
    print('Threshold comparison:')
    display(threshold_table)
    print('Per-generator metrics at the single global Tiny oracle threshold:')
    display(generator_table)

print('\\nArtifact inventory by top-level directory:')
inventory = []
for child in sorted(OUTPUT_ROOT.iterdir()):
    files = [path for path in child.rglob('*') if path.is_file()] if child.is_dir() else [child]
    inventory.append({
        'path': child.name,
        'num_files': len(files),
        'size_mb': sum(path.stat().st_size for path in files) / 1024**2,
    })
display(pd.DataFrame(inventory))

print('\\nSave this directory as the notebook Output:')
print(OUTPUT_ROOT)
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

OUTPUT.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8"
)
print(f"Wrote {OUTPUT}")
