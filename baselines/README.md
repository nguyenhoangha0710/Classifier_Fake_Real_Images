# Baselines

Mỗi thư mục con tương ứng với một mô hình. Code train/test nằm trực tiếp trong
thư mục model; checkpoint, metrics và prediction đã tải về nằm trong
`artifacts/` của chính model đó.

## Cấu trúc chuẩn

```text
baselines/<model>/
├── train_*.py / train_*.ipynb
├── test_*.py / infer_*.py
├── README.md
└── artifacts/
    ├── checkpoints/   # checkpoint canonical để code tái sử dụng
    ├── runs/          # toàn bộ output của từng lần train
    └── evaluations/   # kết quả test-only
```

Các benchmark dùng nhiều mô hình nằm trong `benchmarks/`, không gán vào một
mô hình riêng. Dataset và cache dùng chung không đặt trong `baselines/`.

## Các mô hình có artifact cục bộ

| Model | Checkpoint canonical | Runs / metrics |
|---|---|---|
| CLIP linear probe | `clip_linear_probe/artifacts/checkpoints/clip_linear_head.pt` | `clip_linear_probe/artifacts/runs/` |
| NPR-ResNet18 | `npr_resnet18/artifacts/checkpoints/npr_resnet18_from_scratch.pt` | `npr_resnet18/artifacts/runs/`, `evaluations/` |
| ResNet50 | trong từng run | `resnet50/artifacts/runs/` |
| AIDE gốc — forensic-only ResNet50 | `aide_original_forensic_resnet50/artifacts/checkpoints/model.pt` | `aide_original_forensic_resnet50/artifacts/runs/` |
| AIDE gốc — full semantic + DCT/SRM | `aide_original_full/artifacts/checkpoints/model_trainable.pt` | `aide_original_full/artifacts/runs/`, `evaluations/` |
| NPR patch ResNet18 | trong từng case của run | `npr_patch_resnet18/artifacts/runs/` |
| AIDE + NPR fusion | `fusion_256_2048` hoặc `fusion_256_256` | `aide_npr_fusion/artifacts/runs/` |

Artifact lớn được `.gitignore` bỏ qua. Code, tài liệu và metadata checkpoint nhỏ
vẫn được quản lý bằng Git.
