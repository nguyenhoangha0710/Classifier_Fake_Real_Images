# Cross-Dataset Evaluation

Thư mục này chứa notebook benchmark ngoài dataset chính.

File chính:

```text
train_tiny_combined_test_commfor_eval.ipynb
```

## Mục Tiêu

Train 3 baseline trên Tiny-GenImage `combined`, sau đó evaluate zero-shot trên:

```text
OwensLab/CommunityForensics-Eval
split: CompEval
```

Protocol:

```text
Train: full Tiny-GenImage combined train split
Early stopping: Tiny-GenImage train_inner/val_inner
Internal sanity test: Tiny-GenImage validation split
External benchmark: 1000 random streaming samples from CommunityForensics-Eval CompEval
```

CommunityForensics-Eval không được dùng để:

```text
train
early stopping
threshold tuning
model selection
```

## Baseline Được Chạy

```text
CLIP + linear head
ResNet-50 last layer
NPR + ResNet18 from scratch
```

## Ghi Chú

CommunityForensics-Eval được đọc bằng Hugging Face streaming để tránh tải full dataset về local. Mặc định notebook shuffle streaming bằng seed cố định rồi lấy 1000 ảnh:

```python
MAX_TRAIN_SAMPLES = None
MAX_COMMFOR_EVAL_SAMPLES = 1000
COMMFOR_SHUFFLE_BUFFER_SIZE = 100
RANDOM_SEED = 42
```

## Checkpoint

Sau khi train, notebook lưu weight tốt nhất của từng baseline tại:

```text
outputs/tiny_combined_to_commfor_eval/<baseline_name>/<run_id>/checkpoints/
```

Tên checkpoint:

```text
clip_linear_head.pt
resnet50_last_layer.pt
npr_resnet18_from_scratch.pt
```

Với CLIP baseline, checkpoint chỉ lưu linear head để tránh lưu nguyên CLIP encoder quá nặng:

```text
checkpoint_kind = clip_linear_head_only
```

Với ResNet-50 và NPR-ResNet18, checkpoint lưu model state:

```text
checkpoint_kind = full_model
```

Notebook cũng tạo manifest tổng:

```text
outputs/tiny_combined_to_commfor_eval/checkpoint_manifest_<run_id>.csv
```

File này ghi baseline, checkpoint path, checkpoint type, best epoch và các metric chính để tiện load lại khi test sau.

Trong báo cáo, thí nghiệm này nên được gọi là:

```text
cross-dataset external benchmark
```

Không gọi là in-domain evaluation, vì train trên Tiny-GenImage và test trên CommunityForensics-Eval.
