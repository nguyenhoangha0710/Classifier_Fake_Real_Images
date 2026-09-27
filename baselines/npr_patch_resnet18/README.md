# Native Patch NPR + ResNet18 on Modal

Baseline này giữ ảnh ở kích thước native. Ảnh được chia thành sliding patches,
NPR được tính trên từng patch, ResNet18 dự đoán từng patch và mean logits tạo
ra một dự đoán cấp ảnh.

```text
native RGB image
  -> even-aligned overlapping patches (không resize)
  -> NPR trên từng patch
  -> forensic-friendly ResNet18
  -> mean patch logits
  -> image-level real/fake loss
```

## Bốn chế độ thí nghiệm

Preset `all` tạo 22 model:

- 1 `combined`;
- 7 `in_domain`, một model cho mỗi Tiny-GenImage generator;
- 7 `cross_generator`, leave-one-generator-out;
- 7 `train_one_generator`, train một generator và test tất cả generator.

Chỉ best checkpoint của `combined` được test trên CommunityForensics-Eval.
CommFor được scan để tìm generator, sau đó lấy fake cân bằng theo generator và
dùng chung một real reference cohort. Nhờ đó precision, recall, balanced
accuracy và ROC-AUC theo từng generator đều có đủ hai lớp.

## Modal resources

Script dùng các Modal Volume:

```text
tiny-genimage-data          -> /data
npr-patch-resnet18-outputs -> /outputs
hf-cache                    -> /hf-cache
```

Job train không bắt buộc Kaggle secret: nó đọc dataset đã có trong
`tiny-genimage-data`. Nếu volume còn trống, tạo Modal secret `kaggle-secret`
với `KAGGLE_USERNAME` và `KAGGLE_KEY`, rồi dùng downloader có sẵn:

```powershell
python -m modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py `
  --download-only
```

Việc tách download khỏi job train giúp các lần train/resume không bị chặn chỉ
vì secret không còn được attach.

## Chạy smoke test

Từ repository root:

```powershell
python -m modal run baselines/npr_patch_resnet18/train_npr_patch_resnet18_modal.py `
  --experiment-preset smoke
```

Smoke preset chạy `combined`, tối đa 200 train images, một epoch, và CommFor
quota nhỏ.

## Chạy đủ 22 experiments

```powershell
python -m modal run baselines/npr_patch_resnet18/train_npr_patch_resnet18_modal.py `
  --experiment-preset all `
  --full-train `
  --full-tiny-eval `
  --max-epochs 10 `
  --commfor-fake-per-generator 100 `
  --commfor-real-reference-size 100 `
  --commfor-max-generators 9
```

Các experiment chạy tuần tự thành các Modal function riêng để một experiment
timeout không làm mất checkpoint của experiment trước.

## Resume

Ghi lại `suite_run_id` được in khi bắt đầu, rồi chạy:

```powershell
python -m modal run baselines/npr_patch_resnet18/train_npr_patch_resnet18_modal.py `
  --experiment-preset all `
  --suite-run-id <SUITE_RUN_ID> `
  --full-train `
  --full-tiny-eval
```

Mỗi experiment tự đọc `checkpoints/latest/training_state.pt`. Checkpoint được
lưu ở optimizer boundary nên resume không khôi phục gradient dở dang.

## Chỉ chạy lại CommFor

```powershell
python -m modal run baselines/npr_patch_resnet18/train_npr_patch_resnet18_modal.py `
  --commfor-only `
  --suite-run-id <SUITE_RUN_ID> `
  --commfor-fake-per-generator 100 `
  --commfor-real-reference-size 100
```

Suite phải có sẵn:

```text
/outputs/npr_patch_resnet18/<SUITE_RUN_ID>/combined/checkpoints/best/model.pt
```

## Output quan trọng

```text
/outputs/npr_patch_resnet18/<suite_run_id>/
  combined/
  in_domain_*/
  cross_generator_*/
  train_one_generator_*/
  commfor_combined/
    dataset/selected_samples.csv
    metrics/commfor_generator_metrics.csv
    metrics/macro_summary.json
    predictions/commfor_predictions.csv
  suite_summary/
    experiment_manifest.csv
    all_experiments.csv
    combined.csv
    in_domain.csv
    cross_generator.csv
    train_one_generator.csv
```

Mỗi Tiny experiment lưu `tiny_val_generator_metrics.csv` và
`tiny_test_generator_metrics.csv`. Prediction CSV có thêm patch size, số patch,
native resolution, mean/max/std patch fake probability.

## Kiểm thử cục bộ

```powershell
python -m unittest baselines.npr_patch_resnet18.test_npr_patch_resnet18 -v
```

Các test kiểm tra full image coverage, even alignment của NPR grid, không
resize, sampling tái lập và sự tương đương NPR patch/full-image tại even origin.
