# Kaggle training plan — Spectral-GMM Branch

## Hai thí nghiệm bắt buộc

| Notebook | Local Encoder | Có train reconstructor? | Input bổ sung |
|---|---|---:|---|
| `train_spectral_gmm_mfm_from_scratch_kaggle.ipynb` | ViT-B/16 khởi tạo ngẫu nhiên | Có, masked FrequencyLoss kiểu MFM | Không |
| `train_spectral_gmm_official_mfm_kaggle.ipynb` | Official MFM ViT-B/16 | Không, encoder được freeze | `mfm_pretrain_vit_base.pth` |

Hai notebook dùng cùng ImageNet manifest, seed, split, native-resolution cache,
Global Attention và GMM. Vì vậy đây là ablation chỉ thay đổi cách học Local Encoder.

Notebook A lưu:

```text
checkpoints/local_mfm_last.pt
checkpoints/local_mfm_best.pt
checkpoints/local_encoder_best.pt
```

Notebook B kiểm tra parameter coverage của official checkpoint trước khi cache.

## 1. Mục tiêu của notebook

Notebook Kaggle phải thực hiện tuần tự và có thể resume theo phase:

1. Tự tìm ImageNet tại `ILSVRC/Data/CLS-LOC/train`.
2. Liệt kê 1.000 synset và tạo manifest cố định chỉ từ tên file.
3. Random đúng 100 đường dẫn ảnh trong mỗi synset bằng seed cố định.
4. Chia mỗi synset thành `80 train / 10 validation / 10 calibration`.
5. Tạo một JPEG-recompressed view cho mỗi ảnh với quality ngẫu nhiên trong `[70, 100]` và chroma subsampling ngẫu nhiên.
6. Dùng frozen MFM ViT-B/16 để cache local CLS của bốn view: `raw_low`, `raw_high`, `jpeg_low`, `jpeg_high`.
7. Train Global Transformer bằng masked continuous-feature prediction và JPEG consistency loss.
8. Trích một `Image CLS 768-D` cho mỗi ảnh real train.
9. Fit, chọn và lưu real-only GMM.
10. Calibrate anomaly threshold chỉ bằng real calibration.
11. Lưu manifest, cache index, checkpoint, GMM, config và training history.

## 2. Kaggle inputs bắt buộc

### ImageNet competition data

```text
/kaggle/input/imagenet-object-localization-challenge/
└── ILSVRC/
    └── Data/
        └── CLS-LOC/
            └── train/
                ├── n01440764/
                ├── n01443537/
                └── ... khoảng 1.000 synset
```

### MFM checkpoint

Người dùng phải tạo hoặc thêm một Kaggle Dataset chứa:

```text
mfm_pretrain_vit_base.pth
```

Checkpoint chính thức tham chiếu từ repository MFM:

```text
https://github.com/Jiahao000/MFM
```

Notebook không được âm thầm thay checkpoint này bằng supervised ImageNet ViT. Nếu không tìm thấy hoặc load không tương thích, notebook phải dừng trước khi cache feature.

## 3. Sampling protocol

```text
1000 synset
× 100 ảnh được random độc lập/synset
= 100.000 ảnh real
```

Mỗi synset:

```text
80 ảnh → real_train
10 ảnh → real_validation
10 ảnh → real_calibration
```

Quy tắc:

- Liệt kê file theo thứ tự ổn định trước khi shuffle.
- Seed riêng của synset được suy ra từ `global_seed + synset` bằng SHA-256.
- Không lấy 100 file đầu tiên.
- Không mở/decode ảnh trong bước tạo manifest; ImageNet chuẩn được giả định hợp lệ.
- Lỗi decode, nếu có, sẽ xuất hiện tại DataLoader của phase sử dụng ảnh.
- Folder/synset chỉ phục vụ sampling; model không nhận class ID.
- Manifest được khóa trước khi cache feature và có SHA-256 riêng.

## 4. Tạo manifest không audit ảnh

Pipeline không mở, decode, `stat` hay `verify` 100.000 ảnh trước khi train. Nó chỉ
liệt kê tên file, lấy mẫu xác định bằng seed và lưu cohort vào manifest.

Mỗi dòng manifest:

```text
sample_id
relative_path
synset
split
extension
sampling_seed
```

Summary manifest:

- Số synset và số ảnh được chọn.
- Count theo split.
- Count theo file extension.
- SHA-256 của manifest.
- Corrupt files gặp trong quá trình chọn mẫu.
- Manifest SHA-256.

## 5. JPEG augmentation

Mỗi ảnh tạo hai pixel views trước khi chia patch:

```text
raw_view  = ảnh decode RGB gốc
jpeg_view = raw_view → JPEG encode → JPEG decode
```

Với mỗi sample:

```text
quality     ~ UniformInteger(70, 100)
subsampling ~ Uniform({4:4:4, 4:2:2, 4:2:0})
```

Quality và subsampling được suy ra từ seed của sample để cache có thể tái lập tuyệt đối.

Không lưu các ảnh JPEG trung gian. Chỉ lưu local embedding.

## 6. Native-resolution tiling

```text
Ảnh RGB kích thước gốc
→ reflect padding đến bội số 224
→ chia toàn bộ ảnh thành K vùng 224×224
```

Không resize toàn ảnh về 224×224. Không random crop bỏ nội dung. Ảnh có nhiều patch được xử lý local encoder theo chunk để tránh OOM.

Normalized 2D coordinates của mỗi patch:

```text
y = 2 * (row + 0.5) / rows - 1
x = 2 * (col + 0.5) / cols - 1
```

## 7. Local MFM feature cache

MFM settings phải khớp official ViT-B/16 pretrain:

```text
input size      = 224
token size      = 16
embedding dim   = 768
depth           = 12
heads           = 12
frequency radius= 16
```

Mỗi local patch lưu:

```text
raw_low_cls
raw_high_cls
jpeg_low_cls
jpeg_high_cls
```

Tensor shard:

```text
embeddings: [total_patches_in_shard, 4, 768], float16
positions:  [total_patches_in_shard, 2], float16
offsets:    [num_images_in_shard + 1], int64
```

Cache được chia shard và lưu SHA-256 để:

- Không chạy frozen ViT lại mỗi global epoch.
- Resume khi Kaggle session bị ngắt.
- Đọc trực tiếp previous notebook output khi được Add Input vào run sau.

## 8. Global Transformer training

Target của patch:

```text
t_i = LayerNorm((raw_low_cls_i + raw_high_cls_i) / 2)
```

Student input mỗi epoch:

- Chọn raw hoặc JPEG với xác suất 0,5.
- Trong view đã chọn, chọn low hoặc high với xác suất 0,5.
- Mask 40% local vector.
- Cộng normalized 2D positional encoding.

Global encoder:

```text
dimension = 768
depth     = 4
heads     = 12
Image CLS bottleneck = true
```

Global decoder chỉ nhận `Image CLS` làm memory và positional query làm target query. Nó không nhận trực tiếp target patch token, nhằm buộc `Image CLS` chứa thông tin toàn ảnh.

Loss:

```text
L_global = L_masked_cosine
         + lambda_mse * L_masked_normalized_mse
         + lambda_cons * (1 - cosine(z_raw, z_aug))
```

Default:

```text
lambda_mse  = 1.0
lambda_cons = 0.1
```

Loss được tính trung bình theo ảnh trước khi trung bình batch.

## 9. Checkpoint protocol

Mỗi epoch lưu:

```text
checkpoints/global_encoder_last.pt
```

Khi validation loss tốt hơn:

```text
checkpoints/global_encoder_best.pt
```

Checkpoint chứa:

- Global encoder/decoder state.
- Optimizer và scheduler state.
- Epoch, best validation loss và patience state.
- Full config.
- Manifest SHA-256.
- MFM checkpoint SHA-256.
- Cache index hash.
- RNG states khi có thể.

## 10. GMM fitting

Sau khi load `global_encoder_best.pt`:

```text
real_train raw targets
→ Global Encoder không mask
→ Z_train [80000, 768]
→ StandardScaler fit trên train
→ candidate diagonal GMMs
```

Candidate:

```text
K ∈ {1, 2, 4, 8, 16}
```

Chọn GMM bằng:

- Validation average log-likelihood.
- BIC trên train.
- Component occupancy.
- Không có NaN/Inf hoặc component variance collapse.

Calibration:

```text
score = -log p(z)
threshold_fpr_5 = quantile(calibration_score, 0.95)
threshold_fpr_1 = quantile(calibration_score, 0.99)
```

Không dùng fake hoặc test để chọn GMM/threshold.

## 11. Output layout

```text
/kaggle/working/spectral_gmm_branch/<run_name>/
├── config.json
├── status.json
├── manifests/
│   ├── imagenet_real_100k.csv
│   └── manifest_summary.json
├── feature_cache/
│   ├── train_00000.pt
│   ├── validation_00000.pt
│   ├── calibration_00000.pt
│   └── index.json
├── checkpoints/
│   ├── global_encoder_best.pt
│   └── global_encoder_last.pt
├── gmm/
│   ├── real_distribution_gmm.joblib
│   ├── real_feature_scaler.joblib
│   ├── real_only_thresholds.json
│   └── gmm_component_statistics.npz
├── embeddings/
│   ├── real_train_image_cls.npy
│   ├── real_validation_image_cls.npy
│   └── real_calibration_image_cls.npy
└── metrics/
    ├── global_training_history.csv
    ├── gmm_model_selection.csv
    ├── gmm_component_occupancy.csv
    └── gmm_summary.json
```

## 12. Kaggle execution modes

Notebook A hỗ trợ các phase:

```python
RUN_PHASES = ["local_mfm", "cache", "train", "gmm"]
```

Nếu một session không đủ thời gian:

```python
# Run 1
RUN_PHASES = ["local_mfm"]

# Add output của Run 1 làm Kaggle Input, rồi Run 2
RUN_PHASES = ["cache", "train", "gmm"]
```

Cache có thể chia theo `CACHE_SHARD_START/CACHE_SHARD_STOP` để tạo nhiều version khi cần. Notebook phải ưu tiên đọc shard đã tồn tại trong previous-output input và chỉ tạo shard còn thiếu.

## 13. Acceptance checks

Notebook chỉ được coi là chạy đúng khi:

1. Tìm thấy đúng 1.000 synset ở cấu hình main.
2. Manifest main có đúng 100.000 ảnh, 100 ảnh/synset.
3. Split đúng 80/10/10 trong từng synset.
4. MFM checkpoint load đủ các layer quan trọng; không dùng random/supervised fallback.
5. Cache không chứa NaN/Inf và mỗi ảnh có ít nhất một patch.
6. Global train/validation loss hữu hạn và checkpoint best được lưu.
7. Mỗi ảnh sinh đúng một `Image CLS 768-D`.
8. GMM fit chỉ trên `real_train`.
9. Threshold fit chỉ trên `real_calibration`.
10. Tất cả artifact ghi manifest và checkpoint hash.
