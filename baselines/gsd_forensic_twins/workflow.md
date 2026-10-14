# Workflow GSD–Forensic Twins

## 1. Tổng quan

Workflow học một forensic representation từ ảnh thật bằng cách kết hợp:

- hai crop không giao nhau;
- frozen CLIP làm semantic anchor;
- shared trainable CLIP làm forensic encoder;
- GSD loại semantic projection;
- Barlow Twins học cross-view invariance;
- Gaussian/GMM mô hình hóa real manifold.

```text
Real native RGB image
        |
        v
Shared non-destructive transform
        |
        +-----------------------------+
        |                             |
        v                             v
Disjoint crop x1               Disjoint crop x2
        |                             |
        +-----------+     +-----------+
        |           |     |           |
        v           v     v           v
Frozen CLIP     Shared trainable   Frozen CLIP     Shared trainable
anchor S1       CLIP features F1   anchor S2       CLIP features F2
        |           |     |           |
        +------ semantic basis -------+
                    |
                    v
            Fixed/adaptive GSD
             |             |
             v             v
            z1             z2
             |             |
             v             v
        Shared projector g_phi
             |             |
             +------v------+
                Barlow Twins
```

Sau pretraining:

```text
Discard projector
Freeze encoder + GSD policy
        |
        v
Extract real embeddings
        |
        v
PCA / whitening
        |
        v
Fit real-only Gaussian/GMM
        |
        v
Test anomaly score = -log p(z)
```

## 2. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| `B` | batch size |
| `H, W` | kích thước ảnh native |
| `P` | crop size, mặc định 224 |
| `N` | số patch token trong một crop |
| `d` | hidden dimension của CLIP |
| `k` | rank semantic subspace |
| `D` | forensic representation dimension |
| `D_p` | projector output dimension |
| `x_1, x_2` | hai crop không giao nhau |
| `S_1, S_2` | frozen semantic-anchor tokens |
| `F_1, F_2` | trainable forensic tokens |
| `V` | orthonormal semantic basis |
| `z_1, z_2` | representations sau GSD |
| `e_1, e_2` | projector outputs dùng cho Barlow loss |

## 3. Stage A — Input và augmentation

### 3.1. Input

Ảnh đầu vào được decode về RGB nhưng giữ kích thước native:

```text
I in R^(H x W x 3)
```

Không resize toàn ảnh về `224 x 224` trước khi crop, vì resize có thể thay đổi
micro-statistics.

### 3.2. Shared non-destructive transform

Chỉ dùng transform bảo toàn forensic signal trong MVP:

- horizontal flip;
- vertical flip;
- rotation thuộc `{0, 90, 180, 270}`.

Transform được lấy một lần cho mỗi ảnh và áp dụng trước khi chọn cả hai crop.

Không dùng mặc định:

- arbitrary-angle rotation;
- Gaussian blur;
- random resize;
- color jitter;
- JPEG độc lập giữa hai view.

### 3.3. Disjoint crop sampler

Chọn hai rectangle:

```text
R1 = [x1, x1 + P) x [y1, y1 + P)
R2 = [x2, x2 + P) x [y2, y2 + P)
```

Ràng buộc:

```text
intersection_area(R1, R2) == 0
```

Nếu ảnh quá nhỏ:

1. ưu tiên reflection padding;
2. ghi lại padding trong metadata;
3. không silently resize;
4. có thể loại ảnh nếu kích thước không đạt điều kiện protocol.

Sampler phải lưu:

- `image_id`;
- seed;
- transform;
- tọa độ hai crop;
- native size;
- padding nếu có.

## 4. Stage B — Hai encoder dùng chung initialization

### 4.1. Frozen semantic anchor

`E_s` là CLIP image encoder đóng băng:

```text
requires_grad(E_s) = False
E_s.eval()
```

Với mỗi view:

```text
S_v^l = E_s^l(x_v) in R^(B x N x d)
```

Chỉ dùng non-CLS patch tokens để ước lượng semantic subspace.

Frozen branch luôn chạy trong `no_grad` và không nhận gradient từ Barlow loss.

### 4.2. Shared trainable forensic encoder

`E_f` khởi tạo từ cùng CLIP checkpoint với `E_s`, nhưng được cập nhật bằng
self-supervised loss.

Hai view bắt buộc dùng cùng một `E_f`:

```text
F_1 = E_f(x_1)
F_2 = E_f(x_2)
```

Không tạo hai forensic encoder độc lập.

## 5. Stage C — Semantic basis

### 5.1. Per-view basis

Với frozen patch tokens của view `v`:

```text
S_v in R^(B x N x d)
```

SVD riêng từng view/sample:

```text
S_v = U_v Sigma_v V_v^T
```

Lấy top-`k` right singular vectors:

```text
V_v^(k) in R^(d x k)
```

Chế độ này sát Single-SVD nhưng có thể tạo hai projector khác nhau cho hai
view.

### 5.2. Joint-view basis — mặc định cho MVP

Nối tokens của hai view theo token dimension:

```text
M = concat(S_1, S_2, dim=tokens)
M in R^(B x 2N x d)
```

Có thể tính basis theo từng sample hoặc gộp batch:

```text
M_batch in R^(B * 2N x d)
M_batch = U Sigma V^T
V_k = V[:, :k]
```

Cả hai view dùng chung `V_k`.

### 5.3. Reference basis — ưu tiên cho deployment

Precompute basis trên real reference set:

```text
M_ref = concat frozen CLIP patch tokens từ reference images
M_ref = U_ref Sigma_ref V_ref^T
V_ref_k = V_ref[:, :k]
```

Lưu:

```text
semantic_basis/layer_<l>.pt
semantic_basis/manifest.json
```

Reference basis giúp:

- inference deterministic;
- không phụ thuộc batch composition;
- có thể bỏ frozen anchor forward khi inference;
- giảm chi phí tính toán.

## 6. Stage D — Geometric Semantic Decoupling

Với trainable feature tại layer `l`:

```text
F_v^l in R^(B x (N + 1) x d)
```

Semantic projection:

```text
F_parallel = F_v^l V_k V_k^T
```

Semantic occupancy:

```text
r_v,l = ||F_parallel||_F^2 / (||F_v^l||_F^2 + eps)
```

### 6.1. Fixed GSD

```text
F_hat = F - lambda * F_parallel
```

Đây là cấu hình mặc định của MVP.

### 6.2. Detached-adaptive GSD

```text
lambda_v,l = clamp(alpha_l * stop_gradient(r_v,l), lambda_min, lambda_max)
F_hat = F - lambda_v,l * F_parallel
```

`stop_gradient` ngăn encoder thay đổi occupancy chỉ để giảm suppression.

### 6.3. Learnable GSD — chỉ dùng trong ablation sau

```text
lambda_v,l = lambda_max * sigmoid(a_l * r_v,l + b_l)
```

Phải log:

- mean/std/min/max lambda;
- lambda histogram theo layer;
- lambda theo epoch;
- tương quan lambda với Barlow loss.

Nếu lambda tiến gần 0 trên phần lớn sample, run được đánh dấu là `GSD bypass`.

### 6.4. Vị trí GSD

Thứ tự triển khai:

1. final-layer output;
2. last two transformer blocks;
3. last four transformer blocks.

Không triển khai multi-layer ngay trước khi final-layer GSD chứng minh hiệu quả.

## 7. Stage E — Representation và projector

Representation mặc định lấy từ CLS token sau GSD:

```text
z_v = LayerNorm(F_hat_v[:, CLS, :])
z_v in R^(B x D)
```

Projector dùng chung cho hai view:

```text
e_v = g_phi(z_v)
e_v in R^(B x D_p)
```

Projector có thể là:

```text
Linear(D, hidden)
BatchNorm
GELU
Linear(hidden, hidden)
BatchNorm
GELU
Linear(hidden, D_p)
```

Projector chỉ dùng khi pretrain và bị bỏ trước khi fit density model.

## 8. Stage F — Barlow Twins objective

Chuẩn hóa mỗi projector dimension trên batch:

```text
e1_norm = batch_normalize(e1)
e2_norm = batch_normalize(e2)
```

Cross-correlation:

```text
C = e1_norm^T e2_norm / B
C in R^(D_p x D_p)
```

Loss:

```text
L_invariance = sum_i (1 - C_ii)^2
L_redundancy = sum_i sum_(j != i) C_ij^2
L_BT = L_invariance + beta * L_redundancy
```

Log riêng hai thành phần để phát hiện collapse.

### Effective batch size

Cross-correlation cần batch đủ lớn. Gradient accumulation thông thường không
tương đương một Barlow batch lớn vì correlation đã được tính trong từng
micro-batch.

Nếu GPU không đủ:

- distributed all-gather projector embeddings;
- cross-batch memory có kiểm soát;
- hoặc giảm projector dimension.

Không được chỉ tăng `gradient_accumulation_steps` rồi tuyên bố effective batch
cho Barlow correlation đã tăng.

## 9. Stage G — Checkpoint selection

Checkpoint được chọn bằng real-validation criteria:

- validation Barlow loss;
- embedding effective rank;
- covariance condition number;
- stability giữa hai crop;
- semantic/source probe trên validation nếu protocol cho phép.

Không chọn checkpoint bằng fake test AUROC.

Checkpoint lưu:

```text
encoder_state_dict
gsd_config
semantic_basis hoặc basis manifest
projector_state_dict
optimizer_state_dict
scheduler_state_dict
training_epoch/step
dataset/crop hashes
```

## 10. Stage H — Fit real manifold

### 10.1. Trích embedding

Sau pretraining:

```text
discard g_phi
freeze E_f
freeze GSD policy
```

Với mỗi real calibration image, lấy số patch cố định `K_patch` và thu:

```text
z_(i,1), ..., z_(i,K_patch)
```

Mỗi ảnh phải có cùng trọng số trong tập fit.

### 10.2. Normalization và PCA

Pipeline mặc định:

```text
z
→ optional L2 normalization
→ PCA fit trên real train embeddings
→ optional whitening
→ z_pca in R^(D_gmm)
```

PCA tuyệt đối không fit lại trên test.

### 10.3. Gaussian `K=1`

Ước lượng:

```text
mu = mean(z_pca)
Sigma = covariance(z_pca)
```

Shrinkage:

```text
Sigma_shrunk = (1 - gamma) * Sigma + gamma * sigma2 * I
```

Anomaly score:

```text
s_patch = -log N(z_pca; mu, Sigma_shrunk)
```

### 10.4. GMM extension

Nếu real embeddings multi-modal:

```text
p(z) = sum_(m=1)^K pi_m N(z; mu_m, Sigma_m)
s_patch = -log p(z)
```

Chọn `K` chỉ bằng real-validation likelihood/BIC/AIC trong setting real-only.

## 11. Stage I — Image-level inference

```text
Test image
  → deterministic/native patch sampler
  → K_patch crop embeddings
  → PCA transform
  → K_patch anomaly scores
  → image-level aggregation
```

Các aggregation candidate:

```text
mean_score   = mean(scores)
median_score = median(scores)
max_score    = max(scores)
topk_score   = mean(top_k(scores))
```

Khuyến nghị bắt đầu bằng `median` và `top-k mean`.

Prediction:

```text
fake if image_score >= threshold
real otherwise
```

Threshold được khóa từ real validation quantile hoặc validation protocol riêng.

## 12. Tensor contract tham chiếu

Với CLIP ViT-B/16, crop `224 x 224`:

```text
x1, x2                 [B, 3, 224, 224]
patch grid              14 x 14
non-CLS tokens          [B, 196, 768]
tokens with CLS         [B, 197, 768]
semantic basis V_k      [768, k]
GSD output              [B, 197, 768]
CLS representation z    [B, 768]
projector output e      [B, D_p]
cross-correlation C     [D_p, D_p]
PCA embedding           [num_patches, D_gmm]
patch anomaly scores    [num_patches]
image anomaly score     scalar
```

Các dimension phải đọc từ backbone config thay vì hard-code nếu hỗ trợ backbone
khác.

## 13. Shape và numerical assertions

Code phải kiểm tra tối thiểu:

```python
assert x1.ndim == 4 and x2.ndim == 4
assert x1.shape == x2.shape
assert intersection_area(box1, box2) == 0
assert semantic_tokens.shape[-1] == forensic_tokens.shape[-1]
assert semantic_basis.shape[0] == forensic_tokens.shape[-1]
assert semantic_basis.shape[1] == semantic_rank
assert torch.isfinite(semantic_basis).all()
assert torch.isfinite(z1).all() and torch.isfinite(z2).all()
assert cross_correlation.shape == (projector_dim, projector_dim)
assert torch.isfinite(cross_correlation).all()
assert pca_dim <= representation_dim
assert covariance_condition_number < configured_maximum
```

Kiểm tra orthogonality:

```text
||V_k^T V_k - I||_F < tolerance
```

## 14. Unit tests bắt buộc

### Crop sampler

- hai crop không overlap;
- coverage đúng;
- seed tái lập;
- không resize ngầm;
- padding được ghi metadata;
- cùng image không đi qua hai split.

### GSD

- `lambda=0` trả đúng input;
- `lambda=1` loại đúng projection;
- random orthonormal basis giữ shape;
- occupancy nằm trong khoảng hợp lệ có tolerance số học;
- frozen anchor không nhận gradient;
- hai view dùng chung trainable encoder;
- detached-adaptive lambda không truyền gradient qua occupancy.

### Barlow loss

- identical embeddings có diagonal correlation cao;
- constant embeddings bị phát hiện collapse;
- off-diagonal term được loại đúng khỏi diagonal;
- distributed correlation tương đương single-process trên cùng batch.

### Density model

- PCA chỉ fit train;
- covariance positive definite sau shrinkage;
- NLL finite;
- save/load cho score giống nhau;
- test manifest không xuất hiện trong fit records.

## 15. Logging

Mỗi epoch lưu:

- total Barlow loss;
- invariance term;
- redundancy term;
- embedding variance;
- effective rank;
- semantic occupancy theo layer;
- lambda statistics theo layer;
- learning rate;
- gradient norm;
- GPU memory và throughput.

Sau khi fit density model lưu:

- PCA explained variance;
- covariance eigenvalues;
- condition number;
- real train/validation NLL distribution;
- per-generator test distribution;
- per-real-source false-positive rate.

## 16. Output chuẩn

```text
artifacts/runs/<run_id>/
  config.json
  dataset/
    manifests...
  checkpoints/
    latest/
    best/
  semantic_basis/
    layer_*.pt
    manifest.json
  embeddings/
    real_train.*
    real_validation.*
  density_model/
    pca.*
    gaussian_or_gmm.*
    threshold.json
  predictions/
    patch_predictions.csv
    image_predictions.csv
  metrics/
    overall.json
    generator_metrics.csv
    real_source_metrics.csv
    robustness_metrics.csv
    representation_probes.json
  plots/
    nll_distribution.png
    roc_pr_curves.png
    lambda_by_layer.png
    semantic_source_probe.png
    covariance_spectrum.png
  provenance/
    checkpoints.json
    software_versions.json
    manifest_checksums.json
```

## 17. Điều kiện dừng sớm nghiên cứu

Tạm dừng mở rộng nếu một trong các điều sau xảy ra:

- GSD không vượt random-subspace control;
- semantic probe giảm nhưng AUROC giảm ổn định qua nhiều seed;
- learnable lambda luôn tiến về 0;
- embedding covariance collapse;
- detector chủ yếu phân biệt format/codec;
- real-source shift tạo false-positive rate không chấp nhận được;
- fixed final-layer GSD không cho tín hiệu cải thiện đủ rõ để biện minh cho
  multi-layer/two-encoder compute.

Khi đó cần quay lại kiểm tra residual input, crop policy hoặc objective trước khi
tăng độ phức tạp của model.
