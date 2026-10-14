# Kế hoạch nghiên cứu GSD–Forensic Twins

## 1. Mục tiêu

Xây dựng một visual encoder học **forensic embedding của ảnh thật** mà không cần
quan sát ảnh AI-generated trong giai đoạn self-supervised pretraining.

Phương pháp kết hợp hai ý tưởng:

1. Hai crop không giao nhau từ Forensic Twins làm giảm khả năng dùng nội dung
   semantic để giải pretext task.
2. Geometric Semantic Decoupling (GSD) loại thành phần biểu diễn nằm trong
   semantic subspace của một CLIP encoder đóng băng.
3. Barlow Twins loss giữ lại tín hiệu ổn định giữa hai vùng ảnh, kỳ vọng chủ yếu
   là dấu vết acquisition/generation có tính gần-stationary.
4. Sau pretraining, embedding ảnh thật được dùng để fit Gaussian/GMM. Ảnh test
   được chấm bằng negative log-likelihood dưới real manifold.

Tên làm việc của baseline:

```text
GSD–Forensic Twins
```

Các bài báo nền tảng:

- [Forensic Twins: Self-Supervised Residual Learning for AI-Generated Image Forensics](https://arxiv.org/abs/2609.31514)
- [When Detectors Forget Forensics: Blocking Semantic Shortcuts for Generalizable AI-Generated Image Detection](https://arxiv.org/abs/2603.09242)

## 2. Giả thuyết nghiên cứu

### H1 — Disjoint-view invariance

Hai crop không giao nhau vẫn chia sẻ dấu vết image-formation nhưng chia sẻ ít
cấu trúc semantic hơn hai crop chồng lấp. Vì vậy Barlow Twins trên disjoint
crops sẽ học representation thiên về forensic hơn.

### H2 — Semantic suppression

Loại có kiểm soát các hướng semantic do frozen CLIP xác định sẽ giảm semantic
shortcut và cải thiện khả năng tổng quát sang generator chưa thấy.

### H3 — Real-manifold detection

Embedding sau self-supervised training tạo thành phân phối ảnh thật đủ chặt để
Gaussian/GMM gán anomaly score cao hơn cho ảnh AI-generated chưa xuất hiện lúc
train.

### H4 — Residual signal vẫn cần thiết

GSD chỉ loại semantic component, không tự khuếch đại micro-statistics. Một
residual-aware input/adapter có thể cải thiện representation so với RGB + GSD,
nhưng đóng góp này phải được kiểm chứng bằng ablation.

### H5 — Codec là nuisance domain riêng

Nếu không mô hình hóa JPEG/resize, detector có thể học codec thay vì forensic
evidence. Robustness phải được đo và hiệu chỉnh riêng, không được suy ra từ kết
quả trên ảnh native.

## 3. Phạm vi phiên bản đầu

MVP chỉ giải quyết:

- self-supervised pretraining trên ảnh thật;
- hai native RGB crop không giao nhau;
- frozen CLIP semantic anchor;
- shared trainable CLIP forensic encoder;
- fixed hoặc detached-adaptive GSD;
- Barlow Twins objective;
- Gaussian `K=1` trên embedding đã giảm chiều;
- đánh giá Tiny-GenImage và benchmark ngoài miền;
- robustness với JPEG, resize và real-source shift.

MVP chưa bao gồm:

- MLLM explanation;
- Stage 2 token-space alignment;
- trainable adaptive GSD không ràng buộc;
- end-to-end joint training với fake labels;
- multi-modal density estimator phức tạp;
- localization hoặc heatmap.

## 4. Câu hỏi cần trả lời

1. GSD có làm giảm semantic information trong embedding không?
2. GSD có giữ hoặc tăng thông tin về image source/acquisition process không?
3. GSD có cải thiện real-only anomaly detection so với RGB Barlow Twins không?
4. GSD có cải thiện so với frozen CLIP + GMM đơn giản không?
5. Residual transform có còn cần thiết sau khi đã dùng GSD không?
6. Joint semantic basis có ổn định hơn per-view basis không?
7. Reference basis có giữ hiệu năng trong khi giảm chi phí inference không?
8. Embedding có tổng quát qua generator, real source, dataset và codec không?
9. Improvement đến từ semantic removal hay chỉ từ regularization/giảm dimension?
10. Gaussian `K=1` đã đủ hay cần mixture nhiều component?

## 5. Nguyên tắc thí nghiệm

### 5.1. Không leakage

- Split theo `image_id` trước khi sinh crop.
- Hai crop của cùng một ảnh luôn thuộc cùng split.
- Không dùng test labels để chọn checkpoint, PCA dimension, GMM component,
  aggregation rule hoặc threshold.
- Không dùng ảnh fake test để fit PCA, covariance, Gaussian/GMM hoặc calibrate
  anomaly score.
- Nếu dùng fake validation để chọn hyperparameter, run đó phải được ghi rõ là
  `few-shot/supervised calibration`, không được gọi là real-only zero-shot.

### 5.2. So sánh công bằng

- Các ablation dùng cùng seed, split, image membership và crop manifest.
- Cùng encoder size, số epoch và compute budget khi so GSD/no-GSD.
- Báo cáo cả mean và standard deviation tối thiểu ba seed cho thí nghiệm chính.
- AUROC là metric chính cho anomaly score chưa calibrate.
- Accuracy/F1/balanced accuracy chỉ dùng threshold khóa từ validation.

### 5.3. Provenance

Mỗi run phải lưu:

- git commit;
- config đầy đủ;
- dataset manifest hash;
- crop manifest hash;
- checkpoint hash;
- semantic basis hash;
- PCA/whitening hash;
- Gaussian/GMM parameters hash;
- package versions và hardware.

## 6. Dữ liệu

### 6.1. Pretraining real-only

Ưu tiên nhiều nguồn ảnh thật để tránh học một real domain duy nhất:

- ImageNet train real images;
- MS-COCO train real images;
- real subset của Tiny-GenImage;
- real sources khác nếu license và storage cho phép.

Mỗi sample cần tối thiểu:

```json
{
  "image_id": "stable-id",
  "image_path": "...",
  "dataset_source": "ImageNet",
  "real_source": "imagenet_train",
  "split": "train",
  "native_width": 1024,
  "native_height": 768,
  "format": "JPEG"
}
```

### 6.2. Validation real-only

Dùng để:

- early stopping bằng self-supervised loss;
- fit/validate PCA và density estimator;
- chọn covariance regularization;
- chọn threshold theo target real false-positive rate;
- kiểm tra real-source generalization.

Không trộn ảnh của cùng nguồn file hoặc near-duplicate giữa train và validation.

### 6.3. Test fake

Ít nhất gồm:

- Tiny-GenImage theo generator;
- generator chưa thấy;
- CommunityForensics-Eval nếu cohort đã khóa;
- commercial generators nếu provenance rõ;
- native và post-processed variants.

## 7. Các phase triển khai

### Phase 0 — Data audit và khóa protocol

Mục tiêu:

- tạo manifest train/validation/test;
- kiểm tra duplicate và near-duplicate;
- thống kê resolution, format, JPEG quality và real source;
- định nghĩa crop policy tái lập;
- khóa protocol trước khi train.

Output:

```text
artifacts/<run_id>/dataset/
  real_train_manifest.csv
  real_validation_manifest.csv
  fake_test_manifest.csv
  crop_manifest.csv
  dataset_statistics.json
  manifest_checksums.json
```

Điều kiện qua phase:

- không có image leakage đã biết;
- mỗi ảnh đóng góp cùng số crop;
- test cohort không được dùng trong model selection.

### Phase 1 — Sanity baselines

Chạy các mốc tối thiểu:

1. Frozen CLIP embedding + Gaussian.
2. Trainable RGB Barlow Twins với overlapping crops.
3. Trainable RGB Barlow Twins với disjoint crops.
4. SRM/NPR Barlow Twins không GSD.

Mục tiêu là xác định độ khó thực của protocol trước khi thêm GSD.

### Phase 2 — Fixed GSD MVP

Cấu hình đề xuất:

```text
backbone: CLIP ViT-B/16
crop_size: 224
views: 2 disjoint native crops
semantic_basis: joint-view hoặc fixed reference
gsd_layers: final layer trước
lambda: {0.2, 0.4, 0.6, 0.8}
semantic_rank_k: {8, 16, 64}
projector_dim: 2048 hoặc 4096
objective: Barlow Twins
```

Không học `a_l`, `b_l` trong phase này.

Điều kiện qua phase:

- semantic probe giảm so với no-GSD;
- source/acquisition probe không giảm đáng kể;
- cross-generator AUROC tăng qua nhiều seed;
- GSD tốt hơn random-subspace control.

### Phase 3 — Semantic basis ablation

So sánh:

1. `per_view`: mỗi crop có semantic basis riêng;
2. `joint_view`: hai crop dùng basis chung;
3. `batch`: toàn batch dùng Batch-SVD;
4. `reference`: basis precompute từ real reference set;
5. `random`: random orthogonal subspace cùng rank;
6. `none`: không semantic suppression.

Tiêu chí ưu tiên:

- AUROC và worst-generator AUROC;
- ổn định giữa batch size;
- inference deterministic;
- chi phí tính toán;
- độ phù hợp khi fit GMM.

Mặc định ưu tiên `reference` nếu hiệu năng gần `batch`, vì representation không
phụ thuộc các ảnh khác trong inference batch.

### Phase 4 — Residual-aware extension

So sánh:

- RGB + GSD;
- SRM/FRE + forensic ViT, không GSD;
- RGB + residual adapter + GSD;
- RGB/forensic dual stream + late fusion.

Không chiếu semantic basis của RGB CLIP trực tiếp lên một residual encoder độc
lập nếu hai encoder không còn dùng chung coordinate system.

Residual adapter đề xuất:

```text
RGB patch embedding + alpha * residual patch embedding
```

Khởi tạo `alpha` nhỏ để giữ sự tương thích với CLIP pretrained space.

### Phase 5 — Real-manifold modeling

Thứ tự triển khai:

1. L2 normalization.
2. PCA/whitening với dimension `{64, 128, 192}`.
3. Gaussian `K=1`, covariance shrinkage.
4. GMM `K={2,4,8,16}` nếu `K=1` chưa đủ.
5. Chọn `K` bằng real-validation likelihood hoặc BIC/AIC.

Anomaly score:

```text
score = -log p(z | real manifold)
```

Threshold real-only:

```text
tau = quantile(real_validation_scores, 1 - target_fpr)
```

Các target FPR nên báo cáo: `1%`, `5%`, `10%`.

### Phase 6 — Image-level aggregation

Mỗi ảnh cung cấp số patch cố định `K_patch`. So sánh:

- mean NLL;
- median NLL;
- max NLL;
- top-k mean NLL;
- mean + standard deviation;

Aggregation rule được chọn trên validation và khóa trước test.

### Phase 7 — Robustness

Test riêng:

- JPEG Q95/Q90/Q80/Q70/Q50;
- PNG round-trip;
- resize 0.75x và restore;
- Gaussian blur;
- center crop;
- screenshot/re-upload nếu có.

Ba training/calibration regime:

1. native-only;
2. shared degradation giữa hai view;
3. codec-conditioned real manifold.

Không dùng destructive augmentation độc lập trên hai view trong run chính cho
đến khi ablation chứng minh nó không xóa forensic signal.

### Phase 8 — Integration với pipeline lớn

Chỉ thực hiện khi encoder vượt điều kiện chấp nhận:

- dùng forensic embedding làm nhánh forensic của Stage 1;
- so sánh với NPR-ResNet18 và D2-FAD spectral branch;
- thử fusion với semantic encoder đã được debias;
- freeze encoder trước Stage 2 token-space alignment;
- không để language loss Stage 3 thay đổi real-manifold embedding.

## 8. Ablation matrix bắt buộc

| ID | Crop | Input | GSD | Basis | Loss | Density |
|---|---|---|---|---|---|---|
| A | overlap | RGB | none | none | Barlow | Gaussian |
| B | disjoint | RGB | none | none | Barlow | Gaussian |
| C | disjoint | RGB | fixed | joint | Barlow | Gaussian |
| D | disjoint | RGB | fixed | reference | Barlow | Gaussian |
| E | disjoint | RGB | random | random | Barlow | Gaussian |
| F | disjoint | SRM/NPR | none | none | Barlow | Gaussian |
| G | disjoint | residual adapter | fixed | reference | Barlow | Gaussian |
| H | disjoint | RGB | learnable | joint | Barlow | Gaussian |

Run H phải log phân phối `lambda`. Nếu `lambda` tiến gần 0, adaptive GSD được
xem là đã tìm đường tắt và không được dùng làm kết quả chính.

## 9. Metrics

### Detection

- AUROC;
- Average Precision;
- balanced accuracy;
- real recall/specificity;
- fake recall;
- FPR@95TPR;
- TPR@1%FPR và TPR@5%FPR;
- EER nếu cần.

### Generalization

- macro-generator AUROC;
- worst-generator AUROC;
- standard deviation giữa generator;
- leave-one-generator-family-out;
- leave-one-real-source-out;
- cross-dataset performance.

### Representation probes

- semantic k-NN accuracy: thấp hơn là tốt nếu detection không giảm;
- generator/source k-NN accuracy: cao hơn là tốt;
- semantic occupancy theo layer;
- effective rank của embedding;
- eigenvalue spectrum;
- GMM condition number;
- real/fake NLL separation.

### Robustness

- AUROC drop so với native;
- TPR drop tại fixed real FPR;
- paired probability/NLL shift;
- real false-positive increase.

## 10. Tiêu chí chấp nhận mô hình

Mô hình chỉ được xem là cải thiện nếu đồng thời:

1. Vượt no-GSD baseline trên macro cross-generator AUROC qua tối thiểu ba seed.
2. Vượt random-subspace control.
3. Không làm worst-generator AUROC giảm đáng kể.
4. Semantic probe giảm nhưng source/acquisition probe không sụp.
5. Không chọn threshold hay GMM bằng test labels.
6. Real-only threshold đạt target FPR trên real source chưa thấy.
7. Báo cáo riêng native và post-processing robustness.
8. Kết quả không phụ thuộc bất thường vào một real reference dataset.

Mục tiêu MVP đề xuất:

```text
macro cross-generator AUROC improvement >= 2 điểm
worst-generator AUROC improvement >= 1 điểm
real FPR <= 5% tại threshold real-only đã khóa
JPEG Q90 AUROC drop <= 10 điểm
```

Các mục tiêu trên là tiêu chí nghiên cứu ban đầu, không phải claim cuối cùng.

## 11. Rủi ro và biện pháp giảm thiểu

### GSD tự tắt

Triệu chứng: `lambda -> 0`.

Biện pháp:

- fixed lambda;
- detach occupancy;
- đặt `lambda_min`;
- log histogram theo layer và epoch.

### Representation collapse

Triệu chứng: embedding variance thấp, covariance suy biến.

Biện pháp:

- kiểm tra diagonal/off-diagonal Barlow terms riêng;
- tăng effective batch size;
- distributed cross-correlation;
- theo dõi effective rank.

### Học codec/dataset thay vì forensics

Biện pháp:

- leave-one-real-source-out;
- format-matched cohorts;
- controlled JPEG benchmark;
- source-balanced sampling.

### GMM không ổn định

Biện pháp:

- PCA;
- shrinkage covariance;
- eigenvalue floor;
- log condition number;
- bắt đầu bằng `K=1`.

### Train–inference mismatch

Biện pháp:

- dùng cùng crop size và normalization;
- không train crop rồi infer bằng full-image resize;
- lưu crop/aggregation policy trong checkpoint config.

### Chi phí hai CLIP branch

Biện pháp:

- MVP dùng ViT-B/16;
- frozen branch chạy trong `no_grad`;
- thử reference basis precompute;
- chỉ thêm multi-layer GSD sau khi final-layer GSD có lợi.

## 12. Artifact layout

```text
baselines/gsd_forensic_twins/
  plan.md
  workflow.md
  train_*.py                    # triển khai sau
  evaluate_*.py                 # triển khai sau
  artifacts/
    checkpoints/
    runs/<run_id>/
      config.json
      dataset/
      checkpoints/
      semantic_basis/
      embeddings/
      density_model/
      metrics/
      predictions/
      plots/
      provenance/
```

Checkpoint manifest phải ghi rõ:

- backbone và pretrained tag;
- crop size;
- patch size;
- GSD layers;
- semantic rank;
- basis mode;
- lambda policy;
- representation dimension;
- projector dimension;
- PCA dimension;
- density model và covariance type;
- image-level aggregation rule.

## 13. Thứ tự ưu tiên thực thi

1. Khóa data manifest và controlled cohorts.
2. Viết unit test cho disjoint crop sampler.
3. Chạy frozen CLIP + Gaussian baseline.
4. Chạy RGB Barlow disjoint/no-GSD.
5. Chạy fixed final-layer GSD.
6. Thêm random-subspace control.
7. So joint basis với reference basis.
8. Fit PCA + shrinkage Gaussian.
9. Chạy JPEG/resize robustness.
10. Chỉ sau đó mới thêm residual adapter và adaptive multi-layer GSD.

