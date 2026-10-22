# Twins–GMM: codec augmentation v1

Tài liệu hướng dẫn code cho extension chống codec shortcut. Baseline native và code trong `forensic_twins_original` được giữ để đối chứng; đây không phải cấu hình tái lập paper nguyên bản.

## 1. Mục tiêu và workflow

Real RGB → shared rotation/flip → fork native/random processing → grayscale → frozen FRE → hai residual crops không giao nhau → shared ViT-Tiny → shared projector → Barlow Twins.
Sau train: bỏ projector → freeze encoder → fit lại Gaussian/GMM real-only → anomaly score NLL.

## 2. Kiến trúc đã chốt

| Thành phần | Cấu hình |
|---|---|
| FRE | Checkpoint hiện có; frozen, eval; constrained conv 1→8, 8 filters 15×15, padding 7 |
| Encoder | Implementation hiện có: patch 8, 192-D, 6 blocks, 3 heads, MLP ratio 4, 2D RoPE, CLS |
| Projector | Shared; 192→2048→8192 theo code hiện tại |
| Loss | Barlow Twins; lambda_BT=5e-5, eps=1e-4 |
| GMM | n_components=1, full covariance, reg_covar=1e-6; input 192-D |

Hai forward dùng cùng object encoder/projector. Không concat hai view thành một residual map.

## 3. Dữ liệu

- SSL/fine-tune dùng toàn bộ train_real_200k.csv (100K ImageNet + 100K COCO) từ [twins-gmm-200k-checkpoint](https://www.kaggle.com/datasets/hoangha071005/twins-gmm-200k-checkpoint), relative path forensic_twins_v2_200k/manifests/train_real_200k.csv. Không sampling lại hoặc đổi sang cohort50K/COCO-only.
- Manifest phải loại trùng nội dung và overlap với evaluation; lưu source, split, size, magic format, hash, JPEG metadata.
- Chỉ ở bước density estimation mới reuse split160K GMM-fit (80K mỗi nguồn) / 40K calibration (20K mỗi nguồn). Monitoring lấy subset cố định của train200K, chỉ diagnostic. Không chia lại 80/10/10; xem [kaggle_inputs.md](kaggle_inputs.md).
- Calibration40K disjoint với GMM-fit nhưng đã được encoder thấy trong SSL; không mô tả là encoder-held-out và không đảm bảo FPR5% trên miền mới.
- Test chính reuse bfree_valid_clean_eval.csv từ [evaluation-twins-bfree](https://www.kaggle.com/datasets/hoangha071005/evaluation-twins-bfree), full-run V2 đã đối chiếu: forensic_twins_augmented_gmm_full/manifests/bfree_valid_clean_eval.csv. Cohort 1.557 source IDs = 1.557 real + 9.342 fake; đã loại COCO overlap với train200K, giữ nguyên membership. Mount thêm [B-Free pixels](https://www.kaggle.com/datasets/baopv051/b-free-training-data).
- Fake không vào SSL/PCA/GMM/calibration; không dùng nhãn B-Free chọn checkpoint/threshold. B-Free đã xem kết quả nên là benchmark hồi quy, không phải test mù mới. Tiny/ComFor là bổ sung.
- Chỉ có real JPEG thì không claim loại bỏ hoàn toàn codec shortcut. Real lossless/RAW-derived là extension riêng; JPEG→PNG không phải lossless gốc.

## 4. Tạo hai view

- Decode RGB một lần; rotation uniform {0,90,180,270}; horizontal/vertical flip Bernoulli(0.5), shared giữa hai view.
- View 1 identity. View 2 chọn đúng một: 50% JPEG; 35% downscale→upscale về W×H→JPEG; 15% identity.
- JPEG: integer Q uniform [80,100], subsampling uniform {4:4:4,4:2:2,4:2:0}; Pillow backend/version cố định.
- Scale uniform [0.75,1.0]; interpolation uniform {bilinear,bicubic,Lanczos}, chọn một loại dùng cho cả down/up. Làm tròn kích thước, clamp >=1.
- Processing trên RGB đầy đủ, trong RAM, trước grayscale/FRE. Giữ file nguồn bất biến; không thêm blur/noise ở v1.
- RGB/gray float [0,1]; JPEG round-trip dùng uint8; không chia 255 hai lần.
- Sample hai residual crops vuông từ {64,96,128}; size mỗi view chọn theo iteration, chung trong global batch. Chọn tọa độ theo từng ảnh, intersection=0.
- Resample cặp size nếu không fit; sau 20 lần dùng 64×64 nếu khả thi, nếu không skip/log. Không resize để cứu crop; tránh biên ngoài residual ít nhất 7 pixel.

## 5. Training

- Warm-start encoder và projector từ checkpoint native; FRE dùng đúng checkpoint/hash cũ. Thiếu projector: khởi tạo mới và ghi run metadata.
- Freeze FRE; train toàn encoder/projector; reset optimizer, scheduler, scaler và epoch counter; đây là fine-tune, không resume nguyên trạng.
- AdamW, LR=6e-5, WD=0.0025, betas=(0.9,0.999), eps=1e-8.
- 20 epochs, warm-up 2 epochs, cosine đến 1e-6; gradient clip=1; FP16 AMP trên T4.
- Global batch thật=128 như config native hiện tại: 2 GPU ×64. Nếu OOM phải đổi config và chạy lại control cùng batch.
- Normalize từng chiều trên global batch; C=E1_normalized^T E2_normalized/B. Gradient accumulation không thay thế global batch cho Barlow.
- Save mỗi epoch; epoch 20 là checkpoint chính định trước. Log loss diagonal/off-diagonal, feature variance/effective rank, grad norm, policy counts, skipped samples.

## 6. GMM và đánh giá

- Checkpoint nào có encoder mới thì fit GMM mới. Không dùng GMM native với encoder fine-tuned.
- Fit trên native center crop 256×256 từ real GMM-fit; không augment/TTA ở evaluation chính. Ảnh quá nhỏ skip/log.
- Score=-log p_GMM(z); threshold là percentile 95 của real calibration, khóa trước test.
- So sánh: native checkpoint; native→native train tiếp 20 epochs; native→random train 20 epochs. Hai run train tiếp phải cùng seed/data/batch/steps.
- Report native, JPEG Q95/80/65/50, resize; cùng transform cho cả nhãn; AUROC, per-generator AUROC, TPR@calibrated threshold, real FPR và paired score drift.
- Tách native-confounded, fake-codec-matched, common-reencode protocols; recompress cả hai lớp không xóa khác biệt lịch sử nén.
- Không dùng Tiny/ComFor để chọn checkpoint/hyperparameter. Chọn tham số theo fake validation riêng phải ghi rõ validation-assisted, không còn hoàn toàn real-only model selection.

## 7. Deliverables và acceptance

Encoder-only implementation hiện có: [README.md](README.md), [train_codec.py](train_codec.py), [Kaggle notebook](forensic_twins_codec_encoder_kaggle.ipynb). Notebook này không chạy GMM/test. Dataset publish được chốt là hoangha071005/forensic-twins-codec-checkpoint-encoder; Working giữ cả checkpoints, CSV logs và PNG loss curves.
Triển khai crop dùng halo7 tương đương FRE-full-image rồi crop; pair128/128 được đổi96/128 đồng bộ theo seed/batch để fit interior242 của ảnh min-side256. Checkpoint codec dùng flat last.ckpt trong run root; control native_native phải tách output và tắt publish vào slug codec.

Code extension ở `baselines/forensic_twins_codec/`; reuse core native, thêm policy/config/warm-start/evaluation entrypoints, không đổi mặc định của native.
Lưu config, immutable manifests/hashes, checkpoint/hash FRE, RNG states, environment versions, logs, encoder/projector và GMM/threshold.
Checks: probabilities/seed, output size/range, disjoint crops, same-object weights, FRE frozen, warm-start/reset optimizer, global-batch loss, finite covariance và threshold không đọc test labels.
Workflow tensor: [workflow.md](workflow.md). Kiến trúc: [encoder_v1.drawio](encoder_v1.drawio); plan này ưu tiên nếu figure cũ khác config.
