# Workflow và tensor contract của Forensic Twins

Tài liệu này định nghĩa hai run độc lập:

```text
paper_faithful_full:
  491.690 real images, cấu hình sát paper

genimage_lite_50k:
  50.000 real images, không fake khi train
  đánh giá Tiny-GenImage bằng native + codec-matched protocols

codec_diverse_real:
  50.000 real images, gồm tối đa 10.000 real lossless gốc
  dùng để đo false positive trên PNG/lossless deployment
```

Mọi metric phải mang theo `run_profile` và `eval_protocol`. Không so sánh một metric `native_confounded` với metric paper hoặc `fake_codec_matched` mà không ghi rõ protocol.

## 1. Luồng tổng thể

```text
PRETRAIN — chỉ ảnh thật

RGB image I: H×W×3
        │
        ├─ random rotation {90°, 180°, 270°}
        └─ random horizontal/vertical flip
           (sample một lần, dùng chung cho cả hai view)
        │
        ▼
grayscale + divide by 255: 1×H×W, float [0,1]
        │
        ▼
FRE pretrained, frozen
constrained Conv2d 1→8, kernel 15, padding 7
        │
        ▼
residual map: 8×H×W
        │
        ├──────── strictly-disjoint crop ────────┐
        ▼                                         ▼
R1: 8×H1×W1                              R2: 8×H2×W2
        │                                         │
        └──── shared ViT-Tiny encoder f_theta ────┘
              patch 8, linear stem, 6 blocks,
              2D RoPE, [CLS]
        │                                         │
        ▼                                         ▼
Z1: B×192                                  Z2: B×192
        │                                         │
        └──── shared nonlinear projector g_phi ───┘
        │                                         │
        ▼                                         ▼
E1: B×8192                                 E2: B×8192
        │                                         │
        └──── batch normalization across B ───────┘
                          │
                          ▼
              C = E1^T E2 / B: 8192×8192
                          │
                          ▼
            Barlow Twins loss, lambda = 5e-5


DETECTION — projector bị loại bỏ

test RGB → center crop 256×256 → grayscale /255 → frozen FRE
         → shared ViT encoder → [CLS] Ztest: 192-D
         → score = -log p_GMM(Ztest)

GMM: K=1, full covariance, fit offline chỉ bằng embedding ảnh thật.
Score càng lớn ⇒ ảnh càng bất thường/thiên về AI-generated.
```

## 2. Tensor contract

Quy ước code dùng NCHW; paper mô tả tensor ảnh theo HWC.

| Bước | Tensor | Shape code | Ghi chú |
|---|---|---:|---|
| Decode | `rgb` | `B×3×H×W` | giữ pixel gốc, chưa resize |
| Shared transform | `rgb_aug` | `B×3×H'×W'` | xoay 90/270 có thể đổi H và W |
| Grayscale | `gray` | `B×1×H'×W'` | float, chia 255, miền `[0,1]` |
| FRE | `residual` | `B×8×H'×W'` | frozen; padding giữ spatial size |
| Crop nhánh 1 | `r1` | `B×8×H1×W1` | size chung trong nhánh/batch |
| Crop nhánh 2 | `r2` | `B×8×H2×W2` | không giao `r1`; size có thể khác |
| Patch tokens v1 | `t1` | `B×(H1/8·W1/8 + 1)×192` | `+1` là `[CLS]` |
| Patch tokens v2 | `t2` | `B×(H2/8·W2/8 + 1)×192` | 2D RoPE xử lý grid thay đổi |
| Encoder outputs | `z1,z2` | `B×192` | lấy `[CLS]` |
| Projector outputs | `e1,e2` | `B×8192` | chỉ dùng lúc SSL pretrain |
| Cross-correlation | `C` | `8192×8192` | khoảng 67,1 triệu phần tử |
| GMM mean | `mu` | `192` | fit trên real embeddings |
| GMM covariance | `Sigma` | `192×192` | full covariance |

Riêng inference 256×256:

```text
B×3×256×256
→ B×1×256×256
→ B×8×256×256
→ patch grid 32×32
→ B×1025×192
→ [CLS] B×192
→ GMM NLL B
```

## 3. Workflow train chi tiết

### 3.1 Lấy batch ảnh thật

Input batch chỉ chứa real images. Hai manifest hợp lệ là:

```text
paper_faithful_full:
  ImageNet-1k train   250.000
  COCO train data     241.690

genimage_lite_50k:
  ImageNet-1k train    25.000 = 25 ảnh/class × 1.000 class
  COCO                 25.000
    ưu tiên: 12.235 train2017 + 12.765 unlabeled2017
    fallback hiện tại: 25.000 train2017 nếu input không có unlabeled2017
```

Seed sampling mặc định là 42. Trước sampling phải loại ảnh hỏng, khử trùng lặp SHA-256 và loại mọi content hash xuất hiện trong real evaluation.

Nếu deployment nhận PNG, tạo thêm run `codec_diverse_real` bằng cách thay tối đa 10K ảnh của subset 50K bằng real PNG/TIFF hoặc ảnh phát triển trực tiếp từ RAW, có nguồn và license rõ ràng. Các ảnh phải thật sự lossless từ nguồn; JPEG→PNG không hợp lệ. Giữ cả run 50K gốc làm control để đo tác động của thay đổi real-domain.

Nếu dùng loader Tiny-GenImage hiện có trong repository:

```python
index = build_kaggle_tiny_index(config)
real_pool = index[(index["split"] == "train") & (index["label"] == 0)]
```

Không dùng trực tiếp `build_kaggle_tiny_splits(..., balance_real=True)` cho pretrain vì hàm đó chủ ý trả cả real và fake. Sau khi tạo index, phải hash nội dung vì cùng một real image có thể xuất hiện dưới nhiều generator folders.

Mỗi manifest row tối thiểu có:

```text
sample_id, image_path, source, original_split, class_id_if_available,
width, height, magic_format, sha256, file_size,
jpeg_qtable_hash, jpeg_estimated_quality, jpeg_subsampling
```

Không dùng suffix để kết luận format. Kiểm tra magic bytes/Pillow `image.format` trước khi `.convert("RGB")`.

Nếu kích thước ảnh trong batch khác nhau, collate không được resize tùy tiện vì rescaling phá forensic trace. Decode từng ảnh, tạo residual và crop trước, sau đó stack các crop đã có size chung cho từng nhánh.

### 3.2 Shared non-destructive transform

Với mỗi ảnh, sample transform `T_i` rồi áp dụng một lần lên full RGB image. Hai view không được có transform độc lập. Tập transform paper nêu:

- orthogonal rotations 90°, 180°, 270°;
- horizontal flip;
- vertical flip.

Paper không nêu xác suất từng phép biến đổi, có/không identity, hay thứ tự khi kết hợp rotate và flip. Các giá trị này phải nằm trong phần `provisional` cho đến khi code chính thức được phát hành.

Không dùng random resized crop, arbitrary-angle rotation, resize, blur, noise, JPEG recompression, color jitter hay solarization trong run chính.

JPEG/PNG policy khi train:

```text
source bytes bất biến
→ decode một lần vào RGB tensor
→ không save/re-encode
→ shared rotation/flip
→ grayscale/FRE/crop trong RAM
```

Không chuyển JPEG sang PNG để “đồng bộ”: thao tác này không xóa JPEG artifacts đã nằm trong pixel. Cũng không chuyển toàn bộ real sang cùng JPEG quality vì encoder mới có thể trở thành shortcut chung của lớp real.

### 3.3 Grayscale và FRE

```text
gray = rgb_to_grayscale(rgb_aug)
gray = gray.float() / 255.0
residual = FRE(gray)
```

FRE là mô-đun pretrained lấy từ FSD, gồm 8 prediction-error filters. Implementation công khai của FSD dùng constrained convolution `1→8`, kernel `15×15`, padding `7`, rồi trả `gray - conv(gray)`. Trọng số tâm của kernel bằng 0 và tổng trọng số ngoài tâm bằng 1. FRE chạy `eval()`, không update buffer/parameter và output phải có cùng H/W với input.

Lưu ý tái lập: docstring repo FSD hiện ghi input 0–255, trong khi Forensic Twins v1 ghi rõ grayscale được chia 255 trước FRE. Pipeline paper-faithful dùng `[0,1]`; đồng thời phải lưu test thống kê residual để phát hiện mismatch với checkpoint.

### 3.4 Strictly-disjoint crop sampler

Với hai rectangle `A=(x1,y1,w1,h1)` và `B=(x2,y2,w2,h2)`, điều kiện bắt buộc là:

```text
intersection_width  = max(0, min(x1+w1, x2+w2) - max(x1, x2))
intersection_height = max(0, min(y1+h1, y2+h2) - max(y1, y2))
intersection_area   = intersection_width * intersection_height == 0
```

Sampler hoạt động theo hai cấp:

1. Mỗi iteration sample `(H1,W1)` và `(H2,W2)` độc lập; mọi ảnh trong batch dùng hai cặp size này.
2. Với từng ảnh, sample tọa độ riêng nhưng bắt buộc hai rectangle nằm trong residual map và không giao nhau.

Các size phải tương thích patch 8. Nếu ảnh không đủ chỗ cho cặp crop đã sample, không resize ảnh; sampler phải resample size/tọa độ hoặc bỏ ảnh theo policy đã log.

**Khoảng/phân phối crop size không có trong paper v1.** Đây là blocker lớn nhất đối với exact reproduction. Không được lấy inference size 256 làm train crop size chỉ vì nó là con số duy nhất được công bố.

### 3.5 Encoder

Các thành phần đã được paper xác nhận:

```text
Input channels:       8
Patch size:           8×8
Patch stem:           linear
Transformer depth:    6 blocks
Position encoding:    2D RoPE
Summary:              [CLS]
Embedding dimension:  192
Encoder parameters:   xấp xỉ 3.0M
```

Với view `Hv×Wv`, số patch token là `(Hv/8)×(Wv/8)`. 2D RoPE được dùng vì crop size thay đổi theo iteration. Encoder weights được share tuyệt đối giữa hai nhánh.

Paper không nêu số attention heads, MLP ratio, QKV bias, dropout, stochastic depth và LayerNorm epsilon. Một implementation thử nghiệm có thể dùng giá trị ViT-Tiny thông dụng, nhưng phải đánh dấu là giả định và xác nhận tổng parameter gần 3,0M; không được gọi đó là cấu hình chính thức.

### 3.6 Projector và Barlow Twins

Projector `g_phi` là nonlinear MLP dùng chung, nhận 192-D và trả 8.192-D. Paper không nêu số layer, hidden dimension, activation hoặc vị trí BatchNorm.

Sau khi chuẩn hóa từng dimension trên batch:

```text
C_ij = (1/B) * sum_b E1[b,i] * E2[b,j]

L_BT = sum_i (1 - C_ii)^2
       + lambda * sum_i sum_(j != i) C_ij^2

lambda = 5e-5
```

Trong distributed training, cross-correlation phải dùng **global batch 320**, không tính độc lập trên từng GPU rồi lấy trung bình loss. Cần all-reduce sufficient statistics/`E1^T E2` và normalization statistics để kết quả không phụ thuộc world size. Paper không mô tả implementation distributed cụ thể, nên cách làm này là yêu cầu kỹ thuật để bám đúng định nghĩa global batch.

Ma trận 8.192² có 67.108.864 phần tử, khoảng 256 MiB ở float32 chỉ riêng `C`. Loss/correlation nên tính float32 ngay cả khi forward bfloat16; đây là lựa chọn ổn định số học và phải được ghi trong config vì paper không nêu dtype của loss.

### 3.7 Optimizer và schedule

| Tham số | Giá trị |
|---|---:|
| Epochs | 80 |
| Optimizer | AdamW |
| Base learning rate | 0.0015 |
| Weight decay | 0.0025 |
| Warm-up | linear, 15 epochs |
| Sau warm-up | cosine annealing |
| Final learning rate | 0.000001 |
| Global batch size | 320 |
| Precision | bfloat16 |
| Attention kernel | FlashAttentionV2 |
| Paper hardware | 4× NVIDIA RTX 4090 |

Chưa được paper công bố: Adam betas/epsilon, gradient clipping, LR scaling theo batch, layer-wise decay, seed, epoch checkpoint-selection rule và gradient accumulation. Giữ chúng trong block `provisional_assumptions`.

Với subset 50K và cấu hình đã chốt cho 2×T4:

```text
micro-batch/GPU = 64
world size      = 2
global batch    = 128
steps/epoch     ≈ floor(50.000 / 128) = 390 (drop_last)
80 epochs       ≈ 31.200 optimizer steps
precision       = FP16
learning rate   = 0.0006
```

Không được giả định gradient accumulation thông thường biến nhiều Barlow losses micro-batch thành một Barlow loss batch 320. Muốn giữ global batch 320 phải gather/giữ `E1,E2` với autograd qua các micro-batch rồi chuẩn hóa và tính một ma trận `C` chung. Nếu không đủ bộ nhớ để làm vậy, giảm batch thật và dùng LR provisional:

```text
batch 128 → lr = 0.0015 × 128/320 = 0.0006
batch  64 → lr = 0.0015 ×  64/320 = 0.0003
```

Cross-correlation và batch normalization phải nhìn thấy effective/global batch, không chỉ micro-batch trên từng GPU.

### 3.8 Checkpoint và resume trên Kaggle

Checkpoint là yêu cầu bắt buộc, không phải tùy chọn. Chỉ save sau một optimizer step hoàn chỉnh để không phải phục hồi gradient accumulation dở dang.

Lịch save mặc định:

```text
local rolling checkpoint:  mỗi 250 optimizer steps
epoch checkpoint:          cuối mỗi epoch
persistent upload/version: cuối mỗi epoch khi đã cấu hình Kaggle Dataset slug
local retained milestone:  mỗi 5 epoch
final checkpoint:          cuối epoch 80
```

Với run 50K trên 2×T4, một epoch khoảng 390 optimizer steps ở global batch 128. Vì vậy vừa có checkpoint step 250 vừa có checkpoint cuối epoch.

Mỗi checkpoint phải chứa:

```text
encoder_state_dict
projector_state_dict
optimizer_state_dict
scheduler_state_dict
scaler_state_dict hoặc null
epoch hiện tại
next_batch_index
global_step và optimizer_step
Python / NumPy / Torch CPU / CUDA RNG states
distributed sampler epoch và seed
resolved config + config hash
train manifest hash
FRE checkpoint hash
git commit nếu có
world_size, micro_batch_size, effective_batch_size
loss/correlation statistics gần nhất
```

Quy trình ghi atomic:

```text
save → checkpoint.tmp
fsync/close
validate có đủ key và load thử header
os.replace(checkpoint.tmp, epoch_XXXX.ckpt)
compute SHA-256
update checkpoint_index.json
atomic update last.ckpt
```

Không overwrite checkpoint hợp lệ trực tiếp. Giữ ít nhất ba rolling checkpoints gần nhất và các milestone 5 epoch. Chỉ cleanup checkpoint cũ sau khi bản persistent có checksum khớp.

Thứ tự `resume=auto`:

```text
1. /kaggle/working/.../checkpoints/last.ckpt
2. checkpoint dataset/model được attach dưới /kaggle/input/**
3. local artifact path khi chạy ngoài Kaggle
4. không tìm thấy checkpoint → bắt đầu epoch 0
```

Với mỗi candidate: kiểm tra SHA-256, load trên CPU, kiểm tra schema, rồi so `config_hash`, `manifest_hash`, `fre_hash` và kiến trúc. Nếu khác, từ chối resume mặc định; chỉ cho phép explicit override và phải ghi warning vào run metadata.

Sau restore, log rõ:

```text
RESUMED_FROM=<path>
START_EPOCH=<epoch>
NEXT_BATCH_INDEX=<index>
GLOBAL_STEP=<step>
LR=<restored scheduler lr>
```

Nếu resume giữa epoch, dataloader phải tái tạo đúng sampler seed/epoch và bỏ qua tới `next_batch_index`. Exact replay còn phụ thuộc worker/prefetch, vì vậy checkpoint cuối epoch là mốc phục hồi chuẩn; mid-epoch checkpoint nhằm giảm mất tiến độ nhưng có thể không bit-exact.

`/kaggle/working` chỉ là storage của session hiện tại. Để chống mất session, checkpoint cuối mỗi epoch phải được upload/version hóa thành private Kaggle Dataset/Model hoặc notebook phải được `Save Version` để lưu output. Code hỗ trợ upload mỗi epoch khi credential, slug và network có sẵn; upload thất bại phải cảnh báo nổi bật, không được giả vờ checkpoint đã persistent.

## 4. Workflow GMM và inference

### 4.1 Trích real embedding để fit

Sau pretrain:

1. Bỏ hoàn toàn projector.
2. Freeze FRE và encoder.
3. Trích `[CLS]` 192-D cho cùng real manifest đã pretrain: 50K hoặc 491.690 ảnh.
4. Fit một Gaussian full covariance (`K=1`).

Với `K=1`, mô hình tương đương một multivariate Gaussian:

```text
p(z) = N(z | mu, Sigma)
score(z) = -log p(z)
```

Paper không nói có L2-normalize embedding, PCA, feature standardization hay covariance shrinkage trước GMM. Vì vậy run faithful không thêm các bước đó. Nếu numerical implementation cần jitter, giá trị phải được log và báo cáo như sai khác so với paper.

### 4.2 Preprocessing inference

```text
decode RGB ở kích thước gốc
→ kiểm tra min(H,W) >= 256
→ center crop 256×256
→ grayscale
→ divide by 255
→ frozen FRE
→ ViT encoder
→ lấy [CLS] 192-D
→ GMM negative log-likelihood
```

Không resize ảnh lên/xuống 256. Trong benchmark paper, real image có cạnh ngắn dưới 256 được skip và thay bằng mẫu tiếp theo; generated images đều đủ lớn.

### 4.3 Diễn giải score

- Score nhỏ: embedding có mật độ cao dưới phân phối ảnh thật.
- Score lớn: out-of-distribution so với ảnh thật, được xem là AI-generated.
- AUROC dùng score liên tục, không đảo dấu.
- GMM không chứng minh ảnh là giả; nó chỉ đo độ lệch khỏi manifold ảnh thật đã quan sát.

### 4.4 Codec-matched inference cho GenImage

Native GenImage có nguy cơ `real=JPEG`, `fake=PNG`. Model chỉ nhận pixel tensor nhưng JPEG blocking, ringing, quantization và chroma subsampling vẫn tồn tại sau decode. Vì vậy phải tạo nhánh đánh giá codec-matched:

```text
REAL branch
  native real JPEG
  → decode RGB
  → center crop 256
  → grayscale /255 → FRE → ViT → GMM

FAKE branch
  native fake PNG (giữ bất biến làm source)
  → chọn JPEG profile bằng deterministic hash/seed
  → encode JPEG đúng một lần, giữ nguyên W×H
  → decode RGB
  → center crop 256
  → grayscale /255 → FRE → ViT → GMM
```

JPEG profile không chỉ là một số quality. Một profile gồm:

```text
quantization tables
chroma subsampling
color mode
encoder backend/version
```

Xây empirical profile pool từ real JPEG của cùng evaluation cohort hoặc real training pool. Với mỗi generator, sample profile theo cùng một phân phối; không để profile phụ thuộc label, generator name hoặc image content. Dùng seed cố định và lưu `source_fake_sha256 → profile_id → derived_sha256`.

Không overwrite file PNG. Derived JPEG đặt trong thư mục artifact riêng. Nếu PNG có alpha, không silently composite; phải flag/reject hoặc dùng một policy nền cố định được ghi trong manifest.

Protocol này giảm confound nhưng chưa hoàn hảo vì fake JPEG có thể cùng một encoder backend còn real JPEG đến từ nhiều encoder. Vì vậy phải chạy thêm `common_reencode` sensitivity check, nhưng không dùng nó thay cho kết quả chính vì real có nguy cơ double compression.

## 5. Protocol đánh giá

### 5.1 Benchmark của paper

Paper đánh giá 27 generator families, chia thành 8 GAN, 14 Diffusion/Autoregressive và 5 Commercial. Với mỗi family:

- tối đa 1.000 generated images (Firefly: 800);
- cân bằng với cùng số ảnh thật;
- chạy riêng với real ImageNet-1k validation và MS-COCO 2017 test;
- ảnh không bị resize, crop hoặc re-encode trước khi vào preprocessing riêng của detector;
- metric chính là AUROC.

Kết quả mục tiêu:

| Nhóm | AUROC paper |
|---|---:|
| GAN | 98,17 |
| Diffusion | 98,56 |
| Commercial | 96,10 |
| Global | 97,99 |

Đối với dataset nội bộ/Tiny-GenImage, giữ nguyên model và scoring nhưng báo kết quả ở bảng riêng mang tên `local adaptation`; không so trực tiếp như thể đó là reproduction của benchmark paper.

### 5.2 Tiny-GenImage/GenImage

Tạo ba manifest evaluation bất biến:

| `eval_protocol` | Real | Fake | GMM | Vai trò |
|---|---|---|---|---|
| `native_confounded` | JPEG native | PNG native | native-real GMM | Diagnostic format bias |
| `fake_codec_matched` | JPEG native | fake encode một lần theo empirical real profile | native-real GMM | Kết quả chính |
| `common_reencode` | re-encode cùng pipeline | re-encode cùng pipeline | fit lại trên real cùng pipeline | Sensitivity check |

Quy tắc:

1. Chỉ dùng split real/fake validation của Tiny-GenImage để đánh giá; fake train split không được tham gia pretrain, GMM hay hyperparameter tuning.
2. Cân bằng real/fake trong từng generator evaluation.
3. Một real image dùng làm đối chứng phải có cùng membership giữa ba protocols.
4. Fake codec matching được thực hiện trên full image trước center crop/FRE.
5. Không resize, không đổi aspect ratio và không re-encode nhiều lần.
6. Báo AUROC từng generator, macro-generator AUROC và pooled AUROC.
7. Báo cả `delta_codec = AUROC_native - AUROC_codec_matched`. Delta lớn cho thấy native metric bị thổi phồng bởi format.

### 5.3 Codec-only probe

Fit một probe không nhìn pixel content, chỉ dùng:

```text
magic format, width, height, aspect ratio, file size/pixel,
estimated JPEG quality, qtable hash, chroma subsampling
```

Probe chỉ là công cụ audit, không phải detector cuối. Mục tiêu trên `fake_codec_matched` là AUROC gần 0,5. Nếu probe vẫn cao, dừng việc diễn giải Forensic Twins metric như khả năng phát hiện generator và audit lại codec manifest.

Ngoài ra báo phân phối GMM NLL cho:

```text
real JPEG native
real lossless/PNG thật sự, nếu có
fake PNG native
fake JPEG-matched
```

Không coi JPEG→PNG là real lossless vì artifacts JPEG cũ vẫn tồn tại.

## 6. Assertions và test bắt buộc

```text
assert input_is_real_only_during_pretrain
assert gray.min() >= 0 and gray.max() <= 1
assert residual.shape[1] == 8
assert all(not p.requires_grad for p in FRE.parameters())
assert H1 % 8 == W1 % 8 == H2 % 8 == W2 % 8 == 0
assert intersection_area(crop1, crop2) == 0
assert z.shape[-1] == 192
assert e.shape[-1] == 8192
assert C.shape == (8192, 8192)
assert projector_not_used_for_gmm
assert gmm.n_components == 1
assert gmm.covariance_type == "full"
assert inference_crop == (256, 256)
assert no_inference_resize
assert pretrain_manifest.label.unique() == [0]
assert pretrain_manifest.sha256.is_unique
assert no_hash_overlap(pretrain_manifest, eval_real_manifest)
assert source_files_are_immutable
assert codec_profile_independent_of_label_and_generator
```

Ngoài shape test, cần:

- test cùng seed tạo cùng transform/crop;
- test hai nhánh share đúng object encoder/projector;
- test loss gần 0 khi `C=I`;
- test distributed correlation khớp single-process trên cùng global batch;
- test rotate 90/270 xử lý đúng non-square image;
- test checkpoint resume khôi phục RNG và LR;
- test ngắt giả lập sau một optimizer step rồi resume đúng epoch/batch/global step;
- test checkpoint bị cắt/corrupt bị bỏ qua và fallback sang bản hợp lệ kế tiếp;
- test config/manifest/FRE hash mismatch bị từ chối;
- test atomic save không làm hỏng `last.ckpt` khi tiến trình bị interrupt;
- test GMM serialization cho score giống nhau trong sai số cho phép;
- test format bằng magic bytes thay vì extension;
- test fake codec-matched giữ nguyên width/height và chỉ có đúng một encode operation;
- test cùng seed tạo cùng `profile_id` và cùng derived checksum;
- test phân phối JPEG profile giữa real/fake bằng histogram/JS divergence;
- test codec-only probe và lưu AUROC;
- test common-reencode dùng đúng GMM được refit, không tái sử dụng native GMM.

## 7. Những điểm không được “đoán thành official”

1. Train crop sizes và sampling distribution.
2. Xác suất/khả năng kết hợp rotation và flips.
3. Attention heads và toàn bộ chi tiết block ViT ngoài depth/output.
4. Projector depth, hidden sizes, normalization và activation.
5. AdamW betas/epsilon, gradient clipping và seed.
6. Exact 491.690 file IDs và checkpoint-selection policy.
7. GMM regularization/jitter và việc normalize embedding.
8. Codec matching cho GenImage là protocol của dự án, không phải thành phần trong paper.

Khi code chính thức xuất hiện, đây là checklist audit đầu tiên. Mỗi thay đổi phải cập nhật `paper_faithful.yaml`, version nguồn và kết quả ablation.

## 8. Nguồn

- [Forensic Twins — arXiv HTML v1](https://arxiv.org/html/2609.31514v1)
- [Forensic Twins — arXiv PDF](https://arxiv.org/pdf/2609.31514)
- [Official FSD repository containing FRE](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25)
- [Official FRE source](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/blob/main/fsd/fre.py)
