# Kế hoạch tái lập Forensic Twins

## 1. Mục tiêu và phạm vi

Thư mục này dành cho baseline **Forensic Twins nguyên bản**, tách biệt với thử nghiệm `gsd_forensic_twins`. Mục tiêu là tái lập pipeline trong bản preprint arXiv v1 ngày 25/09/2026:

1. Học biểu diễn forensic chỉ từ ảnh thật, không dùng ảnh AI khi pretrain.
2. Dùng Forensic Residual Extractor (FRE) frozen và hai crop residual không giao nhau.
3. Tối ưu encoder bằng Barlow Twins.
4. Bỏ projector, fit GMM một thành phần trên embedding ảnh thật.
5. Đánh giá zero-shot trên các generator chưa từng thấy khi train.

Đây là kế hoạch **paper-faithful**, không thêm GSD, CLIP, JPEG augmentation, PCA, classifier nhị phân hoặc fake image vào training chính. Mọi thay đổi như vậy phải được đặt trong nhánh/config `extension`, không được báo cáo như kết quả tái lập nguyên bản.

## 2. Mức độ tái lập hiện tại

Paper v1 nói code, weight và exact split sẽ được công bố sau khi bài được chấp nhận, nhưng chưa cung cấp liên kết. Vì vậy không thể cam kết bit-exact ở thời điểm lập kế hoạch này.

| Hạng mục | Paper công bố | Trạng thái triển khai |
|---|---:|---|
| Dữ liệu real pretrain | 250.000 ImageNet-1k train + 241.690 MS-COCO train = 491.690 | Có thể tái lập số lượng; chưa có exact file list |
| Augmentation | xoay trực giao 90/180/270 độ; flip ngang/dọc; cùng phép biến đổi cho hai view | Đủ thông tin |
| Tiền xử lý | RGB → grayscale → chia 255 → FRE frozen | Đủ thông tin |
| FRE | 1 kênh vào, 8 residual channels; checkpoint pretrained từ FSD | Kiến trúc repo FSD là constrained conv 15×15; cần khóa checkpoint/hash |
| View | hai crop residual hoàn toàn không giao nhau; size hai nhánh có thể khác | Thiếu miền/phân phối sampling kích thước crop |
| Encoder | ViT-Tiny, 6 blocks, patch 8, linear stem, 2D RoPE, `[CLS]`, output 192-D | Thiếu một số siêu tham số nội bộ |
| Projector | nonlinear, output 8.192-D | Thiếu số layer, hidden width, activation và BN chi tiết |
| Loss | Barlow Twins, `lambda = 5e-5` ở cấu hình cuối | Đủ công thức; chi tiết distributed BN chưa nêu |
| Train | 80 epochs, AdamW, LR 0.0015, WD 0.0025, warm-up 15 epochs, cosine đến 1e-6, batch 320, bfloat16, FlashAttentionV2 | Hầu hết đủ thông tin |
| Inference | center crop 256×256, không resize trước detector | Đủ thông tin |
| One-class model | GMM `K=1`, full covariance, fit trên cùng ảnh real dùng pretrain | Đủ cấu hình chính; covariance regularization chưa nêu |

Các kết quả tạo ra trước khi code chính thức xuất hiện phải gắn nhãn **provisional reproduction**, không phải exact reproduction.

## 3. Cấu hình mục tiêu bám sát paper

### 3.1 Dữ liệu

- Pretrain chỉ dùng ảnh thật:
  - 250.000 ảnh từ ImageNet-1k training set.
  - 241.690 ảnh từ MS-COCO training data.
  - Tổng cộng 491.690 ảnh.
- Không đọc nhãn semantic khi tối ưu Forensic Twins.
- Không dùng ảnh giả để chọn checkpoint, tinh chỉnh encoder, fit GMM hoặc chọn hyperparameter của run chính.
- Lưu manifest bất biến gồm `sample_id`, nguồn dữ liệu, split gốc, đường dẫn, kích thước, checksum và seed chọn mẫu.
- Exact list chưa được paper công bố; khi tác giả phát hành split, thay manifest provisional và chạy lại toàn bộ.

### 3.2 Kiến trúc

| Thành phần | Cấu hình paper-faithful |
|---|---|
| Input ảnh | RGB, kích thước gốc `H×W×3` |
| FRE input | grayscale float trong `[0,1]`, `1×H×W` |
| FRE | pretrained frozen; constrained conv `1→8`, kernel `15×15`, padding 7 |
| FRE output | `8×H×W` |
| View 1 | `8×H1×W1` |
| View 2 | `8×H2×W2`, không có pixel chung với View 1 |
| Patch embed | linear patch stem, patch `8×8` |
| Encoder | ViT-Tiny, 6 Transformer blocks, 2D rotary positional embeddings |
| Summary token | `[CLS]` |
| Encoder output | 192-D |
| Projector | nonlinear projector, output 8.192-D |
| Train objective | Barlow Twins, redundancy weight `5e-5` |
| Detector | GMM, `K=1`, full covariance trên embedding 192-D |

Paper chỉ cho biết encoder khoảng 3,0M tham số. Số attention heads, MLP ratio, QKV bias, dropout, stochastic depth, LayerNorm epsilon, projector depth/width/activation và cách SyncBN không được nêu. Nếu cần chạy trước khi code chính thức phát hành, dùng config riêng `provisional_v1`, ghi rõ từng giả định; không đưa các giả định đó vào bảng “official”.

### 3.3 Kích thước input

- **Train:** paper không dùng một input size cố định. Hai crop được lấy sau FRE, có thể khác kích thước và được resample độc lập ở mỗi iteration. Trong một mini-batch, tất cả View 1 có cùng `H1×W1`, tất cả View 2 có cùng `H2×W2`; hai size không bắt buộc bằng nhau.
- `H1`, `W1`, `H2`, `W2` phải chia hết cho patch size 8 trong implementation.
- Paper v1 **không công bố khoảng, tập giá trị hoặc phân phối** của bốn kích thước train này. Không được tự ghi 224, 128 hay 256 là “size theo paper”.
- **Inference:** center crop đúng `256×256`; paper không resize ảnh trước detector. Ảnh có cạnh ngắn dưới 256 được bỏ qua/thay bằng ảnh kế tiếp trong benchmark.
- Với input inference 256×256: FRE trả `8×256×256`; patch 8 tạo grid `32×32 = 1.024` patch tokens; cộng `[CLS]` thành 1.025 tokens; output dùng cho GMM là vector 192-D.

## 4. Các giai đoạn triển khai

### Giai đoạn 0 — Khóa nguồn tham chiếu

- Lưu version paper `arXiv:2609.31514v1`, ngày tải và checksum PDF.
- Tải FRE từ repo chính thức của FSD; lưu URL commit, filename và SHA-256.
- Đối chiếu normalization của checkpoint bằng một regression test. Paper Forensic Twins yêu cầu `[0,1]`; docstring hiện tại của repo FSD ghi input `0–255`, nên run faithful ưu tiên mô tả của Forensic Twins nhưng phải ghi lại sai khác này.
- Theo dõi repository chính thức của Forensic Twins. Khi code xuất hiện, audit lại crop sampler, ViT và projector trước khi tuyên bố exact reproduction.

### Giai đoạn 1 — Xây manifest dữ liệu

- Xác thực hai dataset và loại file hỏng.
- Tạo manifest 250.000 ImageNet + 241.690 COCO bằng seed cố định.
- Tạo manifest held-out cho GMM diagnostics và benchmark; kiểm tra không trùng train theo hash, không chỉ theo filename.
- Xuất báo cáo số lượng theo nguồn, kích thước, codec và checksum manifest.

**Điều kiện qua:** đúng 491.690 ảnh real, không có fake, không overlap với real evaluation.

### Giai đoạn 2 — Tiền xử lý và view sampler

- Sample một phép biến đổi không phá hủy trên ảnh RGB: orthogonal rotation và horizontal/vertical flip.
- Áp dụng đúng một transform chung trước khi tạo hai view.
- Chuyển grayscale, chia 255, chạy FRE trong `eval()` và `requires_grad=False`.
- Sample hai hình chữ nhật trên residual map; assert intersection area bằng 0.
- Sample size ở cấp batch, tọa độ ở cấp ảnh.
- Ghi histogram size, aspect ratio và số lần sampler phải thử lại.

**Điều kiện qua:** không có crop overlap, tensor có 8 kênh, không gradient vào FRE, không có resize/blur/JPEG/color jitter trong pipeline chính.

### Giai đoạn 3 — Pretrain Forensic Twins

- Hai view dùng chung encoder `f_theta` và projector `g_phi`.
- Lấy `[CLS]` 192-D từ encoder; projector tạo 8.192-D.
- Batch-normalize embedding theo dimension, tạo cross-correlation 8.192×8.192 và tính Barlow Twins loss.
- Train 80 epochs bằng AdamW:
  - base LR `1.5e-3`;
  - weight decay `2.5e-3`;
  - linear warm-up 15 epochs;
  - cosine decay đến `1e-6`;
  - global batch size 320;
  - bfloat16 và FlashAttentionV2;
  - phần cứng paper: 4× RTX 4090.
- Checkpoint phải lưu encoder, projector, optimizer, scheduler, AMP scaler nếu có, epoch, global step, RNG states và resolved config.

**Điều kiện qua:** loss hữu hạn, diagonal correlation tiến về 1, off-diagonal giảm, embedding không collapse, resume cho kết quả nhất quán.

### Giai đoạn 4 — Fit one-class GMM

- Freeze encoder và bỏ projector.
- Trích `[CLS]` 192-D trên chính 491.690 ảnh real đã dùng pretrain, theo preprocessing inference thống nhất.
- Fit GMM `K=1`, full covariance.
- Lưu mean `192`, covariance `192×192`, precision Cholesky, log-determinant, implementation/version và manifest hash.
- Không PCA, không diagonal covariance và không fake calibration trong run faithful.

**Điều kiện qua:** covariance hữu hạn và positive definite; NLL reproducible; ảnh có score càng cao càng bất thường.

### Giai đoạn 5 — Đánh giá paper-faithful

- Dùng center crop 256×256, không resize hay re-encode trước detector.
- Với mỗi generator, lấy 1.000 ảnh đầu tiên hoặc toàn bộ nếu ít hơn; Firefly trong paper có 800.
- Mỗi generator được cân bằng riêng với cùng số real từ ImageNet-1k validation và COCO 2017 test.
- Báo cáo AUROC cho từng cặp generator/real-domain, trung bình GAN, Diffusion, Commercial và global mean.
- AUROC không cần chọn threshold. Nếu báo accuracy/F1, threshold phải được chọn chỉ từ validation real hợp lệ hoặc protocol phụ và phải tách khỏi kết quả paper.

**Mốc paper:** 97,99% global AUROC; 98,17 GAN; 98,56 Diffusion; 96,10 Commercial.

### Giai đoạn 6 — Reproduce ablation tối thiểu

Chạy theo thứ tự để phát hiện implementation lệch:

1. RGB base, patch 16, `lambda=0.005`: 60,95.
2. SRM input: 84,14.
3. FRE input: 93,77.
4. Destructive augmentation với FRE: 57,84.
5. Overlapped crops: 89,49.
6. Patch 8: 95,47.
7. Conv stem: 92,40.
8. `[CLS]+STD`, GAP, GAP+STD: 91,07 / 91,10 / 92,17.
9. Patch 8 và sweep `lambda`: `1e-3 → 96,69`, `2e-4 → 97,54`, `5e-5 → 97,99`, `1e-5 → 97,77`.

Không yêu cầu khớp tuyệt đối trước khi exact split/code được phát hành, nhưng xu hướng phải tương đồng. Nếu không, ưu tiên audit normalization, crop overlap, transform order và projector/BN.

### Giai đoạn 7 — Robustness và phân tích biểu diễn

- Compression diagnostic ở JPEG Q95/Q80/Q65/Q50/Q35; paper cho thấy AUC giảm mạnh từ 97,99 xuống 71,93/66,35/63,78/59,92/61,63.
- Real-domain ablation: ImageNet-only, COCO-only và combined.
- Semantic/source k-NN probe trên embedding 192-D để kiểm tra representation thiên forensic hơn semantic.
- k-Means source clustering là downstream task phụ, không tham gia train encoder/GMM.

## 5. Cấu trúc artifact dự kiến

```text
forensic_twins_original/
├── plan.md
├── workflow.md
├── configs/
│   ├── paper_faithful.yaml
│   └── provisional_v1.yaml
├── manifests/
├── src/
│   ├── fre.py
│   ├── views.py
│   ├── vit.py
│   ├── projector.py
│   ├── barlow.py
│   ├── train.py
│   ├── fit_gmm.py
│   └── evaluate.py
├── tests/
└── artifacts/
    ├── checkpoints/
    ├── embeddings/
    ├── gmm/
    ├── metrics/
    └── logs/
```

Hai file config phải khác vai trò: `paper_faithful.yaml` chỉ chứa giá trị paper công bố và từ chối chạy nếu tham số bắt buộc còn thiếu; `provisional_v1.yaml` chứa các giả định có chú thích để thử nghiệm sớm.

## 6. Checklist nghiệm thu

- [ ] Chỉ real images đi vào pretrain và GMM.
- [ ] Exact manifest/hash được lưu cùng run.
- [ ] Transform chung cho cả hai view; không có destructive augmentation.
- [ ] Grayscale và `/255` xảy ra trước FRE.
- [ ] FRE frozen, output 8 channels, không gradient.
- [ ] Hai residual crop có intersection area bằng 0.
- [ ] Patch 8, 6 blocks, 2D RoPE, `[CLS]` 192-D.
- [ ] Projector output 8.192-D; `lambda=5e-5`.
- [ ] Global batch 320, 80 epochs, AdamW/LR/schedule đúng paper.
- [ ] Projector bị bỏ khi fit GMM và inference.
- [ ] GMM `K=1`, full covariance, fit bằng real embeddings.
- [ ] Inference center crop 256×256, không resize.
- [ ] AUROC báo riêng theo generator và real domain.
- [ ] Mọi giả định chưa được paper công bố được gắn nhãn provisional.

## 7. Nguồn chính

- [Forensic Twins — arXiv v1](https://arxiv.org/html/2609.31514v1)
- [Forensic Twins — PDF](https://arxiv.org/pdf/2609.31514)
- [FSD/FRE official repository](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25)
- [FRE implementation](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/blob/main/fsd/fre.py)
