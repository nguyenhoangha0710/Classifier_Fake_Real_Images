# Thiết kế encoder forensic real-only với GSD và Gaussian/GMM

Ngày nghiên cứu: 21/10/2026. Đây là đề xuất nghiên cứu, chưa phải cấu hình được thực nghiệm xác nhận. Không thay đổi code hoặc cấu hình training hiện có.

## 1. Kết luận thiết kế

Nên kiểm chứng pipeline hai giai đoạn:

1. Học encoder trên ảnh thật bằng hai crop native không giao nhau, shared GSD encoder và Barlow Twins; VICReg là đối chứng thay loss.
2. Freeze encoder, bỏ projector, fit PCA và Gaussian/GMM trên embedding ảnh thật; calibrate bằng một tập real riêng.

GSD là module can thiệp representation, không phải một encoder hoặc objective hoàn chỉnh. Loại bỏ hướng semantic không tự tạo ra forensic information; objective và cách tạo positive pairs quyết định phần thông tin còn lại mà model học.

Việc giảm phụ thuộc fake đã thấy là động cơ hợp lý. Tuy nhiên, one-class chỉ phát hiện khác biệt với real reference; ảnh thật từ camera/codec mới cũng có thể bị xem là bất thường. Fake có phân phối embedding trùng với real không thể được phân biệt bằng density score đó.

## 2. Nguồn nghiên cứu và điều rút ra

| Nguồn gốc | Kết quả/phương pháp liên quan | Ứng dụng trong đề xuất |
|---|---|---|
| [GSD v2, 03/06/2026](https://arxiv.org/html/2603.09242v2) | Frozen CLIP anchor, SVD semantic subspace, adaptive suppression; bản gốc dùng supervised CE | Giữ module GSD, thay objective bằng real-only SSL; đây là extension |
| [Forensic Twins v1, 25/09/2026](https://arxiv.org/html/2609.31514v1) | Frozen residual extractor, disjoint crops, Barlow Twins; density K=1 full covariance | Pretext task gần mục tiêu nhất; phụ lục JPEG cho thấy real-reference coverage rất quan trọng |
| [Forensic Self-Descriptions, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_Forensic_Self-Descriptions_Are_All_You_Need_for_Zero-Shot_Detection_Open-Set_CVPR_2025_paper.pdf) | Học predictive residual filters từ real và mô hình hóa microstructure nhiều scale | Một nguồn forensic teacher/residual branch cho thí nghiệm mở rộng |
| [Noiseprint](https://arxiv.org/abs/1808.08396) | Học fingerprint camera model và giảm scene content | Gợi ý xây positive pairs theo acquisition pipeline nếu có metadata; camera OOD phải được đánh giá |
| [UniversalFakeDetect, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html) | Binary training có thể làm real thành sink class; frozen pretrained features hữu ích | Cần baseline frozen CLIP + density để đo giá trị thật của fine-tuning |
| [Barlow Twins, ICML 2021](https://arxiv.org/abs/2103.03230) | Cross-view correlation gần identity, không dùng negatives | Baseline loss phù hợp code sẵn có |
| [VICReg, ICLR 2022](https://arxiv.org/abs/2105.04906) | Invariance + variance floor + covariance regularization | Ablation chống collapse tường minh; không đảm bảo hơn Barlow trong forensics |
| [Deep one-class representations, ICLR 2021](https://research.google/pubs/learning-and-evaluating-representations-for-deep-one-class-classification/) | Tách SSL representation learning khỏi one-class classifier | Cơ sở cho quy trình train encoder rồi fit density |
| [ConV, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/2cfa9b0d9be8a5c01cf3eb7f21b4f2b8-Paper-Conference.pdf) | Kiểm tra consistency liên quan natural-image manifold | Hướng score bổ sung về sau; không phải bằng chứng trực tiếp cho GSD+GMM |
| [Nalisnick et al., ICLR 2019](https://arxiv.org/abs/1810.09136) | Một số deep density models cho OOD likelihood cao | Không đồng nhất density với xác suất ảnh thật; không suy rộng rằng mọi GMM đều thất bại |

Các kết quả benchmark thuộc protocol riêng của từng bài, không dự báo trực tiếp kết quả trên Tiny-GenImage/B-Free. Hai bài GSD/Forensic Twins được đọc theo đúng phiên bản ghi trên, không giả định đây là phiên bản mới nhất hoặc code tác giả đã phát hành.

## 3. Liên hệ code và kết quả trong repo

- `reports/gsd_sd15_checkpoint_analysis/analysis_vi.md`: native SD15 BA 99,10%; macro unseen BA 68,52%, macro unseen AUROC 82,47%. Chưa có paired CLIP-FT control trong run này nên chưa tách được đóng góp của GSD. Những con số này được đọc từ báo cáo đã lưu, không chạy lại checkpoint.
- `baselines/gsd_forensic_twins/model.py`: đã có hai CLIP branches, GSD blocks21–24, patch-mean1024 và Barlow Twins. Đây là nền tảng thí nghiệm phù hợp, không cần xây một encoder mới hoàn toàn.
- `baselines/gsd_forensic_twins/density.py`: hiện là PCA + một Gaussian full covariance, không có bước EM học nhiều components. Argument `components=128` trong `fit_gaussian` là chiều PCA, không phải số mixture components.
- `baselines/gsd_forensic_twins/plan.md`: quy định batch basis khi train, sample basis khi extraction. Chuyển protocol này cần audit; kết quả classification của paper không đảm bảo density score giữ nguyên.
- `baselines/forensic_twins_codec/TRAINING_AUDIT_20261020.md`: loss giảm chưa chứng minh representation forensic hoặc chống codec tốt hơn. Cần eigen-spectrum, semantic probe và detection evaluation.

## 4. Kiến trúc baseline nên chạy trước

```text
Real image x
  -> shared flip/rotation 90 degrees
  -> two disjoint native RGB crops x1,x2, each 224x224
  -> shared encoder E_theta
       trainable CLIP forensic stream
       frozen CLIP semantic stream
       GSD after blocks21,22,23,24
  -> patch-mean embedding h1,h2 in R^1024
  -> shared projector q_phi
  -> Barlow Twins loss

After training:
freeze E_theta -> discard projector -> h -> PCA -> Gaussian/GMM -> anomaly score
```

Frozen anchor và forensic branch nên khởi tạo từ cùng pretrained CLIP. Không chỉ huấn luyện classifier rồi lấy embedding. Một nhánh khởi tạo forensic stream từ checkpoint SD15 là ablation riêng: checkpoint đó có thể mang fake-specific bias.

Giữ backbone/readout/projector của code hiện có cho paired baseline. Chỉ giảm projector 8192 xuống1024/2048 trong một ablation có control tương ứng. Projector rộng không tự sai, nhưng correlation matrix centered trên batch B có rank tối đa B-1; không diễn giải loss không về0 là bằng chứng training thất bại.

GSD tại layer l, với frozen patch matrix P_l=U Sigma V^T và V_k gồm k right singular vectors:

```text
F_parallel = (F V_k) V_k^T
r = ||F_parallel||_F^2 / (||F||_F^2 + eps)
lambda = lambda_max sigmoid(a_l r + b_l)
F_hat = F - lambda F_parallel
```

Đây là cơ chế gốc; bắt đầu k64, adaptive suppression hiện có. Mọi thay đổi k hoặc suppression là ablation, không coi hướng có variance lớn nhất của CLIP là semantic thuần. Freeze anchor/basis; gradient vẫn qua forensic tokens và modulation.

Không chiếu token FRE/ViT192 bằng basis CLIP1024. Kể cả thêm linear adapter để bằng chiều cũng chưa đảm bảo cùng hệ tọa độ. Muốn ghép FRE nên dùng late fusion hoặc teacher supervision ở embedding level.

## 5. Positive pairs là phần quan trọng nhất

Hai crop không giao nhau từ cùng ảnh nhằm giữ thống kê acquisition/processing chung, đồng thời giảm khả năng matching bằng object hoặc pixel giống nhau. Đây là giả thuyết cần đo: RGB crops vẫn có thể chia sẻ scene semantics, ánh sáng, color palette, JPEG grid và dataset source.

Recipe đầu tiên:

- Crop native224, không resize toàn ảnh trước crop.
- Dùng shared orthogonal rotation/flip để hai view không khác nhau bởi phép nội suy.
- Không dùng blur/noise/color jitter mạnh trong run đầu.
- Với hai crop224 không giao nhau, cần min(W,H)>=224 và max(W,H)>=448; phải báo số ảnh bị lọc và kiểm tra selection bias.
- Không coi crop size64–128 của encoder patch8 là cấu hình cắm trực tiếp vào CLIP patch14 pretrained224.

Về codec, tách hai câu hỏi:

1. Density coverage: freeze encoder, thêm các phiên bản JPEG của real-fit và real-calibration theo chính sách khóa trước, rồi fit/calibrate lại.
2. Encoder invariance: một run mới dùng native/JPEG pairs và loss tương ứng. Điều này có thể xóa forensic cues nên phải đo cả detection.

Re-encode cả hai lớp không xóa hoàn toàn lịch sử nén cũ. Dataset source, lịch sử resize và JPEG vẫn cần control.

## 6. Loss đề nghị

### 6.1 Baseline: Barlow Twins

Với h_v=E_theta(x_v), z_v=q_phi(h_v); chuẩn hóa mỗi chiều z theo batch, tính C=(z1^T z2)/B:

```text
L_BT = sum_j (1-C_jj)^2 + beta sum_{j!=k} C_jk^2
```

Hai tác dụng: giữ thông tin chung giữa views và giảm redundancy. Code hiện dùng beta=5e-5; giữ nguyên cho phép so sánh đầu tiên. Không lấy hệ số từ loss khác dùng cho tổng có số hạng khác.

Không thêm CE real/fake nếu mục tiêu là real-only. CE với tất cả label real có nghiệm tầm thường và không xác định forensic representation.

### 6.2 Ablation: VICReg

Cho z1,z2 trong R^(B x d), đặt:

```text
L_inv = ||z1-z2||_F^2 / (B*d)
v(Z) = mean_j relu(gamma - sqrt(Var_batch(Z_j)+eps))
L_var = (v(z1)+v(z2))/2
c(Z) = sum_{j!=k} Cov_batch(Z)_jk^2 / d
L_cov = c(z1)+c(z2)
L = 25 L_inv + 25 L_var + L_cov
```

Gamma1 và eps1e-4 là điểm bắt đầu; hệ số25/25/1 theo [official implementation](https://raw.githubusercontent.com/facebookresearch/vicreg/main/main_vicreg.py). Không L2-normalize z trước variance-floor này. Đây là lựa chọn tối ưu hóa để thử, không phải cấu hình đã chứng minh tối ưu cho forensic.

Cần đo collapse trên h trước projector, không chỉ z: feature std, covariance eigen-spectrum/effective rank và nearest-neighbor diversity. Projector có thể che việc h còn semantic hoặc ít chiều hữu dụng. Gradient accumulation không làm tăng batch dùng để ước lượng covariance/correlation nếu không thực sự gom activations.

### 6.3 Optional semantic regularizer

Nếu semantic probe cho thấy h còn chứa content mạnh, có thể thử:

```text
s = stop_gradient(frozen semantic embedding)
L_sem = ||Corr_cross(h,s)||_F^2 / (d_h*d_s)
L_total = L_SSL + eta L_sem
```

Tính correlation sau standardize theo batch; bắt đầu eta0 và sweep nhỏ theo gradient scale. Đây là đề xuất bổ sung, không thuộc loss của GSD paper. Nó chỉ giảm phụ thuộc tuyến tính được đo, không chứng minh độc lập semantic. Khi content và camera/source tương quan, ép decorrelation có thể xóa forensic cues.

GSD cũng không ngăn tuyệt đối việc forensic stream xoay semantic information sang phần bù của basis; vì vậy occupancy thấp không đủ làm acceptance metric.

### 6.4 Không joint-train GMM NLL từ đầu

Với covariance cố định, encoder có thể dồn mọi real embedding về một điểm để giảm NLL. Nếu covariance cũng học và không có ràng buộc, co covariance còn tạo singularity. Real-only BCE hoặc chỉ MSE matching hai views cũng có nghiệm trivial.

Vì vậy train representation trước rồi freeze để fit density. Joint density objective chỉ nên là nghiên cứu sau với anti-collapse, covariance floor và control rõ ràng.

## 7. Gaussian/GMM và inference

Đặt u=PCA(h), fit trên real-fit:

```text
p(u|real) = sum_{k=1}^K pi_k N(u;mu_k,Sigma_k)
score(x) = -log p(PCA(E_theta(x))|real)
```

Khởi đầu PCA128/no whitening và K1/full covariance theo baseline repo. K1 chính là Gaussian đơn, là trường hợp đặc biệt của GMM. Sweep d64/128, K1/2/4/8 sau khi encoder cố định; tăng components không tự giải quyết feature bias.

Chọn K/regularization bằng real validation likelihood hoặc BIC và kiểm tra numerical stability; điều này không tối ưu trực tiếp unseen-fake AUROC. Nếu dùng fake development để lựa chọn thì báo rõ detector được chọn với fake labels, dù fitting vẫn real-only. Thêm covariance regularization có scale phù hợp; fit nhiều seed/init khi K>1.

PCA chỉ giữ hướng variance lớn nên có thể loại mất low-variance forensic cues. Cần đối chứng no-PCA+shrinkage khi đủ mẫu; không mặc định PCA càng nhỏ càng tốt. Không reuse PCA/GMM khi đổi encoder, pooling, basis protocol hoặc preprocessing.

Với GSD có ba protocol:

- Per-sample: feature mỗi ảnh không phụ thuộc cohort, nhưng SVD đắt.
- Fixed reference: basis từ real reference, tái sử dụng; rẻ hơn nhưng có thể bỏ sót semantic directions mới.
- Batch: phải audit batch-size/composition/order dependence và fit/calibrate theo đúng mode đó.

Primary density experiment nên dùng cùng per-sample mode cho fit/calibration/test như plan hiện tại. Batch training -> sample extraction vẫn có distribution shift; so reference training/extraction nhất quán hoặc sample-mode training trên pilot để đo, không tự giả định tương đương. Fixed-reference basis có thể tính offline nhờ anchor frozen, nhưng phải khóa reference membership/hash.

Threshold tau=quantile_(1-alpha)(score(real_calibration)), ví dụ alpha0.05. Đây là ngưỡng nhắm empirical FPR5% trên calibration, không bảo đảm FPR5% trên camera/source/codec mới và không phải P(fake|x).

## 8. Thí nghiệm có sức phân biệt nguyên nhân

| Run | Encoder | Objective | Density | Câu hỏi |
|---|---|---|---|---|
| A0 | Frozen pretrained CLIP | Không fine-tune | Gaussian | Pretraining đã đủ tốt chưa? |
| A1 | CLIP không GSD | Barlow, disjoint RGB | Gaussian | SSL real-only giúp gì? |
| A2 | CLIP+GSD | Barlow, cùng pairs | Gaussian | GSD thêm giá trị gì? |
| A3 | CLIP+GSD | VICReg | Gaussian | Thay loss có giúp không? |
| A4 | Encoder A2 frozen | Không train lại | Native+JPEG real-fit | Density coverage có phải bottleneck? |
| A5 | CLIP+GSD | Codec consistency, run riêng | Refit density | Học invariance có hơn mở rộng real-fit? |
| A6 | Encoder A2 frozen | Không train lại | K>1 | Cần mixture hay K1 đủ? |

Khóa data membership, readout, processing, compute budget và seed cho paired controls. Chạy ít nhất3 seeds cho kết luận chính nếu tài nguyên cho phép. A3 là thử nghiệm mới, không sửa âm thầm run A2 hiện tại. Sau khi A2 thắng có thể thử semantic regularizer hoặc forensic residual teacher, mỗi lần một yếu tố.

Metrics cần cùng xuất hiện:

- Macro AUROC per-generator; TPR tại ngưỡng real-calibration và actual FPR trên real test; worst-generator performance.
- Real camera/source/content/codec OOD: một detector universal không được coi mọi ảnh thật lạ là fake.
- Native và JPEG/resize stress tests với recipe khóa trước; bản gốc và mọi phiên bản của một ảnh giữ trong cùng split.
- Linear semantic probe trên real audit split, có probe-train/probe-test riêng; giảm semantic accuracy phải đi kèm detection tốt và không collapse.
- Effective rank/eigen-spectrum của h; occupancy/lambda từng layer; embedding distance cùng ảnh qua codec.
- Score stability khi đổi batch size/order/composition; VRAM và inference latency.

Tách encoder-train, density-fit, real-calibration và final-test rõ ràng. Calibration chưa được dùng để fit Gaussian nhưng đã xuất hiện trong SSL parent thì phải ghi là density-held-out, không phải encoder-held-out. Khi thiết kế nghiên cứu mới, nên có thêm calibration chưa thấy bởi encoder. Không lựa checkpoint, threshold hoặc loss weights từ final unseen test.

## 9. Hướng mở rộng có forensic supervision rõ hơn

Nếu A2 chưa tốt, ưu tiên hai hướng có thể kiểm chứng:

1. Frozen forensic teacher: dùng FSD/FRE features từ đúng crop để supervise thêm một auxiliary head ở forensic embedding. Stop-gradient teacher; bỏ auxiliary head khi inference. Teacher có thể mang codec/source bias nên vẫn cần camera/codec OOD controls.
2. Có camera metadata: positives từ hai ảnh khác nội dung nhưng cùng camera/pipeline; điều này tách content tốt hơn same-image pairs. Phải có camera-disjoint evaluation; source labels dạng COCO/ImageNet không tương đương camera ground truth.

Không ghép FRE branch, GSD, semantic loss, codec invariance và joint GMM vào cùng run đầu tiên: nếu kết quả thay đổi sẽ khó xác định nguyên nhân. Ưu tiên A0/A1/A2 trước, rồi quyết định bước sau bằng cả representation audit và detection metrics.
