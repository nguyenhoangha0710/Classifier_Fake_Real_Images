# Twins–GSD–Gaussian: native RGB + GSD bốn block cuối

## 1. Phạm vi và nguồn

Train native RGB trước, chưa random processing. Encoder theo cơ chế [GSD paper v2, 03/06/2026, IV-B/C và V-A](https://arxiv.org/html/2603.09242v2). Hai disjoint crops, Barlow Twins real-only và Gaussian là phần kết hợp Twins của dự án; không gọi đây là tái lập supervised detector của paper.

Tài liệu này thay cấu hình cũ ViT-B/16, GSD một layer/K8/alpha0.25 và last2 trainable. Các sơ đồ `option_c_*`, `gsd_twins_native_architecture.drawio/.png` và script `draw_native_architecture.py` hiện thuộc thiết kế cũ, chưa đồng bộ; không dùng làm đặc tả hiện hành. Workflow chuẩn: [workflow.md](workflow.md).

## 2. Kiến trúc đã cập nhật

Real RGB → shared rotation/flip → hai native crops224 không giao nhau → shared GSD encoder → shared projector → Barlow Twins.

| Thành phần | Cấu hình |
|---|---|
| Es / Ef | Cùng OpenAI CLIP ViT-L/14 checkpoint/hash; 24 blocks; hidden1024 |
| Es | Toàn bộ frozen, eval, no_grad |
| Ef | Fine-tune toàn visual backbone, kể cả blocks1–20 |
| GSD | Sau output mỗi block21,22,23,24; suppressed sequence truyền sang block kế tiếp |
| Input/tokens | RGB224, patch14: 256 patch tokens + CLS; [B,257,1024] |
| Train basis | Global-view Batch-SVD, K64; basis riêng theo layer/view |
| Suppression | Toàn token sequence Ef, gồm CLS; adaptive theo sample/layer |
| Modulation | lambda_max=1; bốn cặp learnable a_l,b_l; init a=2,b=1 |
| Readout | Sau GSD24: final LN rồi mean non-CLS patch tokens →1024-D |
| Projector | Shared 1024→2048→8192, pattern activation/normalization của Twins native, khởi tạo mới |
| SSL loss | Barlow Twins, lambda_BT=5e-5, eps=1e-4 |
| Density | PCA128/no whitening → Gaussian K1/full covariance, reg_covar=1e-6 |

Readout final-LN/mean patch tokens là quy ước của bản ghép Twins, không claim paper quy định chính xác readout này. Không fusion bốn CLS, không chỉ suppress CLS cuối, không dùng CLIP visual projection. Không thêm final LN vào mỗi intermediate layer trước SVD: dùng raw block outputs cùng stage Es/Ef.

## 3. GSD và gradient contract

Với view v, layer l thuộc {21,22,23,24}:

1. Es patch tokens P_i,l=[256,1024], bỏ CLS. Ghép toàn global batch **trong cùng view** thành M_v,l=[B_global*256,1024]. SVD FP32 không centering: M=U Sigma Vh; V_v,l=Vh[:64,:].T=[1024,64], detach/no_grad.
2. Ef output trước GSD: F=[B_local,257,1024]. F_parallel=(F@V)@V.T, không dựng projector1024×1024.
3. r=||F_parallel||_F²/max(||F||_F²,1e-12), reduce token và hidden axes, một scalar/sample.
4. lambda_i,l=lambda_max*sigmoid(a_l*r_i,l+b_l), broadcast [B_local,1,1].
5. F_hat=F-lambda*F_parallel. Dùng F_hat làm input block Ef kế tiếp; Es truyền output nguyên bản.

Bốn cặp a_l,b_l shared giữa hai view; basis tính lại từng layer/view. Không ghép hai view để tính SVD. Global-view batch scope là quy ước kết hợp Twins, paper không có Barlow hai-view pipeline.

Gradient qua F, projection, occupancy, lambda và a_l,b_l; chỉ Es/basis stop-gradient, không detach r âm thầm. Projection/occupancy/sigmoid FP32; cast F_hat về backbone activation dtype trước block tiếp theo.

DDP: gather frozen patch tokens không gradient across ranks theo view; rank0 SVD rồi broadcast V. Barlow global normalization/correlation phải differentiable. Gradient accumulation không tăng statistics batch của Barlow hoặc basis batch của GSD.

## 4. Data/preprocessing

- SSL parent `forensic_twins_v2_200k/manifests/train_real_200k.csv` từ [twins-gmm-200k-checkpoint](https://www.kaggle.com/datasets/hoangha071005/twins-gmm-200k-checkpoint), 100K COCO+100K ImageNet. Không đổi cohort hoặc chỉ dùng fit160K cho SSL.
- Native nghĩa là decoded source RGB, không phải camera RAW. Shared rotation {0,90,180,270}/flips; hai crop224 disjoint; không thêm JPEG/resize/blur/noise trong run đầu. File nguồn bất biến.
- Eligible SSL: min(W,H)>=224 và max(W,H)>=448. Lưu derived `ssl_train_eligible.csv`, count thực tế và parent hash; không resize ảnh nhỏ để ép đủ200K.
- Prepare mặc định lọc width/height từ saved manifest, chỉ preflight một số pixel paths; không mở/verify toàn200K ảnh. Missing dimensions fallback đọc header. Full image audit là chế độ riêng `--audit full`. Log tiến độ/ETA, heartbeat và reuse kết quả đã kiểm tra hash; không gọi metadata filtering là full decode audit.
- Fit/calibration reuse saved split160K/40K, lọc center224 riêng và giữ membership; log counts sau filter. Calibration disjoint density-fit nhưng đã nằm trong SSL parent, không gọi encoder-held-out.
- CLIP RGB mean/std đúng checkpoint, không resize thêm sau crop. Inference center224; ảnh thiếu kích thước skip/log.
- B-Free reuse `forensic_twins_augmented_gmm_full/manifests/bfree_valid_clean_eval.csv` từ [evaluation-twins-bfree V2](https://www.kaggle.com/datasets/hoangha071005/evaluation-twins-bfree), mount [B-Free pixels](https://www.kaggle.com/datasets/baopv051/b-free-training-data). Giữ 1.557 source IDs/10.899 ảnh; nếu invalid báo missing/common intersection, không thêm IDs.
- Fake không vào SSL/PCA/Gaussian/calibration; test labels không chọn checkpoint/threshold/protocol. CLIP pretraining overlap không thể bị loại trừ hoàn toàn.
- [Kaggle inputs/provenance](../forensic_twins_codec/kaggle_inputs.md) cung cấp manifest identity; kiến trúc trong plan này ưu tiên khi mô tả cũ khác nhau.

## 5. Training và compute preflight

- Smoke 200–500 steps subset cố định với full gradient Ef/projector/modulation; kiểm tra complete step đầu tiên. Reset initialization/seed trước run chính. Projector-only smoke nếu có chỉ là check phụ.
- AdamW: Ef LR1e-6, projector LR1e-4, modulation LR1e-6; WD0.01 cho matrix weights, modulation không WD; betas=(0.9,0.999), eps1e-8. Modulation LR/WD là lựa chọn triển khai bản ghép.
- 20 epochs, warm-up5% optimizer steps, cosine về1% LR ban đầu, grad clip1. Loss duy nhất Barlow, không supervised classifier/CE.
- Paper train batch128/A100 chỉ là mốc tham khảo. Full ViT-L/14, hai views và bốn SVD cần đo VRAM/step time trước khi chốt hardware/global batch; không mặc định 2×T4 batch32 khả thi.
- Precision auto chọn BF16 nếu hỗ trợ, FP16 trên T4; GSD/projector/Barlow FP32. FP16 GradScaler init128; overflow skip optimizer/scheduler update, giảm scale và log affected parameters/sample IDs; abort sau8 lần liên tiếp. Không bỏ qua nonfinite khi scaler disabled. Cho phép activation checkpointing Ef. Không tự đổi backbone/K/trainable depth khi OOM.
- DataLoader spawn sau CUDA init, worker timeout120s; NCCL timeout600s. Log stage/load/batch/SVD/backward/checkpoint và heartbeat30s; Python stack dump khi stage không đổi300s chỉ phục vụ chẩn đoán, không tự kết luận deadlock.
- Nếu giảm batch/projector dim để fit, lưu config riêng và chạy paired control tương ứng; accumulation chỉ tăng update batch.
- Es eval/no_grad; Ef train. Optimizer chứa toàn visual Ef, projector và đúng8 modulation scalars. Save mỗi epoch; checkpoint chính epoch20 khóa trước test, không chọn chỉ theo loss thấp nhất.
- Log per-layer/view r/lambda/a/b, sigma64/sigma65 gap, basis orthogonality, norms/variance/effective rank, diagonal/offdiagonal loss, gradients, NaN/Inf, actual counts, VRAM/time.
- Occupancy giảm chưa chứng minh semantic debias: cần semantic probe audit split riêng. SVD nonfinite/fail log IDs và fail run nếu lặp lại, không fallback bỏ suppression.

## 6. PCA/Gaussian và inference

Freeze Es/Ef/modulation, bỏ projector. Center224 → encoder/readout1024 → PCA128 → Gaussian → score=-log p(z|real). PCA chỉ fit real-fit; threshold percentile95 từ calibration-real; cùng preprocessing, không thêm L2/TTA.

**Primary: per-sample inference** tại cả bốn layers. SVD riêng từng ảnh P_i,l=[256,1024] → V_i,l=[1024,64]. Dùng thống nhất khi extract real-fit, calibration và test; có thể batch các SVD độc lập. Chọn protocol này để Gaussian representation không phụ thuộc evaluation batch. Paper hỗ trợ protocol này nhưng mặc định benchmark của paper là Batch-Inference.

Batch-Inference là đối chứng phụ: khóa batch size/order/grouping và extract/fit/calibrate lại artifacts riêng. Không đưa features batch-mode vào detector per-sample. Reference inference là extension, chỉ dùng real-fit reference và artifacts riêng.

Lưu encoder/modulation + readout/token-stage/inference mode metadata, PCA/Gaussian/threshold, hashes manifests/checkpoint/preprocessing, RNG/optimizer/scheduler/scaler/sampler và environment. Không reuse embeddings192/768 hoặc density artifacts từ thiết kế cũ.

## 7. Experiments và acceptance

| Run | GSD | Train input | Ưu tiên |
|---|---|---|---|
| B0 | Bypass cả4 modules | Native RGB | Paired control |
| C0 | Adaptive last4 Batch-SVD K64 | Native RGB | Main run đầu tiên |
| B1/C1 | Tương ứng B0/C0 | Codec augmentation | Extension sau, chưa chạy |

B0/C0 cùng CLIP init, full trainable backbone, readout, projector, seeds/crops/manifests, batch/steps và density recipe; detector fit riêng. Run ID mới `gsd_last4_adaptive_k64_native_v2`, không nhầm C0 alpha0.25 cũ. Định trước3 paired seeds cho main report khi compute cho phép. Bổ sung frozen-CLIP Gaussian và random-rank64 control để phân tích hiệu quả semantic basis.

Report AUROC/per-generator, calibrated real FPR/fake TPR, real FPR theo source, robustness JPEG95/80/65/50/resize, latency/memory. B-Free là benchmark hồi quy đã xem kết quả, Tiny/ComFor bổ sung. Tách native-confounded/codec-matched/common-reencode; native training chưa chứng minh codec robustness.

Acceptance trước full train:

- Đúng token shapes/insertion21–24; suppression tác động toàn sequence và truyền sang block kế tiếp.
- K64 orthogonal; bypass/lambda0 identity và fixed-lambda1 residual trực giao là unit-test modes.
- Adaptive gradients finite tới toàn Ef và a_l,b_l; Es/basis không gradient.
- Global-view DDP basis agreement, không trộn views; Barlow gradients đối chiếu single-process reference.
- Per-sample inference invariant với batch composition/order trong tolerance; không áp invariance này cho batch-mode.
- Fit-only PCA/no-test calibration, artifact/resume identity và full-gradient GPU smoke/VRAM/time.

Code deliverables đã triển khai: [model.py](model.py), [data.py](data.py), [prepare.py](prepare.py), [train.py](train.py), [density.py](density.py), [config.json](config.json), notebook Kaggle có source nhúng. Hướng dẫn: [README.md](README.md). Tám CPU tests đã pass, gồm hai-rank Gloo basis/Barlow gradients; full pretrained CUDA/NCCL smoke, VRAM/time và train20 epochs chưa chạy. Robustness sweeps/grouped benchmark reports chưa triển khai trong evaluator đầu tiên.
