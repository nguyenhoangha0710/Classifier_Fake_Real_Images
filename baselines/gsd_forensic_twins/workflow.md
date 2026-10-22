# Workflow — Twins + GSD last4, native RGB

Đặc tả chuẩn: [plan.md](plan.md). GSD theo [paper v2 IV-B/C, V-A](https://arxiv.org/html/2603.09242v2); Barlow và Gaussian là phần ghép Twins. Các sơ đồ cũ một-layer/K8/alpha0.25 chưa đồng bộ.

## 1. Data preflight và views

Reuse parent train200K; derive eligible SSL manifest theo min-side224/max-side448. Fit/calibration reuse saved split160K/40K, lọc center224 riêng, giữ membership, log actual counts/hashes. Calibration đã được thấy trong SSL parent. B-Free reuse clean1.557 IDs/10.899 ảnh, không rebuild cohort. [Dataset inputs](../forensic_twins_codec/kaggle_inputs.md).

```text
Real native RGB → shared rotation/flips
  → rectangles A/B 224×224, intersection=0
  → crop + CLIP normalization
  → x1,x2: [B_local,3,224,224]
```

Không random JPEG/resize/blur ở run đầu. Native không đồng nghĩa camera RAW; không resize để cứu ảnh nhỏ. Return sample IDs/rectangles.

## 2. Shared encoder — forward riêng cho mỗi view

Es/Ef cùng CLIP ViT-L/14 initialization. Es frozen/eval/no_grad; Ef fine-tune toàn visual backbone. Input224/patch14 → [B_local,257,1024].

```text
                         Semantic Es                 Forensic Ef
same crop →              embed + blocks1–20          embed + blocks1–20
                                 ↓                          ↓
                         block21 → patch tokens ──→ block21 → GSD21
                                 ↓                          ↓
                         block22 → patch tokens ──→ block22 → GSD22
                                 ↓                          ↓
                         block23 → patch tokens ──→ block23 → GSD23
                                 ↓                          ↓
                         block24 → patch tokens ──→ block24 → GSD24
                                                            ↓
                                             final LN → mean patch tokens
                                                            ↓
                                                     z [B_local,1024]
```

Mũi tên Es→GSD cung cấp basis cùng layer. F_hat được truyền tiếp vào block Ef sau; Es không suppression. Không fusion bốn CLS. Readout final-LN/mean patch tokens là quy ước bản ghép Twins.

### Tensor contract tại layer l=21,22,23,24

1. P=Es_output[:,1:,:]=[B_local,256,1024], raw block output, không thêm intermediate final LN.
2. Gather frozen P across ranks theo **cùng view**; M=[B_global*256,1024]. Rank0 SVD FP32 không centering; V=Vh[:64,:].T=[1024,64], detach/broadcast.
3. F=Ef block output trước GSD=[B_local,257,1024]. FP32: F_parallel=(F@V)@V.T.
4. r=F_parallel.square().sum((1,2))/F.square().sum((1,2)).clamp_min(1e-12).
5. lambda=sigmoid(a_l*r+b_l), lambda_max1; init a_l2,b_l1. Broadcast [B_local,1,1].
6. F_hat=F-lambda*F_parallel. Cast về backbone activation dtype, dùng cho block tiếp theo hoặc readout sau24.

Hai views share Es/Ef/projector và bốn cặp modulation parameters, có basis riêng theo view/layer; không concat views để SVD. Gradient qua F/r/lambda/a_l/b_l; Es/basis stop-gradient, không detach occupancy.

## 3. SSL training

```text
z1,z2 [B,1024] → shared projector 1024→2048→8192
  → differentiable global-batch normalization/correlation
  → Barlow loss (lambda_BT=5e-5)
  → update toàn Ef + projector + 8 modulation scalars
```

Không classifier CE/fake labels. Full-gradient smoke rồi reset main initialization. Optimizer/schedule theo plan. Chốt actual batch/hardware bằng VRAM/time preflight; full ViT-L/14 last4 chưa được xác nhận fit2×T4. Accumulation không tăng statistics batch của Barlow/Batch-SVD.

## 4. Density fit, calibration và test

```text
Native center224 → Es/Ef/GSD21–24 → LN/mean patches1024
  real-fit     → fit PCA128/no whiten → fit Gaussian K1/full
  calibration  → saved PCA/Gaussian → NLL → threshold p95
  test         → saved PCA/Gaussian → NLL → fake iff score>threshold
```

Freeze encoder/modulation, bỏ projector. **Primary evaluation per-sample**: từng ảnh/layer có SVD[256,1024], V[1024,64]. Dùng cùng mode cho fit/calibration/test; có thể batching SVD độc lập. Paper hỗ trợ per-sample, nhưng paper mặc định benchmark Batch-Inference.

Batch-mode chỉ là đối chứng phụ với batch size/order/grouping khóa và artifacts extraction/PCA/Gaussian/calibration riêng. Không trộn features giữa protocol, không update artifacts trên test.

## 5. Config và kiểm chứng

Khóa checkpoint/hash, ViT-L/14/d1024, crop224/patch14, layers21–24, K64, lambda_max1, a_init2/b_init1, full Ef train, raw block-token stage, final-LN/mean patch readout, train basis global-view batch, primary eval per-sample, hardware/true batch, optimizer/schedule và manifest identities.

Kiểm tra propagation qua4 GSD, token shapes, orthogonal basis, lambda0/1 test modes, adaptive gradients, frozen Es, global-view DDP basis và differentiable Barlow. Per-sample inference phải invariant batch order/composition trong tolerance; Batch-Inference có phụ thuộc batch theo thiết kế. PCA fit-only-real, threshold no-test-labels. Log per-layer r/lambda/a/b/singular gaps, feature rank, gradients/VRAM/time. B0 bypass và C0 adaptive native dùng paired protocol; không reuse checkpoint/GMM từ thiết kế cũ.
