# GSD-CLIP (arXiv:2603.09242v2) - Architecture va cach train paper-faithful

Tai lieu nay tom tat ban **v2 ngay 03/06/2026** cua bai bao *When Detectors Forget Forensics: Blocking Semantic Shortcuts for Generalizable AI-Generated Image Detection*.

- Paper: <https://arxiv.org/abs/2603.09242>
- HTML v2: <https://arxiv.org/html/2603.09242v2>
- PDF v2: <https://arxiv.org/pdf/2603.09242>

> Phan biet quan trong: `baselines/gsd_forensic_twins/` la thiet ke mo rong cua du an, khong phai reproduction day du GSD paper. Ban v2 cua paper dung Batch-SVD theo mini-batch, adaptive suppression va binary supervised training; khong dung Barlow Twins/GMM.

## 1. Architecture chinh xac theo paper

### 1.1 Hai nhanh CLIP bat doi xung

Cung mot anh dau vao duoc dua qua hai CLIP ViT-L/14 khoi tao tu cung checkpoint pre-trained:

1. **Frozen semantic anchor**: dong bang hoan toan. Nhanh nay giu nguyen hinh hoc semantic cua CLIP pre-training va chi cung cap patch tokens de uoc luong semantic subspace.
2. **Trainable forensic branch**: fine-tune end-to-end cho phan loai real/fake. Feature cua nhanh nay bi GSD loai bot thanh phan nam trong semantic subspace cua frozen branch.

CLIP ViT-L/14 co 24 Transformer blocks. Paper chen GSD vao **4 block cuoi** (tuong ung blocks 21-24 neu danh so 1-based). Cac block nong hon khong bi GSD de giu low-level texture/artifact cues.

Theo cau hinh CLIP ViT-L/14 tieu chuan, hidden width `d=1024`, patch size `14`, 24 blocks va 16 attention heads. Neu input 224x224 thi co `N=16x16=256` patch tokens cong 1 CLS token. Cac kich thuoc nay la contract cua backbone CLIP tieu chuan; paper v2 chi neu ten backbone, khong lap lai input resolution/checkpoint cu the.

### 1.2 GSD tai layer `l`

Ky hieu:

- Trainable tokens: `f_(i,l) in R^((N+1) x d)`.
- Frozen anchor tokens: `g_(i,l) in R^((N+1) x d)`.
- Chi dung **non-CLS patch tokens** cua frozen branch de tao basis.

Single-SVD ly tuong (mot SVD moi sample):

```text
Pbar_(i,l) = U_(i,l) Sigma_(i,l) V_(i,l)^T
V_(i,l)^(k) = k right-singular vectors dau tien
```

Training mac dinh dung Batch-SVD. Tai moi layer GSD, ghep patch tokens cua ca batch theo chieu token:

```text
Mbar_l = concat(Pbar_(1,l), ..., Pbar_(B,l)) in R^((B*N) x d)
Mbar_l = U_l^(b) Sigma_l^(b) (V_l^(b))^T
V_l^(b,k) in R^(d x k), k = 64
```

Voi moi sample, chieu **toan bo trainable token matrix** len shared semantic basis:

```text
f_parallel_(i,l) = f_(i,l) V_l^(b,k) (V_l^(b,k))^T
r_(i,l) = ||f_parallel_(i,l)||_F^2 / ||f_(i,l)||_F^2
lambda_(i,l) = lambda_max * sigmoid(a_l * r_(i,l) + b_l)
fhat_(i,l) = f_(i,l) - lambda_(i,l) * f_parallel_(i,l)
```

Thong so mac dinh:

- `k = 64`.
- `lambda_max = 1`.
- Moi GSD layer co hai scalar learnable rieng `a_l`, `b_l`.
- Khoi tao `a_l = 2`, `b_l = 1`.
- Basis den tu frozen branch nen khong nhan gradient cap nhat backbone anchor.

Sau block cuoi, Figure 3 cua paper the hien **Average Pooling -> linear 2-class classifier -> Real/Fake**. Loss la softmax cross-entropy hai lop (Eq. 11), khong co auxiliary loss.

## 2. Cach train paper-faithful

### 2.1 Cau hinh paper cong bo

| Hang muc | Gia tri |
|---|---|
| Backbone | Pre-trained CLIP ViT-L/14 |
| Frozen branch | Dong bang; semantic anchor |
| Trainable branch | Fine-tune toan bo end-to-end |
| GSD placement | 4 Transformer layers cuoi |
| SVD khi train | Batch-SVD tai tung GSD layer |
| Semantic basis | Top `k=64` right singular vectors |
| Adaptive suppression | `lambda_max=1`, `a_l=2`, `b_l=1` luc khoi tao |
| Classifier/loss | Linear 2-class head; cross-entropy |
| Optimizer | AdamW |
| Backbone learning rate | `1e-6` |
| Train batch size | `128` |
| Test batch size | `512` |
| Augmentation | Gaussian blur va JPEG compression theo refs. [53], [54] |
| Hardware | NVIDIA A100 (paper khong neu so GPU) |
| Inference mac dinh | Batch-Inference |

### 2.2 Mot training step

1. Lay mini-batch `B=128` anh va nhan binary real/fake; ap dung preprocessing/augmentation.
2. Dua cung input qua frozen anchor va trainable forensic CLIP.
3. Qua 20 blocks dau nhu CLIP binh thuong.
4. Tai moi trong 4 blocks cuoi:
   - Lay non-CLS tokens tu frozen anchor cho ca batch.
   - Ghep thanh ma tran `(B*N) x d`, chay mot SVD, lay `V[:, :64]`.
   - Chieu tokens cua tung sample o trainable branch len basis nay.
   - Tinh occupancy `r`, roi `lambda` bang hai scalar learnable cua layer.
   - Tru `lambda * f_parallel` khoi trainable representation.
   - Dua representation da decouple sang block tiep theo.
5. Average-pool output, dua qua linear 2-class classifier.
6. Tinh cross-entropy Eq. 11.
7. Backprop chi qua forensic CLIP, classifier va cac `{a_l,b_l}`; frozen anchor khong update.
8. AdamW update voi backbone LR `1e-6`.

## 3. Dataset/protocol train duoc paper dung

Paper bao cao ba protocol supervised rieng:

| Regime | Train | Cross-domain test |
|---|---|---|
| Face cross-dataset | FaceForensics++ `c23` | Celeb-DF-v1/v2, DFD, DFDC, DFDCP |
| Face cross-manipulation | FaceForensics++ `c23` | DF40: UniFace, e4s, FaceDancer, FSGAN, InSwap, SimSwap |
| General synthetic image | ProGAN + real counterparts | UniversalFakeDetect, 19 unseen test sets |
| GenImage | Stable Diffusion v1.4 | Midjourney, SD v1.5, ADM, GLIDE, Wukong, VQDM, BigGAN |

Day la cac run/protocol doc lap; khong nen tron tat ca dataset vao mot training run roi goi la paper-faithful.

## 4. Inference

- **Batch-Inference (mac dinh)**: tinh shared Batch-SVD basis tu chinh test mini-batch tai tung GSD layer; moi sample van co `r` va `lambda` rieng.
- **Per-sample Inference**: Single-SVD tren patch tokens cua tung anh; phu hop streaming nhung cham.
- **Reference-based Inference**: tinh basis tu mot reference set, sau do tai su dung cho anh test. Thi nghiem face dung 512 anh tham chieu tu FF++ hoac CDv2.

## 5. Nhung gi paper v2 KHONG cong bo

Khong the goi mot implementation la “exact reproduction” neu chua co code/config tac gia, vi paper khong neu:

- checkpoint CLIP/provider cu the va input resolution/preprocessing chinh xac;
- so epoch/iteration, learning-rate schedule, warmup, classifier LR;
- AdamW betas, epsilon, weight decay;
- ty le/range Gaussian blur va JPEG, thu tu augmentation, crop/resize/flip;
- sampling frames cho video, class balance va sampler;
- so A100, distributed strategy, mixed precision, gradient clipping/accumulation;
- seed va checkpoint-selection rule;
- SVD precision/solver, xu ly rank thieu, epsilon trong occupancy;
- chi tiet kich thuoc/reference-set construction ngoai ablation 512 anh face.

Vi vay, bang o muc 2.1 la **toan bo recipe chinh xac ma paper cong bo**; cac muc tren phai duoc gan nhan implementation assumption neu code tac gia chua duoc phat hanh.

## 6. File so do

`gsd_clip_v2_architecture.drawio` gom 3 trang:

1. `01 - Architecture`: dual-stream CLIP, 4 GSD blocks, pooling/classifier/loss.
2. `02 - GSD Batch-SVD`: tensor shapes va day du cong thuc projection/occupancy/adaptive subtraction.
3. `03 - Train & Inference`: recipe duoc cong bo, training step, 3 che do inference va cac thong so con thieu.

