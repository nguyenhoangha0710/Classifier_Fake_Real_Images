# Forensic Twins — Kaggle 2×T4 implementation

Đây là implementation `provisional_v1` của Forensic Twins, bám các thông số paper đã công bố và ghi rõ mọi giả định còn thiếu. Encoder/GMM chỉ train bằng ảnh thật.

## Trạng thái triển khai

Đã có:

- manifest real-only 50K;
- tự tìm ImageNet và COCO trong Kaggle Input;
- codec audit, SHA-256, kiểm tra ảnh hỏng và min-side 256;
- shared non-destructive rotation/flip;
- hai crop batch-wise từ `{64,96,128}`, tuyệt đối không giao nhau;
- FRE pretrained frozen `1→8`, kernel 15;
- ViT 6 blocks, patch 8, 2D RoPE, `[CLS]` 192-D;
- projector provisional `192→2048→8192`;
- global Barlow Twins loss qua autograd-aware DDP all-gather;
- training FP16 trên 2×T4;
- atomic checkpoint, checksum, resume model/optimizer/scheduler/RNG;
- trích embedding, fit Gaussian `K=1` full covariance;
- đánh giá Tiny-GenImage native và fake-codec-matched.

Chưa được paper v1 công bố và đang dùng giả định:

- crop-size distribution;
- số attention heads và MLP ratio;
- cấu trúc projector;
- xác suất augmentation;
- AdamW betas/epsilon.

## Kaggle Input cần gắn

Theo cấu trúc hiện tại:

```text
/kaggle/input/imagenet-object-localization-challenge/
└── ILSVRC/Data/CLS-LOC/train/<synset>/*.JPEG

/kaggle/input/coco-2017-dataset/
└── coco2017/train2017/*.jpg
```

COCO input trong ảnh hiện không có `unlabeled2017`. Script tự dùng 25K `train2017` và ghi deviation vào `train_real_50k.summary.json`. Nếu sau này add `unlabeled2017`, script tự chuyển sang 12.235 train + 12.765 unlabeled.

Để evaluation, add thêm Tiny-GenImage có cấu trúc:

```text
<generator>/val/{nature,ai}/...
```

FRE:

- ưu tiên file `fre.pt` ở bất kỳ Kaggle Input nào;
- nếu không tìm thấy, tải từ official FSD release v1.2.0;
- nếu Kaggle Internet tắt, chương trình dừng và yêu cầu add `fre.pt`, không dùng weight ngẫu nhiên.

## Cấu hình 2×T4

File: `configs/kaggle_2xt4.json`.

```text
2 processes / 2 GPUs
micro batch       64/GPU
global batch      128
precision         FP16
base LR           0.0006
epochs            80
warm-up           15 epochs
cosine minimum    1e-6
weight decay      0.0025
```

T4 không có native BF16 như RTX 4090 của paper nên cấu hình dùng FP16. Global batch giảm từ 320 xuống 128 và LR được scale tuyến tính từ 0.0015 xuống 0.0006.

## Chạy trên Kaggle

Notebook tự chứa source code đã chuẩn bị sẵn: `forensic_twins_kaggle_2xt4.ipynb`. Import file này vào Kaggle, chỉ add ImageNet/COCO/Tiny-GenImage rồi điền `CHECKPOINT_DATASET_SLUG` ở cell đầu tiên. Notebook tự bung code, tải `fre.pt` từ official release và tải checkpoint mới nhất bằng Kaggle API khi resume.

Đặt thư mục code này trong `/kaggle/working` hoặc add nó như một Kaggle Dataset. Từ notebook:

```python
import os
from pathlib import Path

CODE_DIR = Path("/kaggle/working/forensic_twins_original")
os.chdir(CODE_DIR)
```

Kiểm tra manifest trước:

```bash
python kaggle_launcher.py manifest \
  --config configs/kaggle_2xt4.json
```

Manifest construction is CPU/I/O-only and intentionally runs in one process.
It samples ImageNet through `LOC_train_solution.csv` and COCO through
`captions_train2017.json` (falling back to `instances_train2017.json`), so it
does not recursively enumerate every training image through the Kaggle mount.

Train hoặc tự resume (cách ngắn nhất):

```bash
python kaggle_launcher.py train \
  --resume auto \
  --kaggle-dataset-slug YOUR_KAGGLE_USERNAME/forensic-twins-checkpoints
```

Lệnh DDP tương đương:

```bash
torchrun --standalone --nproc_per_node=2 train_forensic_twins_kaggle.py \
  --config configs/kaggle_2xt4.json \
  --resume auto
```

Không chạy trực tiếp bằng `python train_forensic_twins_kaggle.py`: config yêu cầu đúng hai DDP processes.

## Checkpoint

Output mặc định:

```text
/kaggle/working/forensic_twins_original/artifacts/checkpoints/
├── last.ckpt
├── last.ckpt.sha256
├── step_XXXXXXXX.ckpt
├── epoch_XXXX.ckpt
└── checkpoint_index.json
```

Lịch save:

- mỗi 250 optimizer steps;
- cuối mỗi epoch;
- giữ milestone local mỗi 5 epoch;
- upload/version checkpoint cuối **mỗi epoch** lên private Kaggle Dataset khi đã cấu hình slug;
- final epoch 80.

`resume=auto` kiểm tra:

1. `/kaggle/working/.../checkpoints`;
2. các Kaggle Inputs có tên chứa `checkpoint`, `forensic` hoặc `twins`;
3. checksum, config hash, manifest hash và FRE hash.

Để checkpoint tồn tại sau khi Kaggle session mất, điền private dataset slug:

```json
"checkpoint": {
  "kaggle_dataset_slug": "YOUR_KAGGLE_USERNAME/forensic-twins-checkpoints"
}
```

Dataset private cần được tạo trước và Kaggle API credential phải hoạt động. Nếu upload/version thất bại, log hiện `persistent-checkpoint:WARNING`. Chỉ ghi `/kaggle/working` không bảo vệ được khi session bị xóa; cần upload dataset hoặc `Save Version` notebook.

Có thể truyền slug từ notebook bằng `--kaggle-dataset-slug`; không bắt buộc sửa file config read-only trong Kaggle Input. Khi slug hợp lệ, checkpoint cuối mỗi epoch được version hóa; checkpoint mỗi 250 step vẫn được giữ local để phục hồi lỗi process trong cùng session.

## Fit GMM và đánh giá

Sau khi có checkpoint:

```bash
python fit_gmm_and_evaluate.py \
  --config configs/kaggle_2xt4.json
```

Hoặc dùng launcher: `python kaggle_launcher.py gmm`.

Script thực hiện:

```text
50K real → frozen FRE/encoder → CLS 192-D
→ fit K=1 full-covariance Gaussian
→ native GenImage evaluation
→ fake PNG encode một lần theo JPEG profile của real đối chứng
→ codec-matched evaluation
```

Kết quả chính là `fake_codec_matched`. `native_confounded` chỉ dùng để định lượng shortcut JPEG-real/PNG-fake.

Artifacts:

```text
/kaggle/working/forensic_twins_original/artifacts/
├── embeddings/train_real_cls192.npy
├── gmm/real_k1_full.npz
├── metrics/native_confounded_summary.json
├── metrics/fake_codec_matched_summary.json
└── metrics/codec_comparison.json
```

## Smoke test

```bash
python tests/test_smoke.py
```

## GMM đa tiền xử lý và B-Free

Notebook `forensic_twins_augmented_gmm_bfree_kaggle.ipynb` đọc frozen encoder
200K, tạo sáu biến thể hợp lệ từ ảnh real, fit các GMM K=1/2/4/8, hiệu chuẩn
q80–q98 trên held-out real rồi đánh giá B-Free `valid_clean`. Quy trình, input,
resume và toàn bộ artifact được mô tả trong `augmented_gmm_workflow.md`.

Pipeline này cần ImageNet, COCO, encoder checkpoint 200K và B-Free; không cần
GMM K=1 cũ vì các GMM/threshold mới được fit lại từ đầu.

## Notebook GMM/evaluation sau khi train

Dung `forensic_twins_gmm_eval_kaggle.ipynb` de fit GMM va danh gia ma khong
train lai encoder. Notebook yeu cau checkpoint Dataset, ImageNet, COCO va
Tiny-GenImage. Quy trinh threshold, metrics va artifact duoc mo ta trong
`gmm_evaluation_workflow.md`.

Test kiểm tra forward/backward, shape, crop không giao nhau, FRE checkpoint loading và atomic checkpoint integrity.

## Nguồn

- [Forensic Twins arXiv v1](https://arxiv.org/html/2609.31514v1)
- [Official FSD/FRE repository](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25)
- [Official FRE release](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/releases/tag/v1.2.0)
