# Forensic Twins 200K checkpoints

Tài liệu này ghi lại hai Kaggle Dataset chứa checkpoint của mô hình Forensic
Twins 200K, vai trò của từng dataset và cách gắn chúng vào notebook Kaggle.

## 1. Encoder checkpoint và FRE

- URL: <https://www.kaggle.com/datasets/hoangha071005/twins-gmm-200k-checkpoint>
- Kaggle slug: `hoangha071005/twins-gmm-200k-checkpoint`

Các file quan trọng:

```text
forensic_twins_v2_200k/
├── artifacts/checkpoints/last.ckpt
├── cache/fre.pt
└── manifests/
    ├── train_real_200k.csv
    └── train_real_200k.summary.json
```

Vai trò:

- `last.ckpt`: trọng số encoder Forensic Twins, projector và trạng thái training;
- `fre.pt`: trọng số Forensic Residual Extractor frozen;
- `train_real_200k.csv`: manifest chính xác của 200K ảnh real đã dùng để train;
- manifest được dùng để audit data leakage khi đánh giá trên dataset mới.

Dataset này **không chứa fitted GMM cuối cùng**. Chỉ có `last.ckpt` là chưa đủ
để dự đoán real/fake: ảnh còn phải được biến đổi thành embedding rồi chấm điểm
bằng GMM và threshold.

## 2. Fitted GMM, thresholds và kết quả evaluation

- URL: <https://www.kaggle.com/datasets/hoangha071005/fit-gmm-checkpoint-200k>
- Kaggle slug: `hoangha071005/fit-gmm-checkpoint-200k`

Các file quan trọng:

```text
forensic_twins_200k_gmm_eval/
├── artifacts/
│   └── gmm/
│       ├── real_k1_full.npz
│       ├── real_k1_full.json
│       └── thresholds.json
├── manifests/
│   ├── gmm_fit_real.csv
│   ├── gmm_calibration_real.csv
│   └── gmm_split_summary.json
├── artifacts/embeddings/
├── artifacts/metrics/
└── artifacts/figures/
```

Vai trò:

- `real_k1_full.npz`: tham số GMM một component, full covariance;
- `real_k1_full.json`: metadata và provenance của GMM;
- `thresholds.json`: các ngưỡng percentile q80 đến q98 tính trên real
  calibration;
- các thư mục embeddings, metrics và figures lưu kết quả Tiny-GenImage đã chạy.

GMM được fit trên embedding real 192 chiều. Quy tắc dự đoán là:

```text
NLL > threshold  -> fake
NLL <= threshold -> real
```

Dataset này **không chứa `last.ckpt` hoặc `fre.pt`**. Vì vậy khi suy luận trên
ảnh mới, phải gắn thêm dataset encoder checkpoint ở mục 1.

## 3. Input cần gắn theo từng tác vụ

### Test trên một dataset mới

Ví dụ test B-Free hoặc Tiny-GenImage:

```text
1. hoangha071005/twins-gmm-200k-checkpoint
2. hoangha071005/fit-gmm-checkpoint-200k
3. dataset ảnh cần test
```

Không cần gắn ImageNet hoặc COCO. Encoder, FRE, GMM và thresholds đã có sẵn.
Manifest trong checkpoint đủ để kiểm tra overlap theo filename/COCO ID mà không
cần đọc lại ảnh COCO gốc.

Với B-Free, dataset ảnh là:

- URL: <https://www.kaggle.com/datasets/baopv051/b-free-training-data>
- slug: `baopv051/b-free-training-data`

### Resume training encoder

Input cần có:

```text
1. hoangha071005/twins-gmm-200k-checkpoint
2. ImageNet Object Localization Challenge
3. awsaf49/coco-2017-dataset
```

ImageNet và COCO vẫn cần trong trường hợp này vì training loader phải mở các
ảnh được ghi trong manifest. `last.ckpt` chỉ lưu trạng thái mô hình và optimizer,
không chứa pixel ảnh training.

### Fit lại GMM từ ảnh real

Nếu muốn trích lại embedding từ đầu rồi fit một GMM mới, cần:

```text
1. hoangha071005/twins-gmm-200k-checkpoint
2. ImageNet Object Localization Challenge
3. awsaf49/coco-2017-dataset
```

Nếu chỉ muốn đọc lại GMM đã fit để chấm điểm embedding/ảnh mới thì không cần
ImageNet hoặc COCO; dùng thêm `fit-gmm-checkpoint-200k` là đủ.

## 4. Đường dẫn Kaggle

Kaggle có thể mount dataset theo một trong hai kiểu:

```text
/kaggle/input/<dataset-slug>/
/kaggle/input/datasets/<owner>/<dataset-slug>/
```

Không nên dùng `rglob("*")` trên toàn bộ `/kaggle/input`, đặc biệt khi B-Free có
hàng trăm nghìn ảnh. Dùng danh sách candidate root cố định:

```python
from pathlib import Path


def resolve_kaggle_dataset(owner: str, slug: str) -> Path:
    candidates = [
        Path("/kaggle/input") / slug,
        Path("/kaggle/input/datasets") / owner / slug,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Dataset chưa được gắn: {owner}/{slug}; candidates={candidates}"
    )


encoder_root = resolve_kaggle_dataset(
    "hoangha071005", "twins-gmm-200k-checkpoint"
)
gmm_root = resolve_kaggle_dataset(
    "hoangha071005", "fit-gmm-checkpoint-200k"
)

encoder_checkpoint = (
    encoder_root
    / "forensic_twins_v2_200k"
    / "artifacts"
    / "checkpoints"
    / "last.ckpt"
)
fre_checkpoint = (
    encoder_root
    / "forensic_twins_v2_200k"
    / "cache"
    / "fre.pt"
)
training_manifest = (
    encoder_root
    / "forensic_twins_v2_200k"
    / "manifests"
    / "train_real_200k.csv"
)

gmm_npz = (
    gmm_root
    / "forensic_twins_200k_gmm_eval"
    / "artifacts"
    / "gmm"
    / "real_k1_full.npz"
)
gmm_metadata = gmm_npz.with_suffix(".json")
thresholds_json = gmm_npz.parent / "thresholds.json"

required = {
    "encoder checkpoint": encoder_checkpoint,
    "FRE checkpoint": fre_checkpoint,
    "training manifest": training_manifest,
    "GMM": gmm_npz,
    "GMM metadata": gmm_metadata,
    "thresholds": thresholds_json,
}
for name, path in required.items():
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {name}: {path}")
    print(f"{name}: {path}")
```

## 5. Luồng inference chuẩn

```text
Ảnh đầu vào
  -> decode RGB
  -> center crop 256x256, không resize
  -> grayscale và chia 255
  -> halo 7 pixel
  -> frozen FRE
  -> frozen Forensic Twins encoder
  -> CLS embedding 192-D
  -> GMM negative log-likelihood (NLL)
  -> so sánh với q80/q86/q95
  -> real hoặc fake
```

Không được fit lại GMM hoặc tính lại threshold bằng nhãn của test set nếu mục
tiêu là đánh giá zero-shot.

## 6. Kiểm tra tương thích trước khi chạy

Trước inference phải xác nhận:

- encoder checkpoint và `fre.pt` load thành công;
- embedding có đúng 192 chiều;
- GMM là `K=1`, full covariance và có cùng embedding dimension;
- center crop inference là 256;
- `thresholds.json` chứa các q80 đến q98;
- q80, q86 và q95 được lấy trực tiếp từ file, không tính từ test labels;
- mọi output được ghi vào `/kaggle/working`, vì `/kaggle/input` là read-only.

## 7. Khuyến nghị đóng gói sau này

Hai dataset checkpoint hiện tách rời encoder/FRE và fitted GMM. Có thể tạo một
private Kaggle Dataset nhỏ gọn dành riêng cho inference, gồm:

```text
inference_bundle/
├── last.ckpt
├── fre.pt
├── real_k1_full.npz
├── real_k1_full.json
├── thresholds.json
├── train_real_200k.csv
└── config.json
```

Khi đó mỗi notebook test chỉ cần gắn `inference_bundle` và dataset ảnh cần đánh
giá.

## 8. Notebook đánh giá B-Free

- Notebook tự chứa source: `forensic_twins_bfree_eval_kaggle.ipynb`
- Script chính: `test_bfree_gmm_kaggle.py`
- Mô tả cohort, manifest, protocol và resume: `bfree_evaluation_workflow.md`

Notebook dùng đồng thời hai checkpoint Dataset ở mục 1–2 và B-Free, tự loại
COCO ID đã xuất hiện trong manifest train 200K, rồi lưu manifest sạch trước khi
trích embedding. Không fit lại GMM hoặc threshold trên B-Free.
