# Data Loader Chung Cho Tiny-GenImage

Folder này chứa code load dữ liệu dùng chung cho các phase của dự án fake image detection:

- Stage 1: train classifier `real/fake`.
- Stage 2: train projector/alignment để MLLM dự đoán token `real/fake`.
- Stage 3 sau này: có thể mở rộng cùng schema để load dữ liệu SFT có `prompt/response/explanation`.

Hiện tại loader mới hỗ trợ Tiny-GenImage từ Hugging Face:

```text
TheKernel01/Tiny-GenImage
```

Dữ liệu gốc vẫn nằm trên Hugging Face. Repo chỉ lưu code loader, split, transform.

Có hai cách load:

- `streaming=True`: đọc trực tiếp/lazy từ Hugging Face khi iterate, không tải toàn bộ dataset về local cache trước.
- `streaming=False`: tải/cache dataset về máy đang chạy để train nhanh và ổn định hơn.

Nếu muốn tránh tải full dataset về local, dùng `streaming=True`.

## 1. Cấu Trúc File

```text
data_loader/
  __init__.py
  schemas.py
  tiny_genimage.py
  splits.py
  transforms.py
  README.md
```

### `schemas.py`

Định nghĩa schema metadata chung.

Các thành phần chính:

- `LABEL_ID_TO_NAME`: map `{0: "real", 1: "fake"}`.
- `LABEL_NAME_TO_ID`: map `{"real": 0, "fake": 1}`.
- `UnifiedSample`: dataclass mô tả metadata chuẩn cho một sample.

`UnifiedSample` có các field quan trọng:

```python
sample_id
label
label_name
dataset_source
generator
split
eval_case
generator_id
image_path
prompt
response
explanation
forensic_attributes
metadata
```

Ý tưởng là mọi dataset sau này, kể cả Holmes SFT, đều nên được normalize về format này.

### `tiny_genimage.py`

Adapter chính cho Tiny-GenImage.

Các hàm/lớp chính:

- `load_tiny_genimage(cache_dir=None, streaming=False)`: gọi `datasets.load_dataset("TheKernel01/Tiny-GenImage", streaming=streaming)`.
- `TinyGenImageDataset`: wrapper PyTorch Dataset, chuyển sample Hugging Face thành dict thống nhất.
- `collate_unified_batch(batch)`: gom batch để dùng với `torch.utils.data.DataLoader`.

`TinyGenImageDataset` hỗ trợ `task_type`:

```text
classification
alignment
sft
inference
```

Trong hiện tại nên dùng:

- `classification` cho Stage 1.
- `alignment` cho Stage 2.

Khi `task_type="alignment"`, dataset sẽ thêm:

```python
target_text = "real" hoặc "fake"
label_token = "real" hoặc "fake"
```

### `splits.py`

Tạo các split evaluation cho Tiny-GenImage.

Các thành phần chính:

- `TinyGenImageSplitConfig`: config chọn evaluation case.
- `build_tiny_genimage_splits(config)`: trả về dict gồm `train`, `eval`, tên split, tên case và ghi chú.

Các `eval_case` đang hỗ trợ:

```text
combined
in_domain
cross_generator
train_one_generator
```

Mặc định split builder dùng:

```python
balance_real=True
seed=42
```

Nghĩa là mỗi split sẽ lấy số ảnh `real` bằng số ảnh `fake`. Cách này tránh bias do class imbalance, đặc biệt trong `in_domain` và `cross_generator`.

Lưu ý quan trọng: bản Hugging Face của Tiny-GenImage là bản flattened, trong đó ảnh real có `generator = Real`. Nó không còn giữ thông tin real thuộc folder generator nào như bản Kaggle/folder. Vì vậy loader sẽ chọn một tập real cân bằng từ toàn bộ real pool bằng seed cố định.

### `transforms.py`

Tạo transform ảnh cho training/evaluation.

Hàm chính:

```python
build_image_transform(...)
```

Các perturbation cho robustness:

```text
none
jpeg
resize
center_crop
blur
```

Các class phụ:

- `JpegCompression`
- `ResizeRoundTrip`
- `CenterCropRatio`
- `GaussianBlur`

## 2. Cài Đặt Thư Viện

Nếu chạy trong notebook:

```python
%pip install -q datasets torchvision matplotlib
```

Nếu chạy script:

```bash
pip install datasets torchvision matplotlib
```

## 3. Load Split Cơ Bản

Ví dụ load case `combined`:

```python
from data_loader import TinyGenImageSplitConfig, build_tiny_genimage_splits

config = TinyGenImageSplitConfig(eval_case="combined")
splits = build_tiny_genimage_splits(config)

print(splits["notes"])
print(len(splits["train"]))
print(len(splits["eval"]))
```

Muốn stream từ Hugging Face, bật:

```python
config = TinyGenImageSplitConfig(
    eval_case="combined",
    streaming=True,
    balance_real=True,
    seed=42,
)
splits = build_tiny_genimage_splits(config)
```

Output `splits` có dạng:

```python
{
    "train": ...,
    "eval": ...,
    "train_split_name": "train",
    "eval_split_name": "validation",
    "eval_case": "combined",
    "dataset_name": "TheKernel01/Tiny-GenImage",
    "streaming": True hoặc False,
    "balance_real": True,
    "train_real_count": 14000,
    "train_fake_count": 14000,
    "eval_real_count": 3500,
    "eval_fake_count": 3500,
    "train_rows": 28000,
    "eval_rows": 7000,
    "notes": "..."
}
```

## 4. Các Trường Hợp Evaluation

### Case 1: `combined`

Train trên toàn bộ generator trong split train, test trên toàn bộ generator trong split validation.

```python
config = TinyGenImageSplitConfig(
    eval_case="combined",
    balance_real=True,
    seed=42,
)
splits = build_tiny_genimage_splits(config)
```

Số lượng:

```text
train = 14,000 real + 14,000 fake = 28,000
eval  = 3,500 real + 3,500 fake = 7,000
```

Dùng cho:

- baseline chính Stage 1,
- Stage 2 alignment clean,
- so sánh model tổng quát trên Tiny-GenImage.

### Case 2: `in_domain`

Train và test cùng một fake generator, cộng với ảnh real.

```python
config = TinyGenImageSplitConfig(
    eval_case="in_domain",
    generator="BigGAN",
    balance_real=True,
    seed=42,
)
splits = build_tiny_genimage_splits(config)
```

Ý nghĩa:

```text
train = real + fake BigGAN từ train split
eval  = real + fake BigGAN từ validation split
```

Số lượng dự kiến:

```text
train = 2,000 real + 2,000 fake BigGAN = 4,000
eval  = 500 real + 500 fake BigGAN = 1,000
```

Dùng để kiểm tra model học tốt trên generator đã thấy chưa.

### Case 3: `cross_generator`

Leave-one-generator-out. Train loại bỏ fake của generator held-out, test trên fake generator đó.

```python
config = TinyGenImageSplitConfig(
    eval_case="cross_generator",
    heldout_generator="GLIDE",
    balance_real=True,
    seed=42,
)
splits = build_tiny_genimage_splits(config)
```

Ý nghĩa:

```text
train = real + fake từ tất cả generator trừ GLIDE
eval  = real + fake GLIDE
```

Số lượng dự kiến:

```text
train = 12,000 real + 12,000 fake non-GLIDE = 24,000
eval  = 500 real + 500 fake GLIDE = 1,000
```

Dùng cho main cross-generator evaluation vì nó đo khả năng generalize sang generator chưa thấy.

### Case 4: `train_one_generator`

Train trên một fake generator, test trên toàn bộ validation.

```python
config = TinyGenImageSplitConfig(
    eval_case="train_one_generator",
    base_generator="BigGAN",
    balance_real=True,
    seed=42,
)
splits = build_tiny_genimage_splits(config)
```

Ý nghĩa:

```text
train = real + fake BigGAN
eval  = real + fake từ tất cả generator validation
```

Số lượng dự kiến:

```text
train = 2,000 real + 2,000 fake BigGAN = 4,000
eval  = 3,500 real + 3,500 fake all generators = 7,000
```

Dùng như stress test. Nếu kết quả thấp trên generator khác thì chứng minh model đang học artifact riêng của generator.

## 5. Tạo PyTorch DataLoader Cho Stage 1

```python
from torch.utils.data import DataLoader
from data_loader import (
    TinyGenImageDataset,
    TinyGenImageIterableDataset,
    TinyGenImageSplitConfig,
    build_image_transform,
    build_tiny_genimage_splits,
    collate_unified_batch,
)

splits = build_tiny_genimage_splits(
    TinyGenImageSplitConfig(
        eval_case="combined",
        streaming=True,
        balance_real=True,
        seed=42,
    )
)

DatasetClass = TinyGenImageIterableDataset if splits["streaming"] else TinyGenImageDataset

train_dataset = DatasetClass(
    splits["train"],
    split_name=splits["train_split_name"],
    eval_case=splits["eval_case"],
    transform=build_image_transform(image_size=224, train=True),
    task_type="classification",
)

eval_dataset = DatasetClass(
    splits["eval"],
    split_name=splits["eval_split_name"],
    eval_case=splits["eval_case"],
    transform=build_image_transform(image_size=224, train=False),
    task_type="classification",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False if splits["streaming"] else True,
    num_workers=2,
    collate_fn=collate_unified_batch,
)

batch = next(iter(train_loader))
print(batch["image"].shape)
print(batch["label"])
print(batch["generator"])
```

Batch trả về:

```python
{
    "image": torch.Tensor,       # [B, 3, H, W]
    "label": torch.LongTensor,   # [B]
    "label_name": list[str],
    "generator": list[str],
    "sample_id": list[str],
    "metadata": list[dict],
}
```

## 6. Dùng Cho Stage 2 Alignment

Stage 2 cần target dạng text/token `real` hoặc `fake`.

```python
alignment_dataset = TinyGenImageDataset(
    splits["train"],
    split_name=splits["train_split_name"],
    eval_case=splits["eval_case"],
    transform=build_image_transform(image_size=224, train=False),
    task_type="alignment",
)

alignment_loader = DataLoader(
    alignment_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_unified_batch,
)

batch = next(iter(alignment_loader))
print(batch["label_token"])
print(batch["target_text"])
```

Batch sẽ có thêm:

```python
label_token: ["real", "fake", ...]
target_text: ["real", "fake", ...]
```

Phần projector/MLLM sau này có thể dùng `target_text` để tạo loss dự đoán token.

## 7. Test Robustness

Robustness không nên trộn vào train. Nên train model trên clean data, sau đó chỉ đổi transform ở eval.

JPEG compression:

```python
robust_transform = build_image_transform(
    image_size=224,
    train=False,
    perturbation="jpeg",
    jpeg_quality=50,
)
```

Resize/downsample:

```python
robust_transform = build_image_transform(
    image_size=224,
    train=False,
    perturbation="resize",
    resize_scale=0.5,
)
```

Center crop:

```python
robust_transform = build_image_transform(
    image_size=224,
    train=False,
    perturbation="center_crop",
    crop_ratio=0.8,
)
```

Blur:

```python
robust_transform = build_image_transform(
    image_size=224,
    train=False,
    perturbation="blur",
    blur_radius=2.0,
)
```

Tạo eval dataset robustness:

```python
robust_eval_dataset = TinyGenImageDataset(
    splits["eval"],
    split_name=splits["eval_split_name"],
    eval_case=f'{splits["eval_case"]}:jpeg_q50',
    transform=robust_transform,
    task_type="classification",
)
```

Metric nên báo:

```text
clean_acc
robust_acc
accuracy_drop = clean_acc - robust_acc
clean_f1
robust_f1
f1_drop = clean_f1 - robust_f1
```

## 8. Streaming Và Cache Dataset

Nếu dùng:

```python
TinyGenImageSplitConfig(streaming=True)
```

loader sẽ dùng Hugging Face streaming. Nó không tải toàn bộ parquet dataset về local cache trước. Dữ liệu được đọc dần khi vòng lặp `DataLoader` chạy.

Điểm đổi lại của streaming:

- Không có `len()` chính xác cho split.
- Không shuffle random toàn dataset như map-style dataset.
- Training có thể chậm hơn vì phụ thuộc mạng.
- Một số operation như `.select(...)` không dùng được; dùng `.take(n)` để test vài mẫu.

Nếu dùng:

```python
TinyGenImageSplitConfig(streaming=False)
```

Hugging Face sẽ tải/cache dataset ở máy đang chạy. Cách này tốn dung lượng nhưng train lặp lại nhanh hơn.

Nếu chạy Colab và muốn cache bền trong Google Drive:

```python
config = TinyGenImageSplitConfig(
    eval_case="combined",
    cache_dir="/content/drive/MyDrive/hf_cache",
    streaming=False,
)
```

Nếu không set `cache_dir`, Colab có thể phải tải lại sau khi runtime mất.

## 9. Notebook Test

Notebook test nằm ở:

```text
notebooks/test_tiny_genimage_loader.ipynb
```

Notebook này kiểm tra:

- import module `data_loader`,
- load Tiny-GenImage từ Hugging Face,
- tạo split theo `EVAL_CASE`,
- tạo PyTorch DataLoader,
- hiển thị vài ảnh trong batch,
- test robustness transform,
- test `task_type="alignment"` cho Stage 2.

## 10. Lưu Ý Khi Dùng Cho Train

- Không lưu ảnh vào repo.
- Chỉ lưu code, config split, metric, prediction, checkpoint.
- Khi báo kết quả, luôn ghi rõ `eval_case`, generator train/heldout, split và perturbation.
- Với `cross_generator`, nên chạy lần lượt nhiều `heldout_generator` rồi lấy trung bình.
- Với robustness, phải dùng cùng checkpoint clean, chỉ thay eval transform.
- Với Stage 2, trước mắt dùng target `real/fake`; sau này nếu có forensic attributes thì thêm target phụ để tránh projector thành classifier head đơn giản.

## 11. Loader Cho Kaggle Folder Tiny-GenImage

Nếu đã add dataset Tiny-GenImage vào Kaggle Input, dùng:

```python
from data_loader import (
    TinyGenImageKaggleConfig,
    TinyGenImageKaggleDataset,
    build_kaggle_tiny_splits,
    find_tiny_genimage_root,
    summarize_index,
)
```

File chính:

```text
data_loader/tiny_genimage_kaggle.py
```

Loader này không dùng Hugging Face streaming. Nó scan cấu trúc folder trong `/kaggle/input`, lưu metadata đường dẫn ảnh vào `DataFrame`, rồi `TinyGenImageKaggleDataset` chỉ mở từng ảnh khi `DataLoader` gọi `__getitem__`.

Cấu trúc folder được hỗ trợ:

```text
tiny-genimage-root/
  BigGAN/
    train/
      ai/
      nature/
    val/
      ai/
      nature/
  GLIDE/
    train/
      ai/
      nature/
    val/
      ai/
      nature/
```

Các alias cũng được hỗ trợ:

```text
split: train, training, val, valid, validation, test
label real: nature, real, 0_real, 0-real, 0
label fake: ai, fake, 1_fake, 1-fake, 1
```

Ví dụ kiểm tra cấu trúc dataset trên Kaggle:

```python
root = find_tiny_genimage_root()
config = TinyGenImageKaggleConfig(dataset_root=str(root), eval_case="combined")
splits = build_kaggle_tiny_splits(config)

print(splits["dataset_root"])
print(splits["notes"])
print(splits["train_real_count"], splits["train_fake_count"])
print(splits["eval_real_count"], splits["eval_fake_count"])
```

Các `eval_case` giữ cùng ý nghĩa với loader Hugging Face:

```text
combined
in_domain
cross_generator
train_one_generator
```

Khác biệt quan trọng: bản Kaggle/folder giữ real ảnh theo từng generator folder (`BigGAN/train/nature`, `GLIDE/train/nature`, ...), nên case `in_domain` và `train_one_generator` công bằng hơn bản Hugging Face flattened.
