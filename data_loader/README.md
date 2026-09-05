# Data Loader Cho Tiny-GenImage

Tài liệu này mô tả dataloader dùng cho các thí nghiệm phân loại ảnh thật/giả trong dự án. Mục tiêu chính là chuẩn hóa cách đọc dữ liệu, cách chia train/eval, và cách trả metadata để các phase sau có thể dùng lại cùng một format.

## 1. Mục Tiêu Thiết Kế

Data loader được xây dựng để phục vụ nhiều phase:

```text
Stage 1: image -> classifier -> real/fake
Stage 2: image -> visual encoder/projector -> MLLM token real/fake
Stage 3: image + prompt -> MLLM -> explanation/answer
```

Vì vậy mỗi sample được normalize về cùng một schema, gồm ảnh, nhãn, generator, split, eval case và metadata. Cách này giúp baseline CLIP, ResNet-50, CNNSpot, hoặc các mô hình VLLM/MLLM sau này dùng chung một interface dữ liệu.

## 2. Nguồn Dữ Liệu Hỗ Trợ

Hiện tại hỗ trợ Tiny-GenImage theo hai dạng.

### Hugging Face Dataset

File chính:

```text
data_loader/tiny_genimage.py
data_loader/splits.py
```

Dataset:

```text
TheKernel01/Tiny-GenImage
```

Dạng này đọc bằng `datasets.load_dataset`. Có hai chế độ:

```python
streaming=True
```

Đọc lazy từ Hugging Face khi iterate. Không tải toàn bộ dataset trước, nhưng tốc độ phụ thuộc mạng và khó dùng một số thao tác random access.

```python
streaming=False
```

Tải/cache dataset về máy chạy. Train ổn định hơn nhưng tốn dung lượng local.

### Kaggle Folder Dataset

File chính:

```text
data_loader/tiny_genimage_kaggle.py
```

Dạng này dùng khi Tiny-GenImage đã được add vào Kaggle Input. Cấu trúc folder thực tế:

```text
tiny-genimage/
  imagenet_ai_0419_biggan/
    train/
      ai/
      nature/
    val/
      ai/
      nature/
  imagenet_ai_0419_vqdm/
  imagenet_ai_0424_sdv5/
  imagenet_ai_0424_wukong/
  imagenet_ai_0508_adm/
  imagenet_glide/
  imagenet_midjourney/
```

Trong đó:

```text
nature -> real -> label 0
ai     -> fake -> label 1
```

Loader Kaggle chỉ scan đường dẫn ảnh vào `DataFrame`, sau đó ảnh được mở lazy trong `__getitem__`. Điều này tránh giữ toàn bộ ảnh trong RAM.

## 3. Schema Chung Của Một Sample

File:

```text
data_loader/schemas.py
```

Class chính:

```python
UnifiedSample
```

Các field quan trọng:

```text
sample_id          ID duy nhất của sample
label              0 hoặc 1
label_name         real hoặc fake
dataset_source     Tiny-GenImage hoặc Tiny-GenImage-Kaggle
generator          tên generator, ví dụ BigGAN hoặc imagenet_ai_0419_biggan
split              train hoặc validation
eval_case          protocol thí nghiệm đang chạy
generator_id       ID generator nếu dataset có ClassLabel
image_path         đường dẫn ảnh nếu có
prompt             dùng cho phase SFT/inference sau này
response           câu trả lời target nếu có
explanation        giải thích nếu có
forensic_attributes thuộc tính forensic nếu có
metadata           dict phụ để mở rộng
```

Quy ước nhãn:

```python
LABEL_ID_TO_NAME = {0: "real", 1: "fake"}
LABEL_NAME_TO_ID = {"real": 0, "fake": 1}
```

## 4. Các File Và Chức Năng

```text
data_loader/
  __init__.py
  schemas.py
  tiny_genimage.py
  tiny_genimage_kaggle.py
  splits.py
  transforms.py
  README.md
```

### `schemas.py`

Định nghĩa schema chuẩn `UnifiedSample`. Đây là hợp đồng dữ liệu chung giữa dataloader và model.

### `tiny_genimage.py`

Đọc Tiny-GenImage từ Hugging Face.

Hàm/lớp chính:

```python
load_tiny_genimage(cache_dir=None, streaming=False)
TinyGenImageDataset
TinyGenImageIterableDataset
collate_unified_batch
```

`TinyGenImageDataset` dùng cho map-style dataset. `TinyGenImageIterableDataset` dùng cho streaming dataset.

### `splits.py`

Tạo các split thí nghiệm cho Tiny-GenImage Hugging Face.

Hàm/lớp chính:

```python
TinyGenImageSplitConfig
build_tiny_genimage_splits(config)
```

### `tiny_genimage_kaggle.py`

Đọc Tiny-GenImage dạng folder trên Kaggle.

Hàm/lớp chính:

```python
TinyGenImageKaggleConfig
find_tiny_genimage_root
build_kaggle_tiny_index
build_kaggle_tiny_splits
TinyGenImageKaggleDataset
summarize_index
```

### `transforms.py`

Tạo transform ảnh cho train/eval và robustness test.

Hàm chính:

```python
build_image_transform(...)
```

## 5. Bốn Trường Hợp Train/Test

Dataloader hỗ trợ 4 protocol chính.

### 5.1. `combined`

Train trên tất cả generator trong train split, test trên tất cả generator trong validation split.

```text
train = real + fake từ tất cả generator/train
test  = real + fake từ tất cả generator/val
```

Số lượng expected với Tiny-GenImage full:

```text
train: 14,000 real + 14,000 fake = 28,000
test:   3,500 real +  3,500 fake =  7,000
```

Mục đích: baseline tổng quát khi train/test cùng toàn bộ distribution Tiny-GenImage.

### 5.2. `in_domain`

Train và test trên cùng một generator.

Ví dụ:

```python
eval_case = "in_domain"
generator = "BigGAN"
```

Ý nghĩa:

```text
train = real + fake BigGAN từ train split
test  = real + fake BigGAN từ validation split
```

Số lượng expected:

```text
train: 2,000 real + 2,000 fake = 4,000
test:    500 real +   500 fake = 1,000
```

Mục đích: kiểm tra model học tốt generator đã thấy hay không.

### 5.3. `cross_generator`

Leave-one-generator-out. Train trên các generator còn lại, test trên generator bị giữ lại.

Ví dụ:

```python
eval_case = "cross_generator"
heldout_generator = "GLIDE"
```

Ý nghĩa:

```text
train = real + fake từ tất cả generator trừ GLIDE
test  = real + fake GLIDE
```

Số lượng expected:

```text
train: 12,000 real + 12,000 fake = 24,000
test:     500 real +    500 fake =  1,000
```

Mục đích: đo khả năng generalize sang generator chưa thấy khi train.

### 5.4. `train_one_generator`

Train trên một generator, test trên tất cả generator.

Ví dụ:

```python
eval_case = "train_one_generator"
base_generator = "BigGAN"
```

Ý nghĩa:

```text
train = real + fake BigGAN
test  = real + fake từ tất cả generator validation
```

Số lượng expected:

```text
train: 2,000 real + 2,000 fake = 4,000
test:  3,500 real + 3,500 fake = 7,000
```

Mục đích: stress test xem model có bị phụ thuộc artifact của một generator hay không.

## 6. Cân Bằng Real/Fake

Config:

```python
balance_real=True
```

Khi bật, mỗi split sẽ lấy số ảnh real bằng số ảnh fake:

```text
num_real = num_fake
```

Việc này quan trọng vì nếu class bị lệch, model có thể đạt accuracy cao bằng cách đoán class chiếm đa số. Với bài toán fake image detection, cân bằng real/fake giúp các metric như accuracy, balanced accuracy, precision, recall phản ánh đúng hơn.

Khi `balance_real=False`, loader giữ phân phối gốc của dataset. Chế độ này có thể dùng để kiểm tra model trong điều kiện phân phối tự nhiên, nhưng không nên dùng làm baseline chính nếu dataset lệch class.

## 7. Khác Biệt Giữa Hugging Face Và Kaggle Folder

Bản Hugging Face Tiny-GenImage là dạng flattened. Ảnh real có thể không còn giữ rõ thông tin real thuộc generator folder nào. Vì vậy trong các case như `in_domain`, loader Hugging Face cân bằng real bằng cách lấy từ real pool chung với seed cố định.

Bản Kaggle folder giữ cấu trúc real/fake theo từng generator:

```text
imagenet_ai_0419_biggan/train/nature
imagenet_ai_0419_biggan/train/ai
```

Do đó `in_domain` và `train_one_generator` trên Kaggle folder công bằng hơn, vì real và fake cùng nằm trong folder generator tương ứng.

## 8. Config Quan Trọng

### `TinyGenImageSplitConfig`

Dùng cho Hugging Face loader.

```python
TinyGenImageSplitConfig(
    eval_case="combined",
    generator=None,
    heldout_generator=None,
    base_generator=None,
    train_split="train",
    validation_split="validation",
    cache_dir=None,
    streaming=False,
    balance_real=True,
    seed=42,
    streaming_shuffle_buffer_size=10000,
)
```

Ý nghĩa:

```text
eval_case       chọn protocol: combined, in_domain, cross_generator, train_one_generator
generator       generator dùng cho in_domain
heldout_generator generator bị giữ lại cho cross_generator
base_generator  generator dùng để train trong train_one_generator
train_split     tên split train
validation_split tên split eval/test
cache_dir       nơi cache dataset nếu streaming=False
streaming       True thì đọc lazy từ Hugging Face
balance_real    True thì cân bằng real/fake
seed            seed để shuffle/sampling tái lập được
streaming_shuffle_buffer_size kích thước buffer shuffle khi streaming
```

### `TinyGenImageKaggleConfig`

Dùng cho Kaggle folder loader.

```python
TinyGenImageKaggleConfig(
    dataset_root=None,
    kaggle_input_root="/kaggle/input",
    eval_case="combined",
    generator=None,
    heldout_generator=None,
    base_generator=None,
    train_split="train",
    validation_split="validation",
    balance_real=True,
    seed=42,
    max_train_samples=None,
    max_eval_samples=None,
)
```

Ý nghĩa:

```text
dataset_root     đường dẫn root Tiny-GenImage; None thì tự tìm trong /kaggle/input
kaggle_input_root root input Kaggle
eval_case        protocol thí nghiệm
generator        generator cho in_domain
heldout_generator generator bị giữ lại cho cross_generator
base_generator   generator train chính cho train_one_generator
train_split      mặc định train
validation_split mặc định validation, có alias val/test
balance_real     cân bằng real/fake
seed             seed sampling
max_train_samples giới hạn sample train để debug
max_eval_samples  giới hạn sample eval/test để debug
```

Loader Kaggle hỗ trợ alias:

```text
split train: train, training
split validation: val, valid, validation, test
real folder: nature, real, 0_real, 0-real, 0
fake folder: ai, fake, 1_fake, 1-fake, 1
generator alias: BigGAN, GLIDE, ADM, VQDM, Wukong, Midjourney, SD15/sdv5, SD14/sdv4
```

## 9. Transform Ảnh

File:

```text
data_loader/transforms.py
```

Hàm:

```python
build_image_transform(
    image_size=224,
    train=False,
    normalize=True,
    perturbation="none",
    jpeg_quality=75,
    resize_scale=0.5,
    crop_ratio=0.8,
    blur_radius=1.0,
)
```

Khi `train=True`:

```text
RandomResizedCrop(image_size)
RandomHorizontalFlip()
ToTensor()
Normalize(ImageNet mean/std)
```

Khi `train=False`:

```text
Resize(int(image_size * 1.15))
CenterCrop(image_size)
ToTensor()
Normalize(ImageNet mean/std)
```

Robustness perturbation hỗ trợ:

```text
none          ảnh clean
jpeg          nén JPEG với jpeg_quality
resize        resize xuống rồi phóng lại với resize_scale
center_crop   crop giữa với crop_ratio rồi resize lại
blur          Gaussian blur với blur_radius
```

Trong báo cáo, robustness nên được đánh giá bằng cùng checkpoint clean, chỉ thay transform ở eval.

## 10. Output Của Dataloader

`collate_unified_batch` gom batch thành:

```python
{
    "image": torch.Tensor,       # [B, 3, H, W] nếu transform trả tensor
    "label": torch.LongTensor,   # [B]
    "label_name": list[str],
    "generator": list[str],
    "sample_id": list[str],
    "metadata": list[dict],
}
```

Nếu `task_type="alignment"`, batch có thêm:

```python
target_text = ["real", "fake", ...]
label_token = ["real", "fake", ...]
```

Trường này dùng cho Stage 2 khi cần target dạng text/token cho MLLM.

## 11. Ví Dụ Sử Dụng Hugging Face Loader

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

config = TinyGenImageSplitConfig(
    eval_case="combined",
    streaming=True,
    balance_real=True,
    seed=42,
)
splits = build_tiny_genimage_splits(config)

DatasetClass = TinyGenImageIterableDataset if splits["streaming"] else TinyGenImageDataset

train_dataset = DatasetClass(
    splits["train"],
    split_name=splits["train_split_name"],
    eval_case=splits["eval_case"],
    transform=build_image_transform(image_size=224, train=True),
    task_type="classification",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False if splits["streaming"] else True,
    num_workers=0,
    collate_fn=collate_unified_batch,
)
```

## 12. Ví Dụ Sử Dụng Kaggle Folder Loader

```python
from torch.utils.data import DataLoader
from data_loader import (
    TinyGenImageKaggleConfig,
    TinyGenImageKaggleDataset,
    build_image_transform,
    build_kaggle_tiny_splits,
    collate_unified_batch,
    find_tiny_genimage_root,
)

root = find_tiny_genimage_root()

config = TinyGenImageKaggleConfig(
    dataset_root=str(root),
    eval_case="combined",
    balance_real=True,
    seed=42,
)
splits = build_kaggle_tiny_splits(config)

train_dataset = TinyGenImageKaggleDataset(
    splits["train_df"],
    eval_case=splits["eval_case"],
    transform=build_image_transform(image_size=224, train=True),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_unified_batch,
)
```

## 13. Config Nên Ghi Trong Báo Cáo

Khi báo cáo kết quả, cần ghi rõ:

```text
dataset source: Hugging Face hoặc Kaggle folder
eval_case: combined/in_domain/cross_generator/train_one_generator
generator: nếu dùng in_domain
heldout_generator: nếu dùng cross_generator
base_generator: nếu dùng train_one_generator
balance_real: True/False
seed: ví dụ 42
train/eval sample count
transform train/eval
robustness perturbation nếu có
batch_size
num_workers
model/backbone
metric chính: balanced_accuracy
```

Với baseline có early stopping, nên ghi thêm:

```text
train_inner: phần train dùng để học
val_inner: phần train tách ra để chọn epoch
test: validation split gốc dùng để báo cáo cuối
MAX_EPOCHS
PATIENCE
MIN_DELTA
best_epoch
```

## 14. Ghi Chú Về Tính Công Bằng Thí Nghiệm

Các nguyên tắc đang dùng:

```text
1. Không trộn validation/test vào train.
2. Với early stopping, tách train_inner/val_inner từ train split gốc.
3. Giữ validation split gốc làm test cuối nếu cần báo cáo kết quả.
4. Cân bằng real/fake bằng seed cố định để kết quả tái lập được.
5. Trong cross_generator, generator held-out không xuất hiện ở fake train set.
6. Robustness chỉ áp dụng ở eval, không trộn vào train clean baseline.
```

Điểm cần lưu ý: nếu dùng bản Hugging Face flattened, real images có thể đến từ real pool chung thay vì real theo từng generator folder. Vì vậy các kết quả cần ghi rõ nguồn loader để tránh so sánh lệch với bản Kaggle folder.
