# Baseline CLIP + Logistic Regression

Tài liệu này mô tả baseline `CLIP + Logistic Regression` dùng cho bài toán phân loại ảnh thật/giả. Đây là baseline quan trọng vì nó kiểm tra xem đặc trưng ảnh tổng quát từ CLIP có đủ phân biệt ảnh real/fake hay không, trước khi chuyển sang các mô hình CNN forensic hoặc MLLM giải thích.

## 1. Ý Tưởng Chính

Pipeline:

```text
image -> frozen CLIP image encoder -> normalized image embedding -> linear classifier -> real/fake
```

Trong baseline này, CLIP chỉ đóng vai trò feature extractor. Toàn bộ CLIP được freeze, không cập nhật trọng số. Phần được train chỉ là classifier tuyến tính ở cuối.

Mục tiêu:

```text
Input: ảnh RGB
Output: xác suất ảnh thuộc class fake
Label: real = 0, fake = 1
```

## 2. Vì Sao Dùng CLIP

CLIP là vision-language model được pretrain trên lượng lớn cặp ảnh-văn bản. Image encoder của CLIP thường học được đặc trưng ngữ nghĩa và đặc trưng thị giác tổng quát. Khi freeze CLIP và chỉ train một classifier tuyến tính, ta có thể đánh giá nhanh:

```text
1. Đặc trưng CLIP có tách được real/fake không.
2. Bài toán có thể giải bằng feature tổng quát hay cần forensic feature chuyên biệt.
3. Model có generalize sang generator chưa thấy không.
```

Baseline này không nhằm đạt SOTA, mà làm mốc so sánh cho các phương pháp sau.

## 3. File Chính

```text
baselines/clip_linear_probe/
  train_clip_linear_probe_loader.ipynb
  train_clip_linear_probe_kaggle_folder.ipynb
  README.md
```

Ý nghĩa:

```text
train_clip_linear_probe_loader.ipynb
  Dùng Hugging Face Tiny-GenImage loader.
  Extract toàn bộ CLIP embedding ra numpy.
  Train sklearn LogisticRegression.

train_clip_linear_probe_kaggle_folder.ipynb
  Dùng Kaggle folder Tiny-GenImage loader.
  Đọc ảnh lazy từ /kaggle/input.
  Encode CLIP theo mini-batch.
  Train linear head bằng PyTorch để giảm RAM.
```

## 4. Hai Cách Triển Khai

### 4.1. Hugging Face + sklearn Logistic Regression

Notebook:

```text
train_clip_linear_probe_loader.ipynb
```

Flow:

```text
DataLoader -> CLIP image encoder -> collect all embeddings -> StandardScaler -> LogisticRegression
```

Classifier:

```python
LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
```

Trước khi Logistic Regression, embedding được chuẩn hóa bằng `StandardScaler`. Việc này giúp các chiều feature có scale ổn định hơn cho mô hình tuyến tính.

Ưu điểm:

```text
1. Đúng nghĩa Logistic Regression cổ điển.
2. Dễ phân tích, dễ lưu model.
3. Train nhanh nếu embedding đã được extract.
```

Nhược điểm:

```text
1. Phải giữ embedding train/eval trong RAM.
2. Có thể gây tràn RAM trên Kaggle khi chạy full data.
```

### 4.2. Kaggle Folder + PyTorch Linear Head

Notebook:

```text
train_clip_linear_probe_kaggle_folder.ipynb
```

Flow:

```text
DataLoader -> CLIP image encoder -> linear layer -> CrossEntropyLoss
```

Classifier:

```python
class ClipLinearHead(nn.Module):
    def __init__(self, in_dim=512, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
```

Loss:

```python
nn.CrossEntropyLoss()
```

Optimizer:

```python
torch.optim.AdamW(
    head.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)
```

Về bản chất, đây vẫn là linear probe trên frozen CLIP embedding. Khác biệt là không gom toàn bộ embedding vào RAM; mỗi batch được encode rồi train head ngay.

Ưu điểm:

```text
1. Ít tốn RAM hơn sklearn Logistic Regression.
2. Phù hợp Kaggle khi dataset nằm trong /kaggle/input.
3. Có thể chạy với batch nhỏ và giới hạn sample để debug.
```

Nhược điểm:

```text
1. Không phải sklearn LogisticRegression cổ điển.
2. Cần chọn số epoch hoặc dùng early stopping nếu mở rộng.
```

## 5. CLIP Encoder

Thư viện:

```python
open_clip_torch
```

Config mặc định:

```python
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
```

Cách load:

```python
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME,
    pretrained=CLIP_PRETRAINED,
    device=DEVICE,
)
```

Freeze CLIP:

```python
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False
```

Encode ảnh:

```python
features = clip_model.encode_image(images)
features = features / features.norm(dim=-1, keepdim=True)
```

Việc L2-normalize embedding giúp vector feature có cùng độ dài, làm classifier tuyến tính ổn định hơn.

## 6. Data Loader

Baseline dùng dataloader chung của dự án:

```python
from data_loader import (
    TinyGenImageDataset,
    TinyGenImageIterableDataset,
    TinyGenImageKaggleDataset,
    build_tiny_genimage_splits,
    build_kaggle_tiny_splits,
    collate_unified_batch,
)
```

Batch trả về:

```python
{
    "image": torch.Tensor,
    "label": torch.LongTensor,
    "label_name": list[str],
    "generator": list[str],
    "sample_id": list[str],
    "metadata": list[dict],
}
```

Với CLIP, transform không dùng `build_image_transform` của ImageNet mà dùng trực tiếp:

```python
clip_preprocess
```

Lý do: CLIP có preprocess riêng gồm resize, crop, convert tensor và normalize theo thống kê mà CLIP đã dùng khi pretrain.

## 7. Protocol Thí Nghiệm

Notebook hỗ trợ 4 case:

```python
EXPERIMENT_CONFIGS = [
    {"name": "combined", "eval_case": "combined"},
    {"name": "in_domain_biggan", "eval_case": "in_domain", "generator": "BigGAN"},
    {"name": "cross_generator_glide", "eval_case": "cross_generator", "heldout_generator": "GLIDE"},
    {"name": "train_one_generator_biggan", "eval_case": "train_one_generator", "base_generator": "BigGAN"},
]
```

Ý nghĩa:

```text
combined
  Train trên tất cả generator/train.
  Test trên tất cả generator/validation.

in_domain
  Train và test trên cùng một generator.
  Ví dụ BigGAN train -> BigGAN test.

cross_generator
  Train trên các generator trừ heldout generator.
  Test trên heldout generator.

train_one_generator
  Train trên một generator.
  Test trên tất cả generator.
```

Các case này dùng để phân biệt:

```text
in-domain performance
cross-generator generalization
generator-specific bias
overall Tiny-GenImage performance
```

## 8. Config Quan Trọng

### Config chung

```python
BALANCE_REAL = True
RANDOM_SEED = 42
BATCH_SIZE = 64       # bản Hugging Face
BATCH_SIZE = 16       # bản Kaggle-folder mặc định
NUM_WORKERS = 0
```

Ý nghĩa:

```text
BALANCE_REAL
  Cân bằng số ảnh real và fake trong mỗi split.

RANDOM_SEED
  Đảm bảo sampling/shuffle tái lập được.

BATCH_SIZE
  Số ảnh xử lý mỗi batch.
  Tăng batch size giúp chạy nhanh hơn nhưng tốn GPU/RAM hơn.

NUM_WORKERS
  Số worker đọc ảnh.
  Trên Kaggle nên để 0 nếu RAM yếu.
```

### Config giới hạn dữ liệu

```python
MAX_TRAIN_SAMPLES = 500
MAX_EVAL_SAMPLES = 300
```

Dùng để debug nhanh. Khi chạy full baseline:

```python
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
```

### Config Hugging Face

```python
STREAMING = True
CACHE_DIR = None
```

Ý nghĩa:

```text
STREAMING=True
  Đọc lazy từ Hugging Face, không tải full dataset trước.

STREAMING=False
  Tải/cache dataset về máy để random access tốt hơn.

CACHE_DIR
  Chỉ dùng khi muốn chỉ định nơi cache dataset.
```

### Config Kaggle Folder

```python
DATASET_ROOT = None
```

Nếu `None`, loader tự tìm Tiny-GenImage trong:

```text
/kaggle/input
```

Nếu auto-detect sai, set thủ công:

```python
DATASET_ROOT = "/kaggle/input/datasets/yangsangtai/tiny-genimage"
```

### Config Linear Head Kaggle

```python
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
```

Ý nghĩa:

```text
NUM_EPOCHS
  Số epoch train linear head.

LEARNING_RATE
  Tốc độ cập nhật trọng số.

WEIGHT_DECAY
  Regularization L2 để giảm overfit.
```

## 9. Metric

Các metric được tính:

```text
accuracy
balanced_accuracy
precision
recall
f1
roc_auc
average_precision
confusion_matrix
```

Vì label quy ước:

```text
real = 0
fake = 1
```

nên với `precision_score`, `recall_score`, `f1_score` mặc định của sklearn, các metric này đang tính cho class positive là:

```text
fake
```

Diễn giải:

```text
precision_fake
  Trong các ảnh model dự đoán fake, bao nhiêu ảnh thật sự fake.

recall_fake
  Trong tất cả ảnh fake thật, model phát hiện được bao nhiêu ảnh.

f1_fake
  Trung bình điều hòa giữa precision_fake và recall_fake.

balanced_accuracy
  Trung bình giữa recall_real và recall_fake.
```

Metric nên dùng làm chính:

```text
balanced_accuracy
```

Lý do: nó công bằng hơn accuracy khi số lượng real/fake hoặc số lượng theo generator bị lệch.

## 10. Output

Mỗi experiment lưu output riêng:

```text
outputs/clip_linear_probe/<experiment_name>/<run_id>/
```

hoặc với bản Kaggle:

```text
outputs/clip_linear_probe_kaggle_folder/<experiment_name>/<run_id>/
```

Các file chính:

```text
checkpoints/
predictions/predictions.csv
metrics/overall_metrics.json
metrics/generator_metrics.csv
plots/roc_curve.png
plots/confusion_matrix.png
```

Ngoài ra có summary:

```text
summary_metrics_<run_id>.csv
```

## 11. Điểm Mạnh Và Hạn Chế

Điểm mạnh:

```text
1. Dễ chạy, dễ tái lập.
2. CLIP đã có visual representation mạnh.
3. Chỉ train classifier tuyến tính nên ít tham số.
4. Phù hợp làm baseline so sánh với ResNet-50, CNNSpot, NPR hoặc MLLM.
```

Hạn chế:

```text
1. CLIP không được pretrain riêng cho forensic artifact.
2. Model có thể dựa vào semantic/context thay vì dấu vết tạo ảnh.
3. Bản sklearn Logistic Regression tốn RAM khi full embedding lớn.
4. Cross-generator có thể giảm mạnh nếu artifact giữa generator khác nhau.
```

## 12. Cách Viết Trong Báo Cáo

Có thể mô tả ngắn gọn như sau:

```text
We use a frozen CLIP ViT-B/32 image encoder pretrained by OpenAI as a generic visual feature extractor. Each image is processed using CLIP's official preprocessing pipeline and encoded into a normalized image embedding. A linear classifier is then trained on top of the frozen features to predict whether the image is real or AI-generated. In the Hugging Face implementation, all CLIP embeddings are extracted first and a scikit-learn Logistic Regression classifier is fitted. In the Kaggle implementation, to reduce memory usage, the same linear-probe idea is implemented as a PyTorch linear head trained mini-batch-wise on frozen CLIP features.
```

Thông tin nên ghi cùng kết quả:

```text
CLIP model: ViT-B-32
pretrained weights: openai
classifier: LogisticRegression hoặc PyTorch LinearHead
CLIP frozen: yes
feature normalization: L2 normalization
class labels: real=0, fake=1
main metric: balanced_accuracy
eval protocol: combined/in_domain/cross_generator/train_one_generator
balance_real: True
seed: 42
batch_size
max_train_samples/max_eval_samples
```
