# ResNet-50 Last-Layer Baseline

Thư mục này chứa baseline **ResNet-only** cho bài toán phân loại ảnh `real/fake`.
Baseline này dùng ảnh RGB trực tiếp, không dùng CLIP embedding, không dùng NPR, không dùng FFT/DCT/SRM.

## 1. Mục Tiêu

Mục tiêu của baseline là kiểm tra năng lực của một CNN pretrained thông thường khi chỉ fine-tune tầng phân loại cuối:

```text
RGB image -> ImageNet-pretrained ResNet-50 frozen backbone -> Linear fc -> real/fake
```

Đây là baseline quan trọng để so sánh với:

```text
CLIP + Logistic Regression
  Dùng CLIP frozen để trích embedding, sau đó train classifier tuyến tính.

NPR + ResNet18
  Dùng NPR residual làm input và train ResNet18 từ scratch.

ResNet-50 last layer
  Dùng RGB trực tiếp, backbone ResNet-50 pretrained, chỉ train fc cuối.
```

## 2. File Chính

```text
train_resnet50_kaggle_folder.ipynb
train_resnet50_hf_loader.ipynb
train_resnet50_kaggle_folder.py
train_resnet50_hf_loader.py
```

Trong đó:

```text
train_resnet50_kaggle_folder.ipynb
  Nên dùng khi chạy trên Kaggle với Tiny-GenImage dạng folder.

train_resnet50_hf_loader.ipynb
  Dùng loader Hugging Face Tiny-GenImage.
```

## 3. Kỹ Thuật Sử Dụng

### 3.1 ImageNet Pretrained Backbone

Model được lấy từ `torchvision`:

```python
from torchvision.models import ResNet50_Weights, resnet50

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
```

Ý nghĩa:

```text
Backbone ResNet-50 đã học đặc trưng thị giác tổng quát từ ImageNet.
Baseline này kiểm tra xem đặc trưng ImageNet có đủ để tách real/fake hay không.
```

### 3.2 Freeze Backbone

Toàn bộ tham số của ResNet-50 được đóng băng:

```python
for param in model.parameters():
    param.requires_grad = False
```

Chỉ tầng cuối được train:

```python
model.fc = nn.Linear(model.fc.in_features, 2)
```

Ý nghĩa:

```text
Không fine-tune toàn bộ CNN.
Không cập nhật conv/bn/layer1/layer2/layer3/layer4.
Chỉ học một decision boundary tuyến tính trên feature cuối của ResNet-50.
```

Vì vậy baseline này còn có thể gọi là:

```text
ResNet-50 frozen backbone + linear classifier
```

### 3.3 Input Và Transform

Input là ảnh RGB sau transform chuẩn ImageNet:

```text
resize/crop -> tensor -> normalize bằng mean/std ImageNet
```

Lý do dùng chuẩn ImageNet:

```text
ResNet-50 pretrained được train với normalization ImageNet.
Nếu normalize sai phân phối, feature backbone có thể bị lệch.
```

Khác với NPR:

```text
ResNet-50 baseline nhận ảnh RGB trực tiếp.
NPR-ResNet18 nhận residual map sau NPR.
```

### 3.4 Loss Function

Loss dùng cho binary classification hai class:

```python
criterion = nn.CrossEntropyLoss()
```

Label mapping:

```text
real = 0
fake = 1
```

Model output có shape:

```text
[batch_size, 2]
```

Hai logit tương ứng với `real` và `fake`.

### 3.5 Optimizer

Optimizer:

```python
optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
```

Chỉ truyền `model.fc.parameters()` vào optimizer, nên chỉ tầng `fc` được update.

### 3.6 Mixed Precision

Nếu có GPU CUDA, code bật AMP:

```python
USE_AMP = torch.cuda.is_available()
```

Trong train/eval:

```python
torch.cuda.amp.autocast(enabled=USE_AMP)
torch.cuda.amp.GradScaler(enabled=USE_AMP)
```

Ý nghĩa:

```text
Giảm VRAM.
Tăng tốc trên GPU.
Không dùng khi chỉ chạy CPU.
```

## 4. Config Hiện Tại

Config mặc định trong script/notebook:

```python
RUN_ALL_CASES = False
SELECTED_EXPERIMENT = "combined"
BALANCE_REAL = True
RANDOM_SEED = 42

MAX_TRAIN_SAMPLES = 500
MAX_TEST_SAMPLES = 300
VAL_FRACTION = 0.2

BATCH_SIZE = 16
NUM_WORKERS = 0
MAX_EPOCHS = 20
PATIENCE = 3
MIN_DELTA = 1e-3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SAVE_PREDICTIONS = True
```

Ý nghĩa các giá trị chính:

```text
MAX_TRAIN_SAMPLES = 500
  Giới hạn số ảnh train để tránh tràn RAM/VRAM khi test nhanh.

MAX_TEST_SAMPLES = 300
  Giới hạn số ảnh test để chạy nhanh.

Nếu muốn chạy full dataset:
  Đổi MAX_TRAIN_SAMPLES = None
  Đổi MAX_TEST_SAMPLES = None

BATCH_SIZE = 16
  Nhẹ hơn batch size 32, phù hợp khi GPU/RAM hạn chế.

MAX_EPOCHS = 20
  Số epoch tối đa.

PATIENCE = 3
  Dừng sớm nếu validation balanced_accuracy không cải thiện.

LEARNING_RATE = 1e-3
  LR cho tầng fc cuối. Vì chỉ train fc nên có thể dùng LR lớn hơn NPR train from scratch.

WEIGHT_DECAY = 1e-4
  Regularization cho linear classifier.
```

## 5. Split Dữ Liệu

### 5.1 Kaggle Folder Loader

Dataset Kaggle có cấu trúc:

```text
tiny-genimage/
  imagenet_ai_0419_biggan/
    train/
      ai/
      nature/
    val/
      ai/
      nature/
```

Mapping:

```text
nature -> real -> label 0
ai     -> fake -> label 1
```

Quy trình split:

```text
train gốc -> train_inner + val_inner
val gốc   -> test cuối
```

Trong đó:

```text
train_inner
  Dùng để update model.

val_inner
  Dùng để chọn best epoch và early stopping.

test cuối
  Dùng để báo cáo metric cuối cùng.
```

### 5.2 Hugging Face Loader

Bản Hugging Face dùng cùng logic split từ `data_loader`.
Nếu `STREAMING=True`, dữ liệu được đọc dạng streaming từ Hugging Face cache/remote interface, nhưng batch ảnh vẫn được tải khi cần dùng.

## 6. Evaluation Cases

Code hỗ trợ các case:

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
  Train trên tất cả generator, test trên tất cả generator.

in_domain
  Train/test cùng một generator.

cross_generator
  Train trên tất cả generator trừ heldout_generator, test trên generator bị heldout.

train_one_generator
  Train trên một generator, test trên tất cả generator.
```

## 7. Early Stopping

Metric dùng để chọn best model:

```text
validation balanced_accuracy
```

Điều kiện cải thiện:

```text
current_val_balanced_accuracy > best_val_balanced_accuracy + MIN_DELTA
```

Nếu không cải thiện trong `PATIENCE` epoch liên tiếp:

```text
early stopping
```

Best checkpoint được lưu:

```text
checkpoints/best_resnet50_fc.pt
```

## 8. Metric Báo Cáo

Các metric chính:

```text
accuracy
balanced_accuracy
precision
recall
f1
roc_auc
average_precision
confusion_matrix
best_epoch
best_val_balanced_accuracy
```

Vì class positive là:

```text
fake = 1
```

nên:

```text
precision, recall, f1 là metric cho class fake.
```

Ngoài overall metric, code còn lưu metric theo từng generator để xem model yếu/mạnh trên generator nào.

## 9. Output

Kaggle folder version lưu tại:

```text
outputs/resnet50_last_layer_kaggle_folder/
```

Hugging Face version lưu tại:

```text
outputs/resnet50_last_layer_hf/
```

Các file thường có:

```text
checkpoints/best_resnet50_fc.pt
metrics/history.csv
metrics/overall_metrics.json
metrics/generator_metrics.csv
predictions/predictions.csv
summary_metrics_<run_id>.csv
```

## 10. Ghi Chú Báo Cáo

Có thể mô tả baseline này trong báo cáo như sau:

```text
We implement a ResNet-50 last-layer baseline for real/fake image classification. The model uses an ImageNet-pretrained ResNet-50 as a frozen feature extractor. All convolutional and batch-normalization layers are frozen, and only the final fully connected layer is replaced and trained for binary classification. The input is the original RGB image normalized with ImageNet statistics. This baseline evaluates whether generic ImageNet visual representations are sufficient for detecting AI-generated images without using explicit forensic residuals or multimodal features.
```

Điểm cần nhấn mạnh:

```text
1. Đây là RGB-only CNN baseline.
2. Dùng ResNet-50 pretrained ImageNet.
3. Freeze toàn bộ backbone.
4. Chỉ train tầng fully connected cuối.
5. Không dùng CLIP embedding.
6. Không dùng NPR/frequency/SRM residual.
7. Split và metric giữ giống các baseline khác để so sánh công bằng.
```
