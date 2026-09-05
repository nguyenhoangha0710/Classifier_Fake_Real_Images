# NPR + ResNet18 From Scratch

Baseline này dùng Neighboring Pixel Relationships (NPR) làm input forensic và train ResNet18 từ đầu, không dùng ImageNet pretrained weights.

## 1. Ý Tưởng

Pipeline:

```text
image -> NPR residual map -> ResNet18(weights=None) -> real/fake
```

Khác với CLIP và ResNet-50 baseline:

```text
CLIP + Logistic Regression
  CLIP frozen, chỉ train classifier tuyến tính.

ResNet-50 last layer
  ResNet-50 pretrained ImageNet, freeze backbone, chỉ train fc cuối.

NPR + ResNet18
  Không dùng pretrained weights.
  Train toàn bộ ResNet18 từ scratch.
  Input không phải RGB trực tiếp mà là NPR residual.
```

## 2. NPR Là Gì

NPR là viết tắt của Neighboring Pixel Relationships. Ý tưởng là ảnh sinh bởi GAN/diffusion thường để lại artifact cục bộ do quá trình upsampling hoặc reconstruction. NPR làm nổi bật sai khác cục bộ bằng cách so sánh ảnh gốc với ảnh được downsample rồi upsample lại.

Công thức:

```python
x_down = interpolate(x, scale_factor=0.5, mode="nearest")
x_up = interpolate(x_down, size=x.shape[-2:], mode="nearest")
npr = x - x_up
```

Trong code:

```python
npr = (x - x_up) * (2.0 / 3.0)
```

Nếu chiều cao hoặc chiều rộng là số lẻ, tensor được crop bớt 1 pixel để downsample/upsample không lệch kích thước.

## 3. Model

File model/training chung:

```text
baselines/npr_resnet18/npr_resnet18.py
```

Class chính:

```python
NPRLayer
NPRResNet18
```

Kiến trúc:

```python
self.npr = NPRLayer()
self.backbone = resnet18(weights=None)
self.backbone.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
)
self.backbone.maxpool = nn.Identity()
self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
```

Toàn bộ model được train:

```text
NPR layer: không có tham số học
ResNet18 conv/bn/layer1-4/fc: train từ đầu
```

Lưu ý về stem:

```text
ResNet18 gốc dùng conv1 7x7 stride=2 và maxpool.
Baseline NPR-ResNet18 đổi stem thành conv1 3x3 stride=1 và bỏ maxpool.
Lý do là NPR nhấn mạnh local pixel relationships/artifacts, nên không giảm resolution quá sớm ở tầng đầu.
```

## 4. Notebook

```text
train_npr_resnet18_kaggle_folder.ipynb
train_npr_resnet18_hf_loader.ipynb
```

Bản nên dùng cho thí nghiệm chính với dataset Kaggle hiện tại:

```text
train_npr_resnet18_kaggle_folder.ipynb
```

Vì bản Kaggle folder giữ rõ cấu trúc:

```text
generator/train/ai
generator/train/nature
generator/val/ai
generator/val/nature
```

## 5. Config Chuẩn Để So Sánh

Các giá trị được set giống nhau giữa hai notebook:

```python
MAX_TRAIN_SAMPLES = None
MAX_TEST_SAMPLES = None
BATCH_SIZE = 32
MAX_EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
```

Các config khác:

```python
BALANCE_REAL = True
RANDOM_SEED = 42
VAL_FRACTION = 0.2
NUM_WORKERS = 0
MIN_DELTA = 1e-3
```

Ý nghĩa:

```text
MAX_TRAIN_SAMPLES/MAX_TEST_SAMPLES = None
  Chạy full dữ liệu.

BATCH_SIZE = 32
  Số ảnh mỗi batch, giữ giống các thí nghiệm so sánh.

MAX_EPOCHS = 10
  Train tối đa 10 epoch.

PATIENCE = 3
  Dừng sớm nếu validation balanced_accuracy không cải thiện 3 epoch liên tiếp.

LEARNING_RATE = 2e-4
  Learning rate nhỏ hơn ResNet pretrained vì model train từ scratch.

WEIGHT_DECAY = 1e-4
  Regularization để giảm overfit.
```

## 6. Split Và Early Stopping

Bản Kaggle folder dùng split sạch:

```text
train gốc -> train_inner + val_inner
val gốc   -> test cuối
```

Trong đó:

```text
train_inner
  Dùng để train model.

val_inner
  Dùng để chọn best epoch và early stopping.

test
  Dùng để báo cáo metric cuối cùng.
```

Metric monitor:

```text
balanced_accuracy
```

Điều kiện dừng:

```text
Nếu balanced_accuracy trên val_inner không tăng ít nhất MIN_DELTA trong PATIENCE epoch liên tiếp thì stop.
```

## 7. Evaluation Cases

Notebook hỗ trợ:

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
  Train tất cả generator, test tất cả generator.

in_domain
  Train/test cùng một generator.

cross_generator
  Train tất cả generator trừ heldout, test heldout.

train_one_generator
  Train một generator, test tất cả generator.
```

## 8. Metric

Các metric lưu lại:

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

Với label:

```text
real = 0
fake = 1
```

`precision`, `recall`, `f1` là metric của class positive `fake`.

## 9. Output

Bản Kaggle lưu ở:

```text
outputs/npr_resnet18_kaggle_folder/<experiment_name>/<run_id>/
```

Bản Hugging Face lưu ở:

```text
outputs/npr_resnet18_hf/<experiment_name>/<run_id>/
```

Các file:

```text
checkpoints/best_npr_resnet18.pt
metrics/history.csv
metrics/overall_metrics.json
metrics/generator_metrics.csv
predictions/predictions.csv
```

## 10. Ghi Chú Báo Cáo

Có thể mô tả trong báo cáo:

```text
We implement an NPR-based forensic baseline using a randomly initialized ResNet18. Instead of feeding the RGB image directly, we first compute the Neighboring Pixel Relationship residual by subtracting a nearest-neighbor downsample-upsample reconstruction from the original image. The residual map emphasizes local pixel-level inconsistencies and upsampling artifacts. The resulting NPR map is passed to a ResNet18 classifier trained from scratch for binary real/fake prediction. To preserve local forensic traces, the standard ImageNet stem is replaced by a 3x3 stride-1 convolution and the initial max-pooling layer is removed. No ImageNet pretrained weights are used.
```

Điểm cần nhấn mạnh:

```text
1. Đây là forensic-oriented baseline.
2. Không dùng pretrained weights.
3. Toàn bộ ResNet18 được train từ scratch.
4. Input là NPR residual, không phải RGB trực tiếp.
5. Stem dùng 3x3 stride=1 và không dùng maxpool để giữ local artifacts.
6. So sánh công bằng bằng cùng split, batch size, epoch, patience, learning rate và weight decay.
```
