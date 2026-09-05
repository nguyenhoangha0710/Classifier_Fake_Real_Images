# NPR + ResNet18 From Scratch

Thư mục này chứa baseline **NPR-ResNet18** cho bài toán phân loại ảnh `real/fake`.
Baseline này dùng Neighboring Pixel Relationships (NPR) để tạo residual map trước khi đưa vào ResNet18.

Khác với ResNet-50 baseline, model này **không dùng pretrained weights** và được train từ đầu.

## 1. Mục Tiêu

Mục tiêu của baseline là kiểm tra xem các local pixel artifacts có giúp phát hiện ảnh sinh bởi AI tốt hơn ảnh RGB trực tiếp hay không.

Pipeline tổng quát:

```text
RGB image -> NPR residual map -> ResNet18(weights=None) -> real/fake
```

So sánh với các baseline khác:

```text
CLIP + Logistic Regression
  CLIP frozen, trích embedding, chỉ train classifier tuyến tính.

ResNet-50 last layer
  RGB image, ResNet-50 pretrained ImageNet, freeze backbone, chỉ train fc cuối.

NPR + ResNet18
  RGB image được chuyển thành NPR residual, ResNet18 khởi tạo ngẫu nhiên, train từ scratch.
```

## 2. Ý Tưởng NPR

NPR là viết tắt của **Neighboring Pixel Relationships**.

Ý tưởng chính:

```text
Ảnh fake thường để lại sai khác cục bộ ở quan hệ giữa các pixel lân cận.
Các artifact này có thể đến từ upsampling, denoising, reconstruction hoặc generator pipeline.
NPR cố làm nổi bật các local artifacts đó thay vì nhìn ảnh RGB như ảnh tự nhiên thông thường.
```

NPR được tính bằng cách lấy ảnh gốc trừ đi ảnh đã downsample rồi upsample lại bằng nearest neighbor:

```python
x_down = F.interpolate(x, scale_factor=0.5, mode="nearest")
x_up = F.interpolate(x_down, size=x.shape[-2:], mode="nearest")
npr = x - x_up
```

Trong code hiện tại:

```python
npr = (x - x_up) * (2.0 / 3.0)
```

Ý nghĩa:

```text
x_up là phiên bản tái tạo thô của ảnh.
x - x_up giữ lại phần residual cục bộ.
Residual này nhấn mạnh biên, texture nhỏ và bất thường pixel-level.
```

Nếu chiều cao hoặc chiều rộng là số lẻ, tensor được crop bớt 1 pixel để downsample/upsample không lệch shape:

```python
if height % 2 == 1:
    x = x[:, :, :-1, :]
if width % 2 == 1:
    x = x[:, :, :, :-1]
```

## 3. Kiến Trúc Model

File model/training chung:

```text
baselines/npr_resnet18/npr_resnet18.py
```

Class chính:

```python
NPRLayer
NPRResNet18
```

Kiến trúc trong notebook và module:

```python
class NPRResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
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
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        residual = self.npr(x)
        return self.backbone(residual)
```

Điểm quan trọng:

```text
weights=None
  Không dùng ImageNet pretrained weights.

NPRLayer
  Không có tham số học, chỉ biến đổi input thành residual map.

ResNet18
  Toàn bộ conv/bn/layer1/layer2/layer3/layer4/fc đều được train từ scratch.
```

## 4. Forensic-Friendly Stem

ResNet18 gốc của ImageNet dùng stem:

```text
conv1: 7x7, stride=2
maxpool: 3x3, stride=2
```

Với NPR residual, tín hiệu quan trọng thường nằm ở local artifacts rất nhỏ.
Nếu giảm resolution quá sớm, model có thể làm mất dấu vết forensic.

Vì vậy baseline này đổi stem thành:

```text
conv1: 3x3, stride=1, padding=1
maxpool: Identity
```

Code:

```python
self.backbone.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
)
self.backbone.maxpool = nn.Identity()
```

Ý nghĩa:

```text
3x3 kernel giữ quan hệ pixel cục bộ tốt hơn 7x7 ở tầng đầu.
stride=1 không giảm kích thước feature map quá sớm.
bỏ maxpool giúp giữ texture/artifact nhỏ sau NPR.
```

Trong báo cáo nên gọi rõ đây là:

```text
NPR-ResNet18 with a forensic-friendly 3x3 stem
```

không phải ResNet18 vanilla hoàn toàn.

## 5. Input Và Transform

Input ban đầu vẫn là ảnh RGB từ dataloader.
Sau transform ảnh được đưa vào model dưới dạng tensor:

```text
[batch_size, 3, H, W]
```

Sau đó NPRLayer tạo residual map có cùng số channel:

```text
[batch_size, 3, H, W]
```

Điểm khác biệt với ResNet-only:

```text
ResNet-50 baseline
  Model nhận RGB image trực tiếp.

NPR-ResNet18
  Model nhận NPR residual map được tính online trong forward pass.
```

NPR được tính trong model, không cần lưu ảnh residual ra disk.

## 6. Loss Function

Loss dùng cho phân loại hai class:

```python
criterion = nn.CrossEntropyLoss()
```

Label mapping:

```text
real = 0
fake = 1
```

Output của model:

```text
[batch_size, 2]
```

Hai logit tương ứng:

```text
logit 0 -> real
logit 1 -> fake
```

## 7. Optimizer Và Training

Optimizer:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
```

Khác với ResNet-50 last-layer baseline:

```text
ResNet-50 last layer
  Chỉ optimizer trên model.fc.parameters().

NPR-ResNet18
  Optimizer trên toàn bộ model.parameters().
```

Lý do:

```text
NPR-ResNet18 train from scratch nên tất cả tham số ResNet18 cần được học.
```

## 8. Mixed Precision

Nếu có GPU CUDA, code bật AMP:

```python
USE_AMP = torch.cuda.is_available()
```

Trong train/eval:

```python
torch.amp.autocast(device_type="cuda", enabled=USE_AMP)
torch.amp.GradScaler("cuda", enabled=USE_AMP)
```

Ý nghĩa:

```text
Giảm VRAM.
Tăng tốc trên GPU.
Không dùng khi chạy CPU.
```

## 9. Config Chuẩn

Config được đặt để so sánh tương đối công bằng với các baseline cùng giai đoạn:

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
SAVE_PREDICTIONS = True
```

Ý nghĩa:

```text
MAX_TRAIN_SAMPLES/MAX_TEST_SAMPLES = None
  Chạy full dataset.

BATCH_SIZE = 32
  Số ảnh mỗi batch.

MAX_EPOCHS = 10
  Số epoch tối đa.

PATIENCE = 3
  Dừng sớm nếu validation balanced_accuracy không cải thiện trong 3 epoch liên tiếp.

LEARNING_RATE = 2e-4
  Nhỏ hơn ResNet-50 last-layer vì model train từ scratch, cập nhật nhiều tham số hơn.

WEIGHT_DECAY = 1e-4
  Regularization để giảm overfit.

BALANCE_REAL = True
  Giữ số lượng real/fake cân bằng trong split.
```

## 10. Split Dữ Liệu

### 10.1 Kaggle Folder Loader

Bản Kaggle đọc Tiny-GenImage theo cấu trúc:

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

Split dùng trong training:

```text
train gốc -> train_inner + val_inner
val gốc   -> test cuối
```

Trong đó:

```text
train_inner
  Dùng để update weight.

val_inner
  Dùng để chọn best epoch và early stopping.

test cuối
  Chỉ dùng sau khi đã chọn best model.
```

### 10.2 Hugging Face Loader

Bản Hugging Face dùng cùng `TinyGenImageSplitConfig` và `build_tiny_genimage_splits`.

Nếu `STREAMING=True`:

```text
Dữ liệu được đọc theo streaming interface.
Không cần tải thủ công toàn bộ dataset trước.
Ảnh vẫn được fetch/cache khi batch được đọc.
```

## 11. Evaluation Cases

Notebook hỗ trợ:

```python
EXPERIMENT_CONFIGS = [
    {"name": "combined", "eval_case": "combined"},
    {"name": "in_domain_biggan", "eval_case": "in_domain", "generator": "BigGAN"},
    {"name": "cross_generator_glide", "eval_case": "cross_generator", "heldout_generator": "GLIDE"},
    {"name": "cross_generator_wukong", "eval_case": "cross_generator", "heldout_generator": "Wukong"},
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

## 12. Early Stopping

Metric monitor:

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

Best checkpoint:

```text
checkpoints/best_npr_resnet18.pt
```

## 13. Metric Báo Cáo

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

Vì class positive là:

```text
fake = 1
```

nên:

```text
precision, recall, f1 là metric cho class fake.
```

Ngoài metric tổng, code còn lưu metric theo từng generator.

## 14. Output

Bản Kaggle lưu ở:

```text
outputs/npr_resnet18_kaggle_folder/<experiment_name>/<run_id>/
```

Bản Hugging Face lưu ở:

```text
outputs/npr_resnet18_hf/<experiment_name>/<run_id>/
```

Các file chính:

```text
checkpoints/best_npr_resnet18.pt
metrics/history.csv
metrics/overall_metrics.json
metrics/generator_metrics.csv
predictions/predictions.csv
summary_metrics_<run_id>.csv
```

## 15. Ghi Chú Báo Cáo

Có thể mô tả baseline này trong báo cáo như sau:

```text
We implement an NPR-based forensic baseline using a randomly initialized ResNet18. Instead of feeding the RGB image directly, we first compute the Neighboring Pixel Relationship residual by subtracting a nearest-neighbor downsample-upsample reconstruction from the original image. The residual map emphasizes local pixel-level inconsistencies and upsampling artifacts. The resulting NPR map is passed to a ResNet18 classifier trained from scratch for binary real/fake prediction. To preserve local forensic traces, the standard ImageNet stem is replaced by a 3x3 stride-1 convolution and the initial max-pooling layer is removed. No ImageNet pretrained weights are used.
```

Điểm cần nhấn mạnh:

```text
1. Đây là forensic-oriented CNN baseline.
2. Input của classifier là NPR residual, không phải RGB trực tiếp.
3. NPR residual được tính online trong forward pass, không lưu ra disk.
4. Không dùng pretrained weights.
5. Toàn bộ ResNet18 được train từ scratch.
6. Stem dùng 3x3 stride=1 và không dùng maxpool để giữ local artifacts.
7. Optimizer cập nhật toàn bộ tham số ResNet18.
8. Split, label mapping và metric giữ thống nhất với các baseline khác.
```
