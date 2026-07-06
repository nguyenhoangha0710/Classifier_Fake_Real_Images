# Kế Hoạch Phân Tích Dữ Liệu Và Train GenImage

File này ghi lại kế hoạch làm việc sau khi đã đọc được dataset GenImage trên Kaggle. Mục tiêu là đi từ kiểm tra dữ liệu, phân tích dữ liệu, feature engineering, baseline đơn giản, rồi mới train deep learning model.

## 1. Trạng Thái Hiện Tại

Dataset đã đọc thành công trên Kaggle.

Tổng số ảnh quét được:

```text
35.000 ảnh
```

Các subset hiện có:

```text
imagenet_ai_0419_biggan
imagenet_ai_0419_vqdm
imagenet_ai_0424_sdv5
imagenet_ai_0424_wukong
imagenet_ai_0508_adm
imagenet_glide
imagenet_midjourney
```

Mỗi subset có:

```text
train/ai       2000 ảnh
train/nature   2000 ảnh
val/ai          500 ảnh
val/nature      500 ảnh
```

Quy ước label:

```text
nature = 0
ai     = 1
```

Các biến chính trong notebook:

```python
DATASET_ROOT  # nơi ảnh gốc nằm trên Kaggle
df            # bảng chứa toàn bộ path ảnh đã quét được
work_df       # bảng sau khi chọn subset / giới hạn số ảnh
train_df      # dữ liệu train
val_df        # dữ liệu validation
train_loader  # DataLoader dùng để train
val_loader    # DataLoader dùng để validate
```

Lưu ý quan trọng:

```text
Ảnh gốc không bị copy/gộp thành folder mới.
Ảnh vẫn nằm nguyên trong /kaggle/input.
Notebook chỉ gộp logic bằng dataframe chứa đường dẫn ảnh.
```

## 2. Pipeline Tổng Thể

Kế hoạch nên đi theo thứ tự:

```text
1. Kiểm tra dữ liệu
2. Phân tích dữ liệu
3. Feature engineering
4. Baseline bằng feature thủ công
5. Baseline deep learning nhỏ
6. Train nghiêm túc
7. Đánh giá theo từng generator
8. Phân tích lỗi
9. Cải thiện model
```

Không nên train model lớn ngay từ đầu. Trước tiên cần hiểu dữ liệu và tạo baseline để biết pipeline đúng.

## 3. Bước 1 - Kiểm Tra Dữ Liệu

Mục tiêu: đảm bảo dataset sạch và đọc đúng.

Cần kiểm tra:

```text
- Tổng số ảnh.
- Số ảnh theo subset.
- Số ảnh theo train/val.
- Số ảnh theo ai/nature.
- Ảnh có mở được không.
- Kích thước ảnh.
- Định dạng ảnh.
- File có bị lỗi hoặc trùng không.
```

Cell gợi ý kiểm tra kích thước ảnh:

```python
from PIL import Image
from tqdm.auto import tqdm

sizes = []

for path in tqdm(df["path"]):
    img = Image.open(path)
    sizes.append(img.size)

df["width"] = [s[0] for s in sizes]
df["height"] = [s[1] for s in sizes]

display(df[["width", "height"]].describe())
display(df.groupby(["subset", "label_name"])[["width", "height"]].describe())
```

Kết quả mong muốn:

```text
- Không có ảnh lỗi.
- Kích thước ảnh không quá bất thường.
- Số lượng giữa ai và nature cân bằng.
```

## 4. Bước 2 - Phân Tích Dữ Liệu

Mục tiêu: hiểu ảnh AI và ảnh thật khác nhau ở điểm nào trước khi train.

Các đặc trưng nên phân tích:

```text
- Width, height.
- Tỉ lệ width / height.
- Dung lượng file.
- Độ sáng trung bình.
- Độ tương phản.
- Trung bình màu RGB.
- Độ lệch chuẩn màu RGB.
- Độ sắc nét / blur.
- Năng lượng tần số FFT.
```

Tạo feature cơ bản:

```python
import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

features = []

for path in tqdm(df["path"]):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    features.append({
        "file_size_kb": os.path.getsize(path) / 1024,
        "width": img.width,
        "height": img.height,
        "aspect_ratio": img.width / img.height,
        "brightness": arr.mean(),
        "contrast": arr.std(),
        "mean_r": arr[:, :, 0].mean(),
        "mean_g": arr[:, :, 1].mean(),
        "mean_b": arr[:, :, 2].mean(),
        "std_r": arr[:, :, 0].std(),
        "std_g": arr[:, :, 1].std(),
        "std_b": arr[:, :, 2].std(),
    })

feature_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(features)], axis=1)
display(feature_df.head())
```

So sánh AI và nature:

```python
display(
    feature_df.groupby("label_name")[
        [
            "file_size_kb",
            "width",
            "height",
            "aspect_ratio",
            "brightness",
            "contrast",
            "mean_r",
            "mean_g",
            "mean_b",
        ]
    ].mean()
)
```

So sánh theo từng generator:

```python
display(
    feature_df.groupby(["subset", "label_name"])[
        ["file_size_kb", "brightness", "contrast"]
    ].mean()
)
```

## 5. Bước 3 - Feature Engineering

Feature engineering có 2 mục đích:

```text
1. Hiểu dữ liệu bằng feature thủ công.
2. Tạo baseline truyền thống trước khi dùng deep learning.
```

### 5.1. Feature độ blur / sharpness

Feature này đo độ sắc nét bằng Laplacian variance.

```python
import cv2
from tqdm.auto import tqdm

def blur_score(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var()

feature_df["blur_score"] = [
    blur_score(p) for p in tqdm(feature_df["path"])
]
```

Ý nghĩa:

```text
blur_score cao  -> ảnh sắc nét hơn
blur_score thấp -> ảnh mờ hơn
```

### 5.2. Feature FFT

Ảnh AI thường có artifact trong miền tần số. FFT là hướng rất đáng thử cho bài fake image detection.

```python
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

def fft_features(path):
    img = Image.open(path).convert("L").resize((224, 224))
    arr = np.array(img) / 255.0

    fft = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.log(np.abs(fft) + 1e-8)

    h, w = mag.shape
    cy, cx = h // 2, w // 2

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    low = mag[dist < 30].mean()
    mid = mag[(dist >= 30) & (dist < 70)].mean()
    high = mag[dist >= 70].mean()

    return {
        "fft_mean": mag.mean(),
        "fft_std": mag.std(),
        "fft_low": low,
        "fft_mid": mid,
        "fft_high": high,
    }

fft_rows = [fft_features(p) for p in tqdm(feature_df["path"])]
fft_df = pd.DataFrame(fft_rows)

feature_df = pd.concat([feature_df.reset_index(drop=True), fft_df], axis=1)
```

### 5.3. Feature edge density

Feature này đo mức độ cạnh trong ảnh.

```python
def edge_density(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    edges = cv2.Canny(img, 100, 200)
    return (edges > 0).mean()

feature_df["edge_density"] = [
    edge_density(p) for p in tqdm(feature_df["path"])
]
```

## 6. Bước 4 - Trực Quan Hóa EDA

Mục tiêu: nhìn sự khác biệt giữa AI và nature.

Nên vẽ:

```text
- Histogram brightness.
- Histogram contrast.
- Histogram blur_score.
- Histogram fft_high.
- Boxplot theo subset.
- Scatter plot giữa các feature.
```

Ví dụ:

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
sns.histplot(data=feature_df, x="brightness", hue="label_name", bins=50, kde=True)
plt.title("Brightness: AI vs Nature")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(data=feature_df, x="blur_score", hue="label_name", bins=50, kde=True)
plt.title("Blur score: AI vs Nature")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=feature_df, x="subset", y="fft_high", hue="label_name")
plt.xticks(rotation=45)
plt.title("FFT high-frequency energy theo subset")
plt.show()
```

## 7. Bước 5 - Baseline Bằng Feature Thủ Công

Trước khi train deep learning, nên train thử RandomForest hoặc Logistic Regression trên feature thủ công.

Mục tiêu:

```text
- Kiểm tra feature có tín hiệu phân biệt AI/nature không.
- Có baseline nhanh.
- Hiểu generator nào dễ/khó.
```

Ví dụ RandomForest:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

feature_cols = [
    "file_size_kb",
    "width",
    "height",
    "aspect_ratio",
    "brightness",
    "contrast",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "blur_score",
    "edge_density",
    "fft_mean",
    "fft_std",
    "fft_low",
    "fft_mid",
    "fft_high",
]

X = feature_df[feature_cols]
y = feature_df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

clf.fit(X_train, y_train)
pred = clf.predict(X_val)

print(classification_report(y_val, pred, target_names=["nature", "ai"]))
print(confusion_matrix(y_val, pred))
```

Feature importance:

```python
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": clf.feature_importances_,
}).sort_values("importance", ascending=False)

display(importances)
```

## 8. Bước 6 - Baseline Deep Learning Nhỏ

Sau baseline feature thủ công, train thử ResNet18.

Mục tiêu:

```text
- Kiểm tra pipeline DataLoader.
- Kiểm tra loss có giảm không.
- Kiểm tra label đúng không.
- Chưa cần đạt accuracy cao nhất.
```

Thiết lập ban đầu nên nhẹ:

```python
USE_SUBSETS = ["imagenet_ai_0419_biggan"]
LIMIT_PER_CLASS = 500
BATCH_SIZE = 32
EPOCHS = 1
```

Sau khi chạy ổn:

```python
USE_SUBSETS = ["imagenet_ai_0419_biggan"]
LIMIT_PER_CLASS = None
EPOCHS = 3
```

Sau đó mới dùng nhiều subset:

```python
USE_SUBSETS = None
LIMIT_PER_CLASS = 1000
EPOCHS = 3
```

## 9. Bước 7 - Train Nghiêm Túc

Sau khi pipeline ổn, tiến hành train theo từng mức:

### Mức 1 - Một generator

Train:

```text
imagenet_ai_0419_biggan
```

Evaluate:

```text
val của biggan
```

Mục tiêu:

```text
Model học được phân biệt AI/nature trên cùng generator.
```

### Mức 2 - Nhiều generator

Train:

```text
biggan + vqdm + sdv5
```

Evaluate:

```text
val từng generator riêng
```

Mục tiêu:

```text
Xem model generalize giữa các generator.
```

### Mức 3 - Tất cả generator

Train:

```text
toàn bộ train của 7 subset
```

Evaluate:

```text
toàn bộ val và val theo từng subset
```

Mục tiêu:

```text
Tạo detector tổng quát hơn.
```

## 10. Bước 8 - Đánh Giá Theo Từng Generator

Không chỉ xem accuracy tổng. Cần báo cáo riêng theo subset.

Các metric cần có:

```text
- Accuracy tổng.
- Accuracy từng subset.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- AI recall.
- Nature recall.
```

Bảng mong muốn:

```text
subset                    accuracy    ai_f1    nature_f1
imagenet_ai_0419_biggan   ...
imagenet_ai_0419_vqdm     ...
imagenet_ai_0424_sdv5     ...
imagenet_ai_0424_wukong   ...
imagenet_ai_0508_adm      ...
imagenet_glide            ...
imagenet_midjourney       ...
```

Ý nghĩa:

```text
Nếu accuracy tổng cao nhưng một subset rất thấp,
model chưa thật sự generalize tốt.
```

## 11. Bước 9 - Phân Tích Lỗi

Sau khi có model, cần xem các ảnh bị dự đoán sai.

Các nhóm lỗi:

```text
- AI bị đoán thành nature.
- Nature bị đoán thành AI.
- Generator nào sai nhiều nhất.
- Ảnh sai có bị blur/JPEG/low quality không.
- Ảnh sai có pattern màu/texture đặc biệt không.
```

Nên hiển thị ảnh sai ra grid:

```python
wrong_df = pred_df[pred_df["label"] != pred_df["pred"]]
display(wrong_df.groupby(["subset", "label_name"]).size())
```

Mục tiêu:

```text
Hiểu model đang học artifact nào,
và bị lừa bởi loại ảnh nào.
```

## 12. Bước 10 - Cải Thiện Model

Sau baseline, có thể cải thiện theo các hướng:

```text
- Dùng pretrained ResNet18/ResNet50.
- Dùng EfficientNet.
- Dùng ConvNeXt/Timm.
- Dùng Swin Transformer.
- Thêm augmentation nhẹ.
- Thêm JPEG compression augmentation.
- Thêm blur augmentation.
- Dùng FFT input hoặc mô hình hai nhánh RGB + FFT.
- Train leave-one-generator-out.
```

Thứ tự cải thiện nên làm:

```text
1. ResNet18 không pretrained.
2. ResNet18 pretrained.
3. ResNet50 pretrained.
4. EfficientNet/ConvNeXt.
5. RGB + FFT hoặc ensemble.
```

## 13. Lộ Trình Khuyến Nghị

Lộ trình cụ thể nên làm:

```text
Ngày 1:
- Đọc dataset.
- Hiển thị ảnh.
- Tạo feature_df.
- Vẽ EDA cơ bản.

Ngày 2:
- Thêm blur_score, edge_density, FFT feature.
- Train RandomForest baseline.
- Xem feature importance.

Ngày 3:
- Train ResNet18 trên BigGAN subset.
- Evaluate val BigGAN.
- Kiểm tra loss/accuracy.

Ngày 4:
- Train ResNet18 trên nhiều subset.
- Evaluate từng subset.
- Phân tích generator nào khó.

Ngày 5:
- Train ResNet50 pretrained.
- So sánh với ResNet18.
- Lưu checkpoint và kết quả.
```

## 14. Output Nên Lưu

Nên lưu các file sau vào `/kaggle/working`:

```text
genimage_index.csv
feature_df.csv
random_forest_baseline_report.txt
resnet18_checkpoint.pth
resnet50_checkpoint.pth
evaluation_by_subset.csv
wrong_predictions.csv
```

Mục đích:

```text
- Dễ kiểm tra lại.
- Dễ viết báo cáo.
- Dễ so sánh model.
```

## 15. Kết Luận

Không nên nhảy thẳng vào train model lớn. Với bài GenImage, phần quan trọng là:

```text
1. Dataset có cân bằng không.
2. AI/nature khác nhau ở feature nào.
3. Model có generalize qua generator khác không.
4. Generator nào khó nhất.
5. Lỗi của model nằm ở đâu.
```

Roadmap tốt nhất:

```text
EDA -> Feature engineering -> Baseline thủ công -> ResNet18 -> ResNet50 -> Evaluate theo generator -> Error analysis
```

