# Pipeline Logistic Regression Baseline

File này mô tả pipeline xử lý dữ liệu của notebook:

```text
train-logistic-regression-colab.ipynb
```

Mục tiêu là giải thích cách một ảnh RGB được biến đổi thành vector feature để huấn luyện Logistic Regression cho bài toán phân loại ảnh thật và ảnh AI-generated:

```text
real = 0
fake = 1
```

## 1. Đầu Vào

Đầu vào của model gồm hai phần:

```text
metadata CSV + ảnh gốc trong dataset Tiny-GenImage
```

Metadata chứa các thông tin tối thiểu:

```text
image_path
relative_path
generator
original_split
label
label_name
width
height
image_format
file_size
sha256
```

Trong notebook train, metadata được đọc từ Google Drive. Ảnh thật sự được lấy từ dataset Tiny-GenImage tải bằng KaggleHub:

```python
kagglehub.dataset_download("yangsangtai/tiny-genimage")
```

Do metadata có thể chứa đường dẫn cũ như `/kaggle/input/...`, notebook ưu tiên dùng:

```python
DATA_ROOT / relative_path
```

để tìm ảnh đúng trong thư mục dataset vừa tải.

## 2. Split Dữ Liệu

Notebook giữ nguyên split gốc của Tiny-GenImage.

Không tự chia lại train, val hoặc test.

Dataset có dạng:

```text
Generator/
    train/
        ai/
        nature/
    val/
        ai/
        nature/
```

Quy ước label:

```text
nature -> real -> 0
ai     -> fake -> 1
```

## 3. Hai Trường Hợp Thí Nghiệm

### TH1: Cross-generator

Ý nghĩa:

```text
Train trên một generator.
Test trên val của tất cả generator.
```

Ví dụ:

```text
train BigGAN -> test BigGAN, VQDM, SDV5, Wukong, ADM, GLIDE, Midjourney
train VQDM   -> test BigGAN, VQDM, SDV5, Wukong, ADM, GLIDE, Midjourney
train ADM    -> test BigGAN, VQDM, SDV5, Wukong, ADM, GLIDE, Midjourney
```

Nếu muốn train từng generator riêng lẻ cho TH1:

```python
EXPERIMENT_CASE = "cross_generator"
BASE_GENERATOR = None
```

Nếu chỉ muốn train nhanh một generator:

```python
EXPERIMENT_CASE = "cross_generator"
BASE_GENERATOR = "imagenet_ai_0419_biggan"
```

### TH2: Combined

Ý nghĩa:

```text
Train trên train của tất cả generator.
Test trên val của tất cả generator.
```

Cấu hình:

```python
EXPERIMENT_CASE = "combined"
BASE_GENERATOR = None
```

TH2 chỉ có một model tổng:

```text
base_generator = all_generators
```

## 4. Pipeline Tổng Thể

```text
metadata CSV
  |
  v
resolve image_path bằng relative_path
  |
  v
PIL.Image.open(...).convert("RGB")
  |
  v
Resize ảnh về IMAGE_SIZE x IMAGE_SIZE
  |
  v
ToTensor()
pixel [0, 255] -> float [0.0, 1.0]
shape = [3, IMAGE_SIZE, IMAGE_SIZE]
  |
  v
Feature engineering bằng PyTorch
  |
  v
StandardScaler
fit trên train, transform train/test
  |
  v
LogisticRegression
  |
  v
fake_probability
  |
  v
predicted_label
  |
  v
metrics + predictions + report
```

## 5. Biến Đổi Ảnh Thành Tensor

Ảnh được đọc bằng PIL:

```python
image = Image.open(row["image_path"]).convert("RGB")
```

`convert("RGB")` đảm bảo mọi ảnh đều có 3 kênh màu:

```text
R, G, B
```

Sau đó ảnh được resize:

```python
transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
```

và chuyển sang tensor:

```python
transforms.ToTensor()
```

`ToTensor()` thực hiện hai việc:

```text
PIL image -> torch.Tensor
pixel integer [0, 255] -> float [0.0, 1.0]
```

Nếu:

```python
IMAGE_SIZE = 64
```

thì một ảnh sau transform có shape:

```text
[3, 64, 64]
```

Trong đó:

```text
3  = số kênh RGB
64 = chiều cao
64 = chiều rộng
```

## 6. Feature Engineering

Logistic Regression không nhận ảnh 2D trực tiếp. Vì vậy notebook chuyển mỗi ảnh thành một vector số 1D.

### 6.1. Flatten Pixel RGB

Tensor ảnh:

```text
[3, IMAGE_SIZE, IMAGE_SIZE]
```

được flatten thành:

```text
[3 * IMAGE_SIZE * IMAGE_SIZE]
```

Ví dụ với:

```python
IMAGE_SIZE = 64
```

số chiều pixel là:

```text
3 * 64 * 64 = 12,288
```

Ý nghĩa:

```text
Flatten pixel giữ lại thông tin màu và vị trí pixel ở dạng vector thô.
```

### 6.2. Mean RGB

Notebook tính trung bình màu theo từng kênh:

```python
mean_rgb = images.mean(dim=(2, 3))
```

Kết quả:

```text
[mean_R, mean_G, mean_B]
```

Số chiều:

```text
3
```

Ý nghĩa:

```text
mean_rgb mô tả xu hướng màu tổng thể của ảnh.
```

### 6.3. Standard Deviation RGB

Notebook tính độ lệch chuẩn màu theo từng kênh:

```python
std_rgb = images.std(dim=(2, 3))
```

Kết quả:

```text
[std_R, std_G, std_B]
```

Số chiều:

```text
3
```

Ý nghĩa:

```text
std_rgb mô tả mức độ biến thiên màu, texture và độ tương phản tổng quát.
```

### 6.4. Ghép Feature

Các feature được nối lại:

```python
features = torch.cat([flat_pixels, mean_rgb, std_rgb], dim=1)
```

Với `IMAGE_SIZE = 64`, tổng số chiều là:

```text
12,288 + 3 + 3 = 12,294
```

Một ảnh trở thành một vector:

```text
image -> feature vector shape [12294]
```

Một batch ảnh trở thành ma trận:

```text
batch images -> feature matrix shape [batch_size, 12294]
```

## 7. Chuẩn Hóa Bằng StandardScaler

Trước khi đưa vào Logistic Regression, feature được chuẩn hóa bằng:

```python
StandardScaler()
```

Công thức:

```text
x_scaled = (x - mean_train) / std_train
```

Quan trọng:

```text
Scaler chỉ fit trên train.
Test chỉ được transform bằng scaler đã fit từ train.
```

Lý do:

```text
Nếu fit scaler trên test thì thông tin thống kê của test bị rò rỉ vào quá trình train.
Đó là data leakage.
```

Trong notebook:

```python
scaler.fit_transform(x_train)
scaler.transform(x_test)
```

## 8. Model Logistic Regression

Model sử dụng:

```python
sklearn.linear_model.LogisticRegression
```

Các tham số được cấu hình ở đầu notebook:

```text
C
max_iter
solver
class_weight
random_state
```

Đầu vào của model:

```text
feature vector đã được scale
```

Đầu ra của model:

```text
fake_probability = xác suất ảnh thuộc class fake
```

Dự đoán label:

```python
predicted_label = 1 nếu fake_probability >= 0.5
predicted_label = 0 nếu fake_probability < 0.5
```

## 9. Đánh Giá

Notebook tính metric tổng thể:

```text
Accuracy
Balanced Accuracy
Precision
Recall
F1-score
ROC-AUC
Average Precision
Confusion Matrix
ROC Curve
```

Ngoài ra còn tính metric riêng theo từng generator:

```text
generator_metrics.csv
```

Với TH1, bảng theo generator trả lời câu hỏi:

```text
Model train trên generator này có dự đoán tốt trên generator khác không?
```

Ví dụ:

```text
base_generator = imagenet_ai_0419_biggan

Kết quả theo generator:
- test BigGAN
- test VQDM
- test SDV5
- test Wukong
- test ADM
- test GLIDE
- test Midjourney
```

## 10. Output

Mỗi lần chạy sẽ sinh các file:

```text
logistic_model.pkl
scaler.pkl
predictions.csv
overall_metrics.json
generator_metrics.csv
confusion_matrix.png
roc_curve.png
logistic_regression_report.md
```

Ý nghĩa:

```text
logistic_model.pkl
    Model Logistic Regression đã train.

scaler.pkl
    StandardScaler đã fit trên train.

predictions.csv
    Dự đoán từng ảnh trong tập test.

overall_metrics.json
    Metric tổng thể của run.

generator_metrics.csv
    Metric riêng theo từng generator.

confusion_matrix.png
    Ma trận nhầm lẫn.

roc_curve.png
    Đường cong ROC.

logistic_regression_report.md
    Báo cáo cấu hình, dataset, metric và kết luận.
```

## 11. Sơ Đồ Một Ảnh Đi Qua Hệ Thống

```text
Ảnh RGB 128x128 hoặc kích thước gốc
  |
  v
Resize về IMAGE_SIZE x IMAGE_SIZE
  |
  v
Tensor [3, IMAGE_SIZE, IMAGE_SIZE]
  |
  v
Pixel float trong [0.0, 1.0]
  |
  +--> Flatten pixel: [3 * IMAGE_SIZE * IMAGE_SIZE]
  |
  +--> Mean RGB: [3]
  |
  +--> Std RGB: [3]
  |
  v
Ghép feature
  |
  v
Vector feature [3 * IMAGE_SIZE * IMAGE_SIZE + 6]
  |
  v
StandardScaler
  |
  v
Logistic Regression
  |
  v
fake_probability
  |
  v
predicted_label real/fake
```

## 12. Giới Hạn Của Baseline

Logistic Regression là baseline tuyến tính, nên khả năng học đặc trưng hình ảnh phức tạp bị giới hạn.

Feature hiện tại chủ yếu gồm:

```text
pixel thô
mean RGB
std RGB
```

Model này hữu ích để làm mốc so sánh ban đầu. Nếu các deep learning model sau này không vượt qua baseline này, cần kiểm tra lại pipeline huấn luyện, dữ liệu, hoặc cách đánh giá.
