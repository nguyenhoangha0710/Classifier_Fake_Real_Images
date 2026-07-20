# Logistic Regression

Thư mục này chứa notebook riêng cho baseline `logistic_regression`.

## File

- `train_colab.ipynb`: notebook chạy trên Google Colab cho model này.
- `README.md`: mô tả mục đích, cấu hình chính và output.

## Mục đích

Notebook này chỉ train và đánh giá model `logistic_regression`.

## Cách chạy hai protocol

Khi chạy notebook từ đầu đến cuối, notebook sẽ lần lượt chạy:

```python
RUN_EXPERIMENT_CASES = ["cross_generator", "combined"]
```

- `cross_generator`: train trên `BASE_GENERATOR`, test trên tất cả generator.
- `combined`: train trên toàn bộ split `train`, test trên split `val`/`test`.

Mỗi protocol tạo một `RUN_DIR` riêng trong Google Drive.

## Cấu hình cần kiểm tra

- `PROJECT_ROOT`: nơi lưu output trên Google Drive.
- `DATA_ROOT`: nơi chứa ảnh thật sự.
- `METADATA_ROOT`: nơi chứa metadata đã chuẩn bị.
- `BASE_GENERATOR`: dùng cho TH1 `cross_generator`.
- `MODEL_NAME`: đã preset là `logistic_regression`.
- `FEATURE_EXTRACTOR`: đã preset là `fft`.

## Output

```text
outputs/<experiment_case>/<model_name>/<run_id>/
  checkpoints/
  features/
  predictions/predictions.csv
  metrics/overall_metrics.json
  metrics/generator_metrics.csv
  plots/
```

Nếu notebook chưa được chạy trên Colab thì chưa có metric thực tế. Không suy diễn kết quả trước khi train/evaluate.

# Baseline Evaluation - Logistic Regression

## 1. Mục tiêu

Thực hiện đánh giá **Logistic Regression** như một baseline truyền thống cho bài toán **phân loại ảnh thật (Real) và ảnh sinh bởi AI (Fake)** trên bộ dữ liệu **Tiny-GenImage**.

Mục tiêu của thí nghiệm là:

- Xây dựng một baseline đơn giản, dễ tái hiện.
- Đánh giá khả năng phân loại của Logistic Regression trong hai giao thức:
  - **Combined Generator**
  - **Cross-Generator**
- Làm cơ sở so sánh với các mô hình học sâu sẽ được triển khai trong các thí nghiệm tiếp theo.

---

# 2. Pipeline của hệ thống

Pipeline xử lý dữ liệu được xây dựng như sau:

```

Metadata CSV

      │

      ▼

Resolve image_path từ relative_path

      │

      ▼

Đọc ảnh bằng PIL

([Image.open](http://Image.open) → RGB)

      │

      ▼

Resize về IMAGE_SIZE × IMAGE_SIZE

      │

      ▼

ToTensor()

Pixel [0,255] → Float [0,1]

      │

      ▼

Feature Engineering

(PyTorch)

      │

      ▼

StandardScaler

(Fit trên train,

Transform train & test)

      │

      ▼

Logistic Regression

      │

      ▼

Fake Probability

      │

      ▼

Predicted Label

      │

      ▼

Evaluation Metrics



```

---

# 3. Kết quả thực nghiệm

## 3.1. Combined Generator

### Kết quả tổng thể


| Metric                 | Giá trị   |
| ---------------------- | --------- |
| Accuracy               | **0.580** |
| Balanced Accuracy      | **0.580** |
| Precision              | **0.580** |
| Recall                 | **0.578** |
| F1-score               | **0.579** |
| ROC-AUC                | **0.608** |
| Average Precision (AP) | **0.608** |


### Kết quả theo từng Generator


| Generator  | Accuracy  | Balanced Accuracy | Precision | Recall    | F1-score  | ROC-AUC   | Average Precision |
| ---------- | --------- | ----------------- | --------- | --------- | --------- | --------- | ----------------- |
| BigGAN     | **0.701** | **0.701**         | 0.664     | **0.812** | **0.731** | **0.772** | **0.756**         |
| ADM        | **0.707** | **0.707**         | 0.667     | **0.826** | **0.738** | **0.760** | **0.713**         |
| VQDM       | 0.618     | 0.618             | 0.608     | 0.664     | 0.635     | 0.664     | 0.625             |
| SDV5       | 0.528     | 0.528             | 0.532     | 0.472     | 0.500     | 0.535     | 0.548             |
| Wukong     | 0.524     | 0.524             | 0.528     | 0.458     | 0.490     | 0.501     | 0.562             |
| GLIDE      | 0.489     | 0.489             | 0.487     | 0.414     | 0.448     | 0.499     | 0.487             |
| Midjourney | 0.493     | 0.493             | 0.491     | 0.398     | 0.440     | 0.517     | 0.513             |


### Nhận xét

Khi được huấn luyện trên dữ liệu của tất cả generator, Logistic Regression đạt Accuracy khoảng **58%**.

Mô hình hoạt động tương đối tốt trên **BigGAN** và **ADM**, tuy nhiên hiệu năng giảm đáng kể trên các generator còn lại, đặc biệt là **GLIDE**, **Midjourney**, **SDV5** và **Wukong**.

Điều này cho thấy mặc dù Logistic Regression có thể học được một phần đặc trưng của dữ liệu khi được tiếp xúc với nhiều generator, khả năng phân biệt vẫn còn hạn chế đối với các generator có chất lượng ảnh cao.

---

## 3.2. Cross-Generator (Train BigGAN)

### Kết quả tổng thể


| Metric                 | Giá trị   |
| ---------------------- | --------- |
| Accuracy               | **0.536** |
| Balanced Accuracy      | **0.536** |
| Precision              | **0.549** |
| Recall                 | **0.404** |
| F1-score               | **0.465** |
| ROC-AUC                | **0.549** |
| Average Precision (AP) | **0.568** |


### Kết quả theo từng Generator


| Generator         | Accuracy  | Balanced Accuracy | Precision | Recall    | F1-score  | ROC-AUC   | Average Precision |
| ----------------- | --------- | ----------------- | --------- | --------- | --------- | --------- | ----------------- |
| **BigGAN (Seen)** | **0.741** | **0.741**         | 0.702     | **0.838** | **0.764** | **0.825** | **0.831**         |
| VQDM              | 0.502     | 0.502             | 0.503     | 0.322     | 0.393     | 0.556     | 0.508             |
| SDV5              | 0.516     | 0.516             | 0.525     | 0.342     | 0.414     | 0.512     | 0.536             |
| ADM               | 0.503     | 0.503             | 0.505     | 0.312     | 0.386     | 0.502     | 0.500             |
| GLIDE             | 0.510     | 0.510             | 0.514     | 0.360     | 0.424     | 0.491     | 0.490             |
| Wukong            | 0.498     | 0.498             | 0.497     | 0.362     | 0.419     | 0.478     | 0.519             |
| Midjourney        | 0.482     | 0.482             | 0.471     | 0.292     | 0.360     | 0.491     | 0.495             |


### Nhận xét

Trong giao thức Cross-Generator, Logistic Regression đạt Accuracy **74.1%** trên **BigGAN**, là generator được sử dụng trong quá trình huấn luyện.

Tuy nhiên, hiệu năng trên các generator chưa từng xuất hiện trong tập huấn luyện chỉ dao động quanh **50%**, gần tương đương mức dự đoán ngẫu nhiên.

Kết quả này cho thấy mô hình chưa học được các đặc trưng có khả năng tổng quát hóa giữa các generator khác nhau.

---

# 4. So sánh hai giao thức


| Metric | Combined Generator | Cross-Generator (Train on BigGAN) |     |
| ------ | ------------------ | --------------------------------- | --- |



|          |           |           |     |
| -------- | --------- | --------- | --- |
| Accuracy | **0.580** | **0.536** |     |



|                   |           |           |     |
| ----------------- | --------- | --------- | --- |
| Balanced Accuracy | **0.580** | **0.536** |     |



|           |           |           |     |
| --------- | --------- | --------- | --- |
| Precision | **0.580** | **0.549** |     |



|        |           |           |     |
| ------ | --------- | --------- | --- |
| Recall | **0.578** | **0.404** |     |



|          |           |           |     |
| -------- | --------- | --------- | --- |
| F1-score | **0.579** | **0.465** |     |



|         |           |           |     |
| ------- | --------- | --------- | --- |
| ROC-AUC | **0.608** | **0.549** |     |



|                        |           |           |     |
| ---------------------- | --------- | --------- | --- |
| Average Precision (AP) | **0.608** | **0.568** |     |


### Nhận xét

Giao thức **Combined** cho kết quả tốt hơn so với **Cross-Generator**.

Nguyên nhân là trong Combined, mô hình được học từ dữ liệu của tất cả generator, trong khi Cross-Generator chỉ được huấn luyện trên một generator duy nhất.

Sự suy giảm hiệu năng phản ánh sự khác biệt về phân bố dữ liệu giữa các generator và cho thấy Logistic Regression chưa có khả năng tổng quát hóa tốt trong bài toán phát hiện ảnh AI.

---

# 5. Kết luận

Logistic Regression được sử dụng như một **baseline truyền thống** cho bài toán phát hiện ảnh AI-generated.

Qua hai thí nghiệm có thể rút ra một số nhận xét:

- Logistic Regression là một baseline đơn giản và có thời gian huấn luyện nhanh.
- Mô hình có khả năng học được đặc trưng phân biệt giữa ảnh thật và ảnh giả khi dữ liệu huấn luyện bao phủ nhiều generator.
- Khi chỉ được huấn luyện trên một generator, hiệu năng giảm rõ rệt khi đánh giá trên các generator chưa từng xuất hiện.
- Điều này cho thấy các đặc trưng được học chưa đủ khả năng biểu diễn những dấu hiệu phổ quát của ảnh AI-generated.

Các kết quả trên cho thấy Logistic Regression có thể hoạt động ở mức chấp nhận được khi dữ liệu huấn luyện bao phủ đầy đủ các generator, nhưng khả năng tổng quát hóa sang các generator chưa từng xuất hiện còn hạn chế.