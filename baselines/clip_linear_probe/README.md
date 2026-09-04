# CLIP Linear Probe

Thư mục này chứa notebook riêng cho baseline `clip_linear_probe`.

## File

- `train_colab.ipynb`: notebook chạy trên Google Colab cho model này.
- `train_clip_linear_probe_loader.ipynb`: notebook mới dùng `data_loader/` chung để load Tiny-GenImage từ Hugging Face, trích CLIP embedding, train Logistic Regression tầng cuối và chạy tuần tự 4 evaluation case.
- `train_clip_linear_probe_kaggle_folder.ipynb`: notebook cho Kaggle Input dạng folder. File này dùng `data_loader/tiny_genimage_kaggle.py`, đọc ảnh lazy từ path local trong `/kaggle/input`, rồi train linear head mini-batch trên frozen CLIP để giảm RAM.
- `README.md`: mô tả mục đích, cấu hình chính và output.

## Mục đích

Notebook này chỉ train và đánh giá model `clip_linear_probe`.

Notebook `train_clip_linear_probe_loader.ipynb` chạy pipeline:

```text
image -> frozen CLIP image encoder -> normalized CLIP embedding -> Logistic Regression -> real/fake
```

CLIP được freeze hoàn toàn. Chỉ classifier tuyến tính ở tầng cuối được train.

Trong notebook mới:

- `STREAMING = True`: đọc lazy từ Hugging Face, không tải full Tiny-GenImage về local cache trước.
- `BALANCE_REAL = True`: cân bằng số ảnh `real` và `fake` trong split.
- `MAX_TRAIN_SAMPLES`, `MAX_EVAL_SAMPLES`: giới hạn sample để test nhanh. Đặt `None` để chạy full split.
- `EXPERIMENT_CONFIGS`: danh sách 4 case chạy tự động gồm `combined`, `in_domain_biggan`, `cross_generator_glide`, `train_one_generator_biggan`.

Mỗi case tạo output riêng:

```text
outputs/clip_linear_probe/<experiment_name>/<run_id>/
```

Sau khi chạy xong, notebook tạo thêm bảng tổng hợp:

```text
outputs/clip_linear_probe/summary_metrics_<run_id>.csv
```

## Chạy Trên Kaggle Với Tiny-GenImage Folder

Khi upload code + dataset vào Kaggle Input, chạy notebook:

```text
baselines/clip_linear_probe/train_clip_linear_probe_kaggle_folder.ipynb
```

Notebook này tự tìm:

```text
/kaggle/input/.../HoangHa_Code/data_loader
/kaggle/input/.../Tiny-GenImage folder
```

Nếu auto-detect dataset sai, sửa trong cell cấu hình:

```python
DATASET_ROOT = "/kaggle/input/<ten-dataset-tiny-genimage>"
```

Mặc định notebook chỉ chạy một case:

```python
RUN_ALL_CASES = False
SELECTED_EXPERIMENT = "combined"
MAX_TRAIN_SAMPLES = 500
MAX_EVAL_SAMPLES = 300
BATCH_SIZE = 16
NUM_WORKERS = 0
```

Đây là cấu hình test RAM trước. Khi đã ổn thì tăng sample dần hoặc bật:

```python
RUN_ALL_CASES = True
```

Khác với notebook Hugging Face cũ, bản Kaggle này không extract toàn bộ CLIP embedding ra `numpy` trước khi train. Nó encode theo mini-batch và chỉ train linear head, nên ít RAM hơn.

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
- `MODEL_NAME`: đã preset là `clip_linear_probe`.
- `FEATURE_EXTRACTOR`: đã preset là `clip`.
- Với notebook mới, kiểm tra thêm `EVAL_CASE`, `STREAMING`, `BALANCE_REAL`, `MAX_TRAIN_SAMPLES`, `MAX_EVAL_SAMPLES`, `CLIP_MODEL_NAME`, `CLIP_PRETRAINED`.

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
