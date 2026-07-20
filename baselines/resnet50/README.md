# ResNet-50

Thư mục này chứa notebook riêng cho baseline `resnet50`.

## File

- `train_colab.ipynb`: notebook chạy trên Google Colab cho model này.
- `README.md`: mô tả mục đích, cấu hình chính và output.

## Mục đích

Notebook này chỉ train và đánh giá model `resnet50`.

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
- `MODEL_NAME`: đã preset là `resnet50`.
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
