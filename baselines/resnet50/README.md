# ResNet-50

Thư mục này chứa notebook riêng cho baseline `resnet50`.

## File

- `train_colab.ipynb`: notebook chạy trên Google Colab cho model này.
- `train_resnet50_hf_loader.ipynb`: notebook dùng `data_loader` Hugging Face Tiny-GenImage.
- `train_resnet50_kaggle_folder.ipynb`: notebook dùng `data_loader/tiny_genimage_kaggle.py` để đọc Tiny-GenImage dạng folder trong Kaggle Input.
- `train_resnet50_hf_loader.py`: script dùng `data_loader` Hugging Face Tiny-GenImage.
- `train_resnet50_kaggle_folder.py`: script dùng `data_loader/tiny_genimage_kaggle.py` để đọc Tiny-GenImage dạng folder trong Kaggle Input.
- `README.md`: mô tả mục đích, cấu hình chính và output.

## Mục đích

Notebook này chỉ train và đánh giá model `resnet50`.

Hai script mới train theo flow:

```text
image -> ImageNet-pretrained ResNet-50 frozen backbone -> fc layer -> real/fake
```

Toàn bộ backbone ResNet-50 được freeze. Chỉ layer cuối:

```python
model.fc = nn.Linear(model.fc.in_features, 2)
```

được train.

## Cách Chạy Nhanh

Trên Kaggle, với code đã add vào Input, nên import notebook:

```text
baselines/resnet50/train_resnet50_kaggle_folder.ipynb
```

Hoặc chạy script tương đương:

```python
%run /kaggle/input/<code-dataset>/HoangHa_Code/baselines/resnet50/train_resnet50_kaggle_folder.py
```

Nếu muốn chạy bản Hugging Face loader, dùng notebook:

```text
baselines/resnet50/train_resnet50_hf_loader.ipynb
```

Hoặc script:

```python
%run /kaggle/input/<code-dataset>/HoangHa_Code/baselines/resnet50/train_resnet50_hf_loader.py
```

Bản Kaggle-folder phù hợp hơn với dataset bạn đang add vào Input:

```text
tiny genimage/
  imagenet_ai_0419_biggan/
    train/ai
    train/nature
    val/ai
    val/nature
```

## Early Stopping

Script dùng early stopping:

```python
MAX_EPOCHS = 20
PATIENCE = 3
MIN_DELTA = 1e-3
```

Điều kiện dừng:

```text
Nếu val balanced_accuracy không tăng ít nhất MIN_DELTA trong PATIENCE epoch liên tiếp thì dừng.
```

Bản Kaggle-folder tách:

```text
train gốc -> train_inner + val_inner
val gốc   -> test cuối
```

Như vậy `val_inner` dùng để chọn epoch, còn `val` gốc của dataset được giữ làm test cuối.

## Cấu Hình RAM An Toàn

Mặc định:

```python
RUN_ALL_CASES = False
SELECTED_EXPERIMENT = "combined"
MAX_TRAIN_SAMPLES = 500
MAX_TEST_SAMPLES = 300
BATCH_SIZE = 16
NUM_WORKERS = 0
```

Khi chạy ổn, tăng `MAX_TRAIN_SAMPLES` trước, sau đó mới bật:

```python
RUN_ALL_CASES = True
```

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
