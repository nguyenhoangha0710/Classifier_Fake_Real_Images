# HoangHa_Code

Thư mục làm việc chính cho project GenImage.

## Cấu trúc

- `download-data-colab.ipynb`: notebook chuẩn bị metadata/dataset tuần 2.
- `train-baseline-colab.ipynb`: notebook tổng, chọn model bằng biến `MODEL_NAME`.
- `baselines/`: notebook riêng cho từng baseline model.
- `outputs/`: nơi gom output hoặc ghi chú output theo từng model.
- `reports/`: nơi gom báo cáo.

## Quy trình đề xuất

1. Chạy `download-data-colab.ipynb` để sinh metadata.
2. Lưu metadata lên Google Drive.
3. Chọn một notebook trong `baselines/<model>/train_colab.ipynb`.
4. Chỉnh `PROJECT_ROOT`, `DATA_ROOT`, `METADATA_ROOT`, `EXPERIMENT_CASE`.
5. Chạy notebook trên Colab.
6. Kiểm tra `predictions`, `metrics`, `plots` và report.
