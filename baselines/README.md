# Baselines

Mỗi thư mục con tương ứng với một baseline model riêng.

## Danh sách model

- `resnet50`
- `swin_t`
- `cnnspot`
- `clip_linear_probe`
- `fft`
- `logistic_regression`
- `linear_svm`
- `random_forest`
- `knn`

## Quy ước chạy

Mỗi notebook chỉ train model trong đúng thư mục đó. Khi chạy notebook từ đầu đến cuối, biến sau sẽ làm notebook chạy cả TH1 và TH2:

```python
RUN_EXPERIMENT_CASES = ["cross_generator", "combined"]
```

Mỗi protocol tạo một `RUN_DIR` riêng trong Google Drive để không ghi đè kết quả.
