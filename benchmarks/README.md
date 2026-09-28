# Benchmarks

Thư mục này chứa các thí nghiệm đánh giá nhiều mô hình trên cùng cohort.

```text
benchmarks/<benchmark>/
├── code/       # script/notebook dựng cohort và chạy inference
└── artifacts/  # manifest, metrics, predictions và provenance đã tải về
```

- `commfor_balanced_four_models`: benchmark CommFor cân bằng 2.000 ảnh.
- `commfor_unseen_three_models`: benchmark unseen-generator CLIP/NPR/AIDE.
- `tiny_combined_to_commfor_eval`: protocol train Tiny combined và test CommFor.

Checkpoint canonical luôn lấy từ `baselines/<model>/artifacts/checkpoints/`.
Report Tiny ba mô hình đã từng được lưu trong `baselines/` được giữ tại
`commfor_unseen_three_models/artifacts/tiny_genimage_three_models_results/`.
