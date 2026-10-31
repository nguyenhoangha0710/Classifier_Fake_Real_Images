# BFree: thay đổi ngưỡng dự đoán

Nguồn: [Kaggle version 1](https://www.kaggle.com/datasets/hadinho0710/eval-capd-test-tiny-bfree).

Tính lại từ 72.128 dự đoán: 10.304 real dùng chung + 6 nhánh fake, mỗi nhánh 10.304 ảnh. Dự đoán fake nếu p >= ngưỡng. Macro BA là trung bình BA của 6 cặp real/fake cân bằng; không phải accuracy trên tập gộp mất cân bằng. Đây là BFree validation.

Đã kiểm tra các confusion count và metrics tại ngưỡng 0,5 trùng bảng gốc cho cả hai pooling. AUC/AP không thay đổi khi chỉ thay ngưỡng.

## lse_full

| Ngưỡng | Macro BA (%) | Recall fake (%) | Recall real (%) | FPR (%) | Real báo nhầm |
| --- | --- | --- | --- | --- | --- |
| 0.500 | 82.54 | 65.85 | 99.22 | 0.78 | 80 |
| 0.300 | 83.14 | 67.35 | 98.92 | 1.08 | 111 |
| 0.200 | 83.41 | 68.26 | 98.56 | 1.44 | 148 |
| 0.100 | 83.60 | 69.66 | 97.53 | 2.47 | 254 |
| 0.050 | 83.50 | 71.14 | 95.86 | 4.14 | 427 |
| 0.020 | 82.79 | 73.48 | 92.11 | 7.89 | 813 |
| 0.010 | 81.72 | 75.65 | 87.78 | 12.22 | 1259 |
| 0.005 | 79.71 | 78.15 | 81.26 | 18.74 | 1931 |
| 0.001 | 71.85 | 85.72 | 57.98 | 42.02 | 4330 |

Ngưỡng chung tối ưu trên chính validation: 0.0845256060; macro BA 83.66%, fake recall 70.02%, real recall 97.30%, FP=278.

| Nhánh | Recall fake @0.5 (%) | @0.1 (%) | @0.02 (%) | @0.01 (%) | BA @0.5 (%) | BA @0.1 (%) | BA @0.02 (%) | BA @0.01 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inpainted_diffcat | 99.24 | 99.59 | 99.76 | 99.84 | 99.23 | 98.56 | 95.93 | 93.81 |
| inpainted_diffcat_origBG | 62.95 | 67.65 | 72.23 | 74.91 | 81.09 | 82.59 | 82.17 | 81.35 |
| inpainted_samecat | 98.05 | 98.98 | 99.45 | 99.58 | 98.64 | 98.26 | 95.78 | 93.68 |
| inpainted_samecat_origBG | 38.75 | 45.16 | 51.61 | 55.58 | 68.99 | 71.35 | 71.86 | 71.68 |
| selfconditioned | 85.35 | 90.28 | 93.94 | 95.23 | 92.28 | 93.91 | 93.03 | 91.50 |
| selfconditioned_origBG | 10.78 | 16.30 | 23.88 | 28.78 | 55.00 | 56.92 | 58.00 | 58.28 |

Ngưỡng tối ưu riêng từng nhánh (oracle, chỉ để phân tích):

| Nhánh | Ngưỡng | BA (%) | Recall fake (%) | Recall real (%) |
| --- | --- | --- | --- | --- |
| inpainted_diffcat | 0.47839105 | 99.24 | 99.27 | 99.20 |
| inpainted_diffcat_origBG | 0.08509407 | 82.74 | 68.18 | 97.30 |
| inpainted_samecat | 0.25873501 | 98.69 | 98.56 | 98.83 |
| inpainted_samecat_origBG | 0.02861015 | 72.04 | 50.11 | 93.96 |
| selfconditioned | 0.08486626 | 94.01 | 90.72 | 97.30 |
| selfconditioned_origBG | 0.00967505 | 58.32 | 29.17 | 87.47 |

## lse_window

| Ngưỡng | Macro BA (%) | Recall fake (%) | Recall real (%) | FPR (%) | Real báo nhầm |
| --- | --- | --- | --- | --- | --- |
| 0.500 | 82.78 | 66.46 | 99.10 | 0.90 | 93 |
| 0.300 | 83.31 | 67.89 | 98.73 | 1.27 | 131 |
| 0.200 | 83.49 | 68.81 | 98.17 | 1.83 | 189 |
| 0.100 | 83.62 | 70.27 | 96.96 | 3.04 | 313 |
| 0.050 | 83.41 | 71.90 | 94.91 | 5.09 | 524 |
| 0.020 | 82.37 | 74.46 | 90.28 | 9.72 | 1002 |
| 0.010 | 80.76 | 76.80 | 84.72 | 15.28 | 1574 |
| 0.005 | 78.38 | 79.56 | 77.19 | 22.81 | 2350 |
| 0.001 | 69.70 | 87.52 | 51.87 | 48.13 | 4959 |

Ngưỡng chung tối ưu trên chính validation: 0.1138782528; macro BA 83.66%, fake recall 70.02%, real recall 97.30%, FP=278.

| Nhánh | Recall fake @0.5 (%) | @0.1 (%) | @0.02 (%) | @0.01 (%) | BA @0.5 (%) | BA @0.1 (%) | BA @0.02 (%) | BA @0.01 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inpainted_diffcat | 99.31 | 99.63 | 99.80 | 99.84 | 99.20 | 98.30 | 95.04 | 92.28 |
| inpainted_diffcat_origBG | 63.78 | 68.40 | 73.46 | 76.26 | 81.44 | 82.68 | 81.87 | 80.49 |
| inpainted_samecat | 98.22 | 99.04 | 99.51 | 99.67 | 98.66 | 98.00 | 94.89 | 92.20 |
| inpainted_samecat_origBG | 39.54 | 46.19 | 53.39 | 57.49 | 69.32 | 71.57 | 71.83 | 71.11 |
| selfconditioned | 86.24 | 90.94 | 94.51 | 95.68 | 92.67 | 93.95 | 92.39 | 90.20 |
| selfconditioned_origBG | 11.65 | 17.43 | 26.10 | 31.87 | 55.37 | 57.20 | 58.19 | 58.30 |

Ngưỡng tối ưu riêng từng nhánh (oracle, chỉ để phân tích):

| Nhánh | Ngưỡng | BA (%) | Recall fake (%) | Recall real (%) |
| --- | --- | --- | --- | --- |
| inpainted_diffcat | 0.56074391 | 99.24 | 99.27 | 99.20 |
| inpainted_diffcat_origBG | 0.11462023 | 82.74 | 68.18 | 97.30 |
| inpainted_samecat | 0.32697861 | 98.69 | 98.56 | 98.83 |
| inpainted_samecat_origBG | 0.03938099 | 72.04 | 50.11 | 93.96 |
| selfconditioned | 0.11412564 | 94.01 | 90.72 | 97.30 |
| selfconditioned_origBG | 0.00883915 | 58.33 | 33.00 | 83.66 |

## Diễn giải

Giảm ngưỡng giúp bắt thêm ảnh giả nhưng làm tăng báo nhầm ảnh thật. Ngưỡng dùng chung được tối ưu trực tiếp trên nhãn validation, nên kết quả tối ưu là mức tham khảo có thiên lệch lựa chọn; cần khóa ngưỡng rồi kiểm tra trên tập độc lập. Ngưỡng riêng từng nhánh còn giả định biết trước loại nhánh, không thể dùng như một bộ phát hiện chung khi nhánh chưa biết.
