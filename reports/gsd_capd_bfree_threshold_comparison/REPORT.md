# GSD và CAPD: quét ngưỡng BFree validation

GSD: [test-gsd-bfree v1](https://www.kaggle.com/datasets/hoangha071005/test-gsd-bfree), checkpoint CLIP+GSD All7 augmented epoch 1. CAPD: [eval-capd-test-tiny-bfree v1](https://www.kaggle.com/datasets/hadinho0710/eval-capd-test-tiny-bfree), checkpoint DINOv3 CAPD Tiny All7 epoch 3.

## Giao thức và đối chiếu

- Không train lại hoặc chạy inference; tính lại quyết định fake nếu probability >= threshold từ CSV nguồn.
- Hai bên trùng đủ 10.304 COCO IDs cho real và từng nhánh fake, đủ 6 nhánh, tổng 72.128 ảnh riêng biệt. Chưa tính lại hash pixel để xác nhận nội dung ảnh byte-identical.
- GSD có 123.648 dòng vì real được suy luận lại trong từng branch/batch context. CAPD có 72.128 dòng và dùng lại cùng real score cho 6 cặp đánh giá. Giữ nguyên các score GSD theo context, không lấy trung bình rồi thay vào.
- BA = balanced accuracy; macro là trung bình 6 nhánh, mỗi nhánh 10.304 real + 10.304 fake. Recall real/FPR của GSD trong bảng là trung bình 6 context; số FP trung bình không phải số ảnh thật riêng biệt.
- Đối chiếu lại 18 confusion matrices ở ngưỡng 0,5 khớp metrics gốc. Có 22 real IDs đổi nhãn dự đoán giữa các context GSD ở ngưỡng 0,5.

## Quét cùng các ngưỡng

| Ngưỡng | GSD BA (%) | CAPD full BA (%) | CAPD window BA (%) | GSD recall fake (%) | GSD FPR (%) | CAPD full FPR (%) | CAPD window FPR (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 81.09 | 82.54 | 82.78 | 62.67 | 0.49 | 0.78 | 0.90 |
| 0.3 | 82.59 | 83.14 | 83.31 | 66.20 | 1.02 | 1.08 | 1.27 |
| 0.2 | 83.44 | 83.41 | 83.49 | 68.58 | 1.71 | 1.44 | 1.83 |
| 0.1 | 84.31 | 83.60 | 83.62 | 72.17 | 3.55 | 2.47 | 3.04 |
| 0.05 | 84.79 | 83.50 | 83.41 | 75.85 | 6.28 | 4.14 | 5.09 |
| 0.02 | 84.08 | 82.79 | 82.37 | 81.32 | 13.16 | 7.89 | 9.72 |
| 0.01 | 81.61 | 81.72 | 80.76 | 86.04 | 22.81 | 12.22 | 15.28 |
| 0.005 | 76.07 | 79.71 | 78.38 | 90.98 | 38.84 | 18.74 | 22.81 |
| 0.001 | 55.92 | 71.85 | 69.70 | 99.01 | 87.18 | 42.02 | 48.13 |

## Ngưỡng 0,5 và ngưỡng chung tối ưu macro BA

| Model | Cách chọn | Ngưỡng | Macro BA (%) | Recall fake (%) | Recall real (%) | FPR (%) | FP trung bình/nhánh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSD | Mặc định | 0.5 | 81.09 | 62.67 | 99.51 | 0.49 | 50.67 |
| GSD | Tối ưu trên validation | 0.041230023 | 84.81 | 76.92 | 92.71 | 7.29 | 751.67 |
| CAPD full | Mặc định | 0.5 | 82.54 | 65.85 | 99.22 | 0.78 | 80.00 |
| CAPD full | Tối ưu trên validation | 0.08452560598 | 83.66 | 70.02 | 97.30 | 2.70 | 278.00 |
| CAPD window | Mặc định | 0.5 | 82.78 | 66.46 | 99.10 | 0.90 | 93.00 |
| CAPD window | Tối ưu trên validation | 0.1138782528 | 83.66 | 70.02 | 97.30 | 2.70 | 278.00 |

Mỗi model dùng một ngưỡng chung cho cả 6 nhánh. Tìm chính xác trên toàn bộ ranh giới score; giữ nguyên nhóm score bằng nhau. Nếu nhiều ngưỡng cùng tối ưu, chọn ngưỡng cao nhất để ưu tiên ít false positive.

## Theo nhánh: dùng ngưỡng chung tối ưu riêng của mỗi model

| Nhánh | GSD BA (%) | CAPD full BA (%) | CAPD window BA (%) | GSD recall fake (%) | CAPD full recall fake (%) | CAPD window recall fake (%) |
| --- | --- | --- | --- | --- | --- | --- |
| inpainted_diffcat | 96.47 | 98.47 | 98.47 | 99.81 | 99.63 | 99.63 |
| inpainted_diffcat_origBG | 83.41 | 82.74 | 82.74 | 74.00 | 68.18 | 68.18 |
| inpainted_samecat | 96.35 | 98.15 | 98.15 | 99.76 | 99.00 | 99.00 |
| inpainted_samecat_origBG | 73.94 | 71.50 | 71.50 | 55.15 | 45.70 | 45.70 |
| selfconditioned | 95.62 | 94.01 | 94.01 | 98.62 | 90.72 | 90.72 |
| selfconditioned_origBG | 63.08 | 57.09 | 57.09 | 34.17 | 16.89 | 16.89 |

## So sánh ở cùng giới hạn false positive

Mỗi model được chọn một ngưỡng chung tối đa hóa macro recall fake dưới giới hạn macro FPR ghi trong bảng. FPR thực tế có thể thấp hơn giới hạn do score rời rạc; đây là ngưỡng tham khảo chọn trực tiếp trên cùng validation.

| FPR tối đa (%) | Model | Ngưỡng | FPR thực tế (%) | Recall fake (%) | Macro BA (%) |
| --- | --- | --- | --- | --- | --- |
| 1 | GSD | 0.3022881 | 1.00 | 66.16 | 82.58 |
| 1 | CAPD full | 0.3652157643 | 1.00 | 66.82 | 82.91 |
| 1 | CAPD window | 0.4446958439 | 1.00 | 66.82 | 82.91 |
| 5 | GSD | 0.066296645 | 5.00 | 74.33 | 84.67 |
| 5 | CAPD full | 0.03741311792 | 5.00 | 71.82 | 83.41 |
| 5 | CAPD window | 0.05132195059 | 5.00 | 71.82 | 83.41 |
| 10 | GSD | 0.02763772 | 9.99 | 79.29 | 84.65 |
| 10 | CAPD full | 0.01364253839 | 10.00 | 74.63 | 82.32 |
| 10 | CAPD window | 0.01888810667 | 10.00 | 74.63 | 82.32 |

## AUC/AP không phụ thuộc ngưỡng

| Model | Macro AUC (%) | Macro AP (%) |
| --- | --- | --- |
| GSD | 90.88 | 92.02 |
| CAPD full | 87.01 | 89.33 |
| CAPD window | 87.00 | 89.33 |

## Nhận xét từ kết quả

- GSD: ngưỡng 0,5 đạt BA 81.09%; giảm về 0.041230023 đạt 84.81%, tăng 3.72 điểm phần trăm.
- Trong các ngưỡng số tròn đã thử, GSD 0,05 đạt BA 84,79%, gần tối ưu 84,81%; 0,1 đạt 84,31% và FPR thấp hơn (3,55% so với 6,28%).
- Tối ưu riêng theo BA: GSD cao hơn CAPD 1,15 điểm phần trăm, nhưng FPR 7,29% so với 2,70%; vì thế cần xem thêm cùng ngân sách false positive.
- Ở FPR 1%, CAPD có recall fake cao hơn nhẹ (66,82% so với 66,16%). Ở FPR 5% và 10%, GSD cao hơn (74,33%/79,29% so với 71,82%/74,63%).
- Các nhánh origBG vẫn là điểm yếu. Ở ngưỡng BA tối ưu chung, selfconditioned_origBG recall fake GSD 34,17%, CAPD 16,89%; cả hai còn bỏ sót phần lớn ảnh giả của nhánh này.

## Giới hạn diễn giải

- Ngưỡng tối ưu và các điểm FPR ở đây được chọn bằng nhãn của chính BFree validation; không phải kết quả test độc lập. Muốn triển khai cần khóa ngưỡng sau calibration và đánh giá trên tập khác.
- So sánh là CLIP+GSD với DINOv3+CAPD; khác cả backbone và cách training/pooling, không đủ để quy toàn bộ chênh lệch cho riêng GSD hay CAPD.
- Một seed và không có ước lượng độ bất định trong báo cáo này. Giá trị probability cùng ngưỡng số học không biểu thị mức calibration tương đương; dùng thêm bảng cùng FPR để đối chiếu.

## File chạy

[compare_thresholds.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_capd_bfree_threshold_comparison/compare_thresholds.py) — đọc hai CSV local, kiểm tra cohort/baseline, xuất báo cáo và [comparison.json](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_capd_bfree_threshold_comparison/comparison.json).

SHA-256 hai CSV nguồn và các kết quả chi tiết từng nhánh/ngưỡng nằm trong comparison.json.
