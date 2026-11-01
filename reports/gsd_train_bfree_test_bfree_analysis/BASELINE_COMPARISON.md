# Đánh giá GSD train BFree so với GSD train Tiny và baseline

## Phạm vi so sánh

- Kết quả mới: CLIP ViT-L/14+GSD train BFree riêng SD2.1_inpainted_samecat_origBG, best virtual epoch2; checkpoint training paused. Evaluation hoàn tất 5/6 nhánh; diffcat_origBG mới có 7.024/10.304 cặp.
- Bảng chính chỉ dùng cùng 5 nhánh đã hoàn tất, cùng COCO IDs/labels. GSD/CAPD dùng probability >=0,5. Twins/GMM dùng score NLL (cao hơn = fake) và pred_q95 từ ngưỡng real-calibration đã lưu, không dùng ngưỡng probability 0,5 cho NLL.
- Đối chiếu theo ID và nhãn, chưa audit lại hash pixel. GSD giữ score theo context gốc; lọc predictions để so cohort không đồng nghĩa chạy lại GSD với batch context mới.
- GSD BFree đã thấy SD2.1/BFree khi training, còn GSD Tiny/CAPD Tiny chuyển miền từ Tiny sang BFree. Đây là so sánh các hệ thống và mức thích nghi dữ liệu, không phải ablation chỉ thay kiến trúc.

## 1. Cùng 10.304 ID, cùng 5 nhánh đủ, ngưỡng 0,5

| Model | Macro BA (%) | Macro AUC (%) | Macro AP (%) | Recall fake (%) | Recall real (%) |
| --- | --- | --- | --- | --- | --- |
| GSD train BFree | 89.66 | 98.89 | 98.98 | 79.58 | 99.73 |
| GSD train Tiny | 82.11 | 90.87 | 91.87 | 64.70 | 99.51 |
| CAPD full | 82.83 | 87.00 | 89.05 | 66.43 | 99.22 |
| CAPD window | 83.04 | 87.00 | 89.05 | 66.99 | 99.10 |

| Nhánh | GSD BFree BA (%) | GSD Tiny BA (%) | CAPD full BA (%) | CAPD window BA (%) |
| --- | --- | --- | --- | --- |
| selfconditioned | 80.73 | 93.71 | 92.28 | 92.67 |
| selfconditioned_origBG | 73.34 | 56.12 | 55.00 | 55.37 |
| inpainted_samecat | 97.00 | 97.34 | 98.64 | 98.66 |
| inpainted_samecat_origBG | 99.72 | 64.84 | 68.99 | 69.32 |
| inpainted_diffcat | 97.51 | 98.52 | 99.23 | 99.20 |

### AUC từng nhánh (không phụ thuộc ngưỡng)

| Nhánh | GSD BFree AUC (%) | GSD Tiny AUC (%) | CAPD full AUC (%) | CAPD window AUC (%) |
| --- | --- | --- | --- | --- |
| selfconditioned | 99.12 | 99.49 | 97.75 | 97.75 |
| selfconditioned_origBG | 95.47 | 71.89 | 61.06 | 61.06 |
| inpainted_samecat | 99.93 | 99.84 | 99.79 | 99.79 |
| inpainted_samecat_origBG | 99.99 | 83.19 | 76.49 | 76.48 |
| inpainted_diffcat | 99.93 | 99.91 | 99.91 | 99.91 |

## 2. Cùng cohort clean 1.557 ID, cùng 5 nhánh

Lọc predictions GSD/CAPD về đúng tập ID của Twins native và codec. Hai bộ Twins có ID trùng nhau, và đều là tập con của official-valid 10.304 ID. Mỗi nhánh có 1.557 real + 1.557 fake. Đây là cohort loại overlap với real200K của Twins; không dùng kết quả Twins trên tập lớn có overlap để xếp hạng.

| Model | Macro BA (%) | Macro AUC (%) | Macro AP (%) | Recall fake (%) | Recall real (%) |
| --- | --- | --- | --- | --- | --- |
| GSD train BFree | 89.59 | 98.90 | 98.95 | 79.37 | 99.81 |
| GSD train Tiny | 81.84 | 90.68 | 91.47 | 64.64 | 99.05 |
| CAPD full | 82.65 | 87.14 | 89.09 | 66.51 | 98.78 |
| Twins native GMM | 46.01 | 45.19 | 44.89 | 26.44 | 65.57 |
| Twins codec GMM | 40.58 | 37.03 | 40.87 | 18.79 | 62.36 |
| CAPD window | 82.88 | 87.13 | 89.08 | 66.97 | 98.78 |

Các cấu hình Twins ở đây là **native K=1 full GMM** và **codec encoder + native GMM**; chưa đại diện mọi lựa chọn mixed/augmented-GMM. AUC <50% là kết quả theo chiều NLL cao=fake đã định trước. Không tự đảo score sau khi nhìn nhãn để báo kết quả đẹp hơn.

## 3. Nhạy với ngưỡng: cùng 5 nhánh trên full valid

| Model | Ngưỡng tối ưu BA | BA (%) | Recall fake (%) | FPR (%) | Recall fake @FPR≤5% (%) |
| --- | --- | --- | --- | --- | --- |
| GSD train BFree | 0.029367857 | 95.72 | 95.24 | 3.79 | 96.14 |
| GSD train Tiny | 0.05465104 | 85.14 | 76.15 | 5.87 | 75.20 |
| CAPD full | 0.0913362653 | 83.85 | 70.26 | 2.56 | 72.12 |
| CAPD window | 0.113878253 | 83.85 | 70.39 | 2.70 | 72.12 |

Tối ưu dùng một ngưỡng chung trên 5 nhánh và chính nhãn validation. Đây là diagnostic in-sample, không phải kết quả test độc lập hoặc threshold đã được xác thực. Không so các số này với báo cáo tối ưu 6 nhánh trước đó vì tập nhánh khác nhau.

## 4. Nhánh diffcat_origBG chưa đủ: cùng 7.024 ID đã có

| Model | Macro BA (%) | Macro AUC (%) | Macro AP (%) | Recall fake (%) | Recall real (%) |
| --- | --- | --- | --- | --- | --- |
| GSD train BFree | 98.92 | 99.95 | 99.95 | 98.09 | 99.74 |
| GSD train Tiny | 75.83 | 91.04 | 92.82 | 52.16 | 99.50 |
| CAPD full | 80.77 | 86.81 | 90.57 | 62.30 | 99.23 |
| CAPD window | 81.12 | 86.80 | 90.57 | 63.14 | 99.10 |

Đã lọc ba baseline về đúng 7.024 ID của partial mới. Đây là tập đã xử lý, không phải mẫu ngẫu nhiên hay kết quả đầy đủ 10.304 ID. Không cộng vào macro 5 nhánh hoặc diễn giải thành macro đủ 6 nhánh.

## 5. CLIP probe và AIDE: số tham khảo khác cohort

Nguồn là output đã lưu trong notebook baselines/NhatNguyen/. Hai baseline dùng 2.217 ID chọn từ train_list.csv, còn GSD/CAPD dùng official valid 10.304 ID. Chưa có predictions/manifest joinable trong lần tổng hợp này để lọc cùng ID. Bảng dưới chỉ lấy cùng tên 5 nhánh; số gốc đã làm tròn, không xếp hạng head-to-head với bảng 1/2.

| Model | ID/nhánh | BA trung bình 5 nhánh (%) | AUC trung bình 5 nhánh (%) |
| --- | --- | --- | --- |
| CLIP ViT-L/14 probe | 2217 | 61.62 | 69.07 |
| AIDE original full | 2217 | 58.10 | 69.14 |

- CLIP notebook này là ViT-L/14 + probe; không nhầm với OpenCLIP ViT-B-32 probe ở benchmark Tiny JPEG-bias.
- AIDE là original full, không phải AIDE forensic-only ResNet50. Không chuyển kết luận nhân quả trong output notebook thành kết luận của báo cáo này.
- Chưa có kết quả BFree được xác minh cho NPR-ResNet18, ResNet50, NPR patch, AIDE+NPR fusion, Qwen hoặc Spectral-GMM trong các artifact đã kiểm tra; không dùng điểm Tiny/CommFor để điền bảng BFree.

## Đánh giá kết quả

- Trên cùng 5 nhánh và 10.304 ID, GSD train BFree tăng BA 7,55 điểm phần trăm và AUC 8,02 điểm so với GSD train Tiny; hơn CAPD window 6,61 điểm BA. AUC tăng cho thấy chênh lệch không chỉ do chọn ngưỡng 0,5.
- Tăng tập trung ở nhánh train samecat_origBG (+34,87 điểm BA so với GSD Tiny) và selfconditioned_origBG (+17,22 điểm), nhưng selfconditioned giảm 12,98 điểm. Hai nhánh inpainting không origBG cũng thấp hơn GSD Tiny/CAPD tại ngưỡng 0,5; chưa phải model thắng đồng đều.
- Selfconditioned của model mới vẫn có AUC 99,12% nhưng recall fake ở 0,5 chỉ 61,75%: đây là dấu hiệu cần phân tích calibration/ngưỡng, không thể chỉ nhìn accuracy để nói mất hoàn toàn khả năng phân biệt.
- Với một ngưỡng chung tối ưu trên chính 5 nhánh validation, GSD BFree đạt BA 95,72% tại 0,029367857, recall fake 95,24%, FPR 3,79%. Đây chỉ là diagnostic có thiên lệch lựa chọn; không được đưa lên bảng kết quả test chính như threshold cố định đã xác thực.
- Baseline Twins native/codec với NLL và ngưỡng calibration đã lưu chuyển sang BFree kém trên cohort clean. Kết luận chỉ áp dụng các cấu hình đã đọc; cần kiểm tra score/calibration/domain shift trước khi quy lỗi cho kiến trúc, và không đảo score theo nhãn test.
- Muốn chứng minh generalization cần hoàn tất nhánh cuối, đánh giá Tiny cùng checkpoint GSD-BFree và các generator ngoài SD2.1. Bảng hiện tại chứng minh khả năng thích nghi với BFree một nhánh trên ID không trùng training, chưa chứng minh tốt hơn trên mọi miền.

## Nguồn và file chạy

- [GSD train BFree → BFree v1](https://www.kaggle.com/datasets/hoangha071005/test-gsd-train-bfree-test-bfree).
- [GSD train Tiny → BFree v1](https://www.kaggle.com/datasets/hoangha071005/test-gsd-bfree).
- [CAPD train Tiny → BFree v1](https://www.kaggle.com/datasets/hadinho0710/eval-capd-test-tiny-bfree).
- [Twins augmented-GMM evaluation v2](https://www.kaggle.com/datasets/hoangha071005/evaluation-twins-bfree), model protocol_native_k1_full.
- [Twins codec evaluation v1](https://www.kaggle.com/datasets/hoangha071005/eval-forensic-twins-codec), protocol native_png.
- [Script so sánh](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_bfree_analysis/compare_baselines.py), [JSON chi tiết/hashes](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_bfree_analysis/baseline_comparison.json).
- [CLIP ViT-L/14 probe](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/NhatNguyen/forensight-clip-bfree.ipynb).
- [AIDE original full](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/NhatNguyen/forensight-aide-bfree.ipynb).
