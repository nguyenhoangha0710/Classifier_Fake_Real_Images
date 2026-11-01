# GSD train BFree → test Tiny-GenImage

Nguồn: [hoangha071005/gsd-train-bfree-test-tiny, version 1](https://www.kaggle.com/datasets/hoangha071005/gsd-train-bfree-test-tiny). Đối chiếu ngày 01/11/2026.

## Trạng thái và protocol

- Evaluation complete, smoke=false; đủ 9 case × 7.000 ảnh = 63.000 dự đoán, không có case partial; khoảng 2,72 giờ. Không có png_roundtrip trong lượt này.
- Model CLIP ViT-L/14 + GSD, train BFree riêng SD2.1_inpainted_samecat_origBG và real COCO, checkpoint best virtual epoch2, không phải model train phối hợp sáu nhánh.
- Checkpoint source: `/kaggle/input/datasets/hoangha071005/check-point-gsd-train-bfee-samecat-ori/gsd_bfree_samecat_origbg_v1/best.pt`; signature `06c7d714b84425bf3248247cb9c1f686c12e5221f530ae325a454b1f88247046`; SHA-256 theo metadata `dab2e8454abf3dc095ea26223ec8afd87f721d2b64752a1db999b1cac811c813`. Không tải weight để tính lại hash trong lượt audit này.
- Test Tiny dùng manifest hadinho0710/mainfest-train-genimage/seed42/test_manifest.csv: 7 generator; mỗi generator 500 real + 500 fake. Đây là cross-dataset evaluation từ BFree/SD2.1 sang Tiny, khác với train Tiny rồi test Tiny.
- Ngưỡng fake probability >=0,5; pooled accuracy = balanced accuracy do hai lớp cân bằng. Các bảng AUC/AP dưới đây dùng pooled 7.000 ảnh, khác macro trung bình AUC từng generator trong macro_all.csv.
- Đã tính lại 72 dòng metrics từ predictions: confusion counts, accuracy, BA, recalls, precision/F1, AUC/AP khớp nguồn trong sai số <1e-12. Đối chiếu cùng ID/label/generator/pixel_hash đã ghi, cùng thứ tự sample/batch giữa 9 case. Pixel_hash là hash nguồn lưu trong manifest, không tính lại ảnh sau biến đổi.
- Cohort native cũng khớp ID/label/pixel_hash với lượt GSD train Tiny augmented trước. Chưa audit near-duplicate/pixel overlap giữa BFree train và Tiny test.

## Kết quả pooled theo case

| Case | Accuracy/BA (%) | AUC (%) | AP (%) | Recall fake (%) | Recall real (%) | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 81.26 | 97.84 | 97.38 | 63.63 | 98.89 | 39 | 1273 |
| fake_jpeg96 | 78.54 | 97.00 | 96.57 | 58.17 | 98.91 | 38 | 1464 |
| all_jpeg96 | 78.49 | 96.87 | 96.39 | 58.11 | 98.86 | 40 | 1466 |
| all_jpeg90 | 78.31 | 95.60 | 95.19 | 58.26 | 98.37 | 57 | 1461 |
| all_jpeg80 | 77.36 | 93.42 | 93.10 | 57.60 | 97.11 | 101 | 1484 |
| resize_square256 | 69.79 | 87.03 | 88.67 | 41.51 | 98.06 | 68 | 2047 |
| resize_square560 | 79.74 | 96.98 | 97.06 | 59.97 | 99.51 | 17 | 1401 |
| blur_sigma1 | 84.33 | 96.49 | 96.30 | 71.34 | 97.31 | 94 | 1003 |
| resize256_jpeg96 | 68.60 | 86.40 | 88.26 | 38.80 | 98.40 | 56 | 2142 |

## Native theo generator

| Generator | Accuracy/BA (%) | AUC (%) | Recall fake (%) | Recall real (%) | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| adm | 74.40 | 96.74 | 49.60 | 99.20 | 4 | 252 |
| biggan | 84.60 | 99.26 | 69.80 | 99.40 | 3 | 151 |
| glide | 81.00 | 97.90 | 62.80 | 99.20 | 4 | 186 |
| midjourney | 67.90 | 95.29 | 36.60 | 99.20 | 4 | 317 |
| sd15 | 87.20 | 98.11 | 76.40 | 98.00 | 10 | 118 |
| vqdm | 85.50 | 98.85 | 72.20 | 98.80 | 6 | 139 |
| wukong | 88.20 | 99.08 | 78.00 | 98.40 | 8 | 110 |

## So với GSD train Tiny All7 augmented

| Case | GSD train BFree acc. (%) | GSD train Tiny acc. (%) | Chênh lệch (điểm %) |
| --- | --- | --- | --- |
| native | 81.26 | 99.86 | -18.60 |
| fake_jpeg96 | 78.54 | 99.67 | -21.13 |
| all_jpeg96 | 78.49 | 99.63 | -21.14 |
| all_jpeg90 | 78.31 | 99.49 | -21.17 |
| all_jpeg80 | 77.36 | 99.47 | -22.11 |
| resize_square256 | 69.79 | 92.20 | -22.41 |
| resize_square560 | 79.74 | 94.76 | -15.01 |
| blur_sigma1 | 84.33 | 99.67 | -15.34 |
| resize256_jpeg96 | 68.60 | 90.09 | -21.49 |

Hai model dùng cùng tập ảnh, nhưng GSD train Tiny đã thấy các generator Tiny trong training; model BFree chưa train trên Tiny. Bảng thể hiện sự khác nhau giữa khả năng trong miền và chuyển miền, chưa phải bằng chứng nhân quả cho riêng kiến trúc.

## Đánh giá

- Native: accuracy 81,26%, AUC 97,84%, AP 97,38%; TN=3.461, FP=39, FN=1.273, TP=2.227. Model ít báo nhầm real (1,11%) nhưng bỏ sót 36,37% fake ở ngưỡng 0,5.
- Midjourney khó nhất: BA 67,90%, fake recall 36,60%; ADM fake recall 49,60%. Wukong và SD1.5 đạt BA cao nhất 88,20% và 87,20%.
- JPEG làm giảm accuracy còn khoảng 77–79%. Resize256 và resize256+JPEG96 giảm mạnh còn 69,79% và 68,60%; AUC cũng giảm, nên ảnh hưởng không chỉ do ngưỡng. Blur sigma1 tăng accuracy lên 84,33% nhưng AUC giảm: không suy ra blur luôn cải thiện khả năng xếp hạng.
- Native AUC cao hơn accuracy khá nhiều gợi ý vấn đề calibration/ngưỡng khi chuyển miền; muốn chọn ngưỡng mới cần calibration độc lập, không chọn bằng nhãn Tiny test rồi gọi là kết quả test xác thực.
- Kết quả BFree tốt của cùng checkpoint không tự chuyển thành hiệu suất cao trên Tiny. Cần báo cáo riêng cross-dataset và in-domain; không gộp 9 biến đổi của cùng 7.000 ảnh thành 63.000 ảnh test độc lập.

## File chạy local

- [download_source.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_tiny_analysis/download_source.py) — tải metadata, bảng kết quả và predictions, pin version trong mỗi request.
- [analyze.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_tiny_analysis/analyze.py) — tính lại metrics, kiểm tra cohort và xuất báo cáo.
- [analysis.json](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_tiny_analysis/analysis.json) — số liệu chi tiết/hashes.
- [Notebook test Tiny9](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/test_gsd_bfree_samecat_origbg_tiny9_kaggle.ipynb).
