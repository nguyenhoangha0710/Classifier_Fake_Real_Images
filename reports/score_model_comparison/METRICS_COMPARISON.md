# So sánh metrics theo tập TRAIN → tập TEST

Tất cả chỉ số là %. Model xác suất dùng ngưỡng0,5; CAPD dùng full pooling. Accuracy=BA ở từng cohort cân bằng. Không trộn các kết quả tối ưu ngưỡng bằng nhãn test vào bảng mặc định.

## I. TRAIN TRÊN TINY-GENIMAGE

### I.1. TEST trên Tiny — native, cùng7.000 ảnh

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD | 99.90 | 100.00 | 100.00 | 99.97 | 99.83 | 99.83 | 99.97 | 99.90 | 99.90 |
| GSD | 99.86 | 100.00 | 100.00 | 99.97 | 99.74 | 99.74 | 99.97 | 99.86 | 99.86 |
| CLIP ViT-B/32 probe | 85.79 | 94.02 | 93.88 | 91.97 | 79.60 | 81.85 | 90.84 | 86.61 | 84.85 |
| NPR-ResNet18 | 96.60 | 99.53 | 99.48 | 97.20 | 96.00 | 96.05 | 97.17 | 96.62 | 96.58 |
| AIDE original full | 92.30 | 98.38 | 98.37 | 87.94 | 96.66 | 96.34 | 88.91 | 91.95 | 92.62 |
| AIDE forensic-only ResNet50 | 92.47 | 97.55 | 97.55 | 86.80 | 98.14 | 97.91 | 88.14 | 92.02 | 92.88 |

Đây là đánh giá trong miền đối với các model train Tiny. AIDE full và forensic-only là hai biến thể riêng. AUC/AP100,00 là làm tròn, không có nghĩa không có lỗi.

### I.2. TEST trên BFree — mean trên cùng5 nhánh, cùng10.304 ID

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD | 82.83 | 87.00 | 89.05 | 66.43 | 99.22 | 97.77 | 79.79 | 72.89 | 87.13 |
| GSD | 82.11 | 90.87 | 91.87 | 64.70 | 99.51 | 98.60 | 78.83 | 71.56 | 86.66 |

Đây là chuyển miền Tiny→BFree. Bảng chính giữ cùng5 nhánh như nhóm train BFree bên dưới để đối chiếu nhất quán. Chưa có predictions BFree cùng cohort được xác minh cho CLIP ViT-B/32 probe, NPR, AIDE full/forensic-only ở bảng I.1; không điền điểm Tiny hoặc notebook khác checkpoint/cohort vào đây.

Kết quả bổ sung trên đủ6 nhánh của hai model train Tiny (không dùng để so với nhóm chỉ đủ5):

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD | 82.54 | 87.01 | 89.33 | 65.85 | 99.22 | 97.94 | 78.62 | 73.56 | 86.61 |
| GSD | 81.09 | 90.88 | 92.02 | 62.67 | 99.51 | 98.67 | 76.97 | 71.07 | 85.64 |

### I.3. TEST Tiny qua các biến đổi — mean/median cùng9 case CAPD/GSD

mean

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD | 98.15 | 99.89 | 99.89 | 99.70 | 96.60 | 97.00 | 99.67 | 98.26 | 98.02 |
| GSD | 97.20 | 99.22 | 99.37 | 95.99 | 98.41 | 98.36 | 96.29 | 97.10 | 97.29 |

median

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD | 99.83 | 99.99 | 99.99 | 99.91 | 99.83 | 99.83 | 99.91 | 99.83 | 99.83 |
| GSD | 99.49 | 99.99 | 99.99 | 99.40 | 99.60 | 99.60 | 99.40 | 99.49 | 99.49 |

## II. TRAIN TRÊN BFREE

CAPD và GSD trong nhóm này train real COCO + fake SD2.1_inpainted_samecat_origBG. Khác backbone: CAPD-DINOv3 so với CLIP+GSD. Không phải recipe train phối hợp6 nhánh của paper B-Free.

### II.1. TEST trên Tiny — native, cùng7.000 ảnh

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD | 90.31 | 98.50 | 98.80 | 81.09 | 99.54 | 99.44 | 84.03 | 89.33 | 91.13 |
| GSD | 81.26 | 97.84 | 97.38 | 63.63 | 98.89 | 98.28 | 73.11 | 77.25 | 84.07 |

Đây là chuyển miền BFree→Tiny. Không đặt cùng bảng xếp hạng trong miền của nhóm I mà không ghi khác biệt dữ liệu train.

### II.2. TEST trên BFree — mean trên cùng5 nhánh hoàn tất, cùng10.304 ID

[Bảng riêng từng case với đầy đủ metrics và nhánh partial matched](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/score_model_comparison/TRAIN_BFREE_TEST_BFREE_CASES.md).

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD | 93.91 | 98.60 | 98.91 | 88.14 | 99.67 | 99.60 | 91.69 | 92.13 | 94.98 |
| GSD | 89.66 | 98.89 | 98.98 | 79.58 | 99.73 | 99.62 | 85.45 | 86.82 | 91.41 |

GSD train BFree còn partial ở diffcat_origBG, nên bảng chính I.2 và II.2 đều dùng cùng5 nhánh. Không dùng kết quả đủ6 nhánh bổ sung ở I.2 để trừ trực tiếp với macro5 của nhóm này.

### II.3. TEST Tiny qua các biến đổi — mean/median cùng9 case CAPD/GSD

mean

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD | 88.18 | 97.97 | 98.38 | 76.79 | 99.56 | 99.45 | 81.31 | 86.51 | 89.46 |
| GSD | 77.38 | 94.18 | 94.32 | 56.38 | 98.38 | 97.13 | 69.59 | 70.87 | 81.43 |

median

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD | 89.80 | 98.40 | 98.64 | 80.14 | 99.54 | 99.43 | 83.36 | 88.71 | 90.70 |
| GSD | 78.49 | 96.49 | 96.30 | 58.17 | 98.40 | 97.28 | 70.24 | 72.98 | 82.13 |

### II.4. TEST BFree — nhánh cuối partial, cùng7.024 ID

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD | 99.59 | 99.99 | 99.99 | 99.54 | 99.64 | 99.64 | 99.54 | 99.59 | 99.59 |
| GSD | 98.92 | 99.95 | 99.95 | 98.09 | 99.74 | 99.74 | 98.12 | 98.91 | 98.93 |

## III. PHỤ LỤC ĐỐI CHIẾU COHORT VÀ TỪNG CASE

Twins/GMM và Spectral train real-only từ COCO/ImageNet, không tự xếp vào nhóm train Tiny hay train BFree. Các cấu hình, ngưỡng và cohort riêng được ghi dưới đây.


**Tất cả bảng là chỉ số đánh giá, đơn vị %.** Probability model dùng ngưỡng0,5; không trộn với kết quả chọn ngưỡng hậu nghiệm. Accuracy=BA trong các cohort cân bằng, nên gộp một cột. Mean/median được tính từ metrics từng case, không phải score.

## 1. Tiny native — cùng7.000 ảnh/nhãn

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 90.31 | 98.50 | 98.80 | 81.09 | 99.54 | 99.44 | 84.03 | 89.33 | 91.13 |
| CAPD train Tiny | 99.90 | 100.00 | 100.00 | 99.97 | 99.83 | 99.83 | 99.97 | 99.90 | 99.90 |
| GSD train BFree | 81.26 | 97.84 | 97.38 | 63.63 | 98.89 | 98.28 | 73.11 | 77.25 | 84.07 |
| GSD train Tiny | 99.86 | 100.00 | 100.00 | 99.97 | 99.74 | 99.74 | 99.97 | 99.86 | 99.86 |
| CLIP ViT-B/32 probe | 85.79 | 94.02 | 93.88 | 91.97 | 79.60 | 81.85 | 90.84 | 86.61 | 84.85 |
| NPR-ResNet18 | 96.60 | 99.53 | 99.48 | 97.20 | 96.00 | 96.05 | 97.17 | 96.62 | 96.58 |
| AIDE original full | 92.30 | 98.38 | 98.37 | 87.94 | 96.66 | 96.34 | 88.91 | 91.95 | 92.62 |
| AIDE forensic-only ResNet50 | 92.47 | 97.55 | 97.55 | 86.80 | 98.14 | 97.91 | 88.14 | 92.02 | 92.88 |

Có3.500 real +3.500 fake/model. CAPD là full pooling. CLIP probe là ViT-B/32, AIDE full khác AIDE forensic-only. Model train Tiny đã thấy các generator Tiny, còn model train BFree đang chuyển miền, không phải ablation chỉ thay kiến trúc.

## 2. Cùng9 case Tiny có đủ CAPD/GSD

### mean

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 88.18 | 97.97 | 98.38 | 76.79 | 99.56 | 99.45 | 81.31 | 86.51 | 89.46 |
| CAPD train Tiny | 98.15 | 99.89 | 99.89 | 99.70 | 96.60 | 97.00 | 99.67 | 98.26 | 98.02 |
| GSD train BFree | 77.38 | 94.18 | 94.32 | 56.38 | 98.38 | 97.13 | 69.59 | 70.87 | 81.43 |
| GSD train Tiny | 97.20 | 99.22 | 99.37 | 95.99 | 98.41 | 98.36 | 96.29 | 97.10 | 97.29 |

### median

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 89.80 | 98.40 | 98.64 | 80.14 | 99.54 | 99.43 | 83.36 | 88.71 | 90.70 |
| CAPD train Tiny | 99.83 | 99.99 | 99.99 | 99.91 | 99.83 | 99.83 | 99.91 | 99.83 | 99.83 |
| GSD train BFree | 78.49 | 96.49 | 96.30 | 58.17 | 98.40 | 97.28 | 70.24 | 72.98 | 82.13 |
| GSD train Tiny | 99.49 | 99.99 | 99.99 | 99.40 | 99.60 | 99.60 | 99.40 | 99.49 | 99.49 |

Gồm all_jpeg80, all_jpeg90, all_jpeg96, blur_sigma1, fake_jpeg96, native, resize256_jpeg96, resize_square256, resize_square560. Mỗi case được trọng số bằng nhau; không gộp prediction để tính AUC của hỗn hợp9 biến đổi.

## 3. Cùng3 case Tiny có đủ baseline CLIP/NPR/AIDE full

### mean

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 90.00 | 98.57 | 98.83 | 80.46 | 99.54 | 99.44 | 83.59 | 88.94 | 90.87 |
| CAPD train Tiny | 99.88 | 100.00 | 100.00 | 99.93 | 99.83 | 99.83 | 99.93 | 99.88 | 99.88 |
| GSD train BFree | 79.43 | 97.24 | 96.78 | 59.97 | 98.89 | 98.17 | 71.21 | 74.43 | 82.79 |
| GSD train Tiny | 99.72 | 99.99 | 99.99 | 99.72 | 99.71 | 99.71 | 99.72 | 99.72 | 99.72 |
| CLIP ViT-B/32 probe | 83.67 | 91.62 | 91.38 | 87.65 | 79.69 | 81.17 | 86.68 | 84.27 | 83.01 |
| NPR-ResNet18 | 70.88 | 73.41 | 75.74 | 45.79 | 95.97 | 87.51 | 68.76 | 53.78 | 78.57 |
| AIDE original full | 90.48 | 97.89 | 97.86 | 84.29 | 96.68 | 96.20 | 86.06 | 89.83 | 91.05 |

### median

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 89.84 | 98.60 | 98.84 | 80.14 | 99.54 | 99.43 | 83.37 | 88.75 | 90.74 |
| CAPD train Tiny | 99.87 | 100.00 | 100.00 | 99.91 | 99.83 | 99.83 | 99.91 | 99.87 | 99.87 |
| GSD train BFree | 78.54 | 97.00 | 96.57 | 58.17 | 98.89 | 98.17 | 70.28 | 73.05 | 82.17 |
| GSD train Tiny | 99.67 | 99.99 | 99.99 | 99.60 | 99.74 | 99.74 | 99.60 | 99.67 | 99.67 |
| CLIP ViT-B/32 probe | 82.67 | 90.49 | 90.23 | 85.49 | 79.60 | 80.93 | 84.62 | 83.15 | 82.17 |
| NPR-ResNet18 | 58.04 | 60.39 | 64.04 | 20.09 | 96.00 | 83.39 | 54.57 | 32.37 | 69.59 |
| AIDE original full | 89.59 | 97.70 | 97.65 | 82.46 | 96.66 | 96.17 | 84.65 | 88.79 | 90.28 |

Gồm native, fake_jpeg96, all_jpeg96. Không dùng mean9 case của CAPD/GSD so với mean3 case của baseline. Khớp ảnh gốc/nhãn; pipeline tiền xử lý của từng model vẫn khác nhau.

## 4. BFree — cùng5 nhánh hoàn tất và10.304 ID

### mean

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 93.91 | 98.60 | 98.91 | 88.14 | 99.67 | 99.60 | 91.69 | 92.13 | 94.98 |
| CAPD train Tiny | 82.83 | 87.00 | 89.05 | 66.43 | 99.22 | 97.77 | 79.79 | 72.89 | 87.13 |
| GSD train BFree | 89.66 | 98.89 | 98.98 | 79.58 | 99.73 | 99.62 | 85.45 | 86.82 | 91.41 |
| GSD train Tiny | 82.11 | 90.87 | 91.87 | 64.70 | 99.51 | 98.60 | 78.83 | 71.56 | 86.66 |

### median

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 99.77 | 99.99 | 99.99 | 99.87 | 99.67 | 99.67 | 99.87 | 99.77 | 99.77 |
| CAPD train Tiny | 92.28 | 97.75 | 98.24 | 85.35 | 99.22 | 99.10 | 87.13 | 91.71 | 92.79 |
| GSD train BFree | 97.00 | 99.93 | 99.92 | 94.24 | 99.75 | 99.73 | 94.55 | 96.91 | 97.08 |
| GSD train Tiny | 93.71 | 99.49 | 99.50 | 87.90 | 99.51 | 99.45 | 89.16 | 93.32 | 94.05 |

Mỗi nhánh10.304 real +10.304 fake. GSD train BFree chỉ đủ5 nhánh nên không dùng macro6 của model khác để đối chiếu. Mean precision/F1 được tính từng nhánh trước rồi lấy trung bình.

## 5. BFree — từng nhánh đầy đủ

### SD2.1_selfconditioned

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 95.90 | 99.79 | 99.80 | 92.14 | 99.67 | 99.64 | 92.69 | 95.74 | 96.05 |
| CAPD train Tiny | 92.28 | 97.75 | 98.24 | 85.35 | 99.22 | 99.10 | 87.13 | 91.71 | 92.79 |
| GSD train BFree | 80.73 | 99.12 | 99.05 | 61.75 | 99.70 | 99.52 | 72.27 | 76.21 | 83.80 |
| GSD train Tiny | 93.71 | 99.49 | 99.50 | 87.90 | 99.51 | 99.45 | 89.16 | 93.32 | 94.05 |

### SD2.1_selfconditioned_origBG

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 74.30 | 93.22 | 94.74 | 48.93 | 99.67 | 99.33 | 66.12 | 65.57 | 79.50 |
| CAPD train Tiny | 55.00 | 61.06 | 65.34 | 10.78 | 99.22 | 93.28 | 52.65 | 19.33 | 68.80 |
| GSD train BFree | 73.34 | 95.47 | 96.02 | 47.00 | 99.68 | 99.32 | 65.29 | 63.81 | 78.90 |
| GSD train Tiny | 56.12 | 71.89 | 74.33 | 12.76 | 99.48 | 96.06 | 53.28 | 22.53 | 69.39 |

### SD2.1_inpainted_samecat

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 99.77 | 100.00 | 100.00 | 99.87 | 99.67 | 99.67 | 99.87 | 99.77 | 99.77 |
| CAPD train Tiny | 98.64 | 99.79 | 99.83 | 98.05 | 99.22 | 99.21 | 98.07 | 98.63 | 98.64 |
| GSD train BFree | 97.00 | 99.93 | 99.92 | 94.24 | 99.75 | 99.73 | 94.55 | 96.91 | 97.08 |
| GSD train Tiny | 97.34 | 99.84 | 99.84 | 95.12 | 99.56 | 99.54 | 95.33 | 97.28 | 97.40 |

### SD2.1_inpainted_samecat_origBG

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 99.77 | 99.99 | 99.99 | 99.87 | 99.67 | 99.67 | 99.87 | 99.77 | 99.77 |
| CAPD train Tiny | 68.99 | 76.49 | 81.92 | 38.75 | 99.22 | 98.04 | 61.83 | 55.55 | 76.19 |
| GSD train BFree | 99.72 | 99.99 | 99.99 | 99.66 | 99.78 | 99.78 | 99.66 | 99.72 | 99.72 |
| GSD train Tiny | 64.84 | 83.19 | 85.79 | 30.15 | 99.53 | 98.48 | 58.76 | 46.17 | 73.90 |

### SD2.1_inpainted_diffcat

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 99.79 | 100.00 | 100.00 | 99.90 | 99.67 | 99.67 | 99.90 | 99.79 | 99.79 |
| CAPD train Tiny | 99.23 | 99.91 | 99.92 | 99.24 | 99.22 | 99.22 | 99.24 | 99.23 | 99.23 |
| GSD train BFree | 97.51 | 99.93 | 99.93 | 95.26 | 99.75 | 99.74 | 95.47 | 97.45 | 97.56 |
| GSD train Tiny | 98.52 | 99.91 | 99.92 | 97.57 | 99.48 | 99.47 | 97.62 | 98.51 | 98.54 |

## 6. BFree — nhánh diffcat_origBG partial, cùng7.024 ID

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 99.59 | 99.99 | 99.99 | 99.54 | 99.64 | 99.64 | 99.54 | 99.59 | 99.59 |
| CAPD train Tiny | 80.77 | 86.81 | 90.57 | 62.30 | 99.23 | 98.78 | 72.47 | 76.41 | 83.76 |
| GSD train BFree | 98.92 | 99.95 | 99.95 | 98.09 | 99.74 | 99.74 | 98.12 | 98.91 | 98.93 |
| GSD train Tiny | 75.83 | 91.04 | 92.82 | 52.16 | 99.50 | 99.05 | 67.53 | 68.34 | 80.46 |

Đã lọc các model về đúng ID đã chạy ở GSD train BFree; đây không phải kết quả đủ10.304 ID của nhánh này.

## 7. Baseline Twins/GMM — cùng cohort clean1.557 ID ×5 nhánh

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 93.55 | 98.58 | 98.86 | 87.49 | 99.61 | 99.53 | 91.17 | 91.71 | 94.66 |
| CAPD train Tiny | 82.65 | 87.14 | 89.09 | 66.51 | 98.78 | 96.60 | 79.73 | 72.82 | 86.92 |
| GSD train BFree | 89.59 | 98.90 | 98.95 | 79.37 | 99.81 | 99.72 | 85.33 | 86.70 | 91.37 |
| GSD train Tiny | 81.84 | 90.68 | 91.47 | 64.64 | 99.05 | 97.36 | 78.71 | 71.36 | 86.40 |
| Twins native GMM | 46.01 | 45.19 | 44.89 | 26.44 | 65.57 | 43.25 | 47.16 | 32.79 | 54.85 |
| Twins codec GMM | 40.58 | 37.03 | 40.87 | 18.79 | 62.36 | 33.24 | 43.44 | 24.01 | 51.21 |

Twins dùng NLL theo chiều đã định cao=fake, quyết định pred_q95 từ real calibration; không áp dụng threshold0,5 cho NLL hoặc đảo score sau khi nhìn nhãn. Tất cả model được lọc đúng cùng ID/nhánh; không so với bảng10.304 ID như cùng cohort.

## 8. Spectral/MFM-GMM — Tiny pipeline riêng

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spectral/MFM-GMM | 54.54 | 77.45 | 72.41 | 13.11 | 95.97 | 76.50 | 52.48 | 22.39 | 67.86 |

Cùng7.000 ảnh gốc nhưng Spectral có JPEG xử lý trước và fixed-grid tiles. Ngưỡng artifact_primary_prediction được lưu từ pipeline; không phải probability0,5, không phải Tiny native pixel pipeline giống hệt các model trên.

## 9. Tiny từng case có prediction

### all_jpeg80

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 87.49 | 97.14 | 97.60 | 75.66 | 99.31 | 99.10 | 80.31 | 85.81 | 88.81 |
| CAPD train Tiny | 99.79 | 99.99 | 99.99 | 99.71 | 99.86 | 99.86 | 99.71 | 99.79 | 99.79 |
| GSD train BFree | 77.36 | 93.42 | 93.10 | 57.60 | 97.11 | 95.23 | 69.61 | 71.78 | 81.09 |
| GSD train Tiny | 99.47 | 99.98 | 99.98 | 99.34 | 99.60 | 99.60 | 99.34 | 99.47 | 99.47 |

### all_jpeg90

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 89.51 | 98.40 | 98.64 | 79.63 | 99.40 | 99.25 | 82.99 | 88.36 | 90.46 |
| CAPD train Tiny | 99.90 | 100.00 | 99.99 | 99.91 | 99.89 | 99.89 | 99.91 | 99.90 | 99.90 |
| GSD train BFree | 78.31 | 95.60 | 95.19 | 58.26 | 98.37 | 97.28 | 70.21 | 72.87 | 81.94 |
| GSD train Tiny | 99.49 | 99.99 | 99.99 | 99.40 | 99.57 | 99.57 | 99.40 | 99.49 | 99.49 |

### all_jpeg96

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 89.84 | 98.61 | 98.85 | 80.14 | 99.54 | 99.43 | 83.37 | 88.75 | 90.74 |
| CAPD train Tiny | 99.87 | 100.00 | 100.00 | 99.91 | 99.83 | 99.83 | 99.91 | 99.87 | 99.87 |
| GSD train BFree | 78.49 | 96.87 | 96.39 | 58.11 | 98.86 | 98.07 | 70.24 | 72.98 | 82.13 |
| GSD train Tiny | 99.63 | 99.99 | 99.99 | 99.60 | 99.66 | 99.66 | 99.60 | 99.63 | 99.63 |
| CLIP ViT-B/32 probe | 82.67 | 90.49 | 90.23 | 85.49 | 79.86 | 80.93 | 84.62 | 83.15 | 82.17 |
| NPR-ResNet18 | 58.00 | 60.39 | 64.04 | 20.09 | 95.91 | 83.10 | 54.55 | 32.35 | 69.55 |
| AIDE original full | 89.59 | 97.70 | 97.65 | 82.46 | 96.71 | 96.17 | 84.65 | 88.79 | 90.28 |

### blur_sigma1

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 89.80 | 97.91 | 98.32 | 80.14 | 99.46 | 99.33 | 83.36 | 88.71 | 90.70 |
| CAPD train Tiny | 99.83 | 99.99 | 99.99 | 99.91 | 99.74 | 99.74 | 99.91 | 99.83 | 99.83 |
| GSD train BFree | 84.33 | 96.49 | 96.30 | 71.34 | 97.31 | 96.37 | 77.25 | 81.99 | 86.13 |
| GSD train Tiny | 99.67 | 99.99 | 99.99 | 99.63 | 99.71 | 99.71 | 99.63 | 99.67 | 99.67 |

### fake_jpeg96

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 89.84 | 98.60 | 98.84 | 80.14 | 99.54 | 99.43 | 83.37 | 88.75 | 90.74 |
| CAPD train Tiny | 99.87 | 100.00 | 100.00 | 99.91 | 99.83 | 99.83 | 99.91 | 99.87 | 99.87 |
| GSD train BFree | 78.54 | 97.00 | 96.57 | 58.17 | 98.91 | 98.17 | 70.28 | 73.05 | 82.17 |
| GSD train Tiny | 99.67 | 99.99 | 99.99 | 99.60 | 99.74 | 99.74 | 99.60 | 99.67 | 99.67 |
| CLIP ViT-B/32 probe | 82.54 | 90.36 | 90.04 | 85.49 | 79.60 | 80.73 | 84.58 | 83.04 | 82.01 |
| NPR-ResNet18 | 58.04 | 60.31 | 63.70 | 20.09 | 96.00 | 83.39 | 54.57 | 32.37 | 69.59 |
| AIDE original full | 89.56 | 97.61 | 97.55 | 82.46 | 96.66 | 96.10 | 84.64 | 88.76 | 90.25 |

### native

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 90.31 | 98.50 | 98.80 | 81.09 | 99.54 | 99.44 | 84.03 | 89.33 | 91.13 |
| CAPD train Tiny | 99.90 | 100.00 | 100.00 | 99.97 | 99.83 | 99.83 | 99.97 | 99.90 | 99.90 |
| GSD train BFree | 81.26 | 97.84 | 97.38 | 63.63 | 98.89 | 98.28 | 73.11 | 77.25 | 84.07 |
| GSD train Tiny | 99.86 | 100.00 | 100.00 | 99.97 | 99.74 | 99.74 | 99.97 | 99.86 | 99.86 |
| CLIP ViT-B/32 probe | 85.79 | 94.02 | 93.88 | 91.97 | 79.60 | 81.85 | 90.84 | 86.61 | 84.85 |
| NPR-ResNet18 | 96.60 | 99.53 | 99.48 | 97.20 | 96.00 | 96.05 | 97.17 | 96.62 | 96.58 |
| AIDE original full | 92.30 | 98.38 | 98.37 | 87.94 | 96.66 | 96.34 | 88.91 | 91.95 | 92.62 |

### png_roundtrip

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 90.31 | 98.50 | 98.80 | 81.09 | 99.54 | 99.44 | 84.03 | 89.33 | 91.13 |
| CAPD train Tiny | 99.90 | 100.00 | 100.00 | 99.97 | 99.83 | 99.83 | 99.97 | 99.90 | 99.90 |
| CLIP ViT-B/32 probe | 85.79 | 94.02 | 93.88 | 91.97 | 79.60 | 81.85 | 90.84 | 86.61 | 84.85 |
| NPR-ResNet18 | 96.60 | 99.53 | 99.48 | 97.20 | 96.00 | 96.05 | 97.17 | 96.62 | 96.58 |
| AIDE original full | 92.30 | 98.38 | 98.37 | 87.94 | 96.66 | 96.34 | 88.91 | 91.95 | 92.62 |

### resize256_jpeg96

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 82.47 | 96.55 | 97.41 | 65.11 | 99.83 | 99.74 | 74.10 | 78.79 | 85.06 |
| CAPD train Tiny | 93.74 | 99.59 | 99.63 | 99.17 | 88.31 | 89.46 | 99.07 | 94.07 | 93.38 |
| GSD train BFree | 68.60 | 86.40 | 88.26 | 38.80 | 98.40 | 96.04 | 61.65 | 55.27 | 75.81 |
| GSD train Tiny | 90.09 | 96.04 | 96.89 | 83.23 | 96.94 | 96.46 | 85.25 | 89.36 | 90.72 |

### resize_square256

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 82.79 | 97.06 | 97.80 | 65.69 | 99.89 | 99.83 | 74.43 | 79.23 | 85.30 |
| CAPD train Tiny | 91.20 | 99.50 | 99.47 | 99.57 | 82.83 | 85.29 | 99.49 | 91.88 | 90.40 |
| GSD train BFree | 69.79 | 87.03 | 88.67 | 41.51 | 98.06 | 95.53 | 62.64 | 57.88 | 76.45 |
| GSD train Tiny | 92.20 | 97.58 | 98.03 | 92.69 | 91.71 | 91.79 | 92.61 | 92.24 | 92.16 |

### resize_square560

| Model | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPD train BFree | 91.54 | 98.92 | 99.11 | 83.54 | 99.54 | 99.46 | 85.81 | 90.81 | 92.17 |
| CAPD train Tiny | 99.21 | 99.92 | 99.93 | 99.17 | 99.26 | 99.26 | 99.17 | 99.21 | 99.21 |
| GSD train BFree | 79.74 | 96.98 | 97.06 | 59.97 | 99.51 | 99.20 | 71.31 | 74.75 | 83.09 |
| GSD train Tiny | 94.76 | 99.46 | 99.49 | 90.49 | 99.03 | 98.94 | 91.23 | 94.52 | 94.97 |

## File chạy và truy nguồn

- [compare_metrics.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/score_model_comparison/compare_metrics.py) tính lại metrics từ các frame đã join ID/label.
- [metrics_comparison.json](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/score_model_comparison/metrics_comparison.json) giữ full precision và TN/FP/FN/TP từng case.
- [Báo cáo score cũ](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/score_model_comparison/COMPARISON.md) có thông tin nguồn và cohort; đây là tài liệu riêng về score, không dùng thay báo cáo metrics này.
