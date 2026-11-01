# CAPD train BFree → Tiny và BFree

Nguồn: [hadinho0710/eval-train-capd-bfee-test-tiny-and-bfree, version 1](https://www.kaggle.com/datasets/hadinho0710/eval-train-capd-bfee-test-tiny-and-bfree). Kiểm tra ngày 01/11/2026. Tất cả tỷ lệ trong bảng là phần trăm.

## Model và kiểm tra dữ liệu

- Run b2_dinov3: DINOv3 ViT-L/16 + LoRA rank8 + patch head CAPD. Train real COCO và nhánh fake SD2.1_inpainted_samecat_origBG, không phải phối hợp sáu nhánh fake.
- Config resume từ hadinho0710/check-point-train-capd-bfree/capd_tn2. Training status complete, images_seen=75.600, global_step=9.450; selected best virtual epoch2, monitor val loss=0,01791314845, stop_reason=all_virtual_epochs_done. Status ve=4 là vị trí vòng lặp sau khi kết thúc, không tự diễn giải thành đã train bốn virtual epoch.
- Cả Tiny và BFree evaluation complete. Tiny đủ 10 case × 7.000 ảnh; mỗi case 3.500 real + 3.500 fake. BFree đủ 10.304 official-valid COCO IDs, mỗi ID có 1 real và 6 fake: 72.128 ảnh riêng biệt.
- Đã đối chiếu trực tiếp ID train_order/validation_ids: không trùng với official-valid BFree. Tập train pool có 38661 ID, internal validation 2000 ID. Kiểm tra theo ID, không tính lại pixel/near-duplicate.
- Tính lại 264 dòng confusion/accuracy/BA/recalls/AUC/AP từ predictions, khớp bảng nguồn trong sai số <1e-12. Bổ sung precision real/fake và F1 từ confusion counts. Tiny ID/label/generator khớp lượt GSD train BFree → Tiny trước.
- Ngưỡng cố định 0,5; BA bằng accuracy vì mỗi cặp evaluation cân bằng hai lớp. AUC/AP Tiny là pooled 7.000 ảnh; BFree macro là trung bình sáu nhánh, không phải cộng lặp real thành các ảnh độc lập.

## Tiny: full pooling, ngưỡng 0,5

| case | BA/Acc | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | 90.31 | 98.50 | 98.80 | 81.09 | 99.54 | 99.44 | 84.03 | 89.33 |
| png_roundtrip | 90.31 | 98.50 | 98.80 | 81.09 | 99.54 | 99.44 | 84.03 | 89.33 |
| fake_jpeg96 | 89.84 | 98.60 | 98.84 | 80.14 | 99.54 | 99.43 | 83.37 | 88.75 |
| all_jpeg96 | 89.84 | 98.61 | 98.85 | 80.14 | 99.54 | 99.43 | 83.37 | 88.75 |
| all_jpeg90 | 89.51 | 98.40 | 98.64 | 79.63 | 99.40 | 99.25 | 82.99 | 88.36 |
| all_jpeg80 | 87.49 | 97.14 | 97.60 | 75.66 | 99.31 | 99.10 | 80.31 | 85.81 |
| resize_square256 | 82.79 | 97.06 | 97.80 | 65.69 | 99.89 | 99.83 | 74.43 | 79.23 |
| resize_square560 | 91.54 | 98.92 | 99.11 | 83.54 | 99.54 | 99.46 | 85.81 | 90.81 |
| blur_sigma1 | 89.80 | 97.91 | 98.32 | 80.14 | 99.46 | 99.33 | 83.36 | 88.71 |
| resize256_jpeg96 | 82.47 | 96.55 | 97.41 | 65.11 | 99.83 | 99.74 | 74.10 | 78.79 |

### Native theo generator

| generator | BA/Acc | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adm | 78.40 | 93.08 | 94.63 | 57.60 | 99.20 | 98.63 | 70.06 | 72.73 |
| biggan | 87.60 | 98.92 | 98.90 | 75.80 | 99.40 | 99.21 | 80.42 | 85.94 |
| glide | 90.80 | 99.31 | 99.31 | 82.20 | 99.40 | 99.28 | 84.81 | 89.93 |
| midjourney | 77.40 | 98.62 | 98.61 | 55.00 | 99.80 | 99.64 | 68.92 | 70.88 |
| sd15 | 99.60 | 100.00 | 100.00 | 99.40 | 99.80 | 99.80 | 99.40 | 99.60 |
| vqdm | 98.70 | 99.94 | 99.94 | 97.80 | 99.60 | 99.59 | 97.84 | 98.69 |
| wukong | 99.70 | 99.96 | 99.96 | 99.80 | 99.60 | 99.60 | 99.80 | 99.70 |

## BFree: full pooling, ngưỡng 0,5

| branch | BA/Acc | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 95.90 | 99.79 | 99.80 | 92.14 | 99.67 | 99.64 | 92.69 | 95.74 |
| SD2.1_selfconditioned_origBG | 74.30 | 93.22 | 94.74 | 48.93 | 99.67 | 99.33 | 66.12 | 65.57 |
| SD2.1_inpainted_samecat | 99.77 | 100.00 | 100.00 | 99.87 | 99.67 | 99.67 | 99.87 | 99.77 |
| SD2.1_inpainted_samecat_origBG | 99.77 | 99.99 | 99.99 | 99.87 | 99.67 | 99.67 | 99.87 | 99.77 |
| SD2.1_inpainted_diffcat | 99.79 | 100.00 | 100.00 | 99.90 | 99.67 | 99.67 | 99.90 | 99.79 |
| SD2.1_inpainted_diffcat_origBG | 99.62 | 99.99 | 99.99 | 99.56 | 99.67 | 99.67 | 99.56 | 99.62 |

### Macro sáu nhánh theo pooling

| Pooling | BA/Acc | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lse_full | 94.86 | 98.83 | 99.09 | 90.05 | 99.67 | 99.61 | 93.00 | 93.38 |
| lse_window | 95.09 | 98.82 | 99.08 | 90.54 | 99.63 | 99.57 | 93.29 | 93.74 |
| lse_inner | 50.01 | 98.14 | 98.38 | 0.01 | 100.00 | 50.00 | 50.00 | 0.02 |
| lse_nopad | 94.86 | 98.83 | 99.09 | 90.05 | 99.67 | 99.61 | 93.00 | 93.38 |

## Các pooling chẩn đoán

- lse_full: LSE trên toàn bộ patch; lse_window: cửa sổ patch, có trong BFree. Tiny nguồn không báo window pooling.
- lse_inner: bỏ hàng cuối và cột cuối patch. lse_nopad: chỉ bỏ patch chứa padding. Đây là các phép can thiệp score, không phải model train riêng.
- BFree ảnh512 chia hết patch16, không padding: full và nopad trùng metrics. inner có BA xấp xỉ50% ở ngưỡng0,5 nhưng AUC vẫn cao; điều này cho thấy score/điểm quyết định rất nhạy với loại patch biên. Chưa đủ để kết luận toàn bộ hiệu suất do shortcut; cần calibration riêng/đối chứng trước khi suy ra nhân quả.

| Tiny case | Pooling | BA/Acc | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native | lse_inner | 50.00 | 97.99 | 98.16 | 0.00 | 100.00 | 0.00 | 50.00 | 0.00 |
| native | lse_nopad | 90.41 | 99.72 | 99.71 | 81.09 | 99.74 | 99.68 | 84.06 | 89.43 |
| png_roundtrip | lse_inner | 50.00 | 97.99 | 98.16 | 0.00 | 100.00 | 0.00 | 50.00 | 0.00 |
| png_roundtrip | lse_nopad | 90.41 | 99.72 | 99.71 | 81.09 | 99.74 | 99.68 | 84.06 | 89.43 |
| fake_jpeg96 | lse_inner | 50.00 | 98.07 | 98.21 | 0.00 | 100.00 | 0.00 | 50.00 | 0.00 |
| fake_jpeg96 | lse_nopad | 89.94 | 99.73 | 99.72 | 80.14 | 99.74 | 99.68 | 83.40 | 88.85 |
| all_jpeg96 | lse_inner | 50.00 | 98.08 | 98.16 | 0.00 | 100.00 | 0.00 | 50.00 | 0.00 |
| all_jpeg96 | lse_nopad | 89.94 | 99.73 | 99.72 | 80.14 | 99.74 | 99.68 | 83.40 | 88.85 |
| all_jpeg90 | lse_inner | 50.00 | 97.87 | 98.05 | 0.00 | 100.00 | 0.00 | 50.00 | 0.00 |
| all_jpeg90 | lse_nopad | 89.71 | 99.76 | 99.75 | 79.63 | 99.80 | 99.75 | 83.05 | 88.56 |
| all_jpeg80 | lse_inner | 50.00 | 96.83 | 97.12 | 0.00 | 100.00 | 0.00 | 50.00 | 0.00 |
| all_jpeg80 | lse_nopad | 87.71 | 99.67 | 99.65 | 75.66 | 99.77 | 99.70 | 80.39 | 86.03 |
| resize_square256 | lse_inner | 50.00 | 94.00 | 95.43 | 0.00 | 100.00 | 0.00 | 50.00 | 0.00 |
| resize_square256 | lse_nopad | 82.79 | 97.06 | 97.80 | 65.69 | 99.89 | 99.83 | 74.43 | 79.23 |
| resize_square560 | lse_inner | 50.00 | 97.90 | 98.20 | 0.00 | 100.00 | 0.00 | 50.00 | 0.00 |
| resize_square560 | lse_nopad | 91.54 | 98.92 | 99.11 | 83.54 | 99.54 | 99.46 | 85.81 | 90.81 |
| blur_sigma1 | lse_inner | 50.00 | 97.53 | 97.84 | 0.00 | 100.00 | 0.00 | 50.00 | 0.00 |
| blur_sigma1 | lse_nopad | 89.97 | 99.67 | 99.66 | 80.14 | 99.80 | 99.75 | 83.40 | 88.88 |
| resize256_jpeg96 | lse_inner | 50.00 | 93.64 | 95.12 | 0.00 | 100.00 | 0.00 | 50.00 | 0.00 |
| resize256_jpeg96 | lse_nopad | 82.47 | 96.55 | 97.41 | 65.11 | 99.83 | 99.74 | 74.10 | 78.79 |

## Đối chiếu GSD train BFree

- Tiny native: CAPD full BA90,31%, AUC98,50%; GSD BA81,26%, AUC97,84% tại cùng ngưỡng0,5 và cùng ID. Khác backbone/training nên không quy toàn bộ chênh lệch cho riêng CAPD/GSD.
- BFree: model CAPD đã có đủ6 nhánh, GSD dataset cũ mới đủ5. Chỉ so matched5, không lấy macro6 CAPD so thẳng macro5 GSD:

| Model | Số nhánh | BA | AUC | Recall fake | Recall real |
| --- | --- | --- | --- | --- | --- |
| CAPD full | 5 | 93.91 | 98.60 | 88.14 | 99.67 |
| CAPD window | 5 | 94.18 | 98.59 | 88.73 | 99.63 |
| GSD train BFree | 5 | 89.66 | 98.89 | 79.58 | 99.73 |

## Nhận xét

- Native Tiny: TP2.838, FN662, TN3.484, FP16; model bỏ sót18,91% fake nhưng chỉ báo nhầm0,46% real ở ngưỡng0,5. Chuyển từ BFree sang Tiny vẫn có sai lệch quyết định; ngưỡng nguồn không tự tối ưu cho miền mới.
- Đọc hiệu suất từng phép biến đổi và từng generator ở bảng trên; không chỉ chọn native hoặc một biến đổi có điểm cao để khẳng định robustness tổng thể.
- Các cột best_ba/best_ba_threshold trong nguồn được tối ưu bằng nhãn evaluation, là diagnostic hậu nghiệm; báo cáo chính giữ ngưỡng0,5, không gọi đó là ngưỡng đã xác thực.

## File chạy và nguồn local

- [download_source.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/download_source.py) — metadata/inventory và tải file nhỏ/predictions theo version.
- [analyze.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/analyze.py) — kiểm tra ID, tính lại metrics và tạo báo cáo.
- [analysis.json](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/analysis.json) — đủ metrics, TN/FP/FN/TP, hashes, mỗi generator/case/pooling.
- [Snapshot capd_all.py từ dataset](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/source/capd_all.py).

## Phụ lục: BFree đủ từng nhánh và pooling

### lse_full

| branch | BA/Acc | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 95.90 | 99.79 | 99.80 | 92.14 | 99.67 | 99.64 | 92.69 | 95.74 |
| SD2.1_selfconditioned_origBG | 74.30 | 93.22 | 94.74 | 48.93 | 99.67 | 99.33 | 66.12 | 65.57 |
| SD2.1_inpainted_samecat | 99.77 | 100.00 | 100.00 | 99.87 | 99.67 | 99.67 | 99.87 | 99.77 |
| SD2.1_inpainted_samecat_origBG | 99.77 | 99.99 | 99.99 | 99.87 | 99.67 | 99.67 | 99.87 | 99.77 |
| SD2.1_inpainted_diffcat | 99.79 | 100.00 | 100.00 | 99.90 | 99.67 | 99.67 | 99.90 | 99.79 |
| SD2.1_inpainted_diffcat_origBG | 99.62 | 99.99 | 99.99 | 99.56 | 99.67 | 99.67 | 99.56 | 99.62 |

| Nhánh | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 10270 | 34 | 810 | 9494 |
| SD2.1_selfconditioned_origBG | 10270 | 34 | 5262 | 5042 |
| SD2.1_inpainted_samecat | 10270 | 34 | 13 | 10291 |
| SD2.1_inpainted_samecat_origBG | 10270 | 34 | 13 | 10291 |
| SD2.1_inpainted_diffcat | 10270 | 34 | 10 | 10294 |
| SD2.1_inpainted_diffcat_origBG | 10270 | 34 | 45 | 10259 |

### lse_window

| branch | BA/Acc | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 96.31 | 99.79 | 99.80 | 92.98 | 99.63 | 99.60 | 93.42 | 96.18 |
| SD2.1_selfconditioned_origBG | 75.30 | 93.16 | 94.70 | 50.96 | 99.63 | 99.28 | 67.01 | 67.35 |
| SD2.1_inpainted_samecat | 99.75 | 100.00 | 100.00 | 99.87 | 99.63 | 99.63 | 99.87 | 99.75 |
| SD2.1_inpainted_samecat_origBG | 99.76 | 99.99 | 99.99 | 99.89 | 99.63 | 99.63 | 99.89 | 99.76 |
| SD2.1_inpainted_diffcat | 99.78 | 100.00 | 100.00 | 99.92 | 99.63 | 99.63 | 99.92 | 99.78 |
| SD2.1_inpainted_diffcat_origBG | 99.62 | 99.99 | 99.99 | 99.61 | 99.63 | 99.63 | 99.61 | 99.62 |

| Nhánh | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 10266 | 38 | 723 | 9581 |
| SD2.1_selfconditioned_origBG | 10266 | 38 | 5053 | 5251 |
| SD2.1_inpainted_samecat | 10266 | 38 | 13 | 10291 |
| SD2.1_inpainted_samecat_origBG | 10266 | 38 | 11 | 10293 |
| SD2.1_inpainted_diffcat | 10266 | 38 | 8 | 10296 |
| SD2.1_inpainted_diffcat_origBG | 10266 | 38 | 40 | 10264 |

### lse_inner

| branch | BA/Acc | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 50.01 | 99.49 | 99.41 | 0.03 | 100.00 | 100.00 | 50.01 | 0.06 |
| SD2.1_selfconditioned_origBG | 50.00 | 89.78 | 91.88 | 0.00 | 100.00 | 0.00 | 50.00 | 0.00 |
| SD2.1_inpainted_samecat | 50.01 | 99.90 | 99.76 | 0.03 | 100.00 | 100.00 | 50.01 | 0.06 |
| SD2.1_inpainted_samecat_origBG | 50.00 | 99.88 | 99.73 | 0.00 | 100.00 | 0.00 | 50.00 | 0.00 |
| SD2.1_inpainted_diffcat | 50.00 | 99.90 | 99.77 | 0.01 | 100.00 | 100.00 | 50.00 | 0.02 |
| SD2.1_inpainted_diffcat_origBG | 50.00 | 99.87 | 99.73 | 0.00 | 100.00 | 0.00 | 50.00 | 0.00 |

| Nhánh | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 10304 | 0 | 10301 | 3 |
| SD2.1_selfconditioned_origBG | 10304 | 0 | 10304 | 0 |
| SD2.1_inpainted_samecat | 10304 | 0 | 10301 | 3 |
| SD2.1_inpainted_samecat_origBG | 10304 | 0 | 10304 | 0 |
| SD2.1_inpainted_diffcat | 10304 | 0 | 10303 | 1 |
| SD2.1_inpainted_diffcat_origBG | 10304 | 0 | 10304 | 0 |

### lse_nopad

| branch | BA/Acc | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 95.90 | 99.79 | 99.80 | 92.14 | 99.67 | 99.64 | 92.69 | 95.74 |
| SD2.1_selfconditioned_origBG | 74.30 | 93.22 | 94.74 | 48.93 | 99.67 | 99.33 | 66.12 | 65.57 |
| SD2.1_inpainted_samecat | 99.77 | 100.00 | 100.00 | 99.87 | 99.67 | 99.67 | 99.87 | 99.77 |
| SD2.1_inpainted_samecat_origBG | 99.77 | 99.99 | 99.99 | 99.87 | 99.67 | 99.67 | 99.87 | 99.77 |
| SD2.1_inpainted_diffcat | 99.79 | 100.00 | 100.00 | 99.90 | 99.67 | 99.67 | 99.90 | 99.79 |
| SD2.1_inpainted_diffcat_origBG | 99.62 | 99.99 | 99.99 | 99.56 | 99.67 | 99.67 | 99.56 | 99.62 |

| Nhánh | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| SD2.1_selfconditioned | 10270 | 34 | 810 | 9494 |
| SD2.1_selfconditioned_origBG | 10270 | 34 | 5262 | 5042 |
| SD2.1_inpainted_samecat | 10270 | 34 | 13 | 10291 |
| SD2.1_inpainted_samecat_origBG | 10270 | 34 | 13 | 10291 |
| SD2.1_inpainted_diffcat | 10270 | 34 | 10 | 10294 |
| SD2.1_inpainted_diffcat_origBG | 10270 | 34 | 45 | 10259 |
