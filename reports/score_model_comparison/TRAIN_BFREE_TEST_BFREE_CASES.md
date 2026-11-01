# TRAIN BFree → TEST BFree: so sánh rõ theo từng case

- CAPD: DINOv3+LoRA+patch head, full pooling. GSD: CLIP ViT-L/14+GSD. Cả hai train real COCO + fake **SD2.1_inpainted_samecat_origBG (case4)**, checkpoint tốt nhất virtual epoch2.
- Probability threshold0,5 cho cả hai; mọi chỉ số trong bảng là %. Accuracy=BA vì mỗi case cân bằng hai lớp.
- Case1–5:10.304 ID/case =10.304 real +10.304 fake, cùng ID/nhãn giữa model.
- Case6: GSD chỉ chạy7.024/10.304 ID. Lọc CAPD về đúng7.024 ID này:14.048 ảnh/model. Không dùng điểm CAPD full10.304 ID để so với GSD partial.
- Mean5 là trung bình số học metrics case1–5; không gộp case6 partial vào macro và không gọi kết quả là đủ6 case.

## CAPD train BFree

| Case | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. selfconditioned | 95.90 | 99.79 | 99.80 | 92.14 | 99.67 | 99.64 | 92.69 | 95.74 | 96.05 |
| 2. selfconditioned_origBG | 74.30 | 93.22 | 94.74 | 48.93 | 99.67 | 99.33 | 66.12 | 65.57 | 79.50 |
| 3. inpainted_samecat | 99.77 | 100.00 | 100.00 | 99.87 | 99.67 | 99.67 | 99.87 | 99.77 | 99.77 |
| 4. inpainted_samecat_origBG **(nhánh train)** | 99.77 | 99.99 | 99.99 | 99.87 | 99.67 | 99.67 | 99.87 | 99.77 | 99.77 |
| 5. inpainted_diffcat | 99.79 | 100.00 | 100.00 | 99.90 | 99.67 | 99.67 | 99.90 | 99.79 | 99.79 |
| 6. inpainted_diffcat_origBG **(partial matched)** | 99.59 | 99.99 | 99.99 | 99.54 | 99.64 | 99.64 | 99.54 | 99.59 | 99.59 |
| **Mean case1–5** | 93.91 | 98.60 | 98.91 | 88.14 | 99.67 | 99.60 | 91.69 | 92.13 | 94.98 |

## GSD train BFree

| Case | Acc./BA | AUC | AP | Recall fake | Recall real | Precision fake | Precision real | F1 fake | F1 real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. selfconditioned | 80.73 | 99.12 | 99.05 | 61.75 | 99.70 | 99.52 | 72.27 | 76.21 | 83.80 |
| 2. selfconditioned_origBG | 73.34 | 95.47 | 96.02 | 47.00 | 99.68 | 99.32 | 65.29 | 63.81 | 78.90 |
| 3. inpainted_samecat | 97.00 | 99.93 | 99.92 | 94.24 | 99.75 | 99.73 | 94.55 | 96.91 | 97.08 |
| 4. inpainted_samecat_origBG **(nhánh train)** | 99.72 | 99.99 | 99.99 | 99.66 | 99.78 | 99.78 | 99.66 | 99.72 | 99.72 |
| 5. inpainted_diffcat | 97.51 | 99.93 | 99.93 | 95.26 | 99.75 | 99.74 | 95.47 | 97.45 | 97.56 |
| 6. inpainted_diffcat_origBG **(partial matched)** | 98.92 | 99.95 | 99.95 | 98.09 | 99.74 | 99.74 | 98.12 | 98.91 | 98.93 |
| **Mean case1–5** | 89.66 | 98.89 | 98.98 | 79.58 | 99.73 | 99.62 | 85.45 | 86.82 | 91.41 |

## Chênh lệch CAPD − GSD, đơn vị điểm phần trăm

| Case | Δ BA | Δ AUC | Δ recall fake | Δ recall real |
| --- | --- | --- | --- | --- |
| 1. selfconditioned | +15.18 | +0.67 | +30.39 | -0.03 |
| 2. selfconditioned_origBG | +0.96 | -2.25 | +1.93 | -0.01 |
| 3. inpainted_samecat | +2.78 | +0.07 | +5.63 | -0.08 |
| 4. inpainted_samecat_origBG | +0.05 | -0.00 | +0.21 | -0.11 |
| 5. inpainted_diffcat | +2.28 | +0.06 | +4.64 | -0.08 |
| 6. inpainted_diffcat_origBG | +0.68 | +0.04 | +1.45 | -0.10 |

## Nhận xét

- Case4 (cùng kiểu fake với training) cả hai gần99,7% BA. Không dùng riêng case4 để kết luận khả năng tổng quát.
- CAPD hơn rõ ở case1 selfconditioned: BA +15,18 điểm, recall fake +30,39 điểm. Hai case inpainting không origBG (3/5) cũng có recall fake cao hơn GSD.
- Case2 selfconditioned_origBG yếu ở cả hai: recall fake CAPD48,93%, GSD47,00%, đều bỏ sót hơn nửa fake. AUC GSD95,47% cao hơn CAPD93,22%, nên không nói CAPD thắng mọi chỉ số.
- Case6 partial: CAPD99,59% BA, GSD98,92% trên cùng7.024 ID; chưa phải kết quả cuối của toàn nhánh.
- Cùng5 case đủ: CAPD93,91% BA và88,14% recall fake; GSD89,66% BA và79,58% recall fake. Đây là kết quả ngưỡng0,5; chưa trộn kết quả tối ưu bằng nhãn evaluation.

## File chạy

- [bfree_case_report.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/score_model_comparison/bfree_case_report.py) đọc JSON kết quả đã audit và lập bảng.
- [metrics_comparison.json](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/score_model_comparison/metrics_comparison.json) chứa TN/FP/FN/TP và số liệu đầy đủ.
