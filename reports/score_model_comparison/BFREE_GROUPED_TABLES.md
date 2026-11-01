# So sánh TRAIN BFree → TEST BFree

Train BFree → Test BFree · ngưỡng 0,5 · CAPD full pooling · đơn vị %. Case 1–5: 10.304 ID/lớp. Case 6*: cùng 7.024 ID/lớp, còn partial; không đưa vào Mean. Cả hai train nhánh case 4.

## Bảng gộp đủ metrics

| Case | **Acc./BA** |  | **AUC** |  | **AP** |  | **Recall fake** |  | **Recall real** |  | **Precision fake** |  | **Precision real** |  | **F1 fake** |  | **F1 real** |  |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | CAPD | GSD | CAPD | GSD | CAPD | GSD | CAPD | GSD | CAPD | GSD | CAPD | GSD | CAPD | GSD | CAPD | GSD | CAPD | GSD |
| 1. selfconditioned | 95.90 | 80.73 | 99.79 | 99.12 | 99.80 | 99.05 | 92.14 | 61.75 | 99.67 | 99.70 | 99.64 | 99.52 | 92.69 | 72.27 | 95.74 | 76.21 | 96.05 | 83.80 |
| 2. selfconditioned_origBG | 74.30 | 73.34 | 93.22 | 95.47 | 94.74 | 96.02 | 48.93 | 47.00 | 99.67 | 99.68 | 99.33 | 99.32 | 66.12 | 65.29 | 65.57 | 63.81 | 79.50 | 78.90 |
| 3. inpainted_samecat | 99.77 | 97.00 | 100.00 | 99.93 | 100.00 | 99.92 | 99.87 | 94.24 | 99.67 | 99.75 | 99.67 | 99.73 | 99.87 | 94.55 | 99.77 | 96.91 | 99.77 | 97.08 |
| 4. inpainted_samecat_origBG | 99.77 | 99.72 | 99.99 | 99.99 | 99.99 | 99.99 | 99.87 | 99.66 | 99.67 | 99.78 | 99.67 | 99.78 | 99.87 | 99.66 | 99.77 | 99.72 | 99.77 | 99.72 |
| 5. inpainted_diffcat | 99.79 | 97.51 | 100.00 | 99.93 | 100.00 | 99.93 | 99.90 | 95.26 | 99.67 | 99.75 | 99.67 | 99.74 | 99.90 | 95.47 | 99.79 | 97.45 | 99.79 | 97.56 |
| 6. inpainted_diffcat_origBG * | 99.59 | 98.92 | 99.99 | 99.95 | 99.99 | 99.95 | 99.54 | 98.09 | 99.64 | 99.74 | 99.64 | 99.74 | 99.54 | 98.12 | 99.59 | 98.91 | 99.59 | 98.93 |
| Mean case 1–5 | 93.91 | 89.66 | 98.60 | 98.89 | 98.91 | 98.98 | 88.14 | 79.58 | 99.67 | 99.73 | 99.60 | 99.62 | 91.69 | 85.45 | 92.13 | 86.82 | 94.98 | 91.41 |

## Kết quả tổng thể

| Case | **Acc./BA** |  | **AUC** |  | **AP** |  |
|---|---:|---:|---:|---:|---:|---:|
| | CAPD | GSD | CAPD | GSD | CAPD | GSD |
| 1. selfconditioned | 95.90 | 80.73 | 99.79 | 99.12 | 99.80 | 99.05 |
| 2. selfconditioned_origBG | 74.30 | 73.34 | 93.22 | 95.47 | 94.74 | 96.02 |
| 3. inpainted_samecat | 99.77 | 97.00 | 100.00 | 99.93 | 100.00 | 99.92 |
| 4. inpainted_samecat_origBG | 99.77 | 99.72 | 99.99 | 99.99 | 99.99 | 99.99 |
| 5. inpainted_diffcat | 99.79 | 97.51 | 100.00 | 99.93 | 100.00 | 99.93 |
| 6. inpainted_diffcat_origBG * | 99.59 | 98.92 | 99.99 | 99.95 | 99.99 | 99.95 |
| Mean case 1–5 | 93.91 | 89.66 | 98.60 | 98.89 | 98.91 | 98.98 |

## Recall

| Case | **Recall fake** |  | **Recall real** |  |
|---|---:|---:|---:|---:|
| | CAPD | GSD | CAPD | GSD |
| 1. selfconditioned | 92.14 | 61.75 | 99.67 | 99.70 |
| 2. selfconditioned_origBG | 48.93 | 47.00 | 99.67 | 99.68 |
| 3. inpainted_samecat | 99.87 | 94.24 | 99.67 | 99.75 |
| 4. inpainted_samecat_origBG | 99.87 | 99.66 | 99.67 | 99.78 |
| 5. inpainted_diffcat | 99.90 | 95.26 | 99.67 | 99.75 |
| 6. inpainted_diffcat_origBG * | 99.54 | 98.09 | 99.64 | 99.74 |
| Mean case 1–5 | 88.14 | 79.58 | 99.67 | 99.73 |

## Precision và F1

| Case | **Precision fake** |  | **Precision real** |  | **F1 fake** |  | **F1 real** |  |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | CAPD | GSD | CAPD | GSD | CAPD | GSD | CAPD | GSD |
| 1. selfconditioned | 99.64 | 99.52 | 92.69 | 72.27 | 95.74 | 76.21 | 96.05 | 83.80 |
| 2. selfconditioned_origBG | 99.33 | 99.32 | 66.12 | 65.29 | 65.57 | 63.81 | 79.50 | 78.90 |
| 3. inpainted_samecat | 99.67 | 99.73 | 99.87 | 94.55 | 99.77 | 96.91 | 99.77 | 97.08 |
| 4. inpainted_samecat_origBG | 99.67 | 99.78 | 99.87 | 99.66 | 99.77 | 99.72 | 99.77 | 99.72 |
| 5. inpainted_diffcat | 99.67 | 99.74 | 99.90 | 95.47 | 99.79 | 97.45 | 99.79 | 97.56 |
| 6. inpainted_diffcat_origBG * | 99.64 | 99.74 | 99.54 | 98.12 | 99.59 | 98.91 | 99.59 | 98.93 |
| Mean case 1–5 | 99.60 | 99.62 | 91.69 | 85.45 | 92.13 | 86.82 | 94.98 | 91.41 |

Case 4 là nhánh cùng kiểu với training. Màu nhấn trong bản HTML chỉ giá trị cao hơn trong cặp, không phải kết luận model thắng mọi mặt. AUC/AP 100,00 có thể do làm tròn.
