# Tổng hợp CAPD: Tiny-GenImage và BFree

Nguồn: [Kaggle dataset, version 1](https://www.kaggle.com/datasets/hadinho0710/eval-capd-test-tiny-bfree).

## Mô hình và cách đánh giá

- Run `r2_dinov3`, backbone DINOv3 ViT-L/16, LoRA rank 8, seed 42, checkpoint epoch 3; validation monitor CE = 0.006938915.
- 3.421.698 tham số trainable / 306.575.874 tổng tham số. Geometry gốc: center crop tối đa 560, replicate-pad tới bội patch 16, không resize.
- Accuracy, balanced accuracy (BA) và recall bên dưới dùng ngưỡng cố định 0,5. AUC/AP được đổi sang phần trăm.
- Tiny: 7.000 ảnh test, gồm 3.500 real và 3.500 fake từ 7 generator đã xuất hiện trong training. Mỗi case là biến đổi của cùng tập test, không phải tập ảnh độc lập.
- BFree: code đọc valid_list.csv và kiểm tra MD5; 10.304 COCO ID, 10.304 ảnh real dùng chung và 6 × 10.304 ảnh fake, tổng cộng 72.128 ảnh riêng biệt. Mỗi nhánh so sánh 20.608 ảnh (real + fake). Đây là BFree validation, dù tên dataset có chữ test.
- BFree lse_full gộp toàn bộ patch; lse_window lấy điểm cao nhất qua các cửa sổ 14×14 patch (224×224 pixel, stride 2 patch).

## Tiny: kết quả tổng thể

| Case | Acc./BA (%) | AUC (%) | AP (%) | Recall fake (%) | Recall real (%) | FP / FN |
| --- | --- | --- | --- | --- | --- | --- |
| native | 99.90 | 100.00 | 100.00 | 99.97 | 99.83 | 6 / 1 |
| png_roundtrip | 99.90 | 100.00 | 100.00 | 99.97 | 99.83 | 6 / 1 |
| fake_jpeg96 | 99.87 | 100.00 | 100.00 | 99.91 | 99.83 | 6 / 3 |
| all_jpeg96 | 99.87 | 100.00 | 100.00 | 99.91 | 99.83 | 6 / 3 |
| all_jpeg90 | 99.90 | 100.00 | 99.99 | 99.91 | 99.89 | 4 / 3 |
| all_jpeg80 | 99.79 | 99.99 | 99.99 | 99.71 | 99.86 | 5 / 10 |
| resize_square256 | 91.20 | 99.50 | 99.47 | 99.57 | 82.83 | 601 / 15 |
| resize_square560 | 99.21 | 99.92 | 99.93 | 99.17 | 99.26 | 26 / 29 |
| blur_sigma1 | 99.83 | 99.99 | 99.99 | 99.91 | 99.74 | 9 / 3 |
| resize256_jpeg96 | 93.74 | 99.59 | 99.63 | 99.17 | 88.31 | 409 / 29 |

Native: 6 ảnh real bị gọi nhầm fake, 1 ảnh fake bị bỏ sót. PNG roundtrip có toàn bộ metrics trùng native. JPEG Q80–96 và blur sigma 1 giữ accuracy ≥99,79%.

Resize 256×256 là điểm yếu lớn nhất: accuracy 91,20%, với 601 false positive / 3.500 ảnh real (17,17%) và 15 false negative. AUC vẫn 99,50%, cho thấy khả năng xếp hạng còn tốt nhưng quyết định tại ngưỡng 0,5 bị lệch. ECE tăng từ 0,00105 (native) lên 0,08740. Resize 256 + JPEG96 đạt 93,74%; resize 560 đạt 99,21%.

## Tiny: theo generator

| Generator | Native acc. (%) | Resize256 acc. (%) | Resize560 acc. (%) |
| --- | --- | --- | --- |
| adm | 99.90 | 90.40 | 98.50 |
| biggan | 99.80 | 91.80 | 99.80 |
| glide | 99.80 | 91.00 | 99.40 |
| midjourney | 100.00 | 91.20 | 97.50 |
| sd15 | 99.90 | 90.40 | 99.80 |
| vqdm | 100.00 | 92.70 | 99.70 |
| wukong | 99.90 | 90.90 | 99.80 |

Mỗi dòng gồm 500 real và 500 fake. Midjourney khó hơn các nguồn khác sau resize: ở 560 có 21/500 fake bị bỏ sót; ở 256 AUC=97,63% và TPR tại FPR 1%=61,60%, thấp hơn các generator còn lại.

## BFree: theo nhánh

| Nhánh | AUC full (%) | BA full (%) | BA window (%) | Recall fake full (%) | Recall fake window (%) |
| --- | --- | --- | --- | --- | --- |
| selfconditioned | 97.75 | 92.28 | 92.67 | 85.35 | 86.24 |
| selfconditioned_origBG | 61.06 | 55.00 | 55.37 | 10.78 | 11.65 |
| inpainted_samecat | 99.79 | 98.64 | 98.66 | 98.05 | 98.22 |
| inpainted_samecat_origBG | 76.49 | 68.99 | 69.32 | 38.75 | 39.54 |
| inpainted_diffcat | 99.91 | 99.23 | 99.20 | 99.24 | 99.31 |
| inpainted_diffcat_origBG | 87.04 | 81.09 | 81.44 | 62.95 | 63.78 |
| macro | 87.01 | 82.54 | 82.78 | 65.85 | 66.46 |

| Pooling | Macro AUC (%) | Macro AP (%) | Macro BA (%) | Macro recall fake (%) | Recall real (%) |
| --- | --- | --- | --- | --- | --- |
| lse_full | 87.01 | 89.33 | 82.54 | 65.85 | 99.22 |
| lse_window | 87.00 | 89.33 | 82.78 | 66.46 | 99.10 |

Window tăng macro BA 0,24 điểm phần trăm và macro fake recall 0,60 điểm; AUC gần như không đổi. False positive trên tập real chung tăng từ 80 lên 93 ảnh.

Các nhánh origBG giảm mạnh: BA full chỉ 55,00–81,09%, trong khi các nhánh tương ứng không origBG đạt 92,28–99,23%. Selfconditioned_origBG yếu nhất: bỏ sót 9.193/10.304 fake với full, hoặc 9.104 với window. Kết quả cho thấy khó phát hiện các mẫu thuộc nhánh này; chưa đủ bằng chứng từ bảng metrics để kết luận nguyên nhân nhân quả.

## Cách diễn giải và giới hạn

- Mô hình rất tốt trên Tiny và bền với các mức JPEG/blur đã thử, nhưng chưa bền với resize 256.
- Khả năng chuyển sang BFree khác biệt mạnh theo nhánh; macro BA 82,54% (full) hoặc 82,78% (window) không có nghĩa mọi nhánh đều đạt mức đó.
- Không gộp các nhánh BFree thành accuracy trên 123.648 ảnh độc lập: tập real được sử dụng lại 6 lần. Các dòng macro là trung bình 6 nhánh.
- Cột best_ba/best_ba_threshold trong nguồn được tìm trực tiếp bằng nhãn tập đánh giá, chỉ là kết quả tham khảo trên tập đó. Không dùng làm kết quả ngưỡng triển khai đã xác thực; muốn đổi ngưỡng cần tập calibration độc lập.
- Chỉ có một seed. Chưa có độ lệch chuẩn hoặc bằng chứng tổng quát cho các mức JPEG/blur/resize ngoài những case liệt kê.
- Đã kiểm tra n=TN+FP+FN+TP và tính lại accuracy/BA từ confusion counts cho toàn bộ 92 dòng không phải macro, đồng thời đối chiếu các macro BFree. Không tính lại AUC/AP từ prediction từng ảnh.

Các file nguồn, inventory đủ 372 file và SHA-256 của 31 file đã tải nằm trong thư mục source/. Không tải checkpoint hoặc bản đồ patch dung lượng lớn.
