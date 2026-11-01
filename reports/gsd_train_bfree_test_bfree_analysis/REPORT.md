# GSD train BFree → đánh giá BFree validation

Nguồn: [hoangha071005/test-gsd-train-bfree-test-bfree, version 1](https://www.kaggle.com/datasets/hoangha071005/test-gsd-train-bfree-test-bfree). Metadata, completion.json và metrics.csv được đối chiếu lại trực tiếp với API Kaggle; hai file kết quả khớp từng byte bản local.

## Trạng thái chạy

- status=paused, reason=time_budget; thời gian đã chạy khoảng 10,50 giờ. Tên completion.json không có nghĩa toàn bộ evaluation đã complete.
- Hoàn tất 5/6 nhánh, mỗi nhánh 20.608 dự đoán. Nhánh inpainted_diffcat_origBG còn dang dở: 14.048/20.608 dòng = 7.024/10.304 cặp real/fake (68,17%). Còn thiếu 6.560 dòng, tương ứng 3.280 cặp.
- Tổng đã lưu: 117.088/123.648 dự đoán theo context, tương ứng 68.848/72.128 ảnh riêng biệt. Không dùng macro 5 nhánh làm kết quả đủ 6 nhánh, không trộn partial vào macro đủ.

## Model và dữ liệu

- Checkpoint: `/kaggle/input/datasets/hoangha071005/check-point-gsd-train-bfee-samecat-ori/gsd_bfree_samecat_origbg_v1/best.pt`; virtual epoch 2, signature `06c7d714b84425bf3248247cb9c1f686c12e5221f530ae325a454b1f88247046`.
- CLIP ViT-L/14 + GSD, seed42; train riêng nhánh SD2.1_inpainted_samecat_origBG và real COCO_real_512. Đây là train BFree một nhánh, không phải recipe phối hợp sáu nhánh đầy đủ của paper B-Free.
- Evaluation: official valid_list.csv, 10.304 COCO IDs; ảnh PNG 512×512, 1.370 token, ngưỡng fake probability >=0,5. Trên mỗi nhánh, số real và fake bằng nhau nên accuracy = balanced accuracy (BA).
- valid_list.csv là tập con của train_list.csv trong bản phát hành. Đã tải train_order.csv và validation_ids.csv từ đúng checkpoint dataset v1: 38.661 train-pool IDs và 2.000 internal-validation IDs; giao với 10.304 IDs evaluation đều bằng 0. Kiểm tra theo COCO ID, không phải audit near-duplicate/pixel hash.
- Đối chiếu từng nhánh đủ: đúng 10.304 real IDs và 10.304 fake IDs theo danh sách official valid đã lưu; partial có cùng 7.024 IDs giữa hai lớp. Không có trùng dòng (case, coco_id, label).

## Kết quả tại ngưỡng 0,5

| Nhánh | Trạng thái | n | BA/Acc (%) | AUC (%) | Recall fake (%) | Recall real (%) | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selfconditioned | Đủ | 20608 | 80.73 | 99.12 | 61.75 | 99.70 | 31 | 3941 |
| selfconditioned_origBG | Đủ | 20608 | 73.34 | 95.47 | 47.00 | 99.68 | 33 | 5461 |
| inpainted_samecat | Đủ | 20608 | 97.00 | 99.93 | 94.24 | 99.75 | 26 | 593 |
| inpainted_samecat_origBG | Đủ | 20608 | 99.72 | 99.99 | 99.66 | 99.78 | 23 | 35 |
| inpainted_diffcat | Đủ | 20608 | 97.51 | 99.93 | 95.26 | 99.75 | 26 | 488 |
| inpainted_diffcat_origBG | CHƯA ĐỦ | 14048 | 98.92 | 99.95 | 98.09 | 99.74 | 18 | 134 |

**Macro chỉ trên 5 nhánh đã hoàn tất:** BA/accuracy 89.66%; AUC 98.89%; AP 98.98%; recall fake 79.58%; recall real 99.73%.

FP = real bị báo nhầm fake; FN = fake bị bỏ sót. Với GSD, real được suy luận lại theo từng branch/batch context, nên không cộng FP của các nhánh rồi gọi là số ảnh real riêng biệt bị báo nhầm.

## Nhận xét

- Nhánh cùng kiểu với training, inpainted_samecat_origBG: BA 99,72%, AUC 99,9933%; chỉ bỏ sót 35/10.304 fake và báo nhầm 23/10.304 real. ID evaluation không nằm trong train pool/internal validation đã đối chiếu.
- Chuyển sang inpainted_samecat và inpainted_diffcat vẫn tốt: BA 97,00% và 97,51%. Nhánh diffcat_origBG mới có kết quả tạm thời 98,92%; chưa suy ra kết quả toàn bộ nhánh.
- Hai nhánh selfconditioned yếu hơn: BA 80,73% và 73,34%, fake recall 61,75% và 47,00%. Real recall vẫn xấp xỉ 99,7%, cho thấy ngưỡng 0,5 bỏ sót nhiều fake. AUC 99,12% và 95,47% cho thấy khả năng xếp hạng tốt hơn kết quả quyết định ở ngưỡng cố định.

## So sánh GSD train Tiny trước đó trên các nhánh hoàn tất

| Nhánh | BA train Tiny (%) | BA train BFree (%) | Thay đổi (điểm %) |
| --- | --- | --- | --- |
| selfconditioned | 93.71 | 80.73 | -12.98 |
| selfconditioned_origBG | 56.12 | 73.34 | +17.22 |
| inpainted_samecat | 97.34 | 97.00 | -0.34 |
| inpainted_samecat_origBG | 64.84 | 99.72 | +34.87 |
| inpainted_diffcat | 98.52 | 97.51 | -1.02 |

So sánh cùng ngưỡng 0,5 và các nhánh đủ. Không so macro 5 nhánh mới với macro 6 nhánh cũ. Train BFree cải thiện mạnh samecat_origBG (+34,87 điểm) và selfconditioned_origBG (+17,22 điểm), nhưng selfconditioned giảm 12,98 điểm; chưa thể nói tốt hơn đồng đều mọi nhánh.

## Việc còn lại

- Resume cùng checkpoint và fingerprint để hoàn thành 6.560 dự đoán còn lại của diffcat_origBG, rồi xuất lại macro đủ 6 nhánh.
- Có thể phân tích ngưỡng cho selfconditioned sau đó; các ngưỡng diagnostic trong analysis.json dùng trực tiếp nhãn evaluation, không phải ngưỡng được xác thực trên tập độc lập.
- Không audit lại toàn bộ pixel/ảnh gần trùng, không tải hoặc thực thi model weights trong lần tổng hợp này.

## File local

- [analyze.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_bfree_analysis/analyze.py): tính lại AUC/AP, confusion và phân bố score từ prediction.
- [audit_and_report.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_bfree_analysis/audit_and_report.py): kiểm tra cohort/overlap, trạng thái partial và tạo báo cáo này.
- [audit_checks.json](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_bfree_analysis/audit_checks.json): kết quả kiểm tra và SHA-256 nguồn.
- [analysis.json](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_bfree_analysis/analysis.json): metrics chi tiết, phân bố điểm, so sánh từng nhánh.
