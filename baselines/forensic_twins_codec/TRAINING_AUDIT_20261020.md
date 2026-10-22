# Audit run codec encoder — 20/10/2026

Nguồn: https://www.kaggle.com/datasets/hoangha071005/checkpoint-encoder-twins-codec-200k . Đọc trực tiếp qua public Kaggle API: config.json, run_metadata.json, logs/epoch_metrics.csv, logs/train_steps.csv, encoder_final.pt và sidecar SHA256. Chỉ audit nhánh forensic_twins_codec/, không đếm bản duplicate publish/.

## Kết quả xác minh

- 20 epoch, native_random, batch128, LR6e-5, warmup2, manifest200K và FRE hash đúng run đã định.
- Native warm-start được metadata ghi nhận; encoder_final checksum khớp và tất cả tensor encoder hữu hạn. Không tải last.ckpt239MB nên chưa audit optimizer/scaler hoặc so trọng số native/final.
- Encoder final ghi optimizer_step31212, đúng tổng successful updates trong epoch CSV. Tối đa31240 attempts nếu chạy liên tục: thiếu28 (~0.09%), phù hợp AMP skipped updates theo code, không có resume history để chứng minh mọi lần skip.
- Rank0 policy tổng: identity300495, JPEG998150, resize-JPEG698923 (~15.04%,49.97%,34.99%). Đúng cấu hình trên các batch optimizer thành công; chưa audit rank1 hoặc transform pixels của run Kaggle.

## Hội tụ

| Epoch | Mean total loss | Mean diagonal correlation |
|---:|---:|---:|
| 1 | 2980.64 | 0.4430 |
| 3 | 1446.88 | 0.6565 |
| 10 | 1341.63 | 0.6738 |
| 18 | 1290.11 | 0.6847 |
| 20 | 1295.47 | 0.6838 |

Loss giảm56.54% từ epoch1, nhưng chỉ10.46% từ epoch3 và3.44% từ epoch10. Epoch12–20 mean1303.57, SD9.52 giữa epoch: plateau thực tế. Epoch18 thấp nhất training loss; điều này không chứng minh detection tốt nhất. Không có cơ sở tự thay checkpoint20 đã định bằng epoch18 theo B-Free.

Epoch20 loss1295.47 = invariance882.74 + weighted redundancy412.72. Raw redundancy8.25M phải nhân5e-5 trước khi diễn giải. Invariance cải thiện, redundancy weighted gần như giữ quanh400–415 từ sau giai đoạn đầu; offdiag abs correlation còn0.293. Chưa đạt biểu diễn decorrelated mạnh, nhưng không thể lấy giá trị loss tuyệt đối để kết luận thất bại vì8192 chiều và batch128, C centered có rank<=127 nên C=I8192 không khả thi trên một batch.

## Những điểm yếu cần nói rõ

1. feature_std192-D encoder khoảng0.25–0.26 cuối run: không có dấu hiệu toàn bộ embedding thành hằng số. Đây không đủ để loại trừ low-rank collapse hoặc duplicated features. Chưa đo effective rank/eigen-spectrum dù plan có yêu cầu.
2. Tất cả330 logged updates có preclip gradient norm>1 (min1400, max39038); mean cuối epoch khoảng9469. Clip1 tác động toàn bộ các điểm được log, scale điển hình~1e-4. Cần ablation; không tự kết luận effective LR giảm1e-4 vì AdamW có normalization riêng và loss là tổng nhiều chiều. Cấu hình clip1 do implementation chọn; thiếu log postclip/update-to-weight ratio.
3. Không có loss theo crop size/policy, không có fixed validation pairs, native-native control, paired codec embedding audit hoặc GMM evaluation. Không thể phân biệt plateau do augmentation khó, crop content, clipping hay objective. Không claim đã chống codec shortcut hoặc học forensic tốt hơn.
4. Không có before-finetune loss dưới cùng augmentation. Mức giảm epoch1→20 chỉ chứng minh tối ưu objective trong run, không đo lợi ích so native.
5. Code log policy counts/feature std từ rank0; loss/correlation là global batch. Epoch metrics khi resume giữa epoch chỉ là đoạn còn lại, phải đọc provenance trước so sánh nếu run có resume.

## Hành động tiếp theo

- Giữ checkpoint20 làm run chính đã định; không train dài hơn chỉ vì muốn loss thấp hơn. Nếu mở rộng phải là ablation mới.
- Fit GMM mới trên saved real-fit160K và calibrate40K; so native/codec trên cùng B-Free clean cohort, native và JPEG95/80/65/50, report per-variant AUROC và real FPR. Không dùng GMM native với encoder mới.
- Trên real audit split cố định: đo embedding eigen-spectrum/effective rank, distance/cosine native-vs-JPEG, so native checkpoint và codec checkpoint; đo cả detection để tránh reward collapse.
- Thử control native-native20epochs; sau đó ablate clipping với cùng data, seed, batch, steps. Đừng đổi đồng thời LR/epoch/loss/augment.

Kết luận: run hoàn tất và numerical health cơ bản tốt; có học ở đầu run, chững rõ cuối run. Đây là checkpoint đủ dùng cho bước đánh giá, chưa phải bằng chứng một detector mạnh hơn native.

Raw audit files và biểu đồ tái dựng được giữ tại artifacts/training_audit_20261020/ (local, gitignored). Scripts audit_download.py và audit_training.py tái lập việc đọc dữ liệu/thống kê; không thay đổi training.
