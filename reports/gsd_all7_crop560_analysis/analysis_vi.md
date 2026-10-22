# Tổng hợp run GSD All7 center crop tối đa 560 — dataset phiên bản 2

Nguồn: https://www.kaggle.com/datasets/hadinho0710/checkpoint-gsd-sdv5-train

Kiểm tra ngày 21/10/2026. Metadata ghi title `checkpoint_gsd_GenImage_train`, version 2.
Run được xác nhận là `gsd_raw_all7_crop560_v1`, không phải run SD15 của version 1.
Chỉ tải báo cáo JSON/CSV; không tải trọng số 4,85 GB, không deserialize `.pt` hoặc chạy lại GPU inference.
Các nhận xét về epoch checkpoint dựa trên selected_checkpoint.json và lịch sử, không xác minh trực tiếp tensors trong best.pt.

## Kết quả chính

Run complete, smoke=false. Dừng sau 4 epoch, chọn epoch 1 theo minimum validation cross entropy.
Test native 7.000 ảnh: TN3498, FP2, FN0, TP3500; accuracy và balanced accuracy 99,97143%;
real recall99,94286%, fake recall100%, fake precision99,94289%, fake F1 99,97144%.
Pooled ROC-AUC99,99989388%, AP99,99989385%. Không nên làm tròn thành 100% rồi bỏ thông tin hai lỗi.

Chỉ có case `native` được chạy. Config `jpeg_eval_qualities=[96]` là tham số kế thừa, nhưng
`eval_cases=['native']` mới quyết định runner test. Không có kết quả JPEG96, JPEG90/80, resize,
CommFor hoặc B-Free trong version này.

## Giao thức

- CLIP ViT-L/14 full fine-tuning; GSD ở 4 block cuối; frozen anchor; Gram-eigh k64; SDPA.
- All7 train: 25.200 ảnh = mỗi generator 1.800 real + 1.800 fake.
- Validation: 2.800 ảnh = mỗi generator 200 real + 200 fake.
- Test: 7.000 ảnh = mỗi generator 500 real + 500 fake.
- 35.000 ảnh được audit ban đầu. Train/validation từ 28.000 source-train; official val dùng test.
- Center crop mỗi chiều tối đa 560, không resize, giữ ảnh nhỏ và padding cạnh phải/dưới tới bội số14.
- JPEG/blur/flip training probabilities đều0. LR backbone/head/modulation đều1e-6, AdamW WD0.01.
- Logical train/test batch8; detector/anchor microbatch1; FP16; gradient checkpointing.
- Seed42, một Tesla T4. Không có đối chứng CLIP không GSD trong run.

## Lịch sử train và early stopping

|Epoch|Train loss|Validation loss|Val accuracy|Val AUC|Val FP|Val FN|AMP skipped|Giây train+val|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|0,034182|0,001931|99,9286%|99,999898%|2|0|6|8907,60|
|2|0,002217|0,001967|99,8929%|99,999796%|2|1|1|8843,68|
|3|0,001803|0,005751|99,8571%|99,999031%|3|1|1|8848,03|
|4|0,001821|0,003871|99,8571%|99,999745%|1|3|2|8830,78|

Validation không cải thiện sau epoch1. Train loss giảm rất mạnh tới epoch3, validation loss tăng:
có dấu hiệu overfit giai đoạn sau epoch1, dù accuracy vẫn gần bão hòa và số lỗi nhỏ. Validation CE
nhạy với độ tự tin của các dự đoán sai. Early stopping đúng patience3 và restore epoch1, không dùng epoch4 cho test.
Không đủ dữ liệu để biết performance JPEG/out-of-domain của từng epoch.

Tổng thời gian train+validation theo history: 35.430,10 giây, khoảng9h50m30s; không gồm toàn bộ scan,
checkpoint writes và test cuối. 12.600 logical batches qua4epoch, tổng10AMP skipped steps; không phải chuỗi
bỏ cập nhật kéo dài. Số epoch ít nhưng epoch1 đã có3.150bước, nên học nhanh không bất ngờ như một run rất nhỏ.

## Test theo generator

|Generator|N|Accuracy/BA|Real recall|Fake recall|ROC-AUC|Confusion [TN FP; FN TP]|
|---|---:|---:|---:|---:|---:|---|
|ADM|1000|100%|100%|100%|100%|[500 0;0 500]|
|BigGAN|1000|100%|100%|100%|100%|[500 0;0 500]|
|GLIDE|1000|100%|100%|100%|100%|[500 0;0 500]|
|Midjourney|1000|100%|100%|100%|100%|[500 0;0 500]|
|SD15|1000|99,9%|99,8%|100%|100%|[499 1;0 500]|
|VQDM|1000|100%|100%|100%|100%|[500 0;0 500]|
|Wukong|1000|99,9%|99,8%|100%|99,9992%|[499 1;0 500]|

Pooled và macro accuracy bằng nhau vì các generator và nhãn cân bằng. Macro AUC99,99988571%
khác rất nhỏ pooled AUC; pooled có xét cặp real/fake khác generator. Tất cả generator đều seen.
Cột `seen=False` của hàng pooled là cách đánh dấu hàng tổng hợp trong code, không có nghĩa generator unseen.

Hai lỗi đều là real bị gán fake với độ tự tin cao:

- Wukong: `imagenet_ai_0424_wukong/val/nature/ILSVRC2012_val_00006695.JPEG`, score fake0,985014,
  ảnh gốc867x996.
- SD15: `imagenet_ai_0424_sdv5/val/nature/ILSVRC2012_val_00029178.JPEG`, score fake0,991456,
  ảnh gốc353x450.

Không suy đoán nội dung hoặc tính đúng nhãn hai ảnh vì không tải/xem ảnh gốc.

## Audit dữ liệu và chỉ số

- Tính lại N, AUC, AP, accuracy/BA, recalls và confusion counts của8hàng từ7.000dự đoán: khớp báo cáo.
- Đối chiếu prediction với test manifest bằng sample_id, generator, label và pixel_hash: khớp.
- Giao train/validation/test bằng pixel_hash, file_hash và sample_id đều bằng0.
- Epoch được chọn khớp minimum val_loss; số history entries khớp training_status.
- Token census đúng công thức grid_w*grid_h+1, max1601, đúng crop cap560.
- Đường dẫn tập test giống7.000/7.000đường dẫn run SD15 cũ. Đây là consistency cohort, không bảo đảm
  cùng protocol; nguồn train, crop, augmentation và batch đều đã thay đổi.
- Hash là số liệu do runner lưu, chưa tính lại từ ảnh gốc. Không loại trừ near-duplicates, tương đồng nội dung
  hoặc trùng sau crop. Cần kiểm tra crop hashes nếu nghi ngờ leakage ở vùng trung tâm.

## Cảnh báo quan trọng: shortcut kích thước/token và codec

Census35.000ảnh có4.287ảnh được crop (12,25%). Median973token, min13, max1601.
Fake có kích thước/token rất đặc trưng theo generator:

|Fake test generator|Tokens mỗi ảnh|
|---|---:|
|BigGAN|101|
|ADM, GLIDE, VQDM|362|
|SD15, Wukong|1370|
|Midjourney|1601|

Real test có nhiều kích thước khác nhau, median973token. Do pipeline giữ variable resolution,
token grid và positional embedding có thể cung cấp dấu hiệu liên quan nhãn. Không cần đọc trực tiếp
metadata chiều rộng/cao bằng classifier để model có khả năng nhạy với cấu trúc kích thước.

Đã thực hiện một diagnostic KHÔNG NHÌN NỘI DUNG ẢNH: học bảng majority label từ TRAIN25.200ảnh
cho mỗi token_count (hoặc input_width,input_height), tie và kích thước chưa thấy dự đoán real.
Giữ nguyên quy tắc khi đánh giá, không chọn threshold bằng test labels.

|Feature diagnostic|Validation accuracy|Test accuracy|Test fake recall|Test real recall|
|---|---:|---:|---:|---:|
|Chỉ số token|97,8214%|97,9143%|100%|95,8286%|
|Chỉ width,height sau crop/pad|97,8214%|97,9143%|100%|95,8286%|

Kết quả này chứng minh bộ dữ liệu/giao thức có tín hiệu shortcut kích thước rất mạnh. Nó KHÔNG
chứng minh GSD thực sự dùng shortcut đó hoặc toàn bộ kết quả của GSD đến từ metadata. Nhưng
99,97% accuracy chưa đủ để kết luận học forensic tổng quát. Quan sát codec real=JPEG, fake=PNG ở
cả train/test bổ sung một yếu tố nhiễu nữa; raw-only training không xử lý được codec bias.

JPEG test vẫn cần thiết, nhưng nén JPEG không thay đổi kích thước ảnh nên không tự loại size/token shortcut.
Ngay cả JPEG test cao cũng cần một cohort kiểm soát kích thước hoặc dữ liệu ngoài miền.

## Kết luận và bước đánh giá tiếp theo

Checkpoint epoch1 là artifact được runner chọn; giữ best.pt để đánh giá, last.pt dùng resume epoch4.
Run chứng minh hiệu năng native cùng7nguồn rất cao, với báo cáo nội bộ nhất quán. Chưa chứng minh
độ bền JPEG, hiệu quả riêng của GSD, generalization unseen hoặc CommFor/B-Free.

Ưu tiên: (1) native và fake-only/all-JPEG theo đúng checkpoint/crop/batch protocol;
(2) cohort kiểm soát size/token bằng một quy tắc label-blind, có báo cáo ảnh/cohort bị thay đổi;
(3) CLIP ViT-L/14 full-finetune đối chứng cùng dữ liệu/preprocessing; (4) CommFor/B-Free với
checkpoint/threshold cố định. Mọi thay preprocessing lúc test cần ghi rõ là stress test, không
trộn với giao thức native; có thể cần train đối chứng mới để đánh giá chiến lược input công bằng.

TestTiny này đã được xem nhiều lần: dùng kết quả cho exploratory diagnostics, không liên tục tune
theo nó rồi coi là test xác nhận độc lập. Không có bằng chứng từ run này để nói crop/GSD riêng lẻ
tạo mức tăng so với run SD15: nhiều yếu tố đã thay cùng lúc.

Files nguồn: `source/config.json`, `source/completion.json`, `source/seed42/clip_gsd/history.json`,
`source/seed42/clip_gsd/selected_checkpoint.json`, `source/seed42/clip_gsd/training_status.json`,
`source/metrics.csv`, `source/predictions.csv`, các manifests và `source/token_census.csv`.
Chỉ số tính lại và hashes file nguồn nằm trong `diagnostics.json`.
