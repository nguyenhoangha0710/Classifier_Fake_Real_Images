# Phân tích checkpoint CLIP + GSD, nguồn train SD15

Nguồn: https://www.kaggle.com/datasets/hadinho0710/checkpoint-gsd-sdv5-train/data

Ngày kiểm tra: 20/10/2026. Các file gốc được tải vào `source/`; không sửa dữ liệu nguồn.
Không tải hoặc thực thi các trọng số `.pt`. Kết luận về checkpoint dựa trên metadata và báo cáo đã lưu, không phải chạy lại inference.

## Kết luận chính

Run hoàn tất và chọn epoch 13 sau khi early stopping tại epoch 16. Mô hình học rất tốt nguồn SD15, nhưng khả năng phát hiện fake ngoài nguồn train chưa tốt và rất không đồng đều. Native SD15 đạt 99,1% balanced accuracy; macro sáu generator unseen chỉ 68,52%, fake recall 37,97%. Wukong mạnh, Midjourney khá; ADM, GLIDE, BigGAN và VQDM còn yếu. Chưa thể coi đây là detector tổng quát hoặc là bằng chứng GSD tốt hơn CLIP fine-tune.

## Kiểm tra tính nhất quán

- `completion.json`: complete, smoke=false.
- Train 3.600, validation 400, test 7.000 ảnh. Nguồn train/validation là SD15. Mỗi generator test có 500 real và 500 fake.
- Tính lại ROC-AUC, balanced accuracy, recalls và confusion counts của toàn bộ 32 dòng metrics từ 28.000 dự đoán: khớp trong sai số số học.
- Kiểm tra cùng tập sample ID và batch ID giữa bốn case: khớp.
- Tính lại flip rate và mean probability delta theo generator/nhãn: khớp paired_summary.
- Hash pixel đã ghi trong manifest không giao nhau giữa train, validation, test. Tập test có 7.000 hash pixel khác nhau.
- Kiểm tra hash không loại trừ ảnh gần trùng, ảnh cùng nội dung hoặc cùng nguồn ImageNet. Không tải toàn bộ ảnh để tính lại hash.
- Một seed (42), một model (`clip_gsd`), một Tesla T4 thực sự được sử dụng. Không có đối chứng `clip_ft` trong run này.
- Nguồn train SD15 khác giao thức SD14 của thí nghiệm GenImage trong paper; Tiny cũng nhỏ hơn full GenImage.

## Diễn biến train

| Epoch | Train loss | Validation loss | Val AUC (%) | Val balanced accuracy (%) |
|---:|---:|---:|---:|---:|
|1|0,662674|0,609068|85,900|64,50|
|2|0,492866|0,361618|97,845|92,00|
|3|0,179241|0,093052|99,635|96,25|
|4|0,044930|0,042275|99,878|98,50|
|5|0,020564|0,051576|99,853|98,00|
|6|0,015040|0,038013|99,888|99,00|
|7|0,007079|0,033456|99,920|99,00|
|8|0,006221|0,033277|99,948|99,00|
|9|0,004945|0,035028|99,940|99,00|
|10|0,003378|0,027768|99,950|99,00|
|11|0,003586|0,024502|99,970|99,25|
|12|0,003007|0,020181|99,983|99,25|
|13|0,003636|0,013950|99,995|99,75|
|14|0,002171|0,016521|99,990|99,50|
|15|0,000869|0,019125|99,988|99,25|
|16|0,000589|0,020002|99,988|99,25|

Epoch 1–4 học nhanh; epoch 5 validation tăng rồi giảm trở lại, không nên gọi là overfit kéo dài ngay tại epoch 5. Validation tiếp tục cải thiện tới epoch 13. Sau đó train loss giảm nhưng validation loss tăng ba epoch liên tiếp: có dấu hiệu overfit giai đoạn cuối trên nguồn SD15. Early stopping hoạt động đúng và báo cáo chọn epoch 13, thay vì epoch 16.

Epoch 13 validation sai 1/400 ảnh. Native test SD15 sai 9/1.000 ảnh. Điều này hỗ trợ khả năng tổng quát trong cùng nguồn, nhưng không chứng minh khả năng tổng quát khác generator. Chỉ có test của checkpoint được chọn, không có test từng epoch; không thể kết luận cross-generator performance bắt đầu giảm từ epoch nào, hay checkpoint sớm hơn sẽ tốt hơn trên unseen.

Tổng cộng ba AMP skipped steps (hai ở epoch 4, một ở epoch 9). Đây là ba lần scaler bỏ qua cập nhật do phát hiện gradient không hữu hạn; run vẫn hoàn tất. Không có chuỗi bỏ cập nhật kéo dài trong history. Không có timestamp từng epoch nên không thể khôi phục chính xác tổng thời gian train từ history.

## Test native ở ngưỡng 0,5

| Generator | AUC (%) | Balanced accuracy (%) | Fake recall (%) | Real recall (%) | Fake bị bỏ sót /500 |
|---|---:|---:|---:|---:|---:|
|SD15, seen|99,98|99,10|99,80|98,40|1|
|Wukong|99,55|96,60|94,60|98,60|27|
|Midjourney|93,99|79,80|60,40|99,20|198|
|VQDM|72,96|63,20|27,00|99,40|365|
|ADM|68,89|57,50|15,00|100,00|425|
|GLIDE|73,82|57,30|16,20|98,40|419|
|BigGAN|85,59|56,70|14,60|98,80|427|

Do mỗi cohort cân bằng, accuracy bằng balanced accuracy. Macro AUC sáu unseen là 82,47%, khác pooled AUC toàn bộ bảy nguồn là 84,72%. Pooled AUC bao gồm các cặp real/fake khác generator; không thay thế macro AUC.

Sáu unseen: TN=2.972, FP=28, FN=1.861, TP=1.139. Tỷ lệ bỏ sót fake 62,03%, báo nhầm real 0,93%. Khoảng cách balanced accuracy SD15 và macro unseen là 30,58 điểm phần trăm.

Nếu chỉ xem ADM, GLIDE, BigGAN và VQDM, macro balanced accuracy native là 58,68%. Vì vậy Wukong/Midjourney nâng số trung bình lên đáng kể. Không đủ bằng chứng quy nguyên nhân Wukong tốt cho việc dùng chung kiến trúc hay artifact với SD15; cần thêm dữ liệu/đối chứng.

## Điểm dự đoán và ngưỡng

BigGAN AUC 85,59% nhưng balanced accuracy 56,7% cho thấy điểm xếp hạng có thông tin, trong khi ngưỡng 0,5 bỏ sót nhiều fake. Median score fake BigGAN khoảng 0,0101. ADM, GLIDE, VQDM có median fake lần lượt khoảng 0,00115; 0,00341; 0,00539. Với SD15, median fake khoảng 0,99959.

Trong 3.000 fake unseen, 1.277 ảnh (42,57%) có score fake dưới 0,01. Đây là dự đoán sai rất tự tin theo score, không chỉ những ảnh nằm sát ngưỡng. Score chưa được chứng minh là xác suất đã hiệu chỉnh.

Hạ threshold có thể tăng fake recall nhưng sẽ tăng false positives, và không tự sửa được AUC yếu của ADM/GLIDE/VQDM. Không chọn threshold bằng nhãn của test này. Nếu hiệu chỉnh, cần tập validation riêng phù hợp miền mục tiêu và quy tắc đã xác định trước.

## Bốn điều kiện xử lý ảnh

| Case | Macro AUC unseen (%) | Macro BA unseen (%) | Fake recall unseen (%) | Real recall unseen (%) |
|---|---:|---:|---:|---:|
|Native|82,47|68,52|37,97|99,07|
|JPEG Q96|77,97|67,73|36,33|99,13|
|Resize square 256|78,89|68,28|39,17|97,40|
|Resize 256 + JPEG Q96|78,76|69,38|41,10|97,67|

Resize+JPEG có BA cao nhất trong các stress test nhưng AUC vẫn thấp hơn native 3,70 điểm. BA tăng không đồng nghĩa khả năng phân biệt tổng thể tăng; một phần là thay đổi điểm dự đoán tại ngưỡng 0,5. Không chọn case này làm pipeline tối ưu chỉ dựa vào test hiện tại.

BigGAN: JPEG Q96 làm AUC từ 85,59% xuống 58,34%, fake recall 14,6% xuống 0,4% (73 xuống 2 fake được phát hiện). Trong paired test, 71 fake đúng chuyển thành sai và không có fake sai chuyển thành đúng. Khi resize rồi JPEG, AUC lại 84,88%, cho thấy tương tác mạnh giữa phép resize và nén, không phải quy luật đơn giản mọi nén đều làm giảm.

GLIDE: JPEG Q96 tăng AUC từ 73,82% lên 81,78%, fake recall từ 16,2% lên 32,6%. Có 90 fake sai thành đúng và 8 đúng thành sai. Do đó JPEG thay đổi tín hiệu theo từng nguồn; không thể kết luận chỉ từ macro.

ADM: resize+JPEG làm AUC còn 53,18%, gần ngẫu nhiên. Wukong giữ AUC cao hơn 98% ở cả bốn case, dù BA giảm nhẹ. Midjourney native BA 79,8%, resize 81,5%, resize+JPEG 76,7%.

## Bias và giới hạn diễn giải

Audit định dạng cho thấy cả train và test đều real=JPEG, fake=PNG. Vì định dạng dự đoán được nhãn hoàn hảo trong bộ này, có nguy cơ model dựa vào dấu vết codec hoặc pipeline. Model đọc pixel RGB nên không nhất thiết nhìn trực tiếp đuôi file; dấu vết nén vẫn có thể hiện diện trong pixel. Augmentation JPEG được áp dụng cho cả nhãn nhưng không chứng minh đã xóa bias.

All-JPEG Q96 không xóa lịch sử nén JPEG trước đó; real có thể bị double compression. Resize square thay đổi aspect ratio và nội dung model thấy sau preprocessing, nên đây là stress test, không phải bằng chứng nhân quả riêng cho size bias. GSD dùng basis chung toàn logical batch nên biến đổi ảnh cũng làm thay đổi basis; delta đo độ nhạy của toàn pipeline, không chỉ riêng một cue nén cục bộ.

Các tham số a,b ở checkpoint epoch 13 lệch khởi tạo tối đa khoảng 4,16e-5 và 4,64e-5. Điều này cho thấy chúng cập nhật rất ít ở LR=1e-6. Không có nghĩa GSD tắt: phép chiếu và hệ số phụ thuộc từng mẫu vẫn hoạt động khi a,b gần cố định. Chưa có log occupancy/lambda, ablation GSD hay baseline cùng protocol để đo đóng góp riêng.

Chỉ một seed và một cohort nhỏ; chưa có đánh giá biến thiên seed, reference/per-sample inference, hay thay đổi batch composition. Các kiểm tra tính nhất quán không chứng minh tái lập GPU. Không dùng test này để lựa checkpoint, threshold hoặc augmentation rồi gọi nó là test độc lập.

## Thứ tự thí nghiệm tiếp theo

1. Chạy CLIP fine-tune không GSD với cùng SD15 split, augmentation, seed, logical batches và quy tắc chọn checkpoint. Đối chiếu per-generator AUC/BA/fake recall.
2. Đánh giá trên cohort kiểm soát codec/resize tốt hơn hoặc dataset độc lập. Tách câu hỏi generalization khỏi bias định dạng của Tiny.
3. Nếu mục tiêu tái lập paper, bổ sung SD14 và train lại theo nguồn đó; báo riêng run SD15 hiện tại.
4. Dùng tập development riêng cho hiệu chỉnh ngưỡng và lựa chọn giao thức; giữ test cuối độc lập. Có thể so checkpoint sớm/muộn như phân tích thăm dò, nhưng phải xác nhận kết luận trên cohort chưa dùng để lựa chọn.
5. Chạy nhiều seed và kiểm tra batch-inference dependence. Log thêm occupancy/lambda/gradient của GSD nếu nghiên cứu cơ chế.

Biểu đồ: `training_and_test.png` và bản vector `training_and_test.svg`.
