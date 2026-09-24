# Phân Tích Chuyên Sâu Và Challenge Của Dự Án Fake Image Detection

File này bổ sung góc nhìn phân tích từ các kết quả baseline và Qwen2.5-VL + LoRA. Mục tiêu không chỉ là so sánh model nào cao hơn, mà còn chỉ ra các vấn đề cốt lõi của bài toán, các rủi ro khi đánh giá, và hướng phát triển hợp lý cho giai đoạn tiếp theo.

## 1. Nhận Định Tổng Quan

Kết quả hiện tại cho thấy bài toán fake image detection không đơn thuần là một bài toán phân loại ảnh nhị phân `real/fake`. Điểm khó nằm ở khả năng **generalization**: model không chỉ cần phân biệt fake trong tập train, mà phải nhận diện được ảnh sinh bởi generator mới, pipeline xử lý mới, compression mới và distribution ảnh thật khác với training data.

Các baseline cho thấy ba kiểu feature chính:

| Nhóm model | Loại feature chính | Điểm mạnh | Điểm yếu |
|---|---|---|---|
| ResNet50 Last Layer | Semantic/object feature từ ImageNet | Nhẹ, dễ train, baseline rõ ràng | Generalization yếu, bỏ sót fake nhiều |
| CLIP Linear Head | Semantic vision-language representation | Mạnh hơn ResNet, recall tốt hơn | Có bias theo distribution, dễ lệch threshold |
| NPR-ResNet18 | Low-level forensic feature | Rất mạnh trong Tiny-GenImage, cross-generator tốt | Có thể nhạy với resize, compression, post-processing |
| Qwen2.5-VL + LoRA | Vision-language + instruction-following | External benchmark rất mạnh | Nặng, khó triển khai, cần kiểm tra calibration và false positive |

Điểm quan trọng: kết quả cao trong `combined` hoặc `in-domain` chưa đủ để chứng minh model mạnh trong thực tế. Các setting như `cross-generator`, `cross-dataset`, `real-only`, và robustness mới phản ánh rõ năng lực thật.

## 2. Góc Nhìn Mới Từ Kết Quả Thí Nghiệm

### 2.1. In-domain Có Thể Tạo Ảo Giác Về Hiệu Năng

In-domain BigGAN cho kết quả rất cao ở cả CLIP, ResNet và NPR. Tuy nhiên, khi chuyển sang Train-one-generator, performance giảm mạnh. Điều này cho thấy model có thể đang học:

- artifact riêng của generator trong train,
- dấu hiệu nén/resize/preprocessing riêng của dataset,
- hoặc đặc trưng distribution của ảnh fake trong dataset,

thay vì học một khái niệm tổng quát về ảnh do AI tạo ra.

Vì vậy, in-domain chỉ nên dùng để kiểm tra model có học được tín hiệu hay không, không nên dùng làm kết luận chính.

### 2.2. Precision Cao Nhưng Recall Thấp Là Dấu Hiệu Model Quá Bảo Thủ

Ở Train-one BigGAN, nhiều model có precision fake rất cao nhưng recall fake rất thấp. Nghĩa là model chỉ dám dự đoán fake khi gặp pattern rất quen thuộc, còn các ảnh fake khác bị đẩy về real.

Đây là một failure mode nguy hiểm nếu mục tiêu hệ thống là phát hiện ảnh giả ngoài thực tế. Model nhìn có vẻ chính xác khi dự đoán fake, nhưng thực ra bỏ sót phần lớn fake mới.

### 2.3. NPR Cho Thấy Low-level Forensic Feature Có Giá Trị Rất Lớn

NPR-ResNet18 vượt mạnh ResNet50 và CLIP trên Tiny-GenImage, đặc biệt ở Combined và Cross-generator. Điều này gợi ý rằng artifact sinh ảnh không chỉ nằm ở semantic level, mà còn tồn tại ở local pixel relationship.

Tuy nhiên, chính vì NPR dựa vào low-level signal, model có thể bị ảnh hưởng bởi:

- JPEG compression,
- resize,
- crop,
- blur,
- social media recompression,
- ảnh RAW chuyển đổi sang JPEG,
- pipeline tiền xử lý khác nhau giữa dataset.

Do đó, NPR mạnh nhưng cần robustness test nghiêm túc.

### 2.4. Qwen2.5-VL + LoRA Mạnh Nhưng Không Nên Xem Là Hộp Đen Hoàn Hảo

Qwen2.5-VL + LoRA đạt kết quả cao nhất trên external evaluation. Đây là tín hiệu rất tốt, nhưng cần hiểu đúng:

- model không chỉ học visual artifact mà còn có prior từ pretraining rất lớn,
- prompt và candidate scoring ảnh hưởng trực tiếp đến prediction,
- xác suất `real/fake` có thể chưa được calibration tốt,
- model có thể bias về fake nếu đã thấy nhiều ảnh AI trong instruction-tuning/pretraining,
- inference chậm và tốn tài nguyên hơn nhiều so với baseline CNN/CLIP.

Nói cách khác, Qwen LoRA là hướng mạnh, nhưng cần đánh giá thêm về độ ổn định, calibration, và false positive trên ảnh thật.

### 2.5. Real-only Evaluation Là Bắt Buộc

Khi model có recall fake rất cao, câu hỏi tiếp theo là: có phải model đang dự đoán fake quá nhiều không?

Vì vậy cần test riêng trên tập toàn ảnh thật như:

- RAISE,
- ảnh camera thật local,
- ảnh từ nhiều thiết bị,
- ảnh sau xử lý JPEG/social media,
- ảnh có noise, blur, low light.

Metric quan trọng trong real-only evaluation là:

| Metric | Ý nghĩa |
|---|---|
| Real Accuracy | Tỷ lệ ảnh real được dự đoán đúng là real |
| False Positive Rate | Tỷ lệ ảnh real bị dự đoán nhầm là fake |
| Mean Fake Probability | Model trung bình nghi ngờ ảnh real là fake bao nhiêu |
| P90/P95/P99 Fake Probability | Nhóm real khó nhất có bị đẩy gần ngưỡng fake không |

Nếu false positive cao, hệ thống sẽ gây hại trong ứng dụng thực tế vì ảnh thật bị gắn nhãn fake.

## 3. Challenge Chính Của Dự Án

### 3.1. Challenge Về Dataset Bias

Tiny-GenImage là dataset tiện để train baseline, nhưng nó có thể chứa bias:

- real/fake có thể khác nhau về preprocessing,
- mỗi generator có folder/split riêng, dễ xuất hiện artifact phụ,
- ảnh real trong từng generator split có thể không đại diện cho ảnh thật ngoài đời,
- ảnh fake có thể cùng pipeline resize/compress,
- model có thể học shortcut từ dataset thay vì học bản chất fake.

Cần kiểm tra kỹ xem model có đang học dấu hiệu từ generator hay từ pipeline tạo dataset.

### 3.2. Challenge Về Data Leakage

Leakage không chỉ là trùng file train/test. Trong fake image detection, leakage có thể xảy ra ở nhiều mức:

| Dạng leakage | Ví dụ |
|---|---|
| Exact duplicate | Cùng ảnh xuất hiện ở train và test |
| Near duplicate | Ảnh crop/resize từ cùng ảnh gốc |
| Source leakage | Ảnh real cùng nguồn xuất hiện ở nhiều split |
| Generator leakage | Test generator đã xuất hiện trong train |
| Preprocessing leakage | Train/test chia khác nhãn nhưng giữ cùng artifact xử lý |

Để tránh leakage, cần group split theo generator/source và kiểm tra duplicate/near-duplicate nếu có thể.

### 3.3. Challenge Về Cross-dataset Generalization

Model train trên GenImage có thể tốt trong GenImage nhưng giảm khi test trên CommunityForensics hoặc RAISE. Nguyên nhân:

- generator khác,
- ảnh có resolution khác,
- prompt/domain khác,
- compression khác,
- real image source khác,
- fake image có thể đã qua post-processing.

Cross-dataset là thước đo quan trọng hơn combined nội bộ. Đây nên là phần đánh giá chính khi viết báo cáo.

### 3.4. Challenge Về Robustness

Fake detector thường rất nhạy với biến đổi ảnh. Một model có thể tốt trên ảnh gốc nhưng giảm mạnh sau JPEG hoặc resize.

Cần thiết kế robustness matrix:

| Biến đổi | Lý do cần test |
|---|---|
| JPEG Q=95/90/80/70 | Ảnh online thường bị nén |
| Resize down-up | Social media và app chat resize ảnh |
| Gaussian blur | Ảnh bị mờ hoặc hậu xử lý |
| Center crop/random crop | Metadata/context bị mất |
| PNG ↔ JPEG | Thay đổi định dạng làm mất artifact |
| Screenshot/re-upload | Trường hợp rất gần thực tế |

Nếu model chỉ hoạt động trên ảnh gốc chưa qua xử lý, khả năng triển khai thực tế sẽ thấp.

### 3.5. Challenge Về Threshold Và Calibration

Các model đang dùng threshold mặc định 0.5 cho fake probability. Nhưng threshold 0.5 chưa chắc tối ưu cho mọi dataset.

Ví dụ:

- nếu cần bắt fake tối đa, có thể giảm threshold,
- nếu cần tránh gắn nhãn sai ảnh thật, cần tăng threshold,
- threshold tốt trên GenImage chưa chắc tốt trên CommFor hoặc RAISE.

Cần báo cáo thêm:

- ROC curve,
- PR curve,
- threshold sweep,
- FPR tại các mức TPR cố định,
- calibration curve nếu có thể.

Trong ứng dụng thật, threshold nên được chọn theo risk: false negative fake hay false positive real nguy hiểm hơn.

### 3.6. Challenge Về Giải Thích Bằng VLLM

Dự án có mục tiêu giải thích bằng VLLM, nhưng explanation là phần rất rủi ro.

VLLM có thể đưa ra lời giải thích nghe hợp lý nhưng không phản ánh đúng lý do model dự đoán. Đây là vấn đề **faithfulness**.

Cần phân biệt:

| Loại explanation | Ý nghĩa |
|---|---|
| Plausible explanation | Nghe hợp lý với con người |
| Faithful explanation | Thật sự phản ánh evidence model dùng |

Nếu chỉ yêu cầu VLLM nói “ảnh này fake vì vùng tóc/da bất thường”, explanation có thể bị hallucinate. Nên kết hợp explanation với evidence định lượng:

- heatmap/saliency,
- vùng ảnh có fake probability cao,
- NPR/DCT/SRM signal,
- so sánh real/fake nearest neighbors,
- confidence và threshold.

## 4. Góc Nhìn Chiến Lược Cho Hướng Phát Triển

### 4.1. Không Nên Chọn Một Model Duy Nhất Quá Sớm

Mỗi nhóm model bắt được một loại evidence khác nhau:

| Evidence | Model phù hợp |
|---|---|
| Semantic inconsistency | CLIP, Qwen |
| Low-level artifact | NPR, SRM, DCT |
| Cross-modal reasoning | Qwen2.5-VL |
| Lightweight deployment | CLIP/ResNet/NPR |

Một hướng mạnh hơn là hybrid hoặc ensemble:

- CLIP embedding + NPR feature,
- Qwen score + NPR score,
- DCT/SRM branch + semantic branch,
- ensemble theo confidence.

### 4.2. Nên Tách Rõ Detector Và Explainer

Detector có nhiệm vụ dự đoán `real/fake`. Explainer có nhiệm vụ giải thích prediction. Không nên để explanation quyết định hoàn toàn prediction nếu chưa kiểm chứng.

Pipeline hợp lý hơn:

1. Detector tạo fake probability.
2. Forensic module tạo evidence map/feature.
3. VLLM nhận image + prediction + evidence.
4. VLLM sinh explanation có ràng buộc theo evidence.

Như vậy explanation ít bị hallucinate hơn.

### 4.3. Cần Error Analysis Theo Nhóm

Không chỉ xem metric tổng thể. Cần phân tích lỗi theo:

- generator,
- image resolution,
- real/fake source,
- fake probability range,
- false positive real,
- false negative fake,
- ảnh có compression,
- ảnh có người/không người,
- ảnh indoor/outdoor nếu metadata cho phép.

Error analysis sẽ cho biết model yếu ở đâu, thay vì chỉ biết accuracy cao/thấp.

## 5. Đề Xuất Thí Nghiệm Tiếp Theo

### 5.1. Real-only Benchmark

Chạy Qwen LoRA, CLIP, NPR trên RAISE hoặc tập ảnh camera thật local.

Mục tiêu:

- đo false positive rate,
- kiểm tra model có bias dự đoán fake không,
- tìm các ảnh real bị xem là fake để phân tích.

### 5.2. Robustness Benchmark

Với cùng một tập ảnh test, tạo nhiều version:

| Version | Thao tác |
|---|---|
| Original | Ảnh gốc |
| JPEG-95 | Nén nhẹ |
| JPEG-80 | Nén trung bình |
| JPEG-70 | Nén mạnh |
| Resize 0.75 | Giảm kích thước rồi restore |
| Blur | Gaussian blur |
| Crop | Center/random crop |

Sau đó so sánh metric từng model. NPR có thể giảm nhiều hơn CLIP/Qwen nếu artifact pixel bị phá.

### 5.3. Threshold Calibration

Dùng validation set riêng để chọn threshold theo mục tiêu:

| Mục tiêu | Threshold strategy |
|---|---|
| Bắt fake tối đa | Chọn threshold để Recall fake cao |
| Tránh vu oan ảnh thật | Chọn threshold để FPR real thấp |
| Cân bằng hai phía | Chọn threshold tối ưu Balanced Accuracy |

### 5.4. False Positive/False Negative Gallery

Tạo gallery ảnh lỗi:

- top real có fake probability cao nhất,
- top fake có fake probability thấp nhất,
- so sánh lỗi của CLIP/NPR/Qwen trên cùng ảnh,
- VLLM giải thích lỗi.

Đây là phần rất có giá trị khi viết báo cáo vì nó cho thấy model fail như thế nào.

### 5.5. Hybrid Model

Các hướng hybrid nên ưu tiên:

| Mô hình | Ý tưởng |
|---|---|
| CLIP + NPR | Semantic + local forensic |
| Qwen score + NPR score | VLM reasoning + pixel artifact |
| DCT-SRM-ResNet | Frequency + residual forensic |
| CLIP + DCT/SRM | Semantic + frequency evidence |

## 6. Rủi Ro Khi Viết Kết Luận Báo Cáo

Không nên kết luận:

> Model đạt accuracy cao nên đã giải quyết tốt fake image detection.

Nên kết luận thận trọng hơn:

> Các kết quả cho thấy model có khả năng học tín hiệu phân biệt real/fake trên Tiny-GenImage và có mức generalization nhất định sang benchmark ngoài miền. Tuy nhiên, hiệu năng phụ thuộc mạnh vào generator distribution, preprocessing pipeline và ngưỡng dự đoán. Do đó cần đánh giá thêm trên real-only, cross-dataset, robustness và calibration trước khi xem model là đáng tin cậy trong thực tế.

## 7. Kết Luận Phân Tích

Các baseline đã tạo được bức tranh khá rõ:

1. **ResNet50** là baseline yếu nhưng hữu ích để chứng minh ImageNet feature không đủ cho forensic detection.
2. **CLIP** mạnh hơn nhờ representation tổng quát, nhưng vẫn bị ảnh hưởng bởi distribution shift.
3. **NPR-ResNet18** cho thấy low-level forensic signal rất quan trọng và đáng phát triển tiếp.
4. **Qwen2.5-VL + LoRA** có kết quả tốt nhất, nhưng cần kiểm tra kỹ false positive, calibration, chi phí inference và độ tin cậy của explanation.

Challenge lớn nhất của dự án không phải là đạt accuracy cao trên một split, mà là xây dựng một detector có thể:

- tổng quát sang generator mới,
- ổn định sau post-processing,
- không vu oan ảnh thật,
- giải thích được dự đoán một cách faithful,
- và có thể tái lập kết quả trên nhiều benchmark.
