# Báo Cáo Kết Quả Base Model Cho Bài Toán Fake Image Classifier

Nguồn tổng hợp: `D:\NguyenHoangHa_nam4\TLCN\Report\Báo cáo Base Model cho bài toán fake classifier.docx`

## 1. Dataset Và Evaluation Protocol

Dataset chính dùng trong các thí nghiệm baseline là **Tiny-GenImage**, gồm ảnh `real` và ảnh `fake` sinh bởi 7 generator:

| Generator |
|---|
| BigGAN |
| VQDM |
| SDv5 |
| Wukong |
| ADM |
| GLIDE |
| Midjourney |

DataLoader được thiết kế theo schema thống nhất để các baseline dùng cùng một cách chia dữ liệu. Điều này giúp giảm sai lệch khi so sánh giữa CLIP, ResNet, NPR-ResNet và các mô hình VLM/LLM sau này.

## 2. Các Protocol Đánh Giá

| Protocol | Train | Test | Mục tiêu |
|---|---|---|---|
| Combined | Tất cả 7 generator | Tất cả 7 generator | Đánh giá hiệu năng tổng thể khi train/test cùng bao phủ toàn bộ generator |
| In-domain | Một generator, ví dụ BigGAN | Cùng generator đó | Kiểm tra khả năng học generator đã xuất hiện trong training |
| Cross-generator | Tất cả generator trừ generator held-out | Generator chưa thấy khi train | Đánh giá generalization sang generator mới |
| Train-one-generator | Chỉ một generator | Tất cả generator | Kiểm tra model học artifact tổng quát hay chỉ học đặc trưng riêng của generator train |

Với cấu hình Combined trên Tiny-GenImage:

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Train | 14,000 | 14,000 | 28,000 |
| Test | 3,500 | 3,500 | 7,000 |

## 3. CLIP + Linear/Logistic Head

Baseline này dùng **CLIP ViT-B/32** như một frozen image encoder. Ảnh được đưa qua CLIP để lấy embedding, sau đó embedding được chuẩn hóa và đưa vào một classifier tuyến tính để dự đoán `real/fake`.

CLIP không được fine-tune. Chỉ classifier phía sau embedding được huấn luyện.

### Kết Quả

| Experiment | Train → Test | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Combined | All 7 → All 7 | 87.21% | 87.21% | 87.40% | 86.97% | 87.18% | 93.96% | 93.69% |
| In-domain BigGAN | BigGAN → BigGAN | 98.80% | 98.80% | 98.80% | 98.80% | 98.80% | 99.96% | 99.96% |
| Cross-generator GLIDE | 6 generators → GLIDE | 91.80% | 91.80% | 85.91% | 100.00% | 92.42% | 99.03% | 98.84% |
| Train-one BigGAN | BigGAN → All 7 | 66.43% | 66.43% | 96.15% | 34.23% | 50.48% | 81.32% | 83.82% |

### Nhận Xét

CLIP frozen + linear head là baseline khá mạnh. Kết quả Combined đạt Balanced Accuracy 87.21% và ROC-AUC 93.96%, cho thấy representation của CLIP chứa thông tin phân biệt ảnh thật và ảnh giả dù CLIP không được train trực tiếp cho bài toán forensic.

Tuy nhiên, Train-one BigGAN cho thấy vấn đề generalization rõ rệt. Precision rất cao 96.15%, nhưng Recall chỉ 34.23%. Nghĩa là khi model dự đoán fake thì thường đúng, nhưng model bỏ sót phần lớn ảnh fake đến từ generator khác. Đây là dấu hiệu model học artifact đặc thù của BigGAN nhiều hơn là học dấu hiệu fake tổng quát.

Cross-generator GLIDE đạt Recall 100% và Balanced Accuracy 91.80%, nhưng GLIDE là generator tương đối dễ hơn so với các generator hiện đại như Wukong hoặc Midjourney. Vì vậy kết quả này tốt nhưng chưa đủ để kết luận model tổng quát tốt trên mọi generator mới.

## 4. ResNet50 Last Layer

Baseline này dùng **ResNet-50 pretrained ImageNet** làm frozen backbone. Chỉ tầng fully connected cuối cùng được thay mới và huấn luyện cho bài toán `real/fake`.

Khác với CLIP, ResNet-50 được pretrained bằng supervised ImageNet classification, nên feature thiên về object/category hơn là alignment semantic ảnh-văn bản.

### Kết Quả

| Experiment | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Combined | 76.34% | 76.34% | 81.58% | 68.06% | 74.21% | 84.71% | 85.08% |
| In-domain BigGAN | 95.20% | 95.20% | 97.48% | 92.80% | 95.08% | 99.26% | 99.23% |
| Cross-generator GLIDE | 82.00% | 82.00% | 83.20% | 80.20% | 81.67% | 90.12% | 90.06% |
| Cross-generator Wukong | 71.70% | 71.70% | 77.61% | 61.00% | 68.31% | 79.90% | 80.50% |
| Train-one BigGAN | 59.87% | 59.87% | 90.22% | 22.14% | 35.56% | 70.96% | 74.04% |

### Nhận Xét

ResNet50 Last Layer yếu hơn CLIP rõ rệt ở Combined. Balanced Accuracy chỉ đạt 76.34%, thấp hơn CLIP khoảng 10.87 điểm phần trăm. Điểm yếu chính nằm ở Recall 68.06%, tức model bỏ sót khá nhiều ảnh fake.

In-domain BigGAN vẫn đạt kết quả rất cao, chứng tỏ ResNet pretrained vẫn học được artifact khi train/test cùng generator. Nhưng khi sang Train-one BigGAN, Recall giảm xuống 22.14%, cho thấy model gần như không nhận diện được fake ngoài distribution đã học.

Cross-generator Wukong thấp hơn Cross-generator GLIDE. Điều này hợp lý vì Wukong là generator mạnh hơn, ảnh sinh ra khó phân biệt hơn. Kết quả này nhấn mạnh rằng benchmark cross-generator phải dùng nhiều generator mạnh, không nên chỉ dựa vào GLIDE.

## 5. NPR-ResNet18 From Scratch

NPR (**Neighboring Pixel Relationships**) là hướng forensic tập trung vào quan hệ pixel lân cận thay vì semantic nội dung ảnh. Ý tưởng là ảnh sinh bởi GAN/diffusion thường để lại pattern cục bộ do quá trình upsampling/generation.

Vì đầu vào đã được biến đổi sang dạng NPR, mô hình dùng **ResNet-18 train from scratch** thay vì dùng pretrained ImageNet. Điều này ép model học low-level forensic evidence thay vì học semantic feature.

### Kết Quả

| Experiment | Eval Case | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Combined | Combined | 96.61% | 96.61% | 96.05% | 97.23% | 96.64% | 99.53% | 99.48% |
| In-domain BigGAN | In-domain | 99.30% | 99.30% | 98.62% | 100.00% | 99.30% | 99.90% | 99.87% |
| Cross-generator GLIDE | Cross-generator | 97.40% | 97.40% | 95.06% | 100.00% | 97.47% | 99.96% | 99.96% |
| Cross-generator Wukong | Cross-generator | 93.00% | 93.00% | 92.66% | 93.40% | 93.03% | 98.16% | 98.23% |
| Train-one BigGAN | Train-one-generator | 65.84% | 65.84% | 95.12% | 33.40% | 49.44% | 72.34% | 78.05% |
| Train-one Wukong | Train-one-generator | 57.69% | 57.69% | 66.36% | 31.17% | 42.42% | 59.33% | 62.48% |

### Nhận Xét

NPR-ResNet18 là baseline mạnh nhất trong nhóm CNN truyền thống. Combined đạt Balanced Accuracy 96.61%, cao hơn đáng kể so với CLIP và ResNet50.

Điểm đáng chú ý là Cross-generator Wukong vẫn đạt Balanced Accuracy 93.00% và Recall 93.40%. Điều này cho thấy NPR giúp model học được dấu hiệu forensic có tính tổng quát hơn so với feature semantic/pretrained thông thường.

Tuy vậy, Train-one BigGAN và Train-one Wukong vẫn thấp. Precision cao nhưng Recall thấp, đặc biệt Train-one BigGAN có Precision 95.12% nhưng Recall chỉ 33.40%. Nghĩa là khi chỉ train trên một generator, NPR vẫn chưa đủ để tổng quát tốt sang toàn bộ generator còn lại.

## 6. Benchmark Tổng Hợp Trên External Evaluation

Bảng sau tổng hợp kết quả so sánh giữa các baseline và Qwen2.5-VL-7B + LoRA. Theo pipeline thí nghiệm, các model được train trên Tiny-GenImage Combined và đánh giá trên benchmark ngoài miền.

| Model | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 Last Layer | 62.60% | 59.90% | 77.56% | 66.38% | 71.54% | 63.25% | 79.41% |
| NPR + ResNet18 | 80.80% | 75.17% | 84.86% | 88.70% | 86.74% | 80.77% | 88.18% |
| CLIP Linear Head | 84.30% | 76.03% | 84.14% | 95.90% | 89.64% | 89.22% | 94.61% |
| Qwen2.5-VL-7B + LoRA | 95.60% | 93.07% | 94.86% | 99.15% | 96.96% | 99.45% | 99.73% |

### Nhận Xét

Qwen2.5-VL-7B + LoRA vượt trội nhất ở external evaluation, đạt Balanced Accuracy 93.07%, ROC-AUC 99.45% và AP 99.73%. Điều này cho thấy VLM sau khi LoRA fine-tune có khả năng tổng quát tốt hơn các baseline thị giác truyền thống.

CLIP Linear Head đứng thứ hai về Balanced Accuracy và có Recall rất cao 95.90%. Tuy nhiên Balanced Accuracy chỉ 76.03%, cho thấy model có xu hướng dự đoán fake mạnh hơn, cần kiểm tra thêm false positive trên ảnh real.

NPR + ResNet18 tốt hơn ResNet50 Last Layer rõ rệt. Điều này củng cố giả thuyết rằng low-level forensic feature có giá trị hơn ImageNet semantic feature trong bài toán fake image detection.

ResNet50 Last Layer là baseline yếu nhất trên external evaluation. Kết quả này cho thấy frozen ImageNet feature không đủ mạnh để generalize sang benchmark ngoài miền.

## 7. Kết Luận Chính

1. **In-domain không phản ánh đủ năng lực thật của detector.** CLIP, ResNet50 và NPR đều đạt rất cao khi train/test cùng generator, nhưng giảm mạnh ở Train-one-generator.

2. **Cross-generator là protocol quan trọng nhất.** Đây là setting gần thực tế hơn vì generator mới liên tục xuất hiện. Kết quả trên Wukong đáng tin hơn GLIDE vì Wukong khó hơn.

3. **Feature forensic giúp tăng generalization.** NPR-ResNet18 vượt ResNet50 rõ rệt, đặc biệt ở Combined và Cross-generator.

4. **CLIP có representation mạnh nhưng vẫn có bias distribution.** CLIP tốt hơn ResNet50, nhưng Train-one BigGAN cho thấy model dễ bỏ sót fake ngoài generator đã học.

5. **Qwen2.5-VL-7B + LoRA hiện là hướng mạnh nhất.** Kết quả external evaluation vượt xa các baseline còn lại, đặc biệt ở Recall, ROC-AUC và AP.

6. **Cần tiếp tục kiểm tra false positive trên ảnh real.** Vì nhiều model có Recall fake cao, cần đánh giá riêng trên tập real-only như RAISE hoặc ảnh camera thật để kiểm tra việc real bị dự đoán nhầm thành fake.

## 8. Hướng Thí Nghiệm Tiếp Theo

Các robustness test nên được chạy để đánh giá độ ổn định:

| Biến thể test | Mục tiêu |
|---|---|
| Original | Kết quả gốc |
| JPEG Q=95/90/80/70 | Kiểm tra ảnh sau nén JPEG |
| Resize 0.75x rồi restore | Kiểm tra độ nhạy với resize |
| Gaussian blur | Kiểm tra khi ảnh bị làm mờ |
| Center crop | Kiểm tra khi mất vùng biên |
| PNG ↔ JPEG conversion | Kiểm tra thay đổi định dạng |

Các hướng hybrid có thể phát triển tiếp:

| Hướng | Ý tưởng |
|---|---|
| DCT-SRM-ResNet | Kết hợp đặc trưng tần số và residual forensic |
| CLIP + NPR-ResNet | Kết hợp semantic feature và low-level forensic feature |
| CLIP + DCT-SRM-ResNet | Kết hợp vision-language representation và frequency/residual evidence |
