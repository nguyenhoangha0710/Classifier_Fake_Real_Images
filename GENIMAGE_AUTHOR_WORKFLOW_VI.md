# Tác Giả GenImage Đã Làm Gì?

File này tóm tắt chi tiết quy trình mà nhóm tác giả dự án **GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image** đã thực hiện, dựa trên README, cấu trúc code và các bảng benchmark trong repo.

Mục tiêu của file:

```text
1. Hiểu tác giả xây dựng benchmark GenImage như thế nào.
2. Hiểu dữ liệu được tổ chức ra sao.
3. Hiểu họ dùng generator nào để tạo ảnh AI.
4. Hiểu họ dùng detector nào để đánh giá.
5. Hiểu các thí nghiệm chính trong benchmark.
6. Rút ra plan nghiên cứu để mình có thể làm lại hoặc mở rộng.
```

## 1. Mục Tiêu Của Tác Giả

Tác giả muốn xây dựng một benchmark lớn cho bài toán:

```text
Phát hiện ảnh do AI sinh ra.
```

Đây là bài toán binary classification:

```text
Input:  một ảnh
Output: ảnh thật hay ảnh AI
```

Quy ước trong dataset:

```text
nature = ảnh thật, lấy từ ImageNet
ai     = ảnh sinh bởi các mô hình generative AI
```

Điểm chính của GenImage:

```text
- Quy mô lớn: hơn 1 triệu cặp ảnh thật/ảnh AI.
- Nội dung phong phú: dùng 1000 class giống ImageNet.
- Nhiều generator khác nhau: Midjourney, Stable Diffusion, ADM, GLIDE, Wukong, VQDM, BigGAN.
- Đánh giá nhiều detector khác nhau.
- Tập trung kiểm tra khả năng generalization giữa các generator.
```

Nói ngắn gọn: tác giả không chỉ tạo một dataset, mà tạo một **benchmark** để đo xem detector có thật sự nhận ra ảnh AI từ nhiều nguồn khác nhau hay không.

## 2. Vấn Đề Tác Giả Muốn Giải Quyết

Ảnh AI ngày càng giống ảnh thật. Một detector có thể học rất tốt trên ảnh từ một generator cụ thể, nhưng khi gặp ảnh từ generator khác thì dễ giảm mạnh.

Ví dụ:

```text
Train trên Stable Diffusion
Test trên Stable Diffusion -> accuracy rất cao
Test trên Midjourney / ADM / BigGAN -> accuracy có thể giảm mạnh
```

Do đó, tác giả muốn trả lời các câu hỏi:

```text
1. Detector hiện tại có phát hiện ảnh AI tốt không?
2. Nếu train trên một generator, có nhận ra ảnh từ generator khác không?
3. Model CNN, Transformer, frequency-based method khác nhau thế nào?
4. Ảnh bị resize, JPEG, blur thì detector còn hoạt động tốt không?
5. Phương pháp nào robust hơn khi dữ liệu bị degrade?
```

## 3. Tác Giả Xây Dataset Như Thế Nào?

### 3.1. Chọn ảnh thật từ ImageNet

Tác giả dùng ảnh thật từ ImageNet làm nhóm `nature`.

Lý do:

```text
- ImageNet có 1000 class.
- Nội dung ảnh đa dạng.
- Có class label rõ ràng.
- Phù hợp để tạo cặp ảnh thật/ảnh AI theo cùng phân bố class.
```

Trong dataset, ảnh thật nằm trong folder:

```text
nature/
```

### 3.2. Tạo hoặc thu thập ảnh AI từ nhiều generator

Tác giả dùng nhiều mô hình sinh ảnh khác nhau:

```text
Midjourney
Stable Diffusion V1.4
Stable Diffusion V1.5
ADM
GLIDE
Wukong
VQDM / VQ-Diffusion
BigGAN
```

Ảnh AI nằm trong folder:

```text
ai/
```

Ý nghĩa của việc dùng nhiều generator:

```text
- Không để benchmark phụ thuộc vào một loại artifact duy nhất.
- Kiểm tra detector có học được đặc điểm tổng quát của ảnh AI không.
- Đánh giá cross-generator generalization.
```

### 3.3. Giữ cấu trúc train/val rõ ràng

Mỗi generator được tổ chức theo cấu trúc:

```text
Generator_Name/
  train/
    ai/
    nature/
  val/
    ai/
    nature/
```

Ví dụ:

```text
Midjourney/
  train/
    ai/
    nature/
  val/
    ai/
    nature/

VQDM/
  train/
    ai/
    nature/
  val/
    ai/
    nature/
```

Trong bản Kaggle tiny dataset bạn đang dùng, cấu trúc tương tự:

```text
imagenet_ai_0419_biggan/
  train/
    ai/
    nature/
  val/
    ai/
    nature/

imagenet_ai_0419_vqdm/
  train/
    ai/
    nature/
  val/
    ai/
    nature/
```

### 3.4. Gộp toàn bộ generator khi muốn train full benchmark

README gốc hướng dẫn nếu muốn train trên toàn bộ GenImage thì gom về một folder chung:

```text
imagenet_ai/
  train/
    ai/
    nature/
  val/
    ai/
    nature/
```

Cách gom:

```text
Midjourney/train/ai        -> imagenet_ai/train/ai
VQDM/train/ai              -> imagenet_ai/train/ai
ADM/train/ai               -> imagenet_ai/train/ai
BigGAN/train/ai            -> imagenet_ai/train/ai

Midjourney/train/nature    -> imagenet_ai/train/nature
VQDM/train/nature          -> imagenet_ai/train/nature
...
```

Tương tự cho `val`.

Lưu ý:

```text
Tác giả vẫn giữ từng generator riêng để benchmark.
Nhưng khi train full detector, có thể gộp tất cả vào imagenet_ai.
```

## 4. Tác Giả Dùng Những Generator Nào?

Repo có folder:

```text
GenImage/generator_codes/
```

Các generator được nhắc đến:

| Generator | Vai trò |
| --- | --- |
| Stable Diffusion | Text-to-image latent diffusion |
| GLIDE | Text-guided diffusion |
| ADM / Guided Diffusion | Diffusion model từ OpenAI |
| VQDM / VQ-Diffusion | Vector Quantized Diffusion |
| BigGAN | GAN class-conditional |
| Midjourney | Generator thương mại / closed model |
| Wukong | Text-to-image model |

Ý nghĩa:

```text
Tác giả cố tình chọn cả GAN, diffusion model, text-to-image model và generator thương mại.
Nhờ vậy benchmark đa dạng hơn.
```

## 5. Tác Giả Dùng Những Detector Nào?

Repo có folder:

```text
GenImage/detector_codes/
```

Các detector được benchmark:

| Detector | Loại | Ý nghĩa |
| --- | --- | --- |
| ResNet-50 | CNN baseline | Baseline phổ biến |
| DeiT-S | Vision Transformer | Kiểm tra transformer nhỏ |
| Swin-T | Hierarchical Vision Transformer | Transformer mạnh hơn, dùng shifted windows |
| CNNSpot | CNN-generated image detector | Detector chuyên cho ảnh sinh bởi CNN/GAN |
| Spec / AutoGAN | Frequency/spectrum-based | Dựa vào artifact miền tần số |
| F3Net | Frequency-aware detector | Khai thác clue tần số |
| GramNet | Texture/Gram feature | Tập trung vào texture artifact |

Nhận xét:

```text
Tác giả không chỉ test CNN thông thường.
Họ so sánh nhiều hướng:
- CNN classification
- Vision Transformer
- Frequency-based detection
- Texture-based detection
```

## 6. Quy Trình Train Detector Của Tác Giả

### 6.1. Bài toán train

Tất cả detector được đưa về bài toán:

```text
Binary classification:
nature -> 0
ai     -> 1
```

Input:

```text
ảnh RGB
```

Output:

```text
xác suất ảnh là nature hoặc ai
```

### 6.2. Ví dụ train ResNet-50

README gốc đưa ví dụ train ResNet-50 bằng `timm`:

```bash
sh ./distributed_train.sh 2 /cache/imagenet_ai/ \
  -b 64 \
  --model resnet50 \
  --sched cosine \
  --epochs 200 \
  --lr 0.05 \
  --amp \
  --remode pixel \
  --reprob 0.6 \
  --aug-splits 3 \
  --aa rand-m9-mstd0.5-inc1 \
  --resplit \
  --jsd \
  --dist-bn reduce \
  --num-classes 2
```

Ý nghĩa các điểm chính:

```text
/cache/imagenet_ai/  -> root dataset đã gộp
-b 64                -> batch size
--model resnet50     -> dùng ResNet-50
--epochs 200         -> train 200 epoch
--num-classes 2      -> binary classification
--amp                -> mixed precision
--sched cosine       -> cosine learning rate schedule
```

### 6.3. Data augmentation

Trong lệnh train có nhiều augmentation:

```text
RandAugment
Random Erasing
AugMix/JSD
Distributed BatchNorm
```

Mục đích:

```text
- Tăng robustness.
- Giảm overfitting vào artifact quá cụ thể.
- Giúp detector chịu được biến đổi ảnh.
```

## 7. Các Thí Nghiệm Chính Tác Giả Đã Làm

README gốc có 3 nhóm benchmark chính.

## 7.1. Thí nghiệm 1 - Train trên SD V1.4, test trên nhiều generator

Mục tiêu:

```text
Kiểm tra nếu detector chỉ học Stable Diffusion V1.4,
nó có phát hiện được ảnh AI từ generator khác không?
```

Thiết kế:

```text
Train:
Stable Diffusion V1.4

Test:
Midjourney
SD V1.4
SD V1.5
ADM
GLIDE
Wukong
VQDM
BigGAN
```

Kết quả chính:

```text
- Test trên SD V1.4 và SD V1.5 rất cao, gần 99%.
- Test trên generator khác giảm mạnh.
- Swin-T có average accuracy cao nhất trong bảng này, khoảng 74.8%.
```

Ý nghĩa:

```text
Detector có thể học rất tốt generator đã thấy,
nhưng chưa chắc generalize tốt sang generator chưa thấy.
```

## 7.2. Thí nghiệm 2 - Cross-validation giữa các generator

Mục tiêu:

```text
Đánh giá khả năng generalization tổng quát giữa nhiều generator.
```

Thiết kế:

```text
Train nhiều mô hình trên nhiều generator khác nhau.
Sau đó test trên từng generator.
Tính average accuracy theo testing subset.
```

Kết quả chính:

```text
Swin-T đạt average accuracy khoảng 70.3%, cao nhất trong bảng.
ResNet-50 khoảng 66.9%.
DeiT-S khoảng 67.6%.
CNNSpot thấp hơn, khoảng 61.7%.
```

Ý nghĩa:

```text
Transformer-based detector như Swin-T có xu hướng mạnh hơn trong benchmark này.
Nhưng accuracy cross-generator vẫn chưa thật sự cao.
```

## 7.3. Thí nghiệm 3 - Đánh giá trên ảnh bị degrade

Mục tiêu:

```text
Kiểm tra detector có còn hoạt động khi ảnh bị biến đổi chất lượng không.
```

Các dạng degradation:

```text
Low resolution 112
Low resolution 64
JPEG quality 65
JPEG quality 30
Blur sigma 3
Blur sigma 5
```

Kết quả chính:

```text
GramNet average khoảng 82.2%, cao nhất trong bảng degrade.
CNNSpot khoảng 78.3%.
ResNet-50 khoảng 70.6%.
Swin-T khoảng 67.0%.
```

Ý nghĩa:

```text
Một số detector mạnh trên ảnh sạch chưa chắc robust trên ảnh bị JPEG/resize/blur.
Texture/frequency method có thể hữu ích trong điều kiện degrade.
```

## 8. Plan Nghiên Cứu Có Thể Suy Ra Từ Tác Giả

Dựa trên README và benchmark, quy trình nghiên cứu của tác giả có thể tóm tắt như sau.

### Bước 1 - Xác định bài toán

```text
Bài toán: AI-generated image detection.
Input: ảnh.
Output: nature hoặc ai.
```

### Bước 2 - Chọn nguồn ảnh thật

```text
Ảnh thật lấy từ ImageNet.
Dùng 1000 class để nội dung đa dạng.
```

### Bước 3 - Chọn nhiều generator AI

```text
Chọn nhiều loại generator:
- GAN
- Diffusion
- Text-to-image
- Commercial generator
```

Mục đích:

```text
Tạo benchmark đa dạng,
không lệ thuộc vào artifact của một generator duy nhất.
```

### Bước 4 - Tạo dataset theo cấu trúc thống nhất

```text
generator/
  train/
    ai/
    nature/
  val/
    ai/
    nature/
```

Mục đích:

```text
Dễ train detector.
Dễ evaluate từng generator.
Dễ gộp full dataset.
```

### Bước 5 - Chọn detector đại diện

Detector gồm nhiều hướng:

```text
CNN baseline: ResNet-50
Transformer: DeiT-S, Swin-T
Specialized fake detector: CNNSpot
Frequency method: Spec, F3Net
Texture method: GramNet
```

Mục đích:

```text
So sánh các họ phương pháp khác nhau.
```

### Bước 6 - Train detector

Train binary classifier:

```text
ai vs nature
```

Có hai chế độ:

```text
1. Train trên một generator cụ thể.
2. Train trên toàn bộ dataset đã gộp.
```

### Bước 7 - Evaluate in-domain

```text
Train generator A
Test generator A
```

Mục tiêu:

```text
Biết model học được generator đã thấy hay không.
```

### Bước 8 - Evaluate cross-generator

```text
Train generator A
Test generator B, C, D...
```

Mục tiêu:

```text
Đo khả năng generalization.
```

### Bước 9 - Evaluate robustness

Test ảnh bị:

```text
resize
JPEG compression
blur
```

Mục tiêu:

```text
Xem detector có bền vững khi ảnh bị xử lý lại không.
```

### Bước 10 - So sánh và rút kết luận

Tác giả so sánh:

```text
- Detector nào tốt nhất trung bình.
- Detector nào generalize tốt hơn.
- Detector nào robust với degrade.
- Generator nào khó phát hiện nhất.
```

## 9. Những Kết Luận Chính Từ Benchmark

### 9.1. Train cùng generator thì dễ

Ví dụ train trên SD V1.4 và test trên SD V1.4:

```text
Accuracy gần 99%.
```

Điều này cho thấy detector có thể học artifact riêng của generator rất tốt.

### 9.2. Cross-generator khó hơn nhiều

Khi test trên generator khác, accuracy giảm rõ rệt.

Ý nghĩa:

```text
Detector có thể overfit vào artifact generator.
Phát hiện ảnh AI tổng quát vẫn là bài toán khó.
```

### 9.3. Swin-T mạnh trong benchmark sạch

Trong các bảng chính, Swin-T thường có average accuracy cao.

Ý nghĩa:

```text
Transformer/hierarchical ViT là hướng đáng thử.
```

### 9.4. Ảnh bị degrade làm detector thay đổi mạnh

JPEG, blur, resize có thể làm detector giảm accuracy.

Ý nghĩa:

```text
Nếu ảnh ngoài đời đã qua mạng xã hội, resize hoặc nén JPEG,
detector cần robustness tốt hơn.
```

### 9.5. Frequency/texture feature vẫn quan trọng

Các method như CNNSpot, Spec, F3Net, GramNet cho thấy:

```text
Artifact không chỉ nằm ở nội dung ảnh,
mà còn nằm ở texture và miền tần số.
```

## 10. Nếu Mình Làm Lại Theo Tác Giả Thì Nên Làm Gì?

Với dataset tiny GenImage trên Kaggle, ta có thể làm phiên bản nhỏ của benchmark.

### Phase 1 - Đọc dataset

```text
- Quét toàn bộ path ảnh.
- Tạo df.
- Kiểm tra số lượng theo subset/split/label.
- Hiển thị ảnh mẫu.
```

### Phase 2 - EDA và feature engineering

```text
- Kiểm tra kích thước ảnh.
- Tính brightness, contrast.
- Tính RGB mean/std.
- Tính blur_score.
- Tính FFT feature.
- Vẽ biểu đồ AI vs nature.
```

### Phase 3 - Baseline truyền thống

```text
- RandomForest trên handcrafted features.
- Logistic Regression.
- Xem feature importance.
```

### Phase 4 - Baseline deep learning

```text
- Train ResNet18 trên BigGAN.
- Validate BigGAN.
- Kiểm tra loss/accuracy.
```

### Phase 5 - Cross-generator nhỏ

```text
Train:
BigGAN

Test:
VQDM
SDV5
Wukong
ADM
GLIDE
Midjourney
```

Mục tiêu:

```text
Làm lại ý tưởng Table 3 ở quy mô nhỏ.
```

### Phase 6 - Multi-generator train

```text
Train:
Nhiều subset hoặc tất cả subset

Test:
Từng subset riêng
```

Mục tiêu:

```text
Làm lại ý tưởng Table 4 ở quy mô nhỏ.
```

### Phase 7 - Robustness

Tự tạo ảnh degrade:

```text
Resize 112
Resize 64
JPEG quality 65
JPEG quality 30
Blur sigma 3
Blur sigma 5
```

Test model trên các ảnh này.

Mục tiêu:

```text
Làm lại ý tưởng Table 5 ở quy mô nhỏ.
```

## 11. Bảng Kế Hoạch Thực Hiện Cho Mình

| Giai đoạn | Việc cần làm | Output |
| --- | --- | --- |
| 1 | Đọc dataset Kaggle | `df`, `genimage_index.csv` |
| 2 | EDA cơ bản | thống kê size, label, subset |
| 3 | Feature engineering | `feature_df.csv` |
| 4 | Baseline RandomForest | report feature baseline |
| 5 | Train ResNet18 BigGAN | checkpoint ResNet18 |
| 6 | Test cross-generator | bảng accuracy theo subset |
| 7 | Train nhiều subset | checkpoint multi-generator |
| 8 | Test degrade | bảng robustness |
| 9 | Error analysis | ảnh dự đoán sai, `wrong_predictions.csv` |

## 12. File Output Nên Có

Nên lưu các file sau:

```text
/kaggle/working/genimage_index.csv
/kaggle/working/feature_df.csv
/kaggle/working/random_forest_report.txt
/kaggle/working/evaluation_by_subset.csv
/kaggle/working/wrong_predictions.csv
/kaggle/working/resnet18_biggan.pth
/kaggle/working/resnet18_multigen.pth
```

## 13. Kết Luận

Tác giả GenImage đã làm một benchmark gồm 4 phần lớn:

```text
1. Xây dataset lớn gồm ảnh thật và ảnh AI từ nhiều generator.
2. Tổ chức dataset theo train/val và ai/nature.
3. Dùng nhiều detector khác nhau để đánh giá.
4. Thực hiện benchmark in-domain, cross-generator và robustness.
```

Điểm quan trọng nhất:

```text
GenImage không chỉ hỏi model có phân biệt AI/nature được không.
Nó hỏi model có phân biệt được ảnh AI từ generator chưa từng thấy không.
```

Vì vậy khi làm lại trên Kaggle tiny dataset, ta nên tập trung vào:

```text
- Evaluate theo từng generator.
- Cross-generator testing.
- Robustness với JPEG/blur/resize.
- Error analysis.
```

