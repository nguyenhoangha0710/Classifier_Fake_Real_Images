# 1. Đặt vấn đề
- Các detector truyền thống huấn luyện trên GAN thường thất bại khi gặp các diffusion  mới. Vì mỗi generator có những đặc trưng tạo ảnh riêng dẫn đến model học thuộc những hình ảnh đấy thay vì học các dấu hiệu ảnh AI. --> Cần data đa dạng và test đa dạng 
- Ngược lại, Multimodal Large Language Model có thể mô tả ảnh và tạo lời giải thích tự nhiên, nhưng vision encoder của MLLM chủ yếu được pretrain để nhận biết semantic content như người, vật thể và bối cảnh. Nó không nhất thiết nhạy với những dấu vết forensic nhỏ như texture bất thường, noise residual, cạnh nhân tạo hoặc dấu vết từ quá trình sinh ảnh.
- Khi tối ưu Classification model sẽ học các artifact, noise, texture, ..... --> model có mất đi các semantic để chuyển thành ngôn ngữ con người dễ hiểu 
- Embedding space của classification sẽ các với token space của LLM --> cần train hoặc thiết kế để LLM hiểu được các evidence từ model classification

# 2. Pipeline đề xuất

Ý tưởng: Một hệ thống explainable fake image detection sẽ hiệu quả hơn nếu năng lực phân loại, năng lực chuyển thông tin forensic sang không gian ngôn ngữ và năng lực reasoning được huấn luyện theo ba giai đoạn riêng biệt nhưng liên kết với nhau.
Cụ thể: 
- Trước tiên huấn luyện một visual expert kép để phân loại real/fake. (classifier)
- Sau đó huấn luyện fused visual embedding để MLLM có thể đọc và dự đoán token `real/fake`, trong khi vẫn giữ năng lực classifier. Chuyển từ classifier embedding sang token embedding
- Cuối cùng mới huấn luyện LLM tạo lời giải thích.


**Pipeline train**

![[Pasted image 20260807094918.png]]
## Công dụng từng stage

### Stage 1 : Classification  Expert Training
Mục tiêu: Xây dựng một visual detector chuyên dụng có khả năng phân biệt ảnh thật và ảnh do AI tạo ra . Nó nhận vào hai tín hiệu embedding : semantic và artifact

Ảnh được đưa đồng thời vào hai nhánh: 
- nhánh CLIP để lấy semantic và general visual representation 
- Nhánh NPR-ResNet để lấy low-level forensic representation 
Việc dùng hai nhánh nhằm tránh phụ thuộc hoàn toàn vào một loại tín hiệu. Tuy nhiên tín hiệu nằm trong classifier embedding space. Chưa đảm bảo MLLM có thể sử dụng các tín hiệu này.


#### CLIP Vision Encoder 
Trích xuất thông tin tổng quát và semantic của ảnh, chẳng hạn: đối tượng xuất hiện trong ảnh, bố cục , ngữ cảnh, mối quan hệ giữa các vật thể, ...... 
#### NPR Transform
NPR biến đổi ảnh đầu vào nhằm nhắn mạnh các mối quan hệ pixel và resdiual. 
NPR làm giảm sự phụ thuộv vào nội dung semantic như (ảnh hoạt hình phần lớn là AI) và làm nổi bật: texture, noise distribution, edge artifacts, ...... 
#### ResNet Blocks
ResNet nhận ảnh đã qua NPR và học cách trích xuất forensic feature
Nhánh này đóng vai trò low-level forensic expert , tập trung vào: texture bất thường, oversmoothing , noise không nhất quán, ......
#### Feature Fusion
Feature từ hai nhánh được đưa về dimension phù hợp. 
Tạo một representation chứa đồng thời: 
- high-level sematic cues
- low-level forensic cues

![[Pasted image 20260807095221.png]]


### Stage 2: Token-Space Forensic Perception Alignment

Mục tiêu: Chuyển fused visual embedding của classifier sang visual-token space mà một MLLM đóng băng có thể sử dụng để dự đoán chính xác token *real/ fake*
Model chỉ yêu cầu trả lời đúng 1 từ *real/fake*

![[Pasted image 20260807095558.png]]

Ở stage 1, model học được cách phân loại nhưng LLM không được trực tiếp vector embedding này vì: 
- dimension khác nhau 
- distribution khác nhau
- Classifier embedding không nằm trong language token space 
- LLM chưa học cách liên hệ vector fused với dự đoán từ real/fake

#### Trainable Projector 
Projector chuyển fused embedding từ visual feature sang hidden space của MLLM 
Công dụng: 
- Khớp dimension giữa visual expert và LLM 
- Tạo visual tokens mà attention layer của MLLM có thể xử lý 
- Bảo tồn thông tin  real/fake từ Stage 1 trong MLLM sử dụng được 

#### Công dụng của token predict 
Nó kiểm tra các tín hiệu trong token space sau đó đưa ra dự đoán token *real/fake*

### Stage 3 - Explanation Reasoning Training 
Mục tiêu: 
- Kết luận real/fake
- Giải thích bằng ngôn ngữ tự nhiên 
- Bằng chứng hỗ trợ
![[Pasted image 20260807095803.png]]

## Tại sao phải freeze Stage1 và Stage2

Trong stage 3, explanation thường dài và nhiều token ngôn ngữ. Nếu geadient language modeling cập nhật trực tiếp visual embedding hay projector, model có thể: 
- Học temple câu trả lời 
- Thay đổi  các tín hiệu để phù hợp với câu trả lời 
- Làm giảm classification accuracy 
- Ưu tiên các chi tiết thường xuất và dễ giải thích hơn các forensic thật 



# Dataset sử dụng

| Stage                     | Mục tiêu                                             | Dataset                                                                                |
| ------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Stage 1**               | Train classifier expert                              | **GenImage, GenImage++, Chameleon, UniversalFakeDetect datasets**                      |
| **Stage 2**               | Train visual representation → MLLM token `real/fake` | **GenImage + self-reconstruction data kiểu Seeing Before Reasoning**                   |
| **Stage 3**               | Train explanation/reasoning                          | **Holmes-SFTSet, ForenX dataset, Forensics-Bench/Seeing Before Reasoning data**        |
| **End-to-end evaluation** | Test classification + explanation + generalization   | **GenImage cross-generator + AIGI-Holmes benchmark + ExplainFake-Bench + ForenX test** |

