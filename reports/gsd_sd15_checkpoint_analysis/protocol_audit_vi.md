# Kiểm tra triển khai và đặc tả giao thức so sánh GSD v1

Ngày: 20/10/2026. Phạm vi: kiểm tra code, báo cáo và cohort hiện có; không thay đổi checkpoint hoặc chạy train mới.

## Kết luận có tác động tới diễn giải

1. CLIP trong benchmark JPEG là OpenAI ViT-B/32 frozen encoder + linear head 512→2, normalize embedding. GSD hiện tại là ViT-L/14 full fine-tuning + patch-mean post-LN 1024→2. Hai model khác cả backbone, readout và cách cập nhật trọng số, chưa phải ablation chỉ thêm/bỏ GSD.
2. Case 3 của benchmark giữ real nguyên và nén fake JPEG Q96. GSD `jpeg96` nén cả hai nhãn, tương ứng `all_jpeg96`. Những so sánh trước đây gộp hai case cần được xem lại.
3. Raw CLIP của chính checkpoint trong benchmark JPEG đạt 85,7857%, không phải 87,21% trong bảng Combined người dùng cung cấp. Chưa có provenance đủ để giải thích chênh lệch; không ghép hai số để đo mức giảm JPEG.
4. Đối chiếu `relative_path` + nhãn: cả hai tập test có cùng 7.000 bản ghi, giao bằng 7.000; không có phần chỉ thuộc một benchmark. Đây là đối chiếu danh tính đường dẫn, không phải tái tải và hash lại toàn bộ ảnh.

## Kiểm tra phần GSD

Đọc runtime và đối chiếu Eq.9–12 của paper v2:

- Basis là top-k right singular vectors của frozen non-CLS tokens, ghép toàn logical batch; SVD không center.
- Projection `F V V^T`, occupancy bằng tỷ lệ năng lượng Frobenius, hệ số `lambda_max * sigmoid(a*r+b)` và phép trừ tương ứng công thức.
- Anchor được freeze và eval. Occupancy không detach; gradient tới forensic branch và a,b được giữ.
- Code chèn GSD trước bốn block cuối. Microbatch chỉ chia tính toán; basis dùng toàn batch 128/512, không riêng từng microbatch.
- Chạy lại 14 CPU tests: pass, gồm gradient, microbatch equivalence, checkpointing/resume, basis collection và overlap.

Giới hạn: các tests xác nhận các thuộc tính toán học và vận hành được kiểm tra, không chứng minh tương đương official code. Pooling post-LN bỏ CLS, preprocessing crop, độ mạnh augmentation, LR của head/modulation, weight decay và early stopping là lựa chọn triển khai. Chưa xác minh bằng checkpoint/code tác giả. Không kiểm chứng FP16/CUDA full ViT-L bằng các tests CPU nhỏ này.

Telemetry còn thiếu trong run cũ: occupancy/lambda/gradient norms. a,b cập nhật ít không đủ chứng minh GSD hỏng hoặc vô tác dụng. Checkpoint được chọn epoch 13, nên mọi đánh giá lại cùng model phải dùng best.pt đó, không dùng last.pt epoch 16.

## Hiệu ứng JPEG đúng cặp checkpoint/cohort

|Model|Raw accuracy|Fake JPEG96 accuracy|Mức giảm, điểm %|Raw fake recall|JPEG fake recall|Raw AUC|JPEG AUC|
|---|---:|---:|---:|---:|---:|---:|---:|
|CLIP linear probe|85,79%|82,54%|3,24|91,97%|85,49%|94,02%|90,36%|
|NPR-ResNet18|96,60%|58,04%|38,56|97,20%|20,09%|99,53%|60,31%|
|Full AIDE|92,30%|89,56%|2,74|87,94%|82,46%|98,38%|97,61%|

Tính lại từ confusion counts và paired predictions, không lấy raw từ bảng thí nghiệm khác.

- CLIP: 239 fake đúng→sai, 12 sai→đúng, giảm ròng 227 ảnh đúng.
- NPR: 2.707 fake đúng→sai, 8 sai→đúng, giảm ròng 2.699.
- AIDE: 209 fake đúng→sai, 17 sai→đúng, giảm ròng 192.
- Real score và real prediction giữ nguyên ở cả ba model trong case fake-only, đúng với implementation độc lập từng ảnh.

NPR fake precision 83,39% không có nghĩa bắt fake tốt: chỉ dự đoán fake ít, trong khi bỏ sót 2.797/3.500 fake. Balanced accuracy 58,04% chỉ cao hơn 8,04 điểm so với luôn dự đoán real trên cohort cân bằng. AUC giảm từ 99,53% xuống 60,31% chứng minh lỗi không chỉ ở threshold.

CLIP và AIDE vẫn giữ tín hiệu phân biệt đáng kể sau biến đổi. AIDE vừa có accuracy tuyệt đối cao nhất, vừa mức giảm nhỏ nhất trong ba checkpoint này. Chưa thể suy ra nhánh nào của AIDE tạo độ bền nếu không ablate các nhánh.

PNG roundtrip cho cùng kết quả raw ở benchmark hiện có. Điều này hỗ trợ rằng việc chỉ đóng gói lại lossless không gây mức suy giảm, còn JPEG lossy có tác động mạnh. Nhưng JPEG vừa đổi codec history vừa sửa pixel/tín hiệu forensic thật; chưa cô lập được shortcut nén với sự mất dấu vết forensic hữu ích. Gọi đây là bằng chứng nhạy với nén/codec là chắc chắn hơn kết luận tất cả hiệu năng raw đều do JPEG bias.

## Đọc bốn thí nghiệm CLIP người dùng cung cấp

- Combined All7→All7 87,21%: đo khả năng trên các nguồn đã thấy, không phải unseen generalization.
- BigGAN→BigGAN 98,80%: rất tốt trong nguồn nhưng không bảo đảm xuyên nguồn.
- 6 nguồn→GLIDE 91,80%, fake recall 100%, AUC99,03%: cho thấy đa dạng hóa nguồn train có thể hữu ích; không có phép đối chứng giữ mọi yếu tố khác nên chưa kết luận nhân quả. Nó không tương đương SD15→GLIDE.
- BigGAN→All7 66,43%, fake recall34,23%, precision96,15%: mô hình bảo thủ khi báo fake, bỏ sót nhiều fake ngoài nguồn. Pattern tương tự GSD SD15 về recall ngoài nguồn, nhưng khác backbone/source và không dùng để đo đóng góp GSD.
- Với GLIDE, chỉ bảng tổng hợp chưa cho phép xác nhận near-duplicate/real-source overlap giữa train sáu nguồn và test GLIDE. Cần manifest trước khi khẳng định generalization hoàn toàn độc lập.

## Giao thức cố định sd15_gsd_ablation_v1

Mục tiêu: cô lập đóng góp của GSD. Không gộp với mục tiêu tái lập SD14 hoặc train đa generator.

|Thành phần|Quy định|
|---|---|
|Nguồn|SD15 train; đúng manifest 3.600 train/400 validation đã lưu|
|Test|Đúng manifest 7.000 ảnh đã lưu; báo riêng SD15 và macro 6 unseen|
|Đối chứng|CLIP ViT-L/14 full fine-tune, cùng patch_mean_post_ln + classifier 1024→2|
|Nhánh GSD|Cùng backbone/head, thêm frozen anchor và GSD hiện tại|
|Seed|42 cho pilot; seed bổ sung là thí nghiệm xác nhận riêng|
|Khởi tạo|Cùng pretrained weights và classifier initialization seed|
|Training|Cùng augmentation/optimizer/config hiện có, batch128, microbatch2|
|Checkpoint|Min SD15 validation CE; patience3/min_delta0,001/max50; không lựa theo test|
|Inference|Logical batch512, cùng ordered IDs và batch IDs ở mọi case/model|
|Ngưỡng|0,5 cho tất cả; hiệu chỉnh ngưỡng phải là protocol riêng trên development set|
|JPEG|Q96, subsampling0, optimize=false, progressive=false; trước preprocessing|
|Case bắt buộc|Native, fake-only JPEG96, all-JPEG96, báo tách biệt|
|Chỉ số|Per-generator AUC/BA/real recall/fake recall, macro unseen; paired delta/flip counts|
|Provenance|SHA256 config/manifests/runtime, model/checkpoint identity và epoch được chọn|

Lưu machine-readable specification cùng hashes tại `protocol_lock_v1.json`. Đây là khóa đặc tả và evidence, không phải tuyên bố đã chạy thêm các case hoặc model còn thiếu. GSD runtime hiện chưa có fake-only case; không gán số all-JPEG cho fake-only.

Do GSD Batch-Inference dùng shared basis, nén fake có thể làm điểm real cùng batch thay đổi dù pixel real nguyên. Khi bổ sung fake-only test phải chạy lại cả batch với biến đổi đó, không ghép real-native score và fake-JPEG score đã có. Case dùng nhãn để tạo biến đổi chỉ là diagnostic, không phải preprocessing triển khai thực tế.

Đối chứng native/JPEG có thể tái sử dụng checkpoint GSD tốt nhất hiện có nếu giữ nguyên training identity, nhưng phải tách evaluation options khỏi training signature khi triển khai runner đánh giá lại. Đổi config hiện tại tùy tiện sẽ bị chặn resume. Không tắt kiểm tra signature để vượt qua.

Test hiện tại đã được xem nhiều lần. Mọi kết luận phát triển phương pháp trên nó là exploratory; cần một cohort chưa dùng để lựa chọn thiết kế cho xác nhận cuối cùng. Không sửa threshold/augmentation theo bảng rồi tiếp tục coi cùng cohort là test độc lập.

## Công việc tiếp theo theo evidence

1. Xác minh provenance của bảng Combined87,21%; không thay raw85,79% của checkpoint JPEG.
2. Hoàn thiện đánh giá fake-only cho GSD trên best.pt, chạy cả batch, giữ cohort/order và ghi provenance.
3. Train CLIP ViT-L/14 không GSD theo protocol v1. Không dùng frozen ViT-B/32 lịch sử làm đối chứng nhân quả.
4. Chỉ sau đối chứng mới quyết định mở rộng all-generator paired training hoặc SD14 reproduction. Các baseline lịch sử vẫn hữu ích làm mốc hệ thống, không đo riêng cơ chế GSD.

Nguồn code: `baselines/clip_gsd_tiny_kaggle/runtime.py`; `baselines/clip_gsd_tiny_kaggle/test_runtime.py`; `benchmarks/tiny_jpeg_bias_three_models/test_tiny_jpeg_bias_three_models_kaggle.py`; `baselines/clip_linear_probe/README.md`.

Nguồn báo cáo: `benchmarks/tiny_jpeg_bias_three_models/tiny_jpeg_bias_three_models_results/metrics/overall_by_model_case.csv`; `metrics/paired_probability_shift.csv`; `provenance/checkpoints.json`; `provenance/codec_transform.json`. Evidence hashes lưu trong protocol lock.

Paper đối chiếu: https://arxiv.org/html/2603.09242v2 , Eq.9–12 và implementation details.
