# Chuyển GMM xuống cấp patch, giữ 3D làm phép thử

**Kết luận chính.** Hướng (1) ở dạng "encoder GSD huấn luyện chỉ trên ảnh thật + một GMM toàn cục trên embedding" **khó thành đóng góp mới**. Mảng real-only đã đông. Forensic Self-Descriptions (CVPR 2025) và Forensic Twins (preprint 9/2026) đạt khoảng **96–98,5% AUC** trên 24–27 generator chưa thấy. Forensic Twins làm đúng công thức "Barlow Twins trên ảnh thật + một Gaussian K=1". Bằng chứng còn cho thấy encoder CLIP/RGB yếu hơn hẳn residual trong bối cảnh này: chỉ đổi đầu vào từ residual sang RGB đã làm AUC rơi từ **93,77 xuống 60,95**. Khoảng trống thật sự nằm ở chỗ khác: **chấm điểm mật độ ở cấp patch, sau khi chiếu bỏ không gian ngữ nghĩa, kèm tham chiếu nội ảnh, cho bài toán inpainting cục bộ dán lại nền gốc (origBG)**. Chưa có bài nào làm việc này, và đây cũng là chỗ detector hiện tại của bạn sụp (fake recall 13–52%). Hướng (2), nâng ảnh 2D lên 3D bằng Pi3/VGGT, **chưa có công bố nào cho ảnh tĩnh** nên mới thật. Rủi ro lại rất cao: Pi3 giữ nguyên encoder DINOv2 đóng băng, ảnh kiểu ImageNet ít cấu trúc hình học, còn với origBG thì hình học toàn cục vẫn là của ảnh thật. Vì vậy hướng 3D chỉ nên chạy như một pilot rẻ, có tiêu chí dừng rõ, và chỉ có giá trị khi chứng minh được tín hiệu đến từ phần 3D chứ không từ DINOv2. Trước cả hai hướng, cần sửa giao thức đánh giá. Con số 99,86% là in-distribution trên 7 generator đã huấn luyện. Mức rơi khi resize, nhất là 90,1% ở resize256+JPEG96, khớp với dấu hiệu shortcut định dạng/kích thước đã được ghi nhận cho GenImage. Reviewer sẽ đọc điểm này trước mọi đóng góp mới.

## Real-only toàn cục đã có chủ: FSD và Forensic Twins đạt khoảng 98 AUC

Hướng "chỉ học từ ảnh thật, coi fake là bất thường" đã phát triển thành một dòng nghiên cứu riêng giai đoạn 2024–2026. Hai mô hình mạnh nhất đều dựa trên **residual điểm ảnh**, không dựa trên đặc trưng ngữ nghĩa.

**FSD** học 8 bộ lọc dự đoán tuyến tính trên ảnh xám thật, khớp lại bộ lọc riêng cho từng ảnh, rồi đặt một GMM lên các tham số đó. Phương pháp đạt **AUC trung bình 0,960 trên 24 generator**. Cùng bảng đó, NPR được 0,926 và UFD 0,798 ([Nguồn](https://arxiv.org/html/2503.21003v1)). Có hai hạn chế. Một là AUC trường hợp xấu nhất rơi còn **0,714** khi ảnh thật lấy từ ImageNet-22k. Hai là chi phí khoảng **2,4 giây mỗi ảnh**, vì mỗi ảnh phải chạy một bài toán tối ưu ([Nguồn](https://arxiv.org/html/2609.31514)).

**Forensic Twins** gần như trùng với ý tưởng của bạn. Mô hình dùng residual FRE đóng băng, đưa qua một ViT-Tiny 3M tham số huấn luyện bằng **Barlow Twins trên hai crop không chồng nhau**, dùng 491.690 ảnh thật và không ảnh fake nào. Bộ phân loại là **một Gaussian duy nhất (GMM K=1)** chấm bằng NLL. Kết quả là **97,99 AUC trên 27 generator**, so với EFFORT (supervised) 98,96, FSD 98,52 và ConV 77,13 ([Nguồn](https://arxiv.org/html/2609.31514)).

Các ablation của Forensic Twins là bằng chứng quan trọng nhất cho hướng (1):

| Cấu hình | AUC toàn cục |
|---|---|
| Cấu hình cuối | 97,99 |
| Residual FRE | 93,77 |
| Crop chồng nhau (dùng chung nội dung) | 89,49 |
| Đầu vào RGB | 60,95 |
| Augmentation phá hủy kiểu DINOv2 | 57,84 |

Nguồn: ([Forensic Twins](https://arxiv.org/html/2609.31514)).

Bảng này nói rằng **nội dung ảnh là kẻ thù số một của mô hình mật độ real-only**. Đó đúng là lý do để dùng GSD. Nhưng nó cũng cảnh báo rằng một CLIP ViT-L lấy RGB làm đầu vào có thể không giữ đủ tín hiệu mức thấp, kể cả sau khi đã bỏ không gian ngữ nghĩa.

Các phương pháp real-only dựa trên đặc trưng nền tảng mà không khử ngữ nghĩa đều đứng cuối bảng. ConV/RIGID dùng DINOv2 và đạt 77,13 AUC trên protocol 27 generator. F-ConV thêm normalizing flow và lên 93,77 AUROC trên benchmark ImageNet của chính tác giả ([Nguồn](https://arxiv.org/html/2511.01293v1)). SDAIE dùng GMM K=5 trên đặc trưng học bằng cách dự đoán EXIF. Ở chế độ one-class, SDAIE chỉ đạt **71,7% accuracy trên GAN**. Tác giả cũng tự thừa nhận mô hình gắn nhãn bất thường cho đồ họa, tranh vẽ và **ảnh đã chỉnh sửa** ([Nguồn](https://arxiv.org/html/2512.05651v2)). SPAI (CVPR 2025) học phổ tần của ảnh thật qua tái tạo tần số có mask và đạt 91,0 AUC trung bình trên 13 generator ([Nguồn](https://arxiv.org/html/2411.19417v2)). Như vậy chính ý tưởng "mô hình hóa ảnh thật, phát hiện độ lệch" đã có nhiều đại diện được peer-review.

Hai điểm yếu chung của cả dòng này trùng đúng với điểm yếu của bạn. Thứ nhất là **nén lại**. Với Forensic Twins, chỉ cần JPEG re-encode lúc test là AUC rơi từ 97,99 xuống **61–71**. Thêm ảnh thật đã re-encode vào tập fit GMM cũng chỉ kéo lên được 64–81. Tác giả kết luận rằng mô hình one-class "không ngoại suy được ra ngoài dải nén đã hiệu chỉnh" ([Nguồn](https://arxiv.org/html/2609.31514)). Thứ hai là **resize chưa được kiểm tra**: cả FSD và Forensic Twins đều không đánh giá resize ([Nguồn](https://arxiv.org/html/2503.21003v1)). Về chọn mô hình mật độ, tài liệu gần như bỏ trống. Không bài nào so sánh có hệ thống K=1, K>1, kNN, flow và Mahalanobis trên cùng một embedding pháp y. So sánh duy nhất là của FSD: GMM 0,986 so với One-Class SVM và Isolation Forest 0,968 trên một tập con ([Nguồn](https://arxiv.org/html/2503.21003v1)). Đây là một ablation rẻ mà bạn nên báo cáo, nhưng tự nó không đủ làm đóng góp chính.

Về độ mới: GSD (Geometric Semantic Decoupling, arXiv 2603.09242) là phương pháp **supervised**. Nó dùng một CLIP đóng băng làm hướng dẫn ngữ nghĩa và chiếu bỏ các hướng ngữ nghĩa tìm được bằng SVD ([Nguồn](https://arxiv.org/abs/2603.09242)). Effort (ICML 2025) cũng tách không gian con bằng SVD, giữ phần chính cố định và chỉ huấn luyện phần dư ([Nguồn](https://arxiv.org/html/2411.15633v4)). Trong các nguồn đã tìm, **chưa có bài nào kết hợp phép chiếu bỏ không gian ngữ nghĩa kiểu GSD với một encoder chỉ học từ ảnh thật và chấm điểm bằng mật độ**. Nhưng nếu đóng góp chỉ là "thay ViT-Tiny+FRE bằng CLIP-L+GSD" mà không vượt được 97,99 của Forensic Twins, không bền hơn với JPEG/resize, và không bắt được inpainting cục bộ, thì reviewer sẽ coi đó là một phép đổi backbone.

## Khoảng trống thật sự: mật độ cấp patch cho inpainting dán lại nền gốc

Tình trạng của bạn (ảnh regenerate toàn phần bắt được 88–98%, origBG chỉ 13–52%) khớp với một kiểu thất bại đã được ghi nhận. TGIF (WIFS 2024) cho thấy bộ định vị giả mạo cổ điển bắt được inpainting kiểu ghép nối (spliced). Ngược lại, detector ảnh tổng hợp bắt được ảnh regenerate toàn phần nhưng **trượt kiểu spliced** ([Nguồn](https://arxiv.org/abs/2407.11566)). Pandolfini và cộng sự đo được rằng với mask **nhỏ hơn 5% diện tích ảnh**, detector toàn cục gần như đoán ngẫu nhiên. Inpainting của Firefly và Flux còn chỉ đạt AUROC 0,41–0,46, tức kém hơn ngẫu nhiên ([Nguồn](https://arxiv.org/html/2512.16688)).

Cơ chế khá rõ. Khi nền thật được dán trở lại, dấu vết VAE/diffusion chỉ còn bên trong vùng mask. Một embedding lấy trung bình trên các patch sẽ **pha loãng tín hiệu theo đúng tỉ lệ diện tích mask**. Đây là suy luận từ thiết kế của TGIF và DinoLizer, chưa phải kết quả đo trực tiếp. DinoLizer xác nhận gián tiếp: mô hình đạt kết quả tốt hơn khi coi vùng tái tạo qua VAE nằm ngoài mask là một lớp riêng ([Nguồn](https://arxiv.org/abs/2511.20722)).

Từ đây có một cách định vị cho hướng (1) mà chưa ai công bố. Trong công nghiệp, PatchCore chỉ học từ ảnh bình thường, chấm từng patch theo khoảng cách tới láng giềng gần nhất trong một memory bank, và đạt **99,6% AUROC ảnh trên MVTec AD** ([Nguồn](https://arxiv.org/abs/2106.08265)). Một tìm kiếm có chủ đích **không thấy bài nào áp dụng PatchCore, PaDiM hay patch-GMM chỉ học từ ảnh thật để phát hiện hoặc định vị inpainting của diffusion**. Các phương pháp định vị deepfake hiện có đều dùng nhãn patch có giám sát ([Nguồn](https://arxiv.org/pdf/2205.14852)). Tiền lệ gần nhất là Noiseprint++ trong TruFor (CVPR 2023). Noiseprint++ chỉ học từ ảnh nguyên bản và phát hiện giả mạo như độ lệch khỏi mẫu nhiễu bình thường của chính ảnh, nhưng trên residual mức thấp chứ không trên token ViT ([Nguồn](https://openaccess.thecvf.com/content/CVPR2023/papers/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.pdf)).

Từ đó, đề xuất cụ thể có ba thành phần, và mỗi thành phần cần ablation riêng:

1. **Biểu diễn patch đã khử ngữ nghĩa.** Chiếu bỏ top-k hướng SVD ước lượng trên patch token của ảnh thật, kiểu GSD/Effort. Trên token CLIP thô, các hướng chính chủ yếu mã hóa vật thể và cảnh. Nếu không khử, mật độ sẽ chỉ đo xem nội dung có điển hình hay không. Điều đó vô ích cho origBG, vì B-Free thay vật thể bằng một vật cùng loại COCO ([Nguồn](https://arxiv.org/html/2412.17671)).
2. **Mật độ cấp patch với pooling max/top-k.** Dùng GMM K>1 chéo hoặc low-rank, hoặc kNN bank có coreset. Ở 224 px, một token CLIP-L/14 phủ khoảng 0,4% ảnh, nên mask 5% tương ứng khoảng 12 token. Con số này đủ để top-k kích hoạt, với điều kiện mỗi token thực sự mang tín hiệu.
3. **Điểm tự nhất quán nội ảnh.** So từng patch với các patch còn lại của *chính ảnh đó*, không chỉ với bank toàn cục. Trong origBG, phần nền thật chính là tham chiếu sẵn có. Đây là phiên bản ngữ nghĩa-đã-khử của ý tưởng Noiseprint++.

Có thể phát biểu đóng góp như sau: *"Mật độ patch real-only trên không gian dư đã khử ngữ nghĩa phát hiện và định vị được inpainting cục bộ chưa thấy, kể cả loại mà detector supervised trượt."* Phát biểu này hẹp, kiểm chứng được, và chưa trùng với bài đã công bố nào.

Rủi ro của hướng này cần nói rõ. Rủi ro lớn nhất là **frozen CLIP-L có thể không đủ tín hiệu mức thấp ở cấp patch**. Trong B-Free, probe tuyến tính trên DINOv2 đóng băng chỉ đạt **80,8 AUC / 68,5 bAcc**, trong khi huấn luyện end-to-end đạt 99,0 / 95,2 ([Nguồn](https://arxiv.org/html/2412.17671)). Thêm vào đó, GSD khử ngữ nghĩa ở các block 21–24, nơi phần lớn thông tin mức thấp đã mất. Vì vậy cần thử các lớp trung gian (8/12/16) và một nhánh residual song song.

Rủi ro thứ hai là **object bias**. TGIF2 cho thấy bộ định vị thường học "tìm vật thể" thay vì "tìm pixel tổng hợp", và điều này lộ ra khi dùng mask ngẫu nhiên không mang ngữ nghĩa ([Nguồn](https://arxiv.org/abs/2603.28613)).

Rủi ro thứ ba là **false positive trên ảnh thật đã qua xử lý hợp lệ** như crop, filter hay nén lại. SDAIE đã ghi nhận vấn đề này ([Nguồn](https://arxiv.org/html/2512.05651v2)). Kết quả nội bộ trước đó của bạn cũng cho thấy điều tương tự: GMM trên đặc trưng phổ đạt macro AUC khoảng 0,77 và real recall 63% tại điểm vận hành.

Rủi ro cuối cùng là một baseline supervised rất rẻ có thể đã đủ tốt. Trong B-Free, chỉ cần thêm fake inpaint (gồm cả bản khôi phục nền) vào huấn luyện là đã nâng từ 94,7/81,4 lên **98,0/92,2 AUC/bAcc** ([Nguồn](https://arxiv.org/html/2412.17671)). Hướng one-class do đó phải chứng minh lợi thế trên **inpainter chưa thấy** (Flux, Firefly, BrushNet), chứ không phải trên đúng bộ sinh SD2.1 của B-Free.

## Hướng 3D mới thật, nhưng phải vượt qua cái bóng DINOv2

Phần đã công bố về hình học trong pháp y ảnh tĩnh chủ yếu dùng **cue hình học trung cấp làm thủ công**.

**Sarkar và cộng sự** ("Shadows Don't Lie…") huấn luyện bộ phân loại chỉ nhìn đoạn thẳng, perspective field và quan hệ vật–bóng, không nhìn pixel. Kết quả là AUC **0,72–0,97**. Trên các tập mà bộ phân loại pixel chỉ đoán ngẫu nhiên, họ vẫn đạt khoảng **0,82–0,84**. Nhưng dữ liệu của họ là cảnh trong nhà/ngoài trời (LSUN, BDD), và họ không có nghiên cứu độ bền với JPEG/resize ([Nguồn](https://arxiv.org/html/2311.17138v2)).

Farid phân tích định tính sự bất nhất của điểm tụ trong ảnh DALL-E 2 ([Nguồn](https://arxiv.org/pdf/2206.14617)). ControlVP cho thấy lỗi điểm tụ vẫn còn ở DALL-E 3 và Midjourney v6.1 ([Nguồn](https://arxiv.org/pdf/2512.07504)). Thêm kênh depth chỉ có bằng chứng peer-review cho deepfake khuôn mặt: DepthFake tăng frame accuracy từ 86,80% lên 91,93% ([Nguồn](https://arxiv.org/html/2208.11074v1)). Grab-3D dùng tính nhất quán hình học 3D theo thời gian, nhưng cho **video** ([Nguồn](https://arxiv.org/abs/2512.13665v1)). GAPL ghi rõ việc mô hình hóa cấu trúc 3D là **hướng tương lai chưa thử** ([Nguồn](https://arxiv.org/html/2512.12982v2)).

Nhiều lượt tìm kiếm có chủ đích đến 10/2026 **không thấy bài nào dùng đặc trưng hoặc point map của VGGT, Pi3, DUSt3R, MoGe hay Depth Anything để phát hiện ảnh tĩnh do AI sinh ra**. Vì vậy có thể bảo vệ claim "lần đầu dùng transformer hình học đa góc nhìn feed-forward làm tín hiệu pháp y". Cần kiểm tra lại trên arXiv full-text và mục "cited by" của VGGT/Pi3 trước khi nộp.

Rủi ro kiến trúc là điểm cốt lõi của hướng này. VGGT (CVPR 2025, khoảng 1,2B tham số) patchify bằng **DINOv2 ViT-L** ([Nguồn](https://arxiv.org/html/2503.11651v1)). Pi3 (ICLR 2026, khoảng 959M tham số) dùng encoder DINOv2 **đóng băng trong lúc huấn luyện, khởi tạo từ VGGT** ([Nguồn](https://arxiv.org/html/2507.13347v2)). Hệ quả là token lấy ngay sau encoder của Pi3 về cơ bản chính là token DINOv2-L. DINOv2 vốn đã mang tín hiệu artifact mức pixel ([Nguồn](https://arxiv.org/pdf/2512.06746)). Mọi cải thiện từ "đặc trưng Pi3" sẽ không diễn giải được nếu thiếu baseline DINOv2-L cùng độ phân giải và cùng head. Phần thực sự "3D" chỉ nằm ở trunk alternating-attention và ở các đầu ra point map, confidence và camera/focal.

Trên một ảnh đơn, global attention của Pi3/VGGT suy biến thành frame attention. Lợi thế đa góc nhìn chỉ quay lại nếu bạn tự tạo **pseudo-multi-view** (crop, rescale, hoặc vùng nghi ngờ so với ngữ cảnh) rồi đo tính nhất quán chéo giữa chúng. Đây là suy luận thiết kế, chưa có bằng chứng.

Ba rủi ro về dữ liệu và hiện tượng cũng nghiêm trọng không kém:

1. **Tiny-GenImage là ảnh ImageNet lấy vật thể làm trung tâm**: cận cảnh động vật, đồ vật, ít đường thẳng và ít phối cảnh. Sarkar và cộng sự đã phải giới hạn dữ liệu vào cảnh có cấu trúc. Trên ImageNet, tín hiệu hình học có thể yếu và lẫn với khác biệt bố cục giữa ảnh thật và ảnh sinh.
2. **Mô hình hình học đơn ảnh được huấn luyện để tạo ra hình học hợp lý**. Pi3 chỉ học trên dữ liệu 3D thật hoặc render, không có ảnh sinh ([Nguồn](https://arxiv.org/html/2507.13347v2)). Do đó nó có thể "làm phẳng" chính các lỗi cần bắt. Chưa có bài nào đo phân phối confidence hay depth của các mô hình này trên ảnh thật so với ảnh sinh.
3. **Với origBG, hình học toàn cục vẫn là của ảnh thật**. Chỉ tín hiệu cục bộ mới có cơ hội, chẳng hạn confidence tụt hoặc gián đoạn depth/normal tại biên mask. Nghĩa là thiết kế phải là bài toán định vị, không phải phân loại toàn cục.

Lập luận "hình học bền với resize/JPEG" nghe hợp lý vì bố cục cảnh sống sót qua downsample. Nhưng **chưa có nghiên cứu nào đo điều này**, nên nó phải được chứng minh bằng một degradation sweep.

Về tính khả thi, chi phí không phải trở ngại. VGGT cần 1,88 GB bộ nhớ đỉnh và 0,04 giây cho một khung 336×518 trên H100 ([Nguồn](https://arxiv.org/html/2503.11651v1)). MoGe-2 có các bản ViT-S 35M đến ViT-L 326M, giấy phép MIT ([Nguồn](https://github.com/microsoft/MoGe)). Trên T4 không có bf16, theo ước lượng (chưa đo) mỗi ảnh mất vài trăm ms, nên trích đặc trưng đóng băng cho 20k ảnh mất vài GPU-giờ. Fine-tune trunk 1B trên 16 GB thì không thực tế.

## Bảy đối chứng mà reviewer sẽ đòi trước mọi claim mới

Con số **99,86% khi huấn luyện trên cả 7 generator và test in-distribution không được tính là kết quả tổng quát hóa**. Các bài top venue gần đây huấn luyện trên một generator (GenImage SD1.4 hoặc ProGAN) rồi test chéo. Họ bổ sung các tập "in the wild" như Chameleon, Synthbuster, WildRF và FakeInversion, và báo cáo AUC/AP kèm bAcc@0.5, tách riêng accuracy cho ảnh thật và ảnh fake ([Nguồn](https://arxiv.org/html/2412.17671v2)). Trên Chameleon, AIDE chỉ đạt **65,77%**, và phần lớn detector gần mức ngẫu nhiên ([Nguồn](https://arxiv.org/html/2406.19435v3)).

Mức rơi khi resize của bạn nhiều khả năng một phần là triệu chứng của **shortcut định dạng/kích thước**, chứ không chỉ là mất tín hiệu tần số cao. GenImage có ảnh thật là JPEG ImageNet, phần lớn QF96, còn ảnh fake là PNG với kích thước cố định theo từng generator. Khử thiên lệch này (nén fake ở QF96, giới hạn kích thước, crop-rồi-resize giống hệt nhau khi train và test) đã tăng accuracy chéo generator thêm **+11,06 điểm** cho ResNet50 ([Nguồn](https://arxiv.org/html/2403.17608v1)). Việc resize256+JPEG96 cho điểm thấp nhất (90,1%) phù hợp với giả thuyết này nhưng chưa chứng minh nó.

Các bài SOTA xử lý resize bằng cách không resize: crop gốc và ghép tile. HiDA-Net đi từ **92,14 với 1 tile lên 96,10 khi phủ toàn ảnh** ([Nguồn](https://arxiv.org/html/2508.17346)). B-Free gần như không suy giảm qua pipeline mạng xã hội mô phỏng (99,3 → 98,5 AUC), trong khi NPR sụp từ 91,6 xuống 43,3 ([Nguồn](https://arxiv.org/html/2412.17671v2)).

Về backbone, probe tuyến tính năm 2026 xếp **PE-Core-G (89,54 F1) và SigLIP2 (86,97) trên CLIP ViT-L/14 (75,63)**. Pooling patch token đưa CLIP-L lên 82,06 ([Nguồn](https://arxiv.org/html/2604.26772v1)). Reviewer sẽ hỏi vì sao chọn CLIP-L.

Bảng dưới gom các đối chứng bắt buộc cho cả hai hướng.

| Đối chứng | Áp dụng cho | Điều nó loại trừ |
|---|---|---|
| Leave-one-generator-out và SD1.4→all trên Tiny-GenImage, kèm Chameleon/Synthbuster/FakeInversion | Cả hai | "Chỉ học thuộc generator đã thấy" |
| Fake nén khớp JPEG QF96 với real; resize giống hệt nhau; kiểm tra trên real PNG không nén | Cả hai | Shortcut định dạng/kích thước |
| Fit GMM trên ảnh thật đã qua cùng pipeline JPEG/resize; sweep resize 0,5–1,5 × JPEG 50–100 | Hướng 1 | Mật độ học "lịch sử JPEG của ImageNet" |
| Forensic Twins, FSD, SPAI, ZED, TruFor; GSD supervised có và không có fake inpaint kiểu B-Free | Hướng 1 | Đóng góp chỉ là đổi backbone, hoặc supervised rẻ đã đủ |
| Raw token so với chiếu bỏ top-k SVD so với chiếu bỏ **subspace ngẫu nhiên cùng hạng**; K=1 / K BIC / kNN / flow | Hướng 1 | Hiệu quả không đến từ khử ngữ nghĩa |
| Mask ngẫu nhiên phi ngữ nghĩa (kiểu TGIF2); AUROC theo kích thước mask; pixel F1/IoU trên TGIF/BR-Gen | Hướng 1 | Object bias, tức "tìm vật thể" thay vì "tìm pixel sinh" |
| DINOv2-L cùng độ phân giải/head; encoder Pi3 so với trunk Pi3 so với geometry-only không pixel; tập con có cấu trúc cảnh | Hướng 2 | Tín hiệu đến từ DINOv2 hoặc từ khác biệt bố cục, không phải 3D |

Cuối cùng, mọi chênh lệch dưới khoảng 1 điểm (ví dụ 99,86 so với 99,4) cần **nhiều seed và khoảng tin cậy bootstrap 95%**. Đây là thực hành lặp lại trong các bài được dẫn, chưa phải quy định chính thức của venue.

## Kế hoạch thí nghiệm theo thứ tự ưu tiên

Thứ tự dưới đây đặt việc rẻ và chặn rủi ro lên trước. Các ngưỡng go/no-go là đề xuất của báo cáo này, suy ra từ mức SOTA đã dẫn; chúng không phải chuẩn được công bố. Giai đoạn P3 (3D) có thể chạy song song với P1 vì chỉ cần trích đặc trưng đóng băng.

| Ưu tiên | Thí nghiệm | Thời gian (ước) | Go | No-go / hành động |
|---|---|---|---|---|
| **P0** | Sửa giao thức: LOGO và SD1.4→all; fake khớp QF96; test real PNG; thêm random-resize 0,5–1,5 và suy luận multi-crop/tile cho GSD supervised; 3 seed + CI. Thêm một biến thể GSD huấn luyện có fake inpaint/origBG kiểu B-Free làm **trần supervised**. | 1–2 tuần | Biết được bao nhiêu phần trong mức rơi resize là do shortcut. Có baseline sạch để so sánh | Nếu accuracy trên real PNG sụp, phải sửa dữ liệu trước mọi hướng khác |
| **P1** | Pilot mật độ patch, không huấn luyện encoder: token CLIP-L ở lớp 8/12/16/24, DINOv2-L, residual FRE/SRM; ±chiếu bỏ SVD (top-k so với ngẫu nhiên); GMM K=1/BIC và kNN-coreset; pooling mean/top-k/max; ±điểm tự nhất quán nội ảnh. Test: B-Free origBG và regenerated, TGIF/BR-Gen, Tiny-GenImage LOGO | 2–3 tuần | AUROC ảnh trên origBG **≥ 0,75**, đồng thời **vượt GSD supervised (không có fake inpaint) ≥ 10 điểm** trên inpainter chưa thấy; khử SVD tốt hơn subspace ngẫu nhiên; mask ngẫu nhiên vẫn phát hiện được | Nếu SVD không hơn subspace ngẫu nhiên thì bỏ claim "ngữ nghĩa", chuyển sang residual thuần. Nếu mọi cấu hình < 0,65 thì dừng hướng 1 |
| **P2** | Chỉ làm khi P1 = go. Huấn luyện encoder real-only: front-end residual hoặc lớp CLIP trung gian + phép chiếu GSD + Barlow Twins trên crop rời nhau, chỉ augment hình học dùng chung; fit mật độ trên real đã augment JPEG/resize | 3–4 tuần | Image-level ngang Forensic Twins (khoảng 98 AUC trên protocol của họ) **hoặc** hơn hẳn họ dưới JPEG/resize hoặc trên origBG | Nếu chỉ ngang mà không có lợi thế độ bền hay định vị thì viết thành bài phân tích/benchmark, không phải bài phương pháp |
| **P3** | Pilot 3D: MoGe-2 ViT-L và Pi3 trên 2–5k ảnh, gồm tập con lọc theo số đoạn thẳng/điểm tụ. Ablation: (a) DINOv2-L probe, (b) encoder Pi3, (c) trunk Pi3, (d) geometry-only không pixel (XYZ, normal, confidence, focal, nhất quán chéo giữa các crop), (e) detector mức thấp + (d) late fusion. Với origBG: confidence và gián đoạn depth tại biên mask | 2 tuần (song song P1) | (d) đạt **AUC > 0,7** trên generator giữ lại; (c) **> (a)** rõ ràng; (d) suy giảm ít hơn detector mức thấp dưới resize; (e) cải thiện AUC trường hợp xấu nhất | Nếu (c) ≈ (a) thì tín hiệu không phải 3D, dừng claim. Nếu (d) chỉ có tác dụng trên tập có cấu trúc cảnh thì thu hẹp phạm vi thành cảnh kiến trúc/trong nhà |
| **P4** | Hợp nhất: GSD supervised (đã sửa resize) + điểm mật độ patch real-only (+ nhánh hình học nếu P3 = go). Đánh giá đầy đủ protocol và degradation grid, ECE/NLL | 2–3 tuần | Cải thiện AUC trường hợp xấu nhất và recall origBG mà không làm tụt in-distribution | — |

## Kết luận

Câu hỏi đáng theo đuổi nhất không phải "GMM trên ảnh thật có phát hiện được fake không", vì FSD và Forensic Twins đã trả lời là có, ở mức khoảng 98 AUC. Câu hỏi đáng theo đuổi là: **một mô hình mật độ chỉ học từ ảnh thật, làm việc ở cấp patch sau khi khử ngữ nghĩa, có bắt được phần nhỏ của ảnh bị sinh lại mà detector supervised bỏ sót hay không**. Đây đúng là chỗ detector của bạn sụp, và cũng là chỗ tài liệu hiện có còn trống. Nếu câu trả lời là có, bạn có một bài phương pháp với claim hẹp nhưng sắc. Nếu là không, chính phép thử "SVD so với subspace ngẫu nhiên" và "frozen CLIP so với residual" vẫn cho ra một kết quả phân tích có giá trị, vì chưa ai đo xem mật độ trên token nền tảng bị nội dung chi phối đến mức nào.

Hướng 3D nên được xem như một giả thuyết khoa học cần bác bỏ nhanh, chưa nên coi là trụ cột của dự án. Giá trị của nó, nếu có, nằm ở chỗ đó là một kênh trực giao, không dùng pixel và có thể bền với resize, chứ không nằm ở chỗ thêm một backbone lớn hơn. Bằng chứng duy nhất có sức thuyết phục là cấu hình geometry-only vượt mức ngẫu nhiên, và trunk Pi3 vượt DINOv2-L trong cùng điều kiện. Dù chọn hướng nào, việc sửa protocol ở P0 là điều kiện tiên quyết. Một phần đáng kể của mức rơi khi resize có thể đến từ dữ liệu chứ không từ mô hình, và mọi so sánh về sau đều phụ thuộc vào việc tách bạch hai nguyên nhân này.
