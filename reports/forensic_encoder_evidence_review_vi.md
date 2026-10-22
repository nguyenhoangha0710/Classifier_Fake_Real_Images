# Học forensic representation: đối chiếu paper, code và phương án kết hợp GSD

Ngày kiểm tra: 21/10/2026. Đây là nghiên cứu nguồn và đề xuất thí nghiệm; không chạy training hoặc xác nhận lại benchmark. Bộ24 label energy/correlation đã thảo luận trước đó không còn là hướng khuyến nghị.

## 1. Kết luận có thể hành động

Có ba nguồn supervision có cơ sở rõ hơn bộ thống kê tự chọn:

1. Metadata của ảnh chụp: học dự đoán hãng/model camera và thứ tự các thông số EXIF, như SDAIE.
2. Quan hệ giữa các patch: cùng ảnh/camera/quá trình xử lý, nhưng phải định nghĩa positive pairs theo nhiệm vụ. Forensic Twins và Noiseprint++ có mục tiêu khác nhau.
3. Một forensic teacher đã được học và đánh giá: chuyển representation sang encoder mới bằng distillation. Đây là hướng mở rộng, không phải ground truth mới.

Với dữ liệu hiện có không đảm bảo EXIF, baseline trực tiếp là residual encoder theo Forensic Twins. Để thêm một hướng độc lập có code/weights tác giả công khai, ưu tiên đánh giá SDAIE pretrained. GSD là thành phần kiểm soát semantic, không thay cho forensic supervision.

Không có bằng chứng từ các nguồn đã kiểm tra rằng GSD+real-only SSL/GMM luôn tốt hơn residual encoder chuyên biệt.

## 2. Evidence map

| Công trình | Nguồn gốc | Supervision | Điều có thể kết luận | Giới hạn |
|---|---|---|---|---|
| [Noiseprint](https://arxiv.org/abs/1808.08396) | TIFS, paper + [repo tác giả](https://github.com/grip-unina/noiseprint) | Quan hệ camera và vị trí patch | Camera-related trace có thể học với Siamese architecture | Dense fingerprint; không tự trở thành global AIGI detector |
| [Forensic Similarity](https://arxiv.org/abs/1902.04684) | TIFS | Same/different forensic trace | CNN + similarity network đánh giá camera/editing traces chưa thấy | Forensic similarity khác với phân loại real/fake |
| [TruFor / Noiseprint++](https://arxiv.org/html/2212.10957) | CVPR2023 + [code](https://github.com/grip-unina/TruFor) | Camera/source, vị trí, editing history | Học dấu vết cả in-camera và out-camera | README chưa cung cấp code train Noiseprint++; full TruFor dùng supervision localization |
| [FSD](https://arxiv.org/html/2503.21003) | CVPR2025 + [code](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25) | Predictive filters trên real; diversity | Cơ sở học residual frontend và descriptor forensic | Original self-description cần fit model từng ảnh; khác feed-forward encoder |
| [Forensic Twins v1](https://arxiv.org/html/2609.31514v1) | Preprint 25/09/2026 | Disjoint residual views + Barlow | Feed-forward forensic embedding, real-only, Gaussian | Bản paper kiểm tra vẫn hẹn link code/splits sau acceptance; local code là reproduction provisional |
| [SDAIE](https://arxiv.org/html/2512.05651) | Paper + repo ghi TPAMI2026 + [official code](https://github.com/Ekko-zn/SDAIE) | EXIF classification/ranking | Học photographic representation rồi fit GMM | Metadata coverage, source/content correlations và khác biệt paper/code |
| [BLADES](https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Bi-Level_Optimization_for_Self-Supervised_AI-Generated_Face_Detection_ICCV_2025_paper.html) | ICCV2025 + [repo](https://github.com/MZMMSEC/AIGFD_BLO) | EXIF + các face manipulations tự tạo; bi-level weights | Bài học về căn chỉnh pretext với downstream | Evidence trên face, không tự suy rộng sang general images |
| [GSD v2](https://arxiv.org/html/2603.09242v2) | Preprint phiên bản03/06/2026 | Real/fake CE trong phương pháp gốc | Suppress semantic directions trong CLIP stream | GSD không phải real-only training objective |
| [RKD](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html) | CVPR2019 + [code](https://github.com/lenscloth/RKD) | Quan hệ giữa teacher embeddings | Có cơ sở chuyển geometry giữa hai không gian khác chiều | Không phải chứng minh trực tiếp cho forensic/GSD student |

Không so sánh AUC giữa các paper như cùng một benchmark. Peer review, nguồn gốc label và khả năng tái lập là ba tiêu chí khác nhau; có paper không đồng nghĩa recipe đã hoàn chỉnh.

## 3. Phương án A: EXIF-supervised residual encoder

### 3.1 Kiến trúc có thể tái lập từ SDAIE

~~~text
Ảnh chụp RGB, kèm metadata EXIF trong tập train
   → lấy16 patch64×64; không giữ tọa độ làm positional embedding
   →30 SRM high-pass filters cho mỗi kênh RGB →90 residual channels/patch
   →CNN local encoder
   →covariance pooling + matrix square-root + upper-triangle vectorization
   →patch descriptor528
   →Transformer giữa các patch
   →image embedding528
   →14 EXIF prediction heads trong training

Inference: bỏ14 heads → image embedding → Gaussian/GMM.
~~~

Các chi tiết input90,11 convolution layers, patch64, covariance528 và Transformer2 layers có thể đối chiếu trực tiếp trong [Net.py, commit đã pin](https://github.com/Ekko-zn/SDAIE/blob/532f0fe3773e591b73d5c28924a579707f789cc4/Classifier/Net.py).

528 xuất phát từ upper triangle của covariance32×32:32×33/2=528. Đây là learned-feature covariance, không phải24 target statistics tự chọn.

### 3.2 Label thật sự đến từ đâu?

SDAIE sử dụng14 EXIF tags. Categorical tags cho class label; ordinal/continuous tags cho thứ tự giữa hai ảnh. Metadata chỉ dùng để tạo supervision trong training, không được đưa làm input model hoặc yêu cầu tồn tại lúc inference.

Ví dụ: ảnh A ghi ISO100, ảnh B ghi ISO800. Đáp án thứ tự cho cặp A,B là “A không lớn hơn B”. Đây là thông tin đo/ghi từ ảnh chụp, khác với việc tự tuyên bố energy/correlation là forensic ground truth.

EXIF vẫn có thể thiếu/sai và tương quan với nội dung. Nó là supervision cho photographic properties, không phải chứng nhận tính xác thực của ảnh. Thiết kế mask tag thiếu, canonicalize tên camera, parse rational exposure values và camera-disjoint audit là lựa chọn triển khai cần thêm nếu dùng dữ liệu riêng.

### 3.3 Loss classification

\[
h_i=E(x_i),\qquad
\mathcal L_{\mathrm{cat}}^{(t)}(i)
=-\log p_t(y_i^{(t)}\mid h_i).
\]

- \(x_i\): ảnh chụp thứ i.
- \(E\): encoder cần học.
- \(h_i\): embedding của ảnh.
- t: một EXIF tag dạng danh mục, như hãng/model camera.
- \(y_i^{(t)}\): giá trị tag thật trong metadata.
- \(p_t\): softmax prediction của head cho tag t.

Cross-entropy làm encoder giữ những đặc điểm pixel hữu ích để dự đoán metadata. Không gán tất cả ảnh nhãn real rồi train BCE.

### 3.4 Loss ranking

Với tag dạng số, như ISO:

\[
q_{ij}^{(t)}=\mathbf1[a_i^{(t)}\ge a_j^{(t)}].
\]

- \(a_i^{(t)}\): giá trị EXIF thật của ảnh i.
- \(q_{ij}^{(t)}\): đáp án0/1 về thứ tự hai ảnh, không phải real/fake.

Mỗi ảnh có một scalar score học được \(r_t(h_i)\). Theo Eq.4 trong paper:

\[
\widehat q_{ij}^{(t)}
=\Phi\left(\frac{r_t(h_i)-r_t(h_j)}{\sqrt2}\right).
\]

\[
\mathcal L_{\mathrm{rank}}^{(t)}(i,j)
=-q\log\widehat q-(1-q)\log(1-\widehat q).
\]

- \(\Phi\): cumulative distribution function của Gaussian chuẩn, biến chênh lệch score thành xác suất.
- Scalar score không phải dự đoán ISO theo đơn vị vật lý; objective chỉ học thứ tự.

Loss tổng là weighted sum các categorical/ranking losses, theo mean của ảnh hoặc các cặp hợp lệ. Các head được bỏ sau pretraining.

**Khác biệt đã kiểm tra:** [backbone_train.py](https://github.com/Ekko-zn/SDAIE/blob/532f0fe3773e591b73d5c28924a579707f789cc4/backbone_train.py) tính0.5×[1+erf(delta/sqrt2)]=Phi(delta), không phải Phi(delta/sqrt2) như Eq.4. Phải chọn paper convention hoặc code convention và lưu rõ; không gọi hai công thức là giống hệt nhau.

### 3.5 Recipe train và dữ liệu

Paper nêu Adam LR1e-4, minibatch64,30k iterations và pretraining trên0.93M YFCC photographs có đủ tags; one-class stage fit K5 trên10k real ImageNet/LSUN. Đây là recipe paper, không dự báo budget của dataset hiện tại.

Code training dùng64 samples/DataLoader trên mỗi DDP rank; mỗi sample chứa hai ảnh để ranking. Vì thế không tự gọi64 là global image batch khi chạy8 GPUs. Script hiện có vòng lặp epochs dài, không explicit stop ở30k updates. Các points này được kiểm tra trong [training script](https://github.com/Ekko-zn/SDAIE/blob/532f0fe3773e591b73d5c28924a579707f789cc4/backbone_train.py) và [dataset code](https://github.com/Ekko-zn/SDAIE/blob/532f0fe3773e591b73d5c28924a579707f789cc4/utils.py).

Không mặc định COCO/ImageNet manifests hiện có mang đủ14 EXIF tags. Cách khả thi trước mắt là đánh giá pretrained encoder trên cùng cohort dự án, hoặc chuẩn bị một corpus có metadata rõ ràng để train từ đầu.

### 3.6 Pooling: khác biệt paper/code cần khóa

Paper mô tả average pooling sau Transformer. Code [Net.py](https://github.com/Ekko-zn/SDAIE/blob/532f0fe3773e591b73d5c28924a579707f789cc4/Classifier/Net.py) lấy output_vit[:,0,:]. Code [vit.py](https://github.com/Ekko-zn/SDAIE/blob/532f0fe3773e591b73d5c28924a579707f789cc4/Classifier/vit.py) không thêm CLS token.

Đây là lấy output token đầu, không phải mean và không phải learned CLS. Không có positional embedding chỉ làm Transformer permutation-equivariant; việc chọn token đầu chưa bảo đảm image descriptor bất biến với thứ tự patch.

Đối chứng cần có: published checkpoint + exact-code pooling, và một run mean pooling riêng. Không đổi pooling rồi reuse GMM cũ.

### 3.7 Tránh nhầm one-class và binary results

SDAIE là nhánh one-class; SDAIE-dagger là nhánh có train real/fake ProGAN. Không dùng số của nhánh binary để minh họa detector “chưa thấy fake”.

Paper TableVII báo one-class full architecture Acc81.02/mAP90.39; bỏ high-pass là66.84/77.90 trong protocol riêng của bài. Đây là evidence high-pass hữu ích trong thí nghiệm đó, không bảo đảm y hệt ở dữ liệu dự án. [Paper](https://arxiv.org/html/2512.05651)

## 4. Phương án B: residual-view self-supervised encoder

### 4.1 Cơ sở Forensic Twins

~~~text
Ảnh thật → shared non-destructive augmentation → grayscale → frozen FRE
  →hai crop residual không giao nhau
  →shared ViT nhỏ → embedding
  →shared projector → Barlow Twins
~~~

Training target là quan hệ “hai views này thuộc cùng ảnh”; không có vector label thủ công và không coi mọi ảnh khác nhau chắc chắn khác camera.

Barlow Twins có cấu trúc:

\[
C_{uv}=\frac1B\sum_i\widetilde p_{i,u}^{(1)}
\widetilde p_{i,v}^{(2)},\quad
\mathcal L_{\mathrm{BT}}=
\sum_u(1-C_{uu})^2+
\beta\sum_{u\ne v}C_{uv}^2.
\]

- B: số ảnh dùng để tính statistics.
- \(p^{(1)},p^{(2)}\): outputs của projector từ hai views.
- Dấu tilde: từng chiều đã chuẩn hóa theo batch.
- C: correlation giữa các chiều ở hai views.
- u,v: chỉ số chiều feature, không phải nhãn ảnh.
- Diagonal term giữ thông tin chung; off-diagonal term giảm redundancy.

Không L2-normalize projector rồi coi là cùng normalization của Barlow. Với DDP phải thực sự tính global statistics có gradient phù hợp; gradient accumulation không tự tăng statistics batch.

### 4.2 Bằng chứng, recipe và giới hạn tái lập

Paper v1 báo RGB-view baseline60.95% global AUC, SRM84.14%, FRE93.77% trong base ablation; điều này ủng hộ việc giảm content ngay ở input. Các số thuộc cùng thí nghiệm tác giả, chưa được chạy lại tại đây.

Recipe công bố:80 epochs, AdamW LR0.0015, WD0.0025, warmup15 epochs rồi cosine về1e-6; batch320, projector8192 và491,690 real images. Final composition dùng patch8, CLS và Barlow coefficient5e-5. Crop distributions, projector internals và một số ViT hyperparameters không được coi là đầy đủ nếu thiếu code/config. [Forensic Twins v1](https://arxiv.org/html/2609.31514v1)

Local implementation chọn ViT6 blocks/width192 và các crop-size/head-count assumptions; phải ghi provenance từng trường. Bản paper có evidence6-block ablation và192-D embedding, nhưng không nên suy diễn mọi cấu hình local đều đã được tác giả xác nhận.

Ở appendix, native GMM có thể giảm mạnh dưới JPEG; thêm real compressed references chỉ phục hồi một phần. Vì vậy đánh giá encoder phải tách khỏi đánh giá density coverage. Số điểm tốt ở native không chứng minh codec invariance.

### 4.3 Vì sao đây khác conditional-statistics proposal?

Không có24 phép đo được chọn làm đáp án phải dự đoán. Encoder tìm feature chung của hai residual views dưới một objective đã có nghiên cứu forensic trực tiếp. Tuy nhiên, same-image pairs vẫn có thể giữ source/codec/content correlations; “không label” không có nghĩa không bias.

## 5. Phương án C: forensic similarity và provenance contrastive learning

[Forensic Similarity](https://arxiv.org/abs/1902.04684) có cơ sở cho việc học embedding/similarity từ same/different camera hoặc editing traces.

Riêng [Noiseprint++ trong TruFor](https://arxiv.org/html/2212.10957), positive patches cùng source, cùng vị trí và cùng editing history; dense residual extractor là DnCNN15 layers, RGB3→residual1. Editing histories là các phép xử lý được biết do người nghiên cứu áp dụng, không phải nhãn generator.

Loss gốc có dạng:

\[
\mathcal L_{\mathrm{NCE}}=
-\sum_i
\log
\frac{\sum_{j\in P_i}\exp[-d(i,j)]}
{\sum_{a\ne i}\exp[-d(i,a)]}.
\]

- i: patch đang xét.
- \(P_i\): tập positives hợp lệ theo source/position/history.
- d: squared Euclidean distance giữa residual outputs.
- Không thêm temperature hoặc L2 normalization rồi gọi là công thức gốc.

Học đúng dense fingerprint khác việc pool thành global forensic embedding. Muốn camera-level global contrastive learning có thể dùng một head riêng với positives cùng camera model nhưng khác cảnh; đây là extension, không phải Noiseprint++ exact.

**Hai điều không được trộn:**

1. Noiseprint++ muốn phân biệt một số editing histories; codec-invariance muốn bỏ qua chúng. Dùng cùng một head/loss cho hai yêu cầu có thể xung đột.
2. Noiseprint++ cần matched spatial position; Forensic Twins dùng disjoint views. Hai pair samplers không tương đương.

Theo [README train của TruFor](https://github.com/grip-unina/TruFor/blob/main/TruFor_train_test/README.md), pretrained Noiseprint++ có sẵn nhưng code train extractor chưa công bố ở phần đã kiểm tra. Không gọi reproduction train là hoàn chỉnh chỉ vì full TruFor có training scripts.

## 6. FSD là residual frontend/teacher, không tự là một encoder feed-forward

FSD học predictive filters: dự đoán pixel từ lân cận, chặn center để không copy, và khuyến khích filter diversity. Sau đó mô tả từng ảnh bằng tham số model residual nhiều scale.

\[
\mathcal L_{\mathrm{FRE}}
=\sum_k\|I-w_k*I\|^2
-\lambda\sum_a\log(\sigma_a(W)+\epsilon).
\]

- I: ảnh grayscale thật.
- \(w_k\): filter dự đoán, center0 và off-center sum1.
- W: matrix xếp filter weights thành hàng.
- \(\sigma_a(W)\): singular values; diversity term hạn chế filters trùng nhau.

Đây là công thức tóm tắt từ [FSD](https://arxiv.org/html/2503.21003). Nhiệm vụ này học residual extractor, không tự bảo đảm final embedding.

**Khác biệt paper/code đã thấy:** paper ghi FRE11×11; [official fre.py](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/blob/50f2eae06efdac2e5a33f407ca9a27a2295133ac/fsd/fre.py) default15×15 và load cấu hình theo shape checkpoint. Paper mô tả per-image iterative optimization; [fsd_computation.py](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/blob/50f2eae06efdac2e5a33f407ca9a27a2295133ac/fsd/fsd_computation.py) dùng constrained least-squares KKT kèm regularization.

Do đó phải pin checkpoint shape, pixel scale, resize và solver. Không trộn mô tả paper11×11 với code/checkpoint15×15 rồi gọi exact reproduction.

## 7. BLADES: bài học về chọn nhiệm vụ, không phải recipe cắm trực tiếp

[BLADES, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Bi-Level_Optimization_for_Self-Supervised_AI-Generated_Face_Detection_ICCV_2025_paper.html) dùng inner-loop các EXIF/manipulation pretexts, outer-loop điều chỉnh task weights theo một proxy manipulation task trên validation. Nó cho thấy bài toán representation cần nhiệm vụ được căn chỉnh với mục đích detection, không chỉ một SSL loss phổ biến.

Đây là evidence ở face domain. Proxy manipulations có đáp án rõ vì do mình tạo, nhưng vẫn không phải ground truth của mọi generative artifact. Tui không khuyên thêm bi-level optimization vào run đầu của general-image encoder: chi phí và khó kiểm soát lớn.

## 8. Kết hợp GSD theo cách nào hợp lệ?

### 8.1 Cách1: GSD encoder học trực tiếp EXIF

~~~text
RGB crop224
 ├─ Frozen CLIP semantic anchor
 └─ Trainable CLIP forensic stream
       GSD sau4 blocks cuối
       →readout embedding1024
       →categorical EXIF heads + ranking heads

Train bằng EXIF losses; bỏ heads; freeze → PCA/GMM.
~~~

Giữ hai streams cùng pretrained coordinate system. Không chiếu thẳng CNN/residual feature bằng basis CLIP chỉ vì đã thêm adapter bằng chiều.

GSD:

\[
F_\parallel=FVV^\top,\quad
o=\frac{\|F_\parallel\|_F^2}{\max(\|F\|_F^2,\epsilon)},\quad
\widehat F=F-\alpha F_\parallel.
\]

\[
\alpha=\alpha_{\max}\operatorname{sigmoid}(a\,o+b).
\]

- F: trainable CLIP tokens tại layer đang xét.
- V: các right singular vectors lấy từ frozen-anchor patch tokens.
- o: tỷ lệ năng lượng nằm trong estimated semantic subspace.
- a,b: modulation parameters học được; alpha là mức subtraction.
- Các ký hiệu alpha ở đây chỉ suppression, khác task weights của EXIF.

[GSD v2](https://arxiv.org/html/2603.09242v2) công bố k64, last4, backbone LR1e-6, a2/b1; objective gốc là real/fake CE. Thay bằng EXIF loss là đề xuất mới. Phải so CLIP+EXIF không GSD để biết GSD có giúp, và so residual SDAIE để biết CLIP backbone có cần thiết.

Không dùng patch64 của SDAIE bằng cách resize lên224 rồi coi là giữ nguyên forensic signal. Với CLIP224 nên lấy native224 riêng; thay patch strategy là một phần của extension.

### 8.2 Cách2: forensic teacher hướng dẫn GSD student

Đề xuất hữu ích khi real images của dự án thiếu EXIF:

~~~text
Frozen forensic teacher: SDAIE pretrained hoặc residual SSL encoder đã đánh giá
Real view ─────────────────────────────────────→ teacher embedding
    └→CLIP student + GSD →student embedding→alignment head
                            ↓
              feature/relational distillation loss
~~~

GSD anchor frozen và forensic teacher frozen là hai vai trò khác nhau. Không lấy frozen semantic CLIP làm teacher forensic.

Loss feature-level đơn giản:

\[
\mathcal L_{\mathrm{align}}
=\frac1B\sum_i
\|A(h_i^{S})-\operatorname{stopgrad}(h_i^{T})\|_2^2.
\]

- \(h^S\): student feature.
- \(h^T\): frozen forensic teacher feature.
- A: alignment head xử lý khác chiều/hệ tọa độ.
- Cả hai cần nhìn cùng field of view; không cho teacher thấy toàn ảnh còn student chỉ thấy một vùng không đại diện.

Nếu muốn teacher geometry chuyển vào student feature trực tiếp, có thể thử normalized pair-distance matching, lấy cảm hứng từ [RKD](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html):

\[
d^S_{ij}=\frac{\|h_i^S-h_j^S\|_2}{\bar d^S+\epsilon},
\qquad
d^T_{ij}=\frac{\|h_i^T-h_j^T\|_2}{\bar d^T+\epsilon},
\]
\[
\mathcal L_{\mathrm{rel}}
=\operatorname{mean}_{i\ne j}
\operatorname{Huber}(d^S_{ij}-\operatorname{stopgrad}(d^T_{ij})).
\]

Trong đó mean distance được tính trên các cặp off-diagonal; Huber là penalty nhỏ gần0 và tuyến tính khi sai khác lớn. Teacher feature được stop-gradient, student nhận gradient qua distance normalization. Đây là adaptation dựa trên RKD, chưa có evidence trực tiếp cho GSD+AIGI trong các nguồn đã kiểm tra.

Có thể thử L_align trước, sau đó thêm L_rel trong ablation riêng. Không gán tùy tiện hệ số1 cho mọi loss; kiểm tra gradient scale trên development split. Nếu bỏ alignment head khi extraction, cần đánh giá raw student h và aligned A(h) riêng, refit density cho từng readout.

Teacher embeddings là learned targets, không phải nhãn forensic hoàn hảo. Phải đánh giá teacher trực tiếp trước; student không được mặc định vượt teacher. So student không GSD nhưng cùng teacher/loss để xác định đóng góp riêng.

### 8.3 Cách3: residual branch và GSD branch giữ riêng

Một residual encoder đã xác nhận có thể chạy song song với GSD encoder. So từng branch và score fusion trước. Không mặc định concat feature rồi fit high-dimensional GMM sẽ tốt hơn; branch scale và covariance estimation có thể làm kết quả xấu hơn.

Đây là lựa chọn ít ép các không gian không tương thích vào cùng projection, nhưng chi phí inference tăng và chưa giải quyết được forensic supervision của branch GSD.

## 9. Recipe đề xuất cho dự án, tách khỏi recipe paper

### Phase0: đo baseline bằng encoder đã có

- Giữ nguyên cohort, pixel preprocessing và split identities.
- Đánh giá local Forensic Twins reproduction và pretrained SDAIE trên cùng real/fake cohort.
- Tách native và codec-shift; report latency, real FPR, unseen macro/worst AUROC.
- Với SDAIE pin exact-code pooling; không tự sửa sang mean rồi dùng checkpoint/GMM cũ.

### Phase1: học representation có supervision rõ

Nếu không có EXIF: residual encoder + disjoint views + Barlow là main real-only training baseline. Giữ recipe hiện có, ghi rõ crop-size/head/precision assumptions; thêm native-vs-codec training như run riêng.

Nếu có corpus EXIF đủ tin cậy: tái lập SDAIE pretraining trước khi thay backbone. Chỉ dùng pixels làm input, metadata làm labels. Handle missing tags bằng mask nếu không lọc complete-case; đó là thay đổi so complete-metadata protocol.

Nếu dùng teacher: freeze pretrained forensic model; cùng real native224 views cho student/teacher theo input contract của từng model. Không cần dataset mới phải có EXIF, nhưng phải khai báo teacher đã học từ external photographic metadata.

### Phase2: GSD ablation

Đề xuất pilot cho CLIP-L/14 extension, không phải source recipe đầy đủ:

- Native224 views; CLIP normalization đúng checkpoint.
- Frozen anchor; train last4 student blocks + readout/alignment heads ở pilot; full fine-tuning là một run riêng.
- GSD last4/k64; start LR1e-6 cho trainable CLIP blocks,1e-4 cho new heads; modulation LR phải log riêng.
- Dùng AdamW, giữ compute/update budget như paired no-GSD control;5% warmup và20-epoch cap là lựa chọn pilot.
- Đo VRAM và step time trước khi chốt batch; không mặc định2×T4 đủ.
- Đầu tiên chọn một objective: EXIF hoặc teacher alignment; chưa ghép nhiều loss chưa có đối chứng.

Để density representation không phụ thuộc test cohort, extraction real-fit/calibration/test dùng cùng per-sample GSD. Train batch-SVD→sample extraction vẫn cần audit. Fixed-reference là đối chứng có artifact riêng; không reuse density qua modes.

### Phase3: freeze và fit density

Fit feature transforms/PCA chỉ trên real-fit; calibrate threshold trên real-calibration riêng. K1/full covariance là control trước K5. K5 từ SDAIE và K1 từ Forensic Twins không phải universal constants.

Nếu refit bằng real codec variants, mọi variants của cùng original ở cùng split. Encoder/GMM/calibration đều phải lưu preprocessing/basis/pooling/checkpoint identities.

## 10. Những bằng chứng cần có để gọi là forensic representation

| Kiểm tra | Tại sao cần |
|---|---|
| Camera/source-held-out | Tránh chỉ ghi nhớ camera hoặc dataset |
| Content probe trên real audit split | Đo semantic leakage; accuracy thấp riêng lẻ chưa chứng minh forensic |
| Source/processing probe | Kiểm tra giữ thông tin hình thành/xử lý ảnh |
| Effective rank, variance, nearest neighbors | Phát hiện collapse hoặc representation dư thừa |
| Native/JPEG/resize + actual real FPR | Tách representation quality khỏi real-reference mismatch |
| No-GSD paired control | Chứng minh lợi ích GSD thay vì backbone/data/objective |
| Teacher-only control | Tránh student phức tạp nhưng kém teacher |
| Per-generator và worst-generator metrics | Tránh một nhóm generator dễ nâng macro |
| Fixed final-test | Không chọn loss/threshold/checkpoint bằng unseen test |

Một representation có nhiều semantic information vẫn có thể detect tốt; không tối ưu semantic probe xuống chance bằng mọi giá. Ngược lại, model không học gì cũng có semantic accuracy thấp.

## 11. Audit nguồn và khả năng tái lập

SDAIE code được đọc và pin tại commit532f0fe3773e591b73d5c28924a579707f789cc4. Đã kiểm tra Net.py, vit.py, backbone_train.py, utils.py và oc_funs.py. Không chạy code, không tải hoặc load pretrained weights.

FSD repo được xác định HEAD50f2eae06efdac2e5a33f407ca9a27a2295133ac; đã đọc fre.py và fsd_computation.py và kiểm tra lại ở ref đã pin.

Forensic Twins dùng đúng arXiv2609.31514v1. Không tìm được link official implementation trong bản paper và lượt tìm kiếm này; điều đó không chứng minh không có code ở nơi khác.

GSD dùng đúng arXiv2603.09242v2. Không gọi implementation local là official code tác giả. Những paper/source summaries online dùng v1 “parameter-free” không được thay thế adaptive GSD v2.

README TruFor vẫn ghi Noiseprint++ training code chưa có tại phần được kiểm tra. Điều này là giới hạn reproduction, không phủ nhận pretrained fingerprint usefulness.

Tài liệu này thay khuyến nghị conditional-statistics encoder bằng các nhiệm vụ có nguồn gốc rõ. Các sơ đồ conditional_residual_encoder cũ vẫn là lịch sử đề xuất chưa được xác thực, không nên dùng làm đặc tả thực nghiệm chính.
