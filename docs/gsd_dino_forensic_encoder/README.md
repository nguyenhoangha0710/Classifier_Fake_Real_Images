# Encoder forensic đề xuất — CLIP + fixed-reference GSD + DINOv2-style SSL

Thiết kế chốt trong cuộc trao đổi ngày 21/10/2026. Đây là tài liệu kiến trúc, chưa phải implementation hoặc kết quả thực nghiệm. Không sửa pipeline GSD/Barlow Twins đang có.

Mở **gsd_dino_forensic_encoder.drawio** trong draw.io / diagrams.net: **File → Open From → Device**. File gồm 7 trang uncompressed XML; hộp, nhãn, công thức và connector đều chỉnh sửa được.

| Trang | Nội dung | Preview |
|---|---|---|
| 01 Workflow | Dữ liệu real 200K, offline reference, online SSL, export | [01.png](01.png) |
| 02 Reference basis | Frozen CLIP, SVD theo layer, lưu và tái sử dụng basis | [02.png](02.png) |
| 03 Teacher student | Hai encoder pretrained + GSD và các head | [03.png](03.png) |
| 04 Inside encoder | Patch embedding, masking, CLIP blocks và GSD 21–24 | [04.png](04.png) |
| 05 Crops and mask | Crop ở preprocessing; mask ở token; cặp DINO/iBOT | [05.png](05.png) |
| 06 Losses | Công thức, mục tiêu, vị trí lấy token của ba loss | [06.png](06.png) |
| 07 Training and export | Gradient, EMA, inference và các thông số chưa chốt | [07.png](07.png) |

[Ảnh tổng hợp](overview.png). Xanh dương: student/dữ liệu; xanh lá: teacher/frozen encoder/output; tím: basis/GSD; cam: loss hoặc thông số cần chốt. Đường tím nét đứt thể hiện basis được đưa vào encoder, không phải gradient hoặc EMA.

## Các quyết định đã thống nhất

- Pool đầu vào: **200K ảnh real**, không thêm fake vào pretraining. Người dùng yêu cầu tạm bỏ ảnh nhỏ; ngưỡng kích thước chưa chốt nên sơ đồ không khẳng định còn đúng 200K ảnh sau lọc.
- Backbone: **image encoder CLIP ViT-L/14 pretrained**, patch size 14, hidden dimension 1024, 24 blocks. Transformer/attention là thành phần bên trong CLIP.
- GSD sau blocks 21,22,23,24; dùng **k=64**. Không thay CLS bằng patch mean trong đường DINO/iBOT này.
- **Cách B:** một basis riêng cho mỗi layer, tính trước từ reference thuộc training, dùng chung cho teacher/student và giữ cố định trong online train/inference. Không cần semantic anchor forward hoặc SVD lại theo batch online.
- Student học bằng gradient; teacher copy student lúc khởi tạo rồi cập nhật bằng EMA. Anchor/reference không phải EMA teacher.
- DINOv2-style multi-crop, masking và các loss **DINO + iBOT + 0.1 KoLeo**. Không thêm CE/BCE real–fake, Barlow Twins hoặc VICReg.
- Sau train giữ teacher EMA encoder và basis để trích CLS/patch features. Việc đánh giá hoặc huấn luyện detector downstream là một giai đoạn khác, không tự động là một phần của pretraining.

## Luồng token

```text
Ảnh raw
  -> preprocessing: crop + augmentation + resize + normalize
  -> CLIP patch embedding
  -> student: replace selected patch embeddings by learned mask token
  -> add CLS + positional embeddings -> CLIP pre-LayerNorm
  -> CLIP blocks 1..20
  -> block21 -> GSD21 -> block22 -> GSD22
  -> block23 -> GSD23 -> block24 -> GSD24
  -> final LayerNorm
  -> CLS / patch outputs
```

Teacher không thay token bằng mask. CLS không bị mask. Teacher và student có patch embedding riêng với trọng số theo hai nhánh; cùng global crop vẫn giữ cùng lưới vị trí. Không dùng text encoder hoặc CLIP visual projection.

Với mỗi crop: global 224×224 có N=256 patches; local 98×98 có N=49. Chuỗi có N+1 token tính cả CLS, hidden dimension D=1024. Positional embedding phải nội suy theo lưới patch; đây là thay đổi cần triển khai cho CLIP. Variable-length groups cần batch handling tương ứng.

GSD tại layer l nhận F_l shape [B,N+1,1024], basis V_l shape [1024,64]. GSD không thay số token hoặc thứ tự patch:

```text
F_parallel = (F_l @ V_l) @ V_l.T
r_b = ||F_parallel_b||_F^2 / (||F_l_b||_F^2 + eps)
lambda_b,l = lambda_max * sigmoid(a_l * r_b + b_l)
F_hat_l = F_l - lambda_b,l * F_parallel
```

Basis cố định không có nghĩa lambda cố định. Student học a_l,b_l; teacher nhận các tham số đó bằng EMA. Do features đầu vào khác nhau, lambda thực tế của hai nhánh có thể khác nhau. GSD áp dụng lên cả CLS và patch tokens theo luồng code GSD hiện có.

## Offline reference

Reference phải thuộc train, không lấy validation/test hoặc ảnh fake đánh giá. Frozen CLIP không mask, chạy eval/no_grad. Lấy non-CLS tokens sau từng block 21–24 của anchor nguyên bản. Gộp token theo từng layer và lấy top-64 right singular vectors bằng **uncentered SVD**; không trộn tensor của nhiều layer làm một basis.

Với P_l shape [M_l,1024], V_l shape [1024,64], kiểm tra tính hữu hạn và V_l.T @ V_l gần I_64. Số ảnh/reference crop policy chưa chốt. Với lượng token lớn cần chọn cách tính có kiểm soát bộ nhớ; eigendecomposition của tổng P_l.T @ P_l trong streaming là một cách toán học tương đương cho các hướng right-singular của SVD không center. Đây là ghi chú triển khai, chưa phải lựa chọn implementation đã chốt.

Local crop 49 tokens vẫn được chiếu lên fixed basis k=64 vì basis fit trên reference đủ lớn; không thực hiện sample-SVD k=64 trên 49 local tokens.

Lưu checkpoint/hash của CLIP anchor, reference manifest/hash, preprocessing policy, layer IDs, k, kiểu dữ liệu và bốn basis. Không fit lại basis sau khi student đã thay đổi trọng số. Không dùng crop đầy đủ của ảnh hiện tại để tạo thêm input basis cho student.

## Công thức loss và normalization

K_D/K_P là số prototypes của heads, khác k=64 của GSD. K_D/K_P và cấu hình hidden/bottleneck chưa chốt.

Gọi H(q,p) = -sum_j q_j log p_j. Student dùng softmax/log-softmax với temperature; teacher dùng teacher temperature và Sinkhorn–Knopp (3 iterations), targets detach. Hai head DINO/iBOT có tham số riêng.

**DINO:** z_s/z_t là CLS backbone outputs. p_s=softmax(head_s(z_s)/tau_s); q_t là teacher target. Tập cặp A gồm teacher g1 -> student g2 và 8 local, teacher g2 -> student g1 và 8 local. Có 18 cặp trên mỗi ảnh:

```text
L_DINO = (1 / (B*|A|)) * sum_b sum_(g,v in A) H(q_t(b,g), p_s(b,v))
```

**iBOT:** dùng patch outputs sau final LayerNorm. Teacher/student so sánh cùng global crop, cùng patch position; chỉ vị trí mask. M_b,g là tập masked indices:

```text
L_iBOT = (1/(2B)) * sum_b sum_g
         [ sum_(i in M_b,g) H(q_t^P(b,g,i), p_s^P(b,g,i))
           / max(1, |M_b,g|) ]
```

Crop không mask đóng góp 0. Giữ mask weights và crop normalization của code DINOv2; không thay bằng trung bình tùy ý trên mọi token của batch. Không có decoder tái tạo RGB.

**KoLeo:** student global CLS, trước head, chuẩn hóa L2. Tính nearest neighbor trong batch ảnh của từng global view; không trộn hai views của cùng ảnh làm ứng viên:

```text
u_b = z_b / ||z_b||_2
d_b = min_(c != b) ||u_b-u_c||_2
L_KoLeo^(g) = -mean_b log(d_b + eps)
L_KoLeo = L_KoLeo^(g1) + L_KoLeo^(g2)

L_total = L_DINO + L_iBOT + 0.1 * L_KoLeo
```

Định nghĩa KoLeo theo tổng hai global groups như implementation phát hành; paper Appendix B.1 mô tả global crop đầu. Sơ đồ giữ normalization của code để không mập mờ hệ số. Với DDP cần quy định rõ nhóm batch dùng cho nearest neighbor; code DINOv2 tính KoLeo theo từng GPU, không gom mọi GPU.

Ba loss không có label real/fake. Tính forensic của representation phải được kiểm chứng; không diễn giải loss giảm hoặc suppression tăng là bằng chứng detection tốt hơn.

## Trình tự cập nhật và các phần còn mở

```text
Build student from pretrained CLIP + GSD + learned mask token + heads.
Copy state to teacher; disable teacher gradients.
Load identical immutable basis buffers into both encoders.
For each batch:
    build views and student masks
    teacher forward under no_grad; build detached targets
    student forward; calculate three losses
    zero_grad; backward; optimizer step on student
    EMA teacher parameters from student parameters
    update schedules, log and save checkpoints
```

Teacher mask-token parameter có thể nằm trong state để copy/EMA đối xứng nhưng không được dùng trong teacher forward không-mask. Basis là buffer không được optimizer hoặc EMA cập nhật. Nếu mixed precision làm optimizer step bị skip, EMA/schedule phải đồng bộ với policy optimizer thực tế; chi tiết này cần khóa khi implementation.

Chưa chốt: reference size/sampling/preprocessing, minimum image size, prototypes/head widths, LR theo nhóm CLIP/GSD/head, batch, số bước/epoch, warmup, temperature/momentum schedules, và validation/model-selection policy. Không áp lịch ViT-g train từ đầu lên CLIP pretrained như một quyết định đã được người dùng duyệt. High-resolution adaptation hoặc model-size distillation chưa được đưa vào bản đề xuất này.

## Nguồn và giới hạn tái dựng

- [DINOv2 paper](https://arxiv.org/html/2304.07193v2): loss, teacher/student, multi-crop và evaluation.
- [GSD paper v2](https://arxiv.org/html/2603.09242v2): semantic anchor, SVD, adaptive suppression, reference-based inference. **Fixed reference basis trong SSL training là lựa chọn của đề xuất này**, không phải kết quả có sẵn của paper.
- [DINO training code](https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/ssl_meta_arch.py), [DINO loss](https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/dino_clstoken_loss.py), [iBOT loss](https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py), [KoLeo loss](https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py).
- [ViT-g overrides](https://github.com/facebookresearch/dinov2/blob/main/dinov2/configs/train/vitg14.yaml), [defaults](https://github.com/facebookresearch/dinov2/blob/main/dinov2/configs/ssl_default_config.yaml), [augmentation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/augmentations.py): multi-crop 224/98 và 8 local crops.
- [Local GSD code](../../baselines/gsd_forensic_twins/model.py): hiện dùng CLIP+GSD, patch-mean và Barlow Twins; chưa phải implementation của thiết kế mới.

## Dựng lại và kiểm tra

Từ repository root: `python docs/gsd_dino_forensic_encoder/build_diagrams.py`.

Script tạo draw.io, 7 PNG, overview và [validation.json](validation.json), kiểm tra XML IDs, connector references, page bounds, chiều cao chữ và PNG dimensions. PNG vẽ từ cùng tọa độ và nhãn với draw.io; đã kiểm tra preview trực quan, chưa export qua ứng dụng draw.io. Dùng ký hiệu toán plain-text cho subscript/transpose để tránh thiếu glyph trong font Windows.
