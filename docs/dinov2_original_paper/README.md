# DINOv2 — sơ đồ paper gốc

Mở **dinov2_original_workflow_architecture.drawio** trong draw.io / diagrams.net bằng **File → Open From → Device**. File gồm 5 trang; mọi hộp, chữ và connector đều sửa được, không phải ảnh nhúng.

| Trang | Nội dung | Preview |
|---|---|---|
| 01 Workflow | Dữ liệu → pretrain → high-resolution adaptation → distill → đánh giá | [01.png](01.png) |
| 02 Architecture | Teacher/student, backbone, CLS/patch tokens, hai head, gradient và EMA | [02.png](02.png) |
| 03 Losses | DINO, iBOT, KoLeo; quy tắc ghép view và patch | [03.png](03.png) |
| 04 Training | Một training step và hyperparameter paper | [04.png](04.png) |
| 05 Distill and inference | Teacher lớn frozen, EMA student nhỏ, embedding khi inference | [05.png](05.png) |

Phạm vi: **DINOv2, arXiv:2304.07193v2, 02/02/2024; TMLR 01/2024**. Các trang tái dựng phương pháp, không phải hình sao chép từ paper. Không thêm thiết kế forensic/GSD, residual/frequency branch hoặc classifier real/fake vào pretraining. Nhãn tiếng Việt là phần giải thích của người dựng sơ đồ.

## Đọc sơ đồ

- Xanh lá: teacher/đầu ra teacher; xanh dương: student; cam: loss/lịch train; tím: EMA/regularization.
- Mũi tên liền: luồng tính toán; nét đứt tím: cập nhật EMA; nét đứt đỏ ở trang 02: backprop student; nét đứt xanh: target teacher đã detach.
- `z` là CLS embedding sau backbone; `P` là patch embeddings. DINO/iBOT heads chuyển token thành prototype scores phục vụ loss, không thành nhãn semantic do con người gán.
- Trang 02 triển khai ViT-g/14 ở độ phân giải global 224; bước 518 được tách ở trang 04. `B` là số ảnh nguồn trong batch.

## Quy tắc loss cần giữ đúng

Ký hiệu `g1,g2` là global views, `l1...l8` là local views; `M_g` là tập vị trí bị mask trên global view `g`. `q_t` là phân phối teacher đã detach; `p_s` là phân phối student.

```text
H(q, p) = -sum_k q[k] * log p[k]

DINO: ghép teacher global với student view khác nó.
       g1 -> g2,l1,...,l8; g2 -> g1,l1,...,l8.

iBOT: ghép teacher và student tại cùng (global view, patch index),
      chỉ tính các vị trí bị mask ở student trong pretraining.

KoLeo: trên CLS student đã L2-normalize;
       -mean_i log(min_{j != i} ||u_i-u_j||_2).

L_total = L_DINO + L_iBOT + 0.1 * L_KoLeo
```

Biểu thức tổng trên là dạng trình bày gộp theo trọng số DINO/iBOT bằng 1 trong cấu hình phát hành. Khi hiện thực, giữ các hệ số chuẩn hóa crop/mask của `ssl_meta_arch.py`; không cộng tùy ý các tổng chưa normalize. Hai head DINO và iBOT độc lập. Không có loss tái tạo RGB.

## Trình tự cập nhật

```text
Khởi tạo teacher = student; tắt requires_grad của teacher.
Lặp qua batch:
    Tạo global/local views; tạo mask trên global student.
    Teacher forward với no_grad; tạo targets có Sinkhorn-Knopp.
    Student forward; tính losses; zero_grad; backward.
    AdamW cập nhật student backbone và các head.
    Teacher <- m * teacher + (1-m) * student.
    Cập nhật lịch LR / weight decay / momentum / temperature.
```

Đây là mô tả thứ tự phụ thuộc, không phải mã chạy được. Teacher không có optimizer riêng. Embedding được tính lại từ trọng số backbone; không phải biến lưu riêng cho từng ảnh.

Distillation là một giai đoạn khác: giữ ViT-g teacher cố định, train student nhỏ và duy trì một bản EMA riêng của student nhỏ để sử dụng cuối cùng. Paper bỏ masking và stochastic depth ở giai đoạn này, nhưng vẫn dùng iBOT trên hai global crops.

## Nguồn và các khác biệt đã giữ riêng

1. [Paper HTML](https://arxiv.org/html/2304.07193v2), [PDF](https://arxiv.org/pdf/2304.07193): §3 dữ liệu; §4 loss; §5 hiệu suất/distillation; Appendix B.1–B.3 và Tables 16–17.
2. [Training implementation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/ssl_meta_arch.py): crop pairing, mask indexing, loss normalization và cập nhật EMA.
3. [Augmentation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/augmentations.py): teacher nhận hai global views của ảnh, student thêm local views; flip/color/grayscale/blur/solarize.
4. [ViT-g config](https://github.com/facebookresearch/dinov2/blob/main/dinov2/configs/train/vitg14.yaml) và [defaults](https://github.com/facebookresearch/dinov2/blob/main/dinov2/configs/ssl_default_config.yaml): cần merge overrides. `local_crops_size=98` từ ViT-g override, không lấy 96 của defaults; 8 local và 224 global từ defaults. ViT-g dùng head tách riêng và Sinkhorn–Knopp. `dataset_path` trong YAML không được xem là thay thế LVD-142M của paper.
5. [Backbone implementation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py): mask token, CLS/patch outputs, embedding dimensions.

Các liên kết code trỏ nhánh `main`, đọc ngày 21/10/2026; không phải snapshot bất biến của code năm 2023. Sơ đồ ưu tiên paper cho cấu hình thí nghiệm được công bố và dùng code để làm rõ luồng tính toán.

| Điểm khác biệt | Paper | Code phát hành được đối chiếu | Cách thể hiện |
|---|---|---|---|
| Hyperparameters | Table 16: 625k iterations, LR 3.5e-4 cho ViT-g, batch 3072, warmup 100k | YAML có base LR/quy tắc scale/epoch riêng | Trang 04 ghi số paper; không ghép mặc định YAML |
| KoLeo | B.1 mô tả CLS của global crop đầu, trong từng GPU | `ssl_meta_arch.py` tính theo cả hai nhóm global CLS, mỗi nhóm riêng | Trang 03 ghi lưu ý; không tính giữa hai view của cùng ảnh |
| ViT-B depth | Table 17 ghi 18 blocks | `vit_base()` đặt `depth=12` | Không dùng ViT-B depth để dựng sơ đồ chính; ghi rõ bất nhất này |
| Registers | Không nằm trong phương pháp paper này | Repo hiện có phiên bản có registers của công trình tiếp theo | Không thêm registers vào sơ đồ |

## Kiểm tra artifact

Chạy `python docs/dinov2_original_paper/build_diagrams.py` từ repository root để dựng lại file draw.io và 5 PNG. Script kiểm tra ID, connector references và chiều cao chữ trong mỗi hộp. PNG được vẽ từ cùng tọa độ/nội dung với XML; đã xem trực quan bản tổng hợp, chưa phải export qua ứng dụng draw.io. `overview.png` là ảnh ghép xem nhanh.
