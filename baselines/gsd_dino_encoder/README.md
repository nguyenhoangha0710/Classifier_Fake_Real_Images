# Real-only GSD encoder + DINOv2-style teacher–student

Implementation của kiến trúc đã chốt: **CLIP ViT-L/14 pretrained + GSD last four layers, fixed reference k=64, DINO + iBOT + 0.1 KoLeo**, pretrain trên pool **200K ảnh real**. Không thay code Barlow Twins hiện có. Không khởi chạy GPU training khi import hoặc khi chạy tests.

[Draw.io và đặc tả](../../docs/gsd_dino_forensic_encoder/README.md).

## Model và luồng dữ liệu

```text
OFFLINE
Real train manifest -> chọn reference theo source
 -> frozen pretrained CLIP -> non-CLS features sau blocks 21..24
 -> uncentered Gram accumulation + eigh -> 4 basis [1024,64]

ONLINE
Real image -> DINO multi-crops (2 global 224, 8 local 98)
Teacher: global unmasked -> CLIP + GSD -> CLS/patch targets
Student: global token-mask + unmasked locals -> CLIP + GSD -> predictions
 -> DINO + iBOT + 0.1 KoLeo -> gradient student -> EMA teacher

INFERENCE
Export EMA teacher encoder + bases; discard prototype heads
 -> CLIP resize-short-side/center-crop/normalize -> CLS [B,1024]
```

Teacher/student có patch embeddings riêng. Mask thay nội dung patch token trước positional embedding và Transformer; CLS không mask. Cả chuỗi token (kể cả CLS) đi qua GSD sau block 21,22,23,24, giữ nguyên số token và vị trí. DINO dùng CLS sau final LayerNorm, iBOT chọn đúng masked patch positions sau LayerNorm. Không dùng CLIP text encoder/visual projection hoặc patch-mean readout.

Basis là buffer cố định, không tính lại theo batch, không nhận gradient và không EMA. Các modulation scalars a,b của GSD học ở student và EMA sang teacher. Không có semantic-anchor forward trong online training. Local crop 49 patch dùng được với k=64 vì basis được tính offline từ nhiều token reference.

## Môi trường

Notebook Kaggle: `gsd_dino_kaggle.ipynb` (tạo lại bằng `python baselines/gsd_dino_encoder/build_notebook.py`). Code được nhúng trong notebook; CLIP được tải trực tiếp từ Hugging Face khi bật Internet. Attach dataset `hoangha071005/twins-gmm-200k-checkpoint`, chứa manifest `forensic_twins_v2_200k/manifests/train_real_200k.csv`, cùng COCO/ImageNet pixels. Notebook kiểm tra hai mount modern/legacy, audit ảnh, fit fixed basis, smoke rồi main qua `RUN_MAIN=True`.

Chạy mọi lệnh từ **repository root** (đường dẫn chứa thư mục `baselines`). Dependencies ở `requirements.txt`; chọn đúng cặp torch/torchvision CUDA theo máy. Đã kiểm tra CPU với torch 2.4.1, torchvision 0.19.1, transformers 4.46.3. Chưa đo VRAM hoặc throughput với pretrained ViT-L trên GPU.

Pipeline tái sử dụng resolver mount COCO/ImageNet và heartbeat từ `baselines/gsd_forensic_twins`, không sửa module đó. Cần giữ cấu trúc repository khi chạy.

Trên Windows, có thể đặt HF cache vào workspace trước khi import Transformers:

```powershell
$env:HF_HOME = Join-Path (Get-Location) '.cache\huggingface'
```

## 1. Chuẩn bị pretrained CLIP

Nếu chưa có local snapshot, dùng downloader có sẵn trong project:

```text
python baselines/gsd_forensic_twins/download_clip.py --cache-dir CLIP_CACHE
```

Lấy `snapshot_path` trong output làm `CLIP_DIR` bên dưới. Downloader ghi resolved Hugging Face commit và chỉ tải một định dạng weight. Các CLI mới chỉ load local weights (`local_files_only=True`), không tải ngầm từ network.

## 2. Audit/filter pool ảnh real

Manifest CSV cần `image_path` hoặc `relative_path`; có thể có `sample_id`, `image_id`, `source`, `label`. Nếu label có mặt, chỉ chấp nhận `real`, `0`, `0.0`. Không có label nghĩa là người dùng chỉ định đây là real training pool; script không tự xác minh nguồn gốc ảnh. `source` nên có để reference sampling cân bằng giữa COCO/ImageNet.

```text
python -m baselines.gsd_dino_encoder.data --parent REAL200K.csv --pixel-root PIXELS --output RUN_DATA/train_real_eligible.csv --min-side 224
```

Mặc định dùng `width`/`height` có sẵn trong manifest, không mở hoặc stat từng ảnh. Chỉ dòng thiếu kích thước hợp lệ mới đọc header; log ghi `metadata` và `headers`. Có thể yêu cầu audit toàn bộ bằng `--audit header`. Ảnh thiếu/hỏng chưa được mở ở bước chuẩn bị sẽ báo lỗi khi reference/training loader đọc. Mặc định lọc min-side<224 là **giá trị triển khai khởi đầu** để thực hiện yêu cầu bỏ ảnh nhỏ. Không yêu cầu hai crop disjoint hoặc max-side>=448. Kết quả ghi actual eligible count và rejected IDs trong sidecar `.csv.json`; không khẳng định vẫn đủ 200K sau lọc. Chặn duplicate sample IDs, duplicate paths và đường dẫn tuyệt đối sau chuẩn hóa chuỗi; không resolve symlink từng file. Chưa có perceptual near-duplicate detection: train/val/test cần được chia và deduplicate upstream.

Trên Kaggle, dùng đúng manifest real200K đã lưu và `--pixel-root /kaggle/input`; resolver chỉ xét các mount COCO/ImageNet đã biết. Không scan đệ quy tìm file hoặc tự đổi cohort.

## 3. Fit fixed reference bases

```text
python -m baselines.gsd_dino_encoder.reference --clip CLIP_DIR --manifest RUN_DATA/train_real_eligible.csv --pixel-root PIXELS --output RUN_DATA/fixed_basis.pt --reference-size 4096 --tokens-per-image 64 --batch-size 8 --device cuda
```

**4096 reference images, tối đa 64 patch tokens/image/layer là cấu hình khởi đầu**, không phải hyperparameter đã chứng minh tối ưu. Sampling deterministic seed42, round-robin theo source để tránh source lớn áp đảo. Reference là subset của train, không dùng validation/test. Default reference preprocessing: EXIF transpose -> RGB -> bicubic resize short side224 -> center crop224 -> CLIP normalize. Không augmentation ngẫu nhiên, không mask.

Mỗi layer có bộ tích lũy riêng. Tính P.T@P không center theo batch, GEMM FP32 (TF32 tắt trên CUDA), cộng vào accumulator CPU FP64, cuối cùng `eigh` lấy top64 eigenvectors. Đây là các hướng right-singular của SVD không center, khác nhau ở sai số số học; không cần giữ toàn bộ P. Kiểm tra numerical rank và orthonormality. Reference features lấy sau block tương ứng của **CLIP anchor nguyên bản**, chưa thêm GSD.

Artifact lưu bases, eigenvalues, reference IDs/sources, SHA256 training/reference manifests và CLIP state, cùng preprocessing. Sidecar `fixed_basis.pt.reference.csv` giữ reference rows. Không overwrite artifact có sẵn. CPU fitting có thể rất chậm với CLIP-L; mặc định yêu cầu CUDA.

## 4. CPU contract tests

```text
python -m unittest discover -s baselines/gsd_dino_encoder/tests -v
```

Tests dùng tiny random CLIP, không download và không cần real200K. Bao gồm numerical loss checks, patch information masking, teacher no-grad/EMA, immutable basis, variable token length, local N<k, streaming Gram vs direct SVD, augmentation reproducibility, export roundtrip, overflow skip và two-rank CPU/Gloo balancing/DDP/EMA.

## 5. GPU smoke trước main training

```text
python -m baselines.gsd_dino_encoder.train --clip CLIP_DIR --basis RUN_DATA/fixed_basis.pt --manifest RUN_DATA/train_real_eligible.csv --pixel-root PIXELS --output RUNS/gsd_dino_smoke --batch-per-gpu 2 --workers 0 --smoke-steps 20
```

Batch2 là **điểm bắt đầu đo VRAM**, không phải khẳng định fits T4. Hai encoder ViT-L, hai heads rộng và 10 crops/source image tốn bộ nhớ đáng kể. Activation checkpointing của student mặc định bật. KoLeo cần **ít nhất 2 source images trên mỗi GPU**; gradient accumulation hoặc global batch2 chia thành mỗi GPU1 không giải quyết được.

Smoke checkpoint có `smoke_only=true`, không được export hoặc resume thành main encoder. Main run bắt đầu lại từ pretrained CLIP ở output directory mới. Không tự chạy main sau smoke.

## 6. Main / DDP / resume

```text
python -m baselines.gsd_dino_encoder.train --clip CLIP_DIR --basis RUN_DATA/fixed_basis.pt --manifest RUN_DATA/train_real_eligible.csv --pixel-root PIXELS --output RUNS/gsd_dino_main --batch-per-gpu LOCKED_BATCH --workers 4

torchrun --standalone --nproc_per_node=2 -m baselines.gsd_dino_encoder.train --clip CLIP_DIR --basis RUN_DATA/fixed_basis.pt --manifest RUN_DATA/train_real_eligible.csv --pixel-root PIXELS --output RUNS/gsd_dino_main_ddp --batch-per-gpu LOCKED_BATCH --workers 4
```

`LOCKED_BATCH` thay bằng số nguyên >=2 đã smoke trên mỗi GPU. Với DDP dùng NCCL cho training CUDA; global teacher-target Sinkhorn reductions gom các rank, KoLeo giữ theo từng GPU như code DINOv2. Workers dùng spawn, timeout và sample/epoch-deterministic augmentation.

```text
python -m baselines.gsd_dino_encoder.train --clip CLIP_DIR --basis RUN_DATA/fixed_basis.pt --manifest RUN_DATA/train_real_eligible.csv --pixel-root PIXELS --output RUNS/gsd_dino_main --batch-per-gpu LOCKED_BATCH --workers 4 --resume RUNS/gsd_dino_main/last.ckpt
```

Resume chỉ từ checkpoint main cuối epoch hoàn chỉnh. Khóa config, hashes, batch/world size, precision và epoch length. Khôi phục student/teacher, optimizer/scaler, successful update count và RNG từng rank. Không mid-epoch resume. `last.ckpt` được thay atomically sau mỗi epoch. Giữ checkpoint riêng nếu cần nghiên cứu nhiều epoch trước khi tiếp tục.

FP16 dùng GradScaler; GSD, heads và loss balancing FP32. Overflow skip cả optimizer, EMA và schedule trên mọi rank; tám lần liên tiếp làm fail. BF16/FP32 nonfinite gradient làm fail. Log ghi attempted/successful steps riêng, ba loss, mask fraction, norm, schedule, GSD a/b và VRAM. Heartbeat30s hiển thị stage khi native calls chạy lâu. Log không tự chứng minh forensic quality.

## Loss và giá trị config khởi đầu

- DINO: cross-entropy giữa teacher targets và student distribution trên **18 cross-view pairs/source** (với 8 locals), không ghép same global view.
- iBOT: cùng global/view/patch index, chỉ masked patches; normalize từng crop theo số mask rồi mean trên 2B crops (không-mask crop đóng góp0).
- KoLeo: normalized student global CLS, nearest other image trên mỗi GPU, tính riêng g1/g2 rồi cộng; coefficient0.1.
- Teacher target: log-domain Sinkhorn–Knopp3 iterations, tương đương phép balancing exp-domain của DINOv2 nhưng tránh overflow. Không CE/BCE real–fake.

`config.json` giữ heads theo ViT-g config DINOv2: K_D=K_P=131072, hidden2048, DINO bottleneck384, iBOT bottleneck256; đầu vào head đổi thành CLIP1024. Ba-layer MLP GELU + L2 bottleneck + weight-normalized linear. Hai head không share parameters.

Các **giá trị đề xuất cho triển khai**, chưa được người dùng xác nhận bằng thực nghiệm: 20epochs, LR backbone/GSD1e-6, heads/mask1e-4, warmup10%, teacher-temperature warmup10% tới0.07 từ0.04, weight decay0.04→0.2, EMA0.994→1, gradient clip3.0, freeze prototype last layers epoch đầu. LR không tự scale theo batch. Giữ cấu trúc loss không có nghĩa hyperparameters này tái lập DINOv2 ViT-g train từ đầu.

## 7. Export và trích CLS embeddings

```text
python -m baselines.gsd_dino_encoder.export export --checkpoint RUNS/gsd_dino_main/last.ckpt --output RUNS/gsd_dino_main/encoder.pt

python -m baselines.gsd_dino_encoder.export extract --encoder RUNS/gsd_dino_main/encoder.pt --manifest EVAL.csv --pixel-root PIXELS --output FEATURES/embeddings.npy --device cuda --batch-size 16
```

Export self-contained gồm EMA encoder + learned mask token (không dùng khi inference) + immutable bases + CLIP architecture config; không cần tải CLIP lại khi load encoder. Loại bỏ SSL heads. Extraction dùng CLS1024, FP32, không masking, không L2-normalize tự động, không lấy patch mean. Đầu ra `.npy` và `.json` giữ sample order/IDs và hashes. Encoder API vẫn trả được patch tokens và xử lý kích thước chia hết14; CLI inference hiện dùng preprocessing canonical224.

Evaluation manifest được phép có real/fake labels; đó không phải pretraining. Linear probe, Gaussian/GMM, thresholds và benchmark reports chưa được tự thêm vào pipeline này.

## Giới hạn verification

Local chỉ có CPU. Tests xác nhận correctness của thuật toán và luồng gradient trên tiny CLIP, không xác nhận chất lượng forensic, full-size VRAM hoặc CUDA/NCCL throughput. Full pretrained GPU smoke cần CLIP weights, pixels/mounted manifests và GPU thực tế. Chưa có checkpoint pretrained forensic mới từ implementation này.

Artifact `.pt/.ckpt` là dữ liệu PyTorch nội bộ, load từ file do pipeline tạo hoặc nguồn đáng tin cậy. Không dùng pickle checkpoint không rõ nguồn.

Nguồn: [DINOv2](https://arxiv.org/abs/2304.07193), [DINOv2 implementation](https://github.com/facebookresearch/dinov2), [GSD](https://arxiv.org/abs/2603.09242). Fixed-reference training + real-only CLIP/GSD/DINOv2 là extension của thiết kế này, chưa phải một kết quả đã được các paper chứng minh.
