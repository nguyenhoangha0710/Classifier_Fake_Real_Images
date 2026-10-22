# Augmented-real GMM workflow

## Mục tiêu

Giữ frozen FRE và encoder Forensic Twins epoch 75, mở rộng real manifold bằng
sáu pipeline tiền xử lý, fit nhiều GMM và tính threshold mới chỉ từ held-out
real calibration. B-Free chỉ được dùng để đánh giá sau khi model/threshold đã
được khóa.

## Kaggle Inputs bắt buộc

1. ImageNet Object Localization Challenge:
   `imagenet-object-localization-challenge`;
2. COCO 2017: `awsaf49/coco-2017-dataset`;
3. encoder/FRE/manifest:
   `hoangha071005/twins-gmm-200k-checkpoint`;
4. B-Free: `baopv051/b-free-training-data`.

Không cần `hoangha071005/fit-gmm-checkpoint-200k`, vì pipeline này fit GMM và
threshold mới. Dataset GMM cũ chỉ cần nếu muốn đối chiếu baseline ngoài
notebook.

## Notebook và script

- `forensic_twins_augmented_gmm_bfree_kaggle.ipynb`;
- `fit_augmented_gmms_bfree.py`.

Notebook chạy theo bốn stage độc lập:

```text
prepare -> extract -> fit -> evaluate
```

- `prepare`: rebase manifest 200K sang Kaggle inputs hiện tại, chia base image
  thành fit/calibration trước khi tạo variant và tạo B-Free `valid_clean`;
- `extract`: trích embedding fit/calibration cho từng variant, dùng `.npy`
  memmap và atomic progress để resume;
- `fit`: fit protocol-specific K=1/full và mixed K=1/2/4/8;
- `evaluate`: trích B-Free native embedding một lần rồi chấm bằng mọi GMM.

## Sáu real variants

| Variant | Xử lý trước center crop 256 |
|---|---|
| `native` | Không thay đổi |
| `resize512_png` | Resize toàn ảnh 512×512 bằng Lanczos; PNG round-trip không cần vì lossless |
| `jpeg_q95` | JPEG quality 95, subsampling 4:2:0 |
| `jpeg_q80` | JPEG quality 80, subsampling 4:2:0 |
| `jpeg_q65` | JPEG quality 65, subsampling 4:2:0 |
| `resize075_blur05_jpeg80` | Down/up 0.75, blur 0.5, JPEG Q80 |

Ảnh biến thể không được ghi ra disk. Transform xảy ra trong DataLoader, sau đó
chỉ embedding 192-D và progress checkpoint được lưu.

## Các GMM candidates

```text
protocol_<variant>_k1_full        # sáu model
mixed_k1_full
mixed_k1_diag
mixed_k2_diag
mixed_k4_diag
mixed_k8_diag
```

`diag` là mặc định cho phép so sánh K>1 khả thi trong 192-D trên Kaggle CPU.
Có thể đặt `MIXED_COVARIANCE="full"`, nhưng K=4/8 full covariance sẽ chậm và
tốn RAM đáng kể.

## Calibration threshold

Với mỗi GMM:

```text
real calibration embedding
-> total mixture log likelihood bằng weighted log-sum-exp
-> NLL = -log p(z)
-> q80 ... q98
```

Quy tắc dự đoán:

```text
NLL > threshold_q -> fake
```

Mỗi model/K có `thresholds.json` riêng. Mixed GMM lưu cả:

- threshold global trên tất cả calibration variants với số lượng bằng nhau;
- threshold theo từng calibration variant.

Không tái sử dụng threshold của GMM cũ hoặc của K khác.

## Pilot và full run

Notebook hiện mặc định chạy full 200K:

```python
MAX_FIT_BASE_IMAGES = None
MAX_CALIBRATION_BASE_IMAGES = None
MAX_BFREE_IDS = None
```

Khoảng 160K base images được dùng để fit và 40K base images held-out được dùng
để hiệu chuẩn threshold, tương ứng khoảng 1,2 triệu embeddings cho sáu variants.
Mặc định `MAX_GMM_FIT_EMBEDDINGS=None`: sáu protocol GMM dùng đủ 160K
embedding/variant và mixed GMM dùng đủ khoảng 960K fit embeddings. Không có
subsample 300K trong full run.
Nếu chỉ muốn chạy pilot để debug, đặt:

```python
MAX_FIT_BASE_IMAGES = 20_000
MAX_CALIBRATION_BASE_IMAGES = 5_000
MAX_GMM_FIT_EMBEDDINGS = 300_000
```

Full run có thể cần nhiều Kaggle sessions.
Điền `RESULT_DATASET_SLUG`; notebook version output sau prepare, sau từng
variant, sau fit và sau evaluate. Lần sau add output Dataset vào Input, điền
`RESUME_OUTPUT_DATASET_ROOT` và chạy lại; embedding đã hoàn thành được tái sử
dụng.

Notebook dùng `kaggle datasets ... -r zip` vì Kaggle CLI mặc định bỏ qua thư
mục con. Dataset mới mặc định là private nên không truyền cờ `--private` (CLI
không hỗ trợ cờ này). Nếu API publish tạm thời lỗi, notebook chỉ cảnh báo và vẫn
tiếp tục; `Save Version` vẫn giữ nội dung `/kaggle/working` trong Output.

Không đổi từ pilot sang full ngay trong cùng một output/resume Dataset. Kích
thước manifest và memmap của hai run khác nhau; hãy dùng một output Dataset
mới cho full run (hoặc lưu pilot sang một slug riêng) để tránh resume nhầm.

## Artifacts

```text
manifests/
├── augmented_gmm_fit_base.csv
├── augmented_gmm_calibration_base.csv
├── augmented_gmm_prepare_summary.json
└── bfree_valid_clean_eval.csv

artifacts/
├── augmented_embeddings/
│   ├── fit_<variant>_cls192.npy
│   ├── fit_<variant>_progress.json
│   ├── calibration_<variant>_cls192.npy
│   └── calibration_<variant>_progress.json
├── augmented_gmms/
│   ├── model_index.json
│   ├── real_only_model_selection.csv
│   └── <model_id>/
│       ├── gmm.npz
│       ├── gmm.json
│       ├── thresholds.json
│       └── calibration_fpr.csv
├── augmented_bfree_metrics/
│   ├── model_comparison.csv
│   └── <model_id>/
└── augmented_evaluation_complete.json
```

## Quy tắc báo cáo

- Chọn K bằng held-out real likelihood/BIC và calibration stability, không
  chọn K bằng toàn bộ nhãn B-Free rồi báo cáo lại trên cùng tập;
- B-Free `native_png` là headline protocol;
- báo cáo macro AUROC/AP, worst-generator AUROC, q80/q86/q95 và bootstrap CI;
- threshold global và threshold `resize512_png` phải được ghi nhãn riêng;
- nếu sử dụng B-Free labels để chọn model, phải chia B-Free theo base ID thành
  development và final test trước khi lựa chọn.
