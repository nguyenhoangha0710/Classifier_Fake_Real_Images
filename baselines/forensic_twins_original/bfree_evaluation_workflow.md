# B-Free zero-shot evaluation workflow

## Mục tiêu

Đánh giá encoder Forensic Twins 200K và fitted real-only GMM trên B-Free mà
không train lại, không fit lại GMM và không hiệu chỉnh threshold bằng nhãn
B-Free. Positive class là `fake`; dự đoán fake khi `NLL > threshold`.

## Kaggle Inputs

1. <https://www.kaggle.com/datasets/baopv051/b-free-training-data>
2. <https://www.kaggle.com/datasets/hoangha071005/twins-gmm-200k-checkpoint>
3. <https://www.kaggle.com/datasets/hoangha071005/fit-gmm-checkpoint-200k>

Không cần add lại ImageNet, COCO hoặc Tiny-GenImage.

## Cohort chính

Script đọc `train_list.csv` và `valid_list.csv`, sau đó lấy COCO ID từ
`train_real_200k.csv`. Mọi B-Free ID từng xuất hiện trong manifest real 200K
đều bị loại cùng toàn bộ sáu fake variant tương ứng.

Audit hiện tại:

| Cohort | Tổng ID | Overlap | Không overlap |
|---|---:|---:|---:|
| B-Free toàn bộ | 51.517 | 43.558 | 7.959 |
| B-Free valid | 10.304 | 8.747 | 1.557 |

Kết quả chính dùng 1.557 `valid_clean` ID, tương ứng 1.557 real và 9.342 fake
thuộc sáu variant. Mỗi variant được đánh giá riêng trên cặp cân bằng 1.557 real
và 1.557 fake; macro-average được tính sau cùng.

## Chạy trên Kaggle

Notebook tự chứa source:

```text
forensic_twins_bfree_eval_kaggle.ipynb
```

Thiết lập nên dùng cho lần chạy chính:

```python
COHORT = "valid_clean"
MAX_IDS = None
RUN_IMAGE_AUDIT = True
BOOTSTRAP_SAMPLES = 1000
USE_BOTH_GPUS = True
```

Để smoke test:

```python
MAX_IDS = 100
BOOTSTRAP_SAMPLES = 0
PROTOCOLS = ["native_png"]
```

## Manifests được lưu

```text
manifests/
├── bfree_id_audit.csv
├── bfree_all_clean_ids.csv
├── bfree_valid_clean_ids.csv
├── bfree_valid_overlap_excluded.csv
├── bfree_valid_clean_eval.csv
├── bfree_valid_clean_evaluated_ids.csv
├── bfree_valid_clean_invalid_groups.csv
├── bfree_manifest_summary.json
└── manifest_ready.json
```

`bfree_valid_clean_eval.csv` là manifest row-level dùng trực tiếp cho inference.
Nó lưu cả `relative_path` để lần sau script có thể rebase sang Kaggle mount mới.

Notebook tạo thêm:

```text
/kaggle/working/forensic_twins_bfree_manifests.zip
```

Để lưu lâu dài, điền `MANIFEST_DATASET_SLUG`. Notebook sẽ tạo hoặc cập nhật
private Kaggle Dataset ngay sau khi manifest hoàn thành, trước khi chạy GPU.

Lần sau add Dataset manifest đó vào Input và đặt:

```python
REUSE_MANIFEST_PATH = "/kaggle/input/.../bfree_valid_clean_eval.csv"
```

## Protocols

- `native_png`;
- `jpeg_q95`, `jpeg_q80`, `jpeg_q65`, `jpeg_q50`;
- `resize_075`, `resize_050`;
- `blur_1`, `blur_2`.

Mọi degradation được áp dụng giống nhau cho real và fake. Inference luôn center
crop 256, grayscale, frozen FRE, frozen encoder, CLS 192-D và frozen GMM.

## Resume

Embedding mỗi protocol được preallocate thành `.npy`. Cứ mỗi
`EMBEDDING_CHECKPOINT_EVERY` ảnh, script flush embedding và cập nhật atomic
`*_progress.json`. Khi chạy lại với cùng manifest/protocol, script tiếp tục từ
`next_index` hoặc tái sử dụng embedding đã hoàn thành.

Nếu điền `RESULT_DATASET_SLUG`, notebook upload output sau từng protocol. Lần
sau add output Dataset đó vào Kaggle Input và điền `RESUME_OUTPUT_DATASET_ROOT`
để copy partial artifacts về `/kaggle/working` rồi tiếp tục.

## Artifacts chính

```text
artifacts/
├── embeddings/
│   ├── <protocol>_cls192.npy
│   └── <protocol>_progress.json
├── metrics/
│   ├── protocol_comparison.csv
│   └── <protocol>/
│       ├── summary.json
│       ├── ranking.csv
│       ├── threshold_metrics.csv
│       ├── predictions.csv
│       └── bootstrap_ci95.json
├── figures/
│   ├── protocol_auroc_comparison.png
│   └── <protocol>/
├── protocol_markers/
├── run_provenance.json
└── evaluation_complete.json
```

Confusion matrices được lưu cho q80, q86 và q95. AUROC/AP được tính riêng cho
sáu fake variant. Bootstrap resample theo B-Free ID và luôn giữ real cùng sáu
fake tương ứng trong cùng một resampling unit.
