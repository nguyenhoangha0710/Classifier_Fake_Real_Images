# Tổng hợp URL data, manifest, checkpoint và kết quả theo thí nghiệm

Ngày đối chiếu: **31/10/2026, giờ Việt Nam**. Phạm vi: các thí nghiệm có code, cấu hình hoặc artifact trong dự án HoangHa_Code và các dataset công khai liên quan được tìm thấy ở hai tài khoản `hadinho0710`, `hoangha071005`.

Các link Kaggle bên dưới là URL dataset; **version và đường dẫn tương đối bên trong dataset** được ghi riêng. Khi tái lập, chọn đúng version, không mặc định lấy bản mới nhất. Version “hiện tại” chỉ phản ánh lúc kiểm tra, không chứng minh đó là version đã mount trong một run cũ nếu provenance không ghi.

**Mức xác nhận:** “API xác nhận” nghĩa là đã đọc metadata hoặc inventory trực tiếp từ Kaggle; “artifact xác nhận” nghĩa là config/provenance/source_paths của run ghi nguồn đó. HTTP 403 chỉ có nghĩa là API công khai không đọc được trong lần kiểm tra, không đủ để kết luận dataset không tồn tại. Những thí nghiệm chưa có URL kết quả xác nhận được được ghi đúng là chưa tìm thấy.

**Cách đọc file chạy local:** các bảng dưới đây trỏ tới file có thật trên máy, ghi riêng notebook train/test, runtime và script phân tích. Notebook chạy trên Kaggle; các script có chữ Modal chạy qua Modal; runtime/module được notebook hoặc entry point gọi, không mặc định là script standalone. Đây là code local phù hợp từng protocol, chưa mặc định là đúng bản notebook đã submit trong lịch sử nếu chưa có hash đối chiếu. Riêng CAPD có thêm snapshot `capd_all.py` tải trực tiếp từ dataset nguồn để đối chiếu code của artifact cũ. Thay đổi code local hoặc chọn notebook phiên bản khác có thể tạo protocol khác run đã báo cáo.

## 1. Data và manifest dùng chung

| Mã | Nguồn | URL | Version / file cần chú ý | Xác nhận |
|---|---|---|---|---|
| D1 | Tiny-GenImage | [yangsangtai/tiny-genimage](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) | Hiện tại v1; ảnh của 7 generator, không phải full GenImage 8 generator | API + config các run Tiny |
| D2 | B-Free | [baopv051/b-free-training-data](https://www.kaggle.com/datasets/baopv051/b-free-training-data) | Hiện tại v2; `train_list.csv`, `valid_list.csv`, `COCO_real_512` và 6 nhánh SD2.1 | API + code/config BFree; version mount lịch sử không luôn được ghi |
| D3 | COCO 2017 | [awsaf49/coco-2017-dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) | Hiện tại v2; pixel nguồn của cohort real Twins | API metadata + tài liệu input Twins |
| D4 | ImageNet | [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) | Competition data; code dùng `ILSVRC/Data/CLS-LOC/train` | URL trang + config/source Twins và Spectral; không tải pixel kiểm tra lại |
| D5 | Community Forensics Eval | [OwensLab/CommunityForensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval) | Split `CompEval`; code benchmark tạo cohort riêng từ nguồn này. Chưa thấy commit revision được pin | Trang Hugging Face + code benchmark |
| M1 | Manifest Tiny All7 seed42 | [hadinho0710/mainfest-train-genimage](https://www.kaggle.com/datasets/hadinho0710/mainfest-train-genimage) | v1: `seed42/train_manifest.csv`, `seed42/validation_manifest.csv`, `seed42/test_manifest.csv`, `seed42/split_summary.json`, `config.json` | API inventory đủ 5 file + source_paths các run |
| W1 | DINOv3 pretrained và repo | [hadinho0710/dinov3-weight](https://www.kaggle.com/datasets/hadinho0710/dinov3-weight) | Hiện tại v1; config CAPD ghi `dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` và `dinov3/dinov3_repo` | API metadata + config CAPD; weight này là pretrained backbone, không phải checkpoint CAPD đã train |

Giữ nguyên chính tả slug **`mainfest-train-genimage`**. Manifest chứa danh sách ảnh/split, không thay thế dataset pixel. M1 là split Tiny All7 25.200 train / 2.800 validation / 7.000 test, không dùng thay manifest train SD15-only hoặc manifest BFree.

## 2. GSD — train Tiny, đánh giá Tiny/BFree

### GSD-01 — CLIP + GSD train SD15-only

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Notebook train/test SD15 | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/train_clip_vs_gsd_tiny_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/train_clip_vs_gsd_tiny_kaggle.ipynb) |
| Runtime SD15 (được notebook nhúng) | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/runtime.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/runtime.py) |
| Phân tích artifact SD15 | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_sd15_checkpoint_analysis/analyze.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_sd15_checkpoint_analysis/analyze.py) |

- **Data:** [D1 — Tiny-GenImage](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage).
- **Manifest, checkpoint và kết quả:** cùng [hadinho0710/checkpoint-gsd-sdv5-train](https://www.kaggle.com/datasets/hadinho0710/checkpoint-gsd-sdv5-train), **version 1**. Báo cáo lịch sử local xác nhận run `clip_gsd_tiny_sd15`, chọn epoch 13.
- **File cần chọn trong v1:** dưới run `clip_gsd_tiny_sd15/`: `seed42/train_manifest.csv`, `seed42/validation_manifest.csv`, `seed42/test_manifest.csv`, `seed42/clip_gsd/best.pt`, `metrics.csv`, `predictions.csv`, `macro_unseen.csv`.
- **Bằng chứng:** [báo cáo SD15 local](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_sd15_checkpoint_analysis/analysis_vi.md), config/selected_checkpoint trong `reports/gsd_sd15_checkpoint_analysis/source/`. API xác nhận dataset có lịch sử v1 và v2; đã tải riêng `config.json`, `selected_checkpoint.json`, `macro_unseen.csv` từ đúng root trên với `datasetVersionNumber=1`, đều HTTP 200; selected epoch 13 khớp báo cáo. Các path còn lại dựa trên artifact/báo cáo lịch sử, chưa liệt kê lại toàn bộ inventory v1.

### GSD-02 — CLIP + GSD All7 raw, crop tối đa 560

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Notebook train All7 raw crop560 | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/train_gsd_raw_all7_crop560_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/train_gsd_raw_all7_crop560_kaggle.ipynb) |
| Notebook test raw checkpoint 10 case | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/test_gsd_all7_crop560_variants_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/test_gsd_all7_crop560_variants_kaggle.ipynb) |
| Runtime test 10 case | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/evaluate_all7_variants.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/evaluate_all7_variants.py) |
| Phân tích checkpoint/native | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_all7_crop560_analysis/analyze.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_all7_crop560_analysis/analyze.py) |
| Quét ngưỡng từ predictions 10 case | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_all7_variants_analysis/threshold_analysis.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_all7_variants_analysis/threshold_analysis.py) |

- **Data:** [D1](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage).
- **Manifest, checkpoint và test native:** [hadinho0710/checkpoint-gsd-sdv5-train](https://www.kaggle.com/datasets/hadinho0710/checkpoint-gsd-sdv5-train), **v2**, root `gsd_raw_all7_crop560_v1/`.
- **Manifest:** `seed42/{train,validation,test}_manifest.csv` dưới root trên. **Checkpoint:** `seed42/clip_gsd/best.pt`, selected epoch 1. **Test:** `metrics.csv`, `predictions.csv`, `macro_all.csv`. API xác nhận các path này. Bản v2 chỉ lưu test `native`, không phải bộ test 10 case.
- **Test 10 case của checkpoint raw:** [hadinho0710/test-gsd-genimage-10case](https://www.kaggle.com/datasets/hadinho0710/test-gsd-genimage-10case), **v1**. Root `gsd_all7_crop560_variants/`; `metrics.csv`, `predictions.csv`, `provenance.json`, `source_paths.json`, `delta_from_native.csv`.
- **Đối chiếu:** source_paths trên server trỏ về checkpoint và test manifest trong GSD-02 v2. `metrics.csv` trên server trùng từng byte với `reports/gsd_all7_variants_analysis/source/metrics.csv`. Không ghép bộ 10 case này với checkpoint augmented.
- **Báo cáo local:** [train/native](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_all7_crop560_analysis/analysis_vi.md), [quét ngưỡng 10 case](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_all7_variants_analysis/threshold_report.md).

### GSD-03 — CLIP + GSD All7 augmented blur/JPEG

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Notebook train All7 augmented | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/train_gsd_all7_crop560_blur_jpeg_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/train_gsd_all7_crop560_blur_jpeg_kaggle.ipynb) |
| Notebook test uploaded checkpoint Tiny (chọn case bằng config) | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/test_gsd_uploaded_best_variants_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/test_gsd_uploaded_best_variants_kaggle.ipynb) |
| Runtime test Tiny | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/evaluate_uploaded_best.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/evaluate_uploaded_best.py) |
| Notebook test uploaded checkpoint BFree valid | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/test_gsd_uploaded_best_bfree_valid_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/test_gsd_uploaded_best_bfree_valid_kaggle.ipynb) |
| Runtime test BFree valid | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/evaluate_bfree_valid.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/evaluate_bfree_valid.py) |
| Phân tích kết quả native | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_augmented_native_analysis/analyze.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_augmented_native_analysis/analyze.py) |
| Phân tích 9 case | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_augmented_nine_cases/analyze.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_augmented_nine_cases/analyze.py) |
| Phân tích BFree valid | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_bfree_valid_analysis/analyze.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_bfree_valid_analysis/analyze.py) |

- **Data:** [D1](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage). **Manifest input:** [M1](https://www.kaggle.com/datasets/hadinho0710/mainfest-train-genimage), v1.
- **Bundle train đầy đủ:** [hadinho0710/checkpoint-augment-gsd-tiny](https://www.kaggle.com/datasets/hadinho0710/checkpoint-augment-gsd-tiny), **v1**. Root `gsd_all7_crop560_blur_jpeg_manifest_v3/`; manifest tại `seed42/`; checkpoint `seed42/clip_gsd/best.pt`, `last.pt`; có `history.json`, `training_status.json`, `validation_metrics.csv`, `validation_protocol.json`. API xác nhận. `training_status.json` ghi epoch 3, `stopped=false`; không gọi bundle này là run đã early-stop hoàn tất.
- **Bản checkpoint thực sự được các evaluator dưới đây dùng:** [hadinho0710/checkpoint-augmented-gsd-train-tinigenimage](https://www.kaggle.com/datasets/hadinho0710/checkpoint-augmented-gsd-train-tinigenimage), **v1**. Checkpoint được upload dưới dạng thư mục **`best/`** chứa cấu trúc serialization PyTorch, không phải file `best.pt` nguyên khối.
- Evaluator `checkpoint_inspection.json` xác nhận epoch **1**, protocol `all7_center_crop560_blur_jpeg_fixedval_v2`, signature `4595045b4c6c76c1ad4aa303a7d2f1690cfac8150add4548d24609a6c980dce6`. Chưa tải và hash toàn bộ trọng số để khẳng định `best.pt` trong bundle train và thư mục `best/` byte-identical.

| Lượt đánh giá | URL chứa kết quả | Version / root / file |
|---|---|---|
| Tiny native | [hoangha071005/evaluate-augmented-tiny-gsd-new](https://www.kaggle.com/datasets/hoangha071005/evaluate-augmented-tiny-gsd-new) | v1; `gsd_native/metrics.csv`, `source_paths.json`, `checkpoint_inspection.json`; dùng root full `gsd_native/`, không lấy `gsd_smoke/` làm kết quả chính |
| Tiny 9 case | [hoangha071005/gsd-test-nine-cases](https://www.kaggle.com/datasets/hoangha071005/gsd-test-nine-cases) | v1; `gsd_uploaded_best_variants_v1/metrics.csv`, `delta_from_native.csv`, `source_paths.json`, `checkpoint_inspection.json`; **9 case tổng cộng = native + 8 biến đổi**, không có `png_roundtrip` |
| BFree validation | [hoangha071005/test-gsd-bfree](https://www.kaggle.com/datasets/hoangha071005/test-gsd-bfree) | v1; `gsd_bfree_valid_v1/metrics.csv`, `macro_metrics.csv`, `bfree_valid_manifest.csv`, `bfree_valid_ids.csv`, `source_paths.json`, `checkpoint_inspection.json` |

Ba lượt test này có source_paths/checkpoint_inspection local xác nhận cùng nguồn checkpoint uploaded và signature nêu trên. Tiny dùng M1; BFree dùng [D2](https://www.kaggle.com/datasets/baopv051/b-free-training-data), manifest evaluation lưu ngay trong dataset kết quả. Báo cáo local: [native](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_augmented_native_analysis/report.md), [các biến đổi](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_augmented_nine_cases/report.md).

### GSD-04 — Train BFree `SD2.1_inpainted_samecat_origBG` (GSD-P2)

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Notebook train GSD-P2 | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/train_gsd_bfree_samecat_origbg_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/train_gsd_bfree_samecat_origbg_kaggle.ipynb) |
| Runtime train GSD-P2 | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/bfree_train.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/bfree_train.py) |
| Runtime audit/split dữ liệu | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/bfree_data.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/bfree_data.py) |
| Notebook test Tiny9 + BFree (URL Tiny9/BFree đã xác nhận ở dưới) | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/test_gsd_bfree_samecat_origbg_tiny9_and_bfree_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/test_gsd_bfree_samecat_origbg_tiny9_and_bfree_kaggle.ipynb) |
| Notebook chỉ test Tiny9 | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/test_gsd_bfree_samecat_origbg_tiny9_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/test_gsd_bfree_samecat_origbg_tiny9_kaggle.ipynb) |
| Notebook chỉ test BFree valid | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/test_gsd_bfree_samecat_origbg_bfree_valid_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/test_gsd_bfree_samecat_origbg_bfree_valid_kaggle.ipynb) |

- **Data:** [D2](https://www.kaggle.com/datasets/baopv051/b-free-training-data), real `COCO_real_512` và fake `SD2.1_inpainted_samecat_origBG`.
- **Audit/split và checkpoint:** [hoangha071005/check-point-gsd-train-bfee-samecat-ori](https://www.kaggle.com/datasets/hoangha071005/check-point-gsd-train-bfee-samecat-ori), **v1**; root `gsd_bfree_samecat_origbg_v1/`.
- **File split thay cho M1:** `audit_train.csv`, `audit_valid.csv`, `train_order.csv`, `validation_ids.csv`, `split_summary.json`. Split ghi 38.661 train-pool IDs, 2.000 validation IDs; mỗi virtual epoch dùng 12.600 IDs = 25.200 ảnh ghép real/fake.
- **Checkpoint:** `best.pt`, `last.pt`, `checkpoint_ve1.pt`, `checkpoint_ve2.pt`, `selected_checkpoint.json`. Selected checkpoint ghi `best_virtual_epoch=2`, trạng thái **`paused_time_budget`**, đã validate 2 virtual epoch. Không ghi run complete.
- **Kết quả train/validation nội bộ:** `validation_predictions_ve1.csv`, `validation_predictions_ve2.csv`, `history.json`, `train_log.csv` trong chính URL checkpoint trên.
- **Đánh giá BFree official valid của checkpoint GSD-P2:** [hoangha071005/test-gsd-train-bfree-test-bfree](https://www.kaggle.com/datasets/hoangha071005/test-gsd-train-bfree-test-bfree), **v1**, root `gsd_bfree_samecat_origbg_best_bfree_valid_v1/`. `checkpoint_inspection.json` xác nhận source checkpoint GSD-P2, virtual epoch 2 và signature tương ứng. Hiện **paused/time_budget**, hoàn tất 5/6 nhánh; `inpainted_diffcat_origBG` có 14.048/20.608 dòng partial. `macro_metrics.csv` chỉ là macro 5 nhánh; chưa phải kết quả đủ 6.
- **Test Tiny9 của GSD-P2:** [hoangha071005/gsd-train-bfree-test-tiny](https://www.kaggle.com/datasets/hoangha071005/gsd-train-bfree-test-tiny), **v1**, root `gsd_bfree_samecat_origbg_best_tiny9_v1/`; `completion.json` xác nhận complete đủ 9 case × 7.000 ảnh. Checkpoint virtual epoch2, signature khớp GSD-P2. [Báo cáo local](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_tiny_analysis/REPORT.md), [script tổng hợp](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_tiny_analysis/analyze.py), [script tải nguồn](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_tiny_analysis/download_source.py). Đối chiếu bổ sung ngày 01/11/2026.
- **Kiểm tra dữ liệu:** đối chiếu trực tiếp train_order.csv, validation_ids.csv của checkpoint và COCO IDs evaluation: không giao nhau. [Báo cáo local](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_bfree_analysis/REPORT.md), [script audit/tổng hợp](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_bfree_analysis/audit_and_report.py), [script tính metrics](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_bfree_analysis/analyze.py).

## 3. CAPD

### CAPD-01 — Train Tiny All7: R1 CLIP và R2 DINOv3

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Notebook train TN1 local hiện tại | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/train_capd_tn1_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/train_capd_tn1_kaggle.ipynb) |
| Bản notebook lưu v1245 (không khẳng định là bản đã submit) | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/train_capd_tn1_kaggle_v1245.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/train_capd_tn1_kaggle_v1245.ipynb) |
| Runtime CAPD local hiện tại | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/capd_runtime.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/capd_runtime.py) |
| Snapshot capd_all.py tải từ dataset checkpoint TN1 | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_checkpoint_analysis/source/capd_all.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_checkpoint_analysis/source/capd_all.py) |
| Kiểm tra checkpoint | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/check_capd_checkpoint.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/check_capd_checkpoint.py) |
| Phân tích kết quả train | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_checkpoint_analysis/analyze.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_checkpoint_analysis/analyze.py) |

- **Data:** [D1](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage). **Manifest:** [M1](https://www.kaggle.com/datasets/hadinho0710/mainfest-train-genimage), v1. **Pretrained R2:** [W1](https://www.kaggle.com/datasets/hadinho0710/dinov3-weight), v1.
- **Checkpoint/train output:** [hadinho0710/checkpoin-capd-train-tiny](https://www.kaggle.com/datasets/hadinho0710/checkpoin-capd-train-tiny), **v1**. Giữ nguyên slug **`checkpoin`**.
- **R1 CLIP:** root `capd_tn1/r1_clip/`; `seed42/best.pt`, `last_state.pt`, `history.json`, manifest ba split. Báo cáo local xác nhận run bị ngắt trong epoch 3, mới hoàn thành 2 epoch; best completed epoch 1. **Chưa có test Tiny/BFree của R1 được xác nhận trong dataset này.**
- **R2 DINOv3:** root `capd_tn1/r2_dinov3/`; `seed42/best.pt`, `selected_checkpoint.json`, `epoch_003.pt`, manifest ba split. Best epoch **3**. Test native ngay trong bundle: `seed42/tiny_eval/metrics.csv`, `predictions_native.csv`.
- **Bằng chứng:** inventory API đầy đủ 65 file và [báo cáo checkpoint local](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_checkpoint_analysis/REPORT.md).

### CAPD-02 — Test R2 Tiny 10 case và BFree validation

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Notebook eval Tiny/BFree local hiện tại | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/eval_capd_tn1_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/eval_capd_tn1_kaggle.ipynb) |
| Bản notebook eval lưu v1245 (không khẳng định là bản đã submit) | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/eval_capd_tn1_kaggle_v1245.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/eval_capd_tn1_kaggle_v1245.ipynb) |
| Snapshot capd_all.py tải từ dataset kết quả CAPD-02 | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_bfree_eval_analysis/source/capd_all.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_bfree_eval_analysis/source/capd_all.py) |
| Tải metrics/config/log từ Kaggle | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_bfree_eval_analysis/download_source.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_bfree_eval_analysis/download_source.py) |
| Tổng hợp và đối chiếu metrics | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_bfree_eval_analysis/analyze.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_bfree_eval_analysis/analyze.py) |
| Quét ngưỡng BFree từ predictions đã tải | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_bfree_eval_analysis/threshold_sweep.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_bfree_eval_analysis/threshold_sweep.py) |

- **Data Tiny/BFree:** [D1](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage), [D2](https://www.kaggle.com/datasets/baopv051/b-free-training-data).
- **Manifest Tiny:** [M1](https://www.kaggle.com/datasets/hadinho0710/mainfest-train-genimage), `seed42/test_manifest.csv`. **BFree:** evaluator đọc `valid_list.csv` của D2, kiểm tra MD5 `72e7b4be2d049029dbedca7e75af91b0`; không phải test split độc lập khác.
- **Checkpoint đầu vào:** [CAPD-01 v1](https://www.kaggle.com/datasets/hadinho0710/checkpoin-capd-train-tiny), `capd_tn1/r2_dinov3/seed42/best.pt`.
- **URL kết quả:** [hadinho0710/eval-capd-test-tiny-bfree](https://www.kaggle.com/datasets/hadinho0710/eval-capd-test-tiny-bfree), **v1**.

| Nội dung | Đường dẫn trong dataset kết quả |
|---|---|
| Tiny đủ 10 case | `capd_eval_tiny/r2_dinov3/seed42/tiny_eval/metrics.csv` và `predictions_<case>.csv` |
| BFree 6 nhánh, full/window pooling | `capd_eval_bfree/r2_dinov3/seed42/bfree_eval/metrics_bfree.csv`, `predictions_bfree.csv` |
| Checkpoint provenance | Hai root `capd_eval_tiny/r2_dinov3/` và `capd_eval_bfree/r2_dinov3/`: `config.json`, `seed42/selected_checkpoint.json`, `seed42/best.pt` |

Lưu ý: `capd_eval_bfree/.../tiny_eval/metrics.csv` chỉ chứa bản Tiny native kế thừa; lấy Tiny **10 case** ở root **`capd_eval_tiny`**. Inventory local đủ 372 file. Kết quả quét ngưỡng BFree mới làm được lưu local, **chưa có URL dataset riêng**: [THRESHOLD_REPORT.md](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_bfree_eval_analysis/THRESHOLD_REPORT.md), [threshold_sweep.json](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_tiny_bfree_eval_analysis/threshold_sweep.json).

### CAPD-03 — Train BFree (TN2: B1 CLIP / B2 DINOv3)

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Notebook train TN2 local hiện tại | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/train_capd_bfree_kaggle_v1830.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/train_capd_bfree_kaggle_v1830.ipynb) |
| Runtime train TN2 | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/capd_bfree.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/capd_bfree.py) |
| Builder notebook TN1/TN2/eval | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/build_capd_notebooks.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/build_capd_notebooks.py) |
| Notebook eval chung (cần đặt checkpoint/run TN2) | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/eval_capd_tn1_kaggle_v1830.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/eval_capd_tn1_kaggle_v1830.ipynb) |

- **Data:** [D2](https://www.kaggle.com/datasets/baopv051/b-free-training-data). **Audit/split reuse theo notebook:** [GSD-P2](https://www.kaggle.com/datasets/hoangha071005/check-point-gsd-train-bfee-samecat-ori), v1. **Pretrained DINOv3:** [W1](https://www.kaggle.com/datasets/hadinho0710/dinov3-weight).
- **Code local:** `baselines/capd/train_capd_bfree_kaggle_v1830.ipynb`, `capd_bfree.py`, `build_capd_notebooks.py` và README TN2.
- **Kết quả TN2 đã xác nhận ngày 01/11/2026:** [hadinho0710/eval-train-capd-bfee-test-tiny-and-bfree](https://www.kaggle.com/datasets/hadinho0710/eval-train-capd-bfee-test-tiny-and-bfree), **v1**, 385 file; run b2_dinov3, complete Tiny10 case và BFree6 nhánh. Config source checkpoint ghi hadinho0710/check-point-train-capd-bfree/capd_tn2; best virtual epoch2. Root kết quả: `capd_eval_tiny/b2_dinov3/seed42/tiny_eval/` và `capd_eval_bfree/b2_dinov3/seed42/bfree_eval/`. Không gán kết quả TN1 sang TN2.
- **Báo cáo/file chạy local:** [REPORT.md](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/REPORT.md), [analyze.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/analyze.py), [download_source.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/download_source.py), [snapshot capd_all.py từ dataset](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/source/capd_all.py).

## 4. Forensic Twins và GMM

### TWINS-01 — Encoder/FRE real50K, GMM50K, test Tiny

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Notebook train encoder50K | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/forensic_twins_kaggle_2xt4.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/forensic_twins_kaggle_2xt4.ipynb) |
| Script train encoder (config quyết định cohort50K/200K) | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/train_forensic_twins_kaggle.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/train_forensic_twins_kaggle.py) |
| Notebook fit GMM50K + test Tiny | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/forensic_twins_gmm_eval_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/forensic_twins_gmm_eval_kaggle.ipynb) |
| Runtime fit GMM/test Tiny | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/fit_gmm_and_evaluate.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/fit_gmm_and_evaluate.py) |

- **Pixel train theo tài liệu/source Twins:** [D3 — COCO](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) và [D4 — ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data); membership chính xác phải đọc manifest50K, không suy ra từ cohort200K. **Test:** [D1](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage).
- **Manifest + encoder:** [hoangha071005/twins-gmm-checkpoint](https://www.kaggle.com/datasets/hoangha071005/twins-gmm-checkpoint), **v1**: `forensic_twins_original/manifests/train_real_50k.csv`, `artifacts/checkpoints/last.ckpt`, `cache/fre.pt` cùng root `forensic_twins_original/`.
- **Fitted GMM + split fit/calibration + test:** [hoangha071005/check-point-gmm-of-twins](https://www.kaggle.com/datasets/hoangha071005/check-point-gmm-of-twins), **v1**, root `forensic_twins_gmm_eval/`.
- **File:** `manifests/gmm_fit_real.csv`, `gmm_calibration_real.csv`, `gmm_split_summary.json`; `artifacts/gmm/real_k1_full.npz`, `thresholds.json`; `artifacts/metrics/native_confounded_summary.json`, `native_confounded_threshold_metrics.csv`. API inventory xác nhận cả hai dataset. Không nhầm đây là fitted GMM200K.

### TWINS-02 — Encoder/FRE real200K, GMM200K, test Tiny

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Notebook train encoder200K | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/forensic_twins_200k_kaggle_2xt4.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/forensic_twins_200k_kaggle_2xt4.ipynb) |
| Script train encoder dùng chung | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/train_forensic_twins_kaggle.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/train_forensic_twins_kaggle.py) |
| Notebook fit GMM200K + test Tiny | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/forensic_twins_200k_gmm_eval_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/forensic_twins_200k_gmm_eval_kaggle.ipynb) |
| Runtime fit GMM/test Tiny dùng chung | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/fit_gmm_and_evaluate.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/fit_gmm_and_evaluate.py) |
| Notebook test native GMM trên BFree | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/forensic_twins_bfree_eval_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/forensic_twins_bfree_eval_kaggle.ipynb) |
| Runtime test native GMM trên BFree | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/test_bfree_gmm_kaggle.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/test_bfree_gmm_kaggle.py) |

- **Data train:** [D3](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) + [D4](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data), cohort 100K COCO + 100K ImageNet theo artifact. **Data test:** [D1](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage).
- **Manifest + encoder/FRE:** [hoangha071005/twins-gmm-200k-checkpoint](https://www.kaggle.com/datasets/hoangha071005/twins-gmm-200k-checkpoint), **v1**, root `forensic_twins_v2_200k/`: `manifests/train_real_200k.csv`, `artifacts/checkpoints/last.ckpt`, `cache/fre.pt`.
- **GMM + manifest fit/calibration + test:** [hoangha071005/fit-gmm-checkpoint-200k](https://www.kaggle.com/datasets/hoangha071005/fit-gmm-checkpoint-200k), **v1**, root `forensic_twins_200k_gmm_eval/`.
- **File:** `manifests/gmm_fit_real.csv`, `gmm_calibration_real.csv`, `eval_codec_matched.csv`; `artifacts/gmm/real_k1_full.npz`, `real_k1_full.json`, `thresholds.json`; `artifacts/metrics/native_raw_summary.json`, `fake_codec_matched_summary.json`, `jpeg_q{35,50,65,80,95}_summary.json` và các bảng threshold tương ứng. API xác nhận full inventory.
- Encoder bundle và fitted-GMM bundle là hai phần khác nhau; inference cần cả hai. [Tài liệu checkpoint local](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/CHECKPOINTS.md).

### TWINS-03 — Native encoder200K + augmented GMM, test BFree clean

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Notebook fit augmented GMM + test BFree | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/forensic_twins_augmented_gmm_bfree_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/forensic_twins_augmented_gmm_bfree_kaggle.ipynb) |
| Runtime fit/test augmented GMM | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/fit_augmented_gmms_bfree.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_original/fit_augmented_gmms_bfree.py) |

- **Data:** pixel fit/calibration [D3](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) + [D4](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data); pixel đánh giá [D2](https://www.kaggle.com/datasets/baopv051/b-free-training-data).
- **Encoder/FRE + parent manifest:** [TWINS-02 encoder v1](https://www.kaggle.com/datasets/hoangha071005/twins-gmm-200k-checkpoint).
- **Manifest fit/calibration và BFree, GMM, kết quả:** [hoangha071005/evaluation-twins-bfree](https://www.kaggle.com/datasets/hoangha071005/evaluation-twins-bfree), **v2**, root chính **`forensic_twins_augmented_gmm_full/`**. API inventory đủ 830 file xác nhận root này.
- **Manifest:** `manifests/augmented_gmm_fit_base.csv`, `augmented_gmm_calibration_base.csv`, `bfree_valid_clean_eval.csv`, `bfree_manifest_summary.json`.
- **GMM:** `artifacts/augmented_gmms/<model_id>/gmm.npz`. **Test:** `artifacts/augmented_bfree_metrics/<model_id>/summary.json`, `ranking.csv`, `threshold_metrics.csv`, `predictions.csv`. Model IDs gồm `mixed_k1_full`, `mixed_k1_diag`, `mixed_k2_diag`, `mixed_k4_diag`, `mixed_k8_diag` và các `protocol_*_k1_full`.
- Cohort clean theo summary đã audit: **1.557 real IDs + 9.342 fake = 10.899 ảnh**, loại overlap với train COCO. Không đánh đồng với 72.128 ảnh BFree valid của CAPD/GSD Tiny-transfer.
- URL `hoangha071005/forensic-twins-augmented-gmms` là slug output lịch sử/code; lần này API trả 403. Dùng URL `evaluation-twins-bfree` v2 đã xác nhận ở trên để lấy bản full được lưu.

### TWINS-04 — Codec encoder200K + GMM mới, test Tiny/BFree

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Notebook train codec encoder | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_codec/forensic_twins_codec_encoder_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_codec/forensic_twins_codec_encoder_kaggle.ipynb) |
| Script train codec encoder | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_codec/train_codec.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_codec/train_codec.py) |
| Notebook fit GMM mới + test Tiny/BFree | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_codec/forensic_twins_codec_gmm_eval_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_codec/forensic_twins_codec_gmm_eval_kaggle.ipynb) |
| Runtime fit GMM/test | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_codec/evaluate_codec_gmm.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_codec/evaluate_codec_gmm.py) |
| Audit train từ artifact | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_codec/audit_training.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/forensic_twins_codec/audit_training.py) |

- **Data:** pixel train/fit [D3](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) + [D4](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data); pixel test [D1](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) + [D2](https://www.kaggle.com/datasets/baopv051/b-free-training-data).
- **Parent manifest/encoder warm-start:** [TWINS-02 encoder v1](https://www.kaggle.com/datasets/hoangha071005/twins-gmm-200k-checkpoint). **Saved split/clean-eval manifest input:** [TWINS-03 v2](https://www.kaggle.com/datasets/hoangha071005/evaluation-twins-bfree).
- **Full codec checkpoint bundle:** [hoangha071005/checkpoint-encoder-twins-codec-200k](https://www.kaggle.com/datasets/hoangha071005/checkpoint-encoder-twins-codec-200k), **v1**, root `forensic_twins_codec/`: `encoder_final.pt`, `encoder_final.pt.sha256`, `fre.pt`, `last.ckpt`, `train_real_200k.csv`, `run_metadata.json`, `logs/epoch_metrics.csv`.
- **Dataset publish trong quá trình train:** [hoangha071005/forensic-twins-codec-checkpoint-encoder](https://www.kaggle.com/datasets/hoangha071005/forensic-twins-codec-checkpoint-encoder), **hiện tại v21**. File nằm ngay root: `encoder_final.pt`, `fre.pt`, `last.ckpt`, `train_real_200k.csv`. Chưa đối chiếu byte-hash toàn bộ hai bundle, không khẳng định hai URL hoàn toàn tương đương.
- **URL chứa GMM mới và kết quả:** [hoangha071005/eval-forensic-twins-codec](https://www.kaggle.com/datasets/hoangha071005/eval-forensic-twins-codec), **v1**, root `forensic_twins_codec_gmm/`. API xác nhận đủ 366 file.
- **Manifest:** `manifests/gmm_fit_real.csv`, `gmm_calibration_real.csv`, `bfree_valid_clean_eval.csv`; Tiny có `tiny/manifests/eval_codec_matched.csv`. **GMM:** `artifacts/gmm/real_k1_full.npz`, `real_k1_full.json`, `thresholds.json`.
- **Kết quả BFree:** `bfree/artifacts/metrics/<protocol>/summary.json`, `ranking.csv`, `threshold_metrics.csv`; có `native_png`, `jpeg_q95`, `jpeg_q80`, `jpeg_q65`, `jpeg_q50`, `resize_075`. **Kết quả Tiny:** xem `tiny/artifacts/metrics/` trong cùng dataset.
- GMM metadata ghi `encoder_sha256=3e6e81adbcf34285c3afd38ee6b7019e59767ac9e28416a4524724d9d771771f` và hash clean manifest `30d0dd7c2eb3063549141df8512c44e33d7fc7719703c8a1473c316390e9ab29`; khi ghép checkpoint cần kiểm tra theo hash này, không reuse fitted GMM native cho codec encoder.

## 5. Spectral/MFM–GMM

### SPECTRAL-01 — ImageNet real100K → local/global encoder → GMM → Tiny

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Notebook train MFM from scratch | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/d2_fad/spectral_gmm_branch/train_spectral_gmm_mfm_from_scratch_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/d2_fad/spectral_gmm_branch/train_spectral_gmm_mfm_from_scratch_kaggle.ipynb) |
| Runtime train/GMM | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/d2_fad/spectral_gmm_branch/train_spectral_gmm_kaggle.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/d2_fad/spectral_gmm_branch/train_spectral_gmm_kaggle.py) |
| Notebook fit stage2 GMM từ stage1 best | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/d2_fad/spectral_gmm_branch/train_stage2_gmm_from_stage1_best_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/d2_fad/spectral_gmm_branch/train_stage2_gmm_from_stage1_best_kaggle.ipynb) |
| Notebook test Tiny | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/d2_fad/spectral_gmm_branch/test_spectral_gmm_tinygenimage_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/d2_fad/spectral_gmm_branch/test_spectral_gmm_tinygenimage_kaggle.ipynb) |
| Script test Tiny | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/d2_fad/spectral_gmm_branch/test_spectral_gmm_tinygenimage_kaggle.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/d2_fad/spectral_gmm_branch/test_spectral_gmm_tinygenimage_kaggle.py) |

- **Data train:** [D4 — ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data). **Test:** [D1 — Tiny](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage).
- **Manifest + checkpoint + fitted GMM:** [hoangha071005/chech-point-gmm-spectral](https://www.kaggle.com/datasets/hoangha071005/chech-point-gmm-spectral), **v1**; giữ nguyên slug **`chech`**. Root `spectral_gmm_branch/spectral_gmm_mfm_scratch_imagenet100k_v1/`.
- **Manifest:** `manifests/imagenet_real_100k.csv`, `manifest_summary.json`. **Checkpoint:** `checkpoints/global_encoder_best.pt`, `local_encoder_best.pt`, `local_mfm_best.pt`. **GMM:** `gmm/real_distribution_gmm.joblib`, `real_feature_scaler.joblib`, `real_only_thresholds.json`. API inventory xác nhận.
- **Kết quả Tiny đã có local:** [artifact_report.json](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/d2_fad/spectral_gmm_branch/output/results/spectral_gmm_tinygenimage_fixed_grid_test/artifact_report.json), thư mục `baselines/d2_fad/spectral_gmm_branch/output/results/` và `results.zip`. Provenance trỏ đúng URL checkpoint trên. **Chưa tìm thấy URL publish riêng cho bộ test Tiny này.** Các metrics train/GMM trong checkpoint dataset không tự được coi là test Tiny.

## 6. CLIP linear probe / NPR / AIDE và benchmark nhiều model

### BASE-01 — Checkpoint dùng cho Tiny JPEG-bias: CLIP probe, NPR-ResNet18, AIDE original full

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Notebook train CLIP probe | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_linear_probe/train_clip_linear_probe_kaggle_folder.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_linear_probe/train_clip_linear_probe_kaggle_folder.ipynb) |
| Notebook train NPR-ResNet18 | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/npr_resnet18/train_npr_resnet18_kaggle_folder.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/npr_resnet18/train_npr_resnet18_kaggle_folder.ipynb) |
| Notebook train AIDE full | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/aide_original_full/train_aide_original_full_tiny_commfor_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/aide_original_full/train_aide_original_full_tiny_commfor_kaggle.ipynb) |
| Script train AIDE full | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/aide_original_full/train_aide_original_full_tiny_commfor_kaggle.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/aide_original_full/train_aide_original_full_tiny_commfor_kaggle.py) |
| Notebook benchmark Tiny JPEG-bias | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/tiny_jpeg_bias_three_models/test_tiny_jpeg_bias_three_models_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/tiny_jpeg_bias_three_models/test_tiny_jpeg_bias_three_models_kaggle.ipynb) |
| Script benchmark Tiny JPEG-bias | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/tiny_jpeg_bias_three_models/test_tiny_jpeg_bias_three_models_kaggle.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/tiny_jpeg_bias_three_models/test_tiny_jpeg_bias_three_models_kaggle.py) |

- **Data benchmark:** [D1](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage).
- **Checkpoint đã API xác nhận:** [hoangha071005/checkpoints-clip-npr-aide](https://www.kaggle.com/datasets/hoangha071005/checkpoints-clip-npr-aide), **v1**, có đúng 4 file:
  - `checkpoints/clip_linear_head.pt` — OpenCLIP **ViT-B-32**, head-only; SHA-256 `788461421cb585bbf489aabc76f3537058cd7753eccf9432d3a792622f738057`.
  - `checkpoints/npr_resnet18_from_scratch.pt` — NPR + ResNet18; SHA-256 `328f9f431d378f9528c86966796e9f3ac604a5ce2bf3d307028062706c1537f0`.
  - `checkpoints/aide_original_full_trainable.pt` — AIDE **full**, không kèm frozen ConvNeXt-XXLarge semantic trunk; SHA-256 `62848895b44255d6a0567754c503a05807ed0a29b8046f3c6956a049844e47c3`.
  - `checkpoints/checkpoint_manifest.json` — metadata vai trò/kiến trúc/hash của ba file trên, đã tải đọc để đối chiếu.
- **AIDE full riêng:** [hoangha071005/aide-checkpoint](https://www.kaggle.com/datasets/hoangha071005/aide-checkpoint), **v1**, `check_point/model_trainable.pt`, `best_checkpoint.json`, `architecture.json`, `training_config.json`. Sidecar metadata ghi cùng SHA-256 AIDE full nêu trên; chưa tải lại file weight để tính hash mới.
- **Manifest và kết quả Tiny JPEG-bias:** hiện xác nhận ở local [thư mục kết quả](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/tiny_jpeg_bias_three_models/tiny_jpeg_bias_three_models_results), gồm `dataset/`, `metrics/`, `predictions/`, `provenance/`; **chưa tìm thấy URL publish kết quả riêng**. Không coi URL checkpoint là URL kết quả benchmark.

### BASE-02 — Tiny test ba model bản cũ: CLIP probe / NPR / AIDE forensic-only

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Script train AIDE forensic-only (Modal) | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/aide_original_forensic_resnet50/train_aide_original_forensic_resnet50_modal.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/aide_original_forensic_resnet50/train_aide_original_forensic_resnet50_modal.py) |
| Notebook test Tiny lưu với tên clip-npr-aide-tiny | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models/code/clip-npr-aide-tiny.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models/code/clip-npr-aide-tiny.ipynb) |
| Notebook test Tiny ba model | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models/code/test_three_models_tiny_genimage_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models/code/test_three_models_tiny_genimage_kaggle.ipynb) |
| Script test Tiny ba model | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models/code/test_three_models_tiny_genimage_kaggle.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models/code/test_three_models_tiny_genimage_kaggle.py) |

- **Data:** [D1](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage).
- **Checkpoint URL lịch sử do provenance ghi:** [hoangha071005/checkpoint-clip-npr-aide](https://www.kaggle.com/datasets/hoangha071005/checkpoint-clip-npr-aide), chú ý **`checkpoint` số ít**, khác URL **`checkpoints` số nhiều** ở BASE-01. API lần này trả **403**, chưa xác minh version/file hiện tại.
- **File được provenance run cũ ghi:** `checkpoints/clip_linear_head.pt`, `checkpoints/npr_resnet18_from_scratch.pt`, **`checkpoints/aide/model.pt`**. AIDE này là **forensic-only ResNet50**, SHA-256 `57024d5c0256869f8855a081abf27e6e882f5bb6de61659c5ef6a28afe8d5b39`, khác AIDE full ở BASE-01.
- **Manifest/kết quả local:** [tiny_genimage_three_models_results](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models/artifacts/tiny_genimage_three_models_results), có `config.json`, `provenance/checkpoints.json` và các bảng kết quả. **Chưa xác nhận URL publish kết quả riêng.**

### BASE-03 — CommFor balanced 2.000 ảnh / CommFor unseen

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| Benchmark CommFor balanced bốn model (Modal) | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_balanced_four_models/code/test_four_models_commfor_balanced_modal.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_balanced_four_models/code/test_four_models_commfor_balanced_modal.py) |
| CommFor unseen ba model (Modal) | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models/code/test_three_models_commfor_unseen_modal.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models/code/test_three_models_commfor_unseen_modal.py) |
| CommFor unseen ba model (Kaggle notebook) | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models/code/test_three_models_commfor_unseen_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models/code/test_three_models_commfor_unseen_kaggle.ipynb) |
| CommFor unseen ba model (Kaggle runtime) | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models/code/test_three_models_commfor_unseen_kaggle.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models/code/test_three_models_commfor_unseen_kaggle.py) |

- **Data:** [D5 — CommunityForensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval), split `CompEval`. Hai benchmark có cohort/manifest riêng; không dùng thay M1.
- **Manifest, checkpoint provenance và kết quả balanced:** [commfor_balanced_2000_v1_download](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_balanced_four_models/artifacts/commfor_balanced_2000_v1_download); lấy `provenance/checkpoints.json` của chính run để xác định bốn checkpoint.
- **Code/artifact unseen:** [commfor_unseen_three_models](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/commfor_unseen_three_models). Không dùng provenance Tiny trong thư mục con `tiny_genimage_three_models_results` để suy ra nguồn checkpoint của mọi lượt CommFor.
- **URL manifest/kết quả publish riêng:** chưa tìm thấy nguồn xác nhận. Checkpoint canonical local theo `baselines/README.md`; không gán đại URL BASE-01 cho mọi model của CommFor.

## 7. Các nguồn đã thấy nhưng chưa đủ bằng chứng ghép vào một run

| URL | Kết quả kiểm tra / cách dùng |
|---|---|
| [hoangha071005/checkpoint](https://www.kaggle.com/datasets/hoangha071005/checkpoint) | API 200, v1; đầu inventory có thư mục serialization `best/`. Tên chung chung, chưa xác định model/signature để ghép vào thí nghiệm. Không tự coi là GSD/CAPD checkpoint |
| [hoangha071005/data-fake-real-images](https://www.kaggle.com/datasets/hoangha071005/data-fake-real-images) | API 200, v1; có `test/FAKE/*.jpg`. Chưa xác định run nào trong dự án dùng nguồn này; không gán thay Tiny/CommFor |
| [hoangha071005/forensic-twins-checkpoints](https://www.kaggle.com/datasets/hoangha071005/forensic-twins-checkpoints) | URL manifest lịch sử theo `kaggle_inputs.md`; API lần này 403. Ưu tiên các manifest đã xác nhận trong TWINS-01/02/03 |
| [hoangha071005/forensic-twins-augmented-gmms](https://www.kaggle.com/datasets/hoangha071005/forensic-twins-augmented-gmms) | Slug output lịch sử/code, API 403; không đồng nhất tự động với TWINS-03 v2 |
| [hoangha071005/forensic-twins-v2-200k-checkpoints](https://www.kaggle.com/datasets/hoangha071005/forensic-twins-v2-200k-checkpoints) | Chỉ là gợi ý trong notebook builder; API 403. URL encoder200K thực tế đã xác nhận là `twins-gmm-200k-checkpoint` |
| [sandeepmnair/openai-clip](https://www.kaggle.com/datasets/sandeepmnair/openai-clip) | API 200, hiện tại v2; pretrained CLIP ViT-L/14 được notebook NhatNguyen tham chiếu, không phải CLIP ViT-B-32 probe đã train của BASE-01 |
| [truongnhatnguyen2805/aide-original-full](https://www.kaggle.com/datasets/truongnhatnguyen2805/aide-original-full) | Notebook NhatNguyen tham chiếu; API 403, không xác nhận version hiện tại hoặc đồng nhất với AIDE bundle của hoangha071005 |

Không tạo URL giả cho slug manifest-only `hoangha071005/forensic-twins-bfree-manifests`: tài liệu local nói đây chỉ là ví dụ, chưa có bằng chứng dataset riêng đã được tạo.

## 8. Những nhánh còn thiếu URL xác nhận

- **GSD Forensic Twins real-only / GSD-DINO encoder:** có source/plan local và input dự kiến từ Twins200K, nhưng chưa tìm thấy checkpoint kết quả đã chạy/publish đủ bằng chứng. Không biến đường dẫn output dự kiến thành URL kết quả.
- **ResNet50, NPR patch ResNet18, AIDE+NPR fusion, Qwen2.5-VL LoRA, CNNSpot, FFT, logistic regression:** code hoặc artifact có trong `baselines/`; chưa tìm thấy URL manifest/checkpoint/test riêng đủ xác nhận trong lần rà soát này. Với các model có artifact, xem `baselines/<model>/artifacts/` và [bảng artifact local](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/README.md).
- **Tiny combined → CommFor:** có code/protocol tại `benchmarks/tiny_combined_to_commfor_eval/`, data Tiny/CommFor như D1/D5; chưa có URL bộ kết quả riêng được xác nhận.
- **Quét ngưỡng BFree CAPD và GSD 10 case:** đã có báo cáo local; chưa publish URL riêng. Không ghi các ngưỡng chọn trên nhãn evaluation là kết quả test độc lập.

## 9. Hồ sơ đối chiếu để kiểm tra lại

- **Bảng so sánh METRICS CAPD/GSD/baseline:** [METRICS_COMPARISON.md](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/score_model_comparison/METRICS_COMPARISON.md), [script compare_metrics.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/score_model_comparison/compare_metrics.py), [JSON](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/score_model_comparison/metrics_comparison.json). Probability threshold0,5; các bảng matched Tiny native/9case/3case và BFree5 nhánh/clean1557; đủ precision/recall/F1 hai lớp, mean/median metrics.
- **So sánh mean/median score CAPD, GSD và baseline:** [COMPARISON.md](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/score_model_comparison/COMPARISON.md), [script compare_scores.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/score_model_comparison/compare_scores.py). Tiny matched native/9case/3case; BFree matched5 nhánh và partial7024ID; Twins/Spectral NLL tách riêng khỏi probability.
- **CAPD mean/median score dự đoán:** [SCORE_REPORT.md](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/SCORE_REPORT.md), [script score_statistics.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/score_statistics.py). Đây là thống kê prob_fake/logit theo nhãn real/fake, khác mean/median các metrics trong THRESHOLD_SUMMARY.md.
- **CAPD train BFree — quét ngưỡng và mean/median Tiny/BFree:** [THRESHOLD_SUMMARY.md](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/THRESHOLD_SUMMARY.md), [script threshold_summary.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/threshold_summary.py), [JSON đầy đủ](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/capd_train_bfree_eval_analysis/threshold_summary.json). Thống kê tách Tiny10, BFree6 và combined16; từng pooling riêng; ngưỡng tối ưu chọn hậu nghiệm trên Tiny.
- **Quét ngưỡng GSD train BFree → Tiny9:** [THRESHOLD_REPORT.md](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_tiny_analysis/THRESHOLD_REPORT.md), [script local threshold_sweep.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_tiny_analysis/threshold_sweep.py), [JSON chi tiết](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_tiny_analysis/threshold_sweep.json). Chọn ngưỡng bằng nhãn Tiny test, chỉ là phân tích hậu nghiệm.
- **So sánh GSD train BFree với GSD Tiny và baseline:** [BASELINE_COMPARISON.md](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_bfree_analysis/BASELINE_COMPARISON.md), [script local compare_baselines.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_train_bfree_test_bfree_analysis/compare_baselines.py). Có bảng matched 10.304 ID/5 nhánh và bảng matched clean 1.557 ID cho Twins/GMM; nhánh partial và CLIP/AIDE khác cohort được ghi riêng.
- **So sánh GSD/CAPD khi chỉnh ngưỡng BFree:** [báo cáo](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_capd_bfree_threshold_comparison/REPORT.md), [kết quả JSON](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_capd_bfree_threshold_comparison/comparison.json), [script chạy local compare_thresholds.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/gsd_capd_bfree_threshold_comparison/compare_thresholds.py). Dùng GSD-03 BFree và CAPD-02 BFree; đã đối chiếu COCO ID/nhánh/nhãn. Đây là phân tích trên predictions đã lưu, chưa có URL publish riêng.
- [URL và file nguồn local](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/experiment_url_audit/local_url_evidence.json).
- [Metadata/API checks đợt 1](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/experiment_url_audit/remote_url_checks.json), [đợt bổ sung](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/experiment_url_audit/extra_remote_url_checks.json): status, thời điểm UTC, version, inventory trang đầu.
- [Danh sách dataset công khai hai tài khoản](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/experiment_url_audit/public_owner_dataset_lists.json). Đây chỉ là danh sách public API trả về, không phải kiểm kê được dataset private.
- Inventory phân trang đầy đủ đã lưu trong `reports/experiment_url_audit/*__inventory.json` cho GSD raw 10 case, Twins augmented, Twins GMM200K, Twins codec evaluation và Spectral. CAPD/GSD augmented có thêm inventory/provenance trong các thư mục báo cáo tương ứng.
- Các config/source_paths/metadata nhỏ đọc từ server được lưu nguyên nội dung ở `reports/experiment_url_audit/small_remote_files/`. Các script audit chỉ đọc API; không thay đổi dataset Kaggle.

File này là bản đồ nguồn và bằng chứng theo run, không phải khẳng định tất cả model trong repository đều đã train xong hoặc có kết quả test công khai.

## 10. File chạy bổ sung cho chuẩn bị dữ liệu và các nhánh chưa có URL kết quả

**File chạy local (đường dẫn tuyệt đối):**

| Vai trò | File local |
|---|---|
| DINOv3 — notebook chuẩn bị pretrained weight/repo | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/dinov3_download_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/capd/dinov3_download_kaggle.ipynb) |
| Manifest All7 — script đóng gói manifest có sẵn | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/package_all7_manifests.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/clip_gsd_tiny_kaggle/package_all7_manifests.py) |
| GSD Forensic Twins — notebook | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_forensic_twins/gsd_twins_native_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_forensic_twins/gsd_twins_native_kaggle.ipynb) |
| GSD Forensic Twins — prepare | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_forensic_twins/prepare.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_forensic_twins/prepare.py) |
| GSD Forensic Twins — train | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_forensic_twins/train.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_forensic_twins/train.py) |
| GSD Forensic Twins — density | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_forensic_twins/density.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_forensic_twins/density.py) |
| GSD-DINO — notebook | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_dino_encoder/gsd_dino_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_dino_encoder/gsd_dino_kaggle.ipynb) |
| GSD-DINO — train module | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_dino_encoder/train.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_dino_encoder/train.py) |
| GSD-DINO — reference basis module | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_dino_encoder/reference.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_dino_encoder/reference.py) |
| GSD-DINO — export embeddings module | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_dino_encoder/export.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/gsd_dino_encoder/export.py) |
| ResNet50 — train Kaggle notebook | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/resnet50/train_resnet50_kaggle_folder.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/resnet50/train_resnet50_kaggle_folder.ipynb) |
| NPR patch ResNet18 — train Modal | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/npr_patch_resnet18/train_npr_patch_resnet18_modal.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/npr_patch_resnet18/train_npr_patch_resnet18_modal.py) |
| AIDE+NPR fusion 256/2048 — train Modal | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/aide_npr_fusion/train_aide_npr_fusion_256_2048_modal.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/aide_npr_fusion/train_aide_npr_fusion_256_2048_modal.py) |
| AIDE+NPR fusion 256/256 — train Modal | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/aide_npr_fusion/train_aide_npr_fusion_256_256_modal.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/aide_npr_fusion/train_aide_npr_fusion_256_256_modal.py) |
| Qwen2.5-VL LoRA — train Kaggle | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_kaggle.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_kaggle.ipynb) |
| Qwen2.5-VL LoRA — train Modal | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py) |
| Qwen2.5-VL LoRA — inference CommFor | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/qwen25vl_lora_word_label/infer_qwen25vl_lora_commfor_modal.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/qwen25vl_lora_word_label/infer_qwen25vl_lora_commfor_modal.py) |
| Logistic regression — train Colab | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/logistic_regression/train_logistic_regression_colab.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/baselines/logistic_regression/train_logistic_regression_colab.ipynb) |
| Tiny combined -> CommFor — notebook | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/tiny_combined_to_commfor_eval/code/train_tiny_combined_test_commfor_eval.ipynb](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/benchmarks/tiny_combined_to_commfor_eval/code/train_tiny_combined_test_commfor_eval.ipynb) |
| Đối chiếu URL — script đọc metadata Kaggle | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/experiment_url_audit/verify_urls.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/experiment_url_audit/verify_urls.py) |
| Đối chiếu artifact — inventory và file nhỏ | [D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/experiment_url_audit/inspect_artifacts.py](D:/LLM/LVLM/MLLM_detect_fake_image/HoangHa_Code/reports/experiment_url_audit/inspect_artifacts.py) |

Chưa tìm thấy file `.py`/`.ipynb` trong hai thư mục `baselines/cnnspot/` và `baselines/fft/` khi rà soát; không tự tạo đường dẫn entry point cho hai nhánh này. Bảng code local không thay đổi trạng thái xác nhận checkpoint/kết quả đã ghi ở mục 8.
