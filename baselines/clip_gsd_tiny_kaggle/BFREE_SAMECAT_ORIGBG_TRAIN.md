# GSD train trên B-Free `SD2.1_inpainted_samecat_origBG` (kế hoạch P2)

Ngày tạo: 30/10/2026. Trạng thái: notebook đã dựng, **chưa chạy trên Kaggle**.

## File
| File | Vai trò |
|---|---|
| `train_gsd_bfree_samecat_origbg_kaggle.ipynb` | Notebook train (tự chứa). Input: chỉ `baopv051/b-free-training-data`. GPU + Internet. |
| `test_gsd_bfree_samecat_origbg_tiny9_kaggle.ipynb` | Test Tiny-GenImage 9 case, bản sao evaluator GSD-aug-Tiny + cell compat. |
| `test_gsd_bfree_samecat_origbg_bfree_valid_kaggle.ipynb` | Test B-Free valid 10.304 ID × 6 nhánh, bản sao evaluator cũ + cell compat. |
| `bfree_data.py` | Lớp dữ liệu không cần torch: audit/mask, split, batch cặp, aug chung cặp, resume. |
| `bfree_train.py` | Vòng train/validate/checkpoint (dùng lại model/basis của runtime GSD-aug-Tiny). |
| `build_bfree_samecat_origbg_notebooks.py` | Dựng lại 3 notebook từ các file .py. |
| `test_bfree_data.py`, `test_bfree_compat.py` | Test offline (CPU, không torch): 10 + 5 test, đều pass. |

## Giao thức (khác GSD-aug-Tiny chỉ ở dữ liệu và cách ghép cặp)
- Dữ liệu: `train_list − valid_list` (41.213 ID; MD5 list chính thức), real `COCO_real_512` vs fake `SD2.1_inpainted_samecat_origBG`.
- Audit: mask = max kênh |fake − real| > 8; loại cặp lỗi/thiếu, mask < 0,1% hoặc > 95%; dừng nếu > 1% cặp lỗi.
- Val 2.000 ID phân tầng 3 nhóm diện tích; train xáo 1 lần; VE = 12.600 ID = 25.200 ảnh; tối đa 3 VE; không lặp ID.
- Batch 8 = 4 ID × (real, fake); blur/JPEG như Tiny nhưng tham số chung cho cặp, theo (seed, ID, VE).
- Model/optimizer/geometry giống hệt GSD-aug-Tiny (kiểm bằng test). Early stopping patience 2 (Tiny: 3).
- Checkpoint: `checkpoint_veN.pt`, `best.pt` theo định dạng evaluator; `config.eval_compat_config` cho evaluator cũ.
- Phiên Kaggle: `time_budget_hours=11`, `last.pt` mỗi 250 bước; resume bằng cách Add Input output lần trước.

## Chưa kiểm chứng
- Phần torch (train/validate/checkpoint) chưa chạy được offline vì môi trường không cài được PyTorch;
  model/basis/optimizer dùng nguyên hàm đã chạy trên Kaggle, nhưng vòng lặp mới chỉ được kiểm tĩnh.
- Thời gian (~4 h/VE trên T4) là ước lượng từ tốc độ đo trước, chưa đo trực tiếp.
- Ngưỡng mask (8 / 0,1% / 95%) đặt trước khi xem histogram; xem `audit_summary.json` sau lần chạy đầu.
