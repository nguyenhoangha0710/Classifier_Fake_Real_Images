# CAPD: TN1 trên Tiny-GenImage (giao thức giống hệt GSD-aug)

CAPD = backbone đóng băng + LoRA r=8 → PatchHead (điểm cho từng patch) → gộp MIL bằng LSE.
Loss = L_img + 0.5·L_neg (+ L_gap, giảm dần về 0 trong 315 bước đầu).
Thiết kế chi tiết: Project docs `claude/model_de_xuat_CAPD.md`, `claude/capd_TN1_plan.md`.

## File

| File | Vai trò |
|---|---|
| `capd_runtime.py` | Code CAPD; chạy SAU các runtime GSD nên dữ liệu, split, augmentation, validation và các case test dùng đúng hàm của GSD-aug |
| `build_capd_notebooks.py` | Dựng 3 notebook tự chứa (nhúng `../clip_gsd_tiny_kaggle/runtime*.py` + `capd_runtime.py`) |
| `dinov3_download_kaggle.ipynb` | Tải weight DINOv3 ViT-L/16 và repo → tạo Kaggle Dataset **private** |
| `train_capd_tn1_kaggle.ipynb` | Train R1 (CLIP, GPU 0) và R2 (DINOv3, GPU 1) song song, sau đó test native |
| `eval_capd_tn1_kaggle.ipynb` | Test Tiny 10 case + B-Free valid 6 nhánh |

Sửa code thì chạy lại: `python baselines/capd/build_capd_notebooks.py`

## Thứ tự chạy trên Kaggle

1. **`dinov3_download_kaggle.ipynb`**: Internet ON, dán link vitl16 từ email → Run All → Save Version → Output → New Dataset (**Private**).
2. **`train_capd_tn1_kaggle.ipynb`**: GPU T4×2, Internet ON. Add Input: tiny-genimage, dataset manifest GSD, dataset DINOv3.
   - `SMOKE=True` → Run All → kiểm tra cả hai log có `[exit] complete`.
   - `SMOKE=False` → Save & Run All (Commit). Hết 12 giờ (`paused_time_budget`) thì Add Input là output phiên trước, điền `RESUME_FROM`, chạy lại.
3. **`eval_capd_tn1_kaggle.ipynb`**: Add thêm B-Free và output của notebook train → điền `TRAIN_OUTPUT`.

## Giữ y hệt GSD-aug

- Manifest All7 seed 42 (train 25.200 / val 2.800 / test 7.000), giữ đúng thứ tự.
- Blur / JPEG với p=0,5, tham số lấy theo `processing_parameters` của GSD (cùng giá trị cho từng ảnh ở từng epoch).
- Thứ tự batch giống `DataLoader(shuffle, seed+epoch)` của GSD; logical batch 8, microbatch 1.
- Center crop ≤560, pad replicate cạnh phải/dưới tới bội của patch (14 với CLIP, 16 với DINOv3).
- val_loss = 0,5·CE(native) + 0,5·CE(processed); early stopping patience 3, min_delta 0,001, tối đa 50 epoch.
- Các case test lấy nguyên `case_spec` / `transform_variant` của evaluator GSD.

## Output (`/kaggle/working/capd_tn1/<run>/seed42/`)

- `epoch_XXX.pt`: mỗi epoch, chỉ LoRA + head (khoảng 14 MB). `best.pt`.
- `last_state.pt`: để resume, lưu mỗi 250 bước.
- `train_steps.csv`: các loss thành phần, λ_gap, lr, grad norm, AMP, NaN, giây/ảnh, VRAM.
- `validation_metrics.csv`, `validation_predictions_epochNNN.csv`, `history.json`.
- `tiny_eval/predictions_<case>.csv`: z, prob, hình học (crop, pad, offset mod 8/16), QF ước lượng, s_max, entropy LSE...; `smaps_<case>.npz` (200 ảnh / case).
- `bfree_eval/predictions_bfree.csv` (`lse_full` và `lse_window`), `metrics_bfree.csv`, `smaps_*.npz` (toàn bộ).

## Trạng thái

Code đã kiểm tra cú pháp và tên biến, và đối chiếu với source DINOv3 (`dinov3/hub/backbones.py`, `layers/block.py`). **Chưa chạy trên GPU**: smoke run là bước kiểm chứng đầu tiên.

## TN2: train trên B-Free (31/10/2026)

| File | Vai trò |
|---|---|
| `capd_bfree.py` | Train CAPD trên B-Free `SD2.1_inpainted_samecat_origBG`. Dùng nguyên `bfree_data.py` + `bfree_train.py` của GSD-P2 cho audit, split, ghép cặp, augmentation và validation |
| `train_capd_bfree_kaggle.ipynb` | T4×2: GPU 0 `b2_dinov3`, GPU 1 `b1_clip`. Input: B-Free, DINOv3, output GSD-P2 (dùng lại audit) |

- Split được kiểm tra trùng GSD-P2: dựng lại split đầy đủ từ audit của GSD và so `validation_order_hash` / `train_order_hash`.
- Checkpoint `ve_XXX.pt` / `best.pt` chỉ chứa LoRA + head (khoảng 26 MB).
- Eval dùng `eval_capd_tn1_kaggle.ipynb` với `EVAL_RUN='b2_dinov3'` hoặc `'b1_clip'`.
- Mọi file prediction có thêm `z_inner` (bỏ hàng/cột patch cuối) và `z_nopad` (bỏ patch có pad), để kiểm tra shortcut cạnh.
