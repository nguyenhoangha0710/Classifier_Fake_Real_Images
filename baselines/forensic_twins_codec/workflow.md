# Workflow code — Twins–GMM codec v1

Đặc tả chính: [plan.md](plan.md). Hyperparameter trong plan là lựa chọn thử nghiệm nội bộ.

Kaggle preflight: [inputs và manifest reuse](kaggle_inputs.md). SSL dùng train_real_200k.csv từ hoangha071005/twins-gmm-200k-checkpoint; test dùng bfree_valid_clean_eval.csv từ full-run V2 của hoangha071005/evaluation-twins-bfree. Resolve/rebase trước GPU; không chia mới hoặc quét lại toàn dataset. Pixel train cần COCO+ImageNet, pixel test cần baopv051/b-free-training-data.

Data contract: SSL200K → encoder mới; sau đó dùng saved split160K để GMM-fit, saved split40K để calibration; B-Free clean1.557 source IDs/10.899 ảnh chỉ test. Calibration held-out khỏi GMM-fit, không held-out khỏi SSL. Không dùng GMM/embedding cũ cho encoder mới.

## Training data flow

1. Dataset trả RGB gốc và metadata; không resize trong loader.
2. Sample shared geometry, clone hai bản; random codec chỉ áp dụng view 2.
3. RGB uint8 ↔ float [0,1] có kiểm soát; grayscale cho từng view.
4. FRE frozen: [1,H,W] → [8,H,W], no_grad/eval.
5. Một sampler trả rectangle A/B không giao nhau trên cùng grid H×W; crop A từ residual 1, B từ residual 2.
6. Collate thành r1=[B,8,h1,w1], r2=[B,8,h2,w2]; size từng nhánh thống nhất across ranks.
7. Shared encoder: z1,z2=[B,192]; shared projector: e1,e2=[B,8192].
8. Differentiable global-batch normalization/correlation → Barlow loss → optimizer step; không detach encoder output.

## Detection data flow

Native RGB → center crop256 → grayscale [0,1] → FRE → encoder CLS192 → GMM NLL.
GMM-fit/calibration/test dùng cùng embedding pipeline và checkpoint ID.
Fit Gaussian trên fit-real; calibrate p95 trên calibration-real; inference không cập nhật state.

## Modules và contracts

- `CodecPolicy(image, rng)`: ảnh cùng W×H và record(policy,Q,scale,interpolation,subsampling).
- `TwinViewCollator`: hai tensor residual, IDs, crop rectangles và policy record; RNG khác nhau theo seed/rank/worker.
- `load_finetune_weights`: encoder/projector strict khi tương thích; không load optimizer của native.
- `fit_detector`: lưu Gaussian, calibration threshold, encoder hash, preprocessing config và manifest hashes.
- Unit checks + CPU smoke; GPU smoke 1 optimizer step để kiểm tra gradient/AMP/DDP, rồi mới chạy dài.

Không có GSD/CLIP/PCA trong phiên bản này. GMM augmented hay inference nhiều view là ablation sau.
