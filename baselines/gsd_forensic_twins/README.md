# Native Twins + GSD last4

Current specification: [plan.md](plan.md), tensor contracts: [workflow.md](workflow.md).
The Kaggle notebook downloads CLIP from Hugging Face once, then training workers use local-only pretrained loading.
No training or Kaggle publishing is triggered automatically. The notebook embeds the sources.

## Inputs and preparation

Enable **Kaggle Settings → Internet**. The notebook downloads `openai/clip-vit-large-patch14` into `/kaggle/working/clip_cache`; a CLIP Kaggle Input is not required.
It saves the resolved Hugging Face commit in `clip_download.json` and downloads only one weight format. Set `CLIP_REVISION` to that commit for repeated experiments; cached files are reused.
Mount the saved real200K manifest and COCO/ImageNet pixels. The notebook's `CLIP` path is populated automatically by the download cell.
The parent manifest is set explicitly to `/kaggle/input/datasets/hoangha071005/twins-gmm-200k-checkpoint/forensic_twins_v2_200k/manifests/train_real_200k.csv`, with one direct fallback to the legacy `/kaggle/input/twins-gmm-200k-checkpoint/` mount. No file discovery or recursive scan is used to locate this manifest.
For CLI runs, execute `python download_clip.py --cache-dir CLIP_CACHE` first and pass its printed snapshot path as `--clip CLIP_DIR`. An existing local CLIP snapshot is still supported.
Manifests must have `image_path` or `relative_path`; optional `sample_id`, `label`, `source`.
Relative paths resolve against `--pixel-root`. For `--pixel-root /kaggle/input`, known COCO/ImageNet source paths are rebased using direct modern/legacy mount checks, cached per source/split. No recursive search is used; unknown layouts fail with the explicit checked paths.
Prepare fit160K and calibration40K independently with `--purpose density`. Preserve saved split membership.
Default preparation uses saved width/height metadata and sampled path checks, without opening all200K images. Missing dimensions fall back to header reads. `--audit header` explicitly reads all headers; `--audit full` also verifies files and may be slow. This metadata mode does not claim to have audited every pixel file; actual decode failures remain fatal during training. Size rejects are recorded in provenance JSON.
Preparation prints progress/rate/ETA every1000 rows plus a heartbeat every30s. `--reuse` accepts an existing output only when parent/output hashes, pixel root and audit/crop policy match. The output CSV is written atomically; a partially completed full audit is not resumable.

```powershell
python prepare.py --parent train_real_200k.csv --pixel-root PIXELS --output ssl_train_eligible.csv --purpose ssl
python -m unittest discover -s . -p "test_*.py" -v
python build_notebook.py
```

Run commands from this directory or provide absolute script paths. GPU training requires recent PyTorch (2.4+) and transformers4.46.3; Pillow9.1+, NumPy and scikit-learn are used. Tests construct tiny random CLIP models without downloading weights.

## GPU smoke and training

Set batch explicitly after memory/time preflight; no claim full ViT-L/14 fits T4. True global batch must be at least2.

```text
torchrun --standalone --nproc_per_node=2 train.py --clip CLIP_DIR --manifest ssl_train_eligible.csv --pixel-root PIXELS --output SMOKE_DIR --batch-per-gpu 1 --smoke-steps 200
torchrun --standalone --nproc_per_node=2 train.py --clip CLIP_DIR --manifest ssl_train_eligible.csv --pixel-root PIXELS --output MAIN_DIR --batch-per-gpu LOCKED_BATCH
```

Smoke checkpoints are marked smoke-only, not accepted for main density extraction. Main training starts fresh from CLIP in a separate output directory.
`--resume MAIN_DIR/last.ckpt` restores model/optimizer/scaler/schedule and per-rank RNG at the next epoch boundary.
Mid-epoch resume is not implemented; rerun the incomplete epoch from the last completed epoch checkpoint.
Each epoch is saved; main result uses epoch20. Output directories cannot be reused without explicit resume.
`run_identity.json` locks configuration, initial CLIP hash, eligible manifest hash, batch/world size and epoch length.
`train.jsonl` contains loss components, per-layer modulation/occupancy, gradient norm and GPU memory/runtime.
The console prints explicit phases (model load, CPU hash, GPU transfer, CPU batch wait, first-step layer/SVD, Barlow, backward, optimizer, checkpoint), plus concise loss every10 steps. All stages heartbeat every30s. If the current stage has not changed for300s, Python stacks are dumped for diagnosis; this does not terminate a CUDA kernel or establish a deadlock by itself.
DataLoader uses `spawn` to avoid forking after CUDA initialization, and `--loader-timeout 120` bounds worker batch waits. `--dist-timeout 600` bounds distributed collectives. `--workers 0` is a debugging option but disables worker timeout protection. `--log-every` controls console frequency. Initial Es hashing happens on CPU before GPU transfer.
Precision defaults to `--precision auto`: BF16 if supported, otherwise FP16. GSD, projector and Barlow remain FP32. FP16 GradScaler starts at128; nonfinite gradients skip the optimizer/scheduler update, reduce the scale and log sample IDs/affected parameters to `overflow_rank*.jsonl`. Eight consecutive overflows abort with diagnostics. BF16/FP32 nonfinite gradients remain fatal. The gradient norm is inspected before clipping without modifying bad gradients. This handles AMP overflow; it does not prove all nonfinite-gradient causes are fixed. Use `--precision fp32` for comparison if overflow persists.
The regenerated notebook uses `gsd_native_smoke_amp_v2` for the corrected smoke run, preserving the failed run's artifacts. Main run precision is included in checkpoint identity.
Tests, prepare, smoke and main are separate notebook cells. `RUN_MAIN=False` is printed explicitly; set it to True only after reviewing smoke results. A running old Kaggle session does not pick up these changes: interrupt it and use the regenerated notebook, preserving existing checkpoints.

For B0, copy config.json and set `bypass=true`, use `--config B0.json`, and retain identical data/seed/batch/steps/readout/projector.
Main C0 fine-tunes all Ef parameters and eight modulation scalars; Es remains frozen. Projector follows native Twins Linear/GELU/Linear.

## Detector

```text
python density.py extract --checkpoint MAIN_DIR/epoch20.ckpt --clip CLIP_DIR --manifest fit_eligible.csv --pixel-root PIXELS --output fit_features.json
python density.py extract --checkpoint MAIN_DIR/epoch20.ckpt --clip CLIP_DIR --manifest calibration_eligible.csv --pixel-root PIXELS --output calibration_features.json
python density.py fit --fit fit_features.json --calibration calibration_features.json --output gaussian.npz
python density.py extract --checkpoint MAIN_DIR/epoch20.ckpt --clip CLIP_DIR --manifest bfree_valid_clean_eval.csv --pixel-root PIXELS --output bfree_features.json
python density.py score --features bfree_features.json --artifact gaussian.npz --output bfree_predictions.csv
```

Extraction is FP32, per-sample basis at each of the four layers; final LN then patch mean gives1024-D.
The `.json` metadata and same-stem `.npy` features must stay together. PCA/Gaussian fit only uses fit-real; p95 threshold only calibration-real.
Gaussian metadata verifies checkpoint/inference identity before scoring. This initial evaluator exports predictions; JPEG/resize protocol sweeps and grouped benchmark reports remain follow-up work.

## Verification limits

CPU tests check four-layer token suppression/propagation, frozen/gradient contracts, activation-checkpoint recomputation, per-sample batch invariance, crop geometry, Gaussian math and two-rank Gloo basis/Barlow gradient agreement.
Full pretrained GPU smoke and NCCL multi-GPU checks require the mounted Kaggle inputs and GPU hardware.
The existing draw.io/PNG still show the earlier one-layer design and are not implementation references.
