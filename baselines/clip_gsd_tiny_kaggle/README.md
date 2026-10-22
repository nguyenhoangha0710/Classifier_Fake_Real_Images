# GSD with early stopping — Tiny-GenImage / Kaggle

Upload **train_clip_vs_gsd_tiny_kaggle.ipynb** to Kaggle, attach `yangsangtai/tiny-genimage`, enable GPU + Internet, Run All. Notebook is self-contained: runtime code is embedded as a readable cell. No repository attachment needed.

Paper source: <https://arxiv.org/html/2603.09242v2>. This is a transparent paper-guided implementation on **Tiny**, not official author code or an exact full-GenImage reproduction.

## Model and data

- Default: train **CLIP + GSD only** (`models=['clip_gsd']`). The previous CLIP fine-tune result is an external baseline; ensure its protocol matches before direct comparisons. Optional CLIP control implementation remains available but is not executed by default.
- Train: balanced SD15 train-folder real/fake only (`source_generator='sd15'`). Source-only 10% stratified validation, minimum validation CE checkpoint. Test: all seven validation/test generator folders, with six unseen generators, never used for checkpoint/threshold selection.
- `yangsangtai/tiny-genimage` explicitly excludes SD14. This is an SD15-source adaptation, not the full GenImage SD14 protocol. `expected_generators` validates the seven actual names. For a separate full eight-generator dataset, explicitly set `source_generator='sd14'` and include `sd14` in `expected_generators`. Use a fresh output folder when changing protocol.
- Decode RGB SHA256 excludes exact pixel overlaps between source train/validation and test. Within-generator duplicate pixels removed. Cross-generator repeated real images recorded; prefer per-generator/macro metrics. This does not detect near duplicates or establish CLIP pretraining independence.
- GSD **before blocks 21–24**, matching Figure 3. Frozen pre-block non-CLS tokens → full logical batch uncentered SVD → top-64 right singular vectors → adaptive projection/subtraction on all detector tokens. Occupancy is differentiable. Eight trainable modulation scalars total.
- No LoRA or partial unfreezing. Optional CLIP control uses identical preprocessing/pooling, but early stopping can produce different training lengths.

## Early stopping

Enabled by default: maximum **50 epochs**, `patience=3`, `min_delta=0.001`, monitor source-validation cross-entropy. A reduction greater than min_delta from the last significant improvement resets the counter. After three consecutive epochs without such improvement, stop. This is a project choice, not a training rule specified in the paper.

`best.pt` always tracks the absolute minimum validation CE, even when the reduction is below min_delta. After stopping, restore that checkpoint for all test cases. `last.pt` preserves the significant-improvement reference, bad-epoch counter and stopped flag. Resuming an already stopped run goes directly to best-checkpoint evaluation. `history.json` and `training_status.json` record progress. Keep the same config to resume; use a fresh output_root for a new training protocol.

## Published settings vs implementation assumptions

Paper settings preserved: ViT-L/14, full forensic fine-tune, frozen anchor, last four GSD layers, k64, a2/b1, lambda_max1, AdamW backbone LR1e-6, batch128 train/512 inference, classification CE, average pooling, JPEG/blur augmentation.

Paper does **not** disclose the complete runnable recipe. Defaults chosen here: maximum 50 epochs with early stopping, constant LR/no warmup, head and a/b LR1e-6, WD.01/betas(.9,.999)/eps1e-8; OpenAI CLIP 224; post-LN patch mean pooling; flip.5, blur.5 sigma[0,2], JPEG.5 Q[75,100], subsampling0. These are explicitly provisional. Figure 3 identifies average pooling but not whether CLS participates or the final normalization placement. Both models use the same explicit choice.

AdamW decay applies uniformly to all groups, including modulation/LN/bias. This is an assumption, configurable via code if conducting another ablation. No undocumented hyperparameter is presented as an official paper value.

## Kaggle memory and runtime

Uses **one GPU**, even if 2×T4 is available. FP16 + checkpointing; micro_batch2, anchor_micro_batch8. Full batch128 is augmented once and retained on CPU; anchor prepass collects tokens for **all 128 samples**, computes four bases, then accumulates CE gradients over GPU micro-batches. No BatchNorm/dropout dependence in the default CLIP; CPU tests compare full and micro-batch gradients. Frozen anchor FP16 is a numerical adaptation to hardware, SVD/projection/occupancy FP32.

Logical batch512 for inference also shares one basis per layer across the whole batch. Final partial inference batch uses all remaining images. Train drops the last partial batch; epoch shuffling changes dropped IDs deterministically. Do not lower logical batch or k silently: these alter GSD. Lower only micro-batch sizes on OOM. Exact SVD four times per logical batch can be costly on T4/P100; no wall-time guarantee.

Start with `smoke=True` and a **different output_root**. Smoke reduces batch to8/k4 and sample count, one epoch, so it is not a paper result. It still loads full pretrained ViT-L/14 and requires GPU. Full mode uses all eligible images. Last checkpoints include optimizer for epoch-boundary resume. Same config/manifests required; extending epochs/config changes requires a new run. Kaggle working storage is ephemeral: save outputs before a reset. Checkpoints can occupy ~10 GB for two models, depending on optimizer serialization.

## Evaluation conditions

Same cohort and label-blind batch membership for both models and all cases:

1. native RGB → standard CLIP preprocessing;
2. both labels JPEG Q96/subsampling0, then CLIP preprocessing;
3. both labels square resize256 bicubic, then CLIP preprocessing;
4. square resize256 then JPEG96, then CLIP preprocessing.

Optional JPEG qualities90/80 via config. Resize is a **processing stress-test**, not a pure isolation of size: it changes aspect ratio, resampling and potentially field of view. All-JPEG does not undo prior real JPEG compression and can create a double-compression difference. No claim of fully unbiased data. PNG roundtrip is not a codec-history fix. Native format/dimensions/quantization-table hashes are audited.

Metrics: ROC-AUC, AP, accuracy, balanced accuracy, fake/real recall, confusion counts, fixed threshold.5. Per-generator, `macro_all.csv` (seven generators) and `macro_unseen.csv` (six generators excluding the configured source); pooled rows explicitly include repeated real samples across generator folders. Per-image native-to-processed delta probability and prediction flips, per-generator/per-class averages; model_comparison includes GSD−FT deltas. Test batches fixed and recorded; their composition affects Batch-Inference and is not used to fit thresholds. Results are fractions 0–1.

One seed is a pilot; use multiple seeds and additional controlled/unbiased datasets before causal/generalization claims. No statistical significance or codec independence is assumed from a raw metric drop alone.

## Files and validation

- `runtime.py`: model, audit/split, training, evaluation and reports.
- `build_notebook.py`: deterministic readable notebook generator.
- `test_runtime.py`: CPU small-CLIP tests for projection, gradients, pre-block basis, micro-batch equivalence, checkpointing, cohort leakage, deterministic augmentation, metrics and paired reports.

Build: `python baselines/clip_gsd_tiny_kaggle/build_notebook.py`

Test (from this folder): `python -m unittest test_runtime -v`

Local tests do not verify Kaggle CUDA memory, full-size SVD throughput, dataset contents or convergence; a Kaggle smoke run is still required.
