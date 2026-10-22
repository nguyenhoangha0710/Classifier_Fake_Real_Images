# All7 native-scale center crop, max 560 pixels

Use `train_gsd_raw_all7_crop560_kaggle.ipynb`. Attach `yangsangtai/tiny-genimage` and enable GPU + Internet.
The standalone notebook embeds the complete runtime and requires no repository input.

## Input policy

For each decoded RGB image, center crop each axis to `min(original_axis, 560)`. Do not resize.
For odd discarded margins, the left/top discarded margin is the integer floor of half the excess.
Keep smaller dimensions unchanged. Pad the right/bottom edges by 0–13 replicated pixels to a
multiple of patch size 14. Finally apply CLIP normalization. No train JPEG, blur or flip.

| Original width x height | Model width x height | Tokens including CLS |
|---|---|---:|
|1024 x 1024|560 x 560|1601|
|512 x 512|518 x 518|1370|
|224 x 224|224 x 224|257|
|1024 x 448|560 x 448|1281|

This preserves source pixel values inside the selected region but discards the exterior of large
images. It is not full-image raw input. Images retain variable shapes and token counts. The cap
must be a positive multiple of 14 so padding does not exceed it; 556 is rejected, 560 is valid.
At this cap, a single image never has more than 1601 tokens. Padding still contributes patch tokens.

Both detector and frozen anchor receive the same preprocessed image and interpolate positional
embeddings for its grid. Mixed logical batches, Gram-eigh basis and SDPA remain unchanged.
Logical batch=8, microbatch=1. The basis pools all patch tokens, so larger inputs contribute more
tokens. Native-resolution checkpoints are not interchangeable with this training protocol.

## Data, checkpoint and evaluation

- Train all seven generators; hash-group split before generator/class balancing.
- Source train/validation separation and official test folders stay as in the All7 native version.
- Train, validation and native test use the same crop/padding policy.
- Early stopping: patience3, min_delta0.001, max50 epochs, min validation CE checkpoint.
- New output: `/kaggle/working/gsd_raw_all7_crop560_v1`.
- Start from pretrained CLIP. Do not resume the failed native training checkpoint as this experiment.
- Default post-training evaluation is native. Optional JPEG evaluation reencodes the original image
  BEFORE center crop. Explicit `size` cases remain resize stress tests; they are never train transforms.
- `token_census.csv` includes original dimensions, crop coordinates/dimensions, input dimensions and
  token count. Config, architecture, signature and completion identify this crop protocol.
- The scan is still run on each invocation in this version; `resume=True` restores epoch-end training
  checkpoints, not scan cache. No batch-level checkpoint resume is added in this change.

First run the SAME crop policy with `smoke=True` in a separate output directory to check T4 memory.
The token cap reduces the previously observed native-resolution memory risk but full Kaggle GPU
training has not been verified locally. No automatic resize, image skip or silent batch reduction.

## Build and verify

```text
python baselines/clip_gsd_tiny_kaggle/build_raw_all7_notebook.py --center-crop
python -m unittest discover -s baselines/clip_gsd_tiny_kaggle -p "test_*.py" -v
```

Crop tests verify exact center pixel values, small-image preservation, odd/non-square sizes,
token bounds, identical native train/test preprocessing and JPEG-before-crop order. Native-input,
GSD mathematics, variable token gradients, early stopping and checkpoint tests also run.
