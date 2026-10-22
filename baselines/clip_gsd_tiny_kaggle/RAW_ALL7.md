# Raw All7 at native resolution

Upload `train_gsd_raw_all7_native_kaggle.ipynb` to Kaggle and attach `yangsangtai/tiny-genimage`.
It embeds both runtime files and needs no repository attachment. Enable GPU + Internet.

This new experiment trains GSD on all seven generators. It is separate from the old SD15/224 run.
Default output: `/kaggle/working/gsd_raw_all7_native_v1`.

## Exact input contract

Decode the original image as RGB, retaining its original width and height. No train JPEG encoding,
blur, flip, interpolation, resize or crop is performed. Right and bottom edges are replicate-padded
by 0–13 pixels to complete 14-pixel patches. CLIP mean/std normalization remains necessary.
Existing JPEG artifacts in original files remain: raw-only does not remove dataset codec bias.

Both branches use positional-embedding interpolation, not image interpolation. A 1024-square
image becomes 1036-square by edge padding, giving a 74x74 patch grid plus CLS (5,477 tokens).
There is no hard-coded 224-pixel image transform, no pixel-size cap and no automatic image dropping.

The default logical train/test batch is 8; detector/anchor microbatch is 1. Logical batches contain
mixed resolutions and retain label-blind membership. Only consecutive equally shaped images are
stacked into compute chunks; no batch-wide padding, label-based grouping or dropped final batches.
At validation/test, ID order and batch membership are deterministic and identical across variants.

All patch tokens of the anchor contribute to the basis. Streaming FP32 Gram matrices M^T M with
`eigh` avoids materializing the full concatenated token matrix. The top eigenvectors span the
same subspace as top right singular vectors of uncentered M in exact arithmetic; Gram computation
can have worse conditioning numerically. No token sampling, centering or per-image SVD is used.
Large images contribute more tokens, an explicit aspect of this protocol rather than an equal-image
weighted basis. Projection/occupancy remains FP32; the anchor is FP16 by default on GPU.

SDPA reduces attention storage. Native-resolution full fine-tuning is still expensive and not
guaranteed to fit a T4 for every image. An OOM stops with an explicit message; no silent downscale,
skip or batch reduction. Use more GPU memory or deliberately choose a different recorded batch
configuration. Microbatch=1 cannot be lowered further. This is not the paper's batch128/224 recipe.

## Data and early stopping

- Scan all seven train folders plus their val/test folders; print progress every 500 usable images.
- Assign each unique decoded-pixel hash to train or validation once, stratified by representative
  generator and label. Repeated real pixels across generators cannot enter both partitions.
- Balance generator/class counts to a common minimum independently in train and validation.
  Counts and exact selected IDs are exported; balancing may exclude surplus images.
- Official val folders are test only. Exclude test pixel hashes present anywhere in source train.
- Maximum 50 epochs; patience3, min_delta0.001, validation CE on raw All7 validation.
- Restore absolute best validation-loss epoch after early stopping. No test-based selection.
- Save `best.pt`, atomic `last.pt`, optimizer/scaler/config/history and stopper state each epoch.
- Resume only with identical config and manifests in the new output directory. Never reuse old
  SD15/224 checkpoint folders for this run.

`token_census.csv` records native dimensions and token counts. Per-split manifests/counts,
history, architecture/config/environment, test predictions and macro_all metrics are also saved.
All test generators are seen: the report does not label six of them unseen anymore.

## Optional evaluation after training

Default is raw native evaluation only. To test additional variants in the same new run:

```python
CONFIG['eval_cases'] = ['native', 'fake_jpeg96', 'jpeg96', 'jpeg90', 'jpeg80', 'size', 'size_jpeg96']
```

This does not alter training pixels. `jpeg*` reencodes both labels, `fake_jpeg*` only fake;
`size` is an explicitly requested 256-square stress test. All variants are normalized and edge-padded
after their test transformation. Fake-only processing is a diagnostic requiring labels, not a deployment
preprocessor. In GSD, changes to fake images can also change real scores via the shared basis.

For later CommFor/B-Free use, retain best.pt and the full config; those evaluation loaders must
implement this same native/padding/interpolation/basis/batch protocol. Existing 224-only evaluation
scripts are not automatically compatible. External evaluation runners are not part of this change.

## Verification

```text
python baselines/clip_gsd_tiny_kaggle/build_raw_all7_notebook.py
python -m unittest discover -s baselines/clip_gsd_tiny_kaggle -p "test_*.py" -v
```

CPU tests use tiny random CLIP models, not pretrained weights. They cover pixel preservation,
edge padding, mixed-resolution interpolation and gradients, Gram/SVD subspace agreement,
All7 hash-group leakage prevention and class balance, complete loader coverage, paired inference,
early stopping and checkpoint resume. Full-size Kaggle CUDA memory/runtime remains to be measured.
