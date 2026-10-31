# All7 crop560 + Gaussian blur/JPEG

Import `train_gsd_all7_crop560_blur_jpeg_kaggle.ipynb` into Kaggle, attach
`yangsangtai/tiny-genimage` AND a previous All7 output containing saved manifests,
enable GPU and Internet. This is a new experiment;
all existing raw-only runtimes and notebooks are preserved.

## Inputs: skip the full scan

Default `data_mode='reuse_manifests'` requires:

1. The SAME Tiny-GenImage dataset version used by the old audit (actual image files).
2. Saved `seed42/train_manifest.csv`, `seed42/validation_manifest.csv`, and
   `seed42/test_manifest.csv`. Recommended alongside: root `config.json` and
   `seed42/split_summary.json` for extra config/order/count checks.

Use `hadinho0710/checkpoint-gsd-sdv5-train`, version containing
`gsd_raw_all7_crop560_v1/seed42/`, or the small `gsd_all7_manifests_seed42.zip` bundle.
No best.pt/last.pt is required or read from this input. Training still initializes
from original pretrained CLIP, downloaded with Internet enabled.

The notebook config now explicitly selects the observed input path
`/kaggle/input/datasets/hadinho0710/mainfest-train-genimage`. This fixes ambiguity when
both the manifest-only dataset and old checkpoint dataset are attached. The spelling
`mainfest` is the actual dataset slug, not a typo to correct in the path.
The notebook uses a fresh output directory ending in `_manifest_v3` to avoid the old
failed run's config.json conflict. The generic runtime default remains `_manifest_v2`;
the notebook's explicit CONFIG takes precedence.

For other inputs, set `manifest_root` to the directory CONTAINING `seed42/`
(a direct `seed42/` path is also accepted), and `dataset_root` to the directory
containing generator folders. `dataset_root=None` still auto-discovers the image dataset.
Only use `manifest_root=None` for discovery if exactly one matching source is attached;
the loader intentionally refuses to choose between different sources automatically.
Old absolute image paths in the CSV are ignored;
relative_path maps to the new Kaggle mount. Merely listing directories is not RGB scanning.

The loader preserves exact saved IDs, labels and ROW ORDER for all three cohorts,
including GSD logical-batch membership for a fixed test_batch. It never re-splits.
`validation_fraction` must match source config when supplied; it does not create a new split.
Smoke takes a small balanced subset within each existing cohort, without moving IDs across splits.

Checks: required metadata schema, all seven balanced cohorts, safe paths, path/label/split
consistency, saved hash leakage/conflicting-label checks, optional split_summary integrity,
and file existence/byte lengths. It NEVER decodes or recomputes content hashes of images.
It trusts the previous audit and an unchanged dataset version. Same-size content edits,
new/unlisted images and inaccurate old width/pixel-hash metadata are NOT independently detected.
Missing/ambiguous/changed input stops with an error; NO automatic full-scan fallback.

Explicit audit-from-scratch alternative: `data_mode='full_scan'`, with a fresh output_root.
This is the only mode that decodes/hashes source images before training.

## Exact contract

- Keep the All7 pixel-hash group split, class/generator balancing, crop560, edge-pad14,
  variable token grids, GSD geometry, batch8/microbatch1, optimizer and classification CE.
- Train only: independent Bernoulli0.5 blur and JPEG gates for each image, each epoch.
  Blur sigma is uniform [0,2]; JPEG quality is uniform integers75–100, subsampling0.
  Both selected: blur then JPEG. Both operate on original decoded RGB BEFORE crop.
  No resize, flip, paired consistency loss, auxiliary network, or extra train forward.
- The augmentation RNG uses run seed, epoch and stable sample_id, never labels/generators,
  row order, worker scheduling or global RNG state. Expected 25% per processing combination,
  not a quota forced for each batch. Small cohorts need not contain exactly 25% of each.
- Frozen anchor and detector consume the SAME transformed tensors from the loader.
- Original source files stay unchanged. Native still means the model's crop/pad pipeline,
  not a guarantee of full-image raw input. Existing JPEG history is not removed.

GSD paper: <https://arxiv.org/html/2603.09242v2#S5.SS1>. JPEG/blur are reported by the
authors; exact probabilities/ranges/order above are our explicit implementation choices.
All7/crop560/batch8 and our early stopping are not the paper's SD14 protocol.

## Fixed validation

Evaluate the SAME validation IDs in the SAME order/logical batches twice per epoch:

1. `native`: no extra augmentation.
2. `processed`: the training processing distribution, sampled by run seed,
   `validation_processing_seed=42` and sample_id. Epoch is intentionally excluded.

CE is computed directly from logits, without probability clipping. Use the mean of
per-image CE in each view, then equal-weight average the two views:

`val_loss = 0.5 * val_native_loss + 0.5 * val_processed_loss`

Both checkpoint selection and early stopping use this monitor. Native and processed
metrics are logged separately. With the default recipe, 25% of the processed view is
expected to receive neither operation; the monitor therefore weights no-extra-processing
images about62.5% overall. This is intentional: the processed view matches train, it is
not a balanced average over distinct corruption severities. No test threshold tuning.

Validation requires two inference passes instead of one. Each view independently
estimates GSD bases from that view's tensors. Do not combine them into a doubled batch.
The anchor stays frozen; only the detector/GSD/head are optimized.

## Artifacts and resume

Notebook output: `/kaggle/working/gsd_all7_crop560_blur_jpeg_manifest_v3`.

- At root: `manifest_reuse.json` records source CSV hashes, paths, checks/trust limitations;
  `reused_manifest_index.csv` stores the resolved selected-image index (not a fresh audit).

- Per seed: existing train/validation/test manifests plus `validation_processing_plan.csv`
  (per-image sigma/quality/gates/batch ID) and `validation_protocol.json` (seed/order/monitor).
- Per model: atomic `best.pt`/`last.pt`, `history.json`, `training_status.json`,
  `validation_metrics.csv`, `validation_predictions_epochNNN.csv`, `selected_checkpoint.json`.
- Validation predictions include both P(fake) scores and batch IDs, with labels/IDs.
- Post-training native test: the existing predictions/metrics/macro/completion outputs.

Start from pretrained CLIP, not the raw-only detector. Never point output_root at an old
experiment. `resume=True` restores only this protocol's last completed epoch with identical
config/manifests, optimizer/scaler/stopper/history. Raw-only checkpoints are rejected.
Resume retains epoch-boundary recovery, and checks the saved manifest/input fingerprint.
No mid-epoch recovery is introduced. Kaggle working storage is ephemeral;
save notebook outputs/download them before a reset. No guarantee against session timeouts.

Run `smoke=True` in a separate output_root first. Smoke retains the max560 input contract,
augmentation and fixed validation; it reduces cohort size, batch and k for one epoch.
Full pretrained CUDA VRAM/runtime on Kaggle has NOT been measured locally.

## Build and verify

```text
python baselines/clip_gsd_tiny_kaggle/build_aug_all7_notebook.py
python -m unittest discover -s baselines/clip_gsd_tiny_kaggle -p "test_*.py" -v
```

New CPU tests use tiny random CLIP models and temporary images. They cover RNG isolation,
label blindness, operation order and pixels, stable validation, native compatibility,
direct-logit CE, per-view batch membership, monitor/checkpoint/early-stop/resume and the
self-contained notebook. They do not substitute for the Kaggle smoke run.

To produce a manifest-only input from saved local outputs:

```text
python baselines/clip_gsd_tiny_kaggle/package_all7_manifests.py --source reports/gsd_all7_crop560_analysis/source --output baselines/clip_gsd_tiny_kaggle/gsd_all7_manifests_seed42.zip
```

This archive contains only the three CSVs, config and split summary. No detector weights,
predictions, credentials or image data are bundled.
