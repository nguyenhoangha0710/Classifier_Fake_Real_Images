# Recovered best checkpoint: image-variant evaluation

Use `test_gsd_uploaded_best_variants_kaggle.ipynb`, not the older raw-only evaluator.
This evaluator accepts the recovered AUGMENTED All7 crop560 checkpoint, signature
`4595045b4c6c76c1ad4aa303a7d2f1690cfac8150add4548d24609a6c980dce6`, epoch1.
It reads the config directly from the checkpoint, does no training and never uses test
labels to tune thresholds or choose model weights.

## Files and Kaggle inputs

The user's `best.zip` (1,212,855,384 bytes) is itself PyTorch's ZIP serialization:
data.pkl plus tensor storages, not an outer ZIP containing best.pt. Prefer uploading
the identical bytes named `best.pt` so the dataset uploader does not unpack its internals.
The loader also handles the original native best.zip, an outer ZIP containing best.pt,
or a folder of unpacked PyTorch internals (reconstructed as a deterministic ZIP container).
Only restricted `torch.load(weights_only=True)` is used. Never load unknown checkpoints.

Attach:

1. Uploaded checkpoint dataset with best.pt (or supported equivalent).
2. `yangsangtai/tiny-genimage`, the same images/version as the audited manifests.
3. `hadinho0710/mainfest-train-genimage` or the manifest-only ZIP. Only the saved
   seed42/test_manifest.csv is required; split_summary.json is optional/recommended.
4. On continuation only: the saved OUTPUT of the previous TEST run.

Internet is needed for ORIGINAL pretrained CLIP ViT-L/14 frozen anchor; an offline
pretrained_model_path may instead point to those original Hugging Face weights/config.
The fine-tuned detector is never substituted for the semantic anchor.

No external config.json, selected_checkpoint.json, train/validation images, last.pt or
optimizer checkpoint is needed for evaluation. The recovered best was saved after
epoch1 validation; it is not evidence of a completed early-stopped training run.

## Protocol and cases

Validate the original 7,000 balanced All7 IDs and order hash
`c544b38bf4276a2fce06fd205fb4dbb86fa772ac26532509679311e7d8a96aa2`.
Check metadata and file paths/byte lengths before inference; no full RGB/hash scan.
The image loader verifies dimensions while decoding each image for actual evaluation.
All transforms operate before the existing crop560/edge-pad14/CLIP normalization.
Keep batch8, Gram-eigh, SDPA, original precision, token interpolation and threshold0.5.

Nine default distinct cases: native, fake_jpeg96, all_jpeg96/90/80,
resize_square256/560, blur_sigma1, resize256_jpeg96. All cases recompute the entire
batch's GSD basis. Fake-only JPEG and all-image JPEG are NOT interchangeable, even
if real scores were nearly equal in an earlier checkpoint.

Lossless PNG roundtrip is not independently executed: it aliases native, with the
decision recorded explicitly in case_plan.json. Duplicate requests and identical
transform spellings normalize to one case. Aliases do not inflate metrics aggregates.
No old raw-only model predictions are reused as a reference for the new checkpoint.

## Runtime safety and resume

GPU notebook default: request ALL NINE cases with no per-invocation case-count limit,
while retaining the 9.5-hour time budget. Its launch cell re-applies the nine-case
plan, max_batches=None and max_new_cases_per_run=None immediately before inference,
so stale native-only/smoke settings cannot silently reduce a full run. SMOKE_ONLY=True
explicitly selects two batches of native/fake-JPEG96 in a separate smoke directory.
The general evaluator API and TPU notebook retain their own configurable case budgets.
Completed cached cases consume no model forwards.
If limited, return normally with status=paused; remaining notebook cells package results,
letting Save & Run All finish before Kaggle's hard timeout. This is not a guarantee
against external termination, hardware stalls, or time spent outside the evaluator.

Save complete-batch shards every10 batches (80 images). A planned stop also flushes
pending complete batches. Hard termination can lose unflushed work, and saved files
must actually be retained as notebook output/downloaded to survive session destruction.

After a paused run, save/download output and attach it as a dataset or notebook output
input on the next run. resume_from='auto' locates matching evaluation_state.json files,
or explicitly set a directory containing that file. Checkpoint hash, test cohort/order,
processing, precision/batches, library versions and GPU type must match. Changed
environment/checkpoint requires a fresh result directory; no silent cross-run score mix.
Do not merely rerun a clean session without the old progress input.

Only case lists, session budget and logging/save intervals may change without invalidating
the fingerprint. Resume is always at a logical-batch boundary, including mixed image sizes.
Native remains first for paired reports. Test labels are used only for metrics and the
explicit fake-only diagnostic transform, not for batch membership.

## Outputs

Root: `/kaggle/working/gsd_uploaded_best_variants_v1`.

- `checkpoint_inspection.json`: actual epoch/signature/config/validation and source SHA256.
- `run_fingerprint.json`, `case_plan.json`, `source_paths.json`, `evaluation_state.json`.
- `progress/<case>/part_*.csv`: reusable complete-batch predictions and metadata.json.
- `predictions_<case>.csv`, `metrics.csv` (pooled and per generator), macro_all.csv,
  paired/delta reports, combined predictions.csv: COMPLETED cases only.
- Partial case scores: predictions_CASE.partial.csv and partial_metrics.csv, explicitly
  incomplete. They are not included in the full-case metrics table.
- Every prediction includes prob_fake=softmax(logits)[:,1], both raw logits, their margin,
  label/ID/generator, dimensions, input token count, and batch ID.
- The final notebook cell exports a results ZIP excluding weights and images.

The old full evaluation notebook is unchanged. New CPU tests cover containers, fingerprint
checks, case deduplication, raw logits, time-budget flushes and cross-session resume.
They are not a pretrained CUDA benchmark; run the provided small smoke configuration first.
