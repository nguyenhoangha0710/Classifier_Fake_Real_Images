# Forensic Twins Codec — encoder-only Kaggle run

Import **forensic_twins_codec_encoder_kaggle.ipynb** into Kaggle. Notebook embeds the native model sources and new trainer; no repository download needed.

## Inputs

1. Competition: imagenet-object-localization-challenge (accept competition rules).
2. Dataset: awsaf49/coco-2017-dataset.
3. Dataset: hoangha071005/twins-gmm-200k-checkpoint.

Do not add B-Free or fitted GMM for this encoder-only run. Enable GPU T4 ×2. Internet/authentication is needed to upload to the explicitly configured destination. No credentials are embedded in code.

Exact native bundle relative paths:

```text
forensic_twins_v2_200k/manifests/train_real_200k.csv
forensic_twins_v2_200k/cache/fre.pt
forensic_twins_v2_200k/artifacts/checkpoints/last.ckpt
```

Preflight verifies the known manifest hash, 200K real membership/source counts, every pixel path and native checkpoint/FRE hashes. Both known dataset mount layouts and ImageNet competition mounts are resolved by native helpers. Unknown/missing inputs stop execution; the notebook does not invent alternative manifests.

## Training and storage

20 epochs, true global batch128, shared native architecture/projector, frozen FRE; warm-start encoder/projector only, reset optimizer/schedule/scaler. Step-keyed augmentations are reproducible after resume. Full RGB processing precedes grayscale and crop+halo. To keep rank-consistent crop sizes inside the min-side256 interior, the impossible square256 pair128/128 becomes96/128 deterministically; all other crop pairs remain sampled from64/96/128.

Working/Output root: /kaggle/working/forensic_twins_codec. Flat publish layout:

```text
last.ckpt + SHA256             # saved each250 optimizer updates and each epoch
encoder_final.pt + SHA256      # inference weights after epoch20
fre.pt, train_real_200k.csv
config.json, run_metadata.json
logs/train_steps.csv           # every100 steps, plus first batch
logs/epoch_metrics.csv
figures/training_curves.png    # loss, invariance, redundancy, LR
figures/epoch_loss.png
```

Every epoch publishes latest checkpoint, provenance, logs.zip and figures.zip to https://www.kaggle.com/datasets/hoangha071005/forensic-twins-codec-checkpoint-encoder . Working also retains epoch5/10/15/20 snapshots. Publishing errors preserve local files and print a warning; Save Version preserves Working as notebook Output. Dataset creation is attempted only on an explicit missing-dataset response; permission failures are reported.

For a new-session resume, add that dataset as Input, set RESUME_CHECKPOINT to its flat last.ckpt path and keep MODE/config/native dataset version unchanged. Resume validates the run identity, restores optimizer/scheduler/scaler/step/epoch, and recreates exact augmentation by sample/epoch/batch. Copy previous logs into Working (or unzip logs.zip) to continue historical plots; otherwise fresh logs describe the remaining segment. Native checkpoints are never accepted as codec resume checkpoints.

MODE=native_native enables the continued-training control; use a separate OUTPUT_ROOT and PUBLISH=False to avoid mixing it into the authorized codec dataset.

## Local maintenance

python test_codec.py checks policy distribution, native-size preservation, reproducibility, disjoint crops, halo equivalence, resume batch selection and encoder/FRE gradient contracts on CPU.
python build_notebook.py regenerates the notebook after source edits. Native implementation is reused without modifying its files.

GPU/DDP throughput and full200K run still require actual Kaggle execution. This notebook performs no GMM fitting, calibration or B-Free testing; those need separate entrypoints and newly fitted density models.
