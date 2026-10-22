# Codec encoder → Gaussian → Tiny / B-Free

Import forensic_twins_codec_gmm_eval_kaggle.ipynb. Source is embedded; no repository download needed. Stages: prepare → fit → tiny → bfree; rerun each stage independently. No encoder training or test-driven threshold selection.

## Required Kaggle Inputs

| Input | Role |
|---|---|
| [checkpoint-encoder-twins-codec-200k](https://www.kaggle.com/datasets/hoangha071005/checkpoint-encoder-twins-codec-200k) | encoder_final.pt, SHA256, fre.pt, train_real_200k.csv under forensic_twins_codec/ |
| [evaluation-twins-bfree](https://www.kaggle.com/datasets/hoangha071005/evaluation-twins-bfree) | Saved full-run V2 manifests under forensic_twins_augmented_gmm_full/manifests/ |
| [ImageNet competition](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) | Original real pixels; accept competition rules |
| [COCO2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) | Original real pixels |
| [Tiny-GenImage](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) | Tiny test pixels |
| [B-Free](https://www.kaggle.com/datasets/baopv051/b-free-training-data) | Clean-manifest test pixels |

No native checkpoint or old fitted GMM required. Dataset root resolver accepts short and owner-qualified Kaggle mounts. Exact directory overrides are exposed in first cell. If saved manifests are ZIP-packaged, unpack only the required manifests into Working and set SAVED_MANIFESTS to their directory; do not regenerate/split a different cohort.

## Data and scoring

- Exact train200K manifest and saved fit160K/calibration40K hashes are checked; their IDs must partition the parent cohort, without intersection.
- B-Free clean manifest hash checked, 1557 source IDs/10899 images, complete1real+6fake groups, no source-ID overlap with COCO train100K.
- Freeze codec epoch20/FRE, drop projector. Legacy evaluation preprocessing is preserved: center crop256, grayscale[0,1], zero halo7, FRE, CLS192. This has the same boundary convention as prior native evaluation.
- Fit Gaussian K1/full covariance with reg_covar1e-6 on native fit160K embeddings. No PCA/extra L2 normalization. Q95 estimated from NLL on native calibration40K; Q90/Q99 supplementary. Calibration was seen during encoder SSL, held out only from Gaussian fitting.
- Fake iff NLL>threshold. Score is anomaly score, not calibrated P(fake). Fixed thresholds reused for all test protocols, without matching calibration to test labels.

## Outputs and resume

/kaggle/working/forensic_twins_codec_gmm contains run_identity.json, preflight.json, copied/rebased manifests, real embedding memmaps/progress, artifacts/gmm/{real_k1_full.npz,metadata,thresholds.json,calibration_nll.npy,calibration_thresholds.png}, tiny/ and bfree/ metrics/predictions/figures.

Embeddings save every1000 images and at completion; same output can resume on rerun. Copy a previous Output tree into OUTPUT_ROOT to continue a new session. Run identity checks encoder/FRE/manifest hashes, dimension, preprocessing and threshold recipe before cache reuse. Keep original directory structure. Save Version persists Working; no automatic publish to an unspecified output Dataset.

Tiny: native, fake-codec-matched, JPEG95/80/65/50, resize0.75; per-generator AUROC/AP, Q90/95/99 threshold metrics and predictions. Codec-matched fake JPEG uses per-image real qtables/subsampling, native real unchanged; remains diagnostic.
B-Free: same robustness protocols, paired balanced per-variant metrics, macro averages, bootstrap CIs grouped by source, histograms/ROC/PR/confusion and protocol comparison. Native B-Free real was previously processed; PNG is not evidence of RAW provenance.

Local verification: test_codec_gmm.py validates Gaussian NLL/serialization, Q95 real FPR on synthetic data and actual application of fake-codec-matched JPEG before crop. Native evaluation tests and notebook embedded-source compilation are also run. Full CUDA extraction requires Kaggle; CPU tests do not establish detector performance.
