# GMM calibration and Tiny-GenImage evaluation

## Inputs

The evaluation notebook requires four Kaggle Inputs:

1. `hoangha071005/twins-gmm-checkpoint`;
2. ImageNet Object Localization Challenge;
3. `awsaf49/coco-2017-dataset`;
4. `yangsangtai/tiny-genimage`.

The checkpoint Dataset supplies `last.ckpt`, `fre.pt`, and the exact 50K-real
training manifest. ImageNet and COCO must be mounted because the manifest stores
paths and hashes, not image bytes.

## Fit and calibration protocol

The 50K real rows are split deterministically with seed 42:

- 80% GMM fit set (40K expected);
- 20% threshold calibration set (10K expected).

ImageNet is stratified by synset. With 25 images per class, five images per
class enter calibration. COCO is deterministically shuffled and split by source.
The two subsets are disjoint.

Both subsets use a 256x256 center crop without resizing, frozen FRE, and the
frozen 192-D encoder. A single full-covariance Gaussian is fit only on the fit
embeddings. The anomaly score is Gaussian negative log likelihood (NLL): higher
means less compatible with the real-image distribution.

Thresholds q80 through q98 are quantiles of calibration-real NLL. The primary
threshold is q95. Prediction is `fake` exactly when `NLL > threshold`. Thresholds
must never be selected by optimizing Tiny-GenImage labels.

## Evaluation protocols

- `native_confounded`: native real JPEG versus native fake files. This can expose
  a JPEG/PNG shortcut and is diagnostic only.
- `fake_codec_matched`: fake images are encoded once using real JPEG quantization
  tables and subsampling profiles. This is the primary result.

The same real-calibrated thresholds are used for both protocols.

## Saved metrics

For pooled data and every generator separately:

- AUROC and average precision;
- TN, FP, FN, TP;
- accuracy and balanced accuracy;
- precision, recall/TPR, specificity/TNR, F1;
- FPR and FNR at every threshold q80 through q98.

The primary q95 confusion matrix is saved as a PNG for pooled data and each
generator. Per-image CSV files contain NLL and all q80-q98 predictions.

## Saved visualizations

For each native/codec-matched protocol:

- pooled and per-generator q95 confusion matrices;
- ROC and precision-recall curves;
- NLL distributions by generator with the q95 line;
- a global PCA scatter of GMM-whitened embeddings colored by generator and true
  label;
- one PCA image per generator colored by TN/FP/FN/TP at q95.

PCA is for visualization only and is never used to calculate NLL or predictions.

## Main artifacts

```text
artifacts/
|-- gmm/
|   |-- real_k1_full.npz
|   |-- real_k1_full.json
|   `-- thresholds.json
|-- embeddings/
|-- metrics/
|   |-- native_confounded_summary.json
|   |-- native_confounded_threshold_metrics.csv
|   |-- native_confounded_predictions.csv
|   |-- fake_codec_matched_summary.json
|   |-- fake_codec_matched_threshold_metrics.csv
|   |-- fake_codec_matched_predictions.csv
|   `-- <protocol>/per_generator/
|-- figures/
|   `-- <protocol>/
`-- evaluation_complete.json
```

The notebook also creates
`/kaggle/working/forensic_twins_gmm_eval_artifacts.zip` for download.
