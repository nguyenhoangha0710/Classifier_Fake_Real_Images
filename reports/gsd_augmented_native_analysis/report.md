# Augmented GSD: native evaluation audit

Dataset v1. GPU Tesla T4, FP16 + SDPA. Checkpoint epoch1, seed42.
Only native is complete (7,000 images). The 16-image smoke and duplicated native export are not independent tests.
No JPEG/resize/blur results are present. status=complete refers only to the requested native case.

|Generator|N|Accuracy|Real recall|Fake recall|AUC|AP|TN|FP|FN|TP|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|pooled|7000|99.8571%|99.7429%|99.9714%|99.9986%|99.9986%|3491|9|1|3499|
|adm|1000|100.0000%|100.0000%|100.0000%|100.0000%|100.0000%|500|0|0|500|
|biggan|1000|99.8000%|99.6000%|100.0000%|99.9984%|99.9984%|498|2|0|500|
|glide|1000|99.6000%|99.2000%|100.0000%|100.0000%|100.0000%|496|4|0|500|
|midjourney|1000|99.8000%|99.8000%|99.8000%|99.9988%|99.9988%|499|1|1|499|
|sd15|1000|99.9000%|99.8000%|100.0000%|100.0000%|100.0000%|499|1|0|500|
|vqdm|1000|100.0000%|100.0000%|100.0000%|100.0000%|100.0000%|500|0|0|500|
|wukong|1000|99.9000%|99.8000%|100.0000%|100.0000%|100.0000%|499|1|0|500|

## Interpretation

Strong in-distribution discrimination at the fixed 0.5 threshold; 9 false positives and 1 false negative.
This does not establish robustness to JPEG/resize/blur or unseen-generator/dataset transfer.
Saved native/processed validation is from the selected checkpoint, not a full training history or proof of convergence.
Old raw-model native accuracy is slightly higher; this alone neither proves overfitting nor establishes augmentation harm.
No threshold was optimized on this test set. Detailed integrity checks, score quantiles and error IDs are in analysis.json.
