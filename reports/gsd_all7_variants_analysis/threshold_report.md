# Post-hoc threshold sweep: GSD All7 image variants

**TEST-LABEL SELECTION. These are in-sample descriptive optima, not independently validated deployment thresholds.**

No model inference or retraining. Original predictions/config/model were not modified. Fake is label 1. Final score is `prob_fake` (class-1 softmax probability). Predict fake iff `prob_fake >= threshold`. Equal scores move as a complete group. For objective ties choose the highest threshold. Precision is recorded as zero if no fake is predicted.

Source: D:\LLM\LVLM\MLLM_detect_fake_image\HoangHa_Code\reports\gsd_all7_variants_analysis\source\predictions.csv

SHA-256: `674a43dc8e740cfa96e3c95732e512d737a6461af42548b1f0baa7e40469b621`

Validated 70,000 rows, ten cases, identical ordered 7,000 IDs and logical batch membership. Each case has 3,500 real and 3,500 fake; each generator has 500+500 images. Baseline at 0.5 agrees with saved metrics; max absolute difference 1.1102230246251565e-16.

## Fake-only JPEG96: threshold objectives

| Strategy | Threshold | Acc=BA (%) | Real recall (%) | Fake recall (%) | Fake precision (%) | Fake F1 (%) | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original threshold | 0.5 | 64.90 | 99.94 | 29.86 | 99.81 | 45.96 | [3498 2; 2455 1045] (real/fake) |
| Best BA / Youden on TEST | 0.00011279297177679837 | 89.24 | 93.94 | 84.54 | 93.31 | 88.71 | [3288 212; 541 2959] (real/fake) |
| Best fake F1 on TEST | 0.00010229985491605476 | 89.14 | 92.29 | 86.00 | 91.77 | 88.79 | [3230 270; 490 3010] (real/fake) |
| Highest recall with FPR <=1% on TEST | 0.00037555920425802469 | 85.14 | 99.00 | 71.29 | 98.62 | 82.75 | [3465 35; 1005 2495] (real/fake) |
| Highest recall with FPR <=5% on TEST | 0.00013032974675297737 | 88.96 | 95.03 | 82.89 | 94.34 | 88.24 | [3326 174; 599 2901] (real/fake) |

ROC-AUC stays 94.17%; AP stays 95.58%. Changing the decision threshold does not change the underlying ranking/scores.

## Fake-only JPEG96: fixed thresholds

| Strategy | Threshold | Acc=BA (%) | Real recall (%) | Fake recall (%) | Fake precision (%) | Fake F1 (%) | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 0.5 | 64.90 | 99.94 | 29.86 | 99.81 | 45.96 | [3498 2; 2455 1045] (real/fake) |
| fixed | 0.10000000000000001 | 69.07 | 99.94 | 38.20 | 99.85 | 55.26 | [3498 2; 2163 1337] (real/fake) |
| fixed | 0.01 | 74.26 | 99.83 | 48.69 | 99.65 | 65.41 | [3494 6; 1796 1704] (real/fake) |
| fixed | 0.001 | 81.50 | 99.57 | 63.43 | 99.33 | 77.42 | [3485 15; 1280 2220] (real/fake) |
| fixed | 0.00050000000000000001 | 83.86 | 99.26 | 68.46 | 98.93 | 80.92 | [3474 26; 1104 2396] (real/fake) |
| fixed | 0.00020000000000000001 | 87.56 | 97.43 | 77.69 | 96.80 | 86.19 | [3410 90; 781 2719] (real/fake) |
| fixed | 0.0001 | 89.04 | 91.77 | 86.31 | 91.30 | 88.74 | [3212 288; 479 3021] (real/fake) |
| fixed | 5.0000000000000002e-05 | 80.11 | 66.77 | 93.46 | 73.77 | 82.46 | [2337 1163; 229 3271] (real/fake) |
| fixed | 1.0000000000000001e-05 | 49.99 | 0.00 | 99.97 | 49.99 | 66.65 | [0 3500; 1 3499] (real/fake) |
| fixed | 0 | 50.00 | 0.00 | 100.00 | 50.00 | 66.67 | [0 3500; 0 3500] (real/fake) |

## Fake-only JPEG96: ONE shared threshold, per generator

Shared threshold selected on the pooled JPEG96 TEST set: `0.00011279297177679837`. The following table does NOT select different thresholds per generator.

| Strategy | Threshold | Acc=BA (%) | Real recall (%) | Fake recall (%) | Fake precision (%) | Fake F1 (%) | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adm | 0.00011279297177679837 | 93.80 | 94.40 | 93.20 | 94.33 | 93.76 | [472 28; 34 466] (real/fake) |
| biggan | 0.00011279297177679837 | 94.70 | 93.80 | 95.60 | 93.91 | 94.75 | [469 31; 22 478] (real/fake) |
| glide | 0.00011279297177679837 | 86.00 | 94.60 | 77.40 | 93.48 | 84.68 | [473 27; 113 387] (real/fake) |
| midjourney | 0.00011279297177679837 | 93.40 | 94.20 | 92.60 | 94.11 | 93.35 | [471 29; 37 463] (real/fake) |
| sd15 | 0.00011279297177679837 | 83.50 | 93.00 | 74.00 | 91.36 | 81.77 | [465 35; 130 370] (real/fake) |
| vqdm | 0.00011279297177679837 | 89.10 | 95.40 | 82.80 | 94.74 | 88.37 | [477 23; 86 414] (real/fake) |
| wukong | 0.00011279297177679837 | 84.20 | 92.20 | 76.20 | 90.71 | 82.83 | [461 39; 119 381] (real/fake) |

## Independently optimized threshold for each case (TEST oracle)

Each row selects a different threshold using labels of that same case. These results are not a single globally calibrated model.

| Strategy | Threshold | Acc=BA (%) | Real recall (%) | Fake recall (%) | Fake precision (%) | Fake F1 (%) | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 0.5397413969039917 | 99.97 | 99.94 | 100.00 | 99.94 | 99.97 | [3498 2; 0 3500] (real/fake) |
| png_roundtrip | 0.5397413969039917 | 99.97 | 99.94 | 100.00 | 99.94 | 99.97 | [3498 2; 0 3500] (real/fake) |
| fake_jpeg96 | 0.00011279297177679837 | 89.24 | 93.94 | 84.54 | 93.31 | 88.71 | [3288 212; 541 2959] (real/fake) |
| all_jpeg96 | 0.00010472557187313214 | 89.47 | 93.23 | 85.71 | 92.68 | 89.06 | [3263 237; 500 3000] (real/fake) |
| all_jpeg90 | 6.6830696596298367e-05 | 70.97 | 81.94 | 60.00 | 76.87 | 67.39 | [2868 632; 1400 2100] (real/fake) |
| all_jpeg80 | 5.6058292102534324e-05 | 63.96 | 73.94 | 53.97 | 67.44 | 59.96 | [2588 912; 1611 1889] (real/fake) |
| resize_square256 | 0.79795074462890625 | 93.19 | 93.69 | 92.69 | 93.62 | 93.15 | [3279 221; 256 3244] (real/fake) |
| resize_square560 | 0.031084952875971794 | 98.76 | 98.57 | 98.94 | 98.58 | 98.76 | [3450 50; 37 3463] (real/fake) |
| blur_sigma1 | 0.998809814453125 | 98.93 | 98.40 | 99.46 | 98.42 | 98.93 | [3444 56; 19 3481] (real/fake) |
| resize256_jpeg96 | 0.00017674513219390064 | 72.57 | 86.54 | 58.60 | 81.32 | 68.12 | [3029 471; 1449 2051] (real/fake) |

## Apply the SAME JPEG96 threshold to every case

This diagnoses transfer only. Do not replace the original global threshold based on the JPEG96 optimum.

| Strategy | Threshold | Acc=BA (%) | Real recall (%) | Fake recall (%) | Fake precision (%) | Fake F1 (%) | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 0.00011279297177679837 | 96.76 | 93.51 | 100.00 | 93.91 | 96.86 | [3273 227; 0 3500] (real/fake) |
| png_roundtrip | 0.00011279297177679837 | 96.76 | 93.51 | 100.00 | 93.91 | 96.86 | [3273 227; 0 3500] (real/fake) |
| fake_jpeg96 | 0.00011279297177679837 | 89.24 | 93.94 | 84.54 | 93.31 | 88.71 | [3288 212; 541 2959] (real/fake) |
| all_jpeg96 | 0.00011279297177679837 | 89.39 | 94.23 | 84.54 | 93.61 | 88.85 | [3298 202; 541 2959] (real/fake) |
| all_jpeg90 | 0.00011279297177679837 | 68.14 | 94.57 | 41.71 | 88.48 | 56.70 | [3310 190; 2040 1460] (real/fake) |
| all_jpeg80 | 0.00011279297177679837 | 60.07 | 95.11 | 25.03 | 83.67 | 38.53 | [3329 171; 2624 876] (real/fake) |
| resize_square256 | 0.00011279297177679837 | 64.36 | 28.74 | 99.97 | 58.38 | 73.72 | [1006 2494; 1 3499] (real/fake) |
| resize_square560 | 0.00011279297177679837 | 90.90 | 81.91 | 99.89 | 84.67 | 91.65 | [2867 633; 4 3496] (real/fake) |
| blur_sigma1 | 0.00011279297177679837 | 60.27 | 20.54 | 100.00 | 55.72 | 71.57 | [719 2781; 0 3500] (real/fake) |
| resize256_jpeg96 | 0.00011279297177679837 | 71.47 | 75.54 | 67.40 | 73.37 | 70.26 | [2644 856; 1141 2359] (real/fake) |

## Interpretation

The large JPEG96 gain confirms a poor operating point at 0.5, but discrimination has still degraded (native AUC nearly 100%, JPEG96 AUC 94.17%). JPEG90/80 remain substantially impaired even at their in-sample best thresholds. The JPEG96 threshold transfers poorly to blur/resize. Use an independent validation/calibration set matching intended conditions, freeze its threshold, and evaluate on a fresh untouched test set before claiming a robustness improvement or comparing against CLIP/AIDE with thresholds selected under a different protocol.

The JSON companion retains all thresholds at full floating-point precision, score quantiles, and explicitly labeled generator-specific oracle optima for diagnostics only.
