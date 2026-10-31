# CAPD Tiny-GenImage checkpoint audit

Source: Kaggle dataset `hadinho0710/checkpoin-capd-train-tiny`, version 1.

## Executive summary

- The package contains two runs: `r1_clip` and `r2_dinov3`.
- `r1_clip` was interrupted during epoch 3 after two completed epochs. Its best completed epoch is epoch 1; it has no Tiny-GenImage test evaluation.
- `r2_dinov3` completed five epochs and stopped early. The selected checkpoint is epoch 3, with the lowest fixed-validation monitor loss, `0.006938915`.
- The selected DINOv3 checkpoint reaches 99.90% accuracy and balanced accuracy on the 7,000-image native Tiny-GenImage test set: TN=3,494, FP=6, FN=1, TP=3,499.
- Only the native test case was executed. The package does not contain JPEG, blur, resize, or external-dataset test results.

## Protocol

- Split: 25,200 train, 2,800 validation, 7,000 official test images.
- Generators: SD1.5, ADM, BigGAN, GLIDE, Midjourney, Wukong, VQDM.
- Training images use random Gaussian blur (`p=0.5`, sigma 0–2) and JPEG (`p=0.5`, quality 75–100).
- Validation uses fixed native and fixed processed views. Early stopping monitors `0.5 * CE(native) + 0.5 * CE(processed)`.
- Input geometry: center crop with maximum side 560, no resize, then replicate-pad to the backbone patch multiple.
- DINOv3 backbone: ViT-L/16 with rank-8 LoRA and a patch-logit head; only about 1.12% of parameters are trainable.

## Run status

| Run | State | Completed epochs | Best epoch | Best monitor CE | Interpretation |
|---|---:|---:|---:|---:|---|
| CLIP ViT-L/14 | Interrupted | 2 | 1 | 0.018441 | Partial result only; do not present it as a finished run. |
| DINOv3 ViT-L/16 | Complete, early-stopped | 5 | 3 | 0.006939 | Recommended checkpoint. |

The CLIP monitor loss worsened from 0.018441 at epoch 1 to 0.027317 at epoch 2 while ECE also worsened, indicating growing overconfidence toward the fake class. DINOv3 improved through epoch 3, then validation CE rose at epochs 4–5 while training loss continued to fall, which is mild overfitting/calibration drift. Selecting epoch 3 is correct.

## DINOv3 training and validation

| Epoch | Train loss | Native val acc. | Processed val acc. | Monitor CE | Note |
|---:|---:|---:|---:|---:|---|
| 1 | 0.084468 | 99.7143% | 99.4286% | 0.010609 | Initial convergence |
| 2 | 0.012066 | 99.8571% | 99.7143% | 0.007034 | Large improvement |
| 3 | 0.006162 | 99.8571% | 99.6786% | **0.006939** | Selected best checkpoint |
| 4 | 0.004291 | 99.7500% | 99.5714% | 0.010706 | Validation CE worsens |
| 5 | 0.003549 | 99.8929% | 99.7857% | 0.010114 | Early stopping triggers |

## Native Tiny-GenImage test, DINOv3 epoch 3

| Generator | N | Accuracy | Real recall | Fake recall | ROC-AUC | AP | Confusion matrix `[TN FP; FN TP]` |
|---|---:|---:|---:|---:|---:|---:|---|
| **Pooled** | 7,000 | **99.90%** | 99.8286% | 99.9714% | 99.9968% | 99.9966% | `[3494 6; 1 3499]` |
| ADM | 1,000 | 99.90% | 100.00% | 99.80% | 99.9992% | 99.9992% | `[500 0; 1 499]` |
| BigGAN | 1,000 | 99.80% | 99.60% | 100.00% | 100.00% | 100.00% | `[498 2; 0 500]` |
| GLIDE | 1,000 | 99.80% | 99.60% | 100.00% | 100.00% | 100.00% | `[498 2; 0 500]` |
| Midjourney | 1,000 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | `[500 0; 0 500]` |
| SD1.5 | 1,000 | 99.90% | 99.80% | 100.00% | 100.00% | 100.00% | `[499 1; 0 500]` |
| VQDM | 1,000 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | `[500 0; 0 500]` |
| Wukong | 1,000 | 99.90% | 99.80% | 100.00% | 99.9864% | 99.9860% | `[499 1; 0 500]` |

At threshold 0.5, there are seven errors: six real images classified as fake and one ADM fake classified as real. Two false positives are highly confident (`p(fake)=0.9958` and `0.99993`), so the near-perfect average metrics do not eliminate rare severe mistakes.

## Audit conclusions

1. Use `r2_dinov3/seed42/best.pt`; it corresponds to epoch 3.
2. Do not use the test-optimized threshold (`0.6396`) as a deployment threshold. Keep 0.5 or select/calibrate a threshold on a separate validation set.
3. Do not claim processing robustness from this package. The processed validation is strong, but only the native official test was run.
4. Run the same checkpoint on JPEG, blur, resize, combined processing, B-Free, and other out-of-domain sets before drawing generalization conclusions.
5. The metadata strings `GSD ...` and `edge_pad14` are inherited labels. The actual DINOv3 data path pads to patch 16 and the model is CAPD, not GSD.

