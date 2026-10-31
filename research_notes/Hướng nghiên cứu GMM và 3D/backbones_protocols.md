# Foundation-model backbones, resize robustness, and evaluation protocols for generalizable AIGI detection

Scope: (a) frozen backbones for AI-generated image (AIGI) detection, (b) why resizing hurts and how SOTA copes, (c) benchmarks, protocols, and controls that reviewers expect in 2025–2026. The research context is a supervised CLIP ViT-L/14 + GSD detector (arXiv 2603.09242) trained on all 7 Tiny-GenImage generators. It gets 99.86% native accuracy and stays above 99.4% under JPEG80 and blur, but drops to 92.2% at square 256, 94.8% at square 560, and 90.1% at resize256+JPEG96.

Status labels: "peer-reviewed" means a confirmed conference or workshop venue. "Preprint" means arXiv with no venue listed on the page I checked.

---

## Q1. Which frozen backbones work best for AIGI detection, and why?

### Takeaway
Among frozen features, the 2025–2026 evidence ranks vision-language contrastive encoders highest: Perception Encoder (PE-Core) and SigLIP2 come first, ahead of CLIP ViT-L/14. Self-supervised DINOv2 and DINOv3 trail them under linear probing. DINOv2 does well once it is fine-tuned end-to-end at native resolution (B-Free). MAE, SAM2 and 3D-model encoders are weak or untested as frozen detectors. Pooled patch tokens beat the CLS token alone by about 3–7 points.

### Cited Findings
- **TAP into the Patch Tokens** (Abdullah, Ebert, Wasenmüller; arXiv 2604.26772, Apr 2026, preprint). Each frozen encoder gets a linear probe, trained on OpenSDI (100k Megalith-10M real images plus 100k SD1.5 full and inpainted images) and tested on SD1.5/2.1/XL/3 and Flux. Mean F1 / Acc at each model's best resolution:
  - DINOv2 ViT-G/14: 73.20 / 71.80 (224px)
  - DINOv3 ViT-L/16: 79.98 / 79.52 (256px)
  - DINOv3 ViT-7B: 83.08 / 82.11
  - SAM2.1 Hiera-L: 62.30 / 68.89
  - CLIP ViT-L/14: 75.63 / 77.45 (336px)
  - SigLIP2 ViT-L/16: 86.97 / 86.45 (512px)
  - SigLIP2 SO400M: 86.67 / 86.72
  - PE-Core-L: 87.52 / 86.24
  - PE-Core-G/14: 89.54 / 89.81 (448px)

  The paper concludes that CLIP-style models (SigLIP2, PE) beat self-supervised DINOv2 and DINOv3. — [arXiv 2604.26772](https://arxiv.org/html/2604.26772v1)
- **Resolution effect, same paper.** Fixed-resolution vision-language encoders improve as input resolution rises: SigLIP2-L goes from 78.65 F1 at 256px to 86.97 at 512px. DINOv2 and DINOv3 get worse as input resolution rises. — [arXiv 2604.26772](https://arxiv.org/html/2604.26772v1)
- **Patch tokens vs CLS, same paper.** Pooling patch tokens (TAP) improves every backbone (OpenSDI F1 / Acc):
  - CLIP-L: 75.63 / 77.45 → 82.06 / 83.33
  - PE-L: 87.52 → 91.64 F1
  - PE-G: 89.54 → 92.94 F1
  - SigLIP2 NaFlex: 80.99 → 88.76 F1

  — [arXiv 2604.26772](https://arxiv.org/html/2604.26772v1)
- **Same paper, standard protocols (trained on SD1.4 GenImage plus ImageNet reals, JPEG30–100 and blur augmentation, one epoch).**
  - GenImage mean accuracy: PE-Core-G 97.16, OMAT 94.63, AIDE 86.88.
  - Chameleon accuracy: PE-Core-G + TAP 95.64, prior best OMAT 66.05.
  - TAP lowers PE-G's GenImage mean to 95.67, and its Midjourney accuracy falls from 94.92 to 82.11. The authors attribute this to overfitting SD1.4 artifacts.

  Treat the Chameleon jump with caution: it is a preprint and the result is not independently replicated. — [arXiv 2604.26772](https://arxiv.org/html/2604.26772v1)
- **CLIP vs DINOv2, other studies.** One ablation swapping CLIP ViT-L/14 for DINOv2 found worse performance, which the authors attribute to CLIP's higher intrinsic dimension. A linear-probe study reports that CLIP beats DINOv2 when the backbone is frozen, and the opposite holds when it is fine-tuned. These claims come from search-result summaries of [arXiv 2411.19117](https://arxiv.org/pdf/2411.19117) and related 2025–2026 preprints (for example [arXiv 2605.13381 "Backbone is All You Need"](https://arxiv.org/pdf/2605.13381)). I did not verify the exact tables, so treat them as indicative.
- **Raising the Bar with CLIP** (Cozzolino et al.; arXiv 2312.00195; the brief lists it as CVPRW 2024, and the arXiv page names no venue). Setup:
  - Features: CLIP ViT-L/14, next-to-last layer, with a linear SVM.
  - Training data: COCO real images paired with Latent Diffusion fakes generated from the same captions, at 1k+1k or 10k+10k pairs. The method still works with 100 pairs.
  - Test set: about 18 generators, including DALL·E 2/3, Midjourney v5 and Firefly.
  - Metrics: AUC, AP, and accuracy at a 0.5 threshold.

  Results:
  - +6% AUC over prior SOTA on out-of-distribution data and +13% on post-processed data.
  - Post-processed test (random crop, resize to 200×200, JPEG65–100): 85.2 AUC vs 71.8 for UnivFD (Ojha).
  - The method is nearly insensitive to JPEG, WebP and resizing.
  - On images from X, Corvi et al.'s CNN gets 88.9 AUC vs 81.2 for the CLIP 10k+ variant.
  - Larger CLIP pretraining data (LAION-400M → CommonPool-12.8B) gives about +10 points.

  — [arXiv 2312.00195](https://arxiv.org/html/2312.00195v2)
- **AIDE** (Yan et al., "A Sanity Check for AI-generated Image Detection", ICLR 2025, peer-reviewed). It combines a semantic branch (OpenCLIP ConvNeXt) with a low-level branch: the 2 highest- and 2 lowest-frequency 32×32 patches, selected by DCT scoring, resized to 256, passed through SRM filters into ResNet-50s. GenImage mean accuracy when trained on SD1.4:
  - AIDE 86.88
  - PatchCraft 82.30
  - GenDet 81.56
  - UnivFD 73.29
  - DIRE 70.66

  — [arXiv 2406.19435](https://arxiv.org/html/2406.19435v3)
- **B-Free** (Guillaro et al.; arXiv 2412.17671 v2, Apr 2025; the arXiv page lists no venue, and I believe it appeared at CVPR 2025 but did not verify that). It fine-tunes DINOv2+registers ViT end-to-end at learning rate 1e-6. Training data: 51,517 COCO real images and 309k self-conditioned SD2.1-inpainting reconstructions, using 504×504 crops with no resizing. Average bAcc on Synthbuster + new generators + WildRF:
  - B-Free 96.3
  - DMID 75.6
  - RINE 65.0
  - UnivFD 62.4
  - AIDE 53.2
  - NPR 42.7

  — [arXiv 2412.17671](https://arxiv.org/html/2412.17671v2)
- **Community Forensics** (Park & Owens, CVPR 2025, peer-reviewed). The dataset has 2.7M images from 4,803 generators. Detection improves as the number of training generators grows, even when they share an architecture. — [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html). Its evaluation set is on Hugging Face (non-commercial use) — [HF CommunityForensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval/blob/main/README.md)
- **HiDA-Net** (Mu et al.; arXiv 2508.17346, Aug 2025, preprint). Frozen CLIP ViT-L/14 with a global 224 path plus native-resolution 224 tiles (no resizing), aggregated by a small Transformer.
  - Chameleon: 79.10% trained on all of GenImage vs 65.77% for AIDE; 73.44% trained on SD1.4 only.
  - GenImage SD1.4 → all: 97.1 vs C2P-CLIP 95.8 and SAFE 95.6.

  — [arXiv 2508.17346](https://arxiv.org/html/2508.17346)

### Inferences
- If the GMM/one-class line keeps a frozen encoder, reviewers will probably ask why it is not PE-Core or SigLIP2, since the 2026 frozen-probe evidence ranks both above CLIP ViT-L/14. A backbone sweep over CLIP-L, SigLIP2-L, PE-L, DINOv2-L and DINOv3-L with the same head is a cheap, expected table.
- I found no frozen-feature evidence for 3D-model encoders (Pi3, VGGT). Because they are DINOv2-initialized, the DINOv2 results above (weaker than vision-language encoders when frozen, and worse at higher resolution) are a reasonable prior. Any 3D-feature claim therefore needs a "DINOv2 alone" control, to show the 3D training adds signal beyond the DINOv2 initialization.
- Pooling patch tokens is a cheap, strong addition to a CLS- or GSD-based head. It is also natural for GMM anomaly scoring, which could model patch-token distributions per patch.

### Gaps
- No DINOv3-specific AIGI detector paper was located and verified (search budget). TAP only reports DINOv3 under a linear probe.
- MAE and BLIP as frozen AIGI backbones: no verified numbers were found.
- I did not fetch UnivFD's original numbers (Ojha et al., CVPR 2023); its numbers appear above only as a baseline in other papers.
- I did not verify the exact tables in "Backbone is All You Need" (arXiv 2605.13381) or arXiv 2411.19117.

---

## Q2. Why does resizing destroy detection cues, and how does SOTA handle it?

### Takeaway
Downscaling acts as a low-pass filter. It removes the high-frequency and upsampling (VAE/decoder) traces that low-level detectors rely on, and it adds its own interpolation traces, which can become a shortcut if real and fake images are resized differently. SOTA handles this in three ways:
- Avoid resizing during training and testing: native-resolution crops, padding, or multi-crop or tile aggregation (B-Free, SPAI, HiDA-Net).
- Use resolution-invariant spectral modeling (SPAI).
- Add resize, blur and JPEG augmentation during training (the "+" variants in Cozzolino et al., and B-Free's inpainted++).

### Cited Findings
- **Fake or JPEG?** (Grommelt et al., ECCV 2024 Workshops, peer-reviewed). In GenImage, real images are ImageNet JPEGs, mostly at quality 96, while fakes are PNGs at fixed sizes (MJ 1024, SD1.4/1.5 and Wukong 512, ADM/GLIDE/VQDM 256, BigGAN 128). Detectors learn these shortcuts:
  - A Midjourney-trained ResNet50 labels uncompressed FFHQ PNG photos as real 80.45% of the time; at JPEG95 it is 94.84%, and at JPEG60 it is 100%.
  - Detectors fail on real images whose size matches their training generator's output size.

  Debiasing (fakes also compressed at JPEG QF96, real images restricted to about 450–550 px, identical crop-then-resize at training and inference) raises cross-generator accuracy by +11.06 points for ResNet50 (71.68 → 82.74) and +11.74 for Swin-T (74.09 → 85.83). The paper also warns that resizing to a fixed input can leak real-versus-fake information through resampling artifacts, especially for non-square images. — [arXiv 2403.17608](https://arxiv.org/html/2403.17608v1); [ECCVW listing](https://mlanthology.org/eccvw/2024/grommelt2024eccvw-fake)
- **B-Free.** The authors avoid resizing because it can "wash out" subtle low-level forensic clues. They train on 504 crops; at test time, images smaller than 504 are padded and larger ones are handled by averaging logits over multiple crops. Augmentation helps most on resizing in their robustness tests (Fig. 6). They also note that resizing all LAION real images to one resolution introduced bias in prior work. On FakeInversion, going from original to a simulated social-network pipeline, AUC / bAcc change as follows:
  - B-Free: 99.3 / 86.3 → 98.5 / 86.4
  - NPR: 91.6 / 87.0 → 43.3 / 49.9
  - AIDE: 85.5 / 76.9 → 67.3 / 56.4
  - UnivFD: 54.9 → 49.7 AUC

  — [arXiv 2412.17671](https://arxiv.org/html/2412.17671v2)
- **SPAI** (Karageorgiou et al., CVPR 2025, peer-reviewed). Setup:
  - Encoder: frozen ViT-B/16, pretrained with masked frequency modeling on real images only (the self-supervised spectral pretext task).
  - Detector training: 180k real and 180k images from one LDM (the Corvi/DMID data), at about 256².
  - Any-resolution handling: Spectral Context Attention runs over K native 224×224 patches, with K=4 during training and the image's actual K at inference. Cost scales linearly, up to 1 GP images.

  Mean AUC over 13 generators (Glide through Flux, SD3, MJv6.1, DALL·E 3 and Firefly):
  - SPAI 91.0
  - RINE 85.5
  - DMID 83.5
  - PatchCraft 80.4
  - UnivFD 67.3
  - NPR 45.4

  Ablations:
  - Center crop without SCA: 83.2 AUC; ten-crop with mean fusion: 85.3.
  - Removing noise and blur augmentation: 84.2; removing JPEG augmentation: 89.1.

  Robustness to JPEG/WebP, blur, noise and resizing (85/70/50%) is shown only as plots, with SPAI above the two best competitors. — [arXiv 2411.19417](https://arxiv.org/html/2411.19417v2); [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html)
- **HiDA-Net** (preprint). It uses a DFT argument and Fig. 2 to show that 448→224 resizing on GenImage SD1.4 removes high-frequency spectral differences. Its fix is full-coverage native tiles plus a global resized view, with an auxiliary JPEG quality-factor estimation loss. Tile-count ablation: 1 tile 92.14, 4 tiles 95.63, 16 tiles 95.89, full coverage 96.10. — [arXiv 2508.17346](https://arxiv.org/html/2508.17346)
- **Other work on padding and scaling.**
  - A NeurIPS 2024 paper ("Breaking Semantic Artifacts") found that zero-padding preserves more low-level features than resizing and improves generalization. — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/6dddcff5b115b40c998a08fbd1cea4d7-Paper-Conference.pdf)
  - A 2026 preprint reports that most existing detectors use center cropping at inference because resizing causes a substantial drop in accuracy. — [arXiv 2607.14684](https://arxiv.org/html/2607.14684v1)
  - Another 2026 preprint finds accuracy lowest at 0.5× downscaling, with moderate degradation at 1.5–2× upscaling from interpolation smoothing. — [arXiv 2607.10695](https://arxiv.org/pdf/2607.10695)
- **Cozzolino et al.** Semantic CLIP features trained with resize and JPEG augmentation ("+") were nearly insensitive to resizing, whereas low-level CNN detectors degraded under strong resizing and WebP. — [arXiv 2312.00195](https://arxiv.org/html/2312.00195v2)
- **AIDE robustness** (averaged over 16 AIGCDetectBenchmark sets).
  - Original: 92.77.
  - JPEG95: 75.54; JPEG50: 69.60.
  - Blur σ=1: 81.88.
  - Resizing is not reported.

  — [arXiv 2406.19435](https://arxiv.org/html/2406.19435v3)

### Inferences
- The researcher's pattern fits this literature: robust to JPEG and blur, which were in the training augmentations, but sensitive to resizing, which was not. Likely causes:
  1. GSD-CLIP is exploiting decoder or upsampling high-frequency traces that resampling removes.
  2. There is a train/test preprocessing mismatch.
  3. A Tiny-GenImage size or format shortcut is being broken. Tiny-GenImage inherits GenImage's ImageNet-JPEG-real vs PNG-fake and fixed-size-fake biases.

  Resize256+JPEG96 is the lowest score (90.1), and JPEG96 mimics the ImageNet real-image format. That is consistent with a format shortcut, which is what Fake-or-JPEG would predict. These are hypotheses to test, not established facts.
- Fixes reviewers would recognize:
  - Random-scale resize augmentation (for example 0.5–1.5× with several interpolation kernels) alongside the existing JPEG and blur.
  - Native-resolution multi-crop or tile inference with logit averaging, as in B-Free, SPAI and HiDA-Net.
  - Matching the format of real and fake images before training, as in the Fake-or-JPEG debiasing.
- Report robustness as curves (resize factor 0.5–1.5, JPEG 50–100, combinations) on unbiased data, not single points.

### Gaps
- I did not verify NPR's (CVPR 2024) own resizing analysis. NPR's collapse under social-network processing comes from the B-Free tables above.
- I found no paper that directly measures CLIP-based detectors' accuracy against downscale factor on Tiny-GenImage.
- SPAI's resize robustness numbers appear only in figures, so no exact values are available.

---

## Q3. Benchmarks, protocols, metrics, SOTA numbers, and the controls reviewers expect (2025–2026)

### Takeaway
Top-venue papers now usually:
1. Train on a single generator (GenImage SD1.4, ProGAN for the AIGCDetectBenchmark/CNNSpot lineage, or one LDM) and test across many generators.
2. Add "in the wild" or hard sets: Chameleon, WildRF, SynthWildX, Synthbuster, FakeInversion, the Community Forensics eval set, and the Flux/SD3.5 sets.
3. Report AUC/AP plus balanced accuracy at a fixed threshold, increasingly with calibration (ECE/NLL).
4. Control for format and size bias.

Training on all 7 Tiny-GenImage generators and testing in-distribution will not count as a generalization result.

### Cited Findings
- **GenImage protocol.** Train on SD v1.4 (324k images) and test on 8 generators. AIGCDetectBenchmark trains on ProGAN (360k) and tests on 16 sets. — [AIDE, arXiv 2406.19435](https://arxiv.org/html/2406.19435v3)
- **Current GenImage SD1.4→all mean accuracy (preprints unless noted).**
  - PE-Core-G (frozen) 97.16 — [TAP](https://arxiv.org/html/2604.26772v1)
  - HiDA-Net 97.1 — [arXiv 2508.17346](https://arxiv.org/html/2508.17346)
  - C2P-CLIP 95.8 and SAFE 95.6, as reported in HiDA-Net — [arXiv 2508.17346](https://arxiv.org/html/2508.17346)
  - AIDE (ICLR 2025) 86.88 — [arXiv 2406.19435](https://arxiv.org/html/2406.19435v3)
- **Chameleon** (AIDE, ICLR 2025). 11,170 fake and 14,863 real images at 720P–4K. Fakes were kept only if two annotators judged them real. Accuracy, shown as overall (fake / real):
  - Trained on all of GenImage: AIDE 65.77 (26.80 / 95.06); UnivFD 60.42; NPR 57.81 (1.68 / 100.00).
  - Most detectors are near chance.
  - Later results: HiDA-Net 79.10 and TAP (PE-G + TAP) 95.64, both preprints.

  Reporting fake and real accuracy separately is standard here. — [arXiv 2406.19435](https://arxiv.org/html/2406.19435v3); [arXiv 2508.17346](https://arxiv.org/html/2508.17346); [arXiv 2604.26772](https://arxiv.org/html/2604.26772v1)
- **B-Free evaluation suite.**
  - Synthbuster (RAW-based real images and raw fakes, which avoids format bias).
  - "Unbiased" GenImage, where fakes are JPEG-compressed to match the real images (5k + 5k).
  - FakeInversion (real images found by reverse image search, so they match the fakes in style).
  - WildRF and SynthWildX (social media).
  - FakeBench, plus 1k each of FLUX and SD3.5.

  Metrics: AUC, bAcc at a 0.5 threshold, binary ECE (15 bins) and balanced NLL. B-Free averages 90.4 bAcc on GenImage + FakeInversion + SynthWildX vs 83.9 for DMID and 64.0 for AIDE. — [arXiv 2412.17671](https://arxiv.org/html/2412.17671v2)
- **SPAI protocol.** One LDM for training, 13 generators for testing, real images from 5 sources (RAISE, FODB, ImageNet, COCO, Open Images). AUC is the main metric, with bAcc@0.5 and AP in the supplement. — [arXiv 2411.19417](https://arxiv.org/html/2411.19417v2)
- **Cozzolino et al. protocol.** About 18 generators including commercial ones, plus Synthbuster and images downloaded from X. Metrics: AUC, AP and accuracy at 0.5. Social-network simulation: random crop, resize to 200, JPEG65–100. — [arXiv 2312.00195](https://arxiv.org/html/2312.00195v2)
- **AI-GenBench** (Pellegrini et al.; arXiv 2504.20865, Apr 2025; reported as IJCNN 2025, which I did not verify). It is a temporal or continual protocol: detectors are trained incrementally on generators in chronological order and tested on newer ones. It aims to stop arbitrary splits through standardized rules and controlled augmentation. A secondary source reports about 180k fakes from 36 generators across 9 sliding windows, which I did not verify. — [arXiv 2504.20865](https://arxiv.org/abs/2504.20865); [secondary review](https://www.themoonlight.io/en/review/ai-genbench-a-new-ongoing-benchmark-for-ai-generated-image-detection)
- **Community Forensics.** Its evaluation set covers many open and commercial generators and is a 2025 cross-dataset target; training-set diversity is a reviewer-visible variable. — [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html)
- **OpenSDI.** Megalith-10M real images vs SD1.5 full and inpainted images, tested on SD2.1/XL/3 and Flux, with F1 and Acc as metrics. It is used as a 2026 protocol by TAP. — [arXiv 2604.26772](https://arxiv.org/html/2604.26772v1)
- **Format and size bias control** (Fake or JPEG?, ECCVW 2024). The controls are: JPEG-match fakes to the real images' QF96, restrict sizes, and use identical crop and resize at training and inference. It also shows a detector's accuracy on uncompressed real PNGs can drop sharply, which is a strong diagnostic that the detector learned "is JPEG" rather than "is fake". — [arXiv 2403.17608](https://arxiv.org/html/2403.17608v1)
- **Ablations seen in these papers, which reviewers will expect similar versions of.**
  - Training-set size: 1k vs 10k vs 100 pairs (Cozzolino).
  - CLIP pretraining-data scale (Cozzolino).
  - Layer choice: next-to-last layer (Cozzolino).
  - Aggregation of crops or tiles (SPAI, HiDA).
  - Augmentation removal (SPAI: without JPEG 89.1, without blur/noise 84.2, full model 91.0 AUC).
  - Backbone swap with the same head (TAP).
  - CLS vs patch tokens (TAP).
  - Type of fake-data generation (B-Free: self-conditioned+inpainted 92.2 vs inpainted++ 96.4).

  — sources as cited above.

### Inferences
- **Minimum protocol package for the GSD and GMM/3D lines.**
  1. Leave-one-generator-out on Tiny-GenImage, or SD1.4-only training, tested on the remaining generators.
  2. Cross-dataset: Chameleon, Synthbuster or unbiased GenImage, WildRF/SynthWildX, plus at least one 2024–2025 generator set (Flux/SD3.5, Community Forensics eval, B-Free's sets).
  3. Robustness grid (JPEG, resize factor, blur, crop, and chains that simulate social media).
  4. Metrics: AUC, AP, bAcc@0.5 with real and fake accuracy shown separately, and ECE/NLL when making calibration claims.
- **Controls specific to GSD (claim: semantic subspace suppression helps).**
  - The same CLIP-L backbone with a linear or SVM probe and no GSD (the Cozzolino/UnivFD-style baseline).
  - The same backbone fine-tuned or LoRA-tuned without GSD.
  - A random subspace of the same rank removed.
  - CLS vs patch-token pooling.
  - Backbone swap (SigLIP2, PE, DINOv2).
  - Several seeds with mean ± std or bootstrap 95% CIs. 99.86% vs 99.4% differences are within noise on small test sets.
  - Format-matched evaluation (JPEG-QF96-matched fakes) to rule out a JPEG shortcut.
- **GMM real-only line.** The closest peer-reviewed analog is SPAI: a model of real images only, with any-resolution handling, and supervised calibration on one LDM. Reviewers will compare against SPAI, plus training-free foundation-model baselines. Example: arXiv 2605.04358 (May 2026 preprint) reports +39.61% AUROC over the best training-free method on Community Forensics-Small; I did not verify the details. — [arXiv 2605.04358](https://arxiv.org/abs/2605.04358)
- **"Near-100% native accuracy" will be read as in-distribution.** Pair it with cross-generator and cross-dataset numbers, where SOTA is about 91 AUC (SPAI, 13 generators), about 90–96 bAcc (B-Free suites), and 65–96% on Chameleon depending on the method.

### Gaps
- Details for OpenFake, DRCT-2M, ImagiNet, WildRF statistics and FakeInversion/SynRIS were not fetched (budget). They appear only as names or through B-Free and HiDA-Net. DRCT numbers: DRCT/SDv1.4 averages 90.8 and DRCT/SDv2 96.6 on the DRCT benchmark, as reported in HiDA-Net ([arXiv 2508.17346](https://arxiv.org/html/2508.17346)).
- I did not confirm whether reviewers at specific 2025–2026 venues formally require CIs or seeds. The "controls" list above is inferred from recurring ablations in the cited papers, not from venue guidelines.
- No verified Tiny-GenImage-specific leaderboard was found.
- I did not confirm that B-Free was accepted at CVPR 2025; its arXiv page lists no venue.
