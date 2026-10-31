# One-class / real-only / zero-shot detection of AI-generated images (state of the art to Oct 2026), and where "GSD encoder + GMM on real" fits

Notes compiled 2026-10-27. "PR" = peer-reviewed venue; "preprint" = arXiv only (as far as found). Numbers are quoted with the benchmark or protocol they come from. Numbers from different papers are **not** directly comparable because each uses different real sets and generator lists.

## Q1. Which papers detect generated images using only real images for training, and how do they perform?

### Takeaway
A mature line of real-only detectors now exists (2024–2026). The strongest are residual-based forensic models: FSD (CVPR 2025) and Forensic Twins (arXiv, Sep 2026). On their own wide benchmarks they reach about 96–98.5% AUC, close to the best supervised detectors. Semantic-feature (DINOv2/CLIP) training-free methods such as ConV and RIGID are clearly weaker, at about 77–87% AUC. Reconstruction and entropy methods (AEROBLADE, ZED) fall further behind on third-party benchmarks. Re-compression and post-processing are the main weakness across this whole family.

### Cited Findings
**Forensic Self-Descriptions (FSD). Nguyen, Azizpour, Stamm (Drexel). CVPR 2025 (PR); arXiv 2503.21003, posted Mar 26, 2025**
- Method: K=8 learnable linear predictive filters (11×11 neighbourhood, centre weight fixed at 0, weights sum to 1) are trained on grayscale real images only. The loss is residual energy plus a spectral-diversity regulariser. Each image then has its own per-image filter model fitted across L=3 scales by AdamW (up to 10,000 iterations), and the fitted parameters are its "self-description". A GMM is fit on self-descriptions of real images, and an image counts as real if p(Φ|Real) ≥ τ. — [arXiv html](https://arxiv.org/html/2503.21003v1)
- Real training sets: COCO2017, ImageNet-1k and ImageNet-22k (100k each), plus MIDB (22,329). Test: 24 generators drawn from OSSIA, DMID, Synthbuster and the authors' own generations, at resolutions from 256² to 1792². — [arXiv html](https://arxiv.org/html/2503.21003v1); [CVPR suppl.](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Nguyen_Forensic_Self-Descriptions_Are_CVPR_2025_supplemental.pdf)
- Average AUC (Table 1): FSD 0.960, NPR 0.926, PatchFor 0.823, LGrad 0.820, UFD 0.798, DE-FAKE 0.731, ZED 0.723, CNNDet 0.722, AEROBLADE 0.674. — [arXiv html](https://arxiv.org/html/2503.21003v1)
- Worst-case AUC (Table 2): FSD 0.892–0.903 on COCO17, IN-1k and MIDB, but **0.714 on IN-22k**. NPR drops as low as 0.239 (IN-1k), ZED to 0.331 (MIDB) and UFD to 0.244 (MIDB). — [arXiv html](https://arxiv.org/html/2503.21003v1)
- JPEG: average AUC 0.972 across QF 50–100 (the authors call this minimal degradation). **Resize was not evaluated.** The authors list post-processed and social-media images as future work. — [arXiv html](https://arxiv.org/html/2503.21003v1)
- Ablation (on a subset, full method 0.986): a fixed 3×3 or 5×5 high-pass filter instead of learned filters gives 0.955 / 0.913; no multi-scale gives 0.956; a one-class SVM or Isolation Forest on the self-descriptions gives 0.968. — [arXiv html](https://arxiv.org/html/2503.21003v1)
- Cost: 2,415 ms per image, because the per-image filter fit is an optimisation (as reported by Forensic Twins). — [Forensic Twins](https://arxiv.org/html/2609.31514)

**Forensic Twins. Muñoz-Haro, Tolosana, Vera-Rodriguez, Morales, Fierrez. arXiv 2609.31514, Sep 25, 2026 (preprint)**
- Method: a frozen FSD residual extractor (FRE, 8 residual channels from grayscale) feeds a ViT-Tiny (6 blocks, CLS token, 2D RoPE, patch 8, about 3.0M parameters). It is trained with **Barlow Twins** on **two spatially disjoint crops** whose sizes are resampled each iteration. Flips and rotations are shared by both views, and destructive augmentations are excluded. Projector dimension is 8192. Best λ = 5×10⁻⁵ (base λ 0.005). — [arXiv html](https://arxiv.org/html/2609.31514)
- Real training data: 250k ImageNet-1k train plus 241,690 MS-COCO train images (491,690 in total). Training ran for 80 epochs at batch 320 on 4× RTX 4090. **No AI-generated image is seen at any stage.** — [arXiv html](https://arxiv.org/html/2609.31514); [arXiv abs](https://arxiv.org/abs/2609.31514)
- Detector: a **one-class GMM with K=1 (single full-covariance Gaussian)** on real embeddings, scored by negative log-likelihood. — [arXiv html](https://arxiv.org/html/2609.31514)
- Global AUC over 27 unseen generators (8 GANs, 14 diffusion/AR, 5 commercial): EFFORT (supervised) 98.96, **FSD 98.52**, **Forensic Twins 97.99**, OmniAID (supervised) 97.67, SPAI 94.88, NPR 90.27, UFD 88.01, **ConV 77.13**, CNNDet 74.46, LGrad 65.87. Forensic Twins per family: GAN 98.17, diffusion 98.56, commercial 96.10. Real sets: ImageNet-1k val and COCO-2017 test. No AP is reported. — [arXiv html](https://arxiv.org/html/2609.31514)
- Ablations (global AUC): RGB input 60.95; SRM filters 84.14; FRE residuals 93.77; **overlapping crops 89.49**; DINOv2-style destructive augmentations 57.84; final configuration 97.99. — [arXiv html](https://arxiv.org/html/2609.31514)
- Speed: 6.44 ms per image, against 2,415 ms for FSD (about 375× faster). Source clustering at k=N: 56.61%, against FSD 50.48, CLIP-L/14 51.73 and ConV 43.98. — [arXiv html](https://arxiv.org/html/2609.31514)
- **Robustness: test-time JPEG re-encoding drops AUC from 97.99 to 61–71.** Adding re-encoded reals (QF 50–100) to the GMM fitting pool raises it to only 64–81. At QF 35 (outside the calibrated range) AUC is 58.95 ± 15.46. Midjourney v6 at QF 50 gets 57.71. Few-shot kNN calibration with 10–25 fakes from one generator recovers 81.9–91.6 under re-encoding. Resize robustness is not tested; images are not resized before detection. — [arXiv html](https://arxiv.org/html/2609.31514)
- The authors name normalizing flows, dense ViT features and few-shot calibration as future work. — [arXiv html](https://arxiv.org/html/2609.31514)

**ZED: Zero-Shot Detection of AI-Generated Images. Cozzolino, Poggi, Nießner, Verdoliva. ECCV 2024 (PR, LNCS 15076, pp. 54–72); arXiv 2409.15875**
- A lossless multi-resolution coder trained on real images predicts each pixel's probability from its context. The detection feature is the coding-cost "surprise" of an image relative to the real-image model, and a single discriminative feature is used. The authors report an average accuracy gain of more than 3% over the prior state of the art across generators. — [arXiv abs](https://arxiv.org/abs/2409.15875); [ECCV pdf](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02665.pdf)
- On independent benchmarks: average AUC 0.723 and worst case 0.331 (MIDB) in the FSD protocol. — [FSD](https://arxiv.org/html/2503.21003v1)

**AEROBLADE. Ricker, Lukovnikov, Fischer. CVPR 2024 (PR); arXiv 2401.17879**
- Training-free. It uses the LDM autoencoder reconstruction error, measured with LPIPS (layer 2 works best), because LDM images are reconstructed more accurately than real ones. It targets SD and Midjourney and can also localise inpainted regions. — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/html/Ricker_AEROBLADE_Training-Free_Detection_of_Latent_Diffusion_Images_Using_Autoencoder_Reconstruction_CVPR_2024_paper.html); [arXiv](https://arxiv.org/pdf/2401.17879)
- Weak outside LDMs: average AUC 0.674 in the FSD protocol; 57.87 AUROC / 57.85 AP on ConV's ImageNet benchmark; 56.46 / 60.19 on Sora/OpenSora. — [FSD](https://arxiv.org/html/2503.21003v1); [ConV](https://arxiv.org/html/2511.01293v1)

**RIGID. He, Chen, Ho. arXiv 2405.20112 (2024; I could not confirm a venue)**
- Training-free. It uses the cosine similarity between foundation-model (DINOv2) embeddings of an image and a slightly noised copy, since real images are more robust to small noise in representation space. The authors claim it beats the best prior training-free method by more than 25% on average. — [arXiv](https://arxiv.org/pdf/2405.20112)
- A follow-up (arXiv 2411.19117) finds that performance tracks the model's robustness (self-supervised models are better), and that Gaussian noise works for general objects but blur works better for faces. — [arXiv 2411.19117](https://arxiv.org/html/2411.19117v1)

**ConV / F-ConV: "Detecting Generated Images by Fitting Natural Image Distributions". arXiv 2511.01293, Nov 3, 2025 (preprint, NeurIPS-format checklist; I could not confirm a venue)**
- ConV is training-free. It uses DINOv2 ViT-L/14 as f₁ and f₂ = f₁∘h, where h is a set of augmentations (flip, colour jitter, blur, n=20). A large change in the self-supervised loss under transformations along the data manifold flags an image as fake. F-ConV trains a RealNVP normalizing flow (2 coupling blocks) on DINOv2 features to map natural images to a Gaussian. — [arXiv html](https://arxiv.org/html/2511.01293v1); [MBZUAI news](https://mbzuai.ac.ae/news/new-synthetic-image-detector-focuses-on-what-makes-real-images-real/)
- ImageNet benchmark, average AUROC/AP: ConV 87.13/85.15; F-ConV 93.77/93.38; FatFormer 93.68/93.11; DRCT 90.36/89.92; NPR 86.00/80.84; Ojha (UFD) 85.35/84.25; AEROBLADE 57.87/57.85; DIRE 52.70/52.18. The paper also evaluates LSUN-Bedroom, GenImage, DRCT-2M and Sora/OpenSora frames (with LAION as reals). — [arXiv html](https://arxiv.org/html/2511.01293v1)
- On Forensic Twins' 27-generator protocol ConV gets only 77.13 AUC. Training-free methods are said to be more robust to JPEG, blur and noise than NPR, but exact values appear only in figures. — [Forensic Twins](https://arxiv.org/html/2609.31514); [ConV](https://arxiv.org/html/2511.01293v1)

**Manifold Induced Biases. Brokman et al. ICLR 2025 (PR); arXiv 2504.15470**
- Zero-shot. It uses score-function estimates (curvature, gradient, bias) of the implicit manifold of a pre-trained diffusion model, with a mixture-of-experts few-shot extension. The authors claim to beat prior zero-shot and few-shot methods across 20 generators. — [arXiv](https://www.arxiv.org/pdf/2504.15470); [lab page](https://guygilboa.net.technion.ac.il/?p=1280)

**SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning. Karageorgiou et al. CVPR 2025 (PR); arXiv 2411.19417**
- It learns the spectral distribution of real images with masked frequency reconstruction (self-supervised), scores generated images as out-of-distribution through "spectral reconstruction similarity", and uses spectral context attention to handle any resolution. The authors report +5.5% absolute AUC over the prior state of the art across 13 generators and robustness to online perturbations. — [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html); [arXiv](https://arxiv.org/html/2411.19417v1)
- Forensic Twins classifies SPAI as **not** real-only and reports 94.88 AUC for it on the 27-generator protocol. — [Forensic Twins](https://arxiv.org/html/2609.31514)

**Beyond Generation: diffusion-based low-level feature extractor. Zhong et al. CVPR 2025 (PR)**
- A diffusion model is used as a denoiser rather than a generator to extract low-level features. The paper frames detection as **one-class classification**: it estimates the distribution of real-photo low-level features and treats deviations as AI-generated. It covers more than 20 generators, including GenImage and DRCT-2M. — [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Zhong_Beyond_Generation_A_Diffusion-based_Low-level_Feature_Extractor_for_Detecting_AI-generated_CVPR_2025_paper.html)

**SDAIE: Self-Supervised AI-Generated Image Detection, a camera-metadata perspective. Zhong, Zou, Xu, Qian, Zhang, Wu, Ma. arXiv 2512.05651 (v2 revised Feb 27, 2026; preprint)**
- The pretext task predicts 7 categorical EXIF tags and ranks 7 ordinal ones, trained only on about 0.93M YFCC100M photos. Input is **16 scrambled 64×64 patches with no position information, each passed through 30 high-pass SRM filters**, explicitly "to suppress semantics". The encoder is CNN + covariance pooling + Transformer, giving a 528-dim feature. — [arXiv html](https://arxiv.org/html/2512.05651v2)
- One-class detector: **GMM with K=5** fit by EM on 10k ImageNet+LSUN photos. The threshold is the 2nd percentile of training log-likelihoods (training false alarms under 2%). No K ablation is reported. — [arXiv html](https://arxiv.org/html/2512.05651v2)
- Accuracy/mAP over 17 generators: one-class SDAIE gets GAN 71.7/83.8 and diffusion 89.3/96.2. The supervised SDAIE† (adds ProGAN fakes) gets 93.7/98.1 and 94.8/99.2. NPR gets 89.7/97.0 and 94.3/99.3; UnivFD 91.9/98.8 and 63.4/81.6. — [arXiv html](https://arxiv.org/html/2512.05651v2)
- Stated limitations: high-frequency traces are fragile under post-processing; the one-class setting flags graphics, paintings and **edited images** as anomalous. — [arXiv html](https://arxiv.org/html/2512.05651v2)

### Inferences
- The best real-only results come from **pixel-residual inputs** (FSD filters, SRM or FRE), not from semantic backbones. In the Forensic Twins ablation, RGB input gives 60.95 AUC versus 93.77 for FRE residuals. That is direct evidence that a CLIP or RGB encoder trained with Barlow Twins will not work without a strong anti-semantic front-end.
- A plausible reason the user's earlier spectral-feature GMM (macro AUC ≈0.77) was weak is that hand-crafted spectral statistics are low-capacity. That is about the level of ConV and Forensic Twins' SRM variant (84).

### Gaps
- I could not retrieve exact ZED or RIGID numbers on a common benchmark beyond the FSD table (ZED 0.723). RIGID is absent from the ConV and Forensic Twins tables.
- I could not access the Beyond Generation full text (403), so its GMM details, K and numbers are unverified.
- Whether SPAI's final detector uses fakes: Forensic Twins says it is not real-only, but I did not verify this in the SPAI paper itself.

## Q2. What density estimators are used (Gaussian, GMM, flows, kNN, Mahalanobis), on which embeddings, and is there evidence on GMM K>1 vs single Gaussian vs kNN?

### Takeaway
GMM on real embeddings is the de facto choice: K=1 full-covariance in Forensic Twins, K=5 in SDAIE, and an unspecified K in FSD. Normalizing flows appear in F-ConV. **None of the papers I found reports a systematic ablation of K=1 vs K>1 vs kNN vs flow** for this task. The only comparison is FSD's: GMM 0.986 versus One-Class SVM and Isolation Forest 0.968 on a subset.

### Cited Findings
- Forensic Twins: single Gaussian (GMM K=1, full covariance) on the ViT-Tiny CLS embedding (exact dimension not confirmed) of FRE residuals, scored by NLL. The visible appendices contain no ablation over K. kNN (cosine, 1-NN or 5-NN) is used only for few-shot calibration and source probes. — [arXiv html](https://arxiv.org/html/2609.31514)
- FSD: GMM on per-image filter parameters (self-description). One-Class SVM or Isolation Forest on the same features gives 0.968 against 0.986 for the full method on the subset. — [arXiv html](https://arxiv.org/html/2503.21003v1)
- SDAIE: GMM K=5 on a 528-dim EXIF-pretext feature; threshold at the 2nd percentile of training likelihood. — [arXiv html](https://arxiv.org/html/2512.05651v2)
- F-ConV: a RealNVP flow (2 coupling blocks) on DINOv2 features. It raises ImageNet AUROC from 87.13 (training-free ConV) to 93.77. — [arXiv html](https://arxiv.org/html/2511.01293v1)
- ZED uses an explicit per-pixel likelihood (lossless coder), compared against the model's expected entropy. — [arXiv abs](https://arxiv.org/abs/2409.15875)
- Generic one-class tooling (CLIP, SigLIP or DINOv2 features with kNN or Mahalanobis) exists, but not as published AIGI detectors. — [imgad GitHub](https://github.com/joheras/imgad)

### Inferences
- With a good self-supervised forensic embedding, even a **single Gaussian** reaches about 98 AUC (Forensic Twins). That suggests representation quality matters far more than the density model. The user should report K=1 and K>1 (selected by BIC) plus kNN-to-real as an ablation, because the literature leaves this open.
- K>1 may model real sub-populations (camera/codec/JPEG-QF clusters). That bears directly on the JPEG failure Forensic Twins reports, where adding re-encoded reals to a single Gaussian helped only partly. This is a hypothesis, not a published finding.

### Gaps
- I found no paper that compares GMM K vs kNN vs flow vs Mahalanobis on the same forensic embedding for AIGI detection.

## Q3. Known failure modes: false positives on OOD real images, compression and resize, likelihood paradoxes, semantic dominance

### Takeaway
The dominant documented failure is **post-processing and compression shift of the real distribution**. In Forensic Twins, JPEG re-encoding cuts AUC from about 98 to 61–71, and calibration does not extrapolate. **Real-domain shift** is the second: FSD's worst-case AUC is 0.714 when reals come from IN-22k. The third is **semantic-feature backbones** being far weaker. Edited and inpainted reals and non-photo reals are flagged as anomalous by construction.

### Cited Findings
- Forensic Twins: JPEG re-encoding gives 61–71 AUC; with re-encoded reals in the fitting pool, 64–81; at QF 35, 58.95 ± 15.46; Midjourney v6 at QF 50, 57.71. "One-class modeling cannot extrapolate beyond the calibrated compression range." — [arXiv html](https://arxiv.org/html/2609.31514)
- FSD: worst-case AUC depends on the real source (0.714 with IN-22k reals against about 0.90 for others). It does not handle social-media or post-processed images. — [arXiv html](https://arxiv.org/html/2503.21003v1)
- SDAIE: the one-class model treats graphics, paintings and edited images as anomalous, and high-frequency traces are fragile under post-processing. JPEG and Gaussian blur are the most damaging perturbations; downsampling is less damaging. — [arXiv html](https://arxiv.org/html/2512.05651v2)
- Semantic dominance: in Forensic Twins, RGB input gives 60.95 versus 93.77 for residuals, and overlapping crops (which share content) give 89.49 versus 97.99 for disjoint crops. — [arXiv html](https://arxiv.org/html/2609.31514)
- Dataset biases (format, JPEG, resolution, content) inflate benchmark scores. B-Free (CVPR 2025) builds semantically aligned fakes from reals with SD conditioning, including inpainting-based content augmentation, to remove those biases. Results span 27 generators, including FLUX and SD3.5. — [B-Free arXiv](https://arxiv.org/pdf/2412.17671); [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.pdf)
- GenImage has a known compression bias (a 2026 paper applies uniform JPEG QF 96 to counter it). AIGIBench (NeurIPS 2025 D&B) reports large drops for detectors on real-world social-media data. — [arXiv 2605.24906](https://arxiv.org/pdf/2605.24906); [AIGIBench NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fb693c67f61e5321746ffce8b6fdd2d0-Abstract-Datasets_and_Benchmarks_Track.html)

### Inferences
- **For the user's B-Free origBG problem**: a global image-level density on a CLS or mean-pooled embedding will probably also miss **local inpainting with the real background pasted back**. Most patches really are real, so the global embedding stays near the real mode. A patch-level score (max or top-k patch NLL, as in anomaly-detection practice) would be needed. No paper I found evaluates one-class detectors on B-Free-style local inpainting. That makes it a gap and an opportunity.
- **Tiny-GenImage caveat**: GenImage reals are ImageNet JPEGs, while many fakes are PNGs. A real-only density can therefore learn "has ImageNet JPEG history", which would give inflated in-distribution AUC and collapse under re-encoding, just as in Forensic Twins. The user should fit on reals with randomised JPEG/resize and test with the fakes JPEG-matched.
- The user's earlier result of 63% real recall at the operating point fits this pattern: one-class thresholds are calibrated on one real distribution and give high false positives on others.

### Gaps
- I found no explicit discussion of the "likelihood paradox" (higher likelihood on simpler or OOD inputs, as in Nalisnick et al.) specifically for AIGI one-class detectors.
- No paper I found quantifies the false-positive rate on real social-media photos from new cameras for these one-class detectors. 2602.07814 and 2608.01258 may contain this, but I did not read them.
- Forensic Twins Appendix E (OOD reals) was not visible in what I retrieved.

## Q4. How do one-class methods compare head-to-head with supervised detectors on unseen generators?

### Takeaway
On their authors' own wide benchmarks, the best real-only methods now sit within about 1 AUC point of the best supervised detectors (EFFORT 98.96, FSD 98.52, Forensic Twins 97.99). They beat older supervised baselines (NPR, UFD, CNNDet, LGrad) and, in the FSD table, are much more stable in the worst case. Weaker one-class variants (SDAIE one-class, ConV) still lag supervised training on some families. I found no head-to-head numbers against AIDE, C2P-CLIP or B-Free.

### Cited Findings
- Forensic Twins 27-generator AUC: EFFORT 98.96 > FSD 98.52 > FT 97.99 > OmniAID 97.67 > SPAI 94.88 > NPR 90.27 > UFD 88.01 > ConV 77.13. — [arXiv html](https://arxiv.org/html/2609.31514)
- FSD 24-generator average AUC: FSD 0.960 vs NPR 0.926 vs UFD 0.798. Worst case: FSD about 0.71–0.90, NPR down to 0.239, UFD down to 0.244. — [arXiv html](https://arxiv.org/html/2503.21003v1)
- ConV ImageNet: F-ConV 93.77 ≈ FatFormer 93.68 > DRCT 90.36 > ConV 87.13 > NPR 86.00 > UFD 85.35. — [arXiv html](https://arxiv.org/html/2511.01293v1)
- SDAIE one-class gets GAN acc 71.7 against NPR 89.7. On diffusion it gets 89.3 against UnivFD 63.4 (UnivFD trained on ProGAN). — [arXiv html](https://arxiv.org/html/2512.05651v2)

### Inferences
- The best evaluation would be supervised GSD and one-class GSD on the **same** splits (Tiny-GenImage cross-generator, B-Free origBG, Synthbuster or Chameleon), at matched JPEG/resize conditions. Published comparisons are self-reported, and each uses different real sets.

### Gaps
- No published one-class vs AIDE, C2P-CLIP or B-Free comparison found. GenImage-specific numbers for FSD and Forensic Twins were not found.

## Q5. Is there published work combining semantic suppression / semantic-subspace removal with one-class density? (Novelty assessment)

### Takeaway
**Semantic suppression for real-only detection is not new as a principle.** Forensic Twins uses residual inputs plus disjoint crops "so the shared signal is the acquisition fingerprint rather than scene content". SDAIE uses scrambled, positionless high-pass patches "to suppress semantics". **Learned semantic-subspace projection (GSD-style SVD removal against a frozen CLIP anchor) inside a real-only, density-scored foundation-model encoder appears unpublished** as of Oct 2026, based on my searches. The novelty claim should be narrow and specific.

### Cited Findings
- GSD (arXiv 2603.09242, Mar 10, 2026; preprint): "When Detectors Forget Forensics: Blocking Semantic Shortcuts for Generalizable AI-Generated Image Detection", by Chao Shuai et al. It identifies "semantic fallback" in CLIP detectors. A parameter-free module uses a frozen VFM as a semantic guide, estimates semantic directions from batch statistics and projects them out. Reported results: 94.4% video-level AUC cross-dataset, +3.0% on DF40, +0.9% on UFD and +1.7% on GenImage. It is supervised. — [arXiv](https://arxiv.org/html/2603.09242)
- DGS-Net (arXiv 2511.13108) also suppresses CLIP semantic components with gradient surgery, but in supervised fine-tuning. — [arXiv](https://arxiv.org/pdf/2511.13108)
- Forensic Twins: content is suppressed through the input (FRE residuals) and the objective (disjoint crops), not through subspace projection. The backbone is a 3M-parameter ViT-Tiny trained from scratch, not CLIP. — [arXiv html](https://arxiv.org/html/2609.31514)
- SDAIE: semantic suppression via scrambled positionless SRM-filtered patches, with a one-class GMM (K=5). — [arXiv html](https://arxiv.org/html/2512.05651v2)
- MPFT (arXiv 2601.03586, Jan 2026): fine-tunes CLIP-ViT with texture-aware masking toward "distributional deviations from real images". The abstract does not say whether fakes are used. It reports up to 98.2% accuracy on GenImage and 94.6% on UFD. — [arXiv](https://arxiv.org/abs/2601.03586)
- Real-only methods built on foundation features (ConV/RIGID with DINOv2, F-ConV flow) do not remove semantics, and they rank lowest among real-only methods (ConV 77.13 on the Forensic Twins protocol). — [Forensic Twins](https://arxiv.org/html/2609.31514); [ConV](https://arxiv.org/html/2511.01293v1)

### Inferences
- A defensible positioning: "Do large VFMs with explicit semantic-subspace removal become competitive real-only forensic encoders?" The answer is uncertain. Evidence (RGB 60.95 in Forensic Twins, ConV 77) suggests that CLIP-L patch tokens, even after semantic removal, may keep too little low-level signal. GSD removes semantic directions in blocks 21–24, but CLIP's low-level information is mostly lost by then. A risk-reducing design would feed **residuals** (FRE or SRM) or use **early or intermediate CLIP layers**, add GSD-style projection, and train with Barlow Twins on disjoint crops. Each piece should then be compared by ablation against Forensic Twins.
- A strong baseline is mandatory: Forensic Twins (97.99 AUC, 3M parameters, real-only, GMM K=1) already is "Barlow Twins on disjoint crops + Gaussian on reals". Unless "GSD + CLIP" beats it, or adds robustness (JPEG/resize) or local-edit detection (origBG), the contribution reduces to swapping the backbone.

### Gaps
- My search may have missed workshop papers or late-2026 preprints that combine subspace removal with one-class scoring. A targeted check on Google Scholar and OpenReview (ICLR 2027 submissions) is advised.

## Q6. Training objectives for real-only forensic encoders and their evidence

### Takeaway
The evidence-backed objectives are: (a) **predictive residual filters** (FSD, linear prediction with diversity regularisation); (b) **Barlow Twins on disjoint residual crops** (Forensic Twins); (c) **lossless-coding likelihood** (ZED); (d) **masked spectral reconstruction** (SPAI); (e) **camera-metadata (EXIF) prediction** (SDAIE); (f) frozen self-supervised features with consistency or noise checks (RIGID, ConV) or a flow (F-ConV). I found no published evidence for VICReg or SimCLR-style contrastive learning in this specific real-only AIGI setting.

### Cited Findings
- Barlow Twins with disjoint crops plus FRE: 97.99 global AUC. Overlapping crops give 89.49; destructive augmentations give 57.84. λ tuning matters (5×10⁻⁵ is best at patch 8). — [Forensic Twins](https://arxiv.org/html/2609.31514)
- Predictive filters (residual energy plus spectral diversity) beat fixed high-pass filters: 0.986 against 0.955 (3×3) and 0.913 (5×5). — [FSD](https://arxiv.org/html/2503.21003v1)
- EXIF prediction on scrambled SRM patches: one-class GAN acc 71.7, diffusion 89.3 (mAP 83.8 / 96.2). — [SDAIE](https://arxiv.org/html/2512.05651v2)
- Self-supervised versus other backbones for training-free detection: robustness of the representation correlates with detection, and SSL models are more reliable. — [arXiv 2411.19117](https://arxiv.org/html/2411.19117v1)
- Masked spectral reconstruction (SPAI): +5.5% AUC over the prior state of the art on 13 generators (protocol as reported by its authors). — [SPAI CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html)
- Barlow Twins original (Zbontar et al., 2021): redundancy reduction through a cross-correlation objective. — [arXiv 2103.03230](https://arxiv.org/abs/2103.03230)

### Inferences
- Practical recipe implied by the evidence: residual front-end → small or medium ViT (or a GSD-filtered CLIP) → Barlow Twins on two disjoint crops with only geometric augmentations shared by both views → GMM fit on reals that include JPEG/resize-augmented copies. Evaluate patch-level max-NLL for local inpainting (B-Free origBG).
- Robustness trade-off: including JPEG/resize in the views as a nuisance to become invariant to may help robustness. However, Forensic Twins shows destructive augmentations collapse AUC to 57.84, so augmentations must be mild and shared, or applied only when fitting the GMM.

### Gaps
- No direct evidence found for VICReg, SimCLR or MAE objectives on real-only AIGI encoders. No evidence on whether initialising from CLIP or DINOv2 weights helps a residual-input encoder.
