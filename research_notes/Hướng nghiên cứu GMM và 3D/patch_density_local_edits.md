# Patch-level density / anomaly scoring and semantic suppression for locally edited (inpainted, origBG) AI images

Research date: 2026-10-27. Source status: "peer-reviewed" means accepted at a conference or workshop as stated by the source. "Preprint" means arXiv only. Items marked **(not re-verified this session)** come from prior knowledge of the paper and were not confirmed against the text during this research. The report writer should treat them as lower confidence.

Naming note: arXiv 2603.09242 calls its method **Geometric Semantic Decoupling (GSD)**, not "semantic subspace suppression" — [arXiv 2603.09242](https://arxiv.org/abs/2603.09242)

---

## Q1. Industrial anomaly detection (PatchCore, PaDiM, SPADE, FastFlow): how they score images from patches, and whether they have been used for image forensics or AIGI localization

### Takeaway
PatchCore-style methods fit a memory bank of patch features from normal images only. They score each test patch by its distance to its nearest neighbour in the bank, which gives a localization map for free, and flag an image if at least one patch is anomalous. I found **no published work that applies PatchCore, PaDiM, SPADE or FastFlow to AI-generated or inpainting localization**. The closest real-only forensic analogues are Noiseprint++ (inside TruFor), ZED and Forensic Self-Descriptions. These model low-level residual statistics, not semantic ViT tokens.

### Cited Findings
- PatchCore (Roth et al., **CVPR 2022**, peer-reviewed):
  - Trains on nominal (normal) images only, using a "maximally representative memory bank of nominal patch-features."
  - Reports image-level AUROC up to **99.6% on MVTec AD**, "more than halving the error" of the next best method, with "competitive inference times."
  - Sources: [arXiv 2106.08265](https://arxiv.org/abs/2106.08265); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)
- PatchCore mechanics:
  - Localization comes from per-patch scores, and an image counts as anomalous if at least one patch is anomalous.
  - A coreset step thins the memory bank so nearest-neighbour search stays feasible.
  - The backbone is a frozen pretrained network with no fine-tuning.
  - Known weakness: outlier patches can enter the bank in high-shot settings and inflate score variance.
  - Sources: [hyper.ai summary of 2106.08265](https://hyper.ai/en/papers/2106.08265); [EmergentMind PatchCore topic](https://www.emergentmind.com/topics/patchcore) (secondary source)
- PatchCore details from the paper **(not re-verified this session)**:
  - Image score = max over patches of the nearest-neighbour distance, with a reweighting term.
  - Features come from WideResNet-50 layers 2+3, aggregated over a 3x3 local neighbourhood ("locally aware patch features").
  - Greedy k-center coreset at about 1–25% of the bank; 1% and 10% are the common settings.
  - Source: [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)
- PaDiM detects and localizes anomalies at once in a one-class setting by producing an anomaly map. It is used, for example, in an industrial defect-localization patent. — [USPTO patent text](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12725047)
- PaDiM's per-position multivariate Gaussian with Mahalanobis scoring **(not re-verified this session)** is effectively a per-location GMM with K=1. That makes it the most direct analogue of the user's "GMM on patch tokens" idea. Caveat: PaDiM assumes spatially aligned objects, which natural photos are not.
- A targeted search for PatchCore or memory-bank patch kNN in deepfake or forgery detection turned up no such paper. Deepfake localization work instead uses supervised patch-level labels:
  - Knowledge-injection patch labels: [arXiv 2503.02503](https://arxiv.org/pdf/2503.02503)
  - Class-token CAM on sequential patches: [arXiv 2207.02803](https://arxiv.org/pdf/2207.02803)
  - DeepShield, which builds patch labels from blended masks: [arXiv 2510.25237](https://arxiv.org/pdf/2510.25237)
- Benchmark and library context for unsupervised anomaly localization: [Benchmarking Unsupervised AD and Localization, arXiv 2205.14852](https://arxiv.org/pdf/2205.14852); [MoViAD library, arXiv 2507.12049](https://arxiv.org/pdf/2507.12049)

### Inferences
- Applying PatchCore or a patch-level GMM to forensic localization looks **novel, or at least not established in the literature**. That is an opportunity, but nothing in the literature shows it works.
- The main risk is **distribution mismatch**. Industrial AD works because "normal" is a narrow visual class, such as one product type. On COCO-like real photos, the normal manifold of CLIP patch tokens is huge and mostly semantic. Without semantic suppression, a rare-but-real object (an unusual animal, text, a logo) may be farther from the memory bank than an inpainted ordinary object. This is the core argument for doing GSD or SVD projection *before* the density step (see Q4).
- Max-pooling over patches gives one-anomalous-patch sensitivity, which is exactly what origBG needs (Q2 shows small masks are near chance for global detectors). It also raises false positives on real images. Top-k mean pooling, for example the top 1–5% of patches, is a common compromise.
- Fit the density on real images processed through the **same pipeline** (resize, JPEG). Otherwise patch-level density will pick up format and resize artifacts. B-Free's whole thesis is that format and content biases dominate.

### Gaps
- I found no paper reporting PatchCore, PaDiM or FastFlow AUROC on TGIF, BR-Gen, SID-Set, CocoGlide or B-Free's origBG.
- FastFlow and SPADE were not verified this session. They are known as normalizing-flow and kNN-on-feature-pyramid methods respectively, but are not cited here with sources.

---

## Q2. State of the art for detecting and localizing partially synthetic images (inpainting, spliced with the real background)

### Takeaway
The literature confirms the user's observation as a **systematic failure mode**:
- Global synthetic-image detectors (SID) catch fully regenerated inpainting but fail when only a small region is synthetic.
- Classical forgery localizers (IFL) such as TruFor and CAT-Net catch spliced inpainting (original background pasted back) but fail on fully regenerated images.

The current best inpainting localizers are supervised, patch-level ViT models (DinoLizer on DINOv2 at 14x14 patch resolution, and VLM or LMM methods). They are trained to treat the VAE-reconstructed region outside the mask as its own class.

### Cited Findings
- **B-Free** (Guillaro et al., arXiv 2412.17671, v1 Dec 2024, v2 Apr 2025; the user cites it as **CVPR 2025**, which the arXiv page does not state):
  - Fakes are generated from real images by Stable Diffusion self-conditioning, plus content augmentation by inpainting. An object is replaced with another of the same COCO category, and a version with the **original background restored** (only the object generated) is also made.
  - "inpainted+" adds same-supercategory objects, and "inpainted++" adds scaling, cut-out, noise and jitter.
  - Each real image yields six fakes.
  - Source: [arXiv html 2412.17671](https://arxiv.org/html/2412.17671)
- B-Free Table 2 (average AUC / bAcc over Synthbuster, new generators and WildRF):

  | Training setting | AUC / bAcc |
  |---|---|
  | Reconstructed (VAE only) | 94.6 / 80.7 |
  | Reconstructed + inpainted | 95.5 / 83.9 |
  | Self-conditioned only | 94.7 / 81.4 |
  | Self-conditioned + cutmix/mixup | 95.3 / 78.6 |
  | Self-conditioned + inpainted | **98.0 / 92.2** |
  | + inpainted+ | 99.0 / 95.2 |
  | + inpainted++ | 99.3 / 96.4 |

  - Calibration improves the same way: NLL goes from 0.53 to 0.10 and ECE from 0.158 to 0.038.
  - **The large gain comes from adding inpainted (including background-restored) fakes, not from self-conditioning alone.**
  - Source: [arXiv html 2412.17671](https://arxiv.org/html/2412.17671)
- B-Free does **not** do pixel-level localization and does not report a separate locally-edited test set. Fig. 5 argues that self-conditioned generation (the full diffusion pass) adds artifacts even at the lowest frequencies, which plain autoencoder reconstruction does not. — [arXiv html 2412.17671](https://arxiv.org/html/2412.17671)
- B-Free backbone: DINOv2 with 4 registers, fine-tuned end-to-end on 504x504 crops without resizing.

  | Backbone / training | AUC / bAcc |
  |---|---|
  | DINOv2+reg, end-to-end | 99.0 / 95.2 |
  | DINOv2+reg, linear probe | **80.8 / 68.5** |
  | DINOv2, end-to-end | 98.4 / 91.1 |
  | SigLIP, end-to-end | 95.4 / 89.9 |

  - The authors credit registers with preserving local patch information.
  - **A frozen foundation model with a linear head is far worse than end-to-end training.**
  - Source: [arXiv html 2412.17671](https://arxiv.org/html/2412.17671)
- **TGIF** (Mareen et al., **IEEE WIFS 2024**, peer-reviewed):
  - About 75k inpainted images from SD2, SDXL and Adobe Firefly, in both spliced and fully regenerated variants.
  - Traditional IFL methods detect **spliced** inpainting but **fail on fully regenerated** inpainting.
  - SID methods may flag regenerated images as fake but **cannot localize** the edit.
  - Both families fail under strong compression and WEBP.
  - Source: [arXiv 2407.11566](https://arxiv.org/abs/2407.11566)
- **TGIF2** (arXiv 2603.28613, Mar 2026; a journal version is hosted on the imec repository, venue not confirmed):
  - Adds FLUX.1 edits and random non-semantic masks.
  - Both IFL and SID methods degrade on FLUX.1.
  - Fine-tuning improves localization on fully regenerated images, but random non-semantic masks reveal **object bias**: localizers learn "find the object," not "find the synthetic pixels."
  - Generative super-resolution significantly weakens forensic traces.
  - Sources: [arXiv 2603.28613](https://arxiv.org/abs/2603.28613); [imec publication](https://imec-publications.be/entities/publication/912a78bc-ba33-4405-a19a-e51040ee95fd)
- **"Detecting Localized Deepfakes: How Well Do Synthetic Image Detectors Handle Inpainting?"** (Pandolfini, Pellegrini, Ferrara, Maltoni, Univ. Bologna; preprint arXiv 2512.16688, Dec 2025). Source for all items: [arXiv html 2512.16688](https://arxiv.org/html/2512.16688)
  - Protocol:
    - Models are trained on AI-GenBench as binary real/fake, with no inpainting data and no localization supervision.
    - Images are resized and classified in a single forward pass. Metric: AUROC.
    - Test sets: BR-Gen, TGIF and TGIF2, with 337k inpainted and 30.6k real images.
  - AUROC by detector:

    | Detector | BR-Gen | TGIF | TGIF2 |
    |---|---|---|---|
    | ResNet-50 CLIP | 0.64 | 0.54 | 0.53 |
    | DINOv2 ViT-L/14 | 0.83 | 0.66 | 0.52 |
    | DINOv3 ViT-L/16 | 0.86 | 0.69 | 0.52 |

    The best prior baseline on BR-Gen (LGrad) scored 0.64.
  - TGIF masked inpainting vs. fully regenerated (AUROC): DINOv3 0.71 vs. 0.60, DINOv2 0.70 vs. 0.60.
  - **Mask-size effect:** near chance for masks **<5% of the image**; performance improves above 20%. DINOv3 reaches full-generation performance at about 65% mask coverage, and DINOv2 needs more than 80%.
  - Generators: Firefly and Flux inpainting score AUROC about 0.41–0.46 (worse than chance), while LaMa, BrushNet, PowerPaint and MAT score about 0.84–0.90.
- **DinoLizer** (Doi, Itier, Butora, Boulanger, Bas; preprint arXiv 2511.20722, v1 Nov 2025, v2 Jul 2026):
  - DINOv2 with LoRA on the Q and V projections, plus one linear layer predicting manipulation at **14x14 patch resolution**, with sliding windows for large images.
  - Key design: **reconstructed (VAE-re-encoded) regions outside the inpainting mask are treated as a separate class**, which the authors report beats the conventional binary mask.
  - Reports about 20% higher IoU than the second-best model on average, and robustness to double JPEG.
  - Source: [arXiv 2511.20722](https://arxiv.org/abs/2511.20722)
- DinoLizer's table (as summarized by search, not read directly), SIDA dataset column, IoU / F1:

  | Method | IoU / F1 |
  |---|---|
  | DinoLizer | ~0.60 / 0.69 |
  | TruFor | 0.18 / 0.25 |
  | CAT-Net v2 | 0.08 / 0.13 |
  | ManTra-Net | 0.03 / 0.05 |

  Source: [arXiv pdf 2511.20722](https://arxiv.org/pdf/2511.20722) (numbers via search snippet; verify against the table)
- **SIDA** (Huang et al., **CVPR 2025**, peer-reviewed):
  - SID-Set has 300K images: 100K real, 100K fully synthetic and 100K tampered.
  - An LMM classifies the image, predicts the tampered mask and explains its judgment.
  - Because FakeShield and some other baselines had no released code, SIDA compared localization against PSCC-Net and LISA.
  - Sources: [CVPR 2025 open access](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_SIDA_Social_Media_Image_Deepfake_Detection_Localization_and_Explanation_with_CVPR_2025_paper.html); [SIDA supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Huang_SIDA_Social_Media_CVPR_2025_supplemental.pdf)
- A 2026 VLM-based method reports **0.65 IoU / 0.87 F1 on SID-Set**, against a strongest prior baseline of about 0.44 IoU. — [arXiv 2603.12930](https://arxiv.org/pdf/2603.12930) (via search snippet)
- A 2026 benchmark shows a fine-tuned SIDA missing most tampered regions on a new VLM-tampering dataset, attributed to mask-based supervision. — [arXiv 2603.20193](https://arxiv.org/pdf/2603.20193) (via search snippet)
- **TruFor** (Guillaro et al., **CVPR 2023**, peer-reviewed):
  - Fuses RGB with **Noiseprint++**, a learned noise fingerprint trained **self-supervised on pristine images only**, so forgeries show up as deviations from an image's normal pattern.
  - Outputs a localization map, an image integrity score and a **reliability map** that reduces false alarms.
  - MiT-B2 backbone; no positional encodings, so it handles arbitrary input sizes.
  - Sources: [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.pdf); [supplement](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Guillaro_TruFor_Leveraging_All-Round_CVPR_2023_supplemental.pdf); [GitHub](https://github.com/grip-unina/TruFor)
- InpDiffusion (inpainting localization via conditional diffusion; arXiv 2501.02816) exists but was not read. — [arXiv 2501.02816](https://arxiv.org/pdf/2501.02816)

### Inferences
- The user's pattern matches the literature closely:
  - Fully regenerated: recall 88–98%. This fits TGIF's finding that SID detects regenerated images.
  - origBG: recall 13–52%. This fits TGIF's finding that SID fails on spliced images and Pandolfini et al.'s finding that masks under 5–20% are near chance.
  - The likely mechanism: when the background is pasted back, the global VAE/diffusion trace survives only inside the mask region. A **patch-mean embedding dilutes that region by roughly the mask area fraction.**
- B-Free shows that **training-side exposure to background-restored inpainted fakes** is the single biggest lever for a supervised detector, reaching 98.0 / 92.2 AUC/bAcc vs. 94.7 / 81.4 for self-conditioning alone. The user's supervised GSD detector would likely benefit from the same augmentation, independently of the GMM plan.
- DinoLizer's "VAE-reconstructed area outside the mask" class tells you what to expect from origBG images. Inside the mask: VAE plus diffusion traces. Outside the mask: clean camera traces, because the real pixels were pasted back. A real-only patch density model should therefore see the origBG mask as a **local island of off-manifold patches**. This is the best-case scenario for max/top-k pooling.
- TGIF2's object-bias finding is a warning for any semantic-feature localizer, including a CLIP-patch GMM: it may highlight salient objects rather than synthetic pixels. Evaluate on random non-semantic masks to check.
- Classical one-class or IFL cues (Noiseprint++, TruFor) are complementary for the origBG case, because there the real background provides the reference camera signature. A fusion of semantic-suppressed CLIP patch density with a Noiseprint-style residual is well motivated, but no paper tests this exact combination.

### Gaps
- B-Free does not report a per-variant origBG test breakdown, so the user's numbers cannot be compared directly with B-Free's own model.
- FakeShield (reported as an ICLR 2025 paper) and HiFi-Net (CVPR 2023), MMFusion and CAT-Net numbers on inpainting were not verified this session. CAT-Net v2 and TruFor appear only through DinoLizer's table.
- No source verified "the global VAE trace disappears when the background is pasted back" as a measured result. It follows logically from TGIF and DinoLizer's setup but is not quantified.
- Exact TGIF2 per-method numbers were not retrieved.

---

## Q3. One-class (real-only) detection and localization methods

### Takeaway
Real-only forensic methods exist and are peer-reviewed:
- Noiseprint++ in TruFor (CVPR 2023)
- ZED, zero-shot detection via lossless-coding surprise
- Forensic Self-Descriptions (CVPR 2025)

They model **low-level residual and microstructure statistics, not semantic embeddings**. Only Noiseprint and TruFor clearly produce localization maps. ZED and Forensic Self-Descriptions are image-level as described in their abstracts.

### Cited Findings
- **Forensic Self-Descriptions** (Nguyen, Azizpour, Stamm, **CVPR 2025**):
  - Learns predictive filters on residuals, modelled at multiple scales, "using only real images in a self-supervised manner."
  - Supports zero-shot detection, open-set attribution and clustering.
  - Localization is not mentioned in the abstract.
  - Source: [arXiv 2503.21003](https://arxiv.org/abs/2503.21003)
- **ZED** (Cozzolino, Poggi, Nießner, Verdoliva; arXiv 2409.15875, Sep 2024; reported as ECCV 2024, venue not confirmed on the arXiv page):
  - A lossless encoder models per-pixel probability from context on real images only.
  - The detector measures the image's "surprise," with a multi-resolution design and a single discriminative feature.
  - Reports a >3% average accuracy gain over prior SOTA.
  - Localization is not described in the abstract.
  - Source: [arXiv 2409.15875](https://arxiv.org/abs/2409.15875)
- **Noiseprint++ / TruFor**: the fingerprint is trained only on pristine images, and forgeries appear as deviations from the expected pattern. TruFor (CVPR 2023) gives pixel maps plus a reliability map. — [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.pdf)
- A broad search for "one-class real-only inpainting localization anomaly detection" returned only industrial AD papers. **No forensic paper applies real-only patch density to localize diffusion inpainting.** — [search results incl. arXiv 2205.14852](https://arxiv.org/pdf/2205.14852)

### Inferences
- Because ZED's per-pixel probabilities are local by construction, a ZED-style surprise map could in principle localize inpainting. This is unverified.
- For a real-only CLIP-patch GMM, the closest validated precedent is the **self-consistency / "deviation from the image's own normal"** idea in Noiseprint++. A strong variant would score each patch against the *same image's* other patches (intra-image density), not only against a global real-patch bank. This targets origBG directly, since the image contains its own real reference background.

### Gaps
- "Fighting Fake News: Image Splice Detection via Learned Self-Consistency" (Huh et al., ECCV 2018), which uses EXIF consistency, was not retrieved this session. It is the canonical self-consistency localizer, but it was not cited here with a verified source.
- I found no numbers for real-only methods on TGIF, BR-Gen or the B-Free origBG data.

---

## Q4. Evidence on semantic suppression: does removing semantic content help generalization or one-class density?

### Takeaway
Several peer-reviewed and preprint works argue that semantic content in CLIP/DINO features drives overfitting and poor cross-domain generalization. Removing or freezing semantic directions, or shuffling patches, consistently helps:
- Effort / Orthogonal Subspace Decomposition (ICML 2025)
- Breaking Semantic Artifacts (NeurIPS 2024)
- GSD (preprint, 2026)
- SemAnti (preprint, 2025)

No source tests this specifically for **one-class density**. The argument that density on raw CLIP tokens would be content-dominated is inferred, not measured.

### Cited Findings
- **GSD, Geometric Semantic Decoupling** (Shuai et al., preprint arXiv 2603.09242, Mar 2026, v2 Jun 2026):
  - Identifies "semantic fallback": forensic fine-tuning leaves representations organized around high-level semantics rather than forensic traces.
  - Uses a frozen CLIP encoder and SVD to estimate the dominant semantic subspace and suppresses it, with suppression strength adapting across samples and layers.
  - Mini-batch SVD cuts overhead by >15x.
  - Offers batch, per-sample and reference-based inference protocols.
  - Source: [arXiv 2603.09242](https://arxiv.org/abs/2603.09242)
- **Orthogonal Subspace Decomposition / Effort** (Yan et al., **ICML 2025**):
  - SVD splits features into a principal (semantic) subspace that stays frozen and a residual subspace that is adapted, with about 0.19M trainable parameters.
  - The authors argue that naive fine-tuning yields a low-rank feature space that overfits to a few fake patterns. They report that the model implicitly learns that "fakes are derived from real images."
  - Sources: [ICML 2025 listing](https://icml.cc/virtual/2025/poster/45843); [arXiv 2411.15633](https://arxiv.org/html/2411.15633v4); [code](https://github.com/YZY-stack/Effort-AIGI-Detection)
- **Breaking Semantic Artifacts** (Zheng et al., **NeurIPS 2024**):
  - Detectors overfit to "semantic artifacts" and drop substantially under cross-scene testing.
  - Patch shuffling plus a patch-based classifier improves cross-scene accuracy by **+2.08%** and open-world accuracy by **+10.59%** on average over 31 test sets.
  - Source: [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6dddcff5b115b40c998a08fbd1cea4d7-Abstract.html)
- **SemAnti, "When Semantics Regulate"** (Chu et al., preprint arXiv 2511.19126, Nov 2025):
  - In CLIP, patch shuffle reduces semantic entropy and makes the feature distributions of real and synthetic images more similar.
  - Freezing the semantic subspace and fine-tuning only artifact-sensitive layers gives SOTA cross-domain results on AIGCDetectBenchmark and GenImage. The abstract gives no numbers.
  - Source: [arXiv 2511.19126](https://arxiv.org/abs/2511.19126)
- "Beyond Semantic Features: Pixel-level Mapping" (preprint arXiv 2512.17350) also lists patch shuffling among ways to limit reliance on global semantics. Not read in detail. — [arXiv html 2512.17350](https://arxiv.org/html/2512.17350v1)
- B-Free's thesis is that detectors learn content and format biases unless real and fake are semantically aligned. Its content-aligned fakes improve generalization across 27 generators. — [arXiv 2412.17671](https://arxiv.org/abs/2412.17671)
- Counter-evidence: in B-Free, a **frozen DINOv2 linear probe reaches only AUC 80.8 / bAcc 68.5**, against 99.0 / 95.2 end-to-end. Frozen foundation features alone are weak forensic features, independent of semantic suppression. — [arXiv html 2412.17671](https://arxiv.org/html/2412.17671)

### Inferences
- For a real-only GMM on CLIP patch tokens, the leading principal directions mostly encode object and scene content. A density on raw tokens will mostly measure "how typical is this content," which is **uninformative for origBG**: the inpainted object comes from the same COCO category as the original.
- Projecting out the top-k SVD directions estimated on real patch tokens (GSD or Effort style), then fitting the GMM or PatchCore bank in the residual subspace, is the principled fix. The number of directions removed, k, should be tuned on a validation set that includes origBG images.
- B-Free's linear-probe result suggests frozen CLIP-L patch tokens may carry weak low-level signal even after suppression. Consider:
  - intermediate layers, which keep more low-level information;
  - higher input resolution with no resizing, since B-Free uses 504 crops;
  - a parallel low-level residual stream (Noiseprint or ZED style).

### Gaps
- I found no paper that measures how much density estimates on CLIP/DINO embeddings are dominated by content, for example the fraction of GMM log-likelihood variance explained by semantic class. This is a testable hypothesis, not an established result.
- C2P-CLIP and D^3 were not retrieved this session.

---

## Q5. Practical considerations: patch size vs. CLIP patch14, memory banks, coreset, compute on Kaggle T4

### Takeaway
CLIP ViT-L/14 tokens give coarse localization: 16x16 tokens at 224 px or 24x24 at 336 px. DinoLizer shows 14x14-patch-resolution prediction with sliding windows is enough for state-of-the-art inpainting localization. Memory for a PatchCore bank of 1024-d tokens grows quickly and needs coreset subsampling or a parametric GMM.

### Cited Findings
- DinoLizer predicts at 14x14 patch resolution on DINOv2 and uses sliding windows because of fixed input size. — [arXiv 2511.20722](https://arxiv.org/abs/2511.20722)
- B-Free trains on 504x504 crops **without resizing**, and registers help preserve local patch information. — [arXiv html 2412.17671](https://arxiv.org/html/2412.17671)
- PatchCore's coreset subsampling exists to keep nearest-neighbour search computationally feasible. — [EmergentMind PatchCore](https://www.emergentmind.com/topics/patchcore); [arXiv 2106.08265](https://arxiv.org/abs/2106.08265)
- Pandolfini et al. found masks under 5% of the image are near chance for global detectors. — [arXiv html 2512.16688](https://arxiv.org/html/2512.16688)

### Inferences
- Token-grid arithmetic (author's calculation):
  - At 224 px, one CLIP-L/14 token covers about 0.4% of the image. A 5% mask covers about 12 tokens, enough for max or top-k pooling to fire if the per-token signal exists.
  - Resizing a 512–1024 px image to 224 px blurs the inpainting boundary and the VAE trace. Native-resolution tiling with 224 or 336 windows is preferable.
- Memory (author's calculation):
  - 256 tokens x 1024 dims x 2 bytes (fp16) ≈ 0.5 MB per image.
  - 10k real images ≈ 5 GB of raw tokens, above comfortable T4 VRAM (16 GB) once activations are counted.
  - A 1% coreset (about 25k vectors) fits easily. After projection to a residual subspace of, say, 256 dimensions, a full bank of about 2.5M vectors (about 1.3 GB in fp16) is feasible with exact or IVF search in faiss-gpu.
- GMM vs. PatchCore: a diagonal or low-rank GMM with K of about 16–64 on residual-subspace tokens has constant memory and fast scoring. It is less sensitive to rare real patterns than kNN but is smoother, which may miss small islands. Try both. PaDiM-style per-position Gaussians do not suit unaligned natural images.
- Evaluation protocol for comparing with the literature:
  - Image-level AUROC on B-Free origBG and fully regenerated images separately.
  - Pixel-level F1 and IoU on TGIF and SID-Set with GT masks.
  - Random-mask checks for object bias, as in TGIF2.
  - JPEG robustness at Q60–80.

### Gaps
- I found no published T4 timing for PatchCore with ViT-L features.
- I found no evidence on which CLIP layer best preserves inpainting traces at patch level for a real-only density model.
