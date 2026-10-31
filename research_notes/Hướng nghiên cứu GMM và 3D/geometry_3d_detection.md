# Geometry / 3D-reconstruction cues (2D→3D lifting) for AI-generated image detection — research notes (as of 2026-10-27)

Scope: whether feed-forward 3D models (π³/Pi3, VGGT, DUSt3R/MASt3R) or monocular geometry models (MoGe, Depth Anything v2, Metric3D, Marigold) are a viable, novel signal for detecting AI-generated images (Tiny-GenImage; B-Free SD2.1 inpainting). The notes are based on about 20 search/fetch calls. Items marked **[unverified in-session]** come from background knowledge and were not confirmed by a fetched source in this session. Check them before citing.

## Q1. What are Pi3, VGGT, DUSt3R and MoGe? Can they take one image, and what do they output?

### Takeaway
All four can run on a single image, and all four use a ViT encoder that comes from (or is initialized from) DINOv2 or CroCo. Pi3 matters most here because its encoder is DINOv2, frozen during training and initialized from VGGT. Any "3D feature" taken from Pi3/VGGT encoder tokens is therefore close to plain DINOv2 features. A new signal is more likely to come from the geometry heads (point maps, confidence, camera) or the alternating-attention trunk.

### Cited Findings
- **π³ / Pi3.** arXiv 2507.13347. v1 was titled "π³: Scalable Permutation-Equivariant Visual Geometry Learning" and later versions "π³: Permutation-Equivariant Visual Geometry Learning". Accepted at **ICLR 2026** (peer-reviewed). Authors: Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang (Shanghai AI Lab), Chunhua Shen (ZJU), Tong He (SII). Code: github.com/yyfz/Pi3 — [arXiv HTML v2](https://arxiv.org/html/2507.13347v2); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10010780); [ML Anthology](https://mlanthology.org/iclr/2026/wang2026iclr-permutationequivariant/)
  - **Architecture:**
    - A DINOv2 encoder turns each view into patch tokens. It is **frozen during training**. Encoder and attention weights are **initialized from pretrained VGGT**. The paper does not state the DINOv2 size; VGGT uses ViT-L, see below.
    - 36 alternating view-wise/global self-attention layers, versus VGGT's 48.
    - No frame-index positional embeddings and no reference/camera tokens, so the model is permutation-equivariant.
    - Three separate lightweight 5-layer transformer decoders for camera, local point map and confidence. The point and confidence heads use an MLP plus pixel shuffle. The camera head is Reloc3r-style and outputs a 9D rotation that is projected to SO(3) via SVD.
    - About **959M parameters** in total (Table 5). — [arXiv HTML v2](https://arxiv.org/html/2507.13347v2)
  - **Input and outputs:** Accepts single images, videos or unordered image sets. Outputs per view:
    - a camera pose in SE(3), with affine-invariant poses;
    - a scale-invariant **local pixel-aligned point map** in that view's camera frame;
    - a **per-pixel confidence map**. — [arXiv HTML v2](https://arxiv.org/html/2507.13347v2)
  - **Training:** 15 datasets (GTA-SfM, CO3D, WildRGB-D, Habitat, ARKitScenes, TartanAir, ScanNet, ScanNet++, BlendedMVG, MatrixCity, MegaDepth, Hypersim, Taskonomy, Mid-Air, plus internal dynamic data), all real or synthetic-rendered 3D data and no generative-model images. Run on 16 to 64 A100s. — [arXiv HTML v2](https://arxiv.org/html/2507.13347v2)
  - **Speed:** 57.4 FPS on KITTI on one A800, versus 43.2 FPS for VGGT. Memory is not reported. — [arXiv HTML v2](https://arxiv.org/html/2507.13347v2)
  - **Monocular depth (Table 6, Abs Rel / δ<1.25), Pi3 vs MoGe v1:**

    | Dataset | Pi3 | MoGe v1 |
    |---|---|---|
    | Sintel | 0.277 / 0.614 | 0.273 / 0.695 |
    | Bonn | 0.044 / 0.976 | 0.050 / 0.976 |
    | KITTI | 0.060 / 0.971 | 0.054 / 0.977 |
    | NYU-v2 | 0.054 / 0.956 | 0.055 / 0.952 |

    Single-image Pi3 is roughly on par with MoGe v1. — [arXiv HTML v2](https://arxiv.org/html/2507.13347v2)
- **VGGT ("Visual Geometry Grounded Transformer").** Wang, Chen, Karaev, Vedaldi, Rupprecht, Novotny (Oxford VGG + Meta). **CVPR 2025**; arXiv 2503.11651 (14 Mar 2025). — [arXiv abs](https://arxiv.org/abs/2503.11651); [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf)
  - **Architecture:**
    - About **1.2B parameters**.
    - Images are patchified with **DINOv2 (ViT-L, 1024-dim)** plus added positional embeddings. A plain 14×14 conv patchifier was also tried and DINOv2 did better.
    - **24 blocks**, each with one frame-wise and one global self-attention layer. — [arXiv HTML](https://arxiv.org/html/2503.11651v1)
  - **Outputs:** Camera extrinsics and intrinsics, depth maps, point maps and 3D point tracks, each with a confidence score. — [GitHub](https://github.com/facebookresearch/vggt); [arXiv abs](https://arxiv.org/abs/2503.11651)
  - **Single image:** The architecture "inherently supports the input of a single image": global attention reduces to frame attention. VGGT was not trained explicitly for single-view input but gives "surprisingly good results". Monocular depth comparisons against Depth Anything V2 and MoGe on the repo come from the community, not the authors. — [arXiv HTML](https://arxiv.org/html/2503.11651v1); [GitHub](https://github.com/facebookresearch/vggt)
  - **Cost (H100, FlashAttention-3, 336×518):**

    | Frames | Time (s) | Peak memory (GB) |
    |---|---|---|
    | 1 | 0.04 | 1.88 |
    | 8 | 0.11 | 3.23 |
    | 100 | 3.12 | 21.15 |

    — [arXiv HTML](https://arxiv.org/html/2503.11651v1)
  - **Features as a backbone:** On TAP-Vid, CoTracker + VGGT features improves AJ from 49.6 to 57.2 (Kinetics), 67.4 to 72.1 (RGB-S) and 61.8 to 64.7 (DAVIS). These are geometric/tracking tasks, not semantic or forensic classification. — [arXiv HTML](https://arxiv.org/html/2503.11651v1)
  - **License:** Repo code allows commercial use (excluding military) since 29 Jul 2025. The original checkpoint is non-commercial; a separate VGGT-1B-Commercial is available by application. bf16 needs Ampere or newer, otherwise the code falls back to fp16. — [GitHub](https://github.com/facebookresearch/vggt)
- **MoGe ("Monocular Geometry Estimation"), Microsoft.** **CVPR 2025 Oral** (pp. 5261–5271). MoGe-2 is arXiv 2507.02546. MoGe-3 is a 2026 release. — [GitHub](https://github.com/microsoft/MoGe)
  - **Input:** Single image. An optional ground-truth FOV can be given.
  - **Outputs:** Metric point map in OpenCV camera coordinates, metric depth, normalized intrinsics, a validity mask and an optional normal map.
  - **Backbone:** The repo includes DINOv2 code. Checkpoint names suggest ViT-S/B/L/G.
  - **Variants:**

    | Checkpoint | Params |
    |---|---|
    | moge-vitl (MoGe-1) | 314M |
    | moge-2-vitl | 326M |
    | moge-2-vitl-normal | 331M |
    | moge-2-vitb-normal | 104M |
    | moge-2-vits-normal | 35M |
    | moge-3-vitl | 370M |
    | moge-3-vitg | 1.25B |

  - **License:** Code is MIT. — [GitHub](https://github.com/microsoft/MoGe)
- **DUSt3R / MASt3R** **[unverified in-session]:**
  - DUSt3R (Wang et al., Naver Labs, **CVPR 2024**) is a pairwise model. A CroCo-pretrained ViT-L encoder and two cross-attending decoders regress two point maps in the first view's frame, plus confidence.
  - Single-image use works by feeding the same image twice.
  - MASt3R (ECCV 2024) adds a dense local-feature matching head.
  - Project site: [dust3r.org](https://dust3r.org/). These details were not re-read in this session.
- **Depth Anything v2** **[unverified in-session]:** NeurIPS 2024. DINOv2 encoder plus a DPT head. Sizes roughly 25M (S), 98M (B), 335M (L), about 1.3B (G). Outputs relative depth, with metric fine-tunes available.
- **Marigold** (repurposes Stable Diffusion for monocular depth): arXiv 2312.02145, CVPR 2024 **[venue unverified in-session]**. — [arXiv](https://arxiv.org/abs/2312.02145)
- **Feat2GS** (arXiv 2412.09606) probes visual foundation models through Gaussian Splatting. In geometry mode it ranks RADIO, MASt3R, DUSt3R and DINO near the top, with Stable Diffusion the worst. — [arXiv](https://arxiv.org/pdf/2412.09606)

### Inferences
- Pi3 and VGGT share the same DINOv2-L patch embedding, and Pi3 keeps it **frozen**. Features taken right after the encoder are therefore essentially DINOv2-L features, possibly as fine-tuned inside VGGT. The VGGT paper does not clearly say whether the encoder was frozen or fine-tuned; this was not verified.
- What is "3D-specific" is:
  - the alternating-attention trunk tokens;
  - the explicit outputs: the point map XYZ, **confidence C**, and camera/intrinsics (focal length).
- On a single image, the global attention in VGGT/Pi3 does nothing beyond frame attention. Their multi-view advantage (cross-view consistency) is lost, unless one deliberately builds pseudo-multi-view inputs such as crops, flips or the inpainted region versus the rest of the image.
- MoGe-2 and Depth Anything v2 are cheaper and trained specifically for one image. They are the more natural single-image geometry extractors. Pi3/VGGT are only worth using for confidence or camera outputs, or for multi-crop consistency tests.

### Gaps
- Exact DINOv2 variant and frozen/fine-tuned status inside VGGT: not confirmed in the fetched text.
- Peak memory of Pi3 for one image: not reported. It is likely similar to or below VGGT's 1.88 GB at 518 px, but that is inference.
- DUSt3R, MASt3R, Depth Anything v2 and Metric3D details were not re-verified in this session.

## Q2. Published work using geometry for fake image detection (methods and numbers)

### Takeaway
There is a small but real literature. The anchor paper is Sarkar et al. (projective geometry: lines, perspective fields, object–shadow). Older forensic work includes Farid's DALL-E 2 perspective and lighting analyses. Depth-as-extra-modality work exists mostly for **face deepfakes** (DepthFake, Leporoni et al. 2024). A 3D-consistency detector exists for **video** (Grab-3D). I found no peer-reviewed work that lifts a generic still image to dense 3D with a modern feed-forward model and classifies on that.

### Cited Findings
- **Sarkar, Mai, Mahapatra, Lazebnik, Forsyth, Bhattad, "Shadows Don't Lie and Lines Can't Bend! Generative Models don't know Projective Geometry…for now."** arXiv 2311.17138 (v1 28 Nov 2023, v2 30 May 2024). — [arXiv](https://arxiv.org/abs/2311.17138); [arXiv HTML v2](https://arxiv.org/html/2311.17138v2)
  - **Venue:** Widely cited as CVPR 2024, e.g. by the AI-GenBench reference list ([AI-GenBench](https://arxiv.org/pdf/2504.20865)). The arXiv page itself states no venue, and the CVF page returned 403, so venue confirmation is indirect. A Liner summary lists conflicting metadata ([Liner](https://liner.com/ko/review/shadows-dont-lie-and-lines-cant-bend-generative-models-dont)).
  - **Method:** Three classifiers that see **no pixels**, only derived geometry:
    - line segments from a pretrained line detector;
    - perspective fields;
    - object–shadow relations (PointNet over geometric representations).
  - **Data:**
    - Generators: SDXL 1.0, **Kandinsky-v3** (main training source), DeepFloyd IF, PixArt-α. Generalization was checked on DALL-E 3, Adobe Firefly and others.
    - Real images: LSUN (indoor), BDD100K and Mapillary (outdoor).
    - 525k real and 525k generated; 100k train, 15k val, 410k test.
    - Images are "prequalified": hard sets that a pixel/signal classifier cannot separate.
  - **Results:**
    - Geometric classifiers reach **AUC 0.72–0.97**.
    - On "unconfident" sets, where the pixel prequalifier is at chance (AUC 0.49–0.51), they reach AUC **0.82** (lines, indoor), **0.84** (perspective field, outdoor) and **0.80** (combined).
    - On "misclassified" sets (prequalifier AUC 0.00) they reach up to **0.82**.
    - On the recent-timestamp set, indoor perspective-field and line classifiers exceed 0.84 AUC on SDXL and PixArt-α. Results on DeepFloyd are weak.
    - Simple logistic-regression statistics baselines reach AUC 0.58–0.75.
    - No named comparison with Corvi or Ojha numbers is given.
  - **Robustness:** No JPEG/resize robustness study. Training used 5% JPEG augmentation. — [arXiv HTML v2](https://arxiv.org/html/2311.17138v2)
- **Farid, "Perspective (In)consistency of Paint by Text"** (arXiv 2206.14617, 2022) and **"Lighting (In)consistency of Paint by Text"** (arXiv 2207.13744). — [Perspective](https://arxiv.org/pdf/2206.14617); [Lighting](https://arxiv.org/pdf/2207.13744)
  - Qualitative forensic analysis of DALL-E 2, using prompts chosen to produce perspective, shadows and reflections.
  - In the perspective paper, the tiled floor had consistent vanishing points. The countertop's vanishing point did not lie on the floor's vanishing line, which is a geometric inconsistency.
  - These are case studies, not a trained detector with accuracy numbers.
- **ControlVP** (arXiv 2512.07504, 2025) is a generation-side fix. It reports vanishing-point inconsistencies in newer generators (**DALL-E 3, Midjourney v6.1**): parallel 3D lines converge at different points. — [arXiv](https://arxiv.org/pdf/2512.07504)
- **DepthFake** (Maiano, Papa, Vocaj, Amerini, Sapienza; arXiv 2208.11074, 2022 preprint). — [arXiv HTML](https://arxiv.org/html/2208.11074v1)
  - **Method:** A FaceDepth (MobileNet) monocular depth map is added as a 4th channel (RGBD) on FaceForensics++.
  - **Frame-level accuracy, Xception backbone:**

    | Split | RGB | RGBD |
    |---|---|---|
    | FULL | 86.80% | 91.93% |
    | NeuralTextures | 76.87% | 85.09% |

  - **Video-level FULL:** 87.00% to 93.00%.
  - **Abstract:** Average gain of +3.20%, and up to +11.7% on some attacks.
  - **Overhead:** The extra channel adds negligible FLOPs, but depth-estimation time was excluded from the timings.
- **Leporoni, Maiano, Papa, Amerini, "A guided-based approach for deepfake detection: RGB-depth integration via features fusion,"** *Pattern Recognition Letters* 181:99–105 (2024, peer-reviewed). — [Sapienza listing](https://www.diag.uniroma1.it/publication/28539); [IRIS](https://iris.uniroma1.it/handle/11573/1710641)
  - Proposes the attention-based "Masked Depthfake Network" for RGB-depth fusion.
  - Reported as more accurate than RGB, and **+3.2% more robust to common adversarial attacks** on average.
  - Face deepfakes only.
- **Grab-3D** (Chen, Karaoglu, Gevers, UvA; arXiv 2512.13665, Dec 2025 preprint). — [arXiv](https://arxiv.org/abs/2512.13665v1)
  - **Detects AI-generated video** using vanishing-point-based 3D geometric temporal consistency.
  - Uses a geometry-aware transformer: geometric positional encoding, temporal-geometric attention, and an EMA-residual head.
  - Dataset: 3,322 real/generated static-scene video pairs.
  - Reports **+11.8% AP and +14.9% F1** over state-of-the-art detectors, with cross-generator generalization.
- **CAM-VFD** (arXiv 2605.17133, 2026 preprint) is a video forgery detector that uses MiDaS DPT-Hybrid depth. It flags distorted depth, flattened facial depth and depth discontinuities at object boundaries. — [arXiv](https://arxiv.org/pdf/2605.17133)
- **ReStraV** (NeurIPS 2025) is a "geometric" detector in representation space only. It measures the temporal curvature of DINOv2 trajectories for video and does not use 3D scene geometry. — [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/118520)
- **GAPL**, "Scaling Up AI-Generated Image Detection with Generator-Aware Prototypes" (arXiv 2512.12982), explicitly lists 3D structure as **future work**. The authors say their method "does not explicitly model the underlying 3D structure of the scene. Incorporating spatial intelligence … could serve as a powerful orthogonal check." This is a proposal, not tested. — [arXiv HTML](https://arxiv.org/html/2512.12982v2)
- **"How Far are AI-generated Videos from Simulating the 3D Visual World"** (arXiv 2406.19568) is a learned 3D evaluation of generated video. It evaluates video quality and is not a detector. — [arXiv](https://arxiv.org/pdf/2406.19568)

### Inferences
- Geometry cues for **still, generic-scene images** have been demonstrated mainly with **hand-crafted mid-level geometry** (lines, perspective fields, shadows), and on curated indoor/outdoor scenes (LSUN, BDD) with SDXL-era generators.
- The strongest number is AUC 0.82–0.84 on hard sets that a pixel classifier cannot separate. This suggests geometry is **complementary** to, not a replacement for, low-level detectors.
- Depth-as-extra-channel has peer-reviewed support only for faces. Gains are modest (+3 to +6 points accuracy), and it was paired with a fully trained CNN.

### Gaps
- Per-generator AUC tables for Sarkar et al. on DALL-E 3 and Firefly were not found in the text.
- I found no reliable numbers for depth or normal features on GenImage-style benchmarks (ADM, BigGAN, GLIDE, Midjourney, SD, VQDM, Wukong).

## Q3. Novelty check: has anyone used VGGT, Pi3 or DUSt3R features, or point-map consistency, for AI-image detection?

### Takeaway
No. Several targeted searches (Oct 2026) found **no paper** that uses VGGT, Pi3, DUSt3R/MASt3R, MoGe or Depth-Anything features or point-map consistency to detect AI-generated **still images**. The closest work is Grab-3D (video, vanishing points) and Sarkar et al. (projective cues). The idea looks novel, with the caveat that preprints appear quickly.

### Cited Findings
- **Searches with no detector hits:**
  - "VGGT DUSt3R features AI-generated image detection" returned only VGGT/DUSt3R background, GeoWorld (3D generation) and Grab-3D (video). — [Grab-3D](https://arxiv.org/abs/2512.13665v1); [GeoWorld](https://arxiv.org/pdf/2511.23191)
  - A 2026-focused query returned VGGT-Det, which uses VGGT internal priors for **multi-view indoor 3D object detection** (CVPR 2026). This shows that repurposing VGGT features works, but not for forensics. Also returned: VGGT-Ω (arXiv 2605.15195), and AIGI papers built on VLMs and visual foundation models, none of which uses 3D priors. — [VGGT-Det](https://arxiv.org/abs/2603.00912); [VGGT-Ω](https://arxiv.org/pdf/2605.15195); ["Simplicity Prevails"](https://arxiv.org/pdf/2602.01738); [UC-VLM](https://arxiv.org/pdf/2608.15238)
- **GAPL** (Dec 2025) names 3D "spatial intelligence" as an untested orthogonal check, which supports the claim that the direction is open. — [arXiv HTML](https://arxiv.org/html/2512.12982v2)

### Inferences
- A defensible novelty claim would be: "first use of feed-forward multi-view geometry transformers (Pi3/VGGT) as a forensic signal for single generated or inpainted images."
- The claim needs care for two reasons:
  1. Sarkar et al. and Farid already claim "geometry reveals fakes".
  2. Detectors built on DINOv2 are already common. The near-duplicate GSD/"GenImage 92.8%" line also uses foundation features ([alphaxiv 2603.09242](https://www.alphaxiv.org/abs/2603.09242v1)).
- So the contribution has to be shown to come from the **3D outputs or trunk**, not the DINOv2 encoder.

### Gaps
- Search engines index fast-moving arXiv content imperfectly. A direct arXiv full-text search for "VGGT" plus "forgery/deepfake/AIGI", and a Google Scholar "cited by" check on VGGT and Pi3, are recommended before submission.

## Q4. Do monocular 3D models hallucinate plausible geometry on generated images? Do diffusion models make geometry errors?

### Takeaway
There is solid evidence that generators make **projective errors**: vanishing points, shadows, lines, and lighting. These errors persist in DALL-E 3 and Midjourney v6.1 per ControlVP. I found **no direct published measurement** of whether depth, normal or point-map statistics from Depth-Anything, MoGe or VGGT differ between real and generated images. Because monocular models are trained to produce plausible geometry, they may "regularize away" errors. That makes the explicit residual signal (confidence, inconsistency between cues) more promising than the raw depth map.

### Cited Findings
- **Generator geometry errors:**
  - Sarkar et al.: classifiers on lines, perspective fields and shadows separate SDXL, Kandinsky-v3 and PixArt-α from real at AUC up to 0.97. Their conclusion is that "current generators cannot reliably reproduce the geometric properties of real images". — [arXiv HTML v2](https://arxiv.org/html/2311.17138v2)
  - DALL-E 2 vanishing-point inconsistency (Farid). — [arXiv](https://arxiv.org/pdf/2206.14617)
  - DALL-E 3 and Midjourney v6.1 vanishing-point inconsistencies (ControlVP). — [arXiv](https://arxiv.org/pdf/2512.07504)
  - In video, VP trajectories of generated clips are "unstable and jittery" compared with smooth real ones. — [Grab-3D](https://arxiv.org/abs/2512.13665v1)
  - Deepfake faces produce depth inconsistencies that a depth channel can exploit. — [DepthFake](https://arxiv.org/html/2208.11074v1); [Leporoni et al. 2024](https://www.diag.uniroma1.it/publication/28539)
- **Monocular depth/normal models as a cue:**
  - Pi3, MoGe and VGGT are trained only on real or rendered 3D datasets. Pi3's list contains no generated images. — [Pi3](https://arxiv.org/html/2507.13347v2)
  - Generated images are therefore out-of-distribution in a subtle way. No paper was found that measures how their confidence or depth statistics behave on such images.
  - An "OOD detection for monocular depth estimation" literature exists ([arXiv 2308.06072](https://arxiv.org/pdf/2308.06072)), but it was not applied to generated images.
  - A third-party review attributes to Leporoni et al. the argument that generation introduces depth inconsistencies. — [Preprints.org review](https://www.preprints.org/manuscript/202501.1685)

### Inferences
- The key hypothesis to test in a pilot is: on real photos, point maps from different views or crops of the same image are mutually consistent, and depth agrees with normals and vanishing points. On generated images, these 3D outputs are less consistent or less confident.
- This hypothesis is unproven. If it holds, it is the novel contribution.
- A danger: Tiny-GenImage classes are ImageNet object-centric, often close-up animals or objects with little perspective structure. Sarkar et al. had to restrict to indoor/outdoor scene categories (LSUN, BDD) where lines and shadows exist.
- On ImageNet-style images, the geometric signal may be weak and confounded with **content or scene-type differences**, since real ImageNet and generated images differ in framing and composition.

### Gaps
- No published distributions of Depth-Anything, MoGe or VGGT depth, normal or confidence statistics for real versus generated images.
- No published study of geometry errors in GenImage's generators specifically (ADM, BigGAN, GLIDE, VQDM, Wukong).

## Q5. Are high-level geometric cues robust to JPEG or resizing compared with low-level traces?

### Takeaway
I found **no direct empirical study**. The robustness argument is plausible but untested. Sarkar et al. did not evaluate JPEG or resize. DepthFake/Leporoni report only adversarial-robustness gains of +3.2% (faces).

### Cited Findings
- Sarkar et al. include no JPEG/resize robustness analysis. Training used 5% JPEG augmentation. — [arXiv HTML v2](https://arxiv.org/html/2311.17138v2)
- Leporoni et al. (PRL 2024): the RGB-D fusion model is +3.2% more robust to common **adversarial** attacks, not compression. — [Sapienza](https://www.diag.uniroma1.it/publication/28539)
- Pi3 was trained at random resolutions from 100k to 255k pixels and aspect ratios from 0.5 to 2.0. The model therefore always sees resized inputs, and its outputs should be fairly stable under rescaling. This is inference from the training setup. — [Pi3](https://arxiv.org/html/2507.13347v2)

### Inferences
- The lines, perspective and depth layout of a scene survive downsampling and moderate JPEG. Classifiers built on them should degrade less than frequency- or noise-residual detectors.
- The counter-risk is that the geometry models themselves resize the input (to about 518 px for VGGT/Pi3/MoGe). Any residual low-level signal that leaks into DINOv2 tokens would be destroyed, which is fine for robustness but limits accuracy.
- This must be demonstrated in the pilot with a degradation sweep.
- For **inpainting with real background** (B-Free SD2.1), global geometry is mostly real, so whole-image geometric statistics will barely change. Only **local** inconsistency between the inpainted region and its surroundings could work. Examples: per-pixel confidence drop, depth or normal discontinuity at the mask boundary, or a different local focal length or vanishing point. That makes it a localization-style design, not a global classifier.

### Gaps
- No paper quantifies AUC under JPEG Q or resize factor for geometry-based detectors versus low-level detectors.

## Q6. Compute cost and feasibility on a Kaggle T4/P100 (16 GB)

### Takeaway
All candidate models fit easily on 16 GB for single-image inference. The practical constraints are that T4 and P100 lack bf16 and FlashAttention-3, so runs use fp16/fp32 and are several times slower than published H100 numbers. Feature extraction for a few thousand to tens of thousands of images is feasible, especially with cached features.

### Cited Findings
- **VGGT-1B:** 1.88 GB peak memory and 0.04 s for one 336×518 frame on an H100 with FA3. On T4/P100 (not Ampere), the example code falls back to fp16. — [arXiv HTML](https://arxiv.org/html/2503.11651v1); [GitHub](https://github.com/facebookresearch/vggt)
- **Pi3:** About 959M parameters, roughly 1.9 GB of weights in fp16 by arithmetic. 57.4 FPS on an A800. — [Pi3](https://arxiv.org/html/2507.13347v2)
- **MoGe-2:** ViT-S 35M, ViT-B 104M, ViT-L 331M; MoGe-3 ViT-G 1.25B. — [MoGe GitHub](https://github.com/microsoft/MoGe)

### Inferences
- **Back-of-envelope (inference only, not measured):**
  - Pi3/VGGT in fp16 on a T4: about 2 to 4 GB VRAM per image at 518 px, and perhaps 0.2 to 0.5 s per image.
  - MoGe-2 ViT-L: well under 2 GB.
  - Extracting features for about 20k images costs roughly 1 to 3 GPU-hours on a T4.
- P100 has poor fp16 throughput. **T4 is preferable**.
- Fine-tuning the 1B geometry trunk on 16 GB is impractical without LoRA and gradient checkpointing. Use **frozen extraction plus a light head**.

### Gaps
- No measured T4/P100 latency or memory numbers for Pi3, VGGT or MoGe were found.

## Q7. Confounder: geometry models build on DINOv2. Are their features better than plain DINOv2 for downstream classification?

### Takeaway
No direct comparison for forensic classification was found. Evidence is indirect:
- VGGT uses the DINOv2-L patchifier.
- Pi3 keeps that DINOv2 encoder **frozen** and initialized from VGGT.
- Probing studies show DINOv2 is already relatively "3D-aware".
- 3D teachers can add 3D awareness to DINOv2 (Splat-and-Distill).

Any pilot **must** include a matched DINOv2-L baseline (same resolution, same head). Without it, gains from "Pi3 features" are uninterpretable.

### Cited Findings
- VGGT uses DINOv2 ViT-L (1024-d) patch tokens, chosen over a raw conv patchifier for better results. — [VGGT arXiv HTML](https://arxiv.org/html/2503.11651v1)
- Pi3's DINOv2 encoder is frozen during training and initialized from VGGT. — [Pi3](https://arxiv.org/html/2507.13347v2)
- "Probing the 3D Awareness of Visual Foundation Models" (El Banani et al., **CVPR 2024**) used frozen-feature probes for depth, normals and correspondence. 3D awareness varies widely; DINO(v2) and Stable Diffusion features probe relatively well, and all models are weak on correspondence. The exact numbers were not retrieved. — [CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Banani_Probing_the_3D_Awareness_of_Visual_Foundation_Models_CVPR_2024_paper.pdf); [arXiv](https://arxiv.org/html/2404.08636v1); [Korean summary](https://rosinality.substack.com/p/2024-4-18)
- "Splat and Distill" (arXiv 2602.06032): visual foundation models remain limited on 3D-aware tasks. Distilling from a feed-forward 3D reconstruction teacher makes DINOv2 features more 3D-aware. — [arXiv HTML](https://arxiv.org/html/2602.06032)
- AlignGemini (arXiv 2512.06746): DINOv2 does better on **pixel artifacts** and VLMs on semantic flaws. Mixing the two kinds of supervision hurts both. This means DINOv2 tokens already carry forensic low-level signal. — [arXiv](https://arxiv.org/pdf/2512.06746)
- VGGT features improve geometric downstream tasks (tracking AJ +7.6 on Kinetics), not semantic ones. — [VGGT](https://arxiv.org/html/2503.11651v1)

### Inferences
- If a Pi3-token detector beats a DINOv2-L detector, the gain must come from the alternating-attention trunk or the heads.
- The cleanest ablations are:
  - (a) frozen DINOv2-L CLS/patch tokens plus a linear probe;
  - (b) Pi3 encoder tokens, which should be about equal to (a) if the encoder is truly DINOv2-equivalent;
  - (c) Pi3 trunk tokens (final alternating-attention layer);
  - (d) **pixel-free geometry only**: point-map XYZ, depth, normals derived from the point map, confidence map, and focal length, fed to a small CNN or PointNet as in Sarkar et al.;
  - (e) the low-level detector, (e) + (a), and (e) + (d) late fusion.
- (d) is the configuration that proves the "geometry" claim, because it sees no pixels. (e) + (d) is the configuration that is likely to be practically useful.

### Gaps
- No paper compares VGGT or Pi3 trunk features against DINOv2 on any image classification or forensic task.

## Pilot design (synthesis drawn from the cited findings above; inference, not literature)
- **Data:**
  - Tiny-GenImage per generator, with leave-one-generator-out evaluation.
  - A **scene-structured subset**: filter images with a line-segment count or VP-detector threshold, because geometry cues need structure (Sarkar et al. used LSUN/BDD).
  - B-Free SD2.1 inpainting with masks, for local tests.
- **Extractors:**
  - MoGe-2 ViT-L (point map, normals, intrinsics, mask) as the cheap single-image baseline.
  - Pi3 run on a **pseudo-multi-view set** of one image: full image plus 2 to 4 overlapping crops/rescales, or inpainted region versus context crops. Measure cross-crop point-map and camera **consistency** and the **confidence** map.
  - This is where Pi3's permutation-equivariant multi-view design adds something DINOv2 cannot.
- **Features:**
  - Global stats: focal length or FOV, depth histogram, normal-orientation histogram, confidence mean and quantiles, Manhattan-ness of normals, and agreement between point-map planes and line-detected vanishing points.
  - Local maps: confidence and depth-gradient maps for a small CNN; boundary discontinuity along inpainting masks.
- **Baselines:** The current low-level detector; frozen DINOv2-L linear probe (the confounder control); Sarkar-style pixel-free classifier.
- **Robustness protocol:** Train on clean images. Test on resize factors 0.5/0.75/1.5, JPEG Q 95/75/50 and blur. Report AUC/AP per degradation for low-level, DINOv2 and geometry features, and their fusion.
- **Go/no-go criterion:**
  - Pixel-free geometry features reach meaningfully above chance (for example AUC > 0.7) on held-out generators.
  - They degrade less than the low-level detector under resize/JPEG.
  - Late fusion improves worst-case AUC.
  - If the Pi3-trunk features do not beat matched DINOv2-L, the signal is not "3D".
