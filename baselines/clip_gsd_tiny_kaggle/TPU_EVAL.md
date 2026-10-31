# Kaggle TPU v5e-8: recovered augmented GSD checkpoint

Notebook: `test_gsd_uploaded_best_variants_tpu_kaggle.ipynb`.
Builder: `build_tpu_evaluation_notebook.py`. Adapter: `evaluate_uploaded_best_tpu.py`.

## Inputs and first run

Select TPU v5e-8 and enable Internet. Attach the augmented checkpoint (native best.pt,
best.zip, or unpacked best/ containing data.pkl + data/ + version), the original
Tiny-GenImage dataset, and the original seed42/test_manifest.csv manifest dataset.
Leave all three path settings None for recursive auto-discovery under /kaggle/input.
Checkpoint validation, restricted deserialization, source remapping, file-size checks,
manifest ID order, and case deduplication are shared with the GPU evaluator.

Default RUN_FULL_TEST=False runs a tiny random-model numerical preflight and then
two complete logical batches for native and fake_jpeg96, using a separate smoke
output directory. These 16 images/case are not a representative validation set and
their accuracy/AUC must not be reported as a full test result. The toy preflight
checks basic backend/GSD operations, NOT numerical parity of the trained checkpoint.
Inspect finite logits, timing, and any XLA CPU fallback counters in
tpu_backend_diagnostics.json before changing RUN_FULL_TEST=True.

Full mode keeps the nine original cases and exactly 7,000 IDs/order/batch membership.
It saves one progress shard per completed batch and pauses normally after at most
three newly started cases or nine hours including preparation. Initial compilation
or a stalled operation can overrun a soft time budget; no protection against hard
platform kills is guaranteed. Preserve actual notebook Output, not merely working
files, for cross-session continuation. Add prior TPU Output as Input; auto resume
only accepts matching checkpoint/cohort/backend/runtime/precision fingerprints.

## Deliberate backend differences

- One process on ONE TPU device, not an eight-chip throughput implementation. No
  distributed split changes the full-batch semantic basis. The other slice devices
  are not used by this first adapter.
- BF16 autocast for CLIP, eager attention rather than CUDA FP16/SDPA. FP32 mode is
  available for diagnosis but is a separate experiment/fingerprint.
- FP32 Gram/projection/occupancy. Four 1024x1024 Gram matrices, not image tensors,
  are transferred to CPU per logical batch. CPU FP32 eigh supplies all top-k basis
  vectors; no token sampling, approximate solver, or per-image basis.
  FP32 matmul precision is explicitly highest; the preflight also compares a
  TPU FP32 Gram product against a CPU FP64 reference to detect unintended downcast.
- Frozen position embeddings are interpolated on CPU using HF4.46.3's bicubic
  formula (align_corners=False) and cached on device with a bounded LRU. Detector
  positions come from the trained checkpoint; anchor positions from original CLIP.
- No additional image resize, token padding, shape bucketing, or order changes.
  With XLA, distinct shapes can recompile and may be slow. This version is not
  advertised as faster than GPU or bit-exact numerically equivalent to GPU.

The original pretrained anchor is cloned BEFORE detector weights load. Only
weights_only=True checkpoint loading is used. No optimizer or training is invoked.
The shared evaluator received optional backend hooks; its default CUDA branch is
unchanged. Separate output roots and environment fingerprints prevent GPU/TPU score
mixing, including when resuming an output dataset containing both backends.

Auto uses installed TorchTPU if present, otherwise installed PyTorch/XLA. It does
not replace torch or install an incompatible backend wheel automatically. A missing
backend or failed hardware probe is an error, not silent CPU inference. The older
XLA path checks torch/XLA major+minor compatibility, disables global BF16 downcast,
marks graph boundaries per image, reports fallback counters, and rejects distributed
launches. A failed runtime initialization is surfaced rather than retried on another
backend after a partly initialized TPU client.

Google now recommends TorchTPU for new workloads; the XLA branch exists for Kaggle
images which still ship it. Sources checked 2026-10-23:

- https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu
- https://github.com/google-pytorch/torch_tpu/blob/main/README.md
- https://github.com/pytorch/xla/blob/v2.8.0/torch_xla/runtime.py

## Verification boundaries

CPU tests exercise cached positional equivalence, complete-token Gram/eigenspace,
the unchanged GSD equation on mixed shapes, FP32/BF16 toy paths, complete-batch CSV
resume, raw logits/softmax consistency, backend fingerprint isolation, and generated
notebook syntax/default smoke configuration. CPU test doubles explicitly use a CPU
device; production prepare() does not.

No TPU hardware is attached to the local workspace. Consequently actual Kaggle TPU
initialization, compilation speed, operator lowering, memory usage, and pretrained
7,000-image inference have NOT been hardware-validated here. Run the notebook's
smoke first and share its complete traceback if the environment needs adjustment.
