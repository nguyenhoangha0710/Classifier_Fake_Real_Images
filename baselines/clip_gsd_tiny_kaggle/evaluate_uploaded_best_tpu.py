"""Single-device TPU inference adapter. No training, distributed split, or CUDA computation.

The image/batch protocol is unchanged, but BF16 + eager attention + CPU eigensolver
is a distinct numerical backend, not a bit-exact reproduction of CUDA FP16/SDPA.
"""
if 'run_uploaded_evaluation' not in globals():
    from pathlib import Path
    _tpu_here = Path(__file__).resolve().parent
    exec(compile((_tpu_here / 'evaluate_uploaded_best.py').read_text(encoding='utf-8'),
                 str(_tpu_here / 'evaluate_uploaded_best.py'), 'exec'), globals())

import importlib.util
import importlib.metadata
import os
from collections import OrderedDict
from contextlib import nullcontext

TPU_ADAPTER_VERSION = 'single_tpu_cpu_eigh_cached_positions_v1'
TPU_DEFAULTS = dict(UPLOAD_DEFAULTS,
    output_root='/kaggle/working/gsd_uploaded_best_tpu_v1',
    tpu_backend='auto', tpu_precision='bf16', cpu_threads=4,
    save_every_batches=1, log_every_batches=1,
    max_new_cases_per_run=3, time_budget_hours=9.0,
    require_preflight=True,
)


def validate_tpu_options(options):
    if options['tpu_backend'] not in ('auto', 'xla', 'torch_tpu'):
        raise ValueError('tpu_backend must be auto, xla, or torch_tpu')
    if options['tpu_precision'] not in ('bf16', 'fp32'):
        raise ValueError('TPU precision must be bf16 or fp32 (not CUDA fp16)')
    if isinstance(options['cpu_threads'], bool) or not isinstance(options['cpu_threads'], int) or options['cpu_threads'] < 1:
        raise ValueError('cpu_threads must be a positive integer')
    if options['micro_batch'] != 1 or options['anchor_micro_batch'] != 1:
        raise ValueError('Initial TPU adapter requires micro_batch=anchor_micro_batch=1; logical batch remains8')
    for key in ('XLA_USE_BF16', 'XLA_DOWNCAST_BF16'):
        if os.environ.get(key, '').lower() in ('1', 'true', 'yes'):
            raise ValueError(f'Unset {key}: global downcasting would invalidate FP32 Gram/GSD projection')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Launch directly in one notebook process; do not use torchrun/xmp.spawn')
    if os.environ.get('XLA_USE_SPMD', '').lower() in ('1', 'true', 'yes'):
        raise ValueError('Disable XLA SPMD for the single-device adapter')


class CachedPositionEmbeddings(nn.Module):
    """Same state_dict names and HF bicubic formula; interpolate frozen weights on CPU.

    This avoids TPU bicubic fallbacks. A bounded device cache supports mixed shapes
    without adding any image resize/pad or changing token membership.
    """
    def __init__(self, original, max_entries=8):
        super().__init__()
        self.class_embedding = original.class_embedding
        self.patch_embedding = original.patch_embedding
        self.position_embedding = original.position_embedding
        self.register_buffer('position_ids', original.position_ids, persistent=False)
        self.patch_size = original.patch_size
        self.image_size = original.image_size
        self.position_cpu = original.position_embedding.weight.detach().float().cpu().clone()
        self.max_entries = max_entries
        self.cache = OrderedDict()

    def positions(self, height, width, device):
        key = (height, width, str(device))
        if key not in self.cache:
            position = self.position_cpu.unsqueeze(0)
            count = position.shape[1] - 1
            if (height // self.patch_size) * (width // self.patch_size) == count and height == width:
                result = position
            else:
                side = math.isqrt(count)
                if side * side != count:
                    raise ValueError('Pretrained position grid must be square')
                patch = position[:, 1:].reshape(1, side, side, position.shape[-1]).permute(0, 3, 1, 2)
                patch = F.interpolate(patch, size=(height // self.patch_size, width // self.patch_size),
                                      mode='bicubic', align_corners=False)
                result = torch.cat((position[:, :1], patch.permute(0, 2, 3, 1).reshape(1, -1, position.shape[-1])), 1)
            self.cache[key] = result.to(device)
            while len(self.cache) > self.max_entries:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def forward(self, pixel_values, interpolate_pos_encoding=True):
        if not interpolate_pos_encoding:
            raise ValueError('TPU adapter requires positional interpolation enabled')
        b, _, h, w = pixel_values.shape
        patch = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype)).flatten(2).transpose(1, 2)
        embeddings = torch.cat((self.class_embedding.expand(b, 1, -1), patch), 1)
        return embeddings + self.positions(h, w, embeddings.device)


def install_cached_positions(vision):
    if not isinstance(vision.embeddings, CachedPositionEmbeddings):
        vision.embeddings = CachedPositionEmbeddings(vision.embeddings)


def cpu_gram_basis(gram, tokens, k):
    """Full-token uncentered PCA; no token subsampling or approximation of eigenvectors."""
    g = gram.detach().float().cpu()
    if k > min(tokens, g.shape[0]) or tokens < 1 or not torch.isfinite(g).all():
        raise ValueError('Invalid Gram matrix/token count/k')
    g = (g + g.T) * (0.5 / tokens)
    values, vectors = torch.linalg.eigh(g)
    basis = vectors[:, -k:].contiguous()
    if not torch.isfinite(basis).all():
        raise FloatingPointError('Nonfinite CPU eigensolver output')
    identity = torch.eye(k, dtype=basis.dtype)
    if not torch.allclose(basis.T @ basis, identity, atol=2e-4, rtol=2e-4):
        raise FloatingPointError('Non-orthonormal GSD basis')
    return basis


class TPUInferenceBackend:
    def __init__(self):
        self.kind, self.device = None, None
        self.runtime_version, self.hardware = None, None
        self.preflight_report = None
        self.batch_calls = 0
        self.diagnostics = {}

    def prepare(self, cfg, options):
        validate_tpu_options(options)
        torch.set_num_threads(options['cpu_threads'])
        self.precision = options['tpu_precision']
        self.output_root = Path(options['output_root'])
        kind = options['tpu_backend']
        if kind == 'auto':
            # Use the TPU runtime actually installed in the Kaggle image. Never pip-replace torch.
            kind = ('torch_tpu' if importlib.util.find_spec('torch_tpu') is not None else 'xla')
        self.kind = kind
        if kind == 'xla':
            if importlib.util.find_spec('torch_xla') is None:
                raise RuntimeError('No TPU backend installed. Use Kaggle TPU runtime; do not pip-upgrade torch alone.')
            os.environ.setdefault('PJRT_DEVICE', 'TPU')
            import torch_xla
            import torch_xla.core.xla_model as xm
            import torch_xla.runtime as xr
            self.xm = xm
            # FP32 tensors alone do not guarantee FP32-quality TPU dot products.
            try:
                from torch_xla.backends import set_mat_mul_precision
                set_mat_mul_precision('highest')
            except ImportError:
                # Legacy Kaggle XLA builds predate the public backends wrapper.
                torch_xla._XLAC._xla_set_mat_mul_precision('highest')
            # In-memory graph cache is still used even if a persistent cache is not available.
            if hasattr(xr, 'initialize_cache') and not os.environ.get('XLA_PERSISTENT_CACHE_PATH'):
                xr.initialize_cache(str(Path(options['checkpoint_cache']).parent / 'gsd_tpu_xla_compile_cache'))
            self.device = torch_xla.device() if hasattr(torch_xla, 'device') else xm.xla_device()
            if xr.device_type() != 'TPU' or xr.world_size() != 1:
                raise RuntimeError('Expected TPU hardware and a single notebook process')
            if hasattr(xr, 'is_spmd') and xr.is_spmd():
                raise RuntimeError('SPMD runtime is already initialized; restart without SPMD')
            self.runtime_version = str(torch_xla.__version__)
            if str(torch.__version__).split('+')[0].split('.')[:2] != self.runtime_version.split('+')[0].split('.')[:2]:
                raise RuntimeError('torch and torch_xla major/minor versions must match; restart with matching TPU image')
            attributes = xr.global_runtime_device_attributes() if hasattr(xr, 'global_runtime_device_attributes') else []
            # Host addresses/physical coords vary between sessions; fingerprint hardware type, not VM identity.
            kinds = sorted({str(item.get('device_kind', item.get('kind', 'unknown'))) for item in attributes})
            self.hardware = dict(device_kinds=kinds, runtime_device_count=len(attributes),
                                 accelerator_type=os.environ.get('TPU_ACCELERATOR_TYPE', 'unspecified'))
        else:
            if importlib.util.find_spec('torch_tpu') is None:
                raise RuntimeError('torch_tpu not installed; choose xla if the Kaggle image provides torch_xla')
            import torch_tpu
            self.device = torch.device('tpu:0')
            torch.set_float32_matmul_precision('highest')
            self.runtime_version = importlib.metadata.version('torch_tpu')
            # Experimental, documented TorchTPU API; do not use removed torch.tpu CUDA-style helpers.
            hardware_name = os.environ.get('TPU_ACCELERATOR_TYPE', 'Kaggle v5e-8 (user-selected)')
            if hasattr(torch, 'tpu') and hasattr(torch.tpu, 'get_device_properties'):
                hardware_name = str(torch.tpu.get_device_properties(self.device).name)
            self.hardware = dict(device_type=hardware_name, devices_used=1)
        # Fail immediately on a missing/unassigned accelerator; never fall back silently to CPU.
        probe = torch.ones(2, 2, device=self.device)
        if not torch.equal((probe @ probe).float().cpu(), torch.full((2, 2), 2.)):
            raise RuntimeError('TPU matrix-multiply probe failed')
        cfg.update(precision=self.precision, attention='eager')
        self.sync()
        print(f'[TPU] backend={self.kind}, device={self.device}, precision={self.precision}; '
              'ONE device, logical batch8; eager attention + CPU FP32 eigh. Not CUDA-equivalent.', flush=True)
        if options.get('require_preflight', True):
            self.preflight_report = self.preflight()
        self.record_diagnostics('after_preflight')
        return self.device

    def sync(self):
        if self.kind == 'xla':
            self.xm.mark_step()
            self.xm.wait_device_ops()
        elif self.kind == 'torch_tpu':
            torch.accelerator.synchronize()

    def context(self):
        return (torch.autocast(self.device.type, dtype=torch.bfloat16)
                if self.precision == 'bf16' else nullcontext())

    def environment(self):
        return dict(torch=str(torch.__version__), transformers=transformers.__version__, pillow=PIL.__version__,
                    backend=self.kind, runtime=self.runtime_version, adapter=TPU_ADAPTER_VERSION,
                    accelerator='TPU', hardware=self.hardware, devices_used=1,
                    precision=self.precision, eigensolver='cpu_fp32_eigh',
                    float32_matmul_precision='highest',
                    positional_interpolation='cpu_fp32_bicubic_align_corners_false',
                    cpu_threads=torch.get_num_threads())

    def record_diagnostics(self, stage):
        if self.kind == 'xla':
            import torch_xla.debug.metrics as met
            fallbacks = {name: met.counter_value(name) for name in met.counter_names()
                         if name.startswith('aten::')}
            self.diagnostics[stage] = dict(cpu_fallback_counters=fallbacks,
                                          report=met.metrics_report())
            if fallbacks:
                print('[TPU/warning] XLA CPU fallback counters: ' + json.dumps(fallbacks) +
                      '; inspect tpu_backend_diagnostics.json before a full run.', flush=True)
        else:
            self.diagnostics[stage] = dict(backend=self.kind,
                note='No XLA counters for this backend; this is not proof of absence of fallback.')
        if hasattr(self, 'output_root'):
            self.output_root.mkdir(parents=True, exist_ok=True)
            atomic_json(self.output_root / 'tpu_backend_diagnostics.json', self.diagnostics)

    def build_models(self, cfg, device, state_dict):
        print('[TPU/model] Loading original pretrained CLIP anchor, then strict detector checkpoint.', flush=True)
        pretrained = CLIPVisionModel.from_pretrained(cfg['model_id'],
            local_files_only=cfg['local_files_only'], attn_implementation='eager')
        vision = pretrained.vision_model
        if (vision.config.hidden_size, vision.config.patch_size) != (1024, 14):
            raise ValueError('Expected original CLIP ViT-L/14')
        # Clone ORIGINAL pretrained anchor before detector checkpoint load.
        anchor = copy.deepcopy(vision).requires_grad_(False).eval()
        model = NativeDetector(vision, cfg, True).requires_grad_(False).eval()
        model.load_state_dict(state_dict, strict=True)
        install_cached_positions(model.vision)
        install_cached_positions(anchor)
        model.to(device); anchor.to(device)
        self.sync()
        self.batch_calls = 0
        print('[TPU/model] Both branches loaded; first real image shapes may trigger compilation.', flush=True)
        return model, anchor

    @torch.no_grad()
    def estimate(self, anchor, images, cfg, selected):
        grams = {l: torch.zeros(anchor.config.hidden_size, anchor.config.hidden_size,
                              dtype=torch.float32, device=self.device) for l in selected}
        tokens = 0
        for offset, x in image_chunks(images, 1):
            if self.batch_calls == 0:
                print(f'[TPU/anchor] image {offset+1}/{len(images)}, shape={tuple(x.shape)}', flush=True)
            with self.context():
                h = anchor.pre_layrnorm(anchor.embeddings(x.to(self.device), interpolate_pos_encoding=True))
            tokens += h.shape[0] * (h.shape[1] - 1)
            # Only pre-block states of selected layers are needed; final anchor block need not run.
            for l, block in enumerate(anchor.encoder.layers):
                if l in grams:
                    m = h[:, 1:].flatten(0, 1).float()
                    grams[l] = grams[l] + m.T @ m  # explicitly OUTSIDE BF16 autocast
                if l == max(selected):
                    break
                with self.context():
                    h = block(h, attention_mask=None, causal_attention_mask=None)[0]
            self.sync()  # bounded XLA graph per image; no per-token host transfer
        return {l: cpu_gram_basis(g, tokens, cfg['k']).to(self.device) for l, g in grams.items()}

    @torch.no_grad()
    def forward_detector(self, model, x, bases):
        with self.context():
            h = model.vision.pre_layrnorm(model.vision.embeddings(x.to(self.device), interpolate_pos_encoding=True))
        for l, block in enumerate(model.vision.encoder.layers):
            if str(l) in model.gsd:
                # Same GSD equation; FP32 occupancy/projection and learned a/b.
                gsd = model.gsd[str(l)]
                f = h.float()
                basis = bases[l].float()
                projection = (f @ basis) @ basis.T
                ratio = projection.square().sum((1, 2)) / f.square().sum((1, 2)).clamp_min(1e-12)
                lam = gsd.maximum * torch.sigmoid(gsd.a.float() * ratio + gsd.b.float())
                h = (f - lam[:, None, None] * projection).to(h.dtype)
            with self.context():
                h = block(h, attention_mask=None, causal_attention_mask=None)[0]
        with self.context():
            logits = model.head(model.vision.post_layernorm(h)[:, 1:].mean(1))
        return logits.float()

    @torch.no_grad()
    def predict_batch(self, model, anchor, images, cfg, device):
        started = time.monotonic()
        bases = self.estimate(anchor, images, cfg, model.selected)
        print(f'[TPU/basis] {len(images)} images, CPU eigh done in {time.monotonic()-started:.1f}s', flush=True)
        outputs = []
        for offset, x in image_chunks(images, 1):
            if self.batch_calls == 0:
                print(f'[TPU/detector] image {offset+1}/{len(images)}, shape={tuple(x.shape)}', flush=True)
            logits = self.forward_detector(model, x, bases)
            self.sync()
            outputs.append(logits.cpu().numpy())
        self.batch_calls += 1
        if self.batch_calls == 1:
            self.record_diagnostics('first_batch')
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    def cleanup(self):
        self.sync()
        gc.collect()

    @torch.no_grad()
    def preflight(self):
        """A small real CLIP/GSD computation on TPU vs CPU, NOT pretrained test accuracy."""
        print('[TPU/preflight] Tiny mixed-shape CLIP + full-batch Gram/GSD; first compile can be slow.', flush=True)
        started = time.monotonic()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(731)
            matrix = torch.randn(128, 32)
            device_matrix = matrix.to(self.device)
            gram_test = (device_matrix.T @ device_matrix).float().cpu()
            gram_reference = matrix.double().T @ matrix.double()
            gram_error = float((gram_test.double() - gram_reference).abs().max() / gram_reference.abs().max())
            tiny = CLIPVisionConfig(image_size=28, patch_size=14, hidden_size=32,
                intermediate_size=64, num_hidden_layers=6, num_attention_heads=4, attention_dropout=0.)
            tiny._attn_implementation = 'eager'
            cfg = dict(DEFAULT_CONFIG, precision='fp32', k=3, micro_batch=1, anchor_micro_batch=1,
                       gradient_checkpointing=False, attention='eager')
            vision = CLIPVisionModel(tiny).vision_model
            model = NativeDetector(copy.deepcopy(vision), cfg, True).requires_grad_(False).eval()
            anchor = copy.deepcopy(vision).requires_grad_(False).eval()
            images = [torch.randn(3, 28, 28), torch.randn(3, 42, 28)]
        basis_cpu = estimate_bases(anchor, images, cfg, torch.device('cpu'), model.selected)
        reference = torch.cat([model(x.unsqueeze(0), basis_cpu) for x in images]).float().numpy()
        install_cached_positions(model.vision); install_cached_positions(anchor)
        model.to(self.device); anchor.to(self.device)
        output = self.predict_batch(model, anchor, images, cfg, self.device)
        error = float(np.max(np.abs(reference - output)))
        limit = 0.15 if self.precision == 'bf16' else 0.01
        report = dict(status='pass' if np.isfinite(output).all() and error <= limit and gram_error <= 1e-4 else 'fail',
            test='tiny_random_clip_mixed_shapes_full_batch_gsd_NOT_checkpoint_parity',
            max_abs_logit_delta=error, tolerance=limit, seconds=time.monotonic()-started,
            fp32_gram_relative_max_error=gram_error, fp32_gram_tolerance=1e-4,
            note='Only a backend sanity check; does not establish CUDA parity for trained checkpoint.')
        print('[TPU/preflight] ' + json.dumps(report), flush=True)
        self.preflight_report = report
        if report['status'] != 'pass':
            raise RuntimeError('TPU numerical preflight failed: ' + json.dumps(report))
        del model, anchor
        self.cleanup()
        return report


def run_uploaded_tpu_evaluation(options=None):
    opts = dict(TPU_DEFAULTS, **(options or {}))
    backend = TPUInferenceBackend()
    try:
        return run_uploaded_evaluation(opts, backend=backend)
    finally:
        if backend.preflight_report is not None:
            out = Path(opts['output_root'])
            out.mkdir(parents=True, exist_ok=True)
            atomic_json(out / 'tpu_preflight.json', backend.preflight_report)
        if backend.kind is not None and backend.device is not None:
            backend.record_diagnostics('session_end')
