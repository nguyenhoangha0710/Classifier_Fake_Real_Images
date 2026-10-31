"""CAPD (Codec-Aligned Patch Detector) — TN1 runtime on Tiny-GenImage, GSD-aug-identical data protocol.

Appended AFTER the GSD runtimes (runtime.py, runtime_raw_all7.py, manifest_reuse.py,
runtime_aug_all7.py) so that manifests, splits, augmentation draws, validation plan and
test-case transforms are byte-for-byte the SAME functions GSD-aug used.

What differs from GSD-aug (the proposed method):
  * backbone frozen + own LoRA (r=8) on qkv/proj/fc1/fc2 (CLIP: q/k/v/out_proj, fc1, fc2)
  * PatchHead (LN -> DWConv3x3 -> GELU -> 1x1 -> GELU -> 1x1) -> patch logits s
  * MIL LSE pooling z = (logsumexp(r s) - log N) / r
  * loss = L_img + lambda_neg * L_neg (+ lambda_gap(t) * L_gap during warm-up)
Backbones: 'clip' (CLIP ViT-L/14, patch 14, exactly GSD geometry) and 'dinov3' (ViT-L/16).
"""
import os
import sys
import time
import shutil
import struct
import traceback

CAPD_VERSION = 'capd_tn1_v1'
CAPD_EVAL_CASES = ['native', 'png_roundtrip', 'fake_jpeg96', 'all_jpeg96', 'all_jpeg90', 'all_jpeg80',
                   'resize_square256', 'resize_square560', 'blur_sigma1', 'resize256_jpeg96']
DINOV3_WEIGHT_NAME = 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'

CAPD_DEFAULTS = dict(DEFAULT_CONFIG)  # GSD-aug defaults (data / augmentation / validation keys)
CAPD_DEFAULTS.update(
    # --- run identity ---
    run_name='r1_clip', backbone='clip', modes=['train'],
    output_root='/kaggle/working/capd_tn1', resume_from=None,
    # --- data: identical to GSD-aug ---
    data_mode='reuse_manifests', input_root='/kaggle/input', dataset_root=None,
    manifest_root='/kaggle/input/datasets/hadinho0710/mainfest-train-genimage',
    seeds=[42], train_generators=GENS7, expected_generators=GENS7, validation_fraction=0.1,
    input_mode='center_crop_max_edge_pad14', crop_max_side=560,
    jpeg_probability=0.5, jpeg_train_quality=[75, 100], jpeg_subsampling=0,
    blur_probability=0.5, blur_sigma=[0.0, 2.0], horizontal_flip_probability=0.0,
    validation_processing_seed=42, validation_monitor=VALIDATION_MONITOR,
    epochs=50, early_stopping=True, patience=3, min_delta=0.001,
    train_batch=8, test_batch=8, micro_batch=1, num_workers=2,
    precision='fp16', gradient_checkpointing=True, threshold=0.5,
    # --- backbones ---
    clip_model_id='openai/clip-vit-large-patch14',
    dinov3_weights=None, dinov3_repo=None,
    # --- CAPD ---
    lora_rank=8, lora_alpha=8.0, head_hidden=256, lse_r=5.0,
    lambda_neg=0.5, lambda_gap_max=0.5, gap_warmup_steps=315,
    lr_lora=1e-4, lr_head=1e-3, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8,
    warmup_steps=500, grad_clip=None,
    # --- bookkeeping ---
    time_budget_hours=11.0, save_every_steps=250, log_every_steps=10, max_nonfinite=50,
    eval_cases=list(CAPD_EVAL_CASES), eval_after_train=['native'], save_smap_per_case=200,
    eval_save_every=500, bfree_root=None, bfree_max_ids=None,
    smoke=False, smoke_train_per_class=8, smoke_eval_per_class=4,
)
# Keys that do not change the experiment (may differ between sessions without breaking resume).
CAPD_VOLATILE_KEYS = {'modes', 'resume_from', 'num_workers', 'time_budget_hours', 'eval_cases',
                      'eval_after_train', 'save_smap_per_case', 'eval_save_every', 'bfree_root',
                      'bfree_max_ids', 'log_every_steps', 'save_every_steps', 'dataset_root',
                      'manifest_root', 'input_root', 'dinov3_weights', 'dinov3_repo', 'output_root'}
_T0 = time.monotonic()


# ----------------------------------------------------------------------------- config
def backbone_spec(cfg):
    if cfg['backbone'] == 'clip':
        return dict(patch=14, mean=(.48145466, .4578275, .40821073), std=(.26862954, .26130258, .27577711))
    if cfg['backbone'] == 'dinov3':
        return dict(patch=16, mean=(.485, .456, .406), std=(.229, .224, .225))
    raise ValueError(f"backbone must be 'clip' or 'dinov3', got {cfg['backbone']!r}")


def validate_capd(cfg):
    backbone_spec(cfg)
    bad = set(cfg) - set(CAPD_DEFAULTS)
    if bad:
        raise ValueError(f'Unknown config keys: {sorted(bad)}')
    if cfg['input_mode'] != 'center_crop_max_edge_pad14' or cfg['crop_max_side'] != 560:
        raise ValueError('TN1 keeps the GSD-aug geometry: center crop <=560, pad to patch multiple')
    if (cfg['train_batch'], cfg['micro_batch']) != (8, 1):
        raise ValueError('TN1 keeps logical batch 8 / microbatch 1 like GSD-aug')
    if cfg['precision'] not in ('fp16', 'fp32'):
        raise ValueError('T4 has no bf16: precision must be fp16 or fp32')
    for m in cfg['modes']:
        if m not in ('train', 'eval_tiny', 'eval_bfree'):
            raise ValueError(f'Unknown mode {m}')
    for c in cfg['eval_cases'] + cfg['eval_after_train']:
        case_spec(c)
    if cfg['validation_monitor'] != VALIDATION_MONITOR:
        raise ValueError('Early stopping must use GSD-aug validation monitor')


def protocol_signature(cfg, train, val, tests):
    stable = {k: v for k, v in cfg.items() if k not in CAPD_VOLATILE_KEYS}
    return digest(dict(config=stable, version=CAPD_VERSION, train=train.sample_id.tolist(),
                       val=val.sample_id.tolist(), test=tests.sample_id.tolist()))


def elapsed_hours():
    return (time.monotonic() - _T0) / 3600.0


def log(*args):
    print(f'[{time.strftime("%H:%M:%S")}]', *args, flush=True)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    tmp.replace(path)


# ----------------------------------------------------------------------------- test-case transforms
# Verbatim copy of evaluate_all7_variants.py (GSD evaluator) so that every case is identical.
def case_spec(name):
    fixed = {'native': dict(kind='native'), 'png_roundtrip': dict(kind='png'),
             'blur_sigma1': dict(kind='blur', sigma=1.), 'blur_sigma2': dict(kind='blur', sigma=2.),
             'resize256_jpeg96': dict(kind='resize_jpeg', side=256, quality=96),
             'resize_long560': dict(kind='resize_long', side=560)}
    if name in fixed:
        return fixed[name]
    for prefix, kind in [('fake_jpeg', 'fake_jpeg'), ('all_jpeg', 'all_jpeg'), ('resize_square', 'resize_square')]:
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            value = int(name[len(prefix):])
            if kind in ('fake_jpeg', 'all_jpeg') and 1 <= value <= 100:
                return dict(kind=kind, quality=value)
            if kind == 'resize_square' and value in (224, 256, 336, 512, 560):
                return dict(kind=kind, side=value)
    raise ValueError(f'Unsupported case: {name}')


def transform_variant(image, label, spec, cfg):
    kind = spec['kind']
    if kind == 'png':
        buf = io.BytesIO(); image.save(buf, format='PNG'); buf.seek(0)
        with Image.open(buf) as decoded:
            image = decoded.convert('RGB').copy()
    elif kind in ('all_jpeg', 'fake_jpeg'):
        if kind == 'all_jpeg' or label == 1:
            image = jpeg(image, spec['quality'], cfg['jpeg_subsampling'])
    elif kind == 'blur':
        image = image.filter(ImageFilter.GaussianBlur(spec['sigma']))
    elif kind in ('resize_square', 'resize_jpeg'):
        image = image.resize((spec['side'], spec['side']), Image.Resampling.BICUBIC)
        if kind == 'resize_jpeg':
            image = jpeg(image, spec['quality'], cfg['jpeg_subsampling'])
    elif kind == 'resize_long':
        w, h = image.size
        ratio = min(1., spec['side'] / max(w, h))
        image = image.resize((max(1, round(w * ratio)), max(1, round(h * ratio))), Image.Resampling.BICUBIC)
    elif kind != 'native':
        raise ValueError(kind)
    return image


# ----------------------------------------------------------------------------- geometry / images
_STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
             14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
             49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def estimate_jpeg_quality(im):
    """Approximate libjpeg quality from the luma table (order-independent sum ratio)."""
    tables = getattr(im, 'quantization', None)
    if not tables or 0 not in tables:
        return float('nan')
    scale = 100.0 * sum(tables[0]) / sum(_STD_LUMA)
    q = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return float(min(100.0, max(1.0, q)))


def capd_geometry(width, height, cfg):
    g = input_geometry(width, height, cfg)  # GSD center-crop coordinates (cap 560)
    p = backbone_spec(cfg)['patch']
    cw, ch = g['crop_width'], g['crop_height']
    pw, ph = -(-cw // p) * p, -(-ch // p) * p
    return dict(crop_left=g['crop_left'], crop_top=g['crop_top'], crop_w=cw, crop_h=ch,
                pad_right=pw - cw, pad_bottom=ph - ch, grid_h=ph // p, grid_w=pw // p,
                n_tokens=(pw // p) * (ph // p))


def image_to_tensor(image, cfg):
    geom = capd_geometry(*image.size, cfg)
    image = image.crop((geom['crop_left'], geom['crop_top'],
                        geom['crop_left'] + geom['crop_w'], geom['crop_top'] + geom['crop_h']))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    x = F.pad(x[None], (0, geom['pad_right'], 0, geom['pad_bottom']), mode='replicate')[0]
    spec = backbone_spec(cfg)
    x = (x - x.new_tensor(spec['mean'])[:, None, None]) / x.new_tensor(spec['std'])[:, None, None]
    return x, geom


class CapdImages(Dataset):
    """mode: train | val_native | val_processed | test (with GSD evaluator case) | bfree."""

    def __init__(self, frame, cfg, seed, mode, epoch=0, case='native'):
        self.rows = frame.to_dict('records')
        self.cfg, self.seed, self.mode, self.epoch, self.case = cfg, seed, mode, epoch, case
        self.spec = case_spec(case) if mode == 'test' else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row, cfg = self.rows[i], self.cfg
        with Image.open(row['path']) as im:
            qf = estimate_jpeg_quality(im) if self.mode in ('test', 'bfree') else float('nan')
            fmt = im.format
            image = im.convert('RGB')
        meta = dict(orig_w=image.width, orig_h=image.height, orig_format=fmt, orig_jpeg_qf_est=qf)
        if self.mode in ('train', 'val_processed'):
            params = processing_parameters(row['sample_id'], cfg, self.seed,
                                           train=self.mode == 'train', epoch=self.epoch)
            image = apply_processing(image, params, cfg)
            meta.update(params)
        elif self.mode == 'test':
            image = transform_variant(image, int(row['label']), self.spec, cfg)
        x, geom = image_to_tensor(image, cfg)
        meta.update(geom)
        meta.update(offset_mod8_x=geom['crop_left'] % 8, offset_mod8_y=geom['crop_top'] % 8,
                    offset_mod16_x=geom['crop_left'] % 16, offset_mod16_y=geom['crop_top'] % 16)
        return x, int(row['label']), i, meta


def capd_collate(items):
    x, y, i, m = zip(*items)
    return list(x), torch.tensor(y, dtype=torch.long), torch.tensor(i, dtype=torch.long), list(m)


def capd_loader(frame, cfg, seed, mode, epoch=0, case='native', batches=None):
    ds = CapdImages(frame, cfg, seed, mode, epoch, case)
    kw = dict(num_workers=cfg['num_workers'], collate_fn=capd_collate, pin_memory=True,
              persistent_workers=False)
    if batches is not None:
        return DataLoader(ds, batch_sampler=batches, **kw)
    return DataLoader(ds, batch_size=cfg['test_batch'], shuffle=False, **kw)


def train_batches(n, cfg, seed, epoch):
    """Same order as GSD's DataLoader(shuffle=True, generator=seed+epoch, drop_last=False)."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    b = cfg['train_batch']
    return [perm[i:i + b] for i in range(0, n, b)]


# ----------------------------------------------------------------------------- model
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = float(alpha) / float(r)

    in_features = property(lambda self: self.base.in_features)
    out_features = property(lambda self: self.base.out_features)
    weight = property(lambda self: self.base.weight)
    bias = property(lambda self: self.base.bias)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


def inject_lora(root, names, r, alpha):
    targets = [(parent, name, child) for parent in root.modules()
               for name, child in parent.named_children()
               if name in names and isinstance(child, nn.Linear)]
    for parent, name, child in targets:
        setattr(parent, name, LoRALinear(child, r, alpha))
    return len(targets)


_SKIP_DIRS = ('nature', 'ai', 'train', 'val', '.git', '__pycache__')


def find_file(root, name, max_depth=9):
    root = Path(root)
    pending = [(root, 0)]
    hits = []
    while pending:
        folder, depth = pending.pop()
        try:
            items = list(folder.iterdir())
        except OSError:
            continue
        for item in items:
            if item.is_file() and item.name == name:
                hits.append(item)
            elif (item.is_dir() and depth < max_depth and item.name not in _SKIP_DIRS
                  and not item.name.startswith(('SD2.1_', 'COCO_real'))):
                pending.append((item, depth + 1))
    return sorted(hits)


def resolve_dinov3(cfg):
    weights = Path(cfg['dinov3_weights']) if cfg['dinov3_weights'] else None
    if weights is None:
        hits = find_file(cfg['input_root'], DINOV3_WEIGHT_NAME)
        if len(hits) != 1:
            raise FileNotFoundError(f'Need exactly one {DINOV3_WEIGHT_NAME} under {cfg["input_root"]}; found '
                                    f'{list(map(str, hits))}. Add the private DINOv3 dataset as Input '
                                    'or set dinov3_weights.')
        weights = hits[0]
    repo = Path(cfg['dinov3_repo']) if cfg['dinov3_repo'] else None
    if repo is None:
        hits = [p.parent.parent.parent for p in find_file(cfg['input_root'], 'backbones.py')
                if p.parent.name == 'hub' and p.parent.parent.name == 'dinov3']
        repo = hits[0] if hits else None
    if repo is None:
        repo = Path('/kaggle/working/dinov3_repo')
        if not (repo / 'dinov3/hub/backbones.py').is_file():
            log('[dinov3] repo not in inputs -> git clone (needs Internet)')
            import subprocess
            subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/facebookresearch/dinov3',
                            str(repo)], check=True)
    return weights, repo


class ClipBackbone(nn.Module):
    lora_names = ('q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2')

    def __init__(self, cfg):
        super().__init__()
        from transformers import CLIPVisionModel
        v = CLIPVisionModel.from_pretrained(cfg['clip_model_id'], attn_implementation='sdpa').vision_model
        if (v.config.hidden_size, v.config.patch_size, v.config.num_hidden_layers) != (1024, 14, 24):
            raise ValueError('Expected CLIP ViT-L/14')
        self.v = v.requires_grad_(False)
        self.ckpt = cfg['gradient_checkpointing']
        self.lora_root = self.v.encoder.layers
        self.expected_lora = 24 * 6

    def forward(self, x):
        h = self.v.pre_layrnorm(self.v.embeddings(x, interpolate_pos_encoding=True))
        for layer in self.v.encoder.layers:
            def fn(t, layer=layer):
                return layer(t, attention_mask=None, causal_attention_mask=None)[0]
            h = checkpoint(fn, h, use_reentrant=False) if self.ckpt and torch.is_grad_enabled() else fn(h)
        return self.v.post_layernorm(h)[:, 1:]


class Dinov3Backbone(nn.Module):
    lora_names = ('qkv', 'proj', 'fc1', 'fc2')

    def __init__(self, cfg):
        super().__init__()
        weights, repo = resolve_dinov3(cfg)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from dinov3.hub.backbones import dinov3_vitl16
        m = dinov3_vitl16(pretrained=False)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        m.load_state_dict(state, strict=True)
        self.m = m.requires_grad_(False)
        self.prefix = 1 + int(m.n_storage_tokens)
        self.ckpt = cfg['gradient_checkpointing']
        self.lora_root = self.m.blocks
        self.expected_lora = 24 * 4
        self.weights_path, self.repo_path = str(weights), str(repo)

    def forward(self, x):
        m = self.m
        t, (H, W) = m.prepare_tokens_with_masks(x, None)
        rope = m.rope_embed(H=H, W=W) if m.rope_embed is not None else None
        for blk in m.blocks:
            def fn(a, blk=blk):
                return blk(a, rope)
            t = checkpoint(fn, t, use_reentrant=False) if self.ckpt and torch.is_grad_enabled() else fn(t)
        return m.norm(t[:, self.prefix:])


class CAPD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = ClipBackbone(cfg) if cfg['backbone'] == 'clip' else Dinov3Backbone(cfg)
        n = inject_lora(self.backbone.lora_root, self.backbone.lora_names, cfg['lora_rank'], cfg['lora_alpha'])
        if n != self.backbone.expected_lora:
            raise RuntimeError(f'LoRA injected into {n} layers, expected {self.backbone.expected_lora}')
        self.patch = backbone_spec(cfg)['patch']
        d, hid = 1024, cfg['head_hidden']
        self.norm = nn.LayerNorm(d)
        self.dw = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.pw1 = nn.Conv2d(d, hid, 1)
        self.pw2 = nn.Conv2d(hid, 1, 1)
        self.gap_head = nn.Linear(d, 1)
        nn.init.zeros_(self.pw2.bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # no RoPE coordinate jitter / dropout in the frozen backbone
        return self

    def lora_parameters(self):
        return [p for n, p in self.named_parameters() if 'lora_' in n]

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith('backbone.')]

    def forward(self, x):
        b, _, hh, ww = x.shape
        h, w = hh // self.patch, ww // self.patch
        tok = self.backbone(x)
        if tok.shape[1] != h * w:
            raise RuntimeError(f'patch tokens {tok.shape[1]} != grid {h}x{w}')
        grid = self.norm(tok).transpose(1, 2).reshape(b, -1, h, w)
        s = self.pw2(F.gelu(self.pw1(F.gelu(self.dw(grid))))).flatten(1)
        g = self.gap_head(tok.mean(1)).squeeze(1)
        return s.float(), g.float(), (h, w)


def lse_pool(s, r):
    return (torch.logsumexp(r * s, dim=1) - math.log(s.shape[1])) / r


def windowed_lse(smap, r, k):
    """smap [h,w] float tensor -> max over k x k windows (stride 2) of LSE (N=k*k)."""
    h, w = smap.shape
    if h < k or w < k:
        return float(lse_pool(smap.reshape(1, -1), r)[0])
    win = smap[None, None].unfold(2, k, 2).unfold(3, k, 2).reshape(-1, k * k)
    return float(lse_pool(win, r).max())


def _lse_subset(grid, r, rows, cols):
    sub = grid[:rows, :cols].reshape(1, -1)
    return float(lse_pool(sub, r)[0]) if sub.numel() else float('nan')


def score_details(s_row, hw, r, pad_right=0, pad_bottom=0):
    """Per-image diagnostics. z_inner drops the LAST row and LAST column of patches (edge/pad
    shortcut check, TN1 31/10); z_nopad drops only the patch row/column that contains padding."""
    s = s_row.float()
    h, w = hw
    p = torch.softmax(r * s, 0)
    k = max(1, int(round(0.05 * s.numel())))
    am = int(torch.argmax(s))
    grid = s.reshape(h, w)
    z_inner = _lse_subset(grid, r, h - 1, w - 1) if h > 1 and w > 1 else float('nan')
    z_nopad = _lse_subset(grid, r, h - int(pad_bottom > 0), w - int(pad_right > 0))
    inner = grid[:max(h - 1, 1), :max(w - 1, 1)].reshape(-1)
    ai = int(torch.argmax(inner))
    return dict(s_max=float(s.max()), s_mean=float(s.mean()), s_top5pct_mean=float(s.topk(k).values.mean()),
                argmax_row=am // w, argmax_col=am % w,
                w_entropy=float(-(p * torch.log(p.clamp_min(1e-12))).sum()),
                frac_patch_gt05=float((s > 0).float().mean()),
                z_inner=z_inner, prob_fake_inner=1 / (1 + math.exp(-z_inner)) if z_inner == z_inner else float('nan'),
                z_nopad=z_nopad, prob_fake_nopad=1 / (1 + math.exp(-z_nopad)),
                argmax_inner_row=ai // max(w - 1, 1), argmax_inner_col=ai % max(w - 1, 1),
                lastcol_mean=float(grid[:, -1].mean()), interior_mean=float(inner.mean()))


# ----------------------------------------------------------------------------- metrics
def extra_metrics(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    best = int(np.argmax(tpr - fpr))
    out = dict(best_ba_threshold=float(thr[best]), best_ba=float((tpr[best] + 1 - fpr[best]) / 2))
    for target in (0.01, 0.05):
        ok = fpr <= target
        out[f'tpr_at_fpr{int(target * 100)}'] = float(tpr[ok].max()) if ok.any() else 0.0
    bins = np.linspace(0, 1, 16)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 14)
    ece = 0.0
    for b in range(15):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    out['ece'] = float(ece)
    return out


# ----------------------------------------------------------------------------- train
def is_trainable_key(k):
    # LoRA factors + head only. NOTE: 'lora_root' aliases the backbone blocks, so a plain
    # "'lora_' in k" test would also match every frozen backbone weight (1.2 GB per file).
    return k.endswith(('.lora_A', '.lora_B')) or not k.startswith('backbone.')


def state_of(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if is_trainable_key(k)}


def save_atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def append_rows(path, rows):
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode='a', header=not Path(path).exists(), index=False)


@torch.no_grad()
def predict_frame(model, frame, cfg, seed, device, mode, case='native', smap_limit=0):
    model.eval()
    z = np.full(len(frame), np.nan)
    rows, smaps = [None] * len(frame), {}
    loss_sum, nonfinite = 0.0, 0
    for bi, (images, labels, indices, metas) in enumerate(capd_loader(frame, cfg, seed, mode, case=case)):
        for x, y, ix, meta in zip(images, labels, indices.tolist(), metas):
            with amp(cfg, device):
                s, _, hw = model(x[None].to(device))
            if not torch.isfinite(s).all():
                nonfinite += 1
                s = torch.nan_to_num(s, nan=0.0, posinf=30.0, neginf=-30.0)
            zz = lse_pool(s, cfg['lse_r'])[0]
            z[ix] = float(zz)
            loss_sum += float(F.binary_cross_entropy_with_logits(zz, y.float().to(device)))
            rows[ix] = dict(meta, **score_details(s[0], hw, cfg['lse_r'], meta.get('pad_right', 0), meta.get('pad_bottom', 0)))
            if len(smaps) < smap_limit:
                smaps[frame.sample_id.iloc[ix]] = s[0].reshape(hw).cpu().numpy().astype(np.float16)
        if bi % 100 == 0:
            log(f'[{mode}/{case}] batch {bi + 1}/{math.ceil(len(frame) / cfg["test_batch"])}')
    if np.isnan(z).any():
        raise RuntimeError('Missing predictions')
    return z, rows, loss_sum / len(frame), nonfinite, smaps


def lr_lambda(cfg):
    w = max(1, cfg['warmup_steps'])
    return lambda step: min(1.0, (step + 1) / w)


def lambda_gap(cfg, step):
    if cfg['gap_warmup_steps'] <= 0:
        return 0.0
    return cfg['lambda_gap_max'] * max(0.0, 1.0 - step / cfg['gap_warmup_steps'])


def train_capd(model, train, val, cfg, seed, device, folder, signature):
    params_lora, params_head = model.lora_parameters(), model.head_parameters()
    optimizer = torch.optim.AdamW([dict(params=params_lora, lr=cfg['lr_lora']),
                                   dict(params=params_head, lr=cfg['lr_head'])],
                                  betas=tuple(cfg['betas']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(cfg))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['precision'] == 'fp16')
    stopper = EarlyStopping(cfg['early_stopping'], cfg['patience'], cfg['min_delta'])
    st = dict(epoch=0, step_in_epoch=0, global_step=0, best=float('inf'), best_epoch=None, history=[],
              run=dict(loss=0., l_img=0., l_neg=0., l_gap=0., images=0, skipped=0, tick=0.),
              nonfinite=0, status='running')
    last = folder / 'last_state.pt'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        if ck['signature'] != signature:
            raise ValueError('Resume state belongs to a different protocol/config; use a new output_root')
        model.load_state_dict(ck['model'], strict=False)
        optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler']); stopper.state.update(ck['stopper'])
        st = ck['st']
        log(f'[resume] epoch {st["epoch"] + 1}, step {st["step_in_epoch"]}, status={st["status"]}')
        del ck
    if st['status'] == 'trained':
        return st

    def checkpoint_state(status):
        st['status'] = status
        save_atomic(last, dict(model=state_of(model), optimizer=optimizer.state_dict(),
                               scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                               stopper=dict(stopper.state), st=st, signature=signature))
        write_json(folder / 'status.json', dict(status=status, epoch=st['epoch'] + 1,
                   step_in_epoch=st['step_in_epoch'], global_step=st['global_step'],
                   hours=elapsed_hours(), best_epoch=st['best_epoch'], early_stopping=dict(stopper.state)))

    trainables = params_lora + params_head
    while st['epoch'] < cfg['epochs'] and not stopper.state['stopped']:
        epoch = st['epoch']
        batches = train_batches(len(train), cfg, seed, epoch)
        model.train()
        tick = time.monotonic()
        step_rows = []
        loader_ = capd_loader(train, cfg, seed, 'train', epoch, batches=batches[st['step_in_epoch']:])
        for images, labels, _, metas in loader_:
            if elapsed_hours() > cfg['time_budget_hours']:
                append_rows(folder / 'train_steps.csv', step_rows)
                checkpoint_state('paused_time_budget')
                log('[time budget] state saved; re-run with this output as resume_from')
                return st
            lam_gap = lambda_gap(cfg, st['global_step'])
            n = len(images)
            n_real = int((labels == 0).sum())
            optimizer.zero_grad(set_to_none=True)
            parts = dict(l_img=0., l_neg=0., l_gap=0.)
            did_backward = False
            for x, y in zip(images, labels):
                yf = y.float().to(device)[None]
                with amp(cfg, device):
                    s, g, hw = model(x[None].to(device))
                z = lse_pool(s, cfg['lse_r'])
                l_img = F.binary_cross_entropy_with_logits(z, yf)
                l_gap = F.binary_cross_entropy_with_logits(g, yf)
                l_neg = F.softplus(s).mean() if int(y) == 0 else s.sum() * 0.0
                loss = (l_img + lam_gap * l_gap) / n
                if n_real:
                    loss = loss + cfg['lambda_neg'] * l_neg / n_real
                if not torch.isfinite(loss):
                    st['nonfinite'] += 1
                    if st['nonfinite'] > cfg['max_nonfinite']:
                        checkpoint_state('failed_nonfinite')
                        raise FloatingPointError('Too many nonfinite losses: rerun with precision=fp32')
                    continue
                scaler.scale(loss).backward()
                did_backward = True
                parts['l_img'] += float(l_img) / n
                parts['l_gap'] += float(l_gap) / n
                parts['l_neg'] += float(l_neg) / max(n_real, 1)
            if not did_backward:
                st['global_step'] += 1; st['step_in_epoch'] += 1
                continue
            scaler.unscale_(optimizer)
            clip = cfg['grad_clip'] if cfg['grad_clip'] else float('inf')
            gnorm = float(torch.nn.utils.clip_grad_norm_(trainables, clip))
            old = scaler.get_scale()
            scaler.step(optimizer); scaler.update(); scheduler.step()
            skipped = int(scaler.get_scale() < old)
            st['global_step'] += 1; st['step_in_epoch'] += 1
            r = st['run']
            total = parts['l_img'] + lam_gap * parts['l_gap'] + cfg['lambda_neg'] * parts['l_neg']
            r['loss'] += total * n; r['l_img'] += parts['l_img'] * n; r['l_neg'] += parts['l_neg'] * n
            r['l_gap'] += parts['l_gap'] * n; r['images'] += n; r['skipped'] += skipped
            step_rows.append(dict(epoch=epoch + 1, step=st['step_in_epoch'], global_step=st['global_step'],
                                  loss=total, l_img=parts['l_img'], l_neg=parts['l_neg'], l_gap=parts['l_gap'],
                                  lambda_gap=lam_gap, lr_lora=optimizer.param_groups[0]['lr'],
                                  lr_head=optimizer.param_groups[1]['lr'], grad_norm=gnorm,
                                  amp_scale=scaler.get_scale(), amp_skipped=skipped, nonfinite_total=st['nonfinite'],
                                  sec_per_img=(time.monotonic() - tick) / max(1, r['images']),
                                  peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
                                  n_tokens_mean=float(np.mean([m['n_tokens'] for m in metas]))))
            if st['global_step'] % cfg['log_every_steps'] == 0:
                log(f'[{cfg["run_name"]} e{epoch + 1}] step {st["step_in_epoch"]}/{len(batches)} '
                    f'loss={r["loss"] / r["images"]:.4f} img={r["l_img"] / r["images"]:.4f} '
                    f'neg={r["l_neg"] / r["images"]:.4f} gap={r["l_gap"] / r["images"]:.4f} '
                    f'lam_gap={lam_gap:.3f} gnorm={gnorm:.2f} vram={torch.cuda.max_memory_allocated() / 1e9:.1f}GB')
                append_rows(folder / 'train_steps.csv', step_rows); step_rows = []
            if st['global_step'] % cfg['save_every_steps'] == 0:
                checkpoint_state('running')
        append_rows(folder / 'train_steps.csv', step_rows)
        # ---------------- validation: GSD-aug monitor (0.5 native + 0.5 processed CE)
        summaries, frames = {}, {}
        for view in ('native', 'processed'):
            zz, rows, loss, nonf, _ = predict_frame(model, val, cfg, seed, device, 'val_' + view)
            p = 1 / (1 + np.exp(-zz))
            y = val.label.to_numpy()
            summaries[view] = dict(loss=loss, nonfinite=nonf, **metric(y, p, cfg['threshold']), **extra_metrics(y, p))
            frames[view] = (zz, p)
        vloss = 0.5 * (summaries['native']['loss'] + summaries['processed']['loss'])
        stop = stopper.update(vloss)
        r = st['run']
        entry = dict(epoch=epoch + 1, train_loss=r['loss'] / max(1, r['images']), train_l_img=r['l_img'] / max(1, r['images']),
                     train_l_neg=r['l_neg'] / max(1, r['images']), train_l_gap=r['l_gap'] / max(1, r['images']),
                     val_loss=vloss, val_native=summaries['native'], val_processed=summaries['processed'],
                     amp_skipped_steps=r['skipped'], train_images=r['images'], nonfinite_total=st['nonfinite'],
                     hours=elapsed_hours(), early_stopping=dict(stopper.state))
        st['history'].append(entry)
        log(f'[{cfg["run_name"]}] epoch {epoch + 1}: val_loss={vloss:.5f} '
            f'native AUC={summaries["native"]["auc"]:.4f} BA={summaries["native"]["balanced_accuracy"]:.4f} '
            f'processed AUC={summaries["processed"]["auc"]:.4f} stop={stop}')
        epoch_state = dict(model=state_of(model), epoch=epoch + 1, val_loss=vloss, validation=summaries,
                           signature=signature, config=cfg)
        save_atomic(folder / f'epoch_{epoch + 1:03d}.pt', epoch_state)
        if vloss < st['best']:
            st['best'], st['best_epoch'] = vloss, epoch + 1
            shutil.copyfile(folder / f'epoch_{epoch + 1:03d}.pt', folder / 'best.pt')
        dump(folder / 'history.json', st['history'])
        pd.DataFrame([dict(epoch=e['epoch'], view=v, monitor_loss=e['val_loss'], **e['val_' + v])
                      for e in st['history'] for v in ('native', 'processed')]).to_csv(
            folder / 'validation_metrics.csv', index=False)
        vf = val[['sample_id', 'generator', 'label']].copy()
        for view, (zz, p) in frames.items():
            vf['z_' + view], vf['prob_fake_' + view] = zz, p
        vf.to_csv(folder / f'validation_predictions_epoch{epoch + 1:03d}.csv', index=False)
        st['epoch'] += 1; st['step_in_epoch'] = 0
        st['run'] = dict(loss=0., l_img=0., l_neg=0., l_gap=0., images=0, skipped=0, tick=0.)
        checkpoint_state('running')
        if stop:
            log('[early stopping] best epoch', st['best_epoch'])
    checkpoint_state('trained')
    dump(folder / 'selected_checkpoint.json', dict(best_epoch=st['best_epoch'], val_loss=st['best'],
         criterion=VALIDATION_MONITOR, signature=signature))
    return st


# ----------------------------------------------------------------------------- evaluation
def load_best(model, folder):
    ck = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ck['model'], strict=False)
    bad = [k for k in missing if is_trainable_key(k)]
    if bad or unexpected:
        raise RuntimeError(f'Checkpoint mismatch: missing={bad[:5]} unexpected={unexpected[:5]}')
    return ck['epoch']


def eval_tiny(model, tests, cfg, seed, device, folder, cases, epoch):
    out = folder / 'tiny_eval'; out.mkdir(exist_ok=True)
    for case in cases:
        target = out / f'predictions_{case}.csv'
        if target.exists():
            log(f'[eval_tiny] {case} already complete'); continue
        if elapsed_hours() > cfg['time_budget_hours']:
            log('[eval_tiny] time budget reached; re-run to continue'); return False
        z, rows, _, nonf, smaps = predict_frame(model, tests, cfg, seed, device, 'test', case, cfg['save_smap_per_case'])
        frame = tests[['sample_id', 'generator', 'label']].copy().reset_index(drop=True)
        frame = pd.concat([frame, pd.DataFrame(rows)], axis=1)
        frame['z'], frame['prob_fake'] = z, 1 / (1 + np.exp(-z))
        frame['case'], frame['run'], frame['seed'], frame['checkpoint_epoch'] = case, cfg['run_name'], seed, epoch
        frame.to_csv(target.with_suffix('.tmp'), index=False); target.with_suffix('.tmp').replace(target)
        np.savez_compressed(out / f'smaps_{case}.npz', **smaps)
        log(f'[eval_tiny] {case}: AUC={metric(frame.label.to_numpy(), frame.prob_fake.to_numpy(), .5)["auc"]:.4f} nonfinite={nonf}')
    rows = []
    for case in cases:
        f = pd.read_csv(out / f'predictions_{case}.csv')
        for gen, g in [('pooled', f), *f.groupby('generator')]:
            for pool, col in (('lse_full', 'prob_fake'), ('lse_inner', 'prob_fake_inner'), ('lse_nopad', 'prob_fake_nopad')):
                if col not in g or g[col].isna().any():
                    continue
                y, p = g.label.to_numpy(), g[col].to_numpy()
                rows.append(dict(run=cfg['run_name'], case=case, generator=gen, pooling=pool,
                                 **metric(y, p, cfg['threshold']), **extra_metrics(y, p)))
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    return True


BFREE_BRANCH_NAMES = ('SD2.1_selfconditioned', 'SD2.1_selfconditioned_origBG', 'SD2.1_inpainted_samecat',
                      'SD2.1_inpainted_samecat_origBG', 'SD2.1_inpainted_diffcat', 'SD2.1_inpainted_diffcat_origBG')
BFREE_VALID_MD5 = '72e7b4be2d049029dbedca7e75af91b0'


def bfree_frame(cfg):
    root = Path(cfg['bfree_root']) if cfg['bfree_root'] else None
    if root is None:
        hits = [p.parent for p in find_file(cfg['input_root'], 'valid_list.csv', 4)
                if (p.parent / 'train_list.csv').is_file()]
        if len(hits) != 1:
            raise FileNotFoundError(f'Add B-Free (baopv051/b-free-training-data) or set bfree_root. Found {hits}')
        root = hits[0]
    raw = (root / 'valid_list.csv').read_bytes()
    if hashlib.md5(raw).hexdigest() != BFREE_VALID_MD5:
        raise ValueError('valid_list.csv differs from the official B-Free list used for GSD')
    ids = pd.read_csv(io.BytesIO(raw), dtype=str).id.str.zfill(12).tolist()
    if cfg['bfree_max_ids']:
        ids = ids[:cfg['bfree_max_ids']]

    def folder(name):
        for cand in (root / name / name, root / name):
            if (cand / (ids[0] + '.png')).is_file():
                return cand
        raise FileNotFoundError(f'B-Free folder {name} not found under {root}')
    rows = []
    real = folder('COCO_real_512')
    rows += [dict(sample_id=f'bfree:real:{i}', coco_id=i, branch='real', label=0, path=str(real / f'{i}.png'))
             for i in ids]
    for b in BFREE_BRANCH_NAMES:
        d = folder(b)
        rows += [dict(sample_id=f'bfree:{b}:{i}', coco_id=i, branch=b, label=1, path=str(d / f'{i}.png'))
                 for i in ids]
    return pd.DataFrame(rows), root


@torch.no_grad()
def eval_bfree(model, cfg, seed, device, folder, epoch):
    out = folder / 'bfree_eval'; out.mkdir(exist_ok=True)
    frame, root = bfree_frame(cfg)
    k = int(round(224 / backbone_spec(cfg)['patch']))
    done = sorted(out.glob('part_*.csv'))
    finished = set(pd.concat([pd.read_csv(p, usecols=['sample_id']) for p in done]).sample_id) if done else set()
    todo = frame[~frame.sample_id.isin(finished)].reset_index(drop=True)
    log(f'[eval_bfree] root={root} total={len(frame)} remaining={len(todo)} window={k}x{k}')
    model.eval()
    buffer, smaps, part = [], {}, len(done)
    ds = CapdImages(todo, cfg, seed, 'bfree')
    loader_ = DataLoader(ds, batch_size=cfg['test_batch'], shuffle=False, num_workers=cfg['num_workers'],
                         collate_fn=capd_collate)

    def flush():
        nonlocal buffer, smaps, part
        if buffer:
            pd.DataFrame(buffer).to_csv(out / f'part_{part:05d}.csv', index=False)
            np.savez_compressed(out / f'smaps_{part:05d}.npz', **smaps)
            part += 1; buffer, smaps = [], {}
    for images, labels, indices, metas in loader_:
        for x, y, ix, meta in zip(images, labels, indices.tolist(), metas):
            with amp(cfg, device):
                s, _, hw = model(x[None].to(device))
            s = torch.nan_to_num(s, nan=0.0, posinf=30.0, neginf=-30.0)
            row = todo.iloc[ix]
            z = float(lse_pool(s, cfg['lse_r'])[0])
            zw = windowed_lse(s[0].reshape(hw), cfg['lse_r'], k)
            buffer.append(dict(sample_id=row.sample_id, coco_id=row.coco_id, branch=row.branch, label=int(y),
                               z=z, prob_fake=1 / (1 + math.exp(-z)), z_window=zw,
                               prob_fake_window=1 / (1 + math.exp(-zw)), **meta,
                               **score_details(s[0], hw, cfg['lse_r'], meta.get('pad_right', 0), meta.get('pad_bottom', 0))))
            smaps[row.sample_id.replace(':', '__')] = s[0].reshape(hw).cpu().numpy().astype(np.float16)
        if len(buffer) >= cfg['eval_save_every']:
            flush(); log(f'[eval_bfree] {part * cfg["eval_save_every"]:,}/{len(todo):,} new images')
        if elapsed_hours() > cfg['time_budget_hours']:
            flush(); log('[eval_bfree] time budget reached; re-run with resume_from to continue'); return False
    flush()
    allp = pd.concat([pd.read_csv(p) for p in sorted(out.glob('part_*.csv'))], ignore_index=True)
    allp['run'], allp['checkpoint_epoch'] = cfg['run_name'], epoch
    allp.to_csv(out / 'predictions_bfree.csv', index=False)
    rows = []
    real = allp[allp.branch == 'real']
    for b in BFREE_BRANCH_NAMES:
        f = pd.concat([real, allp[allp.branch == b]])
        for pool, col in (('lse_full', 'prob_fake'), ('lse_window', 'prob_fake_window'),
                          ('lse_inner', 'prob_fake_inner'), ('lse_nopad', 'prob_fake_nopad')):
            if col not in f or f[col].isna().any():
                continue
            y, p = f.label.to_numpy(), f[col].to_numpy()
            rows.append(dict(run=cfg['run_name'], branch=b, pooling=pool, **metric(y, p, cfg['threshold']),
                             **extra_metrics(y, p)))
    m = pd.DataFrame(rows)
    macro = m.groupby('pooling')[['auc', 'ap', 'balanced_accuracy', 'fake_recall', 'real_recall']].mean().reset_index()
    macro['branch'] = 'macro'
    pd.concat([m, macro], ignore_index=True).to_csv(out / 'metrics_bfree.csv', index=False)
    log('[eval_bfree] done\n' + macro.to_string(index=False))
    return True


# ----------------------------------------------------------------------------- main
def capd_main(config):
    cfg = dict(CAPD_DEFAULTS, **(config or {}))
    validate_capd(cfg)
    if cfg['smoke']:
        cfg.update(epochs=1, save_every_steps=5, log_every_steps=1, warmup_steps=5, gap_warmup_steps=5,
                   eval_cases=['native', 'fake_jpeg96'], bfree_max_ids=cfg['bfree_max_ids'] or 8)
    if not torch.cuda.is_available():
        raise RuntimeError('Needs a Kaggle GPU')
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    seed = cfg['seeds'][0]
    root_out = Path(cfg['output_root']) / cfg['run_name']
    folder = root_out / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if cfg['resume_from']:
        src = Path(cfg['resume_from'])
        cand = [src / cfg['run_name'] / f'seed{seed}', src / f'seed{seed}', src]
        src = next((c for c in cand if (c / 'last_state.pt').is_file() or (c / 'best.pt').is_file()), None)
        if src is None:
            raise FileNotFoundError(f'resume_from has no last_state.pt / best.pt for {cfg["run_name"]}: {cand}')
        if not (folder / 'last_state.pt').exists() and not (folder / 'best.pt').exists():
            log(f'[resume] copying previous output from {src}')
            skip = () if 'train' in cfg['modes'] else ('epoch_*.pt', 'last_state.pt', '*.tmp')
            shutil.copytree(src, folder, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*skip))
            best = folder / 'best.pt'
            if best.is_file() and best.stat().st_size > 200e6:   # legacy checkpoint that still holds the backbone
                ck = torch.load(best, map_location='cpu', weights_only=False)
                ck['model'] = {k: v for k, v in ck['model'].items() if is_trainable_key(k)}
                save_atomic(best, ck)
                log(f'[resume] slimmed legacy best.pt to {best.stat().st_size / 1e6:.1f} MB ({len(ck["model"])} tensors)')
                del ck
            for sub in ('manifest_reuse.json',):
                if (src.parent / sub).is_file():
                    shutil.copyfile(src.parent / sub, root_out / sub)
    write_json(root_out / 'config.json', cfg)
    write_json(root_out / 'environment.json', dict(torch=torch.__version__, python=platform.python_version(),
               gpu=torch.cuda.get_device_name(0), capd_version=CAPD_VERSION, pillow=PIL.__version__))
    seed_all(seed)
    log(f'[{cfg["run_name"]}] backbone={cfg["backbone"]} modes={cfg["modes"]} smoke={cfg["smoke"]}')
    df = scan_dataset(cfg, root_out)                       # GSD manifest reuse (no re-split, no re-scan)
    train, val, tests = make_splits(df, cfg, seed, folder)  # GSD-aug splits + fixed validation plan
    signature = protocol_signature(cfg, train, val, tests)
    model = CAPD(cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json(folder / 'architecture.json', dict(
        backbone=cfg['backbone'], patch=backbone_spec(cfg)['patch'], trainable_parameters=n_train,
        total_parameters=sum(p.numel() for p in model.parameters()), lora_rank=cfg['lora_rank'],
        lora_targets=list(model.backbone.lora_names), head_hidden=cfg['head_hidden'], lse_r=cfg['lse_r'],
        geometry='GSD center crop <=560, replicate pad to patch multiple, no resize',
        signature=signature))
    log(f'[model] trainable params: {n_train:,}')
    if 'train' in cfg['modes']:
        st = train_capd(model, train, val, cfg, seed, device, folder, signature)
        if st['status'] != 'trained':
            return st['status']
        if cfg['eval_after_train']:
            epoch = load_best(model, folder)
            eval_tiny(model, tests, cfg, seed, device, folder, cfg['eval_after_train'], epoch)
    if 'eval_tiny' in cfg['modes'] or 'eval_bfree' in cfg['modes']:
        epoch = load_best(model, folder)
        if 'eval_tiny' in cfg['modes']:
            if not eval_tiny(model, tests, cfg, seed, device, folder, cfg['eval_cases'], epoch):
                return 'paused_time_budget'
        if 'eval_bfree' in cfg['modes']:
            if not eval_bfree(model, cfg, seed, device, folder, epoch):
                return 'paused_time_budget'
    write_json(folder / 'completion.json', dict(status='complete', modes=cfg['modes'], hours=elapsed_hours()))
    return 'complete'


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1].endswith('.json') \
        and json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')).get('task', 'tn1') == 'tn1':
    _cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    _cfg.pop('task', None)
    try:
        log('[exit]', capd_main(_cfg))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
