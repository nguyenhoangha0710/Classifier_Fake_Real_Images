"""Paired full-CLIP/GSD experiment. Self-contained Kaggle runtime.

Paper v2 Fig.3: GSD is BEFORE each of the last four transformer blocks.
No fake-only codec transformations, test tuning, or automatic batch-size changes.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import io
import json
import math
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageFilter, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, average_precision_score
from transformers import CLIPVisionModel, CLIPVisionConfig
from tqdm.auto import tqdm


DEFAULT_CONFIG = dict(
    dataset_root=None, input_root='/kaggle/input', output_root='/kaggle/working/clip_gsd_tiny',
    model_id='openai/clip-vit-large-patch14', local_files_only=False,
    seeds=[42], models=['clip_gsd'], epochs=50,
    early_stopping=True, patience=3, min_delta=0.001,
    train_batch=128, test_batch=512, micro_batch=2, anchor_micro_batch=8,
    lr=1e-6, head_lr=1e-6, modulation_lr=1e-6, weight_decay=0.01,
    betas=[0.9, 0.999], eps=1e-8, k=64, lambda_max=1., a_init=2., b_init=1.,
    validation_fraction=0.1, num_workers=2, gradient_checkpointing=True,
    precision='fp16', svd_driver='gesvdj', resume=True,
    jpeg_probability=0.5, jpeg_train_quality=[75, 100], jpeg_subsampling=0,
    blur_probability=0.5, blur_sigma=[0., 2.], horizontal_flip_probability=0.5,
    jpeg_eval_qualities=[96], size_control=256, threshold=0.5,
    pooling='patch_mean_post_ln', source_generator='sd15', strict_eight_generators=False,
    expected_generators=['sd15', 'adm', 'biggan', 'glide', 'midjourney', 'wukong', 'vqdm'],
    smoke=False, smoke_train_per_class=64, smoke_eval_per_class=4,
)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_generator(name):
    n = ''.join(c for c in name.lower() if c.isalnum())
    aliases = {'sd14': ('sd14', 'sdv14', 'sdv4', 'stablediffusionv14', 'stablediffusion14'),
               'sd15': ('sd15', 'sdv15', 'sdv5', 'stablediffusionv15', 'stablediffusion15'),
               'adm': ('adm', 'guided'), 'biggan': ('biggan',), 'glide': ('glide',),
               'midjourney': ('midjourney',), 'wukong': ('wukong',), 'vqdm': ('vqdm',)}
    matches = [key for key, values in aliases.items() if any(v in n for v in values)]
    return matches[0] if len(matches) == 1 else None


def find_root(cfg):
    if cfg['dataset_root']:
        return Path(cfg['dataset_root'])
    roots = set()
    for p in Path(cfg['input_root']).rglob('*'):
        if p.is_dir() and canonical_generator(p.name):
            if any((p / s).is_dir() for s in ('train', 'training')):
                roots.add(p.parent)
    if len(roots) != 1:
        raise ValueError(f'Set dataset_root explicitly. Candidate roots: {sorted(map(str, roots))}')
    return roots.pop()


def scan_dataset(cfg, out):
    root = find_root(cfg)
    rows, errors = [], []
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    for g in sorted(root.iterdir()):
        gen = canonical_generator(g.name)
        if not g.is_dir() or gen is None:
            continue
        for split, names in [('train', ['train', 'training']), ('test', ['val', 'valid', 'validation', 'test'])]:
            available = [g / s for s in names if (g / s).is_dir()]
            if len(available) != 1:
                raise ValueError(f'{g}: expected exactly one {split} split, found {available}')
            labels = {0: [], 1: []}
            mapping = {'nature': 0, 'real': 0, '0_real': 0, 'ai': 1, 'fake': 1, '1_fake': 1}
            for folder in available[0].iterdir():
                if folder.is_dir() and folder.name.lower() in mapping:
                    labels[mapping[folder.name.lower()]].append(folder)
            if any(len(v) != 1 for v in labels.values()):
                raise ValueError(f'Unexpected label folders: {available[0]} {labels}')
            for label, folders in labels.items():
                for path in sorted(folders[0].rglob('*')):
                    if path.suffix.lower() not in extensions or not path.is_file():
                        continue
                    # Audit source-train only: other generators' train images are not used.
                    if split == 'train' and gen not in cfg.get('train_generators', [cfg['source_generator']]):
                        continue
                    try:
                        raw = path.read_bytes()
                        with Image.open(io.BytesIO(raw)) as im:
                            fmt, w, h = im.format, *im.size
                            qtable = im.quantization if hasattr(im, 'quantization') else None
                            rgb = im.convert('RGB')
                            pixel_hash = hashlib.sha256(f'{w}x{h}:'.encode() + rgb.tobytes()).hexdigest()
                        rel = path.relative_to(root).as_posix()
                        rows.append(dict(sample_id=hashlib.sha256(rel.encode()).hexdigest(),
                                         path=str(path), relative_path=rel, generator=gen, split=split,
                                         label=label, width=w, height=h, format=fmt,
                                         pixel_hash=pixel_hash, file_hash=hashlib.sha256(raw).hexdigest(),
                                         qtable_hash=digest(qtable) if qtable else '', file_bytes=len(raw)))
                        if cfg.get('scan_log_every') and len(rows) % cfg['scan_log_every'] == 0:
                            print(f'[scan] {len(rows):,} images; {gen}/{split}; invalid={len(errors)}', flush=True)
                    except Exception as e:
                        errors.append(dict(path=str(path), error=repr(e)))
    df = pd.DataFrame(rows)
    pd.DataFrame(errors).to_csv(out / 'invalid_images.csv', index=False)
    if df.empty:
        raise ValueError('No usable images')
    if (df.groupby('pixel_hash').label.nunique() > 1).any():
        raise ValueError('Identical decoded pixels have conflicting labels; inspect dataset')
    df.to_csv(out / 'image_audit.csv', index=False)
    df.groupby(['split', 'generator', 'label', 'format']).size().rename('count').to_csv(out / 'format_counts.csv')
    found = set(df.loc[df.split == 'test', 'generator'])
    expected = set(cfg.get('expected_generators') or [])
    if not cfg['smoke'] and expected and found != expected:
        raise ValueError(f'Test generators mismatch: missing={sorted(expected-found)}, '
                         f'unexpected={sorted(found-expected)}, found={sorted(found)}. '
                         'Inspect image_audit.csv and invalid_images.csv.')
    if cfg['strict_eight_generators'] and len(found) != 8:
        raise ValueError(f'Expected eight test generators; found {sorted(found)}')
    for source_gen in cfg.get('train_generators', [cfg['source_generator']]):
        source = df[(df.split == 'train') & (df.generator == source_gen)]
        if set(source.label) != {0, 1}:
            raise ValueError(f"Source {source_gen!r} needs real and fake training images. "
                             'yangsangtai/tiny-genimage excludes SD14; use source_generator=sd15.')
    return df


def balanced(frame, seed, limit=None):
    n = int(frame.label.value_counts().reindex([0, 1], fill_value=0).min())
    n = min(n, limit) if limit else n
    if n == 0:
        raise ValueError('Every cohort must contain real and fake')
    return pd.concat([frame[frame.label == y].sample(n=n, random_state=seed + y) for y in (0, 1)])


def make_splits(df, cfg, seed, out):
    source = df[(df.split == 'train') & (df.generator == cfg['source_generator'])].drop_duplicates('pixel_hash')
    if cfg['smoke']:
        source = balanced(source, seed, cfg['smoke_train_per_class'])
    source = balanced(source, seed)
    tr, va = train_test_split(source, test_size=cfg['validation_fraction'], stratify=source.label, random_state=seed)
    used = set(source.pixel_hash)
    tests = df[df.split == 'test'].copy()
    removed = tests[tests.pixel_hash.isin(used)]
    removed.to_csv(out / 'excluded_test_overlap.csv', index=False)
    tests = tests[~tests.pixel_hash.isin(used)]
    tests = tests.drop_duplicates(['generator', 'pixel_hash'])
    tests = pd.concat([balanced(g, seed, cfg['smoke_eval_per_class'] if cfg['smoke'] else None)
                       for _, g in tests.groupby('generator')])
    # Label-blind order shared by all models and processing cases; never sort batches by label.
    def ordered(frame):
        frame = frame.copy()
        frame['_order'] = frame.sample_id.map(lambda s: digest([seed, s]))
        return frame.sort_values('_order').drop(columns='_order').reset_index(drop=True)
    tr, va, tests = ordered(tr), ordered(va), ordered(tests)
    for name, frame in [('train', tr), ('validation', va), ('test', tests)]:
        frame.to_csv(out / f'{name}_manifest.csv', index=False)
    dump(out / 'split_summary.json', dict(train=len(tr), validation=len(va), test=len(tests),
         test_overlap_removed=len(removed), test_repeated_real_pixels=int(tests[tests.label == 0].pixel_hash.duplicated().sum()),
         train_hash=digest(tr.sample_id.tolist()), validation_hash=digest(va.sample_id.tolist()),
         test_hash=digest(tests.sample_id.tolist()), source_generator=cfg['source_generator'],
         selection=f"source {cfg['source_generator']} train only; no test selection"))
    return tr, va, tests


def jpeg(image, quality, subsampling):
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=int(quality), subsampling=subsampling,
               optimize=False, progressive=False)
    buf.seek(0)
    with Image.open(buf) as decoded:
        return decoded.convert('RGB')


def processing_cases(cfg):
    return ['native', 'size'] + [c for q in cfg['jpeg_eval_qualities'] for c in (f'jpeg{q}', f'size_jpeg{q}')]


def process(image, case, cfg):
    # Size stress-test includes square warp; explicitly NOT claimed to remove all size/history bias.
    if case.startswith('size'):
        image = image.resize((cfg['size_control'], cfg['size_control']), Image.Resampling.BICUBIC)
    if 'jpeg' in case:
        quality = int(case.split('jpeg')[-1])
        image = jpeg(image, quality, cfg['jpeg_subsampling'])
    return image


class Images(Dataset):
    def __init__(self, frame, cfg, seed, train=False, epoch=0, case='native'):
        self.rows = frame.to_dict('records')
        self.cfg, self.seed, self.train, self.epoch, self.case = cfg, seed, train, epoch, case

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r, c = self.rows[i], self.cfg
        with Image.open(r['path']) as im:
            image = im.convert('RGB')
        if self.train:
            rng = random.Random(int(digest([self.seed, self.epoch, r['sample_id']])[:16], 16))
            if rng.random() < c['horizontal_flip_probability']:
                image = ImageOps.mirror(image)
            if rng.random() < c['blur_probability']:
                image = image.filter(ImageFilter.GaussianBlur(rng.uniform(*c['blur_sigma'])))
            if rng.random() < c['jpeg_probability']:
                image = jpeg(image, rng.randint(*c['jpeg_train_quality']), c['jpeg_subsampling'])
        else:
            image = process(image, self.case, c)
        # Explicit standard OpenAI CLIP: shorter-side 224 bicubic, center crop, RGB mean/std.
        w, h = image.size
        image = image.resize((224 if w <= h else int(224 * w / h),
                              224 if h <= w else int(224 * h / w)), Image.Resampling.BICUBIC)
        w, h = image.size
        left, top = int(round((w - 224) / 2)), int(round((h - 224) / 2))
        image = image.crop((left, top, left + 224, top + 224))
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.
        x = (x - torch.tensor([.48145466, .4578275, .40821073])[:, None, None]) / torch.tensor([.26862954, .26130258, .27577711])[:, None, None]
        return x, int(r['label']), i


def loader(frame, cfg, seed, train=False, epoch=0, case='native'):
    ds = Images(frame, cfg, seed, train, epoch, case)
    return DataLoader(ds, batch_size=cfg['train_batch'] if train else cfg['test_batch'],
                      shuffle=train, generator=torch.Generator().manual_seed(seed + epoch),
                      num_workers=cfg['num_workers'], pin_memory=torch.cuda.is_available(),
                      drop_last=train)


class SemanticSuppression(nn.Module):
    def __init__(self, a=2., b=1., maximum=1.):
        super().__init__()
        self.a, self.b = nn.Parameter(torch.tensor(float(a))), nn.Parameter(torch.tensor(float(b)))
        self.maximum = maximum

    def forward(self, features, basis):
        # FP32 projection/occupancy; do not detach occupancy (Eq.10 differentiated end-to-end).
        with torch.autocast(features.device.type, enabled=False):
            f = features.float()
            p = (f @ basis.float()) @ basis.float().T
            r = p.square().sum((1, 2)) / f.square().sum((1, 2)).clamp_min(1e-12)
            lam = self.maximum * torch.sigmoid(self.a * r + self.b)
            return (f - lam[:, None, None] * p).to(features.dtype)


def top_basis(matrix, k, driver=None):
    with torch.autocast(matrix.device.type, enabled=False):
        options = {'driver': driver} if matrix.is_cuda and driver else {}
        _, _, vh = torch.linalg.svd(matrix.float(), full_matrices=False, **options)
        if k > min(matrix.shape):
            raise ValueError('k exceeds available dimensions')
        return vh[:k].T.contiguous().detach()


class Detector(nn.Module):
    def __init__(self, vision, cfg, gsd):
        super().__init__()
        self.vision = vision
        self.cfg = cfg
        self.selected = tuple(range(len(vision.encoder.layers) - 4, len(vision.encoder.layers)))
        d = vision.config.hidden_size
        self.head = nn.Linear(d, 2)
        self.gsd = nn.ModuleDict({str(l): SemanticSuppression(cfg['a_init'], cfg['b_init'], cfg['lambda_max'])
                                  for l in self.selected}) if gsd else nn.ModuleDict()

    def forward(self, x, bases=None):
        h = self.vision.pre_layrnorm(self.vision.embeddings(x))
        for l, block in enumerate(self.vision.encoder.layers):
            if str(l) in self.gsd:
                h = self.gsd[str(l)](h, bases[l])
            def apply(t, layer=block):
                return layer(t, attention_mask=None, causal_attention_mask=None)[0]
            h = checkpoint(apply, h, use_reentrant=False) if self.training and self.cfg['gradient_checkpointing'] else apply(h)
        h = self.vision.post_layernorm(h)
        if self.cfg['pooling'] == 'patch_mean_post_ln':
            h = h[:, 1:].mean(1)
        elif self.cfg['pooling'] == 'all_mean_post_ln':
            h = h.mean(1)
        else:
            raise ValueError('Unsupported pooling')
        return self.head(h)


def amp(cfg, device):
    return torch.autocast(device.type, dtype=torch.float16 if cfg['precision'] == 'fp16' else torch.bfloat16,
                          enabled=device.type == 'cuda' and cfg['precision'] != 'fp32')


@torch.no_grad()
def estimate_bases(anchor, images, cfg, device, selected):
    # Anchor PRE-block states, matching Fig.3. Collect the ENTIRE logical batch on CPU.
    chunks = {l: [] for l in selected}
    for x in images.split(cfg['anchor_micro_batch']):
        with amp(cfg, device):
            h = anchor.pre_layrnorm(anchor.embeddings(x.to(device)))
            for l, layer in enumerate(anchor.encoder.layers):
                if l in chunks:
                    chunks[l].append(h[:, 1:].detach().cpu())
                h = layer(h, attention_mask=None, causal_attention_mask=None)[0]
    bases = {}
    for l in selected:
        m = torch.cat(chunks.pop(l), dim=0).flatten(0, 1).to(device=device, dtype=torch.float32)
        bases[l] = top_basis(m, cfg['k'], cfg['svd_driver'])
        del m
    return bases


def build_models(cfg, device, gsd):
    pretrained = CLIPVisionModel.from_pretrained(cfg['model_id'], local_files_only=cfg['local_files_only'], attn_implementation='eager')
    v = pretrained.vision_model
    assert (v.config.hidden_size, v.config.num_hidden_layers, v.config.patch_size, v.config.image_size) == (1024, 24, 14, 224), 'Expected CLIP ViT-L/14 224'
    model = Detector(v, cfg, gsd).to(device)
    anchor = copy.deepcopy(v).requires_grad_(False).eval().to(device) if gsd else None
    if anchor is not None and device.type == 'cuda' and cfg['precision'] != 'fp32':
        anchor.to(dtype=torch.float16 if cfg['precision'] == 'fp16' else torch.bfloat16)
    return model, anchor


@torch.no_grad()
def predict(model, anchor, frame, cfg, seed, device, case):
    model.eval()
    result = np.full(len(frame), np.nan)
    batches = np.full(len(frame), -1, dtype=np.int64)
    for batch_id, (images, _, indices) in enumerate(tqdm(loader(frame, cfg, seed, case=case), desc=case)):
        bases = estimate_bases(anchor, images, cfg, device, model.selected) if anchor is not None else None
        for start in range(0, len(images), cfg['micro_batch']):
            ix = indices[start:start + cfg['micro_batch']].numpy()
            with amp(cfg, device):
                logits = model(images[start:start + cfg['micro_batch']].to(device), bases)
            result[ix] = logits.float().softmax(-1)[:, 1].cpu().numpy()
            batches[ix] = batch_id
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite or missing predictions')
    return result, batches


def metric(y, p, threshold):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=len(y), auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)), accuracy=float(np.mean(y == pred)),
                fake_recall=float(tp / (tp + fn)), real_recall=float(tn / (tn + fp)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


class EarlyStopping:
    """Track significant validation-loss improvements, separately from best checkpoint."""
    def __init__(self, enabled=True, patience=3, min_delta=0.001, state=None):
        if patience < 1 or min_delta < 0:
            raise ValueError('patience must be >=1; min_delta must be >=0')
        self.enabled, self.patience, self.min_delta = enabled, patience, min_delta
        self.state = dict(reference_loss=float('inf'), bad_epochs=0, stopped=False)
        if state is not None:
            self.state.update(state)

    def update(self, loss):
        if not math.isfinite(loss):
            raise FloatingPointError('Nonfinite validation loss')
        if loss < self.state['reference_loss'] - self.min_delta:
            self.state.update(reference_loss=loss, bad_epochs=0)
        else:
            self.state['bad_epochs'] += 1
        self.state['stopped'] = bool(self.enabled and self.state['bad_epochs'] >= self.patience)
        return self.state['stopped']


def train_model(model, anchor, train, val, cfg, seed, device, folder, signature):
    groups = [dict(params=model.vision.parameters(), lr=cfg['lr']),
              dict(params=model.head.parameters(), lr=cfg['head_lr'])]
    if len(model.gsd):
        groups.append(dict(params=model.gsd.parameters(), lr=cfg['modulation_lr']))
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg['betas']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and cfg['precision'] == 'fp16')
    start, best, history = 0, float('inf'), []
    stopper = EarlyStopping(cfg['early_stopping'], cfg['patience'], cfg['min_delta'])
    latest = folder / 'last.pt'
    if cfg['resume'] and latest.exists():
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        if ck['signature'] != signature:
            raise ValueError('Resume protocol/config/manifest changed; use a fresh output_root')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        start, best, history = ck['epoch'] + 1, ck['best'], ck['history']
        if 'early_stopping' in ck:
            stopper.state.update(ck['early_stopping'])
        del ck
    if len(train) < cfg['train_batch']:
        raise ValueError('Not enough source-training images for a full logical batch')
    for epoch in range(start, cfg['epochs']):
        if stopper.state['stopped']:
            print('Resuming an early-stopped run: restore best checkpoint and evaluate.')
            break
        model.train()
        running, count, skipped = 0., 0, 0
        for images, labels, _ in tqdm(loader(train, cfg, seed, True, epoch), desc=f'epoch {epoch+1}'):
            bases = estimate_bases(anchor, images, cfg, device, model.selected) if anchor is not None else None
            optimizer.zero_grad(set_to_none=True)
            for a in range(0, len(images), cfg['micro_batch']):
                x, y = images[a:a + cfg['micro_batch']].to(device), labels[a:a + cfg['micro_batch']].to(device)
                with amp(cfg, device):
                    logits = model(x, bases)
                    loss = F.cross_entropy(logits.float(), y, reduction='sum') / len(images)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss).backward()
                running += float(loss.detach()) * len(images)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            count += len(images)
        p, _ = predict(model, anchor, val, cfg, seed, device, 'native')
        vloss = float(F.binary_cross_entropy(torch.from_numpy(p).double().clamp(1e-7, 1-1e-7),
                                             torch.tensor(val.label.to_numpy(), dtype=torch.float64)))
        entry = dict(epoch=epoch+1, train_loss=running/count, val_loss=vloss, amp_skipped_steps=skipped,
                     **metric(val.label.to_numpy(), p, cfg['threshold']))
        entry['modulation'] = {l: dict(a=float(g.a), b=float(g.b)) for l, g in model.gsd.items()}
        should_stop = stopper.update(vloss)
        entry['early_stopping'] = dict(stopper.state)
        history.append(entry)
        print(entry)
        if vloss < best:
            best = vloss
            torch.save(dict(model=model.state_dict(), epoch=epoch, signature=signature, config=cfg), folder / 'best.pt')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch, best=best, history=history, signature=signature,
                        early_stopping=dict(stopper.state)), latest)
        dump(folder / 'history.json', history)
        dump(folder / 'training_status.json', dict(epoch=epoch+1, max_epochs=cfg['epochs'],
             early_stopping=dict(stopper.state), patience=cfg['patience'], min_delta=cfg['min_delta']))
        if should_stop:
            print(f"Early stopping at epoch {epoch+1}: no significant val-loss improvement for {cfg['patience']} epochs.")
            break
    ck = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    dump(folder / 'selected_checkpoint.json', dict(epoch=ck['epoch']+1, criterion='minimum source-validation cross entropy', signature=signature))


def summarize(predictions, cfg, out):
    metrics, shifts = [], []
    for (seed, model, case), frame in predictions.groupby(['seed', 'model', 'case']):
        for gen, g in [('pooled', frame)] + list(frame.groupby('generator')):
            metrics.append(dict(seed=seed, model=model, case=case, generator=gen,
                                seen=(gen == cfg['source_generator']), **metric(g.label.to_numpy(), g.prob_fake.to_numpy(), cfg['threshold'])))
    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(out / 'metrics.csv', index=False)
    per_gen = metric_df[metric_df.generator != 'pooled']
    for protocol, g in [('all', per_gen), ('unseen', per_gen[per_gen.generator != cfg['source_generator']])]:
        g.groupby(['seed', 'model', 'case'])[['auc', 'balanced_accuracy', 'fake_recall', 'real_recall', 'accuracy']].mean().to_csv(out / f'macro_{protocol}.csv')
    for (seed, model), group in predictions.groupby(['seed', 'model']):
        base = group[group.case == 'native'].set_index('sample_id')
        for case, changed in group[group.case != 'native'].groupby('case'):
            changed = changed.set_index('sample_id')
            if set(changed.index) != set(base.index):
                raise ValueError('Paired cohorts differ')
            changed = changed.loc[base.index]
            if not np.array_equal(base.batch_id, changed.batch_id):
                raise ValueError('GSD logical batch membership changed')
            delta = changed.prob_fake - base.prob_fake
            paired = base[['generator', 'label']].copy()
            paired['delta'] = delta
            paired['abs_delta'] = delta.abs()
            paired['flip'] = (changed.prob_fake >= cfg['threshold']) != (base.prob_fake >= cfg['threshold'])
            paired['native_prob'] = base.prob_fake
            paired['changed_prob'] = changed.prob_fake
            paired.reset_index().to_csv(out / f'paired_seed{seed}_{model}_{case}.csv', index=False)
            for (gen, label), g in paired.groupby(['generator', 'label']):
                shifts.append(dict(seed=seed, model=model, case=case, generator=gen, label=label,
                                   mean_delta=g.delta.mean(), mean_abs_delta=g.abs_delta.mean(), flip_rate=g.flip.mean(), n=len(g)))
    pd.DataFrame(shifts).to_csv(out / 'paired_summary.csv', index=False)
    comparison = per_gen.pivot(index=['seed', 'case', 'generator'], columns='model', values=['auc', 'balanced_accuracy', 'fake_recall'])
    comparison.to_csv(out / 'model_comparison.csv')
    if {'clip_ft', 'clip_gsd'} <= set(predictions.model):
        for name in ('auc', 'balanced_accuracy', 'fake_recall'):
            comparison[(name, 'gsd_minus_ft')] = comparison[(name, 'clip_gsd')] - comparison[(name, 'clip_ft')]
        comparison.to_csv(out / 'model_comparison.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    plot = per_gen[per_gen.generator != cfg['source_generator']].groupby(['model', 'case']).balanced_accuracy.mean().unstack(0)
    plot.plot.bar(ax=ax)
    ax.set_ylabel('Macro balanced accuracy (unseen generators)')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / 'unseen_balanced_accuracy.png', dpi=160)
    plt.close(fig)
    return metric_df


def run(config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if cfg['smoke']:
        cfg.update(epochs=1, train_batch=8, test_batch=8, k=4, strict_eight_generators=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Use Kaggle GPU; run local unit tests for CPU validation')
    if cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support BF16; use fp16 on T4/P100')
    if min(cfg['epochs'], cfg['micro_batch'], cfg['anchor_micro_batch'], cfg['train_batch'], cfg['test_batch']) < 1:
        raise ValueError('Batch sizes and epoch count must be positive')
    out = Path(cfg['output_root'])
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'config.json', cfg)
    import transformers
    dump(out / 'environment.json', dict(torch=torch.__version__, transformers=transformers.__version__,
         pillow=PIL.__version__, python=platform.python_version(), gpu=torch.cuda.get_device_name(0),
         use_devices=1, paper='https://arxiv.org/html/2603.09242v2', torch_threads=torch.get_num_threads()))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    df = scan_dataset(cfg, out)
    predictions = []
    for seed in cfg['seeds']:
        split_out = out / f'seed{seed}'
        split_out.mkdir(exist_ok=True)
        train, val, tests = make_splits(df, cfg, seed, split_out)
        cfg_hash = digest({**cfg, 'train_files': train.file_hash.tolist(), 'val_files': val.file_hash.tolist(),
                           'test_files': tests.file_hash.tolist()})
        for model_name in cfg['models']:
            if model_name not in ('clip_ft', 'clip_gsd'):
                raise ValueError(model_name)
            seed_all(seed)  # Same backbone AND randomly initialized classifier across the two models.
            folder = split_out / model_name
            folder.mkdir(exist_ok=True)
            model, anchor = build_models(cfg, device, model_name == 'clip_gsd')
            dump(folder / 'architecture.json', dict(backbone=str(model.vision.config),
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 gsd_pre_block_indices=list(model.selected) if anchor else [],
                 initialization_head_sha=digest(model.head.weight.detach().cpu().tolist()),
                 pooling=cfg['pooling'], batch_basis='entire logical batch, independent of microbatch'))
            train_model(model, anchor, train, val, cfg, seed, device, folder, digest([cfg_hash, model_name]))
            for case in processing_cases(cfg):
                p, batches = predict(model, anchor, tests, cfg, seed, device, case)
                frame = tests[['sample_id', 'generator', 'label', 'pixel_hash']].copy()
                frame = frame.assign(seed=seed, model=model_name, case=case, prob_fake=p, batch_id=batches)
                frame.to_csv(folder / f'predictions_{case}.csv', index=False)
                predictions.append(frame)
            del model, anchor
            gc.collect()
            torch.cuda.empty_cache()
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(out / 'predictions.csv', index=False)
    metrics = summarize(predictions, cfg, out)
    dump(out / 'completion.json', dict(status='complete', smoke=cfg['smoke'], config_hash=digest(cfg)))
    print(metrics[metrics.generator == 'pooled'].to_string(index=False))
    return metrics
