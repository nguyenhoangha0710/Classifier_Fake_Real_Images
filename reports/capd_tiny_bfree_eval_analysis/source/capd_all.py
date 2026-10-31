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


"""Native-resolution, raw-only All7 training. Also appended to the base runtime in the notebook."""
if 'DEFAULT_CONFIG' not in globals():
    from pathlib import Path
    _base_file = Path(__file__).with_name('runtime.py')
    exec(compile(_base_file.read_text(encoding='utf-8'), str(_base_file), 'exec'), globals())

import time

GENS7 = ['sd15', 'adm', 'biggan', 'glide', 'midjourney', 'wukong', 'vqdm']
DEFAULT_CONFIG = dict(DEFAULT_CONFIG, output_root='/kaggle/working/gsd_raw_all7_native_v1',
    source_generator='all7', train_generators=GENS7, expected_generators=GENS7,
    train_batch=8, test_batch=8, micro_batch=1, anchor_micro_batch=1, num_workers=0,
    jpeg_probability=0., blur_probability=0., horizontal_flip_probability=0.,
    input_mode='native_rgb_edge_pad14', basis_method='gram_eigh', attention='sdpa',
    eval_cases=['native'], scan_log_every=500)

_base_scan = scan_dataset

def scan_dataset(cfg, out):
    print('[scan] Reading all seven train/val folders; RGB decoding and hashing. No resize.', flush=True)
    start = time.monotonic()
    frame = _base_scan(cfg, out)
    frame.groupby(['split','generator','label']).size().rename('n').to_csv(out/'dataset_counts.csv')
    print(f'[scan] {len(frame):,} images audited in {time.monotonic()-start:.1f}s', flush=True)
    return frame


def make_splits(df, cfg, seed, out):
    source = df[(df.split=='train') & df.generator.isin(cfg['train_generators'])].copy()
    # Assign each decoded-pixel group ONCE, even when it occurs in several generators.
    unique = source.drop_duplicates('pixel_hash')
    strata = unique.generator + ':' + unique.label.astype(str)
    a, b = train_test_split(unique, test_size=cfg['validation_fraction'],
                          stratify=strata, random_state=seed)
    def select(frame, limit):
        groups = [g.drop_duplicates('pixel_hash') for _,g in frame.groupby('generator',sort=True)]
        if set(frame.generator)!=set(cfg['train_generators']):
            raise ValueError('Every configured train generator must have a nonempty cohort')
        n = min(int(g.label.value_counts().reindex([0,1],fill_value=0).min()) for g in groups)
        if limit is not None:
            n = min(n, limit)
        if n < 1:
            raise ValueError('A generator lacks real/fake images after hash-group splitting')
        return pd.concat([balanced(g, seed, n) for g in groups],ignore_index=True)
    tr = select(source[source.pixel_hash.isin(a.pixel_hash)], cfg['smoke_train_per_class'] if cfg['smoke'] else None)
    va = select(source[source.pixel_hash.isin(b.pixel_hash)], cfg['smoke_eval_per_class'] if cfg['smoke'] else None)
    tests = df[df.split=='test'].copy()
    removed = tests[tests.pixel_hash.isin(source.pixel_hash)]
    removed.to_csv(out/'excluded_test_overlap.csv',index=False)
    tests = tests[~tests.pixel_hash.isin(source.pixel_hash)].drop_duplicates(['generator','pixel_hash'])
    tests = pd.concat([balanced(g,seed,cfg['smoke_eval_per_class'] if cfg['smoke'] else None)
                       for _,g in tests.groupby('generator',sort=True)],ignore_index=True)
    def ordered(f):
        return f.assign(_order=f.sample_id.map(lambda s:digest([seed,s]))).sort_values('_order').drop(columns='_order').reset_index(drop=True)
    tr,va,tests=map(ordered,(tr,va,tests))
    assert not set(tr.pixel_hash)&set(va.pixel_hash)
    assert not (set(tr.pixel_hash)|set(va.pixel_hash))&set(tests.pixel_hash)
    for name,frame in [('train',tr),('validation',va),('test',tests)]:
        frame.to_csv(out/f'{name}_manifest.csv',index=False)
        frame.groupby(['generator','label']).size().rename('n').to_csv(out/f'{name}_counts.csv')
    dump(out/'split_summary.json',dict(train=len(tr),validation=len(va),test=len(tests),
        test_overlap_removed=len(removed),selection='All7 raw; unique pixel groups split before per-generator/class balancing',
        train_hash=digest(tr.sample_id.tolist()),validation_hash=digest(va.sample_id.tolist()),test_hash=digest(tests.sample_id.tolist())))
    print(f'[split] train={len(tr):,}, validation={len(va):,}, test={len(tests):,}; all 7 seen',flush=True)
    return tr,va,tests


def processing_cases(cfg):
    return cfg['eval_cases']


def input_geometry(width, height, cfg):
    """Source-pixel crop coordinates and exact model dimensions, without resampling."""
    cap = cfg['crop_max_side'] if cfg['input_mode']=='center_crop_max_edge_pad14' else None
    cw = min(width, cap) if cap else width
    ch = min(height, cap) if cap else height
    left, top = (width-cw)//2, (height-ch)//2
    pw, ph = ((cw+13)//14)*14, ((ch+13)//14)*14
    return dict(crop_left=left,crop_top=top,crop_width=cw,crop_height=ch,
                input_width=pw,input_height=ph,grid_w=pw//14,grid_h=ph//14,
                tokens=(pw//14)*(ph//14)+1)


def protocol_id(cfg):
    if cfg['input_mode']=='center_crop_max_edge_pad14':
        return f"raw_all7_center_crop{cfg['crop_max_side']}_v1"
    return 'raw_all7_native_v1'


class Images(Dataset):
    def __init__(self,frame,cfg,seed,train=False,epoch=0,case='native'):
        self.rows=frame.to_dict('records'); self.cfg=cfg; self.case=case; self.train=train
        if train and case!='native':
            raise ValueError('Training must use native raw images')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        row=self.rows[i]
        with Image.open(row['path']) as im:
            image=im.convert('RGB')
        if not self.train:
            if self.case.startswith('fake_jpeg'):
                if int(row['label'])==1:
                    image=jpeg(image,int(self.case[len('fake_jpeg'):]),self.cfg['jpeg_subsampling'])
            elif self.case!='native':
                image=process(image,self.case,self.cfg)
        geom=input_geometry(*image.size,self.cfg)
        image=image.crop((geom['crop_left'],geom['crop_top'],
                          geom['crop_left']+geom['crop_width'],geom['crop_top']+geom['crop_height']))
        x=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()).permute(2,0,1)/255.
        # Preserve pixels inside the selected crop. No resize or upscaling of smaller images.
        h,w=x.shape[-2:]
        x=F.pad(x.unsqueeze(0),(0,(-w)%14,0,(-h)%14),mode='replicate').squeeze(0)
        mean=x.new_tensor([.48145466,.4578275,.40821073])[:,None,None]
        std=x.new_tensor([.26862954,.26130258,.27577711])[:,None,None]
        return (x-mean)/std,int(row['label']),i


def native_collate(items):
    x,y,i=zip(*items)
    return list(x),torch.tensor(y,dtype=torch.long),torch.tensor(i,dtype=torch.long)


def loader(frame,cfg,seed,train=False,epoch=0,case='native'):
    return DataLoader(Images(frame,cfg,seed,train,epoch,case),
        batch_size=cfg['train_batch'] if train else cfg['test_batch'], shuffle=train,
        generator=torch.Generator().manual_seed(seed+epoch),num_workers=cfg['num_workers'],
        collate_fn=native_collate,pin_memory=torch.cuda.is_available(),drop_last=False)


def image_chunks(images,limit):
    # Keep membership and order of the mixed logical batch, regardless of shape or label.
    start=0
    while start<len(images):
        end=start+1
        while end<min(start+limit,len(images)) and images[end].shape==images[start].shape:
            end+=1
        yield start,torch.stack(images[start:end])
        start=end


class NativeDetector(Detector):
    def forward(self,x,bases=None):
        h=self.vision.pre_layrnorm(self.vision.embeddings(x,interpolate_pos_encoding=True))
        for l,block in enumerate(self.vision.encoder.layers):
            if str(l) in self.gsd:
                h=self.gsd[str(l)](h,bases[l])
            def apply(t,layer=block):
                return layer(t,attention_mask=None,causal_attention_mask=None)[0]
            h=checkpoint(apply,h,use_reentrant=False) if self.training and self.cfg['gradient_checkpointing'] else apply(h)
        return self.head(self.vision.post_layernorm(h)[:,1:].mean(1))


@torch.no_grad()
def estimate_bases(anchor,images,cfg,device,selected):
    # M.T @ M has the same principal right-singular subspace as uncentered M.
    # Stream ALL tokens: no token sampling and no BN x d concatenation on GPU.
    grams={l:torch.zeros(anchor.config.hidden_size,anchor.config.hidden_size,device=device,dtype=torch.float32) for l in selected}
    tokens=0
    for _,x in image_chunks(images,cfg['anchor_micro_batch']):
        with amp(cfg,device):
            h=anchor.pre_layrnorm(anchor.embeddings(x.to(device),interpolate_pos_encoding=True))
            tokens+=h.shape[0]*(h.shape[1]-1)
            for l,layer in enumerate(anchor.encoder.layers):
                if l in grams:
                    with torch.autocast(device.type,enabled=False):
                        m=h[:,1:].flatten(0,1).float()
                        grams[l].add_(m.T@m)
                h=layer(h,attention_mask=None,causal_attention_mask=None)[0]
    if cfg['k']>min(tokens,anchor.config.hidden_size):
        raise ValueError(f"k={cfg['k']} exceeds token/feature dimensions of this logical batch ({tokens})")
    bases={}
    with torch.autocast(device.type,enabled=False):
        for l,g in grams.items():
            _,v=torch.linalg.eigh((g+g.T)*(0.5/tokens))
            bases[l]=v[:,-cfg['k']:].contiguous().detach()
    return bases


def build_models(cfg,device,gsd):
    pretrained=CLIPVisionModel.from_pretrained(cfg['model_id'],local_files_only=cfg['local_files_only'],attn_implementation=cfg['attention'])
    v=pretrained.vision_model
    if (v.config.hidden_size,v.config.patch_size)!=(1024,14):
        raise ValueError('Native All7 protocol requires CLIP ViT-L/14')
    model=NativeDetector(v,cfg,gsd).to(device)
    anchor=copy.deepcopy(v).requires_grad_(False).eval().to(device) if gsd else None
    if anchor is not None and device.type=='cuda' and cfg['precision']!='fp32':
        anchor.to(dtype=torch.float16 if cfg['precision']=='fp16' else torch.bfloat16)
    return model,anchor


@torch.no_grad()
def predict(model,anchor,frame,cfg,seed,device,case):
    model.eval(); result=np.full(len(frame),np.nan); batches=np.full(len(frame),-1)
    for bid,(images,_,indices) in enumerate(loader(frame,cfg,seed,case=case)):
        bases=estimate_bases(anchor,images,cfg,device,model.selected) if anchor is not None else None
        for start,x in image_chunks(images,cfg['micro_batch']):
            with amp(cfg,device):
                logits=model(x.to(device),bases)
            ix=indices[start:start+len(x)].numpy()
            result[ix]=logits.float().softmax(-1)[:,1].cpu().numpy(); batches[ix]=bid
        if bid%25==0:
            print(f'[{case}] batch {bid+1}/{math.ceil(len(frame)/cfg["test_batch"])}',flush=True)
    if not np.isfinite(result).all():
        raise ValueError('Missing/nonfinite prediction')
    return result,batches


def save_checkpoint(path,state):
    temp=path.with_suffix('.tmp')
    torch.save(state,temp)
    temp.replace(path)


def train_model(model,anchor,train,val,cfg,seed,device,folder,signature):
    groups=[dict(params=model.vision.parameters(),lr=cfg['lr']),dict(params=model.head.parameters(),lr=cfg['head_lr'])]
    if len(model.gsd):
        groups.append(dict(params=model.gsd.parameters(),lr=cfg['modulation_lr']))
    optimizer=torch.optim.AdamW(groups,betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and cfg['precision']=='fp16')
    stopper=EarlyStopping(cfg['early_stopping'],cfg['patience'],cfg['min_delta'])
    latest=folder/'last.pt'; start=0; best=float('inf'); history=[]
    if cfg['resume'] and latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck['signature']!=signature:
            raise ValueError('Resume config/manifests changed; use a new output_root')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        stopper.state.update(ck['early_stopping']); start=ck['epoch']+1; best=ck['best']; history=ck['history']
        del ck
        print(f'[resume] next epoch {start+1}, stopped={stopper.state["stopped"]}',flush=True)
    for epoch in range(start,cfg['epochs']):
        if stopper.state['stopped']:
            break
        model.train(); running=0.; count=0; skipped=0; tick=time.monotonic()
        for bid,(images,labels,_) in enumerate(loader(train,cfg,seed,True,epoch)):
            bases=estimate_bases(anchor,images,cfg,device,model.selected) if anchor is not None else None
            optimizer.zero_grad(set_to_none=True)
            for a,x in image_chunks(images,cfg['micro_batch']):
                with amp(cfg,device):
                    logits=model(x.to(device),bases)
                    loss=F.cross_entropy(logits.float(),labels[a:a+len(x)].to(device),reduction='sum')/len(images)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                scaler.scale(loss).backward(); running+=float(loss.detach())*len(images)
            old=scaler.get_scale(); scaler.step(optimizer); scaler.update(); skipped+=int(scaler.get_scale()<old); count+=len(images)
            if bid%10==0:
                print(f'[epoch {epoch+1}] batch {bid+1}/{math.ceil(len(train)/cfg["train_batch"])}; loss={running/count:.5f}',flush=True)
        p,_=predict(model,anchor,val,cfg,seed,device,'native')
        vloss=float(F.binary_cross_entropy(torch.from_numpy(p).double().clamp(1e-7,1-1e-7),torch.tensor(val.label.to_numpy(),dtype=torch.float64)))
        stop=stopper.update(vloss)
        entry=dict(epoch=epoch+1,train_loss=running/count,val_loss=vloss,seconds=time.monotonic()-tick,
            amp_skipped_steps=skipped,train_images=count,**metric(val.label.to_numpy(),p,cfg['threshold']),
            modulation={l:dict(a=float(g.a.detach()),b=float(g.b.detach())) for l,g in model.gsd.items()},early_stopping=dict(stopper.state))
        history.append(entry); print(entry,flush=True)
        if vloss<best:
            best=vloss; save_checkpoint(folder/'best.pt',dict(model=model.state_dict(),epoch=epoch,signature=signature,config=cfg))
        save_checkpoint(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            epoch=epoch,best=best,history=history,signature=signature,config=cfg,early_stopping=dict(stopper.state)))
        dump(folder/'history.json',history)
        dump(folder/'training_status.json',dict(epoch=epoch+1,max_epochs=cfg['epochs'],early_stopping=dict(stopper.state)))
        if stop:
            print('[early stopping] Restore best checkpoint for evaluation.',flush=True)
            break
    ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False); model.load_state_dict(ck['model'])
    dump(folder/'selected_checkpoint.json',dict(epoch=ck['epoch']+1,criterion='minimum All7 raw validation cross entropy',signature=signature))


def summarize(predictions,cfg,out):
    rows=[]
    for (seed,model,case),f in predictions.groupby(['seed','model','case']):
        for gen,g in [('pooled',f),*f.groupby('generator')]:
            rows.append(dict(seed=seed,model=model,case=case,generator=gen,
                seen=gen in cfg['train_generators'],**metric(g.label.to_numpy(),g.prob_fake.to_numpy(),cfg['threshold'])))
    result=pd.DataFrame(rows); result.to_csv(out/'metrics.csv',index=False)
    result[result.generator!='pooled'].groupby(['seed','model','case'])[['auc','balanced_accuracy','fake_recall','real_recall','accuracy']].mean().to_csv(out/'macro_all.csv')
    return result


def validate_raw_config(cfg):
    if set(cfg['train_generators'])!=set(GENS7) or set(cfg['expected_generators'])!=set(GENS7):
        raise ValueError('This protocol requires exactly all seven Tiny generators')
    if any(cfg[k]!=0 for k in ('jpeg_probability','blur_probability','horizontal_flip_probability')):
        raise ValueError('Raw-only training forbids JPEG/blur/flip augmentation')
    if cfg['input_mode'] not in ('native_rgb_edge_pad14','center_crop_max_edge_pad14') or cfg['basis_method']!='gram_eigh':
        raise ValueError('Unsupported native-input protocol')
    if cfg['input_mode']=='center_crop_max_edge_pad14':
        cap=cfg.get('crop_max_side')
        if isinstance(cap,bool) or not isinstance(cap,int) or cap<14 or cap%14:
            raise ValueError('crop_max_side must be a positive integer multiple of 14')
    if cfg['attention']!='sdpa' or cfg['pooling']!='patch_mean_post_ln':
        raise ValueError('Native protocol requires SDPA and patch_mean_post_ln')
    if min(cfg[k] for k in ('train_batch','test_batch','micro_batch','anchor_micro_batch','epochs','k'))<1:
        raise ValueError('Positive batches, epochs and k required')
    if cfg['precision'] not in ('fp16','bf16','fp32'):
        raise ValueError('precision must be fp16, bf16 or fp32')
    if not cfg['eval_cases'] or 'native' not in cfg['eval_cases']:
        raise ValueError('Evaluation must include native')
    for case in cfg['eval_cases']:
        if case not in ('native','size') and not any(case==prefix+str(q) for prefix in ('jpeg','fake_jpeg','size_jpeg') for q in range(1,101)):
            raise ValueError(f'Unknown evaluation case: {case}')


def run(config=None):
    cfg={**DEFAULT_CONFIG,**(config or {})}; validate_raw_config(cfg)
    if cfg['smoke']:
        cfg.update(epochs=1,train_batch=2,test_batch=2,k=4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':
        raise RuntimeError('Use Kaggle GPU; CPU unit tests cover native input separately')
    if cfg['precision']=='bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bf16; use fp16 on T4')
    out=Path(cfg['output_root']); out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():
        previous=json.loads((out/'config.json').read_text(encoding='utf-8'))
        if previous!=cfg:
            raise ValueError('Existing output config differs; choose a new output_root before running')
    dump(out/'config.json',cfg)
    dump(out/'environment.json',dict(torch=torch.__version__,pillow=PIL.__version__,python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0),use_devices=1,protocol=protocol_id(cfg)))
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    df=scan_dataset(cfg,out)
    census=df[['sample_id','split','generator','label','width','height']].copy()
    geometry=pd.DataFrame([input_geometry(int(w),int(h),cfg) for w,h in zip(census.width,census.height)])
    census=pd.concat([census.reset_index(drop=True),geometry],axis=1)
    census.to_csv(out/'token_census.csv',index=False)
    print(f'[{protocol_id(cfg)}] tokens min/median/max: {census.tokens.min()}/{census.tokens.median()}/{census.tokens.max()}. No pixel resampling.',flush=True)
    results=[]
    for seed in cfg['seeds']:
        split_out=out/f'seed{seed}'; split_out.mkdir(exist_ok=True)
        train,val,tests=make_splits(df,cfg,seed,split_out)
        signature=digest(dict(config=cfg,train=train.file_hash.tolist(),val=val.file_hash.tolist(),test=tests.file_hash.tolist(),protocol=protocol_id(cfg)))
        for name in cfg['models']:
            if name not in ('clip_ft','clip_gsd'):
                raise ValueError(name)
            seed_all(seed); folder=split_out/name; folder.mkdir(exist_ok=True)
            print(f'[model] Loading {name}; native position interpolation in BOTH branches.',flush=True)
            model,anchor=build_models(cfg,device,name=='clip_gsd')
            dump(folder/'architecture.json',dict(backbone=str(model.vision.config),input_mode=cfg['input_mode'],
                basis=cfg['basis_method'],attention=cfg['attention'],all_generators_seen=True,patch_size=14,
                protocol=protocol_id(cfg),crop_max_side=cfg.get('crop_max_side'),
                padding='right/bottom edge replication to multiple of 14; no resize',
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
            try:
                train_model(model,anchor,train,val,cfg,seed,device,folder,signature)
                for case in processing_cases(cfg):
                    p,batches=predict(model,anchor,tests,cfg,seed,device,case)
                    f=tests[['sample_id','generator','label','pixel_hash','width','height']].assign(seed=seed,model=name,case=case,prob_fake=p,batch_id=batches)
                    f.to_csv(folder/f'predictions_{case}.csv',index=False); results.append(f)
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError('CUDA OOM under the configured input protocol. No automatic resize, skip or batch reduction. Use more VRAM or explicitly start a smaller-input/batch protocol; last.pt retains completed epochs.') from exc
            del model,anchor; gc.collect(); torch.cuda.empty_cache()
    predictions=pd.concat(results,ignore_index=True); predictions.to_csv(out/'predictions.csv',index=False)
    metrics=summarize(predictions,cfg,out)
    dump(out/'completion.json',dict(status='complete',smoke=cfg['smoke'],protocol=protocol_id(cfg),config_hash=digest(cfg)))
    print(metrics[metrics.generator=='pooled'].to_string(index=False),flush=True)
    return metrics


"""Trusted audit-manifest reuse. Loaded after runtime.py; never decode/hash image content.

The caller deliberately trusts the previous image audit and the same dataset version.
Filesystem stat checks cannot detect content changes that keep the same byte length.
"""
from pathlib import Path, PurePosixPath
import re

MANIFEST_PARTITIONS = ('train', 'validation', 'test')
MANIFEST_COLUMNS = {
    'sample_id', 'relative_path', 'generator', 'split', 'label', 'width', 'height',
    'file_hash', 'pixel_hash', 'file_bytes',
}


def find_manifest_root(cfg):
    def complete(root):
        return all((root / f'seed{seed}' / f'{part}_manifest.csv').is_file()
                   for seed in cfg['seeds'] for part in MANIFEST_PARTITIONS)

    if cfg['manifest_root']:
        root = Path(cfg['manifest_root']).resolve()
        if re.fullmatch(r'seed\d+', root.name):
            root = root.parent
        if not complete(root):
            raise FileNotFoundError(f'Missing saved train/validation/test manifests under {root}/seed<seed>. '
                                    'Attach the previous All7 output; no full-scan fallback is enabled.')
        return root
    search = Path(cfg['input_root'])
    roots = sorted({p.parent.parent.resolve() for p in search.rglob('train_manifest.csv')
                    if p.parent.name == f'seed{cfg["seeds"][0]}' and complete(p.parent.parent)})
    if len(roots) != 1:
        raise ValueError(f'Set manifest_root explicitly. Complete All7-manifest candidates: {[str(p) for p in roots]}. '
                         'Need seed42/train_manifest.csv, validation_manifest.csv and test_manifest.csv; '
                         'weights alone are not enough. No image scan was started.')
    return roots[0]


def validate_manifest_frame(frame, cfg, partition):
    missing = MANIFEST_COLUMNS - set(frame)
    if missing or frame.empty:
        raise ValueError(f'Invalid {partition} manifest: missing columns={sorted(missing)} or empty cohort')
    frame = frame.copy()
    if frame[list(MANIFEST_COLUMNS)].isna().any().any():
        raise ValueError(f'Null required metadata in {partition} manifest')
    if frame.sample_id.duplicated().any() or frame.relative_path.duplicated().any():
        raise ValueError(f'Duplicate IDs/paths in {partition} manifest')
    for column in ('sample_id', 'file_hash', 'pixel_hash'):
        if not frame[column].astype(str).str.fullmatch(r'[0-9a-f]{64}').all():
            raise ValueError(f'Invalid saved {column} in {partition} manifest')
    for column in ('label', 'width', 'height', 'file_bytes'):
        values = pd.to_numeric(frame[column], errors='coerce')
        if not np.isfinite(values).all() or (values != np.floor(values)).any():
            raise ValueError(f'Invalid integer {column} in {partition} manifest')
        frame[column] = values.astype('int64')
    if not frame.label.isin([0, 1]).all() or (frame[['width', 'height', 'file_bytes']] <= 0).any().any():
        raise ValueError(f'Invalid labels/dimensions/byte lengths in {partition} manifest')
    source_split = 'test' if partition == 'test' else 'train'
    if set(frame.split) != {source_split}:
        raise ValueError(f'{partition} manifest must originate only from source {source_split} folders')
    counts = frame.groupby(['generator', 'label']).size()
    expected = {(g, y) for g in cfg['train_generators'] for y in (0, 1)}
    if set(counts.index) != expected or counts.nunique() != 1:
        raise ValueError(f'{partition} manifest must contain balanced real/fake cohorts for all seven generators')
    if frame.duplicated(['generator', 'pixel_hash']).any():
        raise ValueError(f'Duplicate pixels within a generator in {partition} manifest')
    label_names = {'nature': 0, 'real': 0, '0_real': 0, 'ai': 1, 'fake': 1, '1_fake': 1}
    split_names = {'train': {'train', 'training'}, 'test': {'val', 'valid', 'validation', 'test'}}
    for row in frame.itertuples():
        rel = str(row.relative_path)
        parts = PurePosixPath(rel).parts
        if (not parts or len(parts) < 4 or PurePosixPath(rel).is_absolute() or
                '\\' in rel or ':' in rel or '..' in parts or rel != PurePosixPath(rel).as_posix()):
            raise ValueError(f'Unsafe relative image path: {rel}')
        if (canonical_generator(parts[0]) != row.generator or parts[1] not in split_names[source_split] or
                label_names.get(parts[2].lower()) != row.label):
            raise ValueError(f'Path/generator/split/label mismatch: {rel}')
        if row.sample_id != hashlib.sha256(rel.encode()).hexdigest():
            raise ValueError(f'Saved sample_id does not match relative path: {rel}')
    return frame


def read_manifest_cohorts(root, cfg, seed):
    folder = root / f'seed{seed}'
    summary_path = folder / 'split_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else None
    frames, sources = {}, {}
    for partition in MANIFEST_PARTITIONS:
        path = folder / f'{partition}_manifest.csv'
        data = path.read_bytes()  # Hash only the small CSV, NOT any source image.
        frame = pd.read_csv(io.BytesIO(data), dtype={k: str for k in
            ('sample_id', 'relative_path', 'generator', 'split', 'file_hash', 'pixel_hash')})
        frame = validate_manifest_frame(frame, cfg, partition)
        if summary is not None:
            if (summary.get(partition) != len(frame) or
                    summary.get(partition + '_hash') != digest(frame.sample_id.tolist())):
                raise ValueError(f'{path}: ordered cohort does not match saved split_summary.json')
        frames[partition] = frame
        sources[partition] = dict(path=str(path), csv_sha256=hashlib.sha256(data).hexdigest(),
                                  count=len(frame), ordered_ids_hash=digest(frame.sample_id.tolist()))
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if set(frames[a].sample_id) & set(frames[b].sample_id):
            raise ValueError(f'Saved IDs overlap across {a}/{b}')
        if set(frames[a].pixel_hash) & set(frames[b].pixel_hash):
            raise ValueError(f'Saved pixel hashes overlap across {a}/{b}')
    joined = pd.concat(list(frames.values()), ignore_index=True)
    if (joined.groupby('pixel_hash').label.nunique() > 1).any():
        raise ValueError('Conflicting saved labels for identical pixel hashes')
    return frames, sources


def resolve_manifest_dataset(frame, cfg):
    first = frame.relative_path.iloc[0]
    if cfg['dataset_root']:
        candidates = [Path(cfg['dataset_root'])]
    else:
        search = Path(cfg['input_root'])
        candidates = [search / 'tiny-genimage', search / 'datasets/yangsangtai/tiny-genimage']
        generator_dir = PurePosixPath(first).parts[0]
        print('[manifest] Locating dataset root using directory metadata only.', flush=True)
        candidates.extend(p.parent for p in search.rglob(generator_dir) if p.is_dir())
    matches = sorted({p.resolve() for p in candidates if (p / first).is_file()})
    if len(matches) != 1:
        raise ValueError(f'Set dataset_root to the directory containing generator folders. Candidates: '
                         f'{[str(p) for p in matches]}. No RGB/hash scan was started.')
    root = matches[0]
    resolved = frame.copy()
    paths = []
    for index, row in enumerate(resolved.itertuples()):
        path = (root / row.relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError(f'Image path escapes dataset root: {row.relative_path}')
        if not path.is_file() or path.stat().st_size != row.file_bytes:
            raise ValueError(f'Missing or changed file (path/size check): {path}. '
                             'Use the same dataset version as the saved audit.')
        paths.append(str(path))
        if (index + 1) % 5000 == 0:
            print(f'[manifest] {index + 1:,}/{len(frame):,} file paths/sizes checked; no image decode/hash.', flush=True)
    resolved['path'] = paths  # Ignore stale absolute /kaggle/input paths from the previous run.
    return resolved, root


def load_manifest_index(cfg, out):
    root = find_manifest_root(cfg)
    source_config_path = root / 'config.json'
    if source_config_path.is_file():
        source = json.loads(source_config_path.read_text(encoding='utf-8'))
        if source.get('smoke') and not cfg['smoke']:
            raise ValueError('Cannot use smoke-only source manifests for a full training run')
        if set(source.get('train_generators', [])) != set(cfg['train_generators']):
            raise ValueError('Source config is not an All7 experiment')
        if source.get('validation_fraction') != cfg['validation_fraction']:
            raise ValueError('validation_fraction differs from saved split; reused manifests are never re-split')
    all_frames, cohorts, sources = [], {}, {}
    for seed in cfg['seeds']:
        frames, provenance = read_manifest_cohorts(root, cfg, seed)
        all_frames.extend(frames.values())
        cohorts[str(seed)] = {part: f.sample_id.tolist() for part, f in frames.items()}
        sources[str(seed)] = provenance
    merged = pd.concat(all_frames, ignore_index=True)
    # A sample may occur in several seed experiments; metadata must agree.
    for column in sorted(MANIFEST_COLUMNS - {'sample_id'}):
        if (merged.groupby('sample_id')[column].nunique(dropna=False) > 1).any():
            raise ValueError(f'Conflicting saved {column} across seeds/cohorts')
    merged = merged.drop_duplicates('sample_id', keep='first').reset_index(drop=True)
    merged, dataset = resolve_manifest_dataset(merged, cfg)
    provenance = dict(mode='reuse_manifests', manifest_root=str(root), dataset_root=str(dataset),
                      sources=sources, unique_selected_images=len(merged),
                      checks=['schema', 'ordered_split_summary_if_present', 'saved_pixel_hash_leakage',
                              'label_and_split_path_consistency', 'file_exists', 'file_byte_length'],
                      skipped=['image_decode', 'image_content_hash', 'pixel_hash_recomputation',
                               'discovery_of_unlisted_or_new_images'],
                      trust='Saved audit metadata and unchanged dataset version are trusted. '
                            'Same-size content changes are NOT detected.')
    target = out / 'manifest_reuse.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != provenance:
        raise ValueError('Manifest/input fingerprint changed; choose a new output_root before resume')
    dump(target, provenance)
    save_csv(out / 'reused_manifest_index.csv', merged)
    merged.attrs['reused_cohorts'] = cohorts
    merged.attrs['manifest_root'] = str(root)
    print(f'[manifest] Reused {len(merged):,} selected images. No RGB decoding or image hashing; no re-split.', flush=True)
    return merged


def reuse_saved_splits(df, cfg, seed, out):
    cohorts = df.attrs.get('reused_cohorts', {}).get(str(seed))
    if cohorts is None:
        raise ValueError('Missing validated saved-cohort index; refusing to re-split or scan implicitly')
    indexed = df.set_index('sample_id', drop=False)
    frames = []
    for partition in MANIFEST_PARTITIONS:
        frame = indexed.loc[cohorts[partition]].reset_index(drop=True)
        frame.attrs = {}
        if cfg['smoke']:
            limit = cfg['smoke_train_per_class'] if partition == 'train' else cfg['smoke_eval_per_class']
            frame = frame[frame.groupby(['generator', 'label']).cumcount() < limit].reset_index(drop=True)
        save_csv(out / f'{partition}_manifest.csv', frame)
        frame.groupby(['generator', 'label']).size().rename('n').to_csv(out / f'{partition}_counts.csv')
        frames.append(frame)
    tr, va, te = frames
    dump(out / 'split_summary.json', dict(train=len(tr), validation=len(va), test=len(te),
         train_hash=digest(tr.sample_id.tolist()), validation_hash=digest(va.sample_id.tolist()),
         test_hash=digest(te.sample_id.tolist()), selection='Reused ordered audited All7 cohorts; no re-split',
         manifest_root=df.attrs['manifest_root'], smoke_subset=bool(cfg['smoke'])))
    print(f'[split/reused] train={len(tr):,}, validation={len(va):,}, test={len(te):,}; original ID order retained.', flush=True)
    return tr, va, te


"""All7 crop560 + label-independent blur/JPEG + fixed two-view validation.

Append after runtime.py and runtime_raw_all7.py in the standalone notebook.
The raw-only runtimes/notebooks remain unchanged.
"""
if 'GENS7' not in globals():
    from pathlib import Path
    _raw_file = Path(__file__).with_name('runtime_raw_all7.py')
    exec(compile(_raw_file.read_text(encoding='utf-8'), str(_raw_file), 'exec'), globals())

if 'load_manifest_index' not in globals():
    _manifest_file = Path(__file__).with_name('manifest_reuse.py')
    exec(compile(_manifest_file.read_text(encoding='utf-8'), str(_manifest_file), 'exec'), globals())

_raw_images = Images
_raw_validate = validate_raw_config
_raw_splits = make_splits
_raw_run = run
_full_scan = scan_dataset

VALIDATION_MONITOR = 'mean_native_fixed_processed_cross_entropy'
DEFAULT_CONFIG = dict(
    DEFAULT_CONFIG,
    output_root='/kaggle/working/gsd_all7_crop560_blur_jpeg_manifest_v2',
    data_mode='reuse_manifests', manifest_root=None,
    input_mode='center_crop_max_edge_pad14', crop_max_side=560,
    jpeg_probability=0.5, jpeg_train_quality=[75, 100], jpeg_subsampling=0,
    blur_probability=0.5, blur_sigma=[0.0, 2.0], horizontal_flip_probability=0.0,
    validation_processing_seed=42, validation_monitor=VALIDATION_MONITOR,
)


def protocol_id(cfg):
    return f"all7_center_crop{cfg['crop_max_side']}_blur_jpeg_fixedval_v2"


def validate_raw_config(cfg):
    # Reuse geometry, batch, evaluation-case and All7 safeguards, not the raw-only ban.
    _raw_validate(dict(cfg, jpeg_probability=0.0, blur_probability=0.0,
                       horizontal_flip_probability=0.0))
    if cfg['input_mode'] != 'center_crop_max_edge_pad14':
        raise ValueError('Blur/JPEG protocol requires the capped center-crop input mode')
    if cfg['horizontal_flip_probability'] != 0:
        raise ValueError('Only Gaussian blur and JPEG are enabled; flip must remain zero')
    for name in ('jpeg_probability', 'blur_probability'):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be a finite probability in [0, 1]')
    quality = cfg['jpeg_train_quality']
    if (not isinstance(quality, (list, tuple)) or len(quality) != 2 or
            any(isinstance(v, bool) or not isinstance(v, int) for v in quality) or
            not 1 <= quality[0] <= quality[1] <= 100):
        raise ValueError('jpeg_train_quality must be two ordered integers in [1, 100]')
    sigma = cfg['blur_sigma']
    if (not isinstance(sigma, (list, tuple)) or len(sigma) != 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in sigma) or
            not 0 <= sigma[0] <= sigma[1]):
        raise ValueError('blur_sigma must be two finite ordered nonnegative numbers')
    if isinstance(cfg['jpeg_subsampling'], bool) or cfg['jpeg_subsampling'] != 0:
        raise ValueError('This controlled protocol fixes JPEG subsampling=0 (4:4:4)')
    if isinstance(cfg['validation_processing_seed'], bool) or not isinstance(cfg['validation_processing_seed'], int):
        raise ValueError('validation_processing_seed must be an integer')
    if cfg['validation_monitor'] != VALIDATION_MONITOR:
        raise ValueError('Early stopping must monitor equal-weight native/processed validation CE')
    if cfg['data_mode'] not in ('reuse_manifests', 'full_scan'):
        raise ValueError('data_mode must be reuse_manifests or explicit full_scan')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Need a nonempty list of distinct run seeds')


def scan_dataset(cfg, out):
    if cfg['data_mode'] == 'reuse_manifests':
        return load_manifest_index(cfg, out)
    return _full_scan(cfg, out)


def processing_parameters(sample_id, cfg, seed, train=False, epoch=0):
    """Local RNG: independent of labels, row order, workers and global random state.

    Validation deliberately ignores epoch; training changes its draws each epoch.
    The two Bernoulli gates are independent, including when an operation is disabled.
    """
    identity = ['blur_jpeg_v1', int(seed), str(sample_id)]
    identity += ['train', int(epoch)] if train else ['validation', cfg['validation_processing_seed']]
    rng = random.Random(int(digest(identity)[:16], 16))
    blur_applied = rng.random() < cfg['blur_probability']
    jpeg_applied = rng.random() < cfg['jpeg_probability']
    sigma = rng.uniform(*cfg['blur_sigma'])
    quality = rng.randint(*cfg['jpeg_train_quality'])
    return dict(blur_applied=blur_applied, blur_sigma=sigma if blur_applied else None,
                jpeg_applied=jpeg_applied, jpeg_quality=quality if jpeg_applied else None)


def apply_processing(image, parameters, cfg):
    # Operate on the decoded original, BEFORE crop/padding/normalization.
    if parameters['blur_applied']:
        image = image.filter(ImageFilter.GaussianBlur(parameters['blur_sigma']))
    if parameters['jpeg_applied']:
        image = jpeg(image, parameters['jpeg_quality'], cfg['jpeg_subsampling'])
    return image


class Images(_raw_images):
    def __init__(self, frame, cfg, seed, train=False, epoch=0, case='native'):
        super().__init__(frame, cfg, seed, train, epoch, case)
        self.seed, self.epoch = seed, epoch

    def __getitem__(self, i):
        if not self.train and self.case != 'processed':
            # Native/test cases remain byte-for-byte equivalent to the raw evaluator.
            return super().__getitem__(i)
        row = self.rows[i]
        with Image.open(row['path']) as im:
            image = im.convert('RGB')
        parameters = processing_parameters(row['sample_id'], self.cfg, self.seed,
                                           train=self.train, epoch=self.epoch)
        image = apply_processing(image, parameters, self.cfg)
        geom = input_geometry(*image.size, self.cfg)
        image = image.crop((geom['crop_left'], geom['crop_top'],
                            geom['crop_left'] + geom['crop_width'],
                            geom['crop_top'] + geom['crop_height']))
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        h, w = x.shape[-2:]
        x = F.pad(x.unsqueeze(0), (0, (-w) % 14, 0, (-h) % 14), mode='replicate').squeeze(0)
        mean = x.new_tensor([.48145466, .4578275, .40821073])[:, None, None]
        std = x.new_tensor([.26862954, .26130258, .27577711])[:, None, None]
        return (x - mean) / std, int(row['label']), i


def save_csv(path, frame):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temp, index=False)
    temp.replace(path)


def make_splits(df, cfg, seed, out):
    if cfg['data_mode'] == 'reuse_manifests':
        train, val, tests = reuse_saved_splits(df, cfg, seed, out)
    else:
        train, val, tests = _raw_splits(df, cfg, seed, out)
    columns = [c for c in ('sample_id', 'generator', 'label', 'pixel_hash') if c in val]
    plan = pd.concat([
        val[columns].reset_index(drop=True),
        pd.DataFrame([processing_parameters(sid, cfg, seed) for sid in val.sample_id]),
    ], axis=1)
    plan['batch_id'] = np.arange(len(plan)) // cfg['test_batch']
    plan['run_seed'] = seed
    plan['validation_processing_seed'] = cfg['validation_processing_seed']
    save_csv(out / 'validation_processing_plan.csv', plan)
    dump(out / 'validation_protocol.json', dict(
        cases=['native', 'processed'], monitor=VALIDATION_MONITOR,
        weights=dict(native=0.5, processed=0.5),
        processing_order=['decode_rgb', 'blur', 'jpeg', 'center_crop', 'edge_pad14', 'normalize'],
        validation_processing_seed=cfg['validation_processing_seed'], run_seed=seed,
        plan_hash=digest(plan.to_dict('records')),
        source='validation split from train folders only; official test excluded',
        batching='same fixed ID order and membership in both validation views',
        loss='mean per-image cross entropy computed directly from logits',
    ))
    print(f'[validation] Fixed native + processed views; plan: {out / "validation_processing_plan.csv"}', flush=True)
    return train, val, tests


@torch.no_grad()
def predict(model, anchor, frame, cfg, seed, device, case, return_loss=False):
    if len(frame) == 0:
        raise ValueError('Cannot evaluate an empty cohort')
    model.eval()
    result = np.full(len(frame), np.nan)
    batches = np.full(len(frame), -1, dtype=np.int64)
    loss_sum, count = 0.0, 0
    for bid, (images, labels, indices) in enumerate(loader(frame, cfg, seed, case=case)):
        # Both branches see exactly the SAME transformed tensors; no second RNG draw.
        bases = estimate_bases(anchor, images, cfg, device, model.selected) if anchor is not None else None
        for start, x in image_chunks(images, cfg['micro_batch']):
            with amp(cfg, device):
                logits = model(x.to(device), bases)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f'Nonfinite {case} logits')
            ix = indices[start:start + len(x)].numpy()
            result[ix] = logits.float().softmax(-1)[:, 1].cpu().numpy()
            batches[ix] = bid
            if return_loss:
                loss_sum += float(F.cross_entropy(
                    logits.float(), labels[start:start + len(x)].to(device), reduction='sum'))
                count += len(x)
        if bid % 25 == 0:
            print(f'[{case}] batch {bid + 1}/{math.ceil(len(frame) / cfg["test_batch"])}', flush=True)
    if not np.isfinite(result).all() or (batches < 0).any():
        raise ValueError('Missing/nonfinite prediction')
    if return_loss:
        if count != len(frame) or not math.isfinite(loss_sum):
            raise FloatingPointError('Incomplete/nonfinite validation loss')
        return result, batches, loss_sum / count
    return result, batches


def evaluate_validation(model, anchor, val, cfg, seed, device):
    summaries, predictions = {}, {}
    for case in ('native', 'processed'):
        p, batches, loss = predict(model, anchor, val, cfg, seed, device, case, return_loss=True)
        summaries[case] = dict(loss=loss, **metric(val.label.to_numpy(), p, cfg['threshold']))
        predictions[case] = (p, batches)
    if not np.array_equal(predictions['native'][1], predictions['processed'][1]):
        raise ValueError('Validation batch membership changed across views')
    monitor = (summaries['native']['loss'] + summaries['processed']['loss']) / 2.0
    return monitor, summaries, predictions


def train_model(model, anchor, train, val, cfg, seed, device, folder, signature):
    groups = [dict(params=model.vision.parameters(), lr=cfg['lr']),
              dict(params=model.head.parameters(), lr=cfg['head_lr'])]
    if len(model.gsd):
        groups.append(dict(params=model.gsd.parameters(), lr=cfg['modulation_lr']))
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg['betas']), eps=cfg['eps'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and cfg['precision'] == 'fp16')
    stopper = EarlyStopping(cfg['early_stopping'], cfg['patience'], cfg['min_delta'])
    latest = folder / 'last.pt'
    start, best, history = 0, float('inf'), []
    if cfg['resume'] and latest.exists():
        ck = torch.load(latest, map_location='cpu', weights_only=False)
        if ck['signature'] != signature or ck.get('protocol') != protocol_id(cfg) or ck.get('config') != cfg:
            raise ValueError('Resume protocol/config/manifests changed; use a new output_root')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        stopper.state.update(ck['early_stopping'])
        start, best, history = ck['epoch'] + 1, ck['best'], ck['history']
        del ck
        print(f'[resume] next epoch {start + 1}, stopped={stopper.state["stopped"]}', flush=True)
    for epoch in range(start, cfg['epochs']):
        if stopper.state['stopped']:
            break
        model.train()
        running, count, skipped = 0.0, 0, 0
        tick = time.monotonic()
        for bid, (images, labels, _) in enumerate(loader(train, cfg, seed, True, epoch)):
            bases = estimate_bases(anchor, images, cfg, device, model.selected) if anchor is not None else None
            optimizer.zero_grad(set_to_none=True)
            for a, x in image_chunks(images, cfg['micro_batch']):
                with amp(cfg, device):
                    logits = model(x.to(device), bases)
                    loss = F.cross_entropy(logits.float(), labels[a:a + len(x)].to(device), reduction='sum') / len(images)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss).backward()
                running += float(loss.detach()) * len(images)
            old = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old)
            count += len(images)
            if bid % 10 == 0:
                print(f'[epoch {epoch + 1}] batch {bid + 1}/{math.ceil(len(train) / cfg["train_batch"])}; loss={running / count:.5f}', flush=True)
        if count == 0:
            raise ValueError('Empty training cohort')
        vloss, summaries, predictions = evaluate_validation(model, anchor, val, cfg, seed, device)
        stop = stopper.update(vloss)
        entry = dict(epoch=epoch + 1, train_loss=running / count, val_loss=vloss,
                     val_native_loss=summaries['native']['loss'], val_processed_loss=summaries['processed']['loss'],
                     validation_monitor=VALIDATION_MONITOR,
                     val_native=summaries['native'], val_processed=summaries['processed'],
                     seconds=time.monotonic() - tick, amp_skipped_steps=skipped, train_images=count,
                     modulation={l: dict(a=float(g.a.detach()), b=float(g.b.detach())) for l, g in model.gsd.items()},
                     early_stopping=dict(stopper.state))
        history.append(entry)
        print(entry, flush=True)
        state = dict(model=model.state_dict(), epoch=epoch, signature=signature, config=cfg,
                     protocol=protocol_id(cfg), validation=summaries, val_loss=vloss)
        if vloss < best:
            best = vloss
            save_checkpoint(folder / 'best.pt', state)
        save_checkpoint(latest, dict(state, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                                     best=best, history=history, early_stopping=dict(stopper.state)))
        dump(folder / 'history.json', history)
        dump(folder / 'training_status.json', dict(epoch=epoch + 1, max_epochs=cfg['epochs'],
             validation_monitor=VALIDATION_MONITOR, val_loss=vloss, early_stopping=dict(stopper.state)))
        rows = [dict(epoch=e['epoch'], case=case, monitor_loss=e['val_loss'], **e['val_' + case])
                for e in history for case in ('native', 'processed')]
        save_csv(folder / 'validation_metrics.csv', pd.DataFrame(rows))
        columns = [c for c in ('sample_id', 'generator', 'label', 'pixel_hash') if c in val]
        frame = val[columns].copy()
        for case, (p, batches) in predictions.items():
            frame['prob_fake_' + case] = p
            frame['batch_id_' + case] = batches
        save_csv(folder / f'validation_predictions_epoch{epoch + 1:03d}.csv', frame)
        if stop:
            print('[early stopping] Restore best mean native/processed validation CE checkpoint.', flush=True)
            break
    ck = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    dump(folder / 'selected_checkpoint.json', dict(epoch=ck['epoch'] + 1,
         criterion=VALIDATION_MONITOR, val_loss=ck['val_loss'], validation=ck['validation'], signature=signature))


def run(config=None):
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    validate_raw_config(cfg)
    print('[protocol] All7 crop560; train blur/JPEG resampled per epoch, SAME tensors for anchor/detector.', flush=True)
    print('[protocol] Fixed native/processed validation; early stopping = 0.5 * CE(native) + 0.5 * CE(processed).', flush=True)
    print('[protocol] New pretrained run only; do not reuse raw-only checkpoint/output directories.', flush=True)
    print(f'[protocol] Data mode: {cfg["data_mode"]}; manifests may be reused, detector weights are NOT reused.', flush=True)
    return _raw_run(cfg)


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


def score_details(s_row, hw, r):
    s = s_row.float()
    h, w = hw
    p = torch.softmax(r * s, 0)
    k = max(1, int(round(0.05 * s.numel())))
    am = int(torch.argmax(s))
    return dict(s_max=float(s.max()), s_mean=float(s.mean()), s_top5pct_mean=float(s.topk(k).values.mean()),
                argmax_row=am // w, argmax_col=am % w,
                w_entropy=float(-(p * torch.log(p.clamp_min(1e-12))).sum()),
                frac_patch_gt05=float((s > 0).float().mean()))


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
            rows[ix] = dict(meta, **score_details(s[0], hw, cfg['lse_r']))
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
            y, p = g.label.to_numpy(), g.prob_fake.to_numpy()
            rows.append(dict(run=cfg['run_name'], case=case, generator=gen, **metric(y, p, cfg['threshold']),
                             **extra_metrics(y, p)))
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
                               **score_details(s[0], hw, cfg['lse_r'])))
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
        for pool, col in (('lse_full', 'prob_fake'), ('lse_window', 'prob_fake_window')):
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


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
    _cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    try:
        log('[exit]', capd_main(_cfg))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
