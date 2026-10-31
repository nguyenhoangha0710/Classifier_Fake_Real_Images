"""B-Free paired data layer for GSD training on ONE fake branch (torch-free).

Default branch: SD2.1_inpainted_samecat_origBG versus COCO_real_512 of the SAME COCO ID.
Everything here is deterministic and depends only on (seed, COCO ID, virtual epoch),
never on labels, worker order or global RNG state.
"""
import concurrent.futures
import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

BFREE_REAL = 'COCO_real_512'
BFREE_ALL_BRANCHES = (
    'SD2.1_selfconditioned', 'SD2.1_selfconditioned_origBG',
    'SD2.1_inpainted_samecat', 'SD2.1_inpainted_samecat_origBG',
    'SD2.1_inpainted_diffcat', 'SD2.1_inpainted_diffcat_origBG',
)
BFREE_TRAIN_LIST_MD5 = '06581da53bf81741501396d71eb7ec5f'
BFREE_VALID_LIST_MD5 = '72e7b4be2d049029dbedca7e75af91b0'
BFREE_TRAIN_LIST_COUNT, BFREE_VALID_LIST_COUNT = 51517, 10304
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# Keys that only affect where/how long a session runs; excluded from the run identity.
VOLATILE_KEYS = ('input_root', 'bfree_root', 'output_root', 'resume_from', 'time_budget_hours',
                 'num_workers', 'audit_workers', 'save_every_steps', 'log_every_steps',
                 'validation_time_reserve_minutes')


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    temp.replace(path)


def atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temp, index=False)
    temp.replace(path)


def run_identity_hash(cfg):
    return sha({k: v for k, v in cfg.items() if k not in VOLATILE_KEYS})


# ----------------------------------------------------------------------------- dataset discovery
def candidate_mounts(input_root, max_depth=3):
    """Directories that may be a dataset root; never descend into image folders."""
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f'Missing input_root {root}')
    found, pending = [], [(root, 0)]
    skip = {BFREE_REAL, *BFREE_ALL_BRANCHES}
    while pending:
        folder, depth = pending.pop()
        found.append(folder)
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in folder.iterdir() if p.is_dir() and not p.is_symlink())
        except PermissionError:
            continue
        pending.extend((p, depth + 1) for p in children if p.name not in skip)
    return found


def find_bfree_root(cfg):
    if cfg.get('bfree_root'):
        candidates = [Path(cfg['bfree_root'])]
    else:
        candidates = candidate_mounts(cfg['input_root'], max_depth=4)
    matches = [p for p in candidates if (p / 'train_list.csv').is_file() and (p / 'valid_list.csv').is_file()
               and (p / BFREE_REAL).is_dir()]
    if len(matches) != 1:
        raise ValueError('Set bfree_root to the folder containing train_list.csv, valid_list.csv and '
                         f'{BFREE_REAL}. Found: {[str(p) for p in matches]}')
    return matches[0]


def read_bfree_list(path, expected_md5, expected_count, check_md5=True):
    raw = Path(path).read_bytes()
    checksum = hashlib.md5(raw).hexdigest()
    if check_md5 and checksum != expected_md5:
        raise ValueError(f'{path}: MD5 {checksum} differs from the official B-Free list')
    frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    if 'id' not in frame.columns:
        raise ValueError(f'{path}: missing id column')
    if expected_count is not None and len(frame) != expected_count:
        raise ValueError(f'{path}: expected {expected_count} rows, found {len(frame)}')
    if frame.id.duplicated().any() or not frame.id.str.fullmatch(r'\d{12}').all():
        raise ValueError(f'{path}: duplicated or invalid 12-digit COCO IDs')
    return frame, checksum


def bfree_folder(root, name, example_ids):
    """Single (root/name) or double-nested (root/name/name) layout, decided on several IDs."""
    root = Path(root)
    choices = []
    for folder in (root / name / name, root / name):
        hits = sum((folder / f'{i}.png').is_file() for i in example_ids)
        if hits:
            choices.append((hits, folder))
    if not choices:
        raise FileNotFoundError(f'Cannot find {name}/<id>.png (single or double nested) under {root}')
    choices.sort(key=lambda t: -t[0])
    if len(choices) > 1 and choices[0][0] == choices[1][0]:
        raise ValueError(f'Ambiguous layout for {name}: {[str(c[1]) for c in choices]}')
    return choices[0][1]


# ----------------------------------------------------------------------------- pair audit / mask
def load_rgb(path, side):
    with Image.open(path) as image:
        if image.format != 'PNG':
            raise ValueError(f'not PNG ({image.format})')
        if side is not None and image.size != (side, side):
            raise ValueError(f'size {image.size} != {side}x{side}')
        return np.asarray(image.convert('RGB'), dtype=np.uint8)


def pair_statistics(real_path, fake_path, side, diff_threshold):
    """Inpainted-region proxy from the paired difference. origBG pastes the real background back,
    so pixels outside the regenerated object should (almost) equal the real image."""
    try:
        real = load_rgb(real_path, side)
        fake = load_rgb(fake_path, side)
        if real.shape != fake.shape:
            raise ValueError(f'shape mismatch {real.shape} vs {fake.shape}')
        diff = np.abs(fake.astype(np.int16) - real.astype(np.int16)).max(axis=2)
        mask = diff > diff_threshold
        if mask.any():
            ys, xs = np.nonzero(mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        else:
            bbox = [0, 0, 0, 0]
        mask_pixels, total_pixels = int(mask.sum()), int(mask.size)
        return dict(ok=True, error='', height=int(real.shape[0]), width=int(real.shape[1]),
                    changed_any_fraction=float((diff > 0).mean()),
                    mask_pixels=mask_pixels, total_pixels=total_pixels,
                    mask_fraction=mask_pixels / total_pixels, mean_abs_diff=float(diff.mean()),
                    bbox_x0=bbox[0], bbox_y0=bbox[1], bbox_x1=bbox[2], bbox_y1=bbox[3],
                    identical=bool(not diff.any()))
    except Exception as exc:  # recorded and counted, never silently used for training
        return dict(ok=False, error=f'{type(exc).__name__}: {exc}', height=-1, width=-1,
                    changed_any_fraction=float('nan'), mask_pixels=-1, total_pixels=-1,
                    mask_fraction=float('nan'),
                    mean_abs_diff=float('nan'), bbox_x0=-1, bbox_y0=-1, bbox_x1=-1, bbox_y1=-1,
                    identical=False)


def audit_pairs(ids, real_dir, fake_dir, cfg, cohort, log_every=5000):
    ids = list(ids)
    rows = [None] * len(ids)
    start = time.monotonic()

    def work(i):
        coco_id = ids[i]
        real_path, fake_path = Path(real_dir) / f'{coco_id}.png', Path(fake_dir) / f'{coco_id}.png'
        if not real_path.is_file() or not fake_path.is_file():
            stats = pair_statistics(real_path, fake_path, None, 0)  # produces the error row
            stats['error'] = 'missing ' + ('real' if not real_path.is_file() else 'fake')
            stats['ok'] = False
        else:
            stats = pair_statistics(real_path, fake_path, cfg['image_side'], cfg['mask_diff_threshold'])
        return i, dict(coco_id=coco_id, cohort=cohort, real_path=str(real_path), fake_path=str(fake_path), **stats)

    workers = max(1, int(cfg.get('audit_workers', 8)))
    done = 0
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for i, row in pool.map(work, range(len(ids)), chunksize=64):
            rows[i] = row
            done += 1
            if done % log_every == 0 or done == len(ids):
                rate = done / max(time.monotonic() - start, 1e-9)
                print(f'[audit:{cohort}] {done:,}/{len(ids):,} pairs; {rate:.0f} pairs/s', flush=True)
    return pd.DataFrame(rows)


def restore_audit_types(audit):
    """CSV round trip: rebuild booleans and the EXACT mask fraction from integer pixel counts."""
    audit = audit.copy()
    audit['coco_id'] = audit.coco_id.astype(str).str.zfill(12)
    for column in ('ok', 'eligible', 'identical'):
        if column in audit:
            audit[column] = audit[column].astype(str).str.lower() == 'true'
    pixels = pd.to_numeric(audit.mask_pixels, errors='coerce')
    total = pd.to_numeric(audit.total_pixels, errors='coerce')
    audit['mask_pixels'], audit['total_pixels'] = pixels, total
    audit['mask_fraction'] = np.where(audit.ok & (total > 0), pixels / total.where(total > 0, 1), np.nan)
    audit['exclusion_reason'] = audit.exclusion_reason.fillna('').astype(str)
    return audit


def mark_eligibility(audit, cfg):
    audit = audit.copy()
    reason = np.full(len(audit), '', dtype=object)
    reason[~audit.ok.to_numpy(bool)] = 'unreadable_or_missing'
    ok = audit.ok.to_numpy(bool)
    frac = audit.mask_fraction.to_numpy(float)
    reason[ok & (frac < cfg['min_mask_fraction'])] = 'mask_too_small_or_identical'
    reason[ok & (frac > cfg['max_mask_fraction'])] = 'mask_too_large_not_origBG_like'
    audit['exclusion_reason'] = reason
    audit['eligible'] = audit.exclusion_reason == ''
    return audit


def area_tertile_edges(fractions):
    q = np.quantile(np.asarray(fractions, float), [1 / 3, 2 / 3])
    return [float(q[0]), float(q[1])]


def area_group(fractions, edges):
    f = np.asarray(fractions, float)
    return np.where(f <= edges[0], 'small', np.where(f <= edges[1], 'medium', 'large'))


# ----------------------------------------------------------------------------- splits / batches
def id_rank(seed, purpose, coco_id):
    return int(sha([str(purpose), int(seed), str(coco_id)])[:16], 16)


def make_bfree_splits(train_audit, cfg, seed):
    """Validation stratified by mask-area tertile; the rest permuted ONCE into virtual epochs."""
    eligible = train_audit[train_audit.eligible].copy()
    if eligible.coco_id.duplicated().any():
        raise ValueError('Duplicate COCO IDs in the eligible train pool')
    edges = area_tertile_edges(eligible.mask_fraction)
    eligible['area_group'] = area_group(eligible.mask_fraction, edges)
    n_val = int(cfg['val_ids'])
    if n_val < 3 or n_val >= len(eligible):
        raise ValueError(f'val_ids={n_val} incompatible with {len(eligible)} eligible IDs')
    eligible['val_rank'] = [id_rank(seed, 'validation', i) for i in eligible.coco_id]
    quotas = {g: n_val // 3 + (1 if k < n_val % 3 else 0) for k, g in enumerate(('small', 'medium', 'large'))}
    val_parts = []
    for group, quota in quotas.items():
        members = eligible[eligible.area_group == group].sort_values('val_rank')
        if len(members) < quota:
            raise ValueError(f'Not enough {group} IDs for validation')
        val_parts.append(members.head(quota))
    val = pd.concat(val_parts)
    # Fixed evaluation order independent of the stratification.
    val = val.sort_values('val_rank').reset_index(drop=True)
    pool = eligible[~eligible.coco_id.isin(val.coco_id)].copy()
    pool['train_rank'] = [id_rank(seed, 'train_order', i) for i in pool.coco_id]
    pool = pool.sort_values('train_rank').reset_index(drop=True)
    ve_ids = int(cfg['ve_ids'])
    if ve_ids < 1 or ve_ids % cfg['pairs_per_batch']:
        raise ValueError('ve_ids must be a positive multiple of pairs_per_batch')
    n_ve = min(int(cfg['max_virtual_epochs']), len(pool) // ve_ids)
    if n_ve < 1:
        raise ValueError(f'Train pool {len(pool)} smaller than one virtual epoch ({ve_ids})')
    pool['virtual_epoch'] = -1
    pool.loc[:n_ve * ve_ids - 1, 'virtual_epoch'] = np.repeat(np.arange(n_ve), ve_ids)
    summary = dict(eligible=len(eligible), validation=len(val), train_pool=len(pool),
                   virtual_epochs=n_ve, ids_per_virtual_epoch=ve_ids,
                   images_per_virtual_epoch=2 * ve_ids, unused_reserve_ids=int((pool.virtual_epoch < 0).sum()),
                   area_tertile_edges=edges,
                   validation_area_groups=val.area_group.value_counts().to_dict(),
                   validation_order_hash=sha(val.coco_id.tolist()),
                   train_order_hash=sha(pool.coco_id.tolist()))
    if set(val.coco_id) & set(pool.coco_id):
        raise AssertionError('validation/train overlap')
    return val, pool, summary


def pair_rows(frame):
    """Interleave real/fake per ID: row 2k is real (label0), row 2k+1 is fake (label1)."""
    rows = []
    for r in frame.itertuples(index=False):
        rows.append(dict(coco_id=r.coco_id, label=0, path=r.real_path))
        rows.append(dict(coco_id=r.coco_id, label=1, path=r.fake_path))
    return rows


def pair_batches(n_ids, pairs_per_batch):
    """Logical batches of row indices: each batch = `pairs_per_batch` complete real/fake pairs."""
    if n_ids % pairs_per_batch:
        raise ValueError('Number of IDs must be a multiple of pairs_per_batch (complete pairs only)')
    return [list(range(2 * s, 2 * (s + pairs_per_batch))) for s in range(0, n_ids, pairs_per_batch)]


# ----------------------------------------------------------------------------- processing
def pair_processing_parameters(coco_id, cfg, seed, train=False, virtual_epoch=0):
    """Same blur/JPEG draw for the real AND fake image of one COCO ID.
    Train: redrawn every virtual epoch. Validation: fixed by validation_processing_seed."""
    identity = ['bfree_pair_blur_jpeg_v1', int(seed), str(coco_id)]
    identity += ['train', int(virtual_epoch)] if train else ['validation', int(cfg['validation_processing_seed'])]
    rng = random.Random(int(sha(identity)[:16], 16))
    blur_applied = rng.random() < cfg['blur_probability']
    jpeg_applied = rng.random() < cfg['jpeg_probability']
    sigma = rng.uniform(*cfg['blur_sigma'])
    quality = rng.randint(*cfg['jpeg_train_quality'])
    return dict(blur_applied=blur_applied, blur_sigma=sigma if blur_applied else None,
                jpeg_applied=jpeg_applied, jpeg_quality=quality if jpeg_applied else None)


def jpeg_roundtrip(image, quality, subsampling):
    # Byte-for-byte the same call as runtime.jpeg used by GSD-aug-Tiny.
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=int(quality), subsampling=subsampling,
               optimize=False, progressive=False)
    buf.seek(0)
    with Image.open(buf) as decoded:
        return decoded.convert('RGB')


def apply_pair_processing(image, parameters, cfg):
    # Same order as GSD-aug-Tiny: decoded RGB -> blur -> JPEG, BEFORE crop/pad/normalize.
    if parameters['blur_applied']:
        image = image.filter(ImageFilter.GaussianBlur(parameters['blur_sigma']))
    if parameters['jpeg_applied']:
        image = jpeg_roundtrip(image, parameters['jpeg_quality'], cfg['jpeg_subsampling'])
    return image


def crop_geometry(width, height, crop_max_side):
    # Identical to runtime_raw_all7.input_geometry for center_crop_max_edge_pad14.
    cw, ch = min(width, crop_max_side), min(height, crop_max_side)
    left, top = (width - cw) // 2, (height - ch) // 2
    pw, ph = ((cw + 13) // 14) * 14, ((ch + 13) // 14) * 14
    return dict(crop_left=left, crop_top=top, crop_width=cw, crop_height=ch,
                input_width=pw, input_height=ph, tokens=(pw // 14) * (ph // 14) + 1)


def load_processed_crop(path, cfg, parameters=None):
    """Decoded RGB (+ optional pair processing) cropped to <=crop_max_side; pad/normalize happen in torch."""
    with Image.open(path) as source:
        if source.format != 'PNG':
            raise ValueError(f'{path}: expected PNG')
        image = source.convert('RGB')
    if parameters is not None:
        image = apply_pair_processing(image, parameters, cfg)
    g = crop_geometry(*image.size, cfg['crop_max_side'])
    return image.crop((g['crop_left'], g['crop_top'], g['crop_left'] + g['crop_width'], g['crop_top'] + g['crop_height']))


# ----------------------------------------------------------------------------- resume discovery
def find_previous_outputs(cfg, run_hash, exclude=()):
    """Previous sessions' outputs attached as Input: folders with run_state.json of the same run."""
    exclude = {os.path.normcase(os.path.abspath(p)) for p in exclude}
    matches = []
    for folder in candidate_mounts(cfg['input_root'], max_depth=5):
        if os.path.normcase(os.path.abspath(folder)) in exclude:
            continue
        state_path = folder / 'run_state.json'
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding='utf-8'))
            except Exception:
                continue
            if state.get('run_hash') == run_hash:
                matches.append((int(state.get('images_seen', 0)), str(state.get('updated', '')), folder, state))
    matches.sort(key=lambda t: (t[0], t[1]))
    return matches
