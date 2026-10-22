"""Deterministic native RGB pair sampling, independent of DataLoader workers."""
import csv
import hashlib
import random
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

MEAN = torch.tensor([.48145466, .4578275, .40821073])[:, None, None]
STD = torch.tensor([.26862954, .26130258, .27577711])[:, None, None]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty manifest: '+str(path))
    return rows


@lru_cache(maxsize=32)
def mounted_root(root, kind, split):
    """Known Kaggle mounts only; never scan directories or change cohorts."""
    root = Path(root)
    if kind == 'coco':
        candidates = [root/'datasets/awsaf49/coco-2017-dataset/coco2017'/split,
                      root/'coco-2017-dataset/coco2017'/split,
                      root/'coco-2017-dataset'/split, root/'coco2017'/split]
    else:
        candidates = [root/'competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train',
                      root/'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train']
    found = [p for p in candidates if p.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f'Expected exactly one {kind}/{split} mount. Checked: {candidates}; found: {found}')
    return found[0]


def pixel_path(row, root):
    p = Path(row.get('relative_path') or row['image_path'])
    parts = p.as_posix().split('/')
    # Old absolute paths in the saved manifest need rebasing to current mounts.
    # Only apply this translation within Kaggle Input; ordinary local paths stay intact.
    if str(root).replace('\\','/').rstrip('/') == '/kaggle/input':
        if row.get('source') == 'coco2017':
            split = next((s for s in ('train2017','unlabeled2017','val2017') if s in parts), None)
            if split:
                return mounted_root(str(root),'coco',split)/p.name
        if row.get('source') == 'imagenet1k' and 'train' in parts:
            i = parts.index('train')
            return mounted_root(str(root),'imagenet','train').joinpath(*parts[i+1:])
    return p if p.is_absolute() else Path(root)/p


def boxes(w, h, size, rng):
    axes = [axis for axis, length in [(0,w),(1,h)] if length >= 2*size]
    if min(w,h) < size or not axes:
        raise ValueError('Image cannot hold two native disjoint crops')
    axis = rng.choice(axes)
    length = (w,h)[axis]
    cut = rng.randint(size, length-size)
    if axis == 0:
        a = (rng.randint(0,cut-size), rng.randint(0,h-size))
        b = (rng.randint(cut,w-size), rng.randint(0,h-size))
    else:
        a = (rng.randint(0,w-size), rng.randint(0,cut-size))
        b = (rng.randint(0,w-size), rng.randint(cut,h-size))
    result = [(x,y,x+size,y+size) for x,y in (a,b)]
    if rng.random()<.5:
        result.reverse()
    return result


def tensor(image):
    return (torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).permute(2,0,1)/255-MEAN)/STD


class NativePairs(Dataset):
    def __init__(self, rows, root, seed=42, size=224):
        self.rows, self.root, self.seed, self.size, self.epoch = rows, root, seed, size, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sid = row.get('sample_id') or row.get('image_id') or str(pixel_path(row,self.root))
        seed = int(hashlib.sha256(f'{self.seed}:{self.epoch}:{sid}'.encode()).hexdigest()[:16],16)
        rng = random.Random(seed)
        with Image.open(pixel_path(row,self.root)) as src:
            image = src.convert('RGB')
        methods = (None,Image.Transpose.ROTATE_90,Image.Transpose.ROTATE_180,Image.Transpose.ROTATE_270)
        rot = methods[rng.randrange(4)]
        if rot is not None:
            image = image.transpose(rot)
        for op in (Image.Transpose.FLIP_LEFT_RIGHT,Image.Transpose.FLIP_TOP_BOTTOM):
            if rng.random()<.5:
                image = image.transpose(op)
        a,b = boxes(*image.size,self.size,rng)
        return tensor(image.crop(a)),tensor(image.crop(b)),sid,torch.tensor([a,b])


def prepare_manifest(parent, root, destination, pair, audit='metadata', log_every=1000, progress=None):
    """Fast metadata filtering by default; optional explicit header/full pixel audit."""
    rows = read_rows(parent)
    if audit not in ('metadata','header','full'): raise ValueError('Unknown audit mode')
    if log_every < 1: raise ValueError('log_every must be positive')
    began = time.monotonic()
    def report(text):
        if progress: progress.stage(text)
        else: print('[prepare] '+text, flush=True)
    report(f'Loaded {len(rows):,} rows; audit={audit}; no recursive path search')
    # Catch broken mount/path mappings before the expensive loop.
    for i in sorted({0, len(rows)-1, *range(min(16,len(rows))),
                     *(j*len(rows)//16 for j in range(16))}):
        path = pixel_path(rows[i],root)
        if not path.is_file(): raise FileNotFoundError(f'Input path preflight row={i}: {path}')
    report('Sampled input path checks passed; filtering saved dimensions')
    ids, accepted, rejected = set(), [], []
    header_reads = 0
    for index,row in enumerate(rows,1):
        sid = row.get('sample_id') or row.get('image_id') or str(pixel_path(row,root))
        if sid in ids:
            raise ValueError('Duplicate sample identity: '+sid)
        ids.add(sid)
        if row.get('label', 'real').lower() not in ('real','0','0.0'):
            raise ValueError('Non-real row in training/density manifest: '+sid)
        try:
            w,h = int(row.get('width','')),int(row.get('height',''))
        except (ValueError,TypeError):
            w,h = 0,0
        if audit != 'metadata' or min(w,h)<=0:
            path = pixel_path(row,root)
            if progress:
                progress.stage(f'read image {index}/{len(rows)}: {path}',emit=False)
            if index==1:
                report(f'First pixel-header read ({audit}): {path}')
            with Image.open(path) as im:
                w,h = im.size
                if audit == 'full': im.verify()
            header_reads += 1
        if min(w,h)>=224 and (not pair or max(w,h)>=448):
            accepted.append(row)
        else:
            rejected.append(sid)
        if index%log_every==0 or index==len(rows):
            elapsed = time.monotonic()-began
            rate = index/max(elapsed,.001)
            report(f'{index:,}/{len(rows):,} | kept={len(accepted):,} rejected={len(rejected):,} '
                   f'header_reads={header_reads:,} | {rate:.1f} rows/s | ETA={(len(rows)-index)/rate:.0f}s')
    if not accepted:
        raise ValueError('No eligible images')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report('Writing eligible manifest atomically')
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(accepted)
    os.replace(temporary,destination)
    return {'parent_sha256':sha256(parent),'derived_sha256':sha256(destination),
            'parent_count':len(rows),'eligible_count':len(accepted),'rejected_ids':rejected,
            'audit':audit,'header_reads':header_reads,'seconds':time.monotonic()-began,
            'pixel_root':str(Path(root).resolve()),'pair':pair}
