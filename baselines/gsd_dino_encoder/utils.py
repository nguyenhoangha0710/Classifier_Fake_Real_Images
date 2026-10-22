import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def load_config(path):
    base = json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
    supplied = json.loads(Path(path).read_text(encoding='utf-8'))
    unknown = set(supplied) - set(base)
    if unknown:
        raise ValueError('Unknown config keys: ' + ', '.join(sorted(unknown)))
    base.update(supplied)
    if base['k'] != 64 or base['layers'] != [21, 22, 23, 24]:
        raise ValueError('Production configuration requires last-four-layer GSD and k=64')
    if base['global_size'] % 14 or base['local_size'] % 14:
        raise ValueError('Crop sizes must be multiples of patch size 14')
    if base['local_crops'] < 1 or base['epochs'] < 1:
        raise ValueError('Need local views and positive epoch count')
    if not 0 < base['mask_probability'] <= 1:
        raise ValueError('mask_probability must be in (0,1]')
    if not 0 < base['mask_ratio'][0] <= base['mask_ratio'][1] < 1:
        raise ValueError('Invalid mask ratio interval')
    for key in ('dino_prototypes', 'ibot_prototypes', 'head_hidden', 'dino_bottleneck', 'ibot_bottleneck'):
        if base[key] < 2:
            raise ValueError('Invalid ' + key)
    if not 0 <= base['teacher_momentum'] <= base['teacher_momentum_end'] <= 1:
        raise ValueError('Invalid EMA schedule')
    for key in ('warmup_fraction', 'teacher_temperature_warmup_fraction'):
        if not 0 <= base[key] < 1:
            raise ValueError('Invalid ' + key)
    for key in ('lr_encoder', 'lr_gsd', 'lr_heads', 'lr_mask', 'student_temperature',
                'teacher_temperature_start', 'teacher_temperature', 'gradient_clip'):
        if base[key] <= 0:
            raise ValueError('Must be positive: ' + key)
    return base


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
