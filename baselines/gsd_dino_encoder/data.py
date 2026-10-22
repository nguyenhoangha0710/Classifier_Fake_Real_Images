"""Real-only manifests, DINO multi-crops and deterministic block masking."""
import argparse
import csv
import hashlib
import math
import os
import random
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from .utils import sha256, write_json

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def read_manifest(path, real_only=True):
    with open(path, newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Empty manifest')
    identities, paths = set(), set()
    for row in rows:
        pixel = row.get('relative_path') or row.get('image_path')
        if not pixel:
            raise ValueError('Manifest needs image_path or relative_path')
        sid = row.get('sample_id') or row.get('image_id') or pixel
        if sid in identities or pixel in paths:
            raise ValueError('Duplicate identity/path: ' + sid)
        identities.add(sid); paths.add(pixel)
        row['sample_id'] = sid
        if real_only and row.get('label', 'real').strip().lower() not in ('real', '0', '0.0'):
            raise ValueError('Non-real label in training/reference manifest: ' + sid)
    return rows


def image_path(row, root):
    # Preserve the project's exact, non-recursive COCO/ImageNet Kaggle rebasing.
    from baselines.gsd_forensic_twins.data import pixel_path
    return pixel_path(row, root)


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def prepare(parent, root, output, min_side=224, audit='metadata'):
    if min_side < 1:
        raise ValueError('min_side must be positive')
    if audit not in ('metadata', 'header'):
        raise ValueError('Unknown audit mode: ' + audit)
    rows = read_manifest(parent)
    accepted, rejected, resolved = [], [], set()
    metadata_count = header_count = 0
    for i, row in enumerate(rows, 1):
        path = image_path(row, root)
        # Lexical duplicate detection: no per-file stat/resolve on remote mounts.
        canonical = os.path.normcase(os.path.abspath(path))
        if canonical in resolved:
            raise ValueError('Two rows resolve to the same pixel file: ' + canonical)
        resolved.add(canonical)
        try:
            width, height = float(row.get('width', '')), float(row.get('height', ''))
            valid_size = all(math.isfinite(v) and v > 0 and v.is_integer() for v in (width, height))
        except (TypeError, ValueError):
            valid_size = False
        if audit == 'metadata' and valid_size:
            width, height = int(width), int(height)
            metadata_count += 1
        else:
            # Only missing/invalid dimensions require image I/O by default.
            with Image.open(path) as image:
                width, height = image.size
            header_count += 1
        # min(width,height) is invariant under EXIF rotation.
        row = dict(row, width=width, height=height)
        if min(width, height) < min_side:
            rejected.append({'sample_id': row['sample_id'], 'reason': 'small', 'width': width, 'height': height})
        else:
            accepted.append(row)
        if i % 1000 == 0 or i == len(rows):
            print(f'[prepare] {i}/{len(rows)} kept={len(accepted)} small={len(rejected)} '
                  f'metadata={metadata_count} headers={header_count}', flush=True)
    if not accepted:
        raise ValueError('No eligible real images')
    write_manifest(accepted, output)
    meta = {'parent_sha256': sha256(parent), 'manifest_sha256': sha256(output),
            'parent_count': len(rows), 'eligible_count': len(accepted), 'rejected': rejected,
            'min_side': min_side, 'pixel_root': str(Path(root).resolve()), 'audit': audit,
            'metadata_count': metadata_count, 'header_count': header_count}
    write_json(meta, str(output) + '.json')
    return meta


class GaussianBlur:
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, image):
        if random.random() < self.probability:
            return image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2.0)))
        return image


class MultiCrop:
    def __init__(self, config):
        self.local_crops = config['local_crops']
        def geometric(size, scale):
            return T.Compose([T.RandomResizedCrop(size, scale=tuple(scale), interpolation=InterpolationMode.BICUBIC),
                              T.RandomHorizontalFlip(p=0.5)])
        colors = T.Compose([T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                            T.RandomGrayscale(p=0.2)])
        normalize = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
        self.global1 = T.Compose([geometric(config['global_size'], config['global_scale']), colors,
                                  GaussianBlur(1.0), normalize])
        self.global2 = T.Compose([geometric(config['global_size'], config['global_scale']), colors,
                                  GaussianBlur(0.1), T.RandomSolarize(128, p=0.2), normalize])
        self.local = T.Compose([geometric(config['local_size'], config['local_scale']), colors,
                                GaussianBlur(0.5), normalize])

    def __call__(self, image):
        return [self.global1(image), self.global2(image)], [self.local(image) for _ in range(self.local_crops)]


def eval_transform(size=224):
    return T.Compose([T.Resize(size, interpolation=InterpolationMode.BICUBIC), T.CenterCrop(size),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


class RealViews(Dataset):
    def __init__(self, rows, root, config):
        self.rows, self.root, self.seed, self.epoch = rows, root, config['seed'], 0
        self.transform = MultiCrop(config)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        seed = int(hashlib.sha256(f"{self.seed}:{self.epoch}:{row['sample_id']}".encode()).hexdigest()[:15], 16)
        previous = random.getstate()
        try:
            random.seed(seed)
            with torch.random.fork_rng(devices=[]):
                # Seed CPU transforms only; torch.manual_seed would also reseed CUDA
                # when num_workers=0 after the training model has moved to GPU.
                torch.random.default_generator.manual_seed(seed)
                with Image.open(image_path(row, self.root)) as image:
                    image = ImageOps.exif_transpose(image).convert('RGB')
                    global_views, local_views = self.transform(image)
        finally:
            random.setstate(previous)
        return {'global': global_views, 'local': local_views, 'sample_id': row['sample_id'], 'seed': seed}


def block_mask(grid, target, rng):
    """DINO/iBOT-style random rectangles; may stop below target if no rectangle fits."""
    height, width = grid
    mask = torch.zeros(height, width, dtype=torch.bool)
    count = 0
    while count < target:
        added = 0
        for _ in range(10):
            area = rng.uniform(min(4, target - count), target - count)
            aspect = math.exp(rng.uniform(math.log(0.3), math.log(1 / 0.3)))
            h, w = round(math.sqrt(area * aspect)), round(math.sqrt(area / aspect))
            if 0 < h < height and 0 < w < width:
                top, left = rng.randrange(height - h + 1), rng.randrange(width - w + 1)
                rectangle = mask[top:top+h, left:left+w]
                added = h * w - int(rectangle.sum())
                if 0 < added <= target - count:
                    rectangle.fill_(True)
                    break
                added = 0
        if not added:
            break
        count += added
    return mask.flatten()


class CollateViews:
    def __init__(self, config, patch_size=14):
        self.config, self.patch_size = config, patch_size

    def __call__(self, samples):
        b = len(samples)
        global_images = torch.cat([torch.stack([s['global'][g] for s in samples]) for g in range(2)])
        local_images = torch.cat([torch.stack([s['local'][l] for s in samples])
                                  for l in range(self.config['local_crops'])])
        grid = tuple(size // self.patch_size for size in global_images.shape[-2:])
        n = math.prod(grid)
        rng = random.Random(sum(s['seed'] for s in samples))
        mask_count = max(1, int(2 * b * self.config['mask_probability']))
        indices = list(range(2 * b)); rng.shuffle(indices)
        masks = torch.zeros(2 * b, n, dtype=torch.bool)
        lo, hi = self.config['mask_ratio']
        for j, index in enumerate(indices[:mask_count]):
            ratio = rng.uniform(lo + (hi-lo)*j/mask_count, lo + (hi-lo)*(j+1)/mask_count)
            masks[index] = block_mask(grid, max(1, int(n * ratio)), rng)
        if not masks.any():
            raise ValueError('No masked tokens; check global grid/mask settings')
        return {'global': global_images, 'local': local_images, 'masks': masks,
                'sample_ids': [s['sample_id'] for s in samples]}


class ImageRows(Dataset):
    def __init__(self, rows, root, size=224):
        self.rows, self.root, self.transform = rows, root, eval_transform(size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(image_path(row, self.root)) as image:
            image = self.transform(ImageOps.exif_transpose(image).convert('RGB'))
        return image, row['sample_id']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit/filter a user-designated real training pool')
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--pixel-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-side', type=int, default=224)
    parser.add_argument('--audit', choices=['metadata', 'header'], default='metadata',
                        help='Use saved width/height; header fallback only if dimensions are missing')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prepare(args.parent, args.pixel_root, args.output, args.min_side, args.audit)
