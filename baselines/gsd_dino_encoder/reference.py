"""Fit a reusable per-layer top-k right-singular basis from real training images."""
import argparse
from collections import Counter, defaultdict
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ImageRows, read_manifest, write_manifest
from .model import clip_tokens, load_clip
from .utils import atomic_save, load_config, sha256, state_hash


class GramAccumulator:
    """Uncentered P.T@P, not covariance; bounded memory, FP64 accumulation on CPU."""
    def __init__(self, dim):
        self.gram = torch.zeros(dim, dim, dtype=torch.float64)
        self.tokens = 0

    @torch.no_grad()
    def add(self, tokens):
        p = tokens.detach().reshape(-1, tokens.shape[-1])
        if not torch.isfinite(p).all():
            raise FloatingPointError('Nonfinite reference features')
        # GEMM follows input dtype: reference command uses FP32 with TF32 disabled.
        self.gram.add_((p.T @ p).double().cpu())
        self.tokens += len(p)

    def basis(self, k):
        if self.tokens < k or not 1 <= k <= len(self.gram):
            raise ValueError('Not enough reference tokens for k')
        values, vectors = torch.linalg.eigh((self.gram + self.gram.T) * 0.5)
        order = torch.argsort(values, descending=True)
        values, vectors = values[order], vectors[:, order]
        if values[k-1] <= max(1e-12, float(values[0]) * 1e-10):
            raise ValueError('Reference has insufficient numerical rank for k')
        basis = vectors[:, :k].float()
        # Fix arbitrary column signs for stable serialization.
        pivots = basis.abs().argmax(0)
        signs = basis[pivots, torch.arange(k)].sign()
        return basis * signs, values[:k].tolist()


def reference_rows(rows, count, seed):
    if not 1 <= count <= len(rows):
        raise ValueError('reference-size must be between 1 and training row count')
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row.get('source') or 'unknown'].append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        for source in sorted(groups):
            if groups[source] and len(selected) < count:
                selected.append(groups[source].pop())
    return selected


def load_basis(path):
    artifact = torch.load(path, map_location='cpu', weights_only=False)
    if artifact.get('format') != 'gsd-dino-fixed-basis-v1':
        raise ValueError('Unsupported basis artifact')
    if artifact.get('layers') != [21, 22, 23, 24] or artifact.get('k') != 64 or artifact.get('dim') != 1024:
        raise ValueError('Basis does not match agreed CLIP+GSD architecture')
    if set(artifact['bases']) != {'21', '22', '23', '24'}:
        raise ValueError('Incomplete basis layers')
    return artifact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    p.add_argument('--clip', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--pixel-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-size', type=int, default=4096)
    p.add_argument('--tokens-per-image', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.output.exists() or Path(str(args.output)+'.reference.csv').exists():
        raise FileExistsError('Choose a fresh basis artifact path')
    if args.batch_size < 1 or args.workers < 0 or args.tokens_per_image < 1:
        raise ValueError('Invalid reference options')
    config = load_config(args.config)
    rows = reference_rows(read_manifest(args.manifest), args.reference_size, config['seed'])
    print(f'[reference] load pretrained CLIP; selected {len(rows)} real images', flush=True)
    vision = load_clip(args.clip).requires_grad_(False).eval()
    clip_hash = state_hash(vision)
    vision.to(args.device)
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
    dim = vision.config.hidden_size
    accumulators = {layer: GramAccumulator(dim) for layer in config['layers']}
    worker_options = {'multiprocessing_context': 'spawn'} if args.workers else {}
    loader = DataLoader(ImageRows(rows, args.pixel_root, config['global_size']), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False, **worker_options)
    generator = torch.Generator().manual_seed(config['seed'])
    seen = 0
    with torch.inference_mode():
        for images, ids in loader:
            tokens = clip_tokens(vision, images.to(args.device))
            # Identical spatial selection across layers for each reference batch.
            indices = torch.randperm(tokens.shape[1]-1, generator=generator)[:args.tokens_per_image]
            indices = indices.to(args.device)
            for layer, block in enumerate(vision.vision_model.encoder.layers, 1):
                tokens = block(tokens, None, None)[0]
                if layer in accumulators:
                    accumulators[layer].add(tokens[:, 1:][:, indices].float())
            seen += len(images)
            print(f'[reference] {seen}/{len(rows)} images; no gradients, no per-batch model updates', flush=True)
    bases, eigenvalues = {}, {}
    for layer, accumulator in accumulators.items():
        bases[str(layer)], eigenvalues[str(layer)] = accumulator.basis(config['k'])
    sidecar = Path(str(args.output) + '.reference.csv')
    write_manifest(rows, sidecar)
    artifact = {'format': 'gsd-dino-fixed-basis-v1', 'bases': bases, 'eigenvalues': eigenvalues,
                'layers': config['layers'], 'k': config['k'], 'dim': dim,
                'clip_state_sha256': clip_hash, 'training_manifest_sha256': sha256(args.manifest),
                'reference_manifest_sha256': sha256(sidecar), 'reference_ids': [r['sample_id'] for r in rows],
                'reference_sources': dict(Counter(r.get('source', 'unknown') for r in rows)),
                'seed': config['seed'], 'tokens_per_image': min(args.tokens_per_image, (config['global_size']//14)**2),
                'reference_count': len(rows), 'method': 'uncentered_streaming_gram_eigh',
                'preprocessing': {'size': config['global_size'], 'policy': 'CLIP_resize_short_side_center_crop_EXIF_RGB'},
                'vision_config': vision.config.to_dict()}
    atomic_save(artifact, args.output)
    print(f'[reference] saved immutable bases to {args.output}', flush=True)


if __name__ == '__main__':
    main()
