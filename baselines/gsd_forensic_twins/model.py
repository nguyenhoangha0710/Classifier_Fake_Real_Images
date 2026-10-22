"""Native Twins + adaptive, token-level GSD. No projected CLIP embeddings."""
import copy

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint


def full_precision(x):
    return torch.autocast(device_type=x.device.type, enabled=False)


@torch.no_grad()
def semantic_basis(patches, k, mode, progress=None):
    """FP32 uncentered SVD; train scope is one global view, never both views."""
    with full_precision(patches):
        p = patches.float()
        if not torch.isfinite(p).all():
            # Global-mode validation is handled collectively below.
            if mode != 'batch' or not dist.is_initialized():
                raise FloatingPointError('Nonfinite semantic tokens')
        if mode == 'sample':
            if k > min(p.shape[-2:]):
                raise ValueError('K exceeds per-sample matrix rank bound')
            _, s, vh = torch.linalg.svd(p, full_matrices=False)
            v = vh[:, :k].transpose(-1, -2).contiguous()
        elif mode == 'batch':
            if dist.is_initialized():
                if progress: progress(f'GSD gather frozen patches: local_shape={tuple(p.shape)}')
                parts = [torch.empty_like(p) for _ in range(dist.get_world_size())]
                dist.all_gather(parts, p.contiguous())
                p = torch.cat(parts, 0)
            p = p.reshape(-1, p.shape[-1])
            v = torch.empty(p.shape[-1], k, device=p.device)
            gap = torch.zeros((), device=p.device)
            status = torch.ones((), device=p.device)
            error = None
            if not dist.is_initialized() or dist.get_rank() == 0:
                try:
                    if not torch.isfinite(p).all() or k > min(p.shape):
                        raise ValueError('Invalid global semantic matrix')
                    if progress: progress(f'GSD SVD START: matrix={tuple(p.shape)}, dtype={p.dtype}')
                    _, s, vh = torch.linalg.svd(p, full_matrices=False)
                    v.copy_(vh[:k].T)
                    gap.copy_(s[k-1] / s[k].clamp_min(1e-12) if len(s) > k else torch.tensor(float('nan'), device=p.device))
                    if progress: progress('GSD SVD DONE')
                except Exception as exc:
                    status.zero_()
                    error = exc
            if dist.is_initialized():
                if progress: progress('GSD broadcast basis/status')
                dist.broadcast(status, 0)
            if not status.item():
                raise RuntimeError('Global-view SVD failed on rank0') from error
            if dist.is_initialized():
                dist.broadcast(v, 0)
                dist.broadcast(gap, 0)
            return v.detach(), gap.detach()
        else:
            raise ValueError('basis mode must be batch or sample')
        gap = (s[..., k-1] / s[..., k].clamp_min(1e-12)).mean() if s.shape[-1] > k else s.new_tensor(float('nan'))
        return v.detach(), gap.detach()


class AdaptiveGSD(nn.Module):
    def __init__(self, lambda_max=1.):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(2.))
        self.b = nn.Parameter(torch.tensor(1.))
        self.lambda_max = lambda_max

    def forward(self, f, v):
        dtype = f.dtype
        with full_precision(f):
            f = f.float()
            parallel = (f @ v) @ v.transpose(-1, -2)
            r = parallel.square().sum((1, 2)) / f.square().sum((1, 2)).clamp_min(1e-12)
            lam = self.lambda_max * torch.sigmoid(self.a * r + self.b)
            clean = f - lam[:, None, None] * parallel
        stats = {'occupancy': r.detach().mean(), 'lambda_mean': lam.detach().mean(),
                 'lambda_min': lam.detach().min(), 'lambda_max': lam.detach().max(),
                 'a': self.a.detach(), 'b': self.b.detach()}
        return clean.to(dtype), stats


class TwinsGSD(nn.Module):
    def __init__(self, vision, layers=(21, 22, 23, 24), k=64, hidden=2048,
                 output=8192, bypass=False, activation_checkpointing=False):
        super().__init__()
        self.ef = vision
        self.es = copy.deepcopy(vision).requires_grad_(False).eval()
        self.layers = tuple(layers)
        if k < 1:
            raise ValueError('K must be positive')
        depth = len(vision.vision_model.encoder.layers)
        if len(set(layers)) != len(layers) or any(l < 1 or l > depth for l in layers):
            raise ValueError('Invalid GSD insertion layers')
        self.k, self.bypass = k, bypass
        self.activation_checkpointing = activation_checkpointing
        self.gsd = nn.ModuleDict({str(l): AdaptiveGSD() for l in layers})
        dim = vision.config.hidden_size
        self.projector = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, output))
        self.stats = {}
        self.trace = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        from transformers import CLIPVisionModel
        vision = CLIPVisionModel.from_pretrained(path, local_files_only=True)
        c = vision.config
        if (c.hidden_size, c.num_hidden_layers, c.patch_size, c.image_size) != (1024, 24, 14, 224):
            raise ValueError('Expected CLIP ViT-L/14 at 224 resolution')
        return cls(vision, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.es.eval()
        return self

    def encode(self, x, basis_mode='sample', prefix='eval'):
        if self.trace: self.trace(f'{prefix}: embedding input={tuple(x.shape)}')
        ef = self.ef.vision_model
        f = ef.pre_layrnorm(ef.embeddings(x))
        if not self.bypass:
            with torch.no_grad():
                es = self.es.vision_model
                g = es.pre_layrnorm(es.embeddings(x))
        for l, block in enumerate(ef.encoder.layers, 1):
            if self.trace and (l==1 or l%6==0 or l in self.layers):
                self.trace(f'{prefix}: Ef/Es block{l}')
            if self.activation_checkpointing and self.training:
                f = checkpoint(lambda t, block=block: block(t, None, None)[0], f, use_reentrant=False)
            else:
                f = block(f, None, None)[0]
            if not self.bypass:
                with torch.no_grad():
                    g = es.encoder.layers[l-1](g, None, None)[0]
                if l in self.layers:
                    callback = (lambda text: self.trace(f'{prefix}/layer{l}: {text}')) if self.trace else None
                    v, gap = semantic_basis(g[:, 1:], self.k, basis_mode, callback)
                    f, stats = self.gsd[str(l)](f, v)
                    self.stats.update({f'{prefix}/l{l}/{key}': value for key, value in stats.items()})
                    self.stats[f'{prefix}/l{l}/singular_gap'] = gap
        # Explicit Twins readout convention: LN each token, then patch mean.
        return ef.post_layernorm(f)[:, 1:].mean(1)

    def forward(self, x1, x2):
        self.stats = {}
        z1 = self.encode(x1, 'batch', 'view1')
        z2 = self.encode(x2, 'batch', 'view2')
        if self.trace: self.trace('Shared projector forward')
        # Barlow's wide head and small-batch normalization can amplify gradients.
        # Keep the projector in FP32 as well as the correlation, not just the loss.
        with full_precision(z1):
            e1, e2 = self.projector(z1.float()), self.projector(z2.float())
        return z1, z2, e1, e2


def barlow_loss(e1, e2, weight=5e-5, eps=1e-4):
    with full_precision(e1):
        if dist.is_initialized():
            from torch.distributed.nn.functional import all_gather
            e1 = torch.cat(all_gather(e1), 0)
            e2 = torch.cat(all_gather(e2), 0)
        x, y = e1.float(), e2.float()
        if x.shape[0] < 2:
            raise ValueError('Barlow statistics require at least 2 samples')
        x = (x-x.mean(0)) / (x.var(0, unbiased=False)+eps).sqrt()
        y = (y-y.mean(0)) / (y.var(0, unbiased=False)+eps).sqrt()
        c = x.T @ y / len(x)
        diagonal = c.diagonal()
        on = (diagonal-1).square().sum()
        # Avoid cancellation from subtracting the diagonal squared sum.
        off = c.flatten()[:-1].view(c.shape[0]-1, c.shape[0]+1)[:, 1:].square().sum()
        return on + weight*off, {'diagonal_loss': on.detach(), 'offdiagonal_loss': off.detach()}
