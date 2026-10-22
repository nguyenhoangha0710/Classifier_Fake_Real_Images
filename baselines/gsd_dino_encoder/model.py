"""Pretrained CLIP token path with immutable, channel-wise GSD bases."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint


def load_clip(path):
    from transformers import CLIPVisionModel
    vision = CLIPVisionModel.from_pretrained(path, local_files_only=True)
    c = vision.config
    if (c.hidden_size, c.num_hidden_layers, c.patch_size, c.image_size) != (1024, 24, 14, 224):
        raise ValueError('Expected local pretrained CLIP ViT-L/14, image_size=224')
    return vision


def clip_tokens(vision, images, mask=None, mask_token=None):
    """Keep CLIP's learned positions; interpolate only for a different patch grid."""
    emb = vision.vision_model.embeddings
    patch = emb.patch_embedding(images.to(emb.patch_embedding.weight.dtype))
    height, width = patch.shape[-2:]
    tokens = patch.flatten(2).transpose(1, 2)
    if mask is not None:
        if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError('Mask must be boolean [B,N] for this patch grid')
        tokens = torch.where(mask.unsqueeze(-1), mask_token.to(tokens.dtype), tokens)
    cls = emb.class_embedding.to(tokens.dtype).expand(len(images), 1, -1)
    tokens = torch.cat((cls, tokens), dim=1)
    positions = emb.position_embedding.weight
    base = math.isqrt(positions.shape[0] - 1)
    if base * base != positions.shape[0] - 1:
        raise ValueError('Pretrained CLIP positions do not form a square grid')
    if (height, width) != (base, base):
        spatial = positions[1:].float().T.reshape(1, -1, base, base)
        spatial = F.interpolate(spatial, size=(height, width), mode='bicubic', align_corners=False)
        positions = torch.cat((positions[:1], spatial.flatten(2).squeeze(0).T.to(positions.dtype)))
    return vision.vision_model.pre_layrnorm(tokens + positions.to(tokens.dtype))


class FixedGSD(nn.Module):
    def __init__(self, basis, lambda_max=1.0):
        super().__init__()
        basis = basis.detach().float().clone()
        if basis.ndim != 2 or not torch.isfinite(basis).all():
            raise ValueError('Invalid basis')
        gram = basis.T @ basis
        if not torch.allclose(gram, torch.eye(len(gram), device=gram.device), atol=2e-4, rtol=2e-4):
            raise ValueError('Basis columns must be orthonormal')
        if not 0 < lambda_max <= 1:
            raise ValueError('lambda_max must be in (0,1]')
        self.register_buffer('basis', basis)
        self.a = nn.Parameter(torch.tensor(2.0))
        self.b = nn.Parameter(torch.tensor(1.0))
        self.lambda_max = lambda_max

    def forward(self, tokens):
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            features = tokens.float()
            parallel = (features @ self.basis) @ self.basis.T
            occupancy = parallel.square().sum((1, 2)) / features.square().sum((1, 2)).clamp_min(1e-12)
            strength = self.lambda_max * torch.sigmoid(self.a * occupancy + self.b)
            result = features - strength[:, None, None] * parallel
        return result.to(tokens.dtype)


class GSDEncoder(nn.Module):
    def __init__(self, vision, bases, lambda_max=1.0, activation_checkpointing=False):
        super().__init__()
        self.vision = vision
        self.dim = vision.config.hidden_size
        self.patch_size = vision.config.patch_size
        self.activation_checkpointing = activation_checkpointing
        depth = len(vision.vision_model.encoder.layers)
        if not bases or any(not 1 <= int(layer) <= depth for layer in bases):
            raise ValueError('Invalid GSD layer IDs')
        if any(basis.shape[0] != self.dim for basis in bases.values()):
            raise ValueError('Basis/CLIP hidden dimensions do not match')
        self.gsd = nn.ModuleDict({str(layer): FixedGSD(basis, lambda_max) for layer, basis in bases.items()})
        self.mask_token = nn.Parameter(torch.zeros(self.dim))

    def forward(self, images, mask=None):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('Expected image tensor [B,3,H,W]')
        if any(size % self.patch_size for size in images.shape[-2:]):
            raise ValueError('Image dimensions must be multiples of patch size')
        tokens = clip_tokens(self.vision, images, mask, self.mask_token)
        for layer, block in enumerate(self.vision.vision_model.encoder.layers, 1):
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(lambda x, block=block: block(x, None, None)[0],
                                    tokens, use_reentrant=False)
            else:
                tokens = block(tokens, None, None)[0]
            if str(layer) in self.gsd:
                tokens = self.gsd[str(layer)](tokens)
        tokens = self.vision.vision_model.post_layernorm(tokens)
        return {'cls': tokens[:, 0], 'patches': tokens[:, 1:]}


class PrototypeHead(nn.Module):
    """DINOv2-style three-layer GELU MLP + normalized bottleneck + weight-norm logits."""
    def __init__(self, dim, prototypes, hidden, bottleneck):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, bottleneck))
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        self.last_layer = weight_norm(nn.Linear(bottleneck, prototypes, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def forward(self, x):
        # Wide logits, normalization and target balancing deliberately stay FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.normalize(self.mlp(x.float()), dim=-1, eps=1e-12)
            return self.last_layer(x)


class StudentNetwork(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.dino_head = PrototypeHead(encoder.dim, config['dino_prototypes'], config['head_hidden'], config['dino_bottleneck'])
        self.ibot_head = PrototypeHead(encoder.dim, config['ibot_prototypes'], config['head_hidden'], config['ibot_bottleneck'])

    def forward(self, global_images, local_images, masks=None):
        global_out = self.encoder(global_images, masks)
        # Teacher has no local views. Student local outputs still use the same backbone/head.
        cls = global_out['cls']
        if local_images is not None and len(local_images):
            cls = torch.cat((cls, self.encoder(local_images)['cls']), dim=0)
        selected = global_out['patches'][masks] if masks is not None else global_out['patches'].flatten(0, 1)
        return {'cls': global_out['cls'], 'dino': self.dino_head(cls), 'ibot': self.ibot_head(selected)}

    @torch.no_grad()
    def teacher_forward(self, global_images, masks):
        out = self.encoder(global_images)  # Always unmasked, select target positions only AFTER encoding.
        return {'dino': self.dino_head(out['cls']), 'ibot': self.ibot_head(out['patches'][masks])}


def make_teacher(student):
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    teacher.encoder.activation_checkpointing = False
    return teacher


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    if not 0 <= momentum <= 1:
        raise ValueError('Invalid teacher momentum')
    source = dict(student.named_parameters())
    for name, param in teacher.named_parameters():
        param.mul_(momentum).add_(source[name], alpha=1 - momentum)
    # Immutable basis buffers are never EMA-updated or refitted.
