"""Core model components for the provisional Forensic Twins reproduction.

Published components are kept separate from implementation assumptions. The
paper specifies an 8-channel frozen FRE, a 6-block ViT with patch size 8,
2D-RoPE, a 192-D CLS representation, an 8192-D projector output and Barlow
Twins. Head count, MLP ratio and projector internals are provisional here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedConv2d(nn.Module):
    """Prediction-error filters used by the official FSD FRE."""

    def __init__(self, in_channels: int = 1, out_channels: int = 8, kernel_size: int = 15):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.xavier_normal_(weight, gain=1.0 / 3.0)
        self.w = nn.Parameter(weight)
        center = torch.zeros(kernel_size * kernel_size)
        center[kernel_size * kernel_size // 2] = 1.0
        self.register_buffer("one_middle", center, persistent=True)

    def constrained_weight(self, dtype: torch.dtype) -> torch.Tensor:
        weight = self.w.to(dtype=dtype)
        center = self.one_middle.to(dtype=dtype)
        flat = weight.reshape(-1, self.kernel_size * self.kernel_size)
        flat = flat - flat.mean(dim=1, keepdim=True)
        flat = flat + 1.0 / (self.kernel_size * self.kernel_size - 1)
        scale = (flat * (1.0 - center)).sum(dim=1, keepdim=True)
        if torch.any(scale == 0):
            raise FloatingPointError("Degenerate constrained FRE filter with zero off-center sum.")
        flat = flat / scale
        flat = flat * (1.0 - center)
        return flat.reshape_as(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.constrained_weight(x.dtype)
        return F.conv2d(x, weight, padding=self.kernel_size // 2)


class ForensicResidualExtractor(nn.Module):
    """Frozen 1->8 FRE: residual = grayscale - constrained prediction."""

    def __init__(self, kernel_size: int = 15):
        super().__init__()
        self.conv = ConstrainedConv2d(1, 8, kernel_size)

    @property
    def radius(self) -> int:
        return self.conv.kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.conv(x)

    def freeze(self) -> "ForensicResidualExtractor":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    @classmethod
    def from_pretrained(cls, checkpoint: Path | str, map_location: str = "cpu") -> "ForensicResidualExtractor":
        checkpoint = Path(checkpoint)
        try:
            state = torch.load(checkpoint, map_location=map_location, weights_only=True)
        except TypeError:  # PyTorch < 2.0
            state = torch.load(checkpoint, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported FRE checkpoint object: {type(state)!r}")

        def remove_prefix(value: str, prefix: str) -> str:
            return value[len(prefix) :] if value.startswith(prefix) else value

        normalized: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            key = remove_prefix(remove_prefix(key, "module."), "fre.")
            normalized[key] = value
        raw_weight = normalized.get("w", normalized.get("conv.w"))
        if raw_weight is None or raw_weight.ndim != 4:
            raise KeyError("FRE checkpoint must contain 'w' or 'conv.w'.")
        out_channels, in_channels, kh, kw = raw_weight.shape
        if (out_channels, in_channels, kh, kw) != (8, 1, 15, 15):
            raise ValueError(f"Expected FRE weight [8,1,15,15], got {tuple(raw_weight.shape)}")

        model = cls(kernel_size=kh)
        conv_state = {
            remove_prefix(key, "conv."): value
            for key, value in normalized.items()
            if key in {"w", "one_middle"} or key.startswith("conv.")
        }
        missing, unexpected = model.conv.load_state_dict(conv_state, strict=False)
        if "w" in missing:
            raise RuntimeError(f"Could not load FRE weights from {checkpoint}")
        if unexpected:
            raise RuntimeError(f"Unexpected FRE keys: {unexpected}")
        return model.freeze()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _apply_rope_1d(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x [B, heads, tokens, dim], dim must be even."""
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError(f"RoPE dimension must be even, got {dim}")
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(dtype=x.dtype)[None, None]
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(dtype=x.dtype)[None, None]
    return x * cos + _rotate_half(x) * sin


def apply_2d_rope(x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Apply 2D RoPE to patch tokens; leave CLS unchanged."""
    cls, patches = x[:, :, :1], x[:, :, 1:]
    if patches.shape[2] != grid_h * grid_w:
        raise ValueError("Token count does not match the supplied patch grid.")
    head_dim = patches.shape[-1]
    if head_dim % 4:
        raise ValueError("2D RoPE requires head_dim divisible by 4.")
    half = head_dim // 2
    yy, xx = torch.meshgrid(
        torch.arange(grid_h, device=x.device),
        torch.arange(grid_w, device=x.device),
        indexing="ij",
    )
    part_x = _apply_rope_1d(patches[..., :half], xx.flatten())
    part_y = _apply_rope_1d(patches[..., half:], yy.flatten())
    return torch.cat((cls, torch.cat((part_x, part_y), dim=-1)), dim=2)


class RopeAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("Embedding dimension must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 4:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE.")
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, value = qkv.unbind(dim=0)
        q = apply_2d_rope(q, grid_h, grid_w)
        k = apply_2d_rope(k, grid_h, grid_w)
        output = F.scaled_dot_product_attention(
            q,
            k,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RopeAttention(dim, num_heads, qkv_bias=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), grid_h, grid_w)
        return x + self.mlp(self.norm2(x))


class ForensicViT(nn.Module):
    """Provisional ViT-Tiny consistent with the dimensions disclosed by paper."""

    def __init__(
        self,
        in_channels: int = 8,
        patch_size: int = 8,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._initialize()

    def _initialize(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape[-2] % self.patch_size or residual.shape[-1] % self.patch_size:
            raise ValueError("Residual crop dimensions must be divisible by patch size.")
        tokens = self.patch_embed(residual)
        grid_h, grid_w = tokens.shape[-2:]
        tokens = tokens.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens, grid_h, grid_w)
        return self.norm(tokens)[:, 0]


class Projector(nn.Module):
    """Memory-conscious provisional nonlinear projector: 192->2048->8192."""

    def __init__(self, in_dim: int = 192, hidden_dim: int = 2048, out_dim: int = 8192):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ForensicTwins(nn.Module):
    def __init__(
        self,
        fre: ForensicResidualExtractor,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        patch_size: int = 8,
        projector_hidden: int = 2048,
        projector_out: int = 8192,
    ):
        super().__init__()
        self.fre = fre.freeze()
        self.encoder = ForensicViT(
            in_channels=8,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.projector = Projector(embed_dim, projector_hidden, projector_out)

    def train(self, mode: bool = True) -> "ForensicTwins":
        super().train(mode)
        self.fre.eval()
        return self

    def residual_from_halo(self, gray_with_halo: torch.Tensor) -> torch.Tensor:
        """FRE-before-crop equivalent for a local convolution with a 7px halo."""
        with torch.no_grad():
            residual = self.fre(gray_with_halo)
        radius = self.fre.radius
        if radius:
            residual = residual[..., radius:-radius, radius:-radius]
        return residual

    def encode_gray_halo(self, gray_with_halo: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.residual_from_halo(gray_with_halo))

    def forward(self, view1_halo: torch.Tensor, view2_halo: torch.Tensor):
        z1 = self.encode_gray_halo(view1_halo)
        z2 = self.encode_gray_halo(view2_halo)
        return z1, z2, self.projector(z1), self.projector(z2)


def _gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return x
    try:
        from torch.distributed.nn.functional import all_gather

        return torch.cat(tuple(all_gather(x)), dim=0)
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "Autograd-aware all_gather is required for the global Barlow loss. "
            "Upgrade PyTorch or run single-process."
        ) from error


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    n, m = matrix.shape
    if n != m:
        raise ValueError("Expected a square cross-correlation matrix.")
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    redundancy_lambda: float = 5e-5,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Global-batch Barlow Twins objective, computed in float32."""
    x = _gather_with_grad(embedding1).float()
    y = _gather_with_grad(embedding2).float()
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"Expected equal [B,D] tensors, got {x.shape} and {y.shape}")
    x = (x - x.mean(0)) / torch.sqrt(x.var(0, unbiased=False) + eps)
    y = (y - y.mean(0)) / torch.sqrt(y.var(0, unbiased=False) + eps)
    correlation = x.T @ y / x.shape[0]
    diagonal = torch.diagonal(correlation)
    invariance = (diagonal - 1.0).pow(2).sum()
    redundancy = off_diagonal(correlation).pow(2).sum()
    loss = invariance + redundancy_lambda * redundancy
    metrics = {
        "loss": loss.detach(),
        "invariance": invariance.detach(),
        "redundancy": redundancy.detach(),
        "diag_mean": diagonal.mean().detach(),
        "offdiag_abs_mean": off_diagonal(correlation).abs().mean().detach(),
        "global_batch": torch.tensor(x.shape[0], device=x.device),
    }
    return loss, metrics


def encoder_parameter_count(model: ForensicTwins) -> int:
    return sum(parameter.numel() for parameter in model.encoder.parameters())
