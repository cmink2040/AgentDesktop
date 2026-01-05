# layers.py  (UPDATED: remove SystemLayer from here)
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from structures import (
    Components,
    MicroTokens,
    PyramidFeatures,
    Semantics,
    Tensor,
)

# -------------------------
# Helper utilities
# -------------------------

def _make_xy_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    ys = torch.linspace(0.5 / h, 1.0 - 0.5 / h, h, device=device, dtype=dtype)
    xs = torch.linspace(0.5 / w, 1.0 - 0.5 / w, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)  # [H, W, 2]


def _boxes_from_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    grid = _make_xy_grid(h, w, device, dtype)  # centers
    cx, cy = grid[..., 0], grid[..., 1]
    half_w = 0.5 / w
    half_h = 0.5 / h
    x1 = (cx - half_w).clamp(0, 1)
    y1 = (cy - half_h).clamp(0, 1)
    x2 = (cx + half_w).clamp(0, 1)
    y2 = (cy + half_h).clamp(0, 1)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _flatten_hw(x: Tensor) -> Tensor:
    b, c, h, w = x.shape
    return x.view(b, c, h * w).transpose(1, 2).contiguous()


# -------------------------
# M1: Backbone (simple Conv pyramid)
# -------------------------

class ConvStem(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),  # /2
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1),  # /4
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvStage(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, stride: int):
        super().__init__()
        self.down = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.block = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )
        self.skip = nn.Identity() if (ch_in == ch_out and stride == 1) else nn.Conv2d(ch_in, ch_out, 1, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        y = self.down(x)
        y = y + self.skip(x)
        y = y + self.block(y)
        return y


class M1Backbone(nn.Module):
    def __init__(self, in_ch: int = 3, c4: int = 96, c8: int = 192, c16: int = 384, c32: int = 768):
        super().__init__()
        self.stem = ConvStem(in_ch, c4)            # /4
        self.s8 = ConvStage(c4, c8, stride=2)      # /8
        self.s16 = ConvStage(c8, c16, stride=2)    # /16
        self.s32 = ConvStage(c16, c32, stride=2)   # /32

    def forward(self, x: Tensor) -> PyramidFeatures:
        f4 = self.stem(x)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        return PyramidFeatures(f4=f4, f8=f8, f16=f16, f32=f32)


# -------------------------
# M2: Microtokenizer
# -------------------------

@dataclass
class M2Config:
    d_model: int = 256
    n_max: int = 8192
    use_scales: Tuple[int, ...] = (4, 8, 16)
    density_level: int = 8


class M2MicroTokenizer(nn.Module):
    def __init__(self, cfg: M2Config, c4: int, c8: int, c16: int):
        super().__init__()
        self.cfg = cfg

        self.proj4 = nn.Conv2d(c4, cfg.d_model, kernel_size=1)
        self.proj8 = nn.Conv2d(c8, cfg.d_model, kernel_size=1)
        self.proj16 = nn.Conv2d(c16, cfg.d_model, kernel_size=1)

        self.density_head = nn.Sequential(
            nn.Conv2d(c8, c8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c8, 1, kernel_size=1),
        )

    def _level_tokens(self, feat: Tensor, proj: nn.Module) -> Tuple[Tensor, Tensor, Tensor]:
        b, _, h, w = feat.shape
        feat_d = proj(feat)                 # [B, D, H, W]
        tokens = _flatten_hw(feat_d)         # [B, H*W, D]

        grid = _make_xy_grid(h, w, feat.device, feat.dtype)
        boxes = _boxes_from_grid(h, w, feat.device, feat.dtype)
        pos = grid.view(1, h * w, 2).expand(b, -1, -1).contiguous()
        box = boxes.view(1, h * w, 4).expand(b, -1, -1).contiguous()
        return tokens, pos, box

    def forward(self, pyramid: PyramidFeatures) -> MicroTokens:
        density_logits = self.density_head(pyramid.f8)  # [B,1,H8,W8]
        density = torch.sigmoid(density_logits)

        candidates = []
        if 4 in self.cfg.use_scales:
            candidates.append(self._level_tokens(pyramid.f4, self.proj4))
        if 8 in self.cfg.use_scales:
            candidates.append(self._level_tokens(pyramid.f8, self.proj8))
        if 16 in self.cfg.use_scales:
            candidates.append(self._level_tokens(pyramid.f16, self.proj16))

        tokens = torch.cat([c[0] for c in candidates], dim=1)  # [B, Nc, D]
        pos = torch.cat([c[1] for c in candidates], dim=1)     # [B, Nc, 2]
        boxes = torch.cat([c[2] for c in candidates], dim=1)   # [B, Nc, 4]

        b, n, _ = pos.shape
        grid = (pos * 2.0 - 1.0).view(b, n, 1, 2)  # [B,N,1,2]
        sampled = F.grid_sample(density, grid, mode="bilinear", align_corners=False)  # [B,1,N,1]
        importance = sampled.view(b, n)

        k = min(self.cfg.n_max, n)
        topk_vals, topk_idx = torch.topk(importance, k=k, dim=1, largest=True, sorted=False)

        batch_idx = torch.arange(b, device=tokens.device).unsqueeze(-1).expand(b, k)
        sel_tokens = tokens[batch_idx, topk_idx]
        sel_pos = pos[batch_idx, topk_idx]
        sel_boxes = boxes[batch_idx, topk_idx]
        sel_importance = topk_vals

        return MicroTokens(
            tokens=sel_tokens,
            pos=sel_pos,
            boxes=sel_boxes,
            importance=sel_importance,
            density_logits=density_logits,
        )


# -------------------------
# M3: Merger
# -------------------------

@dataclass
class M3Config:
    d_model: int = 256
    k_components: int = 512
    tau_init: float = 1.0


class M3Merger(nn.Module):
    def __init__(self, cfg: M3Config):
        super().__init__()
        self.cfg = cfg
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.assign_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.k_components),
        )
        self.tau = nn.Parameter(torch.tensor(cfg.tau_init))

    def forward(self, micro: MicroTokens) -> Components:
        x = micro.tokens
        p = micro.pos
        x = x + self.pos_mlp(p)

        logits = self.assign_mlp(x)
        tau = torch.clamp(self.tau, 0.1, 10.0)
        A = F.softmax(logits / tau, dim=-1)

        mass = A.sum(dim=1)
        denom = mass.unsqueeze(-1).clamp_min(1e-6)
        comp_tokens = torch.einsum("bnk,bnd->bkd", A, micro.tokens) / denom
        comp_pos = torch.einsum("bnk,bnq->bkq", A, p) / mass.unsqueeze(-1).clamp_min(1e-6)

        return Components(assign_probs=A, comp_tokens=comp_tokens, comp_mass=mass, comp_pos=comp_pos)


# -------------------------
# M4: Semantic Head
# -------------------------

@dataclass
class M4Config:
    d_model: int = 256
    num_classes: int = 32
    attr_dims: Optional[Dict[str, int]] = None


class M4SemanticHead(nn.Module):
    def __init__(self, cfg: M4Config):
        super().__init__()
        self.cfg = cfg
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )
        self.attr_heads = nn.ModuleDict()
        if cfg.attr_dims:
            for k, d in cfg.attr_dims.items():
                self.attr_heads[k] = nn.Sequential(
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, cfg.d_model),
                    nn.GELU(),
                    nn.Linear(cfg.d_model, d),
                )

    def forward(self, comps: Components) -> Semantics:
        z = comps.comp_tokens
        class_logits = self.classifier(z)

        attr_logits = None
        if len(self.attr_heads) > 0:
            attr_logits = {k: head(z) for k, head in self.attr_heads.items()}

        return Semantics(class_logits=class_logits, attr_logits=attr_logits, mask_logits=None)
