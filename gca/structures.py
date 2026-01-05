# structures.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


Tensor = torch.Tensor


@dataclass
class PyramidFeatures:
    """
    Multi-scale feature pyramid.
    Typical strides relative to input: 4, 8, 16, 32
    """
    f4: Tensor   # [B, C4, H/4,  W/4]
    f8: Tensor   # [B, C8, H/8,  W/8]
    f16: Tensor  # [B, C16,H/16, W/16]
    f32: Tensor  # [B, C32,H/32, W/32]


@dataclass
class MicroTokens:
    """
    Dense microtoken representation emitted by M2.
    """
    tokens: Tensor               # [B, N, D]
    pos: Tensor                  # [B, N, 2] normalized xy in [0,1]
    boxes: Tensor                # [B, N, 4] normalized xyxy in [0,1]
    importance: Tensor           # [B, N] (soft) importance weights, differentiable
    density_logits: Tensor       # [B, 1, H8, W8] allocation endpoint (usually at f8)


@dataclass
class Components:
    """
    Merged components emitted by M3.
    """
    assign_probs: Tensor         # [B, N, K] soft assignment matrix
    comp_tokens: Tensor          # [B, K, D] merged tokens
    comp_mass: Tensor            # [B, K] assignment mass per component
    comp_pos: Tensor             # [B, K, 2] soft centroids (normalized xy)


@dataclass
class Semantics:
    """
    M4 semantic outputs.
    """
    class_logits: Tensor         # [B, K, C]
    attr_logits: Optional[Dict[str, Tensor]] = None  # each [B, K, A_i]
    mask_logits: Optional[Tensor] = None             # optional [B, K, H8, W8]


@dataclass
class LearnedState:
    """
    Primary differentiable state flowing through M1->M4.
    """
    pyramid: Optional[PyramidFeatures] = None
    micro: Optional[MicroTokens] = None
    comps: Optional[Components] = None
    sem: Optional[Semantics] = None
    aux: Dict[str, Tensor] = field(default_factory=dict)


@dataclass
class SystemState:
    """
    Secondary (typically detached) outputs used by your System Layer.
    Keep this flexible; many fields may be Python objects.
    """
    micro_boxes_hard: Optional[Tensor] = None        # [B, N, 4] detached
    micro_keep_mask: Optional[Tensor] = None         # [B, N] bool detached
    hard_assign: Optional[Tensor] = None             # [B, N] long detached
    comp_boxes_hard: Optional[Tensor] = None         # [B, K, 4] detached (inactive comps ok)
    pred_classes_hard: Optional[Tensor] = None       # [B, K] long detached
    component_ids: Optional[Any] = None              # list/dict
    ocr_regions: Optional[Any] = None                # list/dict
    deltas: Optional[Any] = None                     # list/dict
    ui_graph: Optional[Any] = None                   # list/dict
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    learned: LearnedState
    system: SystemState
