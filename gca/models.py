# models.py  (UPDATED: import SystemLayer from system.py)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from structures import LearnedState, ModelOutput, SystemState
from layers import (
    M1Backbone,
    M2Config,
    M2MicroTokenizer,
    M3Config,
    M3Merger,
    M4Config,
    M4SemanticHead,
)
from system import SystemLayer


@dataclass
class ModelConfig:
    # Backbone channels
    c4: int = 96
    c8: int = 192
    c16: int = 384
    c32: int = 768

    # Token/model dims
    d_model: int = 256

    # M2
    n_max: int = 8192

    # M3
    k_components: int = 512
    tau_init: float = 1.0

    # M4
    num_classes: int = 32

    # Optional attributes
    attr_dims: Optional[dict] = None


class GCAEnd2EndModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.m1 = M1Backbone(in_ch=3, c4=cfg.c4, c8=cfg.c8, c16=cfg.c16, c32=cfg.c32)

        m2_cfg = M2Config(d_model=cfg.d_model, n_max=cfg.n_max, use_scales=(4, 8, 16), density_level=8)
        self.m2 = M2MicroTokenizer(m2_cfg, c4=cfg.c4, c8=cfg.c8, c16=cfg.c16)

        m3_cfg = M3Config(d_model=cfg.d_model, k_components=cfg.k_components, tau_init=cfg.tau_init)
        self.m3 = M3Merger(m3_cfg)

        m4_cfg = M4Config(d_model=cfg.d_model, num_classes=cfg.num_classes, attr_dims=cfg.attr_dims)
        self.m4 = M4SemanticHead(m4_cfg)

        self.system_layer = SystemLayer()

    def forward(self, image: torch.Tensor, prev_system: Optional[SystemState] = None) -> ModelOutput:
        learned = LearnedState()

        pyramid = self.m1(image)
        learned.pyramid = pyramid

        micro = self.m2(pyramid)
        learned.micro = micro

        comps = self.m3(micro)
        learned.comps = comps

        sem = self.m4(comps)
        learned.sem = sem

        system = self.system_layer(micro=micro, comps=comps, sem=sem, prev_system=prev_system)
        return ModelOutput(learned=learned, system=system)
