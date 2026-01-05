# system.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from structures import Components, MicroTokens, Semantics, SystemState


class SystemLayer(nn.Module):
    """
    Deterministic (non-differentiable) system computations.
    Consumes hard-decoded outputs and produces secondary outputs used for:
      - IDs / tracking (placeholder)
      - OCR region proposals (placeholder)
      - deltas across frames (placeholder)
      - UI graph assembly (placeholder)

    IMPORTANT:
      - Keep this in no_grad().
      - Feed only hard outputs into it for stability.
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(
        self,
        micro: MicroTokens,
        comps: Components,
        sem: Semantics,
        prev_system: Optional[SystemState] = None,
    ) -> SystemState:
        system = SystemState()

        # Hard assign microtokens to components
        hard_assign = torch.argmax(comps.assign_probs, dim=-1)  # [B,N]
        system.hard_assign = hard_assign.detach()

        # Hard predicted classes
        pred_classes = torch.argmax(sem.class_logits, dim=-1)  # [B,K]
        system.pred_classes_hard = pred_classes.detach()

        # Micro boxes + keep mask
        system.micro_boxes_hard = micro.boxes.detach()
        system.micro_keep_mask = torch.ones_like(hard_assign, dtype=torch.bool).detach()

        # Component boxes (xyxy) aggregated from micro boxes
        b, n, _ = micro.boxes.shape
        k = comps.assign_probs.shape[-1]
        comp_boxes = torch.zeros((b, k, 4), device=micro.boxes.device, dtype=micro.boxes.dtype)

        for bi in range(b):
            mb = micro.boxes[bi]     # [N,4]
            ha = hard_assign[bi]     # [N]
            x1 = torch.full((k,), 1.0, device=mb.device, dtype=mb.dtype)
            y1 = torch.full((k,), 1.0, device=mb.device, dtype=mb.dtype)
            x2 = torch.zeros((k,), device=mb.device, dtype=mb.dtype)
            y2 = torch.zeros((k,), device=mb.device, dtype=mb.dtype)

            # amin/amax scatter reduce requires torch>=1.12-ish; works on modern PyTorch.
            x1.scatter_reduce_(0, ha, mb[:, 0], reduce="amin", include_self=True)
            y1.scatter_reduce_(0, ha, mb[:, 1], reduce="amin", include_self=True)
            x2.scatter_reduce_(0, ha, mb[:, 2], reduce="amax", include_self=True)
            y2.scatter_reduce_(0, ha, mb[:, 3], reduce="amax", include_self=True)

            comp_boxes[bi, :, 0] = x1
            comp_boxes[bi, :, 1] = y1
            comp_boxes[bi, :, 2] = x2
            comp_boxes[bi, :, 3] = y2

        system.comp_boxes_hard = comp_boxes.detach()

        # Placeholders: fill these with your real system logic.
        system.component_ids = None
        system.ocr_regions = None
        system.deltas = None
        system.ui_graph = None

        # Optionally stash meta
        system.meta["has_prev"] = prev_system is not None
        return system
