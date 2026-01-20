# system.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

from structures import Components, MicroTokens, Semantics, SystemState


class SystemLayer(nn.Module):
    """
    Deterministic system computations.
    """

    def __init__(self):
        super().__init__()

    def _track_ids(
        self,
        curr_boxes: torch.Tensor,
        prev_system: Optional[SystemState],
        iou_thresh: float = 0.5
    ) -> torch.Tensor:
        """
        Assigns IDs to current components by matching with previous frame via IoU.
        New components get new unique IDs incremented from the previous max.
        """
        b, k, _ = curr_boxes.shape
        device = curr_boxes.device
        
        # [B, K]
        curr_ids = torch.full((b, k), -1, dtype=torch.long, device=device)
        
        # If no history, just assign 0..K-1 individually per batch for now
        # (In a real app, you might want a global ID counter state passed in)
        if prev_system is None or prev_system.component_ids is None:
            for bi in range(b):
                curr_ids[bi] = torch.arange(k, device=device)
            return curr_ids

        prev_boxes = prev_system.comp_boxes_hard  # [B, K, 4]
        prev_ids = prev_system.component_ids      # [B, K]
        
        for bi in range(b):
            pb = prev_boxes[bi]
            cb = curr_boxes[bi]
            pid = prev_ids[bi]

            # IoU matrix: [K_prev, K_curr]
            # M3 usually outputs fixed K components (some active, some not).
            # We match all of them. In a refined version, match only active ones.
            iou = box_iou(pb, cb)
            
            # Cost = 1 - IoU
            cost = 1.0 - iou.cpu().numpy()
            
            # Hungarian matcher
            row_idx, col_idx = linear_sum_assignment(cost)
            
            # Determine next available ID
            if pid.numel() > 0:
                next_new_id = int(pid.max().item()) + 1
            else:
                next_new_id = 0
            
            assigned_mask = torch.zeros(k, dtype=torch.bool, device=device)

            for r, c in zip(row_idx, col_idx):
                # Only carry over ID if spatial overlap is decent
                if iou[r, c] > iou_thresh:
                    curr_ids[bi, c] = pid[r]
                    assigned_mask[c] = True
            
            # Assign new IDs to unmatched
            unmatched_cnt = (~assigned_mask).sum().item()
            if unmatched_cnt > 0:
                new_ids = torch.arange(next_new_id, next_new_id + unmatched_cnt, device=device)
                curr_ids[bi, ~assigned_mask] = new_ids
        
        return curr_ids

    @torch.no_grad()
    def forward(
        self,
        micro: MicroTokens,
        comps: Components,
        sem: Semantics,
        prev_system: Optional[SystemState] = None,
    ) -> SystemState:
        system = SystemState()

        # 1. Hard Assignment
        hard_assign = torch.argmax(comps.assign_probs, dim=-1)  # [B,N]
        system.hard_assign = hard_assign.detach()

        # 2. Predictions
        pred_classes = torch.argmax(sem.class_logits, dim=-1)  # [B,K]
        system.pred_classes_hard = pred_classes.detach()

        # 3. Micro Boxes
        system.micro_boxes_hard = micro.boxes.detach()
        system.micro_keep_mask = torch.ones_like(hard_assign, dtype=torch.bool).detach()

        # 4. Component Aggregation (Boxes)
        b, n, _ = micro.boxes.shape
        k = comps.assign_probs.shape[-1]
        comp_boxes = torch.zeros((b, k, 4), device=micro.boxes.device, dtype=micro.boxes.dtype)

        for bi in range(b):
            mb = micro.boxes[bi]
            ha = hard_assign[bi]
            # Init inverted boxes
            x1 = torch.full((k,), 1.0, device=mb.device, dtype=mb.dtype)
            y1 = torch.full((k,), 1.0, device=mb.device, dtype=mb.dtype)
            x2 = torch.zeros((k,), device=mb.device, dtype=mb.dtype)
            y2 = torch.zeros((k,), device=mb.device, dtype=mb.dtype)

            # Scatter reduce (requires torch >= 1.11 for amin/amax)
            x1.scatter_reduce_(0, ha, mb[:, 0], reduce="amin", include_self=True)
            y1.scatter_reduce_(0, ha, mb[:, 1], reduce="amin", include_self=True)
            x2.scatter_reduce_(0, ha, mb[:, 2], reduce="amax", include_self=True)
            y2.scatter_reduce_(0, ha, mb[:, 3], reduce="amax", include_self=True)

            comp_boxes[bi, :, 0] = x1
            comp_boxes[bi, :, 1] = y1
            comp_boxes[bi, :, 2] = x2
            comp_boxes[bi, :, 3] = y2

        system.comp_boxes_hard = comp_boxes.detach()

        # 5. ID Tracking / Object Permanence
        system.component_ids = self._track_ids(
            system.comp_boxes_hard, 
            prev_system,
            iou_thresh=0.3
        )

        # 6. Deltas (Simple Diff)
        if prev_system and prev_system.component_ids is not None:
            # A simple delta: which IDs are new?
            deltas = []
            for bi in range(b):
                prev_ids_set = set(prev_system.component_ids[bi].cpu().numpy().tolist())
                curr_ids_set = set(system.component_ids[bi].cpu().numpy().tolist())
                
                # We can refine this to ignore "background" or "empty" comps if we had a mask
                added = curr_ids_set - prev_ids_set
                removed = prev_ids_set - curr_ids_set
                deltas.append({"added": list(added), "removed": list(removed)})
            system.deltas = deltas
        else:
            system.deltas = None

        system.meta["has_prev"] = prev_system is not None
        return system
