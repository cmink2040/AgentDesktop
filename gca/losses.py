# losses.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from structures import Components, MicroTokens, Semantics, Tensor


@dataclass
class LossWeights:
    # Task
    task_ce: float = 1.0

    # M2 (allocation)
    budget: float = 0.05
    alloc: float = 0.25

    # M3 (merging)
    cut: float = 0.25
    compact: float = 0.05
    purity: float = 0.05


@dataclass
class BudgetConfig:
    """
    Soft range constraint for expected token count.
    """
    target: float = 4096.0
    slack: float = 512.0


def make_boundary_importance(gt_mask: Tensor) -> Tensor:
    """
    Build a simple boundary importance map from an integer label mask.

    gt_mask: [B, H, W] int64 or int32 labels.

    Returns W: [B, 1, H, W] float in [0,1] (unnormalized is ok).
    """
    # Detect boundaries by comparing adjacent pixels (4-neighborhood).
    # This is cheap and works well for UI-like crisp edges.
    b, h, w = gt_mask.shape
    m = gt_mask

    diff = torch.zeros((b, h, w), device=m.device, dtype=torch.float32)
    diff[:, :, 1:] += (m[:, :, 1:] != m[:, :, :-1]).float()
    diff[:, :, :-1] += (m[:, :, 1:] != m[:, :, :-1]).float()
    diff[:, 1:, :] += (m[:, 1:, :] != m[:, :-1, :]).float()
    diff[:, :-1, :] += (m[:, 1:, :] != m[:, :-1, :]).float()

    # Thicken boundaries a bit using a small maxpool.
    diff = diff.unsqueeze(1)  # [B,1,H,W]
    diff = F.max_pool2d(diff, kernel_size=3, stride=1, padding=1)

    # Normalize per-image to [0,1] (avoid divide-by-zero)
    mx = diff.flatten(1).amax(dim=1).clamp_min(1e-6).view(b, 1, 1, 1)
    W = (diff / mx).clamp(0.0, 1.0)
    return W


def sample_labels_at_positions(gt_mask: Tensor, pos_xy: Tensor) -> Tensor:
    """
    Sample integer GT labels at token positions.

    gt_mask: [B,H,W] integer labels
    pos_xy: [B,N,2] normalized [0,1] x,y

    Returns g: [B,N] int64 labels
    """
    b, h, w = gt_mask.shape
    # grid_sample expects input [B,1,H,W] float; nearest sampling
    inp = gt_mask.unsqueeze(1).float()  # [B,1,H,W]
    grid = pos_xy * 2.0 - 1.0  # to [-1,1]
    grid = grid.view(b, -1, 1, 2)  # [B,N,1,2]
    samp = F.grid_sample(inp, grid, mode="nearest", align_corners=False)  # [B,1,N,1]
    g = samp.view(b, -1).long()
    return g


def loss_m2_budget(micro: MicroTokens, budget: BudgetConfig) -> Tensor:
    """
    Penalize expected token count being outside [target-slack, target+slack].
    Uses density mass proxy from density logits (sigmoid sum).
    """
    # density_logits: [B,1,H8,W8]
    dens = torch.sigmoid(micro.density_logits)
    n_exp = dens.flatten(1).sum(dim=1)  # [B]
    lo = budget.target - budget.slack
    hi = budget.target + budget.slack

    over = (n_exp - hi).clamp_min(0.0)
    under = (lo - n_exp).clamp_min(0.0)
    return (over.pow(2) + under.pow(2)).mean()


def loss_m2_alloc(micro: MicroTokens, gt_mask: Tensor) -> Tensor:
    """
    Encourage density mass to correlate with boundary importance.
    We compute W on the full-res GT mask, then downsample to the density map resolution.
    """
    W = make_boundary_importance(gt_mask)  # [B,1,H,W]
    target = F.interpolate(W, size=micro.density_logits.shape[-2:], mode="bilinear", align_corners=False)

    dens = torch.sigmoid(micro.density_logits)  # [B,1,H8,W8]

    # Normalize both per-image so the model learns a distribution of where to allocate.
    b = dens.shape[0]
    dens_n = dens.flatten(1)
    targ_n = target.flatten(1)
    dens_n = dens_n / dens_n.sum(dim=1, keepdim=True).clamp_min(1e-6)
    targ_n = targ_n / targ_n.sum(dim=1, keepdim=True).clamp_min(1e-6)

    # L1 on normalized distributions
    return (dens_n - targ_n).abs().sum(dim=1).mean()


def loss_m3_cut(comps: Components, g: Tensor, boundary_only: bool = True) -> Tensor:
    """
    Cannot-link / cut loss:
    For token pairs that should be different (cross-label), penalize assignment similarity <Ai, Aj>.

    To keep it cheap, we use local "grid neighbors" by sorting tokens by position is nontrivial.
    Instead, we do a light random pairing which still works surprisingly well early on.
    If you later add adjacency edges (kNN), swap this for true neighbor edges.
    """
    A = comps.assign_probs  # [B,N,K]
    b, n, k = A.shape

    # Random pairs
    idx_i = torch.randint(0, n, (b, n), device=A.device)
    idx_j = torch.randint(0, n, (b, n), device=A.device)

    gi = torch.gather(g, 1, idx_i)  # [B,N]
    gj = torch.gather(g, 1, idx_j)  # [B,N]

    if boundary_only:
        mask = (gi != gj).float()  # cross-boundary
    else:
        mask = torch.ones_like(gi, dtype=torch.float32)

    Ai = torch.gather(A, 1, idx_i.unsqueeze(-1).expand(b, n, k))  # [B,N,K]
    Aj = torch.gather(A, 1, idx_j.unsqueeze(-1).expand(b, n, k))  # [B,N,K]

    sim = (Ai * Aj).sum(dim=-1)  # [B,N]
    return (mask * sim).sum() / (mask.sum().clamp_min(1.0))


def loss_m3_compact(comps: Components, micro: MicroTokens) -> Tensor:
    """
    Spatial compactness:
      sum_k sum_n A[n,k] * ||p_n - mu_k||^2  (mu_k from comps.comp_pos)
    """
    A = comps.assign_probs         # [B,N,K]
    p = micro.pos                  # [B,N,2]
    mu = comps.comp_pos            # [B,K,2]

    # Expand and compute squared distances
    # dist2: [B,N,K]
    dist2 = (p.unsqueeze(2) - mu.unsqueeze(1)).pow(2).sum(dim=-1)
    return (A * dist2).mean()


def loss_m3_purity(comps: Components, g: Tensor, num_labels: int) -> Tensor:
    """
    Component purity: each component should prefer a single GT label.
    We compute soft label histograms per component and penalize entropy.

    num_labels: total label count in gt_mask (including background if used)
    """
    A = comps.assign_probs  # [B,N,K]
    b, n, k = A.shape

    # One-hot token labels: [B,N,C]
    oh = F.one_hot(g.clamp_min(0), num_classes=num_labels).float()

    # q[b,k,c] = sum_n A[b,n,k] * oh[b,n,c]
    q = torch.einsum("bnk,bnc->bkc", A, oh)

    # Normalize label distribution per component
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    # Entropy
    ent = -(q * (q.clamp_min(1e-9)).log()).sum(dim=-1)  # [B,K]
    # Only count "active" components (nontrivial mass)
    active = (comps.comp_mass > 1e-3).float()
    return (ent * active).sum() / active.sum().clamp_min(1.0)


def loss_task_component_ce(sem: Semantics, comp_targets: Tensor, ignore_index: int = -1) -> Tensor:
    """
    Component-level classification CE.
    comp_targets: [B,K] long, with ignore_index for inactive/unused comps.
    """
    b, k, c = sem.class_logits.shape
    return F.cross_entropy(
        sem.class_logits.view(b * k, c),
        comp_targets.view(b * k),
        ignore_index=ignore_index,
    )


def compute_losses(
    micro: MicroTokens,
    comps: Components,
    sem: Semantics,
    gt_mask: Tensor,
    comp_targets: Optional[Tensor],
    num_labels: int,
    weights: LossWeights,
    budget_cfg: BudgetConfig,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Returns (total_loss, dict_of_named_losses).

    gt_mask: [B,H,W] integer segmentation labels for allocation/merge losses.
    comp_targets: [B,K] optional component class targets for CE (if you have them).
    num_labels: number of segmentation labels in gt_mask space.
    """
    out: Dict[str, Tensor] = {}

    # Token-level GT labels at microtoken positions
    g = sample_labels_at_positions(gt_mask, micro.pos)  # [B,N]

    # M2 losses
    out["m2_budget"] = loss_m2_budget(micro, budget_cfg)
    out["m2_alloc"] = loss_m2_alloc(micro, gt_mask)

    # M3 losses
    out["m3_cut"] = loss_m3_cut(comps, g, boundary_only=True)
    out["m3_compact"] = loss_m3_compact(comps, micro)
    out["m3_purity"] = loss_m3_purity(comps, g, num_labels=num_labels)

    # Task loss (optional)
    if comp_targets is not None:
        out["task_ce"] = loss_task_component_ce(sem, comp_targets)
    else:
        out["task_ce"] = torch.zeros((), device=micro.tokens.device)

    total = (
        weights.task_ce * out["task_ce"]
        + weights.budget * out["m2_budget"]
        + weights.alloc * out["m2_alloc"]
        + weights.cut * out["m3_cut"]
        + weights.compact * out["m3_compact"]
        + weights.purity * out["m3_purity"]
    )

    out["total"] = total
    return total, out
