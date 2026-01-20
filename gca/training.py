# training.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from models import GCAEnd2EndModel, ModelConfig
from losses import BudgetConfig, LossWeights, compute_losses


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True

    # Loss configs
    weights: LossWeights = field(default_factory=LossWeights)
    budget: BudgetConfig = field(default_factory=lambda: BudgetConfig(target=4096.0, slack=512.0))

    # Metadata
    num_seg_labels: int = 64  # label-space size for gt_mask


class Trainer:
    def __init__(self, model: GCAEnd2EndModel, cfg: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Expected batch keys:
          - image: [B,3,H,W] float
          - gt_mask: [B,H,W] long (seg labels)  (used for M2/M3 losses)
          - sem_mask: [B,H,W] long (semantic class labels) (used for M4 supervision)
          - comp_targets: OPTIONAL [B,K] long (component classes) for task CE
        """
        image = batch["image"].to(self.device)
        gt_mask = batch["gt_mask"].to(self.device)
        sem_mask = batch["sem_mask"].to(self.device)

        comp_targets = batch.get("comp_targets")
        if comp_targets is not None:
            comp_targets = comp_targets.to(self.device)

        self.model.train()
        self.optim.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(self.scaler.is_enabled())):
            out = self.model(image)

            total, named = compute_losses(
                micro=out.learned.micro,
                comps=out.learned.comps,
                sem=out.learned.sem,
                gt_mask=gt_mask,
                sem_mask=sem_mask,
                comp_targets=comp_targets,
                num_labels=self.cfg.num_seg_labels,
                weights=self.cfg.weights,
                budget_cfg=self.cfg.budget,
            )

        self.scaler.scale(total).backward()

        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        self.scaler.step(self.optim)
        self.scaler.update()

        # Return scalars for logging
        return {k: float(v.detach().cpu()) for k, v in named.items()}

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        image = batch["image"].to(self.device)
        gt_mask = batch["gt_mask"].to(self.device)
        sem_mask = batch["sem_mask"].to(self.device)
        comp_targets = batch.get("comp_targets")
        if comp_targets is not None:
            comp_targets = comp_targets.to(self.device)

        self.model.eval()
        out = self.model(image)

        total, named = compute_losses(
            micro=out.learned.micro,
            comps=out.learned.comps,
            sem=out.learned.sem,
            gt_mask=gt_mask,
            sem_mask=sem_mask,
            comp_targets=comp_targets,
            num_labels=self.cfg.num_seg_labels,
            weights=self.cfg.weights,
            budget_cfg=self.cfg.budget,
        )
        return {k: float(v.detach().cpu()) for k, v in named.items()}


def example_main():
    """
    Minimal example skeleton (no real dataset). Replace with your dataset/dataloader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ModelConfig(
        d_model=256,
        n_max=4096,
        k_components=256,
        num_classes=64,
        attr_dims={"clickable": 2, "enabled": 2, "text_bearing": 2},
    )
    model = GCAEnd2EndModel(model_cfg)

    train_cfg = TrainConfig(
        lr=3e-4,
        amp=True,
        num_seg_labels=64,
    )

    trainer = Trainer(model, train_cfg, device=device)

    # Dummy batch
    batch = {
        "image": torch.rand(2, 3, 1024, 1024),
        "gt_mask": torch.randint(0, 64, (2, 1024, 1024), dtype=torch.long),
        # Optional component targets (must match K used by model): here K=256
        "comp_targets": torch.randint(0, 64, (2, 256), dtype=torch.long),
    }

    logs = trainer.train_step(batch)
    print(logs)


if __name__ == "__main__":
    example_main()
