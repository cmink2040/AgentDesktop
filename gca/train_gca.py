# train_gca.py
import os
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import GCAEnd2EndModel, ModelConfig
from training import Trainer, TrainConfig
from dataset_uivision import UIVisionDataset

def main():
    parser = argparse.ArgumentParser(description="Train GCA on UI-Vision")
    parser.add_argument("--data_dir", type=str, default="./ui_vision_data", help="Root for UI-Vision dataset")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # 1. Setup Data
    print(f"Initializing dataset from {args.data_dir}...")
    dataset = UIVisionDataset(
        root_dir=args.data_dir,
        split="element_grounding", 
        img_size=(1024, 1024),
        instance_mode=True,  # Important for GCA grouping
        download=True
    )
    
    # Simple split (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # 2. Setup Model
    print("Creating model...")
    # Using typical GCA configs
    model_cfg = ModelConfig(
        c4=96, c8=192, c16=384, c32=768,
        d_model=256,
        n_max=4096,
        k_components=256, # We learn up to 256 active components per screen
        num_classes=64,   # Or whatever class count UI-Vision has + background
    )
    model = GCAEnd2EndModel(model_cfg)
    
    # 3. Setup Trainer
    train_cfg = TrainConfig(
        lr=args.lr,
        amp=(args.device == "cuda"),
        num_seg_labels=512, # Must encompass max instance ID in dataset
    )
    trainer = Trainer(model, train_cfg, torch.device(args.device))
    
    # 4. Loop
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        metrics = []
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            log = trainer.train_step(batch)
            metrics.append(log)
            pbar.set_postfix({"loss": f"{log['total']:.4f}"})
            
        avg_train = {k: sum(d[k] for d in metrics)/len(metrics) for k in metrics[0].keys()}
        print(f"Train Loss: {avg_train['total']:.4f} | M2: {avg_train['m2_alloc']:.4f} | M3: {avg_train['m3_purity']:.4f}")
        
        # Eval
        if len(val_data) > 0:
            val_metrics = []
            with torch.no_grad():
                for batch in val_loader:
                    log = trainer.eval_step(batch)
                    val_metrics.append(log)
            avg_val = {k: sum(d[k] for d in val_metrics)/len(val_metrics) for k in val_metrics[0].keys()}
            print(f"Val Loss:   {avg_val['total']:.4f}")
        
        # Save
        path = os.path.join(args.save_dir, f"gca_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), path)
        print(f"Saved to {path}")

if __name__ == "__main__":
    main()
