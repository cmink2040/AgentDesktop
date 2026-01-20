# train_gca.py
import os
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import save_file

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
    default_workers = 0 if os.name == "nt" else 4
    parser.add_argument("--num_workers", type=int, default=default_workers, help="DataLoader workers (use 0 on Windows if you hit WinError 1455)")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from (safetensors or pth)")
    args = parser.parse_args()

    # 1. Setup Data
    print(f"Initializing dataset from {args.data_dir}...")
    dataset = UIVisionDataset(
        root_dir=args.data_dir,
        split="element_grounding", 
        img_size=(1024, 1024),
        instance_mode=True,  # Important for GCA grouping to work on instances
        download=True
    )
    
    # Simple split (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Enable pin_memory for faster data transfer to CUDA
    # Increase workers to offload CPU tasks (image load, resizing, mask generation)
    # On Windows, keep it conservative (e.g. 2-4). On Linux, can go higher.
    num_workers = args.num_workers
    use_pin_memory = "cuda" in args.device
    
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True, 
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0) # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # 2. Setup Model
    print("Creating model...")
    # Using typical GCA configs
    model_cfg = ModelConfig(
        c4=96, c8=192, c16=384, c32=768,
        d_model=256,
        n_max=4096,
        k_components=256, # We learn up to 256 active components per screen
        num_classes=dataset.num_classes,   # From dataset categories (incl. background)
    )
    model = GCAEnd2EndModel(model_cfg)
    # Resume weights if requested
    if args.resume_from:
        print(f"Resuming weights from {args.resume_from}...")
        if args.resume_from.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.resume_from)
            model.load_state_dict(state_dict)
        else:
            # Fallback for old .pth
            model.load_state_dict(torch.load(args.resume_from, map_location="cpu"))
    
    # 
    # 3. Setup Trainer
    train_cfg = TrainConfig(
        lr=args.lr,
        # Check if "cuda" is in device string (e.g. "cuda:0") for AMP
        amp=("cuda" in args.device),
        num_seg_labels=dataset.num_classes, # Matches class labels in gt_mask
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
        val_loss = float('inf')
        if len(val_data) > 0:
            val_metrics = []
            with torch.no_grad():
                for batch in val_loader:
                    log = trainer.eval_step(batch)
                    val_metrics.append(log)
            avg_val = {k: sum(d[k] for d in val_metrics)/len(val_metrics) for k in val_metrics[0].keys()}
            val_loss = avg_val['total']
            print(f"Val Loss:   {val_loss:.4f}")

        # Save snapshot each epoch
        snapshot_path = os.path.join(args.save_dir, f"gca-snapshot-{epoch+1}.safetensors")
        save_file(model.state_dict(), snapshot_path)
        print(f"Snapshot saved to {snapshot_path}")

    # Save final model once after all epochs
    final_path = os.path.join(args.save_dir, "gca.safetensors")
    save_file(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")

if __name__ == "__main__":
    main()
