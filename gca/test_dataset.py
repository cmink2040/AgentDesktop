
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset_uivision import UIVisionDataset

def visualize_sample(dataset, idx=0, output_file="dataset_test.png"):
    # 1. Get raw item data (before rasterization logic, but after JSON parsing)
    raw_item = dataset.items[idx]
    img_path = raw_item['img_path']
    raw_boxes = raw_item['boxes']
    
    # 2. Get processed item (Tensors)
    data = dataset[idx]
    image_tensor = data['image'] # [3, H, W]
    gt_mask = data['gt_mask']    # [H, W] (Instance IDs)
    sem_mask = data['sem_mask']  # [H, W] (Class IDs)
    
    # Convert image for display (C,H,W -> H,W,C)
    img_np = image_tensor.permute(1, 2, 0).numpy()
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot 1: Standard Image with Raw Ground Truth Boxes (Like the user's example)
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Image + Raw JSON Boxes ({len(raw_boxes)} items)")
    axes[0].axis('off')
    
    # We need to scale raw boxes to the resized image size
    h_out, w_out = dataset.img_size
    
    # Let's derive boxes from the Instance Mask unique IDs
    instance_ids = torch.unique(gt_mask)
    detected_boxes = []
    
    for inst_id in instance_ids:
        if inst_id == 0: continue # Background
        # Find coords
        ys, xs = torch.where(gt_mask == inst_id)
        if len(xs) == 0: continue
        
        x1, x2 = xs.min().item(), xs.max().item()
        y1, y2 = ys.min().item(), ys.max().item()
        detected_boxes.append([x1, y1, x2, y2])

    # Draw these derived boxes on the first image
    for box in detected_boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        # Red box, like user example
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)


    # Plot 2: Instance Mask (GT Mask) - Unique colors per object
    # Use a colormap with high contrast
    cmap = plt.get_cmap('tab20')
    # We can just imshow the gt_mask, but let's make it pretty by mapping IDs to colors
    colored_inst = cmap(gt_mask.numpy() % 20)
    # Set background (0) to black
    colored_inst[gt_mask == 0] = [0, 0, 0, 1]
    
    axes[1].imshow(colored_inst)
    axes[1].set_title(f"Generated Instance Mask (Input to M3)\n{len(instance_ids)-1} Unique Objects")
    axes[1].axis('off')


    # Plot 3: Semantic Mask (Sem Mask) - Unique colors per CLASS
    # Map class IDs to colors
    sem_np = sem_mask.numpy()
    classes = np.unique(sem_np)
    colored_sem = cmap(sem_np % 20)
    colored_sem[sem_np == 0] = [0, 0, 0, 1] # Background
    
    axes[2].imshow(colored_sem)
    axes[2].set_title(f"Generated Semantic Mask (Input to M4)\n{len(classes)-1} Unique Classes")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    plt.close()

def main():
    # Use the path from the workspace context
    root_dir = "ui_vision_data"
    
    print(f"Loading dataset from {root_dir}...")
    # Using 'element_grounding' split as it likely has the buttons/icons
    ds = UIVisionDataset(
        root_dir=root_dir, 
        split="element_grounding", 
        img_size=(1024, 1024), 
        instance_mode=True,
        download=True
    )
    
    if len(ds) == 0:
        print("Dataset empty. Check path.")
        return

    # Pick a random sample
    idx = np.random.randint(0, len(ds))
    print(f"Visualizing sample {idx} from {ds.items[idx]['img_path']}...")
    visualize_sample(ds, idx=idx, output_file="test_dataset_mask_gen.png")

if __name__ == "__main__":
    main()
