import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models import GCAEnd2EndModel, ModelConfig

def process_image(image_path, img_size=(1024, 1024), device="cpu"):
    """
    Load and preprocess the image.
    """
    try:
        img_orig = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None, None

    # Resize Image
    img_resized = F.resize(img_orig, img_size)
    img_tensor = F.to_tensor(img_resized).unsqueeze(0) # [1, 3, H, W]
    
    return img_tensor.to(device), img_orig

def visualize_result(img_pil, output, threshold=0.5):
    """
    Visualize the predicted components.
    output.learned.comps gives us component centers and assignments.
    output.learned.sem gives us classification.
    """
    
    # This largely depends on what the model outputs exactly.
    # Looking at models.py/structures.py context from earlier:
    # ModelOutput has .learned and .system
    # .learned.comps has assign_probs [B, N, K] probably? Or [B, HW, K]?
    # Let's assume we can confirm the structure from models.py or just standard GCA understanding.
    # Usually GCA outputs component centers or masks.
    
    # For this simple inference script, let's visualize the "Hard Assignment" from the system layer if available
    # or the soft assignment from comps.
    
    # Let's look at system.py in previous turn.
    # system.hard_assign is [B, N]. N = H*W / (4*4)? Or M2 tokens?
    
    pass

def main():
    parser = argparse.ArgumentParser(description="Run GCA Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")

    # 1. Load Weights first to determine config
    # We need to know num_classes from the checkpoint to init the model correctly
    try:
        if args.checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.checkpoint)
        else:
            state_dict = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint file: {e}")
        return

    # Auto-detect num_classes
    num_classes = 64 # Default
    if "m4.classifier.3.weight" in state_dict:
        num_classes = state_dict["m4.classifier.3.weight"].shape[0]
        print(f"Detected num_classes={num_classes} from checkpoint.")
    
    # 2. Initialize Model Architecture 
    model_cfg = ModelConfig(
        c4=96, c8=192, c16=384, c32=768,
        d_model=256,
        n_max=4096,
        k_components=256, 
        num_classes=num_classes,   
    )
    model = GCAEnd2EndModel(model_cfg)
    
    # 3. Apply Weights
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys in checkpoint: {unexpected}")
    except Exception as e:
        print(f"Failed to load state_dict into model: {e}")

    model.to(args.device)
    model.eval()

    # 3. Process Image
    img_tensor, img_pil = process_image(args.image, device=args.device)
    if img_tensor is None:
        return

    print("Running inference...")
    with torch.no_grad():
        output = model(img_tensor)
    
    # 4. Interpret Output
    system = output.system
    micro = output.learned.micro
    
    # [N]
    hard_assign = system.hard_assign[0].cpu().numpy() 
    # [N, 4] normalized xyxy
    token_boxes = micro.boxes[0].cpu().numpy()
    
    # Resize boxes to image size
    w, h = img_pil.size
    token_boxes[:, 0] *= w
    token_boxes[:, 1] *= h
    token_boxes[:, 2] *= w
    token_boxes[:, 3] *= h
    
    print("Inference finished. Visualizing...")
    
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img_pil)
    
    # Get unique components
    unique_comps = np.unique(hard_assign)
    cmap = matplotlib.colormaps['tab20']
    
    for i, comp_id in enumerate(unique_comps):
        # Background component? Usually we might ignore empty ones or -1
        # Check masks
        mask = (hard_assign == comp_id)
        if not np.any(mask):
            continue
            
        # Get all token boxes for this component
        c_boxes = token_boxes[mask]
        
        # Calculate Union Box (min_x, min_y, max_x, max_y)
        x1 = np.min(c_boxes[:, 0])
        y1 = np.min(c_boxes[:, 1])
        x2 = np.max(c_boxes[:, 2])
        y2 = np.max(c_boxes[:, 3])
        
        # Draw Box
        color = cmap(i % 20)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Optional: Label with Component ID or Class if available
        # ModelOutput.learned.sem.class_logits -> [B, K, C]
        if output.learned.sem is not None:
            # class_logits: [B, K, C]
            # comp_id is 'k'
            logits = output.learned.sem.class_logits[0, comp_id] # [C]
            class_id = torch.argmax(logits).item()
            
            # Print for user
            print(f"\"Class {class_id}\" {{{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}}}")
            
            # Place text BELOW the box (y2)
            # Add a small offset so it doesn't overlap the line exactly
            text_y = y2 + 5
            # Ensure it doesn't go off-screen
            text_y = min(text_y, h - 10)
            
            ax.text(x1, text_y, f"ID:{comp_id} C:{class_id}", color='white', fontsize=8, 
                    bbox=dict(facecolor=color, alpha=0.7, pad=1), va='top')

    plt.axis('off')
    plt.title(f"GCA Prediction: {len(unique_comps)} Components Detected")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
