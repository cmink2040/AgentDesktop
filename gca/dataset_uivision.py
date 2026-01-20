# gca/dataset_uivision.py
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


class UIVisionDataset(Dataset):
    """
    Adapter for ServiceNow/ui-vision dataset to GCA format.
    Automatically downloads from HuggingFace if root_dir doesn't exist.
    
    Transforms:
        - Image: Loaded, resized, converted to Tensor [3, H, W]
        - Annotations (JSON BBoxes): Rasterized into dense segmentation mask [H, W]
    
    The mask values will represent:
        0: Background
        1..N: Instance IDs (if instance_mode=True) OR Class IDs (if instance_mode=False)
    """

    def __init__(
        self,
        root_dir: str = "./ui-vision-data",
        split: str = "element_grounding", # subfolder name in annotations/images
        img_size: Tuple[int, int] = (1024, 1024),
        instance_mode: bool = False, # If True, mask IDs are unique object IDs. If False, mask IDs are semantic classes.
        max_instances: int = 512, # Max objects to encode if using fixed channels
        download: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.instance_mode = instance_mode
        self.max_instances = max_instances
        self.split = split
        self.class_to_id: Dict[str, int] = {"__background__": 0}

        # Auto-download
        if download and not os.path.exists(root_dir):
            if snapshot_download is None:
                raise ImportError("huggingface_hub not installed. Install it or set download=False.")
            print(f"Downloading ServiceNow/ui-vision to {root_dir}...")
            snapshot_download(
                repo_id="ServiceNow/ui-vision",
                repo_type="dataset",
                local_dir=root_dir,
                allow_patterns=["annotations/*", "images/*", "README.md"]
            )
            print("Download complete.")

        # Setup paths
        # Structure:
        # root/
        #   annotations/element_grounding/*.json
        #   images/element_grounding/*.png
        
        self.ann_dir = os.path.join(root_dir, "annotations", split)
        self.img_dir = os.path.join(root_dir, "images", split)

        if not os.path.exists(self.ann_dir):
            # Fallback for flat structure or different splits, adjust as needed
            print(f"Warning: {self.ann_dir} not found. Checking if flat...")
        
        self.items = self._load_index()
        self.num_classes = len(self.class_to_id)
        print(f"Loaded {len(self.items)} samples from {split}")
        print(f"Detected {self.num_classes - 1} classes (plus background)")

    def _load_index(self) -> List[Dict]:
        """
        Parses JSON annotations to build an index.
        Groups flat entries by image_path to form dense annotations.
        """
        # Dictionary to group annotations by image
        # key: full_image_path, value: {'boxes': [], 'classes': []}
        grouped = {}
        
        # Check files
        all_jsons = [f for f in os.listdir(self.ann_dir) if f.endswith('.json') and not f.startswith('._')]
        if not all_jsons:
            print(f"No JSONs found in {self.ann_dir}")
            return []

        print(f"Parsing {len(all_jsons)} annotation files...")

        for j_file in all_jsons:
            path = os.path.join(self.ann_dir, j_file)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Skipping {j_file}: {e}")
                continue
            
            # Expecting a flat list of items
            if isinstance(data, list):
                for entry in data:
                    # Parse entry
                    # Image path (e.g., "element_grounding/xyz.png")
                    rel_path = entry.get('image_path') or entry.get('file_name')
                    # Or generic handling
                    if not rel_path:
                        continue

                    # Construct absolute path
                    # Dataset structure: root/images/ + rel_path
                    full_img_path = os.path.join(self.root_dir, "images", rel_path)
                    
                    if not os.path.exists(full_img_path):
                        # Try relative to the split dir if rel_path is just filename
                        alt_path = os.path.join(self.img_dir, os.path.basename(rel_path))
                        if os.path.exists(alt_path):
                            full_img_path = alt_path
                        else:
                            # Skip if image not found
                            continue

                    # Initialize group if needed
                    if full_img_path not in grouped:
                        grouped[full_img_path] = {'boxes': [], 'classes': []}

                    # Extract Box
                    # Check for 'bbox' (flat list) or 'bounds'
                    bbox = entry.get('bbox') or entry.get('bounds')
                    if not bbox or len(bbox) != 4:
                        continue
                    
                    # Assume bbox is [x1, y1, x2, y2] based on json inspection
                    # If w, h were involved, we'd see vastly distinct values usually.
                    # 1814..1857 is range 43. 0..36 is range 36.
                    # It's likely [x1, y1, x2, y2]
                    x1, y1, x2, y2 = bbox
                    
                    # Basic validation
                    if x2 <= x1 or y2 <= y1:
                        # Maybe it IS xywh?
                        # If x2 < x1, maybe it was w, h?
                        # Let's assume xywh if x2/y2 are small?
                        # But 1814 is large.
                        pass
                    
                    grouped[full_img_path]['boxes'].append([x1, y1, x2, y2])
                    # Resolve class label
                    class_name = entry.get("category") or entry.get("element_type") or entry.get("label") or "unknown"
                    if class_name not in self.class_to_id:
                        self.class_to_id[class_name] = len(self.class_to_id)
                    grouped[full_img_path]['classes'].append(self.class_to_id[class_name])

        # Convert back to list format for dataset
        items = []
        for img_path, ann in grouped.items():
            items.append({
                'img_path': img_path,
                'boxes': ann['boxes'],
                'classes': ann['classes']
            })
            
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Load Image
        try:
            img = Image.open(item['img_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {item['img_path']}: {e}")
            # return dummy or skip
            img = Image.new('RGB', (100, 100))
            
        w_orig, h_orig = img.size
        
        # Resize Image
        img_t = F.resize(img, self.img_size)
        img_t = F.to_tensor(img_t) # [0,1], [3,H,W]
        
        # Create Mask (Rasterization)
        # 0 = background
        h_out, w_out = self.img_size
        gt_mask = torch.zeros((h_out, w_out), dtype=torch.long)
        sem_mask = torch.zeros((h_out, w_out), dtype=torch.long)
        
        # Comp Targets (if using classification)
        # We also need to return the strict list of classes matching the mask IDs
        # This is tricky because rasterization might hide some small boxes behind big ones.
        # Actually in UI, Small buttons are on TOP of Big Windows. 
        # So we paint Largest First, Smallest Last.
        
        # Sort boxes by area (descending) so small ones are painted last (on top)
        boxes = []
        for i, b in enumerate(item['boxes']):
            area = (b[2]-b[0]) * (b[3]-b[1])
            boxes.append((area, b, item['classes'][i]))
            
        # Bigger area first -> painted first. Smaller area last -> paints over.
        boxes.sort(key=lambda x: x[0], reverse=True)
        
        # Scaling factors
        sx = w_out / w_orig
        sy = h_out / h_orig
        
        active_classes = {} # Map ID -> Class
        
        current_id = 1
        for area, box, cls_id in boxes:
            x1, y1, x2, y2 = box
            
            # Scale
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy
            
            # Clip
            x1 = max(0, min(w_out, int(x1)))
            x2 = max(0, min(w_out, int(x2)))
            y1 = max(0, min(h_out, int(y1)))
            y2 = max(0, min(h_out, int(y2)))
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Paint
            # Use instance ID if mode is True, else use Class ID for main mask
            val = current_id if self.instance_mode else cls_id
            
            # Simple rectangle fill
            # Optim: could do this with tensor ops if many boxes
            gt_mask[y1:y2, x1:x2] = val
            sem_mask[y1:y2, x1:x2] = cls_id
            
            if self.instance_mode:
                active_classes[current_id] = cls_id
                current_id += 1
                if current_id >= self.max_instances:
                   break
        
        return {
            "image": img_t,
            "gt_mask": gt_mask,
            "sem_mask": sem_mask,
            # "comp_targets": ... (Optional, can be derived if we have fixed assignment)
        }

def test_loader():
    # Simple debug entry point
    ds = UIVisionDataset("/Volumes/512-GB-FS/Source/AgentDesktop/extern/UI-Vision", split="eval", instance_mode=True)
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        print("Image shape:", sample['image'].shape)
        print("Mask shape:", sample['gt_mask'].shape)
        print("Unique IDs in mask:", torch.unique(sample['gt_mask']))

if __name__ == "__main__":
    test_loader()
