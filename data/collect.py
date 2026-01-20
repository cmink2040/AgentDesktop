import argparse
import sys
import json
import os
import logging
from typing import List, Dict
from PIL import Image
from tqdm import tqdm

# Import Interfaces
from interfaces.base import InterfaceCollector, UIElement
from interfaces.web import WebCollector
from interfaces.macos import MacOSCollector
from interfaces.windows import WindowsCollector
from interfaces.android import AndroidCollector
from interfaces.ios import IOSCollector

from assign_semantic_head import SemanticLabeler
from visualize import draw_bboxes

def get_collector(platform: str, **kwargs) -> InterfaceCollector:
    if platform == 'web':
        return WebCollector(driver_url=kwargs.get('url'))
    elif platform == 'macos':
        return MacOSCollector()
    elif platform == 'windows':
        return WindowsCollector(target_exe=kwargs.get('exe'), target_window_title=kwargs.get('window_title'))
    elif platform == 'android':
        return AndroidCollector(capabilities=kwargs.get('caps'))
    elif platform == 'ios':
        return IOSCollector(capabilities=kwargs.get('caps'))
    else:
        raise ValueError(f"Unknown platform: {platform}")

def save_dataset(image: Image.Image, elements: List[UIElement], output_dir: str, prefix: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Image
    img_path = os.path.join(output_dir, f"{prefix}.png")
    image.save(img_path)
    
    # Save Metadata (Labels)
    data = []
    for el in elements:
        # User requested stripped down format: Only bbox + semantic label
        data.append({
            "bbox": el.bbox,
            "label": el.semantic_label,
            # "id": el.id, # Removed per user request
            # "role": el.role, # Removed per user request 
            # "text": el.text, # Removed per user request
            # "meta": el.metadata # Removed per user request
        })
    
    json_path = os.path.join(output_dir, f"{prefix}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logging.info(f"Saved {img_path} and {json_path}")

def main():
    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(description="UI Data Collection Pipeline")
    parser.add_argument("--platform", type=str, required=True, choices=['web', 'macos', 'windows', 'android', 'ios'])
    parser.add_argument("--output", type=str, default="collected_data")
    parser.add_argument("--label", action="store_true", help="Run semantic labeling with Qwen-VL")
    parser.add_argument("--url", type=str, help="URL for Web collector")
    parser.add_argument("--exe", type=str, help="Path to executable for Windows collector")
    parser.add_argument("--window-title", type=str, help="Title of window to attach to for Windows collector")
    parser.add_argument("--prefix", type=str, default="sample", help="Filename prefix")
    parser.add_argument("--visualize", action="store_true", help="Generate a debug image with red bounding boxes")
    
    args = parser.parse_args()
    
    # 1. Initialize Collector
    try:
        collector = get_collector(args.platform, url=args.url, exe=args.exe, window_title=args.window_title)
    except ImportError as e:
        logging.error(f"Error initializing {args.platform}: {e}")
        sys.exit(1)
        
    logging.info(f"Collector for {args.platform} ready.")
    
    # 2. Collect State
    logging.info("Capturing state...")
    try:
        screenshot, elements = collector.collect_state()
        logging.info(f"Captured screenshot {screenshot.size} and {len(elements)} interactable elements.")
    except Exception as e:
        logging.error(f"Failed to collect state: {e}")
        # For testing purposes, we might want to mock if real collection fails? 
        # But per instructions, 'stock-like file that can do everything' usually implies real logic.
        sys.exit(1)

    # 3. Label (Optional)
    if args.label and len(elements) > 0:
        logging.info("Initializing Semantic Labeler...")
        # Note: Model ID hardcoded in class for now, could be arg
        labeler = SemanticLabeler()
        
        # Limit to reasonable number to prevent freezing on CPU/slow devices
        limit = 50
        if len(elements) > limit:
            logging.warning(f"Notice: Limit enabled. Labeling first {limit} of {len(elements)} elements to avoid indefinite hang.")
            elements_to_label = elements[:limit]
        else:
            elements_to_label = elements

        logging.info(f"Labeling {len(elements_to_label)} elements...")
        for el in tqdm(elements_to_label, desc="Semantic Labeling"):
            labeler.label_element(screenshot, el)
            
    # 4. Save
    save_dataset(screenshot, elements, args.output, args.prefix)

    # 5. Visualize (Red Boxes)
    if args.visualize:
        logging.info("Running Visualization...")
        img_path = os.path.join(args.output, f"{args.prefix}.png")
        json_path = os.path.join(args.output, f"{args.prefix}.json")
        viz_path = os.path.join(args.output, f"{args.prefix}_annotated.png")
        draw_bboxes(img_path, json_path, viz_path)

if __name__ == "__main__":
    main()
