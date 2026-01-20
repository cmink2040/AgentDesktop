import json
import os
import argparse
from typing import List, Dict
from PIL import Image, ImageDraw

def draw_bboxes(image_path: str, json_path: str, output_path: str):
    """
    Loads an image and its corresponding JSON metadata, draws red bounding boxes
    around the UI elements, and saves the result.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    try:
        with open(json_path, 'r') as f:
            elements = json.load(f)
    except Exception as e:
        print(f"Error loading JSON {json_path}: {e}")
        return

    draw = ImageDraw.Draw(image)
    
    count = 0
    for el in elements:
        bbox = el.get("bbox") # [x1, y1, x2, y2]
        if bbox and len(bbox) == 4:
            # Draw rectangle
            draw.rectangle(bbox, outline="red", width=3)
            
            # Optional: Draw ID or Label text slightly above
            label = el.get("label") or el.get("role")
            if label:
                # Basic text drawing
                x1, y1 = bbox[0], bbox[1]
                # Draw a little background for text readability
                text = str(label)
                # left, top, right, bottom = draw.textbbox((x1, y1), text)
                # draw.rectangle((left, top, right, bottom), fill="red")
                draw.text((x1, y1 - 12), text, fill="red")
            
            count += 1

    print(f"Drew {count} bounding boxes on {image_path}")
    image.save(output_path)
    print(f"Saved visualized output to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize UI Elements with Red Boxes")
    parser.add_argument("--image", type=str, required=True, help="Path to source image")
    parser.add_argument("--json", type=str, required=True, help="Path to json metadata")
    parser.add_argument("--output", type=str, default="visualized_debug.png", help="Path to output image")
    
    args = parser.parse_args()
    draw_bboxes(args.image, args.json, args.output)

if __name__ == "__main__":
    main()
