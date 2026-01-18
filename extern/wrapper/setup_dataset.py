import os
import argparse
from datasets import load_dataset
from PIL import Image
import json
import shutil

def setup_dataset(output_dir):
    """
    Downloads and prepares the UI-Vision dataset from Hugging Face.
    """
    print(f"Downloading dataset to {output_dir}...")
    
    # Create necessary directories
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Load the dataset
    # The dataset has multiple configs: 'element_grounding', 'layout_grounding', 'action_prediction'
    # We'll focus on grounding as per previous context, but I should probably check what's available
    
    try:
        # Load element grounding
        print("Loading Element Grounding dataset...")
        ds_element = load_dataset("ServiceNow/ui-vision", "element_grounding", split="test")
        
        # Load layout grounding
        print("Loading Layout Grounding dataset...")
        ds_layout = load_dataset("ServiceNow/ui-vision", "layout_grounding", split="test")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Process Element Grounding
    element_tasks = []
    print("Processing Element Grounding images and metadata...")
    for item in ds_element:
        # Save image
        # Assuming 'image' column contains PIL Image objects or paths
        # Hugging Face datasets usually convert images to PIL
        image = item['image']
        image_filename = f"{item['id']}.png" # Assuming 'id' exists and is unique
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            image.save(image_path)
            
        # Construct task entry compatible with eval_uivision.py
        # Based on previous read of eval_uivision.py logic
        task = {
            "image_path": image_filename,
            "platform": item.get('platform', 'unknown'),
            "prompt_to_evaluate": item['instruction'], # naming might vary
            "image_size": image.size,
            "bbox": item.get('bbox', None), # might need normalization check
            "ground_truth": item.get('bbox', None), # standardized to bbox
            "element_type": item.get('element_type', 'unknown')
        }
        element_tasks.append(task)

    with open(os.path.join(output_dir, "element_grounding_test.json"), "w") as f:
        json.dump(element_tasks, f, indent=4)


    # Process Layout Grounding
    layout_tasks = []
    print("Processing Layout Grounding images and metadata...")
    for item in ds_layout:
        image = item['image']
        image_filename = f"{item['id']}_layout.png"
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            image.save(image_path)

        task = {
            "image_path": image_filename,
            "platform": item.get('platform', 'unknown'),
            "name": item.get('name', ''),
            "explanation": item.get('explanation', ''),
            "image_size": image.size,
            "bbox": item.get('bbox', None)
        }
        layout_tasks.append(task)

    with open(os.path.join(output_dir, "layout_grounding_test.json"), "w") as f:
        json.dump(layout_tasks, f, indent=4)

    print("Dataset setup complete.")
    print(f"Images stored in: {images_dir}")
    print(f"Element Grounding Test File: {os.path.join(output_dir, 'element_grounding_test.json')}")
    print(f"Layout Grounding Test File: {os.path.join(output_dir, 'layout_grounding_test.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store dataset")
    args = parser.parse_args()
    
    setup_dataset(args.output_dir)
