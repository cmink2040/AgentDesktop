import os
import random
import math
import csv
from PIL import Image, ImageDraw

import shutil

def generate_data(num_samples=10000, output_dir="model/data"):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "labels.csv")

    w, h = 600, 350
    colors = ["green", "red", "orange", "blue", "purple"]
    min_dist = 80

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'target_color', 'cursor_x', 'cursor_y', 'target_x', 'target_y', 'dx', 'dy', 'distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_samples):
            if i % 1000 == 0:
                print(f"Generating sample {i}/{num_samples}...")
            
            img = Image.new("RGB", (w, h), "white")
            draw = ImageDraw.Draw(img)

            # Randomize cursor position
            cx = random.randint(20, w - 20)
            cy = random.randint(20, h - 20)
            draw.rectangle([cx-6, cy-6, cx+6, cy+6], fill="black")

            positions = []
            # target_color = random.choice(colors)
            target_color = "red" # Fixed target for single-task learning
            target_center = None

            # Draw buttons
            for color in colors:
                # Try to find a spot
                for _ in range(100):
                    x = random.randint(min_dist, w - min_dist)
                    y = random.randint(min_dist, h - min_dist)
                    
                    # Check overlap with other buttons
                    if all(math.hypot(x - x2, y - y2) > min_dist for x2, y2 in positions):
                        # Check overlap with cursor (don't spawn on top of cursor)
                        if math.hypot(x - cx, y - cy) > 40: 
                            positions.append((x, y))
                            
                            # Draw button (centered at x,y, size 40x30)
                            x1, y1 = x - 20, y - 15
                            x2, y2 = x + 20, y + 15
                            draw.rectangle([x1, y1, x2, y2], fill=color, outline="black")
                            
                            if color == target_color:
                                target_center = (x, y)
                            break
            
            if target_center is None:
                # Should rarely happen with these constraints, but just in case retry or skip
                continue

            tx, ty = target_center
            dx = tx - cx
            dy = ty - cy
            distance = math.hypot(dx, dy)

            filename = f"sample_{i:05d}.png"
            img.save(os.path.join(images_dir, filename))

            writer.writerow({
                'filename': filename,
                'target_color': target_color,
                'cursor_x': cx,
                'cursor_y': cy,
                'target_x': tx,
                'target_y': ty,
                'dx': dx,
                'dy': dy,
                'distance': distance
            })

    print(f"Generated {num_samples} images in {images_dir}")
    print(f"Labels saved to {csv_path}")
    
    # Compress
    print("Compressing dataset...")
    shutil.make_archive(os.path.join(output_dir, "dataset"), 'zip', output_dir)
    print(f"Dataset compressed to {os.path.join(output_dir, 'dataset.zip')}")

if __name__ == "__main__":
    generate_data()
