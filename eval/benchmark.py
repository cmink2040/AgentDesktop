import argparse
import math
import random
import sys
import os
import json
from PIL import Image, ImageDraw

# Ensure we can import from adt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adt.agent_func import Agent
from adt.utility import draw_grid

class Benchmark:
    def __init__(self, agents, env_setup_func):
        """
        :param agents: Dict of {name: agent_func}
        :param env_setup_func: Function that returns (target_coords, image_path, instruction).
        """
        self.agents = agents
        self.env_setup_func = env_setup_func

    def determine_distance(self, x1, y1, xl, yl, x2, y2):
        """
        Calculate distance from point (x2, y2) to the rectangle defined by center (x1, y1) and size (xl, yl).
        If point is inside, distance is 0.
        """
        xmin = x1 - xl / 2
        xmax = x1 + xl / 2
        ymin = y1 - yl / 2
        ymax = y1 + yl / 2

        if x2 < xmin:
            dx = xmin - x2
        elif x2 > xmax:
            dx = x2 - xmax
        else:
            dx = 0
        if y2 < ymin:
            dy = ymin - y2
        elif y2 > ymax:
            dy = y2 - ymax
        else:
            dy = 0
        
        return (dx ** 2 + dy ** 2) ** 0.5

    def simulate_actions(self, actions, start_x, start_y, canvas_width, canvas_height):
        """
        Simulate the cursor movement based on actions.
        Returns final (x, y).
        """
        curr_x, curr_y = start_x, start_y
        
        for action in actions:
            parts = action.split()
            if parts[0] == "move":
                direction = parts[1]
                try:
                    amount = int(parts[2])
                except (IndexError, ValueError):
                    amount = 1 # Default if not specified or parse error, though prompt says amount is there
                
                step = 10 * amount # Agent moves in 10px units? 
                # Wait, prompt says: "Move the cursor by 10px ... up/down/left/right"
                # And example: "move right 1". So 1 unit = 10px.
                
                if direction == "left":  curr_x -= step
                elif direction == "right": curr_x += step
                elif direction == "up":    curr_y -= step
                elif direction == "down":  curr_y += step
                
                # Clamp
                curr_x = max(0, min(curr_x, canvas_width))
                curr_y = max(0, min(curr_y, canvas_height))
                
            elif parts[0] == "click":
                pass # Click doesn't move cursor
                
        return curr_x, curr_y

    def run(self, num_tests=5, output_file="eval/results.json"):
        all_results = {name: [] for name in self.agents}
        print(f"Running {num_tests} tests for agents: {list(self.agents.keys())}...")
        
        for i in range(num_tests):
            print(f"\nTest {i+1}/{num_tests}")
            
            # Setup environment
            target_info, img_path, instruction = self.env_setup_func()
            target_x, target_y, target_w, target_h = target_info
            
            # Initial cursor position (center of image usually)
            img = Image.open(img_path)
            w, h = img.size
            start_x, start_y = w // 2, h // 2
            
            print(f"Instruction: {instruction}")
            
            for agent_name, agent_func in self.agents.items():
                print(f"  Running {agent_name}...")
                try:
                    actions = agent_func(instruction, img_path)
                    print(f"    Actions: {actions}")
                except Exception as e:
                    print(f"    Agent failed: {e}")
                    all_results[agent_name].append({"success": False, "distance": None, "error": str(e)})
                    continue

                # Simulate result
                final_x, final_y = self.simulate_actions(actions, start_x, start_y, w, h)
                
                # Calculate distance
                dist = self.determine_distance(target_x, target_y, target_w, target_h, final_x, final_y)
                
                success = (dist == 0)
                print(f"    Final Pos: ({final_x}, {final_y}), Target: ({target_x}, {target_y}), Dist: {dist:.2f}, Success: {success}")
                
                all_results[agent_name].append({
                    "test_id": i,
                    "success": success,
                    "distance": dist,
                    "actions": actions
                })
            
        # Save to JSON
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_file}")
            
        # Summary
        print("\nBenchmark Complete.")
        for agent_name, results in all_results.items():
            valid_results = [r for r in results if r.get("distance") is not None]
            success_count = sum(1 for r in valid_results if r.get("success"))
            avg_dist = sum(r["distance"] for r in valid_results) / len(valid_results) if valid_results else 0
            print(f"{agent_name}: Success Rate: {success_count}/{num_tests} ({success_count/num_tests*100:.1f}%), Avg Dist: {avg_dist:.2f}px")


def mock_env_setup():
    """
    Generates a synthetic test case without using Tkinter.
    Creates an image with random colored squares and picks one as target.
    """
    w, h = 600, 350
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    
    colors = ["green", "red", "orange", "blue", "purple"]
    targets = []
    
    # Draw cursor at center
    cx, cy = w // 2, h // 2
    draw.rectangle([cx-6, cy-6, cx+6, cy+6], fill="black")
    
    # Draw buttons
    min_dist = 80
    positions = []
    
    # target_color = random.choice(colors)
    target_color = "red" # Fixed target for benchmark fairness
    target_rect = None
    
    for color in colors:
        # Try to find a spot
        for _ in range(100):
            x = random.randint(min_dist, w - min_dist)
            y = random.randint(min_dist, h - min_dist)
            if all(math.hypot(x - x2, y - y2) > min_dist for x2, y2 in positions):
                positions.append((x, y))
                # Draw button (centered at x,y, size 40x30)
                x1, y1 = x - 20, y - 15
                x2, y2 = x + 20, y + 15
                draw.rectangle([x1, y1, x2, y2], fill=color, outline="black")
                
                if color == target_color:
                    target_rect = (x, y, 40, 30) # center_x, center_y, w, h
                break
    
    if not os.path.exists("img"):
        os.makedirs("img")
    img_path = "img/benchmark_test.png"
    img.save(img_path)
    
    return target_rect, img_path, f"click {target_color}"

def get_model_agent(model_path="model/checkpoints/imageshot_model.pth", use_grid=False):
    from model.inference import CursorPredictor
    from adt.utility import draw_grid
    
    predictor = CursorPredictor(model_path=model_path)
    
    def model_agent_func(instruction, img_path):
        # If using grid model, we must superimpose grid first
        inference_img_path = img_path
        if use_grid:
            inference_img_path = "img/benchmark_grid_temp.png"
            draw_grid(img_path, inference_img_path, arrsize=100)

        # Model ignores instruction, just predicts movement
        dx, dy = predictor.predict(inference_img_path)
        # Convert dx, dy to "move" actions
        # dx, dy are pixels. Agent moves in 10px steps.
        actions = []
        
        # X movement
        steps_x = int(dx / 10)
        if steps_x > 0:
            actions.append(f"move right {steps_x}")
        elif steps_x < 0:
            actions.append(f"move left {-steps_x}")
            
        # Y movement
        steps_y = int(dy / 10)
        if steps_y > 0:
            actions.append(f"move down {steps_y}")
        elif steps_y < 0:
            actions.append(f"move up {-steps_y}")
            
        actions.append("click")
        return actions
    return model_agent_func

def get_default_agent():
    agent_instance = Agent()
    # Wrapper to handle new return signature (actions, points)
    def wrapper(instruction, img_path):
        actions, _ = agent_instance.ask(instruction, img_path, mode="Gemini")
        return actions
    return wrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Agent Benchmark")
    parser.add_argument("--tests", type=int, default=5, help="Number of tests to run")
    parser.add_argument("--agent", type=str, default="default", choices=["default", "model", "hybrid", "grid", "all"], help="Agent to use")
    
    args = parser.parse_args()
    
    agents = {}
    
    if args.agent == "default":
        agents["Gemini"] = get_default_agent()
    elif args.agent == "model":
        agents["ImageShot"] = get_model_agent()
    elif args.agent == "grid":
        agents["ImageShot-Grid"] = get_model_agent(model_path="model/checkpoints/imageshot_grid.safetensors", use_grid=True)
    elif args.agent == "hybrid":
        agents["Gemini"] = get_default_agent()
        agents["ImageShot"] = get_model_agent()
    elif args.agent == "all":
        agents["Gemini"] = get_default_agent()
        agents["ImageShot"] = get_model_agent()
        agents["ImageShot-Grid"] = get_model_agent(model_path="model/checkpoints/imageshot_grid.safetensors", use_grid=True)

    benchmark = Benchmark(agents, mock_env_setup)
    benchmark.run(num_tests=args.tests)


