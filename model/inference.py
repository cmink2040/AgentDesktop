import torch
import argparse
import os
from torchvision import transforms
from PIL import Image
from model.imageshot import ImageShotModel
from safetensors.torch import load_file

class CursorPredictor:
    def __init__(self, model_path="model/checkpoints/imageshot_model.pth", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # For Mac M1/M2/M3
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
        else:
            self.device = device
            
        print(f"Loading model from {model_path} on {self.device}...")
        self.model = ImageShotModel(output_dim=2).to(self.device)
        
        if os.path.exists(model_path):
            if model_path.endswith(".safetensors"):
                state_dict = load_file(model_path)
                self.model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            print(f"Warning: Model checkpoint not found at {model_path}. Using random weights.")

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def predict(self, image_path):
        """
        Predicts (dx, dy) for the given image.
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device) # Add batch dimension

        with torch.no_grad():
            output = self.model(image_tensor)
            dx_norm, dy_norm = output[0].cpu().numpy()
            
            # Denormalize
            w, h = 600.0, 350.0
            dx = dx_norm * w
            dy = dy_norm * h
            
        return dx, dy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Cursor Movement Prediction")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--model", type=str, default="model/checkpoints/imageshot_model.pth", help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        exit(1)

    predictor = CursorPredictor(model_path=args.model)
    dx, dy = predictor.predict(args.image_path)
    
    print(f"Predicted Movement:")
    print(f"dx: {dx:.4f}")
    print(f"dy: {dy:.4f}")
