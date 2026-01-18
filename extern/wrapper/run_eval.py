import sys
import os
import argparse

# Config paths
current_dir = os.path.dirname(os.path.abspath(__file__))
# Point to .../extern/UI-Vision/eval/grounding
ui_vision_grounding_dir = os.path.abspath(os.path.join(current_dir, '../UI-Vision/eval/grounding'))

# Add the UI-Vision grounding directory to sys.path so we can import modules from it
if ui_vision_grounding_dir not in sys.path:
    sys.path.insert(0, ui_vision_grounding_dir)

# Add the current directory to sys.path so we can import custom_models
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import eval_uivision
except ImportError:
    print(f"Error: Could not import eval_uivision from {ui_vision_grounding_dir}")
    sys.exit(1)

# Import custom models
try:
    from custom_models.gemini_google import GeminiGoogleModel
except ImportError as e:
    print(f"Error importing custom models: {e}")
    sys.exit(1)

# Save original build_model function
original_build_model = eval_uivision.build_model

def custom_build_model(args):
    """
    Patched build_model function to support custom models.
    """
    model_type = args.model_type
    
    if model_type == "gemini_google":
        if args.model_name_or_path:
             model = GeminiGoogleModel(model_name=args.model_name_or_path)
        else:
             # Default to a recent one if not specified
             model = GeminiGoogleModel(model_name="gemini-2.0-flash-exp")
        model.load_model()
        return model
    else:
        # Fallback to the original build_model for standard models
        return original_build_model(args)

# Apply the patch
eval_uivision.build_model = custom_build_model

def main():
    # Use eval_uivision's argument parser
    args = eval_uivision.parse_args()
    
    # Run the main evaluation loop
    eval_uivision.main(args)

if __name__ == "__main__":
    main()
