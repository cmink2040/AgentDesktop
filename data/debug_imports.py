import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"Torch version: {torch.__version__}")
except ImportError as e:
    print(f"Torch import failed: {e}")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    from transformers import Qwen2_5_VLForConditionalGeneration
    print("Qwen2_5_VLForConditionalGeneration imported successfully")
except ImportError as e:
    print(f"Transformers/Qwen import failed: {e}")
except Exception as e:
    print(f"Transformers/Qwen check failed with {type(e)}: {e}")

try:
    import qwen_vl_utils
    print("qwen_vl_utils imported successfully")
except ImportError as e:
    print(f"qwen_vl_utils import failed: {e}")
