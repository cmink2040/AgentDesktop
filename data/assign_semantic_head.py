import torch
import logging
from PIL import Image
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    # Fallback/Mock if libraries missing
    logging.warning(f"Warning: Semantic Labeling Dependency Missing: {e}")
    Qwen2_5_VLForConditionalGeneration = None

# Fix import relative to execution root
# When imported from collect.py in the same dir, 'interfaces' is directly available
try:
    from interfaces.base import UIElement
except ImportError:
    # If used as part of a package structure in the future
    from .interfaces.base import UIElement

class SemanticLabeler:
    """
    Uses Qwen2.5-VL-3B to assign semantic functional labels to UI elements.
    """
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", device=None):
        # Auto-detect device if not provided
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logging.info(f"Loading VLM: {model_id} on {self.device}...")
        
        if Qwen2_5_VLForConditionalGeneration is None:
            logging.warning("Transformers/Qwen libraries not found. Running in Mock Mode.")
            self.model = None
            return

        try:
            # Note: This requires the model to actually exist on HF. 
            # If 3B VL doesn't exist, try Qwen/Qwen2-VL-2B-Instruct
            
            # Determine dtype
            if self.device == "cuda":
                dtype = torch.float16
            elif self.device == "mps":
                dtype = torch.float16 # MPS supports float16
            else:
                dtype = torch.float32

            logging.info("Downloading/Loading model... (This may take several minutes for the first run)")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=dtype, 
                device_map="auto" if self.device == "cuda" else None, # accelerate auto-map is mostly for CUDA
            )
            if self.device != "cuda" and self.device != "auto":
                self.model.to(self.device)
                
            self.processor = AutoProcessor.from_pretrained(model_id)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model {model_id}: {e}")
            logging.warning("Running in Mock Mode for Semantic Labeling.")
            self.model = None

    def label_element(self, full_image: Image.Image, element: UIElement) -> str:
        if self.model is None:
            return f"mock_label_{element.role}_{element.id[-4:]}"

        # Strategy: Crop with context
        # We add 20% padding to capture context (labels next to checkboxes, etc)
        w, h = full_image.size
        x1, y1, x2, y2 = element.bbox
        
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        
        crop = full_image.crop((cx1, cy1, cx2, cy2))
        
        # Prepare Prompt
        # Updated prompt per user request: "what does this do in a few words"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": crop,
                    },
                    {"type": "text", "text": "What does this do in a few words? Focus on the centered element."},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=32)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        label = output_text[0].strip()
        element.semantic_label = label
        return label
