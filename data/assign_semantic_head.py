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
            # Enhanced CUDA support
            attn_implementation = None
            if self.device == "cuda":
                dtype = torch.float16
                # Force SDPA (Scaled Dot Product Attention) for stability on Windows.
                # 'flash_attention_2' binary incompatibility often causes silent crashes (segfaults).
                logging.info("Using 'sdpa' attention implementation for stability.")
                attn_implementation = "sdpa" 
            elif self.device == "mps":
                dtype = torch.float16 # MPS supports float16
            else:
                dtype = torch.float32

            # Configure Max Memory to use full device capacity if on CUDA
            max_memory = None
            if self.device == "cuda":
                max_memory = {}
                for i in range(torch.cuda.device_count()):
                    # Retrieve total memory in bytes
                    total_mem = torch.cuda.get_device_properties(i).total_memory
                    # Safe margin: 90% instead of 99% to prevent OS/display driver driver conflicts causing crashes
                    max_memory[i] = int(total_mem * 0.90)
                logging.info(f"Configured max_memory usage: {max_memory}")

            logging.info("Downloading/Loading model... (This may take several minutes for the first run)")
            
            # Additional safety params
            # low_cpu_mem_usage=True helps avoid RAM OOM during load
            print("DEBUG: calling from_pretrained...")
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, 
                    dtype=dtype, 
                    device_map="auto" if self.device == "cuda" else None, 
                    max_memory=max_memory if self.device == "cuda" else None,
                    low_cpu_mem_usage=True,
                    attn_implementation=attn_implementation,
                )
                print("DEBUG: from_pretrained returned.")
            except Exception as load_err:
                 print(f"DEBUG: from_pretrained raised exception: {load_err}")
                 raise load_err

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
        tight_crop = full_image.crop((x1, y1, x2, y2))
        
        # Resize to prevent Token Explosion/OOM on large elements (Panes, Groups, etc)
        # 3B model with 12GB VRAM cannot handle 30k+ tokens from 2x 4K crops.
        # 1024px max dimension is usually enough for semantic understanding.
        def safe_resize(img, max_dim=768):
            if max(img.size) > max_dim:
                img = img.copy() # Ensure we don't modify original full_image if it was passed by ref (unlikely here but safe)
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            return img

        crop = safe_resize(crop, 768)
        tight_crop = safe_resize(tight_crop, 512)

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
                    {
                        "type": "image",
                        "image": tight_crop,
                    },
                    {"type": "text", "text": "The first image is the context, and the second image is the specific icon. Generate a 5 word summary of the label."},
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
        try:
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

        except torch.cuda.OutOfMemoryError:
            logging.error(f"OOM Error labeling element {element.id}. Skipping.")
            return "error_oom"
        finally:
            # Aggressive cleanup
            del inputs, image_inputs, video_inputs
            if 'generated_ids' in locals(): del generated_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
