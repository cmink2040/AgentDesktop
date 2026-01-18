import os
import base64
from PIL import Image
from io import BytesIO
from typing import Dict, Any, Union

try:
    from google import genai
    from google.genai import types
except ImportError:
    # Fallback to verify if maybe user has google-generativeai instead, 
    # but request said reqs.txt has google-genai
    print("Warning: google-genai not installed. Gemini Google Model will not work.")
    genai = None

from .base_model import BaseModel
from .utils import extract_first_bounding_box, extract_first_point

class GeminiGoogleModel(BaseModel):
    def __init__(self, model_name="gemini-2.0-flash-exp"): # Default, but can be overridden
        super().__init__(model_name)
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            print("Warning: GOOGLE_API_KEY not found in environment.")
        
        self.client = None
        self.generation_config = {
            "temperature": 0.0,
            "maxOutputTokens": 2048
        }
    
    def load_model(self, model_name_or_path: str = None, **kwargs):
        if model_name_or_path:
            self.model_name = model_name_or_path
        
        if self.api_key and genai:
            self.client = genai.Client(api_key=self.api_key)
        else:
            raise ValueError("Google GenAI client could not be initialized. Check API Key and installation.")

    def set_generation_config(self, **kwargs):
        # Map common keys to google-genai config keys if necessary
        if "max_tokens" in kwargs:
            self.generation_config["maxOutputTokens"] = kwargs.pop("max_tokens")
        self.generation_config.update(kwargs)

    def _get_image_input(self, image_path_or_obj):
        if isinstance(image_path_or_obj, str):
            img = Image.open(image_path_or_obj)
        else:
            img = image_path_or_obj
        return img

    def ground_only_positive(self, instruction: str, image: Union[str, Any], system_prompt: str = None) -> Dict[str, Any]:
        img = self._get_image_input(image)

        if not system_prompt:
             system_prompt = (
                "You are an expert in using electronic devices and interacting with graphic interfaces. "
                "You should not call any external tools."
            )
        
        prompt_text = (
            "You are asked to find the bounding box of an UI element in the given screenshot corresponding to a given instruction.\n"
            "Don't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1. "
            "If it does not fall within this range, please normalize the coordinates.\n"
            "The instruction is:\n"
            f"{instruction}\n"
        )

        try:
             # configuration
            config = types.GenerateContentConfig(
                temperature=self.generation_config.get("temperature", 0.0),
                max_output_tokens=self.generation_config.get("maxOutputTokens", 2048),
                system_instruction=system_prompt
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[img, prompt_text],
                config=config
            )
            
            response_text = response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return {"result": "error", "raw_response": str(e), "point": None}

        # Parse results
        bbox = extract_first_bounding_box(response_text)
        click_point = extract_first_point(response_text)
        
        if not click_point and bbox:
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        return {
            "result": "positive",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response_text
        }

    def layout_gen(self, name: str, explanation: str, image: Union[str, Any]) -> Dict[str, Any]:
        img = self._get_image_input(image)

        system_prompt = (
            "You are an assistant for analyzing UI layouts. Your task is to identify the exact bounding box "
            "of a functional region based on its name and explanation. \n" 
            "Return the bounding box as [[x0, y0, x1, y1]] with values normalized to [0, 1], "
            "ensuring the box is tight and accurate."
        )

        user_prompt = f"Functional Region Name: {name}\nExplanation: {explanation}\nFind the bounding box."
        
        try:
             # configuration
            config = types.GenerateContentConfig(
                temperature=self.generation_config.get("temperature", 0.0),
                max_output_tokens=self.generation_config.get("maxOutputTokens", 2048),
                system_instruction=system_prompt
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[img, user_prompt],
                config=config
            )
            
            response_text = response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return {"bbox": None, "raw_response": str(e)}

        bbox = extract_first_bounding_box(response_text)
        return {
            "bbox": bbox,
            "raw_response": response_text
        }
