from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

class BaseModel(ABC):
    """
    Abstract base class for models in the UI-Vision benchmark.
    This serves as a template for integrating new models.
    """

    def __init__(self, model_name: str = None, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def load_model(self, model_name_or_path: str = None, **kwargs):
        """
        Load the model and its resources.
        """
        pass

    @abstractmethod
    def set_generation_config(self, **kwargs):
        """
        Set or update generation configuration (temperature, tokens, etc).
        """
        pass

    @abstractmethod
    def ground_only_positive(self, instruction: str, image: Union[str, Any], system_prompt: str = None) -> Dict[str, Any]:
        """
        Perform element grounding task.
        
        Args:
            instruction (str): The prompt/instruction for what to find.
            image (str or Image): Path to image or Image object.
            system_prompt (str, optional): System prompt override.

        Returns:
            dict: {
                "result": "positive",
                "bbox": [x1, y1, x2, y2], # normalized 0-1 (optional)
                "point": [x, y],          # normalized 0-1 (required)
                "raw_response": str       # full model response
            }
        """
        pass

    @abstractmethod
    def layout_gen(self, name: str, explanation: str, image: Union[str, Any]) -> Dict[str, Any]:
        """
        Perform layout generation task.

        Args:
            name (str): Name of element.
            explanation (str): description.
            image: input image.
            
        Returns:
            dict: {
                "bbox": [x1,y1,x2,y2],
                "raw_response": str
            }
        """
        pass
