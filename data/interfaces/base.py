from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image

@dataclass
class UIElement:
    """
    Represents a segmented interactive element from the UI.
    """
    id: str  # Unique identifier (e.g., XPath, accessibility ID, memory address)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    role: str  # e.g., "button", "link", "input"
    text: Optional[str] = None
    is_interactive: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Placeholder for the semantic label assigned by VLM later
    semantic_label: Optional[str] = None 

    def area(self) -> int:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

class InterfaceCollector(ABC):
    """
    Abstract base class for platform-specific UI collectors.
    """
    
    @abstractmethod
    def capture_screenshot(self) -> Image.Image:
        """
        Captures the current screen or window state as an image.
        """
        pass

    @abstractmethod
    def get_accessibility_tree(self) -> Any:
        """
        Retrieves the raw accessibility tree or DOM from the platform.
        """
        pass

    @abstractmethod
    def segment_interactables(self, raw_tree: Any) -> List[UIElement]:
        """
        Parses the raw tree to Extract interactable UIElements with valid bounding boxes.
        """
        pass

    def collect_state(self) -> Tuple[Image.Image, List[UIElement]]:
        """
        Main entry method: captures screenshot and segments interactables.
        """
        screenshot = self.capture_screenshot()
        raw_tree = self.get_accessibility_tree()
        elements = self.segment_interactables(raw_tree)
        
        # Initial filter: Remove elements outside screen bounds or empty
        valid_elements = []
        w, h = screenshot.size
        for el in elements:
            x1, y1, x2, y2 = el.bbox
            if x2 <= x1 or y2 <= y1:
                continue
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                # Optionally clip or discard. Discarding for now.
                continue
            valid_elements.append(el)
            
        return screenshot, valid_elements
