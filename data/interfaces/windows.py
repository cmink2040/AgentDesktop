from typing import List, Any
from PIL import Image, ImageGrab
from .base import InterfaceCollector, UIElement

try:
    from pywinauto import Desktop
    import pywinauto
except ImportError:
    pywinauto = None

class WindowsCollector(InterfaceCollector):
    def __init__(self):
        if not pywinauto:
            raise ImportError("pywinauto not installed. Run `pip install pywinauto`.")
        self.desktop = Desktop(backend="uia")

    def capture_screenshot(self) -> Image.Image:
        return ImageGrab.grab()

    def get_accessibility_tree(self) -> Any:
        # Return the wrapper for the active window or root
        # Returning root elements list
        return self.desktop.windows()

    def _walk_uia_tree(self, wrapper) -> List[UIElement]:
        elements = []
        
        # Check current wrapper
        if wrapper.is_visible():
            rect = wrapper.rectangle() # (left, top, right, bottom)
            
            # Simple heuristic for interactivity based on control type
            ctype = wrapper.element_info.control_type
            is_interactive = ctype in ["Button", "Edit", "ListItem", "MenuItem", "CheckBox", "ComboBox"]
            
            if is_interactive:
                elements.append(UIElement(
                    id=str(wrapper.handle if hasattr(wrapper, 'handle') else id(wrapper)),
                    bbox=(rect.left, rect.top, rect.right, rect.bottom),
                    role=ctype,
                    text=wrapper.window_text(),
                    is_interactive=True
                ))
            
            # Recurse
            # pywinauto walk is implicit via descendants, but that can be slow.
            # `wrapper.children()` provides immediate children.
            for child in wrapper.children():
                elements.extend(self._walk_uia_tree(child))
                
        return elements

    def segment_interactables(self, raw_tree: Any) -> List[UIElement]:
        # raw_tree is a list of top-level windows
        all_elements = []
        for win in raw_tree:
            # Maybe only focus on the active one?
            # For collecting data, we might want everything visible.
            if win.is_visible():
                all_elements.extend(self._walk_uia_tree(win))
        return all_elements
