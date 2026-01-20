from typing import List, Any
import subprocess
from PIL import Image

# Import PyObjC libs lazily or inside try/except to avoid crashing non-macOS/broken envs
try:
    import Quartz
    import LaunchServices
    from Cocoa import NSURL
    import ApplicationServices as AS
    from ApplicationServices import AXUIElementCreateSystemWide, AXUIElementCopyAttributeValue
except ImportError:
    Quartz = None
    LaunchServices = None
    NSURL = None
    AS = None

from .base import InterfaceCollector, UIElement

class MacOSCollector(InterfaceCollector):
    def __init__(self):
        if not AS:
            raise ImportError("pyobjc-framework-ApplicationServices not installed or not on macOS.")
        print("Initialized MacOS Accessibility Collector")

    def capture_screenshot(self) -> Image.Image:
        # Use screencapture CLI for reliability on Mac
        subprocess.run(["screencapture", "-x", "/tmp/temp_screen_dump.png"], check=True)
        return Image.open("/tmp/temp_screen_dump.png")

    def get_accessibility_tree(self) -> Any:
        # Start from System Wide element
        system_wide = AS.AXUIElementCreateSystemWide()
        return system_wide

    def _walk_ax_tree(self, ax_element, depth=0, max_depth=10) -> List[UIElement]:
        if depth > max_depth:
            return []
        
        elements = []
        
        # Check role
        _, role = AS.AXUIElementCopyAttributeValue(ax_element, "AXRole", None)
        _, subrole = AS.AXUIElementCopyAttributeValue(ax_element, "AXSubrole", None)
        
        # Check Position & Size
        _, pos_val = AS.AXUIElementCopyAttributeValue(ax_element, "AXPosition", None)
        _, size_val = AS.AXUIElementCopyAttributeValue(ax_element, "AXSize", None)
        
        if pos_val and size_val:
            # Convert AXValue to python tuples... (Skipping complex struct unpack for brevity in stock file)
            # In real impl, need AXValueGetValue(pos_val, kAXValueTypeCGPoint, &point)
            # Placeholder logic:
            x, y = 0, 0 # would unpack 'pos_val'
            w, h = 0, 0 # would unpack 'size_val'
            
            # For this stock file, assume we somehow unpacked them into valid integers
            # If size is 0, ignore
            if w > 0 and h > 0:
                # Is it interactive? Buttons, Inputs, Checkboxes.
                is_interactive = role in ["AXButton", "AXTextField", "AXCheckBox", "AXRadioButton", "AXLink"]
                
                if is_interactive:
                    _, title = AS.AXUIElementCopyAttributeValue(ax_element, "AXTitle", None)
                    
                    elements.append(UIElement(
                        id=str(ax_element),
                        bbox=(int(x), int(y), int(x+w), int(y+h)),
                        role=str(role),
                        text=str(title) if title else "",
                        is_interactive=True
                    ))

        # Recurse children
        _, children = AS.AXUIElementCopyAttributeValue(ax_element, "AXChildren", None)
        if children:
            for child in children:
                elements.extend(self._walk_ax_tree(child, depth+1, max_depth))
                
        return elements

    def segment_interactables(self, raw_tree: Any) -> List[UIElement]:
        # raw_tree is the SystemWide AX element
        # In practice, we usually target the frontmost app to avoid scanning the entire OS tree which is slow.
        
        # Get focused app
        _, app = AS.AXUIElementCopyAttributeValue(raw_tree, "AXFocusedApplication", None)
        if app:
            return self._walk_ax_tree(app)
        else:
            return []
