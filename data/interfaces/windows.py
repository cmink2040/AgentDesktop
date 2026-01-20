from typing import List, Any
from PIL import Image, ImageGrab
from .base import InterfaceCollector, UIElement

try:
    from pywinauto import Desktop, Application
    import pywinauto
    import time
except ImportError:
    pywinauto = None

class WindowsCollector(InterfaceCollector):
    def __init__(self, target_exe: str = None, target_window_title: str = None):
        if not pywinauto:
            raise ImportError("pywinauto not installed. Run `pip install pywinauto`.")
        
        self.desktop = Desktop(backend="uia")
        self.app = None
        self.target_window = None

        if target_exe:
            # Launch the application
            print(f"Launching {target_exe}...")
            self.app = Application(backend="uia").start(target_exe)
            # Wait for the app to initialize - simple wait logic, can be improved
            self.app.wait_cpu_usage_lower(threshold=5) 
            # Try to grab the top window
            self.target_window = self.app.top_window()
        
        elif target_window_title:
            # Attach to existing window
            print(f"Connecting to window with title containing '{target_window_title}'...")
            
            # 1. Try finding by title case-insensitive scan
            # This is more robust than pywinauto's regex which can be case-sensitive
            found_window = None
            try:
                for w in self.desktop.windows():
                    if w.is_visible() and target_window_title.lower() in w.window_text().lower():
                        found_window = w
                        break
            except Exception as e:
                print(f"Error scanning windows: {e}")

            if found_window:
                self.target_window = found_window
                print(f"Connected to: '{self.target_window.window_text()}'")
                try:
                    self.target_window.set_focus()
                except Exception as e:
                    print(f"Warning: Could not set focus (might be minimized or background): {e}")

            else:
                 print(f"Warning: Window '{target_window_title}' not found.")
                 print("Listing available visible windows (Note: Some system apps like Task Manager require Admin privileges to see):")
                 try:
                     for w in self.desktop.windows():
                         if w.is_visible():
                             print(f" - '{w.window_text()}'")
                 except Exception as list_err:
                     print(f"Error listing windows: {list_err}")
                 self.target_window = None

    def capture_screenshot(self) -> Image.Image:
        if self.target_window:
            # Ensure window is in foreground before value capture
            try:
                if self.target_window.is_minimized():
                    self.target_window.restore()
                self.target_window.set_focus()
                # Give the OS time to repaint and animate
                time.sleep(0.5) 
            except Exception as e:
                print(f"Warning: Could not set focus before screenshot: {e}")
                
            return self.target_window.capture_as_image()
        
        # Default: Full usage
        return ImageGrab.grab()

    def get_accessibility_tree(self) -> Any:
        if self.target_window:
            return [self.target_window]
            
        # Return the wrapper for the active window or root
        # Returning root elements list
        return self.desktop.windows()

    def _walk_uia_tree(self, wrapper, offset_x=0, offset_y=0) -> List[UIElement]:
        elements = []
        
        # Check current wrapper
        try:
            if wrapper.is_visible():
                rect = wrapper.rectangle() # (left, top, right, bottom) - Screen Coords
                
                # Simple heuristic for interactivity based on control type
                ctype = wrapper.element_info.control_type
                
                # Expanded heuristic to catch Qt widgets (Maya), TreeItems (Outliner), etc.
                interactive_types = [
                    "Button", "Edit", "ListItem", "MenuItem", "CheckBox", "ComboBox",
                    "RadioButton", "TabItem", "TreeItem", "Hyperlink", "SplitButton", 
                    "Document", "Custom", "DataGrid", "DataItem"
                ]
                
                is_interactive = ctype in interactive_types
                
                if is_interactive:
                    # Adjust to relative coordinates if offset provided
                    bbox = (
                        rect.left - offset_x, 
                        rect.top - offset_y, 
                        rect.right - offset_x, 
                        rect.bottom - offset_y
                    )
                    
                    elements.append(UIElement(
                        id=str(wrapper.handle if hasattr(wrapper, 'handle') else id(wrapper)),
                        bbox=bbox,
                        role=ctype,
                        text=wrapper.window_text(),
                        is_interactive=True
                    ))
                
                # Recurse
                children = wrapper.children()
                for child in children:
                    elements.extend(self._walk_uia_tree(child, offset_x, offset_y))
        except Exception as e:
            # UIA elements can be flaky (disappear during walk)
            pass
                
        return elements

    def segment_interactables(self, raw_tree: Any) -> List[UIElement]:
        # raw_tree is a list of top-level windows
        all_elements = []
        
        # Calculate offset if target_window exists (we are capturing only that window)
        offset_x, offset_y = 0, 0
        if self.target_window:
             win_rect = self.target_window.rectangle()
             offset_x, offset_y = win_rect.left, win_rect.top

        for win in raw_tree:
            # Maybe only focus on the active one?
            # For collecting data, we might want everything visible.
            if win.is_visible():
                all_elements.extend(self._walk_uia_tree(win, offset_x, offset_y))
        return all_elements
