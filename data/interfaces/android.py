from typing import List, Any
import io
import base64
from PIL import Image
from .base import InterfaceCollector, UIElement

try:
    from appium import webdriver
    from appium.webdriver.common.appiumby import AppiumBy
except ImportError:
    webdriver = None

class AndroidCollector(InterfaceCollector):
    def __init__(self, appium_server_url='http://localhost:4723', capabilities=None):
        if not webdriver:
            raise ImportError("Appium-Python-Client not installed.")
        
        default_caps = dict(
            platformName='Android',
            automationName='UiAutomator2',
            deviceName='Android Emulator',
            # appPackage='...',
            # appActivity='...'
        )
        if capabilities:
            default_caps.update(capabilities)
            
        self.driver = webdriver.Remote(appium_server_url, options=default_caps)

    def capture_screenshot(self) -> Image.Image:
        png_data = self.driver.get_screenshot_as_png()
        return Image.open(io.BytesIO(png_data))

    def get_accessibility_tree(self) -> Any:
        # We can dump using page_source (XML) or find_elements
        # Finding elements is better for objects.
        # Querying all typically visible elements
        return self.driver.find_elements(AppiumBy.XPATH, "//*[@clickable='true' or @scrollable='true' or @checkablde='true']")

    def segment_interactables(self, raw_elements: List[Any]) -> List[UIElement]:
        elements = []
        for el in raw_elements:
            if not el.is_displayed():
                continue
            
            # Appium returns bounds as string like "[12,34][100,200]" or has .rect
            # .rect usually works in python client
            rect = el.rect
            bbox = (
                int(rect['x']),
                int(rect['y']),
                int(rect['x'] + rect['width']),
                int(rect['y'] + rect['height'])
            )
            
            text = el.text or el.get_attribute("content-desc") or ""
            cls_name = el.get_attribute("className")
            
            elements.append(UIElement(
                id=el.id,
                bbox=bbox,
                role=cls_name.split('.')[-1], # e.g. widget.Button
                text=text,
                is_interactive=True
            ))
        return elements
