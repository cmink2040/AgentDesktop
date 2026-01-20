from typing import List, Any
import io
from PIL import Image
from .base import InterfaceCollector, UIElement

try:
    from appium import webdriver
    from appium.webdriver.common.appiumby import AppiumBy
except ImportError:
    webdriver = None

class IOSCollector(InterfaceCollector):
    def __init__(self, appium_server_url='http://localhost:4723', capabilities=None):
        if not webdriver:
            raise ImportError("Appium-Python-Client not installed.")
        
        default_caps = dict(
            platformName='iOS',
            automationName='XCUITest',
            deviceName='iPhone Simulator',
            # bundleId='...',
        )
        if capabilities:
            default_caps.update(capabilities)
            
        self.driver = webdriver.Remote(appium_server_url, options=default_caps)

    def capture_screenshot(self) -> Image.Image:
        png_data = self.driver.get_screenshot_as_png()
        return Image.open(io.BytesIO(png_data))

    def get_accessibility_tree(self) -> Any:
        # XCUITest allows querying by class chain or predicate string for better performance
        # Finding elements that are visible and interactive (Buttons, TextFields, etc)
        # Simplified query:
        return self.driver.find_elements(AppiumBy.IOS_CLASS_CHAIN, "**/XCUIElementTypeButton") + \
               self.driver.find_elements(AppiumBy.IOS_CLASS_CHAIN, "**/XCUIElementTypeTextField") + \
               self.driver.find_elements(AppiumBy.IOS_CLASS_CHAIN, "**/XCUIElementTypeCell")

    def segment_interactables(self, raw_elements: List[Any]) -> List[UIElement]:
        elements = []
        for el in raw_elements:
            if not el.is_displayed():
                continue
            
            rect = el.rect # x,y,width,height
            bbox = (
                int(rect['x']),
                int(rect['y']),
                int(rect['x'] + rect['width']),
                int(rect['y'] + rect['height'])
            )
            
            text = el.text or el.get_attribute("label") or el.get_attribute("name") or ""
            role = el.tag_name # XCUIElementType...
            
            elements.append(UIElement(
                id=el.id,
                bbox=bbox,
                role=role.replace('XCUIElementType', ''),
                text=text,
                is_interactive=True
            ))
        return elements
