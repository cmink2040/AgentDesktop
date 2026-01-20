from typing import List, Any
import io
from PIL import Image
from .base import InterfaceCollector, UIElement

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.remote.webelement import WebElement
except ImportError:
    webdriver = None

class WebCollector(InterfaceCollector):
    def __init__(self, driver_url: str = None):
        if not webdriver:
            raise ImportError("Selenium not installed. Run `pip install selenium`.")
        
        # In a real scenario, this would attach to an existing session or start one
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=options)
        if driver_url:
            self.driver.get(driver_url)

    def capture_screenshot(self) -> Image.Image:
        png_data = self.driver.get_screenshot_as_png()
        return Image.open(io.BytesIO(png_data))

    def get_accessibility_tree(self) -> List[WebElement]:
        # Web doesn't return a single tree object easily in Selenium, 
        # but we can query for all potentially interactive elements.
        # A true accessibility tree fetch requires CDP (Chrome DevTools Protocol).
        # For this 'segment interactables' scope, we'll query generic interactive tags.
        selectors = ["button", "a", "input", "textarea", "select", "[role='button']", "[onclick]"]
        css_query = ", ".join(selectors)
        return self.driver.find_elements(By.CSS_SELECTOR, css_query)

    def segment_interactables(self, raw_elements: List[WebElement]) -> List[UIElement]:
        ui_elements = []
        
        # Calculate Pixel Ratio correction
        # Selenium coordinates are in 'CSS pixels'
        # Screenshots are in 'Physical pixels'
        # On Retina Mac, ratio is 2.0. On standard, 1.0.
        window_width = self.driver.execute_script("return window.innerWidth;")
        # We need the screenshot width from the last capture, but we don't have it explicitly stored unless we capture again.
        # Efficient hack: Get one screenshot size now to determine ratio.
        png_data = self.driver.get_screenshot_as_png()
        pil_img = Image.open(io.BytesIO(png_data))
        img_width = pil_img.width
        
        scale_factor = img_width / window_width
        
        for el in raw_elements:
            if not el.is_displayed():
                continue
                
            rect = el.rect  # {'x':, 'y':, 'width':, 'height':}
            if rect['width'] == 0 or rect['height'] == 0:
                continue

            bbox = (
                int(rect['x'] * scale_factor),
                int(rect['y'] * scale_factor),
                int((rect['x'] + rect['width']) * scale_factor),
                int((rect['y'] + rect['height']) * scale_factor)
            )
            
            # Extract basic accessibility info
            role = el.get_attribute("role") or el.tag_name
            text = el.text or el.get_attribute("value") or el.get_attribute("aria-label") or ""
            
            # Unique ID (XPath or Selenium int ID)
            el_id = el.id
            
            ui_elements.append(UIElement(
                id=el_id,
                bbox=bbox,
                role=role,
                text=text[:100], # Trucate
                is_interactive=True,
                metadata={
                    "tag": el.tag_name,
                    "href": el.get_attribute("href")
                }
            ))
            
        return ui_elements
