import os
import re
from PIL import Image, ImageDraw
from dotenv import load_dotenv

def get_api_key() -> str:
    """Return API key from environment or raise a clear error.

    Checks common variable names so the project supports both direct
    environment variables and a .env file loaded by python-dotenv.
    """
    # load .env (if present) into environment
    load_dotenv()
    
    # prefer explicit API_KEY, then other common names
    key = os.getenv("API_KEY") or os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "API key not found. Set environment variable API_KEY or create a .env file with API_KEY=your_key"
        )
    return key

def draw_grid(input_path: str,
              output_path: str,
              arrsize: int = 100,
              line_color: str = "red",
              line_width: int = 1):
    img = Image.open(input_path)
    w, h = img.size

    draw = ImageDraw.Draw(img)
    for x in range(0, w, arrsize):
        draw.line([(x, 0), (x, h)], fill=line_color, width=line_width)
    for y in range(0, h, arrsize):
        draw.line([(0, y), (w, y)], fill=line_color, width=line_width)

    img.save(output_path)

def parse_moves(response_text: str) -> list[str]:
    # find the bracketed part
    m = re.search(r'\[(.*)\]', response_text, re.S)
    if not m:
        return []
    inside = m.group(1)
    # split on commas or newlines, strip whitespace, drop empty strings
    parts = re.split(r'[,\n]+', inside)
    return [p.strip() for p in parts if p.strip()]
