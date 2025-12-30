import os
from google import genai
from PIL import Image, ImageDraw
import re
from dotenv import load_dotenv

# load .env (if present) into environment
load_dotenv()


def _get_api_key() -> str:
    """Return API key from environment or raise a clear error.

    Checks common variable names so the project supports both direct
    environment variables and a .env file loaded by python-dotenv.
    """
    
    key = os.environ["API_KEY"]
    return key

class Agent:
    def __init__(self, arrsize=100):
        self.arrsize = arrsize

    def draw_grid(self, input_path: str,
              output_path: str,
              line_color: str = "red",
              line_width: int = 1):
        img = Image.open(input_path)
        w, h = img.size

        draw = ImageDraw.Draw(img)
        for x in range(0, w, self.arrsize):
            draw.line([(x, 0), (x, h)], fill=line_color, width=line_width)
        for y in range(0, h, self.arrsize):
            draw.line([(0, y), (w, y)], fill=line_color, width=line_width)

        img.save(output_path)

    def parse_moves(self, response_text: str) -> list[str]:
        # find the bracketed part
        m = re.search(r'\[(.*)\]', response_text, re.S)
        if not m:
            return []
        inside = m.group(1)
        # split on commas or newlines, strip whitespace, drop empty strings
        parts = re.split(r'[,\n]+', inside)
        return [p.strip() for p in parts if p.strip()]


    def ask(self, cmd: str, img_path: str) -> str:
        client = genai.Client(api_key=_get_api_key())
        self.draw_grid(img_path, "img/grid.png")

        image = Image.open("img/grid.png")

        prompt = f"""You are a model specializing in GUI work. Attached is an image and an instruction. The image has a grid of redlines of it, each symbolizing {self.arrsize} pixels. The cursor is that of a black square. Your two actions are as follows:
1. Click the screen.
2. Move the cursor by 10px (1/{self.arrsize/10} red grid units) up/down/left/right.

Here is your goal: {cmd}
Output a specific list of actions of [click] or [move left/down/up/right amount], or state NA if not possible. An example output may be Response: [move right 1, move left 20, click]. Only include the list of actions and nothing else. Now go."""

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[image, prompt]
        )
        return self.parse_moves(response.text)


    def consult(self, cmd: str, img_path: str) -> str:
        client = genai.Client(api_key=_get_api_key())
        self.draw_grid(img_path, "img/grid.png")

        image = Image.open("img/grid.png")

        prompt = f"You are a model specializing in GUI work. Attached is an image and an instruction. The image has a grid of red lines of it, each symbolizing {self.arrsize}  pixels. The cursor is that of a black square. Here is the instruction: {cmd}. How much red squares do you think you need to move the cursor to complete the instruction? Now let's say you can only move 10px. How many of those 10px moves do you need?"

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[image, prompt]
        )
        return response.text

