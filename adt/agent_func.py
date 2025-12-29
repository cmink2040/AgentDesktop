from google import genai
from PIL import Image
from adt.utility import get_api_key, draw_grid, parse_moves
from model.inference import CursorPredictor

class Agent:
    def __init__(self, arrsize=100):
        self.arrsize = arrsize
        self.predictor = None

    def _get_predictor(self):
        if self.predictor is None:
            self.predictor = CursorPredictor()
        return self.predictor

    def _actions_to_dxdy(self, actions):
        dx, dy = 0, 0
        for action in actions:
            parts = action.split()
            if parts[0] == "move":
                direction = parts[1]
                try:
                    amount = int(parts[2])
                except (IndexError, ValueError):
                    amount = 1
                step = 10 * amount
                if direction == "left": dx -= step
                elif direction == "right": dx += step
                elif direction == "up": dy -= step
                elif direction == "down": dy += step
        return dx, dy

    def _dxdy_to_actions(self, dx, dy):
        actions = []
        # X movement
        steps_x = int(dx / 10)
        if steps_x > 0:
            actions.append(f"move right {steps_x}")
        elif steps_x < 0:
            actions.append(f"move left {-steps_x}")
            
        # Y movement
        steps_y = int(dy / 10)
        if steps_y > 0:
            actions.append(f"move down {steps_y}")
        elif steps_y < 0:
            actions.append(f"move up {-steps_y}")
            
        actions.append("click")
        return actions

    def ask(self, cmd: str, img_path: str, mode: str = "Gemini") -> tuple[list[str], list[dict]]:
        """
        Returns:
            actions: list of action strings
            points: list of dicts {'label': str, 'dx': float, 'dy': float, 'color': str}
        """
        points = []
        gemini_actions = []
        imageshot_actions = []

        # Run Gemini if needed
        if mode in ["Gemini", "Hybrid"]:
            client = genai.Client(api_key=get_api_key())
            draw_grid(img_path, "img/grid.png", self.arrsize)
            image = Image.open("img/grid.png")

            prompt = f"""You are a model specializing in GUI work. Attached is an image and an instruction. The image has a grid of redlines of it, each symbolizing {self.arrsize} pixels. The cursor is that of a black square. Your two actions are as follows:
1. Click the screen.
2. Move the cursor by 10px (1/{self.arrsize/10} red grid units) up/down/left/right.

Here is your goal: {cmd}
Output a specific list of actions of [click] or [move left/down/up/right amount], or state NA if not possible. An example output may be Response: [move right 1, move left 20, click]. Only include the list of actions and nothing else. Now go."""

            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[image, prompt]
            )
            gemini_actions = parse_moves(response.text)
            gdx, gdy = self._actions_to_dxdy(gemini_actions)
            points.append({"label": "Gemini", "dx": gdx, "dy": gdy, "color": "blue"})

        # Run ImageShot if needed
        if mode in ["ImageShot", "Hybrid"]:
            pred = self._get_predictor()
            idx, idy = pred.predict(img_path)
            imageshot_actions = self._dxdy_to_actions(idx, idy)
            points.append({"label": "ImageShot", "dx": idx, "dy": idy, "color": "green"})

        # Decide which actions to return
        if mode == "ImageShot":
            return imageshot_actions, points
        elif mode == "Hybrid":
            # For Hybrid, we return Gemini actions but show both points
            return gemini_actions, points
        else: # Gemini
            return gemini_actions, points



    def consult(self, cmd: str, img_path: str) -> str:
        client = genai.Client(api_key=get_api_key())
        draw_grid(img_path, "img/grid.png", self.arrsize)

        image = Image.open("img/grid.png")

        prompt = f"You are a model specializing in GUI work. Attached is an image and an instruction. The image has a grid of red lines of it, each symbolizing {self.arrsize}  pixels. The cursor is that of a black square. Here is the instruction: {cmd}. How much red squares do you think you need to move the cursor to complete the instruction? Now let's say you can only move 10px. How many of those 10px moves do you need?"

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[image, prompt]
        )
        return response.text
