import tkinter as tk
import random
import math
import mss
import mss.tools

class VDesktop(tk.Tk):
    def __init__(self, agent):
        super().__init__()
        self.title("vdesktop")
        self.geometry("600x400")
        self.resizable(False, False)

        # keep reference to agent
        self.agent = agent

        # Canvas area
        self.canvas_width = 600
        self.canvas_height = 350
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        # Randomly place 5 non-overlapping buttons and remember their window ids and colors
        self.button_data = []  # list of tuples: (window_id, color)
        positions = []
        min_dist = 80  # minimum separation between button centers
        self.button_colors = ["green", "red", "orange", "blue", "purple"]
        for color in self.button_colors:
            while True:
                x = random.randint(min_dist, self.canvas_width - min_dist)
                y = random.randint(min_dist, self.canvas_height - min_dist)
                if all(math.hypot(x - x2, y - y2) > min_dist for x2, y2 in positions):
                    positions.append((x, y))
                    break
            btn = tk.Frame(self.canvas, bg=color, width=40, height=30, bd=2, relief="raised")
            win_id = self.canvas.create_window(x, y, window=btn)
            self.button_data.append((win_id, color))

        # Cursor as a small square widget, initially in the middle
        self.cursor_size = 12
        start_x = self.canvas_width // 2
        start_y = self.canvas_height // 2
        self.cursor_widget = tk.Frame(self.canvas, width=self.cursor_size, height=self.cursor_size, bg="black")
        self.cursor = self.canvas.create_window(start_x, start_y, window=self.cursor_widget)
        self.canvas.tag_raise(self.cursor)

        # Controls frame (arrow keys, click, screenshot, toggle)
        self.controls_frame = tk.Frame(self, height=40)
        self.controls_frame.pack(side="bottom", fill="x")
        self._build_controls()

        # Input frame (text entry and submit/back)
        self.input_frame = tk.Frame(self, height=40)
        self._build_input_view()

        # start with controls visible
        self.show_controls()

    def _build_controls(self):
        cf = self.controls_frame
        # arrow and action buttons
        btn_left   = tk.Button(cf, text="←", width=4, command=lambda: self.move_cursor("left"))
        btn_up     = tk.Button(cf, text="↑", width=4, command=lambda: self.move_cursor("up"))
        btn_down   = tk.Button(cf, text="↓", width=4, command=lambda: self.move_cursor("down"))
        btn_right  = tk.Button(cf, text="→", width=4, command=lambda: self.move_cursor("right"))
        btn_click  = tk.Button(cf, text="Click", width=6, command=self.check_click)
        btn_ss     = tk.Button(cf, text="Grab Window", width=10, command=self.screenshot)
        btn_toggle = tk.Button(cf, text="Agent Mode", width=10, command=self.show_input)
        for btn in (btn_left, btn_up, btn_down, btn_right, btn_click, btn_ss, btn_toggle):
            btn.pack(side="left", padx=5, pady=5)

    def _build_input_view(self):
        iv = self.input_frame
        self.entry = tk.Entry(iv)
        self.entry.pack(side="left", padx=5, pady=5)
        # bind Enter key to on_submit
        self.entry.bind("<Return>", self.on_submit)

        # Mode selection
        self.mode_var = tk.StringVar(value="Gemini")
        modes = ["Gemini", "ImageShot", "Hybrid"]
        mode_menu = tk.OptionMenu(iv, self.mode_var, *modes)
        mode_menu.pack(side="left", padx=5)

        btn_submit = tk.Button(iv, text="Submit", width=8, command=self.on_submit)
        btn_back   = tk.Button(iv, text="Manual Mode", width=8, command=self.show_controls)
        btn_submit.pack(side="left", padx=5)
        btn_back.pack(side="left", padx=5)

    def show_controls(self):
        self.input_frame.pack_forget()
        self.controls_frame.pack(side="bottom", fill="x")

    def show_input(self):
        self.controls_frame.pack_forget()
        self.input_frame.pack(side="bottom", fill="x")

    def move_cursor(self, direction, times = 1):
        for _ in range(times):
            dx = dy = 0
            step = 10
            if direction == "left":  dx = -step
            elif direction == "right": dx = step
            elif direction == "up":    dy = -step
            elif direction == "down":  dy = step
            self.canvas.move(self.cursor, dx, dy)
            # clamp within canvas
            x, y = self.canvas.coords(self.cursor)
            half = self.cursor_size // 2
            x1, y1, x2, y2 = x - half, y - half, x + half, y + half
            if x1 < 0: self.canvas.move(self.cursor, -x1, 0)
            if y1 < 0: self.canvas.move(self.cursor, 0, -y1)
            if x2 > self.canvas_width:  self.canvas.move(self.cursor, self.canvas_width - x2, 0)
            if y2 > self.canvas_height: self.canvas.move(self.cursor, 0, self.canvas_height - y2)
            self.canvas.tag_raise(self.cursor)

    def check_click(self):
        bbox_cursor = self.canvas.bbox(self.cursor)
        if not bbox_cursor:
            return
        x1_c, y1_c, x2_c, y2_c = bbox_cursor
        for win_id, color in self.button_data:
            bbox_btn = self.canvas.bbox(win_id)
            if not bbox_btn:
                continue
            x1_b, y1_b, x2_b, y2_b = bbox_btn
            if x1_c < x2_b and x2_c > x1_b and y1_c < y2_b and y2_c > y1_b:
                print(f"CLICKED {color}")
                return
        print("Click found nothing")

    def screenshot(self):
        self.update_idletasks()
        x1 = self.winfo_rootx()
        y1 = self.winfo_rooty()
        width = self.winfo_width()
        height = self.winfo_height()

        with mss.mss() as sct:
            monitor = {"top": y1, "left": x1, "width": width, "height": height}
            output = "img/tk_window.png"
            sct_img = sct.grab(monitor)
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

    def on_submit(self, event=None):
        text = self.entry.get()
        self.entry.delete(0, tk.END)

        self.screenshot()

        if text != "":
            # Clear previous debug markers
            self.canvas.delete("debug_marker")
            
            output, points = self.agent.ask(text, "img/tk_window.png", mode=self.mode_var.get())
            
            # Draw debug points
            cx, cy = self.canvas.coords(self.cursor)
            # coords returns center because it's a window object? No, create_window coords are center.
            # Wait, create_window coords are where the window is placed.
            # Let's verify. Yes, create_window(x, y, ...) places center at x,y by default.
            
            for p in points:
                dx, dy = p['dx'], p['dy']
                tx, ty = cx + dx, cy + dy
                color = p['color']
                # Draw a small circle
                r = 5
                self.canvas.create_oval(tx-r, ty-r, tx+r, ty+r, fill=color, outline="white", width=2, tags="debug_marker")
                # Draw label
                self.canvas.create_text(tx, ty-15, text=p['label'], fill=color, font=("Arial", 8), tags="debug_marker")

            debug = self.agent.consult(text, "img/tk_window.png")
            print(f"Response: {output} | DEBUG: {debug}")
            print(output)
            for action in output:
               self.execute(action)
            

    def execute(self, command: str):
        parts = command.split()
        action = parts[0]
        params = parts[1:]

        tasks = {
            "move": lambda direction, times="1": self.move_cursor(direction, int(times)),
            "click": lambda: self.check_click(),
        }

        if action in tasks:
            tasks[action](*params)
        else:
            print(f"Unknown action: {action!r}")

