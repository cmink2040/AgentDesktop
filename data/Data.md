# Data Generation Pipeline

This module provides a unified interface to collect, segment, and label UI elements from various platforms (Web, MacOS, Windows, Android, iOS).

## 📦 Setup (uv)

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

1.  **Initialize Environment:**
    ```bash
    cd data
    uv sync
    ```
    This creates a `.venv` with all necessary dependencies including PyTorch, Transformers (for VLM), and Selenium/Appium clients.

2.  **Platform Prerequisites:**
    *   **Web:** Chrome installed. `selenium` uses `webdriver_manager` or system driver usually, but `chromedriver` might be needed in PATH.
    *   **MacOS:** Must be running on macOS. Requires Accessibility permissions for your Terminal/IDE.
    *   **Android/iOS:** Requires a running Appium Server (`npm install -g appium && appium`) and a connected device/emulator.

## 🚀 Usage

The main entry point is `collect.py`.

### Web Collection (Target a Website)
To collect DOM segments from a website and visualize the result with red bounding boxes:

```bash
uv run python collect.py --platform web --url "https://www.google.com" --label --visualize
```

*   `--platform web`: Selects the Web interface collector.
*   `--url ...`: The target URL.
*   `--label`: Runs the cropped elements through Qwen2.5-VL to generate semantic descriptions (e.g. "Search Button").
*   `--visualize`: Automatically runs the visualization step to draw red boxes around detected elements.

**Output:**
*   `collected_data/sample.png`: Raw screenshot.
*   `collected_data/sample.json`: List of UIElements with bounding boxes and labels.
*   `collected_data/sample_annotated.png`: Debug image with red boxes.

### Desktop & Mobile Analysis

**MacOS:**
```bash
uv run python collect.py --platform macos --label --visualize
```
(Captures the currently focused window using Accessibility APIs).

**Android:**
```bash
uv run python collect.py --platform android --label --visualize
```
(Requires running Appium server).

## 🛠 Visualization (Standalone)

If you already have a `sample.png` and `sample.json`, you can re-run the red-box visualization manually:

```bash
uv run python visualize.py --image collected_data/sample.png --json collected_data/sample.json --output debug_view.png
```

## 🧠 Semantic Labeling

The semantic labeling uses `Qwen/Qwen2.5-VL-3B-Instruct`. 
*   It crops the UI element from the screenshot (with 20% context padding).
*   It asks the VLM: *"Describe the function of the centered UI element concisely."*
*   You need a GPU for reasonable performance, though it works on CPU (slowly).