# GCA: Generalized Component Analysis

GCA is an End-to-End UI Understanding Model designed to "read" user interfaces by grouping pixels into functional units ("Components") rather than just drawing bounding boxes. It focuses on object permanence and semantic understanding without imposing artificial hierarchies.

## 🧠 Architecture

The model consists of four learned stages (M1-M4) and one deterministic system layer:

1.  **M1 (Backbone):** Extracts multi-scale visual features from the raw image.
2.  **M2 (MicroTokenizer):** Converts features into atomic "micro-tokens" representing small visual patches.
3.  **M3 (Merger):** The core innovation. Softly assigns micro-tokens to $K$ Component centers, fusion visual text and graphical elements into single entities.
4.  **M4 (Semantic Head):** Classifies these fused components (e.g., "Delete Button", "Search Bar") and predicts attributes.
5.  **System Layer:** Handles "Object Permanence". It assigns unique tracking IDs to components across frames and calculates state deltas (added/removed elements).

## 🚀 Getting Started

This project uses [`uv`](https://github.com/astral-sh/uv) for fast Python package and environment management.

### Prerequisites

- Python 3.10+
- `uv` installed (`pip install uv`)

### Installation

Initialize the environment and install dependencies:

```bash
uv sync
```

This will create a `.venv` directory and install PyTorch, TorchVision, and other requirements specified in `pyproject.toml`.

## 📚 Dataset

We use the [ServiceNow/ui-vision](https://huggingface.co/datasets/ServiceNow/ui-vision) dataset for training. 

The training script `train_gca.py` is configured to **automatically download** this dataset from HuggingFace on the first run. It creates a local folder `ui_vision_data/` containing the images and annotations.

## 🏋️‍♀️ Training

To start training the model, simply run:

```bash
uv run python train_gca.py
```

### Configuration
You can customize the training via command-line arguments:

- `--batch_size`: Number of images per batch (default: `2`). Reduced for local debugging; increase for GPU servers.
- `--epochs`: Total training epochs (default: `10`).
- `--lr`: Learning rate (default: `3e-4`).
- `--data_dir`: Location to store/load dataset (default: `./ui_vision_data`).
- `--save_dir`: Where to save model checkpoints (default: `checkpoints/`).

Example:
```bash
uv run python train_gca.py --batch_size 8 --epochs 50 --lr 1e-4
```

## 📂 File Structure

- **`models.py`**: The top-level `GCAEnd2EndModel` definition.
- **`layers.py`**: Implementation of M1, M2, M3, and M4 modules.
- **`system.py`**: Non-differentiable logic for ID tracking (Hungarian matching) and delta generation.
- **`training.py`**: The `Trainer` class, optimization loop, and complex loss functions (`compute_losses`).
- **`losses.py`**: Definitions for Budget, Allocation, Cut, Compactness, and Purity losses.
- **`dataset_uivision.py`**: Adapter that rasterizes UI-Vision JSON bounding boxes into the dense integer masks GCA requires.
- **`train_gca.py`**: The main entry point script.
