#!/bin/bash
set -e

# Config
OUTPUT_DIR="./uivision_data"
WRAPPER_DIR="./extern/wrapper"
PYTHON_CMD="python3"

echo "=============================================="
echo "      UI-Vision Benchmark Setup Script        "
echo "=============================================="

# 1. Install Dependencies
echo "[1/3] Installing Dependencies..."
pip install huggingface_hub datasets flask google-genai pillow

# 2. Download and Setup Dataset
echo "[2/3] Setting up Dataset from Hugging Face..."
if [ -d "$OUTPUT_DIR" ]; then
    echo "Dataset directory $OUTPUT_DIR already exists. Skipping download."
else
    # Assuming setup_dataset.py is in the wrapper dir
    $PYTHON_CMD $WRAPPER_DIR/setup_dataset.py --output_dir "$OUTPUT_DIR"
fi

# 3. Start Model Server (Optional / Demonstration)
echo "[3/3] Setup Complete!"
echo ""
echo "You can now run the evaluation using:"
echo "  export GOOGLE_API_KEY=your_key"
echo "  $PYTHON_CMD $WRAPPER_DIR/run_eval.py --model_type gemini_google --uivision_imgs $OUTPUT_DIR/images --uivision_test_file $OUTPUT_DIR/element_grounding_test.json --task element --log_path results.json"
echo ""
echo "Or start the model server endpoint:"
echo "  $PYTHON_CMD $WRAPPER_DIR/model_server.py"
echo "=============================================="
