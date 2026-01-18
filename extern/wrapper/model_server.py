from flask import Flask, request, jsonify
import sys
import os
import io
import base64
from PIL import Image
import threading

# Add custom models path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from custom_models.gemini_google import GeminiGoogleModel
except ImportError:
    print("Could not import GeminiGoogleModel")

app = Flask(__name__)

# Global model instance (lazy loaded or initialized here)
# For this example, we assume we might want to switch models dynamically or hold one
model_instance = None

def get_model(model_name="gemini-2.0-flash-exp"):
    global model_instance
    if model_instance is None:
        model_instance = GeminiGoogleModel(model_name=model_name)
        model_instance.load_model()
    return model_instance

@app.route('/ground_element', methods=['POST'])
def ground_element():
    """
    Endpoint for Element Grounding.
    Expects JSON:
    {
        "instruction": "Click on the...",
        "image": "base64_string...",
        "model_name": "optional_model_name"
    }
    """
    data = request.json
    instruction = data.get('instruction')
    image_b64 = data.get('image')
    model_name = data.get('model_name', "gemini-2.0-flash-exp")

    if not instruction or not image_b64:
        return jsonify({"error": "Missing instruction or image"}), 400

    try:
        # Decode image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        
        model = get_model(model_name)
        # Update model name if switched? For now simpler to re-use or just update name
        if model.model_name != model_name:
             model.model_name = model_name
             # Assuming load_model doesn't need to be re-run for simple prompt change 
             # but strictly speaking for GeminiGoogleModel it just sets self.client
        
        result = model.ground_only_positive(instruction, image)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_layout', methods=['POST'])
def generate_layout():
    """
    Endpoint for Layout Grounding.
    Expects JSON:
    {
        "name": "Component Name",
        "explanation": "Component Description",
        "image": "base64_string...",
        "model_name": "optional_model_name"
    }
    """
    data = request.json
    name = data.get('name')
    explanation = data.get('explanation')
    image_b64 = data.get('image')
    model_name = data.get('model_name', "gemini-2.0-flash-exp")

    if not name or not explanation or not image_b64:
        return jsonify({"error": "Missing name, explanation or image"}), 400

    try:
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        
        model = get_model(model_name)
        
        result = model.layout_gen(name, explanation, image)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_server(port=5000):
    app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    run_server()
