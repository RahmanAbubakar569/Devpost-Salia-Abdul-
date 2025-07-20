import os 
import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Class labels for cassava diseases
class_names = ['bacterial blight', 'brown spot', 'green mite', 'healthy', 'mosaic']

# Device setup
device = torch.device('cpu')  # For Raspberry Pi

# Model path and architecture
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_efficientnet_cassava.pth")
MODEL_ARCH = "efficientnet_b0"

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load the model
def load_model():
    try:
        model = models.efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Model loaded and ready!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Image preprocessing
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(device)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        raise

# Prediction logic
def predict_image(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]
        return {"predicted_class": predicted_class}

# Load the model at app startup
model = load_model()

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        image_tensor = preprocess_image(image_bytes)
        result = predict_image(image_tensor)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1010, debug=False)