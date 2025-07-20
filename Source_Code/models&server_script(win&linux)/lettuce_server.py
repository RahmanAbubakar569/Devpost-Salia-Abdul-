import os
import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Class labels
class_names = ['Bacterial', 'Fungal', 'Healthy']

# Device setup
device = torch.device('cpu')  # For Raspberry Pi

# Model path and architecture
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_efficientnet.pth")
# MODEL_PATH = "best_efficientnet.pth"  
MODEL_ARCH = "efficientnet_b0"  # Change to 'resnet50' if needed

# Image transformation (mobile camera-friendly)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
def load_model(arch='efficientnet_b0'):
    if arch == 'resnet50':
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
    elif arch == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model ({arch}) loaded and ready!")
    return model

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
model = load_model(arch=MODEL_ARCH)

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
    app.run(host='0.0.0.0', port=8080, debug=False)