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

# Device setup - Windows can utilize GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model path and architecture - Windows-compatible path handling
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_efficientnet.pth")
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
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        
        print(f"Loading model: {arch} with {len(class_names)} classes")
        
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
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        print(f"Model ({arch}) loaded successfully on {device}!")
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
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]
        confidence = probabilities[preds.item()].item()
        
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "model_used": MODEL_ARCH,
            "all_probabilities": {
                class_names[i]: round(probabilities[i].item(), 4) 
                for i in range(len(class_names))
            }
        }

# Load the model at app startup
model = load_model(arch=MODEL_ARCH)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'device': str(device),
        'model': MODEL_ARCH,
        'classes': class_names
    })

# Model info endpoint
@app.route('/info', methods=['GET'])
def model_info():
    return jsonify({
        'model_name': MODEL_ARCH,
        'device': str(device),
        'num_classes': len(class_names),
        'class_names': class_names,
        'model_path': MODEL_PATH,
        'classification_type': 'disease_classification'
    })

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    image_bytes = image_file.read()

    try:
        image_tensor = preprocess_image(image_bytes)
        result = predict_image(image_tensor)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Entry point
if __name__ == '__main__':
    print(f"Starting Flask app on Windows...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Device: {device}")
    print(f"Model: {MODEL_ARCH}")
    print(f"Classes: {class_names}")
    
    # Windows-friendly host and port configuration
    app.run(host='127.0.0.1', port=8080, debug=False, threaded=True)