import os
import io 
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Class labels for maize diseases
class_names = ['fall armyworm', 'grasshoper', 'healthy', 
               'leaf beetle', 'leaf blight', 'leaf spot', 
               'streak virus']

# Device setup - Windows can utilize GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model path and architecture - Windows-compatible path handling
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maize_model.pth")
MODEL_NAME = "EfficientNet-V2-S"  # Default model

# Image transformation
transform = transforms.Compose([
    transforms.Resize(int(224 * 1.15)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model
def load_model():
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        # Get model info
        model_name = checkpoint.get('model_name', 'EfficientNet-V2-S')
        num_classes = len(class_names)
        
        print(f"Loading model: {model_name} with {num_classes} classes")
        
        # Create model architecture
        if 'EfficientNet-B0' in model_name:
            model = models.efficientnet_b0(weights=None)
            dropout_rate = 0.5
        elif 'EfficientNet-B2' in model_name:
            model = models.efficientnet_b2(weights=None)
            dropout_rate = 0.4
        elif 'EfficientNet-V2-S' in model_name:
            model = models.efficientnet_v2_s(weights=None)
            dropout_rate = 0.3
        else:
            model = models.efficientnet_b0(weights=None)
            dropout_rate = 0.5
        
        # Modify classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print(f"Model ({model_name}) loaded successfully on {device}!")
        return model, model_name
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
            "model_used": model_name,
            "all_probabilities": {
                class_names[i]: round(probabilities[i].item(), 4) 
                for i in range(len(class_names))
            }
        }

# Load the model at app startup
model, model_name = load_model()

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'device': str(device),
        'model': model_name,
        'classes': class_names
    })

# Model info endpoint
@app.route('/info', methods=['GET'])
def model_info():
    return jsonify({
        'model_name': model_name,
        'device': str(device),
        'num_classes': len(class_names),
        'class_names': class_names,
        'model_path': MODEL_PATH,
        'crop_type': 'maize'
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
    print(f"Model: {model_name}")
    print(f"Classes: {class_names}")
    
    # Windows-friendly host and port configuration
    app.run(host='127.0.0.1', port=4040, debug=False, threaded=True)