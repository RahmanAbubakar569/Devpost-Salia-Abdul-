import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np

def load_model(model_path, device='cpu'):
    """Load the trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model info
    model_name = checkpoint.get('model_name', 'EfficientNet-V2-S')
    num_classes = checkpoint.get('num_classes', 7)  # Updated for 7 maize classes
    input_size = checkpoint.get('input_size', 224)
    classes = checkpoint.get('classes', ['fall armyworm', 'grasshoper', 'healthy', 
                                       'leaf beetle', 'leaf blight', 'leaf spot', 
                                       'streak virus'])
    
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
    
    # Modify classifier for 7 classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, classes, input_size

def preprocess_image(image_path, input_size):
    """Preprocess image for prediction"""
    transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, classes, device='cpu'):
    """Make prediction"""
    model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    predicted_class = classes[predicted.item()]
    
    return predicted_class

# usage
if __name__ == "__main__":
    
    import os
    MODEL_PATH = os.path.join("Maize", "maize_model.pth")  # Updated path
    IMAGE_PATH = os.path.join("Maize", "fall armyworm257_.jpg")   # Updated path
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Load model
        model, classes, input_size = load_model(MODEL_PATH, DEVICE)
        print(f"Model loaded. Classes: {classes}")
        
        # Preprocess image
        if not os.path.exists(IMAGE_PATH):
            raise FileNotFoundError(f"Image file not found at {IMAGE_PATH}")
            
        image_tensor = preprocess_image(IMAGE_PATH, input_size)
        
        # Make prediction
        predicted_class = predict(model, image_tensor, classes, DEVICE)
        
        # Print result
        print(f"Predicted class: {predicted_class}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")