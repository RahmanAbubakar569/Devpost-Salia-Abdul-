import os
import time
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import ImageFolder
from google.colab import drive
import warnings
from typing import Any, Tuple
import random
import json
import matplotlib.pyplot as plt

# GOOGLE DRIVE MOUNTING
def mount_google_drive():
    """Mount Google Drive if not already mounted"""
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
        print("Google Drive mounted successfully!")
    else:
        print("Google Drive is already mounted")

# SIMPLE IMAGE LOADING

def safe_load_image(path):
    """Safely load an image with fallback strategies"""
    try:
        with Image.open(path) as img:
            img = img.convert('RGB')
            return img.copy()
    except Exception as e1:
        try:
            # Try with PIL's robust loading
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(path) as img:
                img = img.convert('RGB')
                return img.copy()
        except Exception as e2:
            warnings.warn(f"Cannot load image {path}: {str(e1)}, {str(e2)}")
            return None

# SIMPLE DATASET CLASS
class SimpleImageFolder(Dataset):
    """Simple dataset that loads images without validation"""
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root)
        self.transform = transform
        print(f"SimpleImageFolder initialized with {len(self.dataset)} images")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]

        img = safe_load_image(path)
        if img is None:
            # Return a gray fallback image for corrupt files
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            print(f"Using fallback for corrupt image: {path}")

        if self.transform:
            img = self.transform(img)

        return img, label

# MODEL SELECTION
def get_appropriate_model(dataset_size: int, num_classes: int) -> Tuple[Any, int, str]:
    """Select model based on dataset size"""
    print(f"Selecting model for dataset size: {dataset_size}, classes: {num_classes}")

    if dataset_size < 500:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        input_size = 224
        model_name = "EfficientNet-B0"
        dropout_rate = 0.5
    elif 500 <= dataset_size < 2000:
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
        model = models.efficientnet_b2(weights=weights)
        input_size = 260
        model_name = "EfficientNet-B2"
        dropout_rate = 0.4
    else:
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_s(weights=weights)
        input_size = 384
        model_name = "EfficientNet-V2-S"
        dropout_rate = 0.3

    # Modify classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    # Freeze early layers, unfreeze later layers and classifier
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        elif dataset_size >= 500:
            if "features.6" in name or "features.7" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False

    print(f"Selected model: {model_name}")
    return model, input_size, model_name

# DATA TRANSFORMS
def get_train_transform(input_size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_test_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# MIXUP AUGMENTATION
def mixup_data(x, y, alpha=1.0):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# TRAINING FUNCTIONS
def train_model(model, train_loader, val_loader, config, device):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    print(f"Starting training for {config['num_epochs']} epochs...")

    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Apply mixup occasionally
            if config['use_mixup'] and random.random() > 0.5:
                mixed_data, y_a, y_b, lam = mixup_data(data, target, alpha=0.2)
                output = model(mixed_data)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            if not config['use_mixup'] or random.random() <= 0.5:
                _, predicted = torch.max(output.data, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()

            # Print progress less frequently
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{config["num_epochs"]}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()

        # Calculate metrics
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0
        val_accuracy = 100 * correct_val / total_val

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        epoch_time = time.time() - epoch_start_time

        print(f'\nEpoch {epoch+1}/{config["num_epochs"]} Summary:')
        print(f'  Time: {epoch_time:.1f}s')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f'   New best model saved!')
        else:
            patience_counter += 1
            print(f'   Patience: {patience_counter}/{config["early_stopping_patience"]}')

        if patience_counter >= config['early_stopping_patience']:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

        print('-' * 60)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state")

    return model, train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader, device, classes):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    print("Evaluating model on test set...")

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Per-class accuracy
            for i in range(target.size(0)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == target[i]:
                    class_correct[label] += 1

    accuracy = 100 * correct / total
    class_accuracies = {classes[k]: 100 * class_correct[k] / class_total[k]
                       for k in class_total.keys()}

    return accuracy, class_accuracies

def plot_training_history(train_losses, val_losses, val_accuracies, save_path):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot validation accuracy
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved to: {save_path}")

def train_crop_model(crop_name, config):
    """Train a model for a specific crop"""
    print(f"\n Training {crop_name} classification model...")
    start_time = time.time()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data directories
    crop_data_dir = os.path.join(config['split_data_path'], crop_name)
    train_dir = os.path.join(crop_data_dir, 'train')
    valid_dir = os.path.join(crop_data_dir, 'valid')
    test_dir = os.path.join(crop_data_dir, 'test')

    # Check if directories exist
    for dir_path, dir_name in [(train_dir, 'train'), (valid_dir, 'valid'), (test_dir, 'test')]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{dir_name} directory not found: {dir_path}")

    # Get dataset info quickly
    temp_dataset = ImageFolder(train_dir)
    num_classes = len(temp_dataset.classes)
    classes = temp_dataset.classes

    # Get dataset sizes
    train_size = len(temp_dataset)
    valid_size = len(ImageFolder(valid_dir))
    test_size = len(ImageFolder(test_dir))
    total_size = train_size + valid_size + test_size

    print(f"Dataset info:")
    print(f"  Classes: {classes}")
    print(f"  Train: {train_size}, Valid: {valid_size}, Test: {test_size}")
    print(f"  Total: {total_size} images")

    # Get model
    model, input_size, model_name = get_appropriate_model(total_size, num_classes)
    model = model.to(device)

    # Create datasets - simple and fast
    train_transform = get_train_transform(input_size)
    test_transform = get_test_transform(input_size)

    train_dataset = SimpleImageFolder(train_dir, transform=train_transform)
    val_dataset = SimpleImageFolder(valid_dir, transform=test_transform)
    test_dataset = SimpleImageFolder(test_dir, transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['training_params']['batch_size'],
                            shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training_params']['batch_size'],
                          shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training_params']['batch_size'],
                           shuffle=False, num_workers=0, pin_memory=True)

    # Train model
    model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, config['training_params'], device)

    # Evaluate model
    test_accuracy, class_accuracies = evaluate_model(model, test_loader, device, classes)

    print(f"\n Final Results:")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("Per-class accuracy:")
    for class_name, acc in class_accuracies.items():
        print(f"  {class_name}: {acc:.2f}%")

    # Save model
    model_filename = f"{crop_name.lower()}_model.pth"
    model_path = os.path.join(config['model_save_dir'], model_filename)

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'input_size': input_size,
        'num_classes': num_classes,
        'classes': classes,
        'test_accuracy': test_accuracy,
        'class_accuracies': class_accuracies,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'config': config
    }, model_path)

    # Save training history
    history_path = os.path.join(config['model_save_dir'], f"{crop_name.lower()}_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'test_accuracy': test_accuracy,
            'class_accuracies': class_accuracies
        }, f, indent=2)

    # Plot and save training history
    plot_path = os.path.join(config['model_save_dir'], f"{crop_name.lower()}_training_plot.png")
    plot_training_history(train_losses, val_losses, val_accuracies, plot_path)

    training_time = (time.time() - start_time) / 60

    return {
        'crop': crop_name,
        'model_path': model_path,
        'model_name': model_name,
        'classes': classes,
        'test_accuracy': test_accuracy,
        'class_accuracies': class_accuracies,
        'training_time_min': training_time,
        'history_path': history_path,
        'plot_path': plot_path
    }

def train_multiple_crops(config):
    """Train models for multiple crops"""
    all_results = []

    # Get list of available crops
    if not os.path.exists(config['split_data_path']):
        raise FileNotFoundError(f"Split data directory not found: {config['split_data_path']}")

    available_crops = [d for d in os.listdir(config['split_data_path'])
                      if os.path.isdir(os.path.join(config['split_data_path'], d))]

    if config['crops_to_train'] == 'all':
        crops_to_train = available_crops
    else:
        crops_to_train = config['crops_to_train']

    print(f"Available crops: {available_crops}")
    print(f"Crops to train: {crops_to_train}")

    for crop in crops_to_train:
        if crop not in available_crops:
            print(f" Warning: {crop} not found in available crops. Skipping...")
            continue

        try:
            result = train_crop_model(crop, config)
            all_results.append(result)
            print(f" Successfully trained {crop} model")
        except Exception as e:
            print(f" Error training {crop} model: {str(e)}")
            import traceback
            traceback.print_exc()

    return all_results

def print_final_summary(results):
    """Print final summary of all trained models"""
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    total_time = sum(result['training_time_min'] for result in results)

    print(f" Total models trained: {len(results)}")
    print(f" Total training time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")

    print("\n Model Performance:")
    print("-" * 80)
    print(f"{'Crop':<15} {'Model':<20} {'Accuracy':<10} {'Time (min)':<12}")
    print("-" * 80)

    for result in results:
        print(f"{result['crop']:<15} {result['model_name']:<20} {result['test_accuracy']:<10.2f} {result['training_time_min']:<12.1f}")

    print("-" * 80)
    avg_accuracy = sum(result['test_accuracy'] for result in results) / len(results)
    print(f"Average accuracy: {avg_accuracy:.2f}%")

    print("\n Saved files:")
    for result in results:
        print(f"  {result['crop']}:")
        print(f"    Model: {result['model_path']}")
        print(f"    History: {result['history_path']}")
        print(f"    Plot: {result['plot_path']}")

# MAIN EXECUTION
if __name__ == "__main__":
    # Mount Google Drive
    mount_google_drive()

    # Configuration
    config = {
        'split_data_path': "/content/drive/MyDrive/CCMT_Split_Data",
        'model_save_dir': "/content/drive/MyDrive/CCMT_Models",
        'crops_to_train': ["Maize"],
        'training_params': {
            'num_epochs': 12,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'early_stopping_patience': 7,
            'use_mixup': True
        }
    }

    # Create model directory
    os.makedirs(config['model_save_dir'], exist_ok=True)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("Starting training process...")
    print(f"Split data path: {config['split_data_path']}")
    print(f"Model save path: {config['model_save_dir']}")
    print(f"Crops to train: {config['crops_to_train']}")
    print(f"Training parameters: {config['training_params']}")

    try:
        # Train the models
        results = train_multiple_crops(config)

        if results:
            print_final_summary(results)
            print("\n ALL TRAINING COMPLETED SUCCESSFULLY!")
            print("="*80)
        else:
            print(" No models were trained successfully.")

    except Exception as e:
        print(f" Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\nTraining script finished.")