# split_data.py
""" Script to split image dataset into training, validation, and test sets."""
""" This script is designed to run in Google Colab with Google Drive integration."""
import os
import shutil
import random
from pathlib import Path
from PIL import Image
import warnings
from google.colab import drive

# GOOGLE DRIVE MOUNTING
def mount_google_drive():
    """Mount Google Drive if not already mounted"""
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
        print("Google Drive mounted successfully!")
    else:
        print("Google Drive is already mounted")

# IMAGE VALIDATION
def is_valid_image(image_path):
    """Check if image is valid and not corrupted"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (IOError, OSError, Image.DecompressionBombWarning) as e:
        warnings.warn(f"Invalid image {image_path}: {str(e)}")
        return False

# DATA SPLITTING FUNCTIONS
def split_dataset(source_dir, output_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets

    Args:
        source_dir: Path to source dataset directory
        output_dir: Path to output directory where split data will be saved
        train_ratio: Ratio of training data (default 0.7)
        valid_ratio: Ratio of validation data (default 0.15)
        test_ratio: Ratio of test data (default 0.15)
    """

    # Validate ratios
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Ratios must sum to 1.0")

    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all class directories
    class_dirs = [d for d in os.listdir(source_dir)
                  if os.path.isdir(os.path.join(source_dir, d))]

    print(f"Found {len(class_dirs)} classes: {class_dirs}")

    total_images = 0
    total_valid_images = 0
    split_summary = {}

    for class_name in class_dirs:
        class_path = os.path.join(source_dir, class_name)

        # Get all image files in this class
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        all_images = []

        for file in os.listdir(class_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(class_path, file)
                if is_valid_image(image_path):
                    all_images.append(file)
                    total_valid_images += 1
                total_images += 1

        print(f"Class '{class_name}': {len(all_images)} valid images out of {len(os.listdir(class_path))} files")

        # Shuffle images
        random.shuffle(all_images)

        # Calculate split sizes
        num_images = len(all_images)
        num_train = int(num_images * train_ratio)
        num_valid = int(num_images * valid_ratio)
        num_test = num_images - num_train - num_valid

        # Split images
        train_images = all_images[:num_train]
        valid_images = all_images[num_train:num_train + num_valid]
        test_images = all_images[num_train + num_valid:]

        # Create class directories in each split
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Copy images to respective directories
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)

        for img in valid_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(valid_dir, class_name, img)
            shutil.copy2(src, dst)

        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_dir, class_name, img)
            shutil.copy2(src, dst)

        split_summary[class_name] = {
            'total': num_images,
            'train': len(train_images),
            'valid': len(valid_images),
            'test': len(test_images)
        }

    # Print summary
    print("\n" + "="*60)
    print("DATASET SPLIT SUMMARY")
    print("="*60)
    print(f"Total images processed: {total_images}")
    print(f"Valid images: {total_valid_images}")
    print(f"Invalid/corrupted images: {total_images - total_valid_images}")
    print("\nPer-class breakdown:")
    print(f"{'Class':<15} {'Total':<8} {'Train':<8} {'Valid':<8} {'Test':<8}")
    print("-" * 60)

    total_train = total_valid = total_test = 0
    for class_name, counts in split_summary.items():
        print(f"{class_name:<15} {counts['total']:<8} {counts['train']:<8} {counts['valid']:<8} {counts['test']:<8}")
        total_train += counts['train']
        total_valid += counts['valid']
        total_test += counts['test']

    print("-" * 60)
    print(f"{'TOTAL':<15} {total_train+total_valid+total_test:<8} {total_train:<8} {total_valid:<8} {total_test:<8}")
    print(f"\nSplit ratios achieved:")
    print(f"Train: {total_train/(total_train+total_valid+total_test)*100:.1f}%")
    print(f"Valid: {total_valid/(total_train+total_valid+total_test)*100:.1f}%")
    print(f"Test: {total_test/(total_train+total_valid+total_test)*100:.1f}%")

    return split_summary

def split_multiple_crops(base_data_path, output_base_path, crops_to_split,
                        train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    """Split multiple crop datasets"""

    results = {}

    for crop in crops_to_split:
        print(f"\n Splitting {crop} dataset...")

        source_dir = os.path.join(base_data_path, crop)
        output_dir = os.path.join(output_base_path, crop)

        if not os.path.exists(source_dir):
            print(f" Source directory not found: {source_dir}")
            continue

        try:
            summary = split_dataset(source_dir, output_dir, train_ratio, valid_ratio, test_ratio)
            results[crop] = summary
            print(f" {crop} dataset split successfully!")

        except Exception as e:
            print(f" Error splitting {crop}: {str(e)}")
            results[crop] = None

    return results

# MAIN EXECUTION
if __name__ == "__main__":
    # Mount Google Drive
    mount_google_drive()

    # Set random seed for reproducibility
    random.seed(42)

    # Configuration
    config = {
        'base_data_path': "/content/drive/MyDrive/CCMT Dataset_1",  # Original dataset path
        'output_base_path': "/content/drive/MyDrive/CCMT_Split_Data",  # Where to save split data
        'crops_to_split': ["Maize"],  # List of crops to split, can be extended ["Tomato", "Rice", etc.]
        'split_ratios': {
            'train': 0.7, # Ratio for training data 70%
            'valid': 0.15, # Ratio for validation data 15%
            'test': 0.15 # Ratio for test data 15%
        }
    }

    # Create output directory
    os.makedirs(config['output_base_path'], exist_ok=True)

    print("Starting dataset splitting process...")
    print(f"Source: {config['base_data_path']}")
    print(f"Output: {config['output_base_path']}")
    print(f"Crops to split: {config['crops_to_split']}")
    print(f"Split ratios - Train: {config['split_ratios']['train']}, Valid: {config['split_ratios']['valid']}, Test: {config['split_ratios']['test']}")

    # Split datasets
    results = split_multiple_crops(
        config['base_data_path'],
        config['output_base_path'],
        config['crops_to_split'],
        config['split_ratios']['train'],
        config['split_ratios']['valid'],
        config['split_ratios']['test']
    )

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    for crop, summary in results.items():
        if summary:
            print(f"\n {crop}:")
            print(f"   Split data saved to: {os.path.join(config['output_base_path'], crop)}")
            total_images = sum(class_data['total'] for class_data in summary.values())
            print(f"   Total images processed: {total_images}")
        else:
            print(f"\n {crop}: Failed to split")

    print(f"\n Dataset splitting completed!")
    print(f"Split datasets are ready for training at: {config['output_base_path']}")