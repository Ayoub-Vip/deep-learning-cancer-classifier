from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from cancer_classifier.config import MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASSES

import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def adjust_image_contrast(image, target_size=(256, 256)):


    # CLAHE: ajust contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(1, 1))
    image_clahe = clahe.apply(image)

    # image_clahe = cv2.resize(image_clahe, target_size)

    # tensor = transforms.ToTensor()
    # image_tensor = tensor(image_clahe)

    return image_clahe

def crop_image(image_path, size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # assert image is not None, f"Failed to load image: {image_path}"
    
    image = adjust_image_contrast(image)
    # Compute non-zero pixel indices
    non_zero_coords = np.argwhere(image > 0)  # Pixels greater than zero (non-black)
    
    # Find boundaries
    top_left = non_zero_coords.min(axis=0)  # Minimum row, col
    bottom_right = non_zero_coords.max(axis=0)  # Maximum row, col
    
    # Crop the image
    cropped_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    
    return cropped_image

def resize_image_tensor(image_tensor, size=(256, 256)):
    """Resizes an image to the given size."""
    return transforms.functional.resize(image_tensor, size)

def normalize_image_tensor(image_tensor):
    """Normalizes pixel values to range [0,1]."""
    return image_tensor / 255.0

def augment_image_tensor(image_tensor):
    # Apply random transformations to augment the image
    flipped_H_image = transforms.functional.hflip(image_tensor)  # Flip horizontally
    flipped_V_image = transforms.functional.vflip(image_tensor)  # Flip vertically
    rotated_image_90 = transforms.functional.rotate(image_tensor, 90)  # Rotate 90 degrees
    rotated_image_270 = transforms.functional.rotate(image_tensor, 270)

    return [image_tensor, flipped_H_image, flipped_V_image, rotated_image_90, rotated_image_270]

def process_dataset(test_size=0.1, val_size=0.1, random_state=42):
    X, y = [], []
    
    # Load all images and labels
    for label, cls in enumerate(CLASSES):
        cls_dir = os.path.join(RAW_DATA_DIR, cls)
        for img_name in tqdm(os.listdir(cls_dir), desc=f"Processing {cls}"):
            img_path = os.path.join(cls_dir, img_name)
            img_processed = preprocess_image(img_path)
            X.append(img_processed)
            y.append(label)
    
    X = torch.stack(X) 
    y = torch.tensor(y)
    
    # Split into train (80%), val (10%), test (10%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y.numpy(), random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size/(1-test_size), stratify=y_train.numpy(), random_state=random_state
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_processed_images(X, y, subset="train"):
    """Save preprocessed tensor images to disk"""
    for i, (img_tensor, label) in tqdm(enumerate(zip(X, y)), desc=f"Saving {subset}"):
        cls = CLASSES[label]
        save_path = os.path.join(PROCESSED_DATA_DIR, subset, cls, f"img_{i}.jpg")
        
        img_np = img_tensor.permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C] to be compatible with open cv
        img_np = (img_np * 255).astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

# def save_processed_image()







#########################################################################
app = typer.Typer()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()