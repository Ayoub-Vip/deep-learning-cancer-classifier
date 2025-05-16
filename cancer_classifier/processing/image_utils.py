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


def adjust_image_contrast(image, clip_limit=2.0, tile_size=(1, 1)):

    # CLAHE: ajust contrast 
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    image_clahe = clahe.apply(image)

    return image_clahe

def crop_image(image_path, size=(256, 256), clip_limit=2.0, tile_size=(1, 1)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load {image_path}")
    img = adjust_image_contrast(img, clip_limit, tile_size)

    _, mask = cv2.threshold(img, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imshow("mask", mask); cv2.waitKey()

    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        raise ValueError("No foreground found in " + image_path)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # +1 to include the edge pixel

    cropped = img[y0:y1, x0:x1]
    cropped = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)
    return cropped

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