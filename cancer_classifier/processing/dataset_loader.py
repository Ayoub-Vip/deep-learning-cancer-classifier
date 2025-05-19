from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from cancer_classifier.processing.image_utils import crop_image, adjust_image_contrast
from cancer_classifier.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, CLASSES

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class BrainTumorDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        idx = 0 #to assign class indices starting from 0 if other files are present
        for cls in sorted(os.listdir(root_dir)):
            if os.path.isdir(os.path.join(root_dir, cls)):
                self.class_to_idx[cls] = idx
                idx += 1
        
        for cls in self.class_to_idx:
            cls_dir = os.path.join(root_dir, cls)
            for img_file in os.listdir(cls_dir):
                if img_file.lower().endswith(('jpg')):
                    self.image_paths.append(os.path.join(cls_dir, img_file))
                    self.labels.append(self.class_to_idx[cls])
                    
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
            self.labels = self.labels[:max_samples]
        
        self.transform = transform if transform is not None else transforms.ToTensor()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def get_dataloaders(self, batch_size, train_ratio=0.8, val_ratio=0.1):
        rand_gen = torch.Generator().manual_seed(142)
        test_ratio = 1.0 - train_ratio - val_ratio
        train_dataset, val_dataset, test_dataset = random_split(
            dataset = self,
            lengths = [train_ratio, val_ratio, test_ratio],
            generator = rand_gen
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader


    def get_mean_std(self, batch_size=256):
        loader = DataLoader(self, batch_size=batch_size, shuffle=False)
        mean = 0.0
        std = 0.0
        total = 0.0
        for images, _ in loader:
            batch = images.size(0)
            images = images.view(batch, -1)
            mean += images.mean(1).sum(0)
            std += images.std(1).sum(0)
            total += batch
        mean /= total
        std /= total
        return torch.tensor([mean]), torch.tensor([std])

def unnormalize(img, input_mean, input_std):
    img = img.cpu().numpy().squeeze(0)
    img = input_std * img + input_mean
    img = np.clip(img, 0, 1)
    return img

def preprocess_images(src_dir=RAW_DATA_DIR, dist_dir=PROCESSED_DATA_DIR,img_size=(256, 256), clip_limit=2.0, tile_size=(1, 1)):
    for i, cls in tqdm(enumerate(CLASSES), colour="blue", desc="Processing classes"):
        cls_dir = os.path.join(src_dir, cls)
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            img = crop_image(img_path, img_size, clip_limit, tile_size)
            # img = adjust_image_contrast(img)
            # save in processed directory
            cv2.imwrite(os.path.join(dist_dir, cls, img_name), img)
            





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

    ROOT_DIR = "../../data/raw/"
    BATCH_SIZE = 32
    MAX_SAMPLES = 1000
    
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    raw_dataset = BrainTumorDataset(root_dir=ROOT_DIR, transform=transform_pipeline, max_samples=MAX_SAMPLES)
    
    mean, std = raw_dataset.get_mean_std()
    print("Mean:", mean)
    print("Std:", std)
    
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])
    
    dataset = BrainTumorDataset(root_dir=ROOT_DIR, transform=transform_pipeline, max_samples=MAX_SAMPLES)
    
    train_loader, val_loader, test_loader = dataset.get_dataloader(BATCH_SIZE)
    
    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    train_images, _ = next(iter(train_loader))
    test_images, _ = next(iter(test_loader))
    val_images, _ = next(iter(val_loader))

    train_img = train_images[0]
    test_img = test_images[0]
    val_img = val_images[0]

    train_img_np = unnormalize(train_img, mean, std)
    test_img_np = unnormalize(test_img, mean, std)
    val_img_np = unnormalize(val_img, mean, std)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(train_img_np)
    plt.title("Train Set Sample")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(val_img_np)
    plt.title("Validation Set Sample")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(test_img_np)
    plt.title("Test Set Sample")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()
