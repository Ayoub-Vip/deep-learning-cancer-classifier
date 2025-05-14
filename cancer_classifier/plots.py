from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from cancer_classifier.config import FIGURES_DIR, PROCESSED_DATA_DIR, CLASSES

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from PIL import Image
# import cv2
import torch

def visualize_sample_images(dataset, num_samples=5):
  class_indices = dataset.class_to_idx

  fig, axes = plt.subplots(len(class_indices), num_samples, figsize=(15, 10))
  for i, class_name in enumerate(class_indices.keys()):
       class_idx = class_indices[class_name]
       class_images = [img for img, label in dataset.imgs if label == class_idx][:num_samples]

       for j, img_path in enumerate(class_images):
           img = plt.imread(img_path, format='gray')
           axes[i, j].imshow(img)
           axes[i, j].set_title(class_name)
           axes[i, j].axis('off')

  plt.tight_layout()
  plt.savefig(FIGURES_DIR / "visualize_sample_images.png")
  plt.show()

def plot_confusion_matrix(true_classes, predicted_classes, model_name=None):
    
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    file_name = model_name + "_confusion_matrix.png"
    plt.savefig(FIGURES_DIR / file_name)
    plt.show()



app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
