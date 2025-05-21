from cancer_classifier.config import FIGURES_DIR, CLASSES

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def visualize_sample_images(dataset, num_samples=5):
  class_indices = dataset.class_to_idx

  fig, axes = plt.subplots(len(class_indices), num_samples, figsize=(15, 10))
  
  for i, class_name in enumerate(class_indices.keys()):
       class_idx = class_indices[class_name]
       class_paths = [
            path for path, lbl in zip(dataset.image_paths, dataset.labels)
            if lbl == class_idx
        ][:num_samples]
       for j, img_path in enumerate(class_paths):
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

import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curves(
    train_batch_losses: list[list[float]],
    test_batch_losses:  list[list[float]],
    labels:             list[str],
    batch_size:         int
):
    fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Training loss subplot
    for losses, name in zip(train_batch_losses, labels):
        x = np.arange(1, len(losses) + 1)
        ax_tr.plot(x, losses, label=name)
    ax_tr.set_yscale('log')
    ax_tr.set_xlabel('number of batches')
    ax_tr.set_ylabel('Loss')
    ax_tr.set_title('Training Loss')
    ax_tr.grid(which='both', linestyle='--', linewidth=0.5)
    ax_tr.legend(loc='upper right')

    # Test loss subplot
    for losses, name in zip(test_batch_losses, labels):
        x = np.arange(1, len(losses) + 1)
        ax_te.plot(x, losses, label=name)
    ax_te.set_yscale('log')
    ax_te.set_xlabel('number of bztches')
    ax_te.set_title('Test Loss')
    ax_te.grid(which='both', linestyle='--', linewidth=0.5)
    ax_te.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_acc_curves_no_log(
    train_batch_accuracies: list[list[float]],
    test_batch_accuracies:  list[list[float]],
    labels:             list[str],
    batch_size:         int
):
    fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(18, 5), sharey=True)

    # Training loss subplot
    for accuracies, name in zip(train_batch_accuracies, labels):
        x = np.arange(1, len(accuracies) + 1)*15
        ax_tr.plot(x, accuracies, label=name, linewidth=.45)
    # ax_tr.set_yscale('log')
    ax_tr.set_xlabel('number of batches')
    ax_tr.set_ylabel('Accuracy')
    ax_tr.set_title('Training accuracy per batch')
    ax_tr.grid(which='both', linestyle='--', linewidth=0.5)
    ax_tr.legend(loc='upper right')

    # Test loss subplot
    for accuracies, name in zip(test_batch_accuracies, labels):
        x = np.arange(1, len(accuracies) + 1)*10
        ax_te.plot(x, accuracies, label=name, linewidth=.55)
    # ax_te.set_yscale('log')
    ax_te.set_xlabel('number of batches')
    ax_te.set_title('Test accuracy per batch')
    ax_te.grid(which='both', linestyle='--', linewidth=0.5)
    ax_te.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_loss_curves_no_log(
    train_batch_losses: list[list[float]],
    test_batch_losses:  list[list[float]],
    labels:             list[str],
    batch_size:         int
):
    fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(18, 5), sharey=True)

    # Training loss subplot
    for losses, name in zip(train_batch_losses, labels):
        x = np.arange(1, len(losses) + 1)*15
        ax_tr.plot(x, losses, label=name, linewidth=.45)
    # ax_tr.set_yscale('log')
    ax_tr.set_xlabel('number of batches')
    ax_tr.set_ylabel('Loss')
    ax_tr.set_title('Training losses per batch')
    ax_tr.grid(which='both', linestyle='--', linewidth=0.5)
    ax_tr.legend(loc='upper right')

    # Test loss subplot
    for losses, name in zip(test_batch_losses, labels):
        x = np.arange(1, len(losses) + 1)*10
        ax_te.plot(x, losses, label=name, linewidth=.55)
    # ax_te.set_yscale('log')
    ax_te.set_xlabel('number of batches')
    ax_te.set_title('Test losses per batch')
    ax_te.grid(which='both', linestyle='--', linewidth=0.5)
    ax_te.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
