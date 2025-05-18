import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class BrainTumorDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(sorted(os.listdir(root_dir)))
            if os.path.isdir(os.path.join(root_dir, cls))
            }
        
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
    
    def get_dataloader(self, batch_size, train_split=0.8, val_split=0.1):
        train_size = int(train_split * len(self))
        val_size = int(val_split * len(self))
        test_size = len(self) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(self, [train_size, val_size, test_size])
        
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

if __name__ == "__main__":
    
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
    
    a, b = train_loader
    print(a, b)
    

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
