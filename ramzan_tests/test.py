from torch import nn
import torch

import time
import numpy as np

from test2 import BrainTumorDataset
from torchvision import transforms
from tqdm import tqdm
import os


def conv_block(in_channels, out_channels, kernel_size=3,
                stride=1, padding=1, pool_kernel_size=2,
                pool_stride=2, dropout_prob=0.2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout_prob),
        nn.MaxPool2d(pool_kernel_size, pool_stride)
    )

class CNNModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNModel, self).__init__()

        self.conv_block1 = conv_block(1, 64, 3)
        self.conv_block2 = conv_block(64, 128, 5)
        self.conv_block3 = conv_block(128, 256, 5)
        self.conv_block4 = conv_block(256, 512, 3, 2)

        dummy_input = torch.randn(1, 1, 256, 256) # Sadece bir görsel için çıkış boyutu hesaplar
        dummy_output = self.conv_block4(self.conv_block3(self.conv_block2(self.conv_block1(dummy_input))))

        flattened_size = torch.flatten(dummy_output, 1).size(1)

        print(f'the full architecture parameters size: {flattened_size}')

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(flattened_size, 512), # Hesaplanan boyutu kullan
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
            )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = torch.flatten(x, 1)

        x = self.fc_layers(x)

        return x


def train(config, checkpoint_dir=None):
    print("training started")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BrainTumorDataset(root_dir=config["data_path"], max_samples=config.get("max_samples"))
    mean, std = dataset.get_mean_std()

    transform_pipeline = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])
    
    dataset.transform = transform_pipeline
    
    print("Checking:")
    all_labels = [label for _, label in dataset]
    unique_labels = set(all_labels)
    print(f"Unique labels: {unique_labels}")
    
    train_loader, val_loader, test_loader = dataset.get_dataloader(batch_size=config["batch_size"])

    model = CNNModel(num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weights_decay"])
    criterion = nn.CrossEntropyLoss()

    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    avg_train_losses_history = []
    avg_test_losses_history = []
    
    for epoch in tqdm(range(config["epochs"]), desc="Training"):
        epoch_start = time.time()
        model.train()
        train_losses, train_accs = [], []
        
        # training loop
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append((outputs.argmax(dim=1) == labels).float().mean().item())
            
        # evaluation loop
        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_losses.append(loss.item())
                val_accs.append((outputs.argmax(dim=1) == labels).float().mean().item())
                
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)
        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accs)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Test Acc: {avg_val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        
        # if we want to save epochs we can do it like this:
        # if (epoch + 1) % save_freq == 0:
        #     checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pt")
        #     torch.save(model.state_dict(), checkpoint_path)
        #     print(f"Saved checkpoint to {checkpoint_path}")
            
        
        avg_train_losses_history.append(avg_train_loss)
        avg_test_losses_history.append(avg_val_loss)

    # optionally we could plot the losses here:
    # plot_losses(avg_train_losses_history, avg_test_losses_history)
    
    
if __name__ == "__main__":   
    
    config = {
        "learning_rate": 1e-3,
        #"dropout_prob": tune.uniform(0.2, 0.7),
        "batch_size": 32,
        #"img_size": tune.choice([224, 256]),
        "img_size": 256,
        #"model_type": "tune.choice(["cnn", "vit"])",
        #"num_hiddens": tune.choice([126, 252, 516]),
        #"num_heads": tune.choice([8, 12]),
        #"mlp_num_hiddens": 156,
        #"num_blks": tune.choice([2, 3, 5]),
        #"blk_dropout": tune.uniform(0.0, 0.2),
        #"emb_dropout": tune.uniform(0.0, 0.2),
        "weights_decay": 0.001,
        "epochs": 1,
        "data_path": os.path.abspath("../data/raw/"),
        #"num_classes": 3
    }
    
    train(config)
    