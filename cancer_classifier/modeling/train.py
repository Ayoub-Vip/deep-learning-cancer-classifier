"""
Example:
    python train.py --data_path ./data --save_path ./checkpoints --epochs 50 --batch_size 32 ...
"""

import os
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torch import nn

from models.vitmodel import ViTClassifier # Placeholder for vit model
from models.cnnmodel import CNNClassifier # Placeholder for cnn model

from cancer_classifier.processing.data_loader import BrainTumorDataset

def get_model(model_type, args, device):
    
    if model_type == "cnn":
        model = CNNClassifier(dropout_prob=args.dropout_prob)
        
    elif model_type == "vit":
        model = ViTClassifier(
            img_size=args.img_size[0],
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dim=args.num_hiddens,
            depth=args.num_blks,
            heads=args.num_heads,
            mlp_dim=args.mlp_num_hiddens,
            dropout=args.blk_dropout,
            emb_dropout=args.emb_dropout
        )
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model.to(device)

def train(model, train_loader, val_loader, optimizer, criterion, epochs, device, save_freq, save_path):
    
    avg_train_losses_history = []
    avg_test_losses_history = []
    
    for epoch in tqdm(range(epochs), desc="Training"):
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
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
              f"Test Loss: {avg_val_loss:.4f}, Test Acc: {avg_val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # if we want to save epochs we can do it like this:
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        avg_train_losses_history.append(avg_train_loss)
        avg_test_losses_history.append(avg_val_loss)

    # optionally we could plot the losses here:
    # plot_losses(avg_train_losses_history, avg_test_losses_history)

if __name__ == "__main__":
    #app()
    parser = get_args_parser() # Need to implement parser
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_path, exist_ok=True)
    
    dataset = BrainTumorDataset(root_dir=args.data_path, max_samples=args.max_samples)
    mean, std = dataset.get_mean_std()
    
    # Preprocessing here, for now I ignore and do a simple thing.
    
    transform_pipeline = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])
    
    dataset.transform = transform_pipeline
    train_loader, val_loader, test_loader = dataset.get_dataloader(batch_size=args.batch_size)
    
    model = get_model(args.model_type, args, device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weights_decay)
    criterion = nn.CrossEntropyLoss()
    
    train(model, train_loader, val_loader, optimizer, criterion, epochs=args.epochs, device=device, save_freq=args.save_freq, save_path=args.save_path)
    
    # Save final model
    os.makedirs(args.save_path, exist_ok=True)
    final_model_local_path = os.path.join(args.save_path, 'final_model.pt')
    torch.save({'model_state_dict': model.state_dict()}, final_model_local_path)
    print(f"Final model saved locally to {final_model_local_path}")
    
    
