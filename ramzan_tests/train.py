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
from ray import tune
import torch

from test import CNNModel

#from models.vitmodel import ViTClassifier # Placeholder for vit model
#from models.cnnmodel import CNNClassifier # Placeholder for cnn model

from test2 import BrainTumorDataset

def get_model(model_type, args, device):
    if model_type == "cnn":
        model = CNNModel(num_classes=3)
    elif model_type == "vit":
        # model = ViTClassifier(
        #     img_size=args["img_size"],
        #     patch_size=args["patch_size"],
        #     num_classes=args["num_classes"],
        #     dim=args["num_hiddens"],
        #     depth=args["num_blks"],
        #     heads=args["num_heads"],
        #     mlp_dim=args["mlp_num_hiddens"],
        #     dropout=args["blk_dropout"],
        #     emb_dropout=args["emb_dropout"]
        # )
        model = CNNModel(num_classes=3)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model.to(device)

def train(config, checkpoint_dir=None):
    #print("training started")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")
    dataset = BrainTumorDataset(root_dir=config["data_path"], max_samples=config.get("max_samples"))
    mean, std = dataset.get_mean_std()

    transform_pipeline = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])
    
    dataset.transform = transform_pipeline
    train_loader, val_loader, test_loader = dataset.get_dataloader(batch_size=config["batch_size"])

    model = get_model(config["model_type"], config, device)
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
        
        """
        print(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Test Acc: {avg_val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
        """
        tune.report({"val_accuracy" :avg_val_acc})
        
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
    #app()
    # parser = get_args_parser() # Need to implement parser
    # args = parser.parse_args()
    
    # device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # os.makedirs(args.save_path, exist_ok=True)
    
    # dataset = BrainTumorDataset(root_dir=args.data_path, max_samples=args.max_samples)
    # mean, std = dataset.get_mean_std()
    
    # # Preprocessing here, for now I ignore and do a simple thing.
    
    # transform_pipeline = transforms.Compose([
    #     transforms.Resize(args.img_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    # ])
    
    # dataset.transform = transform_pipeline
    # train_loader, val_loader, test_loader = dataset.get_dataloader(batch_size=args.batch_size)
    
    # model = get_model(args.model_type, args, device)
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weights_decay)
    # criterion = nn.CrossEntropyLoss()
    
    # train(model, train_loader, val_loader, optimizer, criterion, epochs=args.epochs, device=device, save_freq=args.save_freq, save_path=args.save_path)
    
    # # Save final model
    # os.makedirs(args.save_path, exist_ok=True)
    # final_model_local_path = os.path.join(args.save_path, 'final_model.pt')
    # torch.save({'model_state_dict': model.state_dict()}, final_model_local_path)
    # print(f"Final model saved locally to {final_model_local_path}")
    config = {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        #"dropout_prob": tune.uniform(0.2, 0.7),
        "batch_size": tune.choice([16, 32, 64]),
        #"img_size": tune.choice([224, 256]),
        "img_size": 256,
        #"model_type": "tune.choice(["cnn", "vit"])",
        "model_type": "cnn",
        "patch_size": tune.choice([16, 32]),
        #"num_hiddens": tune.choice([126, 252, 516]),
        #"num_heads": tune.choice([8, 12]),
        #"mlp_num_hiddens": 156,
        #"num_blks": tune.choice([2, 3, 5]),
        #"blk_dropout": tune.uniform(0.0, 0.2),
        #"emb_dropout": tune.uniform(0.0, 0.2),
        "weights_decay": tune.choice([0.0001, 0.001]),
        "epochs": 1,
        "data_path": os.path.abspath("../data/raw/"),
        #"num_classes": 3
    }
    trainable_with_resources = tune.with_resources(train, {"cpu": 2, "gpu": 1})
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=-1,
            metric="val_accuracy",
            mode="max"
        ),
        run_config=tune.RunConfig(name="brain_tumor_ray_tuning"),
    )

    results = tuner.fit()
    print("Best config:", results.get_best_result(metric="val_accuracy", mode="max").config)

