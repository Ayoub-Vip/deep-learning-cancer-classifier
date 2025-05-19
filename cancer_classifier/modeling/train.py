import os
import time
import torch
import numpy as np

from pathlib import Path
import torch.nn as nn
import numpy as np
from ray import tune
#from loguru import logger
from tqdm import tqdm
import typer

from cancer_classifier.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def get_model(model_name, model_class, args, device):
    if model_name == "cnn":
        model = model_class(
            fc_dropout_prob=args['fc_dropout_prob'],
            num_classes=3
        )
    elif model_name == "vit":
        model = model_class(
            img_size=args["img_size"],
            patch_size=args["patch_size"],
            num_hiddens=args["num_hiddens"],
            mlp_num_hiddens=args["mlp_num_hiddens"],
            num_heads=args["num_heads"],
            num_blks=args["num_blks"],
            emb_dropout=args["emb_dropout"],
            blk_dropout=args["blk_dropout"],
            use_bias=args["use_bias"],
            num_classes=args["num_classes"]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    return model.to(device)


def train(config, model_class, device, dataset, checkpoint_dir=None):
    # Get dataloaders from config (injected via tune.with_parameters)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config["batch_size"]
    )

    # Build model
    model = get_model(config["model_name"], model_class, config, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weights_decay"]
    )
    criterion = nn.CrossEntropyLoss()

    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in tqdm(range(config["epochs"]), desc="Training"):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            train_losses.append(loss.item())
            train_accuracies.append(acc)

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                acc = (outputs.argmax(dim=1) == labels).float().mean().item()
                val_losses.append(loss.item())
                val_accuracies.append(acc)

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    avg_train_acc = np.mean(train_accuracies)
    avg_val_acc = np.mean(val_accuracies)

    tune.report({
        "val_accuracy" : avg_val_acc,
        "train_losses" : train_losses,
        "val_losses" : val_losses,
        "train_accuracies" : train_accuracies,
        "val_accuracies" : val_accuracies,
        "avg_train_loss" : avg_train_loss,
        "avg_val_loss" : avg_val_loss,
        "avg_train_acc" : avg_train_acc,
        "avg_val_acc" : avg_val_acc
        }
    )

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # logger.info("Training some model...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Modeling training complete.")
    # # -----------------------------------------


    # I commented the following section out because there is a lot of unknowns for me. I need to know how 
    
    # dataset = OURDATASET()
    # train_data, test_data = get_dataloader()
    # MODEL = OURMODEL(NUM_MATE_FEATURES, NUM_CLASSES).to(device)
    # optimizer = torch.optim.Adam(MODEL.parameters(), lr=lr)
    # criterion = torch.nn.CrossEntropyLoss()
    
    # train(MODEL, train_loader, test_loader, optimizer, criterion, epochs=epochs, input_device=device, save_freq=save_freq, save_path=save_path)
    
    # # Save the final model at save_path"
    # final_model_local_path = save_final_model(MODEL, save_path)
    pass

if __name__ == "__main__":
    app()
