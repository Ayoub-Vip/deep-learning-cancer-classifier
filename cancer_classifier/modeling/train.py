import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from ray import tune

def get_model(model_name, model_class, args, device):
    if model_name == "cnn":
        model = model_class(
            fc_dropout_prob=args['fc_dropout_prob'],
            dropout_prob=args['dropout_prob'],
            num_classes=args['num_classes']
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

def get_optimizer(config, model):
    if config["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weights_decay"]
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    return optimizer

def train(config, model_class, device, train_data, val_data, test_data, checkpoint_dir=None, tune=False):

    train_loader, val_loader, test_loader = train_data, val_data, test_data

    model = get_model(config["model_name"], model_class, config, device)

    optimizer = get_optimizer(config, model)
    criterion = config["loss_fn"]

    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []
    true_labels, pred_labels = [], []

    for epoch in tqdm(range(config["epochs"]), desc="Training"):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            acc = (pred.argmax(dim=1) == y).float().mean().item()
            train_losses.append(loss.item())
            train_accuracies.append(acc)

        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                pred = model(X)
                loss = criterion(pred, y)

                acc = (pred.argmax(dim=1) == y).float().mean().item()
                val_losses.append(loss.item())
                val_accuracies.append(acc)
        
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)

                pred = model(X)
                loss = criterion(pred, y)

                acc = (pred.argmax(dim=1) == y).float().mean().item()
                test_losses.append(loss.item())
                test_accuracies.append(acc)
                true_labels.extend(y.cpu().numpy())
                pred_labels.extend(pred.argmax(1).cpu().numpy())

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    avg_test_loss = np.mean(test_losses)
    
    avg_train_acc = np.mean(train_accuracies)
    avg_val_acc = np.mean(val_accuracies)
    avg_test_acc = np.mean(test_accuracies)

    if tune:
        tune.report({
            "train_losses" : train_losses,
            "val_losses" : val_losses,
            "test_losses" : test_losses,
            
            "avg_train_loss" : avg_train_loss,
            "avg_val_loss" : avg_val_loss,
            "avg_test_loss" : avg_test_loss,
            
            "train_accuracies" : train_accuracies,
            "val_accuracies" : val_accuracies,
            "test_accuracies" : test_accuracies,
            
            "avg_train_acc" : avg_train_acc,
            "avg_val_acc" : avg_val_acc,
            "avg_test_acc" : avg_test_acc,
            
            "true_labels" : true_labels,
            "pred_labels" : pred_labels
            }
        )
    else:
        return {
            "train_losses" : train_losses,
            "val_losses" : val_losses,
            "test_losses" : test_losses,
            
            "avg_train_loss" : avg_train_loss,
            "avg_val_loss" : avg_val_loss,
            "avg_test_loss" : avg_test_loss,
            
            "train_accuracies" : train_accuracies,
            "val_accuracies" : val_accuracies,
            "test_accuracies" : test_accuracies,
            
            "avg_train_acc" : avg_train_acc,
            "avg_val_acc" : avg_val_acc,
            "avg_test_acc" : avg_test_acc,
            
            "true_labels" : true_labels,
            "pred_labels" : pred_labels
        }
