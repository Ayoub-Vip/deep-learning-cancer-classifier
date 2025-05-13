import os
import time
import torch
import numpy as np

from pathlib import Path
#from loguru import logger
from tqdm import tqdm
import typer

from cancer_classifier.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def train(input_model, input_train_loader, input_test_loader, input_optimizer,
          input_criterion, epochs, input_device, save_freq, save_path):
    
    avg_train_losses_history = []
    avg_test_losses_history = []
    
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_start = time.time()
        input_model.train()
        train_losses, train_accs = [], []
        
        # training loop
        for images, labels in input_train_loader:
            images = images.to(input_device)
            labels = labels.to(input_device)
            
            outputs = input_model(images)
            loss = input_criterion(outputs, labels)
            
            input_optimizer.zero_grad()
            loss.backward()
            input_optimizer.step()

            train_losses.append(loss.item())
            train_accs.append((outputs.argmax(dim=1) == labels).float().mean().item())
            
        # evaluation loop
        input_model.eval()
        test_losses, test_accs = [], []
        with torch.no_grad():
            for images, labels in input_test_loader:
                images = images.to(input_device)
                labels = labels.to(input_device)

                outputs = input_model(images)
                loss = input_criterion(outputs, labels)
                
                test_losses.append(loss.item())
                test_accs.append((outputs.argmax(dim=1) == labels).float().mean().item())
                
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)
        avg_test_loss = np.mean(test_losses)
        avg_test_acc = np.mean(test_accs)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # if we want to save epochs we can do it like this:
        # if (epoch + 1) % save_freq == 0:
        #     checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pt")
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': input_model.state_dict(),
        #         'optimizer_state_dict': input_optimizer.state_dict(),
        #         'train_loss': avg_train_loss,
        #         'test_loss': avg_test_loss,
        #         'train_acc': avg_train_acc,
        #         'test_acc': avg_test_acc,
        #     }, checkpoint_path)
        #     print(f"Saved checkpoint to {checkpoint_path}")
        
        avg_train_losses_history.append(avg_train_loss)
        avg_test_losses_history.append(avg_test_loss)

    # optionally we could plot the losses here:
    # plot_losses(avg_train_losses_history, avg_test_losses_history)
                
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
