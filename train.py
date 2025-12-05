import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# --- IMPORT YOUR CUSTOM MODULES ---
from dataset import DeepfakeDataset
from model import DeepfakeDetector

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16       # Lower this if you get "Out of Memory" errors
EPOCHS = 20           # How many times to go through the dataset
LEARNING_RATE = 1e-4  # 0.0001 is a good starting point for fine-tuning
IMG_SIZE = 299
NUM_WORKERS = 2       # Set to 0 if you are on Windows and get a "BrokenPipe" error

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    # Progress Bar
    loop = tqdm(loader, desc="Training", leave=False)

    for rgb_imgs, dct_imgs, labels in loop:
        rgb_imgs, dct_imgs, labels = rgb_imgs.to(device), dct_imgs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward Pass
        logits, _ = model(rgb_imgs, dct_imgs)
        
        # Calculate Loss
        # Squeeze logits to match label shape: [batch, 1] -> [batch]
        loss = criterion(logits.squeeze(1), labels)

        # Backward Pass & Optimize
        loss.backward()
        optimizer.step()

        # Track Stats
        running_loss += loss.item()
        
        # Convert logits to probabilities (Sigmoid) for metrics
        probs = torch.sigmoid(logits).detach().cpu().numpy().squeeze()
        all_preds.extend(np.atleast_1d(probs))
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        loop = tqdm(loader, desc="Validating", leave=False)
        for rgb_imgs, dct_imgs, labels in loop:
            rgb_imgs, dct_imgs, labels = rgb_imgs.to(device), dct_imgs.to(device), labels.to(device)

            logits, _ = model(rgb_imgs, dct_imgs)
            loss = criterion(logits.squeeze(1), labels)

            running_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            all_preds.extend(np.atleast_1d(probs))
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    
    # Calculate Metrics
    # Handle edge case where batch size is 1 or all labels are same
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5 # Default if only one class is present in batch
        
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])

    return avg_loss, acc, auc

def main():
    print(f"Using Device: {DEVICE}")
    
    # 1. Define Transforms
    # Note: dataset.py handles resize/to_tensor if transform is None, 
    # but explicit transforms are safer.
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Prepare Datasets
    print("Loading Training Data...")
    train_dataset = DeepfakeDataset(root_dir="./processed_data", split="train", transform=data_transforms)
    
    print("Loading Validation Data...")
    val_dataset = DeepfakeDataset(root_dir="./processed_data", split="validation", transform=data_transforms)
    
    # Check if data exists
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Dataset empty. Please check path and preprocessing.")
        return

    # 3. Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 4. Initialize Model
    print("Initializing Dual-Stream Model...")
    model = DeepfakeDetector().to(DEVICE)
    
    # 5. Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss() # Best for binary classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    best_val_acc = 0.0

    print(f"\nStarting Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, DEVICE)
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"--> New Best Model Saved! (Acc: {val_acc:.4f})")

    print("\n--- Training Complete ---")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print("Model saved as 'best_model.pth'")

if __name__ == "__main__":
    main()