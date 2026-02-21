import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# --- IMPORT CUSTOM MODULES ---
from dct_transformation import DeepfakeDataset
from model import DeepfakeDetector

# --- CONFIGURATION ---
DEVICE = "mps"
BATCH_SIZE = 32       
EPOCHS = 10
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 1e-4
IMG_SIZE = 299
NUM_WORKERS = 4
LABEL_SMOOTHING = 0.05   # Smooths 0.0->0.05 and 1.0->0.95
EARLY_STOP_PATIENCE = 7  # Stop if val AUC doesn't improve for this many epochs
GRAD_CLIP_MAX_NORM = 1.0

# --- FOCAL LOSS ---
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Down-weights well-classified (easy) examples so the model focuses on
    hard examples. Particularly useful when there is an imbalance between
    easy/hard samples in deepfake detection.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        # pt = p if target=1, else 1-p
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce
        return loss.mean()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    # Progress Bar
    loop = tqdm(loader, desc="Training", leave=False)

    for rgb_imgs, dct_imgs, labels, _ in loop:
        rgb_imgs, dct_imgs, labels = rgb_imgs.to(device), dct_imgs.to(device), labels.to(device)

        # Label Smoothing: 0.0 -> 0.05, 1.0 -> 0.95
        smoothed_labels = labels * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / 2.0

        # Zero gradients
        optimizer.zero_grad()

        # Forward Pass
        logits, _ = model(rgb_imgs, dct_imgs)
        
        # Calculate Loss with smoothed labels
        loss = criterion(logits.squeeze(1), smoothed_labels)

        # Backward Pass
        loss.backward()
        
        # Gradient Clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        
        # Optimize
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
        for rgb_imgs, dct_imgs, labels, _ in loop:
            rgb_imgs, dct_imgs, labels = rgb_imgs.to(device), dct_imgs.to(device), labels.to(device)

            logits, _ = model(rgb_imgs, dct_imgs)
            loss = criterion(logits.squeeze(1), labels)

            running_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            all_preds.extend(np.atleast_1d(probs))
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    
    # Calculate Metrics
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5  # Default if only one class is present in batch
        
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])

    return avg_loss, acc, auc

def main():
    print(f"Using Device: {DEVICE}")
    
    # 1. Define Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Prepare Datasets
    print("Loading Training Data...")
    train_dataset = DeepfakeDataset(
        root_dir="./processed_data", split="train",
        transform=data_transforms, training_mode=True  # CRITICAL: enable augmentations
    )
    
    print("Loading Validation Data...")
    val_dataset = DeepfakeDataset(
        root_dir="./processed_data", split="validation",
        transform=data_transforms, training_mode=False
    )
    
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
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning Rate Scheduler: Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    # 6. Training Loop with Early Stopping
    best_val_auc = 0.0
    early_stop_counter = 0

    print(f"\nStarting Training for up to {EPOCHS} epochs (early stop patience: {EARLY_STOP_PATIENCE})...")
    
    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} (LR: {current_lr:.6f}) ---")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Validate
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, DEVICE)
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        # Save Best Model (tracked by AUC, not accuracy)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"--> New Best Model Saved! (AUC: {val_auc:.4f})")
        else:
            early_stop_counter += 1
            print(f"    No improvement. Early stop counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}")
        
        # Early Stopping
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    print("\n--- Training Complete ---")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    print("Model saved as 'best_model.pth'")

if __name__ == "__main__":
    main()
