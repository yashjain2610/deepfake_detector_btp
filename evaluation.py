import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- IMPORT CUSTOM MODULES ---
from dct_transformation import DeepfakeDataset
from model import DeepfakeDetector

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
IMG_SIZE = 299
NUM_WORKERS = 2

# Name of the split folder to test on
# Options: "validation" (FF++), "test_celebdf" (Celeb-DF)"
TEST_SPLIT_NAME = "test_celebdf" 

def test_model():
    print(f"Using Device: {DEVICE}")
    
    # 1. Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Load Dataset
    print(f"Loading Test Data from: {TEST_SPLIT_NAME}...")
    test_dataset = DeepfakeDataset(root_dir="./processed_data", split=TEST_SPLIT_NAME, transform=data_transforms)
    
    if len(test_dataset) == 0:
        print(f"Error: No images found in processed_data/{TEST_SPLIT_NAME}")
        return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 3. Load Model
    print("Loading Trained Model...")
    model = DeepfakeDetector().to(DEVICE)
    
    # Load weights
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
        print("Loaded 'best_model.pth' successfully.")
    except FileNotFoundError:
        print("Error: 'best_model.pth' not found. Train the model first!")
        return

    model.eval()

    # 4. Evaluation Loop
    all_labels = []
    all_preds = []
    
    print("Running Evaluation...")
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing")
        for rgb_imgs, dct_imgs, labels in loop:
            rgb_imgs = rgb_imgs.to(DEVICE)
            dct_imgs = dct_imgs.to(DEVICE)
            
            # Forward pass
            logits, _ = model(rgb_imgs, dct_imgs)
            
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            all_preds.extend(np.atleast_1d(probs))
            all_labels.extend(labels.numpy())

    # 5. Calculate Metrics
    # Binary predictions (0 or 1)
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    
    acc = accuracy_score(all_labels, binary_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5
        
    print("\n" + "="*30)
    print(f"RESULTS ON {TEST_SPLIT_NAME}")
    print("="*30)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"AUC Score: {auc:.4f}")
    print("-" * 30)
    print("\nClassification Report:")
    print(classification_report(all_labels, binary_preds, target_names=["Real", "Fake"]))
    
    # 6. Confusion Matrix
    cm = confusion_matrix(all_labels, binary_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {TEST_SPLIT_NAME}')
    plt.savefig(f"confusion_matrix_{TEST_SPLIT_NAME}.png")
    print(f"Confusion matrix saved as confusion_matrix_{TEST_SPLIT_NAME}.png")

if __name__ == "__main__":
    test_model()