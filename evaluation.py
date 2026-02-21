import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io

# --- IMPORT CUSTOM MODULES ---
from dct_transformation import DeepfakeDataset, compute_dct
from model import DeepfakeDetector

# --- CONFIGURATION ---
DEVICE = "mps"
BATCH_SIZE = 16
IMG_SIZE = 299
NUM_WORKERS = 2

# Name of the split folder to test on
# Options: "validation" (FF++), "test_celebdf" (Celeb-DF)
TEST_SPLIT_NAME = "test_celebdf"


def load_model():
    """Load the trained DeepfakeDetector model."""
    model = DeepfakeDetector().to(DEVICE)
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
        print("Loaded 'best_model.pth' successfully.")
    except FileNotFoundError:
        print("Error: 'best_model.pth' not found. Train the model first!")
        return None
    model.eval()
    return model


def get_video_id(img_path):
    """Extract video folder name from an image path for video-level grouping."""
    return Path(img_path).parent.name


def test_model():
    """
    Main evaluation: frame-level AND video-level metrics.
    """
    print(f"Using Device: {DEVICE}")
    
    # 1. Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Load Dataset
    print(f"Loading Test Data from: {TEST_SPLIT_NAME}...")
    test_dataset = DeepfakeDataset(
        root_dir="./processed_data", split=TEST_SPLIT_NAME,
        transform=data_transforms, training_mode=False
    )
    
    if len(test_dataset) == 0:
        print(f"Error: No images found in processed_data/{TEST_SPLIT_NAME}")
        return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 3. Load Model
    print("Loading Trained Model...")
    model = load_model()
    if model is None:
        return

    # 4. Evaluation Loop -- collect frame-level predictions AND image paths
    all_labels = []
    all_preds = []
    all_paths = []
    
    print("Running Evaluation...")
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing")
        for rgb_imgs, dct_imgs, labels, img_paths in loop:
            rgb_imgs = rgb_imgs.to(DEVICE)
            dct_imgs = dct_imgs.to(DEVICE)
            
            # Forward pass
            logits, _ = model(rgb_imgs, dct_imgs)
            
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            all_preds.extend(np.atleast_1d(probs))
            all_labels.extend(labels.numpy())
            all_paths.extend(img_paths)

    # ==========================================
    # 5. FRAME-LEVEL METRICS
    # ==========================================
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    
    acc = accuracy_score(all_labels, binary_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.5
        
    print("\n" + "=" * 50)
    print(f"FRAME-LEVEL RESULTS ON {TEST_SPLIT_NAME}")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"AUC Score: {auc:.4f}")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(all_labels, binary_preds, target_names=["Real", "Fake"]))
    
    # Frame-level Confusion Matrix
    cm = confusion_matrix(all_labels, binary_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Frame-Level Confusion Matrix - {TEST_SPLIT_NAME}')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{TEST_SPLIT_NAME}_new_1.png")
    plt.close()
    print(f"Frame-level confusion matrix saved as confusion_matrix_{TEST_SPLIT_NAME}_new.png")

    # ==========================================
    # 6. VIDEO-LEVEL METRICS (Aggregated)
    # ==========================================
    video_preds = defaultdict(list)
    video_labels = {}

    for path, pred, label in zip(all_paths, all_preds, all_labels):
        vid_id = get_video_id(path)
        video_preds[vid_id].append(pred)
        video_labels[vid_id] = label  # All frames in a video share the same label

    # Average frame probabilities per video
    vid_ids = sorted(video_preds.keys())
    vid_avg_preds = [np.mean(video_preds[v]) for v in vid_ids]
    vid_true_labels = [video_labels[v] for v in vid_ids]
    vid_binary_preds = [1 if p > 0.5 else 0 for p in vid_avg_preds]

    vid_acc = accuracy_score(vid_true_labels, vid_binary_preds)
    try:
        vid_auc = roc_auc_score(vid_true_labels, vid_avg_preds)
    except Exception:
        vid_auc = 0.5

    print("\n" + "=" * 50)
    print(f"VIDEO-LEVEL RESULTS ON {TEST_SPLIT_NAME}")
    print("=" * 50)
    print(f"Total Videos: {len(vid_ids)}")
    print(f"Accuracy: {vid_acc:.4f} ({vid_acc*100:.2f}%)")
    print(f"AUC Score: {vid_auc:.4f}")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(vid_true_labels, vid_binary_preds, target_names=["Real", "Fake"]))

    # Video-level Confusion Matrix
    vid_cm = confusion_matrix(vid_true_labels, vid_binary_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(vid_cm, annot=True, fmt='d', cmap='Greens', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Video-Level Confusion Matrix - {TEST_SPLIT_NAME}')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_video_{TEST_SPLIT_NAME}_new_1.png")
    plt.close()
    print(f"Video-level confusion matrix saved as confusion_matrix_video_{TEST_SPLIT_NAME}.png")


def test_compression_resilience():
    """
    Compression Robustness Stress Test.
    Tests the model on the same dataset at various JPEG quality factors
    to measure how accuracy/AUC degrade under compression.
    Saves a line plot to compression_resilience.png.
    """
    print(f"\n{'='*50}")
    print("COMPRESSION RESILIENCE STRESS TEST")
    print(f"{'='*50}")
    print(f"Using Device: {DEVICE}")

    # Load model
    model = load_model()
    if model is None:
        return

    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset to get image paths and labels
    test_dataset = DeepfakeDataset(
        root_dir="./processed_data", split=TEST_SPLIT_NAME,
        transform=data_transforms, training_mode=False
    )

    if len(test_dataset) == 0:
        print(f"Error: No images found in processed_data/{TEST_SPLIT_NAME}")
        return

    quality_factors = [100, 80, 60, 40, 20]
    results = {"qf": [], "accuracy": [], "auc": []}

    for qf in quality_factors:
        print(f"\n--- Testing at JPEG Quality Factor: {qf} ---")
        all_labels = []
        all_preds = []

        for idx in tqdm(range(len(test_dataset)), desc=f"QF={qf}", leave=False):
            img_path = test_dataset.image_paths[idx]
            label = test_dataset.labels[idx]

            # Load and compress the image at the specified quality factor
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # JPEG compress in memory
            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG', quality=qf)
            buffer.seek(0)
            img_compressed = Image.open(buffer).convert('RGB')

            # Compute DCT on the compressed image
            img_np = np.array(img_compressed)
            dct_map = compute_dct(img_np)
            dct_pil = Image.fromarray((dct_map * 255).astype(np.uint8))

            # Apply transforms
            img_tensor = data_transforms(img_compressed).unsqueeze(0).to(DEVICE)
            dct_tensor = data_transforms(dct_pil).unsqueeze(0).to(DEVICE)

            # Inference
            with torch.no_grad():
                logits, _ = model(img_tensor, dct_tensor)
                prob = torch.sigmoid(logits).cpu().item()

            all_preds.append(prob)
            all_labels.append(label)

        # Compute metrics for this quality factor
        binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
        acc = accuracy_score(all_labels, binary_preds)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except Exception:
            auc = 0.5

        results["qf"].append(qf)
        results["accuracy"].append(acc)
        results["auc"].append(auc)

        print(f"  QF={qf}: Accuracy={acc:.4f}, AUC={auc:.4f}")

    # Print summary table
    print("\n" + "=" * 50)
    print("COMPRESSION RESILIENCE SUMMARY")
    print("=" * 50)
    print(f"{'QF':<10}{'Accuracy':<15}{'AUC':<15}")
    print("-" * 40)
    for qf, acc, auc in zip(results["qf"], results["accuracy"], results["auc"]):
        print(f"{qf:<10}{acc:<15.4f}{auc:<15.4f}")

    # Plot results
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel('JPEG Quality Factor')
    ax1.set_ylabel('Score')
    ax1.plot(results["qf"], results["accuracy"], 'b-o', label='Accuracy', linewidth=2)
    ax1.plot(results["qf"], results["auc"], 'r-s', label='AUC', linewidth=2)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(max(quality_factors) + 5, min(quality_factors) - 5)  # High to low QF
    ax1.legend(loc='lower left')
    ax1.set_title('Compression Resilience: Accuracy & AUC vs JPEG Quality')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("compression_resilience.png", dpi=150)
    plt.close()
    print("\nPlot saved as compression_resilience.png")


if __name__ == "__main__":
    # Run standard evaluation (frame-level + video-level)
    test_model()
    
    # # Run compression resilience stress test
    # test_compression_resilience()
