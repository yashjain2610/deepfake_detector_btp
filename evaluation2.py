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

from dct_transformation import DeepfakeDataset
from model import DeepfakeDetector

# --- CONFIGURATION ---
DEVICE = "mps"
BATCH_SIZE = 16
IMG_SIZE = 299
NUM_WORKERS = 2
TEST_SPLIT_NAME = "validation"  # FaceForensics++ test split

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

def test_faceforensics():
    """
    Evaluate on FaceForensics++ validation set.
    Computes frame-level AND video-level metrics.
    """
    print(f"Using Device: {DEVICE}")
    print(f"Testing on FaceForensics++ ({TEST_SPLIT_NAME} split)")
    
    # 1. Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Load Dataset
    print(f"Loading Test Data from: {TEST_SPLIT_NAME}...")
    test_dataset = DeepfakeDataset(
        root_dir="./processed_data", 
        split=TEST_SPLIT_NAME,
        transform=data_transforms, 
        training_mode=False
    )
    
    if len(test_dataset) == 0:
        print(f"Error: No images found in processed_data/{TEST_SPLIT_NAME}")
        return

    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )

    # 3. Load Model
    print("Loading Trained Model...")
    model = load_model()
    if model is None:
        return

    # 4. Evaluation Loop
    all_labels = []
    all_preds = []
    all_paths = []
    
    print("Running Evaluation...")
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing FaceForensics++")
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
    print(f"FRAME-LEVEL RESULTS ON FACEFORENSICS++")
    print("=" * 50)
    print(f"Total Frames: {len(all_labels)}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"AUC Score: {auc:.4f}")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(all_labels, binary_preds, target_names=["Real", "Fake"]))
    
    # Frame-level Confusion Matrix
    cm = confusion_matrix(all_labels, binary_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=["Real", "Fake"], 
        yticklabels=["Real", "Fake"]
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Frame-Level Confusion Matrix - FaceForensics++')
    plt.tight_layout()
    plt.savefig("confusion_matrix_faceforensics_frame.png")
    plt.close()
    print("Frame-level confusion matrix saved as confusion_matrix_faceforensics_frame.png")

    # ==========================================
    # 6. VIDEO-LEVEL METRICS (Aggregated)
    # ==========================================
    video_preds = defaultdict(list)
    video_labels = {}

    for path, pred, label in zip(all_paths, all_preds, all_labels):
        vid_id = get_video_id(path)
        video_preds[vid_id].append(pred)
        video_labels[vid_id] = label

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
    print(f"VIDEO-LEVEL RESULTS ON FACEFORENSICS++")
    print("=" * 50)
    print(f"Total Videos: {len(vid_ids)}")
    print(f"Accuracy: {vid_acc:.4f} ({vid_acc*100:.2f}%)")
    print(f"AUC Score: {vid_auc:.4f}")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(
        vid_true_labels, vid_binary_preds, 
        target_names=["Real", "Fake"]
    ))

    # Video-level Confusion Matrix
    vid_cm = confusion_matrix(vid_true_labels, vid_binary_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        vid_cm, annot=True, fmt='d', cmap='Greens', 
        xticklabels=["Real", "Fake"], 
        yticklabels=["Real", "Fake"]
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Video-Level Confusion Matrix - FaceForensics++')
    plt.tight_layout()
    plt.savefig("confusion_matrix_faceforensics_video.png")
    plt.close()
    print("Video-level confusion matrix saved as confusion_matrix_faceforensics_video.png")

    # ==========================================
    # 7. PER-MANIPULATION METHOD BREAKDOWN (Optional)
    # ==========================================
    # FaceForensics++ has multiple manipulation methods
    # You can analyze performance per method
    method_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for path, pred, label in zip(all_paths, binary_preds, all_labels):
        # Extract manipulation method from path
        # Assuming path structure: .../processed_data/validation/fake/method_name/video_id/frame.jpg
        parts = Path(path).parts
        if 'fake' in parts:
            fake_idx = parts.index('fake')
            if fake_idx + 1 < len(parts):
                method = parts[fake_idx + 1]
                method_stats[method]["total"] += 1
                if pred == label:
                    method_stats[method]["correct"] += 1
    
    if method_stats:
        print("\n" + "=" * 50)
        print("PER-MANIPULATION METHOD ACCURACY")
        print("=" * 50)
        for method, stats in sorted(method_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{method:20s}: {acc:.4f} ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    test_faceforensics()