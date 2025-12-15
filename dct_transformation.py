import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
from torchvision import transforms
from PIL import Image

# --- CONFIGURATION ---
IMG_SIZE = 299  # Matches XceptionNet requirement
# Mean and Std for ImageNet (Standard for XceptionNet/ViT)
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

def compute_dct(img_array):
    """
    Converts an RGB image to its Frequency Domain representation (DCT).
    
    1. Converts RGB to Grayscale.
    2. Applies Discrete Cosine Transform (DCT).
    3. Converts to Log-Magnitude scale (so the model can 'see' the frequencies).
    """
    # 1. Convert to grayscale (Frequency artifacts are most visible in luma)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 2. Convert to float32 for DCT
    gray_float = np.float32(gray) / 255.0
    
    # 3. Apply DCT
    dct_obj = cv2.dct(gray_float)
    
    # 4. Convert to Log-Magnitude Spectrum
    # We use log because DCT values have a massive range. 
    # This makes the patterns visible to a Neural Network.
    dct_log = np.log(np.abs(dct_obj) + 1e-6)
    
    # 5. Normalize to 0-1 range for stability
    dct_norm = (dct_log - np.min(dct_log)) / (np.max(dct_log) - np.min(dct_log))
    
    # 6. Stack to make it 3-channel (so standard CNNs can accept it)
    dct_3ch = np.stack([dct_norm]*3, axis=-1)
    
    return dct_3ch

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to the 'processed_data' folder.
            split (str): 'train', 'validation', or 'test_celecdf'.
            transform (callable, optional): PyTorch transforms for augmentation.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Collect all image paths
        self.image_paths = []
        self.labels = [] # 0 for Real, 1 for Fake
        
        # Path to the split folder (e.g., ./processed_data/train)
        split_dir = self.root_dir / split
        
        # Check if folder exists
        if not split_dir.exists():
            print(f"Warning: Split directory not found: {split_dir}")
            return

        # --- DYNAMIC FOLDER LOADING ---
        # Instead of hardcoding "real" and "fake", we iterate through all folders.
        # If folder name is "real" -> Label 0.
        # Any other folder name (fake, face2face, deepfakes, etc.) -> Label 1.
        
        print(f"Scanning directory: {split_dir}")
        
        for class_folder in split_dir.iterdir():
            if not class_folder.is_dir():
                continue
                
            folder_name = class_folder.name.lower()
            
            # Determine label
            if folder_name == "real":
                current_label = 0.0
                print(f"  Found 'Real' data in: {folder_name}")
            else:
                current_label = 1.0
                print(f"  Found 'Fake' data in: {folder_name}")

            # Iterate through video folders inside (e.g., train/face2face/video01_frames)
            for vid_folder in class_folder.iterdir():
                if vid_folder.is_dir():
                    # Find all PNG images in the video folder
                    for img_file in vid_folder.glob("*.png"):
                        self.image_paths.append(str(img_file))
                        self.labels.append(current_label)

        print(f"Total loaded: {len(self.image_paths)} images for split: {split}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open as RGB
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Compute Frequency Map (DCT) BEFORE transforms
        # (We want the frequency of the raw image, not the resized/augmented one)
        dct_map = compute_dct(img_rgb)
        
        # 3. Convert to PIL for PyTorch Transforms
        img_pil = Image.fromarray(img_rgb)
        dct_pil = Image.fromarray((dct_map * 255).astype(np.uint8))
        
        # 4. Apply Transforms (Resize, Normalize, Tensor conversion)
        if self.transform:
            img_tensor = self.transform(img_pil)
            dct_tensor = self.transform(dct_pil)
        else:
            # Fallback if no transform provided
            t = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor()
            ])
            img_tensor = t(img_pil)
            dct_tensor = t(dct_pil)
            
        return img_tensor, dct_tensor, torch.tensor(label, dtype=torch.float32)

# --- QUICK TEST BLOCK WITH VISUALIZATION ---
if __name__ == "__main__":
    # Define standard transforms
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    # Initialize dataset (Change path to your actual processed_data folder)
    print("--- Testing Dataset Loading ---")
    dataset = DeepfakeDataset(root_dir="./processed_data", split="train", transform=train_transforms)
    
    if len(dataset) > 0:
        # Get one sample
        rgb_data, dct_data, label = dataset[0]
        
        print(f"\nSample Loaded Successfully:")
        print(f"RGB Tensor Shape: {rgb_data.shape}")
        print(f"DCT Tensor Shape: {dct_data.shape}")
        print(f"Label: {label}")
        
        # --- VISUALIZATION LOGIC ---
        print("\n--- Generating Visualization Previews ---")
        
        # Helper to un-normalize tensor back to image
        def unnormalize_and_convert(tensor):
            # 1. Un-normalize: x = x_norm * std + mean
            # (Assuming mean=0.5, std=0.5 for all channels)
            unnorm = tensor * 0.5 + 0.5
            
            # 2. Convert to numpy (Channels, Height, Width) -> (Height, Width, Channels)
            img_np = unnorm.permute(1, 2, 0).numpy()
            
            # 3. Scale to 0-255
            img_np = (img_np * 255).astype(np.uint8)
            
            # 4. Convert RGB to BGR for OpenCV saving
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Save RGB Face
        face_img = unnormalize_and_convert(rgb_data)
        cv2.imwrite("face_preview.png", face_img)
        print("Saved: face_preview.png (The face image the model sees)")
        
        # Save DCT Map
        dct_img = unnormalize_and_convert(dct_data)
        cv2.imwrite("dct_preview.png", dct_img)
        print("Saved: dct_preview.png (The Frequency map the model sees)")
        
    else:
        print("Dataset is empty")