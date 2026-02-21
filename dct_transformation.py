import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageFilter
import random
import io

# --- CONFIGURATION ---
IMG_SIZE = 299

def frequency_masking(dct_map, max_mask_pct=0.15):
    """
    Implements Frequency-Domain Masking (FDM).
    Randomly zeros out a block of the frequency spectrum.
    """
    h, w, c = dct_map.shape
    
    # 50% chance to apply masking
    if random.random() > 0.5:
        return dct_map
        
    # Define mask size
    mask_h = int(h * random.uniform(0.05, max_mask_pct))
    mask_w = int(w * random.uniform(0.05, max_mask_pct))
    
    # Random position
    y = random.randint(0, h - mask_h)
    x = random.randint(0, w - mask_w)
    
    # Apply mask (set to 0)
    dct_map[y:y+mask_h, x:x+mask_w, :] = 0
    
    return dct_map

def compute_dct(img_array):
    """ 
    Converts RGB to Per-Channel Log-Magnitude DCT Map.
    Processes R, G, B channels independently to preserve color-channel
    forensic information that deepfake generators leave behind.
    """
    # Ensure even dimensions (OpenCV DCT requires even sizes)
    h, w = img_array.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    
    if pad_h or pad_w:
        img_array = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    
    channels = cv2.split(img_array)  # Split into R, G, B
    dct_channels = []
    for ch in channels:
        # Convert to float and normalize to 0-1
        ch_float = np.float32(ch) / 255.0
        
        # Apply Discrete Cosine Transform
        dct_obj = cv2.dct(ch_float)
        
        # Log-Magnitude scaling (to make features visible)
        dct_log = np.log(np.abs(dct_obj) + 1e-6)
        
        # Normalize to 0-1 range
        dct_norm = (dct_log - dct_log.min()) / (dct_log.max() - dct_log.min() + 1e-8)
        dct_channels.append(dct_norm)
    
    dct_result = np.stack(dct_channels, axis=-1)
    
    # Crop back to original size if we padded
    if pad_h or pad_w:
        dct_result = dct_result[:h, :w, :]
    
    return dct_result

def jpeg_compress(img_pil, quality_range=(30, 95)):
    """
    Simulates real JPEG compression by saving to an in-memory buffer
    at a random quality factor and re-reading. This produces authentic
    block artifacts and quantization noise that Gaussian blur cannot replicate.
    """
    buffer = io.BytesIO()
    q = random.randint(*quality_range)
    img_pil.save(buffer, format='JPEG', quality=q)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')

def resolution_degrade(img_pil, scale_range=(0.5, 0.9)):
    """
    Simulates low-resolution video by downscaling then upscaling.
    This teaches the model to handle varying input resolutions.
    """
    w, h = img_pil.size
    scale = random.uniform(*scale_range)
    small = img_pil.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, training_mode=False):
        """
        Args:
            root_dir (str): Path to processed_data folder
            split (str): 'train', 'validation', or 'test_celecdf'
            transform (callable): PyTorch transforms
            training_mode (bool): If True, enables FDM and Robust Augmentations.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.training_mode = training_mode 
        
        self.image_paths = []
        self.labels = []
        
        split_dir = self.root_dir / split
        if not split_dir.exists():
            print(f"Warning: Split directory not found: {split_dir}")
            return

        # --- Dynamic Folder Loading ---
        # Iterates through 'real', 'fake', etc.
        for class_folder in split_dir.iterdir():
            if not class_folder.is_dir(): continue
            
            folder_name = class_folder.name.lower()
            
            # Label 0 for Real, 1 for everything else (Fake, Deepfakes, Face2Face)
            current_label = 0.0 if folder_name == "real" else 1.0
            
            # Check if this has method subfolders (faceshifter, faceswap, etc.)
            # or direct video folders
            subfolders = [f for f in class_folder.iterdir() if f.is_dir()]
            
            for subfolder in subfolders:
                # Check if this subfolder contains video folders or frames directly
                contents = list(subfolder.iterdir())
                has_png_files = any(f.suffix.lower() == '.png' for f in contents if f.is_file())
                
                if has_png_files:
                    # This is a video folder with frames directly
                    for img_file in subfolder.glob("*.png"):
                        self.image_paths.append(str(img_file))
                        self.labels.append(current_label)
                else:
                    # This is a method folder (faceshifter/faceswap/etc), go one level deeper
                    for vid_folder in subfolder.iterdir():
                        if vid_folder.is_dir():
                            for img_file in vid_folder.glob("*.png"):
                                self.image_paths.append(str(img_file))
                                self.labels.append(current_label)

        print(f"[{split.upper()}] Loaded {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 1. Load Image using OpenCV
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # Fallback for corrupted images
            print(f"Error loading: {img_path}")
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.tensor(label), img_path

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # --- ROBUST AUGMENTATION PIPELINE (Only in Training) ---
        if self.training_mode:
            # A. Spatial Augmentations
            
            # Random Horizontal Flip
            if random.random() > 0.5:
                img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Compression Simulation (Gaussian Blur)
            # This teaches the model to survive low-quality video inputs
            if random.random() > 0.5:
                img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
            
            # B. JPEG Compression Augmentation
            # Simulates real compression artifacts (block artifacts, quantization noise)
            if random.random() > 0.6:
                img_pil = jpeg_compress(img_pil, quality_range=(30, 95))
            
            # C. Resolution Degradation
            # Simulates low-resolution video sources
            if random.random() > 0.6:
                img_pil = resolution_degrade(img_pil, scale_range=(0.5, 0.9))
                
            # D. Color Jitter (Lighting differences)
            color_transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            img_pil = color_transform(img_pil)

        # 2. Compute DCT (Frequency Domain)
        # CRITICAL: Compute DCT *after* spatial augmentations (like Blur) so the 
        # frequency map reflects the "compressed" state.
        img_aug_np = np.array(img_pil)
        dct_map = compute_dct(img_aug_np)
        
        # --- FREQUENCY DOMAIN MASKING (FDM) ---
        if self.training_mode:
            dct_map = frequency_masking(dct_map)
        
        # Convert DCT back to PIL for final transforms
        dct_pil = Image.fromarray((dct_map * 255).astype(np.uint8))
        
        # 3. Apply Final Transforms (Resize, Tensor, Normalize)
        if self.transform:
            img_tensor = self.transform(img_pil)
            dct_tensor = self.transform(dct_pil)
        else:
            # Default transform if none provided
            t = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor()
            ])
            img_tensor = t(img_pil)
            dct_tensor = t(dct_pil)
            
        return img_tensor, dct_tensor, torch.tensor(label, dtype=torch.float32), img_path
