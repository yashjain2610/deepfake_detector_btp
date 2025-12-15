import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = Path("./processed_data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
SPLIT_RATIO = 0.2  # 20% of videos go to validation

def create_split():
    if not TRAIN_DIR.exists():
        print(f"Error: Training directory not found at {TRAIN_DIR}")
        return

    # Create validation directory if it doesn't exist
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Iterate over each class folder (real, fake, face2face, etc.)
    # We want to maintain the same balance in validation
    class_folders = [f for f in TRAIN_DIR.iterdir() if f.is_dir()]
    
    print(f"Found class folders: {[f.name for f in class_folders]}")
    
    for class_folder in class_folders:
        print(f"\nProcessing class: {class_folder.name}...")
        
        # 1. Create corresponding folder in validation
        # e.g., processed_data/validation/real
        val_class_dir = VAL_DIR / class_folder.name
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. List all VIDEO folders inside
        # (We split by video folder, not by image file, to prevent leakage)
        video_folders = [f for f in class_folder.iterdir() if f.is_dir()]
        total_videos = len(video_folders)
        
        if total_videos == 0:
            print(f"  No videos found in {class_folder.name}, skipping.")
            continue
            
        # 3. Shuffle and Pick 20%
        random.shuffle(video_folders)
        num_val = int(total_videos * SPLIT_RATIO)
        
        # Ensure at least one video goes to val if possible (unless only 1 exists)
        if num_val == 0 and total_videos > 1:
            num_val = 1
            
        val_videos = video_folders[:num_val]
        
        print(f"  Moving {num_val} out of {total_videos} videos to Validation...")
        
        # 4. Move the folders
        for vid_folder in tqdm(val_videos, desc=f"Moving {class_folder.name}", unit="vid"):
            # Source: processed_data/train/real/video_01
            # Dest:   processed_data/validation/real/video_01
            destination = val_class_dir / vid_folder.name
            
            # Move the directory
            shutil.move(str(vid_folder), str(destination))
            
    print("\n--- Split Complete ---")
    print(f"Validation data created at: {VAL_DIR}")

if __name__ == "__main__":
    # Safety check
    response = input(f"This will move 20% of folders from {TRAIN_DIR} to {VAL_DIR}. Proceed? (y/n): ")
    if response.lower() == 'y':
        create_split()
    else:
        print("Operation cancelled.")