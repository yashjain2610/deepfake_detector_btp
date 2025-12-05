import cv2
import os
from tqdm import tqdm
from pathlib import Path
from mtcnn.mtcnn import MTCNN
import sys

# --- CONFIGURATION: YOU MUST EDIT THIS SECTION ---

# 1. Set the desired output size for your models
# 299x299 is for XceptionNet. Use 224x224 for ViT if preferred.
IMG_SIZE = 299

# 2. Set the margin to add around the detected face (e.g., 0.2 = 20%)
# This helps include context like forehead and chin.
CROP_MARGIN = 0.2

# 3. Set how many frames to skip. 
# 1 = process every frame (slow)
# 10 = process every 10th frame (faster, recommended)
FRAME_INTERVAL = 20

# 4. Set the main output directory for your processed faces
OUTPUT_DIR = Path("./processed_data")

# 5. Set the paths to your DOWNLOADED datasets.
# This example assumes you have downloaded FaceForensics++ and Celeb-DF.
# You MUST change these paths to match your local machine.
DATASET_ROOTS = {
    "train_ff_real": Path(r"C:\Users\yash jain\Desktop\folders\btp\FaceForensics++_C23\original"),
    # "train_ff_fake_deepfakes": Path("/path/to/your/FaceForensics++/Deepfakes"),
    # "train_ff_fake_face2face": Path("/path/to/your/FaceForensics++/Face2Face"),
    # "train_ff_fake_faceswap": Path("/path/to/your/FaceForensics++/FaceSwap"),
    # "train_ff_fake_faceshifter": Path("/path/to/your/FaceForensics++/FaceShifter"),
    # "train_ff_fake_neuraltextures": Path("/path/to/your/FaceForensics++/NeuralTextures"),
    
    # "test_celecdf_real": Path("/path/to/your/Celeb-DF-v2/Celeb-real"),
    # "test_celecdf_fake": Path("/path/to/your/Celeb-DF-v2/Celeb-synthesis"),
    
    # Add paths for DFDC or other FF++ manipulations (NeuralTextures, etc.) here
    # "test_dfdc_real": Path("/path/to/your/DFDC/real"),
    # "test_dfdc_fake": Path("/path/to/your/DFDC/fake"),
}

# --- END OF CONFIGURATION ---


def process_video(video_path, output_dir, detector, split_name, label):
    """
    Reads a video, detects faces, crops, and saves them.
    
    :param video_path: Path object for the input video file.
    :param output_dir: Path object for the base output directory.
    :param detector: Initialized MTCNN detector.
    :param split_name: The name of the dataset split (e.g., "train", "test_celecdf").
    :param label: "real" or "fake".
    """
    video_name = video_path.stem  # Get video name without extension
    
    # Create the final output directory for this video's frames
    # e.g., ./processed_data/train/real/video1_frames
    frame_output_dir = output_dir / split_name / label / f"{video_name}_frames"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            # Use tqdm.write for thread-safe printing with the progress bar
            tqdm.write(f"Error: Could not open video file {video_path}", file=sys.stderr)
            return

        frame_count = 0
        saved_frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Process only every N-th frame
            if frame_count % FRAME_INTERVAL == 0:
                # MTCNN expects RGB, OpenCV gives BGR
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                detections = detector.detect_faces(frame_rgb)
                
                if not detections:
                    # No face detected in this frame, skip
                    frame_count += 1
                    continue

                # Find the face with the largest bounding box
                largest_detection = max(detections, key=lambda d: d['box'][2] * d['box'][3])
                x, y, w, h = largest_detection['box']
                
                # Ensure coordinates are positive
                x, y = max(0, x), max(0, y)
                
                # Add margin
                x1 = int(x - w * CROP_MARGIN)
                y1 = int(y - h * CROP_MARGIN)
                x2 = int(x + w + w * CROP_MARGIN)
                y2 = int(y + h + h * CROP_MARGIN)

                # Get frame dimensions
                frame_h, frame_w = frame.shape[:2]

                # --- Critical: Clamp coordinates to be within frame boundaries ---
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_w - 1, x2)
                y2 = min(frame_h - 1, y2)

                # Ensure the crop is valid (has width and height)
                if x1 >= x2 or y1 >= y2:
                    frame_count += 1
                    continue
                    
                # Crop the face
                cropped_face = frame[y1:y2, x1:x2]
                
                # Resize to standard size
                resized_face = cv2.resize(cropped_face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                
                # Save the frame
                output_filename = frame_output_dir / f"frame_{saved_frame_count:04d}.png"
                cv2.imwrite(str(output_filename), resized_face)
                
                saved_frame_count += 1
                
            frame_count += 1

        # Use tqdm.write to print without breaking the progress bar
        tqdm.write(f"Processed {video_path.name}: {saved_frame_count} frames saved.")
        cap.release()

    except Exception as e:
        # Use tqdm.write for thread-safe printing with the progress bar
        tqdm.write(f"Error processing video {video_path}: {e}", file=sys.stderr)


def main():
    """
    Main function to initialize and run the preprocessing pipeline.
    """
    print("Initializing MTCNN face detector...")
    # This might take a moment and download model weights on first run
    detector = MTCNN()
    print("Detector initialized.")

    # Create the base output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Output will be saved to: {OUTPUT_DIR.resolve()}")

    # Loop through all the dataset paths defined in the config
    for key, root_path in DATASET_ROOTS.items():
        if not root_path.exists():
            # Use print here, as it's before the tqdm loop
            print(f"Warning: Path not found, skipping: {root_path}", file=sys.stderr)
            continue
            
        # Determine split name (e.g., "train") and label (e.g., "real")
        if "train" in key:
            split_name = "train"
        elif "val" in key:
            split_name = "validation"
        elif "test" in key:
            split_name = key.replace("_real", "").replace("_fake", "") # e.g., "test_celecdf"
        else:
            split_name = "unknown"

        label = "real" if "real" in key else "fake"
        
        print(f"\nProcessing {key} (Split: {split_name}, Label: {label})...")
        print(f"Source: {root_path}")

        video_files = list(root_path.glob("*.mp4")) + list(root_path.glob("*.avi"))
        if not video_files:
            # Use print here, as it's before the tqdm loop
            print(f"Warning: No .mp4 or .avi videos found in {root_path}", file=sys.stderr)
            continue

        print(f"Found {len(video_files)} videos.")

        # Process each video with a tqdm progress bar
        for video_path in tqdm(video_files, desc=f"Processing {key}", unit="video"):
            process_video(video_path, OUTPUT_DIR, detector, split_name, label)

    print("\n--- Preprocessing Complete ---")
    print(f"All processed data saved in: {OUTPUT_DIR.resolve()}")
    print("You can now proceed to model training.")


if __name__ == "__main__":
    main()

