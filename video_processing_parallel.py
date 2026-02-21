import os
import logging

# --- Disable TensorFlow and MTCNN verbosity ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.keras.utils.disable_interactive_logging()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress absl logging
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
from tqdm import tqdm
from pathlib import Path
from mtcnn.mtcnn import MTCNN
import sys
import contextlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# Patch TensorFlow's Model.predict to always use verbose=0
original_predict = tf.keras.Model.predict

def silent_predict(self, *args, **kwargs):
    kwargs['verbose'] = 0
    return original_predict(self, *args, **kwargs)

tf.keras.Model.predict = silent_predict

IMG_SIZE = 299
CROP_MARGIN = 0.2
FRAME_INTERVAL = 20
OUTPUT_DIR = Path("./processed_data")

# Configure your dataset paths here
DATASET_ROOTS = {
    # FaceForensics++ - Training data (real videos)
    # "train_ff_real": Path("/Users/int1967/Desktop/btp/FaceForensics++_C23/original"),
    
    # # FaceForensics++ - Training data (all manipulation types)
    # "train_ff_fake_deepfakes": Path("/Users/int1967/Desktop/btp/FaceForensics++_C23/Deepfakes"),
    # "train_ff_fake_face2face": Path("/Users/int1967/Desktop/btp/FaceForensics++_C23/Face2Face"),
    # "train_ff_fake_faceswap": Path("/Users/int1967/Desktop/btp/FaceForensics++_C23/FaceSwap"),
    # "train_ff_fake_neuraltextures": Path("/Users/int1967/Desktop/btp/FaceForensics++_C23/NeuralTextures"),
    # #"train_ff_fake_deepfakedetection": Path("/Users/int1967/Desktop/btp/FaceForensics++_C23/DeepFakeDetection"),
    # "train_ff_fake_faceshifter": Path("/Users/int1967/Desktop/btp/FaceForensics++_C23/FaceShifter"),
    
    # Celeb-DF - Test data
    # "test_celebdf_real": Path("/Users/int1967/Desktop/btp/celebdf/Celeb-real"),
    # "test_celebdf_fake": Path("/Users/int1967/Desktop/btp/celebdf/Celeb-synthesis"),
    #"test_celebdf_youtube_real": Path("/Users/int1967/Desktop/btp/celebdf/YouTube-real")

    "train_ff_fake_faceswap": Path("/Users/int1967/Desktop/btp/FaceForensics++_C23/FaceSwap"),
    "train_ff_fake_neuraltextures": Path("/Users/int1967/Desktop/btp/FaceForensics++_C23/NeuralTextures"),
    #"train_ff_fake_deepfakedetection": Path("/Users/int1967/Desktop/btp/FaceForensics++_C23/DeepFakeDetection"),
    "train_ff_fake_faceshifter": Path("/Users/int1967/Desktop/btp/FaceForensics++_C23/FaceShifter"),
}

# Number of threads to use
MAX_WORKERS = os.cpu_count() # Adjust based on your CPU cores

# Optional: Limit number of videos per folder (set to None to process all)
MAX_VIDEOS_PER_FOLDER = None  # Change to a number like 100 to limit processing

# Force reprocessing of videos even if frames already exist
FORCE_REPROCESS = False  # Set to True to reprocess all videos

print(f"Using {MAX_WORKERS} threads")

# --- END OF CONFIGURATION ---


def process_video(video_path, output_dir, detector, split_name, label, manipulation_type="unknown"):
    """
    Reads a video, detects faces, crops, and saves them.
    """
    video_name = video_path.stem
    # Include manipulation type in path to prevent collisions between datasets
    frame_output_dir = output_dir / split_name / label / manipulation_type / f"{video_name}_frames"
    
    # Check if already processed (skip if frames exist)
    if not FORCE_REPROCESS and frame_output_dir.exists():
        existing_frames = list(frame_output_dir.glob("*.png"))
        if len(existing_frames) > 0:
            # Silently skip already processed videos
            return len(existing_frames)
    
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}", file=sys.stderr)
            return 0

        frame_count = 0
        saved_frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_INTERVAL == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Suppress TensorFlow output during face detection
                with contextlib.redirect_stdout(io.StringIO()):
                    detections = detector.detect_faces(frame_rgb)
                
                if not detections:
                    frame_count += 1
                    continue

                largest_detection = max(detections, key=lambda d: d['box'][2] * d['box'][3])
                x, y, w, h = largest_detection['box']
                x, y = max(0, x), max(0, y)

                x1 = int(x - w * CROP_MARGIN)
                y1 = int(y - h * CROP_MARGIN)
                x2 = int(x + w + w * CROP_MARGIN)
                y2 = int(y + h + h * CROP_MARGIN)

                frame_h, frame_w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w - 1, x2), min(frame_h - 1, y2)

                if x1 >= x2 or y1 >= y2:
                    frame_count += 1
                    continue

                cropped_face = frame[y1:y2, x1:x2]
                resized_face = cv2.resize(cropped_face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                output_filename = frame_output_dir / f"frame_{saved_frame_count:04d}.png"
                cv2.imwrite(str(output_filename), resized_face)
                saved_frame_count += 1

            frame_count += 1

        cap.release()
        # Silently return frame count (progress bar shows overall progress)
        return saved_frame_count

    except Exception as e:
        print(f"Error processing video {video_path}: {e}", file=sys.stderr)
        return 0


def main():
    print("Initializing MTCNN face detector...")
    detector = MTCNN()
    print("Detector initialized.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR.resolve()}")

    # First pass: collect all videos from all datasets
    all_tasks = []  # List of tuples: (video_path, split_name, label, dataset_key)
    
    for key, root_path in DATASET_ROOTS.items():
        if not root_path.exists():
            print(f"Warning: Path not found, skipping: {root_path}", file=sys.stderr)
            continue

        if "train" in key:
            split_name = "train"
        elif "val" in key:
            split_name = "validation"
        elif "test" in key:
            split_name = key.replace("_real", "").replace("_fake", "")
        else:
            split_name = "unknown"

        label = "real" if "real" in key else "fake"
        
        # Extract manipulation type from key
        # e.g., "train_ff_fake_deepfakes" -> "deepfakes"
        # e.g., "train_ff_real" -> "original"
        # e.g., "test_celebdf_real" -> "celebdf_real"
        if "real" in key and "ff" in key:
            manipulation_type = "original"
        elif "celebdf" in key or "youtube" in key:
            # For Celeb-DF, use the last part of the key
            manipulation_type = "_".join(key.split("_")[1:])  # e.g., "celebdf_real"
        else:
            # For fake videos, extract the manipulation type
            parts = key.split("_")
            manipulation_type = parts[-1] if len(parts) > 0 else "unknown"
        
        video_files = list(root_path.glob("*.mp4")) + list(root_path.glob("*.avi"))
        
        # Apply video limit if configured
        if MAX_VIDEOS_PER_FOLDER is not None:
            video_files = video_files[:MAX_VIDEOS_PER_FOLDER]
        
        if not video_files:
            print(f"Warning: No .mp4 or .avi videos found in {root_path}", file=sys.stderr)
            continue
            
        print(f"Found {len(video_files)} videos in {key} (Split: {split_name}, Label: {label}, Type: {manipulation_type})")
        
        # Add to task list with manipulation type
        for v in video_files:
            all_tasks.append((v, split_name, label, manipulation_type, key))
    
    print(f"\nTotal videos to process: {len(all_tasks)}")
    print(f"Starting parallel processing with {MAX_WORKERS} threads...\n")
    
    # Process all videos with a single progress bar
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_video, v, OUTPUT_DIR, MTCNN(), split, lbl, manip_type): (v, key) 
            for v, split, lbl, manip_type, key in all_tasks
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Processing videos", unit="video", 
                          leave=True, position=0, colour='green'):
            _ = future.result()

    print("\n--- Preprocessing Complete ---")
    print(f"All processed data saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
