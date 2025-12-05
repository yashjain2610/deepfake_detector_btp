import os
import logging

# --- Disable TensorFlow and MTCNN verbosity ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


import cv2
from tqdm import tqdm
from pathlib import Path
from mtcnn.mtcnn import MTCNN
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION: YOU MUST EDIT THIS SECTION ---

IMG_SIZE = 299
CROP_MARGIN = 0.2
FRAME_INTERVAL = 20
OUTPUT_DIR = Path("./processed_data")

DATASET_ROOTS = {
    #"train_ff_real": Path(r"C:\Users\yash jain\Desktop\folders\btp\FaceForensics++_C23\original"),
    #"train_ff_fake_deepfakes": Path(r"C:\Users\yash jain\Desktop\folders\btp\FaceForensics++_C23\Deepfakes")
    "train_ff_fake_face2face": Path(r"C:\Users\yash jain\Desktop\folders\btp\FaceForensics++_C23\Face2Face")
}

# Number of threads to use
MAX_WORKERS = os.cpu_count()  # Adjust based on your CPU cores
print(f"Using {MAX_WORKERS} threads")

# --- END OF CONFIGURATION ---


def process_video(video_path, output_dir, detector, split_name, label):
    """
    Reads a video, detects faces, crops, and saves them.
    """
    video_name = video_path.stem
    frame_output_dir = output_dir / "train" / "face2face" / f"{video_name}_frames"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            tqdm.write(f"Error: Could not open video file {video_path}", file=sys.stderr)
            return 0

        frame_count = 0
        saved_frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_INTERVAL == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        tqdm.write(f"Processed {video_path.name}: {saved_frame_count} frames saved.")
        return saved_frame_count

    except Exception as e:
        tqdm.write(f"Error processing video {video_path}: {e}", file=sys.stderr)
        return 0


def main():
    print("Initializing MTCNN face detector...")
    detector = MTCNN()
    print("Detector initialized.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR.resolve()}")

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

        print(f"\nProcessing {key} (Split: {split_name}, Label: {label})...")
        print(f"Source: {root_path}")

        video_files = list(root_path.glob("*.mp4")) + list(root_path.glob("*.avi"))
        video_files = video_files[801:1000]
        if not video_files:
            print(f"Warning: No .mp4 or .avi videos found in {root_path}", file=sys.stderr)
            continue

        print(f"Found {len(video_files)} videos.")
        
        # --- Parallel Execution ---
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_video, v, OUTPUT_DIR, MTCNN(), split_name, label): v for v in video_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {key}", unit="video"):
                _ = future.result()  # just to trigger exceptions if any

    print("\n--- Preprocessing Complete ---")
    print(f"All processed data saved in: {OUTPUT_DIR.resolve()}")
    print("You can now proceed to model training.")


if __name__ == "__main__":
    main()