# scripts/extract_frames.py (With "Body Shot" Filter)

import cv2
import os
from tqdm import tqdm
from ultralytics import YOLO

# --- Configuration ---
VIDEO_FILENAME = "2025-05-27_13-46-16_UTC.mp4" 
FRAMES_PER_SECOND = 5 # Scan at a higher rate so we don't miss good frames
BODY_SHOT_AREA_THRESHOLD = 0.20 # A person must cover at least 20% of the frame area
# --------------------


def main():
    """
    Extracts only "body shot" frames from a video where a person is clearly visible.
    """
    print("--- Starting Body Shot Frame Extraction ---")
    
    # --- Setup ---
    VIDEO_DIR = '../videos/'
    OUTPUT_DIR_BASE = '../extracted_frames/'
    video_path = os.path.join(VIDEO_DIR, VIDEO_FILENAME)
    
    video_name_without_ext = os.path.splitext(VIDEO_FILENAME)[0]
    output_dir = os.path.join(OUTPUT_DIR_BASE, f"{video_name_without_ext}_body_shots")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print("\nLoading YOLOv8n model for person detection...")
    # This will download the model automatically on the first run
    model = YOLO("yolov8n.pt")
    print("Model loaded.")

    # --- Frame Extraction and Filtering Logic ---
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps / FRAMES_PER_SECOND) or 1

    print(f"\nScanning video at {FRAMES_PER_SECOND} FPS...")
    pbar = tqdm(total=total_frames, desc="Scanning for body shots")

    frame_count = 0
    saved_frame_count = 0
    while True:
        success, frame_bgr = vidcap.read()
        if not success: break
        
        pbar.update(1)

        if frame_count % frame_interval == 0:
            # --- THE NEW LOGIC IS HERE ---
            frame_height, frame_width, _ = frame_bgr.shape
            frame_area = frame_height * frame_width
            
            # Run YOLO to find people (class 0)
            results = model(frame_bgr, classes=[0], verbose=False)
            
            # Check if any detected person is large enough
            is_body_shot = False
            best_person_box = None
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    b_width = box.xywh[0][2]
                    b_height = box.xywh[0][3]
                    box_area = b_width * b_height
                    
                    if (box_area / frame_area) > BODY_SHOT_AREA_THRESHOLD:
                        is_body_shot = True
                        best_person_box = box # Keep the box to draw it later
                        break 
            
            if is_body_shot:
                # Draw a green box on the frame to show which person triggered the save
                x1, y1, x2, y2 = map(int, best_person_box.xyxy[0])
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Save the annotated frame
                frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(frame_filename, frame_bgr)
                saved_frame_count += 1
        
        frame_count += 1
    
    pbar.close()
    vidcap.release()

    print(f"\n✅ Done! Found and saved {saved_frame_count} 'body shot' frames to the folder '{output_dir}'.")

if __name__ == '__main__':
    main()