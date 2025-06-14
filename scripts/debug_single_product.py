import torch
import pandas as pd
from PIL import Image
import numpy as np
import os
import clip
from ultralytics import YOLO
import faiss
import json
import cv2
from tqdm import tqdm

# --- Configuration ---
# Paste the video filename you want to test
VIDEO_TO_TEST = '2025-05-28_13-42-32_UTC.mp4' 

# Paste the Product ID you want to track
TARGET_PRODUCT_ID = 15339

# Paste the list of FAISS indices you found in the Inspector Notebook
TARGET_FAISS_INDICES = [1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660]
# --------------------

def main():
    """Main execution function."""
    print(f"--- STARTING DEBUG RUN FOR PRODUCT ID: {TARGET_PRODUCT_ID} ---")

    # --- Setup Paths ---
    MODELS_DIR = '../models/'
    DATA_DIR = '../data/'
    VIDEO_DIR = '../videos/'
    
    # --- Device Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Load Assets ---
    print("Loading assets...")
    yolo_model = YOLO(os.path.join(MODELS_DIR, 'best.pt')).to(device)
    # Using the large model assets
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    index = faiss.read_index(os.path.join(MODELS_DIR, "catalog_index_large.faiss"))
    df_catalog = pd.read_csv(os.path.join(DATA_DIR, 'catalog_full.csv'))
    print("Assets loaded.\n")

    if not TARGET_FAISS_INDICES:
        print(f"Error: The FAISS indices list for product {TARGET_PRODUCT_ID} is empty. Please check the ID and re-run the Inspector notebook.")
        return

    # 1. Get the embedding vectors for our target product from the FAISS index
    target_embeddings = np.array([index.reconstruct(i) for i in TARGET_FAISS_INDICES])
    
    # --- Process Video ---
    video_path = os.path.join(VIDEO_DIR, VIDEO_TO_TEST)
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    print(f"Processing video: {VIDEO_TO_TEST}")
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps) # Analyze 1 frame per second
    frame_count = 0
    
    pbar = tqdm(total=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Analyzing Frames")

    all_frame_similarities = []

    while True:
        success, frame_bgr = vidcap.read()
        if not success: break
        
        pbar.update(1)
        
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)
            
            # --- Get the embedding for the item in this video frame ---
            results = yolo_model(frame_image, verbose=False)
            best_box = None
            max_area = 0
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                    if area > max_area:
                        max_area = area
                        best_box = box

            if best_box:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cropped_image = frame_image.crop((x1, y1, x2, y2))
                
                image_input = preprocess(cropped_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    video_embedding = clip_model.encode_image(image_input)
                    video_embedding /= video_embedding.norm(dim=-1, keepdim=True)
                
                # --- 2. Calculate Similarity ---
                # Compare the video item's embedding against ALL embeddings of our target product
                # We use matrix multiplication (@) for this. The result is cosine similarity.
                similarities = video_embedding.cpu().numpy() @ target_embeddings.T
                
                # Find the highest similarity score out of all the target's angles/images
                max_similarity_for_frame = similarities.max()
                
                all_frame_similarities.append(max_similarity_for_frame)
                
        frame_count += 1
        
    vidcap.release()
    pbar.close()

    # --- 3. Report the Results ---
    print("\n\n--- DEBUG RESULTS ---")
    if all_frame_similarities:
        max_overall_similarity = max(all_frame_similarities)
        avg_similarity = sum(all_frame_similarities) / len(all_frame_similarities)
        
        print(f"Similarity scores for Product ID {TARGET_PRODUCT_ID} across all processed frames:")
        # Print scores formatted nicely
        formatted_scores = [f"{score:.4f}" for score in all_frame_similarities]
        print(formatted_scores)

        print(f"\nMaximum similarity found: {max_overall_similarity:.4f}")
        print(f"Average similarity found: {avg_similarity:.4f}")

        if max_overall_similarity > 0.75:
            print("\nConclusion: The model sees this product as a POTENTIAL match.")
        else:
            print("\nConclusion: The model DOES NOT see this product as a good match (scores are too low).")
    else:
        print("No items were detected in the video to compare against.")


if __name__ == '__main__':
    # Set the environment variable to avoid the OMP error
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()
