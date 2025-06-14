# run_pipeline.py (STABLE BASELINE - Attribute Bonus Scoring)

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
import collections
import gc

def load_main_assets(device):
    """Loads assets for the detailed analysis stage (best.pt, Large CLIP)."""
    print("--- Loading Main Assets (Fashion Model, Large CLIP, FAISS) ---")
    MODELS_DIR = '../models/'
    DATA_DIR = '../data/'
    assets = {}
    
    assets['yolo_model'] = YOLO(os.path.join(MODELS_DIR, 'best.pt')).to(device)
    assets['clip_model'], assets['preprocess'] = clip.load("ViT-L/14", device=device)
    assets['index'] = faiss.read_index(os.path.join(MODELS_DIR, "catalog_index_large.faiss"))
    with open(os.path.join(MODELS_DIR, "product_id_map_large.json"), 'r') as f:
        assets['product_id_map'] = json.load(f)
    assets['df_catalog'] = pd.read_csv(os.path.join(DATA_DIR, 'catalog_full.csv')).set_index('id')
    
    print(f"FAISS index (large) loaded with {assets['index'].ntotal} vectors.")
    print("--- Main Assets Loaded Successfully ---")
    return assets

def process_single_frame(frame_image, assets, device):
    """Processes one frame using the Attribute-Bonus scoring approach."""
    detected_products = []
    yolo_model, clip_model, preprocess = assets['yolo_model'], assets['clip_model'], assets['preprocess']
    faiss_index, id_map, df_catalog = assets['index'], assets['product_id_map'], assets['df_catalog']

    results = yolo_model(frame_image, verbose=False)
    
    best_box = None
    max_area = 0
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            area = (box.xywh[0][2] * box.xywh[0][3]).item()
            if area > max_area:
                max_area = area
                best_box = box

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cropped_image = frame_image.crop((x1, y1, x2, y2))
        image_input = preprocess(cropped_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # Predict Attributes from the video frame image
        text_descs = {"type": ["a top", "a dress", "pants", "a skirt", "shorts", "a jacket"],
                      "color": ["white", "black", "blue", "red", "green", "pink", "purple", "orange", "yellow", "brown", "grey"]}
        
        predicted_attrs = {}
        with torch.no_grad():
            for attr, labels in text_descs.items():
                tokens = clip.tokenize(labels).to(device)
                text_feats = clip_model.encode_text(tokens)
                text_feats /= text_feats.norm(dim=-1, keepdim=True)
                sims = (image_features @ text_feats.T).squeeze()
                predicted_attrs[attr] = labels[sims.argmax().item()].replace("a ", "")

        embedding_np = image_features.cpu().numpy()
        k = 100
        distances, indices = faiss_index.search(embedding_np, k)
        
        for i in range(k):
            pid = id_map[indices[0][i]]
            dist = distances[0][i]
            visual_sim = 1 / (1 + float(dist))

            try:
                info = df_catalog.loc[pid]
                if isinstance(info, pd.DataFrame): info = info.iloc[0]
                
                final_score = visual_sim
                if predicted_attrs['color'] == str(info.get('color', 'N/A')).lower():
                    final_score *= 1.2
                if predicted_attrs['type'] in str(info.get('product_type', 'N/A')).lower():
                    final_score *= 1.2
                
                detected_products.append({'id': pid, 'final_score': final_score})
            except KeyError:
                continue
    return detected_products

def main():
    """Main execution function with the two-stage pipeline."""
    VIDEO_DIR = '../videos/'
    OUTPUTS_DIR = '../outputs/'
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    main_assets = load_main_assets(device)
    
    print("\nLoading helper model for person detection (yolov8n.pt)...")
    yolo_person_model = YOLO("yolov8n.pt").to(device)
    print("All models loaded.\n")

    for video_filename in os.listdir(VIDEO_DIR):
        if not any(video_filename.lower().endswith(ext) for ext in ['.mp4', '.mov']):
            continue
            
        print(f"--- Processing Video: {video_filename} ---")
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        print("Stage 1: Identifying 'body shot' frames...")
        shoppable_frames = []
        vidcap = cv2.VideoCapture(video_path)
        
        process_rate = 5
        fps = vidcap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = int(fps / process_rate) or 1
        
        pbar1 = tqdm(total=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Scanning for people")
        frame_count = 0
        while True:
            success, frame_bgr = vidcap.read()
            if not success: break
            pbar1.update(1)
            if frame_count % frame_interval == 0:
                frame_height, frame_width, _ = frame_bgr.shape
                frame_area = frame_height * frame_width
                person_results = yolo_person_model(frame_bgr, classes=[0], verbose=False)
                if len(person_results[0].boxes) > 0:
                    for box in person_results[0].boxes:
                        box_area = (box.xywh[0][2] * box.xywh[0][3]).item()
                        if (box_area / frame_area) > 0.20:
                            shoppable_frames.append(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
                            break
            frame_count += 1
        vidcap.release()
        pbar1.close()
        
        if not shoppable_frames:
            print("No 'body shot' frames found. Skipping video.")
            continue
        print(f"Found {len(shoppable_frames)} high-quality frames.")

        print("\nStage 2: Running detailed fashion analysis...")
        all_detections = []
        pbar2 = tqdm(total=len(shoppable_frames), desc="Analyzing products")
        for frame_image in shoppable_frames:
            frame_detections = process_single_frame(frame_image, main_assets, device)
            all_detections.extend(frame_detections)
            pbar2.update(1)
        pbar2.close()

        if not all_detections:
            final_json = {"video_id": os.path.splitext(video_filename)[0], "vibes": [], "products": []}
        else:
            grouped_by_id = collections.defaultdict(list)
            for d in all_detections:
                grouped_by_id[d['id']].append(d['final_score'])

            best_scores = [{'id': pid, 'confidence': max(scores)} for pid, scores in grouped_by_id.items()]
            best_scores.sort(key=lambda x: x['confidence'], reverse=True)
            # You can change the number of results here, e.g., to 8
            final_products = best_scores[:50]
            
            output_products = []
            for match in final_products:
                try:
                    info = main_assets['df_catalog'].loc[match['id']]
                    if isinstance(info, pd.DataFrame): info = info.iloc[0]
                    sim = match['confidence']
                    output_products.append({
                        "type": str(info.get('product_type', '')),
                        "color": str(info.get('color', '')),
                        "match_type": "exact" if sim > 1.2 else "similar",
                        "matched_product_id": str(match['id']),
                        "confidence": float(round(sim, 4))
                    })
                except KeyError: pass
            
            video_id = os.path.splitext(video_filename)[0]
            # Vibe classification is currently disabled
            final_json = {"video_id": video_id, "vibes": [], "products": output_products}

        output_path = os.path.join(OUTPUTS_DIR, f"{video_id}.json")
        with open(output_path, 'w') as f:
            json.dump(final_json, f, indent=4)
        
        print(f"\n✅ Successfully processed and saved output for {video_filename}")
        print(json.dumps(final_json, indent=2))
        print("-" * 50)
        
        gc.collect()

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()