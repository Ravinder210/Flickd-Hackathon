
import os
import json
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel
import gc

# We will now import the functions directly from our other scripts
# This requires making sure they are importable.
# Let's put all the core logic in this one master file for simplicity.

# ----- Copied from run_advanced_pipeline.py -----
from PIL import Image
import numpy as np
import clip
from ultralytics import YOLO
import faiss
import cv2
from tqdm import tqdm
import collections
import torch

def load_main_assets(device):
    print("--- Loading Main Assets (Fashion Model, Large CLIP, FAISS) ---")
    MODELS_DIR = '../models/'; DATA_DIR = '../data/'
    assets = {'yolo_model': YOLO(os.path.join(MODELS_DIR, 'best.pt')).to(device),
              'clip_model': clip.load("ViT-L/14", device=device)[0],
              'clip_preprocess': clip.load("ViT-L/14", device=device)[1],
              'index': faiss.read_index(os.path.join(MODELS_DIR, "catalog_index_large.faiss")),
              'product_id_map': json.load(open(os.path.join(MODELS_DIR, "product_id_map_large.json"))),
              'df_catalog': pd.read_csv(os.path.join(DATA_DIR, 'catalog_full.csv')).set_index('id')}
    print("--- Main Assets Loaded Successfully ---")
    return assets

def process_single_frame(frame_image, assets, device):
    # This function remains the same as our best version.
    # [I'm omitting the long code here for brevity, but it's the one we perfected]
    detected_products = []
    yolo_model, clip_model, clip_preprocess = assets['yolo_model'], assets['clip_model'], assets['clip_preprocess']
    faiss_index, id_map, df_catalog = assets['index'], assets['product_id_map'], assets['df_catalog']
    results = yolo_model(frame_image, verbose=False)
    best_box = None; max_area = 0
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            area = (box.xywh[0][2] * box.xywh[0][3]).item()
            if area > max_area:
                max_area = area; best_box = box
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0]); cropped_image = frame_image.crop((x1, y1, x2, y2))
        image_input_clip = clip_preprocess(cropped_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input_clip)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        embedding_np = image_features.cpu().numpy()
        k=80; distances, indices = faiss_index.search(embedding_np, k) # Increased candidates for LLM
        for i in range(k):
            detected_products.append({'id': id_map[indices[0][i]], 'visual_sim': 1 / (1 + float(distances[0][i]))})
    return detected_products
# ----- End of Copied Section -----

# ----- Copied from vertex_visual_rerank.py -----
from vertexai.generative_models import Part, Image as VertexImage

def get_best_frame_for_llm(video_path: str):
    print("Finding the best reference frame from the video...")
    yolo_fashion_model = YOLO("../models/best.pt")
    vidcap = cv2.VideoCapture(video_path); best_frame_data = None; max_box_area = 0
    pbar = tqdm(total=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Scanning for best fashion shot")
    frame_count = 0
    frame_interval = int(vidcap.get(cv2.CAP_PROP_FPS) or 30) // 2
    if frame_interval == 0: frame_interval = 1
    while True:
        success, frame_bgr = vidcap.read()
        if not success: break
        pbar.update(1)
        if frame_count % frame_interval == 0:
            results = yolo_fashion_model(frame_bgr, verbose=False)
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    box_area = (box.xywh[0][2] * box.xywh[0][3]).item()
                    if box_area > max_box_area:
                        max_box_area = box_area
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        best_frame_data = frame_bgr[y1:y2, x1:x2]
        frame_count +=1
    vidcap.release(); pbar.close()
    if best_frame_data is not None:
        print("Best reference frame found and cropped.")
        success, encoded_image = cv2.imencode('.png', best_frame_data)
        if success:
            return VertexImage.from_bytes(encoded_image.tobytes())
    return None
# ----- End of Copied Section -----

def main():
    """THE MASTER ORCHESTRATOR"""
    PROJECT_ID = "mnm10807"
    LOCATION = "us-central1"
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    assets = load_main_assets(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    yolo_person_model = YOLO("yolov8n.pt")
    vibe_model = GenerativeModel("gemini-2.0-flash-lite-001") # Corrected model
    rerank_model = GenerativeModel("gemini-2.0-flash-lite-001") # The Pro vision model
    
    VIDEO_DIR = '../videos/'; OUTPUTS_DIR = '../outputs/'
    if not os.path.exists(OUTPUTS_DIR): os.makedirs(OUTPUTS_DIR)

    df_catalog = assets['df_catalog']
    with open('../data/vibeslist.json', 'r') as f:
        vibes_list = json.load(f)

    for video_filename in os.listdir(VIDEO_DIR):
        if not any(video_filename.lower().endswith(ext) for ext in ['.mp4', '.mov']): continue
        
        print(f"\n{'='*20} PROCESSING NEW VIDEO: {video_filename} {'='*20}")
        video_path = os.path.join(VIDEO_DIR, video_filename)

        # Step 1: Candidate Generation (All in memory now)
        print("Stage 1: Finding shoppable frames and generating candidates...")
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS) or 30; frame_interval = int(fps / 5) or 1
        pbar1 = tqdm(total=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Scanning & Analyzing")
        all_detections = []; frame_count = 0
        while True:
            success, frame_bgr = vidcap.read()
            if not success: break
            pbar1.update(1)
            if frame_count % frame_interval == 0:
                h, w, _ = frame_bgr.shape; frame_area = h * w
                person_results = yolo_person_model(frame_bgr, classes=[0], verbose=False)
                if len(person_results[0].boxes) > 0:
                    for box in person_results[0].boxes:
                        if (box.xywh[0][2] * box.xywh[0][3]).item() / frame_area > 0.20:
                            all_detections.extend(process_single_frame(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), assets, torch.device("mps")))
                            break
            frame_count += 1
        vidcap.release(); pbar1.close()

        if not all_detections: continue
        
        grouped = collections.defaultdict(list)
        for d in all_detections: grouped[d['id']].append(d['visual_sim'])
        best_scores = [{'id': pid, 'confidence': max(scores)} for pid, scores in grouped.items()]
        best_scores.sort(key=lambda x: x['confidence'], reverse=True)
        top_candidates = best_scores[:8]

        # Step 2: Visual Re-ranking with LLM
        print("\nStage 2: Visually re-ranking candidates with Gemini Pro Vision...")
        best_frame_image = get_best_frame_for_llm(video_path)
        if not best_frame_image: continue

        candidate_details = ""; winner_id = None
        for i, product in enumerate(top_candidates):
            info = df_catalog.loc[product['id']]; candidate_details += f"\nCandidate {i+1}:\n- Product ID: {product['id']}\n- Title: {info.get('title', '')}\n- Description: {info.get('description', '')}\n"
        
        rerank_prompt = [best_frame_image, f"You are a fashion expert. Look at the image and find the best match from the candidates. Respond with ONLY the numeric Product ID.\n### Candidates:{candidate_details}\n### Best Match (ID only):"]
        response = rerank_model.generate_content(rerank_prompt);
        try:
            winner_id = int(''.join(filter(str.isdigit, response.text)))
            print(f"LLM chose Product ID: {winner_id}")
        except (ValueError, TypeError):
             print(f"LLM failed to re-rank. Using top visual candidate. Raw response: {response.text}")
             winner_id = top_candidates[0]['id'] if top_candidates else None
        
        if not winner_id: continue

        # Step 3: Vibe Classification for the winning product
        print(f"\nStage 3: Classifying vibe for winning product {winner_id}...")
        winner_info = df_catalog.loc[winner_id]
        product_text = f"Title: {winner_info.get('title', '')}\nDescription: {winner_info.get('description', '')}\nTags: {winner_info.get('product_tags', '')}"
        vibe_prompt = f"You are a fashion stylist. Choose 1-3 vibes from this list: {json.dumps(vibes_list)} that best fit the product described here: {product_text}. Respond with ONLY a valid JSON list of strings."
        response = vibe_model.generate_content(vibe_prompt)
        try:
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            classified_vibes = json.loads(cleaned_response)
            print(f"LLM chose vibes: {classified_vibes}")
        except Exception:
            classified_vibes = []

        # Step 4: Final Assembly
        winning_product_details = next((p for p in top_candidates if p['id'] == winner_id), top_candidates[0])
        final_output = {"video_id": os.path.splitext(video_filename)[0], "vibes": classified_vibes,
                        "products": [{"type": str(winner_info.get('product_type', '')), "color": str(winner_info.get('color', '')),
                                      "match_type": "similar", "matched_product_id": str(winner_id),
                                      "confidence": float(round(winning_product_details['confidence'], 4))}]}
        
        output_path = os.path.join(OUTPUTS_DIR, f"{os.path.splitext(video_filename)[0]}.json");
        with open(output_path, 'w') as f: json.dump(final_output, f, indent=4)
        print("\n\n--- FINAL COMPLETE OUTPUT ---"); print(json.dumps(final_output, indent=2))
        print(f"{'='*20} FINISHED {video_filename} {'='*20}")
        gc.collect()

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()
