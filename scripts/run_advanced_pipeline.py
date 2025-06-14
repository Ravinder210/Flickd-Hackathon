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
from sentence_transformers import SentenceTransformer, util
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_main_assets(device):
    """Loads all models and data assets for the full pipeline."""
    print("--- Loading Main Assets ---")
    MODELS_DIR = '../models/'
    DATA_DIR = '../data/'
    assets = {}
    
    # Computer Vision Models
    assets['yolo_model'] = YOLO(os.path.join(MODELS_DIR, 'best.pt')).to(device)
    assets['clip_model'], assets['clip_preprocess'] = clip.load("ViT-L/14", device=device)
    
    # Text and Image-to-Text Models
    print("Loading Sentence Transformer model...")
    assets['text_model'] = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print("Loading Image Captioning model (BLIP)...")
    # Load BLIP to CPU to save GPU memory; it's fast enough for a few captions.
    blip_device = "cpu" if not torch.cuda.is_available() else device
    assets['caption_processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    assets['caption_model'] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(blip_device)
    
    # Data assets
    assets['index'] = faiss.read_index(os.path.join(MODELS_DIR, "catalog_index_large.faiss"))
    with open(os.path.join(MODELS_DIR, "product_id_map_large.json"), 'r') as f:
        assets['product_id_map'] = json.load(f)
    assets['df_catalog'] = pd.read_csv(os.path.join(DATA_DIR, 'catalog_full.csv')).set_index('id')
    
    print(f"FAISS index (large) loaded with {assets['index'].ntotal} vectors.")
    print("--- Main Assets Loaded Successfully ---")
    return assets

def process_single_frame(frame_image, assets, device):
    """Processes one frame using the full advanced pipeline."""
    detected_products = []
    
    yolo_model, clip_model, clip_preprocess = assets['yolo_model'], assets['clip_model'], assets['clip_preprocess']
    faiss_index, id_map, df_catalog = assets['index'], assets['product_id_map'], assets['df_catalog']
    text_model, caption_processor, caption_model = assets['text_model'], assets['caption_processor'], assets['caption_model']
    blip_device = caption_model.device

    results = yolo_model(frame_image, verbose=False)
    
    best_box = None; max_area = 0
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            area = (box.xywh[0][2] * box.xywh[0][3]).item()
            if area > max_area:
                max_area = area; best_box = box

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cropped_image = frame_image.crop((x1, y1, x2, y2))
        
        # --- A. CANDIDATE GENERATION ---
        image_input_clip = clip_preprocess(cropped_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input_clip)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        embedding_np = image_features.cpu().numpy()
        k=30; distances, indices = faiss_index.search(embedding_np, k)
        
        visual_candidates, candidate_ids, candidate_docs = [], [], []
        for i in range(k):
            pid = id_map[indices[0][i]]; visual_sim = 1 / (1 + float(distances[0][i]))
            if visual_sim > 0.70:
                try:
                    info = df_catalog.loc[pid]
                    if isinstance(info, pd.DataFrame): info = info.iloc[0]
                    doc = f"{info.get('title', '')}. {info.get('description', '')}. Tags: {info.get('product_tags', '')}"
                    candidate_docs.append(doc); candidate_ids.append(pid); visual_candidates.append(visual_sim)
                except KeyError: continue

        if not candidate_docs: return []

        # --- B. RICH QUERY GENERATION ---
        blip_inputs = caption_processor(cropped_image, return_tensors="pt").to(blip_device)
        with torch.no_grad():
            out = caption_model.generate(**blip_inputs, max_new_tokens=25)
        query_doc = caption_processor.decode(out[0], skip_special_tokens=True)
        
        # --- C. TEXT RE-RANKING ---
        query_embedding = text_model.encode(query_doc, convert_to_tensor=True)
        candidate_embeddings = text_model.encode(candidate_docs, convert_to_tensor=True)
        text_similarities = util.pytorch_cos_sim(query_embedding, candidate_embeddings)[0]

        # --- D. HYBRID SCORING ---
        weight_visual = 0.4; weight_text = 0.6
        for i in range(len(candidate_ids)):
            visual_score = visual_candidates[i]; text_score = text_similarities[i].item()
            final_score = (visual_score * weight_visual) + (text_score * weight_text)
            detected_products.append({'id': candidate_ids[i], 'final_score': final_score})

    return detected_products

# The MAIN function remains the same as the last version I provided, as it is already
# correctly structured to use the two-stage approach and call the advanced process_single_frame.
# I am including it here for completeness.

def main():
    """Main execution function with the two-stage pipeline."""
    VIDEO_DIR = '../videos/'; OUTPUTS_DIR = '../outputs/'
    if not os.path.exists(OUTPUTS_DIR): os.makedirs(OUTPUTS_DIR)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    main_assets = load_main_assets(device)
    yolo_person_model = YOLO("yolov8n.pt").to(device)
    print("All models loaded.\n")

    for video_filename in os.listdir(VIDEO_DIR):
        if not any(video_filename.lower().endswith(ext) for ext in ['.mp4', '.mov']): continue
        
        print(f"--- Processing Video: {video_filename} ---")
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        print("Stage 1: Identifying 'body shot' frames...")
        shoppable_frames = []
        vidcap = cv2.VideoCapture(video_path); fps = vidcap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = int(fps / 5) or 1
        pbar1 = tqdm(total=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Scanning for people")
        frame_count = 0
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
                            shoppable_frames.append(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
                            break
            frame_count += 1
        vidcap.release(); pbar1.close()
        
        if not shoppable_frames:
            print("No 'body shot' frames found. Skipping video."); continue
        print(f"Found {len(shoppable_frames)} high-quality frames.")

        print("\nStage 2: Running detailed fashion analysis...")
        all_detections = []
        pbar2 = tqdm(total=len(shoppable_frames), desc="Analyzing products")
        for frame_image in shoppable_frames:
            all_detections.extend(process_single_frame(frame_image, main_assets, device))
            pbar2.update(1)
        pbar2.close()

        if not all_detections:
            final_json = {"video_id": os.path.splitext(video_filename)[0], "vibes": [], "products": []}
        else:
            grouped = collections.defaultdict(list)
            for d in all_detections: grouped[d['id']].append(d['final_score'])
            best_scores = [{'id': pid, 'confidence': max(scores)} for pid, scores in grouped.items()]
            best_scores.sort(key=lambda x: x['confidence'], reverse=True)
            final_products = best_scores[:8]
            
            output_products = []
            for match in final_products:
                try:
                    info = main_assets['df_catalog'].loc[match['id']]
                    if isinstance(info, pd.DataFrame): info = info.iloc[0]
                    output_products.append({
                        "type": str(info.get('product_type', '')), "color": str(info.get('color', '')),
                        "match_type": "similar", "matched_product_id": str(match['id']),
                        "confidence": float(round(match['confidence'], 4))
                    })
                except KeyError: pass
            video_id = os.path.splitext(video_filename)[0]
            final_json = {"video_id": video_id, "vibes": [], "products": output_products}

        output_path = os.path.join(OUTPUTS_DIR, f"{video_id}.json");
        with open(output_path, 'w') as f: json.dump(final_json, f, indent=4)
        print(f"\n✅ Successfully processed and saved output for {video_filename}")
        print(json.dumps(final_json, indent=2)); print("-" * 50)
        gc.collect()

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'; main()