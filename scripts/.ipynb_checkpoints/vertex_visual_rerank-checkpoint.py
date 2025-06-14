import os
import json
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image as VertexImage

# --- Configuration ---
PROJECT_ID = "mnm10807"  # <--- REPLACE with your Project ID
LOCATION = "us-central1"           # <--- REPLACE with your region

# Path to the JSON file from the first script
INPUT_JSON_FILE = "../outputs/2025-05-28_13-42-32_UTC.json"

# Path to the corresponding original video file
VIDEO_FILE = "../videos/2025-05-28_13-42-32_UTC.mp4"
# --------------------

# This is a helper function to find the best frame, no changes needed here.
def get_best_frame_from_video(video_path: str):
    """
    Scans a video to find the single best, clearest frame of a FASHION ITEM,
    using our specialized 'best.pt' model.
    """
    print("Finding the best reference frame using the expert fashion model...")
    from ultralytics import YOLO # Import locally
    
    # --- THIS IS THE KEY CHANGE ---
    # Load our expert fashion model instead of the generic one.
    yolo_fashion_model = YOLO("../models/best.pt")
    # ------------------------------

    vidcap = cv2.VideoCapture(video_path)
    
    best_frame_data = None
    max_box_area = 0
    
    pbar = tqdm(total=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Scanning for best fashion shot")
    frame_count = 0
    # Scan at a reasonable rate
    frame_interval = int(vidcap.get(cv2.CAP_PROP_FPS) or 30) // 4 # Check ~4fps
    if frame_interval == 0: frame_interval = 1

    while True:
        success, frame_bgr = vidcap.read()
        if not success: break
        pbar.update(1)
        
        if frame_count % frame_interval == 0:
            # We no longer need to filter for 'person' (class 0)
            # The fashion model will find dresses, tops, etc. directly.
            results = yolo_fashion_model(frame_bgr, verbose=False)
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    box_area = (box.xywh[0][2] * box.xywh[0][3]).item()
                    # Find the frame that contains the largest detected FASHION ITEM
                    if box_area > max_box_area:
                        max_box_area = box_area
                        # Get the coordinates to crop the item
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Crop the original BGR frame to the fashion item
                        cropped_bgr = frame_bgr[y1:y2, x1:x2]
                        best_frame_data = cropped_bgr
        frame_count += 1
    
    vidcap.release()
    pbar.close()

    if best_frame_data is not None:
        print("Best reference frame found and cropped to the item.")
        # Convert the final BGR numpy array to a Vertex AI Image object
        success, encoded_image = cv2.imencode('.png', best_frame_data)
        if success:
            return VertexImage.from_bytes(encoded_image.tobytes())
    return None

    if best_frame_data is not None:
        print("Best reference frame found and cropped.")
        # Convert the final BGR numpy array to a Vertex AI Image object
        # This requires encoding the image to a byte stream (like PNG)
        success, encoded_image = cv2.imencode('.png', best_frame_data)
        if success:
            return VertexImage.from_bytes(encoded_image.tobytes())
    return None

def get_product_details_for_prompt(candidate_products: list, df_catalog: pd.DataFrame):
     # This function is the same as before
    prompt_text = ""
    for i, product in enumerate(candidate_products):
        try:
            product_id = int(product['matched_product_id'])
            info = df_catalog.loc[product_id]
            if isinstance(info, pd.DataFrame): info = info.iloc[0]
            prompt_text += f"\nCandidate {i+1}:\n- Product ID: {product_id}\n- Title: {info.get('title', '')}\n- Description: {info.get('description', '')}\n"
        except (KeyError, ValueError): continue
    return prompt_text

def main():
    if not os.path.exists(INPUT_JSON_FILE) or not os.path.exists(VIDEO_FILE):
        print("Error: Input JSON or Video file not found."); return
        
    # --- STEP 1: Initialization and Data Loading ---
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    # Load the multimodal model
    model = GenerativeModel(model_name="gemini-2.0-flash-lite-001")
    df_catalog = pd.read_csv('../data/catalog_full.csv').set_index('id')
    with open(INPUT_JSON_FILE, 'r') as f: data = json.load(f)
    print(f"--- Starting Vertex AI Visual Re-ranking for video: {data['video_id']} ---")

    # --- STEP 2: PREPARE THE PROMPT INPUTS ---
    # Get the best frame from the video as an image object
    image_for_prompt = get_best_frame_from_video(VIDEO_FILE)
    if not image_for_prompt:
        print("Could not find a suitable frame. Aborting."); return
    
    # Get the text descriptions of the candidates
    candidate_details = get_product_details_for_prompt(data['products'], df_catalog)

    # The prompt text is now simpler
    text_prompt = f"""
You are a world-class fashion expert. Your task is to look at the attached image of an outfit and find the best match from a list of candidates.

Analyze the image carefully. Then, review the text descriptions of the catalog candidates.

Choose the single best match. Respond with ONLY the numeric Product ID of your choice. No other text or explanation.

### Catalog Candidates:
{candidate_details}

### Final Judgement (Product ID only):"""
    
    # --- STEP 3: SEND THE MULTIMODAL REQUEST ---
    # We combine the image and the text into one prompt
    prompt_parts = [
        image_for_prompt,
        text_prompt,
    ]
    
    print("\nSending multimodal request to Vertex AI Gemini Pro Vision...")
    response = model.generate_content(prompt_parts)
    
    # --- STEP 4: PARSE AND DISPLAY THE RESULT ---
    try:
        result_text = response.text
        # Extract the first number found in the LLM's response
        best_id_str = ''.join(filter(str.isdigit, result_text.splitlines()[0]))
        
        final_product_id = None
        if best_id_str:
            final_product_id = int(best_id_str)
            print(f"LLM chose Product ID: {final_product_id}")
    
        # --- SAVE THE RE-RANKED JSON ---
        if final_product_id:
            # Create a new list with the LLM's winning product at the top
            reranked_products = sorted(data['products'], key=lambda p: int(p['matched_product_id']) != final_product_id)
        else:
            # If LLM fails for any reason, just use the original order from the candidate generator
            print("LLM re-ranking failed or returned invalid ID. Using original candidate order.")
            reranked_products = data['products']
    
        # Create the new JSON data structure for saving
        output_data = {
            "video_id": data['video_id'],
            "vibes": [], # Vibes will be added by the next script
            "products": reranked_products # The new re-ranked list
        }
        
        # Define the output path
        OUTPUTS_DIR = "../outputs_reranked/"
        if not os.path.exists(OUTPUTS_DIR):
            os.makedirs(OUTPUTS_DIR)
        
        video_id = data['video_id']
        # Create a new filename for the re-ranked output
        output_filename = os.path.join(OUTPUTS_DIR, f"{video_id}_reranked.json")
        
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        print(f"\n✅ Successfully re-ranked and saved top products to: {output_filename}")
        print("\nFinal re-ranked product list:")
        print(json.dumps(output_data, indent=2))
    
    except Exception as e:
        print(f"An error occurred while processing the Vertex AI response: {e}")
        # Still save the original list if the API call fails, so we don't lose progress
        print("Saving the original candidate list as fallback.")
        output_data = data
        OUTPUTS_DIR = "../outputs_reranked/"
        if not os.path.exists(OUTPUTS_DIR): os.makedirs(OUTPUTS_DIR)
        video_id = data['video_id']
        output_filename = os.path.join(OUTPUTS_DIR, f"{video_id}_reranked_fallback.json")
        with open(output_filename, 'w') as f: json.dump(output_data, f, indent=4)

if __name__ == '__main__':
    # We need to install ultralytics if it's not present for the helper function
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics for helper function...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO
        
    main()