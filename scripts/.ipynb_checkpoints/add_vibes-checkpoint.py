import os
import json
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel

# --- Configuration ---
PROJECT_ID = "mnm10807"  # <--- REPLACE with your Project ID
LOCATION = "us-central1"           # <--- REPLACE with your region

# Path to the RE-RANKED JSON file
INPUT_JSON_FILE = "../outputs_reranked/2025-05-28_13-42-32_UTC_reranked.json"
# --------------------


def main():
    if not os.path.exists(INPUT_JSON_FILE):
        print("Error: Re-ranked JSON file not found."); return

    # --- Initialization and Loading ---
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    # Use the text-only model for this task, it's faster and cheaper
    model = GenerativeModel(model_name="gemini-2.0-flash-lite-001")
    
    df_catalog = pd.read_csv('../data/catalog_full.csv').set_index('id')
    
    with open('../data/vibeslist.json', 'r') as f:
        candidate_vibes = json.load(f)
    
    with open(INPUT_JSON_FILE, 'r') as f:
        data = json.load(f)

    # --- Get Info for Top Product ---
    if not data['products']:
        print("No products found in the JSON file. Cannot determine vibe.")
        return
        
    top_product_id = int(data['products'][0]['matched_product_id'])
    
    try:
        info = df_catalog.loc[top_product_id]
        if isinstance(info, pd.DataFrame): info = info.iloc[0]
        
        # Create a rich text document from the winning product's data
        product_text = f"Title: {info.get('title', '')}\nDescription: {info.get('description', '')}\nCollections: {info.get('product_collections', '')}"
    except (KeyError, ValueError):
        print(f"Could not find details for winning product ID {top_product_id}.")
        return

    # --- Create the Vibe Classification Prompt ---
    prompt = f"""
You are a fashion expert and trend analyst. Based on the following product information, classify its style into 1 to 3 "vibes" from the provided list.

### Product Information:
{product_text}

### Vibe List:
{', '.join(candidate_vibes)}

Respond with a comma-separated list of the 1 to 3 best-fitting vibes from the list. For example: "Coquette, Y2K"
"""

    print(f"--- Classifying Vibe for Product ID: {top_product_id} ---")
    response = model.generate_content(prompt)
    
    try:
        # Clean up the LLM's response
        result_text = response.text.strip()
        # Split by comma and strip whitespace from each vibe
        detected_vibes = [vibe.strip() for vibe in result_text.split(',') if vibe.strip() in candidate_vibes]
        
        print(f"Detected Vibes: {detected_vibes}")
        
        # --- THIS IS THE CORRECTED SAVING LOGIC ---
        data['vibes'] = detected_vibes
        
        # 1. Define the final output directory and create it if it doesn't exist
        FINAL_OUTPUT_DIR = "../outputs_final/"
        if not os.path.exists(FINAL_OUTPUT_DIR):
            os.makedirs(FINAL_OUTPUT_DIR)
            
        # 2. Get the base name of the input file
        base_filename = os.path.basename(INPUT_JSON_FILE)
        # 3. Create the final filename by removing '_reranked'
        final_filename = base_filename.replace('_reranked', '')
        # 4. Join the directory and the final filename to get the full path
        output_path = os.path.join(FINAL_OUTPUT_DIR, final_filename)
        # --- END OF CORRECTION ---
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"\n✅ Successfully added vibes. Final complete output saved to: {output_path}")
        print(json.dumps(data, indent=2))
        
    except Exception as e:
        print(f"An error occurred while processing vibes: {e}")
        print("Raw response:", response.text)


if __name__ == '__main__':
    main()