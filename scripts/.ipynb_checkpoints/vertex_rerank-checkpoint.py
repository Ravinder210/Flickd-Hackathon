import os
import json
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# --- Configuration ---
# Your Google Cloud Project ID and Location
PROJECT_ID = "mnm10807"  # <--- REPLACE with your Project ID
LOCATION = "us-central1"           # <--- REPLACE with your region

# Path to the JSON file from the first script
INPUT_JSON_FILE = "../outputs/2025-05-28_13-40-09_UTC.json"
# --------------------

def load_data_assets():
    """Loads the catalog dataframe."""
    DATA_DIR = '../data/'
    assets = {}
    assets['df_catalog'] = pd.read_csv(os.path.join(DATA_DIR, 'catalog_full.csv')).set_index('id')
    return assets

def get_product_details_for_prompt(candidate_products: list, assets: dict):
    """Formats the candidate product details into a string for the LLM prompt."""
    prompt_text = ""
    for i, product in enumerate(candidate_products):
        try:
            # We need to handle both int and str IDs from the JSON
            product_id = int(product['matched_product_id'])
            info = assets['df_catalog'].loc[product_id]
            if isinstance(info, pd.DataFrame): info = info.iloc[0]
            
            prompt_text += f"\nCandidate {i+1}:\n"
            prompt_text += f"- Product ID: {product_id}\n"
            prompt_text += f"- Title: {info.get('title', '')}\n"
            prompt_text += f"- Description: {info.get('description', '')}\n"
        except (KeyError, ValueError):
            continue
    return prompt_text

def main():
    if not os.path.exists(INPUT_JSON_FILE):
        print("Error: Input JSON file not found.")
        return
        
    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    # Load the Gemini Pro model
    # We don't need the vision model here, since we are sending a text-only prompt
    model = GenerativeModel(model_name="gemini-2.0-flash-lite-001")
    
    assets = load_data_assets()
    
    with open(INPUT_JSON_FILE, 'r') as f:
        data = json.load(f)
    
    video_id = data['video_id']
    candidate_products = data['products']
    
    print(f"--- Starting Vertex AI Re-ranking for video: {video_id} ---")
    
    candidate_details = get_product_details_for_prompt(candidate_products, assets)
    
    # --- Create the Prompt ---
    # This prompt is carefully engineered for this task
    prompt = """
You are a world-class fashion expert. Your task is to act as a final judge to find the single best product match for an item of clothing.

I have already done a visual search and narrowed it down to these top candidates. Your job is to analyze their detailed text descriptions and select the one that is the most likely correct match.

The first candidate in the list has the highest combined visual and text score from my previous analysis, but it may not be the best one. Use your reasoning to make the final choice.

Please respond with ONLY the numeric Product ID of the single best candidate. Do not provide any other text, explanation, or justification. Just the number.

### Catalog Candidates:
{candidate_details}

### Final Judgement (Product ID only):"""

    # Format the prompt with our candidate details
    final_prompt = prompt.format(candidate_details=candidate_details)
    
    print("Sending request to Vertex AI Gemini model...")
    
    # Send the request to the API
    response = model.generate_content(final_prompt)
    
    try:
        result_text = response.text
        # Extract the first number found in the LLM's response
        best_id_str = ''.join(filter(str.isdigit, result_text))
        
        if best_id_str:
            final_product_id = int(best_id_str)
            print("\n--- FINAL RESULT ---")
            print(f"✅ The LLM has judged that the best match is Product ID: {final_product_id}")
            
            # Find the full data for the winning product
            winning_product = next((p for p in candidate_products if int(p['matched_product_id']) == final_product_id), None)
            
            if winning_product:
                print("\nDetails of the winning product:")
                print(json.dumps(winning_product, indent=4))
            else:
                print(f"Warning: The winning product ID {final_product_id} was not in the original candidate list.")
        else:
            print("LLM did not return a valid numeric ID.")
            print(f"Raw response: {result_text}")

    except Exception as e:
        print(f"An error occurred while processing the Vertex AI response: {e}")
        print("Full response object:", response)


if __name__ == '__main__':
    main()