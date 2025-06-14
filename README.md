Flickd AI Hackathon: Smart Tagging & Vibe Engine
This repository contains the full implementation for the Flickd AI Hackathon. The project is a multi-stage, multi-modal AI pipeline designed to accurately identify fashion products from short video clips, re-rank them for maximum relevance, and classify their fashion "vibe".

The final architecture evolved significantly from a simple visual search to a sophisticated, production-grade system that leverages multiple specialized AI models to overcome real-world challenges like irrelevant video frames and visual ambiguity in large product catalogs.

Live Demo
[Link to your 5-minute Loom video demo here]

Final Architecture Overview
Our final system is a three-script workflow designed for maximum accuracy and modularity. Each script performs a distinct, specialized task, feeding its output to the next stage.


Script 1: run_advanced_pipeline.py - The Candidate Generator
This script's goal is to intelligently process a raw video and produce a high-quality list of the top 8-10 most plausible product matches.

Two-Stage Frame Filtering: To handle noisy, real-world video clips (e.g., Reels), the pipeline first performs a "Body Shot" scan. It uses a fast yolov8n model to identify frames where a person's body is clearly visible (taking up >20% of the frame area), discarding irrelevant close-ups or filler scenes. This ensures that only high-quality, "shoppable" frames proceed to the next stage.
Expert Fashion Detection: On this filtered set of frames, we use our fine-tuned best.pt fashion model to accurately detect specific clothing items (dress, top, etc.).
Hybrid Scoring (Visual + Text): For each detected item, we generate a final score based on a weighted combination of two signals:
Visual Similarity (40% weight): A powerful ViT-L/14 CLIP model finds visually similar items from a 7,922-item FAISS index.
Textual Relevance (60% weight): A BLIP Image Captioning model dynamically generates a rich description of the item in the video (e.g., "a white blouse with ruffled sleeves"). This rich query is then compared against the detailed descriptions of catalog candidates using a Sentence-Transformer model.
Output: The script produces a JSON file containing a list of the top candidate products, ranked by this robust hybrid score.
Script 2: vertex_visual_rerank.py - The LLM Adjudicator
This script acts as our "final expert judge" to select the single best match from the high-quality candidate list.

Input: Takes the candidate JSON from Script 1 and the original video.
Best Evidence Selection: It scans the video with the best.pt model to find the single clearest, best-cropped image of the target fashion item.
Multimodal Prompting: It presents this best image directly to the powerful Google Gemini Pro Vision model via the Vertex AI API. The prompt includes both the image and the text descriptions of the top candidates.
Final Judgement: We ask the LLM to perform its advanced reasoning to look at the visual evidence and the text descriptions simultaneously and select the single best product_id.
Output: It saves a new JSON file with the product list re-ranked to place the LLM's choice at the very top.
Script 3: add_vibes.py - The Vibe Tagger
The final script completes the required output by adding fashion "vibes".

Input: Takes the re-ranked JSON from Script 2.
Contextual Analysis: It looks at the winning product's title, description, and product_collections to create a rich text profile.
LLM Vibe Classification: It sends this text profile to the Gemini Pro model and asks it to classify the item into 1-3 vibes from the provided vibes_list.json.
Final Output: It updates the JSON file with the vibes array, creating the final deliverable for the hackathon.
How to Run the Project
1. Setup
Prerequisites:

Python 3.10+
A Google Cloud account with the Vertex AI API enabled.
The gcloud CLI tool installed.
Installation:

# Clone the repository
git clone [your-repo-url]
cd [your-repo-name]

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all required libraries
pip install -r requirements.txt
Authentication:
Authenticate with Google Cloud to use the Vertex AI API. This will open a browser window for you to log in.

gcloud auth application-default login
2. Prepare Models & Data
Place the provided video files in the /videos directory.
Place product_data.xlsx, images.csv, and vibes_list.json in the /data directory.
Download the pre-trained best.pt model from the Kaggle link provided and place it in the /models directory.
Run the data preparation and index-building notebooks (01_... to 04b_...) or provide the final FAISS index (catalog_index_large.faiss and product_id_map_large.json) in the /models folder.
(Optional but Recommended) Run the scripts/download_model.py script to pre-cache all Hugging Face models locally.
3. Execute the Full Pipeline
The workflow consists of running the three main scripts in sequence for each video. The scripts are designed to find and process the outputs from the previous step.

Step 1: Generate Candidates
This script will process all videos in the /videos folder and create initial candidate JSON files in /outputs.

python3 scripts/run_advanced_pipeline.py
Step 2: Re-rank with Gemini Vision
This script will process all candidate files in /outputs and create re-ranked files in /outputs_reranked.

python3 scripts/vertex_visual_rerank.py
(Note: You will need to edit the script to loop through all files, or run it manually for each one by changing the INPUT_JSON_FILE variable).

Step 3: Add Vibe Information
This final script processes files in /outputs_reranked to produce the final, complete JSON files in /outputs_final.

python3 scripts/add_vibes.py
(Note: This script will also need to be modified to loop through all re-ranked files).

Conclusion
This project successfully demonstrates an advanced, multi-modal pipeline for fashion product recognition. By evolving from a simple visual search to a sophisticated system with relevance filtering and LLM-based re-ranking, we were able to achieve highly accurate and context-aware results, overcoming the challenges of real-world video data and visual ambiguity in a large catalog.