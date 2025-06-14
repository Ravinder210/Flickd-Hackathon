from transformers import pipeline
import os

# This prevents the OMP error just in case


# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
MODEL_NAME = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

print(f"--- Attempting to download and cache the model: {MODEL_NAME} ---")
print("This may take a while depending on your network connection. Please be patient.")

try:
    # This one line will trigger the download and save it to the cache
    pipeline("zero-shot-classification", model=MODEL_NAME)
    print("\n\n✅ Success! Model has been downloaded and cached successfully.")
    print("You can now run the main 'run_pipeline.py' script.")

except Exception as e:
    print(f"\n\n❌ An error occurred during download: {e}")
    print("Please check your internet connection and try running this script again.")
