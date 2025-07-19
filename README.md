# 👗 Flickd AI Hackathon: Smart Tagging & Vibe Engine

Welcome to the **Flickd AI Hackathon** submission — a full-fledged, multi-modal AI pipeline built to **identify fashion products from short videos**, **rank them for relevance**, and **classify their fashion "vibe"**.

> ⚡ What started as a simple visual search evolved into a production-grade system tackling challenges like irrelevant frames, visual ambiguity, and large catalog search — using CLIP, YOLO, BLIP, Sentence Transformers, and Google Gemini.

---

## 🏗️ Final Architecture Overview

Our final system is modular, consisting of **three specialized scripts**, each performing a core function:

```
🎥 Video → 🧠 Frame Filtering → 🧥 Product Detection → 🔍 Candidate Scoring → 🤖 LLM Reranking → ✨ Vibe Tagging
```

---

## 📜 Scripts Breakdown

### 1️⃣ `run_advanced_pipeline.py` — **Candidate Generator**

Produces a high-quality shortlist (top 8–10) of fashion products per video.

* 🧠 **Two-Stage Frame Filtering**

  * **YOLOv8n** filters for "body shot" frames (body ≥ 20% frame area).
* 👗 **Fashion Item Detection**

  * Uses custom-trained `best.pt` model.
* 🔗 **Hybrid Scoring: Visual + Text**

  * **40% Visual Similarity**: CLIP (ViT-L/14) via FAISS index.
  * **60% Textual Match**: BLIP captioning + Sentence Transformers.

📦 **Output**: Ranked `candidates.json` per video in `/outputs`.

---

### 2️⃣ `vertex_visual_rerank.py` — **LLM Adjudicator**

Re-ranks candidates to find the **best product match**.

* 🖼️ **Best Frame Extraction**

  * Uses `best.pt` to pick the clearest single frame.
* 💡 **Multimodal Prompting**

  * Passes frame + candidates to **Gemini Pro Vision**.
* 🏆 **Final Judgment**

  * Gemini selects the best `product_id`.

📦 **Output**: Re-ranked `candidates_reranked.json` in `/outputs_reranked`.

---

### 3️⃣ `add_vibes.py` — **Vibe Classifier**

Adds fashion “vibes” to each product.

* 📚 **Contextual Understanding**

  * Title, description & collections are merged into a rich text profile.
* 🎨 **LLM-Based Vibe Tagging**

  * Gemini classifies into 1–3 vibes from `vibes_list.json`.

📦 **Output**: Final enriched JSONs in `/outputs_final`.

---

## 🚀 How to Run the Project

### 🔧 1. Setup

#### ✅ Prerequisites

* Python 3.10+
* Google Cloud account with Vertex AI enabled
* `gcloud` CLI installed

#### 📦 Installation

```bash
git clone https://github.com/Ravinder210/Flickd-Hackathon
cd Flickd-Hackathon

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

#### 🔐 Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

---

### 📁 2. Prepare Models & Data

* Drop **videos** into: `/videos`
* Place:

  * `product_data.xlsx`
  * `images.csv`
  * `vibes_list.json`
    into: `/data`
* Download `best.pt` from the provided Kaggle link → place in `/models`
* Run notebooks `01_...` to `04b_...` or place:

  * `catalog_index_large.faiss`
  * `product_id_map_large.json`
    into `/models`
* *(Optional)* Run:

```bash
python scripts/download_model.py
```

To cache all Hugging Face models locally.

---

### ▶️ 3. Execute the Full Pipeline

#### 🌀 Step 1: Generate Candidates

```bash
python3 scripts/run_advanced_pipeline.py
```

Creates JSON outputs in `/outputs`.

---

#### 🔁 Step 2: Re-rank Using Gemini

```bash
python3 scripts/vertex_visual_rerank.py
```

> ⚠️ Script currently processes one file.
> Edit `INPUT_JSON_FILE` or loop through `/outputs`.

---

#### 🌈 Step 3: Add Vibe Information

```bash
python3 scripts/add_vibes.py
```

> ⚠️ Also designed for single-file processing.
> Modify to loop through `/outputs_reranked`.

---

## 🏁 Final Thoughts

This project demonstrates a **cutting-edge, multi-stage AI pipeline** for fashion video understanding — merging:

* 🖼️ **Computer Vision**
* 🧾 **Captioning**
* 🔍 **Semantic Search**
* 💬 **Multimodal LLM Reasoning**

By combining smart filtering, hybrid matching, and LLM judgement, we accurately detect fashion products *and* assign them an expressive, style-aware **vibe**.

---

## 💡 Technologies Used

* **YOLOv8n** — Frame Filtering
* **CLIP (ViT-L/14)** — Visual Embeddings
* **BLIP** — Image Captioning
* **Sentence Transformers** — Text Similarity
* **Gemini Pro Vision (Vertex AI)** — Multimodal Reasoning
* **FAISS** — Fast Product Retrieval

---

## 📂 Project Structure

```
📁 /scripts
 ├─ run_advanced_pipeline.py
 ├─ vertex_visual_rerank.py
 └─ add_vibes.py
📁 /models
📁 /data
📁 /videos
📁 /outputs
📁 /outputs_reranked
📁 /outputs_final
```

---

## 🧠 Team & Credits

Made with ❤️ for the Flickd AI Hackathon
By [Ravinder](https://github.com/Ravinder210)


