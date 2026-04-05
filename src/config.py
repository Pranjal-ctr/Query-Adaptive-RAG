"""
Centralized configuration for the Adaptive RAG Research Project.
All constants, paths, and model settings in one place.
"""

import os
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_FILE = os.path.join(DATA_DIR, "yelp_reviews_sample.jsonl")
DB_PATH = os.path.join(PROJECT_ROOT, "faiss_food_db")
SCORE_CURVES_DIR = os.path.join(DATA_DIR, "score_curves")
EVAL_DIR = os.path.join(PROJECT_ROOT, "eval")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ─── Embedding Model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding kwargs — normalize for proper cosine similarity
EMBEDDING_MODEL_KWARGS = {"device": DEVICE}
EMBEDDING_ENCODE_KWARGS = {
    "normalize_embeddings": True,
    "batch_size": 64,
}

# ─── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
# Sentence-aware separators for cleaner splitting
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

# ─── Retrieval ────────────────────────────────────────────────────────────────
K_MAX = 20          # Retrieve this many candidates for score curve analysis
K_BASELINE = 5      # Default K for baseline generation (current system)

# Intent-based K (baseline / fallback)
INTENT_K_MAP = {
    "recommendation": 8,
    "complaint": 6,
    "aspect": 5,
    "general": 4,
}

# ─── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL = "mistral"

# ─── Aspect Detection ────────────────────────────────────────────────────────
ASPECT_KEYWORDS = {
    "food": ["taste", "delicious", "bland", "fresh", "flavor", "cooked",
             "raw", "spicy", "sweet", "sour", "crispy", "tender", "dry",
             "overcooked", "undercooked", "portion", "dish", "meal"],
    "service": ["service", "staff", "waiter", "waitress", "server", "rude",
                "friendly", "attentive", "slow", "fast", "helpful", "manager"],
    "price": ["cheap", "expensive", "price", "cost", "affordable", "value",
              "overpriced", "reasonable", "budget", "worth"],
    "ambience": ["ambience", "atmosphere", "crowded", "quiet", "cozy", "noisy",
                 "clean", "dirty", "decor", "vibe", "music", "seating"],
}

# ─── Score Logging ────────────────────────────────────────────────────────────
SCORE_LOG_FILE = os.path.join(SCORE_CURVES_DIR, "retrieval_scores.csv")
CURVE_DATA_FILE = os.path.join(SCORE_CURVES_DIR, "curve_data.json")

# ─── Ensure output directories exist ─────────────────────────────────────────
for _dir in [SCORE_CURVES_DIR, EVAL_DIR, RESULTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
