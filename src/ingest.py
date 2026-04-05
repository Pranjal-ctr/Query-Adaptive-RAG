"""
Ingestion Pipeline — Phase 0 Baseline
Loads Yelp reviews, chunks them with sentence-aware splitting (512/100),
embeds with BGE-base-en-v1.5 (normalized), and stores in FAISS.

Usage:
    cd d:/Programming/AI/RAG/food_domain_rag
    python src/ingest.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
from textblob import TextBlob

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import (
    DATA_FILE, DB_PATH, EMBEDDING_MODEL,
    EMBEDDING_MODEL_KWARGS, EMBEDDING_ENCODE_KWARGS,
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS,
    ASPECT_KEYWORDS,
)


def detect_aspects(text: str) -> list:
    """Detect all matching aspects in a review (returns list, not single)."""
    text_lower = text.lower()
    found = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(aspect)
                break  # one match per aspect is enough
    return found if found else ["general"]


def main():
    print("=" * 60)
    print("  PHASE 0 — Ingestion Pipeline")
    print(f"  Model:      {EMBEDDING_MODEL}")
    print(f"  Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    print(f"  Data:       {DATA_FILE}")
    print("=" * 60)

    # ── Load reviews ──
    documents = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading reviews")):
            text = json.loads(line)["text"]
            sentiment = TextBlob(text).sentiment.polarity
            aspects = detect_aspects(text)

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "sentiment": round(sentiment, 4),
                        "aspects": aspects,
                        "review_length": len(text),
                        "review_index": line_num,
                    }
                )
            )

    print(f"\n  Loaded {len(documents)} reviews")

    # ── Sentence-aware chunking ──
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"  Split into {len(chunks)} chunks")

    # ── Embed with BGE (normalized) ──
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=EMBEDDING_ENCODE_KWARGS,
    )

    print("  Embedding chunks (this may take a few minutes)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_PATH)

    print(f"\n  ✓ FAISS DB saved to {DB_PATH}")
    print(f"  ✓ {len(chunks)} chunks indexed")
    print("=" * 60)


if __name__ == "__main__":
    main()
    
