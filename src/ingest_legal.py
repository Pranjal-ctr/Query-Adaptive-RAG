"""
Legal Domain Ingestion — Task 1
Loads Indian legal statute chunks from data/legal/chunks.jsonl,
embeds with BGE-base-en-v1.5 (same as Yelp), and stores in a
SEPARATE FAISS index at data/legal/faiss_index/.

Usage:
    cd d:/Programming/AI/RAG/food_domain_rag
    python src/ingest_legal.py
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import (
    EMBEDDING_MODEL, EMBEDDING_MODEL_KWARGS, EMBEDDING_ENCODE_KWARGS,
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS, PROJECT_ROOT,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
LEGAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "legal")
LEGAL_CHUNKS_FILE = os.path.join(LEGAL_DATA_DIR, "chunks.jsonl")
LEGAL_DB_PATH = os.path.join(LEGAL_DATA_DIR, "faiss_index")
LEGAL_METADATA_FILE = os.path.join(LEGAL_DATA_DIR, "chunks_metadata.json")


def load_legal_chunks(filepath: str) -> list:
    """
    Load legal chunks from JSONL.
    Each record has flat fields: id, doc_id, source, type, section_no, title, text.
    """
    documents = []
    skipped = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading legal chunks"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            text = record.get("text", "").strip()
            if not text:
                skipped += 1
                continue

            # Build metadata from flat fields
            metadata = {
                "chunk_id": record.get("id", ""),
                "doc_id": record.get("doc_id", ""),
                "source": record.get("source", ""),
                "type": record.get("type", "statute"),
                "section_no": record.get("section_no", ""),
                "title": record.get("title", ""),
                "text_length": len(text),
                "domain": "legal",
            }

            documents.append(Document(page_content=text, metadata=metadata))

    if skipped > 0:
        print(f"  Skipped {skipped} empty/malformed records")

    return documents


def main():
    parser = argparse.ArgumentParser(description="Legal Domain Ingestion (Task 1)")
    parser.add_argument("--input", type=str, default=LEGAL_CHUNKS_FILE,
                        help="Path to legal chunks JSONL")
    parser.add_argument("--output", type=str, default=LEGAL_DB_PATH,
                        help="Path for FAISS index output")
    args = parser.parse_args()

    print("=" * 60)
    print("  TASK 1 — Legal Domain Ingestion")
    print(f"  Model:      {EMBEDDING_MODEL}")
    print(f"  Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    print(f"  Data:       {args.input}")
    print(f"  Index:      {args.output}")
    print("=" * 60)

    # ── Step 1: Load raw chunks ──
    print("\n  Step 1: Loading legal chunks...")
    documents = load_legal_chunks(args.input)
    print(f"  Loaded {len(documents)} documents")

    if not documents:
        print("  ERROR: No documents loaded. Check input file.")
        sys.exit(1)

    # Print source distribution
    sources = {}
    types = {}
    for doc in documents:
        src = doc.metadata.get("source", "unknown")
        typ = doc.metadata.get("type", "unknown")
        sources[src] = sources.get(src, 0) + 1
        types[typ] = types.get(typ, 0) + 1

    print(f"\n  Sources ({len(sources)} unique):")
    for src, count in sorted(sources.items(), key=lambda x: -x[1])[:10]:
        print(f"    {src}: {count}")
    print(f"\n  Types: {types}")

    # ── Step 2: Sentence-aware chunking ──
    print("\n  Step 2: Chunking (same params as Yelp)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    chunks = splitter.split_documents(documents)

    # Add chunk index
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"  Split into {len(chunks)} chunks")
    print(f"  (from {len(documents)} original records)")

    # Text length stats
    lengths = [len(c.page_content) for c in chunks]
    print(f"  Chunk lengths: min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths)/len(lengths):.0f}")

    # ── Step 3: Save chunk metadata ──
    print("\n  Step 3: Saving chunk metadata...")
    metadata_list = []
    for i, chunk in enumerate(chunks):
        metadata_list.append({
            "chunk_index": i,
            "text_preview": chunk.page_content[:100],
            "text_length": len(chunk.page_content),
            **{k: v for k, v in chunk.metadata.items() if k != "chunk_index"},
        })

    os.makedirs(os.path.dirname(LEGAL_METADATA_FILE), exist_ok=True)
    with open(LEGAL_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved: {LEGAL_METADATA_FILE}")

    # ── Step 4: Embed with BGE (normalized) ──
    print("\n  Step 4: Embedding chunks (BGE-base-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=EMBEDDING_ENCODE_KWARGS,
    )

    t_start = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    t_embed = time.time() - t_start

    # ── Step 5: Save FAISS index ──
    os.makedirs(args.output, exist_ok=True)
    vectorstore.save_local(args.output)

    # Index size on disk
    index_size = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if os.path.isfile(os.path.join(args.output, f))
    )

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  TASK 1 COMPLETE — Legal Domain Ingested")
    print(f"  {'─' * 40}")
    print(f"  Original records:   {len(documents)}")
    print(f"  Total chunks:       {len(chunks)}")
    print(f"  Embedding time:     {t_embed:.1f}s")
    print(f"  Index size:         {index_size / 1024 / 1024:.2f} MB")
    print(f"  Index path:         {args.output}")
    print(f"  Metadata path:      {LEGAL_METADATA_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
