"""
diagnostic.py — Run this to find the root cause of low similarity scores
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import DB_PATH, EMBEDDING_MODEL, EMBEDDING_MODEL_KWARGS, EMBEDDING_ENCODE_KWARGS, BGE_QUERY_PREFIX

# Load vectorstore
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs=EMBEDDING_MODEL_KWARGS,
    encode_kwargs=EMBEDDING_ENCODE_KWARGS,
)
vs = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

# Test query
query = "best pizza place"
prefixed = BGE_QUERY_PREFIX + query

# Get RAW scores before any conversion
raw_results = vs.similarity_search_with_score(prefixed, k=20)

print("=== RAW FAISS SCORES (before conversion) ===")
for i, (doc, score) in enumerate(raw_results[:10]):
    print(f"  [{i+1}] raw_score={score:.4f}  |  text preview: {doc.page_content[:60]}")

print(f"\nRaw score range: min={min(s for _,s in raw_results):.4f}  max={max(s for _,s in raw_results):.4f}")

# Check what FAISS index type is being used
print(f"\nFAISS index type: {type(vs.index)}")
print(f"Total vectors in index: {vs.index.ntotal}")

# Manually compute embedding similarity to verify
query_emb = embeddings.embed_query(prefixed)
doc_emb = embeddings.embed_documents([raw_results[0][0].page_content])

query_vec = np.array(query_emb)
doc_vec = np.array(doc_emb[0])

# Check norms (should be ~1.0 if normalized)
print(f"\nQuery embedding norm: {np.linalg.norm(query_vec):.4f}  (should be 1.0)")
print(f"Doc embedding norm:   {np.linalg.norm(doc_vec):.4f}  (should be 1.0)")

# Manual cosine similarity
cosine_sim = np.dot(query_vec, doc_vec)
print(f"\nManual cosine similarity (top doc): {cosine_sim:.4f}")
print(f"Your converted similarity:           {max(0, 1 - raw_results[0][1]/2):.4f}")
print(f"Direct FAISS score:                  {raw_results[0][1]:.4f}")

print("\n=== INTERPRETATION ===")
raw = raw_results[0][1]
if raw > 1.5:
    print("  FAISS is using L2 distance. Scores > 2.0 mean very dissimilar.")
    print("  Your conversion formula is correct but semantic similarity is genuinely low.")
elif raw < 0.5:
    print("  Scores are low — FAISS may be using inner product, score IS similarity.")
    print("  Your conversion formula is WRONG — you're shrinking already-small scores.")
else:
    print("  Scores in mid range — check manual cosine similarity above for ground truth.")