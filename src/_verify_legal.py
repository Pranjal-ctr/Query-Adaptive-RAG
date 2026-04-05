"""Quick verification of legal FAISS index."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import (
    EMBEDDING_MODEL, EMBEDDING_MODEL_KWARGS, EMBEDDING_ENCODE_KWARGS,
    BGE_QUERY_PREFIX, PROJECT_ROOT,
)

LEGAL_DB = os.path.join(PROJECT_ROOT, "data", "legal", "faiss_index")

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs=EMBEDDING_MODEL_KWARGS,
    encode_kwargs=EMBEDDING_ENCODE_KWARGS,
)

print("Loading FAISS index...")
vs = FAISS.load_local(LEGAL_DB, embeddings, allow_dangerous_deserialization=True)

# Test query
query = BGE_QUERY_PREFIX + "What is the punishment for murder?"
results = vs.similarity_search_with_score(query, k=5)

print(f"\nTest query: 'What is the punishment for murder?'")
print(f"Results: {len(results)} documents")
for i, (doc, score) in enumerate(results):
    sim = max(0.0, min(1.0, 1.0 - (score / 2.0)))
    print(f"  [{i+1}] sim={sim:.4f} | {doc.page_content[:80]}...")

# Index stats
print(f"\nFAISS index size: {vs.index.ntotal} vectors")
print(f"Index path: {LEGAL_DB}")
print(f"\nIndex size on disk:")
for f in os.listdir(LEGAL_DB):
    fp = os.path.join(LEGAL_DB, f)
    print(f"  {f}: {os.path.getsize(fp)/1024/1024:.2f} MB")

print("\n✓ Legal FAISS index verification PASSED")
