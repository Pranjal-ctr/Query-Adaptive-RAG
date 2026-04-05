"""
RAG Query Pipeline — Phase 0 Baseline with Score Collection

Uses similarity_search_with_score() to capture raw similarity scores for
every retrieval. Scores are logged to CSV/JSON for Phase 1 analysis.

Two modes:
    - Interactive: `python src/rag_query.py`
    - Batch:       `python src/rag_query.py --batch queries.txt`

Usage:
    cd d:/Programming/AI/RAG/food_domain_rag
    python src/rag_query.py
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from config import (
    DB_PATH, EMBEDDING_MODEL, EMBEDDING_MODEL_KWARGS,
    EMBEDDING_ENCODE_KWARGS, LLM_MODEL,
    K_MAX, INTENT_K_MAP, BGE_QUERY_PREFIX,
)
from intent_classifier import classify_intent
from score_logger import ScoreLogger


# ─── Prompt Template ─────────────────────────────────────────────────────────
PROMPT = PromptTemplate(
    template="""You are a food and restaurant analysis assistant.
Answer strictly using the customer reviews provided below.
Mention food quality, service, ambience, and price where relevant.
Do not hallucinate restaurant facts — only use information from the context.

Context (customer reviews):
{context}

Question:
{question}

Answer:""",
    input_variables=["context", "question"],
)


def load_vectorstore():
    """Load FAISS vectorstore with BGE embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=EMBEDDING_ENCODE_KWARGS,
    )
    return FAISS.load_local(
        DB_PATH, embeddings, allow_dangerous_deserialization=True
    )


def retrieve_with_scores(vectorstore, query: str, k: int = K_MAX):
    """
    Retrieve top-k candidates WITH similarity scores.

    FAISS returns (Document, score) pairs where score is L2 distance.
    Lower distance = higher similarity. We convert to similarity.

    Returns:
        List of (Document, similarity_score) tuples, sorted by score descending.
    """
    # Prepend BGE query instruction for proper retrieval
    prefixed_query = BGE_QUERY_PREFIX + query

    # Get candidates with distance scores
    results = vectorstore.similarity_search_with_score(prefixed_query, k=k)

    # FAISS with normalized embeddings: distance = 2*(1 - cosine_sim)
    # So cosine_sim = 1 - distance/2
    # But with IP (inner product) on normalized vectors, score IS similarity
    # LangChain FAISS uses L2 by default, so we convert:
    converted = []
    for doc, score in results:
        # For normalized embeddings with L2: similarity = 1 - (score / 2)
        # Clamp to [0, 1]
        similarity = max(0.0, min(1.0, 1.0 - (score / 2.0)))
        converted.append((doc, similarity))

    # Sort by similarity descending
    converted.sort(key=lambda x: x[1], reverse=True)
    return converted


def format_docs(docs_with_scores, k: int):
    """Format top-k documents for the prompt context."""
    top_k = docs_with_scores[:k]
    parts = []
    for i, (doc, score) in enumerate(top_k, 1):
        parts.append(f"[Review {i} | sim={score:.4f}]\n{doc.page_content}")
    return "\n\n".join(parts)


def get_baseline_k(intent: str) -> int:
    """Get baseline K from intent (current system's approach)."""
    return INTENT_K_MAP.get(intent, 4)


def query_rag(
    vectorstore,
    llm,
    query: str,
    score_logger: ScoreLogger,
    verbose: bool = True,
):
    """
    Run a single RAG query with full score collection.

    1. Classify intent
    2. Retrieve K_MAX candidates with similarity scores
    3. Use baseline K for generation
    4. Log all scores for Phase 1 analysis
    """
    # ── Step 1: Intent classification ──
    intent = classify_intent(query)
    baseline_k = get_baseline_k(intent)

    # ── Step 2: Retrieve K_MAX candidates with scores ──
    results = retrieve_with_scores(vectorstore, query, k=K_MAX)

    # Extract all similarity scores (the research data!)
    all_scores = [score for _, score in results]

    # ── Step 3: Generate answer using baseline K ──
    context = format_docs(results, k=baseline_k)
    response = llm.invoke(PROMPT.format(context=context, question=query))

    # ── Step 4: Log everything ──
    query_id = score_logger.log(
        query_text=query,
        intent=intent,
        similarity_scores=all_scores,
        num_used=baseline_k,
        response=response,
    )

    # ── Print results ──
    if verbose:
        print(f"\n{'─' * 50}")
        print(f"  Query ID:  {query_id}")
        print(f"  Intent:    {intent}")
        print(f"  K used:    {baseline_k} / {K_MAX} retrieved")
        print(f"  Scores:    max={all_scores[0]:.4f}  min={all_scores[-1]:.4f}"
              f"  range={all_scores[0] - all_scores[-1]:.4f}")
        print(f"  Top-5 sim: {[round(s, 4) for s in all_scores[:5]]}")
        print(f"{'─' * 50}")
        print(f"\nAssistant: {response}")

    return {
        "query_id": query_id,
        "intent": intent,
        "response": response,
        "scores": all_scores,
        "k_used": baseline_k,
    }


def run_interactive(vectorstore, llm, score_logger):
    """Interactive query loop."""
    print("\n" + "=" * 60)
    print("  PHASE 0 — RAG Baseline with Score Collection")
    print(f"  Retrieving K_MAX={K_MAX} candidates per query")
    print(f"  Scores saved to: {score_logger.csv_path}")
    print("  Type 'exit' to quit, 'stats' for logging stats")
    print("=" * 60)

    while True:
        query = input("\nUser: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            print(f"\n  ✓ {score_logger.get_curve_count()} curves saved. Goodbye!")
            break
        if query.lower() == "stats":
            count = score_logger.get_curve_count()
            print(f"  📊 {count} score curves logged so far")
            continue

        query_rag(vectorstore, llm, query, score_logger)


def run_batch(vectorstore, llm, score_logger, queries_file: str):
    """Batch mode — run a list of queries from a file."""
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f
                   if line.strip() and not line.strip().startswith("#")]

    print(f"\n  Running batch: {len(queries)} queries from {queries_file}")
    print(f"  Scores saved to: {score_logger.csv_path}")

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {query}")
        query_rag(vectorstore, llm, query, score_logger, verbose=True)

    print(f"\n  ✓ Batch complete. {score_logger.get_curve_count()} total curves saved.")


def main():
    parser = argparse.ArgumentParser(description="RAG Query with Score Collection")
    parser.add_argument("--batch", type=str, help="Path to queries file for batch mode")
    args = parser.parse_args()

    print("  Loading vectorstore...")
    vectorstore = load_vectorstore()

    print("  Loading LLM (Ollama Mistral)...")
    llm = Ollama(model=LLM_MODEL)

    score_logger = ScoreLogger()

    if args.batch:
        run_batch(vectorstore, llm, score_logger, args.batch)
    else:
        run_interactive(vectorstore, llm, score_logger)


if __name__ == "__main__":
    main()
