"""
Legal Domain Full Pipeline - Task 3
Runs the complete analysis pipeline on the legal domain:

  1. Load legal FAISS index
  2. Run all legal queries through retrieval (K_MAX=20)
  3. Log score curves
  4. Run curve analysis -> results/legal/
  5. Run evaluation -> results/legal/
  6. Generate regime report

Usage:
    cd d:/Programming/AI/RAG/food_domain_rag
    python src/run_legal_analysis.py
"""

import io
import json
import os
import sys
import time

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('cp'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import (
    EMBEDDING_MODEL, EMBEDDING_MODEL_KWARGS, EMBEDDING_ENCODE_KWARGS,
    K_MAX, BGE_QUERY_PREFIX, PROJECT_ROOT,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
LEGAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "legal")
LEGAL_DB_PATH = os.path.join(LEGAL_DATA_DIR, "faiss_index")
LEGAL_QUERIES_FILE = os.path.join(LEGAL_DATA_DIR, "legal_queries.json")
LEGAL_CURVES_DIR = os.path.join(LEGAL_DATA_DIR, "score_curves")
LEGAL_CURVE_DATA = os.path.join(LEGAL_CURVES_DIR, "curve_data.json")
LEGAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "legal")

os.makedirs(LEGAL_CURVES_DIR, exist_ok=True)
os.makedirs(LEGAL_RESULTS_DIR, exist_ok=True)


def load_legal_vectorstore():
    """Load the legal FAISS vectorstore."""
    print("  Loading legal FAISS index...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=EMBEDDING_ENCODE_KWARGS,
    )
    vs = FAISS.load_local(LEGAL_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    print(f"  [OK] Legal index loaded from {LEGAL_DB_PATH}")
    return vs


def retrieve_with_scores(vectorstore, query: str, k: int = K_MAX):
    """Retrieve top-k with similarity scores (L2 → cosine conversion)."""
    prefixed = BGE_QUERY_PREFIX + query
    results = vectorstore.similarity_search_with_score(prefixed, k=k)
    converted = []
    for doc, score in results:
        similarity = max(0.0, min(1.0, 1.0 - (score / 2.0)))
        converted.append((doc, similarity))
    converted.sort(key=lambda x: x[1], reverse=True)
    return converted


def run_retrieval_pipeline(vectorstore, queries: list) -> list:
    """Run all queries through retrieval and collect score curves."""
    curves = []
    total = len(queries)

    for i, q_item in enumerate(queries):
        query = q_item["query"]
        intent = q_item.get("intent", "unknown")

        results = retrieve_with_scores(vectorstore, query, k=K_MAX)
        scores = [score for _, score in results]
        chunks = [doc.page_content for doc, _ in results]

        query_id = f"legal_q_{int(time.time())}_{i+1:04d}"

        curve = {
            "query_id": query_id,
            "query_text": query,
            "intent": intent,
            "domain": "legal",
            "similarity_scores": [round(float(s), 6) for s in scores],
            "num_candidates": len(scores),
            "num_used": min(5, len(scores)),
            "retrieved_chunks": chunks[:5],  # Save top-5 for evaluation
        }
        curves.append(curve)

        if (i + 1) % 10 == 0 or i == 0:
            score_range = scores[0] - scores[-1] if scores else 0
            print(f"  [{i+1:>3}/{total}] {intent:<13} range={score_range:.4f}  "
                  f"max={scores[0]:.4f}  {query[:50]}...")

    return curves


def run_curve_analysis(curves: list):
    """Run analyze_curves.py on the legal curves."""
    from analyze_curves import analyze_domain
    print("\n  Running curve analysis...")
    features = analyze_domain(curves, "Legal", LEGAL_RESULTS_DIR, n_clusters=4)

    # Also save as curve_features.json (compare_domains.py expects this name)
    compat_path = os.path.join(LEGAL_RESULTS_DIR, "curve_features.json")
    with open(compat_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
    print(f"  [OK] Features also saved -> {compat_path}")

    return features


def run_evaluation(curves: list):
    """Run evaluate_rag.py metrics on the legal curves."""
    from evaluate_rag import compute_distribution_metrics

    print("\n  Running evaluation metrics...")
    results = []
    for curve in curves:
        scores = curve["similarity_scores"]
        k_used = curve.get("num_used", 5)

        metrics = compute_distribution_metrics(scores, k=k_used)
        metrics["query_id"] = curve["query_id"]
        metrics["query_text"] = curve["query_text"]
        metrics["intent"] = curve["intent"]
        metrics["k_used"] = k_used
        metrics["has_ground_truth"] = False
        metrics["faithfulness"] = None
        metrics["relevancy"] = None
        metrics["completeness"] = None
        results.append(metrics)

    # Save evaluation report
    eval_path = os.path.join(LEGAL_RESULTS_DIR, "evaluation_report.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] Evaluation saved -> {eval_path}")

    # Print summary
    print_eval_summary(results)
    return results


def print_eval_summary(results: list):
    """Print a summary of evaluation metrics."""
    def safe_mean(values):
        clean = [v for v in values if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else None

    intents = sorted(set(r["intent"] for r in results))
    total = len(results)

    print(f"\n  {'='*60}")
    print(f"  LEGAL DOMAIN EVALUATION SUMMARY")
    print(f"  {total} queries")
    print(f"  {'='*60}")

    print(f"\n  DISTRIBUTION SHAPE:")
    print(f"  Entropy (avg):          {safe_mean([r['entropy'] for r in results])}")
    print(f"  Kurtosis (avg):         {safe_mean([r['kurtosis'] for r in results])}")
    print(f"  Score Range (avg):      {safe_mean([r['score_range'] for r in results])}")
    print(f"  Drop Ratio @K5 (avg):   {safe_mean([r['drop_ratio_k5'] for r in results])}")

    print(f"\n  PER-INTENT:")
    print(f"  {'Intent':<16} {'N':>3}  {'Entropy':>8}  {'Range':>7}  {'DropR@5':>8}")
    print(f"  {'-'*50}")
    for intent in intents:
        g = [r for r in results if r["intent"] == intent]
        print(f"  {intent:<16} {len(g):>3}"
              f"  {safe_mean([r['entropy'] for r in g]):>8}"
              f"  {safe_mean([r['score_range'] for r in g]):>7}"
              f"  {safe_mean([r['drop_ratio_k5'] for r in g]):>8}")

    # Regime diagnosis
    avg_range = safe_mean([r['score_range'] for r in results])
    print(f"\n  REGIME: ", end="")
    if avg_range and avg_range < 0.05:
        print(f"FLAT (avg_range={avg_range})")
    elif avg_range and avg_range > 0.15:
        print(f"DISCRIMINATIVE (avg_range={avg_range})")
    else:
        print(f"MIXED (avg_range={avg_range})")


def main():
    print("=" * 60)
    print("  TASK 3 - Legal Domain Full Pipeline")
    print("=" * 60)

    # Step 1: Load queries
    print(f"\n  Loading queries from {LEGAL_QUERIES_FILE}...")
    if not os.path.exists(LEGAL_QUERIES_FILE):
        print("  ERROR: Run generate_legal_queries.py first (Task 2)")
        sys.exit(1)
    with open(LEGAL_QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"  Loaded {len(queries)} queries")

    # Step 2: Load vectorstore
    vectorstore = load_legal_vectorstore()

    # Step 3: Run retrieval pipeline
    print(f"\n  Running retrieval on {len(queries)} queries (K_MAX={K_MAX})...")
    t_start = time.time()
    curves = run_retrieval_pipeline(vectorstore, queries)
    t_retrieval = time.time() - t_start
    print(f"\n  [OK] Retrieval complete in {t_retrieval:.1f}s")

    # Step 4: Save curves
    with open(LEGAL_CURVE_DATA, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Curves saved -> {LEGAL_CURVE_DATA}")

    # Step 5: Curve analysis
    features = run_curve_analysis(curves)

    # Step 6: Evaluation metrics
    eval_results = run_evaluation(curves)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  TASK 3 COMPLETE - Legal Domain Pipeline")
    print(f"  {'-' * 40}")
    print(f"  Queries processed:  {len(curves)}")
    print(f"  Retrieval time:     {t_retrieval:.1f}s")
    print(f"  Curves saved:       {LEGAL_CURVE_DATA}")
    print(f"  Results dir:        {LEGAL_RESULTS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
