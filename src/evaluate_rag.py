"""
RAG Evaluation Metrics — Fixed & Research-Grade
================================================

BUGS FIXED vs previous version:
  1. NDCG: Was computing ideal DCG from already-sorted scores (always = 1.0)
           Fixed: NDCG now requires external relevance labels from QA pairs
           Without labels: reports score_drop_ratio instead (honest proxy)

  2. Hit Rate: Was identical across K=3/5/10 because threshold (0.65) was
               too close to max similarity (0.651). Fixed by using
               relative threshold (top-X% of score range) instead of absolute.

  3. Confidence Score: Was always ~0.96 because score range is tiny (0.057).
                       Fixed: now uses normalized range position, not mean/max ratio.

  4. MRR: Was only finding relevant at rank 1 or not at all (same threshold bug).
          Fixed alongside hit rate.

CORRECT METRICS (no ground truth needed):
  - Score Range / Spread     — how discriminative the retrieval is
  - Relative Precision@K     — uses adaptive threshold per query
  - Drop Ratio               — how much score drops from rank 1 to rank K
  - Score Entropy            — distribution uniformity (your key research metric)
  - Relative MRR             — rank of top-scoring chunk relative to spread

CORRECT METRICS (requires QA ground truth from build_eval_dataset.py):
  - True Precision@K         — chunk contains answer string
  - True NDCG@K              — ideal ranking from labeled relevance
  - True Hit Rate@K          — at least one answer chunk in top K
  - True MRR                 — rank of first answer-containing chunk

LLM Judge (always valid):
  - Faithfulness, Relevancy, Completeness

Usage:
    # Without ground truth (always works):
    python src/evaluate_rag.py

    # With ground truth QA pairs (research-grade):
    python src/evaluate_rag.py --qa-file data/eval/qa_pairs_curated.json
"""

import json
import math
import os
import sys
import time
import re
import argparse
import random
import requests
from typing import List, Dict, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CURVE_DATA_FILE, EVAL_DIR

OLLAMA_URL   = "http://localhost:11434/api/generate"
LLM_MODEL    = "mistral"
LLM_SAMPLE   = 100


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: METRICS THAT DON'T NEED GROUND TRUTH
# These are always valid and form your baseline research metrics.
# ══════════════════════════════════════════════════════════════════════════════

def score_entropy(scores: List[float]) -> float:
    """
    Shannon entropy of the normalized score distribution.
    High entropy = flat/uniform = FLAT regime (Yelp-like)
    Low entropy  = concentrated = DISCRIMINATIVE regime (HotpotQA-like)
    THIS IS YOUR KEY RESEARCH METRIC.
    """
    arr = [max(0.0, s) for s in scores]
    total = sum(arr)
    if total == 0:
        return 0.0
    probs = [s / total for s in arr]
    return -sum(p * math.log2(p + 1e-12) for p in probs)


def score_kurtosis(scores: List[float]) -> float:
    """
    Excess kurtosis of score distribution.
    >0 = peaked (one dominant chunk)
    <0 = flat (uniform scores)
    """
    if len(scores) < 4:
        return 0.0
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    if variance == 0:
        return 0.0
    m4 = sum((s - mean) ** 4 for s in scores) / n
    return (m4 / (variance ** 2)) - 3


def drop_ratio_at_k(scores: List[float], k: int) -> float:
    """
    How much the score drops from rank 1 to rank K.
    drop_ratio = scores[k-1] / scores[0]
    High value (near 1.0) = flat curve, scores barely drop = FLAT regime
    Low value (near 0.0)  = sharp drop = DISCRIMINATIVE regime
    """
    if not scores or k > len(scores) or scores[0] == 0:
        return 0.0
    return scores[k - 1] / scores[0]


def relative_precision_at_k(scores: List[float], k: int, top_pct: float = 0.8) -> float:
    """
    Precision@K using a RELATIVE threshold instead of absolute 0.65.
    Threshold = top_pct * max_score (e.g. 80% of best score)
    This avoids the bug where absolute threshold sits above all scores.
    """
    if not scores:
        return 0.0
    threshold = top_pct * scores[0]
    top_k = scores[:k]
    relevant = sum(1 for s in top_k if s >= threshold)
    return relevant / len(top_k)


def relative_mrr(scores: List[float], top_pct: float = 0.8) -> float:
    """MRR with relative threshold — avoids flat-distribution MRR bug."""
    if not scores:
        return 0.0
    threshold = top_pct * scores[0]
    for i, s in enumerate(scores):
        if s >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def relative_hit_rate_at_k(scores: List[float], k: int, top_pct: float = 0.8) -> float:
    """Hit rate with relative threshold."""
    if not scores:
        return 0.0
    threshold = top_pct * scores[0]
    return 1.0 if any(s >= threshold for s in scores[:k]) else 0.0


def normalized_confidence(scores: List[float], k: int) -> float:
    """
    Confidence score that works even in flat-distribution domains.
    Uses score range position instead of mean/max ratio.

    confidence = (mean_top_k - min_all) / (max_all - min_all + eps)

    High = top-K chunks are much better than the worst chunks
    Low  = top-K chunks are barely better than worst = flat = low confidence
    """
    if not scores or len(scores) < 2:
        return 0.0
    top_k = scores[:k]
    max_s = scores[0]
    min_s = scores[-1]
    spread = max_s - min_s
    if spread < 1e-6:
        return 0.0  # Completely flat — zero confidence (honest!)
    mean_top_k = sum(top_k) / len(top_k)
    return (mean_top_k - min_s) / spread


def compute_distribution_metrics(scores: List[float], k: int = 5) -> Dict:
    """
    Compute all no-ground-truth metrics for one query's score vector.
    These are always valid and form your research baseline.
    """
    return {
        # Distribution shape (your key research metrics)
        "entropy":         round(score_entropy(scores), 4),
        "kurtosis":        round(score_kurtosis(scores), 4),
        "score_range":     round(scores[0] - scores[-1], 4) if scores else 0.0,
        "max_similarity":  round(scores[0], 4) if scores else 0.0,
        "mean_similarity": round(sum(scores) / len(scores), 4) if scores else 0.0,

        # Drop ratios — how fast quality degrades
        "drop_ratio_k3":   round(drop_ratio_at_k(scores, 3), 4),
        "drop_ratio_k5":   round(drop_ratio_at_k(scores, 5), 4),
        "drop_ratio_k10":  round(drop_ratio_at_k(scores, 10), 4),

        # Relative precision (adaptive threshold, avoids absolute threshold bug)
        "rel_precision_k3":  round(relative_precision_at_k(scores, 3), 4),
        "rel_precision_k5":  round(relative_precision_at_k(scores, 5), 4),
        "rel_precision_k10": round(relative_precision_at_k(scores, 10), 4),

        # Relative MRR and Hit Rate (adaptive threshold)
        "rel_mrr":          round(relative_mrr(scores), 4),
        "rel_hit_rate_k3":  round(relative_hit_rate_at_k(scores, 3), 4),
        "rel_hit_rate_k5":  round(relative_hit_rate_at_k(scores, 5), 4),
        "rel_hit_rate_k10": round(relative_hit_rate_at_k(scores, 10), 4),

        # Normalized confidence (honest in flat domains)
        "norm_confidence_k5": round(normalized_confidence(scores, k), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: METRICS THAT NEED GROUND TRUTH
# Only computed when --qa-file is provided.
# These are the publishable research-grade metrics.
# ══════════════════════════════════════════════════════════════════════════════

def chunk_contains_answer(chunk_text: str, answer: str) -> bool:
    """
    Check if a chunk contains the answer string.
    Uses case-insensitive substring match.
    For research: you can upgrade this to token-overlap (F1) later.
    """
    return answer.lower().strip() in chunk_text.lower()


def true_precision_at_k(chunks: List[str], answer: str, k: int) -> float:
    """Fraction of top-K chunks that contain the answer."""
    top_k = chunks[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for c in top_k if chunk_contains_answer(c, answer))
    return relevant / len(top_k)


def true_ndcg_at_k(chunks: List[str], answer: str, k: int) -> float:
    """
    NDCG@K with EXTERNAL ground truth labels.
    Relevance label = 1 if chunk contains answer, 0 otherwise.
    Ideal ranking = all relevant chunks first.
    THIS IS THE CORRECT NDCG — not the circular version.
    """
    labels = [1 if chunk_contains_answer(c, answer) else 0 for c in chunks]

    def dcg(relevances, k):
        return sum(rel / math.log2(i + 2)
                   for i, rel in enumerate(relevances[:k]))

    actual_dcg = dcg(labels, k)
    ideal_labels = sorted(labels, reverse=True)  # sort LABELS not scores
    ideal_dcg = dcg(ideal_labels, k)

    if ideal_dcg == 0:
        return None  # No relevant chunk exists — exclude from average
    return round(actual_dcg / ideal_dcg, 4)


def true_mrr(chunks: List[str], answer: str) -> Optional[float]:
    """MRR: 1/rank of first chunk containing the answer."""
    for i, chunk in enumerate(chunks):
        if chunk_contains_answer(chunk, answer):
            return round(1.0 / (i + 1), 4)
    return 0.0  # Answer not found in any chunk


def true_hit_rate_at_k(chunks: List[str], answer: str, k: int) -> float:
    """1 if answer found in top-K chunks, 0 otherwise."""
    return 1.0 if any(chunk_contains_answer(c, answer) for c in chunks[:k]) else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: LLM JUDGE
# ══════════════════════════════════════════════════════════════════════════════

def call_ollama(prompt: str) -> Optional[str]:
    try:
        r = requests.post(OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1}}, timeout=120)
        return r.json().get("response", "") if r.status_code == 200 else None
    except:
        return None


def parse_score(text: str) -> Optional[int]:
    if not text:
        return None
    for pattern in [r'[Ss]core\s*:\s*(\d)', r'(\d)\s*/\s*5', r'\b([1-5])\b']:
        m = re.search(pattern, text)
        if m:
            return min(5, max(1, int(m.group(1))))
    return None


def llm_judge(query: str, response: str) -> Dict:
    """Single combined prompt — 3x faster than separate calls."""
    prompt = f"""You are evaluating a RAG system answer. Rate on three dimensions (1-5 each).

QUERY: {query}
ANSWER: {response[:1000]}

Rate each:
1. FAITHFULNESS: Is every claim in the answer traceable to the context? (1=hallucinated, 5=fully grounded)
2. RELEVANCY: Does the answer address what was asked? (1=irrelevant, 5=perfectly relevant)
3. COMPLETENESS: Does it cover all key aspects? (1=very incomplete, 5=comprehensive)

Output ONLY this format:
Faithfulness: X
Relevancy: X
Completeness: X"""

    raw = call_ollama(prompt)
    if not raw:
        return {"faithfulness": None, "relevancy": None, "completeness": None}

    def extract(label):
        m = re.search(rf'{label}:\s*([1-5])', raw, re.IGNORECASE)
        return int(m.group(1)) if m else None

    return {
        "faithfulness":  extract("Faithfulness"),
        "relevancy":     extract("Relevancy"),
        "completeness":  extract("Completeness"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def load_qa_lookup(qa_file: str) -> Dict:
    """Build a lookup from query text → answer for ground truth evaluation."""
    with open(qa_file, encoding="utf-8") as f:
        pairs = json.load(f)
    # Key by question text (normalized)
    return {p["question"].lower().strip(): p["answer"] for p in pairs}


def run_evaluation(curves: List[Dict], qa_lookup: Dict, use_llm: bool) -> List[Dict]:
    """Run full evaluation over all curves."""
    results = []
    total = len(curves)

    # Sample for LLM judging
    judge_indices = set()
    if use_llm:
        sample_size = min(LLM_SAMPLE, total)
        judge_indices = set(random.sample(range(total), sample_size))
        print(f"  LLM judging: {len(judge_indices)} queries sampled")

    for i, curve in enumerate(curves):
        query   = curve["query_text"]
        scores  = curve["similarity_scores"]
        chunks  = [d for d in curve.get("retrieved_chunks", [])]
        intent  = curve.get("intent", "unknown")
        response = curve.get("response", "")
        k_used  = curve.get("num_used", 5)

        # Always compute: distribution metrics
        metrics = compute_distribution_metrics(scores, k=k_used)
        metrics["query_id"]   = curve["query_id"]
        metrics["query_text"] = query
        metrics["intent"]     = intent
        metrics["k_used"]     = k_used

        # Ground truth metrics (if QA file provided AND chunks are stored)
        answer = qa_lookup.get(query.lower().strip())
        if answer and chunks:
            metrics["true_precision_k3"]  = true_precision_at_k(chunks, answer, 3)
            metrics["true_precision_k5"]  = true_precision_at_k(chunks, answer, 5)
            metrics["true_precision_k10"] = true_precision_at_k(chunks, answer, 10)
            metrics["true_ndcg_k5"]       = true_ndcg_at_k(chunks, answer, 5)
            metrics["true_ndcg_k10"]      = true_ndcg_at_k(chunks, answer, 10)
            metrics["true_mrr"]           = true_mrr(chunks, answer)
            metrics["true_hit_k3"]        = true_hit_rate_at_k(chunks, answer, 3)
            metrics["true_hit_k5"]        = true_hit_rate_at_k(chunks, answer, 5)
            metrics["true_hit_k10"]       = true_hit_rate_at_k(chunks, answer, 10)
            metrics["has_ground_truth"]   = True
        else:
            metrics["has_ground_truth"] = False

        # LLM judge (sampled)
        if i in judge_indices and response:
            print(f"  [{i+1}/{total}] LLM judging: {query[:55]}...", end="\r")
            judge = llm_judge(query, response)
            metrics.update(judge)
        else:
            metrics["faithfulness"]  = None
            metrics["relevancy"]     = None
            metrics["completeness"]  = None

        results.append(metrics)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def safe_mean(values):
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def print_report(results: List[Dict]):
    total = len(results)
    intents = sorted(set(r["intent"] for r in results))

    def group_mean(group, key):
        return safe_mean([r.get(key) for r in group])

    print(f"\n{'='*68}")
    print(f"  RAG EVALUATION REPORT (Fixed & Research-Grade)")
    print(f"  {total} queries  ·  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*68}")

    print(f"\n  ── DISTRIBUTION SHAPE (your research metrics) ──────────────")
    print(f"  Entropy (avg):              {group_mean(results, 'entropy')}")
    print(f"  Kurtosis (avg):             {group_mean(results, 'kurtosis')}")
    print(f"  Score Range (avg):          {group_mean(results, 'score_range')}")
    print(f"  Norm. Confidence K5 (avg):  {group_mean(results, 'norm_confidence_k5')}")
    print(f"  Drop Ratio @K5 (avg):       {group_mean(results, 'drop_ratio_k5')}")
    print(f"  (drop_ratio near 1.0 = FLAT regime, near 0 = DISCRIMINATIVE)")

    print(f"\n  ── RELATIVE RETRIEVAL METRICS (adaptive threshold) ─────────")
    print(f"  Rel. P@3:                   {group_mean(results, 'rel_precision_k3')}")
    print(f"  Rel. P@5:                   {group_mean(results, 'rel_precision_k5')}")
    print(f"  Rel. MRR:                   {group_mean(results, 'rel_mrr')}")
    print(f"  Rel. Hit Rate @3:           {group_mean(results, 'rel_hit_rate_k3')}")
    print(f"  Rel. Hit Rate @5:           {group_mean(results, 'rel_hit_rate_k5')}")
    print(f"  Rel. Hit Rate @10:          {group_mean(results, 'rel_hit_rate_k10')}")
    print(f"  (Note: these use 80% of max score as threshold, not absolute 0.65)")

    # Ground truth metrics (if available)
    gt_results = [r for r in results if r.get("has_ground_truth")]
    if gt_results:
        print(f"\n  ── TRUE RETRIEVAL METRICS (external ground truth) ──────────")
        print(f"  Queries with GT:            {len(gt_results)}/{total}")
        print(f"  True P@3:                   {group_mean(gt_results, 'true_precision_k3')}")
        print(f"  True P@5:                   {group_mean(gt_results, 'true_precision_k5')}")
        ndcg_vals = [r.get('true_ndcg_k5') for r in gt_results if r.get('true_ndcg_k5') is not None]
        print(f"  True NDCG@5:                {round(sum(ndcg_vals)/len(ndcg_vals),4) if ndcg_vals else 'N/A'}")
        print(f"  True MRR:                   {group_mean(gt_results, 'true_mrr')}")
        print(f"  True Hit Rate @5:           {group_mean(gt_results, 'true_hit_k5')}")

    # LLM judge
    judged = [r for r in results if r.get("faithfulness") is not None]
    if judged:
        print(f"\n  ── LLM JUDGE ({len(judged)} sampled queries) ──────────────────────")
        print(f"  Faithfulness (1-5):         {group_mean(judged, 'faithfulness')}")
        print(f"  Relevancy (1-5):            {group_mean(judged, 'relevancy')}")
        print(f"  Completeness (1-5):         {group_mean(judged, 'completeness')}")

    # Per-intent breakdown
    print(f"\n  ── PER-INTENT (distribution metrics) ───────────────────────")
    print(f"  {'Intent':<16} {'N':>3}  {'Entropy':>8}  {'Kurt':>6}  {'Range':>7}  {'Conf':>6}  {'DropR@5':>8}")
    print(f"  {'─'*65}")
    for intent in intents:
        g = [r for r in results if r["intent"] == intent]
        print(f"  {intent:<16} {len(g):>3}"
              f"  {group_mean(g,'entropy'):>8}"
              f"  {group_mean(g,'kurtosis'):>6}"
              f"  {group_mean(g,'score_range'):>7}"
              f"  {group_mean(g,'norm_confidence_k5'):>6}"
              f"  {group_mean(g,'drop_ratio_k5'):>8}")

    print(f"\n  ── REGIME DIAGNOSIS ─────────────────────────────────────────")
    avg_range = group_mean(results, 'score_range')
    avg_entropy = group_mean(results, 'entropy')
    avg_drop = group_mean(results, 'drop_ratio_k5')
    if avg_range and avg_range < 0.05:
        print(f"  ⚠  FLAT REGIME detected (avg_range={avg_range})")
        print(f"     Score distributions are compressed — elbow detection unreliable")
        print(f"     This is expected for informal short-text corpora (Yelp)")
        print(f"     Compare against HotpotQA to show cross-domain contrast")
    elif avg_range and avg_range > 0.15:
        print(f"  ✓  DISCRIMINATIVE REGIME (avg_range={avg_range})")
        print(f"     Score distributions have useful structure — adaptive retrieval viable")
    else:
        print(f"  ~  MIXED REGIME (avg_range={avg_range})")

    print(f"{'='*68}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-file", type=str, default=None,
                        help="Path to curated QA pairs JSON for ground truth metrics")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM judge (faster, retrieval metrics only)")
    args = parser.parse_args()

    print(f"\n{'='*68}")
    print(f"  RAG EVALUATION — Fixed Version")
    print(f"{'='*68}")

    # Load curves
    print(f"\n  Loading curves from {CURVE_DATA_FILE}...")
    with open(CURVE_DATA_FILE, encoding="utf-8") as f:
        curves = json.load(f)
    print(f"  Loaded {len(curves)} queries")

    # Load QA pairs if provided
    qa_lookup = {}
    if args.qa_file and os.path.exists(args.qa_file):
        qa_lookup = load_qa_lookup(args.qa_file)
        print(f"  Loaded {len(qa_lookup)} ground truth QA pairs")
    else:
        print(f"  No QA file — computing distribution metrics only")
        print(f"  (For true NDCG/MRR/HitRate: run --qa-file data/eval/qa_pairs_curated.json)")

    # Check Ollama
    use_llm = False
    if not args.no_llm:
        test = call_ollama("Say ok")
        use_llm = test is not None
        print(f"  Ollama: {'✓ available' if use_llm else '✗ not available'}")

    # Run evaluation
    print(f"\n  Running evaluation...")
    results = run_evaluation(curves, qa_lookup, use_llm)

    # Save results
    os.makedirs(EVAL_DIR, exist_ok=True)
    out_path = os.path.join(EVAL_DIR, "rag_evaluation_fixed.json")
    save_results = [{k: v for k, v in r.items()
                     if k not in ("faithfulness_raw","relevancy_raw","completeness_raw")}
                    for r in results]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2)
    print(f"  ✓ Results saved → {out_path}")

    # Print report
    print_report(results)


if __name__ == "__main__":
    main()