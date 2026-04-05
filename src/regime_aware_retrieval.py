"""
Regime-Aware Retrieval — Task 6
Uses the trained regime classifier to adapt K at retrieval time.

Strategy:
  - Flat regime (predicted): Use higher K (10-15), weight chunks equally
  - Discriminative regime: Use lower K (3-5), trust top chunks

Compares baseline (fixed K=5) vs regime-aware K on both domains.

Usage:
    cd d:/Programming/AI/RAG/food_domain_rag
    python src/regime_aware_retrieval.py
"""

import json
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import spacy

from config import RESULTS_DIR, PROJECT_ROOT
from evaluate_rag import compute_distribution_metrics

# ─── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "regime_classifier.pkl")
YELP_CURVES = os.path.join(PROJECT_ROOT, "data", "score_curves", "curve_data.json")
LEGAL_CURVES = os.path.join(PROJECT_ROOT, "data", "legal", "score_curves", "curve_data.json")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "regime_aware_evaluation.json")

# ─── K Selection Strategies ──────────────────────────────────────────────────
K_FLAT = 12           # Higher K for flat regime (need quantity)
K_DISCRIMINATIVE = 4  # Lower K for discriminative regime (trust quality)
K_BASELINE = 5        # Fixed baseline


def load_classifier():
    """Load the trained regime classifier."""
    if not os.path.exists(MODEL_PATH):
        print(f"  ✗ No classifier found at {MODEL_PATH}")
        print(f"    Run: python src/regime_classifier.py (Task 5)")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data


def extract_features_for_query(text: str, nlp) -> dict:
    """Extract features for a single query (same as regime_classifier.py)."""
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space]
    n_tokens = len(tokens)
    n_chars = len(text)
    n_entities = len(doc.ents)
    n_entity_types = len(set(ent.label_ for ent in doc.ents))

    pos_counts = {}
    for token in tokens:
        pos = token.pos_
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    n_nouns = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    n_verbs = pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)
    n_adj = pos_counts.get("ADJ", 0)
    n_adv = pos_counts.get("ADV", 0)

    noun_ratio = n_nouns / max(n_tokens, 1)
    verb_ratio = n_verbs / max(n_tokens, 1)
    adj_ratio = n_adj / max(n_tokens, 1)

    def get_depth(token):
        depth = 0
        while token.head != token:
            depth += 1
            token = token.head
        return depth

    depths = [get_depth(t) for t in tokens] if tokens else [0]
    max_depth = max(depths)
    mean_depth = sum(depths) / len(depths)

    text_lower = text.lower()
    has_question_mark = 1 if "?" in text else 0
    question_words = ["what", "which", "how", "where", "when", "who", "why",
                      "does", "is", "are", "can", "do", "compare", "difference"]
    n_question_words = sum(1 for w in question_words if w in text_lower)

    legal_keywords = ["section", "act", "article", "clause", "provision",
                      "penalty", "offence", "court", "law", "legal", "rights",
                      "contract", "statute", "bail", "criminal", "civil"]
    food_keywords = ["restaurant", "food", "menu", "dish", "taste", "service",
                     "chef", "dining", "cuisine", "meal", "review", "price",
                     "ambience", "waiter", "portion"]

    n_legal = sum(1 for k in legal_keywords if k in text_lower)
    n_food = sum(1 for k in food_keywords if k in text_lower)
    avg_word_len = sum(len(t.text) for t in tokens) / max(n_tokens, 1)

    return {
        "n_tokens": n_tokens, "n_chars": n_chars,
        "n_entities": n_entities, "n_entity_types": n_entity_types,
        "noun_ratio": round(noun_ratio, 4), "verb_ratio": round(verb_ratio, 4),
        "adj_ratio": round(adj_ratio, 4),
        "n_nouns": n_nouns, "n_verbs": n_verbs, "n_adj": n_adj, "n_adv": n_adv,
        "max_depth": max_depth, "mean_depth": round(mean_depth, 4),
        "has_question_mark": has_question_mark,
        "n_question_words": n_question_words,
        "n_legal_keywords": n_legal, "n_food_keywords": n_food,
        "avg_word_len": round(avg_word_len, 4),
    }


def evaluate_with_k(curves: list, k: int) -> dict:
    """Evaluate all curves using a specific K, return mean metrics."""
    all_metrics = []
    for curve in curves:
        scores = curve["similarity_scores"]
        metrics = compute_distribution_metrics(scores, k=k)
        all_metrics.append(metrics)

    def safe_mean(key):
        vals = [m.get(key) for m in all_metrics if m.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "k": k,
        "n_queries": len(all_metrics),
        "entropy": safe_mean("entropy"),
        "score_range": safe_mean("score_range"),
        "rel_precision_k": safe_mean(f"rel_precision_k{min(k, 10)}"),
        "norm_confidence": safe_mean("norm_confidence_k5"),
        "drop_ratio_k5": safe_mean("drop_ratio_k5"),
    }


def regime_aware_evaluate(curves: list, classifier_data: dict, nlp) -> dict:
    """Evaluate using regime-aware K selection."""
    model = classifier_data["model"]
    feature_keys = classifier_data["feature_keys"]

    flat_count = 0
    disc_count = 0
    all_metrics = []

    for curve in curves:
        query_text = curve["query_text"]
        scores = curve["similarity_scores"]

        # Predict regime
        feats = extract_features_for_query(query_text, nlp)
        X = np.array([[feats[k] for k in feature_keys]])
        predicted_regime = model.predict(X)[0]

        # Select K
        if predicted_regime == 0:  # Flat
            k = K_FLAT
            flat_count += 1
        else:  # Discriminative
            k = K_DISCRIMINATIVE
            disc_count += 1

        metrics = compute_distribution_metrics(scores, k=min(k, len(scores)))
        metrics["predicted_regime"] = "flat" if predicted_regime == 0 else "discriminative"
        metrics["k_selected"] = k
        all_metrics.append(metrics)

    def safe_mean(key):
        vals = [m.get(key) for m in all_metrics if m.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "strategy": "regime_aware",
        "n_queries": len(all_metrics),
        "n_flat": flat_count,
        "n_discriminative": disc_count,
        "entropy": safe_mean("entropy"),
        "score_range": safe_mean("score_range"),
        "norm_confidence": safe_mean("norm_confidence_k5"),
        "drop_ratio_k5": safe_mean("drop_ratio_k5"),
        "per_query": all_metrics,
    }


def main():
    print("=" * 60)
    print("  TASK 6 — Regime-Aware Retrieval Evaluation")
    print("=" * 60)

    # Load classifier
    print("\n  Loading regime classifier...")
    classifier_data = load_classifier()
    print(f"  ✓ Loaded: {classifier_data['best_name']}")

    # Load spaCy
    print("  Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")

    # Load curves
    results = {}

    for domain, curves_path in [("yelp", YELP_CURVES), ("legal", LEGAL_CURVES)]:
        if not os.path.exists(curves_path):
            print(f"\n  ⚠ Skipping {domain}: no curves at {curves_path}")
            continue

        with open(curves_path, "r", encoding="utf-8") as f:
            curves = json.load(f)
        print(f"\n  === {domain.upper()} ({len(curves)} queries) ===")

        # Baseline (fixed K=5)
        baseline = evaluate_with_k(curves, K_BASELINE)
        print(f"  Baseline (K={K_BASELINE}):")
        print(f"    Confidence: {baseline['norm_confidence']}")
        print(f"    Entropy:    {baseline['entropy']}")

        # Regime-aware
        aware = regime_aware_evaluate(curves, classifier_data, nlp)
        print(f"  Regime-Aware (K={K_DISCRIMINATIVE} disc / K={K_FLAT} flat):")
        print(f"    Flat predicted:  {aware['n_flat']}")
        print(f"    Disc predicted:  {aware['n_discriminative']}")
        print(f"    Confidence:      {aware['norm_confidence']}")
        print(f"    Entropy:         {aware['entropy']}")

        # Compare
        if baseline["norm_confidence"] and aware["norm_confidence"]:
            improvement = aware["norm_confidence"] - baseline["norm_confidence"]
            pct = (improvement / baseline["norm_confidence"]) * 100 if baseline["norm_confidence"] else 0
            print(f"\n  Confidence improvement: {improvement:+.4f} ({pct:+.1f}%)")

        results[domain] = {
            "baseline": baseline,
            "regime_aware": {k: v for k, v in aware.items() if k != "per_query"},
            "regime_aware_per_query": aware.get("per_query", []),
        }

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  ✓ Results saved → {OUTPUT_PATH}")

    print(f"\n{'=' * 60}")
    print(f"  TASK 6 COMPLETE — Regime-Aware Retrieval Evaluated")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
