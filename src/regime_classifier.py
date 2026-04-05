"""
Pre-Retrieval Regime Classifier — Task 5
Trains a classifier to predict retrieval regime (flat vs discriminative)
from query text alone, using NLP features from spaCy.

Features:
  - Token count, character count
  - Entity count (from spaCy NER)
  - POS distribution (noun/verb/adj/adv ratios)
  - Dependency tree depth
  - Question word presence
  - Domain-specific keyword counts

Uses combined Yelp + Legal data for training.
Saves the best model for use in regime-aware retrieval (Task 6).

Usage:
    cd d:/Programming/AI/RAG/food_domain_rag
    python src/regime_classifier.py
"""

import json
import os
import sys
import pickle
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from config import RESULTS_DIR, PROJECT_ROOT

# ─── Paths ────────────────────────────────────────────────────────────────────
YELP_EVAL = os.path.join(RESULTS_DIR, "yelp", "evaluation_report.json")
LEGAL_EVAL = os.path.join(RESULTS_DIR, "legal", "evaluation_report.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "regime_classifier.pkl")
RESULTS_PATH = os.path.join(RESULTS_DIR, "classifier_results.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Regime threshold ────────────────────────────────────────────────────────
REGIME_THRESHOLD = 0.05  # score_range < 0.05 = flat, >= 0.05 = discriminative


def load_data():
    """Load evaluation data from both domains."""
    all_data = []

    for domain, path in [("yelp", YELP_EVAL), ("legal", LEGAL_EVAL)]:
        if not os.path.exists(path):
            print(f"  ⚠ Missing {domain} data: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            item["domain"] = domain
        all_data.extend(data)
        print(f"  Loaded {len(data)} queries from {domain}")

    return all_data


def extract_features(text: str, nlp) -> dict:
    """Extract NLP features from a query text using spaCy."""
    doc = nlp(text)

    # Basic counts
    tokens = [t for t in doc if not t.is_space]
    n_tokens = len(tokens)
    n_chars = len(text)

    # Entity counts
    n_entities = len(doc.ents)
    entity_types = set(ent.label_ for ent in doc.ents)
    n_entity_types = len(entity_types)

    # POS distribution
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

    # Dependency depth
    def get_depth(token):
        depth = 0
        while token.head != token:
            depth += 1
            token = token.head
        return depth

    depths = [get_depth(t) for t in tokens] if tokens else [0]
    max_depth = max(depths)
    mean_depth = sum(depths) / len(depths)

    # Question features
    text_lower = text.lower()
    has_question_mark = 1 if "?" in text else 0
    question_words = ["what", "which", "how", "where", "when", "who", "why",
                      "does", "is", "are", "can", "do", "compare", "difference"]
    n_question_words = sum(1 for w in question_words if w in text_lower)

    # Domain keywords
    legal_keywords = ["section", "act", "article", "clause", "provision",
                      "penalty", "offence", "court", "law", "legal", "rights",
                      "contract", "statute", "bail", "criminal", "civil"]
    food_keywords = ["restaurant", "food", "menu", "dish", "taste", "service",
                     "chef", "dining", "cuisine", "meal", "review", "price",
                     "ambience", "waiter", "portion"]

    n_legal = sum(1 for k in legal_keywords if k in text_lower)
    n_food = sum(1 for k in food_keywords if k in text_lower)

    # Complexity features
    avg_word_len = sum(len(t.text) for t in tokens) / max(n_tokens, 1)

    return {
        "n_tokens": n_tokens,
        "n_chars": n_chars,
        "n_entities": n_entities,
        "n_entity_types": n_entity_types,
        "noun_ratio": round(noun_ratio, 4),
        "verb_ratio": round(verb_ratio, 4),
        "adj_ratio": round(adj_ratio, 4),
        "n_nouns": n_nouns,
        "n_verbs": n_verbs,
        "n_adj": n_adj,
        "n_adv": n_adv,
        "max_depth": max_depth,
        "mean_depth": round(mean_depth, 4),
        "has_question_mark": has_question_mark,
        "n_question_words": n_question_words,
        "n_legal_keywords": n_legal,
        "n_food_keywords": n_food,
        "avg_word_len": round(avg_word_len, 4),
    }


FEATURE_KEYS = [
    "n_tokens", "n_chars", "n_entities", "n_entity_types",
    "noun_ratio", "verb_ratio", "adj_ratio",
    "n_nouns", "n_verbs", "n_adj", "n_adv",
    "max_depth", "mean_depth",
    "has_question_mark", "n_question_words",
    "n_legal_keywords", "n_food_keywords", "avg_word_len",
]


def train_and_evaluate(X, y, labels_map):
    """Train multiple classifiers and report results."""
    classifiers = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=1000)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    best_name = None
    best_score = 0
    best_model = None

    print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1-macro':>10} {'AUC':>10}")
    print(f"  {'─'*58}")

    for name, pipeline in classifiers.items():
        acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")

        try:
            auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
            auc_mean = auc_scores.mean()
        except Exception:
            auc_mean = 0.0

        acc_mean = acc_scores.mean()
        f1_mean = f1_scores.mean()

        print(f"  {name:<25} {acc_mean:>10.4f} {f1_mean:>10.4f} {auc_mean:>10.4f}")

        results[name] = {
            "accuracy": round(float(acc_mean), 4),
            "accuracy_std": round(float(acc_scores.std()), 4),
            "f1_macro": round(float(f1_mean), 4),
            "auc": round(float(auc_mean), 4),
        }

        if acc_mean > best_score:
            best_score = acc_mean
            best_name = name
            best_model = pipeline

    # Train best model on full data
    best_model.fit(X, y)

    # Feature importance
    clf_step = best_model.named_steps["clf"]
    if hasattr(clf_step, "feature_importances_"):
        importances = dict(zip(FEATURE_KEYS, clf_step.feature_importances_))
    elif hasattr(clf_step, "coef_"):
        importances = dict(zip(FEATURE_KEYS, abs(clf_step.coef_[0])))
    else:
        importances = {}

    # Print feature importance
    if importances:
        print(f"\n  Feature Importance ({best_name}):")
        for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
            bar = "█" * int(imp * 50)
            print(f"    {feat:<22} {imp:.4f}  {bar}")

    # Confusion matrix on full data
    y_pred = best_model.predict(X)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=list(labels_map.values()),
                                   output_dict=True)

    return best_model, best_name, results, importances, cm, report


def main():
    print("=" * 60)
    print("  TASK 5 — Pre-Retrieval Regime Classifier")
    print("=" * 60)

    # Load spaCy
    print("\n  Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("  Downloading en_core_web_sm...")
        os.system(f"{sys.executable} -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    print("  ✓ spaCy loaded")

    # Load data
    print("\n  Loading evaluation data...")
    data = load_data()
    if not data:
        print("  ERROR: No data available. Run Tasks 0-3 first.")
        sys.exit(1)

    # Label regimes
    labels = []
    texts = []
    for item in data:
        score_range = item.get("score_range", 0)
        regime = 1 if score_range >= REGIME_THRESHOLD else 0  # 1=discriminative, 0=flat
        labels.append(regime)
        texts.append(item.get("query_text", ""))

    labels_map = {0: "Flat", 1: "Discriminative"}
    y = np.array(labels)
    print(f"  Flat: {sum(y == 0)}, Discriminative: {sum(y == 1)}")

    # Extract features
    print("\n  Extracting NLP features...")
    features_list = []
    for text in texts:
        feats = extract_features(text, nlp)
        features_list.append(feats)

    X = np.array([[f[k] for k in FEATURE_KEYS] for f in features_list])
    print(f"  Feature matrix: {X.shape}")

    # Train classifiers
    print("\n  Training classifiers (5-fold CV)...")
    best_model, best_name, results, importances, cm, report = \
        train_and_evaluate(X, y, labels_map)

    print(f"\n  ✓ Best model: {best_name} (acc={results[best_name]['accuracy']:.4f})")

    # Save model
    model_data = {
        "model": best_model,
        "feature_keys": FEATURE_KEYS,
        "labels_map": labels_map,
        "threshold": REGIME_THRESHOLD,
        "best_name": best_name,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  ✓ Model saved → {MODEL_PATH}")

    # Save results
    classifier_results = {
        "models": results,
        "best_model": best_name,
        "best_accuracy": results[best_name]["accuracy"],
        "feature_importance": {k: round(float(v), 4) for k, v in importances.items()} if importances else {},
        "class_distribution": {"flat": int(sum(y == 0)), "discriminative": int(sum(y == 1))},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "threshold": REGIME_THRESHOLD,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(classifier_results, f, indent=2)
    print(f"  ✓ Results saved → {RESULTS_PATH}")

    print(f"\n{'=' * 60}")
    print(f"  TASK 5 COMPLETE — Regime Classifier Trained")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
