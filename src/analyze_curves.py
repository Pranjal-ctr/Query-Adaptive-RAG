"""
Curve Analysis — Phase 1 Core Research Script

Loads saved score curves and produces the key research findings:

  1. Per-curve feature extraction (kurtosis, entropy, slope, range)
  2. K-means clustering to discover distribution shape taxonomy
  3. Cluster-to-intent correlation (chi-square test)
  4. Failure prediction experiment (if ground truth labels available)
  5. Cross-domain comparison (Yelp vs Legal) when both datasets present
  6. All plots saved to results/

This script produces the content of Section 3 (Analysis) in the paper.

Usage:
    python src/analyze_curves.py                              # Yelp only
    python src/analyze_curves.py --domain legal               # Legal only
    python src/analyze_curves.py --compare --legal_curves ... # Both domains
    python src/analyze_curves.py --gt eval/ground_truth.json  # With failure prediction
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from config import CURVE_DATA_FILE, RESULTS_DIR


# ─── Feature Extraction ──────────────────────────────────────────────────────

def compute_curve_features(scores: list) -> dict:
    """
    Compute 6 distribution shape features for a single score curve.
    Captures: peakedness, uniformity, decay rate, spread, top-chunk dominance, bimodality.
    """
    arr = np.array(scores, dtype=np.float64)

    if len(arr) < 4:
        return {k: 0.0 for k in ["kurtosis", "entropy", "slope", "score_range",
                                   "top1_dominance", "bimodality_coeff"]}

    kurt    = float(sp_stats.kurtosis(arr, fisher=True))
    shifted = arr - arr.min() + 1e-10
    probs   = shifted / shifted.sum()
    entropy = float(sp_stats.entropy(probs))

    n_top      = max(2, len(arr) // 2)
    top_scores = arr[:n_top]
    ranks      = np.arange(n_top, dtype=float)
    slope      = float(np.polyfit(ranks, top_scores, 1)[0]) if len(ranks) > 1 else 0.0

    score_range    = float(arr[0] - arr[-1])
    top1_dominance = float(arr[0] - arr[1]) if len(arr) > 1 else 0.0

    skew = float(sp_stats.skew(arr))
    n    = len(arr)
    bc_denom = kurt + (3 * ((n-1)**2) / ((n-2)*(n-3))) if n > 3 else 1.0
    bimodality_coeff = (skew**2 + 1) / (bc_denom + 1e-10)

    return {
        "kurtosis":         round(kurt, 4),
        "entropy":          round(entropy, 4),
        "slope":            round(slope, 6),
        "score_range":      round(score_range, 4),
        "top1_dominance":   round(top1_dominance, 4),
        "bimodality_coeff": round(bimodality_coeff, 4),
    }


# ─── Clustering ──────────────────────────────────────────────────────────────

FEATURE_KEYS = ["kurtosis", "entropy", "slope", "score_range",
                "top1_dominance", "bimodality_coeff"]


def cluster_curves(features_list, n_clusters=4):
    X = np.array([[f[k] for k in FEATURE_KEYS] for f in features_list])
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km      = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels  = km.fit_predict(X_scaled)
    return labels, X_scaled, scaler, km


def name_cluster(centroid_dict):
    kurt  = centroid_dict["kurtosis"]
    ent   = centroid_dict["entropy"]
    bm    = centroid_dict["bimodality_coeff"]
    slope = centroid_dict["slope"]
    top1  = centroid_dict["top1_dominance"]

    if kurt > 2.0 and top1 > 0.02:
        return "Peaked"
    elif bm > 0.555:
        return "Bimodal"
    elif ent > 2.65 and abs(slope) < 0.002:
        return "Flat"
    elif slope < -0.005:
        return "Steep-decay"
    else:
        return "Gradual-decay"


# ─── Printing ────────────────────────────────────────────────────────────────

def print_intent_summary(features_list):
    print(f"\n  SUMMARY BY INTENT:")
    print(f"  {'─'*75}")
    intents = sorted(set(f["intent"] for f in features_list))
    for intent in intents:
        g  = [f for f in features_list if f["intent"] == intent]
        k  = [f["kurtosis"]    for f in g]
        e  = [f["entropy"]     for f in g]
        r  = [f["score_range"] for f in g]
        b  = [f["bimodality_coeff"] for f in g]
        print(f"  {intent:<16} n={len(g):>3}"
              f"  kurtosis={np.mean(k):>6.3f}±{np.std(k):.3f}"
              f"  entropy={np.mean(e):>6.3f}"
              f"  range={np.mean(r):>7.4f}"
              f"  bimodality={np.mean(b):>5.3f}")


def print_cluster_intent_matrix(features_list, labels, cluster_names):
    intents     = sorted(set(f["intent"] for f in features_list))
    cluster_ids = sorted(set(labels))

    print(f"\n  CLUSTER x INTENT CROSS-TABULATION")
    print(f"  {'Cluster':<16}", end="")
    for intent in intents:
        print(f"  {intent:>12}", end="")
    print(f"  {'TOTAL':>7}")
    print(f"  {'─' * (16 + 14*len(intents) + 9)}")

    contingency = []
    for cid in cluster_ids:
        name = cluster_names.get(cid, f"C{cid}")
        print(f"  {name:<16}", end="")
        row = []
        for intent in intents:
            count = sum(1 for f, l in zip(features_list, labels)
                        if l == cid and f["intent"] == intent)
            print(f"  {count:>12}", end="")
            row.append(count)
        print(f"  {sum(row):>7}")
        contingency.append(row)

    contingency = np.array(contingency)
    if contingency.min() >= 1 and len(intents) > 1:
        chi2, p, dof, _ = sp_stats.chi2_contingency(contingency)
        print(f"\n  Chi-square: chi2={chi2:.3f}, df={dof}, p={p:.4f}")
        if p < 0.05:
            print(f"  FINDING: Cluster shape significantly associated with intent (p<0.05)")
        else:
            print(f"  No significant association (p={p:.3f}) — domain may be too flat.")
            print(f"  Expected for Yelp. Re-run on legal domain for paper's main result.")


def print_summary_table(features_list, labels, cluster_names):
    print(f"\n  {'Query':<42} {'Intent':<14} {'Cluster':<15}"
          f"  {'Kurtosis':>9} {'Entropy':>8} {'Range':>7}")
    print(f"  {'─'*100}")
    for f, lbl in zip(features_list, labels):
        name = cluster_names.get(lbl, f"C{lbl}")
        print(f"  {f['query_text']:<42} {f['intent']:<14} {name:<15}"
              f"  {f['kurtosis']:>9.3f} {f['entropy']:>8.3f} {f['score_range']:>7.4f}")


# ─── Failure Prediction ──────────────────────────────────────────────────────

def failure_prediction_experiment(features_list, ground_truth_path):
    print(f"\n  FAILURE PREDICTION EXPERIMENT (RQ3)")
    print(f"  {'─'*50}")

    with open(ground_truth_path) as f:
        gt = {item["query_id"]: item["correct"] for item in json.load(f)}

    matched = [(f, gt[f["query_id"]]) for f in features_list if f["query_id"] in gt]
    if len(matched) < 20:
        print(f"  Only {len(matched)} matched. Need 20+ for reliable AUC.")
        return

    X = np.array([[f[k] for k in FEATURE_KEYS] for f, _ in matched])
    y = np.array([label for _, label in matched])

    X_scaled = StandardScaler().fit_transform(X)
    clf  = LogisticRegression(random_state=42, max_iter=1000)
    aucs = cross_val_score(clf, X_scaled, y, cv=5, scoring="roc_auc")

    print(f"  Logistic regression AUC (5-fold CV): {aucs.mean():.3f} ± {aucs.std():.3f}")
    if aucs.mean() > 0.65:
        print(f"  FINDING: Shape features predict retrieval failure (AUC={aucs.mean():.3f})")
    elif aucs.mean() > 0.55:
        print(f"  Weak signal AUC={aucs.mean():.3f} — re-run on legal domain.")
    else:
        print(f"  No signal (AUC~0.5) — distributions too flat (Yelp effect, expected).")

    # Feature importance
    clf.fit(X_scaled, y)
    coefs = dict(zip(FEATURE_KEYS, abs(clf.coef_[0])))
    print(f"\n  Feature importance:")
    for feat, imp in sorted(coefs.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 25)
        print(f"    {feat:<22} {imp:.4f}  {bar}")


# ─── Cross-Domain Comparison ─────────────────────────────────────────────────

def cross_domain_comparison(yelp_feats, legal_feats):
    print(f"\n  CROSS-DOMAIN COMPARISON")
    print(f"  {'─'*65}")
    metrics = ["kurtosis", "entropy", "score_range", "top1_dominance"]
    print(f"  {'Metric':<22} {'Yelp mean±std':<22} {'Legal mean±std':<22} {'p-value':>10}")
    print(f"  {'─'*80}")
    for m in metrics:
        yv = [f[m] for f in yelp_feats]
        lv = [f[m] for f in legal_feats]
        _, p = sp_stats.ttest_ind(yv, lv) if (len(yv) > 2 and len(lv) > 2) else (0, 1.0)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {m:<22} {np.mean(yv):>6.3f}±{np.std(yv):.3f}      "
              f"{np.mean(lv):>6.3f}±{np.std(lv):.3f}      {p:>8.4f} {sig}")
    print(f"\n  Yelp n={len(yelp_feats)}, Legal n={len(legal_feats)}")
    print(f"  *** p<0.001  ** p<0.01  * p<0.05  ns=not significant")


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_curves_by_cluster(curves, features_list, labels, cluster_names, domain, save_dir):
    cluster_ids = sorted(set(labels))
    fig, axes   = plt.subplots(1, len(cluster_ids), figsize=(5*len(cluster_ids), 4))
    if len(cluster_ids) == 1:
        axes = [axes]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for ax, cid in zip(axes, cluster_ids):
        name    = cluster_names.get(cid, f"Cluster {cid}")
        indices = [i for i, l in enumerate(labels) if l == cid]
        color   = colors[cid % len(colors)]

        for idx in indices[:5]:
            ax.plot(curves[idx]["similarity_scores"], alpha=0.35, linewidth=1.2, color=color)

        all_scores = [curves[i]["similarity_scores"] for i in indices]
        min_len    = min(len(s) for s in all_scores)
        mean_curve = np.mean([s[:min_len] for s in all_scores], axis=0)
        ax.plot(mean_curve, linewidth=2.5, color=color, label="Mean")

        intent_counts = {}
        for i in indices:
            k = features_list[i]["intent"]
            intent_counts[k] = intent_counts.get(k, 0) + 1
        top_intent = max(intent_counts, key=intent_counts.get)

        ax.set_title(f"{name}\nn={len(indices)} | {top_intent}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Rank", fontsize=9)
        ax.set_ylabel("Similarity", fontsize=9)

    plt.suptitle(f"Distribution Shape Taxonomy — {domain}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(save_dir, f"taxonomy_{domain.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_distributions(features_list, domain, save_dir):
    intents = sorted(set(f["intent"] for f in features_list))
    keys    = ["kurtosis", "entropy", "score_range", "top1_dominance"]
    labels  = ["Kurtosis", "Entropy", "Score Range", "Top-1 Dominance"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, key, label in zip(axes, keys, labels):
        data = [[f[key] for f in features_list if f["intent"] == intent]
                for intent in intents]
        valid = [(i, d) for i, d in zip(intents, data) if d]
        if not valid:
            continue
        vi, vd = zip(*valid)
        parts = ax.violinplot(vd, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#2196F3")
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(vi)+1))
        ax.set_xticklabels(vi, rotation=20, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold")

    plt.suptitle(f"Feature Distributions by Intent — {domain}", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(save_dir, f"features_by_intent_{domain.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cross_domain(yelp_feats, legal_feats, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, feats, domain, color in [
        (axes[0], yelp_feats,  "Yelp (informal reviews)", "#FF5722"),
        (axes[1], legal_feats, "Legal (CUAD contracts)",  "#2196F3"),
    ]:
        ranges = [f["score_range"] for f in feats]
        ax.hist(ranges, bins=20, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(ranges), color="black", linestyle="--", linewidth=1.5,
                   label=f"mean={np.mean(ranges):.3f}")
        ax.set_title(domain, fontsize=11, fontweight="bold")
        ax.set_xlabel("Score Range (max - min similarity)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
    plt.suptitle("Score Range Distribution: Yelp vs Legal\n"
                 "(Higher = more discriminative retrieval signal)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = os.path.join(save_dir, "cross_domain_score_range.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_curves(path):
    if not os.path.exists(path):
        print(f"  No curve data at {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def analyze_domain(curves, domain, save_dir, n_clusters=4, gt_path=None):
    if len(curves) < 10:
        print(f"  Only {len(curves)} curves. Need 10+ for analysis.")
        return []

    print(f"\n{'='*65}")
    print(f"  CURVE ANALYSIS — {domain} — {len(curves)} distributions")
    print(f"{'='*65}")

    features_list = []
    for curve in curves:
        feats = compute_curve_features(curve["similarity_scores"])
        feats["query_id"]   = curve["query_id"]
        feats["query_text"] = curve["query_text"][:50]
        feats["intent"]     = curve.get("intent", "unknown")
        features_list.append(feats)

    print_intent_summary(features_list)

    n_c = max(2, min(n_clusters, len(curves) // 5))
    labels, X_scaled, scaler, km = cluster_curves(features_list, n_c)

    cluster_names = {}
    for cid in range(n_c):
        centroid = scaler.inverse_transform(km.cluster_centers_[cid].reshape(1, -1))[0]
        cluster_names[cid] = name_cluster(dict(zip(FEATURE_KEYS, centroid)))

    print_summary_table(features_list, labels, cluster_names)
    print_cluster_intent_matrix(features_list, labels, cluster_names)

    plot_curves_by_cluster(curves, features_list, labels, cluster_names, domain, save_dir)
    plot_feature_distributions(features_list, domain, save_dir)

    if gt_path:
        failure_prediction_experiment(features_list, gt_path)

    out = os.path.join(save_dir, f"curve_features_{domain.lower()}.json")
    with open(out, "w") as f:
        json.dump(features_list, f, indent=2)
    print(f"  Features saved: {out}")

    return features_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",       default="yelp")
    parser.add_argument("--curves",       default=None)
    parser.add_argument("--compare",      action="store_true")
    parser.add_argument("--yelp_curves",  default=None)
    parser.add_argument("--legal_curves", default=None)
    parser.add_argument("--gt",           default=None)
    parser.add_argument("--clusters",     type=int, default=4)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.compare:
        yelp_path  = args.yelp_curves  or CURVE_DATA_FILE
        legal_path = args.legal_curves
        if not legal_path or not os.path.exists(legal_path):
            print("  --compare requires --legal_curves <path>")
            print("  Complete CUAD ingestion first (Week 5 in your roadmap).")
            sys.exit(1)
        yf = analyze_domain(load_curves(yelp_path),  "Yelp",  RESULTS_DIR, args.clusters)
        lf = analyze_domain(load_curves(legal_path), "Legal", RESULTS_DIR, args.clusters)
        if yf and lf:
            cross_domain_comparison(yf, lf)
            plot_cross_domain(yf, lf, RESULTS_DIR)
    else:
        curves_path = args.curves or CURVE_DATA_FILE
        analyze_domain(load_curves(curves_path), args.domain,
                       RESULTS_DIR, args.clusters, args.gt)

    print(f"\n  Done. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()