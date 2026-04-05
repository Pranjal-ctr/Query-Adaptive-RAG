"""
Cross-Domain Comparison — Task 4
Loads Yelp and Legal results side-by-side, computes statistical comparison,
generates comparison table + visualization.

Usage:
    cd d:/Programming/AI/RAG/food_domain_rag
    python src/compare_domains.py
"""

import io
import json
import os
import sys

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('cp'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from config import RESULTS_DIR, PROJECT_ROOT

# ─── Paths ────────────────────────────────────────────────────────────────────
YELP_EVAL = os.path.join(RESULTS_DIR, "yelp", "evaluation_report.json")
LEGAL_EVAL = os.path.join(RESULTS_DIR, "legal", "evaluation_report.json")
YELP_FEATURES = os.path.join(RESULTS_DIR, "yelp", "curve_features.json")
LEGAL_FEATURES = os.path.join(RESULTS_DIR, "legal", "curve_features.json")

OUTPUT_TABLE = os.path.join(RESULTS_DIR, "comparison_table.json")
OUTPUT_PLOT = os.path.join(RESULTS_DIR, "cross_domain_comparison.png")


def load_json(path):
    if not os.path.exists(path):
        print(f"  ✗ Missing: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_comparison(yelp_data, legal_data):
    """Compute per-metric comparison with t-tests between domains."""
    metrics = [
        "entropy", "kurtosis", "score_range", "max_similarity",
        "mean_similarity", "drop_ratio_k5", "rel_precision_k5",
        "norm_confidence_k5", "drop_ratio_k3", "drop_ratio_k10",
    ]

    comparison = []
    for m in metrics:
        yv = [r.get(m) for r in yelp_data if r.get(m) is not None]
        lv = [r.get(m) for r in legal_data if r.get(m) is not None]

        if not yv or not lv:
            continue

        t_stat, p_val = sp_stats.ttest_ind(yv, lv) if (len(yv) > 2 and len(lv) > 2) else (0, 1.0)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))

        row = {
            "metric": m,
            "yelp_mean": round(float(np.mean(yv)), 4),
            "yelp_std": round(float(np.std(yv)), 4),
            "legal_mean": round(float(np.mean(lv)), 4),
            "legal_std": round(float(np.std(lv)), 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "significance": sig,
            "yelp_n": len(yv),
            "legal_n": len(lv),
        }
        comparison.append(row)

    return comparison


def print_comparison_table(comparison):
    """Pretty print the comparison table."""
    print(f"\n  {'='*90}")
    print(f"  CROSS-DOMAIN COMPARISON: Yelp vs Legal")
    print(f"  {'='*90}")
    print(f"  {'Metric':<22} {'Yelp (mean±std)':<20} {'Legal (mean±std)':<20} {'p-value':>10} {'Sig':>5}")
    print(f"  {'-'*82}")

    for row in comparison:
        yelp_str = f"{row['yelp_mean']:.4f}±{row['yelp_std']:.4f}"
        legal_str = f"{row['legal_mean']:.4f}±{row['legal_std']:.4f}"
        print(f"  {row['metric']:<22} {yelp_str:<20} {legal_str:<20} {row['p_value']:>10.6f} {row['significance']:>5}")

    print(f"  {'-'*82}")
    print(f"  *** p<0.001  ** p<0.01  * p<0.05  ns=not significant")


def plot_comparison(yelp_data, legal_data, comparison, save_path):
    """Generate a multi-panel comparison plot."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#1a1a2e')

    plot_metrics = [
        ("entropy", "Score Entropy", "Higher = more uniform"),
        ("kurtosis", "Kurtosis", "Higher = more peaked"),
        ("score_range", "Score Range", "Higher = more discriminative"),
        ("drop_ratio_k5", "Drop Ratio @K5", "Higher = flatter curve"),
        ("norm_confidence_k5", "Norm. Confidence", "Higher = more confident"),
        ("max_similarity", "Max Similarity", "Highest retrieval score"),
    ]

    colors_yelp = "#FF6B6B"
    colors_legal = "#4ECDC4"

    for ax, (metric, title, subtitle) in zip(axes.flatten(), plot_metrics):
        ax.set_facecolor('#16213e')

        yv = [r.get(metric) for r in yelp_data if r.get(metric) is not None]
        lv = [r.get(metric) for r in legal_data if r.get(metric) is not None]

        if not yv or not lv:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color="white",
                    transform=ax.transAxes)
            continue

        # Find p-value
        row = next((c for c in comparison if c["metric"] == metric), None)
        sig = row["significance"] if row else "?"

        # Histogram overlay
        bins = 25
        ax.hist(yv, bins=bins, alpha=0.6, color=colors_yelp, edgecolor="white",
                linewidth=0.5, label=f"Yelp (μ={np.mean(yv):.3f})", density=True)
        ax.hist(lv, bins=bins, alpha=0.6, color=colors_legal, edgecolor="white",
                linewidth=0.5, label=f"Legal (μ={np.mean(lv):.3f})", density=True)

        # Mean lines
        ax.axvline(np.mean(yv), color=colors_yelp, linestyle="--", linewidth=2)
        ax.axvline(np.mean(lv), color=colors_legal, linestyle="--", linewidth=2)

        ax.set_title(f"{title}\n{subtitle} [{sig}]", fontsize=11, fontweight="bold",
                     color="white", pad=8)
        ax.legend(fontsize=8, loc="upper right", facecolor="#16213e",
                  edgecolor="#333", labelcolor="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333")

    plt.suptitle("Cross-Domain Comparison: Yelp (Food Reviews) vs Legal (Statutes)",
                 fontsize=14, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='#1a1a2e')
    plt.close()
    print(f"  [OK] Plot saved -> {save_path}")


def main():
    print("=" * 60)
    print("  TASK 4 — Cross-Domain Comparison")
    print("=" * 60)

    # Load data
    yelp_eval = load_json(YELP_EVAL)
    legal_eval = load_json(LEGAL_EVAL)

    if not yelp_eval or not legal_eval:
        print("  ERROR: Need both Yelp and Legal evaluation reports.")
        print("  Run Tasks 0-3 first.")
        sys.exit(1)

    print(f"  Yelp:  {len(yelp_eval)} queries")
    print(f"  Legal: {len(legal_eval)} queries")

    # Compute comparison
    comparison = compute_comparison(yelp_eval, legal_eval)

    # Print table
    print_comparison_table(comparison)

    # Save table
    with open(OUTPUT_TABLE, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  [OK] Table saved -> {OUTPUT_TABLE}")

    # Plot
    plot_comparison(yelp_eval, legal_eval, comparison, OUTPUT_PLOT)

    print(f"\n{'=' * 60}")
    print(f"  TASK 4 COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
