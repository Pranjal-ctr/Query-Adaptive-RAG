"""
Score Logger — Saves similarity score curves to disk for research analysis.

Every retrieval produces a vector of similarity scores. These are the raw
research data for Phase 1 (distribution shape taxonomy). This module handles
persistent storage of those curves.

CHANGES FROM PREVIOUS VERSION:
- JSON no longer re-reads entire file on every write (was O(n) per query)
- Added load_and_verify() to confirm scores are in correct 0.6-0.8 range
- Added export_for_analysis() to dump clean numpy-ready arrays
"""

import csv
import json
import os
import time
from typing import List, Optional

from config import SCORE_LOG_FILE, CURVE_DATA_FILE, SCORE_CURVES_DIR


class ScoreLogger:
    """Logs retrieval similarity scores to CSV and JSON for later analysis."""

    def __init__(self, csv_path: str = SCORE_LOG_FILE, json_path: str = CURVE_DATA_FILE):
        self.csv_path = csv_path
        self.json_path = json_path
        self._query_counter = 0
        self._buffer = []  # In-memory buffer, flushed to disk

        # Initialize CSV with header if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "query_id", "timestamp", "query_text", "intent",
                    "num_candidates", "num_used", "similarity_scores",
                    "max_score", "min_score", "mean_score", "score_range"
                ])

        # Load existing JSON into buffer (avoids full re-read on every write)
        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as f:
                self._buffer = json.load(f)
        else:
            self._buffer = []
            self._flush_json()

    def log(
        self,
        query_text: str,
        intent: str,
        similarity_scores: List[float],
        num_used: int,
        response: Optional[str] = None,
    ) -> str:
        """
        Log a single retrieval event.

        Args:
            query_text: The user's original query
            intent: Classified intent type
            similarity_scores: Full list of CONVERTED similarity scores (0.0-1.0),
                               already processed by retrieve_with_scores()
            num_used: How many chunks were actually used for generation
            response: The LLM's generated response (optional)

        Returns:
            query_id for reference
        """
        self._query_counter += 1
        query_id = f"q_{int(time.time())}_{self._query_counter:04d}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Ensure scores are plain Python floats and sorted descending
        scores = sorted([float(s) for s in similarity_scores], reverse=True)

        # Sanity check: warn if scores look like raw L2 distances
        if scores and max(scores) < 0.3:
            print(f"  WARNING: max similarity={max(scores):.4f} is very low.")
            print(f"    Check that retrieve_with_scores() is applying the L2->cosine conversion.")

        # CSV row
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                query_id,
                timestamp,
                query_text,
                intent,
                len(scores),
                num_used,
                json.dumps(scores),
                round(max(scores), 6) if scores else 0,
                round(min(scores), 6) if scores else 0,
                round(sum(scores) / len(scores), 6) if scores else 0,
                round(max(scores) - min(scores), 6) if scores else 0,
            ])

        # JSON (full detail) — append to buffer, flush to disk
        entry = {
            "query_id": query_id,
            "timestamp": timestamp,
            "query_text": query_text,
            "intent": intent,
            "similarity_scores": [round(s, 6) for s in scores],
            "num_candidates": len(scores),
            "num_used": num_used,
            "response": response,
        }
        self._buffer.append(entry)
        self._flush_json()

        return query_id

    def _flush_json(self):
        """Write buffer to disk."""
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self._buffer, f, indent=2, ensure_ascii=False)

    def get_all_curves(self) -> list:
        """Return all saved score curves (from in-memory buffer)."""
        return self._buffer

    def get_curve_count(self) -> int:
        """How many curves have been logged so far."""
        return len(self._buffer)

    def load_and_verify(self):
        """
        Diagnostic: print score range stats to confirm scores are correctly
        converted (should be 0.6-0.8 for Yelp, not 0.05-0.15).
        Call this after your batch run to confirm data quality.
        """
        if not self._buffer:
            print("  No curves logged yet.")
            return

        all_maxes = [max(c["similarity_scores"]) for c in self._buffer]
        all_mins  = [min(c["similarity_scores"])  for c in self._buffer]
        all_ranges = [max(c["similarity_scores"]) - min(c["similarity_scores"])
                      for c in self._buffer]

        print(f"\n  === Score Verification ===")
        print(f"  Curves logged:      {len(self._buffer)}")
        print(f"  Max sim (mean):     {sum(all_maxes)/len(all_maxes):.4f}  "
              f"(expect ~0.70-0.80 for Yelp, ~0.80-0.95 for legal)")
        print(f"  Min sim (mean):     {sum(all_mins)/len(all_mins):.4f}")
        print(f"  Score range (mean): {sum(all_ranges)/len(all_ranges):.4f}  "
              f"(expect ~0.03 for Yelp, ~0.20+ for legal)")
        print(f"  ==========================\n")

    def export_for_analysis(self, output_path: str = None) -> list:
        """
        Export clean list of dicts ready for analyze_curves.py.
        Each dict: query_id, query_text, intent, similarity_scores (list of floats)
        """
        clean = [
            {
                "query_id": c["query_id"],
                "query_text": c["query_text"],
                "intent": c["intent"],
                "similarity_scores": c["similarity_scores"],
            }
            for c in self._buffer
        ]

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(clean, f, indent=2)
            print(f"  Exported {len(clean)} curves to {output_path}")

        return clean