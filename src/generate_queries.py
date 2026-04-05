"""
Query Dataset Generator — Research Grade
Generates 300 balanced, diverse queries from Yelp reviews using Mistral 7B (Ollama).

Produces exactly:
  75 FACTOID queries     (simple, single-aspect, peaked curve expected)
  75 COMPARATIVE queries (comparing two things, staircase curve expected)
  75 AGGREGATION queries (summarizing across reviews, flat curve expected)
  75 EXPLORATORY queries (open discovery, noisy curve expected)

Total: 300 queries with balanced intent distribution for research analysis.

Usage:
    cd d:/Programming/AI/RAG/food_domain_rag
    python src/generate_queries.py
"""

import json
import os
import sys
import random
import time
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from config import DATA_DIR, DATA_FILE

# ── Output path ──
OUTPUT_FILE = os.path.join(DATA_DIR, "sample_queries_research.txt")
OUTPUT_JSON = os.path.join(DATA_DIR, "sample_queries_research.json")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

# ── Target counts per intent ──
TARGET_PER_INTENT = 75
INTENTS = ["factoid", "comparative", "aggregation", "exploratory"]

# ── How many reviews to sample per batch ──
REVIEWS_PER_BATCH = 8
QUERIES_PER_BATCH = 5  # Ask for 5 queries per call, loop until target reached


# ─── Prompts ──────────────────────────────────────────────────────────────────

INTENT_CONFIGS = {

    "factoid": {
        "description": "Simple, single-aspect questions with one correct answer. "
                       "These ask about one specific thing: quality of one dish, "
                       "one aspect of service, whether one place has something.",
        "examples": [
            "Is the sushi fresh at this restaurant?",
            "Does this place have outdoor seating?",
            "How is the portion size at this cafe?",
            "Is the pizza crust thin or thick here?",
            "Does this restaurant have vegetarian options?",
            "How long are the wait times typically?",
            "Is parking available near this place?",
            "Is the coffee strong or mild here?",
            "Does this place get noisy on weekends?",
            "Is the steak cooked to order at this steakhouse?",
        ],
        "anti_examples": [
            "What are the best restaurants?",       # too vague
            "Tell me about the food",               # not a question
            "Compare the service here vs elsewhere", # that's comparative
        ],
        "curve_hint": "PEAKED — one dominant relevant chunk expected",
    },

    "comparative": {
        "description": "Questions that explicitly compare two things: two dishes, "
                       "two aspects, two types of restaurants, lunch vs dinner, "
                       "weekday vs weekend, dine-in vs takeout.",
        "examples": [
            "Is the lunch menu better or worse than dinner here?",
            "Do chain restaurants get better reviews than local spots?",
            "Which is more important to diners: food quality or service speed?",
            "Is takeout quality as good as dining in?",
            "Are newer restaurants rated higher than established ones?",
            "Do expensive restaurants get better food reviews than budget ones?",
            "Is weekend service better or worse than weekday service?",
            "Which gets more complaints: slow service or cold food?",
            "Do places with small menus get better reviews than large menus?",
            "Is breakfast or dinner rated higher at brunch spots?",
        ],
        "anti_examples": [
            "What is good food?",               # not comparative
            "Tell me about service",            # single aspect
            "What are popular dishes?",         # aggregation
        ],
        "curve_hint": "STAIRCASE — two groups of relevant chunks expected",
    },

    "aggregation": {
        "description": "Questions that require synthesizing information ACROSS MANY reviews. "
                       "These ask about patterns, trends, common opinions, overall sentiment, "
                       "or what people generally think about something.",
        "examples": [
            "What do people generally think about the ambience at local restaurants?",
            "What are the most common complaints about dining out?",
            "Summarize what reviewers say about portion sizes overall",
            "What food aspects do customers praise most frequently?",
            "What patterns appear in five-star restaurant reviews?",
            "What are the most mentioned dishes in positive reviews?",
            "How do customers generally describe the value for money?",
            "What themes appear repeatedly in one-star reviews?",
            "What makes customers become repeat visitors?",
            "What service behaviors do reviewers mention most often?",
        ],
        "anti_examples": [
            "Is the sushi fresh?",              # that's factoid
            "Which place has better service?",  # that's comparative
            "Tell me about one restaurant",     # not aggregation
        ],
        "curve_hint": "FLAT — many equally relevant chunks expected",
    },

    "exploratory": {
        "description": "Open-ended discovery questions with no single correct answer. "
                       "These seek recommendations, hidden gems, unusual experiences, "
                       "or ask the system to surface surprising or unexpected information.",
        "examples": [
            "What are some hidden gem restaurants that people discovered by accident?",
            "What unusual or quirky food items have reviewers been surprised by?",
            "Tell me about memorable dining experiences people have described",
            "What restaurants have dramatically exceeded customer expectations?",
            "What are some underrated places that deserve more attention?",
            "What creative or unusual menu items have people loved?",
            "Describe some unexpectedly good experiences at average-looking places",
            "What restaurants have the most devoted loyal customer base?",
            "What dining experiences do people describe as life-changing?",
            "What places do locals recommend that tourists miss?",
        ],
        "anti_examples": [
            "What is the best pizza?",              # too specific, factoid
            "Compare brunch spots",                 # comparative
            "What do reviews generally say?",       # aggregation
        ],
        "curve_hint": "NOISY/MIXED — unpredictable distribution expected",
    },
}


BATCH_PROMPT_TEMPLATE = """You are generating a research dataset of information retrieval queries for a food review system.

Your task: Generate exactly {n} queries of type: {intent_type}

INTENT DEFINITION:
{description}

GOOD EXAMPLES of this query type:
{examples}

BAD EXAMPLES (do NOT generate these — wrong type):
{anti_examples}

CONTEXT — here are some real restaurant reviews to ground your queries:
---
{reviews}
---

STRICT RULES:
1. Generate EXACTLY {n} queries, numbered 1 to {n}
2. Every query MUST match the intent type: {intent_type}
3. Each query must be DIFFERENT from the examples above — no copying
4. Each query must be grounded in realistic restaurant/food scenarios
5. Vary the topics: food, service, price, ambience, specific dishes, wait times, parking, etc.
6. Queries should be 8-25 words long
7. Do NOT generate meta-questions about reviews ("what do reviews say about reviews")
8. Do NOT repeat similar queries — make each one distinct
9. Output ONLY the numbered list, no explanations, no headers, no extra text

OUTPUT FORMAT (exactly this, nothing else):
1. [query here]
2. [query here]
3. [query here]
4. [query here]
5. [query here]"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_reviews(data_file: str, n: int = 200) -> list:
    """Load a sample of reviews from the Yelp JSONL file."""
    reviews = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                if len(text) > 50:
                    reviews.append(text)
            except json.JSONDecodeError:
                continue
            if len(reviews) >= n:
                break
    random.shuffle(reviews)
    return reviews


def sample_reviews(reviews: list, n: int = REVIEWS_PER_BATCH) -> str:
    """Pick n random reviews and format them for the prompt."""
    sampled = random.sample(reviews, min(n, len(reviews)))
    parts = []
    for i, r in enumerate(sampled, 1):
        # Truncate long reviews to keep prompt manageable
        truncated = r[:300] + "..." if len(r) > 300 else r
        parts.append(f"Review {i}: {truncated}")
    return "\n\n".join(parts)


def call_mistral(prompt: str, temperature: float = 0.8) -> str:
    """Call Mistral via Ollama API."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": 400,
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("  ✗ Cannot connect to Ollama. Is it running?")
        print("    Run: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Ollama error: {e}")
        return ""


def parse_queries(raw_output: str) -> list:
    """
    Extract numbered queries from Mistral's output.
    Handles messy formatting Mistral sometimes produces.
    """
    queries = []
    lines = raw_output.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match patterns like "1. query" or "1) query" or "1 query"
        match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if match:
            query = match.group(1).strip()
            # Basic quality filters
            if len(query) < 8:
                continue
            if len(query) > 200:
                continue
            if query.lower().startswith("query"):
                continue  # Mistral sometimes outputs "Query: ..."
            queries.append(query)

    return queries


def deduplicate(queries: list, existing: list) -> list:
    """Remove queries too similar to existing ones (simple substring check)."""
    existing_lower = {q.lower() for q in existing}
    unique = []
    for q in queries:
        q_lower = q.lower()
        # Skip near-duplicates
        is_dup = False
        for ex in existing_lower:
            # If 60%+ of words overlap, consider duplicate
            q_words = set(q_lower.split())
            ex_words = set(ex.split())
            if len(q_words) > 0:
                overlap = len(q_words & ex_words) / len(q_words)
                if overlap > 0.7:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(q)
            existing_lower.add(q_lower)
    return unique


# ─── Generation Loop ──────────────────────────────────────────────────────────

def generate_for_intent(
    intent: str,
    target: int,
    reviews: list,
    verbose: bool = True,
) -> list:
    """Generate `target` queries of a given intent type."""
    config = INTENT_CONFIGS[intent]
    collected = []
    attempts = 0
    max_attempts = target * 4  # Safety limit

    print(f"\n  Generating {target} {intent.upper()} queries...")

    while len(collected) < target and attempts < max_attempts:
        attempts += QUERIES_PER_BATCH

        # Build prompt
        prompt = BATCH_PROMPT_TEMPLATE.format(
            n=QUERIES_PER_BATCH,
            intent_type=intent.upper(),
            description=config["description"],
            examples="\n".join(f"  - {e}" for e in config["examples"]),
            anti_examples="\n".join(f"  - {e}" for e in config["anti_examples"]),
            reviews=sample_reviews(reviews),
        )

        # Call Mistral with slightly varying temperature for diversity
        temperature = 0.7 + random.uniform(0, 0.3)
        raw = call_mistral(prompt, temperature=temperature)

        if not raw:
            continue

        # Parse and deduplicate
        new_queries = parse_queries(raw)
        new_queries = deduplicate(new_queries, collected)
        collected.extend(new_queries)

        if verbose:
            print(f"    [{len(collected):>3}/{target}] +{len(new_queries)} new  "
                  f"(attempt {attempts})", end="\r")

        # Small delay to avoid overwhelming Ollama
        time.sleep(0.5)

    # Trim to exact target
    collected = collected[:target]

    print(f"    ✓ {len(collected)} {intent} queries generated          ")
    return collected


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  QUERY DATASET GENERATOR")
    print(f"  Target: {TARGET_PER_INTENT * len(INTENTS)} queries "
          f"({TARGET_PER_INTENT} per intent × {len(INTENTS)} intents)")
    print(f"  Model:  {MODEL} via Ollama")
    print("=" * 60)

    # Load reviews
    print(f"\n  Loading reviews from {DATA_FILE}...")
    reviews = load_reviews(DATA_FILE, n=500)
    print(f"  Loaded {len(reviews)} reviews for grounding")

    # Generate per intent
    all_queries = {}
    for intent in INTENTS:
        queries = generate_for_intent(intent, TARGET_PER_INTENT, reviews)
        all_queries[intent] = queries

    # ── Save .txt (for --batch mode) ──
    total = sum(len(v) for v in all_queries.values())
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Research Query Dataset — {total} queries\n")
        f.write(f"# Generated by Mistral 7B from Yelp reviews\n")
        f.write(f"# Format: plain queries, one per line, usable with --batch\n\n")

        for intent in INTENTS:
            f.write(f"\n# ── {intent.upper()} queries ({len(all_queries[intent])}) ──\n")
            for q in all_queries[intent]:
                f.write(q + "\n")

    # ── Save .json (with intent labels, for analysis) ──
    json_data = []
    for intent in INTENTS:
        for q in all_queries[intent]:
            json_data.append({
                "query": q,
                "intent_ground_truth": intent,
                "curve_expectation": INTENT_CONFIGS[intent]["curve_hint"],
            })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  GENERATION COMPLETE")
    print(f"  {'─' * 40}")
    for intent in INTENTS:
        print(f"  {intent:<15} {len(all_queries[intent]):>3} queries")
    print(f"  {'─' * 40}")
    print(f"  TOTAL:          {total:>3} queries")
    print(f"\n  Saved to:")
    print(f"    {OUTPUT_FILE}")
    print(f"    {OUTPUT_JSON}")
    print(f"\n  Next step:")
    print(f"    python src/rag_query.py --batch {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()