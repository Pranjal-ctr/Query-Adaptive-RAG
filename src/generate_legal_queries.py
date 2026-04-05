"""
Legal Domain Query Generator — Task 2
Generates 100 balanced legal-domain queries (25 per intent × 4 intents)
using Mistral 7B via Ollama, grounded from legal statute/contract chunks.

Intents:
  25 FACTOID       — specific clause or penalty questions
  25 COMPARATIVE   — comparing legal provisions, jurisdictions, or clauses
  25 AGGREGATION   — summarizing patterns across legal texts
  25 EXPLORATORY   — open-ended legal discovery questions

Usage:
    cd d:/Programming/AI/RAG/food_domain_rag
    python src/generate_legal_queries.py
"""

import json
import os
import sys
import random
import time
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from config import PROJECT_ROOT

# ── Paths ──
LEGAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "legal")
LEGAL_CHUNKS_FILE = os.path.join(LEGAL_DATA_DIR, "chunks.jsonl")
OUTPUT_JSON = os.path.join(LEGAL_DATA_DIR, "legal_queries.json")
OUTPUT_TXT = os.path.join(LEGAL_DATA_DIR, "legal_queries.txt")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

# ── Target counts ──
TARGET_PER_INTENT = 25
INTENTS = ["factoid", "comparative", "aggregation", "exploratory"]
CHUNKS_PER_BATCH = 6
QUERIES_PER_BATCH = 5


# ─── Intent Configs ─────────────────────────────────────────────────────────

INTENT_CONFIGS = {

    "factoid": {
        "description": "Specific legal questions with a concrete answer found in one clause or section. "
                       "These ask about a particular penalty, provision, definition, time limit, "
                       "threshold, or legal requirement.",
        "examples": [
            "What is the minimum sentence for theft under Section 379 of IPC?",
            "What does Section 420 of the Indian Penal Code define?",
            "What is the limitation period for filing a civil suit for property disputes?",
            "Under which section can preventive detention be challenged?",
            "What is the maximum fine for defamation under IPC?",
            "What constitutes 'grievous hurt' under the Indian Penal Code?",
            "What is the bail provision for non-bailable offences?",
            "Under which act is cybercrime defined in Indian law?",
        ],
        "anti_examples": [
            "Compare two legal provisions",
            "What patterns exist in legal texts?",
            "Tell me about legal reform",
        ],
    },

    "comparative": {
        "description": "Questions comparing two legal provisions, penalties, acts, jurisdictions, "
                       "clauses, or rights. Must explicitly compare two things.",
        "examples": [
            "How does IPC Section 302 differ from Section 304?",
            "What is the difference between bailable and non-bailable offences?",
            "How do civil and criminal remedies differ for breach of contract?",
            "Compare the powers of High Courts and Supreme Court under Article 226 and 32",
            "Is the punishment for robbery harsher than for theft?",
            "How does negligence differ from strict liability in tort law?",
            "What distinguishes a warrant case from a summons case?",
            "Compare the rights of tenants under old and new rent control acts",
        ],
        "anti_examples": [
            "What is the penalty for murder?",
            "Tell me about criminal law",
            "Summarize contract provisions",
        ],
    },

    "aggregation": {
        "description": "Questions requiring synthesis across multiple legal provisions, sections or "
                       "documents. These ask about common patterns, general provisions, overall "
                       "frameworks, or typical legal approaches across multiple clauses.",
        "examples": [
            "What are the common grounds for challenging a contract's validity?",
            "What types of offences carry mandatory minimum sentences?",
            "What fundamental rights are guaranteed under Part III of the Constitution?",
            "What are the typical remedies available in breach of contract cases?",
            "What provisions exist for protecting women's rights across Indian law?",
            "What are the common defences available in criminal proceedings?",
            "What types of disputes can be resolved through arbitration?",
            "What provisions address environmental protection across different acts?",
        ],
        "anti_examples": [
            "What is Section 420?",
            "Compare murder and manslaughter",
            "Tell me something interesting about law",
        ],
    },

    "exploratory": {
        "description": "Open-ended legal discovery questions seeking insights, implications, "
                       "analysis, or lesser-known provisions. No single correct answer.",
        "examples": [
            "What are some lesser-known rights available to accused persons?",
            "What unusual provisions exist in Indian contract law?",
            "What are the most debated aspects of Indian constitutional law?",
            "What legal protections might surprise someone unfamiliar with Indian law?",
            "What innovative legal remedies have courts developed?",
            "What areas of Indian law need the most reform?",
            "What interesting intersection exists between technology and Indian law?",
            "What overlooked provisions could help small businesses?",
        ],
        "anti_examples": [
            "What is the penalty under Section 420?",
            "Compare two sections",
            "What common patterns exist?",
        ],
    },
}


BATCH_PROMPT_TEMPLATE = """You are generating a research dataset of information retrieval queries for an Indian legal text search system.

Your task: Generate exactly {n} queries of type: {intent_type}

INTENT DEFINITION:
{description}

GOOD EXAMPLES of this query type:
{examples}

BAD EXAMPLES (do NOT generate these — wrong type):
{anti_examples}

CONTEXT — here are some real legal text chunks to ground your queries:
---
{chunks}
---

STRICT RULES:
1. Generate EXACTLY {n} queries, numbered 1 to {n}
2. Every query MUST match the intent type: {intent_type}
3. Each query must be DIFFERENT from the examples above — no copying
4. Each query must be grounded in realistic Indian legal scenarios
5. Vary topics: criminal law, civil law, constitutional law, contract law, property law, etc.
6. Queries should be 10-30 words long
7. Do NOT generate meta-questions about legal texts
8. Do NOT repeat similar queries — make each one distinct
9. Output ONLY the numbered list, no explanations, no headers, no extra text

OUTPUT FORMAT (exactly this, nothing else):
1. [query here]
2. [query here]
3. [query here]
4. [query here]
5. [query here]"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_legal_chunks(filepath: str, n: int = 200) -> list:
    """Load a sample of legal chunks from JSONL."""
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                text = record.get("text", "").strip()
                if len(text) > 80:
                    chunks.append(text)
            except json.JSONDecodeError:
                continue
            if len(chunks) >= n:
                break
    random.shuffle(chunks)
    return chunks


def sample_chunks(chunks: list, n: int = CHUNKS_PER_BATCH) -> str:
    """Pick n legal chunks and format them for the prompt."""
    sampled = random.sample(chunks, min(n, len(chunks)))
    parts = []
    for i, c in enumerate(sampled, 1):
        truncated = c[:400] + "..." if len(c) > 400 else c
        parts.append(f"Legal text {i}: {truncated}")
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
            "num_predict": 500,
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
    """Extract numbered queries from Mistral's output."""
    queries = []
    for line in raw_output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+[\.)\]]\s*(.+)$", line)
        if match:
            query = match.group(1).strip()
            if len(query) < 10 or len(query) > 250:
                continue
            if query.lower().startswith("query"):
                continue
            queries.append(query)
    return queries


def deduplicate(queries: list, existing: list) -> list:
    """Remove queries too similar to existing ones."""
    existing_lower = {q.lower() for q in existing}
    unique = []
    for q in queries:
        q_lower = q.lower()
        is_dup = False
        for ex in existing_lower:
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


# ─── Generation Loop ─────────────────────────────────────────────────────────

def generate_for_intent(intent: str, target: int, chunks: list) -> list:
    """Generate `target` queries of a given intent type."""
    config = INTENT_CONFIGS[intent]
    collected = []
    attempts = 0
    max_attempts = target * 6

    print(f"\n  Generating {target} {intent.upper()} queries...")

    while len(collected) < target and attempts < max_attempts:
        attempts += QUERIES_PER_BATCH

        prompt = BATCH_PROMPT_TEMPLATE.format(
            n=QUERIES_PER_BATCH,
            intent_type=intent.upper(),
            description=config["description"],
            examples="\n".join(f"  - {e}" for e in config["examples"]),
            anti_examples="\n".join(f"  - {e}" for e in config["anti_examples"]),
            chunks=sample_chunks(chunks),
        )

        temperature = 0.7 + random.uniform(0, 0.3)
        raw = call_mistral(prompt, temperature=temperature)

        if not raw:
            continue

        new_queries = parse_queries(raw)
        new_queries = deduplicate(new_queries, collected)
        collected.extend(new_queries)

        print(f"    [{len(collected):>3}/{target}] +{len(new_queries)} new  "
              f"(attempt {attempts})", end="\r")

        time.sleep(0.5)

    collected = collected[:target]
    print(f"    ✓ {len(collected)} {intent} queries generated          ")
    return collected


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TASK 2 — Legal Domain Query Generator")
    print(f"  Target: {TARGET_PER_INTENT * len(INTENTS)} queries "
          f"({TARGET_PER_INTENT} per intent × {len(INTENTS)} intents)")
    print(f"  Model:  {MODEL} via Ollama")
    print("=" * 60)

    # Load chunks
    print(f"\n  Loading legal chunks from {LEGAL_CHUNKS_FILE}...")
    chunks = load_legal_chunks(LEGAL_CHUNKS_FILE, n=500)
    print(f"  Loaded {len(chunks)} chunks for grounding")

    if not chunks:
        print("  ERROR: No chunks loaded. Check data/legal/chunks.jsonl")
        sys.exit(1)

    # Generate per intent
    all_queries = {}
    for intent in INTENTS:
        queries = generate_for_intent(intent, TARGET_PER_INTENT, chunks)
        all_queries[intent] = queries

    # ── Save .txt ──
    total = sum(len(v) for v in all_queries.values())
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"# Legal Domain Query Dataset — {total} queries\n")
        f.write(f"# Generated by Mistral 7B from Indian legal texts\n\n")
        for intent in INTENTS:
            f.write(f"\n# ── {intent.upper()} queries ({len(all_queries[intent])}) ──\n")
            for q in all_queries[intent]:
                f.write(q + "\n")

    # ── Save .json ──
    json_data = []
    for intent in INTENTS:
        for q in all_queries[intent]:
            json_data.append({
                "query": q,
                "intent": intent,
                "domain": "legal",
            })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  TASK 2 COMPLETE — Legal Query Generation")
    print(f"  {'─' * 40}")
    for intent in INTENTS:
        print(f"  {intent:<15} {len(all_queries[intent]):>3} queries")
    print(f"  {'─' * 40}")
    print(f"  TOTAL:          {total:>3} queries")
    print(f"\n  Saved to:")
    print(f"    {OUTPUT_TXT}")
    print(f"    {OUTPUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
