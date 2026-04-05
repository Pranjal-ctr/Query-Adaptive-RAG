# paste this as: python src/filter_queries.py
import json

INPUT = "data/sample_queries_research.json"
OUTPUT = "data/sample_queries_research_filtered.json"

# Reject queries containing these — signs of hallucinated or off-topic content
REJECT_KEYWORDS = [
    "recall", "salon", "hotel lobby", "supermarket", "gym",
    "dentist", "clinic", "hospital", "school", "airline",
]

with open(INPUT) as f:
    data = json.load(f)

kept, rejected = [], []
for item in data:
    q = item["query"].lower()
    if any(kw in q for kw in REJECT_KEYWORDS):
        rejected.append(item)
    elif len(item["query"].split()) < 6:  # too short
        rejected.append(item)
    else:
        kept.append(item)

print(f"Kept: {len(kept)}  |  Rejected: {len(rejected)}")
for r in rejected:
    print(f"  REJECTED [{r['intent_ground_truth']}]: {r['query']}")

with open(OUTPUT, "w") as f:
    json.dump(kept, f, indent=2)

# Also save filtered .txt for --batch
TXT_OUTPUT = OUTPUT.replace(".json", ".txt")
with open(TXT_OUTPUT, "w") as f:
    for item in kept:
        f.write(item["query"] + "\n")
print(f"\nSaved {len(kept)} queries → {OUTPUT}")