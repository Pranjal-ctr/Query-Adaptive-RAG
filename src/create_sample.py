import json
import os

INPUT_FILE = "data/yelp_academic_dataset_review.json"
OUTPUT_FILE = "data/yelp_reviews_sample.jsonl"
MAX_REVIEWS = 5000

os.makedirs("data", exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for i, line in enumerate(fin):
        if i >= MAX_REVIEWS:
            break
        review = json.loads(line)
        fout.write(json.dumps({"text": review["text"]}) + "\n")

print(" Sample dataset created:", OUTPUT_FILE)

