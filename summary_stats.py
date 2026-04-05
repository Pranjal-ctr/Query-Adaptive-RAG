"""Quick summary of collected curve features."""
import json
import numpy as np

data = json.load(open(r"d:\Programming\AI\RAG\food_domain_rag\results\curve_features.json"))

intents = {}
for d in data:
    intents.setdefault(d["intent"], []).append(d)

print(f"TOTAL CURVES: {len(data)}")
print(f"\nINTENT DISTRIBUTION:")
for k, v in sorted(intents.items()):
    print(f"  {k}: {len(v)} queries")

ks = [d["kurtosis"] for d in data]
es = [d["entropy"] for d in data]
ss = [d["slope"] for d in data]
rs = [d["score_range"] for d in data]

print(f"\nFEATURE RANGES:")
print(f"  Kurtosis:    min={min(ks):.3f}  max={max(ks):.3f}  mean={np.mean(ks):.3f}  std={np.std(ks):.3f}")
print(f"  Entropy:     min={min(es):.3f}  max={max(es):.3f}  mean={np.mean(es):.3f}  std={np.std(es):.3f}")
print(f"  Slope:       min={min(ss):.6f}  max={max(ss):.6f}  mean={np.mean(ss):.6f}")
print(f"  Score Range: min={min(rs):.4f}  max={max(rs):.4f}  mean={np.mean(rs):.4f}")

print(f"\nBY INTENT:")
print(f"  {'Intent':<15} {'N':>3}  {'Kurtosis':>9}  {'Entropy':>8}  {'Slope':>11}  {'Range':>7}")
print(f"  {'-'*60}")
for k, v in sorted(intents.items()):
    n = len(v)
    mk = np.mean([d["kurtosis"] for d in v])
    me = np.mean([d["entropy"] for d in v])
    ms = np.mean([d["slope"] for d in v])
    mr = np.mean([d["score_range"] for d in v])
    print(f"  {k:<15} {n:>3}  {mk:>9.3f}  {me:>8.3f}  {ms:>11.6f}  {mr:>7.4f}")
