import json
from collections import Counter
rows = []
with open("runs/rep_sample_nonretr_v1/gates.jsonl") as f:
    for line in f:
        rows.append(json.loads(line))
print("gate_types", Counter(r["gate"] for r in rows))
for g in sorted(set(r["gate"] for r in rows)):
    sub = [r for r in rows if r["gate"] == g]
    passed = sum(1 for r in sub if r.get("pass") is True)
    print(g, "n", len(sub), "pass", passed, "rate", passed / len(sub) if sub else None)
s = json.load(open("runs/rep_sample_nonretr_v1/summary.json"))
print("summary.gates", s.get("gates") or s.get("gate_stats"))
