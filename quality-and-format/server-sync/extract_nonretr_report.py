import json
from pathlib import Path
root = Path("runs/protect_multipage_nonretr_v1")
s = json.load(open(root / "summary.json"))
d = json.load(open(root / "design.json"))
print("===CONFIG===")
print(json.dumps({
    "exp_id": d.get("exp_id"),
    "template": d.get("template"),
    "seed": d.get("seed"),
    "b": d.get("b"),
    "g": d.get("g"),
    "L_p": d.get("L_p"),
    "strategies": d.get("strategies"),
    "budgets": d.get("budgets"),
    "harms_path": d.get("harms_path"),
    "page_scores_path": d.get("page_scores_path"),
}, indent=2))
print("===WALL/GATES===")
print("elapsed_s", s.get("elapsed_s"))
print("n_metrics", s.get("n_metrics"), "n_metrics_file", s.get("n_metrics_file"))
print("stop_expand", s.get("stop_expand"))
print(json.dumps(s.get("gates"), indent=2))
print("===TABLE mean S / H_pos / n_protect / n_quant===")
for k, v in sorted(s["table_S_by_strategy_budget"].items()):
    print(f"{k}: S_mean={v['S_mean_mean']:.6e} H_pos={v['H_pos_mean_mean']:.6e} n={v['n']} n_prot={v['mean_n_protect']:.3f} n_quant={v['mean_n_quant']:.3f}")
print("===COMPARISONS===")
print(json.dumps(s.get("comparisons"), indent=2))
print("===RECOVERY===")
print(json.dumps(s.get("recovery_vs_none"), indent=2))
# vs retrieval
r = json.load(open("runs/protect_multipage_v1/summary.json"))
print("===VS RETRIEVAL (mean S)===")
for k in sorted(set(s["table_S_by_strategy_budget"]) | set(r["table_S_by_strategy_budget"])):
    a = s["table_S_by_strategy_budget"].get(k, {})
    b = r["table_S_by_strategy_budget"].get(k, {})
    print(f"{k}: nonretr={a.get('S_mean_mean')} retr={b.get('S_mean_mean')}")
print("retr_elapsed", r.get("elapsed_s"), "retr_gates", r.get("gates"))
print("retr_comparisons", json.dumps(r.get("comparisons"), indent=2)[:1500])
