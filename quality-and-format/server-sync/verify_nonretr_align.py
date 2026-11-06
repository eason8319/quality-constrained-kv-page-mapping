import json
from pathlib import Path
from transformers import AutoTokenizer
from experiments.page_paired_sensitivity.prefix_builder import build_nonretr_long_prefixes, summarize_prefix

snap = Path("/cluster/home/zengy/hf-cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots")
load_from = str(list(snap.iterdir())[0])
tok = AutoTokenizer.from_pretrained(load_from, local_files_only=True)
prefs = build_nonretr_long_prefixes(tok, L_p=64, n=32, seed=20261106)
saved = json.load(open("runs/rep_sample_nonretr_v1/prefixes.json"))
ok = 0
for a, b in zip(prefs, saved):
    sa = summarize_prefix(a)
    assert sa["prefix_id"] == b["prefix_id"]
    assert sa["shared_len"] == b["shared_len"], (sa["prefix_id"], sa["shared_len"], b["shared_len"])
    assert sa["page_a"] == b["page_a"] and sa["page_b"] == b["page_b"]
    assert sa["ans_jia_ids"] == b["ans_jia_ids"]
    ok += 1
print("prefix_align_ok", ok)
prefs_bad = build_nonretr_long_prefixes(tok, L_p=64, n=2, seed=20261105)
sb = summarize_prefix(prefs_bad[0])
print(
    "seed20261105 nr000 shared",
    sb["shared_len"],
    "saved",
    saved[0]["shared_len"],
    "match",
    sb["shared_len"] == saved[0]["shared_len"],
)
