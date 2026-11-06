#!/usr/bin/env python3
"""Equal-budget multi-page protect re-run (true joint quant of unprotected pages).

Measures residual end-to-end S when selected pages stay FP and all other closed
pages are quantized jointly (same b/g). Offline H-coverage is NOT a substitute.

Default out: runs/protect_multipage_v1/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.page_paired_sensitivity.prefix_builder import (
    build_pilot_prefixes,
    summarize_prefix,
)
from experiments.page_paired_sensitivity.kv_fork import (
    run_fp_trajectory,
    run_fork_pair_pages,
)

# Frozen heads from rep_sample_v1
HEADS = [(10, 3), (2, 1), (22, 2), (22, 4), (24, 6)]
FIXED_BUCKET_SCORE = {
    "earlier_history": 3.0,
    "first_token_page": 2.0,
    "recent_closed": 1.0,
}
STRATEGIES_DEFAULT = ("oracle_H", "bucket", "random", "struct+content", "none")
BUDGETS_DEFAULT = (0.25, 0.50)
RANDOM_SEED = 20261105


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--hf-home", default="/cluster/home/zengy/hf-cache")
    p.add_argument("--L-p", type=int, default=64)
    p.add_argument("--g", type=int, default=64)
    p.add_argument("--b", type=int, default=4)
    p.add_argument("--n-prefix", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--template", default="retrieval", choices=["retrieval", "nonretr"])
    p.add_argument("--harms", default=None, help="page_harms.jsonl path (oracle_H / page list)")
    p.add_argument("--page-scores", default=None, help="page_scores.jsonl for struct+content")
    p.add_argument("--strategies", default=",".join(STRATEGIES_DEFAULT))
    p.add_argument("--budgets", default="0.25,0.50")
    p.add_argument("--intervene", default="kv", help="comma list; primary kv")
    p.add_argument("--max-prefixes", type=int, default=None)
    p.add_argument("--zero-pert-every", type=int, default=8, help="run zero_pert on none every N prefixes")
    p.add_argument("--smoke", action="store_true", help="1 prefix, 1 head, budgets 0.25 only")
    return p.parse_args()


def k_for_frac(n: int, f: float) -> int:
    if n <= 0:
        return 0
    if f >= 1.0:
        return n
    if f <= 0:
        return 0
    return max(1, int(math.ceil(f * n)))


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def fp_correct(answer_logprobs: List[float]) -> bool:
    if not answer_logprobs:
        return False
    nll = [-lp for lp in answer_logprobs]
    return (sum(nll) / len(nll)) < 2.0


def index_harms(rows: List[dict], intervene: str) -> Dict[Tuple[str, int, int], List[dict]]:
    """(prefix_id,l,h) -> list of page rows (unique page_index)."""
    grouped: Dict[Tuple[str, int, int], Dict[int, dict]] = defaultdict(dict)
    for r in rows:
        if r["intervene"] != intervene:
            continue
        key = (r["prefix_id"], int(r["l"]), int(r["h"]))
        pi = int(r["page_index"])
        grouped[key][pi] = r
    out = {}
    for k, d in grouped.items():
        pages = [d[i] for i in sorted(d.keys())]
        out[k] = pages
    return out


def index_scores(rows: List[dict], intervene: str, template: str) -> Dict[Tuple[str, int, int, int], float]:
    """(prefix_id,l,h,page_index) -> struct_content_score"""
    out = {}
    for r in rows:
        if r.get("intervene") != intervene:
            continue
        if template and r.get("template") and r["template"] != template:
            continue
        key = (r["prefix_id"], int(r["l"]), int(r["h"]), int(r["page_index"]))
        out[key] = float(r["struct_content_score"])
    return out


def select_protect(
    pages: List[dict],
    strategy: str,
    budget: float,
    *,
    seed: int,
    scores: Dict[Tuple[str, int, int, int], float],
    prefix_id: str,
    layer: int,
    kv_head: int,
    intervene: str,
) -> List[int]:
    """Return sorted page_index list to KEEP at FP (protect)."""
    n = len(pages)
    if strategy == "none" or budget <= 0:
        return []
    k = k_for_frac(n, budget)
    idxs = list(range(n))

    if strategy == "oracle_H":
        order = sorted(idxs, key=lambda i: (-max(0.0, float(pages[i]["H"])), pages[i]["page_index"]))
    elif strategy == "bucket":
        scores_b = [FIXED_BUCKET_SCORE.get(pages[i]["page_pos_bucket"], 0.0) for i in idxs]
        order = sorted(idxs, key=lambda i: (-scores_b[i], pages[i]["page_index"]))
    elif strategy == "random":
        rng = random.Random(seed + (abs(hash((prefix_id, layer, kv_head, intervene))) % 10_000_000))
        order = idxs[:]
        rng.shuffle(order)
    elif strategy == "struct+content":
        sc = []
        for i in idxs:
            pi = int(pages[i]["page_index"])
            sc.append(scores.get((prefix_id, layer, kv_head, pi), float("-inf")))
        order = sorted(idxs, key=lambda i: (-sc[i], pages[i]["page_index"]))
    else:
        raise ValueError(strategy)

    protect = [int(pages[i]["page_index"]) for i in order[:k]]
    return sorted(protect)


def spans_to_quantize(pages: List[dict], protect: List[int], L_p: int) -> List[Tuple[int, int, int]]:
    prot = set(protect)
    spans = []
    for p in pages:
        pi = int(p["page_index"])
        if pi in prot:
            continue
        t0, t1 = pi * L_p, (pi + 1) * L_p
        spans.append((pi, t0, t1))
    spans.sort(key=lambda x: x[1])
    return spans


def mean_or_none(xs):
    return (sum(xs) / len(xs)) if xs else None


def main():
    args = parse_args()
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(args.hf_home, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(args.hf_home, "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    intervenes = [x.strip() for x in args.intervene.split(",") if x.strip()]
    heads = list(HEADS)

    if args.smoke:
        args.max_prefixes = 1
        budgets = [0.25]
        heads = [HEADS[0]]
        strategies = [s for s in strategies if s in ("oracle_H", "bucket", "random", "none", "struct+content")]

    if args.template == "retrieval":
        default_harms = ROOT / "runs" / "rep_sample_v1" / "page_harms.jsonl"
        template_tag = "retrieval"
    else:
        default_harms = ROOT / "runs" / "rep_sample_nonretr_v1" / "page_harms.jsonl"
        template_tag = "nonretr"

    harms_path = Path(args.harms) if args.harms else default_harms
    scores_path = Path(args.page_scores) if args.page_scores else (ROOT / "runs" / "protect_close_feat_v1" / "page_scores.jsonl")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.template == "nonretr":
        out_dir = ROOT / "runs" / "protect_multipage_nonretr_v1"
    else:
        out_dir = ROOT / "runs" / "protect_multipage_v1"
    if args.smoke and args.out_dir is None:
        out_dir = ROOT / "runs" / (
            "protect_multipage_nonretr_v1_smoke" if args.template == "nonretr" else "protect_multipage_v1_smoke"
        )
    ckpt_dir = out_dir / "ckpts"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    traj_path = out_dir / "trajectories.jsonl"
    metrics_path = out_dir / "metrics.jsonl"
    gates_path = out_dir / "gates.jsonl"
    residuals_path = out_dir / "residuals.jsonl"
    summary_path = out_dir / "summary.json"
    progress_path = out_dir / "progress.json"

    # Load rankings sources
    print(f"[pmp] load harms {harms_path}", flush=True)
    harms_rows = load_jsonl(harms_path)
    print(f"[pmp] load scores {scores_path}", flush=True)
    score_rows = load_jsonl(scores_path) if scores_path.exists() else []
    if "struct+content" in strategies and not score_rows:
        raise FileNotFoundError(f"struct+content requested but missing {scores_path}")

    snap = Path(args.hf_home) / "hub" / "models--meta-llama--Llama-3.1-8B-Instruct" / "snapshots"
    load_from = str(list(snap.iterdir())[0]) if snap.exists() and list(snap.iterdir()) else args.model

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[pmp] load {load_from} device={device} dtype={dtype} b={args.b}", flush=True)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        print(f"[pmp] gpu={torch.cuda.get_device_name(device)} free_gb={free/1e9:.2f}/{total/1e9:.2f}", flush=True)

    tok = AutoTokenizer.from_pretrained(load_from, local_files_only=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        load_from, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    print(f"[pmp] model ready in {time.time()-t_load:.1f}s", flush=True)

    if args.template == "retrieval":
        prefixes = build_pilot_prefixes(tok, L_p=args.L_p, n=args.n_prefix, seed=args.seed)
    else:
        from experiments.page_paired_sensitivity.prefix_builder import build_nonretr_long_prefixes
        prefixes = build_nonretr_long_prefixes(tok, L_p=args.L_p, n=args.n_prefix, seed=args.seed)
    if args.max_prefixes:
        prefixes = prefixes[: args.max_prefixes]
    prefix_by_id = {sp.prefix_id: sp for sp in prefixes}

    design = {
        "exp_id": out_dir.name,
        "arm": "protect_multipage_equal_budget",
        "template": args.template,
        "seed": args.seed,
        "n_prefix": len(prefixes),
        "L_p": args.L_p,
        "g": args.g,
        "b": args.b,
        "heads": heads,
        "intervenes": intervenes,
        "strategies": strategies,
        "budgets": budgets,
        "harms_path": str(harms_path),
        "page_scores_path": str(scores_path),
        "bucket_fixed_order": ["earlier_history", "first_token_page", "recent_closed"],
        "k_rule": "k = max(1, ceil(frac * n_pages)) for frac>0; none -> k=0",
        "metric": "residual S under joint quant of unprotected closed pages; chunk-matched FP baseline",
        "recovery": "1 - S_strategy/S_none when S_none>0 (computed on means)",
        "random_seed": RANDOM_SEED,
        "N_prefix_frozen": 32,
        "compare_to": ["protect_strategy_v1", "protect_close_feat_v1", "pilot_multipage_v1"],
        "smoke": bool(args.smoke),
    }
    (out_dir / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    (out_dir / "prefixes.json").write_text(
        json.dumps([summarize_prefix(sp) for sp in prefixes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "layer_heads.json").write_text(json.dumps(heads), encoding="utf-8")

    done = set(p.stem for p in ckpt_dir.glob("*.done"))
    print(f"[pmp] resume done={len(done)} strategies={strategies} budgets={budgets} heads={heads}", flush=True)

    gate_stats = {
        "hash_checks": 0,
        "hash_fail": 0,
        "fp_both_correct": 0,
        "fp_checked": 0,
        "zero_pert_checks": 0,
        "zero_pert_fail": 0,
        "empty_protect_ok": 0,
        "prefix_failures": [],
        "missing_score_pages": 0,
    }

    def append_jsonl(path: Path, obj: dict):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    t0 = time.time()
    n_traj = 0
    n_metrics = 0
    stop_expand = False

    # Pre-index per intervene
    harms_by_iv = {iv: index_harms(harms_rows, iv) for iv in intervenes}
    scores_by_iv = {iv: index_scores(score_rows, iv, template_tag if args.template == "retrieval" else "nonretr") for iv in intervenes}

    with torch.no_grad():
        for pi, sp in enumerate(prefixes):
            if stop_expand:
                break
            prefix_t0 = time.time()
            print(
                f"[pmp] === {sp.prefix_id} ({pi+1}/{len(prefixes)}) shared={len(sp.shared_ids)} ===",
                flush=True,
            )

            # Mono FP for correctness gate
            fp_mono = {}
            for q in ("jia", "yi"):
                r = run_fp_trajectory(model, sp, question=q, device=device)
                fp_mono[q] = r
            both_ok = fp_correct(fp_mono["jia"].answer_logprobs) and fp_correct(fp_mono["yi"].answer_logprobs)
            gate_stats["fp_checked"] += 1
            if both_ok:
                gate_stats["fp_both_correct"] += 1
            else:
                gate_stats["prefix_failures"].append({"prefix_id": sp.prefix_id, "reason": "fp_not_both_correct"})
                print(f"[pmp] WARN FP both-correct FAIL {sp.prefix_id}", flush=True)

            # Cache chunk-matched FP NLL by t_close
            fp_chunk_cache: Dict[int, Dict[str, List[float]]] = {}

            def get_fp_chunk(t_close: int) -> Dict[str, List[float]]:
                if t_close not in fp_chunk_cache:
                    fp_chunk_cache[t_close] = {}
                    for q in ("jia", "yi"):
                        r = run_fp_trajectory(model, sp, question=q, device=device, chunk_at=t_close)
                        fp_chunk_cache[t_close][q] = [-lp for lp in r.answer_logprobs]
                return fp_chunk_cache[t_close]

            for layer, kv_head in heads:
                for iv in intervenes:
                    pages = harms_by_iv[iv].get((sp.prefix_id, layer, kv_head))
                    if not pages:
                        print(f"[pmp] WARN no harms for {sp.prefix_id} L{layer}H{kv_head} {iv}", flush=True)
                        continue
                    score_map = scores_by_iv[iv]

                    # Build work list: (strategy, budget_label, budget_val)
                    work = []
                    for strat in strategies:
                        if strat == "none":
                            work.append((strat, 0.0, 0.0))
                        else:
                            for bud in budgets:
                                work.append((strat, bud, bud))

                    for strat, bud_label, bud_val in work:
                        ckpt_key = f"{sp.prefix_id}_{strat}_b{bud_label}_L{layer}_H{kv_head}_{iv}_bits{args.b}"
                        # sanitize filename
                        ckpt_key = ckpt_key.replace("+", "plus").replace(".", "p")
                        if ckpt_key in done:
                            continue

                        protect = select_protect(
                            pages, strat, bud_val,
                            seed=RANDOM_SEED, scores=score_map,
                            prefix_id=sp.prefix_id, layer=layer, kv_head=kv_head, intervene=iv,
                        )
                        if strat == "struct+content":
                            for p in pages:
                                key = (sp.prefix_id, layer, kv_head, int(p["page_index"]))
                                if key not in score_map:
                                    gate_stats["missing_score_pages"] += 1

                        q_spans = spans_to_quantize(pages, protect, args.L_p)
                        n_pages = len(pages)
                        n_protect = len(protect)
                        n_quant = len(q_spans)

                        if n_quant == 0:
                            # all protected / all FP — residual S = 0
                            gate_stats["empty_protect_ok"] += 1  # misnomer kept for all-fp
                            for q in ("jia", "yi"):
                                append_jsonl(
                                    traj_path,
                                    {
                                        "prefix_id": sp.prefix_id,
                                        "template": args.template,
                                        "strategy": strat,
                                        "budget": bud_label,
                                        "l": layer,
                                        "h": kv_head,
                                        "intervene": iv,
                                        "question": q,
                                        "mode": "all_fp",
                                        "S": 0.0,
                                        "S_key": 0.0,
                                        "n_pages": n_pages,
                                        "n_protect": n_protect,
                                        "n_quant": 0,
                                        "protect_pages": protect,
                                        "page_hash": None,
                                        "bits": args.b,
                                    },
                                )
                                n_traj += 1
                            row = {
                                "prefix_id": sp.prefix_id,
                                "template": args.template,
                                "strategy": strat,
                                "budget": bud_label,
                                "l": layer,
                                "h": kv_head,
                                "intervene": iv,
                                "bits": args.b,
                                "n_pages": n_pages,
                                "n_protect": n_protect,
                                "n_quant": 0,
                                "protect_pages": protect,
                                "quant_pages": [],
                                "S_jia": 0.0,
                                "S_yi": 0.0,
                                "S_mean": 0.0,
                                "S_key_jia": 0.0,
                                "S_key_yi": 0.0,
                                "S_key_mean": 0.0,
                                "H_pos_mean": 0.0,
                                "page_hash": None,
                            }
                            append_jsonl(metrics_path, row)
                            append_jsonl(residuals_path, row)
                            n_metrics += 1
                            (ckpt_dir / f"{ckpt_key}.done").write_text(str(time.time()), encoding="utf-8")
                            done.add(ckpt_key)
                            continue

                        t_close = max(t1 for _, _, t1 in q_spans)
                        fp_n = get_fp_chunk(t_close)

                        rj, ry, joint_hash, _ = run_fork_pair_pages(
                            model, sp,
                            page_spans=q_spans,
                            intervene=iv, mode="quant",
                            layer=layer, kv_head=kv_head,
                            L_p=args.L_p, bits=args.b, group=args.g,
                            device=device,
                            fp_nlls_jia=fp_n["jia"],
                            fp_nlls_yi=fp_n["yi"],
                            page_label=f"unprot_{strat}_k{n_protect}",
                        )

                        gate_stats["hash_checks"] += 1
                        if rj.page_hash != ry.page_hash:
                            gate_stats["hash_fail"] += 1
                            stop_expand = True
                            append_jsonl(gates_path, {
                                "prefix_id": sp.prefix_id, "gate": "hash", "pass": False,
                                "strategy": strat, "budget": bud_label,
                            })

                        S_mean = 0.5 * (rj.S + ry.S)
                        S_key_mean = 0.5 * (rj.S_key + ry.S_key)
                        H_pos = 0.5 * (max(0.0, rj.S) + max(0.0, ry.S))

                        for r in (rj, ry):
                            append_jsonl(
                                traj_path,
                                {
                                    "prefix_id": sp.prefix_id,
                                    "template": args.template,
                                    "strategy": strat,
                                    "budget": bud_label,
                                    "l": layer,
                                    "h": kv_head,
                                    "intervene": iv,
                                    "question": r.question,
                                    "mode": "quant",
                                    "S": r.S,
                                    "S_key": r.S_key,
                                    "nll_run": r.nll_run,
                                    "n_pages": n_pages,
                                    "n_protect": n_protect,
                                    "n_quant": n_quant,
                                    "protect_pages": protect,
                                    "quant_pages": [pi_ for pi_, _, _ in q_spans],
                                    "page_hash": r.page_hash,
                                    "bits": args.b,
                                    "t_close": t_close,
                                },
                            )
                            n_traj += 1

                        row = {
                            "prefix_id": sp.prefix_id,
                            "template": args.template,
                            "strategy": strat,
                            "budget": bud_label,
                            "l": layer,
                            "h": kv_head,
                            "intervene": iv,
                            "bits": args.b,
                            "n_pages": n_pages,
                            "n_protect": n_protect,
                            "n_quant": n_quant,
                            "protect_pages": protect,
                            "quant_pages": [pi_ for pi_, _, _ in q_spans],
                            "S_jia": rj.S,
                            "S_yi": ry.S,
                            "S_mean": S_mean,
                            "S_key_jia": rj.S_key,
                            "S_key_yi": ry.S_key,
                            "S_key_mean": S_key_mean,
                            "H_pos_mean": H_pos,
                            "page_hash": joint_hash,
                            "t_close": t_close,
                        }
                        append_jsonl(metrics_path, row)
                        append_jsonl(residuals_path, row)
                        n_metrics += 1

                        # zero-pert gate: on none strategy periodically
                        if (
                            strat == "none"
                            and (pi % max(1, args.zero_pert_every) == 0)
                            and layer == heads[0][0]
                            and kv_head == heads[0][1]
                        ):
                            zj, zy, _, _ = run_fork_pair_pages(
                                model, sp,
                                page_spans=q_spans,
                                intervene=iv, mode="zero_pert",
                                layer=layer, kv_head=kv_head,
                                L_p=args.L_p, bits=args.b, group=args.g,
                                device=device,
                                fp_nlls_jia=fp_n["jia"],
                                fp_nlls_yi=fp_n["yi"],
                                page_label="zero_pert_none",
                            )
                            for r in (zj, zy):
                                gate_stats["zero_pert_checks"] += 1
                                ok = abs(r.S) <= 1e-4 and abs(r.S_key) <= 1e-4
                                if not ok:
                                    gate_stats["zero_pert_fail"] += 1
                                    stop_expand = True
                                append_jsonl(
                                    gates_path,
                                    {
                                        "prefix_id": sp.prefix_id,
                                        "gate": "zero_pert",
                                        "pass": ok,
                                        "S": r.S,
                                        "S_key": r.S_key,
                                        "q": r.question,
                                        "strategy": "none",
                                    },
                                )
                                print(f"[pmp] zero_pert {sp.prefix_id} {r.question}: S={r.S:.3e} pass={ok}", flush=True)

                        (ckpt_dir / f"{ckpt_key}.done").write_text(str(time.time()), encoding="utf-8")
                        done.add(ckpt_key)

                        if stop_expand:
                            break
                    if stop_expand:
                        break
                if stop_expand:
                    break

            progress_path.write_text(
                json.dumps(
                    {
                        "prefix_id": sp.prefix_id,
                        "done_ckpts": len(done),
                        "n_traj": n_traj,
                        "n_metrics": n_metrics,
                        "elapsed_s": time.time() - t0,
                        "gate_stats": {k: v for k, v in gate_stats.items() if k != "prefix_failures"},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[pmp] prefix {sp.prefix_id} done in {time.time()-prefix_t0:.1f}s ckpts={len(done)}", flush=True)

    # Finalize from metrics (resume-safe: re-read file)
    rows = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    # Pool by strategy x budget x intervene
    def pool_key(r):
        return (r["strategy"], float(r["budget"]), r["intervene"])

    pools: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in rows:
        pools[pool_key(r)].append(r)

    table = {}
    for (strat, bud, iv), sub in sorted(pools.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
        s_mean = [r["S_mean"] for r in sub]
        s_jia = [r["S_jia"] for r in sub]
        s_yi = [r["S_yi"] for r in sub]
        hpos = [r["H_pos_mean"] for r in sub]
        sk = [r["S_key_mean"] for r in sub]
        table[f"{strat}|budget={bud}|{iv}"] = {
            "n": len(sub),
            "S_mean_mean": mean_or_none(s_mean),
            "S_mean_median": pct(s_mean, 50),
            "S_mean_p95": pct(s_mean, 95),
            "S_jia_mean": mean_or_none(s_jia),
            "S_yi_mean": mean_or_none(s_yi),
            "S_key_mean_mean": mean_or_none(sk),
            "H_pos_mean_mean": mean_or_none(hpos),
            "mean_n_protect": mean_or_none([r["n_protect"] for r in sub]),
            "mean_n_quant": mean_or_none([r["n_quant"] for r in sub]),
        }

    # Recovery vs none: matched by (prefix,l,h,intervene,budget) — for none use budget 0
    none_map = {}
    for r in rows:
        if r["strategy"] == "none":
            none_map[(r["prefix_id"], r["l"], r["h"], r["intervene"])] = r

    recovery = {}
    for strat in strategies:
        if strat == "none":
            continue
        for bud in budgets:
            for iv in intervenes:
                ratios = []
                abs_red = []
                paired = 0
                for r in rows:
                    if r["strategy"] != strat or float(r["budget"]) != float(bud) or r["intervene"] != iv:
                        continue
                    nref = none_map.get((r["prefix_id"], r["l"], r["h"], iv))
                    if not nref:
                        continue
                    paired += 1
                    sn = nref["S_mean"]
                    ss = r["S_mean"]
                    if sn > 1e-12:
                        ratios.append(1.0 - ss / sn)
                    abs_red.append(sn - ss)
                recovery[f"{strat}|budget={bud}|{iv}"] = {
                    "n_paired": paired,
                    "recovery_mean": mean_or_none(ratios),
                    "recovery_median": pct(ratios, 50),
                    "abs_S_reduction_mean": mean_or_none(abs_red),
                    "S_strategy_mean": table.get(f"{strat}|budget={bud}|{iv}", {}).get("S_mean_mean"),
                    "S_none_mean": table.get(f"none|budget=0.0|{iv}", {}).get("S_mean_mean"),
                }

    # Head-to-head: oracle vs bucket, bucket vs random, struct+content vs random at 0.25/0.50
    comparisons = {}
    for bud in budgets:
        for iv in intervenes:
            def getS(name):
                return table.get(f"{name}|budget={bud}|{iv}", {}).get("S_mean_mean")
            def getR(name):
                return recovery.get(f"{name}|budget={bud}|{iv}", {}).get("recovery_mean")
            s_oracle = getS("oracle_H")
            s_bucket = getS("bucket")
            s_rand = getS("random")
            s_sc = getS("struct+content")
            s_none = table.get(f"none|budget=0.0|{iv}", {}).get("S_mean_mean")
            comparisons[f"budget={bud}|{iv}"] = {
                "S_none": s_none,
                "S_oracle_H": s_oracle,
                "S_bucket": s_bucket,
                "S_random": s_rand,
                "S_struct+content": s_sc,
                "oracle_beats_bucket": (s_oracle is not None and s_bucket is not None and s_oracle < s_bucket),
                "bucket_beats_random": (s_bucket is not None and s_rand is not None and s_bucket < s_rand),
                "struct_content_beats_random": (s_sc is not None and s_rand is not None and s_sc < s_rand),
                "oracle_beats_random": (s_oracle is not None and s_rand is not None and s_oracle < s_rand),
                "recovery_oracle": getR("oracle_H"),
                "recovery_bucket": getR("bucket"),
                "recovery_random": getR("random"),
                "recovery_struct+content": getR("struct+content"),
            }

    fp_rate = gate_stats["fp_both_correct"] / gate_stats["fp_checked"] if gate_stats["fp_checked"] else None
    hash_rate = 1.0 - gate_stats["hash_fail"] / gate_stats["hash_checks"] if gate_stats["hash_checks"] else None
    zp_rate = (
        1.0 - gate_stats["zero_pert_fail"] / gate_stats["zero_pert_checks"]
        if gate_stats["zero_pert_checks"]
        else None
    )

    summary = {
        "config": {
            "model": load_from,
            "L_p": args.L_p,
            "g": args.g,
            "b": args.b,
            "n_prefix": len(prefixes),
            "template": args.template,
            "heads": heads,
            "intervenes": intervenes,
            "strategies": strategies,
            "budgets": budgets,
            "device": str(device),
            "dtype": args.dtype,
            "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "arm": "protect_multipage_equal_budget",
            "seed": args.seed,
            "harms": str(harms_path),
            "page_scores": str(scores_path),
            "primary_metric": "mean residual S_mean = 0.5*(S_jia+S_yi) under joint unprot quant",
        },
        "elapsed_s": time.time() - t0,
        "n_traj_written": n_traj,
        "n_metrics": n_metrics if n_metrics else len(rows),
        "n_metrics_file": len(rows),
        "stop_expand": stop_expand,
        "gates": {
            "hash_equality_rate": hash_rate,
            "hash_checks": gate_stats["hash_checks"],
            "hash_fail": gate_stats["hash_fail"],
            "fp_both_correct_rate": fp_rate,
            "fp_checked": gate_stats["fp_checked"],
            "zero_pert_pass_rate": zp_rate,
            "zero_pert_checks": gate_stats["zero_pert_checks"],
            "zero_pert_fail": gate_stats["zero_pert_fail"],
            "missing_score_pages": gate_stats["missing_score_pages"],
            "prefix_failures": gate_stats["prefix_failures"],
        },
        "table_S_by_strategy_budget": table,
        "recovery_vs_none": recovery,
        "comparisons": comparisons,
        "gate_stats_raw": {k: v for k, v in gate_stats.items()},
        "note_free_gen": "Free-gen error rate not measured; teacher-forced NLL residual S is primary. FP both-correct uses mean NLL<2 gate.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # README
    lines = [
        f"# {out_dir.name} — equal-budget multi-page protect re-run",
        "",
        "True joint quant of unprotected closed pages; protected pages stay FP.",
        "Primary metric: residual mean S = 0.5*(S_jia+S_yi) vs chunk-matched FP.",
        "",
        f"- template: {args.template}",
        f"- bits/group/L_p: {args.b}/{args.g}/{args.L_p}",
        f"- heads: {heads}",
        f"- intervenes: {intervenes}",
        f"- strategies: {strategies}",
        f"- budgets: {budgets} (none runs once at budget=0)",
        f"- elapsed_s: {summary['elapsed_s']:.1f}",
        f"- n_metrics: {len(rows)}",
        f"- stop_expand: {stop_expand}",
        f"- gates: {json.dumps(summary['gates'], ensure_ascii=False)}",
        "",
        "## Table (mean residual S)",
        "",
        "```",
        json.dumps(table, indent=2),
        "```",
        "",
        "## Recovery vs none",
        "",
        "```",
        json.dumps(recovery, indent=2),
        "```",
        "",
        "## Comparisons",
        "",
        "```",
        json.dumps(comparisons, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- oracle_H uses measured H from prior single-page page_harms (upper bound).",
        "- bucket / random / struct+content do not peek target H at selection time.",
        "- Offline H-coverage ≠ residual S under multi-page joint quant (this arm).",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"table": table, "comparisons": comparisons, "gates": summary["gates"]}, indent=2), flush=True)
    print(f"[pmp] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
