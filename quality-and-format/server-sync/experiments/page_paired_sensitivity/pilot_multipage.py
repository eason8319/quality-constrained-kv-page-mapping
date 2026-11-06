#!/usr/bin/env python3
"""Multi-page joint quant pilot (and optional b=2 multipage).

Variants:
  AB   — quantize fact pages A and B together
  Aadj — quantize fact page A + one adjacent closed page

Metrics:
  - For AB: report S_jia / S_yi (both answers' pages are perturbed); classic D N/A.
  - For Aadj: report S_jia (need A) vs S_yi (need-not A) gap = S_jia - S_yi;
    also report mean S. Classic single-page D is NOT claimed.

Default out: runs/pilot_multipage_v1/
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
    enrich_meta,
    summarize_prefix,
)
from experiments.page_paired_sensitivity.layer_heads import sample_layer_heads
from experiments.page_paired_sensitivity.kv_fork import (
    run_fp_trajectory,
    run_fork_pair_pages,
)


INTERVENES = ("k", "v", "kv")


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
    p.add_argument("--variants", default="AB,Aadj", help="comma list: AB,Aadj")
    p.add_argument("--max-prefixes", type=int, default=None)
    p.add_argument("--zero-pert-every", type=int, default=1)
    return p.parse_args()


def fp_correct(answer_logprobs: List[float]) -> bool:
    if not answer_logprobs:
        return False
    nll = [-lp for lp in answer_logprobs]
    return (sum(nll) / len(nll)) < 2.0


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


def adjacent_page_index(sp, L_p: int) -> Optional[int]:
    """Pick a closed page adjacent to A, preferring +1 if not B, else -1."""
    n = len(sp.shared_ids) // L_p
    a = sp.page_a_index
    b = sp.page_b_index
    for cand in (a + 1, a - 1, a + 2, a - 2):
        if 0 <= cand < n and cand != a and cand != b:
            return cand
    # fallback: any non-A non-B
    for cand in range(n):
        if cand != a and cand != b:
            return cand
    return None


def spans_for_variant(sp, variant: str, L_p: int) -> Tuple[str, List[Tuple[int, int, int]]]:
    a0, a1 = sp.page_a
    b0, b1 = sp.page_b
    if variant == "AB":
        spans = [
            (sp.page_a_index, a0, a1),
            (sp.page_b_index, b0, b1),
        ]
        # stable order by t0
        spans = sorted(spans, key=lambda x: x[1])
        return "AB", spans
    if variant == "Aadj":
        adj = adjacent_page_index(sp, L_p)
        if adj is None:
            raise RuntimeError(f"no adjacent page for {sp.prefix_id}")
        t0, t1 = adj * L_p, (adj + 1) * L_p
        spans = [
            (sp.page_a_index, a0, a1),
            (adj, t0, t1),
        ]
        spans = sorted(spans, key=lambda x: x[1])
        return f"Aadj_p{adj}", spans
    raise ValueError(variant)


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

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / "pilot_multipage_v1"
    ckpt_dir = out_dir / "ckpts"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"
    gates_path = out_dir / "gates.jsonl"
    summary_path = out_dir / "summary.json"
    progress_path = out_dir / "progress.json"
    metrics_path = out_dir / "metrics.jsonl"

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    snap = Path(args.hf_home) / "hub" / "models--meta-llama--Llama-3.1-8B-Instruct" / "snapshots"
    load_from = str(list(snap.iterdir())[0]) if snap.exists() and list(snap.iterdir()) else args.model

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[mp] load {load_from} device={device} dtype={dtype} b={args.b} variants={variants}", flush=True)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        print(f"[mp] gpu={torch.cuda.get_device_name(device)} free_gb={free/1e9:.2f}", flush=True)

    tok = AutoTokenizer.from_pretrained(load_from, local_files_only=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        load_from, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    print(f"[mp] model ready in {time.time()-t_load:.1f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    lh_list = sample_layer_heads(n_layers=n_layers, n_kv_heads=n_kv, seed=20261105)
    print(f"[mp] (l,h) combos ({len(lh_list)}): {lh_list}", flush=True)

    prefixes = build_pilot_prefixes(tok, L_p=args.L_p, n=args.n_prefix, seed=args.seed)
    if args.max_prefixes:
        prefixes = prefixes[: args.max_prefixes]

    (out_dir / "design.json").write_text(
        json.dumps(
            {
                "template_family": "archive_alpha_beta_pilot",
                "arm": "multipage_joint",
                "variants": variants,
                "seed": args.seed,
                "n_prefix": len(prefixes),
                "L_p": args.L_p,
                "g": args.g,
                "b": args.b,
                "metric_def": {
                    "AB": "Joint quant of pages A+B. Report mean S_jia and S_yi (both fact pages perturbed). Classic paired D not applicable.",
                    "Aadj": "Joint quant of page A + one adjacent closed page (not B). Report S_jia (need A), S_yi (need-not A), gap=S_jia-S_yi.",
                },
                "compare_to": "pilot_32 b=4 single-page D; pilot_b2_v1 if present",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "prefixes.json").write_text(
        json.dumps([summarize_prefix(sp) for sp in prefixes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "layer_heads.json").write_text(json.dumps(lh_list), encoding="utf-8")

    done = set(p.stem for p in ckpt_dir.glob("*.done"))
    print(f"[mp] resume done={len(done)}", flush=True)

    gate_stats = {
        "hash_checks": 0,
        "hash_fail": 0,
        "fp_both_correct": 0,
        "fp_checked": 0,
        "zero_pert_checks": 0,
        "zero_pert_fail": 0,
        "prefix_failures": [],
    }

    def append_jsonl(path: Path, obj: dict):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    t0 = time.time()
    n_traj = 0
    stop_expand = False

    with torch.no_grad():
        for pi, sp in enumerate(prefixes):
            if stop_expand:
                break
            prefix_t0 = time.time()
            print(
                f"[mp] === {sp.prefix_id} ({pi+1}/{len(prefixes)}) shared={len(sp.shared_ids)} "
                f"A={sp.page_a_index} B={sp.page_b_index} ===",
                flush=True,
            )

            fp_mono = {}
            for q in ("jia", "yi"):
                r = run_fp_trajectory(model, sp, question=q, device=device)
                fp_mono[q] = r
                append_jsonl(
                    traj_path,
                    {
                        "prefix_id": sp.prefix_id,
                        "variant": None,
                        "l": None,
                        "h": None,
                        "page": "-",
                        "question": q,
                        "intervene": "none",
                        "mode": "fp",
                        "S": 0.0,
                        "S_key": 0.0,
                        "nll_run": r.nll_run,
                        "page_hash": None,
                        "bits": args.b,
                    },
                )
                n_traj += 1

            both_ok = fp_correct(fp_mono["jia"].answer_logprobs) and fp_correct(fp_mono["yi"].answer_logprobs)
            gate_stats["fp_checked"] += 1
            if both_ok:
                gate_stats["fp_both_correct"] += 1
            else:
                gate_stats["prefix_failures"].append({"prefix_id": sp.prefix_id, "reason": "fp_not_both_correct"})
                print(f"[mp] WARN FP both-correct FAIL {sp.prefix_id}", flush=True)

            # Precompute chunk-matched FP per variant (chunk_at = max t1)
            fp_chunk: Dict[str, Dict[str, List[float]]] = {}
            span_map: Dict[str, List[Tuple[int, int, int]]] = {}
            label_map: Dict[str, str] = {}
            for vname in variants:
                label, spans = spans_for_variant(sp, vname, args.L_p)
                span_map[vname] = spans
                label_map[vname] = label
                t_close = max(t1 for _, _, t1 in spans)
                fp_chunk[vname] = {}
                for q in ("jia", "yi"):
                    r = run_fp_trajectory(model, sp, question=q, device=device, chunk_at=t_close)
                    fp_chunk[vname][q] = [-lp for lp in r.answer_logprobs]

            for li, (layer, kv_head) in enumerate(lh_list):
                for vname in variants:
                    ckpt_key = f"{sp.prefix_id}_{vname}_L{layer}_H{kv_head}_b{args.b}"
                    if ckpt_key in done:
                        continue
                    spans = span_map[vname]
                    label = label_map[vname]
                    for iv in INTERVENES:
                        rj, ry, joint_hash, _ = run_fork_pair_pages(
                            model, sp,
                            page_spans=spans,
                            intervene=iv, mode="quant",
                            layer=layer, kv_head=kv_head,
                            L_p=args.L_p, bits=args.b, group=args.g,
                            device=device,
                            fp_nlls_jia=fp_chunk[vname]["jia"],
                            fp_nlls_yi=fp_chunk[vname]["yi"],
                            page_label=label,
                        )
                        # hash gate: jia/yi share joint_hash by construction
                        gate_stats["hash_checks"] += 1
                        if rj.page_hash != ry.page_hash:
                            gate_stats["hash_fail"] += 1
                            stop_expand = True
                            append_jsonl(gates_path, {"prefix_id": sp.prefix_id, "gate": "hash", "pass": False})

                        for r in (rj, ry):
                            append_jsonl(
                                traj_path,
                                {
                                    "prefix_id": sp.prefix_id,
                                    "variant": vname,
                                    "page_label": label,
                                    "page_spans": [{"pi": pi_, "t0": t0, "t1": t1} for pi_, t0, t1 in spans],
                                    "l": layer,
                                    "h": kv_head,
                                    "page": label,
                                    "question": r.question,
                                    "intervene": iv,
                                    "mode": "quant",
                                    "S": r.S,
                                    "S_key": r.S_key,
                                    "nll_run": r.nll_run,
                                    "page_hash": r.page_hash,
                                    "bits": args.b,
                                    "t_key": r.t_key,
                                    "t_page_end": r.t_page_end,
                                    "d": r.d,
                                },
                            )
                            n_traj += 1

                        gap = rj.S - ry.S  # for Aadj: need - need-not; for AB: S_jia - S_yi exploratory
                        append_jsonl(
                            metrics_path,
                            {
                                "prefix_id": sp.prefix_id,
                                "variant": vname,
                                "page_label": label,
                                "l": layer,
                                "h": kv_head,
                                "intervene": iv,
                                "bits": args.b,
                                "S_jia": rj.S,
                                "S_yi": ry.S,
                                "S_key_jia": rj.S_key,
                                "S_key_yi": ry.S_key,
                                "gap_jia_minus_yi": gap,
                                "page_hash": joint_hash,
                            },
                        )

                    # zero-pert once per prefix on first LH, first variant
                    if li == 0 and vname == variants[0] and (pi % max(1, args.zero_pert_every) == 0):
                        rj, ry, _, _ = run_fork_pair_pages(
                            model, sp,
                            page_spans=spans,
                            intervene="k", mode="zero_pert",
                            layer=layer, kv_head=kv_head,
                            L_p=args.L_p, bits=args.b, group=args.g,
                            device=device,
                            fp_nlls_jia=fp_chunk[vname]["jia"],
                            fp_nlls_yi=fp_chunk[vname]["yi"],
                            page_label=label,
                        )
                        for r in (rj, ry):
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
                                    "variant": vname,
                                },
                            )
                            print(f"[mp] zero_pert {sp.prefix_id} {r.question}: S={r.S:.3e} pass={ok}", flush=True)

                    (ckpt_dir / f"{ckpt_key}.done").write_text(str(time.time()), encoding="utf-8")
                    done.add(ckpt_key)
                    progress_path.write_text(
                        json.dumps(
                            {
                                "prefix_id": sp.prefix_id,
                                "variant": vname,
                                "layer": layer,
                                "kv_head": kv_head,
                                "done_ckpts": len(done),
                                "n_traj": n_traj,
                                "elapsed_s": time.time() - t0,
                                "gate_stats": {k: v for k, v in gate_stats.items() if k != "prefix_failures"},
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    if stop_expand:
                        break
                if stop_expand:
                    break

            print(f"[mp] prefix {sp.prefix_id} done in {time.time()-prefix_t0:.1f}s", flush=True)

    # Finalize
    rows = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    headline = {}
    for vname in variants:
        for iv in INTERVENES:
            sub = [r for r in rows if r["variant"] == vname and r["intervene"] == iv]
            s_jia = [r["S_jia"] for r in sub]
            s_yi = [r["S_yi"] for r in sub]
            gaps = [r["gap_jia_minus_yi"] for r in sub]
            headline[f"{vname}_{iv}"] = {
                "n": len(sub),
                "S_jia_mean": (sum(s_jia) / len(s_jia)) if s_jia else None,
                "S_jia_median": pct(s_jia, 50),
                "S_jia_p95": pct(s_jia, 95),
                "S_yi_mean": (sum(s_yi) / len(s_yi)) if s_yi else None,
                "S_yi_median": pct(s_yi, 50),
                "S_yi_p95": pct(s_yi, 95),
                "gap_jia_minus_yi_mean": (sum(gaps) / len(gaps)) if gaps else None,
                "gap_jia_minus_yi_median": pct(gaps, 50),
                "gap_jia_minus_yi_p95": pct(gaps, 95),
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
            "variants": variants,
            "layer_heads": lh_list,
            "device": str(device),
            "dtype": args.dtype,
            "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "arm": "multipage_joint",
            "seed": args.seed,
            "metric_def": {
                "AB": "S_jia/S_yi under joint A+B quant; gap exploratory",
                "Aadj": "gap = S_jia - S_yi under A+adjacent quant (need vs need-not for A)",
            },
        },
        "elapsed_s": time.time() - t0,
        "n_traj_written": n_traj,
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
            "prefix_failures": gate_stats["prefix_failures"],
        },
        "headline": headline,
        "gate_stats_raw": gate_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# pilot_multipage_v1 — joint multi-page quant\n\n"
        f"- bits: {args.b}\n"
        f"- variants: {variants}\n"
        f"- elapsed_s: {summary['elapsed_s']}\n"
        f"- n_traj: {n_traj}\n"
        f"- stop_expand: {stop_expand}\n"
        f"- gates: {json.dumps(summary['gates'], ensure_ascii=False)}\n\n"
        "## Metric definition\n\n"
        "- **AB**: quantize pages A and B together. Report mean S_jia / S_yi. Classic paired D N/A.\n"
        "- **Aadj**: quantize page A + one adjacent closed page. gap = S_jia − S_yi (need A vs need-not A).\n\n"
        f"## Headline\n\n```\n{json.dumps(headline, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(headline, indent=2), flush=True)
    print(f"[mp] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
