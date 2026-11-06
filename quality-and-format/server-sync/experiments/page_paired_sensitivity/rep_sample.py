#!/usr/bin/env python3
"""Representative page sampling + protection-coverage pilot (v1).

Reuses the 32 retrieval-style prefixes (NOT a naturalistic load sample —
report template bias). Enumerates every closed L_p page, measures S/S_key
under K/V/KV for a documented stronger-(l,h) subset, then builds
within-prefix protection curves: fraction of positive page-harm mass
covered by protecting top-k% pages.
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
    summarize_prefix,
)
from experiments.page_paired_sensitivity.pages import closed_pages
from experiments.page_paired_sensitivity.kv_fork import (
    run_fp_trajectory,
    run_fork_pair_span,
)

# Stronger heads from pilot_32 breakdown (early/late concentration)
DEFAULT_HEADS = [(10, 3), (2, 1), (22, 2), (22, 4), (24, 6)]
INTERVENES = ("k", "v", "kv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-home", default="/cluster/home/zengy/hf-cache")
    p.add_argument("--L-p", type=int, default=64)
    p.add_argument("--g", type=int, default=64)
    p.add_argument("--b", type=int, default=4)
    p.add_argument("--n-prefix", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--max-prefixes", type=int, default=None)
    p.add_argument("--intervenes", default="k,v,kv")
    p.add_argument("--heads", default="10:3,2:1,22:2,22:4,24:6")
    return p.parse_args()


def parse_heads(s: str) -> List[Tuple[int, int]]:
    out = []
    for part in s.split(","):
        a, b = part.strip().split(":")
        out.append((int(a), int(b)))
    return out


def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def protection_curve(harms: List[float], fracs=(0.05, 0.1, 0.2, 0.25, 0.5, 1.0)) -> dict:
    """harms: per-page positive harm H_p >= 0. Returns coverage at top fractions + pages for 50/80/95%."""
    pos = sorted([max(0.0, h) for h in harms], reverse=True)
    total = sum(pos)
    n = len(pos)
    if n == 0:
        return {"n_pages": 0, "total_pos_H": 0.0, "coverage_at_frac": {}, "pages_for_mass": {}}
    csum = []
    s = 0.0
    for h in pos:
        s += h
        csum.append(s)

    def cov_at_k(k_pages: int) -> float:
        if total <= 0:
            return 1.0 if k_pages >= 0 else 0.0
        k = max(0, min(n, k_pages))
        if k == 0:
            return 0.0
        return csum[k - 1] / total

    coverage_at_frac = {}
    for f in fracs:
        k = max(1, int(math.ceil(f * n))) if f < 1 else n
        if f == 0:
            k = 0
        coverage_at_frac[str(f)] = {
            "k_pages": k,
            "frac_pages": k / n,
            "coverage": cov_at_k(k),
        }

    pages_for_mass = {}
    for mass in (0.5, 0.8, 0.95):
        if total <= 0:
            pages_for_mass[str(mass)] = {"k_pages": 0, "frac_pages": 0.0}
            continue
        k = n
        for i, cs in enumerate(csum):
            if cs / total >= mass:
                k = i + 1
                break
        pages_for_mass[str(mass)] = {"k_pages": k, "frac_pages": k / n}

    return {
        "n_pages": n,
        "total_pos_H": total,
        "max_H": pos[0] if pos else 0.0,
        "coverage_at_frac": coverage_at_frac,
        "pages_for_mass": pages_for_mass,
    }


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
    heads = parse_heads(args.heads)
    intervenes = tuple(x.strip() for x in args.intervenes.split(",") if x.strip())

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / "rep_sample_v1"
    ckpt_dir = out_dir / "ckpts"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"
    gates_path = out_dir / "gates.jsonl"
    pages_path = out_dir / "page_harms.jsonl"
    progress_path = out_dir / "progress.json"

    snap = Path(args.hf_home) / "hub" / "models--meta-llama--Llama-3.1-8B-Instruct" / "snapshots"
    load_from = str(list(snap.iterdir())[0]) if snap.exists() and list(snap.iterdir()) else "meta-llama/Llama-3.1-8B-Instruct"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[rep] load {load_from} device={device}", flush=True)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        print(f"[rep] gpu={torch.cuda.get_device_name(device)} free_gb={free/1e9:.2f}", flush=True)

    tok = AutoTokenizer.from_pretrained(load_from, local_files_only=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        load_from, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    ).to(device).eval()
    print(f"[rep] model ready in {time.time()-t_load:.1f}s", flush=True)

    prefixes = build_pilot_prefixes(tok, L_p=args.L_p, n=args.n_prefix, seed=args.seed)
    if args.max_prefixes:
        prefixes = prefixes[: args.max_prefixes]

    design = {
        "caveat": (
            "Prefixes are the same retrieval-style paired templates as pilot_32 "
            "(Archive Alpha/Beta codewords). Retrieval frequency is experimenter-set; "
            "do NOT treat page-sensitivity rates as naturalistic load estimates."
        ),
        "n_prefix": len(prefixes),
        "L_p": args.L_p,
        "g": args.g,
        "b": args.b,
        "heads": heads,
        "heads_rationale": "Stronger |mean D| cells from pilot_32 breakdown (early/late).",
        "intervenes": list(intervenes),
        "model": load_from,
        "harm_def": "H_p = max(0,S_jia)+max(0,S_yi) for a fixed (l,h,intervene); protection keeps top pages FP.",
    }
    (out_dir / "design.json").write_text(json.dumps(design, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "prefixes.json").write_text(
        json.dumps([summarize_prefix(sp) for sp in prefixes], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    done = set(p.stem for p in ckpt_dir.glob("*.done"))
    print(f"[rep] resume done={len(done)} heads={heads} intervenes={intervenes}", flush=True)

    gate_stats = {
        "hash_checks": 0,
        "hash_fail": 0,
        "zero_pert_checks": 0,
        "zero_pert_fail": 0,
    }
    t0 = time.time()
    n_traj = 0
    stop = False

    with torch.no_grad():
        for pi, sp in enumerate(prefixes):
            if stop:
                break
            pages = closed_pages(sp, args.L_p)
            print(
                f"[rep] === {sp.prefix_id} ({pi+1}/{len(prefixes)}) n_pages={len(pages)} "
                f"buckets={[p['page_pos_bucket'] for p in pages]} ===",
                flush=True,
            )
            prefix_t0 = time.time()

            # FP chunk baselines keyed by chunk_at (= page end exclusive = t1)
            # Need one baseline per distinct t1 among pages
            t1s = sorted({p["t1"] for p in pages})
            fp_chunk: Dict[int, Dict[str, List[float]]] = {}
            for t1 in t1s:
                fp_chunk[t1] = {}
                for q in ("jia", "yi"):
                    r = run_fp_trajectory(model, sp, question=q, device=device, chunk_at=t1)
                    fp_chunk[t1][q] = [-lp for lp in r.answer_logprobs]

            # Zero-pert once per prefix on first page × first head × k
            if pages and heads:
                p0 = pages[0]
                L0, H0 = heads[0]
                ck_zp = f"{sp.prefix_id}_zp"
                if ck_zp not in done:
                    rj, ry, _, _ = run_fork_pair_span(
                        model, sp,
                        page_index=p0["page_index"], t0=p0["t0"], t1=p0["t1"],
                        intervene="k", mode="zero_pert",
                        layer=L0, kv_head=H0, L_p=args.L_p, bits=args.b, group=args.g,
                        device=device,
                        fp_nlls_jia=fp_chunk[p0["t1"]]["jia"],
                        fp_nlls_yi=fp_chunk[p0["t1"]]["yi"],
                    )
                    for r in (rj, ry):
                        gate_stats["zero_pert_checks"] += 1
                        ok = abs(r.S) <= 1e-4 and abs(r.S_key) <= 1e-4
                        if not ok:
                            gate_stats["zero_pert_fail"] += 1
                            stop = True
                        append_jsonl(
                            gates_path,
                            {"prefix_id": sp.prefix_id, "gate": "zero_pert", "pass": ok, "S": r.S, "S_key": r.S_key, "q": r.question},
                        )
                        print(f"[rep] zero_pert {sp.prefix_id} {r.question}: S={r.S:.3e} pass={ok}", flush=True)
                    (ckpt_dir / f"{ck_zp}.done").write_text(str(time.time()), encoding="utf-8")
                    done.add(ck_zp)
                    if stop:
                        print("[rep] STOP: zero-pert gate failed", flush=True)
                        break

            for page in pages:
                for layer, kv_head in heads:
                    for iv in intervenes:
                        ck = f"{sp.prefix_id}_pg{page['page_index']}_L{layer}_H{kv_head}_{iv}"
                        if ck in done:
                            continue
                        rj, ry, packed, _ = run_fork_pair_span(
                            model, sp,
                            page_index=page["page_index"], t0=page["t0"], t1=page["t1"],
                            intervene=iv, mode="quant",  # type: ignore
                            layer=layer, kv_head=kv_head, L_p=args.L_p, bits=args.b, group=args.g,
                            device=device,
                            fp_nlls_jia=fp_chunk[page["t1"]]["jia"],
                            fp_nlls_yi=fp_chunk[page["t1"]]["yi"],
                        )
                        gate_stats["hash_checks"] += 1
                        if rj.page_hash != ry.page_hash:
                            gate_stats["hash_fail"] += 1
                            stop = True
                            append_jsonl(gates_path, {"prefix_id": sp.prefix_id, "gate": "hash", "pass": False, "ck": ck})
                            print(f"[rep] HASH FAIL {ck}", flush=True)

                        H = max(0.0, rj.S) + max(0.0, ry.S)
                        H_key = max(0.0, rj.S_key) + max(0.0, ry.S_key)
                        for r in (rj, ry):
                            append_jsonl(
                                traj_path,
                                {
                                    "prefix_id": sp.prefix_id,
                                    "page_index": page["page_index"],
                                    "page_kind": page["kind"],
                                    "page_pos_bucket": page["page_pos_bucket"],
                                    "d": page["d"],
                                    "t_page_end": page["t_page_end"],
                                    "l": layer,
                                    "h": kv_head,
                                    "intervene": iv,
                                    "mode": "quant",
                                    "question": r.question,
                                    "S": r.S,
                                    "S_key": r.S_key,
                                    "page_hash": r.page_hash,
                                    "t_key": r.t_key,
                                },
                            )
                            n_traj += 1
                        append_jsonl(
                            pages_path,
                            {
                                "prefix_id": sp.prefix_id,
                                "page_index": page["page_index"],
                                "page_kind": page["kind"],
                                "page_pos_bucket": page["page_pos_bucket"],
                                "d": page["d"],
                                "l": layer,
                                "h": kv_head,
                                "intervene": iv,
                                "S_jia": rj.S,
                                "S_yi": ry.S,
                                "S_key_jia": rj.S_key,
                                "S_key_yi": ry.S_key,
                                "H": H,
                                "H_key": H_key,
                                "page_hash": rj.page_hash,
                            },
                        )
                        (ckpt_dir / f"{ck}.done").write_text(str(time.time()), encoding="utf-8")
                        done.add(ck)
                        if stop:
                            break
                    if stop:
                        break
                if stop:
                    break

            progress_path.write_text(
                json.dumps(
                    {
                        "prefix_id": sp.prefix_id,
                        "done_ckpts": len(done),
                        "n_traj": n_traj,
                        "elapsed_s": time.time() - t0,
                        "gate_stats": gate_stats,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[rep] prefix {sp.prefix_id} done in {time.time()-prefix_t0:.1f}s", flush=True)

    summary = build_summary(out_dir, design, gate_stats, heads, intervenes, t0, n_traj, stop, device, load_from, args)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out_dir, summary)
    print(json.dumps(summary.get("headline", {}), indent=2, ensure_ascii=False), flush=True)
    print(f"[rep] wrote {out_dir/'summary.json'} stop={stop}", flush=True)


def build_summary(out_dir, design, gate_stats, heads, intervenes, t0, n_traj, stop, device, load_from, args):
    pages_path = out_dir / "page_harms.jsonl"
    rows = []
    if pages_path.exists():
        for line in pages_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    # Per (prefix,l,h,iv) protection curves
    groups = defaultdict(list)
    for r in rows:
        key = (r["prefix_id"], r["l"], r["h"], r["intervene"])
        groups[key].append(r)

    curves = []
    for key, items in groups.items():
        items = sorted(items, key=lambda x: x["page_index"])
        harms = [x["H"] for x in items]
        curve = protection_curve(harms)
        curves.append(
            {
                "prefix_id": key[0],
                "l": key[1],
                "h": key[2],
                "intervene": key[3],
                "n_pages": curve["n_pages"],
                "total_pos_H": curve["total_pos_H"],
                "coverage_at_frac": curve["coverage_at_frac"],
                "pages_for_mass": curve["pages_for_mass"],
                "by_bucket_mean_H": _bucket_means(items),
            }
        )

    # Aggregate mean coverage across prefixes for each (l,h,iv) at fixed fracs
    agg = defaultdict(lambda: defaultdict(list))
    pages_for = defaultdict(lambda: defaultdict(list))
    for c in curves:
        gkey = (c["l"], c["h"], c["intervene"])
        for f, v in c["coverage_at_frac"].items():
            agg[gkey][f].append(v["coverage"])
        for m, v in c["pages_for_mass"].items():
            pages_for[gkey][m].append(v["frac_pages"])

    headline = {}
    for (L, H, iv), fmap in sorted(agg.items()):
        headline[f"L{L}_H{H}_{iv}"] = {
            "mean_coverage_at_frac": {f: (sum(vs) / len(vs) if vs else None) for f, vs in fmap.items()},
            "mean_frac_pages_for_mass": {
                m: (sum(vs) / len(vs) if vs else None) for m, vs in pages_for[(L, H, iv)].items()
            },
            "n_prefix_curves": len(next(iter(fmap.values()))) if fmap else 0,
        }

    # Pooled across all heads for each intervene (mean of prefix curves averaged over heads)
    pool = {}
    for iv in intervenes:
        cov = defaultdict(list)
        mass = defaultdict(list)
        for (L, H, iv2), fmap in agg.items():
            if iv2 != iv:
                continue
            for f, vs in fmap.items():
                cov[f].extend(vs)
            for m, vs in pages_for[(L, H, iv)].items():
                mass[m].extend(vs)
        pool[iv] = {
            "mean_coverage_at_frac": {f: sum(vs) / len(vs) if vs else None for f, vs in cov.items()},
            "mean_frac_pages_for_mass": {m: sum(vs) / len(vs) if vs else None for m, vs in mass.items()},
            "n_curves": sum(len(vs) for vs in cov.values()) // max(1, len(cov)),
        }

    # Bucket stats of H
    bucket_H = defaultdict(list)
    kind_H = defaultdict(list)
    for r in rows:
        bucket_H[(r["page_pos_bucket"], r["intervene"])].append(r["H"])
        kind_H[(r["page_kind"], r["intervene"])].append(r["H"])

    def _summ(xs):
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "mean": sum(xs) / len(xs), "median": sorted(xs)[len(xs) // 2], "p95": sorted(xs)[int(0.95 * (len(xs) - 1))]}

    hash_rate = 1.0 - gate_stats["hash_fail"] / gate_stats["hash_checks"] if gate_stats["hash_checks"] else None
    zp_rate = 1.0 - gate_stats["zero_pert_fail"] / gate_stats["zero_pert_checks"] if gate_stats["zero_pert_checks"] else None

    # Persist per-curve detail
    (out_dir / "protection_curves.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in curves) + ("\n" if curves else ""),
        encoding="utf-8",
    )

    return {
        "design": design,
        "elapsed_s": time.time() - t0,
        "n_traj": n_traj,
        "n_page_harm_rows": len(rows),
        "n_curves": len(curves),
        "stop_expand": stop,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gates": {
            "hash_equality_rate": hash_rate,
            "hash_checks": gate_stats["hash_checks"],
            "hash_fail": gate_stats["hash_fail"],
            "zero_pert_pass_rate": zp_rate,
            "zero_pert_checks": gate_stats["zero_pert_checks"],
            "zero_pert_fail": gate_stats["zero_pert_fail"],
        },
        "headline": {
            "pooled_by_intervene": pool,
            "by_lh_intervene": headline,
            "H_by_bucket": {f"{b}|{iv}": _summ(vs) for (b, iv), vs in sorted(bucket_H.items())},
            "H_by_kind": {f"{k}|{iv}": _summ(vs) for (k, iv), vs in sorted(kind_H.items())},
        },
    }


def _bucket_means(items):
    d = defaultdict(list)
    for x in items:
        d[x["page_pos_bucket"]].append(x["H"])
    return {b: (sum(vs) / len(vs) if vs else 0.0) for b, vs in d.items()}


def write_readme(out_dir: Path, summary: dict):
    pool = summary["headline"]["pooled_by_intervene"]
    lines = [
        "# Representative sampling / protection coverage v1",
        "",
        summary["design"]["caveat"],
        "",
        f"- elapsed_s: {summary['elapsed_s']:.1f}",
        f"- n_traj: {summary['n_traj']}, n_curves: {summary['n_curves']}",
        f"- gates: {json.dumps(summary['gates'])}",
        f"- heads: {summary['design']['heads']}",
        "",
        "## Pooled protection curve (mean over prefix×head curves)",
        "",
    ]
    for iv, v in pool.items():
        lines.append(f"### intervene={iv}")
        lines.append(f"- mean coverage @ top frac pages: {json.dumps(v['mean_coverage_at_frac'])}")
        lines.append(f"- mean frac pages to cover mass: {json.dumps(v['mean_frac_pages_for_mass'])}")
        lines.append("")
    lines.append("## H by bucket")
    lines.append(json.dumps(summary["headline"]["H_by_bucket"], indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("## H by page kind (fact_A/B vs other)")
    lines.append(json.dumps(summary["headline"]["H_by_kind"], indent=2, ensure_ascii=False))
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
