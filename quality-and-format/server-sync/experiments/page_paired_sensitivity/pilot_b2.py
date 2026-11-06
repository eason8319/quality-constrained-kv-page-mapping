#!/usr/bin/env python3
"""b=2 single-page paired pilot (stronger quant vs pilot_32 b=4).

Default out: runs/pilot_b2_v1/. Same templates/heads as pilot_32.
"""
from __future__ import annotations

import argparse
import json
import os
import time
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
    run_fork_pair,
)


INTERVENES = ("k", "v", "kv")
PAGES = ("A", "B")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--hf-home", default="/cluster/home/zengy/hf-cache")
    p.add_argument("--L-p", type=int, default=64)
    p.add_argument("--g", type=int, default=64)
    p.add_argument("--b", type=int, default=2)
    p.add_argument("--n-prefix", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--zero-pert-every", type=int, default=1, help="run zero-pert on first LH for every N prefixes (1=all)")
    p.add_argument("--max-prefixes", type=int, default=None, help="debug cap")
    p.add_argument("--skip-manip", action="store_true")
    return p.parse_args()


def _row(base: dict, **extra) -> dict:
    r = dict(base)
    r.update(extra)
    return r


def fp_correct(answer_logprobs: List[float], ans_ids: List[int], model_logits_ok: bool = True) -> bool:
    """Heuristic: teacher-forcing mean NLL low enough that argmax likely matches (nll < 2.0 nats)."""
    if not answer_logprobs:
        return False
    nll = [-lp for lp in answer_logprobs]
    return (sum(nll) / len(nll)) < 2.0


def compute_D(s_map: Dict[Tuple[str, str], float]) -> Optional[float]:
    """D = 1/2 * [(S_A,jia - S_A,yi) + (S_B,yi - S_B,jia)] for one intervene."""
    try:
        return 0.5 * (
            (s_map[("A", "jia")] - s_map[("A", "yi")])
            + (s_map[("B", "yi")] - s_map[("B", "jia")])
        )
    except KeyError:
        return None


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

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / "pilot_b2_v1"
    ckpt_dir = out_dir / "ckpts"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"
    gates_path = out_dir / "gates.jsonl"
    summary_path = out_dir / "summary.json"
    progress_path = out_dir / "progress.json"

    snap = Path(args.hf_home) / "hub" / "models--meta-llama--Llama-3.1-8B-Instruct" / "snapshots"
    load_from = str(list(snap.iterdir())[0]) if snap.exists() and list(snap.iterdir()) else args.model

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[b2] load {load_from} device={device} dtype={dtype}", flush=True)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        print(f"[b2] gpu={torch.cuda.get_device_name(device)} free_gb={free/1e9:.2f}", flush=True)

    tok = AutoTokenizer.from_pretrained(load_from, local_files_only=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        load_from, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    print(f"[b2] model ready in {time.time()-t_load:.1f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    lh_list = sample_layer_heads(n_layers=n_layers, n_kv_heads=n_kv, seed=args.seed)
    print(f"[b2] (l,h) combos ({len(lh_list)}): {lh_list}", flush=True)

    prefixes = build_pilot_prefixes(tok, L_p=args.L_p, n=args.n_prefix, seed=args.seed)
    (out_dir / "design.json").write_text(
        json.dumps({
            "template_family": "archive_alpha_beta_pilot",
            "arm": "b2_single_page",
            "seed": args.seed,
            "n_prefix": args.n_prefix,
            "L_p": args.L_p, "g": args.g, "b": args.b,
            "compare_to": "runs/pilot_32 (b=4 single-page)",
            "caveat": "Same templates/heads as pilot_32; only bits changed to 2 to test whether stronger quant raises paired D.",
        }, indent=2), encoding="utf-8")
    if args.max_prefixes:
        prefixes = prefixes[: args.max_prefixes]
    (out_dir / "prefixes.json").write_text(
        json.dumps([summarize_prefix(sp) for sp in prefixes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "layer_heads.json").write_text(json.dumps(lh_list), encoding="utf-8")

    # Resume: completed ckpt keys
    done = set(p.stem for p in ckpt_dir.glob("*.done"))
    print(f"[b2] resume done={len(done)}", flush=True)

    gate_stats = {
        "hash_checks": 0,
        "hash_fail": 0,
        "fp_both_correct": 0,
        "fp_checked": 0,
        "zero_pert_checks": 0,
        "zero_pert_fail": 0,
        "manip_checks": 0,
        "manip_pass": 0,
        "prefix_failures": [],
    }

    t0 = time.time()
    n_traj_written = 0
    stop_expand = False

    def append_jsonl(path: Path, obj: dict):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with torch.no_grad():
        for pi, sp in enumerate(prefixes):
            if stop_expand:
                break
            prefix_t0 = time.time()
            meta_jia = enrich_meta(sp, "jia")
            meta_yi = enrich_meta(sp, "yi")
            print(
                f"[b2] === {sp.prefix_id} ({pi+1}/{len(prefixes)}) "
                f"shared={len(sp.shared_ids)} A={sp.page_a} B={sp.page_b} "
                f"bucketA={meta_jia['bucket_a']} dA={meta_jia['d_a']} ===",
                flush=True,
            )

            # --- FP monolithic + chunked baselines ---
            fp_mono = {}
            fp_chunk = {"A": {}, "B": {}}  # page -> question -> nlls
            for q in ("jia", "yi"):
                r = run_fp_trajectory(model, sp, question=q, device=device)
                fp_mono[q] = r
                ans = sp.ans_jia_ids if q == "jia" else sp.ans_yi_ids
                append_jsonl(
                    traj_path,
                    _row(
                        {
                            "prefix_id": sp.prefix_id,
                            "l": None,
                            "h": None,
                            "c": f"b{args.b}_g{args.g}",
                            "page": "-",
                            "question": q,
                            "role": None,
                            "intervene": "none",
                            "mode": "fp",
                            "S": 0.0,
                            "S_key": 0.0,
                            "nll_run": r.nll_run,
                            "page_hash": None,
                            "t_key": r.t_key,
                            "t_page_end": None,
                            "d": None,
                            "page_pos_bucket": None,
                            "answer_logprobs": r.answer_logprobs,
                        }
                    ),
                )
                n_traj_written += 1

            both_ok = fp_correct(fp_mono["jia"].answer_logprobs, sp.ans_jia_ids) and fp_correct(
                fp_mono["yi"].answer_logprobs, sp.ans_yi_ids
            )
            gate_stats["fp_checked"] += 1
            if both_ok:
                gate_stats["fp_both_correct"] += 1
            else:
                gate_stats["prefix_failures"].append({"prefix_id": sp.prefix_id, "reason": "fp_not_both_correct"})
                print(f"[b2] WARN FP both-correct FAIL {sp.prefix_id}", flush=True)

            for page, chunk_at in (("A", sp.page_a[1]), ("B", sp.page_b[1])):
                for q in ("jia", "yi"):
                    r = run_fp_trajectory(model, sp, question=q, device=device, chunk_at=chunk_at)
                    fp_chunk[page][q] = [-lp for lp in r.answer_logprobs]

            # --- Manipulation check (TF): replace Alpha-page codeword, keep asking original answers ---
            if not args.skip_manip:
                from experiments.page_paired_sensitivity.prefix_builder import build_prefix

                # Rebuild with a different Alpha-side raw fact; preserve swap/trailing/layout.
                # After build, force teacher-forced answers to the ORIGINAL sp answers.
                # Change only the Alpha-page stored codeword; keep Beta-page fact identical.
                if not sp.swap_labels:
                    alt = build_prefix(
                        tok, L_p=args.L_p, prefix_id=sp.prefix_id + "_manip",
                        fact_a="ZZZZMANIP", fact_b=sp.fact_b, swap_labels=False,
                        n_trailing_pages=sp.n_trailing_pages,
                        put_a_on_first_content_page=(sp.page_a_index == 0),
                    )
                else:
                    # Alpha page content comes from fact_b arg under swap
                    alt = build_prefix(
                        tok, L_p=args.L_p, prefix_id=sp.prefix_id + "_manip",
                        fact_a=sp.fact_b, fact_b="ZZZZMANIP", swap_labels=True,
                        n_trailing_pages=sp.n_trailing_pages,
                        put_a_on_first_content_page=(sp.page_a_index == 0),
                    )
                alt.ans_jia_ids = list(sp.ans_jia_ids)
                alt.ans_yi_ids = list(sp.ans_yi_ids)
                alt.q_jia_ids = list(sp.q_jia_ids)
                alt.q_yi_ids = list(sp.q_yi_ids)
                r_m_jia = run_fp_trajectory(model, alt, question="jia", device=device)
                r_m_yi = run_fp_trajectory(model, alt, question="yi", device=device)
                rise = r_m_jia.nll_run - fp_mono["jia"].nll_run
                stab = abs(r_m_yi.nll_run - fp_mono["yi"].nll_run)
                # Alpha answer should get harder; Beta answer should stay roughly stable.
                manip_ok = (rise > 0.5) and (stab < 2.0)
                gate_stats["manip_checks"] += 1
                if manip_ok:
                    gate_stats["manip_pass"] += 1
                append_jsonl(
                    gates_path,
                    {
                        "prefix_id": sp.prefix_id,
                        "gate": "manip",
                        "pass": manip_ok,
                        "rise_jia": rise,
                        "stab_yi": stab,
                        "nll_jia_orig": fp_mono["jia"].nll_run,
                        "nll_jia_manip": r_m_jia.nll_run,
                        "nll_yi_orig": fp_mono["yi"].nll_run,
                        "nll_yi_manip": r_m_yi.nll_run,
                    },
                )
                print(f"[b2] manip {sp.prefix_id}: pass={manip_ok} rise_jia={rise:.3f} stab_yi={stab:.3f}", flush=True)

            # --- Per (l,h) quant trajectories ---
            for li, (layer, kv_head) in enumerate(lh_list):
                ckpt_key = f"{sp.prefix_id}_L{layer}_H{kv_head}"
                if ckpt_key in done:
                    continue

                # S maps for D per intervene
                s_by_iv: Dict[str, Dict[Tuple[str, str], float]] = {iv: {} for iv in INTERVENES}
                sk_by_iv: Dict[str, Dict[Tuple[str, str], float]] = {iv: {} for iv in INTERVENES}
                hash_fail_local = False

                for page in PAGES:
                    bucket = meta_jia["bucket_a"] if page == "A" else meta_jia["bucket_b"]
                    for iv in INTERVENES:
                        rj, ry, packed, _ = run_fork_pair(
                            model, sp,
                            target_page=page, intervene=iv, mode="quant",
                            layer=layer, kv_head=kv_head,
                            L_p=args.L_p, bits=args.b, group=args.g,
                            device=device,
                            fp_nlls_jia=fp_chunk[page]["jia"],
                            fp_nlls_yi=fp_chunk[page]["yi"],
                        )
                        gate_stats["hash_checks"] += 1
                        if rj.page_hash != ry.page_hash:
                            gate_stats["hash_fail"] += 1
                            hash_fail_local = True
                            print(f"[b2] HASH FAIL {ckpt_key} page={page} iv={iv}", flush=True)

                        for r in (rj, ry):
                            m = meta_jia if r.question == "jia" else meta_yi
                            d = m["d_a"] if page == "A" else m["d_b"]
                            bkt = m["bucket_a"] if page == "A" else m["bucket_b"]
                            append_jsonl(
                                traj_path,
                                {
                                    "prefix_id": sp.prefix_id,
                                    "l": layer,
                                    "h": kv_head,
                                    "c": f"b{args.b}_g{args.g}",
                                    "page": page,
                                    "question": r.question,
                                    "role": r.role,
                                    "intervene": iv,
                                    "mode": "quant",
                                    "S": r.S,
                                    "S_key": r.S_key,
                                    "nll_run": r.nll_run,
                                    "page_hash": r.page_hash,
                                    "t_key": r.t_key,
                                    "t_page_end": r.t_page_end,
                                    "d": d,
                                    "page_pos_bucket": bkt,
                                    "answer_logprobs": r.answer_logprobs,
                                },
                            )
                            n_traj_written += 1
                            s_by_iv[iv][(page, r.question)] = r.S
                            sk_by_iv[iv][(page, r.question)] = r.S_key

                if hash_fail_local:
                    gate_stats["prefix_failures"].append(
                        {"prefix_id": sp.prefix_id, "lh": [layer, kv_head], "reason": "hash_mismatch"}
                    )
                    # Protocol: rerun prefix cluster — mark not done, stop expansion batch
                    stop_expand = True
                    append_jsonl(gates_path, {"prefix_id": sp.prefix_id, "gate": "hash", "pass": False, "lh": [layer, kv_head]})
                    break

                # D aggregates for this (prefix,l,h)
                for iv in INTERVENES:
                    D = compute_D(s_by_iv[iv])
                    Dk = compute_D(sk_by_iv[iv])
                    append_jsonl(
                        out_dir / "D_by_prefix_lh.jsonl",
                        {
                            "prefix_id": sp.prefix_id,
                            "l": layer,
                            "h": kv_head,
                            "intervene": iv,
                            "D": D,
                            "D_key": Dk,
                            "S_need_minus_unneed_A": s_by_iv[iv][("A", "jia")] - s_by_iv[iv][("A", "yi")],
                            "S_need_minus_unneed_B": s_by_iv[iv][("B", "yi")] - s_by_iv[iv][("B", "jia")],
                        },
                    )

                # Zero-pert spot: first LH of selected prefixes
                if li == 0 and (pi % max(1, args.zero_pert_every) == 0):
                    for page in ("A",):
                        rj, ry, _, _ = run_fork_pair(
                            model, sp,
                            target_page=page, intervene="k", mode="zero_pert",
                            layer=layer, kv_head=kv_head,
                            L_p=args.L_p, bits=args.b, group=args.g,
                            device=device,
                            fp_nlls_jia=fp_chunk[page]["jia"],
                            fp_nlls_yi=fp_chunk[page]["yi"],
                        )
                        for r in (rj, ry):
                            gate_stats["zero_pert_checks"] += 1
                            ok = abs(r.S) <= 1e-4 and abs(r.S_key) <= 1e-4
                            if not ok:
                                gate_stats["zero_pert_fail"] += 1
                                stop_expand = True
                            append_jsonl(
                                traj_path,
                                {
                                    "prefix_id": sp.prefix_id,
                                    "l": layer,
                                    "h": kv_head,
                                    "c": f"b{args.b}_g{args.g}",
                                    "page": page,
                                    "question": r.question,
                                    "role": r.role,
                                    "intervene": "k",
                                    "mode": "zero_pert",
                                    "S": r.S,
                                    "S_key": r.S_key,
                                    "nll_run": r.nll_run,
                                    "page_hash": r.page_hash,
                                    "t_key": r.t_key,
                                    "t_page_end": r.t_page_end,
                                    "d": r.d,
                                    "page_pos_bucket": None,
                                    "zero_pert_pass": ok,
                                },
                            )
                            n_traj_written += 1
                            append_jsonl(
                                gates_path,
                                {"prefix_id": sp.prefix_id, "gate": "zero_pert", "pass": ok, "S": r.S, "S_key": r.S_key, "q": r.question},
                            )
                            print(f"[b2] zero_pert {sp.prefix_id} {r.question}: S={r.S:.3e} pass={ok}", flush=True)
                        if stop_expand:
                            break

                # mark ckpt done
                (ckpt_dir / f"{ckpt_key}.done").write_text(str(time.time()), encoding="utf-8")
                done.add(ckpt_key)

                progress_path.write_text(
                    json.dumps(
                        {
                            "prefix_id": sp.prefix_id,
                            "layer": layer,
                            "kv_head": kv_head,
                            "done_ckpts": len(done),
                            "n_traj_written": n_traj_written,
                            "elapsed_s": time.time() - t0,
                            "gate_stats": gate_stats,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                if stop_expand:
                    break

            print(f"[b2] prefix {sp.prefix_id} done in {time.time()-prefix_t0:.1f}s", flush=True)

            # Gate: FP rate running
            if gate_stats["fp_checked"] >= 5:
                rate = gate_stats["fp_both_correct"] / gate_stats["fp_checked"]
                if rate < 0.90:
                    print(f"[b2] FP both-correct rate {rate:.2%} < 90% — continuing but will flag", flush=True)
            if gate_stats["manip_checks"] >= 8:
                mrate = gate_stats["manip_pass"] / gate_stats["manip_checks"]
                if mrate < 0.95:
                    print(f"[b2] manip rate {mrate:.2%} < 95% — STOP expand (protocol)", flush=True)
                    stop_expand = True

    # Finalize summary
    summary = finalize_summary(out_dir, gate_stats, lh_list, args, t0, n_traj_written, load_from, device, stop_expand)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["headline"], indent=2, ensure_ascii=False), flush=True)
    print(f"[b2] wrote {summary_path}", flush=True)
    (out_dir / "README.md").write_text(
        "# pilot_b2_v1 — single-page b=2\n\n"
        f"- elapsed_s: {summary.get('elapsed_s')}\n"
        f"- gates: {json.dumps(summary.get('gates'), ensure_ascii=False)}\n"
        f"- stop_expand: {summary.get('stop_expand')}\n\n"
        "## Headline D vs pilot_32 (b=4: K 8.75e-5 / V 6.76e-5 / KV 5.00e-5)\n\n"
        f"```\n{json.dumps(summary.get('headline'), indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(f"[b2] wrote README", flush=True)


def finalize_summary(out_dir, gate_stats, lh_list, args, t0, n_traj_written, load_from, device, stop_expand):
    import math
    import statistics as stats

    D_rows = []
    dpath = out_dir / "D_by_prefix_lh.jsonl"
    if dpath.exists():
        for line in dpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                D_rows.append(json.loads(line))

    traj = []
    tpath = out_dir / "trajectories.jsonl"
    if tpath.exists():
        for line in tpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                traj.append(json.loads(line))

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

    headline = {}
    for iv in INTERVENES:
        vals = [r["D"] for r in D_rows if r.get("intervene") == iv and r.get("D") is not None]
        headline[f"D_{iv}"] = {
            "n": len(vals),
            "mean": (sum(vals) / len(vals)) if vals else None,
            "median": pct(vals, 50),
            "p95": pct(vals, 95),
        }

    # S_need - S_unneed by intervene and bucket
    need_gap = []
    for r in traj:
        if r.get("mode") != "quant":
            continue
        need_gap.append(r)

    by_iv_bucket = {}
    # pair within prefix,l,h,page,intervene
    from collections import defaultdict

    pairs = defaultdict(dict)
    for r in traj:
        if r.get("mode") != "quant":
            continue
        key = (r["prefix_id"], r["l"], r["h"], r["page"], r["intervene"])
        pairs[key][r["question"]] = r
    gaps_by = defaultdict(list)
    for key, qd in pairs.items():
        if "jia" in qd and "yi" in qd:
            page = key[3]
            iv = key[4]
            if page == "A":
                gap = qd["jia"]["S"] - qd["yi"]["S"]  # need - unneed
                bkt = qd["jia"].get("page_pos_bucket")
            else:
                gap = qd["yi"]["S"] - qd["jia"]["S"]
                bkt = qd["yi"].get("page_pos_bucket")
            gaps_by[(iv, bkt)].append(gap)
            gaps_by[(iv, "ALL")].append(gap)

    gap_stats = {}
    for k, vals in sorted(gaps_by.items(), key=lambda x: str(x[0])):
        gap_stats[str(k)] = {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "median": pct(vals, 50),
            "p95": pct(vals, 95),
        }

    fp_rate = (gate_stats["fp_both_correct"] / gate_stats["fp_checked"]) if gate_stats["fp_checked"] else None
    hash_rate = (
        1.0 - gate_stats["hash_fail"] / gate_stats["hash_checks"] if gate_stats["hash_checks"] else None
    )
    zp_rate = (
        1.0 - gate_stats["zero_pert_fail"] / gate_stats["zero_pert_checks"]
        if gate_stats["zero_pert_checks"]
        else None
    )
    manip_rate = (
        gate_stats["manip_pass"] / gate_stats["manip_checks"] if gate_stats["manip_checks"] else None
    )

    return {
        "config": {
            "model": load_from,
            "L_p": args.L_p,
            "g": args.g,
            "b": args.b,
            "n_prefix": args.n_prefix,
            "layer_heads": lh_list,
            "device": str(device),
            "dtype": args.dtype,
            "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "arm": "b2_single_page",
            "seed": args.seed,
        },
        "elapsed_s": time.time() - t0,
        "n_traj_written": n_traj_written,
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
            "manip_pass_rate": manip_rate,
            "manip_checks": gate_stats["manip_checks"],
            "prefix_failures": gate_stats["prefix_failures"],
        },
        "headline": headline,
        "S_need_minus_unneed": gap_stats,
        "gate_stats_raw": gate_stats,
    }


if __name__ == "__main__":
    main()
