#!/usr/bin/env python3
"""1 prefix × 1 (layer, kv_head) paired-sensitivity smoke test.

Runs:
  - FP for Q_jia and Q_yi
  - Quant perturb page A, K-only, for both Q
  - Zero-pert full path (A, K-only intervene path with FP bytes) for both Q

Mandatory fork order is implemented in kv_fork.run_fork_trajectory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def _add_path():
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_add_path()

from experiments.page_paired_sensitivity.prefix_builder import build_smoke_prefix, summarize_prefix
from experiments.page_paired_sensitivity.kv_fork import run_fp_trajectory, run_fork_trajectory


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--hf-home", default="/cluster/home/zengy/hf-cache")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--kv-head", type=int, default=0)
    p.add_argument("--L-p", type=int, default=64)
    p.add_argument("--g", type=int, default=64)
    p.add_argument("--b", type=int, default=4)
    p.add_argument("--device", default=None, help="cuda:0 / cuda:1 / cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(args.hf_home, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(args.hf_home, "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = args.model
    # Prefer local snapshot if present
    snap = Path(args.hf_home) / "hub" / "models--meta-llama--Llama-3.1-8B-Instruct" / "snapshots"
    local_model = None
    if snap.exists():
        snaps = list(snap.iterdir())
        if snaps:
            local_model = str(snaps[0])

    load_from = local_model or model_id
    print(f"[smoke] loading tokenizer/model from {load_from}", flush=True)
    print(f"[smoke] device={device} dtype={dtype}", flush=True)
    if device.type == "cuda":
        print(f"[smoke] cuda_device={torch.cuda.get_device_name(device)}", flush=True)
        free, total = torch.cuda.mem_get_info(device)
        print(f"[smoke] mem_free_gb={free/1e9:.2f} total_gb={total/1e9:.2f}", flush=True)

    tok = AutoTokenizer.from_pretrained(load_from, local_files_only=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        load_from,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    print(f"[smoke] model loaded in {time.time()-t0:.1f}s", flush=True)

    sp = build_smoke_prefix(tok, L_p=args.L_p, prefix_id="smoke000")
    print("[smoke] prefix:", json.dumps(summarize_prefix(sp)), flush=True)

    results = []
    # FP both Q
    with torch.no_grad():
        fp = {}
        # Monolithic FP (protocol reference) + chunked FP matching page-A close schedule
        # (used as S baseline for forked runs to isolate intervention from SDPA-split noise).
        chunk_at = sp.page_a[1]
        for q in ("jia", "yi"):
            r = run_fp_trajectory(model, sp, question=q, device=device)
            results.append(r.__dict__)
            print(f"[smoke] FP {q}: mean_nll={r.nll_run:.6f} nlls={[round(-lp,6) for lp in r.answer_logprobs]}", flush=True)
            r_c = run_fp_trajectory(model, sp, question=q, device=device, chunk_at=chunk_at)
            fp[q] = [-lp for lp in r_c.answer_logprobs]
            results.append(r_c.__dict__)
            print(f"[smoke] FP-chunk@{chunk_at} {q}: mean_nll={r_c.nll_run:.6f}", flush=True)

        # Perturb A K-only both Q
        hashes = {}
        for q in ("jia", "yi"):
            r, packed = run_fork_trajectory(
                model, sp,
                target_page="A", question=q, intervene="k", mode="quant",
                layer=args.layer, kv_head=args.kv_head,
                L_p=args.L_p, bits=args.b, group=args.g,
                device=device, fp_nlls=fp[q],
            )
            hashes[q] = r.page_hash
            results.append(r.__dict__)
            print(
                f"[smoke] QUANT A/K {q}: S={r.S:.6e} S_key={r.S_key:.6e} hash={r.page_hash[:16]}… role={r.role}",
                flush=True,
            )

        hash_eq = hashes["jia"] == hashes["yi"]
        print(f"[smoke] hash_equality jia==yi: {hash_eq} ({hashes['jia']})", flush=True)

        # Zero-pert full path both Q
        for q in ("jia", "yi"):
            r, packed = run_fork_trajectory(
                model, sp,
                target_page="A", question=q, intervene="k", mode="zero_pert",
                layer=args.layer, kv_head=args.kv_head,
                L_p=args.L_p, bits=args.b, group=args.g,
                device=device, fp_nlls=fp[q],
            )
            results.append(r.__dict__)
            ok = abs(r.S) <= 1e-4 and abs(r.S_key) <= 1e-4
            print(
                f"[smoke] ZERO-PERT A/K {q}: S={r.S:.6e} S_key={r.S_key:.6e} pass={ok}",
                flush=True,
            )

    out = {
        "config": {
            "model": load_from,
            "layer": args.layer,
            "kv_head": args.kv_head,
            "L_p": args.L_p,
            "g": args.g,
            "b": args.b,
            "device": str(device),
            "dtype": args.dtype,
            "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "prefix": summarize_prefix(sp),
        "hash_equality_A_K": hash_eq,
        "hash_jia": hashes["jia"],
        "hash_yi": hashes["yi"],
        "results": results,
    }

    out_path = Path(args.out) if args.out else Path(__file__).resolve().parents[2] / "runs" / "smoke_1x1" / "result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[smoke] wrote {out_path}", flush=True)

    # Gate summary
    zp = [r for r in results if r["mode"] == "zero_pert"]
    zp_pass = all(abs(r["S"]) <= 1e-4 and abs(r["S_key"]) <= 1e-4 for r in zp)
    print(
        json.dumps(
            {
                "hash_equality": hash_eq,
                "zero_pert_pass": zp_pass,
                "S_A_K_jia": next(r["S"] for r in results if r["name"] == "quant_A_k_jia"),
                "S_A_K_yi": next(r["S"] for r in results if r["name"] == "quant_A_k_yi"),
                "S_key_A_K_jia": next(r["S_key"] for r in results if r["name"] == "quant_A_k_jia"),
                "S_key_A_K_yi": next(r["S_key"] for r in results if r["name"] == "quant_A_k_yi"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
