#!/usr/bin/env python3
"""Re-aggregate protect_multipage metrics on FP both-correct prefixes only (CPU)."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def pct(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def mean_or_none(xs):
    return sum(xs) / len(xs) if xs else None


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def fp_fail_ids(summary: dict) -> List[str]:
    fails = []
    for item in summary.get("gates", {}).get("prefix_failures", []) or []:
        if item.get("reason") == "fp_not_both_correct" and item.get("prefix_id"):
            fails.append(item["prefix_id"])
    return sorted(set(fails))


def all_prefix_ids(rows: List[dict], prefixes_path: Path) -> List[str]:
    ids = sorted({r["prefix_id"] for r in rows})
    if prefixes_path.exists():
        pref = load_json(prefixes_path)
        if isinstance(pref, list):
            ids2 = [p.get("prefix_id") or p.get("id") for p in pref if isinstance(p, dict)]
            ids2 = [x for x in ids2 if x]
            if ids2:
                ids = ids2
        elif isinstance(pref, dict) and "prefixes" in pref:
            ids2 = [p.get("prefix_id") or p.get("id") for p in pref["prefixes"] if isinstance(p, dict)]
            ids2 = [x for x in ids2 if x]
            if ids2:
                ids = ids2
    return ids


def reaggregate(parent: Path, out: Path) -> dict:
    summary = load_json(parent / "summary.json")
    rows_all = load_jsonl(parent / "metrics.jsonl")
    if not rows_all:
        rows_all = load_jsonl(parent / "residuals.jsonl")

    fails = fp_fail_ids(summary)
    all_ids = all_prefix_ids(rows_all, parent / "prefixes.json")
    pass_ids = [pid for pid in all_ids if pid not in set(fails)]
    pass_set = set(pass_ids)

    rows = [r for r in rows_all if r["prefix_id"] in pass_set]

    strategies = summary.get("config", {}).get("strategies") or sorted({r["strategy"] for r in rows})
    budgets = summary.get("config", {}).get("budgets") or sorted(
        {float(r["budget"]) for r in rows if r["strategy"] != "none"}
    )
    intervenes = summary.get("config", {}).get("intervenes") or sorted({r["intervene"] for r in rows})

    pools: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in rows:
        pools[(r["strategy"], float(r["budget"]), r["intervene"])].append(r)

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

    comparisons = {}
    for bud in budgets:
        for iv in intervenes:
            def getS(name, bud=bud, iv=iv):
                return table.get(f"{name}|budget={bud}|{iv}", {}).get("S_mean_mean")

            def getR(name, bud=bud, iv=iv):
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

    parent_cmp = summary.get("comparisons", {})
    parent_table = summary.get("table_S_by_strategy_budget", {})

    design = {
        "arm": "protect_multipage_fp_both_correct_subset",
        "parent_run": str(parent),
        "filter": {
            "rule": "keep prefixes where FP both-correct gate passed; exclude gates.prefix_failures with reason=fp_not_both_correct",
            "source": "parent summary.json gates.prefix_failures (verified against parent fp_both_correct_rate)",
            "excluded_prefix_ids": fails,
            "kept_prefix_ids": pass_ids,
            "n_prefixes_parent": summary.get("config", {}).get("n_prefix", len(all_ids)),
            "n_prefixes": len(pass_ids),
            "n_excluded": len(fails),
        },
        "aggregation": "CPU re-pool of parent metrics.jsonl rows for kept prefixes only; same pooling as protect_multipage.py finalize",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_b": summary.get("config", {}).get("b"),
        "parent_template": summary.get("config", {}).get("template"),
        "strategies": strategies,
        "budgets": budgets,
        "intervenes": intervenes,
        "note": "metrics/residuals not copied; filter applied at aggregate time. Lower S = better.",
    }

    out_summary = {
        "config": {
            **{k: v for k, v in summary.get("config", {}).items()},
            "n_prefix": len(pass_ids),
            "n_prefix_parent": summary.get("config", {}).get("n_prefix"),
            "subset": "fp_both_correct",
            "parent_run": parent.name,
            "analysis": "reagg_protect_fp_subset",
        },
        "elapsed_s": None,
        "n_traj_written": None,
        "n_metrics": len(rows),
        "n_metrics_file": len(rows),
        "n_metrics_parent": len(rows_all),
        "stop_expand": False,
        "gates": {
            **summary.get("gates", {}),
            "fp_both_correct_rate": 1.0,
            "fp_checked": len(pass_ids),
            "fp_both_correct": len(pass_ids),
            "prefix_failures": [],
            "subset_note": "Re-report on FP-pass prefixes only; parent failures listed under design.filter.excluded_prefix_ids",
        },
        "table_S_by_strategy_budget": table,
        "recovery_vs_none": recovery,
        "comparisons": comparisons,
        "parent_comparisons": parent_cmp,
        "parent_table_S_by_strategy_budget": {
            k: {
                "S_mean_mean": v.get("S_mean_mean"),
                "H_pos_mean_mean": v.get("H_pos_mean_mean"),
                "n": v.get("n"),
            }
            for k, v in parent_table.items()
        },
        "n_prefixes": len(pass_ids),
        "excluded_prefix_ids": fails,
        "kept_prefix_ids": pass_ids,
        "gate_stats_raw": {
            "fp_both_correct": len(pass_ids),
            "fp_checked": len(pass_ids),
            "excluded_from_parent": fails,
        },
        "note_free_gen": summary.get("note_free_gen"),
        "analysis_note": "CPU-only re-aggregation; no GPU re-run. Filter applied at aggregate time (no filtered metrics.jsonl copy).",
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "design.json").write_text(json.dumps(design, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(out_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x):
        return "n/a" if x is None else f"{x:.6g}"

    lines = [
        f"# {out.name} — FP both-correct subset re-report",
        "",
        f"Parent: `{parent.name}` (b={summary.get('config', {}).get('b')}, template={summary.get('config', {}).get('template')}).",
        "CPU-only re-aggregation of parent `metrics.jsonl`; no GPU re-run.",
        "",
        f"- kept prefixes: **{len(pass_ids)}** / {summary.get('config', {}).get('n_prefix', len(all_ids))}",
        f"- excluded (fp_not_both_correct): `{fails}`",
        "- primary metric: mean residual S_mean (lower better); also H_pos_mean",
        "- filter applied at aggregate time (metrics/residuals not copied)",
        "",
        "## Comparisons (subset vs parent full-32)",
        "",
        "| budget | metric | none | oracle_H | bucket | random | struct+content | bucket>random | oracle>bucket |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for bud in budgets:
        for iv in intervenes:
            key = f"budget={bud}|{iv}"
            c = comparisons[key]
            pc = parent_cmp.get(key, {})
            lines.append(
                "| {bud} subset | S | {none} | {o} | {b} | {r} | {sc} | {br} | {ob} |".format(
                    bud=bud,
                    none=fmt(c["S_none"]),
                    o=fmt(c["S_oracle_H"]),
                    b=fmt(c["S_bucket"]),
                    r=fmt(c["S_random"]),
                    sc=fmt(c["S_struct+content"]),
                    br=c["bucket_beats_random"],
                    ob=c["oracle_beats_bucket"],
                )
            )
            lines.append(
                "| {bud} full32 | S | {none} | {o} | {b} | {r} | {sc} | {br} | {ob} |".format(
                    bud=bud,
                    none=fmt(pc.get("S_none")),
                    o=fmt(pc.get("S_oracle_H")),
                    b=fmt(pc.get("S_bucket")),
                    r=fmt(pc.get("S_random")),
                    sc=fmt(pc.get("S_struct+content")),
                    br=pc.get("bucket_beats_random"),
                    ob=pc.get("oracle_beats_bucket"),
                )
            )

            def H(name, bud=bud, iv=iv, tbl=table):
                keyn = f"none|budget=0.0|{iv}" if name == "none" else f"{name}|budget={bud}|{iv}"
                return tbl.get(keyn, {}).get("H_pos_mean_mean")

            def Hp(name, bud=bud, iv=iv, tbl=parent_table):
                keyn = f"none|budget=0.0|{iv}" if name == "none" else f"{name}|budget={bud}|{iv}"
                return tbl.get(keyn, {}).get("H_pos_mean_mean")

            lines.append(
                "| {bud} subset | H_pos | {none} | {o} | {b} | {r} | {sc} |  |  |".format(
                    bud=bud,
                    none=fmt(H("none")),
                    o=fmt(H("oracle_H")),
                    b=fmt(H("bucket")),
                    r=fmt(H("random")),
                    sc=fmt(H("struct+content")),
                )
            )
            lines.append(
                "| {bud} full32 | H_pos | {none} | {o} | {b} | {r} | {sc} |  |  |".format(
                    bud=bud,
                    none=fmt(Hp("none")),
                    o=fmt(Hp("oracle_H")),
                    b=fmt(Hp("bucket")),
                    r=fmt(Hp("random")),
                    sc=fmt(Hp("struct+content")),
                )
            )

    lines += [
        "",
        "## Table (subset mean residual S)",
        "",
        "```",
        json.dumps(table, indent=2),
        "```",
        "",
        "## Comparisons JSON",
        "",
        "```",
        json.dumps(comparisons, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- Lower S = better under joint unprot quant.",
        "- Head-to-head flags use mean S across kept prefixes × heads.",
        "- Parent full-32 numbers copied under summary.parent_comparisons for side-by-side.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "kept_prefixes.json").write_text(
        json.dumps({"kept": pass_ids, "excluded": fails}, indent=2) + "\n", encoding="utf-8"
    )
    return out_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    summary = reaggregate(args.parent.resolve(), args.out.resolve())
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_prefixes": summary["n_prefixes"],
                "excluded": summary["excluded_prefix_ids"],
                "comparisons": summary["comparisons"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
