#!/usr/bin/env python3
"""Protect-strategy A/B on existing page_harms (CPU-only). Offline; no GPU."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

FRACS = (0.05, 0.1, 0.2, 0.25, 0.5)
MASSES = (0.5, 0.8, 0.95)
RANDOM_SEED = 20261105
N_RANDOM_DRAWS = 20

FIXED_BUCKET_SCORE = {
    "earlier_history": 3.0,
    "first_token_page": 2.0,
    "recent_closed": 1.0,
}

STRATEGIES = ("oracle_H", "random", "recency", "bucket", "bucket_loo", "lh_band")


def layer_band(l: int, n_layers: int = 32) -> str:
    if l < 11:
        return "early"
    if l < 22:
        return "mid"
    return "late"


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def k_for_frac(n: int, f: float) -> int:
    if n <= 0:
        return 0
    if f >= 1.0:
        return n
    if f <= 0:
        return 0
    return max(1, int(math.ceil(f * n)))


def coverage_from_order(pages: List[dict], order_idx: List[int], fracs=FRACS, masses=MASSES) -> dict:
    n = len(pages)
    Hs = [max(0.0, float(pages[i]["H"])) for i in range(n)]
    total = sum(Hs)
    ordered_H = [Hs[i] for i in order_idx]
    csum = []
    s = 0.0
    for h in ordered_H:
        s += h
        csum.append(s)

    def cov_at_k(k: int) -> float:
        if total <= 0:
            return 1.0
        if k <= 0:
            return 0.0
        k = min(n, k)
        return csum[k - 1] / total

    coverage_at_frac = {}
    for f in fracs:
        k = k_for_frac(n, f)
        coverage_at_frac[str(f)] = {
            "k_pages": k,
            "frac_pages": (k / n) if n else 0.0,
            "coverage": cov_at_k(k),
        }

    pages_for_mass = {}
    for mass in masses:
        if total <= 0 or n == 0:
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
        "coverage_at_frac": coverage_at_frac,
        "pages_for_mass": pages_for_mass,
    }


def mean_dicts(dicts: List[dict]) -> dict:
    if not dicts:
        return {}
    base = dicts[0]
    out = {
        "n_pages": base["n_pages"],
        "total_pos_H": base["total_pos_H"],
        "coverage_at_frac": {},
        "pages_for_mass": {},
        "n_draws": len(dicts),
    }
    for f, v0 in base["coverage_at_frac"].items():
        covs = [d["coverage_at_frac"][f]["coverage"] for d in dicts]
        out["coverage_at_frac"][f] = {
            "k_pages": v0["k_pages"],
            "frac_pages": v0["frac_pages"],
            "coverage": sum(covs) / len(covs),
        }
    for m, v0 in base["pages_for_mass"].items():
        fracs = [d["pages_for_mass"][m]["frac_pages"] for d in dicts]
        ks = [d["pages_for_mass"][m]["k_pages"] for d in dicts]
        out["pages_for_mass"][m] = {
            "k_pages": sum(ks) / len(ks),
            "frac_pages": sum(fracs) / len(fracs),
        }
    return out


def rank_oracle(pages: List[dict]) -> List[int]:
    return sorted(range(len(pages)), key=lambda i: (-max(0.0, float(pages[i]["H"])), pages[i]["page_index"]))


def rank_recency(pages: List[dict]) -> List[int]:
    return sorted(range(len(pages)), key=lambda i: (-pages[i]["page_index"],))


def rank_by_score(pages: List[dict], scores: List[float], older_first_within: bool = True) -> List[int]:
    if older_first_within:
        return sorted(range(len(pages)), key=lambda i: (-scores[i], pages[i]["page_index"]))
    return sorted(range(len(pages)), key=lambda i: (-scores[i], -pages[i]["page_index"]))


def rank_bucket_fixed(pages: List[dict]) -> List[int]:
    scores = [FIXED_BUCKET_SCORE.get(p["page_pos_bucket"], 0.0) for p in pages]
    return rank_by_score(pages, scores, older_first_within=True)


def build_loo_bucket_means(all_rows: List[dict], intervene: str) -> Dict[str, Dict[str, float]]:
    total_sum: Dict[str, float] = defaultdict(float)
    total_n: Dict[str, int] = defaultdict(int)
    per_prefix: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for r in all_rows:
        if r["intervene"] != intervene:
            continue
        b = r["page_pos_bucket"]
        h = max(0.0, float(r["H"]))
        total_sum[b] += h
        total_n[b] += 1
        per_prefix[r["prefix_id"]][b].append(h)

    out: Dict[str, Dict[str, float]] = {}
    for pid, bmap in per_prefix.items():
        scores = {}
        for b in total_sum.keys():
            p_sum = sum(bmap.get(b, []))
            p_n = len(bmap.get(b, []))
            rem_n = total_n[b] - p_n
            if rem_n <= 0:
                scores[b] = 0.0
            else:
                scores[b] = (total_sum[b] - p_sum) / rem_n
        out[pid] = scores
    return out


def build_loo_band_bucket_means(all_rows: List[dict], intervene: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    total_sum: Dict[Tuple[str, str], float] = defaultdict(float)
    total_n: Dict[Tuple[str, str], int] = defaultdict(int)
    per_prefix: Dict[str, Dict[Tuple[str, str], List[float]]] = defaultdict(lambda: defaultdict(list))

    for r in all_rows:
        if r["intervene"] != intervene:
            continue
        band = layer_band(int(r["l"]))
        b = r["page_pos_bucket"]
        h = max(0.0, float(r["H"]))
        key = (band, b)
        total_sum[key] += h
        total_n[key] += 1
        per_prefix[r["prefix_id"]][key].append(h)

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pid, keymap in per_prefix.items():
        bands: Dict[str, Dict[str, float]] = defaultdict(dict)
        for (band, b), tsum in total_sum.items():
            p_list = keymap.get((band, b), [])
            rem_n = total_n[(band, b)] - len(p_list)
            if rem_n <= 0:
                bands[band][b] = 0.0
            else:
                bands[band][b] = (tsum - sum(p_list)) / rem_n
        out[pid] = dict(bands)
    return out


def process_template(name: str, harms_path: Path, out_rows: List[dict]) -> dict:
    rows = load_jsonl(harms_path)
    intervenes = sorted({r["intervene"] for r in rows})
    prefixes = sorted({r["prefix_id"] for r in rows})
    heads = sorted({(r["l"], r["h"]) for r in rows})

    loo_bucket = {iv: build_loo_bucket_means(rows, iv) for iv in intervenes}
    loo_band = {iv: build_loo_band_bucket_means(rows, iv) for iv in intervenes}

    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in rows:
        key = (r["prefix_id"], int(r["l"]), int(r["h"]), r["intervene"])
        groups[key].append(r)

    n_groups = 0
    for key, items in sorted(groups.items()):
        items = sorted(items, key=lambda x: x["page_index"])
        pid, L, H, iv = key
        n_groups += 1
        band = layer_band(L)

        orders = {
            "oracle_H": rank_oracle(items),
            "recency": rank_recency(items),
            "bucket": rank_bucket_fixed(items),
        }

        bscore_map = loo_bucket[iv].get(pid, {})
        scores_loo = [bscore_map.get(p["page_pos_bucket"], 0.0) for p in items]
        orders["bucket_loo"] = rank_by_score(items, scores_loo, older_first_within=True)

        band_map = loo_band[iv].get(pid, {}).get(band, {})
        if not band_map:
            scores_lh = [FIXED_BUCKET_SCORE.get(p["page_pos_bucket"], 0.0) for p in items]
        else:
            scores_lh = [band_map.get(p["page_pos_bucket"], 0.0) for p in items]
        orders["lh_band"] = rank_by_score(items, scores_lh, older_first_within=True)

        rng = random.Random(RANDOM_SEED + (abs(hash((pid, L, H, iv))) % 10_000_000))
        rand_curves = []
        for _ in range(N_RANDOM_DRAWS):
            order = list(range(len(items)))
            rng.shuffle(order)
            rand_curves.append(coverage_from_order(items, order))
        random_curve = mean_dicts(rand_curves)

        for strat, order in orders.items():
            curve = coverage_from_order(items, order)
            out_rows.append(
                {
                    "template": name,
                    "prefix_id": pid,
                    "l": L,
                    "h": H,
                    "band": band,
                    "intervene": iv,
                    "strategy": strat,
                    "n_pages": curve["n_pages"],
                    "total_pos_H": curve["total_pos_H"],
                    "coverage_at_frac": curve["coverage_at_frac"],
                    "pages_for_mass": curve["pages_for_mass"],
                }
            )

        out_rows.append(
            {
                "template": name,
                "prefix_id": pid,
                "l": L,
                "h": H,
                "band": band,
                "intervene": iv,
                "strategy": "random",
                "n_pages": random_curve["n_pages"],
                "total_pos_H": random_curve["total_pos_H"],
                "coverage_at_frac": random_curve["coverage_at_frac"],
                "pages_for_mass": random_curve["pages_for_mass"],
                "n_random_draws": N_RANDOM_DRAWS,
                "random_seed_base": RANDOM_SEED,
            }
        )

    return {
        "template": name,
        "n_rows_in": len(rows),
        "n_prefixes": len(prefixes),
        "n_groups": n_groups,
        "n_heads": len(heads),
        "heads": heads,
        "intervenes": intervenes,
        "harms_path": str(harms_path),
    }


def pool_summary(coverage_rows: List[dict]) -> dict:
    buckets: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in coverage_rows:
        buckets[(r["template"], r["intervene"], r["strategy"])].append(r)

    pooled = {}
    for (tmpl, iv, strat), rs in sorted(buckets.items()):
        cov = defaultdict(list)
        mass = defaultdict(list)
        for r in rs:
            for f, v in r["coverage_at_frac"].items():
                cov[f].append(v["coverage"])
            for m, v in r["pages_for_mass"].items():
                mass[m].append(v["frac_pages"])
        key = f"{tmpl}|{iv}|{strat}"
        pooled[key] = {
            "template": tmpl,
            "intervene": iv,
            "strategy": strat,
            "n_curves": len(rs),
            "mean_coverage_at_frac": {f: sum(vs) / len(vs) for f, vs in cov.items()},
            "mean_frac_pages_for_mass": {m: sum(vs) / len(vs) for m, vs in mass.items()},
        }
    return pooled


def gap_vs_oracle(pooled: dict, frac: str = "0.25") -> dict:
    out = {}
    by_ti: Dict[Tuple[str, str], dict] = defaultdict(dict)
    for k, v in pooled.items():
        by_ti[(v["template"], v["intervene"])][v["strategy"]] = v
    for (tmpl, iv), smap in by_ti.items():
        oracle = smap.get("oracle_H", {}).get("mean_coverage_at_frac", {}).get(frac)
        if oracle is None:
            continue
        gaps = {}
        for strat, v in smap.items():
            c = v["mean_coverage_at_frac"].get(frac)
            if c is None:
                continue
            gaps[strat] = {"coverage": c, "gap_vs_oracle": oracle - c}
        out[f"{tmpl}|{iv}"] = {"frac": frac, "oracle_coverage": oracle, "strategies": gaps}
    return out


def write_readme(out_dir: Path, summary: dict):
    lines = [
        "# Protect strategy A/B v1 (CPU-only on existing page_harms)",
        "",
        "Compare deployable heuristics vs oracle top-H / random / recency at equal page budgets.",
        "Heuristics never use the target page's measured H (except `oracle_H`).",
        "",
        f"- random: mean of {N_RANDOM_DRAWS} seeded shuffles per group (base seed {RANDOM_SEED})",
        "- bucket: fixed order earlier_history > first_token_page > recent_closed; older first within bucket",
        "- bucket_loo: leave-one-prefix-out mean H of bucket (same template+intervene)",
        "- lh_band: LOO mean H stratified by layer band (early/mid/late thirds) x bucket for the curve's (l,h) band",
        "- recency: highest page_index first",
        "",
        f"- gates: {json.dumps(summary['gates'])}",
        "",
        "## Pooled mean coverage (all prefix x lh curves)",
        "",
    ]
    pooled = summary["pooled"]
    for tmpl in ("retrieval", "nonretr"):
        lines.append(f"### template={tmpl}")
        lines.append("")
        for iv in ("kv", "k", "v"):
            lines.append(f"#### intervene={iv}")
            lines.append("")
            lines.append("| strategy | @5% | @10% | @20% | @25% | @50% | frac->50% | frac->80% | frac->95% |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for strat in STRATEGIES:
                key = f"{tmpl}|{iv}|{strat}"
                v = pooled.get(key)
                if not v:
                    continue
                c = v["mean_coverage_at_frac"]
                m = v["mean_frac_pages_for_mass"]

                def fmt(x):
                    return f"{x:.4f}" if x is not None else "NA"

                lines.append(
                    "| {s} | {a} | {b} | {c20} | {c25} | {c50} | {m50} | {m80} | {m95} |".format(
                        s=strat,
                        a=fmt(c.get("0.05")),
                        b=fmt(c.get("0.1")),
                        c20=fmt(c.get("0.2")),
                        c25=fmt(c.get("0.25")),
                        c50=fmt(c.get("0.5")),
                        m50=fmt(m.get("0.5")),
                        m80=fmt(m.get("0.8")),
                        m95=fmt(m.get("0.95")),
                    )
                )
            lines.append("")
    lines.append("## Gap vs oracle @25% budget")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["gap_vs_oracle_at_0.25"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Design notes")
    lines.append("")
    lines.append("- Harm: H_p = max(0,S_jia)+max(0,S_yi) from rep_sample page_harms.")
    lines.append("- Coverage = sum H of selected pages / total positive H in the (prefix,l,h,iv) group.")
    lines.append("- No GPU; no re-quantization. Ranking quality on measured single-page H (proxy for protect-and-keep-FP).")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/cluster/home/zengy/quality-constrained-kv-page-mapping"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root
    out_dir = args.out or (root / "runs" / "protect_strategy_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    design = {
        "exp_id": "protect_strategy_v1",
        "cpu_only": True,
        "reuse_page_harms": [
            "runs/rep_sample_v1/page_harms.jsonl",
            "runs/rep_sample_nonretr_v1/page_harms.jsonl",
        ],
        "model": "Llama-3.1-8B-Instruct (from source runs)",
        "L_p": 64,
        "g": 64,
        "b": 4,
        "n_prefix_per_template": 32,
        "harm_def": "H_p = max(0,S_jia)+max(0,S_yi); coverage = selected pos-H / total pos-H",
        "budget_fracs": list(FRACS),
        "mass_targets": list(MASSES),
        "strategies": list(STRATEGIES),
        "bucket_fixed_order": ["earlier_history", "first_token_page", "recent_closed"],
        "layer_bands": {"early": "[0,11)", "mid": "[11,22)", "late": "[22,32)"},
        "random_n_draws": N_RANDOM_DRAWS,
        "random_seed": RANDOM_SEED,
        "no_peek": "heuristics do not use target-page measured H; oracle_H does",
        "frozen_from_source_runs": {
            "heads": [[10, 3], [2, 1], [22, 2], [22, 4], [24, 6]],
            "intervenes": ["k", "v", "kv"],
        },
    }
    (out_dir / "design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")

    coverage_rows: List[dict] = []
    meta = []
    meta.append(process_template("retrieval", root / "runs" / "rep_sample_v1" / "page_harms.jsonl", coverage_rows))
    meta.append(process_template("nonretr", root / "runs" / "rep_sample_nonretr_v1" / "page_harms.jsonl", coverage_rows))

    cov_path = out_dir / "coverage_by_strategy.jsonl"
    with cov_path.open("w", encoding="utf-8") as f:
        for r in coverage_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pooled = pool_summary(coverage_rows)
    gaps = gap_vs_oracle(pooled, "0.25")

    flags = {}
    for tmpl in ("retrieval", "nonretr"):
        smap = {s: pooled.get(f"{tmpl}|kv|{s}") for s in STRATEGIES}

        def cov25(s, _smap=smap):
            v = _smap.get(s)
            return None if not v else v["mean_coverage_at_frac"].get("0.25")

        rnd, rec, buck, lh = cov25("random"), cov25("recency"), cov25("bucket"), cov25("lh_band")
        flags[tmpl] = {
            "bucket_beats_random_at_25": (buck is not None and rnd is not None and buck > rnd),
            "bucket_beats_recency_at_25": (buck is not None and rec is not None and buck > rec),
            "lh_band_beats_random_at_25": (lh is not None and rnd is not None and lh > rnd),
            "lh_band_beats_recency_at_25": (lh is not None and rec is not None and lh > rec),
            "cov25": {
                "random": rnd,
                "recency": rec,
                "bucket": buck,
                "lh_band": lh,
                "oracle_H": cov25("oracle_H"),
                "bucket_loo": cov25("bucket_loo"),
            },
        }

    summary = {
        "design": design,
        "gates": {
            "templates": meta,
            "n_coverage_rows": len(coverage_rows),
            "expected_rows_per_group": len(STRATEGIES),
            "strategies": list(STRATEGIES),
        },
        "pooled": pooled,
        "gap_vs_oracle_at_0.25": gaps,
        "flags_kv_at_0.25": flags,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_readme(out_dir, summary)
    print(json.dumps({"out_dir": str(out_dir), "n_coverage_rows": len(coverage_rows), "flags_kv_at_0.25": flags}, indent=2))


if __name__ == "__main__":
    main()