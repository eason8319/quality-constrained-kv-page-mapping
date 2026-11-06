# quality-constrained-kv-page-mapping (server)

Fresh project. Do **not** restore `phd-research`. Leave alone: `attention-accelerator-phd`, `hf-cache`, `Soc`, `tmp`.

## Pilot 32 — COMPLETE (2026-11-05 Asia/Shanghai)

- Code: `experiments/page_paired_sensitivity/` (`pilot_32.py`, fork/quantize/prefixes)
- Out: `runs/pilot_32/` — `summary.json`, `trajectories.jsonl` (4736 rows), `D_by_prefix_lh.jsonl`, `gates.jsonl`, `ckpts/` (384)
- Log: `logs/pilot_32_20261105_194939.log`
- Smoke: `runs/smoke_1x1/`
- Docs: `docs/`

### Gates (all pass)
- hash equality 100% (2304/2304)
- FP both-Q correct 100% (32/32)
- zero-pert 100% (64/64, S=S_key=0 vs chunk-matched FP)
- manip 100% (32/32)

### Re-run / resume
See `experiments/page_paired_sensitivity/README.md`.
