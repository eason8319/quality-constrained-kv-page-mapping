# page_paired_sensitivity

Minimal package for KV page paired-sensitivity smoke + 32-prefix pilot.

## Entrypoints

- `smoke_1x1.py` — 1×1 smoke
- `pilot_32.py` — formal 32-prefix trial (12 layer/head × 12 quant + FP)

## Re-run pilot

```bash
cd /cluster/home/zengy/quality-constrained-kv-page-mapping
# env: r1-kv-baseline, HF_HOME, CUDA_VISIBLE_DEVICES=1
srun --jobid=21429 --overlap bash -lc '
python -m experiments.page_paired_sensitivity.pilot_32 \
  --n-prefix 32 --L-p 64 --g 64 --b 4 --dtype bfloat16 --device cuda:0 \
  --out-dir runs/pilot_32
'
```

Resumes via `runs/pilot_32/ckpts/*.done`. Outputs: `trajectories.jsonl`, `D_by_prefix_lh.jsonl`, `summary.json`, `gates.jsonl`.

## Representative sampling (rep_sample_v1)

```bash
python -m experiments.page_paired_sensitivity.rep_sample \
  --n-prefix 32 --heads 10:3,2:1,22:2,22:4,24:6 --intervenes k,v,kv \
  --out-dir runs/rep_sample_v1
```

Outputs protection curves under `runs/rep_sample_v1/`. Resumes via `ckpts/*.done`.
