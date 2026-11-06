#!/usr/bin/env bash
set -euo pipefail
cd /cluster/home/zengy/quality-constrained-kv-page-mapping
module load miniconda3/2024
source /cluster/apps/miniconda3/etc/profile.d/conda.sh
conda activate r1-kv-baseline
export HF_HOME=/cluster/home/zengy/hf-cache
export HUGGINGFACE_HUB_CACHE=/cluster/home/zengy/hf-cache/hub
export TRANSFORMERS_CACHE=/cluster/home/zengy/hf-cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
mkdir -p logs runs/protect_multipage_nonretr_v1
# refuse if retrieval out dir would be targeted
test -f runs/protect_multipage_v1/summary.json
LOG=logs/protect_multipage_nonretr_v1_$(date +%Y%m%d_%H%M%S).log
echo "LOG=$LOG"
# seed 20261106 matches rep_sample_nonretr_v1 prefix builder (NOT retrieval frozen 20261105)
timeout 86400 srun --jobid=21429 --overlap bash -lc '
set -euo pipefail
cd /cluster/home/zengy/quality-constrained-kv-page-mapping
module load miniconda3/2024
source /cluster/apps/miniconda3/etc/profile.d/conda.sh
conda activate r1-kv-baseline
export HF_HOME=/cluster/home/zengy/hf-cache HUGGINGFACE_HUB_CACHE=/cluster/home/zengy/hf-cache/hub
export TRANSFORMERS_CACHE=/cluster/home/zengy/hf-cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
python -m experiments.page_paired_sensitivity.protect_multipage \
  --template nonretr \
  --seed 20261106 \
  --intervene kv \
  --budgets 0.25,0.50 \
  --strategies oracle_H,bucket,random,struct+content,none \
  --n-prefix 32 --L-p 64 --g 64 --b 4 --dtype bfloat16 --device cuda:0 \
  --harms runs/rep_sample_nonretr_v1/page_harms.jsonl \
  --page-scores runs/protect_close_feat_v1/page_scores.jsonl \
  --out-dir runs/protect_multipage_nonretr_v1
' 2>&1 | tee "$LOG"
echo DONE_LOG=$LOG
