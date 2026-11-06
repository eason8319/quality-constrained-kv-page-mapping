#!/bin/bash
set -euo pipefail
cd /cluster/home/zengy/quality-constrained-kv-page-mapping
module load miniconda3/2024
source /cluster/apps/miniconda3/etc/profile.d/conda.sh
conda activate r1-kv-baseline
export HF_HUB_OFFLINE=1
export HF_HOME=/cluster/home/zengy/hf-cache
export TRANSFORMERS_OFFLINE=1
export HUGGINGFACE_HUB_CACHE=/cluster/home/zengy/hf-cache/hub
export TRANSFORMERS_CACHE=/cluster/home/zengy/hf-cache/hub
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
mkdir -p logs runs/rep_sample_Lp128_v1
TS=$(date +%Y%m%d_%H%M%S)
LOG=logs/rep_sample_Lp128_v1_${TS}.log
echo LOG=$LOG
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader || true
python -m experiments.page_paired_sensitivity.rep_sample \
  --n-prefix 32 --L-p 128 --g 128 --b 4 --dtype bfloat16 --device cuda:0 \
  --heads 10:3,2:1,22:2,22:4,24:6 --intervenes k,v,kv \
  --seed 20261105 \
  --out-dir runs/rep_sample_Lp128_v1 \
  2>&1 | tee "$LOG"
echo REP_DONE