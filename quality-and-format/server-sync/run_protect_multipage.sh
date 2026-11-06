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
python -c "import sys; print(sys.version); import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
MODE=${1:-smoke}
if [[ "$MODE" == "smoke" ]]; then
  OUT=runs/protect_multipage_v1_smoke
  rm -rf "$OUT"
  python -m experiments.page_paired_sensitivity.protect_multipage --smoke --out-dir "$OUT"
else
  OUT=runs/protect_multipage_v1
  mkdir -p "$OUT"
  python -m experiments.page_paired_sensitivity.protect_multipage \
    --template retrieval \
    --intervene kv \
    --budgets 0.25,0.50 \
    --strategies oracle_H,bucket,random,struct+content,none \
    --out-dir "$OUT"
fi
echo DONE_MODE=$MODE
