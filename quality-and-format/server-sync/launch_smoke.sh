#!/bin/bash
set -euo pipefail
cd /cluster/home/zengy/quality-constrained-kv-page-mapping
tar xzf /tmp/kv-page-smoke.tar.gz --strip-components=1
cp /tmp/probe_cache_api.py /tmp/probe_cache_api.py
module load miniconda3/2024
source /cluster/apps/miniconda3/etc/profile.d/conda.sh
conda activate r1-kv-baseline
# quick API probe on GPU via srun (short)
timeout 300 srun --jobid=21429 --overlap bash -lc '
module load miniconda3/2024
source /cluster/apps/miniconda3/etc/profile.d/conda.sh
conda activate r1-kv-baseline
export HF_HOME=/cluster/home/zengy/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
python /tmp/probe_cache_api.py
' | tee /tmp/probe_cache_out.txt
echo PROBE_DONE
nohup bash /tmp/kv_smoke_run.sh > /tmp/kv_smoke_outer.log 2>&1 &
echo SMOKE_STARTED
sleep 2
head -5 /tmp/kv_smoke_outer.log
