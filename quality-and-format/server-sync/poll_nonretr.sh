#!/usr/bin/env bash
set -euo pipefail
cd /cluster/home/zengy/quality-constrained-kv-page-mapping
log=$(ls -t logs/protect_multipage_nonretr_v1_20261106_*.log | head -1)
echo "LOG=$log"
for i in $(seq 1 60); do
  if grep -q '\[pmp\] wrote' "$log" 2>/dev/null; then
    echo DONE_HIT
    tail -20 "$log"
    break
  fi
  if grep -q 'CUDA out of memory\|Traceback (most recent call last)' "$log" 2>/dev/null; then
    echo ERR_HIT
    tail -50 "$log"
    break
  fi
  prog=""
  if [[ -f runs/protect_multipage_nonretr_v1/progress.json ]]; then
    prog=$(python3 -c 'import json;d=json.load(open("runs/protect_multipage_nonretr_v1/progress.json"));print(d.get("prefix_id"), "ckpts",d.get("done_ckpts"), "elapsed", round(d.get("elapsed_s",0),1), "fp", d.get("gate_stats",{}).get("fp_both_correct"), "/", d.get("gate_stats",{}).get("fp_checked"))')
  fi
  echo "wait_$i $(date +%H:%M:%S) $prog"
  sleep 30
done
echo ===SUMMARY===
ls -la runs/protect_multipage_nonretr_v1/
test -f runs/protect_multipage_nonretr_v1/summary.json && echo HAS_SUMMARY
echo RETR_MD5=$(md5sum runs/protect_multipage_v1/summary.json)
