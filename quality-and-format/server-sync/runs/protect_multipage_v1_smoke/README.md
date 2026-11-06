# protect_multipage_v1_smoke — equal-budget multi-page protect re-run

True joint quant of unprotected closed pages; protected pages stay FP.
Primary metric: residual mean S = 0.5*(S_jia+S_yi) vs chunk-matched FP.

- template: retrieval
- bits/group/L_p: 4/64/64
- heads: [(10, 3)]
- intervenes: ['kv']
- strategies: ['oracle_H', 'bucket', 'random', 'struct+content', 'none']
- budgets: [0.25] (none runs once at budget=0)
- elapsed_s: 4.1
- n_metrics: 5
- stop_expand: False
- gates: {"hash_equality_rate": 1.0, "hash_checks": 5, "hash_fail": 0, "fp_both_correct_rate": 1.0, "fp_checked": 1, "zero_pert_pass_rate": 1.0, "zero_pert_checks": 2, "zero_pert_fail": 0, "missing_score_pages": 0, "prefix_failures": []}

## Table (mean residual S)

```
{
  "bucket|budget=0.25|kv": {
    "n": 1,
    "S_mean_mean": 0.00014418081127587357,
    "S_mean_median": 0.00014418081127587357,
    "S_mean_p95": 0.00014418081127587357,
    "S_jia_mean": 0.00010695225500967354,
    "S_yi_mean": 0.0001814093675420736,
    "S_key_mean_mean": 7.358621223829687e-05,
    "H_pos_mean_mean": 0.00014418081127587357,
    "mean_n_protect": 1.0,
    "mean_n_quant": 2.0
  },
  "none|budget=0.0|kv": {
    "n": 1,
    "S_mean_mean": 0.00015042246081975463,
    "S_mean_median": 0.00015042246081975463,
    "S_mean_p95": 0.00015042246081975463,
    "S_jia_mean": 0.00010813643166329712,
    "S_yi_mean": 0.00019270848997621215,
    "S_key_mean_mean": 0.00010670871415641159,
    "H_pos_mean_mean": 0.00015042246081975463,
    "mean_n_protect": 0.0,
    "mean_n_quant": 3.0
  },
  "oracle_H|budget=0.25|kv": {
    "n": 1,
    "S_mean_mean": -1.5212926882668398e-06,
    "S_mean_median": -1.5212926882668398e-06,
    "S_mean_p95": -1.5212926882668398e-06,
    "S_jia_mean": 5.993933882564306e-08,
    "S_yi_mean": -3.1025247153593227e-06,
    "S_key_mean_mean": -8.401591912843287e-06,
    "H_pos_mean_mean": 2.996966941282153e-08,
    "mean_n_protect": 1.0,
    "mean_n_quant": 2.0
  },
  "random|budget=0.25|kv": {
    "n": 1,
    "S_mean_mean": 0.00014418081127587357,
    "S_mean_median": 0.00014418081127587357,
    "S_mean_p95": 0.00014418081127587357,
    "S_jia_mean": 0.00010695225500967354,
    "S_yi_mean": 0.0001814093675420736,
    "S_key_mean_mean": 7.358621223829687e-05,
    "H_pos_mean_mean": 0.00014418081127587357,
    "mean_n_protect": 1.0,
    "mean_n_quant": 2.0
  },
  "struct+content|budget=0.25|kv": {
    "n": 1,
    "S_mean_mean": 0.00014418081127587357,
    "S_mean_median": 0.00014418081127587357,
    "S_mean_p95": 0.00014418081127587357,
    "S_jia_mean": 0.00010695225500967354,
    "S_yi_mean": 0.0001814093675420736,
    "S_key_mean_mean": 7.358621223829687e-05,
    "H_pos_mean_mean": 0.00014418081127587357,
    "mean_n_protect": 1.0,
    "mean_n_quant": 2.0
  }
}
```

## Recovery vs none

```
{
  "oracle_H|budget=0.25|kv": {
    "n_paired": 1,
    "recovery_mean": 1.0101134676296097,
    "recovery_median": 1.0101134676296097,
    "abs_S_reduction_mean": 0.00015194375350802147,
    "S_strategy_mean": -1.5212926882668398e-06,
    "S_none_mean": 0.00015042246081975463
  },
  "bucket|budget=0.25|kv": {
    "n_paired": 1,
    "recovery_mean": 0.04149413265722457,
    "recovery_median": 0.04149413265722457,
    "abs_S_reduction_mean": 6.241649543881067e-06,
    "S_strategy_mean": 0.00014418081127587357,
    "S_none_mean": 0.00015042246081975463
  },
  "random|budget=0.25|kv": {
    "n_paired": 1,
    "recovery_mean": 0.04149413265722457,
    "recovery_median": 0.04149413265722457,
    "abs_S_reduction_mean": 6.241649543881067e-06,
    "S_strategy_mean": 0.00014418081127587357,
    "S_none_mean": 0.00015042246081975463
  },
  "struct+content|budget=0.25|kv": {
    "n_paired": 1,
    "recovery_mean": 0.04149413265722457,
    "recovery_median": 0.04149413265722457,
    "abs_S_reduction_mean": 6.241649543881067e-06,
    "S_strategy_mean": 0.00014418081127587357,
    "S_none_mean": 0.00015042246081975463
  }
}
```

## Comparisons

```
{
  "budget=0.25|kv": {
    "S_none": 0.00015042246081975463,
    "S_oracle_H": -1.5212926882668398e-06,
    "S_bucket": 0.00014418081127587357,
    "S_random": 0.00014418081127587357,
    "S_struct+content": 0.00014418081127587357,
    "oracle_beats_bucket": true,
    "bucket_beats_random": false,
    "struct_content_beats_random": false,
    "oracle_beats_random": true,
    "recovery_oracle": 1.0101134676296097,
    "recovery_bucket": 0.04149413265722457,
    "recovery_random": 0.04149413265722457,
    "recovery_struct+content": 0.04149413265722457
  }
}
```

## Notes

- oracle_H uses measured H from prior single-page page_harms (upper bound).
- bucket / random / struct+content do not peek target H at selection time.
- Offline H-coverage ≠ residual S under multi-page joint quant (this arm).
