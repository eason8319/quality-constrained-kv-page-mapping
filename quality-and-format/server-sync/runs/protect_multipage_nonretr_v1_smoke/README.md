# protect_multipage_nonretr_v1_smoke — equal-budget multi-page protect re-run

True joint quant of unprotected closed pages; protected pages stay FP.
Primary metric: residual mean S = 0.5*(S_jia+S_yi) vs chunk-matched FP.

- template: nonretr
- bits/group/L_p: 4/64/64
- heads: [(10, 3)]
- intervenes: ['kv']
- strategies: ['oracle_H', 'bucket', 'random', 'struct+content', 'none']
- budgets: [0.25] (none runs once at budget=0)
- elapsed_s: 6.9
- n_metrics: 5
- stop_expand: False
- gates: {"hash_equality_rate": 1.0, "hash_checks": 5, "hash_fail": 0, "fp_both_correct_rate": 0.0, "fp_checked": 1, "zero_pert_pass_rate": 1.0, "zero_pert_checks": 2, "zero_pert_fail": 0, "missing_score_pages": 0, "prefix_failures": [{"prefix_id": "nr000", "reason": "fp_not_both_correct"}]}

## Table (mean residual S)

```
{
  "bucket|budget=0.25|kv": {
    "n": 1,
    "S_mean_mean": 0.025894844895810818,
    "S_mean_median": 0.025894844895810818,
    "S_mean_p95": 0.025894844895810818,
    "S_jia_mean": 0.015398339333478362,
    "S_yi_mean": 0.036391350458143276,
    "S_key_mean_mean": -0.00020516221411526203,
    "H_pos_mean_mean": 0.025894844895810818,
    "mean_n_protect": 3.0,
    "mean_n_quant": 9.0
  },
  "none|budget=0.0|kv": {
    "n": 1,
    "S_mean_mean": 0.018132362688629656,
    "S_mean_median": 0.018132362688629656,
    "S_mean_p95": 0.018132362688629656,
    "S_jia_mean": -0.00010535941109992563,
    "S_yi_mean": 0.036370084788359236,
    "S_key_mean_mean": 3.8692960515618324e-06,
    "H_pos_mean_mean": 0.018185042394179618,
    "mean_n_protect": 0.0,
    "mean_n_quant": 12.0
  },
  "oracle_H|budget=0.25|kv": {
    "n": 1,
    "S_mean_mean": 0.017006241488707018,
    "S_mean_median": 0.017006241488707018,
    "S_mean_p95": 0.017006241488707018,
    "S_jia_mean": -0.0023972512397449465,
    "S_yi_mean": 0.036409734217158984,
    "S_key_mean_mean": 3.0362862162292004e-06,
    "H_pos_mean_mean": 0.018204867108579492,
    "mean_n_protect": 3.0,
    "mean_n_quant": 9.0
  },
  "random|budget=0.25|kv": {
    "n": 1,
    "S_mean_mean": 0.009231759767862968,
    "S_mean_median": 0.009231759767862968,
    "S_mean_p95": 0.009231759767862968,
    "S_jia_mean": 2.477486850693822e-05,
    "S_yi_mean": 0.018438744667219,
    "S_key_mean_mean": -0.00021212827414274216,
    "H_pos_mean_mean": 0.009231759767862968,
    "mean_n_protect": 3.0,
    "mean_n_quant": 9.0
  },
  "struct+content|budget=0.25|kv": {
    "n": 1,
    "S_mean_mean": -0.0027435908647021265,
    "S_mean_median": -0.0027435908647021265,
    "S_mean_p95": -0.0027435908647021265,
    "S_jia_mean": -0.04194421689608134,
    "S_yi_mean": 0.03645703516667709,
    "S_key_mean_mean": 5.894224159419537e-06,
    "H_pos_mean_mean": 0.018228517583338544,
    "mean_n_protect": 3.0,
    "mean_n_quant": 9.0
  }
}
```

## Recovery vs none

```
{
  "oracle_H|budget=0.25|kv": {
    "n_paired": 1,
    "recovery_mean": 0.06210559645538083,
    "recovery_median": 0.06210559645538083,
    "abs_S_reduction_mean": 0.0011261211999226382,
    "S_strategy_mean": 0.017006241488707018,
    "S_none_mean": 0.018132362688629656
  },
  "bucket|budget=0.25|kv": {
    "n_paired": 1,
    "recovery_mean": -0.42810097837105454,
    "recovery_median": -0.42810097837105454,
    "abs_S_reduction_mean": -0.007762482207181162,
    "S_strategy_mean": 0.025894844895810818,
    "S_none_mean": 0.018132362688629656
  },
  "random|budget=0.25|kv": {
    "n_paired": 1,
    "recovery_mean": 0.49086834813579094,
    "recovery_median": 0.49086834813579094,
    "abs_S_reduction_mean": 0.008900602920766687,
    "S_strategy_mean": 0.009231759767862968,
    "S_none_mean": 0.018132362688629656
  },
  "struct+content|budget=0.25|kv": {
    "n_paired": 1,
    "recovery_mean": 1.1513090661054648,
    "recovery_median": 1.1513090661054648,
    "abs_S_reduction_mean": 0.020875953553331782,
    "S_strategy_mean": -0.0027435908647021265,
    "S_none_mean": 0.018132362688629656
  }
}
```

## Comparisons

```
{
  "budget=0.25|kv": {
    "S_none": 0.018132362688629656,
    "S_oracle_H": 0.017006241488707018,
    "S_bucket": 0.025894844895810818,
    "S_random": 0.009231759767862968,
    "S_struct+content": -0.0027435908647021265,
    "oracle_beats_bucket": true,
    "bucket_beats_random": false,
    "struct_content_beats_random": true,
    "oracle_beats_random": false,
    "recovery_oracle": 0.06210559645538083,
    "recovery_bucket": -0.42810097837105454,
    "recovery_random": 0.49086834813579094,
    "recovery_struct+content": 1.1513090661054648
  }
}
```

## Notes

- oracle_H uses measured H from prior single-page page_harms (upper bound).
- bucket / random / struct+content do not peek target H at selection time.
- Offline H-coverage ≠ residual S under multi-page joint quant (this arm).
