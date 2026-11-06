# protect_multipage_nonretr_b2_v1 — equal-budget multi-page protect re-run

True joint quant of unprotected closed pages; protected pages stay FP.
Primary metric: residual mean S = 0.5*(S_jia+S_yi) vs chunk-matched FP.

- template: nonretr
- bits/group/L_p: 2/64/64
- heads: [(10, 3), (2, 1), (22, 2), (22, 4), (24, 6)]
- intervenes: ['kv']
- strategies: ['oracle_H', 'bucket', 'random', 'struct+content', 'none']
- budgets: [0.25, 0.5] (none runs once at budget=0)
- elapsed_s: 969.6
- n_metrics: 1440
- stop_expand: False
- gates: {"hash_equality_rate": 1.0, "hash_checks": 1440, "hash_fail": 0, "fp_both_correct_rate": 0.8125, "fp_checked": 32, "zero_pert_pass_rate": 1.0, "zero_pert_checks": 8, "zero_pert_fail": 0, "missing_score_pages": 0, "prefix_failures": [{"prefix_id": "nr000", "reason": "fp_not_both_correct"}, {"prefix_id": "nr003", "reason": "fp_not_both_correct"}, {"prefix_id": "nr010", "reason": "fp_not_both_correct"}, {"prefix_id": "nr014", "reason": "fp_not_both_correct"}, {"prefix_id": "nr015", "reason": "fp_not_both_correct"}, {"prefix_id": "nr023", "reason": "fp_not_both_correct"}]}

## Table (mean residual S)

```
{
  "bucket|budget=0.25|kv": {
    "n": 160,
    "S_mean_mean": -0.001256655966374574,
    "S_mean_median": 1.1781315456952753e-05,
    "S_mean_p95": 0.00807172883357794,
    "S_jia_mean": -0.0009296126934975029,
    "S_yi_mean": -0.0015836992392516441,
    "S_key_mean_mean": 0.0003964437275499222,
    "H_pos_mean_mean": 0.0017531107137637035,
    "mean_n_protect": 4.15625,
    "mean_n_quant": 11.65625
  },
  "bucket|budget=0.5|kv": {
    "n": 160,
    "S_mean_mean": -0.0014782195556360383,
    "S_mean_median": -9.294261502645895e-06,
    "S_mean_p95": 0.00826571747278271,
    "S_jia_mean": -0.001448980610372648,
    "S_yi_mean": -0.0015074585008994271,
    "S_key_mean_mean": 0.00024932296328188385,
    "H_pos_mean_mean": 0.001606205232509576,
    "mean_n_protect": 7.90625,
    "mean_n_quant": 7.90625
  },
  "none|budget=0.0|kv": {
    "n": 160,
    "S_mean_mean": -0.0020387209600636535,
    "S_mean_median": 2.0197834298111655e-05,
    "S_mean_p95": 0.008264250764292089,
    "S_jia_mean": -0.0017874032770945785,
    "S_yi_mean": -0.002290038643032731,
    "S_key_mean_mean": 0.0013554748440128606,
    "H_pos_mean_mean": 0.0018438383304622743,
    "mean_n_protect": 0.0,
    "mean_n_quant": 15.8125
  },
  "oracle_H|budget=0.25|kv": {
    "n": 160,
    "S_mean_mean": -0.003694105008879818,
    "S_mean_median": -7.882564252289134e-05,
    "S_mean_p95": 0.004278977859498432,
    "S_jia_mean": -0.0040549429600768945,
    "S_yi_mean": -0.0033332670576827423,
    "S_key_mean_mean": -0.0002552871696934744,
    "H_pos_mean_mean": 0.0007721917949246184,
    "mean_n_protect": 4.15625,
    "mean_n_quant": 11.65625
  },
  "oracle_H|budget=0.5|kv": {
    "n": 160,
    "S_mean_mean": -0.003449790641232891,
    "S_mean_median": -0.00011828542872081016,
    "S_mean_p95": 0.001922651161767148,
    "S_jia_mean": -0.0035673045232460774,
    "S_yi_mean": -0.0033322767592197044,
    "S_key_mean_mean": -0.0002330436317151907,
    "H_pos_mean_mean": 0.0005307376406625507,
    "mean_n_protect": 7.90625,
    "mean_n_quant": 7.90625
  },
  "random|budget=0.25|kv": {
    "n": 160,
    "S_mean_mean": -0.0005863699852570528,
    "S_mean_median": 3.3070858042568338e-06,
    "S_mean_p95": 0.009234773469746271,
    "S_jia_mean": -0.0005165602386814072,
    "S_yi_mean": -0.0006561797318326988,
    "S_key_mean_mean": 0.0006767489683170425,
    "H_pos_mean_mean": 0.0020170245657504904,
    "mean_n_protect": 4.15625,
    "mean_n_quant": 11.65625
  },
  "random|budget=0.5|kv": {
    "n": 160,
    "S_mean_mean": -0.0010392988175726807,
    "S_mean_median": -3.7803358736709924e-06,
    "S_mean_p95": 0.003912968085930224,
    "S_jia_mean": -0.0010107222744642937,
    "S_yi_mean": -0.0010678753606810675,
    "S_key_mean_mean": -6.168224708744675e-05,
    "H_pos_mean_mean": 0.0016075009876175225,
    "mean_n_protect": 7.90625,
    "mean_n_quant": 7.90625
  },
  "struct+content|budget=0.25|kv": {
    "n": 160,
    "S_mean_mean": -0.0014487733038389791,
    "S_mean_median": -3.4989207335911487e-06,
    "S_mean_p95": 0.008179170524105453,
    "S_jia_mean": -0.00201802100487702,
    "S_yi_mean": -0.000879525602800939,
    "S_key_mean_mean": 0.000729857187207017,
    "H_pos_mean_mean": 0.0018239938886164112,
    "mean_n_protect": 4.15625,
    "mean_n_quant": 11.65625
  },
  "struct+content|budget=0.5|kv": {
    "n": 160,
    "S_mean_mean": -0.0009331990827505029,
    "S_mean_median": -7.521196744164626e-06,
    "S_mean_p95": 0.00810309014293201,
    "S_jia_mean": -0.0015703070069373124,
    "S_yi_mean": -0.00029609115856369316,
    "S_key_mean_mean": 2.98398102131614e-05,
    "H_pos_mean_mean": 0.0017698768050255728,
    "mean_n_protect": 7.90625,
    "mean_n_quant": 7.90625
  }
}
```

## Recovery vs none

```
{
  "oracle_H|budget=0.25|kv": {
    "n_paired": 160,
    "recovery_mean": -12.988842581922258,
    "recovery_median": 0.945450823105774,
    "abs_S_reduction_mean": 0.0016553840488161644,
    "S_strategy_mean": -0.003694105008879818,
    "S_none_mean": -0.0020387209600636535
  },
  "oracle_H|budget=0.5|kv": {
    "n_paired": 160,
    "recovery_mean": -11.369367589751473,
    "recovery_median": 1.0253265127458233,
    "abs_S_reduction_mean": 0.0014110696811692363,
    "S_strategy_mean": -0.003449790641232891,
    "S_none_mean": -0.0020387209600636535
  },
  "bucket|budget=0.25|kv": {
    "n_paired": 160,
    "recovery_mean": 3.096607282902551,
    "recovery_median": 0.09104591016250418,
    "abs_S_reduction_mean": -0.0007820649936890808,
    "S_strategy_mean": -0.001256655966374574,
    "S_none_mean": -0.0020387209600636535
  },
  "bucket|budget=0.5|kv": {
    "n_paired": 160,
    "recovery_mean": -17.50318712713038,
    "recovery_median": 0.15443649838211948,
    "abs_S_reduction_mean": -0.0005605014044276169,
    "S_strategy_mean": -0.0014782195556360383,
    "S_none_mean": -0.0020387209600636535
  },
  "random|budget=0.25|kv": {
    "n_paired": 160,
    "recovery_mean": -23.245736530168436,
    "recovery_median": 0.4728783224051807,
    "abs_S_reduction_mean": -0.0014523509748066016,
    "S_strategy_mean": -0.0005863699852570528,
    "S_none_mean": -0.0020387209600636535
  },
  "random|budget=0.5|kv": {
    "n_paired": 160,
    "recovery_mean": -0.38298565029407095,
    "recovery_median": 0.8578153537335439,
    "abs_S_reduction_mean": -0.0009994221424909746,
    "S_strategy_mean": -0.0010392988175726807,
    "S_none_mean": -0.0020387209600636535
  },
  "struct+content|budget=0.25|kv": {
    "n_paired": 160,
    "recovery_mean": -22.75534774730292,
    "recovery_median": 0.07479163122683219,
    "abs_S_reduction_mean": -0.0005899476562246756,
    "S_strategy_mean": -0.0014487733038389791,
    "S_none_mean": -0.0020387209600636535
  },
  "struct+content|budget=0.5|kv": {
    "n_paired": 160,
    "recovery_mean": -20.543679142881665,
    "recovery_median": 0.2408266390212167,
    "abs_S_reduction_mean": -0.0011055218773131517,
    "S_strategy_mean": -0.0009331990827505029,
    "S_none_mean": -0.0020387209600636535
  }
}
```

## Comparisons

```
{
  "budget=0.25|kv": {
    "S_none": -0.0020387209600636535,
    "S_oracle_H": -0.003694105008879818,
    "S_bucket": -0.001256655966374574,
    "S_random": -0.0005863699852570528,
    "S_struct+content": -0.0014487733038389791,
    "oracle_beats_bucket": true,
    "bucket_beats_random": true,
    "struct_content_beats_random": true,
    "oracle_beats_random": true,
    "recovery_oracle": -12.988842581922258,
    "recovery_bucket": 3.096607282902551,
    "recovery_random": -23.245736530168436,
    "recovery_struct+content": -22.75534774730292
  },
  "budget=0.5|kv": {
    "S_none": -0.0020387209600636535,
    "S_oracle_H": -0.003449790641232891,
    "S_bucket": -0.0014782195556360383,
    "S_random": -0.0010392988175726807,
    "S_struct+content": -0.0009331990827505029,
    "oracle_beats_bucket": true,
    "bucket_beats_random": true,
    "struct_content_beats_random": false,
    "oracle_beats_random": true,
    "recovery_oracle": -11.369367589751473,
    "recovery_bucket": -17.50318712713038,
    "recovery_random": -0.38298565029407095,
    "recovery_struct+content": -20.543679142881665
  }
}
```

## Notes

- oracle_H uses measured H from prior single-page page_harms (upper bound).
- bucket / random / struct+content do not peek target H at selection time.
- Offline H-coverage ≠ residual S under multi-page joint quant (this arm).
