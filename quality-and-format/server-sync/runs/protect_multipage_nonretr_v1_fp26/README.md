# protect_multipage_nonretr_v1_fp26 — FP both-correct subset re-report

Parent: `protect_multipage_nonretr_v1` (b=4, template=nonretr).
CPU-only re-aggregation of parent `metrics.jsonl`; no GPU re-run.

- kept prefixes: **26** / 32
- excluded (fp_not_both_correct): `['nr000', 'nr003', 'nr010', 'nr014', 'nr015', 'nr023']`
- primary metric: mean residual S_mean (lower better); also H_pos_mean
- filter applied at aggregate time (metrics/residuals not copied)

## Comparisons (subset vs parent full-32)

| budget | metric | none | oracle_H | bucket | random | struct+content | bucket>random | oracle>bucket |
|---|---|---|---|---|---|---|---|---|
| 0.25 subset | S | 0.00040984 | 0.000220608 | 0.000271124 | 0.000305422 | 0.000462139 | True | True |
| 0.25 full32 | S | -0.000344365 | -0.000632756 | 0.00052322 | -0.000367706 | -0.000310257 | False | True |
| 0.25 subset | H_pos | 0.000564202 | 0.000488306 | 0.000411252 | 0.000429839 | 0.000580319 |  |  |
| 0.25 full32 | H_pos | 0.00129653 | 0.00101427 | 0.00159564 | 0.00114049 | 0.00123809 |  |  |
| 0.5 subset | S | 0.00040984 | -0.000177272 | 0.000529996 | 3.26392e-05 | 0.00035062 | False | True |
| 0.5 full32 | S | -0.000344365 | -0.00139644 | -1.13208e-06 | -0.000641222 | -0.000359459 | False | True |
| 0.5 subset | H_pos | 0.000564202 | 0.000198629 | 0.000691427 | 0.000370384 | 0.000507271 |  |  |
| 0.5 full32 | H_pos | 0.00129653 | 0.000453756 | 0.00123002 | 0.000826392 | 0.000934099 |  |  |

## Table (subset mean residual S)

```
{
  "bucket|budget=0.25|kv": {
    "n": 130,
    "S_mean_mean": 0.00027112440440812583,
    "S_mean_median": 4.101641176627407e-07,
    "S_mean_p95": 0.0031729730138465514,
    "S_jia_mean": 0.0004867495254926815,
    "S_yi_mean": 5.549928332357013e-05,
    "S_key_mean_mean": 0.0003860725376468438,
    "H_pos_mean_mean": 0.0004112521265121143,
    "mean_n_protect": 4.1923076923076925,
    "mean_n_quant": 11.73076923076923
  },
  "bucket|budget=0.5|kv": {
    "n": 130,
    "S_mean_mean": 0.0005299964025006322,
    "S_mean_median": 1.303280072306734e-06,
    "S_mean_p95": 0.004603812326589226,
    "S_jia_mean": 0.0010478389509133207,
    "S_yi_mean": 1.2153854087943885e-05,
    "S_key_mean_mean": 0.0008902200679026114,
    "H_pos_mean_mean": 0.0006914270082958241,
    "mean_n_protect": 7.961538461538462,
    "mean_n_quant": 7.961538461538462
  },
  "none|budget=0.0|kv": {
    "n": 130,
    "S_mean_mean": 0.0004098400331925562,
    "S_mean_median": 1.985984386389343e-05,
    "S_mean_p95": 0.004505719204000039,
    "S_jia_mean": 0.0007944350365806211,
    "S_yi_mean": 2.5245029804491187e-05,
    "S_key_mean_mean": 0.000772821052501408,
    "H_pos_mean_mean": 0.0005642015052386831,
    "mean_n_protect": 0.0,
    "mean_n_quant": 15.923076923076923
  },
  "oracle_H|budget=0.25|kv": {
    "n": 130,
    "S_mean_mean": 0.00022060817673529224,
    "S_mean_median": -1.2692175488382418e-06,
    "S_mean_p95": 0.0027001318010483424,
    "S_jia_mean": 0.00041412750422960715,
    "S_yi_mean": 2.7088849240977327e-05,
    "S_key_mean_mean": 3.2085979070801004e-05,
    "H_pos_mean_mean": 0.0004883063017284537,
    "mean_n_protect": 4.1923076923076925,
    "mean_n_quant": 11.73076923076923
  },
  "oracle_H|budget=0.5|kv": {
    "n": 130,
    "S_mean_mean": -0.00017727159870759853,
    "S_mean_median": -2.447131930694013e-05,
    "S_mean_p95": 0.0005616253249880284,
    "S_jia_mean": -0.00021946882636615764,
    "S_yi_mean": -0.0001350743710490394,
    "S_key_mean_mean": 2.0441061440998546e-05,
    "H_pos_mean_mean": 0.00019862866904921144,
    "mean_n_protect": 7.961538461538462,
    "mean_n_quant": 7.961538461538462
  },
  "random|budget=0.25|kv": {
    "n": 130,
    "S_mean_mean": 0.0003054222106208015,
    "S_mean_median": 2.301420674181524e-06,
    "S_mean_p95": 0.0011464002890609944,
    "S_jia_mean": 0.0006580703581406711,
    "S_yi_mean": -4.7225936899068095e-05,
    "S_key_mean_mean": 0.0006663404752124244,
    "H_pos_mean_mean": 0.00042983948422227643,
    "mean_n_protect": 4.1923076923076925,
    "mean_n_quant": 11.73076923076923
  },
  "random|budget=0.5|kv": {
    "n": 130,
    "S_mean_mean": 3.26391965996191e-05,
    "S_mean_median": -8.143591458065202e-07,
    "S_mean_p95": 0.0007936726882557343,
    "S_jia_mean": 5.754715662868502e-05,
    "S_yi_mean": 7.731236570553165e-06,
    "S_key_mean_mean": 0.00012321626339913705,
    "H_pos_mean_mean": 0.0003703839346969773,
    "mean_n_protect": 7.961538461538462,
    "mean_n_quant": 7.961538461538462
  },
  "struct+content|budget=0.25|kv": {
    "n": 130,
    "S_mean_mean": 0.00046213891012920763,
    "S_mean_median": 3.921989423361083e-06,
    "S_mean_p95": 0.004061278224361329,
    "S_jia_mean": 0.0009609552868269661,
    "S_yi_mean": -3.6677466568550865e-05,
    "S_key_mean_mean": 0.0008445967645876898,
    "H_pos_mean_mean": 0.0005803192557927053,
    "mean_n_protect": 4.1923076923076925,
    "mean_n_quant": 11.73076923076923
  },
  "struct+content|budget=0.5|kv": {
    "n": 130,
    "S_mean_mean": 0.000350619921834757,
    "S_mean_median": 7.43825340426459e-06,
    "S_mean_p95": 0.003108760604309654,
    "S_jia_mean": 0.0006571938085331415,
    "S_yi_mean": 4.404603513637248e-05,
    "S_key_mean_mean": 0.0004734855039994447,
    "H_pos_mean_mean": 0.0005072706518146584,
    "mean_n_protect": 7.961538461538462,
    "mean_n_quant": 7.961538461538462
  }
}
```

## Comparisons JSON

```
{
  "budget=0.25|kv": {
    "S_none": 0.0004098400331925562,
    "S_oracle_H": 0.00022060817673529224,
    "S_bucket": 0.00027112440440812583,
    "S_random": 0.0003054222106208015,
    "S_struct+content": 0.00046213891012920763,
    "oracle_beats_bucket": true,
    "bucket_beats_random": true,
    "struct_content_beats_random": false,
    "oracle_beats_random": true,
    "recovery_oracle": -28.77425479195018,
    "recovery_bucket": -2.223501103450473,
    "recovery_random": -5.379903234185543,
    "recovery_struct+content": -30.41981911037085
  },
  "budget=0.5|kv": {
    "S_none": 0.0004098400331925562,
    "S_oracle_H": -0.00017727159870759853,
    "S_bucket": 0.0005299964025006322,
    "S_random": 3.26391965996191e-05,
    "S_struct+content": 0.000350619921834757,
    "oracle_beats_bucket": true,
    "bucket_beats_random": false,
    "struct_content_beats_random": false,
    "oracle_beats_random": true,
    "recovery_oracle": -13.844525462254223,
    "recovery_bucket": -5.813822574108428,
    "recovery_random": 1.5208393183380504,
    "recovery_struct+content": -5.754863094899619
  }
}
```

## Notes

- Lower S = better under joint unprot quant.
- Head-to-head flags use mean S across kept prefixes × heads.
- Parent full-32 numbers copied under summary.parent_comparisons for side-by-side.
