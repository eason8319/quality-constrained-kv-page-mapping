# protect_multipage_nonretr_b2_v1_fp26 — FP both-correct subset re-report

Parent: `protect_multipage_nonretr_b2_v1` (b=2, template=nonretr).
CPU-only re-aggregation of parent `metrics.jsonl`; no GPU re-run.

- kept prefixes: **26** / 32
- excluded (fp_not_both_correct): `['nr000', 'nr003', 'nr010', 'nr014', 'nr015', 'nr023']`
- primary metric: mean residual S_mean (lower better); also H_pos_mean
- filter applied at aggregate time (metrics/residuals not copied)

## Comparisons (subset vs parent full-32)

| budget | metric | none | oracle_H | bucket | random | struct+content | bucket>random | oracle>bucket |
|---|---|---|---|---|---|---|---|---|
| 0.25 subset | S | 0.000224166 | -0.000948454 | -5.85694e-05 | -9.91587e-05 | 5.00998e-05 | False | True |
| 0.25 full32 | S | -0.00203872 | -0.00369411 | -0.00125666 | -0.00058637 | -0.00144877 | True | True |
| 0.25 subset | H_pos | 0.000938131 | 0.00026144 | 0.000612118 | 0.000457364 | 0.000686936 |  |  |
| 0.25 full32 | H_pos | 0.00184384 | 0.000772192 | 0.00175311 | 0.00201702 | 0.00182399 |  |  |
| 0.5 subset | S | 0.000224166 | -0.00079855 | 5.28271e-05 | -0.000401287 | -7.61699e-05 | False | True |
| 0.5 full32 | S | -0.00203872 | -0.00344979 | -0.00147822 | -0.0010393 | -0.000933199 | True | True |
| 0.5 subset | H_pos | 0.000938131 | 0.000277764 | 0.000593699 | 0.000237856 | 0.000628271 |  |  |
| 0.5 full32 | H_pos | 0.00184384 | 0.000530738 | 0.00160621 | 0.0016075 | 0.00176988 |  |  |

## Table (subset mean residual S)

```
{
  "bucket|budget=0.25|kv": {
    "n": 130,
    "S_mean_mean": -5.856936186390637e-05,
    "S_mean_median": 3.163827408722379e-05,
    "S_mean_p95": 0.004567389579483461,
    "S_jia_mean": -0.0002944451565598718,
    "S_yi_mean": 0.00017730643283205913,
    "S_key_mean_mean": 0.00046583817400880015,
    "H_pos_mean_mean": 0.0006121176997630055,
    "mean_n_protect": 4.1923076923076925,
    "mean_n_quant": 11.73076923076923
  },
  "bucket|budget=0.5|kv": {
    "n": 130,
    "S_mean_mean": 5.282710233075248e-05,
    "S_mean_median": -3.076367988796844e-06,
    "S_mean_p95": 0.004486529130264216,
    "S_jia_mean": -0.00013056287518947894,
    "S_yi_mean": 0.00023621707985098394,
    "S_key_mean_mean": 0.0002463627971547584,
    "H_pos_mean_mean": 0.0005936987665944716,
    "mean_n_protect": 7.961538461538462,
    "mean_n_quant": 7.961538461538462
  },
  "none|budget=0.0|kv": {
    "n": 130,
    "S_mean_mean": 0.00022416592562432214,
    "S_mean_median": 3.6009103458430517e-05,
    "S_mean_p95": 0.006869646383415782,
    "S_jia_mean": 1.8850463684663287e-06,
    "S_yi_mean": 0.00044644680488017794,
    "S_key_mean_mean": 0.0016243469352327968,
    "H_pos_mean_mean": 0.0009381312797398196,
    "mean_n_protect": 0.0,
    "mean_n_quant": 15.923076923076923
  },
  "oracle_H|budget=0.25|kv": {
    "n": 130,
    "S_mean_mean": -0.0009484535226042768,
    "S_mean_median": -2.37487110590943e-05,
    "S_mean_p95": 0.0006188437010659223,
    "S_jia_mean": -0.0017646513546654974,
    "S_yi_mean": -0.00013225569054305618,
    "S_key_mean_mean": -0.00034155952408372495,
    "H_pos_mean_mean": 0.0002614398460505713,
    "mean_n_protect": 4.1923076923076925,
    "mean_n_quant": 11.73076923076923
  },
  "oracle_H|budget=0.5|kv": {
    "n": 130,
    "S_mean_mean": -0.0007985496173145416,
    "S_mean_median": -6.770739392247075e-05,
    "S_mean_p95": 0.0006231475356041946,
    "S_jia_mean": -0.0013965925881263257,
    "S_yi_mean": -0.00020050664650275742,
    "S_key_mean_mean": -0.00017308533004744767,
    "H_pos_mean_mean": 0.0002777635944382312,
    "mean_n_protect": 7.961538461538462,
    "mean_n_quant": 7.961538461538462
  },
  "random|budget=0.25|kv": {
    "n": 130,
    "S_mean_mean": -9.915873365537792e-05,
    "S_mean_median": 2.729093023390305e-06,
    "S_mean_p95": 0.0016333077505294113,
    "S_jia_mean": -0.00014099506096438465,
    "S_yi_mean": -5.732240634637119e-05,
    "S_key_mean_mean": 0.0008060215920312867,
    "H_pos_mean_mean": 0.00045736411009551,
    "mean_n_protect": 4.1923076923076925,
    "mean_n_quant": 11.73076923076923
  },
  "random|budget=0.5|kv": {
    "n": 130,
    "S_mean_mean": -0.0004012866969999498,
    "S_mean_median": -1.395442647359839e-07,
    "S_mean_p95": 0.0011897511756450286,
    "S_jia_mean": -0.0008961587987292683,
    "S_yi_mean": 9.358540472936883e-05,
    "S_key_mean_mean": -8.06138377251605e-05,
    "H_pos_mean_mean": 0.00023785624213922883,
    "mean_n_protect": 7.961538461538462,
    "mean_n_quant": 7.961538461538462
  },
  "struct+content|budget=0.25|kv": {
    "n": 130,
    "S_mean_mean": 5.0099807250504475e-05,
    "S_mean_median": 7.737763705032801e-06,
    "S_mean_p95": 0.004506895772124153,
    "S_jia_mean": -0.00023052146997373804,
    "S_yi_mean": 0.000330721084474747,
    "S_key_mean_mean": 0.000870114800356812,
    "H_pos_mean_mean": 0.0006869355278493207,
    "mean_n_protect": 4.1923076923076925,
    "mean_n_quant": 11.73076923076923
  },
  "struct+content|budget=0.5|kv": {
    "n": 130,
    "S_mean_mean": -7.616989556308266e-05,
    "S_mean_median": -6.365888763563522e-06,
    "S_mean_p95": 0.004769348186251439,
    "S_jia_mean": -0.00022543113315674207,
    "S_yi_mean": 7.309134203057675e-05,
    "S_key_mean_mean": 3.3456884221567845e-05,
    "H_pos_mean_mean": 0.0006282714067367724,
    "mean_n_protect": 7.961538461538462,
    "mean_n_quant": 7.961538461538462
  }
}
```

## Comparisons JSON

```
{
  "budget=0.25|kv": {
    "S_none": 0.00022416592562432214,
    "S_oracle_H": -0.0009484535226042768,
    "S_bucket": -5.856936186390637e-05,
    "S_random": -9.915873365537792e-05,
    "S_struct+content": 5.0099807250504475e-05,
    "oracle_beats_bucket": true,
    "bucket_beats_random": false,
    "struct_content_beats_random": false,
    "oracle_beats_random": true,
    "recovery_oracle": -16.80394751179507,
    "recovery_bucket": 2.5301638396125323,
    "recovery_random": -27.742439761072653,
    "recovery_struct+content": -26.28212943437418
  },
  "budget=0.5|kv": {
    "S_none": 0.00022416592562432214,
    "S_oracle_H": -0.0007985496173145416,
    "S_bucket": 5.282710233075248e-05,
    "S_random": -0.0004012866969999498,
    "S_struct+content": -7.616989556308266e-05,
    "oracle_beats_bucket": true,
    "bucket_beats_random": false,
    "struct_content_beats_random": false,
    "oracle_beats_random": true,
    "recovery_oracle": -15.221411500258753,
    "recovery_bucket": -20.30203055618498,
    "recovery_random": -0.8506505391528568,
    "recovery_struct+content": -23.40129541842791
  }
}
```

## Notes

- Lower S = better under joint unprot quant.
- Head-to-head flags use mean S across kept prefixes × heads.
- Parent full-32 numbers copied under summary.parent_comparisons for side-by-side.
