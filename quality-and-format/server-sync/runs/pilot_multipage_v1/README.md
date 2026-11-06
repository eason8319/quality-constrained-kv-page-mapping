# pilot_multipage_v1 — joint multi-page quant

- bits: 4
- variants: ['AB', 'Aadj']
- elapsed_s: 1167.9026639461517
- n_traj: 4672
- stop_expand: False
- gates: {"hash_equality_rate": 1.0, "hash_checks": 2304, "hash_fail": 0, "fp_both_correct_rate": 1.0, "fp_checked": 32, "zero_pert_pass_rate": 1.0, "zero_pert_checks": 64, "zero_pert_fail": 0, "prefix_failures": []}

## Metric definition

- **AB**: quantize pages A and B together. Report mean S_jia / S_yi. Classic paired D N/A.
- **Aadj**: quantize page A + one adjacent closed page. gap = S_jia − S_yi (need A vs need-not A).

## Headline

```
{
  "AB_k": {
    "n": 384,
    "S_jia_mean": -8.408066522877079e-05,
    "S_jia_median": 5.307473657012452e-07,
    "S_jia_p95": 6.926805442617479e-05,
    "S_yi_mean": 4.569895018492352e-05,
    "S_yi_median": 1.4998652630993092e-07,
    "S_yi_p95": 0.0006742278356644723,
    "gap_jia_minus_yi_mean": -0.00012977961541369435,
    "gap_jia_minus_yi_median": -6.548476865949244e-07,
    "gap_jia_minus_yi_p95": 0.0002830878772783769
  },
  "AB_v": {
    "n": 384,
    "S_jia_mean": -8.510214119567847e-05,
    "S_jia_median": 3.463112989265937e-07,
    "S_jia_p95": 8.404025626020938e-05,
    "S_yi_mean": 5.809098560613322e-05,
    "S_yi_median": -3.010325334192506e-07,
    "S_yi_p95": 0.0006727424860249208,
    "gap_jia_minus_yi_mean": -0.00014319312680181174,
    "gap_jia_minus_yi_median": 1.9817981969557272e-06,
    "gap_jia_minus_yi_p95": 0.0003144165547610102
  },
  "AB_kv": {
    "n": 384,
    "S_jia_mean": -8.901305153723449e-05,
    "S_jia_median": 4.916230295748392e-07,
    "S_jia_p95": 8.411734781077485e-05,
    "S_yi_mean": 5.994355841243908e-05,
    "S_yi_median": -2.817952614956691e-07,
    "S_yi_p95": 0.0006695190713344346,
    "gap_jia_minus_yi_mean": -0.0001489566099496736,
    "gap_jia_minus_yi_median": 7.099177641597028e-07,
    "gap_jia_minus_yi_p95": 0.00022878178049268175
  },
  "Aadj_k": {
    "n": 384,
    "S_jia_mean": -5.066644886909049e-05,
    "S_jia_median": 2.0832220570810023e-07,
    "S_jia_p95": 0.0001447170075152826,
    "S_yi_mean": 1.940424275589596e-05,
    "S_yi_median": -3.4743599523305113e-07,
    "S_yi_p95": 0.00042678453040707697,
    "gap_jia_minus_yi_mean": -7.007069162498646e-05,
    "gap_jia_minus_yi_median": 2.7263728270554574e-06,
    "gap_jia_minus_yi_p95": 0.0003513058864609775
  },
  "Aadj_v": {
    "n": 384,
    "S_jia_mean": -1.9177959662069415e-05,
    "S_jia_median": 1.3386943464865908e-07,
    "S_jia_p95": 0.0001077393295418006,
    "S_yi_mean": 2.8938648637360457e-05,
    "S_yi_median": -3.420445864321664e-07,
    "S_yi_p95": 0.0004207443766893,
    "gap_jia_minus_yi_mean": -4.811660829942983e-05,
    "gap_jia_minus_yi_median": 2.1242166402165212e-06,
    "gap_jia_minus_yi_p95": 0.00042958970509251586
  },
  "Aadj_kv": {
    "n": 384,
    "S_jia_mean": 1.7868507428402887e-05,
    "S_jia_median": 1.910337156611301e-07,
    "S_jia_p95": 0.00013876791320702474,
    "S_yi_mean": 4.288341221246764e-06,
    "S_yi_median": -5.880829121451825e-07,
    "S_yi_p95": 0.0003648362633248339,
    "gap_jia_minus_yi_mean": 1.3580166207156154e-05,
    "gap_jia_minus_yi_median": 1.4700884851966596e-06,
    "gap_jia_minus_yi_p95": 0.00040117855107382684
  }
}
```
