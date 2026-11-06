# pilot_multipage_v1 — joint multi-page quant

- bits: 2
- variants: ['AB', 'Aadj']
- elapsed_s: 1168.8413813114166
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
    "S_jia_mean": -6.060464268295738e-05,
    "S_jia_median": 1.1896281648660079e-07,
    "S_jia_p95": 0.00012332166157496768,
    "S_yi_mean": 7.054749696258079e-05,
    "S_yi_median": 1.465556177511947e-06,
    "S_yi_p95": 0.000674976217131492,
    "gap_jia_minus_yi_mean": -0.00013115213964553824,
    "gap_jia_minus_yi_median": -5.60592703398773e-06,
    "gap_jia_minus_yi_p95": 0.00028549121739918884
  },
  "AB_v": {
    "n": 384,
    "S_jia_mean": 5.653272259240616e-05,
    "S_jia_median": 6.960794962651562e-08,
    "S_jia_p95": 0.0001262184820006952,
    "S_yi_mean": 5.0617721515439714e-05,
    "S_yi_median": 4.461525549004364e-08,
    "S_yi_p95": 0.0006750697015453019,
    "gap_jia_minus_yi_mean": 5.9150010769664606e-06,
    "gap_jia_minus_yi_median": -3.6846801701055178e-06,
    "gap_jia_minus_yi_p95": 0.00033818552142861357
  },
  "AB_kv": {
    "n": 384,
    "S_jia_mean": 7.768761827752337e-05,
    "S_jia_median": 5.143419912201352e-07,
    "S_jia_p95": 0.000148707052852615,
    "S_yi_mean": 6.639086616166226e-05,
    "S_yi_median": 4.1649923332442995e-07,
    "S_yi_p95": 0.0007997844690786829,
    "gap_jia_minus_yi_mean": 1.1296752115861068e-05,
    "gap_jia_minus_yi_median": -2.6024342882162015e-06,
    "gap_jia_minus_yi_p95": 0.0003331973119657051
  },
  "Aadj_k": {
    "n": 384,
    "S_jia_mean": 0.00010667975793327381,
    "S_jia_median": -3.422933938660814e-07,
    "S_jia_p95": 0.0002011258457343958,
    "S_yi_mean": -1.982688904993915e-06,
    "S_yi_median": -1.3296165093379386e-06,
    "S_yi_p95": 0.0005794606908693843,
    "gap_jia_minus_yi_mean": 0.00010866244683826772,
    "gap_jia_minus_yi_median": 8.0030688422994e-07,
    "gap_jia_minus_yi_p95": 0.0005558039995889892
  },
  "Aadj_v": {
    "n": 384,
    "S_jia_mean": 0.00017985566356404606,
    "S_jia_median": 7.770654519845266e-08,
    "S_jia_p95": 0.00018371398491581206,
    "S_yi_mean": 3.032668752829687e-06,
    "S_yi_median": -1.8002971652701185e-06,
    "S_yi_p95": 0.0004085162502065035,
    "gap_jia_minus_yi_mean": 0.00017682299481121635,
    "gap_jia_minus_yi_median": 1.139277810580097e-06,
    "gap_jia_minus_yi_p95": 0.0004951078085559861
  },
  "Aadj_kv": {
    "n": 384,
    "S_jia_mean": 0.00010458732251032504,
    "S_jia_median": -1.3998736676512635e-06,
    "S_jia_p95": 0.00022859636394514384,
    "S_yi_mean": -2.8575111659767716e-06,
    "S_yi_median": -6.0570405821636085e-06,
    "S_yi_p95": 0.0006611038850314803,
    "gap_jia_minus_yi_mean": 0.0001074448336763018,
    "gap_jia_minus_yi_median": 7.429649940604575e-07,
    "gap_jia_minus_yi_p95": 0.0006340143921827733
  }
}
```
