# protect_multipage_v1 — equal-budget multi-page protect re-run

True joint quant of unprotected closed pages; protected pages stay FP.
Primary metric: residual mean S = 0.5*(S_jia+S_yi) vs chunk-matched FP.

- template: retrieval
- bits/group/L_p: 4/64/64
- heads: [(10, 3), (2, 1), (22, 2), (22, 4), (24, 6)]
- intervenes: ['kv']
- strategies: ['oracle_H', 'bucket', 'random', 'struct+content', 'none']
- budgets: [0.25, 0.5] (none runs once at budget=0)
- elapsed_s: 730.2
- n_metrics: 1440
- stop_expand: False
- gates: {"hash_equality_rate": 1.0, "hash_checks": 1440, "hash_fail": 0, "fp_both_correct_rate": 1.0, "fp_checked": 32, "zero_pert_pass_rate": 1.0, "zero_pert_checks": 8, "zero_pert_fail": 0, "missing_score_pages": 0, "prefix_failures": []}

## Table (mean residual S)

```
{
  "bucket|budget=0.25|kv": {
    "n": 160,
    "S_mean_mean": -4.6146202525374504e-05,
    "S_mean_median": -2.8152367917755327e-06,
    "S_mean_p95": 0.00031146458849775656,
    "S_jia_mean": -0.00012636494221048624,
    "S_yi_mean": 3.407253715973725e-05,
    "S_key_mean_mean": -0.0002922439738654248,
    "H_pos_mean_mean": 5.973506100061421e-05,
    "mean_n_protect": 1.75,
    "mean_n_quant": 3.875
  },
  "bucket|budget=0.5|kv": {
    "n": 160,
    "S_mean_mean": -4.090167757918965e-05,
    "S_mean_median": -8.490629359888165e-07,
    "S_mean_p95": 0.00013584326863338002,
    "S_jia_mean": -7.698662756541334e-05,
    "S_yi_mean": -4.816727592965959e-06,
    "S_key_mean_mean": -0.00020120970701782427,
    "H_pos_mean_mean": 3.275998262529889e-05,
    "mean_n_protect": 3.0,
    "mean_n_quant": 2.625
  },
  "none|budget=0.0|kv": {
    "n": 160,
    "S_mean_mean": -0.00012562091843814884,
    "S_mean_median": -2.6745591715856187e-06,
    "S_mean_p95": 0.0001338492490475574,
    "S_jia_mean": -0.00022292840067504946,
    "S_yi_mean": -2.831343620124826e-05,
    "S_key_mean_mean": -0.0005181335445627155,
    "H_pos_mean_mean": 2.704787151078135e-05,
    "mean_n_protect": 0.0,
    "mean_n_quant": 5.625
  },
  "oracle_H|budget=0.25|kv": {
    "n": 160,
    "S_mean_mean": -7.895813393714886e-05,
    "S_mean_median": -1.9851998141954635e-06,
    "S_mean_p95": 0.0002123545537835983,
    "S_jia_mean": -0.00014861658411663352,
    "S_yi_mean": -9.299683757664219e-06,
    "S_key_mean_mean": -0.00033749104821936273,
    "H_pos_mean_mean": 3.607218827320698e-05,
    "mean_n_protect": 1.75,
    "mean_n_quant": 3.875
  },
  "oracle_H|budget=0.5|kv": {
    "n": 160,
    "S_mean_mean": -5.2189937378003626e-05,
    "S_mean_median": -4.221779715862795e-06,
    "S_mean_p95": 5.2588545297567204e-05,
    "S_jia_mean": -8.908435043916067e-05,
    "S_yi_mean": -1.5295524316846598e-05,
    "S_key_mean_mean": -0.00019508722815544388,
    "H_pos_mean_mean": 2.8260845553420562e-05,
    "mean_n_protect": 3.0,
    "mean_n_quant": 2.625
  },
  "random|budget=0.25|kv": {
    "n": 160,
    "S_mean_mean": 9.404705663668507e-06,
    "S_mean_median": -1.6572423646721294e-07,
    "S_mean_p95": 0.00023866435722879966,
    "S_jia_mean": 2.2765246045016396e-06,
    "S_yi_mean": 1.653288672283537e-05,
    "S_key_mean_mean": 2.0607501460290224e-05,
    "H_pos_mean_mean": 7.713591276253003e-05,
    "mean_n_protect": 1.75,
    "mean_n_quant": 3.875
  },
  "random|budget=0.5|kv": {
    "n": 160,
    "S_mean_mean": -4.3495182810806555e-05,
    "S_mean_median": -1.6935274326594178e-06,
    "S_mean_p95": 0.00013474836345892538,
    "S_jia_mean": -6.852502632456255e-05,
    "S_yi_mean": -1.8465339297050566e-05,
    "S_key_mean_mean": -0.00019631535501503096,
    "H_pos_mean_mean": 3.4207273191289773e-05,
    "mean_n_protect": 3.0,
    "mean_n_quant": 2.625
  },
  "struct+content|budget=0.25|kv": {
    "n": 160,
    "S_mean_mean": -9.486578155861726e-06,
    "S_mean_median": -1.9984324467259284e-06,
    "S_mean_p95": 0.00031146458849775656,
    "S_jia_mean": -4.9925484564757515e-05,
    "S_yi_mean": 3.095232825303403e-05,
    "S_key_mean_mean": -0.0001383114021223264,
    "H_pos_mean_mean": 5.8036024806199885e-05,
    "mean_n_protect": 1.75,
    "mean_n_quant": 3.875
  },
  "struct+content|budget=0.5|kv": {
    "n": 160,
    "S_mean_mean": -4.62105389085791e-05,
    "S_mean_median": -1.8367497887084028e-06,
    "S_mean_p95": 0.00013476841564852293,
    "S_jia_mean": -8.092039771663432e-05,
    "S_yi_mean": -1.1500680100523905e-05,
    "S_key_mean_mean": -0.00021309465926151462,
    "H_pos_mean_mean": 3.053285752879552e-05,
    "mean_n_protect": 3.0,
    "mean_n_quant": 2.625
  }
}
```

## Recovery vs none

```
{
  "oracle_H|budget=0.25|kv": {
    "n_paired": 160,
    "recovery_mean": -28.622216160701377,
    "recovery_median": 0.8393999838740528,
    "abs_S_reduction_mean": -4.666278450099999e-05,
    "S_strategy_mean": -7.895813393714886e-05,
    "S_none_mean": -0.00012562091843814884
  },
  "oracle_H|budget=0.5|kv": {
    "n_paired": 160,
    "recovery_mean": 9.266794045327728,
    "recovery_median": 1.1500368467772386,
    "abs_S_reduction_mean": -7.343098106014522e-05,
    "S_strategy_mean": -5.2189937378003626e-05,
    "S_none_mean": -0.00012562091843814884
  },
  "bucket|budget=0.25|kv": {
    "n_paired": 160,
    "recovery_mean": -26.578833461005377,
    "recovery_median": 0.30151081060625173,
    "abs_S_reduction_mean": -7.947471591277435e-05,
    "S_strategy_mean": -4.6146202525374504e-05,
    "S_none_mean": -0.00012562091843814884
  },
  "bucket|budget=0.5|kv": {
    "n_paired": 160,
    "recovery_mean": 42.63732592120655,
    "recovery_median": 0.4361418957813292,
    "abs_S_reduction_mean": -8.471924085895921e-05,
    "S_strategy_mean": -4.090167757918965e-05,
    "S_none_mean": -0.00012562091843814884
  },
  "random|budget=0.25|kv": {
    "n_paired": 160,
    "recovery_mean": 8.676241630453672,
    "recovery_median": 0.4595846428014453,
    "abs_S_reduction_mean": -0.0001350256241018174,
    "S_strategy_mean": 9.404705663668507e-06,
    "S_none_mean": -0.00012562091843814884
  },
  "random|budget=0.5|kv": {
    "n_paired": 160,
    "recovery_mean": 10.690461717066555,
    "recovery_median": 1.037365751701481,
    "abs_S_reduction_mean": -8.21257356273423e-05,
    "S_strategy_mean": -4.3495182810806555e-05,
    "S_none_mean": -0.00012562091843814884
  },
  "struct+content|budget=0.25|kv": {
    "n_paired": 160,
    "recovery_mean": -26.218506570030694,
    "recovery_median": 0.3132662852771281,
    "abs_S_reduction_mean": -0.00011613434028228713,
    "S_strategy_mean": -9.486578155861726e-06,
    "S_none_mean": -0.00012562091843814884
  },
  "struct+content|budget=0.5|kv": {
    "n_paired": 160,
    "recovery_mean": 43.04566090611998,
    "recovery_median": 0.5910099296540215,
    "abs_S_reduction_mean": -7.941037952956976e-05,
    "S_strategy_mean": -4.62105389085791e-05,
    "S_none_mean": -0.00012562091843814884
  }
}
```

## Comparisons

```
{
  "budget=0.25|kv": {
    "S_none": -0.00012562091843814884,
    "S_oracle_H": -7.895813393714886e-05,
    "S_bucket": -4.6146202525374504e-05,
    "S_random": 9.404705663668507e-06,
    "S_struct+content": -9.486578155861726e-06,
    "oracle_beats_bucket": true,
    "bucket_beats_random": true,
    "struct_content_beats_random": true,
    "oracle_beats_random": true,
    "recovery_oracle": -28.622216160701377,
    "recovery_bucket": -26.578833461005377,
    "recovery_random": 8.676241630453672,
    "recovery_struct+content": -26.218506570030694
  },
  "budget=0.5|kv": {
    "S_none": -0.00012562091843814884,
    "S_oracle_H": -5.2189937378003626e-05,
    "S_bucket": -4.090167757918965e-05,
    "S_random": -4.3495182810806555e-05,
    "S_struct+content": -4.62105389085791e-05,
    "oracle_beats_bucket": true,
    "bucket_beats_random": false,
    "struct_content_beats_random": true,
    "oracle_beats_random": true,
    "recovery_oracle": 9.266794045327728,
    "recovery_bucket": 42.63732592120655,
    "recovery_random": 10.690461717066555,
    "recovery_struct+content": 43.04566090611998
  }
}
```

## Notes

- oracle_H uses measured H from prior single-page page_harms (upper bound).
- bucket / random / struct+content do not peek target H at selection time.
- Offline H-coverage ≠ residual S under multi-page joint quant (this arm).

## Augmented: H_pos and matched pairwise

Mean S under `none` is slightly negative on this retrieval set, so `1 - S/S_none` recovery is noisy.
Below: positive residual harm and matched pairwise (lower S / lower H is better).

```
{
  "budget=0.25|kv": {
    "oracle_vs_bucket": {
      "n": 160,
      "mean_dS_a_minus_b": -3.2811931411774376e-05,
      "mean_dH_a_minus_b": -2.3662872727407226e-05,
      "frac_a_lower_S": 0.5375,
      "frac_a_lower_H": 0.40625,
      "a_beats_b_on_mean_S": true,
      "a_beats_b_on_mean_H": true
    },
    "bucket_vs_random": {
      "n": 160,
      "mean_dS_a_minus_b": -5.5550908189043e-05,
      "mean_dH_a_minus_b": -1.7400851761915823e-05,
      "frac_a_lower_S": 0.44375,
      "frac_a_lower_H": 0.43125,
      "a_beats_b_on_mean_S": true,
      "a_beats_b_on_mean_H": true
    },
    "struct_content_vs_random": {
      "n": 160,
      "mean_dS_a_minus_b": -1.8891283819530245e-05,
      "mean_dH_a_minus_b": -1.9099887956330153e-05,
      "frac_a_lower_S": 0.425,
      "frac_a_lower_H": 0.41875,
      "a_beats_b_on_mean_S": true,
      "a_beats_b_on_mean_H": true
    },
    "oracle_vs_random": {
      "n": 160,
      "mean_dS_a_minus_b": -8.836283960081737e-05,
      "mean_dH_a_minus_b": -4.106372448932305e-05,
      "frac_a_lower_S": 0.5,
      "frac_a_lower_H": 0.41875,
      "a_beats_b_on_mean_S": true,
      "a_beats_b_on_mean_H": true
    },
    "oracle_vs_struct_content": {
      "n": 160,
      "mean_dS_a_minus_b": -6.947155578128714e-05,
      "mean_dH_a_minus_b": -2.19638365329929e-05,
      "frac_a_lower_S": 0.50625,
      "frac_a_lower_H": 0.4125,
      "a_beats_b_on_mean_S": true,
      "a_beats_b_on_mean_H": true
    },
    "bucket_vs_struct_content": {
      "n": 160,
      "mean_dS_a_minus_b": -3.665962436951275e-05,
      "mean_dH_a_minus_b": 1.6990361944143234e-06,
      "frac_a_lower_S": 0.10625,
      "frac_a_lower_H": 0.10625,
      "a_beats_b_on_mean_S": true,
      "a_beats_b_on_mean_H": false
    }
  },
  "budget=0.5|kv": {
    "oracle_vs_bucket": {
      "n": 160,
      "mean_dS_a_minus_b": -1.1288259798813987e-05,
      "mean_dH_a_minus_b": -4.499137071878321e-06,
      "frac_a_lower_S": 0.525,
      "frac_a_lower_H": 0.5375,
      "a_beats_b_on_mean_S": true,
      "a_beats_b_on_mean_H": true
    },
    "bucket_vs_random": {
      "n": 160,
      "mean_dS_a_minus_b": 2.5935052316169126e-06,
      "mean_dH_a_minus_b": -1.4472905659908728e-06,
      "frac_a_lower_S": 0.38125,
      "frac_a_lower_H": 0.35625,
      "a_beats_b_on_mean_S": false,
      "a_beats_b_on_mean_H": true
    },
    "struct_content_vs_random": {
      "n": 160,
      "mean_dS_a_minus_b": -2.7153560977725404e-06,
      "mean_dH_a_minus_b": -3.6744156624942312e-06,
      "frac_a_lower_S": 0.41875,
      "frac_a_lower_H": 0.3875,
      "a_beats_b_on_mean_S": true,
      "a_beats_b_on_mean_H": true
    },
    "oracle_vs_random": {
      "n": 160,
      "mean_dS_a_minus_b": -8.694754567197068e-06,
      "mean_dH_a_minus_b": -5.9464276378691995e-06,
      "frac_a_lower_S": 0.4625,
      "frac_a_lower_H": 0.4375,
      "a_beats_b_on_mean_S": true,
      "a_beats_b_on_mean_H": true
    },
    "oracle_vs_struct_content": {
      "n": 160,
      "mean_dS_a_minus_b": -5.979398469424524e-06,
      "mean_dH_a_minus_b": -2.272011975374964e-06,
      "frac_a_lower_S": 0.49375,
      "frac_a_lower_H": 0.50625,
      "a_beats_b_on_mean_S": true,
      "a_beats_b_on_mean_H": true
    },
    "bucket_vs_struct_content": {
      "n": 160,
      "mean_dS_a_minus_b": 5.308861329389457e-06,
      "mean_dH_a_minus_b": 2.227125096503357e-06,
      "frac_a_lower_S": 0.0875,
      "frac_a_lower_H": 0.08125,
      "a_beats_b_on_mean_S": false,
      "a_beats_b_on_mean_H": false
    }
  }
}
```

### H_pos table

```
{
  "bucket|budget=0.25|kv": {
    "n": 160,
    "H_pos_mean": 5.973506100061421e-05,
    "H_pos_median": 3.746591966091728e-06,
    "H_pos_p95": 0.00031898370887120187,
    "S_mean_mean": -4.6146202525374504e-05,
    "S_mean_median": -2.8152367917755327e-06,
    "frac_S_positive": 0.4625,
    "mean_n_protect": 1.75,
    "mean_n_quant": 3.875
  },
  "bucket|budget=0.5|kv": {
    "n": 160,
    "H_pos_mean": 3.275998262529889e-05,
    "H_pos_median": 3.288793539013568e-06,
    "H_pos_p95": 0.00013584326863338002,
    "S_mean_mean": -4.090167757918965e-05,
    "S_mean_median": -8.490629359888165e-07,
    "frac_S_positive": 0.4625,
    "mean_n_protect": 3.0,
    "mean_n_quant": 2.625
  },
  "none|budget=0.0|kv": {
    "n": 160,
    "H_pos_mean": 2.704787151078135e-05,
    "H_pos_median": 1.8371344291760274e-06,
    "H_pos_p95": 0.00013404726330747967,
    "S_mean_mean": -0.00012562091843814884,
    "S_mean_median": -2.6745591715856187e-06,
    "frac_S_positive": 0.38125,
    "mean_n_protect": 0.0,
    "mean_n_quant": 5.625
  },
  "oracle_H|budget=0.25|kv": {
    "n": 160,
    "H_pos_mean": 3.607218827320698e-05,
    "H_pos_median": 2.7382343432691414e-06,
    "H_pos_p95": 0.00021356615016429744,
    "S_mean_mean": -7.895813393714886e-05,
    "S_mean_median": -1.9851998141954635e-06,
    "frac_S_positive": 0.425,
    "mean_n_protect": 1.75,
    "mean_n_quant": 3.875
  },
  "oracle_H|budget=0.5|kv": {
    "n": 160,
    "H_pos_mean": 2.8260845553420562e-05,
    "H_pos_median": 5.381747769206413e-07,
    "H_pos_p95": 5.307052925616064e-05,
    "S_mean_mean": -5.2189937378003626e-05,
    "S_mean_median": -4.221779715862795e-06,
    "frac_S_positive": 0.36875,
    "mean_n_protect": 3.0,
    "mean_n_quant": 2.625
  },
  "random|budget=0.25|kv": {
    "n": 160,
    "H_pos_mean": 7.713591276253003e-05,
    "H_pos_median": 4.375372782305931e-06,
    "H_pos_p95": 0.00025829514040083264,
    "S_mean_mean": 9.404705663668507e-06,
    "S_mean_median": -1.6572423646721294e-07,
    "frac_S_positive": 0.49375,
    "mean_n_protect": 1.75,
    "mean_n_quant": 3.875
  },
  "random|budget=0.5|kv": {
    "n": 160,
    "H_pos_mean": 3.4207273191289773e-05,
    "H_pos_median": 1.5019186472879179e-06,
    "H_pos_p95": 0.00014444858875322107,
    "S_mean_mean": -4.3495182810806555e-05,
    "S_mean_median": -1.6935274326594178e-06,
    "frac_S_positive": 0.43125,
    "mean_n_protect": 3.0,
    "mean_n_quant": 2.625
  },
  "struct+content|budget=0.25|kv": {
    "n": 160,
    "H_pos_mean": 5.8036024806199885e-05,
    "H_pos_median": 3.746591966091728e-06,
    "H_pos_p95": 0.00031898370887120187,
    "S_mean_mean": -9.486578155861726e-06,
    "S_mean_median": -1.9984324467259284e-06,
    "frac_S_positive": 0.4625,
    "mean_n_protect": 1.75,
    "mean_n_quant": 3.875
  },
  "struct+content|budget=0.5|kv": {
    "n": 160,
    "H_pos_mean": 3.053285752879552e-05,
    "H_pos_median": 2.8480073581249597e-06,
    "H_pos_p95": 0.00013584326863338002,
    "S_mean_mean": -4.62105389085791e-05,
    "S_mean_median": -1.8367497887084028e-06,
    "frac_S_positive": 0.44375,
    "mean_n_protect": 3.0,
    "mean_n_quant": 2.625
  }
}
```

### Recovery on H_pos (when H_none>0)

```
{
  "oracle_H|budget=0.25|kv": {
    "n_with_H_none_pos": 107,
    "recovery_H_mean": -245.06227240104738,
    "recovery_H_median": 0.20681104306423792,
    "frac_lower_H_than_none": 0.5887850467289719
  },
  "oracle_H|budget=0.5|kv": {
    "n_with_H_none_pos": 107,
    "recovery_H_mean": -272.59560247511735,
    "recovery_H_median": 0.8211985375417324,
    "frac_lower_H_than_none": 0.6822429906542056
  },
  "bucket|budget=0.25|kv": {
    "n_with_H_none_pos": 107,
    "recovery_H_mean": -248.50601546470668,
    "recovery_H_median": 0.1918023789322233,
    "frac_lower_H_than_none": 0.5607476635514018
  },
  "bucket|budget=0.5|kv": {
    "n_with_H_none_pos": 107,
    "recovery_H_mean": -31.0067247716688,
    "recovery_H_median": 0.08036007748340102,
    "frac_lower_H_than_none": 0.6074766355140186
  },
  "random|budget=0.25|kv": {
    "n_with_H_none_pos": 107,
    "recovery_H_mean": -239.0480501851568,
    "recovery_H_median": 0.04149413265722457,
    "frac_lower_H_than_none": 0.5327102803738317
  },
  "random|budget=0.5|kv": {
    "n_with_H_none_pos": 107,
    "recovery_H_mean": -235.61792938107453,
    "recovery_H_median": 0.47254552818508355,
    "frac_lower_H_than_none": 0.6822429906542056
  },
  "struct+content|budget=0.25|kv": {
    "n_with_H_none_pos": 107,
    "recovery_H_mean": -254.03446253703956,
    "recovery_H_median": 0.24474612328607026,
    "frac_lower_H_than_none": 0.5607476635514018
  },
  "struct+content|budget=0.5|kv": {
    "n_with_H_none_pos": 107,
    "recovery_H_mean": -28.015646324987017,
    "recovery_H_median": 0.31983729928640814,
    "frac_lower_H_than_none": 0.6355140186915887
  }
}
```

Non-retrieval not run in this arm.
