# Protect strategy A/B v1 (CPU-only on existing page_harms)

Compare deployable heuristics vs oracle top-H / random / recency at equal page budgets.
Heuristics never use the target page's measured H (except `oracle_H`).

- random: mean of 20 seeded shuffles per group (base seed 20261105)
- bucket: fixed order earlier_history > first_token_page > recent_closed; older first within bucket
- bucket_loo: leave-one-prefix-out mean H of bucket (same template+intervene)
- lh_band: LOO mean H stratified by layer band (early/mid/late thirds) x bucket for the curve's (l,h) band
- recency: highest page_index first

- gates: {"templates": [{"template": "retrieval", "n_rows_in": 2700, "n_prefixes": 32, "n_groups": 480, "n_heads": 5, "heads": [[2, 1], [10, 3], [22, 2], [22, 4], [24, 6]], "intervenes": ["k", "kv", "v"], "harms_path": "runs/rep_sample_b2_v1/page_harms.jsonl"}, {"template": "nonretr", "n_rows_in": 7590, "n_prefixes": 32, "n_groups": 480, "n_heads": 5, "heads": [[2, 1], [10, 3], [22, 2], [22, 4], [24, 6]], "intervenes": ["k", "kv", "v"], "harms_path": "runs/rep_sample_nonretr_b2_v1/page_harms.jsonl"}], "n_coverage_rows": 5760, "expected_rows_per_group": 6, "strategies": ["oracle_H", "random", "recency", "bucket", "bucket_loo", "lh_band"]}

## Pooled mean coverage (all prefix x lh curves)

### template=retrieval

#### intervene=kv

| strategy | @5% | @10% | @20% | @25% | @50% | frac->50% | frac->80% | frac->95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_H | 0.6641 | 0.6641 | 0.7795 | 0.8331 | 0.9564 | 0.2377 | 0.3639 | 0.5207 |
| random | 0.2035 | 0.2035 | 0.2766 | 0.3248 | 0.5559 | 0.5899 | 0.7280 | 0.8354 |
| recency | 0.1929 | 0.1929 | 0.2411 | 0.2797 | 0.4795 | 0.6403 | 0.7667 | 0.8521 |
| bucket | 0.2371 | 0.2371 | 0.3343 | 0.4059 | 0.6118 | 0.5377 | 0.6787 | 0.8157 |
| bucket_loo | 0.2292 | 0.2292 | 0.3404 | 0.4121 | 0.6126 | 0.5388 | 0.6777 | 0.8136 |
| lh_band | 0.2064 | 0.2064 | 0.3350 | 0.4051 | 0.5944 | 0.5503 | 0.6937 | 0.8290 |

#### intervene=k

| strategy | @5% | @10% | @20% | @25% | @50% | frac->50% | frac->80% | frac->95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_H | 0.6319 | 0.6319 | 0.7551 | 0.8067 | 0.9506 | 0.2537 | 0.3934 | 0.5234 |
| random | 0.1967 | 0.1967 | 0.2691 | 0.3166 | 0.5473 | 0.5950 | 0.7548 | 0.8337 |
| recency | 0.1433 | 0.1433 | 0.2161 | 0.2391 | 0.4965 | 0.6461 | 0.8049 | 0.8605 |
| bucket | 0.2100 | 0.2100 | 0.3119 | 0.3865 | 0.6300 | 0.5411 | 0.6965 | 0.8137 |
| bucket_loo | 0.2106 | 0.2106 | 0.3163 | 0.3909 | 0.5557 | 0.5793 | 0.7461 | 0.8236 |
| lh_band | 0.1842 | 0.1842 | 0.2911 | 0.3686 | 0.5498 | 0.5982 | 0.7501 | 0.8264 |

#### intervene=v

| strategy | @5% | @10% | @20% | @25% | @50% | frac->50% | frac->80% | frac->95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_H | 0.6029 | 0.6029 | 0.7328 | 0.7966 | 0.9486 | 0.2628 | 0.4009 | 0.5392 |
| random | 0.2076 | 0.2076 | 0.2806 | 0.3301 | 0.5473 | 0.5871 | 0.7564 | 0.8455 |
| recency | 0.1917 | 0.1917 | 0.2679 | 0.3102 | 0.5337 | 0.6204 | 0.7762 | 0.8788 |
| bucket | 0.2022 | 0.2022 | 0.2732 | 0.3489 | 0.5475 | 0.5758 | 0.7497 | 0.8417 |
| bucket_loo | 0.1832 | 0.1832 | 0.2591 | 0.3347 | 0.5326 | 0.5947 | 0.7718 | 0.8759 |
| lh_band | 0.1855 | 0.1855 | 0.2710 | 0.3349 | 0.5008 | 0.6169 | 0.7871 | 0.8798 |

### template=nonretr

#### intervene=kv

| strategy | @5% | @10% | @20% | @25% | @50% | frac->50% | frac->80% | frac->95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_H | 0.3717 | 0.5979 | 0.7899 | 0.8386 | 0.9795 | 0.1434 | 0.2589 | 0.3940 |
| random | 0.0668 | 0.1338 | 0.2303 | 0.2667 | 0.5039 | 0.5307 | 0.7605 | 0.8782 |
| recency | 0.0744 | 0.1261 | 0.2100 | 0.2508 | 0.4574 | 0.5706 | 0.7904 | 0.8905 |
| bucket | 0.0736 | 0.1456 | 0.2390 | 0.2779 | 0.4940 | 0.5301 | 0.7695 | 0.8786 |
| bucket_loo | 0.0507 | 0.1096 | 0.2053 | 0.2477 | 0.4905 | 0.5646 | 0.7891 | 0.9012 |
| lh_band | 0.0775 | 0.1585 | 0.2674 | 0.3197 | 0.5269 | 0.5101 | 0.7649 | 0.8882 |

#### intervene=k

| strategy | @5% | @10% | @20% | @25% | @50% | frac->50% | frac->80% | frac->95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_H | 0.3624 | 0.5847 | 0.7875 | 0.8394 | 0.9799 | 0.1469 | 0.2629 | 0.3904 |
| random | 0.0635 | 0.1304 | 0.2249 | 0.2615 | 0.4992 | 0.5337 | 0.7675 | 0.8843 |
| recency | 0.0637 | 0.1052 | 0.1947 | 0.2347 | 0.4696 | 0.5565 | 0.8040 | 0.8975 |
| bucket | 0.0740 | 0.1543 | 0.2188 | 0.2627 | 0.5079 | 0.5275 | 0.7656 | 0.8763 |
| bucket_loo | 0.0680 | 0.1532 | 0.2188 | 0.2627 | 0.5078 | 0.5304 | 0.7716 | 0.8802 |
| lh_band | 0.0971 | 0.1677 | 0.2466 | 0.2939 | 0.5135 | 0.5127 | 0.7589 | 0.8796 |

#### intervene=v

| strategy | @5% | @10% | @20% | @25% | @50% | frac->50% | frac->80% | frac->95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_H | 0.3625 | 0.5866 | 0.7886 | 0.8411 | 0.9813 | 0.1500 | 0.2603 | 0.3870 |
| random | 0.0634 | 0.1289 | 0.2308 | 0.2661 | 0.4990 | 0.5331 | 0.7671 | 0.8788 |
| recency | 0.0784 | 0.1346 | 0.2223 | 0.2678 | 0.4910 | 0.5307 | 0.7787 | 0.8868 |
| bucket | 0.0661 | 0.1213 | 0.2086 | 0.2554 | 0.4777 | 0.5522 | 0.7803 | 0.8848 |
| bucket_loo | 0.0562 | 0.1346 | 0.2156 | 0.2558 | 0.5051 | 0.5336 | 0.7814 | 0.8917 |
| lh_band | 0.0661 | 0.1442 | 0.2308 | 0.2702 | 0.5058 | 0.5328 | 0.7702 | 0.8891 |

## Gap vs oracle @25% budget

```json
{
  "nonretr|k": {
    "frac": "0.25",
    "oracle_coverage": 0.839414886902734,
    "strategies": {
      "bucket": {
        "coverage": 0.2626829425878915,
        "gap_vs_oracle": 0.5767319443148425
      },
      "bucket_loo": {
        "coverage": 0.26265417245359685,
        "gap_vs_oracle": 0.5767607144491371
      },
      "lh_band": {
        "coverage": 0.2938777082957732,
        "gap_vs_oracle": 0.5455371786069608
      },
      "oracle_H": {
        "coverage": 0.839414886902734,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.26147362987683903,
        "gap_vs_oracle": 0.577941257025895
      },
      "recency": {
        "coverage": 0.23469772350632345,
        "gap_vs_oracle": 0.6047171633964106
      }
    }
  },
  "nonretr|kv": {
    "frac": "0.25",
    "oracle_coverage": 0.8386378557057725,
    "strategies": {
      "bucket": {
        "coverage": 0.2778863392997616,
        "gap_vs_oracle": 0.560751516406011
      },
      "bucket_loo": {
        "coverage": 0.24769446209245163,
        "gap_vs_oracle": 0.5909433936133209
      },
      "lh_band": {
        "coverage": 0.31972778765671656,
        "gap_vs_oracle": 0.5189100680490559
      },
      "oracle_H": {
        "coverage": 0.8386378557057725,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.26668427606498113,
        "gap_vs_oracle": 0.5719535796407914
      },
      "recency": {
        "coverage": 0.25079157138821445,
        "gap_vs_oracle": 0.587846284317558
      }
    }
  },
  "nonretr|v": {
    "frac": "0.25",
    "oracle_coverage": 0.8410579599152079,
    "strategies": {
      "bucket": {
        "coverage": 0.2553908680419964,
        "gap_vs_oracle": 0.5856670918732115
      },
      "bucket_loo": {
        "coverage": 0.2557689776698542,
        "gap_vs_oracle": 0.5852889822453538
      },
      "lh_band": {
        "coverage": 0.27018474181911556,
        "gap_vs_oracle": 0.5708732180960924
      },
      "oracle_H": {
        "coverage": 0.8410579599152079,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.2660920484270983,
        "gap_vs_oracle": 0.5749659114881096
      },
      "recency": {
        "coverage": 0.26776704130605494,
        "gap_vs_oracle": 0.573290918609153
      }
    }
  },
  "retrieval|k": {
    "frac": "0.25",
    "oracle_coverage": 0.806682666393678,
    "strategies": {
      "bucket": {
        "coverage": 0.38646108983230426,
        "gap_vs_oracle": 0.42022157656137377
      },
      "bucket_loo": {
        "coverage": 0.3908964495242887,
        "gap_vs_oracle": 0.4157862168693893
      },
      "lh_band": {
        "coverage": 0.3685857264466168,
        "gap_vs_oracle": 0.4380969399470612
      },
      "oracle_H": {
        "coverage": 0.806682666393678,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.3166494854450914,
        "gap_vs_oracle": 0.4900331809485866
      },
      "recency": {
        "coverage": 0.23910437118524897,
        "gap_vs_oracle": 0.5675782952084291
      }
    }
  },
  "retrieval|kv": {
    "frac": "0.25",
    "oracle_coverage": 0.8331029161487375,
    "strategies": {
      "bucket": {
        "coverage": 0.4059297870055964,
        "gap_vs_oracle": 0.42717312914314115
      },
      "bucket_loo": {
        "coverage": 0.41207029368415826,
        "gap_vs_oracle": 0.4210326224645793
      },
      "lh_band": {
        "coverage": 0.4051092623504178,
        "gap_vs_oracle": 0.4279936537983198
      },
      "oracle_H": {
        "coverage": 0.8331029161487375,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.3247987375912402,
        "gap_vs_oracle": 0.5083041785574973
      },
      "recency": {
        "coverage": 0.2796783134480719,
        "gap_vs_oracle": 0.5534246027006656
      }
    }
  },
  "retrieval|v": {
    "frac": "0.25",
    "oracle_coverage": 0.7966378425806291,
    "strategies": {
      "bucket": {
        "coverage": 0.34890737065904676,
        "gap_vs_oracle": 0.44773047192158233
      },
      "bucket_loo": {
        "coverage": 0.3347446344882473,
        "gap_vs_oracle": 0.46189320809238177
      },
      "lh_band": {
        "coverage": 0.3348875242303767,
        "gap_vs_oracle": 0.4617503183502524
      },
      "oracle_H": {
        "coverage": 0.7966378425806291,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.33011142856247255,
        "gap_vs_oracle": 0.46652641401815653
      },
      "recency": {
        "coverage": 0.3102173225660278,
        "gap_vs_oracle": 0.4864205200146013
      }
    }
  }
}
```

## Design notes

- Harm: H_p = max(0,S_jia)+max(0,S_yi) from rep_sample page_harms.
- Coverage = sum H of selected pages / total positive H in the (prefix,l,h,iv) group.
- No GPU; no re-quantization. Ranking quality on measured single-page H (proxy for protect-and-keep-FP).
