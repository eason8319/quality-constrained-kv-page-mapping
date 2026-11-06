# Protect strategy A/B v1 (CPU-only on existing page_harms)

Compare deployable heuristics vs oracle top-H / random / recency at equal page budgets.
Heuristics never use the target page's measured H (except `oracle_H`).

- random: mean of 20 seeded shuffles per group (base seed 20261105)
- bucket: fixed order earlier_history > first_token_page > recent_closed; older first within bucket
- bucket_loo: leave-one-prefix-out mean H of bucket (same template+intervene)
- lh_band: LOO mean H stratified by layer band (early/mid/late thirds) x bucket for the curve's (l,h) band
- recency: highest page_index first

- gates: {"templates": [{"template": "retrieval", "n_rows_in": 2700, "n_prefixes": 32, "n_groups": 480, "n_heads": 5, "heads": [[2, 1], [10, 3], [22, 2], [22, 4], [24, 6]], "intervenes": ["k", "kv", "v"], "harms_path": "/cluster/home/zengy/quality-constrained-kv-page-mapping/runs/rep_sample_v1/page_harms.jsonl"}, {"template": "nonretr", "n_rows_in": 7590, "n_prefixes": 32, "n_groups": 480, "n_heads": 5, "heads": [[2, 1], [10, 3], [22, 2], [22, 4], [24, 6]], "intervenes": ["k", "kv", "v"], "harms_path": "/cluster/home/zengy/quality-constrained-kv-page-mapping/runs/rep_sample_nonretr_v1/page_harms.jsonl"}], "n_coverage_rows": 5760, "expected_rows_per_group": 6, "strategies": ["oracle_H", "random", "recency", "bucket", "bucket_loo", "lh_band"]}

## Pooled mean coverage (all prefix x lh curves)

### template=retrieval

#### intervene=kv

| strategy | @5% | @10% | @20% | @25% | @50% | frac->50% | frac->80% | frac->95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_H | 0.6373 | 0.6373 | 0.7476 | 0.8049 | 0.9504 | 0.2593 | 0.3885 | 0.5302 |
| random | 0.1920 | 0.1920 | 0.2636 | 0.3154 | 0.5423 | 0.6013 | 0.7578 | 0.8422 |
| recency | 0.1663 | 0.1663 | 0.2520 | 0.2932 | 0.5809 | 0.6066 | 0.7430 | 0.8316 |
| bucket | 0.1619 | 0.1619 | 0.2540 | 0.3385 | 0.5627 | 0.5926 | 0.7395 | 0.8165 |
| bucket_loo | 0.2053 | 0.2053 | 0.2439 | 0.2812 | 0.5160 | 0.6323 | 0.7591 | 0.8400 |
| lh_band | 0.1959 | 0.1959 | 0.2345 | 0.2718 | 0.5043 | 0.6406 | 0.7653 | 0.8442 |

#### intervene=k

| strategy | @5% | @10% | @20% | @25% | @50% | frac->50% | frac->80% | frac->95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_H | 0.6124 | 0.6124 | 0.7528 | 0.8160 | 0.9591 | 0.2466 | 0.3904 | 0.5174 |
| random | 0.2044 | 0.2044 | 0.2740 | 0.3239 | 0.5519 | 0.5933 | 0.7509 | 0.8273 |
| recency | 0.1778 | 0.1778 | 0.2622 | 0.3019 | 0.5646 | 0.5909 | 0.7644 | 0.8395 |
| bucket | 0.2019 | 0.2019 | 0.2672 | 0.3540 | 0.5741 | 0.5757 | 0.7312 | 0.8049 |
| bucket_loo | 0.1815 | 0.1815 | 0.2196 | 0.2583 | 0.5015 | 0.6051 | 0.7898 | 0.8645 |
| lh_band | 0.1845 | 0.1845 | 0.2262 | 0.2648 | 0.5288 | 0.5904 | 0.7684 | 0.8483 |

#### intervene=v

| strategy | @5% | @10% | @20% | @25% | @50% | frac->50% | frac->80% | frac->95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_H | 0.6302 | 0.6302 | 0.7517 | 0.8076 | 0.9522 | 0.2565 | 0.3897 | 0.5290 |
| random | 0.1987 | 0.1987 | 0.2723 | 0.3215 | 0.5514 | 0.5947 | 0.7471 | 0.8366 |
| recency | 0.1880 | 0.1880 | 0.2631 | 0.3034 | 0.5552 | 0.6076 | 0.7582 | 0.8378 |
| bucket | 0.2056 | 0.2056 | 0.2749 | 0.3420 | 0.5685 | 0.5847 | 0.7410 | 0.8283 |
| bucket_loo | 0.1752 | 0.1752 | 0.2173 | 0.2743 | 0.5282 | 0.6062 | 0.7686 | 0.8533 |
| lh_band | 0.1787 | 0.1787 | 0.2261 | 0.2831 | 0.5334 | 0.6052 | 0.7697 | 0.8533 |

### template=nonretr

#### intervene=kv

| strategy | @5% | @10% | @20% | @25% | @50% | frac->50% | frac->80% | frac->95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_H | 0.3489 | 0.5717 | 0.7796 | 0.8270 | 0.9748 | 0.1537 | 0.2717 | 0.3997 |
| random | 0.0663 | 0.1317 | 0.2309 | 0.2647 | 0.4999 | 0.5337 | 0.7706 | 0.8813 |
| recency | 0.0714 | 0.1121 | 0.1980 | 0.2413 | 0.4881 | 0.5562 | 0.7844 | 0.8987 |
| bucket | 0.0732 | 0.1513 | 0.2300 | 0.2703 | 0.5301 | 0.5007 | 0.7517 | 0.8636 |
| bucket_loo | 0.0464 | 0.1160 | 0.2242 | 0.2595 | 0.5096 | 0.5282 | 0.7641 | 0.8739 |
| lh_band | 0.0427 | 0.1090 | 0.2222 | 0.2581 | 0.4880 | 0.5330 | 0.7825 | 0.8831 |

#### intervene=k

| strategy | @5% | @10% | @20% | @25% | @50% | frac->50% | frac->80% | frac->95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_H | 0.3363 | 0.5699 | 0.7805 | 0.8276 | 0.9778 | 0.1533 | 0.2721 | 0.3952 |
| random | 0.0631 | 0.1293 | 0.2281 | 0.2621 | 0.4953 | 0.5374 | 0.7748 | 0.8814 |
| recency | 0.0747 | 0.1227 | 0.2043 | 0.2513 | 0.5285 | 0.5254 | 0.7536 | 0.8731 |
| bucket | 0.0588 | 0.1403 | 0.2040 | 0.2367 | 0.4913 | 0.5190 | 0.7683 | 0.8753 |
| bucket_loo | 0.0396 | 0.1140 | 0.2122 | 0.2509 | 0.4658 | 0.5545 | 0.7746 | 0.8785 |
| lh_band | 0.0416 | 0.1195 | 0.2071 | 0.2438 | 0.4732 | 0.5429 | 0.7727 | 0.8791 |

#### intervene=v

| strategy | @5% | @10% | @20% | @25% | @50% | frac->50% | frac->80% | frac->95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_H | 0.3376 | 0.5526 | 0.7731 | 0.8207 | 0.9765 | 0.1539 | 0.2749 | 0.4091 |
| random | 0.0673 | 0.1297 | 0.2295 | 0.2642 | 0.5039 | 0.5319 | 0.7802 | 0.8855 |
| recency | 0.0832 | 0.1301 | 0.2241 | 0.2686 | 0.5018 | 0.5226 | 0.7690 | 0.8887 |
| bucket | 0.0698 | 0.1494 | 0.2316 | 0.2723 | 0.5040 | 0.5206 | 0.7759 | 0.8760 |
| bucket_loo | 0.0603 | 0.1377 | 0.2276 | 0.2698 | 0.4981 | 0.5252 | 0.7768 | 0.8749 |
| lh_band | 0.0692 | 0.1671 | 0.2469 | 0.2891 | 0.5191 | 0.5171 | 0.7605 | 0.8737 |

## Gap vs oracle @25% budget

```json
{
  "nonretr|k": {
    "frac": "0.25",
    "oracle_coverage": 0.8276243997360092,
    "strategies": {
      "bucket": {
        "coverage": 0.23672706184606226,
        "gap_vs_oracle": 0.5908973378899469
      },
      "bucket_loo": {
        "coverage": 0.2508941381392672,
        "gap_vs_oracle": 0.576730261596742
      },
      "lh_band": {
        "coverage": 0.24375794252588232,
        "gap_vs_oracle": 0.5838664572101269
      },
      "oracle_H": {
        "coverage": 0.8276243997360092,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.26207679617658297,
        "gap_vs_oracle": 0.5655476035594262
      },
      "recency": {
        "coverage": 0.2513489460176155,
        "gap_vs_oracle": 0.5762754537183937
      }
    }
  },
  "nonretr|kv": {
    "frac": "0.25",
    "oracle_coverage": 0.8269866950070061,
    "strategies": {
      "bucket": {
        "coverage": 0.2702785058083549,
        "gap_vs_oracle": 0.5567081891986512
      },
      "bucket_loo": {
        "coverage": 0.2594687267025086,
        "gap_vs_oracle": 0.5675179683044975
      },
      "lh_band": {
        "coverage": 0.2581270646703028,
        "gap_vs_oracle": 0.5688596303367033
      },
      "oracle_H": {
        "coverage": 0.8269866950070061,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.26470520412850707,
        "gap_vs_oracle": 0.562281490878499
      },
      "recency": {
        "coverage": 0.2412638786250036,
        "gap_vs_oracle": 0.5857228163820025
      }
    }
  },
  "nonretr|v": {
    "frac": "0.25",
    "oracle_coverage": 0.8206766560148038,
    "strategies": {
      "bucket": {
        "coverage": 0.27231560975074276,
        "gap_vs_oracle": 0.548361046264061
      },
      "bucket_loo": {
        "coverage": 0.26983521177657005,
        "gap_vs_oracle": 0.5508414442382337
      },
      "lh_band": {
        "coverage": 0.28905562051067774,
        "gap_vs_oracle": 0.531621035504126
      },
      "oracle_H": {
        "coverage": 0.8206766560148038,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.26419590416055816,
        "gap_vs_oracle": 0.5564807518542456
      },
      "recency": {
        "coverage": 0.26858706642216906,
        "gap_vs_oracle": 0.5520895895926348
      }
    }
  },
  "retrieval|k": {
    "frac": "0.25",
    "oracle_coverage": 0.8160060085708505,
    "strategies": {
      "bucket": {
        "coverage": 0.3540282262677322,
        "gap_vs_oracle": 0.46197778230311826
      },
      "bucket_loo": {
        "coverage": 0.2582627507919685,
        "gap_vs_oracle": 0.557743257778882
      },
      "lh_band": {
        "coverage": 0.2648410743534114,
        "gap_vs_oracle": 0.5511649342174391
      },
      "oracle_H": {
        "coverage": 0.8160060085708505,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.32386241662360404,
        "gap_vs_oracle": 0.49214359194724644
      },
      "recency": {
        "coverage": 0.30194141291603693,
        "gap_vs_oracle": 0.5140645956548135
      }
    }
  },
  "retrieval|kv": {
    "frac": "0.25",
    "oracle_coverage": 0.8048543859338612,
    "strategies": {
      "bucket": {
        "coverage": 0.33845284666093406,
        "gap_vs_oracle": 0.4664015392729271
      },
      "bucket_loo": {
        "coverage": 0.28121031490843024,
        "gap_vs_oracle": 0.5236440710254309
      },
      "lh_band": {
        "coverage": 0.2717702374483993,
        "gap_vs_oracle": 0.5330841484854618
      },
      "oracle_H": {
        "coverage": 0.8048543859338612,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.3154242812307911,
        "gap_vs_oracle": 0.4894301047030701
      },
      "recency": {
        "coverage": 0.293216253007095,
        "gap_vs_oracle": 0.5116381329267662
      }
    }
  },
  "retrieval|v": {
    "frac": "0.25",
    "oracle_coverage": 0.8075694396724739,
    "strategies": {
      "bucket": {
        "coverage": 0.3419732569289635,
        "gap_vs_oracle": 0.46559618274351033
      },
      "bucket_loo": {
        "coverage": 0.2742707336580755,
        "gap_vs_oracle": 0.5332987060143983
      },
      "lh_band": {
        "coverage": 0.28305159744994024,
        "gap_vs_oracle": 0.5245178422225336
      },
      "oracle_H": {
        "coverage": 0.8075694396724739,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.3215049722715651,
        "gap_vs_oracle": 0.48606446740090875
      },
      "recency": {
        "coverage": 0.30336742676278566,
        "gap_vs_oracle": 0.5042020129096882
      }
    }
  }
}
```

## Design notes

- Harm: H_p = max(0,S_jia)+max(0,S_yi) from rep_sample page_harms.
- Coverage = sum H of selected pages / total positive H in the (prefix,l,h,iv) group.
- No GPU; no re-quantization. Ranking quality on measured single-page H (proxy for protect-and-keep-FP).
