# Protect close-time content features v1

Offline page-close prediction: structural vs content (range/scale/outlier) features.
Reuse measured H from rep_sample page_harms; do not recompute S.

## Features at close (no future / no d)

- outlier_thr = 3.0 on MAD rule: count |x-median| > thr * 1.4826 * MAD over page K or V tensor
- k_range_p50/p95: percentiles of per-channel (max-min along L_p) on K
- v_range_p50/p95: percentiles of per-token (max-min along D) on V
- k_scale_mean / v_scale_mean: mean asymmetric uniform pack scales (b=4, g=64, intervene=kv pack)
- k_sat_frac / v_sat_frac: fraction of codes at 0 or qmax
- struct: page_pos_bucket + page_index + first_token (bucket encodes first_token)
- **Excluded from deployable score:** d (needs t_key / answer position)

## Strategies

- oracle_H: rank by measured H (peek)
- random: mean of 20 seeded shuffles
- bucket: fixed earlier_history > first_token_page > recent_closed (same as protect_strategy_v1)
- struct: continuous struct_score = bucket*1e6 - page_index
- content: LOO Spearman-fit weights on z-scored content features (fit on other prefixes, same intervene)
- struct+content: z_loo(struct_score) + content_score

- leakage: {"feature_score_jia_yi_identical": {"n_checked": 3430, "n_pass": 3430, "pass_rate": 1.0}, "dup_feature_mismatch": 0, "traj_jia_yi_page_hash": {"n_checked": 10290, "n_pass": 10290, "pass_rate": 1.0}, "overall_pass_rate": 1.0, "note": "Scores are question-agnostic (close-time only); jia/yi necessarily identical.", "pass_rate_100": true}
- extract: {"phase": "extract", "wall_s": 24.449578523635864, "n_written_this_run": 3430, "feat_path": "runs/protect_close_feat_v1/close_features.jsonl", "device": "cuda:0", "heads": [[10, 3], [2, 1], [22, 2], [22, 4], [24, 6]], "outlier_thr": 3.0, "L_p": 64, "g": 64, "b": 4, "note": "One shared prefill per prefix; page K/V sliced from final past (immutable cache)."}

## Pooled mean coverage

### template=retrieval

#### intervene=kv

| strategy | @5% | @10% | @20% | @25% | @50% |
|---|---:|---:|---:|---:|---:|
| oracle_H | 0.6373 | 0.6373 | 0.7476 | 0.8049 | 0.9504 |
| random | 0.1959 | 0.1959 | 0.2717 | 0.3204 | 0.5450 |
| bucket | 0.1619 | 0.1619 | 0.2540 | 0.3385 | 0.5627 |
| struct | 0.1619 | 0.1619 | 0.2540 | 0.3385 | 0.5627 |
| content | 0.2040 | 0.2040 | 0.2811 | 0.3201 | 0.5595 |
| struct+content | 0.2147 | 0.2147 | 0.2776 | 0.3278 | 0.5651 |

#### intervene=k

| strategy | @5% | @10% | @20% | @25% | @50% |
|---|---:|---:|---:|---:|---:|
| oracle_H | 0.6124 | 0.6124 | 0.7528 | 0.8160 | 0.9591 |
| random | 0.2071 | 0.2071 | 0.2777 | 0.3260 | 0.5474 |
| bucket | 0.2019 | 0.2019 | 0.2672 | 0.3540 | 0.5741 |
| struct | 0.2019 | 0.2019 | 0.2672 | 0.3540 | 0.5741 |
| content | 0.1804 | 0.1804 | 0.2518 | 0.3206 | 0.5594 |
| struct+content | 0.2025 | 0.2025 | 0.2837 | 0.3545 | 0.5758 |

#### intervene=v

| strategy | @5% | @10% | @20% | @25% | @50% |
|---|---:|---:|---:|---:|---:|
| oracle_H | 0.6302 | 0.6302 | 0.7517 | 0.8076 | 0.9522 |
| random | 0.1956 | 0.1956 | 0.2647 | 0.3176 | 0.5527 |
| bucket | 0.2056 | 0.2056 | 0.2749 | 0.3420 | 0.5685 |
| struct | 0.2056 | 0.2056 | 0.2749 | 0.3420 | 0.5685 |
| content | 0.2120 | 0.2120 | 0.3004 | 0.3583 | 0.5819 |
| struct+content | 0.1950 | 0.1950 | 0.2926 | 0.3546 | 0.5913 |

### template=nonretr

#### intervene=kv

| strategy | @5% | @10% | @20% | @25% | @50% |
|---|---:|---:|---:|---:|---:|
| oracle_H | 0.3489 | 0.5717 | 0.7796 | 0.8270 | 0.9748 |
| random | 0.0661 | 0.1327 | 0.2284 | 0.2635 | 0.5049 |
| bucket | 0.0732 | 0.1513 | 0.2300 | 0.2703 | 0.5301 |
| struct | 0.0732 | 0.1513 | 0.2300 | 0.2703 | 0.5301 |
| content | 0.0814 | 0.1411 | 0.2175 | 0.2583 | 0.5244 |
| struct+content | 0.0786 | 0.1228 | 0.2273 | 0.2676 | 0.5106 |

#### intervene=k

| strategy | @5% | @10% | @20% | @25% | @50% |
|---|---:|---:|---:|---:|---:|
| oracle_H | 0.3363 | 0.5699 | 0.7805 | 0.8276 | 0.9778 |
| random | 0.0644 | 0.1319 | 0.2284 | 0.2632 | 0.5001 |
| bucket | 0.0588 | 0.1403 | 0.2040 | 0.2367 | 0.4913 |
| struct | 0.0588 | 0.1403 | 0.2040 | 0.2367 | 0.4913 |
| content | 0.0651 | 0.1184 | 0.2101 | 0.2439 | 0.4843 |
| struct+content | 0.0680 | 0.1116 | 0.2128 | 0.2423 | 0.4893 |

#### intervene=v

| strategy | @5% | @10% | @20% | @25% | @50% |
|---|---:|---:|---:|---:|---:|
| oracle_H | 0.3376 | 0.5526 | 0.7731 | 0.8207 | 0.9765 |
| random | 0.0674 | 0.1344 | 0.2323 | 0.2672 | 0.4983 |
| bucket | 0.0698 | 0.1494 | 0.2316 | 0.2723 | 0.5040 |
| struct | 0.0698 | 0.1494 | 0.2316 | 0.2723 | 0.5040 |
| content | 0.0596 | 0.1232 | 0.2243 | 0.2485 | 0.4803 |
| struct+content | 0.0644 | 0.1392 | 0.2263 | 0.2588 | 0.5105 |

## Gap vs oracle @25%

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
      "content": {
        "coverage": 0.2439243666311861,
        "gap_vs_oracle": 0.5837000331048231
      },
      "oracle_H": {
        "coverage": 0.8276243997360092,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.26316500974843215,
        "gap_vs_oracle": 0.564459389987577
      },
      "struct": {
        "coverage": 0.23672706184606226,
        "gap_vs_oracle": 0.5908973378899469
      },
      "struct+content": {
        "coverage": 0.2423174536235432,
        "gap_vs_oracle": 0.585306946112466
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
      "content": {
        "coverage": 0.2582751117203308,
        "gap_vs_oracle": 0.5687115832866753
      },
      "oracle_H": {
        "coverage": 0.8269866950070061,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.26354502574483957,
        "gap_vs_oracle": 0.5634416692621664
      },
      "struct": {
        "coverage": 0.2702785058083549,
        "gap_vs_oracle": 0.5567081891986512
      },
      "struct+content": {
        "coverage": 0.2675909571586329,
        "gap_vs_oracle": 0.5593957378483732
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
      "content": {
        "coverage": 0.2485267896668207,
        "gap_vs_oracle": 0.572149866347983
      },
      "oracle_H": {
        "coverage": 0.8206766560148038,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.2671597784162968,
        "gap_vs_oracle": 0.553516877598507
      },
      "struct": {
        "coverage": 0.27231560975074276,
        "gap_vs_oracle": 0.548361046264061
      },
      "struct+content": {
        "coverage": 0.2588135263109439,
        "gap_vs_oracle": 0.5618631297038599
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
      "content": {
        "coverage": 0.32062470823865213,
        "gap_vs_oracle": 0.49538130033219835
      },
      "oracle_H": {
        "coverage": 0.8160060085708505,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.3259589641723532,
        "gap_vs_oracle": 0.4900470443984973
      },
      "struct": {
        "coverage": 0.3540282262677322,
        "gap_vs_oracle": 0.46197778230311826
      },
      "struct+content": {
        "coverage": 0.3544864517230476,
        "gap_vs_oracle": 0.4615195568478029
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
      "content": {
        "coverage": 0.3201444512565387,
        "gap_vs_oracle": 0.4847099346773225
      },
      "oracle_H": {
        "coverage": 0.8048543859338612,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.3204094600281603,
        "gap_vs_oracle": 0.4844449259057009
      },
      "struct": {
        "coverage": 0.33845284666093406,
        "gap_vs_oracle": 0.4664015392729271
      },
      "struct+content": {
        "coverage": 0.32783830963266525,
        "gap_vs_oracle": 0.4770160763011959
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
      "content": {
        "coverage": 0.35831282512676765,
        "gap_vs_oracle": 0.4492566145457062
      },
      "oracle_H": {
        "coverage": 0.8075694396724739,
        "gap_vs_oracle": 0.0
      },
      "random": {
        "coverage": 0.31760615794587715,
        "gap_vs_oracle": 0.4899632817265967
      },
      "struct": {
        "coverage": 0.3419732569289635,
        "gap_vs_oracle": 0.46559618274351033
      },
      "struct+content": {
        "coverage": 0.3545754343143247,
        "gap_vs_oracle": 0.45299400535814915
      }
    }
  }
}
```

