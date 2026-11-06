# Strong-pair pilot v1

Template family: strong_pair_registry (multi-token rare phrases + near-miss distractors).

- elapsed_s: 1993.687444448471
- n_traj: 4736
- stop_expand: False
- gates: {"hash_equality_rate": 1.0, "hash_checks": 2304, "hash_fail": 0, "fp_both_correct_rate": 1.0, "fp_checked": 32, "zero_pert_pass_rate": 1.0, "zero_pert_checks": 64, "zero_pert_fail": 0, "manip_pass_rate": 1.0, "manip_checks": 32, "free_gen_both_correct_rate": 1.0, "free_gen_checked": 32, "free_gen_both_correct": 32, "free_gen_wrong_n": 0, "prefix_failures": []}

## Headline D (vs pilot_32 K 8.75e-5 / V 6.76e-5 / KV 5.00e-5)

```
{
  "D_k": {
    "n": 384,
    "mean": -1.8651905858829041e-06,
    "median": -2.3729541055221457e-07,
    "p95": 1.3554958449191208e-05
  },
  "D_v": {
    "n": 384,
    "mean": -2.3684029968629376e-06,
    "median": -2.3309904702747042e-07,
    "p95": 9.763399079574249e-06
  },
  "D_kv": {
    "n": 384,
    "mean": -2.6877054322068674e-06,
    "median": -2.1275769637914742e-07,
    "p95": 1.2167950208653425e-05
  }
}
```

## S_need - S_unneed

```
{
  "('k', 'ALL')": {
    "n": 768,
    "mean": -1.8651905858829044e-06,
    "median": -2.9902513481212214e-07,
    "p95": 1.826550489823829e-05
  },
  "('k', 'earlier_history')": {
    "n": 720,
    "mean": -1.8678126151725381e-06,
    "median": -1.9303767896330797e-07,
    "p95": 1.945781093764609e-05
  },
  "('k', 'first_token_page')": {
    "n": 48,
    "mean": -1.8258601465383945e-06,
    "median": -1.4059872930667033e-06,
    "p95": 2.0352302789206074e-06
  },
  "('kv', 'ALL')": {
    "n": 768,
    "mean": -2.6877054322068708e-06,
    "median": -2.205334172660716e-07,
    "p95": 1.2930215458796712e-05
  },
  "('kv', 'earlier_history')": {
    "n": 720,
    "mean": -2.8273521831855135e-06,
    "median": -2.468679757125106e-07,
    "p95": 1.4481867260168856e-05
  },
  "('kv', 'first_token_page')": {
    "n": 48,
    "mean": -5.930041675272251e-07,
    "median": -3.8772820687427146e-08,
    "p95": 3.1378454090713605e-06
  },
  "('v', 'ALL')": {
    "n": 768,
    "mean": -2.3684029968629355e-06,
    "median": -6.831305898101298e-08,
    "p95": 1.0906382563409988e-05
  },
  "('v', 'earlier_history')": {
    "n": 720,
    "mean": -2.4485326545200338e-06,
    "median": -4.257108815728488e-08,
    "p95": 1.1309215168304421e-05
  },
  "('v', 'first_token_page')": {
    "n": 48,
    "mean": -1.1664581320064691e-06,
    "median": -2.2650280442348958e-07,
    "p95": 2.7248917450606306e-06
  }
}
```
