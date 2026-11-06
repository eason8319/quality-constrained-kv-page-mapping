# Representative sampling / protection coverage — NON-RETRIEVAL long context v1

NON-RETRIEVAL diary/narrative templates with buried keywords (Maya/Jonah), longer shared context (~12-20 L_p pages). Still synthetic — not a naturalistic load sample. Purpose: check whether protection-curve concentration holds off the Archive Alpha/Beta retrieval family used in rep_sample_v1.

- elapsed_s: 4952.5
- n_traj: 15180, n_curves: 480
- gates: {"hash_equality_rate": 1.0, "hash_checks": 7590, "hash_fail": 0, "zero_pert_pass_rate": 1.0, "zero_pert_checks": 64, "zero_pert_fail": 0}
- heads: [(10, 3), (2, 1), (22, 2), (22, 4), (24, 6)]

## Pooled protection curve (mean over prefix×head curves)

### intervene=k
- mean coverage @ top frac pages: {"0.05": 0.36239447015038884, "0.1": 0.5847169715558193, "0.2": 0.7875256542195771, "0.25": 0.8394148869027342, "0.5": 0.9799194805235187, "1.0": 1.0}
- mean frac pages to cover mass: {"0.5": 0.14691220238095243, "0.8": 0.26291914682539674, "0.95": 0.39044394841269836}

### intervene=v
- mean coverage @ top frac pages: {"0.05": 0.36252016693355893, "0.1": 0.5866064356213635, "0.2": 0.788579862269325, "0.25": 0.8410579599152077, "0.5": 0.98133360344464, "1.0": 1.0}
- mean frac pages to cover mass: {"0.5": 0.15002728174603175, "0.8": 0.2603410218253968, "0.95": 0.3869655257936508}

### intervene=kv
- mean coverage @ top frac pages: {"0.05": 0.37171992673246873, "0.1": 0.5978886100119831, "0.2": 0.7899212052433091, "0.25": 0.8386378557057725, "0.5": 0.9794936362972002, "1.0": 1.0}
- mean frac pages to cover mass: {"0.5": 0.14342757936507947, "0.8": 0.2589347718253968, "0.95": 0.3939608134920636}

## H by bucket
{
  "earlier_history|k": {
    "n": 2050,
    "mean": 0.002468339892233497,
    "median": 3.4798060141838505e-05,
    "p95": 0.015880574902985244
  },
  "earlier_history|kv": {
    "n": 2050,
    "mean": 0.0026408146178025426,
    "median": 3.255659066780936e-05,
    "p95": 0.0157386374767763
  },
  "earlier_history|v": {
    "n": 2050,
    "mean": 0.0020375005710715293,
    "median": 2.402774531446994e-05,
    "p95": 0.013075419734377647
  },
  "first_token_page|k": {
    "n": 160,
    "mean": 0.0019948437732202487,
    "median": 4.157396699611127e-05,
    "p95": 0.013536510634003206
  },
  "first_token_page|kv": {
    "n": 160,
    "mean": 0.0020560229259335623,
    "median": 3.391075491284331e-05,
    "p95": 0.011249406833667309
  },
  "first_token_page|v": {
    "n": 160,
    "mean": 0.0018994409126143724,
    "median": 1.9470434835966444e-05,
    "p95": 0.008863909957426586
  },
  "recent_closed|k": {
    "n": 320,
    "mean": 0.0018913205428973613,
    "median": 1.4030535203346517e-05,
    "p95": 0.014729551749769598
  },
  "recent_closed|kv": {
    "n": 320,
    "mean": 0.0026814514597247787,
    "median": 2.489011239958927e-05,
    "p95": 0.02097279576701112
  },
  "recent_closed|v": {
    "n": 320,
    "mean": 0.0024177418706999043,
    "median": 1.7865328845800833e-05,
    "p95": 0.016637772514892884
  }
}

## H by page kind (fact_A/B vs other)
{
  "fact_A|k": {
    "n": 160,
    "mean": 0.002485402008834588,
    "median": 3.711721692525316e-05,
    "p95": 0.01861209609778598
  },
  "fact_A|kv": {
    "n": 160,
    "mean": 0.0027803843237205455,
    "median": 0.00013002947562199552,
    "p95": 0.016109563408463146
  },
  "fact_A|v": {
    "n": 160,
    "mean": 0.0018017681066992467,
    "median": 5.6421833505737595e-05,
    "p95": 0.009163989449158786
  },
  "fact_B|k": {
    "n": 160,
    "mean": 0.002484494306708409,
    "median": 4.1918299302778905e-05,
    "p95": 0.016424862367936537
  },
  "fact_B|kv": {
    "n": 160,
    "mean": 0.0026002272386942507,
    "median": 3.616578639290916e-05,
    "p95": 0.014231015549739823
  },
  "fact_B|v": {
    "n": 160,
    "mean": 0.0014482803779944603,
    "median": 1.464151318941731e-05,
    "p95": 0.006707629177253693
  },
  "intro_or_first|k": {
    "n": 110,
    "mean": 0.0011609079227162056,
    "median": 4.747499838231306e-05,
    "p95": 0.00424767120651571
  },
  "intro_or_first|kv": {
    "n": 110,
    "mean": 0.0010588143422773663,
    "median": 3.1671857641413226e-05,
    "p95": 0.002546822056001474
  },
  "intro_or_first|v": {
    "n": 110,
    "mean": 0.000889862984368764,
    "median": 2.2551045185537076e-05,
    "p95": 0.0020808806607419683
  },
  "trailing_or_mid|k": {
    "n": 2100,
    "mean": 0.002410290987873999,
    "median": 3.083661340497201e-05,
    "p95": 0.015706933435649263
  },
  "trailing_or_mid|kv": {
    "n": 2100,
    "mean": 0.00267777651148554,
    "median": 2.9386039773271478e-05,
    "p95": 0.016847403725478216
  },
  "trailing_or_mid|v": {
    "n": 2100,
    "mean": 0.0022078911569082603,
    "median": 2.169093204429373e-05,
    "p95": 0.015179945534327999
  }
}
