# Representative sampling / protection coverage — NON-RETRIEVAL long context v1

NON-RETRIEVAL diary/narrative templates with buried keywords (Maya/Jonah), longer shared context (~12-20 L_p pages). Still synthetic — not a naturalistic load sample. Purpose: check whether protection-curve concentration holds off the Archive Alpha/Beta retrieval family used in rep_sample_v1.

- elapsed_s: 3909.7
- n_traj: 15180, n_curves: 480
- gates: {"hash_equality_rate": 1.0, "hash_checks": 7590, "hash_fail": 0, "zero_pert_pass_rate": 1.0, "zero_pert_checks": 64, "zero_pert_fail": 0}
- heads: [(10, 3), (2, 1), (22, 2), (22, 4), (24, 6)]

## Pooled protection curve (mean over prefix×head curves)

### intervene=k
- mean coverage @ top frac pages: {"0.05": 0.3363207046604251, "0.1": 0.5698707912033909, "0.2": 0.7804530575374716, "0.25": 0.8276243997360091, "0.5": 0.977825632877998, "1.0": 1.0}
- mean frac pages to cover mass: {"0.5": 0.1533345734126985, "0.8": 0.2720994543650793, "0.95": 0.3951599702380951}

### intervene=v
- mean coverage @ top frac pages: {"0.05": 0.3375687652248941, "0.1": 0.5526335374120069, "0.2": 0.7731299016108151, "0.25": 0.8206766560148038, "0.5": 0.976548847960278, "1.0": 1.0}
- mean frac pages to cover mass: {"0.5": 0.15386532738095243, "0.8": 0.2749144345238095, "0.95": 0.40905753968253933}

### intervene=kv
- mean coverage @ top frac pages: {"0.05": 0.348900629240391, "0.1": 0.5717118240013591, "0.2": 0.7795607045234824, "0.25": 0.8269866950070057, "0.5": 0.9747817920744458, "1.0": 1.0}
- mean frac pages to cover mass: {"0.5": 0.1537413194444444, "0.8": 0.27171130952380956, "0.95": 0.3997470238095236}

## H by bucket
{
  "earlier_history|k": {
    "n": 2050,
    "mean": 0.002207317993634605,
    "median": 3.45300859407871e-05,
    "p95": 0.01461045970208943
  },
  "earlier_history|kv": {
    "n": 2050,
    "mean": 0.002131848792956871,
    "median": 2.9630024982907344e-05,
    "p95": 0.014637098694220186
  },
  "earlier_history|v": {
    "n": 2050,
    "mean": 0.0022921024813693175,
    "median": 3.510841603808027e-05,
    "p95": 0.017387456737924366
  },
  "first_token_page|k": {
    "n": 160,
    "mean": 0.0013613045886620042,
    "median": 9.863010745903011e-06,
    "p95": 0.005356679202122905
  },
  "first_token_page|kv": {
    "n": 160,
    "mean": 0.001723434978931048,
    "median": 1.9091451804342796e-05,
    "p95": 0.012684558589656567
  },
  "first_token_page|v": {
    "n": 160,
    "mean": 0.0014257469854753665,
    "median": 1.4298026144388132e-05,
    "p95": 0.013003651984035969
  },
  "recent_closed|k": {
    "n": 320,
    "mean": 0.00240584438801682,
    "median": 3.0795523343840614e-05,
    "p95": 0.017802383363596165
  },
  "recent_closed|kv": {
    "n": 320,
    "mean": 0.0023903492488874765,
    "median": 2.0388057843471568e-05,
    "p95": 0.017803642184298953
  },
  "recent_closed|v": {
    "n": 320,
    "mean": 0.0022115083051140157,
    "median": 2.0407680040079867e-05,
    "p95": 0.016359361659018152
  }
}

## H by page kind (fact_A/B vs other)
{
  "fact_A|k": {
    "n": 160,
    "mean": 0.0017401511216939742,
    "median": 2.3432134184986353e-05,
    "p95": 0.008117752615362406
  },
  "fact_A|kv": {
    "n": 160,
    "mean": 0.001867376258167563,
    "median": 3.168524199281819e-05,
    "p95": 0.012736168960202576
  },
  "fact_A|v": {
    "n": 160,
    "mean": 0.002265631300367682,
    "median": 8.825146755953028e-05,
    "p95": 0.01514390781289876
  },
  "fact_B|k": {
    "n": 160,
    "mean": 0.0014297406863824588,
    "median": 2.0992144709452986e-05,
    "p95": 0.006354158214526251
  },
  "fact_B|kv": {
    "n": 160,
    "mean": 0.0018647144545225797,
    "median": 1.5091403232266506e-05,
    "p95": 0.016516197283635847
  },
  "fact_B|v": {
    "n": 160,
    "mean": 0.0020707197754838086,
    "median": 2.0314596440584864e-05,
    "p95": 0.017571048714671633
  },
  "intro_or_first|k": {
    "n": 110,
    "mean": 0.0005937830672081127,
    "median": 1.660723796703678e-05,
    "p95": 0.0012895648579274166
  },
  "intro_or_first|kv": {
    "n": 110,
    "mean": 0.000723997026288447,
    "median": 1.9041580628709198e-05,
    "p95": 0.0014720460436971432
  },
  "intro_or_first|v": {
    "n": 110,
    "mean": 0.00033188674128182644,
    "median": 1.533890872451593e-05,
    "p95": 0.0010953987948596478
  },
  "trailing_or_mid|k": {
    "n": 2100,
    "mean": 0.0023524676183891043,
    "median": 3.476075562502956e-05,
    "p95": 0.016745028544998302
  },
  "trailing_or_mid|kv": {
    "n": 2100,
    "mean": 0.0022543700928154262,
    "median": 2.9630024982907344e-05,
    "p95": 0.015345771185820922
  },
  "trailing_or_mid|v": {
    "n": 2100,
    "mean": 0.0023353754992582877,
    "median": 3.289207643319969e-05,
    "p95": 0.017387456737924366
  }
}
