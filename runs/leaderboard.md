# Leaderboard

Evaluated patches: **40**. Passing: **20**. Updated 2026-11-23 16:20:41.

> **Honesty note.** "Pass" means a patch beat these specific controls (untrained model, shuffled labels, J-lens-only delta R, held-out task, random subspace) by the stated thresholds on this micro dataset with this tiny model. It is not evidence about real LLMs, and with many patches tried, some passes will be selection noise. Re-run with new seeds before believing any single row.

## Top 10

| # | key | pass | obj | readable | persist | causal | verbal | patch |
|--:|---|:-:|--:|--:|--:|--:|--:|---|
| 1 | `97da54bc1b` | yes | 1.240 | 0.82 | 1.00 | 0.46 | 0.74 | locus=raw instrument=jacobian phi=linear rank=16 granularity=layer layer=-1 tau_coord=True modulation=False seed=0 |
| 2 | `a24216b5ca` | yes | 1.240 | 0.82 | 1.00 | 0.46 | 0.74 | locus=raw instrument=jacobian phi=linear rank=16 granularity=layer layer=1 tau_coord=True modulation=False seed=0 |
| 3 | `bf15e9571b` | yes | 1.237 | 0.82 | 1.00 | 0.46 | 0.74 | locus=raw instrument=jacobian phi=linear rank=16 granularity=layer layer=-1 tau_coord=True modulation=False seed=1 |
| 4 | `4d4a10ee37` | yes | 1.237 | 0.80 | 1.00 | 0.46 | 0.72 | locus=raw instrument=jacobian phi=linear rank=16 granularity=layer layer=-1 tau_coord=False modulation=False seed=0 |
| 5 | `a682ba3e74` | yes | 1.236 | 0.82 | 1.00 | 0.46 | 0.73 | locus=raw instrument=jacobian phi=linear rank=16 granularity=layer layer=-1 tau_coord=True modulation=False seed=2 |
| 6 | `733dd07fe5` | yes | 1.208 | 0.83 | 1.00 | 0.68 | 0.74 | locus=joint instrument=jacobian phi=linear rank=16 granularity=layer layer=-1 tau_coord=True modulation=False seed=0 |
| 7 | `245989625b` | yes | 1.203 | 0.83 | 1.00 | 0.68 | 0.76 | locus=joint instrument=jacobian phi=linear rank=16 granularity=layer layer=-1 tau_coord=True modulation=True seed=2 |
| 8 | `18c68271a0` | yes | 1.197 | 0.84 | 1.00 | 0.46 | 0.84 | locus=raw instrument=jacobian phi=lowrank rank=16 granularity=layer layer=-1 tau_coord=True modulation=False seed=0 |
| 9 | `cab26cd0b5` | yes | 1.197 | 0.82 | 1.00 | 0.44 | 0.74 | locus=raw instrument=jacobian phi=linear rank=16 granularity=model layer=-1 tau_coord=True modulation=True seed=0 |
| 10 | `ee747b9547` | yes | 1.196 | 0.84 | 1.00 | 0.76 | 0.76 | locus=joint instrument=jacobian phi=linear rank=32 granularity=model layer=-1 tau_coord=True modulation=False seed=0 |

## Per-knob marginal means

Mean objective (100 per pass + clipped margin sum) over all evaluated patches with that knob value, and pass rate. Marginals are confounded by which patches the climb chose to visit.

| knob | value | n | mean obj | pass rate |
|---|---|--:|--:|--:|
| locus | raw | 13 | 62.55 | 0.62 |
| locus | readout | 1 | 0.80 | 0.00 |
| locus | gates | 4 | 0.26 | 0.00 |
| locus | joint | 22 | 55.47 | 0.55 |
| instrument | sae | 7 | 0.74 | 0.00 |
| instrument | jacobian | 25 | 73.01 | 0.72 |
| instrument | covariance | 1 | 0.64 | 0.00 |
| instrument | probe | 2 | 50.74 | 0.50 |
| instrument | gramian | 5 | 20.56 | 0.20 |
| phi | linear | 32 | 54.04 | 0.53 |
| phi | lowrank | 6 | 50.86 | 0.50 |
| phi | mlp | 2 | 0.40 | 0.00 |
| rank | 4 | 6 | 17.32 | 0.17 |
| rank | 8 | 1 | 0.42 | 0.00 |
| rank | 16 | 26 | 62.50 | 0.62 |
| rank | 32 | 7 | 43.72 | 0.43 |
| granularity | head | 4 | 25.78 | 0.25 |
| granularity | layer | 25 | 52.88 | 0.52 |
| granularity | model | 11 | 55.48 | 0.55 |
| layer | 0 | 5 | 0.47 | 0.00 |
| layer | 1 | 6 | 34.13 | 0.33 |
| layer | -1 | 29 | 63.05 | 0.62 |
| tau_coord | False | 14 | 50.86 | 0.50 |
| tau_coord | True | 26 | 50.90 | 0.50 |
| modulation | False | 33 | 52.41 | 0.52 |
| modulation | True | 7 | 43.70 | 0.43 |
| seed | 0 | 30 | 50.88 | 0.50 |
| seed | 1 | 5 | 40.85 | 0.40 |
| seed | 2 | 5 | 60.94 | 0.60 |

## Controls summary

Margin = value minus threshold-adjusted control (positive clears). Mean over all patches, and fraction clearing.

| margin | mean | best | fraction >= 0 |
|---|--:|--:|--:|
| causal_vs_random | 0.172 | 0.720 | 0.50 |
| delta_R | 0.053 | 0.115 | 0.82 |
| heldout_ratio | 0.190 | 0.261 | 1.00 |
| readable_vs_shuffled | 0.448 | 0.534 | 1.00 |
| readable_vs_untrained | 0.276 | 0.382 | 0.97 |

