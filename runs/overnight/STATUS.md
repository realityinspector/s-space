# Overnight autoresearch: status

Updated 2026-11-24T08:48:06 · round 7 · started 2026-11-23T23:42:25 · deadline 2026-11-24 07:00

**3101 candidates**, 3523 evaluations, 0 errors. Levels under the *current* thresholds: L0 2997 · L1 10 · L2 51 · **L3 robust 43**.

> L1 = passes on the base model · L2 = also passes with fitting seeds 1 and 2 · L3 = also passes on 2 independently trained models. Thresholds only rise. This is a 32k-parameter model on synthetic micro tasks, so a robust patch is a hypothesis to test on real models, not a finding about them.

## Current thresholds

| control | start | now |
|---|--:|--:|
| readable_vs_untrained | 0.200 | **0.230** |
| readable_vs_shuffled | 0.300 | **0.330** |
| delta_R | 0.050 | **0.070** |
| causal_vs_random | 0.250 | **0.300** |
| heldout_ratio | 0.900 | **0.915** |

## Top candidates

| # | level | min-margin sum | readable | persist | causal | verbal | weakest control | patch |
|--:|:-:|--:|--:|--:|--:|--:|---|---|
| 1 | L3 | +0.837 | 0.86 | 1.00 | 0.72 | 0.82 | delta_R +0.008 | locus=raw instrument=jacobian phi=lowrank rank=32 granularity=model layer=-1 tau_coord=False modulation=False |
| 2 | L3 | +0.834 | 0.86 | 1.00 | 0.72 | 0.83 | delta_R +0.011 | locus=raw instrument=jacobian phi=lowrank rank=32 granularity=model layer=-1 tau_coord=False modulation=True |
| 3 | L3 | +0.833 | 0.86 | 1.00 | 0.82 | 0.82 | delta_R +0.011 | locus=joint instrument=jacobian phi=lowrank rank=32 granularity=model layer=-1 tau_coord=False modulation=True |
| 4 | L3 | +0.832 | 0.86 | 1.00 | 0.72 | 0.85 | delta_R +0.006 | locus=raw instrument=jacobian phi=lowrank rank=32 granularity=model layer=-1 tau_coord=True modulation=False |
| 5 | L3 | +0.830 | 0.86 | 1.00 | 0.82 | 0.84 | delta_R +0.006 | locus=joint instrument=jacobian phi=lowrank rank=32 granularity=model layer=-1 tau_coord=True modulation=False |
| 6 | L3 | +0.829 | 0.87 | 1.00 | 0.82 | 0.84 | delta_R +0.004 | locus=joint instrument=jacobian phi=lowrank rank=32 granularity=model layer=-1 tau_coord=True modulation=True |
| 7 | L3 | +0.828 | 0.84 | 1.00 | 0.72 | 0.74 | delta_R +0.045 | locus=raw instrument=jacobian phi=linear rank=32 granularity=model layer=-1 tau_coord=False modulation=False |
| 8 | L3 | +0.827 | 0.86 | 1.00 | 0.72 | 0.83 | delta_R +0.004 | locus=raw instrument=jacobian phi=lowrank rank=32 granularity=model layer=-1 tau_coord=True modulation=True |
| 9 | L3 | +0.820 | 0.84 | 1.00 | 0.72 | 0.76 | readable_vs_untrained +0.049 | locus=raw instrument=jacobian phi=linear rank=32 granularity=model layer=-1 tau_coord=False modulation=True |
| 10 | L3 | +0.818 | 0.86 | 1.00 | 0.82 | 0.82 | delta_R +0.008 | locus=joint instrument=jacobian phi=lowrank rank=32 granularity=model layer=-1 tau_coord=False modulation=False |
| 11 | L3 | +0.814 | 0.84 | 1.00 | 0.72 | 0.76 | readable_vs_untrained +0.044 | locus=raw instrument=jacobian phi=linear rank=32 granularity=model layer=-1 tau_coord=True modulation=True |
| 12 | L3 | +0.813 | 0.84 | 1.00 | 0.72 | 0.77 | readable_vs_untrained +0.050 | locus=raw instrument=jacobian phi=linear rank=32 granularity=model layer=-1 tau_coord=True modulation=False |
| 13 | L3 | +0.812 | 0.83 | 1.00 | 0.82 | 0.74 | delta_R +0.045 | locus=joint instrument=jacobian phi=linear rank=32 granularity=model layer=-1 tau_coord=False modulation=False |
| 14 | L3 | +0.810 | 0.83 | 1.00 | 0.82 | 0.76 | delta_R +0.049 | locus=joint instrument=jacobian phi=linear rank=32 granularity=model layer=-1 tau_coord=False modulation=True |
| 15 | L3 | +0.798 | 0.84 | 1.00 | 0.82 | 0.76 | readable_vs_untrained +0.049 | locus=joint instrument=jacobian phi=linear rank=32 granularity=model layer=-1 tau_coord=True modulation=False |

## What robust patches have in common

Share of evaluated candidates with each knob value that reached L3. Read it as a tendency: the search visits values unevenly.

| knob | value | tried | robust | rate |
|---|---|--:|--:|--:|
| granularity | model | 836 | 25 | 0.03 |
| granularity | layer | 1146 | 18 | 0.02 |
| granularity | head | 1119 | 0 | 0.00 |
| instrument | jacobian | 656 | 41 | 0.06 |
| instrument | probe | 598 | 2 | 0.00 |
| instrument | sae | 606 | 0 | 0.00 |
| instrument | covariance | 616 | 0 | 0.00 |
| instrument | gramian | 625 | 0 | 0.00 |
| layer | -1 | 2209 | 43 | 0.02 |
| layer | 0 | 892 | 0 | 0.00 |
| locus | raw | 793 | 22 | 0.03 |
| locus | joint | 777 | 21 | 0.03 |
| locus | readout | 753 | 0 | 0.00 |
| locus | gates | 778 | 0 | 0.00 |
| modulation | False | 1541 | 23 | 0.01 |
| modulation | True | 1560 | 20 | 0.01 |
| phi | linear | 1028 | 28 | 0.03 |
| phi | lowrank | 1029 | 15 | 0.01 |
| phi | mlp | 1044 | 0 | 0.00 |
| rank | 32 | 772 | 30 | 0.04 |
| rank | 16 | 776 | 13 | 0.02 |
| rank | 4 | 758 | 0 | 0.00 |
| rank | 8 | 795 | 0 | 0.00 |
| tau_coord | False | 1571 | 23 | 0.01 |
| tau_coord | True | 1530 | 20 | 0.01 |

## Threshold history

- round 0 (2026-11-23T23:42:25): readable_vs_untrained 0.200, readable_vs_shuffled 0.300, delta_R 0.050, causal_vs_random 0.250, heldout_ratio 0.900
- round 1 (2026-11-24T00:22:51): readable_vs_untrained 0.230, readable_vs_shuffled 0.330, delta_R 0.070, causal_vs_random 0.300, heldout_ratio 0.915

## Notes

- 2026-11-23T23:46:35 round 1: new robust best 92087cd2a6 (locus=raw instrument=jacobian phi=lowrank rank=16 granularity=model layer=-1 tau_coord=False modulation=True)
- 2026-11-23T23:52:52 round 1: new robust best 08281e7648 (locus=raw instrument=jacobian phi=lowrank rank=16 granularity=model layer=-1 tau_coord=False modulation=False)
- 2026-11-23T23:54:07 round 1: new robust best 4614ce2c3e (locus=joint instrument=jacobian phi=lowrank rank=16 granularity=model layer=-1 tau_coord=False modulation=False)
- 2026-11-24T00:22:51 round 1: ratchet → {'readable_vs_untrained': 0.23, 'readable_vs_shuffled': 0.33, 'delta_R': 0.07, 'causal_vs_random': 0.3, 'heldout_ratio': 0.915}
- 2026-11-24T00:26:41 round 2: new robust best 141f164970 (locus=raw instrument=jacobian phi=lowrank rank=32 granularity=model layer=-1 tau_coord=False modulation=True)
- 2026-11-24T00:29:51 round 2: new robust best e0aa01c7ea (locus=raw instrument=jacobian phi=lowrank rank=32 granularity=model layer=-1 tau_coord=False modulation=False)
