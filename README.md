# s-space

**A working hypothesis:** selective state-space models such as Mamba have an *S-space*, an interpretable and steerable subspace of the recurrent state. It would play the role for Mamba that *J-space* plays for a transformer's residual stream.

This repo is a place to work that idea out: name it, state its tenets, and build a way to measure it.

> **Where it stands (2026-11-24): not found yet.** On a toy 32k-parameter Mamba-2, an automated search found a rank-32 subspace of the recurrent state, located with Jacobians. It is readable and causal, and it survives seed and model replication. But it adds only a little over the residual-stream J-lens (delta R 0.07–0.13). The S-space-specific ideas didn't show up: the Gramian basis, decay timescale and gating modulation had no effect. So far this is evidence that the model uses part of its state, not evidence for a distinct workspace. Details are under [First overnight run](#first-overnight-run-toy-model-only), and the next step is under [Next](#next).

---

## Background

### J-space (transformers)

Gurnee et al. (2026), [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html) ([arXiv:2607.15495](https://arxiv.org/abs/2607.15495)), introduce the **Jacobian lens (J-lens)**. The J-lens finds representations in the residual stream that the model is poised to verbalize. Unlike the logit lens, it corrects for how coordinates change across layers. The span of those directions is **J-space**. The paper reports that J-space:

- carries coherent content only in an intermediate band of layers,
- holds on the order of tens of concepts at a time,
- is broadcast more widely by the weights than other representations,
- is causal: swapping J-space content changes what the model reports.

### Attention as a non-Markov process

A single softmax attention step resembles a Markov transition: its rows are non-negative and sum to 1, and it updates a state by weighted mixing. Over a whole sequence, though, attention is built to break the Markov property. It reads the full KV cache directly, and its transition weights are computed from context rather than fixed.

### Mamba as a Markov process

SSMs bring the Markov constraint back. All of the past is folded into a fixed-size state:

```
h_t = Ā_t · h_{t-1} + B̄_t · x_t        y_t = C_t · h_t
```

In Mamba, `Ā_t`, `B̄_t` and `C_t` are *selective*: they depend on the current input. The model therefore decides per token what to keep, what to write, and what to expose. Given the inputs, the update is linear in `h`, and `h_t` is a sufficient statistic of the prefix.

### Hybrids

Hybrid models interleave Mamba-2 layers with a small number of attention layers. NVIDIA's Nemotron 3 Ultra is one, and Jamba is another. The Mamba layers act as cheap working memory that runs forward with constant memory. The attention layers provide non-Markov deep recall of exact details. Such a model would have both a J-space (in the residual stream) and a candidate S-space (in the recurrent states). That makes it the natural place to study how the two hand content to each other.

---

## Prior work

| Work | What it does | Relation to S-space |
|---|---|---|
| Wang & Reid (2026), [*Residual Streams Read, Recurrent States Remember: The Global Workspace in Mamba Models*](https://arxiv.org/abs/2609.32102) | Fits Jacobian lenses to both the residual stream and the recurrent state of Mamba-1, -2 and -3. Reports that on Mamba-2 the **state lens** often beats both the residual J-lens and the logit lens at recovering intermediate concepts, and that joint readouts do better still. Earlier content stays readable in the state even as its residual visibility changes. | Closest prior work. It establishes J-space-style content in SSM state. It calls the tool a "state lens" and does not name the space. |
| Mohan et al. (2026), [*Interpreting and Steering State-Space Models via Activation Subspace Bottlenecks*](https://arxiv.org/abs/2602.22719) (ICML 2026) | Finds a bottleneck subspace (tied to `dt_proj.bias` in Mamba-130M). Scaling it at test time improves results, with an average gain of about 8% across 5 SSMs and 6 benchmarks. | Shows that subspaces of the state machinery can be steered. |
| Ali et al. (2024), [*The Hidden Attention of Mamba Models*](https://arxiv.org/abs/2403.01590) | Unrolls selective SSMs into implicit attention matrices. | Gives a way to attribute state content back to tokens. |
| *Superposition and Polysemanticity in Mamba's Selective State Space* (SSRN) | Reported to train SAEs on Mamba state channels and to find a geometry different from the transformer residual stream. | **Unverified.** I could not locate this paper. |

As far as I know, nobody uses **"S-space"** as a name, or frames the state as a spectral/centrality object.

---

## Proposed tenets of S-space

**Definition.** S-space is the interpretable, steering-accessible subspace of a selective SSM's recurrent state `h_t`.

1. **Decay stratification.** The state is ordered by half-life, which `Δ` and `A` control. Fast-decaying modes carry local syntax. Slow-decaying modes carry semantic anchors such as task, persona and entities.
2. **Gated write/overwrite.** Content does not add up freely the way it does in a residual stream. It competes for a fixed state capacity, and enters only when the selective gates (`Δ_t`, `B_t`) admit it.
3. **Channel-wise resonators.** Each state channel or head acts as a resonator tuned to a structural rhythm, such as bracket depth, entity tracking or topic drift. Superposition happens *within* this bank of channels rather than across one flat vector.
4. **Markovian sufficiency.** Two prompts that reach the same point in S-space produce the same future. S-space is therefore a true phase space for generation.

---

## Sketch: a spectral S-space fingerprint

The aim is to compress the selective state into a low-dimensional spectral signature that can be compared with a J-space readout.

1. **Collect selective metadata** per layer and step: `Ā_t` (retain/forget), `B̄_t` (write) and `C_t` (read).
2. **Build the cumulative transition** over a window of `k` steps: `Φ_{t,k} = Ā_t Ā_{t-1} … Ā_{t-k+1}`.
3. **Eigen-reduce.** Dominant modes are the persistent attractors. Sub-dominant modes are transient.
4. **Read out.** Project the top-`r` modes through `C_t` to get a manifold fingerprint. Compare it with the J-lens readout to catch the moment when state content is *promoted* into the workspace.

### ⚠️ Known problem with the naive version

In Mamba-1, `Ā_t` is **diagonal**. In Mamba-2 it is a **scalar times the identity** per head. That makes the eigenvectors of `Φ` the coordinate axes (Mamba-1), or makes every vector an eigenvector (Mamba-2). The eigen-reduction of `Φ` alone is therefore trivial. The decay spectrum is useful, but the structure has to come from somewhere else:

- **Controllability/observability Gramians**, `W_c = Σ Φ B̄ B̄ᵀ Φᵀ` and `W_o = Σ Φᵀ Cᵀ C Φ`, or their balanced-truncation product. These mix the input and output maps with the decay and produce non-trivial principal directions.
- **Empirical state covariance** over a corpus, weighted by decay.
- **The state Jacobian** from Wang & Reid. Here S-space would be the J-lens fitted on `h`, with the spectral work coming afterwards.
- **Mamba-3** uses complex/rotary dynamics. Its modes come as 2-D rotating blocks, which carry phase as well as decay.

Choosing among these is the first real research question.

---

## Open questions

- Is S-space a subspace, or a manifold that depends on the input?
- Does the state lens's advantage over the residual J-lens on Mamba-2 hold up in hybrids?
- In a hybrid, can we watch content move from S-space to J-space around the attention layers?
- How does steering along S-space modes compare with steering J-space directions, and with bottleneck scaling?
- Recovering content, keeping it over time, and redirecting the model are related but distinct (Wang & Reid). Which one should define S-space?

## Autoresearch harness

S-space is not defined by fiat here. Each open decision above (which part of the state, which instrument, which readout, which layer, ...) becomes a **knob**. A **patch** is one setting of all the knobs, like a patch on a modular synth. An automated search ("autoresearch") tries patches and keeps only those that beat strict controls.

**Knobs** (see `sspace/patch.py`): `locus` (raw state, readout, gates, joint), `instrument` (SAE, Jacobian, covariance, probe, Gramian), `phi` (linear, low-rank, MLP readout), `rank`, `granularity` (head, layer, model), `layer`, `tau_coord`, `modulation`, `seed`.

**Four scores**, all on held-out test data from a micro synthetic dataset:

- **readable**: a readout from the S-space features predicts the per-position label (macro-average over 5 tasks).
- **persistent**: the same, for the recall task at positions well after the key was written.
- **causal**: swapping the S-space component of the state between two examples flips the model's answer.
- **verbal**: the lens's readout agrees with what the model actually predicts.

**Controls** (the falsifier). A patch passes only if it clears every one:

1. untrained model: same patch on a randomly initialised model (readable +0.10)
2. shuffled labels: same patch with training labels permuted within each task (readable +0.10)
3. J-lens-only: same readout fitted on the residual stream alone; the joint state+residual readout must add at least 0.02 (delta R)
4. held-out task: basis fitted on 4 tasks, read out on the 5th (ratio >= 0.8)
5. random subspace (for causal): a random subspace of equal dimension must be beaten by +0.10

The loop does a hill climb (1-2 knob mutations of the best patch, plus random restarts), logs every evaluation to `runs/ledger.jsonl`, and rewrites `runs/leaderboard.md` each iteration. Passing means "beats these controls on this micro dataset", nothing more.

**Layout**

- `sspace/`: `model.py` and `train.py` (tiny Mamba-2), `patch.py` and `eval.py` (patches and scoring), `loop.py` (search), `data.py` (loader), `CONTRACT.md` (interfaces)
- `patches/`: patch settings as TOML (`default.toml`)
- `runs/`: checkpoints, `train_metrics.json`, `eval_*.json`, `ledger.jsonl`, `leaderboard.md` (checkpoints and ledger are git-ignored)
- `data/`: micro dataset and generator (`data/README.md`)
- `explorer/`: local page for working through the design decisions

**Quickstart**

```
make data       # generate the micro dataset
make train      # train tiny model + untrained twin
make eval       # score patches/default.toml with all controls
make loop       # autoresearch for 30 minutes
make fake-loop  # test the loop on a synthetic landscape (writes *.fake.* only)
make explorer   # decision explorer
```

## First overnight run (toy model only)

`make overnight` runs `sspace/overnight.py`. It searches the patch knobs and escalates each candidate:

- **L1**: passes every control on the base model.
- **L2**: also passes with fitting seeds 1 and 2.
- **L3 ("robust")**: also passes on two independently trained models.

The bar is stricter than the `make loop` defaults above, and it ratchets upward whenever robust patches become common:

| control | start | after round 1 |
|---|--:|--:|
| readable vs untrained | 0.20 | 0.23 |
| readable vs shuffled | 0.30 | 0.33 |
| delta R | 0.05 | 0.07 |
| causal vs random | 0.25 | 0.30 |
| held-out ratio | 0.90 | 0.915 |

Night of 2026-11-23, 32k-parameter Mamba-2, micro dataset, CPU: **3,101 candidates and 3,523 evaluations with 0 errors. 43 are robust**, roughly 65% of the non-redundant patch space. See `runs/overnight/STATUS.md` and `journal.md`.

What it suggests, as hypotheses for real models:

1. **Readability is cheap, causality is not.** Every instrument reaches about 0.85 readability. The SAE, covariance and Gramian bases never beat a norm-matched random subspace on causal patching, across 1,847 tries. 41 of the 43 robust patches use the **Jacobian** instrument and the other 2 use a probe.
2. **The S-space is whole-model or last-layer, and it is not small.** No head-level patch and no layer-0 patch passed. Rank 32 does better than rank 16, and ranks 4 and 8 never passed.
3. **The added value over the J-lens is real but thin.** Delta R, the gain from reading the state on top of the residual stream, is the weakest control for most leaders. Its worst case across the 5 replicates runs from 0.074 to 0.125, against a bar of 0.07. The leader is `locus=raw instrument=jacobian phi=lowrank rank=32 granularity=model`, with readable 0.86, persistent 1.00, causal 0.64–0.92 against a random 0.00, and delta R 0.078–0.127.
4. **τ-as-coordinate and modulation are null knobs here.** Robust rates are the same with them on or off. That could mean the tasks never exercise timescale or gating structure, so it does not count as evidence against the "modular synth" view.
5. **The Gramian, the instrument the theory favours, failed on causality.** It is the clearest tension with the spectral sketch above, worth checking at a scale where Ā is not close to diagonal.
6. **The search saturated early.** It found no new leader after 00:30, and rounds 3–7 added only 10 robust patches. The next step is harder tasks or a real model, not more search.

**Verdict: not S-space yet.** The model does have a readable, causal subspace in its state. But on a 2-layer model that solves every task perfectly, "a rank-32 causal subspace of the state" may just be the part of the state the model uses at all. The defining claim is that the state holds content the residual stream does not, and that claim gets only weak support (finding 3). The one instrument derived from the theory, the Gramian, failed (finding 5).

## Next

The question that settles it: **does delta R grow with scale and with the length of the gap the model must remember across?**

- If delta R stays near 0.08, S-space is mostly the J-lens seen from a different angle.
- If it grows, and the Gramian or decay-timescale knobs start to matter, there is a distinct S-space.

To get there:

1. **Real checkpoint.** Run the harness on a pretrained Mamba-2 (130M runs on CPU), with `state_hook` and `return_cache` adapted to its blocks.
2. **Tasks the residual stream can't shortcut.** Use long gaps and multiple pieces of interfering information, so the answer has to live in `h`. Report delta R as a function of the gap length.
3. **Retest the Gramian** where Ā is not close to diagonal, using Mamba-3's rotating dynamics, or using a trained model whose decay rates are spread out.
4. **Hybrid.** Look for content moving from S-space into J-space around the attention layers.

## Status

The repo has a micro synthetic dataset, a tiny Mamba-2 (32k parameters, test accuracy 1.0 on all tasks, plus two replicate seeds), the patch/eval code with controls, a search loop, and an overnight escalating search with replication. The `--fake` loop mode runs on a synthetic landscape purely to test the machinery. There are **no results on real LLMs**, and the findings above hold only for this toy setting.
