# s-space

**A working hypothesis:** selective state-space models such as Mamba have an *S-space*, an interpretable and steerable subspace of the recurrent state. It would play the role for Mamba that *J-space* plays for a transformer's residual stream.

This repo is a place to work that idea out: name it, state its tenets, and build a way to measure it.

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

## Status

These are ideas only. There is no code yet.
