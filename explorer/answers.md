# S-space · design decisions

_Updated 2026-11-23 22:45 UTC_

## 1. Locus — Which object does S-space live in?

**Choice:** _undecided_

**Synthesis:**

it should be learning like "autoresearch" these are properties we would have to test not choices I can make today.

## 2. Membership criterion — What property qualifies a direction for S-space?

**Choice:** All four

```tex
\mathcal S = \mathcal S_{\text{read}} \cap \mathcal S_{\text{persist}} \cap \mathcal S_{\text{causal}} \cap \mathcal S_{\text{verbal}}
```

## 3. Instrument — How is S-space measured?

**Choice:** _undecided_

**Synthesis:**

probably d but again "autoresearch" them persistently over time is the solution

## 4. Directionality — Does S-space look forward, backward, or both?

**Choice:** Balanced (both) _(recommended)_

```tex
T W_c T^{\top} = T^{-\top} W_o T^{-1} = \operatorname{diag}(\sigma_1,\dots,\sigma_n)
```

## 5. Shape — Fixed subspace, per-prompt subspace, or curved manifold?

**Choice:** Input-dependent manifold

```tex
\operatorname{vec}h_t \in \mathcal M = \{\varphi(z) : z\in\mathbb{R}^{r}\}
```

**Commentary:**

parameterize it

## 6. Granularity — What is the unit of an S-space?

**Choice:** Whole model

```tex
\mathcal S \subseteq \textstyle\bigoplus_{\ell,h} \mathbb{R}^{P\times N}
```

**Synthesis:**

we don't know yet  probably whole model?

## 7. Timescale — How is time scale handled?

**Choice:** τ as a coordinate

```tex
\mathcal S \subseteq \mathbb{R}^{PN}\times\mathbb{R}_{>0},\qquad (h,\ \tau)
```

**Commentary:**

hard t

## 8. Unit of content — What counts as one item in S-space?

**Choice:** _undecided_

**Synthesis:**

would think of it more like a modular synthesizer that can modulate across variables and parameterize things like modulation

## 9. Scope — Which architectures must the definition cover?

**Choice:** Staged _(recommended)_

```tex
\begin{gathered}\text{Mamba-2} \to \text{hybrids} \to \text{linear RNNs} \\ \text{Mamba-3 as phase extension}\end{gathered}
```

## 10. Falsifier — What result would prove S-space wrong?

**Choice:** _undecided_

**Synthesis:**

that's like moog asking what songs can be written with a modular circuit board - we don't know yet but we should have strong evals, probably these

## Working definition

_not yet written_
