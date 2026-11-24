# sspace interface contract

These are the shared interfaces for the three parallel workstreams. **Do not change a signature here without the owner's agreement.** Each file has exactly one owner, and nobody edits another owner's files.

The context for all of this is in `README.md` (the S-space hypothesis), `data/README.md` (the micro dataset) and `explorer/answers.md`-style decisions, summarised below.

Everything runs on CPU with Python 3.14, torch 2.12 and numpy, and nothing else. Do not install packages. Read config with `tomllib`, not yaml. Run modules from the repo root as `python3 -m sspace.<module>`.

## Ownership

| owner | files |
|---|---|
| shared (already written; do not edit) | `sspace/data.py`, `sspace/CONTRACT.md`, `data/` |
| **A: model** | `sspace/model.py`, `sspace/train.py`, `runs/base.pt`, `runs/untrained.pt`, `runs/train_metrics.json` |
| **B: patch + eval** | `sspace/patch.py`, `sspace/eval.py`, `sspace/instruments.py` (optional helpers), `patches/*.toml` |
| **C: loop + docs** | `sspace/loop.py`, `sspace/__init__.py`, `runs/ledger.jsonl`, `runs/leaderboard.md`, `README.md`, `.gitignore`, `Makefile` |

## A: `sspace/model.py`

```python
@dataclass
class Config:
    vocab: int = 63; d_model: int = 64; n_layer: int = 2
    n_head: int = 4; head_dim: int = 16      # P;  d_inner = n_head * head_dim
    d_state: int = 8                          # N;  Mamba-2 style, B and C shared across heads (ngroups = 1)
    max_len: int = 32

class TinyMamba2(nn.Module):
    cfg: Config
    def forward(self, ids: LongTensor[B, T], *, state_hook=None, return_cache: bool = False
                ) -> tuple[Tensor[B, T, V], dict | None]

def load(path: str | Path) -> TinyMamba2      # eval mode, CPU
```

- **Recurrence.** Each layer runs h_t = a_t · h_{t−1} + x_t ⊗ B_t and y_t = h_t · C_t (+ D·x_t) with a_t = exp(−dt_t · exp(A_log)), so a_t ∈ (0, 1) holds one scalar per head. It runs as an explicit **sequential loop over t**, since T ≤ 32 and clarity beats speed here.
- **`state_hook(layer: int, t: int, h: Tensor[B, H, P, N]) -> Tensor[B, H, P, N]`** is called right after every state update. Whatever it returns becomes h_t and carries forward. This is how patching, steering and gradient injection happen.
- **`cache`** (when `return_cache=True`) is a dict of lists indexed by layer:
  - `"h"`: `[B, T, H, P, N]`, the post-update state
  - `"a"`: `[B, T, H]`, decay ā_t
  - `"dt"`: `[B, T, H]`
  - `"B"`: `[B, T, N]`
  - `"C"`: `[B, T, N]`
  - `"x"`: `[B, T, H, P]`, the SSM input
  - `"y"`: `[B, T, H, P]`, the SSM output before the gate
  - `"resid"`: a list of length `n_layer + 1` of `[B, T, d_model]`. Entry ℓ is the residual stream entering layer ℓ, and the last entry is after the final layer, before the final norm.
- Tensors in the cache keep their graph if the inputs require grad (needed for Jacobians).

## A: `sspace/train.py`

`python3 -m sspace.train [--steps N] [--seed S]` does the following:
1. Trains on `load_split("train")` with next-token loss on all non-pad positions, with the **answer token weighted ×5**.
2. Saves `runs/base.pt` (trained) and `runs/untrained.pt` (same init seed, no training).
3. Writes `runs/train_metrics.json`: `{"test_acc": {task: acc_at_query_pos}, "steps", "seconds", "params"}`.

Targets: about 100k parameters at most, under 5 minutes on 4 CPU threads, and test accuracy at `query_pos` of at least 0.9 on recall, count and topic (brackets and entity at ≥ 0.7 is acceptable).

## B: `sspace/patch.py`

```python
KNOBS = {   # the loop searches over exactly these
  "locus": ["raw", "readout", "gates", "joint"],
  "instrument": ["sae", "jacobian", "covariance", "probe", "gramian"],
  "phi": ["linear", "lowrank", "mlp"],
  "rank": [4, 8, 16, 32],
  "granularity": ["head", "layer", "model"],
  "layer": [0, 1, -1],          # used when granularity != "model"; -1 = last
  "tau_coord": [False, True],
  "modulation": [False, True],  # add gate-derivative (modulation-route) features
  "seed": [0, 1, 2],
}
@dataclass(frozen=True)
class Patch:  # one field per KNOBS key, defaults = first value except instrument="sae"
    def to_dict(self) -> dict; def key(self) -> str          # stable short hash
    @classmethod def from_dict(cls, d) -> "Patch"; @classmethod def from_toml(cls, path) -> "Patch"
def default() -> Patch
```

## B: `sspace/eval.py`

`evaluate(patch: Patch, *, base="runs/base.pt", untrained="runs/untrained.pt", controls=True) -> dict`, which returns JSON-serialisable data:

```python
{
 "patch": dict, "key": str,
 "scores":   {"readable": f, "persistent": f, "persistent_by_band": {"fast": f, "mid": f, "slow": f},
              "causal": f, "verbal": f},
 "controls": {"untrained": {...same score keys...}, "shuffled": {...}, "jlens_only": {"readable": f, "delta_R": f},
              "random_subspace": {"causal": f}, "heldout": {"readable": f, "ratio": f}},
 "margins":  {name: float},   # positive = S-space beats that control
 "pass": bool,                # True only if every margin clears its threshold
 "elapsed_s": f,
}
```

`python3 -m sspace.eval [patches/x.toml] [--no-controls]` prints a table and writes `runs/eval_<key>.json`. **A single evaluate with controls must take under 120 s on CPU.**

The scores (all in [0, 1], computed on test):
- **readable**: a φ readout from the S-space features predicts the per-position label (see `data/README.md`) at every labelled position. Report macro-average accuracy over the 5 tasks.
- **persistent**: readable accuracy on `recall` at positions ≥ 8 tokens after the key write. Also report accuracy by band.
- **causal**: on `load_cf("recall")`, replace the S-space component of a's state with b's at a middle position (layer per patch), using `state_hook`. Report the fraction of examples where the model's output at `query_pos` flips to b's key.
- **verbal**: agreement between the lens's readout at `query_pos` and the model's actual argmax prediction.

The controls:
- **untrained**: the same patch on `runs/untrained.pt`.
- **shuffled**: the same patch with train labels permuted within each task.
- **jlens_only**: the same φ fitted on `resid[-1]` alone, with delta_R = R(joint) − R(resid).
- **random_subspace**: the causal score using a random subspace of equal dimension.
- **heldout**: the basis fitted on 4 tasks, the readout on the 5th, averaged over the 5 folds. Report `ratio = heldout / in-dist`.

Pass thresholds:
- readable beats untrained by ≥ 0.10
- readable beats shuffled by ≥ 0.10
- delta_R ≥ 0.02
- causal beats random_subspace by ≥ 0.10
- heldout ratio ≥ 0.8

## C: `sspace/loop.py`

`python3 -m sspace.loop [--iters N | --minutes M] [--fake] [--seed S]` runs an "autoresearch" search over `KNOBS`:
1. It is resumable from `runs/ledger.jsonl`. Each line holds one `evaluate()` result plus `{"iter", "parent", "mutation", "t"}`, and patches are deduplicated by `key`.
2. The search mixes 1–2-knob mutations of the current best (an evolutionary hill climb) with random restarts.
3. The objective is to pass first, then maximise the sum of clipped margins.
4. It rewrites `runs/leaderboard.md` after every iteration with the top 10 patches, a per-knob marginal summary (the mean objective for each knob value) and the controls.
5. `--fake` uses a synthetic `evaluate` with the same schema, so the loop can be tested without A or B.
