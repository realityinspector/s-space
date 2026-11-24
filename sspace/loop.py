"""Autoresearch loop: hill-climb over Patch knobs, keep what beats the controls.

    python3 -m sspace.loop [--iters N | --minutes M] [--fake] [--seed S]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIP = 0.3  # each margin is clipped to [-CLIP, CLIP] in the objective
RESTART_P = 0.2
RESTART_AFTER = 8  # non-improving iterations before a forced restart

FALLBACK_KNOBS = {
    "locus": ["raw", "readout", "gates", "joint"],
    "instrument": ["sae", "jacobian", "covariance", "probe", "gramian"],
    "phi": ["linear", "lowrank", "mlp"],
    "rank": [4, 8, 16, 32],
    "granularity": ["head", "layer", "model"],
    "layer": [0, 1, -1],
    "tau_coord": [False, True],
    "modulation": [False, True],
    "seed": [0, 1, 2],
}


# ---------------------------------------------------------------- fake backend
def _h(*parts) -> float:
    s = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(s[:8], "big") / 2**64


def fake_key(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


def fake_evaluate(d: dict) -> dict:
    """Synthetic landscape with the contract schema. Not a result of anything."""
    t0 = time.time()
    g = lambda k, v: _h("fx", k, v)  # per-knob-value fixed effect in [0,1)
    base = 0.30 * g("locus", d["locus"]) + 0.25 * g("instrument", d["instrument"])
    base += 0.1 * g("phi", d["phi"]) + 0.05 * math.log2(d["rank"]) / 5
    if d["locus"] == "joint" and d["instrument"] in ("jacobian", "gramian"):
        base += 0.25  # interaction
    if d["granularity"] == "model" and d["phi"] == "mlp":
        base -= 0.15  # interaction
    if d["granularity"] != "model" and d["layer"] == -1:
        base += 0.08
    base += 0.06 * d["tau_coord"] + 0.05 * d["modulation"] * (d["locus"] in ("gates", "joint"))
    rng = random.Random(fake_key(d))
    noise = lambda s=0.05: rng.gauss(0, s)
    readable = min(1.0, max(0.0, 0.35 + 0.5 * base + noise()))
    untr = max(0.0, readable - 0.18 * base * 2 + noise(0.06) - 0.02)
    shuf = max(0.0, 0.2 + noise(0.04) + 0.1 * (1 - base))
    causal = min(1.0, max(0.0, 0.2 + 0.7 * base + noise(0.08)))
    rand_c = max(0.0, 0.1 + noise(0.05))
    delta_R = 0.12 * (base - 0.3) + noise(0.02)
    ratio = min(1.2, max(0.0, 0.55 + 0.5 * base + noise(0.07)))
    scores = {
        "readable": readable,
        "persistent": max(0.0, readable - 0.1 + noise()),
        "persistent_by_band": {b: max(0.0, readable - o + noise()) for b, o in
                               (("fast", 0.25), ("mid", 0.1), ("slow", 0.05))},
        "causal": causal,
        "verbal": max(0.0, min(1.0, readable - 0.05 + noise())),
    }
    ctrl_scores = lambda r: {**scores, "readable": r}
    margins = {
        "readable_vs_untrained": readable - untr - 0.10,
        "readable_vs_shuffled": readable - shuf - 0.10,
        "delta_R": delta_R - 0.02,
        "causal_vs_random": causal - rand_c - 0.10,
        "heldout_ratio": ratio - 0.8,
    }
    return {
        "patch": d, "key": fake_key(d), "scores": scores,
        "controls": {
            "untrained": ctrl_scores(untr), "shuffled": ctrl_scores(shuf),
            "jlens_only": {"readable": readable - delta_R, "delta_R": delta_R},
            "random_subspace": {"causal": rand_c},
            "heldout": {"readable": readable * ratio, "ratio": ratio},
        },
        "margins": margins, "pass": all(m >= 0 for m in margins.values()),
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------- backend glue
class Backend:
    def __init__(self, fake: bool):
        self.fake = fake
        self.knobs = FALLBACK_KNOBS
        self._Patch = self._eval = None
        if not fake:
            from sspace.patch import KNOBS, Patch  # lazy: owned by workstream B
            from sspace.eval import evaluate
            self.knobs, self._Patch, self._eval = KNOBS, Patch, evaluate

    def key(self, d):
        return fake_key(d) if self.fake else self._Patch.from_dict(d).key()

    def evaluate(self, d):
        if self.fake:
            return fake_evaluate(d)
        return self._eval(self._Patch.from_dict(d))


# ---------------------------------------------------------------- objective
def objective(r: dict) -> tuple[int, float]:
    s = sum(max(-CLIP, min(CLIP, m)) for m in r["margins"].values())
    return (1 if r["pass"] else 0, s)


def scalar(r):  # for sorting/means
    p, s = objective(r)
    return 100 * p + s


# ---------------------------------------------------------------- ledger
def load_ledger(path: Path) -> list[dict]:
    rows, seen = [], set()
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn last line after a hard kill
            if r["key"] in seen:
                continue
            seen.add(r["key"])
            rows.append(r)
    return rows


def append(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


# ---------------------------------------------------------------- leaderboard
def write_leaderboard(path: Path, rows: list[dict], knobs: dict, fake: bool):
    ranked = sorted(rows, key=lambda r: objective(r), reverse=True)
    npass = sum(r["pass"] for r in rows)
    L = []
    L.append("# Leaderboard" + (" (FAKE: synthetic landscape, not real results)" if fake else ""))
    L.append("")
    L.append(f"Evaluated patches: **{len(rows)}**. Passing: **{npass}**. "
             f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}.")
    L.append("")
    L.append("> **Honesty note.** \"Pass\" means a patch beat these specific controls "
             "(untrained model, shuffled labels, J-lens-only delta R, held-out task, "
             "random subspace) by the stated thresholds on this micro dataset with this "
             "tiny model. It is not evidence about real LLMs, and with many patches "
             "tried, some passes will be selection noise. Re-run with new seeds before "
             "believing any single row.")
    L.append("")
    mnames = sorted({m for r in rows for m in r["margins"]})
    L.append("## Top 10")
    L.append("")
    L.append("| # | key | pass | obj | readable | persist | causal | verbal | patch |")
    L.append("|--:|---|:-:|--:|--:|--:|--:|--:|---|")
    for i, r in enumerate(ranked[:10], 1):
        s = r["scores"]
        pd = " ".join(f"{k}={v}" for k, v in r["patch"].items())
        L.append(f"| {i} | `{r['key']}` | {'yes' if r['pass'] else 'no'} | {objective(r)[1]:.3f} | "
                 f"{s['readable']:.2f} | {s['persistent']:.2f} | {s['causal']:.2f} | "
                 f"{s['verbal']:.2f} | {pd} |")
    L.append("")
    L.append("## Per-knob marginal means")
    L.append("")
    L.append("Mean objective (100 per pass + clipped margin sum) over all evaluated patches "
             "with that knob value, and pass rate. Marginals are confounded by which "
             "patches the climb chose to visit.")
    L.append("")
    L.append("| knob | value | n | mean obj | pass rate |")
    L.append("|---|---|--:|--:|--:|")
    for k, vals in knobs.items():
        for v in vals:
            sub = [r for r in rows if r["patch"].get(k) == v]
            if not sub:
                L.append(f"| {k} | {v} | 0 | - | - |")
                continue
            L.append(f"| {k} | {v} | {len(sub)} | {sum(map(scalar, sub))/len(sub):.2f} | "
                     f"{sum(r['pass'] for r in sub)/len(sub):.2f} |")
    L.append("")
    L.append("## Controls summary")
    L.append("")
    L.append("Margin = value minus threshold-adjusted control (positive clears). "
             "Mean over all patches, and fraction clearing.")
    L.append("")
    L.append("| margin | mean | best | fraction >= 0 |")
    L.append("|---|--:|--:|--:|")
    for m in mnames:
        v = [r["margins"][m] for r in rows if m in r["margins"]]
        if v:
            L.append(f"| {m} | {sum(v)/len(v):.3f} | {max(v):.3f} | {sum(x >= 0 for x in v)/len(v):.2f} |")
    L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(L) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------- search
def random_patch(knobs, rng):
    return {k: rng.choice(v) for k, v in knobs.items()}


def mutate(d, knobs, rng):
    n = rng.choice([1, 2])
    ks = rng.sample(list(knobs), n)
    new = dict(d)
    for k in ks:
        opts = [v for v in knobs[k] if v != d[k]]
        new[k] = rng.choice(opts)
    return new, ks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.iters is None and a.minutes is None:
        a.iters = 50

    be = Backend(a.fake)
    runs = ROOT / "runs"
    ledger = runs / ("ledger.fake.jsonl" if a.fake else "ledger.jsonl")
    board = runs / ("leaderboard.fake.md" if a.fake else "leaderboard.md")
    rows = load_ledger(ledger)
    seen = {r["key"] for r in rows}
    it = max((r["iter"] for r in rows), default=-1) + 1
    # mix resumed iteration count into the rng so a resumed run explores new points
    rng = random.Random(f"{a.seed}:{it}")
    best = max(rows, key=objective, default=None)
    stale = 0
    t_end = time.time() + a.minutes * 60 if a.minutes else None
    done = 0
    print(f"[loop] {'FAKE ' if a.fake else ''}resuming with {len(rows)} rows, next iter {it}")
    total = len(list(__import__('itertools').product(*be.knobs.values())))

    try:
        while True:
            if a.iters is not None and done >= a.iters:
                break
            if t_end and time.time() >= t_end:
                break
            if len(seen) >= total:
                print("[loop] search space exhausted")
                break
            # propose
            cand = parent = mut = None
            for _ in range(200):
                if best is None or rng.random() < RESTART_P or stale >= RESTART_AFTER:
                    c, m, p = random_patch(be.knobs, rng), ["restart"], None
                else:
                    c, m = mutate(best["patch"], be.knobs, rng)
                    p = best["key"]
                if be.key(c) not in seen:
                    cand, mut, parent = c, m, p
                    break
            if cand is None:  # neighbourhood exhausted; force restart
                stale = RESTART_AFTER
                continue
            if mut == ["restart"]:
                stale = 0
            t0 = time.time()
            r = be.evaluate(cand)
            r["key"] = be.key(cand)
            r.update(iter=it, parent=parent, mutation=mut, t=time.time())
            append(ledger, r)
            rows.append(r)
            seen.add(r["key"])
            improved = best is None or objective(r) > objective(best)
            if improved:
                best, stale = r, 0
            else:
                stale += 1
            write_leaderboard(board, rows, be.knobs, a.fake)
            print(f"[loop] iter {it:4d} {r['key']} pass={int(r['pass'])} "
                  f"obj={objective(r)[1]:+.3f} {'*' if improved else ' '} "
                  f"mut={','.join(map(str, mut))} ({time.time()-t0:.1f}s)")
            it += 1
            done += 1
    except KeyboardInterrupt:
        print("\n[loop] interrupted; ledger is already flushed")
    write_leaderboard(board, rows, be.knobs, a.fake)
    print(f"[loop] done: {len(rows)} total, {sum(r['pass'] for r in rows)} passing -> {board}")


if __name__ == "__main__":
    main()
