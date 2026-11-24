"""Overnight autoresearch: search S-space definitions all night under rising controls.

    python3 -m sspace.overnight --until 07:00         # or --hours 8
    caffeinate -i python3 -m sspace.overnight ...     # keep a Mac awake (make overnight does this)

What one night does, in rounds (default 40 min each):

  1. SEARCH: a hill climb with restarts over the patch knobs. `seed` is no longer a knob:
     it becomes a replicate (see below).
  2. ESCALATE: each candidate faces harder tests in turn, and a candidate only
     moves up a level by passing every control at that level:
       L1  passes on the base model, fitting seed 0
       L2  also passes with fitting seeds 1 and 2         (not a lucky fit)
       L3  also passes on two independently trained models (not one model's quirk)
     Only L3 counts as "robust".
  3. RATCHET: if a round produces too many robust candidates, the thresholds rise.
     They never fall. Status is always recomputed from the stored raw scores under
     the *current* thresholds, so earlier winners have to keep clearing the bar.
  4. JOURNAL: runs/overnight/journal.md gets a summary per round, and
     runs/overnight/STATUS.md is the one-page morning read.

Resumable: the state lives in runs/overnight/{ledger.jsonl,state.json}. Ctrl-C or a
kill loses at most one evaluation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import eval as E
from .loop import CLIP, mutate, random_patch
from .patch import KNOBS, Patch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "overnight"
LEDGER, STATE = OUT / "ledger.jsonl", OUT / "state.json"
JOURNAL, STATUS = OUT / "journal.md", OUT / "STATUS.md"

SEARCH_KNOBS = {k: v for k, v in KNOBS.items() if k != "seed"}
FIT_SEEDS = (0, 1, 2)
MODEL_SEEDS = (0, 1, 2)

# The first real loop showed the contract thresholds were too easy: 50% of patches
# passed, and shuffled/heldout were cleared 100% of the time. Start tighter.
START_THR = {"readable_vs_untrained": 0.20, "readable_vs_shuffled": 0.30, "delta_R": 0.05,
             "causal_vs_random": 0.25, "heldout_ratio": 0.90}
RATCHET = {"readable_vs_untrained": 0.03, "readable_vs_shuffled": 0.03, "delta_R": 0.02,
           "causal_vs_random": 0.05, "heldout_ratio": 0.015}
CAPS = {"readable_vs_untrained": 0.6, "readable_vs_shuffled": 0.8, "delta_R": 0.3,
        "causal_vs_random": 0.9, "heldout_ratio": 0.99}
RATCHET_FRACTION = 0.15   # robust fraction of a round's new candidates that triggers tightening
RATCHET_MIN = 3


# ───────────────────────────── models ─────────────────────────────
def model_paths(ms: int) -> tuple[str, str]:
    if ms == 0:
        return "runs/base.pt", "runs/untrained.pt"
    return f"runs/models/s{ms}/base.pt", f"runs/models/s{ms}/untrained.pt"


def ensure_models(log):
    for ms in MODEL_SEEDS:
        base, _ = model_paths(ms)
        if (ROOT / base).exists():
            continue
        out = "runs" if ms == 0 else f"runs/models/s{ms}"
        log(f"training replicate model seed={ms} -> {out} (~3 min)")
        t = time.time()
        subprocess.run([sys.executable, "-m", "sspace.train", "--seed", str(ms), "--out", out],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        m = json.loads((ROOT / out / "train_metrics.json").read_text())
        log(f"  done in {time.time()-t:.0f}s, test acc {m['test_acc']}")


# ───────────────────────────── margins under any thresholds ─────────────────────────────
def margins(row: dict, thr: dict) -> dict:
    s, c = row["scores"], row["controls"]
    return {
        "readable_vs_untrained": s["readable"] - c["untrained"]["readable"] - thr["readable_vs_untrained"],
        "readable_vs_shuffled": s["readable"] - c["shuffled"]["readable"] - thr["readable_vs_shuffled"],
        "delta_R": c["jlens_only"]["delta_R"] - thr["delta_R"],
        "causal_vs_random": s["causal"] - c["random_subspace"]["causal"] - thr["causal_vs_random"],
        "heldout_ratio": c["heldout"]["ratio"] - thr["heldout_ratio"],
    }


def passes(row, thr) -> bool:
    return "error" not in row and all(v >= 0 for v in margins(row, thr).values())


def msum(row, thr) -> float:
    if "error" in row:
        return -9.0
    return sum(max(-CLIP, min(CLIP, v)) for v in margins(row, thr).values())


# ───────────────────────────── ledger / state ─────────────────────────────
def cand_key(d: dict) -> str:
    return Patch.from_dict({**d, "seed": 0}).key()


def load_rows() -> list[dict]:
    rows = []
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def append(row):
    OUT.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"thr": dict(START_THR), "round": 0, "thr_history": [{"round": 0, "t": now(), "thr": dict(START_THR)}]}


def save_state(st):
    tmp = STATE.with_suffix(".tmp"); tmp.write_text(json.dumps(st, indent=1)); tmp.replace(STATE)


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


# ───────────────────────────── candidate status ─────────────────────────────
def index(rows):
    """cand -> {(fit_seed, model_seed): row}  (first row wins)."""
    ix: dict[str, dict] = {}
    for r in rows:
        ix.setdefault(r["cand"], {}).setdefault((r["fit_seed"], r["model_seed"]), r)
    return ix


def level(evals: dict, thr) -> int:
    def ok(fs, ms):
        r = evals.get((fs, ms))
        return r is not None and passes(r, thr)
    if not ok(0, 0):
        return 0
    if not all(ok(fs, 0) for fs in FIT_SEEDS):
        return 1
    if not all(ok(0, ms) for ms in MODEL_SEEDS):
        return 2
    return 3


def cand_obj(evals, thr) -> tuple[int, float]:
    return level(evals, thr), min(msum(r, thr) for r in evals.values())


# ───────────────────────────── evaluation with escalation ─────────────────────────────
class Runner:
    def __init__(self, st, log):
        self.st, self.log = st, log
        self.rows = load_rows()
        self.consec_err = 0

    def run_one(self, d, fs, ms, rnd, parent, mutation):
        base, untr = model_paths(ms)
        p = Patch.from_dict({**d, "seed": fs})
        row = {"cand": cand_key(d), "patch": p.to_dict(), "fit_seed": fs, "model_seed": ms,
               "round": rnd, "parent": parent, "mutation": mutation, "t": now()}
        try:
            r = E.evaluate(p, base=base, untrained=untr)
            row.update({k: r[k] for k in ("scores", "controls", "head", "elapsed_s")})
            self.consec_err = 0
        except Exception as e:  # keep going all night; record what broke
            row["error"] = f"{type(e).__name__}: {e}"
            row["trace"] = traceback.format_exc()[-1500:]
            self.consec_err += 1
            self.log(f"  ERROR {row['cand']} fs={fs} ms={ms}: {row['error']}")
        append(row)
        self.rows.append(row)
        return row

    def escalate(self, d, rnd, parent, mutation):
        thr = self.st["thr"]
        evals = index(self.rows).get(cand_key(d), {})
        steps = [(0, 0)] + [(fs, 0) for fs in FIT_SEEDS[1:]] + [(0, ms) for ms in MODEL_SEEDS[1:]]
        for fs, ms in steps:
            if (fs, ms) not in evals:
                evals[(fs, ms)] = self.run_one(d, fs, ms, rnd, parent, mutation)
            if not passes(evals[(fs, ms)], thr):
                break
        return cand_obj(evals, thr)


# ───────────────────────────── reporting ─────────────────────────────
def summarize(rows, thr):
    ix = index(rows)
    cands = []
    for ck, ev in ix.items():
        lv, ob = cand_obj(ev, thr)
        d = {k: v for k, v in next(iter(ev.values()))["patch"].items() if k != "seed"}
        good = [r for r in ev.values() if "error" not in r]
        mean = lambda k: sum(r["scores"][k] for r in good) / len(good) if good else float("nan")
        mins = {m: min(margins(r, thr)[m] for r in good) for m in START_THR} if good else {}
        cands.append({"cand": ck, "level": lv, "obj": ob, "patch": d, "n": len(ev),
                      "readable": mean("readable"), "persistent": mean("persistent"),
                      "causal": mean("causal"), "verbal": mean("verbal"), "min_margins": mins,
                      "round": min(r["round"] for r in ev.values())})
    cands.sort(key=lambda c: (c["level"], c["obj"]), reverse=True)
    return cands


def enrichment(cands, lv=3):
    top = [c for c in cands if c["level"] >= lv]
    lines = []
    for k, vals in SEARCH_KNOBS.items():
        for v in vals:
            a = sum(c["patch"][k] == v for c in cands)
            b = sum(c["patch"][k] == v for c in top)
            if a:
                lines.append((k, v, a, b, b / a))
    return lines


def ptxt(d):
    return " ".join(f"{k}={v}" for k, v in d.items())


def write_status(st, rows, started, deadline, notes):
    thr = st["thr"]
    cands = summarize(rows, thr)
    by = {lv: sum(c["level"] == lv for c in cands) for lv in range(4)}
    errs = sum("error" in r for r in rows)
    L = [f"# Overnight autoresearch: status", "",
         f"Updated {now()} · round {st['round']} · started {started} · deadline {deadline}", "",
         f"**{len(cands)} candidates**, {len(rows)} evaluations, {errs} errors. "
         f"Levels under the *current* thresholds: L0 {by[0]} · L1 {by[1]} · L2 {by[2]} · **L3 robust {by[3]}**.", "",
         "> L1 = passes on the base model · L2 = also passes with fitting seeds 1 and 2 · L3 = also passes on 2 independently "
         "trained models. Thresholds only rise. This is a 32k-parameter model on synthetic micro tasks, so a robust "
         "patch is a hypothesis to test on real models, not a finding about them.", "",
         "## Current thresholds", "",
         "| control | start | now |", "|---|--:|--:|"]
    L += [f"| {k} | {START_THR[k]:.3f} | **{thr[k]:.3f}** |" for k in START_THR]
    L += ["", "## Top candidates", "",
          "| # | level | min-margin sum | readable | persist | causal | verbal | weakest control | patch |",
          "|--:|:-:|--:|--:|--:|--:|--:|---|---|"]
    for i, c in enumerate(cands[:15], 1):
        weak = min(c["min_margins"].items(), key=lambda kv: kv[1]) if c["min_margins"] else ("-", 0)
        L.append(f"| {i} | L{c['level']} | {c['obj']:+.3f} | {c['readable']:.2f} | {c['persistent']:.2f} | "
                 f"{c['causal']:.2f} | {c['verbal']:.2f} | {weak[0]} {weak[1]:+.3f} | {ptxt(c['patch'])} |")
    enr = [e for e in enrichment(cands, 3)]
    if by[3]:
        L += ["", "## What robust patches have in common", "",
              "Share of evaluated candidates with each knob value that reached L3. Read it as a "
              "tendency: the search visits values unevenly.", "", "| knob | value | tried | robust | rate |", "|---|---|--:|--:|--:|"]
        L += [f"| {k} | {v} | {a} | {b} | {r:.2f} |" for k, v, a, b, r in sorted(enr, key=lambda e: (e[0], -e[4]))]
    L += ["", "## Threshold history", ""]
    L += [f"- round {h['round']} ({h['t']}): " + ", ".join(f"{k} {v:.3f}" for k, v in h["thr"].items()) for h in st["thr_history"]]
    if notes:
        L += ["", "## Notes", ""] + [f"- {n}" for n in notes[-15:]]
    tmp = STATUS.with_suffix(".tmp"); tmp.write_text("\n".join(L) + "\n"); tmp.replace(STATUS)
    return cands


def journal(text):
    OUT.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a") as f:
        f.write(text + "\n")


# ───────────────────────────── main ─────────────────────────────
def parse_deadline(a) -> dt.datetime:
    n = dt.datetime.now()
    if a.hours is not None:
        return n + dt.timedelta(hours=a.hours)
    hh, mm = map(int, a.until.split(":"))
    d = n.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return d if d > n else d + dt.timedelta(days=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--until", default="07:00", help="local HH:MM deadline (default 07:00)")
    ap.add_argument("--hours", type=float, default=None, help="run for N hours instead of --until")
    ap.add_argument("--round-minutes", type=float, default=40)
    ap.add_argument("--restart-p", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=None, help="search RNG seed (default: from round number)")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    deadline = parse_deadline(a)
    started = now()
    notes: list[str] = []

    def log(msg):
        line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)

    st = load_state()
    log(f"overnight start; deadline {deadline:%Y-%m-%d %H:%M}; thresholds {st['thr']}")
    journal(f"\n# Night started {started} (deadline {deadline:%Y-%m-%d %H:%M})\n")
    ensure_models(log)
    E.THRESHOLDS.update(st["thr"])  # evaluate()'s own pass flag follows ours (we recompute anyway)
    R = Runner(st, log)

    try:
        while dt.datetime.now() < deadline:
            st["round"] += 1
            rnd = st["round"]
            rng = random.Random(a.seed if a.seed is not None else 7919 * rnd)
            thr = st["thr"]
            round_end = min(deadline, dt.datetime.now() + dt.timedelta(minutes=a.round_minutes))
            seen = set(index(R.rows))
            new_cands, stall = [], 0
            cands = summarize(R.rows, thr)
            best = cands[0] if cands else None
            best_obj = (best["level"], best["obj"]) if best else (-1, -9e9)
            parent = best["patch"] if best else random_patch(SEARCH_KNOBS, rng)
            log(f"round {rnd}: {len(seen)} known candidates; best L{best_obj[0]} {best_obj[1]:+.3f}")
            # warm start: on a fresh night, first escalate the top patches from the plain loop's ledger
            warm = []
            if not seen and (ROOT / "runs" / "ledger.jsonl").exists():
                prev = [json.loads(l) for l in (ROOT / "runs" / "ledger.jsonl").read_text().splitlines() if l.strip()]
                prev.sort(key=lambda r: (r.get("pass", False), sum(max(-CLIP, min(CLIP, v)) for v in r.get("margins", {}).values())), reverse=True)
                for r in prev:
                    d = {k: v for k, v in r["patch"].items() if k != "seed"}
                    if cand_key(d) not in {cand_key(w) for w in warm}:
                        warm.append(d)
                    if len(warm) >= 6:
                        break
                log(f"  warm start from runs/ledger.jsonl: {len(warm)} prior leaders")
            for d in warm:
                ck = cand_key(d)
                seen.add(ck); new_cands.append(ck)
                ob = R.escalate(d, rnd, None, "warm-start")
                log(f"  {ck} L{ob[0]} {ob[1]:+.3f}  {'warm-start':<22} {ptxt(Patch.from_dict({**d, 'seed': 0}).to_dict())}")
                if ob > best_obj:
                    best_obj, parent = ob, d
            while dt.datetime.now() < round_end:
                if R.consec_err >= 10:
                    raise RuntimeError("10 consecutive evaluation errors; stopping (see ledger 'trace')")
                if not seen or rng.random() < a.restart_p or stall >= 8:
                    d, mut, par = random_patch(SEARCH_KNOBS, rng), "restart", None
                    stall = 0
                else:
                    d, ks = mutate(parent, SEARCH_KNOBS, rng)
                    mut, par = ",".join(ks), cand_key(parent)
                ck = cand_key(d)
                if ck in seen:
                    stall += 1
                    continue
                seen.add(ck)
                new_cands.append(ck)
                ob = R.escalate(d, rnd, par, mut)
                log(f"  {ck} L{ob[0]} {ob[1]:+.3f}  {mut:<22} {ptxt(Patch.from_dict({**d, 'seed': 0}).to_dict())}")
                if ob > best_obj:
                    best_obj, parent, stall = ob, d, 0
                    if ob[0] == 3:
                        notes.append(f"{now()} round {rnd}: new robust best {ck} ({ptxt(d)})")
                else:
                    stall += 1
            # ── end of round: ratchet, journal, status
            cands = write_status(st, R.rows, started, f"{deadline:%Y-%m-%d %H:%M}", notes)
            newset = set(new_cands)
            robust_new = [c for c in cands if c["cand"] in newset and c["level"] == 3]
            frac = len(robust_new) / max(1, len(newset))
            msg = (f"## Round {rnd}, {now()}\n\n- new candidates: {len(newset)}; reached L3 this round: {len(robust_new)} "
                   f"({frac:.0%})\n- totals under current thresholds: " +
                   ", ".join(f"L{lv} {sum(c['level']==lv for c in cands)}" for lv in range(4)))
            if cands:
                msg += f"\n- leader: L{cands[0]['level']} {cands[0]['obj']:+.3f} `{cands[0]['cand']}` {ptxt(cands[0]['patch'])}"
            if len(robust_new) >= RATCHET_MIN and frac >= RATCHET_FRACTION:
                old = dict(thr)
                for k in thr:
                    thr[k] = round(min(CAPS[k], thr[k] + RATCHET[k]), 4)
                E.THRESHOLDS.update(thr)
                st["thr_history"].append({"round": rnd, "t": now(), "thr": dict(thr)})
                still = sum(c["level"] == 3 for c in summarize(R.rows, thr))
                msg += (f"\n- **ratchet**: {len(robust_new)} robust ({frac:.0%}) ≥ trigger, so thresholds rose: " +
                        ", ".join(f"{k} {old[k]:.3f}→{thr[k]:.3f}" for k in thr) +
                        f". Robust under the new bar: {still}.")
                notes.append(f"{now()} round {rnd}: ratchet → {thr}")
                write_status(st, R.rows, started, f"{deadline:%Y-%m-%d %H:%M}", notes)
            journal(msg + "\n")
            save_state(st)
            log(msg.replace("\n", " | "))
    except KeyboardInterrupt:
        log("interrupted; ledger and state are on disk")
    except Exception as e:
        notes.append(f"{now()} stopped on error: {e}")
        journal(f"\n**Stopped on error** {now()}: {e}\n")
        log(f"fatal: {e}")
        raise
    finally:
        save_state(st)
        write_status(st, R.rows, started, f"{deadline:%Y-%m-%d %H:%M}", notes)
        journal(f"\n_Night ended {now()}: {len(index(R.rows))} candidates, {len(R.rows)} evaluations._\n")
        log(f"done -> {STATUS}")


if __name__ == "__main__":
    main()
