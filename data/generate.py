#!/usr/bin/env python3
"""Generate the s-space micro dataset: tiny, synthetic, deterministic, CPU-only.

Five tasks, each probing one claim about what a recurrent state must hold:

  recall    delayed key recall at controlled lags      -> persistence, timescale bands
  brackets  running bracket depth                      -> non-word content (a counter)
  count     running count of a marker token            -> non-word content, integration
  entity    name:colour bindings, queried later        -> binding / slot memory
  topic     a topic token biases every later word      -> slow semantic anchor

Every example carries per-position concept labels (`labels`), so a lens fitted on
the state at position t can be scored against what *should* be held at t.
`recall` and `topic` also ship counterfactual pairs (`cf.jsonl`) that differ in
exactly one token, for patching and steering tests.

    python3 data/generate.py            # writes data/micro/
    python3 data/generate.py --seed 1   # a different but equally tiny draw

Stdlib only. Output is ~0.5 MB.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

# ───────────────────────────── vocabulary ─────────────────────────────
SPECIAL = ["<pad>", "<bos>", "<eos>", "?", "=", ":", "(", ")", "*"]
KEYS = [f"K{c}" for c in "ABCDEFGH"]
NAMES = ["ana", "bo", "cy", "di"]
COLOURS = ["red", "blue", "green", "gold"]
TOPICS = {
    "SEA": ["wave", "fish", "shell", "tide", "salt"],
    "SKY": ["cloud", "bird", "star", "wind", "moon"],
    "FOREST": ["tree", "moss", "deer", "fern", "owl"],
}
FILLER = [f"f{i}" for i in range(10)]
DIGITS = [str(i) for i in range(10)]
VOCAB = SPECIAL + KEYS + NAMES + COLOURS + list(TOPICS) + [w for ws in TOPICS.values() for w in ws] + FILLER + DIGITS
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
assert len(TOK2ID) == len(VOCAB), "duplicate token"

MAX_LEN = 32
LAGS = [2, 4, 8, 16, 24]
BAND = {2: "fast", 4: "fast", 8: "mid", 16: "mid", 24: "slow"}


def example(task, tokens, labels, meta, uid):
    """tokens end with the answer; the model predicts tokens[-1] from position len-2."""
    assert len(tokens) <= MAX_LEN, (task, len(tokens))
    assert len(labels) == len(tokens)
    return {
        "id": uid,
        "task": task,
        "tokens": tokens,
        "ids": [TOK2ID[t] for t in tokens],
        "query_pos": len(tokens) - 2,
        "target": tokens[-1],
        "labels": labels,
        "meta": meta,
    }


# ───────────────────────────── tasks ─────────────────────────────
def gen_recall(r: random.Random, uid: str, lag: int | None = None, key: str | None = None, filler=None):
    lag = lag or r.choice(LAGS)
    key = key or r.choice(KEYS)
    filler = filler or [r.choice(FILLER) for _ in range(lag)]
    toks = ["<bos>", key] + filler + ["?", key]
    labels = [None, key] + [key] * lag + [key, key]
    return example("recall", toks, labels, {"lag": lag, "band": BAND[lag], "key": key}, uid), filler


def gen_brackets(r: random.Random, uid: str):
    n = r.randint(12, 26)
    toks, labels, d = ["<bos>"], [0], 0
    for _ in range(n):
        x = r.random()
        if x < 0.4 and d < 9:
            t, d = "(", d + 1
        elif x < 0.75 and d > 0:
            t, d = ")", d - 1
        else:
            t = r.choice(FILLER)
        toks.append(t)
        labels.append(d)
    toks += ["=", str(d)]
    labels += [d, d]
    return example("brackets", toks, labels, {"final_depth": d, "max_depth": max(labels)}, uid)


def gen_count(r: random.Random, uid: str):
    n = r.randint(10, 24)
    toks, labels, c = ["<bos>"], [0], 0
    for _ in range(n):
        if r.random() < 0.3 and c < 9:
            toks.append("*")
            c += 1
        else:
            toks.append(r.choice(FILLER))
        labels.append(c)
    toks += ["=", str(c)]
    labels += [c, c]
    return example("count", toks, labels, {"count": c}, uid)


def gen_entity(r: random.Random, uid: str):
    k = r.randint(2, 4)
    names, cols = r.sample(NAMES, k), r.sample(COLOURS, k)
    toks, labels, bound = ["<bos>"], [{}], {}
    for nm, co in zip(names, cols):
        gap = [r.choice(FILLER) for _ in range(r.randint(0, 3))]
        for t in [nm, ":", co] + gap:
            if t in COLOURS:
                bound = {**bound, nm: t}
            toks.append(t)
            labels.append(dict(bound))
    gap = [r.choice(FILLER) for _ in range(r.randint(0, 4))]
    qi = r.randrange(k)
    for t in gap + ["?", cols[qi], "=", names[qi]]:
        toks.append(t)
        labels.append(dict(bound))
    return example("entity", toks, labels, {"n_pairs": k, "query_colour": cols[qi], "answer": names[qi], "query_index": qi}, uid)


def gen_topic(r: random.Random, uid: str, topic: str | None = None, words=None):
    topic = topic or r.choice(list(TOPICS))
    n = r.randint(14, 26)
    if words is None:
        # slots: True = topic word, False = filler; indices let counterfactuals swap topic cleanly
        # topic words only in the first half: the second half is pure filler, so the
        # answer must come from memory, not from the last few tokens
        words = [(j < n // 2 and r.random() < 0.7, r.randrange(5), r.choice(FILLER)) for j in range(n)]
    body = [TOPICS[topic][i] if is_t else f for is_t, i, f in words]
    toks = ["<bos>", topic] + body + ["?", topic]
    labels = [None] + [topic] * (len(toks) - 1)
    return example("topic", toks, labels, {"topic": topic, "topic_word_frac": round(sum(w[0] for w in words) / len(words), 3)}, uid), words


# ───────────────────────────── build ─────────────────────────────
def build(seed: int, n_train: int, n_test: int):
    r = random.Random(seed)
    splits = {"train": [], "test": []}
    cf = []
    for split, n in (("train", n_train), ("test", n_test)):
        for i in range(n):
            # stratify recall by lag so every band is equally represented
            ex, _ = gen_recall(r, f"{split}-recall-{i}", lag=LAGS[i % len(LAGS)])
            splits[split].append(ex)
            splits[split].append(gen_brackets(r, f"{split}-brackets-{i}"))
            splits[split].append(gen_count(r, f"{split}-count-{i}"))
            splits[split].append(gen_entity(r, f"{split}-entity-{i}"))
            ex, _ = gen_topic(r, f"{split}-topic-{i}", topic=list(TOPICS)[i % 3])
            splits[split].append(ex)

    # counterfactual pairs (test-time only): identical except the one causal token
    for i in range(n_test):
        lag = LAGS[i % len(LAGS)]
        k1, k2 = r.sample(KEYS, 2)
        a, fill = gen_recall(r, f"cf-recall-{i}-a", lag=lag, key=k1)
        b, _ = gen_recall(r, f"cf-recall-{i}-b", lag=lag, key=k2, filler=fill)
        cf.append({"pair": f"cf-recall-{i}", "task": "recall", "diff_pos": 1, "a": a, "b": b})
        t1, t2 = r.sample(list(TOPICS), 2)
        a, words = gen_topic(r, f"cf-topic-{i}-a", topic=t1)
        b, _ = gen_topic(r, f"cf-topic-{i}-b", topic=t2, words=words)
        cf.append({"pair": f"cf-topic-{i}", "task": "topic", "diff_pos": None, "a": a, "b": b,
                   "note": "topic token and every topic word differ; fillers identical"})
    for p in cf:
        if p["task"] == "recall":
            diff = [j for j, (x, y) in enumerate(zip(p["a"]["tokens"], p["b"]["tokens"])) if x != y]
            assert diff == [1, len(p["a"]["tokens"]) - 1], diff
    return splits, cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=int, default=200, help="examples per task in train")
    ap.add_argument("--test", type=int, default=50, help="examples per task in test (and cf pairs per task)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "micro")
    a = ap.parse_args()

    splits, cf = build(a.seed, a.train, a.test)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(a.out / f"{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
    with open(a.out / "cf.jsonl", "w") as f:
        for row in cf:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    (a.out / "vocab.json").write_text(json.dumps({"tokens": VOCAB, "max_len": MAX_LEN, "pad_id": 0, "bos_id": 1}, indent=1))

    stats = {"seed": a.seed, "vocab_size": len(VOCAB), "max_len": MAX_LEN, "splits": {}}
    for name, rows in splits.items():
        stats["splits"][name] = {
            "n": len(rows),
            "by_task": dict(Counter(x["task"] for x in rows)),
            "mean_len": round(sum(len(x["tokens"]) for x in rows) / len(rows), 2),
            "recall_by_band": dict(Counter(x["meta"]["band"] for x in rows if x["task"] == "recall")),
            "target_majority_rate": {t: round(max(Counter(x["target"] for x in rows if x["task"] == t).values()) / sum(1 for x in rows if x["task"] == t), 3)
                                     for t in sorted({x["task"] for x in rows})},
        }
    stats["cf_pairs"] = dict(Counter(p["task"] for p in cf))
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
