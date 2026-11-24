# s-space micro dataset

A tiny, fully synthetic dataset for testing S-space ideas on a laptop CPU. It has 63 tokens, sequences of at most 32 tokens, 1,250 examples, and takes about 700 KB. It is deterministic given a seed and needs only the Python standard library.

```sh
python3 data/generate.py              # → data/micro/
python3 data/generate.py --seed 1     # another draw
python3 data/generate.py --train 50 --test 10   # even smaller
```

## Tasks

Each task targets one claim about what a recurrent state has to hold. Train has 200 examples per task and test has 50.

| task | example | what the state must hold | tests |
|---|---|---|---|
| `recall` | `<bos> KC f3 f1 … ? KC` | one key across a lag k ∈ {2, 4, 8, 16, 24} | persistence; **timescale bands** (`meta.band` is fast, mid or slow) |
| `brackets` | `<bos> ( f2 ( ) … = 1` | running bracket depth, 0–9 | content that is not a word (a counter) |
| `count` | `<bos> f4 * f1 * … = 2` | running count of `*` | content that is not a word (integration) |
| `entity` | `<bos> cy : green … ana : red … ? green = cy` | 2–4 name:colour bindings | binding and slot memory |
| `topic` | `<bos> SEA salt tide … f9 f7 ? SEA` | a topic set at the start | slow semantic anchor |

In `topic`, topic words appear only in the first half of each sequence. The second half is filler, so the answer has to come from memory. A check confirmed that no topic word appears in the last six tokens before the query.

## Files

| file | contents |
|---|---|
| `micro/train.jsonl`, `micro/test.jsonl` | one example per line |
| `micro/cf.jsonl` | 100 counterfactual pairs (`a`, `b`), for patching and steering |
| `micro/vocab.json` | the token list, `max_len`, `pad_id` and `bos_id` |
| `micro/stats.json` | counts, mean length, band balance, majority-class baselines |

## Row schema

```json
{
  "id": "test-recall-0", "task": "recall",
  "tokens": ["<bos>","KA","f0","f9","?","KA"],
  "ids":    [1, 9, 43, 52, 3, 9],
  "query_pos": 4,            // predict tokens[-1] from the state at this position
  "target": "KA",
  "labels": [null,"KA","KA","KA","KA","KA"],   // what should be held at each position
  "meta": {"lag": 2, "band": "fast", "key": "KA"}
}
```

`labels` holds the per-position ground truth for lens scoring. A state lens read at position t should recover `labels[t]`:

- `recall`: the key
- `brackets`: the depth
- `count`: the count
- `entity`: the bindings dict so far
- `topic`: the topic

## Counterfactual pairs (`cf.jsonl`)

- **`recall`**: `a` and `b` are identical except for the key at position 1 (and the answer). Use them for activation patching. Swap the state at position t from `b` into `a` and check whether the output flips, at each lag. This is the causal half of decision 10.
- **`topic`**: the topic token and the topic words differ, and the filler is identical. Use these for steering: add the mean `b − a` state difference and see whether the answer moves.

## Chance baselines (test)

These are majority-class rates: recall 0.20, brackets 0.18, count 0.22, entity 0.38, topic 0.34. A lens or model has to beat these rates.

## Scale note

This data is for checking that the pipeline is correct: that lenses, Gramians, patching and band splits all run end to end on a tiny SSM within minutes on a CPU. It says nothing about whether S-space exists in real language models.
