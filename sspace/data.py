"""Loader for data/micro. Shared by train, eval and loop. Stdlib + torch only."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MICRO = ROOT / "data" / "micro"
TASKS = ("recall", "brackets", "count", "entity", "topic")


@lru_cache(maxsize=None)
def vocab() -> dict:
    return json.loads((MICRO / "vocab.json").read_text())


def tok2id() -> dict[str, int]:
    return {t: i for i, t in enumerate(vocab()["tokens"])}


@lru_cache(maxsize=None)
def _load(name: str) -> tuple:
    path = MICRO / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run: python3 data/generate.py")
    return tuple(json.loads(l) for l in path.open())


def load_split(name: str, tasks: tuple[str, ...] | None = None) -> list[dict]:
    """name in {'train','test'}. Rows follow data/README.md."""
    rows = _load(name)
    return [r for r in rows if tasks is None or r["task"] in tasks]


def load_cf(task: str | None = None) -> list[dict]:
    """Counterfactual pairs: {'pair','task','diff_pos','a':row,'b':row}."""
    return [p for p in _load("cf") if task is None or p["task"] == task]


def batchify(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """-> ids LongTensor [B, T] right-padded with pad_id=0, lengths LongTensor [B]."""
    T = max(len(r["ids"]) for r in rows)
    ids = torch.zeros(len(rows), T, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r["ids"])] = torch.tensor(r["ids"])
    return ids, torch.tensor([len(r["ids"]) for r in rows])
