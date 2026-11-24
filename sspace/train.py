"""python3 -m sspace.train [--steps N] [--seed S]"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import ROOT, TASKS, batchify, load_split
from .model import Config, TinyMamba2, load, save


def weights(rows, ids, lengths, ans_w=5.0):
    """Loss weight for predicting ids[:,1:]; answer token (last token) x5."""
    B, T = ids.shape
    w = (ids[:, 1:] != 0).float()
    for i, L in enumerate(lengths.tolist()):
        w[i, L - 2] = ans_w  # target is ids[L-1]
    return w


@torch.no_grad()
def test_acc(model):
    out = {}
    rows = load_split("test")
    for task in TASKS:
        rs = [r for r in rows if r["task"] == task]
        ids, _ = batchify(rs)
        logits, _ = model(ids)
        ok = [int(logits[i, r["query_pos"]].argmax()) == r["ids"][r["query_pos"] + 1] for i, r in enumerate(rs)]
        out[task] = sum(ok) / len(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=6e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--threads", type=int, default=1)  # tiny ops: 1 thread is faster than 4 here
    ap.add_argument("--d_state", type=int, default=Config.d_state)
    ap.add_argument("--n_head", type=int, default=Config.n_head)
    ap.add_argument("--n_layer", type=int, default=Config.n_layer)
    ap.add_argument("--d_model", type=int, default=Config.d_model)
    ap.add_argument("--ans_w", type=float, default=5.0)
    ap.add_argument("--extra", type=int, default=1500, help="extra synthetic train examples per task (data/generate.py, fresh seed)")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    model = TinyMamba2(Config(d_model=a.d_model, n_layer=a.n_layer, n_head=a.n_head, d_state=a.d_state))
    runs = ROOT / a.out
    runs.mkdir(parents=True, exist_ok=True)
    save(model, runs / "untrained.pt")
    params = sum(p.numel() for p in model.parameters())
    rows = load_split("train")
    if a.extra:
        import importlib.util
        spec = importlib.util.spec_from_file_location("gen", ROOT / "data" / "generate.py")
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        held = {tuple(r["ids"]) for r in load_split("test")}
        rg = random.Random(1000 + a.seed)
        extra = []
        for i in range(a.extra):
            for f in (lambda u: gen.gen_recall(rg, u, lag=gen.LAGS[i % len(gen.LAGS)])[0], lambda u: gen.gen_brackets(rg, u),
                      lambda u: gen.gen_count(rg, u), lambda u: gen.gen_entity(rg, u),
                      lambda u: gen.gen_topic(rg, u, topic=list(gen.TOPICS)[i % 3])[0]):
                try:
                    extra.append(f(f"extra-{i}"))
                except AssertionError:  # generator occasionally overflows max_len
                    pass
        rows = rows + [r for r in extra if tuple(r["ids"]) not in held]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    t0 = time.time()
    model.train()
    for step in range(a.steps):
        batch = random.sample(rows, a.bs)
        ids, lengths = batchify(batch)
        logits, _ = model(ids)
        w = weights(batch, ids, lengths, a.ans_w)
        loss = (F.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none") * w).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == a.steps - 1:
            model.eval()
            print(f"step {step} loss {loss.item():.3f} acc {test_acc(model)} t={time.time()-t0:.0f}s", flush=True)
            model.train()
    secs = time.time() - t0
    model.eval()
    save(model, runs / "base.pt")
    acc = test_acc(load(runs / "base.pt"))
    m = {"test_acc": acc, "steps": a.steps, "seconds": secs, "params": params}
    (runs / "train_metrics.json").write_text(json.dumps(m, indent=1))
    print(m)


if __name__ == "__main__":
    main()
