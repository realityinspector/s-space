"""Evaluate a Patch: how readable / persistent / causal / verbal is the S-space it defines?

Pipeline (per checkpoint):  features(locus, granularity) -> instrument (basis/encoder, rank r)
-> S-coordinates (+ optional tau / modulation coords) -> phi (linear | lowrank | mlp) -> per-position label.

Approximations (see also instruments in this file):
  * head choice (granularity="head"): the head with best *train* accuracy of a ridge-onehot readout on
    PCA(rank) coordinates of that head's features (cheap proxy for "best train readable score").
  * modulation features: temporal first difference of the gates (dt, B, C) of the selected layers, i.e. a
    finite-difference estimate of how the input modulates the gates. Appended to the S-coordinates.
  * tau_coord: log tau = -log(-log a_t) per selected head, appended to the S-coordinates.
  * Non-raw loci for causal patching: the S-subspace lives in feature space; it is pulled back to state space
    through a ridge regression feature = state @ W (state directions = W q), then orthonormalised.
  * tau / modulation coords are input-derived and are NOT patched by the causal test.
  * jacobian instrument: gradients of {target log-prob, target logit, a random logit direction} at query_pos
    w.r.t. the state at earlier positions (via a zero-delta state_hook), a stand-in for the logit Jacobian rows.
  * seed only affects instrument / phi randomness (SAE init, phi init, random-subspace draw), never data/model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import ROOT, TASKS, batchify, load_cf, load_split, tok2id
from .patch import Patch, default

torch.set_num_threads(4)

THRESHOLDS = {"readable_vs_untrained": 0.10, "readable_vs_shuffled": 0.10, "delta_R": 0.02,
              "causal_vs_random": 0.10, "heldout_ratio": 0.8}
TRAIN_CAP = 2000   # max train positions per task
JAC_EX = 24        # examples per task for the jacobian instrument


def _load_model(path):
    impl = os.environ.get("SSPACE_MODEL_IMPL")  # test hook: module name providing load()
    if impl:
        import importlib
        return importlib.import_module(impl).load(path)
    from .model import load
    return load(path)


# ----------------------------------------------------------------------------- data / caches
class Ctx:
    """Flattened labelled positions of one split under one checkpoint (all arrays indexed by sample m)."""


_CTX: dict = {}
_JAC: dict = {}
_MODEL: dict = {}


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _ckpt_key(path):
    p = _abs(path)
    return (str(p), p.stat().st_mtime_ns if p.exists() else 0)


def get_model(path):
    k = _ckpt_key(path)
    if k not in _MODEL:
        _MODEL[k] = _load_model(_abs(path))
    return _MODEL[k]


def _label_positions(r):
    """Yield (t, label_str) of labelled positions t <= query_pos for a row."""
    q = r["query_pos"]
    out = []
    for t, lab in enumerate(r["labels"][: q + 1]):
        if lab is None:
            continue
        if r["task"] == "entity":
            if lab != r["labels"][q]:
                continue  # only positions after the last binding
            lab = r["target"]
        out.append((t, str(lab)))
    return out


def get_ctx(path, split):
    k = (_ckpt_key(path), split)
    if k in _CTX:
        return _CTX[k]
    model = get_model(path)
    rows = load_split(split)
    rng = np.random.RandomState(0)
    items = []  # (row_idx, t, label)
    for i, r in enumerate(rows):
        for t, lab in _label_positions(r):
            items.append((i, t, lab))
    if split == "train":
        keep = []
        for task in TASKS:
            idx = [j for j, it in enumerate(items) if rows[it[0]]["task"] == task]
            if len(idx) > TRAIN_CAP:
                idx = list(rng.choice(idx, TRAIN_CAP, replace=False))
            keep += idx
        items = [items[j] for j in sorted(keep)]
    ri = np.array([it[0] for it in items]); ti = np.array([it[1] for it in items])
    c = Ctx()
    c.rows, c.ri, c.ti = rows, ri, ti
    c.label = [it[2] for it in items]
    c.task = np.array([rows[i]["task"] for i in ri])
    c.isq = np.array([t == rows[i]["query_pos"] for i, t in zip(ri, ti)])
    c.rel = np.array([t - 1 for t in ti])  # tokens after the key write (recall: key at pos 1)
    c.band = np.array([rows[i]["meta"].get("band", "") for i in ri])
    c.lag = np.array([rows[i]["meta"].get("lag", 0) for i in ri])
    keys = ["h", "a", "dt", "B", "C", "x", "y"]
    parts = {k_: [] for k_ in keys + ["dg", "resid"]}
    argmax = np.zeros(len(rows), dtype=int)
    with torch.no_grad():
        for s in range(0, len(rows), 250):
            sub = rows[s: s + 250]
            ids, _ = batchify(sub)
            logits, cache = model(ids, return_cache=True)
            for j, r in enumerate(sub):
                argmax[s + j] = int(logits[j, r["query_pos"]].argmax())
            sel = (ri >= s) & (ri < s + 250)
            b = torch.from_numpy(ri[sel] - s); t_ = torch.from_numpy(ti[sel])
            tp = (t_ - 1).clamp(min=0); z = (t_ > 0).float()
            for k_ in keys:
                parts[k_].append(torch.stack([cache[k_][l][b, t_] for l in range(len(cache[k_]))], 1))
            gl = []
            for l in range(len(cache["h"])):
                g = torch.cat([cache["dt"][l], cache["B"][l], cache["C"][l]], -1)
                gl.append((g[b, t_] - g[b, tp]) * z[:, None])
            parts["dg"].append(torch.stack(gl, 1))
            parts["resid"].append(torch.stack([r_[b, t_] for r_ in cache["resid"]], 1))
    for k_, v in parts.items():
        setattr(c, k_, torch.cat(v))
    c.argmax_q = argmax
    c.L = c.h.shape[1]
    _CTX[k] = c
    return c


def get_jac(path, ctx_tr):
    """Gradient samples G [n, L, H*P*N] and their task (see module docstring)."""
    k = _ckpt_key(path)
    if k in _JAC:
        return _JAC[k]
    model = get_model(path)
    rows = ctx_tr.rows
    g = torch.Generator().manual_seed(0)
    G, tasks = [], []
    for task in TASKS:
        rs = [r for r in rows if r["task"] == task][:JAC_EX]
        ids, _ = batchify(rs)
        T = ids.shape[1]
        qp = torch.tensor([r["query_pos"] for r in rs]); tgt = torch.tensor([r["ids"][r["query_pos"] + 1] for r in rs])
        rnd = torch.randn(len(rs), model.cfg.vocab, generator=g)
        for mode in range(3):
            deltas = {}
            def hook(layer, t, h):
                d = torch.zeros_like(h, requires_grad=True)
                deltas[(layer, t)] = d
                return h + d
            logits, _ = model(ids, state_hook=hook)
            lq = logits[torch.arange(len(rs)), qp]
            if mode == 0:
                f = F.log_softmax(lq, -1)[torch.arange(len(rs)), tgt]
            elif mode == 1:
                f = lq[torch.arange(len(rs)), tgt]
            else:
                f = (lq * rnd).sum(-1)
            keys = list(deltas)
            grads = torch.autograd.grad(f.sum(), [deltas[k_] for k_ in keys], allow_unused=True)
            gd = {k_: gr for k_, gr in zip(keys, grads)}
            for i in range(len(rs)):
                for t in range(1, int(qp[i]) + 1):
                    row = torch.stack([gd[(l, t)][i].reshape(-1) if gd[(l, t)] is not None else torch.zeros(deltas[(l, t)][i].numel())
                                       for l in range(model.cfg.n_layer)])
                    G.append(row); tasks.append(task)
    out = (torch.stack(G).detach(), np.array(tasks))
    _JAC[k] = out
    return out


# ----------------------------------------------------------------------------- feature construction
class Spec:
    """Which slice of the model a patch looks at."""

    def __init__(self, patch: Patch, L: int, H: int, head: int | None, locus: str | None = None):
        self.locus = locus or patch.locus
        self.layers = list(range(L)) if patch.granularity == "model" else [patch.layer % L]
        self.head = head if patch.granularity == "head" else None
        self.resid_idx = L if patch.granularity == "model" else self.layers[0] + 1
        self.H = H
        self.tau, self.mod = patch.tau_coord, patch.modulation

    def hs(self, hh):  # [M,L,H,P,N] -> sel state flat [M, D]
        x = hh[:, self.layers]
        if self.head is not None:
            x = x[:, :, self.head]
        return x.reshape(x.shape[0], -1)

    def _heads(self):
        return [self.head] if self.head is not None else list(range(self.H))

    def feats(self, c: Ctx):
        """Locus features F [M, D] and selected state S [M, Ds]."""
        S = self.hs(c.h)
        hd = self._heads()
        if self.locus == "raw":
            Fv = S
        elif self.locus == "readout":
            Fv = c.y[:, self.layers][:, :, hd].reshape(len(S), -1)
        else:
            g = torch.cat([c.dt[:, self.layers][:, :, hd].reshape(len(S), -1),
                           c.a[:, self.layers][:, :, hd].reshape(len(S), -1),
                           c.B[:, self.layers].reshape(len(S), -1), c.C[:, self.layers].reshape(len(S), -1)], 1)
            if self.locus == "gates":
                Fv = g
            else:  # joint: resid + state
                Fv = torch.cat([c.resid[:, self.resid_idx], S], 1)
        return Fv, S

    def extras(self, c: Ctx):
        e = []
        hd = self._heads()
        if self.tau:
            a = c.a[:, self.layers][:, :, hd].clamp(1e-6, 1 - 1e-6)
            e.append((-torch.log(-torch.log(a))).reshape(len(a), -1))  # log tau
        if self.mod:
            dg = c.dg[:, self.layers]  # [M,l,H+2N]
            H = c.a.shape[2]
            e.append(torch.cat([dg[:, :, hd].reshape(len(dg), -1), dg[:, :, H:].reshape(len(dg), -1)], 1))
        return torch.cat(e, 1) if e else None


def _std(X, mu=None, sd=None):
    if mu is None:
        mu = X.mean(0); sd = X.std(0)
        sd = sd + 0.05 * sd.mean() + 1e-8
    return (X - mu) / sd, mu, sd


def _ridge(A, B, lam=1e-2):
    """argmin ||A W - B||^2 + lam*tr-scaled ridge."""
    G = A.T @ A
    G = G + lam * G.diagonal().mean() * torch.eye(G.shape[0])
    return torch.linalg.solve(G, A.T @ B)


# ----------------------------------------------------------------------------- instruments
class Instrument:
    """encode(F_raw, S_raw) -> S-coords; dirs(): state-space (enc, dec) matrices for causal patching."""

    def __init__(self, name, rank, seed):
        self.name, self.rank, self.seed = name, rank, seed
        self.enc_dirs = None  # feature-space directions Q [D, r]

    def fit(self, Fv, S, mask, labels, jac, spec, ctx):
        """Fv/S: train features; mask: bool rows usable (held-out tasks excluded); labels: task->(rows, ylab, C)."""
        r, name = self.rank, self.name
        g = torch.Generator().manual_seed(self.seed)
        self.mu_f = Fv[mask].mean(0)
        sd = Fv[mask].std(0)
        self.sd_f = sd + 0.05 * sd.mean() + 1e-8
        X = (Fv[mask] - self.mu_f) / self.sd_f
        self.spec = spec
        if name == "covariance":
            # PCA: top-r right singular vectors of centred standardised features
            _, _, V = torch.linalg.svd(X, full_matrices=False)
            self.Q = V[:r].T
        elif name == "probe":
            # supervised: multinomial-logistic probe per task, SVD of the stacked probe weights, top-r directions
            Ws = []
            for task, (rows, y, C) in labels.items():
                rows = rows[mask[rows]] if False else rows
                m = mask[rows]
                if m.sum() < 5 or C < 2:
                    continue
                Xt = (Fv[rows[m]] - self.mu_f) / self.sd_f
                Ws.append(_fit_logreg(Xt, y[m], C, wd=1e-2, iters=40))
            W = torch.cat(Ws, 1)
            U, _, _ = torch.linalg.svd(W, full_matrices=False)
            self.Q = U[:, :r]
        elif name == "sae":
            # tiny ReLU SAE: dictionary 4r, L1 on codes, 300 Adam steps; Q = top-r right sing. vectors of
            # activation-weighted decoder (state/feature-space directions of the dominant atoms).
            nf = 4 * r
            torch.manual_seed(self.seed)
            D = X.shape[1]
            We = torch.nn.Parameter(torch.randn(D, nf) / D ** 0.5); be = torch.nn.Parameter(torch.zeros(nf))
            Wd = torch.nn.Parameter(torch.randn(nf, D) / nf ** 0.5)
            opt = torch.optim.Adam([We, be, Wd], lr=3e-3)
            for _ in range(300):
                idx = torch.randint(0, len(X), (min(1024, len(X)),), generator=g)
                xb = X[idx]
                z = F.relu(xb @ We + be)
                loss = F.mse_loss(z @ Wd, xb) + 3e-3 * z.abs().mean() * nf ** 0.5
                opt.zero_grad(); loss.backward(); opt.step()
            self.We, self.be = We.detach(), be.detach()
            with torch.no_grad():
                act = F.relu(X @ self.We + self.be).mean(0)
                _, _, V = torch.linalg.svd(Wd.detach() * (act[:, None] + 1e-6), full_matrices=False)
            self.Q = V[:r].T
        elif name == "jacobian":
            # SVD of logit-gradient samples mapped into feature space (grad_F = R grad_S, R: F -> S ridge fit)
            G, gt = jac
            gm = torch.from_numpy(np.isin(gt, [t for t in labels]))
            Gs = spec.hs(G.reshape(len(G), G.shape[1], ctx.h.shape[2], *ctx.h.shape[3:]))[gm]
            if spec.locus == "raw":
                GF = Gs * self.sd_f  # dS/dF_std = sd
            else:
                Rm = _ridge(X, (S[mask] - S[mask].mean(0)), 1e-2)  # F_std -> S
                GF = Gs @ Rm.T
            _, _, V = torch.linalg.svd(GF, full_matrices=False)
            self.Q = V[:r].T
        elif name == "gramian":
            self._fit_gramian(S, mask, spec, ctx)
            return self
        else:
            raise ValueError(name)
        if spec.locus != "raw":
            self.Wreg = _ridge(S[mask] - S[mask].mean(0), X, 1e-2)  # state -> F_std (for pullback)
            self.mu_s = S[mask].mean(0)
        return self

    def _fit_gramian(self, S, mask, spec, ctx):
        """Empirical balanced truncation in state space.
        W_c = E[(x⊗B)(x⊗B)^T] / (1 - abar^2) per head (diag-decay discounted sum of written increments),
        W_o = I_P ⊗ E[C C^T] / (1 - abar^2) per head (readout y = h·C is linear in h; exact, empirical C).
        Balanced modes via eig of Wc^1/2 Wo Wc^1/2; coordinates z = T^-1 h (top-r Hankel modes).
        Locus-independent: S-coordinates are always balanced state coordinates."""
        c = ctx
        r = self.rank
        hd = spec._heads()
        Hn, P, N = c.h.shape[2], c.h.shape[3], c.h.shape[4]
        m = torch.from_numpy(mask)
        us, blocks = [], []
        for l in spec.layers:
            u = c.x[m, l][:, :, :, None] * c.B[m, l][:, None, None, :]  # [M,H,P,N]
            abar = c.a[m, l].mean(0)  # [H]
            disc = 1.0 / (1 - abar ** 2).clamp(min=1e-3)
            u = u * disc.sqrt()[None, :, None, None]
            Cc = (c.C[m, l][:, :, None] * c.C[m, l][:, None, :]).mean(0)  # [N,N]
            for h in hd:
                blocks.append(torch.kron(torch.eye(P), Cc) * disc[h])
            us.append(u[:, hd].reshape(len(u), -1))
        U = torch.cat(us, 1)
        Wc = U.T @ U / len(U)
        Dh = Wc.shape[0]
        Wo = torch.block_diag(*blocks)
        Wc = Wc + 1e-6 * Wc.diagonal().mean() * torch.eye(Dh)
        lam, V = torch.linalg.eigh(Wc)
        lam = lam.clamp(min=1e-10)
        Sh = V @ torch.diag(lam.sqrt()) @ V.T
        Shi = V @ torch.diag(lam.rsqrt()) @ V.T
        Lam, Qm = torch.linalg.eigh(Sh @ Wo @ Sh)
        Lam, Qm = Lam.flip(0).clamp(min=1e-12), Qm.flip(1)
        Tm = Sh @ Qm[:, :r] * Lam[:r].pow(-0.25)
        Tinv = (Lam[:r].pow(0.25)[:, None] * Qm[:, :r].T) @ Shi  # [r, Dh]
        self.Enc, self.Dec = Tinv.T, Tm
        self.hankel = Lam[:r].sqrt()

    def encode(self, Fv, S):
        if self.name == "gramian":
            return S @ self.Enc
        X = (Fv - self.mu_f) / self.sd_f
        if self.name == "sae":
            return F.relu(X @ self.We + self.be)
        return X @ self.Q

    def state_proj(self):
        """(Enc_s [Ds,k], Dec_s [Ds,k]): patch Δ -> Dec_s Enc_s^T Δ in selected-state space."""
        if self.name == "gramian":
            return self.Enc, self.Dec
        if self.spec.locus == "raw":
            E = self.Q / self.sd_f[:, None]
        else:
            E = self.Wreg @ self.Q
        V, _ = torch.linalg.qr(E)
        return V, V


def _fit_logreg(X, y, C, wd=1e-3, iters=50, k=None, hidden=None, seed=0):
    """Multinomial logistic (optionally rank-k bottleneck or 1-hidden-layer MLP). Returns weights or module."""
    torch.manual_seed(seed)
    D = X.shape[1]
    if hidden:
        net = torch.nn.Sequential(torch.nn.Linear(D, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, C))
    elif k:
        net = torch.nn.Sequential(torch.nn.Linear(D, k, bias=False), torch.nn.Linear(k, C))
    else:
        net = torch.nn.Linear(D, C)
    ps = list(net.parameters())
    opt = torch.optim.LBFGS(ps, lr=1, max_iter=iters, history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(net(X), y) + wd * sum((p ** 2).sum() for p in ps if p.dim() > 1)
        loss.backward()
        return loss
    opt.step(closure)
    if hidden or k:
        return net
    return net.weight.detach().T  # [D, C] for probe use / logistic


def _phi_fit_predict(kind, Xtr, ytr, C, Xte, rank, seed):
    if kind == "linear":
        net = torch.nn.Linear(Xtr.shape[1], C)
        _train(net, Xtr, ytr, 60, seed)
    elif kind == "lowrank":
        net = _lr_net(Xtr.shape[1], C, max(2, min(Xtr.shape[1], rank // 4)), seed)
        _train(net, Xtr, ytr, 80, seed)
    else:
        net = _mlp_net(Xtr.shape[1], C, 48, seed)
        _train(net, Xtr, ytr, 100, seed)
    with torch.no_grad():
        return net(Xte).argmax(-1)


def _lr_net(D, C, k, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(D, k, bias=False), torch.nn.Linear(k, C))


def _mlp_net(D, C, h, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(D, h), torch.nn.Tanh(), torch.nn.Linear(h, C))


def _train(net, X, y, iters, seed):
    torch.manual_seed(seed)
    ps = list(net.parameters())
    opt = torch.optim.LBFGS(ps, lr=1, max_iter=iters, history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(net(X), y) + 1e-3 * sum((p ** 2).sum() for p in ps if p.dim() > 1)
        loss.backward()
        return loss
    opt.step(closure)


# ----------------------------------------------------------------------------- readability pipeline
def _task_labels(ctx_tr, ctx_te, perm_seed=None):
    """Per task: (train rows idx, train y, C, test rows idx, test y (-1 unseen))."""
    out = {}
    rng = np.random.RandomState(perm_seed) if perm_seed is not None else None
    for task in TASKS:
        tr = np.where(ctx_tr.task == task)[0]; te = np.where(ctx_te.task == task)[0]
        classes = sorted(set(ctx_tr.label[i] for i in tr))
        cid = {c: i for i, c in enumerate(classes)}
        ytr = np.array([cid[ctx_tr.label[i]] for i in tr])
        if rng is not None:
            ytr = rng.permutation(ytr)
        yte = np.array([cid.get(ctx_te.label[i], -1) for i in te])
        out[task] = (torch.from_numpy(tr), torch.from_numpy(ytr), len(classes), torch.from_numpy(te),
                     torch.from_numpy(yte), classes)
    return out


def _choose_head(patch, tr, ctx_tr, labs):
    """Best head by train accuracy of a ridge-onehot readout on rank-r PCA coordinates (cheap proxy)."""
    H = ctx_tr.h.shape[2]
    best, bs = 0, -1
    for h in range(H):
        sp = Spec(patch, ctx_tr.L, H, h)
        Fv, S = sp.feats(ctx_tr)
        X, _, _ = _std(Fv)
        _, _, V = torch.linalg.svd(X, full_matrices=False)
        Z = X @ V[:patch.rank].T
        Z = torch.cat([Z, torch.ones(len(Z), 1)], 1)
        accs = []
        for task, (rows, y, C, *_r) in labs.items():
            Zt = Z[rows]
            Y = F.one_hot(y, C).float()
            W = _ridge(Zt, Y, 1e-2)
            accs.append(((Zt @ W).argmax(1) == y).float().mean().item())
        s = float(np.mean(accs))
        if s > bs:
            best, bs = h, s
    return best


_HEAD: dict = {}


def run_readable(patch, base_path, head, *, shuffle=False, locus=None, hold=None, return_inst=False,
                 only_resid=False):
    """Fit instrument on train, phi on train, score on test. Returns dict with per-task accuracy etc."""
    ctx_tr, ctx_te = get_ctx(base_path, "train"), get_ctx(base_path, "test")
    labs = _task_labels(ctx_tr, ctx_te, perm_seed=1234 if shuffle else None)
    spec = Spec(patch, ctx_tr.L, ctx_tr.h.shape[2], head, locus)
    if only_resid:
        Ftr, Fte = ctx_tr.resid[:, -1], ctx_te.resid[:, -1]
        Xtr, mu, sd = _std(Ftr); Xte, _, _ = _std(Fte, mu, sd)
        inst, extra_tr, extra_te = None, None, None
    else:
        Ftr, Str = spec.feats(ctx_tr); Fte, Ste = spec.feats(ctx_te)
        mask = np.ones(len(ctx_tr.task), bool)
        if hold is not None:
            mask = ctx_tr.task != hold
        used = {t: labs[t][:3] for t in TASKS if t != hold}
        jac = get_jac(base_path, ctx_tr) if patch.instrument == "jacobian" else None
        inst = Instrument(patch.instrument, patch.rank, patch.seed).fit(Ftr, Str, mask, used, jac, spec, ctx_tr)
        Xtr, Xte = inst.encode(Ftr, Str), inst.encode(Fte, Ste)
        et, ee = spec.extras(ctx_tr), spec.extras(ctx_te)
        if et is not None:
            Xtr, Xte = torch.cat([Xtr, et], 1), torch.cat([Xte, ee], 1)
        Xtr, mu, sd = _std(Xtr); Xte, _, _ = _std(Xte, mu, sd)
    acc, preds = {}, {}
    tasks = [hold] if hold else TASKS
    for task in tasks:
        tr, ytr, C, te, yte, classes = labs[task]
        p = _phi_fit_predict(patch.phi, Xtr[tr], ytr, C, Xte[te], patch.rank, patch.seed)
        acc[task] = (p == yte).float().mean().item()
        preds[task] = (te, p, classes)
    out = {"acc": acc, "preds": preds, "inst": inst, "spec": spec, "readable": float(np.mean(list(acc.values())))}
    return out


def scores_from(res, patch, base_path):
    ctx_te = get_ctx(base_path, "test")
    t2i = tok2id()
    # persistent
    te_all = {}
    ok = np.zeros(len(ctx_te.task), bool)
    for task, (te, p, classes) in res["preds"].items():
        te_n = te.numpy()
        pred_lab = np.array([classes[i] for i in p.tolist()])
        truth = np.array([ctx_te.label[i] for i in te_n])
        ok[te_n] = pred_lab == truth
    rc = ctx_te.task == "recall"
    far = rc & (ctx_te.rel >= 8)
    pers = float(ok[far].mean()) if far.any() else 0.0
    byb = {}
    for b in ("fast", "mid", "slow"):
        m = rc & (ctx_te.band == b) & (ctx_te.rel >= np.minimum(8, ctx_te.lag))
        byb[b] = float(ok[m].mean()) if m.any() else 0.0
    # verbal: lens readout vs model argmax at query_pos
    ag = []
    for task, (te, p, classes) in res["preds"].items():
        te_n = te.numpy(); q = ctx_te.isq[te_n]
        pred_tok = np.array([t2i.get(classes[i], -1) for i in p.tolist()])[q]
        model_tok = ctx_te.argmax_q[ctx_te.ri[te_n][q]]
        ag.append(float((pred_tok == model_tok).mean()))
    return pers, byb, float(np.mean(ag))


# ----------------------------------------------------------------------------- causal
def causal_score(patch, base_path, inst, spec, random_basis=False, seed=0):
    """Swap a's S-space component for b's at the middle of the gap; fraction whose argmax flips to b's key."""
    model = get_model(base_path)
    pairs = load_cf("recall")
    ra = [p["a"] for p in pairs]; rb = [p["b"] for p in pairs]
    ids_a, _ = batchify(ra); ids_b, _ = batchify(rb)
    qp = torch.tensor([r["query_pos"] for r in ra])
    tmid = ((1 + qp) // 2).clamp(min=1)
    tgt_b = torch.tensor([r["ids"][r["query_pos"] + 1] for r in rb])
    with torch.no_grad():
        _, ca = model(ids_a, return_cache=True)
        _, cb = model(ids_b, return_cache=True)
        ar = torch.arange(len(ra))
        def sel_state(cache):
            hh = torch.stack([cache["h"][l][ar, tmid] for l in range(len(cache["h"]))], 1)  # [n,L,H,P,N]
            return spec.hs(hh)
        d = sel_state(cb) - sel_state(ca)  # [n, Ds]
        if random_basis:
            g = torch.Generator().manual_seed(1000 + seed)
            k = inst.state_proj()[0].shape[1]
            V, _ = torch.linalg.qr(torch.randn(d.shape[1], k, generator=g))
            delta = (d @ V) @ V.T
            # norm-match to the real S-space intervention so the control isn't trivially weaker
            E_, Dc_ = inst.state_proj()
            ref = ((d @ E_) @ Dc_.T).norm(dim=1, keepdim=True)
            delta = delta / delta.norm(dim=1, keepdim=True).clamp(min=1e-8) * ref
        else:
            E, Dc = inst.state_proj()
            delta = (d @ E) @ Dc.T
        L = len(spec.layers); H, P, N = ca["h"][0].shape[2:]
        hs = [spec.head] if spec.head is not None else list(range(H))
        per = len(hs) * P * N
        delta = delta.reshape(len(ra), L, len(hs), P, N)

        def hook(layer, t, h):
            if layer not in spec.layers:
                return h
            m = (tmid == t)
            if not m.any():
                return h
            li = spec.layers.index(layer)
            h = h.clone()
            if spec.head is None:
                h[m] = h[m] + delta[m, li]
            else:
                h[m, spec.head] = h[m, spec.head] + delta[m, li, 0]
            return h
        logits, _ = model(ids_a, state_hook=hook)
        pred = logits[ar, qp].argmax(-1)
    return float((pred == tgt_b).float().mean())


# ----------------------------------------------------------------------------- evaluate
def _head_for(patch, base_path):
    if patch.granularity != "head":
        return None
    k = (_ckpt_key(base_path), patch.locus, patch.layer, patch.rank, patch.tau_coord, patch.modulation)
    if k not in _HEAD:
        ctx_tr, ctx_te = get_ctx(base_path, "train"), get_ctx(base_path, "test")
        labs = _task_labels(ctx_tr, ctx_te)
        _HEAD[k] = _choose_head(patch, None, ctx_tr, labs)
    return _HEAD[k]


def _full_scores(patch, path, head):
    res = run_readable(patch, path, head)
    pers, byb, verbal = scores_from(res, patch, path)
    causal = causal_score(patch, path, res["inst"], res["spec"])
    return res, {"readable": res["readable"], "persistent": pers, "persistent_by_band": byb,
                 "causal": causal, "verbal": verbal}


def evaluate(patch: Patch, *, base="runs/base.pt", untrained="runs/untrained.pt", controls=True) -> dict:
    t0 = time.time()
    head = _head_for(patch, base)
    res, sc = _full_scores(patch, base, head)
    out = {"patch": patch.to_dict(), "key": patch.key(), "scores": sc, "controls": {}, "margins": {}, "pass": False}
    out["head"] = head
    if controls:
        ctl = out["controls"]
        h_u = _head_for(patch, untrained)
        _, ctl["untrained"] = _full_scores(patch, untrained, h_u)
        sh = run_readable(patch, base, head, shuffle=True)
        pers, byb, verbal = scores_from(sh, patch, base)
        ctl["shuffled"] = {"readable": sh["readable"], "persistent": pers, "persistent_by_band": byb,
                           "causal": causal_score(patch, base, sh["inst"], sh["spec"]), "verbal": verbal}
        # jlens only: phi on resid[-1] alone vs joint-locus S-space
        rr = run_readable(patch, base, head, only_resid=True)
        if patch.locus == "joint":
            rj = sc["readable"]
        else:
            rj = run_readable(patch, base, head, locus="joint")["readable"]
        ctl["jlens_only"] = {"readable": rr["readable"], "delta_R": rj - rr["readable"]}
        rands = [causal_score(patch, base, res["inst"], res["spec"], True, s) for s in range(3)]
        ctl["random_subspace"] = {"causal": float(np.mean(rands))}
        ho = [run_readable(patch, base, head, hold=t)["acc"][t] for t in TASKS]
        hr = float(np.mean(ho))
        ctl["heldout"] = {"readable": hr, "ratio": hr / max(sc["readable"], 1e-9)}
        m = out["margins"]
        m["readable_vs_untrained"] = sc["readable"] - ctl["untrained"]["readable"] - THRESHOLDS["readable_vs_untrained"]
        m["readable_vs_shuffled"] = sc["readable"] - ctl["shuffled"]["readable"] - THRESHOLDS["readable_vs_shuffled"]
        m["delta_R"] = ctl["jlens_only"]["delta_R"] - THRESHOLDS["delta_R"]
        m["causal_vs_random"] = sc["causal"] - ctl["random_subspace"]["causal"] - THRESHOLDS["causal_vs_random"]
        m["heldout_ratio"] = ctl["heldout"]["ratio"] - THRESHOLDS["heldout_ratio"]
        out["pass"] = bool(all(v >= 0 for v in m.values()))
    out["elapsed_s"] = time.time() - t0
    return out


def _table(r):
    s = r["scores"]
    lines = [f"patch {r['key']}  {r['patch']}  head={r.get('head')}", "",
             f"{'score':<22}{'value':>8}", "-" * 30]
    for k in ("readable", "persistent", "causal", "verbal"):
        lines.append(f"{k:<22}{s[k]:>8.3f}")
    for b, v in s["persistent_by_band"].items():
        lines.append(f"  persistent[{b}]".ljust(22) + f"{v:>8.3f}")
    if r["controls"]:
        c = r["controls"]
        lines += ["", "controls", "-" * 30,
                  f"{'untrained.readable':<22}{c['untrained']['readable']:>8.3f}",
                  f"{'untrained.causal':<22}{c['untrained']['causal']:>8.3f}",
                  f"{'shuffled.readable':<22}{c['shuffled']['readable']:>8.3f}",
                  f"{'jlens_only.readable':<22}{c['jlens_only']['readable']:>8.3f}",
                  f"{'random.causal':<22}{c['random_subspace']['causal']:>8.3f}",
                  f"{'heldout.readable':<22}{c['heldout']['readable']:>8.3f}",
                  f"{'heldout.ratio':<22}{c['heldout']['ratio']:>8.3f}", "",
                  "margins (>=0 clears)", "-" * 30]
        lines += [f"{k:<22}{v:>+8.3f}  {'ok' if v >= 0 else 'FAIL'}" for k, v in r["margins"].items()]
        lines.append(f"\nPASS: {r['pass']}")
    lines.append(f"elapsed {r['elapsed_s']:.1f}s")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patch", nargs="?", default=None)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--base", default="runs/base.pt")
    ap.add_argument("--untrained", default="runs/untrained.pt")
    a = ap.parse_args()
    p = Patch.from_toml(a.patch) if a.patch else default()
    r = evaluate(p, base=a.base, untrained=a.untrained, controls=not a.no_controls)
    print(_table(r))
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / f"eval_{r['key']}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
