"""TinyMamba2: a small Mamba-2-style selective SSM with an explicit sequential recurrence."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 63
    d_model: int = 64
    n_layer: int = 2
    n_head: int = 4
    head_dim: int = 16  # P; d_inner = n_head * head_dim
    d_state: int = 8  # N
    max_len: int = 32
    d_conv: int = 4


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H, P, N = cfg.n_head, cfg.head_dim, cfg.d_state
        self.d_inner = H * P
        self.norm = RMSNorm(cfg.d_model)
        # x, z, B, C, dt
        self.in_proj = nn.Linear(cfg.d_model, 2 * self.d_inner + 2 * N + H, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, cfg.d_conv, groups=self.d_inner, padding=cfg.d_conv - 1)
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(torch.empty(H).uniform_(0.01, 0.5))))
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(0.5, 8.0)))
        self.D = nn.Parameter(torch.ones(H))
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

    def forward(self, u, layer, state_hook, cache):
        cfg = self.cfg
        Bsz, T, _ = u.shape
        H, P, N = cfg.n_head, cfg.head_dim, cfg.d_state
        proj = self.in_proj(self.norm(u))
        x, z, Bm, Cm, dt = torch.split(proj, [self.d_inner, self.d_inner, N, N, H], dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)  # causal depthwise conv
        x = F.silu(x).view(Bsz, T, H, P)
        dt = F.softplus(dt + self.dt_bias)  # [B,T,H]
        a = torch.exp(-dt * torch.exp(self.A_log))  # [B,T,H]
        h = x.new_zeros(Bsz, H, P, N)
        hs, ys = [], []
        for t in range(T):
            h = a[:, t, :, None, None] * h + x[:, t, :, :, None] * Bm[:, t, None, None, :]
            if state_hook is not None:
                h = state_hook(layer, t, h)
            y = (h * Cm[:, t, None, None, :]).sum(-1) + self.D[None, :, None] * x[:, t]
            hs.append(h)
            ys.append(y)
        y = torch.stack(ys, 1)  # [B,T,H,P]
        out = self.out_proj((y.reshape(Bsz, T, -1)) * F.silu(z))
        if cache is not None:
            cache["h"].append(torch.stack(hs, 1))
            cache["a"].append(a)
            cache["dt"].append(dt)
            cache["B"].append(Bm)
            cache["C"].append(Cm)
            cache["x"].append(x)
            cache["y"].append(y)
        return u + out


class TinyMamba2(nn.Module):
    def __init__(self, cfg: Config | None = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.embed = nn.Embedding(self.cfg.vocab, self.cfg.d_model)
        self.layers = nn.ModuleList(Block(self.cfg) for _ in range(self.cfg.n_layer))
        self.norm_f = RMSNorm(self.cfg.d_model)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab, bias=False)
        self.lm_head.weight = self.embed.weight  # tied
        nn.init.normal_(self.embed.weight, std=0.1)

    def forward(self, ids, *, state_hook=None, return_cache: bool = False):
        cache = None
        if return_cache:
            cache = {k: [] for k in ("h", "a", "dt", "B", "C", "x", "y")}
            cache["resid"] = []
        u = self.embed(ids)
        for l, blk in enumerate(self.layers):
            if cache is not None:
                cache["resid"].append(u)
            u = blk(u, l, state_hook, cache)
        if cache is not None:
            cache["resid"].append(u)
        return self.lm_head(self.norm_f(u)), cache


def save(model: TinyMamba2, path):
    torch.save({"cfg": asdict(model.cfg), "state_dict": model.state_dict()}, path)


def load(path: str | Path) -> TinyMamba2:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = TinyMamba2(Config(**ck["cfg"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval()
