"""Patch = one point in the S-space design space. The loop searches over KNOBS."""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

N_LAYER = 2  # sspace.model.Config.n_layer; used only to canonicalise layer=-1 vs layer=N_LAYER-1

KNOBS = {
    "locus": ["raw", "readout", "gates", "joint"],
    "instrument": ["sae", "jacobian", "covariance", "probe", "gramian"],
    "phi": ["linear", "lowrank", "mlp"],
    "rank": [4, 8, 16, 32],
    "granularity": ["head", "layer", "model"],
    "layer": [0, 1, -1],  # used when granularity != "model"; -1 = last
    "tau_coord": [False, True],
    "modulation": [False, True],
    "seed": [0, 1, 2],
}


@dataclass(frozen=True)
class Patch:
    locus: str = "raw"
    instrument: str = "sae"
    phi: str = "linear"
    rank: int = 4
    granularity: str = "head"
    layer: int = 0
    tau_coord: bool = False
    modulation: bool = False
    seed: int = 0

    def __post_init__(self):
        for k, opts in KNOBS.items():
            v = getattr(self, k)
            if v not in opts:
                raise ValueError(f"knob {k}={v!r} not in {opts}")
        if self.granularity == "model" and self.layer != -1:
            # layer is unused for model granularity: canonicalise so redundant patches share a key
            object.__setattr__(self, "layer", -1)
        if self.layer == N_LAYER - 1:
            # the last layer has two names (N_LAYER-1 and -1); canonicalise to -1
            object.__setattr__(self, "layer", -1)

    def to_dict(self) -> dict:
        return asdict(self)

    def key(self) -> str:
        s = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:10]

    @classmethod
    def from_dict(cls, d: dict) -> "Patch":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown knobs: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_toml(cls, path) -> "Patch":
        with open(Path(path), "rb") as f:
            d = tomllib.load(f)
        return cls.from_dict(d.get("patch", d))


def default() -> Patch:
    return Patch()
