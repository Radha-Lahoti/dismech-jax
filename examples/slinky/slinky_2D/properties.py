from dataclasses import dataclass

import jax

@dataclass
class Properties:
    length: float | None = None
    r0: float = 0.005
    axs: float | None = None
    jxs: float | None = None
    ixs1: float | None = None
    ixs2: float | None = None
    density: float = 1000.0
    E: float = 1e6
    N: int = 20
    start: jax.Array | None = None
    end: jax.Array | None = None
    mass: float | None = None


## =========================================================
# DLO properties
# =========================================================
@dataclass
class SlinkyN3Properties(Properties):
    length: float = 0.20
    r0: float = 0.005
    density: float = 1200.0
    E: float = 1e6
    N: int = 3
    mass: float = 0.3

@dataclass
class SlinkyN5Properties(Properties):
    length: float = 0.33
    r0: float = 0.005
    density: float = 1200.0
    E: float = 1e6
    N: int = 7 # since we add clamped points at the ends artificially in the data
    mass: float = 0.1

@dataclass
class StripN9Properties(Properties):
    length: float = 0.55
    r0: float = 0.005
    density: float = 700.0
    E: float = 1e6
    N: int = 9

@dataclass
class TubeN7Properties(Properties):
    length: float = 0.45
    r0: float = 0.005
    density: float = 700.0
    E: float = 1e6
    N: int = 7