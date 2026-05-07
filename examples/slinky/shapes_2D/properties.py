from dataclasses import dataclass, field

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
    density: float = 800.0
    E: float = 1e6
    N: int = 3
    mass: float = 0.2

@dataclass
class SlinkyN5Properties(Properties):
    length: float = 0.33
    r0: float = 0.005
    density: float = 1200.0
    E: float = 1e6
    N: int = 7 # since we add clamped points at the ends artificially in the data
    mass: float = 0.05

@dataclass
class StripN9Properties(Properties):
    length: float = 0.55
    r0: float = 0.005
    density: float = 700.0
    E: float = 1e6
    N: int = 9

@dataclass
class TubeN7Properties(Properties):
    # length: float = 0.5
    start: jax.Array = field(default_factory=lambda: jax.numpy.array([0.0, 0.0, 0.0]))
    end: jax.Array = field(default_factory=lambda: jax.numpy.array([0.46, 0.0, 0.03]))
    r0: float = 0.005
    density: float = 600.0
    E: float = 1e6
    N: int = 7

@dataclass
class TapeN11Properties(Properties):
    # length: float = 1.2
    start: jax.Array = field(default_factory=lambda: jax.numpy.array([0.0, 0.0, 0.0]))
    end: jax.Array = field(default_factory=lambda: jax.numpy.array([1.05660479, 0.0, 0.04239192]))
    r0: float = 0.005
    density: float = 600.0
    E: float = 1e6
    N: int = 11
    mass: float = 0.005
