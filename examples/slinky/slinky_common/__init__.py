"""
Shared utilities for the ``examples/slinky`` notebooks.

Import from here when the notebook working directory is ``examples/slinky``::

    from slinky_common.jax_math import inv_softplus
    from slinky_common.direct_bc import DirectBCFactory, make_bc_from_testcase
    from slinky_common.energy_plots import plot_energy_2d_from_traj
"""

from .jax_math import inv_softplus
from .direct_bc import DirectBCFactory, make_bc_from_testcase
from .energy_plots import (
    QSpaceEnergySlice,
    plot_energy_1d_from_traj,
    plot_energy_2d_from_traj,
    plot_energy_xz_slice,
)

__all__ = [
    "DirectBCFactory",
    "QSpaceEnergySlice",
    "inv_softplus",
    "make_bc_from_testcase",
    "plot_energy_1d_from_traj",
    "plot_energy_2d_from_traj",
    "plot_energy_xz_slice",
]
