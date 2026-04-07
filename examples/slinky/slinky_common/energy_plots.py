"""
Matplotlib visualization helpers for learned strain energy models.

These were previously copy-pasted across several 3D slinky matrix notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_energy_2d_from_traj(
    model: Any,
    strain_trajs: jax.Array | list[jax.Array] | tuple[jax.Array, ...],
    idx1: int,
    idx2: int,
    *,
    pad: float = 0.05,
    n: int = 200,
    center: str | int = "mean",
) -> None:
    """
    Contour plot of ``model(strain)`` in the plane spanned by two strain components.

    ``strain_trajs`` may be a single array ``(N, 5)`` or a list of trajectories;
    all points set the plotting window and (by default) the fixed reference strain.
    """
    if isinstance(strain_trajs, (list, tuple)):
        traj_list = [jnp.asarray(tr) for tr in strain_trajs]
        all_strains = jnp.concatenate(traj_list, axis=0)
    else:
        traj_list = [jnp.asarray(strain_trajs)]
        all_strains = traj_list[0]

    if center == "mean":
        strain_ref = jnp.mean(all_strains, axis=0)
    else:
        strain_ref = all_strains[int(center)]

    s1_data_all = all_strains[:, idx1]
    s2_data_all = all_strains[:, idx2]

    s1_vals = jnp.linspace(s1_data_all.min() - pad, s1_data_all.max() + pad, n)
    s2_vals = jnp.linspace(s2_data_all.min() - pad, s2_data_all.max() + pad, n)

    s1, s2 = jnp.meshgrid(s1_vals, s2_vals, indexing="xy")

    def energy_at(a: jax.Array, b: jax.Array) -> jax.Array:
        strain = strain_ref.at[idx1].set(a).at[idx2].set(b)
        return model(strain)

    e_grid = jax.vmap(lambda row_s1, row_s2: jax.vmap(energy_at)(row_s1, row_s2))(s1, s2)

    s1_np = np.array(s1)
    s2_np = np.array(s2)
    e_np = np.array(e_grid)

    plt.figure(figsize=(6, 5))
    cp = plt.contourf(s1_np, s2_np, e_np, levels=40)
    plt.colorbar(cp, label="Energy")

    for k, traj in enumerate(traj_list):
        c1 = np.array(traj[:, idx1])
        c2 = np.array(traj[:, idx2])
        color = f"C{k}"
        plt.plot(c1, c2, "-o", ms=3, lw=1.5, color=color, label="trajectory" if k == 0 else None)
        plt.plot(c1[0], c2[0], marker="o", markersize=8, color=color, linestyle="None", label="start" if k == 0 else None)
        plt.plot(c1[-1], c2[-1], marker="*", markersize=12, color=color, linestyle="None", label="end" if k == 0 else None)

        step = max(1, len(c1) // 12)
        for j in range(0, len(c1) - 1, step):
            plt.annotate(
                "",
                xy=(c1[j + 1], c2[j + 1]),
                xytext=(c1[j], c2[j]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.8,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=14,
                ),
                zorder=5,
            )

    plt.xlabel(f"del_strain[{idx1}]")
    plt.ylabel(f"del_strain[{idx2}]")
    plt.title("Energy slice (trajectory-based window)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_energy_1d_from_traj(
    model: Any,
    strain_trajs: jax.Array | list[jax.Array] | tuple[jax.Array, ...],
    idx: int,
    *,
    pad: float = 0.05,
    n: int = 200,
    center: str | int = "mean",
) -> None:
    """Plot scalar energy along one strain direction, overlaying trajectory samples."""
    if isinstance(strain_trajs, (list, tuple)):
        traj_list = [jnp.asarray(tr) for tr in strain_trajs]
        all_strains = jnp.concatenate(traj_list, axis=0)
    else:
        traj_list = [jnp.asarray(strain_trajs)]
        all_strains = traj_list[0]

    if center == "mean":
        strain_ref = jnp.mean(all_strains, axis=0)
    else:
        strain_ref = all_strains[int(center)]

    s_data_all = all_strains[:, idx]
    s_vals = jnp.linspace(s_data_all.min() - pad, s_data_all.max() + pad, n)

    def energy_at(s: jax.Array) -> jax.Array:
        strain = strain_ref.at[idx].set(s)
        return model(strain)

    e_line = jax.vmap(energy_at)(s_vals)

    plt.figure(figsize=(6, 4))
    plt.plot(np.array(s_vals), np.array(e_line), label="Energy")

    for k, traj in enumerate(traj_list):
        s_data = traj[:, idx]
        e_data = jax.vmap(energy_at)(s_data)
        s_np = np.array(s_data)
        e_np = np.array(e_data)
        plt.scatter(s_np, e_np, c="red", s=12, label="trajectory points" if k == 0 else None)
        plt.plot(s_np[-1], e_np[-1], marker="*", markersize=12, color="blue", label="end point" if k == 0 else None)

    plt.xlabel(f"del_strain[{idx}]")
    plt.ylabel("Energy")
    plt.title(f"Energy slice along strain[{idx}] (trajectory-based)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


@dataclass
class QSpaceEnergySlice:
    """
    Evaluate total energy on a 2D slice of configuration space (two free coordinates).

    Typical use: fix all but two entries of ``q`` (e.g. in-plane marker positions)
    and visualize the learned energy landscape around a predicted trajectory.
    """

    rods: Any
    model: Any
    aux0: Any
    q_ref: jax.Array
    lam_ref: jax.Array
    x_dof: int = 4
    z_dof: int = 6

    def energy_at(self, x: jax.Array, z: jax.Array) -> jax.Array:
        q = self.q_ref.at[self.x_dof].set(x).at[self.z_dof].set(z)
        return self.rods.get_E(self.lam_ref, q, self.model, self.aux0)


def plot_energy_xz_slice(
    slice_eval: QSpaceEnergySlice,
    pred: jax.Array,
    *,
    case_idx: int = 0,
    x_pad: float = 0.02,
    z_pad: float = 0.02,
    grid_n: int = 120,
) -> None:
    """
    Contour plot in ``(x, z)`` using ``pred[case_idx, :, x_dof]`` and ``pred[..., z_dof]``
    to set bounds and overlay the trajectory.
    """
    x_traj = pred[case_idx, :, slice_eval.x_dof]
    z_traj = pred[case_idx, :, slice_eval.z_dof]

    x_vals = jnp.linspace(x_traj.min() - x_pad, x_traj.max() + x_pad, grid_n)
    z_vals = jnp.linspace(z_traj.min() - z_pad, z_traj.max() + z_pad, grid_n)
    x_grid, z_grid = jnp.meshgrid(x_vals, z_vals, indexing="ij")

    def row_energy(x_row: jax.Array, z_row: jax.Array) -> jax.Array:
        return jax.vmap(lambda xv, zv: slice_eval.energy_at(xv, zv))(x_row, z_row)

    e_grid = jax.vmap(row_energy)(x_grid, z_grid)

    plt.figure(figsize=(6, 5))
    cf = plt.contourf(np.array(x_grid), np.array(z_grid), np.array(e_grid), levels=40)
    plt.xlabel("x coordinate")
    plt.ylabel("z coordinate")
    plt.title("Energy landscape in (x, z)")
    plt.colorbar(cf, label="Energy")
    plt.plot(np.array(x_traj), np.array(z_traj), "r.-", markersize=4, linewidth=1)
    plt.tight_layout()
    plt.show()
