import os
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import dismech_jax as djx


# =========================================================
# Config
# =========================================================
@dataclass
class EnergyLandscapeSpec:
    strain_x_idx: int = 0
    strain_y_idx: int = 1
    triplet_idx: int = 0
    traj_idx: int = 0

    # grid / view
    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None
    n_grid: int = 161
    n_levels: int = 18

    # how to freeze non-plotted strain components
    fixed_strains: Optional[np.ndarray] = None
    fixed_mode: str = "first_path_point"   # "zeros" or "first_path_point"

    # plotting
    contour_style: str = "lines"           # "lines", "filled", "both"
    show_colorbar: bool = False
    show_path_points: bool = True
    show_start_end: bool = True
    path_linewidth: float = 1.8
    contour_linewidth: float = 0.9

    # labels
    strain_labels: Optional[Sequence[str]] = None
    title: Optional[str] = None


# =========================================================
# Solve + strain history
# =========================================================
def solve_with_strain_history(
    model,
    base,
    aux,
    idx_b,
    xb,
    lambdas,
    max_dlambda=5e-3,
    iters=5,
    ls_steps=10,
):
    """
    Solve equilibrium and return configuration and reduced strain history.
    """
    if xb.ndim == 2:
        bc = djx.DirectBC(idx_b=idx_b, xb=xb, lambdas=lambdas)
    elif xb.ndim == 3:
        bc = djx.BatchedDirectBC(idx_b=idx_b, xb=xb, lambdas=lambdas)
    else:
        raise ValueError(f"Expected xb.ndim in {{2, 3}}, got shape {xb.shape}")

    rod = base.with_bc(bc)
    qs, auxs = rod.solve_with_aux(
        model,
        lambdas,
        aux,
        max_dlambda=max_dlambda,
        iters=iters,
        ls_steps=ls_steps,
    )
    del_strains = rod.get_del_strain_history(qs, auxs)
    return qs, auxs, del_strains


# =========================================================
# Strain-path extraction
# =========================================================
def extract_strain_path(
    del_strains,
    traj_idx=0,
    triplet_idx=0,
):
    """
    Returns
    -------
    path : (T, n_strain)
    """
    arr = np.asarray(del_strains)

    if arr.ndim == 4:      # (B, T, n_triplets, n_strain)
        arr = arr[traj_idx]
    elif arr.ndim != 3:    # (T, n_triplets, n_strain)
        raise ValueError(
            "del_strains must have shape (T, n_triplets, n_strain) "
            "or (B, T, n_triplets, n_strain)"
        )

    return arr[:, triplet_idx, :]


def project_path_to_2d(
    path,
    strain_x_idx,
    strain_y_idx,
):
    x_path = np.asarray(path[:, strain_x_idx])
    y_path = np.asarray(path[:, strain_y_idx])
    return x_path, y_path


# =========================================================
# Grid helpers
# =========================================================
def _auto_limits_from_path(x_path, y_path, pad_frac=0.35, min_pad=1e-3):
    x_min, x_max = float(np.min(x_path)), float(np.max(x_path))
    y_min, y_max = float(np.min(y_path)), float(np.max(y_path))
    dx = max(x_max - x_min, min_pad)
    dy = max(y_max - y_min, min_pad)
    return (
        (x_min - pad_frac * dx, x_max + pad_frac * dx),
        (y_min - pad_frac * dy, y_max + pad_frac * dy),
    )


def _make_fixed_strains(path, spec: EnergyLandscapeSpec):
    n_strain = path.shape[1]

    if spec.fixed_strains is not None:
        fixed = np.asarray(spec.fixed_strains, dtype=float).copy()
        if fixed.shape != (n_strain,):
            raise ValueError(
                f"fixed_strains should have shape ({n_strain},), got {fixed.shape}"
            )
        return fixed

    if spec.fixed_mode == "zeros":
        fixed = np.zeros(n_strain, dtype=float)
    elif spec.fixed_mode == "first_path_point":
        fixed = np.asarray(path[0], dtype=float).copy()
    else:
        raise ValueError(
            f"Unknown fixed_mode='{spec.fixed_mode}'. "
            "Use 'zeros' or 'first_path_point'."
        )

    fixed[spec.strain_x_idx] = 0.0
    fixed[spec.strain_y_idx] = 0.0
    return fixed


def evaluate_energy_on_grid(model, spec: EnergyLandscapeSpec, fixed_strains):
    xs = jnp.linspace(spec.xlim[0], spec.xlim[1], spec.n_grid)
    ys = jnp.linspace(spec.ylim[0], spec.ylim[1], spec.n_grid)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")

    fixed = jnp.asarray(fixed_strains, dtype=float)

    def energy_at(x, y):
        eps = fixed.at[spec.strain_x_idx].set(x)
        eps = eps.at[spec.strain_y_idx].set(y)
        return model(eps)

    E = jax.vmap(lambda yy: jax.vmap(lambda xx: energy_at(xx, yy))(xs))(ys)
    return np.asarray(X), np.asarray(Y), np.asarray(E)


# =========================================================
# Plotting
# =========================================================
def _get_axis_labels(spec: EnergyLandscapeSpec):
    if spec.strain_labels is None:
        return f"strain[{spec.strain_x_idx}]", f"strain[{spec.strain_y_idx}]"
    return (
        spec.strain_labels[spec.strain_x_idx],
        spec.strain_labels[spec.strain_y_idx],
    )


def plot_energy_landscape_from_path(
    model,
    path,
    spec: EnergyLandscapeSpec,
    ax=None,
):
    """
    Plot energy contours and overlay a visited strain trajectory.
    """
    x_path, y_path = project_path_to_2d(
        path, spec.strain_x_idx, spec.strain_y_idx
    )

    xlim = spec.xlim
    ylim = spec.ylim
    if xlim is None or ylim is None:
        auto_x, auto_y = _auto_limits_from_path(x_path, y_path)
        if xlim is None:
            xlim = auto_x
        if ylim is None:
            ylim = auto_y

    spec_eval = replace(spec, xlim=xlim, ylim=ylim)

    fixed_strains = _make_fixed_strains(path, spec_eval)
    X, Y, E = evaluate_energy_on_grid(model, spec_eval, fixed_strains)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
    else:
        fig = ax.figure

    contour_artist = None

    if spec_eval.contour_style in ("filled", "both"):
        contour_artist = ax.contourf(X, Y, E, levels=spec_eval.n_levels)

    if spec_eval.contour_style in ("lines", "both"):
        contour_artist = ax.contour(
            X,
            Y,
            E,
            levels=spec_eval.n_levels,
            linewidths=spec_eval.contour_linewidth,
        )

    ax.plot(x_path, y_path, linewidth=spec_eval.path_linewidth, zorder=3)

    if spec_eval.show_path_points:
        ax.scatter(x_path, y_path, s=22, zorder=4)

    if spec_eval.show_start_end:
        ax.scatter([x_path[0]], [y_path[0]], marker="o", s=70, zorder=5, label="start")
        ax.scatter([x_path[-1]], [y_path[-1]], marker="*", s=120, zorder=5, label="end")
        ax.legend(frameon=True)

    xlabel, ylabel = _get_axis_labels(spec_eval)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        spec_eval.title
        or f"Energy landscape with visited strain path (triplet {spec_eval.triplet_idx})"
    )

    if spec_eval.show_colorbar and contour_artist is not None:
        cbar = fig.colorbar(contour_artist, ax=ax)
        cbar.set_label("Energy")

    out = {
        "X": X,
        "Y": Y,
        "E": E,
        "path": np.asarray(path),
        "x_path": np.asarray(x_path),
        "y_path": np.asarray(y_path),
        "fixed_strains": np.asarray(fixed_strains),
        "spec": spec_eval,
    }
    return fig, ax, out


# =========================================================
# Snapshot helper during training
# =========================================================
def _snapshot_stem(epoch=None, tag=None):
    if tag is not None:
        return f"energy_landscape_{tag}"
    if epoch is None:
        raise ValueError("Either epoch or tag must be provided.")
    return f"energy_landscape_epoch_{int(epoch):04d}"


def save_energy_landscape_snapshot_from_solution(
    model,
    del_strains,
    spec: EnergyLandscapeSpec,
    save_dir,
    epoch=None,
    tag=None,
    qs=None,
    dpi=180,
    close_fig=True,
):
    os.makedirs(save_dir, exist_ok=True)
    stem = _snapshot_stem(epoch=epoch, tag=tag)

    path = extract_strain_path(
        del_strains,
        traj_idx=spec.traj_idx,
        triplet_idx=spec.triplet_idx,
    )

    fig, ax, out = plot_energy_landscape_from_path(
        model=model,
        path=path,
        spec=spec,
        ax=None,
    )

    fig.savefig(
        os.path.join(save_dir, f"{stem}.png"),
        bbox_inches="tight",
        dpi=dpi,
    )

    save_dict = dict(
        X=out["X"],
        Y=out["Y"],
        E=out["E"],
        path=out["path"],
        x_path=out["x_path"],
        y_path=out["y_path"],
        fixed_strains=out["fixed_strains"],
    )
    if qs is not None:
        save_dict["qs"] = np.asarray(qs)
    save_dict["del_strains"] = np.asarray(del_strains)

    np.savez(
        os.path.join(save_dir, f"{stem}.npz"),
        **save_dict,
    )

    if close_fig:
        plt.close(fig)

    return out


def make_energy_snapshot_fn(
    spec,
    save_dir,
    use_valid=True,
    traj_idx=0,
    max_dlambda=5e-3,
    iters=5,
    ls_steps=10,
    dpi=180,
    n_grid_snapshot=None,
):
    """
    Returns a snapshot callback for train_model(...).

    Efficiency:
    - does exactly ONE solve_with_aux(...) for the selected trajectory
    - then plots from the extracted strain history
    """
    os.makedirs(save_dir, exist_ok=True)

    def snapshot_fn(model, epoch, base, aux, train, valid, train_loss=None, val_loss=None):
        ds = valid if use_valid else train

        if ds.idx_b.ndim == 1:
            idx_b_plot = ds.idx_b
        else:
            idx_b_plot = ds.idx_b[traj_idx]

        xb_plot = ds.xb[traj_idx]

        if ds.lambdas.ndim == 1:
            lambdas_plot = ds.lambdas
        else:
            lambdas_plot = ds.lambdas[traj_idx]

        spec_local = replace(spec)
        spec_local.traj_idx = 0  # local solve is single-trajectory now

        if n_grid_snapshot is not None:
            spec_local.n_grid = n_grid_snapshot

        # Single solve only for the chosen trajectory
        qs, auxs, del_strains = solve_with_strain_history(
            model=model,
            base=base,
            aux=aux,
            idx_b=idx_b_plot,
            xb=xb_plot,
            lambdas=lambdas_plot,
            max_dlambda=max_dlambda,
            iters=iters,
            ls_steps=ls_steps,
        )

        _ = auxs  # retained for debugging if you later want to save/use it

        tag = "initial" if epoch == -1 else None
        save_energy_landscape_snapshot_from_solution(
            model=model,
            del_strains=del_strains,
            spec=spec_local,
            save_dir=save_dir,
            epoch=None if tag is not None else epoch,
            tag=tag,
            qs=qs,
            dpi=dpi,
            close_fig=True,
        )

    return snapshot_fn
