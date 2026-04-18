import os
from typing import Optional

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


def _to_numpy(x):
    return np.asarray(x)


def _get_center_node_xz_indices(qs):
    qs = _to_numpy(qs)
    if qs.ndim < 2:
        raise ValueError(f"Expected trajectory array with at least 2 dims, got shape {qs.shape}")

    dof = qs.shape[-1]
    n_nodes = (dof + 1) // 4
    if 4 * n_nodes - 1 != dof:
        raise ValueError(
            f"Expected dof to match 2D rod layout 4*N-1, got dof={dof}."
        )

    center_node = n_nodes // 2
    x_idx = 4 * center_node
    z_idx = 4 * center_node + 2
    return center_node, x_idx, z_idx


def plot_loss_curves(
    train_hist,
    valid_hist,
    title: str,
    save_path: Optional[str] = None,
    show: bool = False,
    logy: bool = True,
):
    train_hist = _to_numpy(train_hist)
    valid_hist = _to_numpy(valid_hist)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(train_hist, linewidth=2.0, label="Train")
    ax.plot(valid_hist, linewidth=2.0, label="Valid")

    if logy:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_prediction_vs_truth(
    pred,
    truth,
    split_name: str,
    title: str,
    save_path: Optional[str] = None,
    show: bool = False,
    x_idx: Optional[int] = None,
    z_idx: Optional[int] = None,
):
    """
    Overlays prediction vs truth for all trajectories in one plot.

    For each case:
      - x DOF is plotted with full opacity
      - z DOF is plotted with lower opacity
      - solid = prediction
      - dashed = truth
    """
    pred = _to_numpy(pred)
    truth = _to_numpy(truth)

    center_node, default_x_idx, default_z_idx = _get_center_node_xz_indices(truth)
    if x_idx is None:
        x_idx = default_x_idx
    if z_idx is None:
        z_idx = default_z_idx

    n_cases = pred.shape[0]
    colors = cm.viridis(np.linspace(0, 1, n_cases))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for i in range(n_cases):
        c = colors[i]

        ax.plot(pred[i, :, x_idx], color=c, linestyle="-", linewidth=1.8)
        ax.plot(truth[i, :, x_idx], color=c, linestyle="--", linewidth=1.8)

        ax.plot(pred[i, :, z_idx], color=c, linestyle="-", linewidth=1.4, alpha=0.6)
        ax.plot(truth[i, :, z_idx], color=c, linestyle="--", linewidth=1.4, alpha=0.6)

    pred_line = mlines.Line2D([], [], color="black", linestyle="-", label="Prediction")
    truth_line = mlines.Line2D([], [], color="black", linestyle="--", label="Truth")
    ax.legend(handles=[pred_line, truth_line], loc="best")

    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=0, vmax=max(n_cases - 1, 1)),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Case index")

    ax.set_title(f"{title} | {split_name}")
    ax.set_xlabel("lambda index")
    ax.set_ylabel(f"Center Node {center_node} Position (m)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_prediction_vs_truth_separate_components(
    pred,
    truth,
    split_name: str,
    title: str,
    save_path: Optional[str] = None,
    show: bool = False,
    x_idx: Optional[int] = None,
    z_idx: Optional[int] = None,
):
    """
    Cleaner 1x2 figure:
      - left: x trajectories
      - right: z trajectories
    """
    pred = _to_numpy(pred)
    truth = _to_numpy(truth)

    center_node, default_x_idx, default_z_idx = _get_center_node_xz_indices(truth)
    if x_idx is None:
        x_idx = default_x_idx
    if z_idx is None:
        z_idx = default_z_idx

    n_cases = pred.shape[0]
    colors = cm.viridis(np.linspace(0, 1, n_cases))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax1, ax2 = axes

    for i in range(n_cases):
        c = colors[i]
        ax1.plot(pred[i, :, x_idx], color=c, linestyle="-", linewidth=1.8)
        ax1.plot(truth[i, :, x_idx], color=c, linestyle="--", linewidth=1.8)

        ax2.plot(pred[i, :, z_idx], color=c, linestyle="-", linewidth=1.8)
        ax2.plot(truth[i, :, z_idx], color=c, linestyle="--", linewidth=1.8)

    pred_line = mlines.Line2D([], [], color="black", linestyle="-", label="Prediction")
    truth_line = mlines.Line2D([], [], color="black", linestyle="--", label="Truth")
    fig.legend(handles=[pred_line, truth_line], loc="upper center", ncol=2)

    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=0, vmax=max(n_cases - 1, 1)),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Case index")

    ax1.set_title("x component")
    ax2.set_title("z component")

    for ax in axes:
        ax.set_xlabel("lambda index")
        ax.set_ylabel(f"Center Node {center_node} Position (m)")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{title} | {split_name}", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _load_scalar(data, key: str):
    value = np.asarray(data[key])
    if value.shape == ():
        return value.item()
    return value


def plot_saved_results(
    results_path: str,
    output_dir: Optional[str] = None,
    show: bool = False,
    make_component_plots: bool = True,
    logy: bool = True,
):
    data = np.load(results_path, allow_pickle=False)

    arch_name = _load_scalar(data, "arch_name")
    model_cls = _load_scalar(data, "model_cls")
    which_case = _load_scalar(data, "which_case")
    seed = _load_scalar(data, "seed")
    title = f"{arch_name} | {model_cls} | {which_case} | seed={seed}"

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    def maybe_path(filename: str):
        if output_dir is None:
            return None
        return os.path.join(output_dir, filename)

    plot_loss_curves(
        train_hist=data["train_hist"],
        valid_hist=data["valid_hist"],
        title=title,
        save_path=maybe_path("loss_curves.png"),
        show=show,
        logy=logy,
    )

    plot_prediction_vs_truth(
        pred=data["train_pred"],
        truth=data["train_truth"],
        split_name="train",
        title=title,
        save_path=maybe_path("pred_vs_truth_train_overlay.png"),
        show=show,
    )

    plot_prediction_vs_truth(
        pred=data["valid_pred"],
        truth=data["valid_truth"],
        split_name="valid",
        title=title,
        save_path=maybe_path("pred_vs_truth_valid_overlay.png"),
        show=show,
    )

    if make_component_plots:
        plot_prediction_vs_truth_separate_components(
            pred=data["train_pred"],
            truth=data["train_truth"],
            split_name="train",
            title=title,
            save_path=maybe_path("pred_vs_truth_train_xz.png"),
            show=show,
        )

        plot_prediction_vs_truth_separate_components(
            pred=data["valid_pred"],
            truth=data["valid_truth"],
            split_name="valid",
            title=title,
            save_path=maybe_path("pred_vs_truth_valid_xz.png"),
            show=show,
        )


def _load_results_file(results_path: str) -> dict:
    data = np.load(results_path, allow_pickle=False)
    return {key: np.asarray(data[key]) for key in data.files}


def _scalar_str(data: dict, key: str) -> str:
    value = np.asarray(data[key])
    if value.shape == ():
        return str(value.item())
    return str(value)


def _find_architecture_results_path(results_dir: str, arch_name: str) -> str:
    matches = []
    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if not entry.startswith(f"{arch_name}__"):
            continue
        results_path = os.path.join(entry_path, "results.npz")
        if os.path.isfile(results_path):
            matches.append(results_path)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No saved results found for architecture '{arch_name}' in '{results_dir}'."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple saved results found for architecture '{arch_name}' in '{results_dir}': "
            f"{matches}. Please keep one run per architecture in that directory."
        )
    return matches[0]


def _paper_rc_params():
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
    }


def _architecture_colors(architectures: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, max(len(architectures), 1)))
    return {arch: colors[i] for i, arch in enumerate(architectures)}


def _save_pdf(fig, path: str):
    fig.savefig(path, format="pdf", dpi=600, bbox_inches="tight", transparent=False)


def _plot_loss_comparison(
    results_by_arch: dict[str, dict],
    architectures: list[str],
    colors: dict[str, tuple],
    which: str,
    save_path: str,
):
    ylabel = "Training Loss" if which == "train" else "Validation Loss"
    hist_key = "train_hist" if which == "train" else "valid_hist"

    with plt.rc_context(_paper_rc_params()):
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        ax.set_prop_cycle(cycler(color=[colors[a] for a in architectures]))

        for arch in architectures:
            hist = np.asarray(results_by_arch[arch][hist_key], dtype=float)
            epochs = np.arange(hist.shape[0], dtype=int)
            ax.plot(epochs, hist, linewidth=1.8, label=arch)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
        ax.legend(frameon=False, ncol=2)
        _save_pdf(fig, save_path)
        plt.close(fig)


def _plot_trajectory_comparison(
    results_by_arch: dict[str, dict],
    architectures: list[str],
    colors: dict[str, tuple],
    split: str,
    save_path: str,
    traj_idx: Optional[int],
    x_idx: Optional[int],
    z_idx: Optional[int],
):
    pred_key = "train_pred" if split == "train" else "valid_pred"
    truth_key = "train_truth" if split == "train" else "valid_truth"
    lambdas_key = "train_lambdas" if split == "train" else "valid_lambdas"

    first = results_by_arch[architectures[0]]
    truth = np.asarray(first[truth_key], dtype=float)
    lambdas = np.asarray(first[lambdas_key], dtype=float)

    if truth.ndim != 3:
        raise ValueError(f"Expected {truth_key} to have shape (n_traj, T, dof), got {truth.shape}")

    center_node, default_x_idx, default_z_idx = _get_center_node_xz_indices(truth)
    if x_idx is None:
        x_idx = default_x_idx
    if z_idx is None:
        z_idx = default_z_idx

    if traj_idx is None:
        lam_plot = np.arange(truth.shape[0] * truth.shape[1], dtype=int)
        truth_x = truth[:, :, x_idx].reshape(-1)
        truth_z = truth[:, :, z_idx].reshape(-1)
        xlabel = "BC Step Index"
    else:
        if traj_idx >= truth.shape[0]:
            raise IndexError(
                f"traj_idx={traj_idx} is out of bounds for split '{split}' with {truth.shape[0]} trajectories."
            )
        lam_plot = lambdas if lambdas.ndim == 1 else lambdas[traj_idx]
        truth_x = truth[traj_idx, :, x_idx]
        truth_z = truth[traj_idx, :, z_idx]
        xlabel = r"$\lambda$"

    with plt.rc_context(_paper_rc_params()):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
        ax_x, ax_z = axes

        for arch in architectures:
            pred = np.asarray(results_by_arch[arch][pred_key], dtype=float)
            if traj_idx is None:
                pred_x = pred[:, :, x_idx].reshape(-1)
                pred_z = pred[:, :, z_idx].reshape(-1)
            else:
                pred_x = pred[traj_idx, :, x_idx]
                pred_z = pred[traj_idx, :, z_idx]
            color = colors[arch]

            ax_x.plot(lam_plot, pred_x, color=color, linewidth=1.6, label=arch)
            ax_z.plot(lam_plot, pred_z, color=color, linewidth=1.6, label=arch)

        ax_x.plot(lam_plot, truth_x, color="black", linestyle="--", linewidth=1.8, label="ground truth")
        ax_z.plot(lam_plot, truth_z, color="black", linestyle="--", linewidth=1.8, label="ground truth")

        ax_x.set_xlabel(xlabel)
        ax_z.set_xlabel(xlabel)
        ax_x.set_ylabel(f"Center Node {center_node} x Position (m)")
        ax_z.set_ylabel(f"Center Node {center_node} z Position (m)")
        ax_x.grid(True, alpha=0.18, linewidth=0.6)
        ax_z.grid(True, alpha=0.18, linewidth=0.6)

        handles, labels = ax_x.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.05),
        )
        _save_pdf(fig, save_path)
        plt.close(fig)


def plot_architecture_comparison_paper(
    architectures: list[str],
    results_dir: str,
    output_dir: Optional[str] = None,
    traj_idx: Optional[int] = None,
    x_idx: Optional[int] = None,
    z_idx: Optional[int] = None,
):
    """
    Build paper-ready comparison figures across architectures from saved results.

    Parameters
    ----------
    architectures
        List of architecture names, e.g. ["diag_energy_baseline", "chol_energy_baseline"].
    results_dir
        Directory containing experiment subdirectories created by run_architectures.py.
        Each requested architecture must have exactly one matching subdirectory in this folder.
    output_dir
        Directory where PDF figures will be written. Defaults to
        ``<results_dir>/paper_ready_architecture_comparison``.
    traj_idx
        Trajectory index to use for the train/validation trajectory comparison figures.
        Use ``None`` to plot the complete concatenated trajectory across all saved BCs.
    x_idx, z_idx
        Optional DOF indices used for the 2-panel trajectory figures.
        If omitted, they are inferred from the center node via
        ``x_idx = 4 * (n_nodes // 2)`` and ``z_idx = 4 * (n_nodes // 2) + 2``.
    """
    if len(architectures) == 0:
        raise ValueError("architectures must contain at least one architecture name.")

    if output_dir is None:
        output_dir = os.path.join(results_dir, "paper_ready_architecture_comparison")
    os.makedirs(output_dir, exist_ok=True)

    results_by_arch = {}
    for arch in architectures:
        results_path = _find_architecture_results_path(results_dir, arch)
        results_by_arch[arch] = _load_results_file(results_path)

    colors = _architecture_colors(architectures)

    _plot_loss_comparison(
        results_by_arch=results_by_arch,
        architectures=architectures,
        colors=colors,
        which="train",
        save_path=os.path.join(output_dir, "training_loss_comparison.pdf"),
    )
    _plot_loss_comparison(
        results_by_arch=results_by_arch,
        architectures=architectures,
        colors=colors,
        which="valid",
        save_path=os.path.join(output_dir, "validation_loss_comparison.pdf"),
    )
    _plot_trajectory_comparison(
        results_by_arch=results_by_arch,
        architectures=architectures,
        colors=colors,
        split="train",
        save_path=os.path.join(output_dir, "training_trajectory_comparison.pdf"),
        traj_idx=traj_idx,
        x_idx=x_idx,
        z_idx=z_idx,
    )
    _plot_trajectory_comparison(
        results_by_arch=results_by_arch,
        architectures=architectures,
        colors=colors,
        split="valid",
        save_path=os.path.join(output_dir, "validation_trajectory_comparison.pdf"),
        traj_idx=traj_idx,
        x_idx=x_idx,
        z_idx=z_idx,
    )

    return {
        "training_loss": os.path.join(output_dir, "training_loss_comparison.pdf"),
        "validation_loss": os.path.join(output_dir, "validation_loss_comparison.pdf"),
        "training_trajectory": os.path.join(output_dir, "training_trajectory_comparison.pdf"),
        "validation_trajectory": os.path.join(output_dir, "validation_trajectory_comparison.pdf"),
        "colors": colors,
    }
