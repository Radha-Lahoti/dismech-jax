import os
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Optional

import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from run_architectures import (
    build_architecture_registry,
    run_one_architecture,
    subset_brazier_stiffness_only,
    subset_tape_tube_candidates,
)


def run_seed_hessian_diagnostics(output_dir: str, **diagnostic_kwargs):
    """
    Run post-training Hessian diagnostics for a saved seed ablation directory.

    This is a thin convenience wrapper around
    compute_architecture_hessian_diagnostics.py so seed sweeps keep Hessian
    work separate from training, matching run_architectures.py.
    """
    from types import SimpleNamespace

    from compute_architecture_hessian_diagnostics import (
        _find_experiment_dirs,
        compute_for_experiment,
        write_hessian_summary_csv,
    )

    defaults = dict(
        use_predicted=True,
        splits=("train", "valid"),
        max_trajectories=1,
        all_trajectories=False,
        stride=10,
        train_file=None,
        valid_file=None,
        properties_class=None,
        force_key=None,
        fail_on_nonconvergence=False,
        summary_csv="hessian_diagnostics_table.csv",
    )
    defaults.update(diagnostic_kwargs)
    args = SimpleNamespace(**defaults)

    exp_dirs = _find_experiment_dirs(output_dir)
    n_ok = 0
    for exp_dir in exp_dirs:
        try:
            n_ok += int(compute_for_experiment(exp_dir, args))
        except Exception as exc:
            print(f"[failed] {exp_dir}: {exc!r}")

    if args.summary_csv:
        write_hessian_summary_csv(output_dir, exp_dirs=exp_dirs, csv_name=args.summary_csv)

    print(f"Finished Hessian diagnostics for {n_ok}/{len(exp_dirs)} seed runs.")
    return n_ok


# =========================================================
# Helpers
# =========================================================
def _is_successful_result(r: dict) -> bool:
    return bool(r.get("success", True))


def _is_nonfinite_training_failure(r: dict) -> bool:
    return "nonfinite_training_history" in str(r.get("failure_reason", "")).lower()


def _successful_results(seed_results: list[dict]) -> list[dict]:
    return [r for r in seed_results if _is_successful_result(r)]


def _final_or_nan(hist):
    if hist is None:
        return np.nan
    arr = np.asarray(hist, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(arr[-1])


def _history_key(which: str) -> str:
    aliases = {
        "train": "train_hist",
        "valid": "valid_hist",
        "train_displacement": "train_displacement_hist",
        "valid_displacement": "valid_displacement_hist",
        "train_force": "train_force_hist",
        "valid_force": "valid_force_hist",
    }
    if which not in aliases:
        raise ValueError(f"which must be one of {tuple(aliases)}")
    return aliases[which]


def _has_history(seed_results: list[dict], which: str) -> bool:
    key = _history_key(which)
    return any(r.get(key, None) is not None for r in seed_results)


def _trajectory_valid_mask(result: dict, split: str, traj_idx: int, length: int) -> np.ndarray:
    mask_key = f"{split}_valid_mask"
    if mask_key not in result or result[mask_key] is None:
        return np.ones(length, dtype=bool)

    mask = np.asarray(result[mask_key], dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected {mask_key} to have shape (n_traj, T), got {mask.shape}")
    if not (0 <= traj_idx < mask.shape[0]):
        raise IndexError(
            f"traj_idx={traj_idx} is out of bounds for {mask_key} with {mask.shape[0]} trajectories."
        )
    if mask.shape[1] != length:
        raise ValueError(
            f"Expected {mask_key} time dimension to be {length}, got {mask.shape[1]}."
        )

    traj_mask = mask[traj_idx]
    if not np.any(traj_mask):
        raise ValueError(f"{mask_key}[{traj_idx}] contains no valid samples to plot.")

    return traj_mask


def _trajectory_lambda_axis(result: dict, split: str, traj_idx: int, valid_mask: np.ndarray) -> tuple[np.ndarray, str]:
    lambdas_key = f"{split}_lambdas"
    if lambdas_key not in result or result[lambdas_key] is None:
        return np.arange(np.sum(valid_mask)), "valid lambda index"

    lambdas = np.asarray(result[lambdas_key], dtype=float)
    if lambdas.ndim == 1:
        if lambdas.shape[0] != valid_mask.shape[0]:
            raise ValueError(
                f"Expected {lambdas_key} length to be {valid_mask.shape[0]}, got {lambdas.shape[0]}."
            )
        return lambdas[valid_mask], r"$\lambda$"
    if lambdas.ndim == 2:
        if not (0 <= traj_idx < lambdas.shape[0]):
            raise IndexError(
                f"traj_idx={traj_idx} is out of bounds for {lambdas_key} with {lambdas.shape[0]} trajectories."
            )
        if lambdas.shape[1] != valid_mask.shape[0]:
            raise ValueError(
                f"Expected {lambdas_key} time dimension to be {valid_mask.shape[0]}, got {lambdas.shape[1]}."
            )
        return lambdas[traj_idx, valid_mask], r"$\lambda$"

    raise ValueError(f"Expected {lambdas_key} to have shape (T,) or (n_traj, T), got {lambdas.shape}")


def _load_scalar_npz(data, key: str):
    value = np.asarray(data[key])
    if value.shape == ():
        return value.item()
    return value


def _result_from_npz(results_path: str) -> dict:
    with np.load(results_path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]) for key in data.files}

    seed = int(_load_scalar_npz(result, "seed")) if "seed" in result else -1
    result["cfg"] = SimpleNamespace(seed=seed)
    result["success"] = True
    result["results_path"] = results_path
    return result


def load_seed_ablation_results(results_dir: str, architectures: Optional[list[str]] = None) -> dict:
    """
    Reconstruct all_seed_results from saved per-seed results.npz files.

    results_dir should be the directory that contains architecture/seed run
    subdirectories, each with a results.npz file written by run_architectures.py.
    If a seed_ablation_summary directory is passed, the parent directory is used.
    """
    if os.path.basename(os.path.normpath(results_dir)) == "seed_ablation_summary":
        results_dir = os.path.dirname(os.path.normpath(results_dir))

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")

    requested = None if architectures is None else set(architectures)
    all_seed_results = {}

    for entry in sorted(os.listdir(results_dir)):
        results_path = os.path.join(results_dir, entry, "results.npz")
        if not os.path.isfile(results_path):
            continue

        result = _result_from_npz(results_path)
        if "arch_name" in result:
            arch_name = str(_load_scalar_npz(result, "arch_name"))
        else:
            arch_name = entry.split("__", 1)[0]

        if requested is not None and arch_name not in requested:
            continue

        all_seed_results.setdefault(arch_name, []).append(result)

    if len(all_seed_results) == 0:
        raise FileNotFoundError(
            f"No saved results.npz files found in '{results_dir}'. "
            "Expected subdirectories like <arch>__...__seed_<n>/results.npz."
        )

    for arch_name in all_seed_results:
        all_seed_results[arch_name].sort(key=lambda r: int(r["cfg"].seed))

    return all_seed_results


# =========================================================
# 1) Run seed ablation
# =========================================================
def run_seed_ablation(
    properties,
    train_file: str,
    valid_file: str,
    base_cfg,
    selected_architectures: list[str],
):
    """
    Run multiple seeds for each selected architecture.

    Parameters
    ----------
    properties : any
        Passed into run_one_architecture.
    train_file, valid_file : str
        Dataset files.
    base_cfg : SweepConfig
        Must contain all base settings. Uses base_cfg.seed_list.
    selected_architectures : list[str]
        Architecture names from build_architecture_registry().

    Returns
    -------
    all_results : dict
        all_results[arch_name] = [result_seed0, result_seed1, ...]
    """
    registry = build_architecture_registry()

    unknown = [name for name in selected_architectures if name not in registry]
    if len(unknown) > 0:
        raise ValueError(f"Unknown architecture names: {unknown}")

    if not hasattr(base_cfg, "seed_list"):
        raise AttributeError(
            "base_cfg must have attribute 'seed_list'. "
            "Add seed_list: tuple[int, ...] to SweepConfig."
        )

    all_results = {}

    for arch_name in selected_architectures:
        spec = registry[arch_name]
        arch_results = []

        print("\n" + "=" * 110)
        print(f"SEED ABLATION for architecture: {arch_name}")
        print("=" * 110)

        for seed in base_cfg.seed_list:
            cfg = replace(base_cfg, seed=int(seed))

            print(f"\n--- Running seed {seed} for {arch_name} ---\n")

            result = run_one_architecture(
                properties=properties,
                train_file=train_file,
                valid_file=valid_file,
                spec=spec,
                cfg=cfg,
            )
            arch_results.append(result)

            if _is_nonfinite_training_failure(result):
                print(
                    f"[FAILED] {arch_name}: nonfinite training history for seed {seed}. "
                    "Skipping remaining seeds for this architecture."
                )
                break

        all_results[arch_name] = arch_results

    return all_results


# =========================================================
# 2) Summaries
# =========================================================
def get_seed_summary(seed_results: list[dict]) -> dict:
    seeds = np.array([r["cfg"].seed for r in seed_results], dtype=int)
    success_mask = np.array([_is_successful_result(r) for r in seed_results], dtype=bool)

    final_train = np.array([_final_or_nan(r.get("train_hist", None)) for r in seed_results], dtype=float)
    final_valid = np.array([_final_or_nan(r.get("valid_hist", None)) for r in seed_results], dtype=float)
    final_train_displacement = np.array(
        [_final_or_nan(r.get("train_displacement_hist", None)) for r in seed_results],
        dtype=float,
    )
    final_valid_displacement = np.array(
        [_final_or_nan(r.get("valid_displacement_hist", None)) for r in seed_results],
        dtype=float,
    )
    final_train_force = np.array(
        [_final_or_nan(r.get("train_force_hist", None)) for r in seed_results],
        dtype=float,
    )
    final_valid_force = np.array(
        [_final_or_nan(r.get("valid_force_hist", None)) for r in seed_results],
        dtype=float,
    )

    successful = _successful_results(seed_results)

    if len(successful) > 0:
        successful_final_valid = np.array(
            [_final_or_nan(r.get("valid_hist", None)) for r in successful],
            dtype=float,
        )
        best_success_idx_local = int(np.nanargmin(successful_final_valid))
        best_success_result = successful[best_success_idx_local]
        best_seed = int(best_success_result["cfg"].seed)

        valid_success_vals = successful_final_valid
        train_success_vals = np.array(
            [_final_or_nan(r.get("train_hist", None)) for r in successful],
            dtype=float,
        )

        mean_final_train = float(np.nanmean(train_success_vals))
        std_final_train = float(np.nanstd(train_success_vals))
        mean_final_valid = float(np.nanmean(valid_success_vals))
        std_final_valid = float(np.nanstd(valid_success_vals))
        median_final_valid = float(np.nanmedian(valid_success_vals))
        min_final_valid = float(np.nanmin(valid_success_vals))
        max_final_valid = float(np.nanmax(valid_success_vals))
    else:
        best_success_result = None
        best_seed = None
        mean_final_train = np.nan
        std_final_train = np.nan
        mean_final_valid = np.nan
        std_final_valid = np.nan
        median_final_valid = np.nan
        min_final_valid = np.nan
        max_final_valid = np.nan

    failed_seed_info = []
    for r in seed_results:
        if not _is_successful_result(r):
            failed_seed_info.append(
                {
                    "seed": int(r["cfg"].seed),
                    "failure_reason": r.get("failure_reason", "unknown_failure"),
                }
            )

    return {
        "n_seeds": len(seed_results),
        "n_success": int(np.sum(success_mask)),
        "n_failed": int(np.sum(~success_mask)),
        "success_mask": success_mask,
        "seeds": seeds,
        "final_train": final_train,
        "final_valid": final_valid,
        "final_train_displacement": final_train_displacement,
        "final_valid_displacement": final_valid_displacement,
        "final_train_force": final_train_force,
        "final_valid_force": final_valid_force,
        "best_seed": best_seed,
        "best_result": best_success_result,
        "mean_final_train": mean_final_train,
        "std_final_train": std_final_train,
        "mean_final_valid": mean_final_valid,
        "std_final_valid": std_final_valid,
        "median_final_valid": median_final_valid,
        "min_final_valid": min_final_valid,
        "max_final_valid": max_final_valid,
        "failed_seed_info": failed_seed_info,
    }


def print_seed_summary(all_seed_results: dict):
    print("\n" + "#" * 110)
    print("SEED ABLATION SUMMARY")
    print("#" * 110)

    for arch_name, seed_results in all_seed_results.items():
        s = get_seed_summary(seed_results)

        print(f"\nArchitecture: {arch_name}")
        print(f"  n_seeds           : {s['n_seeds']}")
        print(f"  n_success         : {s['n_success']}")
        print(f"  n_failed          : {s['n_failed']}")
        print(f"  best_seed         : {s['best_seed']}")
        print(f"  best final valid  : {s['min_final_valid']:.6e}")
        print(f"  worst final valid : {s['max_final_valid']:.6e}")
        print(f"  mean final valid  : {s['mean_final_valid']:.6e}")
        print(f"  std  final valid  : {s['std_final_valid']:.6e}")
        print(f"  median final valid: {s['median_final_valid']:.6e}")
        print(f"  mean final train  : {s['mean_final_train']:.6e}")
        print(f"  std  final train  : {s['std_final_train']:.6e}")
        if np.any(np.isfinite(s["final_valid_displacement"])):
            print(f"  mean valid disp   : {np.nanmean(s['final_valid_displacement']):.6e}")
        if np.any(np.isfinite(s["final_valid_force"])):
            print(f"  mean valid force  : {np.nanmean(s['final_valid_force']):.6e}")

        if s["n_failed"] > 0:
            print("  failed seeds      :")
            for item in s["failed_seed_info"]:
                print(f"    seed {item['seed']}: {item['failure_reason']}")


def save_seed_summary_json(all_seed_results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    payload = {}
    for arch_name, seed_results in all_seed_results.items():
        s = get_seed_summary(seed_results)
        payload[arch_name] = {
            "n_seeds": int(s["n_seeds"]),
            "n_success": int(s["n_success"]),
            "n_failed": int(s["n_failed"]),
            "seeds": [int(x) for x in s["seeds"]],
            "success_mask": [bool(x) for x in s["success_mask"]],
            "final_train": [float(x) if np.isfinite(x) else None for x in s["final_train"]],
            "final_valid": [float(x) if np.isfinite(x) else None for x in s["final_valid"]],
            "final_train_displacement": [
                float(x) if np.isfinite(x) else None for x in s["final_train_displacement"]
            ],
            "final_valid_displacement": [
                float(x) if np.isfinite(x) else None for x in s["final_valid_displacement"]
            ],
            "final_train_force": [
                float(x) if np.isfinite(x) else None for x in s["final_train_force"]
            ],
            "final_valid_force": [
                float(x) if np.isfinite(x) else None for x in s["final_valid_force"]
            ],
            "best_seed": None if s["best_seed"] is None else int(s["best_seed"]),
            "mean_final_train": None if not np.isfinite(s["mean_final_train"]) else float(s["mean_final_train"]),
            "std_final_train": None if not np.isfinite(s["std_final_train"]) else float(s["std_final_train"]),
            "mean_final_valid": None if not np.isfinite(s["mean_final_valid"]) else float(s["mean_final_valid"]),
            "std_final_valid": None if not np.isfinite(s["std_final_valid"]) else float(s["std_final_valid"]),
            "median_final_valid": None if not np.isfinite(s["median_final_valid"]) else float(s["median_final_valid"]),
            "min_final_valid": None if not np.isfinite(s["min_final_valid"]) else float(s["min_final_valid"]),
            "max_final_valid": None if not np.isfinite(s["max_final_valid"]) else float(s["max_final_valid"]),
            "failed_seed_info": s["failed_seed_info"],
        }

    with open(os.path.join(output_dir, "seed_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)


# =========================================================
# 3) Plot helpers
# =========================================================
def plot_seed_loss_envelope(
    seed_results: list[dict],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
    which: str = "valid",
):
    hist_key = _history_key(which)

    good_results = _successful_results(seed_results)
    if len(good_results) == 0:
        print(f"[plot_seed_loss_envelope] No successful runs available for {which}. Skipping plot.")
        return

    losses = []
    seeds = []

    for r in good_results:
        if r.get(hist_key, None) is None:
            continue
        losses.append(r[hist_key])
        seeds.append(r["cfg"].seed)

    if len(losses) == 0:
        print(f"[plot_seed_loss_envelope] No {which} histories available. Skipping plot.")
        return

    losses = np.asarray(losses, dtype=float)
    seeds = np.asarray(seeds, dtype=int)

    q05 = np.percentile(losses, 5, axis=0)
    q25 = np.percentile(losses, 25, axis=0)
    q50 = np.percentile(losses, 50, axis=0)
    q75 = np.percentile(losses, 75, axis=0)
    q95 = np.percentile(losses, 95, axis=0)

    final_losses = losses[:, -1]
    best_idx = int(np.argmin(final_losses))
    best_seed = int(seeds[best_idx])

    fig, ax = plt.subplots(figsize=(8.0, 5.4))

    for i in range(losses.shape[0]):
        ax.plot(losses[i], linewidth=1.0, alpha=0.18)

    ax.fill_between(np.arange(losses.shape[1]), q05, q95, alpha=0.18, label="5-95%")
    ax.fill_between(np.arange(losses.shape[1]), q25, q75, alpha=0.28, label="25-75%")

    ax.plot(q50, linestyle="--", linewidth=2.2, label="Median across seeds")
    ax.plot(
        losses[best_idx],
        linewidth=2.5,
        label=f"Best seed = {best_seed} (final {which} = {final_losses[best_idx]:.3e})",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    label = which.replace("_", " ").capitalize()
    ax.set_ylabel(f"{label} MSE loss")
    ax.set_title(title if title is not None else f"{which.capitalize()} loss across seeds")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_seed_prediction_envelope(
    seed_results: list[dict],
    split: str = "valid",
    component: str = "x",
    traj_idx: int = 0,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
    x_idx: int = 4,
    z_idx: int = 6,
):
    """
    Percentile envelope version for prediction curves across seeds.
    """
    if split not in ("train", "valid"):
        raise ValueError("split must be 'train' or 'valid'")
    if component not in ("x", "z"):
        raise ValueError("component must be 'x' or 'z'")

    good_results = _successful_results(seed_results)
    if len(good_results) == 0:
        print(f"[plot_seed_prediction_envelope] No successful runs available. Skipping plot.")
        return

    comp_idx = x_idx if component == "x" else z_idx
    pred_key = f"{split}_pred"
    truth_key = f"{split}_truth"

    seeds = np.array([r["cfg"].seed for r in good_results], dtype=int)
    full_truth = np.asarray(good_results[0][truth_key][traj_idx, :, comp_idx], dtype=float)
    valid_mask = _trajectory_valid_mask(
        good_results[0],
        split=split,
        traj_idx=traj_idx,
        length=full_truth.shape[0],
    )
    preds = np.asarray(
        [np.asarray(r[pred_key][traj_idx, :, comp_idx], dtype=float)[valid_mask] for r in good_results],
        dtype=float,
    )
    truth = full_truth[valid_mask]
    x_plot, xlabel = _trajectory_lambda_axis(good_results[0], split, traj_idx, valid_mask)

    final_valid = np.asarray([r["valid_hist"][-1] for r in good_results], dtype=float)
    best_idx = int(np.argmin(final_valid))
    best_seed = int(seeds[best_idx])

    q05 = np.percentile(preds, 5, axis=0)
    q25 = np.percentile(preds, 25, axis=0)
    q50 = np.percentile(preds, 50, axis=0)
    q75 = np.percentile(preds, 75, axis=0)
    q95 = np.percentile(preds, 95, axis=0)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    ax.fill_between(x_plot, q05, q95, alpha=0.18, label="5-95%")
    ax.fill_between(x_plot, q25, q75, alpha=0.28, label="25-75%")
    ax.plot(x_plot, q50, linestyle="--", linewidth=2.0, label="Median across seeds")
    ax.plot(x_plot, preds[best_idx], linewidth=2.6, label=f"Best seed = {best_seed}")
    ax.plot(x_plot, truth, color="black", linewidth=2.3, label="Ground truth")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{component}-component position (m)")
    ax.set_title(
        title if title is not None else
        f"{split.capitalize()} envelope across seeds | traj {traj_idx} | {component}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_seed_prediction_all_curves(
    seed_results: list[dict],
    split: str = "valid",
    component: str = "x",
    traj_idx: int = 0,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
    x_idx: int = 4,
    z_idx: int = 6,
):
    """
    All seeds in light alpha, best seed bold, truth in black.
    """
    if split not in ("train", "valid"):
        raise ValueError("split must be 'train' or 'valid'")
    if component not in ("x", "z"):
        raise ValueError("component must be 'x' or 'z'")

    good_results = _successful_results(seed_results)
    if len(good_results) == 0:
        print(f"[plot_seed_prediction_all_curves] No successful runs available. Skipping plot.")
        return

    comp_idx = x_idx if component == "x" else z_idx
    pred_key = f"{split}_pred"
    truth_key = f"{split}_truth"

    seeds = np.array([r["cfg"].seed for r in good_results], dtype=int)
    full_truth = np.asarray(good_results[0][truth_key][traj_idx, :, comp_idx], dtype=float)
    valid_mask = _trajectory_valid_mask(
        good_results[0],
        split=split,
        traj_idx=traj_idx,
        length=full_truth.shape[0],
    )
    preds = np.asarray(
        [np.asarray(r[pred_key][traj_idx, :, comp_idx], dtype=float)[valid_mask] for r in good_results],
        dtype=float,
    )
    truth = full_truth[valid_mask]
    x_plot, xlabel = _trajectory_lambda_axis(good_results[0], split, traj_idx, valid_mask)

    final_valid = np.asarray([r["valid_hist"][-1] for r in good_results], dtype=float)
    best_idx = int(np.argmin(final_valid))
    best_seed = int(seeds[best_idx])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    for i in range(preds.shape[0]):
        ax.plot(x_plot, preds[i], linewidth=1.0, alpha=0.18)

    ax.plot(
        x_plot,
        preds[best_idx],
        linewidth=2.8,
        label=f"Best seed = {best_seed} (test MSE = {final_valid[best_idx]:.3e})",
    )
    ax.plot(x_plot, truth, color="black", linewidth=2.3, label="Ground truth")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{component}-component position (m)")
    ax.set_title(
        title if title is not None else
        f"{split.capitalize()} all-seed curves | traj {traj_idx} | {component}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_seed_prediction_xz_envelope(
    seed_results: list[dict],
    split: str = "valid",
    traj_idx: int = 0,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
    x_idx: int = 4,
    z_idx: int = 6,
):
    """
    x-z phase-space trajectory envelope across seeds.
    """
    if split not in ("train", "valid"):
        raise ValueError("split must be 'train' or 'valid'")

    good_results = _successful_results(seed_results)
    if len(good_results) == 0:
        print(f"[plot_seed_prediction_xz_envelope] No successful runs available. Skipping plot.")
        return

    pred_key = f"{split}_pred"
    truth_key = f"{split}_truth"

    truth_full = np.asarray(good_results[0][truth_key][traj_idx], dtype=float)
    valid_mask = _trajectory_valid_mask(
        good_results[0],
        split=split,
        traj_idx=traj_idx,
        length=truth_full.shape[0],
    )

    preds_x = np.asarray(
        [np.asarray(r[pred_key][traj_idx, :, x_idx], dtype=float)[valid_mask] for r in good_results],
        dtype=float,
    )
    preds_z = np.asarray(
        [np.asarray(r[pred_key][traj_idx, :, z_idx], dtype=float)[valid_mask] for r in good_results],
        dtype=float,
    )
    truth_x = truth_full[valid_mask, x_idx]
    truth_z = truth_full[valid_mask, z_idx]

    seeds = np.array([r["cfg"].seed for r in good_results], dtype=int)
    final_valid = np.asarray([r["valid_hist"][-1] for r in good_results], dtype=float)
    best_idx = int(np.argmin(final_valid))
    best_seed = int(seeds[best_idx])

    q05_x, q25_x, q50_x, q75_x, q95_x = np.percentile(preds_x, [5, 25, 50, 75, 95], axis=0)
    q05_z, q25_z, q50_z, q75_z, q95_z = np.percentile(preds_z, [5, 25, 50, 75, 95], axis=0)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))

    stride = max(1, truth_x.shape[0] // 35)
    for i in range(0, truth_x.shape[0], stride):
        ax.fill(
            [q05_x[i], q95_x[i], q95_x[i], q05_x[i]],
            [q05_z[i], q05_z[i], q95_z[i], q95_z[i]],
            alpha=0.06,
            color="C0",
            linewidth=0,
        )
        ax.fill(
            [q25_x[i], q75_x[i], q75_x[i], q25_x[i]],
            [q25_z[i], q25_z[i], q75_z[i], q75_z[i]],
            alpha=0.10,
            color="C0",
            linewidth=0,
        )

    ax.plot(q50_x, q50_z, linestyle="--", linewidth=2.0, color="C0", label="Median across seeds")
    ax.plot(
        preds_x[best_idx],
        preds_z[best_idx],
        linewidth=2.6,
        color="C1",
        label=f"Best seed = {best_seed} (test MSE = {final_valid[best_idx]:.3e})",
    )
    ax.plot(truth_x, truth_z, color="black", linewidth=2.3, label="Ground truth")

    q95_handle = mlines.Line2D([], [], color="C0", linewidth=8, alpha=0.12, label="5-95% x-z envelope")
    q50_handle = mlines.Line2D([], [], color="C0", linewidth=8, alpha=0.22, label="25-75% x-z envelope")
    handles, labels = ax.get_legend_handles_labels()

    ax.set_xlabel("x-component position (m)")
    ax.set_ylabel("z-component position (m)")
    ax.set_title(
        title if title is not None else
        f"{split.capitalize()} x-z envelope across seeds | traj {traj_idx}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(handles=[q95_handle, q50_handle] + handles, labels=[q95_handle.get_label(), q50_handle.get_label()] + labels)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_seed_final_loss_bar(
    all_seed_results: dict,
    save_path: Optional[str] = None,
    title: str = "Final validation loss across seeds",
    show: bool = False,
):
    arch_names = list(all_seed_results.keys())
    means = []
    stds = []
    bests = []

    for arch_name in arch_names:
        seed_results = all_seed_results[arch_name]
        good_results = _successful_results(seed_results)

        if len(good_results) == 0:
            means.append(np.nan)
            stds.append(np.nan)
            bests.append(np.nan)
            continue

        final_valid = np.asarray([r["valid_hist"][-1] for r in good_results], dtype=float)
        means.append(np.mean(final_valid))
        stds.append(np.std(final_valid))
        bests.append(np.min(final_valid))

    x = np.arange(len(arch_names))

    fig, ax = plt.subplots(figsize=(max(9.5, 0.8 * len(arch_names)), 5.8))
    ax.bar(x, means, yerr=stds, capsize=4, alpha=0.85, label="Mean ± std")
    ax.scatter(x, bests, marker="x", s=70, label="Best seed")

    ax.set_yscale("log")
    ax.set_ylabel("Final validation loss")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(arch_names, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# 4) Make all standard plots
# =========================================================
def make_seed_ablation_plots(
    all_seed_results: Optional[dict] = None,
    output_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    architectures: Optional[list[str]] = None,
    traj_idx: int = 0,
    x_idx: int = 4,
    z_idx: int = 6,
):
    if isinstance(all_seed_results, (str, bytes, os.PathLike)):
        if results_dir is not None:
            raise ValueError("Pass saved results directory either as first argument or results_dir, not both.")
        results_dir = all_seed_results
        all_seed_results = None

    if all_seed_results is None:
        if results_dir is None:
            if output_dir is None:
                raise ValueError("Provide either all_seed_results or results_dir.")
            if os.path.basename(os.path.normpath(output_dir)) == "seed_ablation_summary":
                results_dir = os.path.dirname(os.path.normpath(output_dir))
            else:
                results_dir = output_dir
        all_seed_results = load_seed_ablation_results(results_dir, architectures=architectures)

        if output_dir is None or os.path.normpath(output_dir) == os.path.normpath(results_dir):
            output_dir = os.path.join(results_dir, "seed_ablation_summary")
    elif output_dir is None:
        raise ValueError("output_dir must be provided when plotting from in-memory all_seed_results.")

    os.makedirs(output_dir, exist_ok=True)

    for arch_name, seed_results in all_seed_results.items():
        arch_dir = os.path.join(output_dir, arch_name)
        os.makedirs(arch_dir, exist_ok=True)

        plot_seed_loss_envelope(
            seed_results,
            which="train",
            save_path=os.path.join(arch_dir, "train_loss_across_seeds.png"),
            title=f"{arch_name} | train loss across seeds",
            show=False,
        )

        plot_seed_loss_envelope(
            seed_results,
            which="valid",
            save_path=os.path.join(arch_dir, "valid_loss_across_seeds.png"),
            title=f"{arch_name} | valid loss across seeds",
            show=False,
        )

        for which, filename, title_suffix in (
            ("train_displacement", "train_displacement_loss_across_seeds.png", "train displacement"),
            ("valid_displacement", "valid_displacement_loss_across_seeds.png", "valid displacement"),
            ("train_force", "train_force_loss_across_seeds.png", "train force"),
            ("valid_force", "valid_force_loss_across_seeds.png", "valid force"),
        ):
            if _has_history(seed_results, which):
                plot_seed_loss_envelope(
                    seed_results,
                    which=which,
                    save_path=os.path.join(arch_dir, filename),
                    title=f"{arch_name} | {title_suffix} loss across seeds",
                    show=False,
                )

        for component in ("x", "z"):
            plot_seed_prediction_envelope(
                seed_results,
                split="valid",
                component=component,
                traj_idx=traj_idx,
                x_idx=x_idx,
                z_idx=z_idx,
                save_path=os.path.join(arch_dir, f"valid_{component}_envelope.png"),
                title=f"{arch_name} | valid envelope | {component} | traj {traj_idx}",
                show=False,
            )

            plot_seed_prediction_all_curves(
                seed_results,
                split="valid",
                component=component,
                traj_idx=traj_idx,
                x_idx=x_idx,
                z_idx=z_idx,
                save_path=os.path.join(arch_dir, f"valid_{component}_all_seeds.png"),
                title=f"{arch_name} | valid all-seed curves | {component} | traj {traj_idx}",
                show=False,
            )

        plot_seed_prediction_xz_envelope(
            seed_results,
            split="valid",
            traj_idx=traj_idx,
            x_idx=x_idx,
            z_idx=z_idx,
            save_path=os.path.join(arch_dir, "valid_xz_envelope.png"),
            title=f"{arch_name} | valid x-z envelope | traj {traj_idx}",
            show=False,
        )

    plot_seed_final_loss_bar(
        all_seed_results,
        save_path=os.path.join(output_dir, "final_valid_loss_summary.png"),
        title="Final validation loss across seeds",
        show=False,
    )

    save_seed_summary_json(all_seed_results, output_dir=output_dir)
