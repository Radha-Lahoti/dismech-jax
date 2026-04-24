import os
import json
from dataclasses import dataclass, asdict, replace
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from run_architectures import (
    ArchSpec,
    SweepConfig,
    build_architecture_registry,
    run_one_architecture,
)


# =========================================================
# Config
# =========================================================
@dataclass(frozen=True)
class MaxDlambdaAblationConfig:
    # -------------------------
    # Keep these aligned with SweepConfig defaults / fields
    # -------------------------
    der_K_diag: tuple[float, float] = (0.2, 0.01)
    der_K_chol: tuple[float, float, float] = (0.2, 0.0, 0.01)

    hidden: tuple[int, ...] = (10,)
    corr_factor: float = 1.0
    input_mode: str = "raw"
    only_stretching_NN: bool = False
    zero_reference: bool = True
    activation: str = "softplus"

    n_epochs: int = 100
    lr: float = 1e-2
    seed: int = 0
    seed_list: tuple[int, ...] = (0,)

    valid_every: int = 1
    max_dlambda_values: tuple[float, ...] = (
        1e-2, 2e-2, 5e-2,
        1e-1, 2e-1, 5e-1,
        1.0,
    )
    iters: int = 5
    ls_steps: int = 10
    abs_tol: float = 1e-8
    rel_tol: float = 1e-6
    fail_on_nonconvergence: bool = False

    output_dir: str = "max_dlambda_ablation_outputs"
    save_npz: bool = True
    save_plots: bool = True
    verbose: bool = True
    continue_on_failure: bool = True

    # -------------------------
    # Ablation-only controls
    # -------------------------
    stop_after_first_failure: bool = False
    strict_finite_check: bool = False
    save_summary_json: bool = True
    save_summary_plots: bool = True


# =========================================================
# Helpers
# =========================================================
def _to_numpy(x):
    return np.asarray(x)


def _all_finite(x) -> bool:
    arr = np.asarray(x, dtype=float)
    return np.all(np.isfinite(arr))


def _safe_last(hist):
    if hist is None:
        return np.nan
    arr = np.asarray(hist, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(arr[-1])


def _safe_min(hist):
    if hist is None:
        return np.nan
    arr = np.asarray(hist, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.nan
    return float(np.nanmin(arr))


def _make_sweep_cfg(cfg: MaxDlambdaAblationConfig, *, seed: int, max_dlambda: float) -> SweepConfig:
    """Build the exact SweepConfig consumed by run_one_architecture(...)."""
    return SweepConfig(
        der_K_diag=cfg.der_K_diag,
        der_K_chol=cfg.der_K_chol,
        hidden=cfg.hidden,
        corr_factor=cfg.corr_factor,
        input_mode=cfg.input_mode,
        only_stretching_NN=cfg.only_stretching_NN,
        zero_reference=cfg.zero_reference,
        activation=cfg.activation,
        n_epochs=cfg.n_epochs,
        lr=cfg.lr,
        seed=seed,
        valid_every=cfg.valid_every,
        max_dlambda=max_dlambda,
        iters=cfg.iters,
        ls_steps=cfg.ls_steps,
        abs_tol=cfg.abs_tol,
        rel_tol=cfg.rel_tol,
        fail_on_nonconvergence=cfg.fail_on_nonconvergence,
        output_dir=cfg.output_dir,
        save_npz=cfg.save_npz,
        save_plots=cfg.save_plots,
        verbose=cfg.verbose,
        continue_on_failure=cfg.continue_on_failure,
    )


def _record_from_result(
    result: dict,
    spec: ArchSpec,
    seed: int,
    max_dlambda: float,
    strict_finite_check: bool,
) -> dict:
    train_hist = result.get("train_hist", None)
    valid_hist = result.get("valid_hist", None)

    train_hist_finite = _all_finite(train_hist) if train_hist is not None else False
    valid_hist_finite = _all_finite(valid_hist) if valid_hist is not None else False

    success = bool(result.get("success", False))
    failure_reason = str(result.get("failure_reason", ""))

    # Optional extra strictness, disabled by default so behavior
    # stays as close as possible to run_architectures.py.
    if strict_finite_check:
        if not train_hist_finite:
            success = False
            failure_reason = "nonfinite_train_hist"
        elif not valid_hist_finite:
            success = False
            failure_reason = "nonfinite_valid_hist"

    return {
        "arch_name": spec.name,
        "model_cls": spec.model_cls.__name__,
        "which_case": spec.which_case,
        "seed": int(seed),
        "max_dlambda": float(max_dlambda),
        "iters": int(result["cfg"].iters),
        "ls_steps": int(result["cfg"].ls_steps),
        "abs_tol": float(result["cfg"].abs_tol),
        "rel_tol": float(result["cfg"].rel_tol),
        "fail_on_nonconvergence": bool(result["cfg"].fail_on_nonconvergence),
        "n_epochs": int(result["cfg"].n_epochs),
        "lr": float(result["cfg"].lr),
        "success": bool(success),
        "failure_reason": failure_reason,
        "train_hist_finite": bool(train_hist_finite),
        "valid_hist_finite": bool(valid_hist_finite),
        "final_train_loss": _safe_last(train_hist),
        "final_valid_loss": _safe_last(valid_hist),
        "min_train_loss": _safe_min(train_hist),
        "min_valid_loss": _safe_min(valid_hist),
        "exp_dir": result.get("exp_dir", None),
        "exp_name": result.get("exp_name", None),
    }


def _make_arch_dir(root: str, arch_name: str) -> str:
    d = os.path.join(root, arch_name)
    os.makedirs(d, exist_ok=True)
    return d


def _make_seed_dir(root: str, arch_name: str, seed: int) -> str:
    d = os.path.join(root, arch_name, f"seed_{seed}")
    os.makedirs(d, exist_ok=True)
    return d


# =========================================================
# Plots
# =========================================================
def plot_architecture_curve(records, arch_name, save_path=None, show=False):
    if len(records) == 0:
        return

    xs = np.array([r["max_dlambda"] for r in records], dtype=float)
    train_last = np.array([r["final_train_loss"] for r in records], dtype=float)
    valid_last = np.array([r["final_valid_loss"] for r in records], dtype=float)
    success = np.array([bool(r["success"]) for r in records], dtype=bool)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ok_train = success & np.isfinite(train_last)
    ok_valid = success & np.isfinite(valid_last)
    fail = ~success

    if np.any(ok_train):
        ax.plot(xs[ok_train], train_last[ok_train], marker="o", linewidth=2.0, label="Train")
    if np.any(ok_valid):
        ax.plot(xs[ok_valid], valid_last[ok_valid], marker="s", linewidth=2.0, label="Valid")

    if np.any(fail):
        finite_pos = np.concatenate([
            train_last[np.isfinite(train_last) & (train_last > 0)],
            valid_last[np.isfinite(valid_last) & (valid_last > 0)],
        ])
        y_fail = np.min(finite_pos) if finite_pos.size > 0 else 1.0
        ax.scatter(xs[fail], np.full(np.sum(fail), y_fail), marker="x", s=80, label="Failed")

    ax.set_xscale("log")
    if np.any(np.isfinite(np.concatenate([train_last, valid_last])) &
              (np.concatenate([train_last, valid_last]) > 0)):
        ax.set_yscale("log")

    ax.set_xlabel("max_dlambda")
    ax.set_ylabel("Final loss")
    ax.set_title(f"{arch_name}: max_dlambda sweep")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_seed_envelope(records, arch_name, save_path=None, show=False):
    if len(records) == 0:
        return

    seeds = sorted(set(int(r["seed"]) for r in records))
    xs_all = sorted(set(float(r["max_dlambda"]) for r in records))

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for seed in seeds:
        seed_records = [r for r in records if int(r["seed"]) == seed]
        seed_records = sorted(seed_records, key=lambda r: r["max_dlambda"])

        xs = np.array([r["max_dlambda"] for r in seed_records], dtype=float)
        ys = np.array([r["final_valid_loss"] for r in seed_records], dtype=float)
        ok = np.array([bool(r["success"]) for r in seed_records], dtype=bool)

        good = ok & np.isfinite(ys) & (ys > 0)
        if np.any(good):
            ax.plot(xs[good], ys[good], marker="o", linewidth=1.5, alpha=0.7, label=f"seed {seed}")

    all_valid = []
    for x in xs_all:
        vals = [
            float(r["final_valid_loss"])
            for r in records
            if float(r["max_dlambda"]) == x
            and bool(r["success"])
            and np.isfinite(r["final_valid_loss"])
            and float(r["final_valid_loss"]) > 0
        ]
        if len(vals) > 0:
            all_valid.append((x, np.min(vals), np.median(vals), np.max(vals)))

    if len(all_valid) > 0:
        xs = np.array([t[0] for t in all_valid], dtype=float)
        y_min = np.array([t[1] for t in all_valid], dtype=float)
        y_med = np.array([t[2] for t in all_valid], dtype=float)
        y_max = np.array([t[3] for t in all_valid], dtype=float)

        ax.fill_between(xs, y_min, y_max, alpha=0.2)
        ax.plot(xs, y_med, linewidth=3.0, label="median valid")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("max_dlambda")
    ax.set_ylabel("Final valid loss")
    ax.set_title(f"{arch_name}: seed sweep envelope")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_success_summary(all_results, save_path=None, show=False):
    arch_names = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(arch_names)), 5.0))

    for i, arch_name in enumerate(arch_names):
        records = all_results[arch_name]
        xs = np.array([r["max_dlambda"] for r in records], dtype=float)
        ys = np.full_like(xs, i, dtype=float)
        ok = np.array([bool(r["success"]) for r in records], dtype=bool)

        if np.any(ok):
            ax.scatter(xs[ok], ys[ok], marker="o", s=60)
        if np.any(~ok):
            ax.scatter(xs[~ok], ys[~ok], marker="x", s=70)

    ax.set_xscale("log")
    ax.set_yticks(np.arange(len(arch_names)))
    ax.set_yticklabels(arch_names)
    ax.set_xlabel("max_dlambda")
    ax.set_title("Stable vs failed runs across architectures")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# One architecture
# =========================================================
def run_one_architecture_max_dlambda_ablation(
    properties,
    train_file: str,
    valid_file: str,
    spec: ArchSpec,
    cfg: MaxDlambdaAblationConfig,
):
    root_dir = cfg.output_dir
    os.makedirs(root_dir, exist_ok=True)
    arch_dir = _make_arch_dir(root_dir, spec.name)

    records = []

    for seed in cfg.seed_list:
        seed_dir = _make_seed_dir(root_dir, spec.name, seed)

        stop_this_seed = False
        for max_dlambda in cfg.max_dlambda_values:
            if stop_this_seed:
                break

            run_cfg = _make_sweep_cfg(cfg, seed=seed, max_dlambda=max_dlambda)

            if cfg.verbose:
                print("=" * 100)
                print(f"Architecture : {spec.name}")
                print(f"which_case   : {spec.which_case}")
                print(f"seed         : {seed}")
                print(f"max_dlambda  : {max_dlambda:.3e}")
                print(f"iters        : {run_cfg.iters}")
                print(f"ls_steps     : {run_cfg.ls_steps}")
                print(f"abs_tol      : {run_cfg.abs_tol:.3e}")
                print(f"rel_tol      : {run_cfg.rel_tol:.3e}")
                print(f"fail_on_nonconvergence : {run_cfg.fail_on_nonconvergence}")
                print("=" * 100)

            # IMPORTANT:
            # We call the SAME runner as run_architectures.py.
            result = run_one_architecture(
                properties=properties,
                train_file=train_file,
                valid_file=valid_file,
                spec=spec,
                cfg=run_cfg,
            )

            rec = _record_from_result(
                result=result,
                spec=spec,
                seed=seed,
                max_dlambda=max_dlambda,
                strict_finite_check=cfg.strict_finite_check,
            )
            records.append(rec)

            if cfg.verbose:
                tag = "SUCCESS" if rec["success"] else f"FAIL ({rec['failure_reason']})"
                train_str = f"{rec['final_train_loss']:.3e}" if np.isfinite(rec["final_train_loss"]) else "nan"
                valid_str = f"{rec['final_valid_loss']:.3e}" if np.isfinite(rec["final_valid_loss"]) else "nan"
                print(
                    f"[{spec.name}] seed={seed} | max_dlambda={max_dlambda:.3e} | "
                    f"train={train_str} | valid={valid_str} | {tag}"
                )

            if (not rec["success"]) and cfg.stop_after_first_failure:
                stop_this_seed = True
                if cfg.verbose:
                    print(f"Stopping larger max_dlambda values for {spec.name}, seed={seed} after first failure.")

        if cfg.save_summary_json:
            seed_records = [r for r in records if int(r["seed"]) == int(seed)]
            with open(os.path.join(seed_dir, "summary.json"), "w") as f:
                json.dump(
                    {
                        "arch_name": spec.name,
                        "seed": int(seed),
                        "records": seed_records,
                        "cfg": asdict(cfg),
                    },
                    f,
                    indent=2,
                )

    if cfg.save_summary_json:
        with open(os.path.join(arch_dir, "summary_all_seeds.json"), "w") as f:
            json.dump(
                {
                    "arch_name": spec.name,
                    "records": records,
                    "cfg": asdict(cfg),
                },
                f,
                indent=2,
            )

    if cfg.save_summary_plots and len(records) > 0:
        # Per-seed individual curves
        for seed in sorted(set(int(r["seed"]) for r in records)):
            seed_records = [r for r in records if int(r["seed"]) == seed]
            plot_architecture_curve(
                seed_records,
                arch_name=f"{spec.name} | seed {seed}",
                save_path=os.path.join(arch_dir, f"max_dlambda_curve_seed_{seed}.png"),
                show=False,
            )

        # Aggregate across seeds
        if len(set(int(r["seed"]) for r in records)) > 1:
            plot_seed_envelope(
                records,
                arch_name=spec.name,
                save_path=os.path.join(arch_dir, "max_dlambda_seed_envelope.png"),
                show=False,
            )

    return records


# =========================================================
# Multi-architecture runner
# =========================================================
def run_max_dlambda_ablation(
    properties,
    train_file: str,
    valid_file: str,
    cfg: MaxDlambdaAblationConfig,
    selected_architectures: Optional[list[str]] = None,
):
    registry = build_architecture_registry()

    if selected_architectures is None:
        arch_names = list(registry.keys())
    else:
        unknown = [name for name in selected_architectures if name not in registry]
        if len(unknown) > 0:
            raise ValueError(f"Unknown architecture names: {unknown}")
        arch_names = selected_architectures

    all_results = {}
    for arch_name in arch_names:
        spec = registry[arch_name]
        all_results[arch_name] = run_one_architecture_max_dlambda_ablation(
            properties=properties,
            train_file=train_file,
            valid_file=valid_file,
            spec=spec,
            cfg=cfg,
        )

    if cfg.save_summary_plots:
        plot_success_summary(
            all_results,
            save_path=os.path.join(cfg.output_dir, "success_summary.png"),
            show=False,
        )

    if cfg.save_summary_json:
        with open(os.path.join(cfg.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    return all_results


# =========================================================
# Optional summaries
# =========================================================
def summarize_best_stable_step(all_results):
    summary = {}
    for arch_name, records in all_results.items():
        by_seed = {}
        seeds = sorted(set(int(r["seed"]) for r in records))

        for seed in seeds:
            stable = [
                r for r in records
                if int(r["seed"]) == seed
                and bool(r["success"])
                and np.isfinite(r["final_valid_loss"])
            ]
            if len(stable) == 0:
                by_seed[str(seed)] = {
                    "largest_stable_max_dlambda": None,
                    "best_valid_loss_among_stable": None,
                }
            else:
                by_seed[str(seed)] = {
                    "largest_stable_max_dlambda": max(r["max_dlambda"] for r in stable),
                    "best_valid_loss_among_stable": min(r["final_valid_loss"] for r in stable),
                }

        summary[arch_name] = by_seed

    return summary


# =========================================================
# Example main
# =========================================================
if __name__ == "__main__":
    train_file = "../experiment_data/n5_combined_train_dataset.npz"
    valid_file = "../experiment_data/n5_combined_test_dataset.npz"

    from properties import Properties

    properties = Properties(
        length=0.2,
        r0=0.005,
        axs=None,
        jxs=None,
        ixs1=None,
        ixs2=None,
        density=800.0,
        E=1e6,
        N=5,
        mass=0.3,
    )

    cfg = MaxDlambdaAblationConfig(
        hidden=(10,),
        corr_factor=1.0,
        input_mode="raw",
        zero_reference=True,
        activation="softplus",

        # Keep these identical to your run_architectures.py test
        n_epochs=100,
        lr=1e-2,
        seed_list=(0,),
        valid_every=1,

        max_dlambda_values=(1e-2, 5e-2, 1e-1, 5e-1, 1.0),
        iters=5,
        ls_steps=10,
        abs_tol=1e-8,
        rel_tol=1e-6,
        fail_on_nonconvergence=False,

        output_dir="max_dlambda_ablation_outputs",
        save_npz=True,
        save_plots=True,
        verbose=True,
        continue_on_failure=True,

        stop_after_first_failure=False,

        # Keep False if you want behavior as close as possible
        # to run_architectures.py.
        strict_finite_check=False,

        save_summary_json=True,
        save_summary_plots=True,
    )

    selected_architectures = [
        "mlp_energy",
        "icnn_energy",
        "diag_energy_baseline",
        "chol_stiffness_mlp",
    ]

    results = run_max_dlambda_ablation(
        properties=properties,
        train_file=train_file,
        valid_file=valid_file,
        cfg=cfg,
        selected_architectures=selected_architectures,
    )

    summary = summarize_best_stable_step(results)
    print("\nLargest stable max_dlambda by architecture and seed:")
    print(json.dumps(summary, indent=2))
