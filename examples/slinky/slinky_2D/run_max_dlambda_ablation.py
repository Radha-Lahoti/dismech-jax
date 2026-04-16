import os
import json
from dataclasses import dataclass, asdict, replace
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from util import Dataset, get_slinky, predict, train_model
from run_architectures import (
    ArchSpec,
    SweepConfig,
    build_architecture_registry,
    make_model_params,
)


# =========================================================
# Config
# =========================================================
@dataclass(frozen=True)
class MaxDlambdaAblationConfig:
    # ---- reuse most of your architecture/training flags ----
    der_K_diag: tuple[float, float] = (0.1, 0.1)
    der_K_chol: tuple[float, float, float] = (0.1, 0.0, 0.1)

    hidden: tuple[int, ...] = (10, 10)
    corr_factor: float = 1.0
    input_mode: str = "raw"
    zero_reference: bool = True
    activation: str = "softplus"
    seed: int = 0

    n_epochs: int = 200
    lr: float = 1e-2
    valid_every: int = 1

    # ---- solver settings for this ablation ----
    iters: int = 20
    ls_steps: int = 10

    # max_dlambda sweep: default logspace from 1e-2 to 1
    max_dlambda_values: tuple[float, ...] = (
        1e-2, 2e-2, 5e-2,
        1e-1, 2e-1, 5e-1,
        1.0,
    )

    # stop testing larger values for one architecture once it fails
    stop_after_first_failure: bool = True

    # optional: also reject if final prediction contains NaN/Inf
    check_predictions: bool = True

    output_dir: str = "max_dlambda_ablation_outputs"
    save_json: bool = True
    save_npz: bool = True
    save_plots: bool = True
    verbose: bool = True


# =========================================================
# Helpers
# =========================================================
def _to_float(x):
    return float(np.asarray(x))


def _all_finite(x) -> bool:
    x = np.asarray(x)
    return np.all(np.isfinite(x))


def _make_sweep_cfg(ab_cfg: MaxDlambdaAblationConfig) -> SweepConfig:
    """Convert ablation config into the SweepConfig expected by
    make_model_params from run_architectures.py.
    """
    return SweepConfig(
        der_K_diag=ab_cfg.der_K_diag,
        der_K_chol=ab_cfg.der_K_chol,
        hidden=ab_cfg.hidden,
        corr_factor=ab_cfg.corr_factor,
        input_mode=ab_cfg.input_mode,
        zero_reference=ab_cfg.zero_reference,
        activation=ab_cfg.activation,
        n_epochs=ab_cfg.n_epochs,
        lr=ab_cfg.lr,
        seed=ab_cfg.seed,
        output_dir=ab_cfg.output_dir,   # not used directly here
        save_npz=ab_cfg.save_npz,
        save_plots=ab_cfg.save_plots,
        verbose=ab_cfg.verbose,
    )


def _make_arch_dir(root: str, arch_name: str) -> str:
    d = os.path.join(root, arch_name)
    os.makedirs(d, exist_ok=True)
    return d


def _make_run_dir(root: str, arch_name: str, max_dlambda: float) -> str:
    tag = f"maxdl_{max_dlambda:.3e}".replace("+", "")
    d = os.path.join(root, arch_name, tag)
    os.makedirs(d, exist_ok=True)
    return d


def _check_run_stability(
    model,
    train_hist,
    valid_hist,
    base,
    aux,
    train_data,
    valid_data,
    max_dlambda,
    iters,
    ls_steps,
    check_predictions=True,
):
    out = {
        "train_hist_finite": _all_finite(train_hist),
        "valid_hist_finite": _all_finite(valid_hist),
        "train_pred_finite": True,
        "valid_pred_finite": True,
        "success": False,
        "failure_reason": "",
    }

    if not out["train_hist_finite"]:
        out["failure_reason"] = "nonfinite_train_hist"
        return out

    if not out["valid_hist_finite"]:
        out["failure_reason"] = "nonfinite_valid_hist"
        return out

    if check_predictions:
        train_pred = predict(
            model, base, aux,
            train_data.idx_b, train_data.xb, train_data.lambdas,
            max_dlambda=max_dlambda,
            iters=iters,
            ls_steps=ls_steps,
        )
        valid_pred = predict(
            model, base, aux,
            valid_data.idx_b, valid_data.xb, valid_data.lambdas,
            max_dlambda=max_dlambda,
            iters=iters,
            ls_steps=ls_steps,
        )

        out["train_pred_finite"] = _all_finite(train_pred)
        out["valid_pred_finite"] = _all_finite(valid_pred)

        if not out["train_pred_finite"]:
            out["failure_reason"] = "nonfinite_train_pred"
            return out
        if not out["valid_pred_finite"]:
            out["failure_reason"] = "nonfinite_valid_pred"
            return out

    out["success"] = True
    return out


def plot_architecture_curve(records, arch_name, save_path=None, show=False):
    xs = np.array([r["max_dlambda"] for r in records], dtype=float)
    train_last = np.array([r["final_train_loss"] for r in records], dtype=float)
    valid_last = np.array([r["final_valid_loss"] for r in records], dtype=float)
    success = np.array([1.0 if r["success"] else 0.0 for r in records], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(xs, train_last, marker="o", linewidth=2.0, label="Train final loss")
    ax.plot(xs, valid_last, marker="s", linewidth=2.0, label="Valid final loss")

    failed = success < 0.5
    if np.any(failed):
        ax.scatter(xs[failed], valid_last[failed], marker="x", s=80, label="Failed")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("max_dlambda")
    ax.set_ylabel("Final loss")
    ax.set_title(f"{arch_name}: continuation-step ablation")
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
        ok = np.array([r["success"] for r in records], dtype=bool)

        ax.scatter(xs[ok], ys[ok], marker="o", s=60)
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

    train_data = Dataset.load(train_file)
    valid_data = Dataset.load(valid_file)
    base, aux = get_slinky(properties)

    sweep_cfg = _make_sweep_cfg(cfg)

    records = []
    for max_dlambda in cfg.max_dlambda_values:
        run_dir = _make_run_dir(root_dir, spec.name, max_dlambda)

        if cfg.verbose:
            print("=" * 100)
            print(f"Architecture : {spec.name}")
            print(f"which_case   : {spec.which_case}")
            print(f"max_dlambda  : {max_dlambda:.3e}")
            print(f"iters        : {cfg.iters}")
            print(f"ls_steps     : {cfg.ls_steps}")
            print("=" * 100)

        params = make_model_params(sweep_cfg, spec)

        success = False
        failure_reason = ""
        train_hist = [np.nan]
        valid_hist = [np.nan]

        try:
            model, train_hist, valid_hist = train_model(
                properties=properties,
                model_cls=spec.model_cls,
                params=params,
                train_file=train_file,
                valid_file=valid_file,
                n_epochs=cfg.n_epochs,
                lr=cfg.lr,
                valid_every=cfg.valid_every,
                max_dlambda=max_dlambda,
                iters=cfg.iters,
                ls_steps=cfg.ls_steps,
            )

            status = _check_run_stability(
                model=model,
                train_hist=train_hist,
                valid_hist=valid_hist,
                base=base,
                aux=aux,
                train_data=train_data,
                valid_data=valid_data,
                max_dlambda=max_dlambda,
                iters=cfg.iters,
                ls_steps=cfg.ls_steps,
                check_predictions=cfg.check_predictions,
            )
            success = status["success"]
            failure_reason = status["failure_reason"]

        except Exception as e:
            success = False
            failure_reason = f"exception: {repr(e)}"

        rec = {
            "arch_name": spec.name,
            "model_cls": spec.model_cls.__name__,
            "which_case": spec.which_case,
            "max_dlambda": float(max_dlambda),
            "iters": int(cfg.iters),
            "ls_steps": int(cfg.ls_steps),
            "success": bool(success),
            "failure_reason": failure_reason,
            "final_train_loss": _to_float(train_hist[-1]) if len(train_hist) else np.nan,
            "final_valid_loss": _to_float(valid_hist[-1]) if len(valid_hist) else np.nan,
            "min_train_loss": float(np.nanmin(np.asarray(train_hist, dtype=float))),
            "min_valid_loss": float(np.nanmin(np.asarray(valid_hist, dtype=float))),
            "n_epochs": int(cfg.n_epochs),
            "seed": int(cfg.seed),
        }
        records.append(rec)

        if cfg.save_json:
            with open(os.path.join(run_dir, "result.json"), "w") as f:
                json.dump(rec, f, indent=2)

        if cfg.save_npz:
            np.savez(
                os.path.join(run_dir, "histories.npz"),
                train_hist=np.asarray(train_hist, dtype=float),
                valid_hist=np.asarray(valid_hist, dtype=float),
            )

        if cfg.verbose:
            tag = "SUCCESS" if success else f"FAIL ({failure_reason})"
            print(
                f"[{spec.name}] max_dlambda={max_dlambda:.3e} | "
                f"train={rec['final_train_loss']:.3e} | "
                f"valid={rec['final_valid_loss']:.3e} | {tag}"
            )

        if (not success) and cfg.stop_after_first_failure:
            if cfg.verbose:
                print(
                    f"Stopping larger max_dlambda values for {spec.name} "
                    f"after first failure."
                )
            break

    if cfg.save_json:
        with open(os.path.join(arch_dir, "summary.json"), "w") as f:
            json.dump(
                {
                    "arch_name": spec.name,
                    "records": records,
                    "cfg": asdict(cfg),
                },
                f,
                indent=2,
            )

    if cfg.save_plots and len(records) > 0:
        plot_architecture_curve(
            records,
            arch_name=spec.name,
            save_path=os.path.join(arch_dir, "max_dlambda_curve.png"),
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
        if unknown:
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

    if cfg.save_plots:
        plot_success_summary(
            all_results,
            save_path=os.path.join(cfg.output_dir, "success_summary.png"),
            show=False,
        )

    if cfg.save_json:
        with open(os.path.join(cfg.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    return all_results


# =========================================================
# Optional helper: best stable max_dlambda per architecture
# =========================================================
def summarize_best_stable_step(all_results):
    summary = {}
    for arch_name, records in all_results.items():
        stable = [r for r in records if r["success"]]
        if len(stable) == 0:
            summary[arch_name] = {
                "largest_stable_max_dlambda": None,
                "best_valid_loss_among_stable": None,
            }
        else:
            summary[arch_name] = {
                "largest_stable_max_dlambda": max(r["max_dlambda"] for r in stable),
                "best_valid_loss_among_stable": min(r["final_valid_loss"] for r in stable),
            }
    return summary


# =========================================================
# Example main
# =========================================================
if __name__ == "__main__":
    # train_file = "../experiment_data/train_dataset.npz"
    # valid_file = "../experiment_data/test_dataset.npz"

    train_file = "../experiment_data/n5_combined_train_dataset.npz"
    valid_file = "../experiment_data/n5_combined_test_dataset.npz"

    # replace with your actual properties object
    properties = None

    cfg = MaxDlambdaAblationConfig(
        hidden=(10,),
        corr_factor=1.0,
        input_mode="raw",
        zero_reference=True,
        activation="softplus",
        seed=0,
        n_epochs=200,
        lr=1e-3,
        iters=20,
        ls_steps=10,
        max_dlambda_values=(1e-2, 5e-2, 1e-1, 5e-1, 1.0),
        stop_after_first_failure=True,
        check_predictions=True,
        output_dir="max_dlambda_ablation_outputs",
        save_json=True,
        save_npz=True,
        save_plots=True,
        verbose=True,
    )

    # Example: choose only selected architectures
    selected_architectures = [
        "diag_energy_baseline",
        "chol_stiffness_mlp",
        # add whatever you want here
    ]

    from properties import Properties

    properties = Properties(
        length = 0.2,
        r0 = 0.005,
        axs = None,
        jxs = None,
        ixs1 = None,
        ixs2 = None,
        density = 800.0,
        E = 1e6,
        N = 5, # 3 or 5
        # start = jax.numpy.array([0, 0, 0]),
        # end = jax.numpy.array([0.266166, 0., 0.01240256]),
        mass = 0.3
    )

    results = run_max_dlambda_ablation(
        properties=properties,
        train_file=train_file,
        valid_file=valid_file,
        cfg=cfg,
        selected_architectures=selected_architectures,
    )

    summary = summarize_best_stable_step(results)
    print("\nLargest stable max_dlambda by architecture:")
    print(json.dumps(summary, indent=2))