import os
import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import dismech_jax as djx
from dismech_jax.solver_with_info import solve_with_info

from util import Dataset, get_slinky, validate_dataset_compatibility
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
class MaxDlambdaConvergenceProfileConfig:
    der_K_diag: tuple[float, float] = (0.2, 0.01)
    der_K_chol: tuple[float, float, float] = (0.2, 0.0, 0.01)

    hidden: tuple[int, ...] = (10,)
    corr_factor: float = 1.0
    input_mode: str = "raw"
    only_stretching_NN: bool = False
    only_bending_NN: bool = False
    zero_reference: bool = True
    activation: str = "softplus"

    n_epochs: int = 100
    lr: float = 1e-2
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

    # Keep training resilient; the point of this script is to measure
    # convergence fractions after training, not to abort the training run.
    fail_on_nonconvergence: bool = False

    output_dir: str = "max_dlambda_convergence_profile_outputs"
    save_json: bool = True
    save_plots: bool = True
    verbose: bool = True
    continue_on_failure: bool = True

    # Stop sweeping larger max_dlambda values once the profile is fully broken.
    stop_after_zero_converged: bool = True


# =========================================================
# Helpers
# =========================================================
def _make_sweep_cfg(
    cfg: MaxDlambdaConvergenceProfileConfig,
    *,
    seed: int,
    max_dlambda: float,
) -> SweepConfig:
    return SweepConfig(
        der_K_diag=cfg.der_K_diag,
        der_K_chol=cfg.der_K_chol,
        hidden=cfg.hidden,
        corr_factor=cfg.corr_factor,
        input_mode=cfg.input_mode,
        only_stretching_NN=cfg.only_stretching_NN,
        only_bending_NN=cfg.only_bending_NN,
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
        save_npz=False,
        save_plots=False,
        verbose=cfg.verbose,
        continue_on_failure=cfg.continue_on_failure,
        save_energy_landscapes=False,
    )


def _broadcast_idx_all(data: Dataset) -> np.ndarray:
    if data.idx_b.ndim == 1:
        return np.broadcast_to(np.asarray(data.idx_b), (data.qs.shape[0], data.idx_b.shape[0]))
    return np.asarray(data.idx_b)


def _broadcast_lam_all(data: Dataset) -> np.ndarray:
    if data.lambdas.ndim == 1:
        return np.broadcast_to(np.asarray(data.lambdas), (data.qs.shape[0], data.lambdas.shape[0]))
    return np.asarray(data.lambdas)


def _evaluate_trajectory_convergence(
    model,
    base,
    aux,
    idx_b,
    xb,
    lambdas,
    valid,
    *,
    max_dlambda: float,
    iters: int,
    ls_steps: int,
    abs_tol: float,
    rel_tol: float,
) -> dict:
    bc = djx.DirectBC(idx_b=idx_b, xb=xb, lambdas=lambdas)
    rod = base.with_bc(bc)

    try:
        _, info = solve_with_info(
            model=model,
            lambdas=lambdas,
            q0=rod.q0,
            aux=aux,
            sys=rod,
            iters=iters,
            ls_steps=ls_steps,
            c1=1e-4,
            max_dt=max_dlambda,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            fail_on_nonconvergence=False,
        )

        converged_steps = np.asarray(info["converged"], dtype=bool)
        valid_steps = np.asarray(valid, dtype=bool)
        active_steps = converged_steps[valid_steps]

        if active_steps.size == 0:
            traj_converged = True
            n_active = 0
            n_conv = 0
            first_fail = None
        else:
            fail_idx = np.where(~active_steps)[0]
            traj_converged = fail_idx.size == 0
            n_active = int(active_steps.size)
            n_conv = int(np.sum(active_steps))
            first_fail = None if traj_converged else int(fail_idx[0])

        return {
            "traj_converged": bool(traj_converged),
            "n_valid_steps": int(n_active),
            "n_converged_steps": int(n_conv),
            "first_failed_valid_step": first_fail,
            "exception": "",
        }
    except Exception as e:
        valid_steps = np.asarray(valid, dtype=bool)
        return {
            "traj_converged": False,
            "n_valid_steps": int(np.sum(valid_steps)),
            "n_converged_steps": 0,
            "first_failed_valid_step": 0 if np.any(valid_steps) else None,
            "exception": repr(e),
        }


def evaluate_dataset_convergence(
    model,
    base,
    aux,
    data: Dataset,
    *,
    max_dlambda: float,
    iters: int,
    ls_steps: int,
    abs_tol: float,
    rel_tol: float,
) -> dict:
    idx_all = _broadcast_idx_all(data)
    lam_all = _broadcast_lam_all(data)

    traj_records = []
    for traj_idx in range(data.qs.shape[0]):
        rec = _evaluate_trajectory_convergence(
            model,
            base,
            aux,
            idx_all[traj_idx],
            np.asarray(data.xb[traj_idx]),
            lam_all[traj_idx],
            np.asarray(data.valid[traj_idx]),
            max_dlambda=max_dlambda,
            iters=iters,
            ls_steps=ls_steps,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        rec["traj_idx"] = int(traj_idx)
        traj_records.append(rec)

    n_total = len(traj_records)
    n_converged = int(sum(int(r["traj_converged"]) for r in traj_records))

    return {
        "n_total": int(n_total),
        "n_converged": int(n_converged),
        "frac_converged": float(n_converged / max(n_total, 1)),
        "traj_records": traj_records,
    }


def plot_convergence_profile(records, arch_name: str, save_path: Optional[str] = None, show: bool = False):
    if len(records) == 0:
        return

    seeds = sorted(set(int(r["seed"]) for r in records))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    ax_train, ax_valid = axes

    for seed in seeds:
        seed_records = sorted(
            [r for r in records if int(r["seed"]) == seed],
            key=lambda r: r["max_dlambda"],
        )
        xs = np.asarray([r["max_dlambda"] for r in seed_records], dtype=float)
        train_frac = np.asarray([r["train_frac_converged"] for r in seed_records], dtype=float)
        valid_frac = np.asarray([r["valid_frac_converged"] for r in seed_records], dtype=float)

        ax_train.plot(xs, train_frac, marker="o", linewidth=2.0, label=f"seed {seed}")
        ax_valid.plot(xs, valid_frac, marker="o", linewidth=2.0, label=f"seed {seed}")

    for ax, title in zip(axes, ("Train", "Valid")):
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("max_dlambda")
        ax.set_ylabel("Converged fraction")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    handles, labels = ax_valid.get_legend_handles_labels()
    if len(handles) > 0:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 4))

    fig.suptitle(f"{arch_name}: trajectory convergence profile", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# Runner
# =========================================================
def run_max_dlambda_convergence_profile(
    properties,
    train_file: str,
    valid_file: str,
    cfg: MaxDlambdaConvergenceProfileConfig,
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

    base, aux = get_slinky(properties)
    train_data = Dataset.load(train_file)
    valid_data = Dataset.load(valid_file)
    validate_dataset_compatibility(base, train_data, "train")
    validate_dataset_compatibility(base, valid_data, "valid")

    os.makedirs(cfg.output_dir, exist_ok=True)
    all_results = {}

    for arch_name in arch_names:
        spec = registry[arch_name]
        arch_records = []
        arch_dir = os.path.join(cfg.output_dir, arch_name)
        os.makedirs(arch_dir, exist_ok=True)

        for seed in cfg.seed_list:
            for max_dlambda in cfg.max_dlambda_values:
                run_cfg = _make_sweep_cfg(cfg, seed=int(seed), max_dlambda=float(max_dlambda))

                if cfg.verbose:
                    print("=" * 100)
                    print(f"Convergence profile | arch={arch_name} | seed={seed} | max_dlambda={max_dlambda:.3e}")
                    print("=" * 100)

                result = run_one_architecture(
                    properties=properties,
                    train_file=train_file,
                    valid_file=valid_file,
                    spec=spec,
                    cfg=run_cfg,
                )

                if result["success"] and result["model"] is not None:
                    train_profile = evaluate_dataset_convergence(
                        result["model"],
                        base,
                        aux,
                        train_data,
                        max_dlambda=max_dlambda,
                        iters=cfg.iters,
                        ls_steps=cfg.ls_steps,
                        abs_tol=cfg.abs_tol,
                        rel_tol=cfg.rel_tol,
                    )
                    valid_profile = evaluate_dataset_convergence(
                        result["model"],
                        base,
                        aux,
                        valid_data,
                        max_dlambda=max_dlambda,
                        iters=cfg.iters,
                        ls_steps=cfg.ls_steps,
                        abs_tol=cfg.abs_tol,
                        rel_tol=cfg.rel_tol,
                    )
                else:
                    train_profile = {
                        "n_total": int(train_data.qs.shape[0]),
                        "n_converged": 0,
                        "frac_converged": 0.0,
                        "traj_records": [],
                    }
                    valid_profile = {
                        "n_total": int(valid_data.qs.shape[0]),
                        "n_converged": 0,
                        "frac_converged": 0.0,
                        "traj_records": [],
                    }

                record = {
                    "arch_name": spec.name,
                    "model_cls": spec.model_cls.__name__,
                    "which_case": spec.which_case,
                    "seed": int(seed),
                    "max_dlambda": float(max_dlambda),
                    "training_success": bool(result["success"]),
                    "training_failure_reason": str(result.get("failure_reason", "")),
                    "final_train_loss": float(np.asarray(result["train_hist"], dtype=float)[-1]),
                    "final_valid_loss": float(np.asarray(result["valid_hist"], dtype=float)[-1]),
                    "train_n_converged": int(train_profile["n_converged"]),
                    "train_n_total": int(train_profile["n_total"]),
                    "train_frac_converged": float(train_profile["frac_converged"]),
                    "valid_n_converged": int(valid_profile["n_converged"]),
                    "valid_n_total": int(valid_profile["n_total"]),
                    "valid_frac_converged": float(valid_profile["frac_converged"]),
                    "train_traj_records": train_profile["traj_records"],
                    "valid_traj_records": valid_profile["traj_records"],
                    "exp_dir": result.get("exp_dir", None),
                    "exp_name": result.get("exp_name", None),
                }
                arch_records.append(record)

                if cfg.verbose:
                    print(
                        f"[{arch_name}] seed={seed} | max_dlambda={max_dlambda:.3e} | "
                        f"train {record['train_n_converged']}/{record['train_n_total']} | "
                        f"valid {record['valid_n_converged']}/{record['valid_n_total']}"
                    )

                if cfg.stop_after_zero_converged:
                    if record["train_n_converged"] == 0 and record["valid_n_converged"] == 0:
                        if cfg.verbose:
                            print(
                                f"Stopping larger max_dlambda values for {arch_name}, seed={seed} "
                                "because no train or valid trajectories converged."
                            )
                        break

        all_results[arch_name] = arch_records

        if cfg.save_json:
            with open(os.path.join(arch_dir, "convergence_profile.json"), "w") as f:
                json.dump(
                    {
                        "arch_name": arch_name,
                        "cfg": asdict(cfg),
                        "records": arch_records,
                    },
                    f,
                    indent=2,
                )

        if cfg.save_plots:
            plot_convergence_profile(
                arch_records,
                arch_name=arch_name,
                save_path=os.path.join(arch_dir, "convergence_profile.png"),
                show=False,
            )

    if cfg.save_json:
        with open(os.path.join(cfg.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    train_file = "../experiment_data/n5_combined_train_dataset.npz"
    valid_file = "../experiment_data/n5_combined_test_dataset.npz"

    from properties import Properties

    properties = Properties(
        length=0.2,
        r0=0.005,
        density=800.0,
        E=1e6,
        N=5,
        mass=0.3,
    )

    cfg = MaxDlambdaConvergenceProfileConfig(
        hidden=(10,),
        corr_factor=1.0,
        input_mode="raw",
        zero_reference=True,
        activation="softplus",
        n_epochs=100,
        lr=1e-2,
        seed_list=(0,),
        max_dlambda_values=(1e-2, 5e-2, 1e-1, 5e-1, 1.0),
        iters=5,
        ls_steps=10,
        abs_tol=1e-8,
        rel_tol=1e-6,
        output_dir="max_dlambda_convergence_profile_outputs",
    )

    selected_architectures = [
        "mlp_energy",
        "icnn_energy",
        "diag_energy_baseline",
        "chol_stiffness_mlp",
    ]

    run_max_dlambda_convergence_profile(
        properties=properties,
        train_file=train_file,
        valid_file=valid_file,
        cfg=cfg,
        selected_architectures=selected_architectures,
    )
