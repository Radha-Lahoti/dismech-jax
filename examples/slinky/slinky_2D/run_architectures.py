import os
import json
from dataclasses import dataclass, replace, asdict, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

import jax
import jax.numpy as jnp

from util import Dataset, get_slinky, predict, train_model
from util_energy_plots import EnergyLandscapeSpec, make_energy_snapshot_fn
from Energy_NN_architectures import (
    ModelParams,
    DiagonalPlusEnergyNN,
    CholeskyPlusEnergyNN,
    DiagonalPlusStiffnessNN,
    CholeskyPlusStiffnessNN,
    CholeskyPlusStiffnessSignedNN,
)


# =========================================================
# 1) Architecture registry
# =========================================================
@dataclass(frozen=True)
class ArchSpec:
    name: str
    model_cls: type
    which_case: str


def build_architecture_registry() -> dict[str, ArchSpec]:
    return {
        # -------------------------
        # Energy families
        # -------------------------
        "diag_energy_baseline": ArchSpec(
            name="diag_energy_baseline",
            model_cls=DiagonalPlusEnergyNN,
            which_case="baseline",
        ),
        "diag_energy_mlp": ArchSpec(
            name="diag_energy_mlp",
            model_cls=DiagonalPlusEnergyNN,
            which_case="MLP",
        ),
        "diag_energy_icnn": ArchSpec(
            name="diag_energy_icnn",
            model_cls=DiagonalPlusEnergyNN,
            which_case="ICNN",
        ),
        "chol_energy_baseline": ArchSpec(
            name="chol_energy_baseline",
            model_cls=CholeskyPlusEnergyNN,
            which_case="baseline",
        ),
        "chol_energy_mlp": ArchSpec(
            name="chol_energy_mlp",
            model_cls=CholeskyPlusEnergyNN,
            which_case="MLP",
        ),
        "chol_energy_icnn": ArchSpec(
            name="chol_energy_icnn",
            model_cls=CholeskyPlusEnergyNN,
            which_case="ICNN",
        ),

        # -------------------------
        # Stiffness families
        # -------------------------
        "diag_stiffness_mlp": ArchSpec(
            name="diag_stiffness_mlp",
            model_cls=DiagonalPlusStiffnessNN,
            which_case="MLP",
        ),
        "diag_stiffness_icnn": ArchSpec(
            name="diag_stiffness_icnn",
            model_cls=DiagonalPlusStiffnessNN,
            which_case="ICNN",
        ),
        "chol_stiffness_mlp": ArchSpec(
            name="chol_stiffness_mlp",
            model_cls=CholeskyPlusStiffnessNN,
            which_case="MLP",
        ),
        "chol_stiffness_icnn": ArchSpec(
            name="chol_stiffness_icnn",
            model_cls=CholeskyPlusStiffnessNN,
            which_case="ICNN",
        ),
        "chol_stiffness_signed_mlp": ArchSpec(
            name="chol_stiffness_signed_mlp",
            model_cls=CholeskyPlusStiffnessSignedNN,
            which_case="MLP",
        ),
        "chol_stiffness_signed_icnn": ArchSpec(
            name="chol_stiffness_signed_icnn",
            model_cls=CholeskyPlusStiffnessSignedNN,
            which_case="ICNN",
        ),
    }


# =========================================================
# 2) Sweep config
# =========================================================
@dataclass(frozen=True)
class SweepConfig:
    # Separate initializations because diagonal and Cholesky
    # models expect different der_K shapes.
    der_K_diag: tuple[float, float] = (0.2, 0.01)
    der_K_chol: tuple[float, float, float] = (0.2, 0.0, 0.01)

    hidden: tuple[int, ...] = (10,)
    corr_factor: float = 1.0
    input_mode: str = "raw"          # "raw" or "invariant"
    zero_reference: bool = True      # relevant for energy-correction families
    activation: str = "softplus"     # relevant for MLP nets
    n_epochs: int = 100
    lr: float = 1e-2
    seed: int = 0

    # solver / training-loop compatibility with new util.py
    valid_every: int = 1
    max_dlambda: float = 1e-2
    iters: int = 5
    ls_steps: int = 10
    abs_tol: float = 1e-8
    rel_tol: float = 1e-6
    fail_on_nonconvergence: bool = False

    output_dir: str = "arch_sweep_outputs"
    save_npz: bool = True
    save_plots: bool = True
    verbose: bool = True
    seed_list: tuple[int, ...] = (0, 1, 2, 3, 4)

    # whether sweep should continue if one architecture fails
    continue_on_failure: bool = True

    # -------------------------
    # Energy-landscape snapshots
    # -------------------------
    save_energy_landscapes: bool = True
    energy_snapshot_initial: bool = True
    energy_snapshot_final: bool = True
    energy_snapshot_epochs: tuple[int, ...] = ()
    energy_snapshot_every: Optional[int] = None
    energy_snapshot_use_valid: bool = True
    energy_snapshot_dpi: int = 180
    energy_snapshot_n_grid: Optional[int] = None
    energy_snapshot_dirname: str = "energy_landscapes"
    energy_landscape_spec: EnergyLandscapeSpec = field(
        default_factory=EnergyLandscapeSpec
    )


# =========================================================
# 3) Config / params helpers
# =========================================================
def is_diagonal_family(model_cls: type) -> bool:
    return model_cls in (DiagonalPlusEnergyNN, DiagonalPlusStiffnessNN)


def is_cholesky_family(model_cls: type) -> bool:
    return model_cls in (
        CholeskyPlusEnergyNN,
        CholeskyPlusStiffnessNN,
        CholeskyPlusStiffnessSignedNN,
    )


def get_der_K_for_model(cfg: SweepConfig, model_cls: type) -> jax.Array:
    if is_diagonal_family(model_cls):
        return jnp.asarray(cfg.der_K_diag)
    if is_cholesky_family(model_cls):
        return jnp.asarray(cfg.der_K_chol)
    raise ValueError(f"Unsupported model class: {model_cls}")


def make_model_params(cfg: SweepConfig, spec: ArchSpec) -> ModelParams:
    key = jax.random.PRNGKey(cfg.seed)
    der_K = get_der_K_for_model(cfg, spec.model_cls)

    return ModelParams(
        der_K=der_K,
        key=key,
        hidden=cfg.hidden,
        which_case=spec.which_case,
        corr_factor=cfg.corr_factor,
        input_mode=cfg.input_mode,
        zero_reference=cfg.zero_reference,
        activation=cfg.activation,
    )


def experiment_name(spec: ArchSpec, cfg: SweepConfig) -> str:
    hidden_str = "x".join(str(h) for h in cfg.hidden)
    zr = "zr1" if cfg.zero_reference else "zr0"
    return (
        f"{spec.name}"
        f"__hid_{hidden_str}"
        f"__inp_{cfg.input_mode}"
        f"__act_{cfg.activation}"
        f"__corr_{cfg.corr_factor:g}"
        f"__{zr}"
        f"__seed_{cfg.seed}"
        f"__mdl_{cfg.max_dlambda:g}"
        f"__it_{cfg.iters}"
    )


def make_experiment_dir(spec: ArchSpec, cfg: SweepConfig) -> str:
    exp_dir = os.path.join(cfg.output_dir, experiment_name(spec, cfg))
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def _classify_exception_message(msg: str) -> str:
    msg_low = msg.lower()
    if "did not converge" in msg_low or "nonconverge" in msg_low or "converge" in msg_low:
        return "convergence_failure"
    if "nan" in msg_low:
        return "nan_exception"
    if "inf" in msg_low:
        return "inf_exception"
    return f"exception: {msg}"


def _build_energy_snapshot_controls(cfg: SweepConfig, exp_dir: str):
    """
    Build the snapshot callback + schedule knobs for train_model.

    Default behavior:
      - initial landscape before training
      - final landscape after last epoch

    Optional extras:
      - any explicit epoch numbers in energy_snapshot_epochs
      - periodic snapshots via energy_snapshot_every
    """
    if not (cfg.save_plots and cfg.save_energy_landscapes):
        return None, None, None

    snapshot_dir = os.path.join(exp_dir, cfg.energy_snapshot_dirname)
    os.makedirs(snapshot_dir, exist_ok=True)

    snapshot_fn = make_energy_snapshot_fn(
        spec=cfg.energy_landscape_spec,
        save_dir=snapshot_dir,
        use_valid=cfg.energy_snapshot_use_valid,
        traj_idx=cfg.energy_landscape_spec.traj_idx,
        max_dlambda=cfg.max_dlambda,
        iters=cfg.iters,
        ls_steps=cfg.ls_steps,
        dpi=cfg.energy_snapshot_dpi,
        n_grid_snapshot=cfg.energy_snapshot_n_grid,
    )

    epoch_list = list(cfg.energy_snapshot_epochs)
    if cfg.energy_snapshot_final and cfg.n_epochs > 0:
        epoch_list.append(cfg.n_epochs - 1)
    snapshot_epochs = tuple(sorted(set(epoch_list))) if len(epoch_list) > 0 else None

    return snapshot_fn, snapshot_epochs, cfg.energy_snapshot_initial


# =========================================================
# 4) Plotting utilities
# =========================================================
def _to_numpy(x):
    return np.asarray(x)


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
    x_idx: int = 4,
    z_idx: int = 6,
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

    n_cases = pred.shape[0]
    colors = cm.viridis(np.linspace(0, 1, n_cases))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for i in range(n_cases):
        c = colors[i]

        # x component
        ax.plot(pred[i, :, x_idx], color=c, linestyle="-", linewidth=1.8)
        ax.plot(truth[i, :, x_idx], color=c, linestyle="--", linewidth=1.8)

        # z component
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
    ax.set_ylabel("Position (m)")
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
    x_idx: int = 4,
    z_idx: int = 6,
):
    """
    Cleaner 1x2 figure:
      - left: x trajectories
      - right: z trajectories
    """
    pred = _to_numpy(pred)
    truth = _to_numpy(truth)

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
        ax.set_ylabel("Position (m)")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{title} | {split_name}", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# 5) Saving helpers
# =========================================================
def save_config_json(cfg: SweepConfig, spec: ArchSpec, exp_dir: str):
    payload = asdict(cfg)
    payload["arch_name"] = spec.name
    payload["model_cls"] = spec.model_cls.__name__
    payload["which_case"] = spec.which_case

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(payload, f, indent=2)


def save_results_npz(
    exp_dir: str,
    spec: ArchSpec,
    cfg: SweepConfig,
    train_hist,
    valid_hist,
    train_pred,
    valid_pred,
    train_truth,
    valid_truth,
    train_lambdas,
    valid_lambdas,
):
    np.savez(
        os.path.join(exp_dir, "results.npz"),
        arch_name=spec.name,
        model_cls=spec.model_cls.__name__,
        which_case=spec.which_case,
        hidden=np.asarray(cfg.hidden, dtype=int),
        input_mode=cfg.input_mode,
        activation=cfg.activation,
        corr_factor=cfg.corr_factor,
        zero_reference=int(cfg.zero_reference),
        seed=cfg.seed,
        der_K_diag=np.asarray(cfg.der_K_diag, dtype=float),
        der_K_chol=np.asarray(cfg.der_K_chol, dtype=float),
        max_dlambda=float(cfg.max_dlambda),
        iters=int(cfg.iters),
        ls_steps=int(cfg.ls_steps),
        abs_tol=float(cfg.abs_tol),
        rel_tol=float(cfg.rel_tol),
        fail_on_nonconvergence=int(cfg.fail_on_nonconvergence),
        train_hist=np.asarray(train_hist, dtype=float),
        valid_hist=np.asarray(valid_hist, dtype=float),
        train_pred=np.asarray(train_pred, dtype=float),
        valid_pred=np.asarray(valid_pred, dtype=float),
        train_truth=np.asarray(train_truth, dtype=float),
        valid_truth=np.asarray(valid_truth, dtype=float),
        train_lambdas=np.asarray(train_lambdas, dtype=float),
        valid_lambdas=np.asarray(valid_lambdas, dtype=float),
    )


# =========================================================
# 6) Run one experiment
# =========================================================
def run_one_architecture(
    properties,
    train_file: str,
    valid_file: str,
    spec: ArchSpec,
    cfg: SweepConfig,
):
    exp_dir = make_experiment_dir(spec, cfg)

    if cfg.verbose:
        print("=" * 90)
        print(f"Running architecture: {spec.name}")
        print(f"  model_cls               : {spec.model_cls.__name__}")
        print(f"  which_case              : {spec.which_case}")
        print(f"  hidden                  : {cfg.hidden}")
        print(f"  input_mode              : {cfg.input_mode}")
        print(f"  activation              : {cfg.activation}")
        print(f"  corr_factor             : {cfg.corr_factor}")
        print(f"  zero_reference          : {cfg.zero_reference}")
        print(f"  seed                    : {cfg.seed}")
        print(f"  max_dlambda             : {cfg.max_dlambda}")
        print(f"  iters                   : {cfg.iters}")
        print(f"  ls_steps                : {cfg.ls_steps}")
        print(f"  abs_tol                 : {cfg.abs_tol}")
        print(f"  rel_tol                 : {cfg.rel_tol}")
        print(f"  fail_on_nonconvergence  : {cfg.fail_on_nonconvergence}")
        print(f"  exp_dir                 : {exp_dir}")
        if cfg.save_energy_landscapes:
            print(f"  energy snapshots        : initial={cfg.energy_snapshot_initial}, final={cfg.energy_snapshot_final}, epochs={cfg.energy_snapshot_epochs}, every={cfg.energy_snapshot_every}")
            print(f"  energy snapshot split   : {'valid' if cfg.energy_snapshot_use_valid else 'train'}")
        print("=" * 90)

    params = make_model_params(cfg, spec)
    snapshot_fn, snapshot_epochs, snapshot_before_training = _build_energy_snapshot_controls(
        cfg, exp_dir
    )

    try:
        # -------------------------
        # Train
        # -------------------------
        model, train_hist, valid_hist = train_model(
            properties=properties,
            model_cls=spec.model_cls,
            params=params,
            train_file=train_file,
            valid_file=valid_file,
            n_epochs=cfg.n_epochs,
            lr=cfg.lr,
            snapshot_fn=snapshot_fn,
            snapshot_every=cfg.energy_snapshot_every,
            snapshot_epochs=snapshot_epochs,
            snapshot_before_training=snapshot_before_training,
            valid_every=cfg.valid_every,
            max_dlambda=cfg.max_dlambda,
            iters=cfg.iters,
            ls_steps=cfg.ls_steps,
            abs_tol=cfg.abs_tol,
            rel_tol=cfg.rel_tol,
            fail_on_nonconvergence=cfg.fail_on_nonconvergence,
        )

        # -------------------------
        # Predict
        # -------------------------
        base, aux = get_slinky(properties)
        train_data = Dataset.load(train_file)
        valid_data = Dataset.load(valid_file)

        train_pred = predict(
            model, base, aux,
            train_data.idx_b, train_data.xb, train_data.lambdas,
            max_dlambda=cfg.max_dlambda,
            iters=cfg.iters,
            ls_steps=cfg.ls_steps,
            abs_tol=cfg.abs_tol,
            rel_tol=cfg.rel_tol,
            fail_on_nonconvergence=cfg.fail_on_nonconvergence,
        )
        valid_pred = predict(
            model, base, aux,
            valid_data.idx_b, valid_data.xb, valid_data.lambdas,
            max_dlambda=cfg.max_dlambda,
            iters=cfg.iters,
            ls_steps=cfg.ls_steps,
            abs_tol=cfg.abs_tol,
            rel_tol=cfg.rel_tol,
            fail_on_nonconvergence=cfg.fail_on_nonconvergence,
        )

        # -------------------------
        # Save config / arrays
        # -------------------------
        save_config_json(cfg, spec, exp_dir)

        if cfg.save_npz:
            save_results_npz(
                exp_dir=exp_dir,
                spec=spec,
                cfg=cfg,
                train_hist=train_hist,
                valid_hist=valid_hist,
                train_pred=train_pred,
                valid_pred=valid_pred,
                train_truth=train_data.qs,
                valid_truth=valid_data.qs,
                train_lambdas=train_data.lambdas,
                valid_lambdas=valid_data.lambdas,
            )

        # -------------------------
        # Plots
        # -------------------------
        if cfg.save_plots:
            title = experiment_name(spec, cfg)

            plot_loss_curves(
                train_hist=train_hist,
                valid_hist=valid_hist,
                title=title,
                save_path=os.path.join(exp_dir, "loss_curves.png"),
                show=False,
                logy=True,
            )

            plot_prediction_vs_truth(
                pred=train_pred,
                truth=train_data.qs,
                split_name="train",
                title=title,
                save_path=os.path.join(exp_dir, "pred_vs_truth_train_overlay.png"),
                show=False,
            )

            plot_prediction_vs_truth(
                pred=valid_pred,
                truth=valid_data.qs,
                split_name="valid",
                title=title,
                save_path=os.path.join(exp_dir, "pred_vs_truth_valid_overlay.png"),
                show=False,
            )

            plot_prediction_vs_truth_separate_components(
                pred=train_pred,
                truth=train_data.qs,
                split_name="train",
                title=title,
                save_path=os.path.join(exp_dir, "pred_vs_truth_train_xz.png"),
                show=False,
            )

            plot_prediction_vs_truth_separate_components(
                pred=valid_pred,
                truth=valid_data.qs,
                split_name="valid",
                title=title,
                save_path=os.path.join(exp_dir, "pred_vs_truth_valid_xz.png"),
                show=False,
            )

        result = {
            "spec": spec,
            "cfg": cfg,
            "params": params,
            "model": model,
            "train_hist": np.asarray(train_hist, dtype=float),
            "valid_hist": np.asarray(valid_hist, dtype=float),
            "train_pred": np.asarray(train_pred, dtype=float),
            "valid_pred": np.asarray(valid_pred, dtype=float),
            "train_truth": np.asarray(train_data.qs),
            "valid_truth": np.asarray(valid_data.qs),
            "train_lambdas": np.asarray(train_data.lambdas),
            "valid_lambdas": np.asarray(valid_data.lambdas),
            "exp_dir": exp_dir,
            "exp_name": experiment_name(spec, cfg),
            "success": True,
            "failure_reason": "",
        }
        return result

    except Exception as e:
        failure_reason = _classify_exception_message(repr(e))

        # still save minimal config + failure record
        save_config_json(cfg, spec, exp_dir)
        failure_payload = {
            "arch_name": spec.name,
            "model_cls": spec.model_cls.__name__,
            "which_case": spec.which_case,
            "success": False,
            "failure_reason": failure_reason,
        }
        with open(os.path.join(exp_dir, "failure.json"), "w") as f:
            json.dump(failure_payload, f, indent=2)

        if cfg.verbose:
            print(f"[FAILED] {spec.name}: {failure_reason}")

        result = {
            "spec": spec,
            "cfg": cfg,
            "params": params,
            "model": None,
            "train_hist": np.array([np.nan]),
            "valid_hist": np.array([np.nan]),
            "train_pred": None,
            "valid_pred": None,
            "train_truth": None,
            "valid_truth": None,
            "train_lambdas": None,
            "valid_lambdas": None,
            "exp_dir": exp_dir,
            "exp_name": experiment_name(spec, cfg),
            "success": False,
            "failure_reason": failure_reason,
        }
        return result


# =========================================================
# 7) Run sweep
# =========================================================
def run_architecture_sweep(
    properties,
    train_file: str,
    valid_file: str,
    cfg: SweepConfig,
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

    results = {}
    for arch_name in arch_names:
        spec = registry[arch_name]
        try:
            results[arch_name] = run_one_architecture(
                properties=properties,
                train_file=train_file,
                valid_file=valid_file,
                spec=spec,
                cfg=cfg,
            )
        except Exception as e:
            # extra guard, though run_one_architecture already catches failures
            failure_reason = _classify_exception_message(repr(e))
            if cfg.verbose:
                print(f"[SWEEP FAILED] {arch_name}: {failure_reason}")

            results[arch_name] = {
                "spec": spec,
                "cfg": cfg,
                "params": None,
                "model": None,
                "train_hist": np.array([np.nan]),
                "valid_hist": np.array([np.nan]),
                "train_pred": None,
                "valid_pred": None,
                "train_truth": None,
                "valid_truth": None,
                "train_lambdas": None,
                "valid_lambdas": None,
                "exp_dir": None,
                "exp_name": None,
                "success": False,
                "failure_reason": failure_reason,
            }
            if not cfg.continue_on_failure:
                raise

    return results


# =========================================================
# 8) Convenience subsets
# =========================================================
def subset_all() -> list[str]:
    return list(build_architecture_registry().keys())


def subset_energy_only() -> list[str]:
    return [
        "diag_energy_baseline",
        "diag_energy_mlp",
        "diag_energy_icnn",
        "chol_energy_baseline",
        "chol_energy_mlp",
        "chol_energy_icnn",
    ]


def subset_stiffness_only() -> list[str]:
    return [
        "diag_stiffness_mlp",
        "diag_stiffness_icnn",
        "chol_stiffness_mlp",
        "chol_stiffness_icnn",
        "chol_stiffness_signed_mlp",
        "chol_stiffness_signed_icnn",
    ]


def subset_main_paper_candidates() -> list[str]:
    return [
        "diag_energy_baseline",
        "chol_energy_baseline",
        "chol_energy_icnn",
        "chol_stiffness_mlp",
    ]


def subset_coupled_only() -> list[str]:
    return [
        "chol_energy_baseline",
        "chol_energy_mlp",
        "chol_energy_icnn",
        "chol_stiffness_mlp",
        "chol_stiffness_icnn",
        "chol_stiffness_signed_mlp",
        "chol_stiffness_signed_icnn",
    ]


# =========================================================
# 9) Optional grid runner
# =========================================================
def run_flag_grid(
    properties,
    train_file: str,
    valid_file: str,
    base_cfg: SweepConfig,
    selected_architectures: Optional[list[str]],
    hidden_list: list[tuple[int, ...]],
    input_modes: list[str],
    activations: list[str],
    seeds: list[int],
):
    all_results = {}

    for hidden in hidden_list:
        for input_mode in input_modes:
            for activation in activations:
                for seed in seeds:
                    cfg = replace(
                        base_cfg,
                        hidden=hidden,
                        input_mode=input_mode,
                        activation=activation,
                        seed=seed,
                    )
                    tag = (
                        f"hid={hidden}_inp={input_mode}"
                        f"_act={activation}_seed={seed}"
                    )
                    print(f"\n##### GRID RUN: {tag} #####\n")
                    all_results[tag] = run_architecture_sweep(
                        properties=properties,
                        train_file=train_file,
                        valid_file=valid_file,
                        cfg=cfg,
                        selected_architectures=selected_architectures,
                    )

    return all_results


# =========================================================
# 10) Optional summary plots across many models
# =========================================================
def plot_summary_final_losses(results: dict, save_path: Optional[str] = None, show: bool = False):
    names = list(results.keys())
    train_last = []
    valid_last = []

    for k in names:
        r = results[k]
        train_hist = np.asarray(r["train_hist"], dtype=float)
        valid_hist = np.asarray(r["valid_hist"], dtype=float)
        train_last.append(train_hist[-1])
        valid_last.append(valid_hist[-1])

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(names)), 5.5))
    ax.bar(x - width / 2, train_last, width=width, label="Train")
    ax.bar(x + width / 2, valid_last, width=width, label="Valid")

    ax.set_yscale("log")
    ax.set_ylabel("Final loss")
    ax.set_title("Final train/valid loss by architecture")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
