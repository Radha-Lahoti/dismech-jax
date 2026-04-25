import os
import json
from dataclasses import dataclass, replace, asdict, field
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

from util import Dataset, get_slinky, predict, train_model
from architecture_plots import (
    plot_baseline_stiffness_history,
    plot_loss_curves,
    plot_prediction_vs_truth,
    plot_prediction_vs_truth_separate_components,
)
from util_energy_plots import EnergyLandscapeSpec, make_energy_snapshot_fn
from Energy_NN_architectures import (
    ModelParams,
    ScalarEnergyNN,
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
        "mlp_energy": ArchSpec(
            name="mlp_energy",
            model_cls=ScalarEnergyNN,
            which_case="MLP",
        ),
        "icnn_energy": ArchSpec(
            name="icnn_energy",
            model_cls=ScalarEnergyNN,
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
    only_stretching_NN: bool = False
    only_bending_NN: bool = False
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
    abs_tol: float = 1e-4
    rel_tol: float = 1e-4
    # Keep strict convergence checks during optimizer steps, while validation
    # loss inside util.train_model remains non-strict and prediction is
    # controlled separately below.
    train_fail_on_nonconvergence: bool = True
    prediction_fail_on_nonconvergence: bool = False
    fail_on_nonconvergence: Optional[bool] = None  # legacy alias for training

    # Optional Hessian spectral regularizer in util.py.
    hessian_reg_strength: float = 0.0
    hessian_reg_probes: int = 1
    hessian_reg_seed: int = 0

    output_dir: str = "arch_sweep_outputs"
    save_npz: bool = True
    save_model: bool = True
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


def is_scalar_energy_family(model_cls: type) -> bool:
    return model_cls is ScalarEnergyNN


def get_der_K_for_model(cfg: SweepConfig, model_cls: type) -> jax.Array:
    if is_scalar_energy_family(model_cls):
        # ScalarEnergyNN does not use baseline stiffness entries, but ModelParams
        # still carries der_K, so provide a harmless placeholder.
        return jnp.asarray(cfg.der_K_diag)
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
        only_stretching_NN=cfg.only_stretching_NN,
        only_bending_NN=cfg.only_bending_NN,
        zero_reference=cfg.zero_reference,
        activation=cfg.activation,
    )


def experiment_name(spec: ArchSpec, cfg: SweepConfig) -> str:
    hidden_str = "x".join(str(h) for h in cfg.hidden)
    zr = "zr1" if cfg.zero_reference else "zr0"
    name = (
        f"{spec.name}"
        f"__hid_{hidden_str}"
        f"__inp_{cfg.input_mode}"
        f"__stretchNN_{int(cfg.only_stretching_NN)}"
        f"__bendNN_{int(cfg.only_bending_NN)}"
        f"__act_{cfg.activation}"
        f"__corr_{cfg.corr_factor:g}"
        f"__{zr}"
        f"__seed_{cfg.seed}"
        f"__mdl_{cfg.max_dlambda:g}"
        f"__it_{cfg.iters}"
    )
    if cfg.hessian_reg_strength != 0.0:
        name += (
            f"__hreg_{cfg.hessian_reg_strength:g}"
            f"__hprobe_{cfg.hessian_reg_probes}"
            f"__hseed_{cfg.hessian_reg_seed}"
        )
    return name


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


def _train_fail_on_nonconvergence(cfg: SweepConfig) -> bool:
    if cfg.fail_on_nonconvergence is not None:
        return bool(cfg.fail_on_nonconvergence)
    return bool(cfg.train_fail_on_nonconvergence)


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


def _supports_baseline_stiffness_history(model_or_cls) -> bool:
    supported_types = (
        DiagonalPlusEnergyNN,
        CholeskyPlusEnergyNN,
        DiagonalPlusStiffnessNN,
        CholeskyPlusStiffnessNN,
    )
    if isinstance(model_or_cls, type):
        return issubclass(model_or_cls, supported_types)
    return isinstance(model_or_cls, supported_types)


def _baseline_stiffness_labels(model_or_cls) -> tuple[str, ...]:
    if model_or_cls in (DiagonalPlusEnergyNN, DiagonalPlusStiffnessNN) or isinstance(
        model_or_cls, (DiagonalPlusEnergyNN, DiagonalPlusStiffnessNN)
    ):
        return ("k_s", "k_b")
    if model_or_cls in (CholeskyPlusEnergyNN, CholeskyPlusStiffnessNN) or isinstance(
        model_or_cls, (CholeskyPlusEnergyNN, CholeskyPlusStiffnessNN)
    ):
        return ("k_ss", "k_sb", "k_bb")
    raise ValueError(f"Unsupported model type for baseline stiffness labels: {model_or_cls}")


def _extract_baseline_stiffness_entries(model) -> Optional[np.ndarray]:
    if not _supports_baseline_stiffness_history(model):
        return None
    return np.asarray(model.get_baseline_K_entries(), dtype=float)


def _make_baseline_history_callback(model_cls):
    history_epochs = []
    history_values = []

    def callback(model, epoch, train_loss=None, val_loss=None):
        _ = train_loss, val_loss
        values = _extract_baseline_stiffness_entries(model)
        if values is None:
            return
        history_epochs.append(int(epoch))
        history_values.append(values)

    labels = _baseline_stiffness_labels(model_cls) if _supports_baseline_stiffness_history(model_cls) else ()

    return callback, history_epochs, history_values, labels


# =========================================================
# 4) Saving helpers
# =========================================================
def save_config_json(cfg: SweepConfig, spec: ArchSpec, exp_dir: str):
    payload = asdict(cfg)
    payload["arch_name"] = spec.name
    payload["model_cls"] = spec.model_cls.__name__
    payload["which_case"] = spec.which_case
    payload["effective_train_fail_on_nonconvergence"] = _train_fail_on_nonconvergence(cfg)
    payload["validation_loss_fail_on_nonconvergence"] = False

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
    train_valid_mask,
    valid_valid_mask,
    baseline_history_epochs=None,
    baseline_history_values=None,
    baseline_history_labels=None,
):
    save_dict = dict(
        arch_name=spec.name,
        model_cls=spec.model_cls.__name__,
        which_case=spec.which_case,
        hidden=np.asarray(cfg.hidden, dtype=int),
        input_mode=cfg.input_mode,
        activation=cfg.activation,
        corr_factor=cfg.corr_factor,
        only_stretching_NN=int(cfg.only_stretching_NN),
        only_bending_NN=int(cfg.only_bending_NN),
        zero_reference=int(cfg.zero_reference),
        seed=cfg.seed,
        der_K_diag=np.asarray(cfg.der_K_diag, dtype=float),
        der_K_chol=np.asarray(cfg.der_K_chol, dtype=float),
        max_dlambda=float(cfg.max_dlambda),
        iters=int(cfg.iters),
        ls_steps=int(cfg.ls_steps),
        abs_tol=float(cfg.abs_tol),
        rel_tol=float(cfg.rel_tol),
        train_fail_on_nonconvergence=int(_train_fail_on_nonconvergence(cfg)),
        validation_loss_fail_on_nonconvergence=0,
        prediction_fail_on_nonconvergence=int(cfg.prediction_fail_on_nonconvergence),
        fail_on_nonconvergence=int(_train_fail_on_nonconvergence(cfg)),
        hessian_reg_strength=float(cfg.hessian_reg_strength),
        hessian_reg_probes=int(cfg.hessian_reg_probes),
        hessian_reg_seed=int(cfg.hessian_reg_seed),
        train_hist=np.asarray(train_hist, dtype=float),
        valid_hist=np.asarray(valid_hist, dtype=float),
        train_pred=np.asarray(train_pred, dtype=float),
        valid_pred=np.asarray(valid_pred, dtype=float),
        train_truth=np.asarray(train_truth, dtype=float),
        valid_truth=np.asarray(valid_truth, dtype=float),
        train_lambdas=np.asarray(train_lambdas, dtype=float),
        valid_lambdas=np.asarray(valid_lambdas, dtype=float),
        train_valid_mask=np.asarray(train_valid_mask, dtype=bool),
        valid_valid_mask=np.asarray(valid_valid_mask, dtype=bool),
    )
    if baseline_history_epochs is not None and baseline_history_values is not None:
        save_dict["baseline_history_epochs"] = np.asarray(baseline_history_epochs, dtype=int)
        save_dict["baseline_history_values"] = np.asarray(baseline_history_values, dtype=float)
    if baseline_history_labels is not None and len(baseline_history_labels) > 0:
        save_dict["baseline_history_labels"] = np.asarray(baseline_history_labels, dtype=str)

    np.savez(os.path.join(exp_dir, "results.npz"), **save_dict)


def save_model_artifact(exp_dir: str, model):
    eqx.tree_serialise_leaves(os.path.join(exp_dir, "model.eqx"), model)


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
        print(f"  only_stretching_NN      : {cfg.only_stretching_NN}")
        print(f"  only_bending_NN         : {cfg.only_bending_NN}")
        print(f"  activation              : {cfg.activation}")
        print(f"  corr_factor             : {cfg.corr_factor}")
        print(f"  zero_reference          : {cfg.zero_reference}")
        print(f"  seed                    : {cfg.seed}")
        print(f"  max_dlambda             : {cfg.max_dlambda}")
        print(f"  iters                   : {cfg.iters}")
        print(f"  ls_steps                : {cfg.ls_steps}")
        print(f"  abs_tol                 : {cfg.abs_tol}")
        print(f"  rel_tol                 : {cfg.rel_tol}")
        print(f"  training fail_on_nonconvergence       : {_train_fail_on_nonconvergence(cfg)}")
        print("  validation loss fail_on_nonconvergence: False")
        print(f"  prediction fail_on_nonconvergence     : {cfg.prediction_fail_on_nonconvergence}")
        print(f"  hessian_reg_strength    : {cfg.hessian_reg_strength}")
        print(f"  hessian_reg_probes      : {cfg.hessian_reg_probes}")
        print(f"  hessian_reg_seed        : {cfg.hessian_reg_seed}")
        print(f"  exp_dir                 : {exp_dir}")
        if cfg.save_energy_landscapes:
            print(f"  energy snapshots        : initial={cfg.energy_snapshot_initial}, final={cfg.energy_snapshot_final}, epochs={cfg.energy_snapshot_epochs}, every={cfg.energy_snapshot_every}")
            print(f"  energy snapshot split   : {'valid' if cfg.energy_snapshot_use_valid else 'train'}")
        print("=" * 90)

    params = make_model_params(cfg, spec)
    snapshot_fn, snapshot_epochs, snapshot_before_training = _build_energy_snapshot_controls(
        cfg, exp_dir
    )
    baseline_callback = None
    baseline_history_epochs = None
    baseline_history_values = None
    baseline_history_labels = ()
    if spec.model_cls in (
        DiagonalPlusEnergyNN,
        CholeskyPlusEnergyNN,
        DiagonalPlusStiffnessNN,
        CholeskyPlusStiffnessNN,
    ):
        (
            baseline_callback,
            baseline_history_epochs,
            baseline_history_values,
            baseline_history_labels,
        ) = _make_baseline_history_callback(spec.model_cls)

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
            epoch_callback=baseline_callback,
            valid_every=cfg.valid_every,
            max_dlambda=cfg.max_dlambda,
            iters=cfg.iters,
            ls_steps=cfg.ls_steps,
            abs_tol=cfg.abs_tol,
            rel_tol=cfg.rel_tol,
            fail_on_nonconvergence=_train_fail_on_nonconvergence(cfg),
            hessian_reg_strength=cfg.hessian_reg_strength,
            hessian_reg_probes=cfg.hessian_reg_probes,
            hessian_reg_seed=cfg.hessian_reg_seed,
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
            fail_on_nonconvergence=cfg.prediction_fail_on_nonconvergence,
        )
        valid_pred = predict(
            model, base, aux,
            valid_data.idx_b, valid_data.xb, valid_data.lambdas,
            max_dlambda=cfg.max_dlambda,
            iters=cfg.iters,
            ls_steps=cfg.ls_steps,
            abs_tol=cfg.abs_tol,
            rel_tol=cfg.rel_tol,
            fail_on_nonconvergence=cfg.prediction_fail_on_nonconvergence,
        )

        # -------------------------
        # Save config / arrays
        # -------------------------
        save_config_json(cfg, spec, exp_dir)

        if cfg.save_model:
            save_model_artifact(exp_dir, model)

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
                train_valid_mask=train_data.valid,
                valid_valid_mask=valid_data.valid,
                baseline_history_epochs=baseline_history_epochs,
                baseline_history_values=baseline_history_values,
                baseline_history_labels=baseline_history_labels,
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

            if baseline_history_values is not None and len(baseline_history_values) > 0:
                plot_baseline_stiffness_history(
                    epochs=baseline_history_epochs,
                    values=baseline_history_values,
                    labels=baseline_history_labels,
                    title=f"{title} | baseline stiffness",
                    save_path=os.path.join(exp_dir, "baseline_stiffness_history.png"),
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
            "train_valid_mask": np.asarray(train_data.valid),
            "valid_valid_mask": np.asarray(valid_data.valid),
            "baseline_history_epochs": None if baseline_history_epochs is None else np.asarray(baseline_history_epochs, dtype=int),
            "baseline_history_values": None if baseline_history_values is None else np.asarray(baseline_history_values, dtype=float),
            "baseline_history_labels": baseline_history_labels,
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
            "train_valid_mask": None,
            "valid_valid_mask": None,
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
        "mlp_energy",
        "icnn_energy",
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
