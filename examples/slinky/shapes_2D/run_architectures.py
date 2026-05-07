import json
import os
from dataclasses import dataclass, replace, asdict, field, fields
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

from util import Dataset, get_slinky, predict, predict_reaction_force, train_model
from architecture_plots import (
    plot_baseline_stiffness_history,
    plot_force_prediction_vs_truth,
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
    StructuredBrazierCholeskyEnergyNN,
    StructuredBrazierDiagonalEnergyNN,
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
        "brazier_diag_stiffness_baseline": ArchSpec(
            name="brazier_diag_stiffness_baseline",
            model_cls=StructuredBrazierDiagonalEnergyNN,
            which_case="baseline",
        ),
        "brazier_diag_stiffness_mlp": ArchSpec(
            name="brazier_diag_stiffness_mlp",
            model_cls=StructuredBrazierDiagonalEnergyNN,
            which_case="MLP",
        ),
        "brazier_diag_stiffness_icnn": ArchSpec(
            name="brazier_diag_stiffness_icnn",
            model_cls=StructuredBrazierDiagonalEnergyNN,
            which_case="ICNN",
        ),
        "brazier_chol_stiffness_baseline": ArchSpec(
            name="brazier_chol_stiffness_baseline",
            model_cls=StructuredBrazierCholeskyEnergyNN,
            which_case="baseline",
        ),
        "brazier_chol_stiffness_mlp": ArchSpec(
            name="brazier_chol_stiffness_mlp",
            model_cls=StructuredBrazierCholeskyEnergyNN,
            which_case="MLP",
        ),
        "brazier_chol_stiffness_icnn": ArchSpec(
            name="brazier_chol_stiffness_icnn",
            model_cls=StructuredBrazierCholeskyEnergyNN,
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
    mode: Optional[str] = None        # relevant for structured Brazier families
    n_epochs: int = 100
    lr: float = 1e-2
    weight_decay: float = 0.0
    seed: int = 0

    # solver / training-loop compatibility with the training utilities
    valid_every: int = 1
    max_dlambda: float = 1e-2
    iters: int = 5
    ls_steps: int = 10
    abs_tol: float = 1e-4
    rel_tol: float = 1e-4
    early_stop: bool = True
    # Keep strict convergence checks during optimizer steps, while validation
    # loss inside util.train_model remains non-strict and prediction is
    # controlled separately below.
    train_fail_on_nonconvergence: bool = True
    prediction_fail_on_nonconvergence: bool = False
    fail_on_nonconvergence: Optional[bool] = None  # legacy alias for training

    # Optional Hessian spectral regularizer.
    hessian_reg_strength: float = 0.0
    hessian_reg_probes: int = 1
    hessian_reg_seed: int = 0

    # Optional reaction-force loss in util.py.
    force_key: Optional[str] = None
    force_loss_strength: float = 0.0
    force_components: tuple[int, ...] = (0, 1, 2)
    force_sign: float = 1.0
    return_loss_components: bool = False

    # Optimizer-level early stopping to reduce overfitting.
    early_stopping: bool = True
    early_stopping_patience: Optional[int] = 25
    early_stopping_min_delta: float = 0.0
    early_stopping_warmup_epochs: int = 0
    restore_best_model: bool = True

    # Stored Hessian-path diagnostics for trained models. Prefer running these
    # with compute_architecture_hessian_diagnostics.py after the sweep finishes.
    save_hessian_diagnostics: bool = False
    hessian_diagnostics_use_predicted: bool = False
    hessian_diagnostics_splits: tuple[str, ...] = ("train", "valid")
    hessian_diagnostics_max_trajectories: Optional[int] = 1
    hessian_diagnostics_stride: int = 10

    # Store and plot predicted reaction-force trajectories when force loss is active.
    save_force_predictions: bool = True
    plot_force_predictions: bool = True

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
    return model_cls in (
        DiagonalPlusEnergyNN,
        DiagonalPlusStiffnessNN,
        StructuredBrazierDiagonalEnergyNN,
    )


def is_cholesky_family(model_cls: type) -> bool:
    return model_cls in (
        CholeskyPlusEnergyNN,
        CholeskyPlusStiffnessNN,
        CholeskyPlusStiffnessSignedNN,
        StructuredBrazierCholeskyEnergyNN,
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
        mode=cfg.mode,
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
    if cfg.weight_decay != 0.0:
        name += f"__wd_{cfg.weight_decay:g}"
    if cfg.force_loss_strength != 0.0:
        comps = "-".join(str(c) for c in cfg.force_components)
        name += (
            f"__floss_{cfg.force_loss_strength:g}"
            f"__fcomp_{comps}"
            f"__fsign_{cfg.force_sign:g}"
        )
    if cfg.mode is not None and spec.model_cls in (
        StructuredBrazierCholeskyEnergyNN,
        StructuredBrazierDiagonalEnergyNN,
    ):
        name += f"__mode_{cfg.mode}"
    return name


def make_experiment_dir(spec: ArchSpec, cfg: SweepConfig) -> str:
    exp_dir = os.path.join(cfg.output_dir, experiment_name(spec, cfg))
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def _classify_exception_message(msg: str) -> str:
    msg_low = msg.lower()
    if "did not converge" in msg_low or "nonconverge" in msg_low or "converge" in msg_low:
        return "convergence_failure"
    if "nonfinite_training_history" in msg_low:
        return "nonfinite_training_history"
    if "nan" in msg_low:
        return "nan_exception"
    if "inf" in msg_low:
        return "inf_exception"
    return f"exception: {msg}"


def _history_has_nonfinite(hist) -> bool:
    if hist is None:
        return False
    arr = np.asarray(hist, dtype=float)
    return bool(arr.size > 0 and not np.all(np.isfinite(arr)))


def _nonfinite_training_history_reason(**histories) -> Optional[str]:
    bad_names = [name for name, hist in histories.items() if _history_has_nonfinite(hist)]
    if len(bad_names) == 0:
        return None
    return "nonfinite_training_history: " + ",".join(bad_names)


def _tree_has_nonfinite_numeric(tree) -> bool:
    for leaf in jax.tree_util.tree_leaves(tree):
        try:
            arr = np.asarray(leaf)
        except Exception:
            continue
        if not np.issubdtype(arr.dtype, np.inexact):
            continue
        if arr.size > 0 and not np.all(np.isfinite(arr)):
            return True
    return False


def _train_fail_on_nonconvergence(cfg: SweepConfig) -> bool:
    if cfg.fail_on_nonconvergence is not None:
        return bool(cfg.fail_on_nonconvergence)
    return bool(cfg.train_fail_on_nonconvergence)


def _build_energy_snapshot_controls(cfg: SweepConfig, exp_dir: str):
    """
    Build the snapshot callback + schedule knobs for train_model.

    Default behavior:
      - initial landscape before training
      - final landscape for the model returned by training

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


def _dataset_has_force_for_loss(cfg: SweepConfig, data: Dataset) -> bool:
    return (
        cfg.force_loss_strength != 0.0
        and cfg.save_force_predictions
        and data.forces is not None
    )


def _predict_dataset_reaction_forces(model, base, aux, data: Dataset, cfg: SweepConfig):
    if data.forces is None:
        return None

    n_traj = data.qs.shape[0]
    if data.idx_b.ndim == 1:
        idx_all = jnp.broadcast_to(data.idx_b, (n_traj, data.idx_b.shape[0]))
    else:
        idx_all = data.idx_b

    if data.lambdas.ndim == 1:
        lambdas_all = jnp.broadcast_to(data.lambdas, (n_traj, data.lambdas.shape[0]))
    else:
        lambdas_all = data.lambdas

    preds = []
    for traj_idx in range(n_traj):
        pred_force = predict_reaction_force(
            model,
            base,
            aux,
            lambdas_all[traj_idx],
            data.qs[traj_idx],
            idx_all[traj_idx],
            data.valid[traj_idx],
            force_components=cfg.force_components,
            force_sign=cfg.force_sign,
        )
        force_true = data.forces[traj_idx]
        if force_true.ndim == 1:
            pred_force = pred_force[:, 0]
        else:
            n_force = min(force_true.shape[-1], pred_force.shape[-1])
            pred_force = pred_force[:, :n_force]
        preds.append(np.asarray(pred_force, dtype=float))

    return np.stack(preds, axis=0)


# =========================================================
# 4) Saving helpers
# =========================================================
def _jsonable_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable_value(v) for v in value]
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return arr.item()
        return arr.tolist()
    except Exception:
        return repr(value)


def _serialize_properties(properties) -> dict:
    payload = {"class_name": properties.__class__.__name__}
    for key, value in vars(properties).items():
        payload[key] = _jsonable_value(value)
    return payload


def save_config_json(
    cfg: SweepConfig,
    spec: ArchSpec,
    exp_dir: str,
    properties=None,
    train_file: Optional[str] = None,
    valid_file: Optional[str] = None,
):
    payload = asdict(cfg)
    payload["arch_name"] = spec.name
    payload["model_cls"] = spec.model_cls.__name__
    payload["which_case"] = spec.which_case
    payload["effective_train_fail_on_nonconvergence"] = _train_fail_on_nonconvergence(cfg)
    payload["validation_loss_fail_on_nonconvergence"] = False
    if properties is not None:
        payload["properties"] = _serialize_properties(properties)
    if train_file is not None:
        payload["train_file"] = os.path.abspath(train_file)
    if valid_file is not None:
        payload["valid_file"] = os.path.abspath(valid_file)

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
    train_displacement_hist=None,
    train_force_hist=None,
    valid_displacement_hist=None,
    valid_force_hist=None,
    train_force_pred=None,
    valid_force_pred=None,
    train_force_truth=None,
    valid_force_truth=None,
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
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        der_K_diag=np.asarray(cfg.der_K_diag, dtype=float),
        der_K_chol=np.asarray(cfg.der_K_chol, dtype=float),
        max_dlambda=float(cfg.max_dlambda),
        iters=int(cfg.iters),
        ls_steps=int(cfg.ls_steps),
        abs_tol=float(cfg.abs_tol),
        rel_tol=float(cfg.rel_tol),
        early_stop=int(cfg.early_stop),
        train_fail_on_nonconvergence=int(_train_fail_on_nonconvergence(cfg)),
        validation_loss_fail_on_nonconvergence=0,
        prediction_fail_on_nonconvergence=int(cfg.prediction_fail_on_nonconvergence),
        fail_on_nonconvergence=int(_train_fail_on_nonconvergence(cfg)),
        hessian_reg_strength=float(cfg.hessian_reg_strength),
        hessian_reg_probes=int(cfg.hessian_reg_probes),
        hessian_reg_seed=int(cfg.hessian_reg_seed),
        force_key="" if cfg.force_key is None else cfg.force_key,
        force_loss_strength=float(cfg.force_loss_strength),
        force_components=np.asarray(cfg.force_components, dtype=int),
        force_sign=float(cfg.force_sign),
        early_stopping=int(cfg.early_stopping),
        early_stopping_patience=-1 if cfg.early_stopping_patience is None else int(cfg.early_stopping_patience),
        early_stopping_min_delta=float(cfg.early_stopping_min_delta),
        restore_best_model=int(cfg.restore_best_model),
        save_force_predictions=int(cfg.save_force_predictions),
        plot_force_predictions=int(cfg.plot_force_predictions),
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
    if train_displacement_hist is not None:
        save_dict["train_displacement_hist"] = np.asarray(train_displacement_hist, dtype=float)
    if train_force_hist is not None:
        save_dict["train_force_hist"] = np.asarray(train_force_hist, dtype=float)
    if valid_displacement_hist is not None:
        save_dict["valid_displacement_hist"] = np.asarray(valid_displacement_hist, dtype=float)
    if valid_force_hist is not None:
        save_dict["valid_force_hist"] = np.asarray(valid_force_hist, dtype=float)
    if train_force_pred is not None and train_force_truth is not None:
        save_dict["train_force_pred"] = np.asarray(train_force_pred, dtype=float)
        save_dict["train_force_truth"] = np.asarray(train_force_truth, dtype=float)
    if valid_force_pred is not None and valid_force_truth is not None:
        save_dict["valid_force_pred"] = np.asarray(valid_force_pred, dtype=float)
        save_dict["valid_force_truth"] = np.asarray(valid_force_truth, dtype=float)
    np.savez(os.path.join(exp_dir, "results.npz"), **save_dict)


def save_model_artifact(exp_dir: str, model):
    eqx.tree_serialise_leaves(os.path.join(exp_dir, "model.eqx"), model)


# =========================================================
# 5) Post-training energy landscape generation
# =========================================================
def _sweep_config_from_payload(payload: dict) -> SweepConfig:
    field_names = {f.name for f in fields(SweepConfig)}
    kwargs = {key: value for key, value in payload.items() if key in field_names}

    tuple_keys = {
        "der_K_diag",
        "der_K_chol",
        "hidden",
        "force_components",
        "hessian_diagnostics_splits",
        "seed_list",
        "energy_snapshot_epochs",
    }
    for key in tuple_keys:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = tuple(kwargs[key])

    spec_payload = kwargs.get("energy_landscape_spec")
    if isinstance(spec_payload, dict):
        spec_fields = {f.name for f in fields(EnergyLandscapeSpec)}
        spec_kwargs = {
            key: value for key, value in spec_payload.items() if key in spec_fields
        }
        kwargs["energy_landscape_spec"] = EnergyLandscapeSpec(**spec_kwargs)

    return SweepConfig(**kwargs)


def _properties_from_payload(payload: dict):
    import properties as properties_module

    prop_payload = payload.get("properties")
    if prop_payload is None:
        raise ValueError(
            "No properties metadata found in config.json. Re-run the sweep with a "
            "newer run_architectures.py or add properties metadata to the config."
        )

    class_name = prop_payload["class_name"]
    prop_cls = getattr(properties_module, class_name)
    valid_fields = {f.name for f in fields(prop_cls)}
    kwargs = {
        key: value
        for key, value in prop_payload.items()
        if key != "class_name" and key in valid_fields
    }
    return prop_cls(**kwargs)


def _resolve_saved_data_path(payload: dict, key: str, exp_dir: str) -> str:
    path = payload.get(key)
    if path is None:
        raise ValueError(f"No {key} stored in {os.path.join(exp_dir, 'config.json')}.")
    if os.path.isabs(path):
        return path

    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(exp_dir, path)),
        os.path.abspath(os.path.join(os.path.dirname(exp_dir), path)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def _find_saved_experiment_dirs(output_dir: str) -> list[str]:
    output_dir = os.path.abspath(output_dir)
    if os.path.isfile(os.path.join(output_dir, "config.json")):
        return [output_dir]

    exp_dirs = []
    for root, dirs, files in os.walk(output_dir):
        if "config.json" in files:
            exp_dirs.append(root)
            dirs[:] = []
    return sorted(exp_dirs)


def generate_energy_landscapes_for_experiment(
    exp_dir: str,
    *,
    include_initial: bool = False,
    include_final: bool = True,
    use_valid: Optional[bool] = None,
    traj_idx: Optional[int] = None,
    output_dir: Optional[str] = None,
    dpi: Optional[int] = None,
    n_grid: Optional[int] = None,
) -> dict[str, str]:
    """Recreate energy landscape plots from saved config/model/data artifacts."""
    config_path = os.path.join(exp_dir, "config.json")
    model_path = os.path.join(exp_dir, "model.eqx")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)
    if include_final and not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    with open(config_path) as f:
        payload = json.load(f)

    cfg = _sweep_config_from_payload(payload)
    registry = build_architecture_registry()
    arch_name = payload.get("arch_name")
    if arch_name not in registry:
        raise ValueError(f"Unknown or missing arch_name in {config_path}: {arch_name}")
    spec = registry[arch_name]

    properties = _properties_from_payload(payload)
    train_file = _resolve_saved_data_path(payload, "train_file", exp_dir)
    valid_file = _resolve_saved_data_path(payload, "valid_file", exp_dir)

    params = make_model_params(cfg, spec)
    base, aux = get_slinky(properties)
    train = Dataset.load(train_file, force_key=False)
    valid = Dataset.load(valid_file, force_key=False)

    spec_local = replace(cfg.energy_landscape_spec)
    if traj_idx is not None:
        spec_local.traj_idx = int(traj_idx)

    save_dir = output_dir or os.path.join(exp_dir, cfg.energy_snapshot_dirname)
    snapshot_fn = make_energy_snapshot_fn(
        spec=spec_local,
        save_dir=save_dir,
        use_valid=cfg.energy_snapshot_use_valid if use_valid is None else bool(use_valid),
        traj_idx=spec_local.traj_idx,
        max_dlambda=cfg.max_dlambda,
        iters=cfg.iters,
        ls_steps=cfg.ls_steps,
        dpi=cfg.energy_snapshot_dpi if dpi is None else int(dpi),
        n_grid_snapshot=cfg.energy_snapshot_n_grid if n_grid is None else int(n_grid),
    )

    written = {}
    if include_initial:
        initial_model = spec.model_cls(params)
        snapshot_fn(
            model=initial_model,
            epoch=-1,
            base=base,
            aux=aux,
            train=train,
            valid=valid,
        )
        written["initial"] = os.path.join(save_dir, "energy_landscape_initial.png")

    if include_final:
        model = eqx.tree_deserialise_leaves(model_path, spec.model_cls(params))
        snapshot_fn(
            model=model,
            epoch="final",
            base=base,
            aux=aux,
            train=train,
            valid=valid,
        )
        written["final"] = os.path.join(save_dir, "energy_landscape_final.png")

    return written


def generate_energy_landscapes_for_sweep(
    output_dir: str,
    *,
    architectures: Optional[list[str]] = None,
    include_initial: bool = False,
    include_final: bool = True,
    use_valid: Optional[bool] = None,
    traj_idx: Optional[int] = None,
    dpi: Optional[int] = None,
    n_grid: Optional[int] = None,
    continue_on_failure: bool = True,
) -> dict[str, dict[str, str]]:
    """Generate post-training energy landscapes for all saved architecture runs."""
    requested = None if architectures is None else set(architectures)
    outputs = {}
    for exp_dir in _find_saved_experiment_dirs(output_dir):
        config_path = os.path.join(exp_dir, "config.json")
        try:
            with open(config_path) as f:
                payload = json.load(f)
            arch_name = payload.get("arch_name", os.path.basename(exp_dir))
            if requested is not None and arch_name not in requested:
                continue

            outputs[arch_name] = generate_energy_landscapes_for_experiment(
                exp_dir,
                include_initial=include_initial,
                include_final=include_final,
                use_valid=use_valid,
                traj_idx=traj_idx,
                dpi=dpi,
                n_grid=n_grid,
            )
            print(f"[ok] {arch_name}: {outputs[arch_name]}")
        except Exception as exc:
            if not continue_on_failure:
                raise
            arch_name = os.path.basename(exp_dir)
            outputs[arch_name] = {"error": str(exc)}
            print(f"[skip] {exp_dir}: {exc}")

    return outputs


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
        print(f"  n_epochs                : {cfg.n_epochs}")
        print(f"  lr                      : {cfg.lr}")
        print(f"  weight_decay            : {cfg.weight_decay}")
        print(f"  max_dlambda             : {cfg.max_dlambda}")
        print(f"  iters                   : {cfg.iters}")
        print(f"  ls_steps                : {cfg.ls_steps}")
        print(f"  abs_tol                 : {cfg.abs_tol}")
        print(f"  rel_tol                 : {cfg.rel_tol}")
        print(f"  early_stop              : {cfg.early_stop}")
        print(f"  training fail_on_nonconvergence       : {_train_fail_on_nonconvergence(cfg)}")
        print("  validation loss fail_on_nonconvergence: False")
        print(f"  prediction fail_on_nonconvergence     : {cfg.prediction_fail_on_nonconvergence}")
        print(f"  early_stopping          : {cfg.early_stopping}")
        print(f"  early_stopping_patience : {cfg.early_stopping_patience}")
        print(f"  early_stopping_warmup   : {cfg.early_stopping_warmup_epochs}")
        print(f"  restore_best_model      : {cfg.restore_best_model}")
        print(f"  hessian_reg_strength    : {cfg.hessian_reg_strength}")
        print(f"  hessian_reg_probes      : {cfg.hessian_reg_probes}")
        print(f"  hessian_reg_seed        : {cfg.hessian_reg_seed}")
        print(f"  force_key               : {cfg.force_key}")
        print(f"  force_loss_strength     : {cfg.force_loss_strength}")
        print(f"  force_components        : {cfg.force_components}")
        print(f"  force_sign              : {cfg.force_sign}")
        print(f"  save_force_predictions  : {cfg.save_force_predictions}")
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
        train_result = train_model(
            properties=properties,
            model_cls=spec.model_cls,
            params=params,
            train_file=train_file,
            valid_file=valid_file,
            force_key=cfg.force_key,
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
            early_stop=cfg.early_stop,
            hessian_reg_strength=cfg.hessian_reg_strength,
            hessian_reg_probes=cfg.hessian_reg_probes,
            hessian_reg_seed=cfg.hessian_reg_seed,
            force_loss_strength=cfg.force_loss_strength,
            force_components=cfg.force_components,
            force_sign=cfg.force_sign,
            weight_decay=cfg.weight_decay,
            early_stopping_patience=(
                cfg.early_stopping_patience if cfg.early_stopping else None
            ),
            early_stopping_min_delta=cfg.early_stopping_min_delta,
            early_stopping_warmup_epochs=cfg.early_stopping_warmup_epochs,
            restore_best_model=cfg.restore_best_model,
            return_loss_components=cfg.return_loss_components,
        )
        if cfg.return_loss_components:
            (
                model,
                train_hist,
                valid_hist,
                train_displacement_hist,
                train_force_hist,
                valid_displacement_hist,
                valid_force_hist,
            ) = train_result
        else:
            model, train_hist, valid_hist = train_result
            train_displacement_hist = None
            train_force_hist = None
            valid_displacement_hist = None
            valid_force_hist = None

        nonfinite_reason = _nonfinite_training_history_reason(
            train_hist=train_hist,
            valid_hist=valid_hist,
            train_displacement_hist=train_displacement_hist,
            train_force_hist=train_force_hist,
            valid_displacement_hist=valid_displacement_hist,
            valid_force_hist=valid_force_hist,
        )
        if nonfinite_reason is not None:
            raise FloatingPointError(nonfinite_reason)
        if _tree_has_nonfinite_numeric(model):
            raise FloatingPointError("nonfinite_training_history: model_parameters")

        # -------------------------
        # Predict
        # -------------------------
        base, aux = get_slinky(properties)
        train_data = Dataset.load(train_file, force_key=cfg.force_key)
        valid_data = Dataset.load(valid_file, force_key=cfg.force_key)

        train_pred = predict(
            model, base, aux,
            train_data.idx_b, train_data.xb, train_data.lambdas,
            max_dlambda=cfg.max_dlambda,
            iters=cfg.iters,
            ls_steps=cfg.ls_steps,
            abs_tol=cfg.abs_tol,
            rel_tol=cfg.rel_tol,
            fail_on_nonconvergence=cfg.prediction_fail_on_nonconvergence,
            early_stop=cfg.early_stop,
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
            early_stop=cfg.early_stop,
        )

        train_force_pred = None
        valid_force_pred = None
        train_force_truth = None
        valid_force_truth = None
        if _dataset_has_force_for_loss(cfg, train_data):
            train_force_pred = _predict_dataset_reaction_forces(model, base, aux, train_data, cfg)
            train_force_truth = np.asarray(train_data.forces, dtype=float)
        if _dataset_has_force_for_loss(cfg, valid_data):
            valid_force_pred = _predict_dataset_reaction_forces(model, base, aux, valid_data, cfg)
            valid_force_truth = np.asarray(valid_data.forces, dtype=float)

        # -------------------------
        # Save config / arrays
        # -------------------------
        save_config_json(
            cfg,
            spec,
            exp_dir,
            properties=properties,
            train_file=train_file,
            valid_file=valid_file,
        )

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
                train_displacement_hist=train_displacement_hist,
                train_force_hist=train_force_hist,
                valid_displacement_hist=valid_displacement_hist,
                valid_force_hist=valid_force_hist,
                train_force_pred=train_force_pred,
                valid_force_pred=valid_force_pred,
                train_force_truth=train_force_truth,
                valid_force_truth=valid_force_truth,
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

            if cfg.plot_force_predictions and train_force_pred is not None and train_force_truth is not None:
                plot_force_prediction_vs_truth(
                    pred_force=train_force_pred,
                    true_force=train_force_truth,
                    lambdas=train_data.lambdas,
                    valid=train_data.valid,
                    split_name="train",
                    title=title,
                    save_path=os.path.join(exp_dir, "force_pred_vs_truth_train.png"),
                    show=False,
                )

            if cfg.plot_force_predictions and valid_force_pred is not None and valid_force_truth is not None:
                plot_force_prediction_vs_truth(
                    pred_force=valid_force_pred,
                    true_force=valid_force_truth,
                    lambdas=valid_data.lambdas,
                    valid=valid_data.valid,
                    split_name="valid",
                    title=title,
                    save_path=os.path.join(exp_dir, "force_pred_vs_truth_valid.png"),
                    show=False,
                )

        result = {
            "spec": spec,
            "cfg": cfg,
            "params": params,
            "model": model,
            "train_hist": np.asarray(train_hist, dtype=float),
            "valid_hist": np.asarray(valid_hist, dtype=float),
            "train_displacement_hist": None if train_displacement_hist is None else np.asarray(train_displacement_hist, dtype=float),
            "train_force_hist": None if train_force_hist is None else np.asarray(train_force_hist, dtype=float),
            "valid_displacement_hist": None if valid_displacement_hist is None else np.asarray(valid_displacement_hist, dtype=float),
            "valid_force_hist": None if valid_force_hist is None else np.asarray(valid_force_hist, dtype=float),
            "train_force_pred": None if train_force_pred is None else np.asarray(train_force_pred, dtype=float),
            "valid_force_pred": None if valid_force_pred is None else np.asarray(valid_force_pred, dtype=float),
            "train_force_truth": None if train_force_truth is None else np.asarray(train_force_truth, dtype=float),
            "valid_force_truth": None if valid_force_truth is None else np.asarray(valid_force_truth, dtype=float),
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
        save_config_json(
            cfg,
            spec,
            exp_dir,
            properties=properties,
            train_file=train_file,
            valid_file=valid_file,
        )
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
            "train_displacement_hist": None,
            "train_force_hist": None,
            "valid_displacement_hist": None,
            "valid_force_hist": None,
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
                "train_displacement_hist": None,
                "train_force_hist": None,
                "valid_displacement_hist": None,
                "valid_force_hist": None,
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
    return subset_energy_only() + subset_stiffness_only()


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


def subset_brazier_stiffness_only() -> list[str]:
    return [
        "brazier_diag_stiffness_baseline",
        "brazier_diag_stiffness_mlp",
        "brazier_diag_stiffness_icnn",
        "brazier_chol_stiffness_baseline",
        "brazier_chol_stiffness_mlp",
        "brazier_chol_stiffness_icnn",
    ]


def subset_tape_tube_candidates() -> list[str]:
    return subset_energy_only() + subset_brazier_stiffness_only()


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
