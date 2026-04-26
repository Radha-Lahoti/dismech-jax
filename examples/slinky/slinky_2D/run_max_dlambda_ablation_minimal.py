import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from architecture_plots import (
    plot_force_prediction_vs_truth,
    plot_loss_curves,
    plot_prediction_vs_truth,
    plot_prediction_vs_truth_separate_components,
)
from util import Dataset, get_slinky, predict, predict_reaction_force, train_model
from run_architectures import (
    ArchSpec,
    build_architecture_registry,
    experiment_name as _architecture_experiment_name,
    make_model_params as _make_architecture_params,
    subset_all,
    subset_energy_only,
    subset_main_paper_candidates,
)


@dataclass(frozen=True)
class MaxDlambdaAblationConfig:
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

    max_dlambda_values: tuple[float, ...] = (1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0)
    iters: int = 20
    ls_steps: int = 10
    abs_tol: float = 1e-8
    rel_tol: float = 1e-6
    early_stop: bool = True

    # This is passed only into train_model(...). Validation loss is already
    # computed with fail_on_nonconvergence=False.
    train_fail_on_nonconvergence: bool = True

    # This is passed only into final predict(...), after training completes.
    prediction_fail_on_nonconvergence: bool = False

    # Optional Hessian spectral regularizer.
    hessian_reg_strength: float = 0.0
    hessian_reg_probes: int = 1
    hessian_reg_seed: int = 0

    # Optional reaction-force loss.
    force_key: Optional[str] = None
    force_loss_strength: float = 0.0
    force_components: tuple[int, ...] = (0, 1, 2)
    force_sign: float = 1.0
    return_loss_components: bool = False

    # Optimizer-level early stopping.
    early_stopping: bool = True
    early_stopping_patience: Optional[int] = 25
    early_stopping_min_delta: float = 0.0
    restore_best_model: bool = True

    # Hessian diagnostics are intentionally post-processed by
    # compute_architecture_hessian_diagnostics.py, not run during this sweep.
    save_hessian_diagnostics: bool = False
    hessian_diagnostics_use_predicted: bool = False
    hessian_diagnostics_splits: tuple[str, ...] = ("train", "valid")
    hessian_diagnostics_max_trajectories: Optional[int] = 1
    hessian_diagnostics_stride: int = 10

    # Store and plot predicted reaction-force trajectories when force loss is active.
    save_force_predictions: bool = True
    plot_force_predictions: bool = True

    output_dir: str = "max_dlambda_ablation_minimal_outputs"
    save_npz: bool = True
    save_model: bool = True
    save_plots: bool = True
    save_summary_json: bool = True
    strict_finite_check: bool = True
    stop_after_first_failure: bool = False
    verbose: bool = True


def _make_params(
    cfg: MaxDlambdaAblationConfig,
    spec: ArchSpec,
    seed: int,
    max_dlambda: float,
):
    return _make_architecture_params(
        _cfg_for_architecture_helpers(cfg, seed, max_dlambda),
        spec,
    )


def _cfg_for_architecture_helpers(
    cfg: MaxDlambdaAblationConfig,
    seed: int,
    max_dlambda: float,
):
    # run_architectures helper functions are intentionally duck-typed here:
    # they need the shared architecture/model fields, not a SweepConfig instance.
    return type(
        "MaxDlambdaRunConfigView",
        (),
        {
            **cfg.__dict__,
            "seed": int(seed),
            "max_dlambda": float(max_dlambda),
            "fail_on_nonconvergence": None,
        },
    )()


def _experiment_name(spec: ArchSpec, cfg: MaxDlambdaAblationConfig, seed: int, max_dlambda: float) -> str:
    return _architecture_experiment_name(
        spec,
        _cfg_for_architecture_helpers(cfg, seed, max_dlambda),
    )


def _classify_exception(exc: Exception) -> str:
    msg = repr(exc).lower()
    if "did not converge" in msg or "nonconverge" in msg or "converge" in msg:
        return "convergence_failure"
    if "nonfinite_training_history" in msg:
        return "nonfinite_training_history"
    if "nan" in msg:
        return "nan_exception"
    if "inf" in msg:
        return "inf_exception"
    return f"exception: {repr(exc)}"


def _finite_array(x) -> bool:
    arr = np.asarray(x, dtype=float)
    return bool(np.all(np.isfinite(arr)))


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


def _is_nonfinite_training_failure(record: dict) -> bool:
    return "nonfinite_training_history" in str(record.get("failure_reason", "")).lower()


def _safe_last(x) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr[-1])


def _safe_min(x) -> float:
    arr = np.asarray(x, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float("nan")
    return float(np.nanmin(arr))


def _write_json(path: str, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


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


def _make_record(
    spec: ArchSpec,
    cfg: MaxDlambdaAblationConfig,
    seed: int,
    max_dlambda: float,
    exp_dir: str,
    *,
    success: bool,
    failure_reason: str,
    train_hist=None,
    valid_hist=None,
) -> dict:
    train_hist = np.asarray([np.nan] if train_hist is None else train_hist, dtype=float)
    valid_hist = np.asarray([np.nan] if valid_hist is None else valid_hist, dtype=float)

    train_finite = _finite_array(train_hist)
    valid_finite = _finite_array(valid_hist)

    if cfg.strict_finite_check and success:
        if not train_finite:
            success = False
            failure_reason = "nonfinite_train_hist"
        elif not valid_finite:
            success = False
            failure_reason = "nonfinite_valid_hist"

    return {
        "arch_name": spec.name,
        "model_cls": spec.model_cls.__name__,
        "which_case": spec.which_case,
        "seed": int(seed),
        "max_dlambda": float(max_dlambda),
        "iters": int(cfg.iters),
        "ls_steps": int(cfg.ls_steps),
        "abs_tol": float(cfg.abs_tol),
        "rel_tol": float(cfg.rel_tol),
        "early_stop": bool(cfg.early_stop),
        "n_epochs": int(cfg.n_epochs),
        "lr": float(cfg.lr),
        "train_fail_on_nonconvergence": bool(cfg.train_fail_on_nonconvergence),
        "validation_loss_fail_on_nonconvergence": False,
        "prediction_fail_on_nonconvergence": bool(cfg.prediction_fail_on_nonconvergence),
        "hessian_reg_strength": float(cfg.hessian_reg_strength),
        "hessian_reg_probes": int(cfg.hessian_reg_probes),
        "hessian_reg_seed": int(cfg.hessian_reg_seed),
        "force_key": cfg.force_key,
        "force_loss_strength": float(cfg.force_loss_strength),
        "force_components": [int(c) for c in cfg.force_components],
        "force_sign": float(cfg.force_sign),
        "early_stopping": bool(cfg.early_stopping),
        "early_stopping_patience": None if cfg.early_stopping_patience is None else int(cfg.early_stopping_patience),
        "early_stopping_min_delta": float(cfg.early_stopping_min_delta),
        "restore_best_model": bool(cfg.restore_best_model),
        "success": bool(success),
        "failure_reason": failure_reason,
        "train_hist_finite": train_finite,
        "valid_hist_finite": valid_finite,
        "final_train_loss": _safe_last(train_hist),
        "final_valid_loss": _safe_last(valid_hist),
        "min_train_loss": _safe_min(train_hist),
        "min_valid_loss": _safe_min(valid_hist),
        "exp_dir": exp_dir,
    }


def _save_config(
    exp_dir: str,
    spec: ArchSpec,
    cfg: MaxDlambdaAblationConfig,
    seed: int,
    max_dlambda: float,
    properties=None,
    train_file: Optional[str] = None,
    valid_file: Optional[str] = None,
):
    payload = asdict(cfg)
    payload.update(
        {
            "arch_name": spec.name,
            "model_cls": spec.model_cls.__name__,
            "which_case": spec.which_case,
            "seed": int(seed),
            "max_dlambda": float(max_dlambda),
            "effective_train_fail_on_nonconvergence": bool(cfg.train_fail_on_nonconvergence),
            "validation_loss_fail_on_nonconvergence": False,
        }
    )
    if properties is not None:
        payload["properties"] = _serialize_properties(properties)
    if train_file is not None:
        payload["train_file"] = os.path.abspath(train_file)
    if valid_file is not None:
        payload["valid_file"] = os.path.abspath(valid_file)
    _write_json(os.path.join(exp_dir, "config.json"), payload)


def _save_npz(
    exp_dir: str,
    spec: ArchSpec,
    cfg: MaxDlambdaAblationConfig,
    seed: int,
    max_dlambda: float,
    train_hist,
    valid_hist,
    train_pred,
    valid_pred,
    train_data: Dataset,
    valid_data: Dataset,
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
        seed=int(seed),
        max_dlambda=float(max_dlambda),
        hidden=np.asarray(cfg.hidden, dtype=int),
        input_mode=cfg.input_mode,
        activation=cfg.activation,
        corr_factor=float(cfg.corr_factor),
        only_stretching_NN=int(cfg.only_stretching_NN),
        only_bending_NN=int(cfg.only_bending_NN),
        zero_reference=int(cfg.zero_reference),
        der_K_diag=np.asarray(cfg.der_K_diag, dtype=float),
        der_K_chol=np.asarray(cfg.der_K_chol, dtype=float),
        iters=int(cfg.iters),
        ls_steps=int(cfg.ls_steps),
        abs_tol=float(cfg.abs_tol),
        rel_tol=float(cfg.rel_tol),
        early_stop=int(cfg.early_stop),
        train_fail_on_nonconvergence=bool(cfg.train_fail_on_nonconvergence),
        validation_loss_fail_on_nonconvergence=False,
        prediction_fail_on_nonconvergence=bool(cfg.prediction_fail_on_nonconvergence),
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
        train_truth=np.asarray(train_data.qs, dtype=float),
        valid_truth=np.asarray(valid_data.qs, dtype=float),
        train_lambdas=np.asarray(train_data.lambdas, dtype=float),
        valid_lambdas=np.asarray(valid_data.lambdas, dtype=float),
        train_valid_mask=np.asarray(train_data.valid, dtype=bool),
        valid_valid_mask=np.asarray(valid_data.valid, dtype=bool),
    )
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


def _dataset_has_force_for_loss(cfg: MaxDlambdaAblationConfig, data: Dataset) -> bool:
    return (
        cfg.force_loss_strength != 0.0
        and cfg.save_force_predictions
        and data.forces is not None
    )


def _predict_dataset_reaction_forces(model, base, aux, data: Dataset, cfg: MaxDlambdaAblationConfig):
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


def run_one_max_dlambda_case(
    properties,
    train_file: str,
    valid_file: str,
    spec: ArchSpec,
    cfg: MaxDlambdaAblationConfig,
    seed: int,
    max_dlambda: float,
) -> dict:
    exp_name = _experiment_name(spec, cfg, seed, max_dlambda)
    exp_dir = os.path.join(cfg.output_dir, spec.name, f"seed_{seed}", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    _save_config(
        exp_dir,
        spec,
        cfg,
        seed,
        max_dlambda,
        properties=properties,
        train_file=train_file,
        valid_file=valid_file,
    )

    params = _make_params(cfg, spec, seed, max_dlambda)

    if cfg.verbose:
        print("=" * 100)
        print(f"Architecture : {spec.name}")
        print(f"which_case   : {spec.which_case}")
        print(f"seed         : {seed}")
        print(f"max_dlambda  : {max_dlambda:.3e}")
        print(f"iters        : {cfg.iters}")
        print(f"ls_steps     : {cfg.ls_steps}")
        print(f"abs_tol      : {cfg.abs_tol:.3e}")
        print(f"rel_tol      : {cfg.rel_tol:.3e}")
        print(f"early_stop   : {cfg.early_stop}")
        print(f"training fail_on_nonconvergence      : {cfg.train_fail_on_nonconvergence}")
        print("validation loss fail_on_nonconvergence: False")
        print(f"prediction fail_on_nonconvergence    : {cfg.prediction_fail_on_nonconvergence}")
        print(f"early_stopping                       : {cfg.early_stopping}")
        print(f"early_stopping_patience              : {cfg.early_stopping_patience}")
        print(f"restore_best_model                   : {cfg.restore_best_model}")
        print(f"hessian_reg_strength                 : {cfg.hessian_reg_strength:.3e}")
        print(f"hessian_reg_probes                   : {cfg.hessian_reg_probes}")
        print(f"hessian_reg_seed                     : {cfg.hessian_reg_seed}")
        print(f"force_key                            : {cfg.force_key}")
        print(f"force_loss_strength                  : {cfg.force_loss_strength:.3e}")
        print(f"force_components                     : {cfg.force_components}")
        print(f"force_sign                           : {cfg.force_sign:g}")
        print("=" * 100)

    try:
        train_result = train_model(
            properties=properties,
            model_cls=spec.model_cls,
            params=params,
            train_file=train_file,
            valid_file=valid_file,
            force_key=cfg.force_key,
            n_epochs=cfg.n_epochs,
            lr=cfg.lr,
            valid_every=cfg.valid_every,
            max_dlambda=max_dlambda,
            iters=cfg.iters,
            ls_steps=cfg.ls_steps,
            abs_tol=cfg.abs_tol,
            rel_tol=cfg.rel_tol,
            fail_on_nonconvergence=cfg.train_fail_on_nonconvergence,
            early_stop=cfg.early_stop,
            hessian_reg_strength=cfg.hessian_reg_strength,
            hessian_reg_probes=cfg.hessian_reg_probes,
            hessian_reg_seed=cfg.hessian_reg_seed,
            force_loss_strength=cfg.force_loss_strength,
            force_components=cfg.force_components,
            force_sign=cfg.force_sign,
            early_stopping_patience=(
                cfg.early_stopping_patience if cfg.early_stopping else None
            ),
            early_stopping_min_delta=cfg.early_stopping_min_delta,
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
    except Exception as exc:
        failure_reason = f"training_{_classify_exception(exc)}"
        _write_json(os.path.join(exp_dir, "failure.json"), {"success": False, "failure_reason": failure_reason})
        if cfg.verbose:
            print(f"[FAILED] {spec.name}: {failure_reason}")
        return _make_record(
            spec,
            cfg,
            seed,
            max_dlambda,
            exp_dir,
            success=False,
            failure_reason=failure_reason,
        )

    try:
        base, aux = get_slinky(properties)
        train_data = Dataset.load(train_file, force_key=cfg.force_key)
        valid_data = Dataset.load(valid_file, force_key=cfg.force_key)
        train_pred = predict(
            model,
            base,
            aux,
            train_data.idx_b,
            train_data.xb,
            train_data.lambdas,
            max_dlambda=max_dlambda,
            iters=cfg.iters,
            ls_steps=cfg.ls_steps,
            abs_tol=cfg.abs_tol,
            rel_tol=cfg.rel_tol,
            fail_on_nonconvergence=cfg.prediction_fail_on_nonconvergence,
            early_stop=cfg.early_stop,
        )
        valid_pred = predict(
            model,
            base,
            aux,
            valid_data.idx_b,
            valid_data.xb,
            valid_data.lambdas,
            max_dlambda=max_dlambda,
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
    except Exception as exc:
        failure_reason = f"prediction_{_classify_exception(exc)}"
        _write_json(os.path.join(exp_dir, "failure.json"), {"success": False, "failure_reason": failure_reason})
        if cfg.verbose:
            print(f"[FAILED] {spec.name}: {failure_reason}")
        return _make_record(
            spec,
            cfg,
            seed,
            max_dlambda,
            exp_dir,
            success=False,
            failure_reason=failure_reason,
            train_hist=train_hist,
            valid_hist=valid_hist,
        )

    if cfg.save_model:
        eqx.tree_serialise_leaves(os.path.join(exp_dir, "model.eqx"), model)

    if cfg.save_npz:
        _save_npz(
            exp_dir,
            spec,
            cfg,
            seed,
            max_dlambda,
            train_hist,
            valid_hist,
            train_pred,
            valid_pred,
            train_data,
            valid_data,
            train_displacement_hist=train_displacement_hist,
            train_force_hist=train_force_hist,
            valid_displacement_hist=valid_displacement_hist,
            valid_force_hist=valid_force_hist,
            train_force_pred=train_force_pred,
            valid_force_pred=valid_force_pred,
            train_force_truth=train_force_truth,
            valid_force_truth=valid_force_truth,
        )

    if cfg.save_plots:
        title = exp_name
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

    return _make_record(
        spec,
        cfg,
        seed,
        max_dlambda,
        exp_dir,
        success=True,
        failure_reason="",
        train_hist=train_hist,
        valid_hist=valid_hist,
    )


def run_max_dlambda_ablation(
    properties,
    train_file: str,
    valid_file: str,
    cfg: MaxDlambdaAblationConfig,
    selected_architectures: Optional[list[str]] = None,
) -> dict[str, list[dict]]:
    registry = build_architecture_registry()
    arch_names = subset_all() if selected_architectures is None else selected_architectures

    unknown = [name for name in arch_names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown architecture names: {unknown}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    all_results = {}

    for arch_name in arch_names:
        spec = registry[arch_name]
        records = []
        stop_this_architecture = False

        for seed in cfg.seed_list:
            if stop_this_architecture:
                break

            for max_dlambda in cfg.max_dlambda_values:
                rec = run_one_max_dlambda_case(
                    properties=properties,
                    train_file=train_file,
                    valid_file=valid_file,
                    spec=spec,
                    cfg=cfg,
                    seed=int(seed),
                    max_dlambda=float(max_dlambda),
                )
                records.append(rec)

                if cfg.verbose:
                    status = "SUCCESS" if rec["success"] else f"FAIL ({rec['failure_reason']})"
                    train = rec["final_train_loss"]
                    valid = rec["final_valid_loss"]
                    train_s = f"{train:.3e}" if np.isfinite(train) else "nan"
                    valid_s = f"{valid:.3e}" if np.isfinite(valid) else "nan"
                    print(
                        f"[{arch_name}] seed={seed} | max_dlambda={max_dlambda:.3e} | "
                        f"train={train_s} | valid={valid_s} | {status}"
                    )

                if _is_nonfinite_training_failure(rec):
                    stop_this_architecture = True
                    if cfg.verbose:
                        print(
                            f"Stopping remaining max_dlambda values and seeds for {arch_name} "
                            "after nonfinite training history. Moving to next architecture."
                        )
                    break

                if cfg.stop_after_first_failure and not rec["success"]:
                    stop_this_architecture = True
                    if cfg.verbose:
                        print(
                            f"Stopping larger max_dlambda values for {arch_name} "
                            "after first failure. Moving to next architecture."
                        )
                    break

        all_results[arch_name] = records

        if cfg.save_summary_json:
            out = os.path.join(cfg.output_dir, arch_name)
            os.makedirs(out, exist_ok=True)
            _write_json(os.path.join(out, "summary_all_seeds.json"), records)

    if cfg.save_summary_json:
        _write_json(os.path.join(cfg.output_dir, "all_results.json"), all_results)

    return all_results


def run_max_dlambda_hessian_diagnostics(output_dir: str, **diagnostic_kwargs):
    """
    Run post-training Hessian diagnostics for saved max_dlambda ablation outputs.

    Diagnostics are intentionally separate from training. This wrapper mirrors
    seed_ablation_utils.run_seed_hessian_diagnostics for notebook convenience.
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

    print(f"Finished Hessian diagnostics for {n_ok}/{len(exp_dirs)} max_dlambda runs.")
    return n_ok


def summarize_best_stable_step(all_results: dict[str, list[dict]]) -> dict:
    summary = {}
    for arch_name, records in all_results.items():
        by_seed = {}
        seeds = sorted(set(int(r["seed"]) for r in records))
        for seed in seeds:
            stable = [
                r
                for r in records
                if int(r["seed"]) == seed
                and bool(r["success"])
                and np.isfinite(r["final_valid_loss"])
            ]
            if not stable:
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
