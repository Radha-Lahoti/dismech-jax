import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from Energy_NN_architectures import (
    CholeskyPlusEnergyNN,
    CholeskyPlusStiffnessNN,
    CholeskyPlusStiffnessSignedNN,
    DiagonalPlusEnergyNN,
    DiagonalPlusStiffnessNN,
    ModelParams,
    ScalarEnergyNN,
)
from util import Dataset, get_slinky, predict, train_model


@dataclass(frozen=True)
class ArchSpec:
    name: str
    model_cls: type
    which_case: str


def build_architecture_registry() -> dict[str, ArchSpec]:
    return {
        "diag_energy_baseline": ArchSpec("diag_energy_baseline", DiagonalPlusEnergyNN, "baseline"),
        "diag_energy_mlp": ArchSpec("diag_energy_mlp", DiagonalPlusEnergyNN, "MLP"),
        "diag_energy_icnn": ArchSpec("diag_energy_icnn", DiagonalPlusEnergyNN, "ICNN"),
        "mlp_energy": ArchSpec("mlp_energy", ScalarEnergyNN, "MLP"),
        "icnn_energy": ArchSpec("icnn_energy", ScalarEnergyNN, "ICNN"),
        "chol_energy_baseline": ArchSpec("chol_energy_baseline", CholeskyPlusEnergyNN, "baseline"),
        "chol_energy_mlp": ArchSpec("chol_energy_mlp", CholeskyPlusEnergyNN, "MLP"),
        "chol_energy_icnn": ArchSpec("chol_energy_icnn", CholeskyPlusEnergyNN, "ICNN"),
        "diag_stiffness_mlp": ArchSpec("diag_stiffness_mlp", DiagonalPlusStiffnessNN, "MLP"),
        "diag_stiffness_icnn": ArchSpec("diag_stiffness_icnn", DiagonalPlusStiffnessNN, "ICNN"),
        "chol_stiffness_mlp": ArchSpec("chol_stiffness_mlp", CholeskyPlusStiffnessNN, "MLP"),
        "chol_stiffness_icnn": ArchSpec("chol_stiffness_icnn", CholeskyPlusStiffnessNN, "ICNN"),
        "chol_stiffness_signed_mlp": ArchSpec(
            "chol_stiffness_signed_mlp", CholeskyPlusStiffnessSignedNN, "MLP"
        ),
        "chol_stiffness_signed_icnn": ArchSpec(
            "chol_stiffness_signed_icnn", CholeskyPlusStiffnessSignedNN, "ICNN"
        ),
    }


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


def subset_main_paper_candidates() -> list[str]:
    return [
        "diag_energy_baseline",
        "diag_energy_mlp",
        "diag_energy_icnn",
        "chol_stiffness_mlp",
    ]


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

    max_dlambda_values: tuple[float, ...] = (1e-2, 5e-2, 1e-1, 5e-1, 1.0)
    iters: int = 10
    ls_steps: int = 10
    abs_tol: float = 1e-8
    rel_tol: float = 1e-6

    # This is passed only into train_model(...). Inside util.py, validation loss
    # is already computed with fail_on_nonconvergence=False.
    train_fail_on_nonconvergence: bool = True

    # This is passed only into final predict(...), after training completes.
    prediction_fail_on_nonconvergence: bool = False

    output_dir: str = "max_dlambda_ablation_minimal_outputs"
    save_npz: bool = True
    save_model: bool = False
    save_summary_json: bool = True
    strict_finite_check: bool = True
    stop_after_first_failure: bool = False
    verbose: bool = True


def _der_K_for_model(cfg: MaxDlambdaAblationConfig, model_cls: type) -> jax.Array:
    if model_cls is ScalarEnergyNN:
        return jnp.asarray(cfg.der_K_diag)
    if model_cls in (DiagonalPlusEnergyNN, DiagonalPlusStiffnessNN):
        return jnp.asarray(cfg.der_K_diag)
    if model_cls in (
        CholeskyPlusEnergyNN,
        CholeskyPlusStiffnessNN,
        CholeskyPlusStiffnessSignedNN,
    ):
        return jnp.asarray(cfg.der_K_chol)
    raise ValueError(f"Unsupported model class: {model_cls}")


def _make_params(cfg: MaxDlambdaAblationConfig, spec: ArchSpec, seed: int) -> ModelParams:
    return ModelParams(
        der_K=_der_K_for_model(cfg, spec.model_cls),
        key=jax.random.PRNGKey(seed),
        hidden=cfg.hidden,
        which_case=spec.which_case,
        corr_factor=cfg.corr_factor,
        input_mode=cfg.input_mode,
        only_stretching_NN=cfg.only_stretching_NN,
        only_bending_NN=cfg.only_bending_NN,
        zero_reference=cfg.zero_reference,
        activation=cfg.activation,
    )


def _experiment_name(spec: ArchSpec, cfg: MaxDlambdaAblationConfig, seed: int, max_dlambda: float) -> str:
    hidden = "x".join(str(h) for h in cfg.hidden)
    zr = "zr1" if cfg.zero_reference else "zr0"
    return (
        f"{spec.name}"
        f"__hid_{hidden}"
        f"__inp_{cfg.input_mode}"
        f"__stretchNN_{int(cfg.only_stretching_NN)}"
        f"__bendNN_{int(cfg.only_bending_NN)}"
        f"__act_{cfg.activation}"
        f"__corr_{cfg.corr_factor:g}"
        f"__{zr}"
        f"__seed_{seed}"
        f"__mdl_{max_dlambda:g}"
        f"__it_{cfg.iters}"
    )


def _classify_exception(exc: Exception) -> str:
    msg = repr(exc).lower()
    if "did not converge" in msg or "nonconverge" in msg or "converge" in msg:
        return "convergence_failure"
    if "nan" in msg:
        return "nan_exception"
    if "inf" in msg:
        return "inf_exception"
    return f"exception: {repr(exc)}"


def _finite_array(x) -> bool:
    arr = np.asarray(x, dtype=float)
    return bool(np.all(np.isfinite(arr)))


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
        "n_epochs": int(cfg.n_epochs),
        "lr": float(cfg.lr),
        "train_fail_on_nonconvergence": bool(cfg.train_fail_on_nonconvergence),
        "validation_loss_fail_on_nonconvergence": False,
        "prediction_fail_on_nonconvergence": bool(cfg.prediction_fail_on_nonconvergence),
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


def _save_config(exp_dir: str, spec: ArchSpec, cfg: MaxDlambdaAblationConfig, seed: int, max_dlambda: float):
    payload = asdict(cfg)
    payload.update(
        {
            "arch_name": spec.name,
            "model_cls": spec.model_cls.__name__,
            "which_case": spec.which_case,
            "seed": int(seed),
            "max_dlambda": float(max_dlambda),
            "validation_loss_fail_on_nonconvergence": False,
        }
    )
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
):
    np.savez(
        os.path.join(exp_dir, "results.npz"),
        arch_name=spec.name,
        model_cls=spec.model_cls.__name__,
        which_case=spec.which_case,
        seed=int(seed),
        max_dlambda=float(max_dlambda),
        train_fail_on_nonconvergence=bool(cfg.train_fail_on_nonconvergence),
        validation_loss_fail_on_nonconvergence=False,
        prediction_fail_on_nonconvergence=bool(cfg.prediction_fail_on_nonconvergence),
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
    _save_config(exp_dir, spec, cfg, seed, max_dlambda)

    params = _make_params(cfg, spec, seed)

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
        print(f"training fail_on_nonconvergence      : {cfg.train_fail_on_nonconvergence}")
        print("validation loss fail_on_nonconvergence: False")
        print(f"prediction fail_on_nonconvergence    : {cfg.prediction_fail_on_nonconvergence}")
        print("=" * 100)

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
            abs_tol=cfg.abs_tol,
            rel_tol=cfg.rel_tol,
            fail_on_nonconvergence=cfg.train_fail_on_nonconvergence,
        )
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
        train_data = Dataset.load(train_file)
        valid_data = Dataset.load(valid_file)
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
        )
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
