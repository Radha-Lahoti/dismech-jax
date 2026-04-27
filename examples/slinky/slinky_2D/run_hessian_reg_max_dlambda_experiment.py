"""Sweep Hessian regularization strength, then test maximum stable load step.

This experiment is designed for the paper claim:

    stronger Hessian regularization improves the learned energy landscape
    enough to permit larger continuation steps at inference time.

Unlike run_max_dlambda_ablation_minimal.py, each model is trained with the
same conservative max_dlambda. After training, the model is frozen and tested
with increasingly large max_dlambda values using fail_on_nonconvergence=True.

Example:
    uv run python examples/slinky/slinky_2D/run_hessian_reg_max_dlambda_experiment.py \
        --properties-class StripN9Properties \
        --train-file examples/slinky/experiment_data/n9_strip_train_dataset.npz \
        --valid-file examples/slinky/experiment_data/n9_strip_test_dataset.npz \
        --architectures diag_energy_mlp,icnn_energy,diag_stiffness_mlp \
        --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import properties as properties_module
from compute_architecture_hessian_diagnostics import (
    _hessian_path_diagnostics,
    _sample_dataset,
)
from run_architectures import (
    SweepConfig,
    build_architecture_registry,
    experiment_name,
    run_one_architecture,
    subset_brazier_stiffness_only,
    subset_tape_tube_candidates,
)
from util import Dataset, get_slinky, predict


DEFAULT_ARCHITECTURES = (
    "diag_energy_mlp",
    "icnn_energy",
    "diag_stiffness_mlp",
)
DEFAULT_HESSIAN_REG_STRENGTHS = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
DEFAULT_EVAL_MAX_DLAMBDAS = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1)
DEFAULT_SEEDS = (0,)


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(v.strip()) for v in value.split(",") if v.strip())


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return arr.item()
        return arr.tolist()
    except Exception:
        return repr(value)


def _properties_from_name(class_name: str, mass: Optional[float]):
    cls = getattr(properties_module, class_name)
    if mass is None:
        return cls()
    return cls(mass=mass)


def _masked_displacement_mse(pred, truth, valid) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    err = (pred - truth) ** 2
    masked = np.where(valid[..., None], err, 0.0)
    return float(np.sum(masked) / max(float(np.sum(valid)), 1.0))


def _is_finite_tree(tree) -> bool:
    for leaf in jax.tree_util.tree_leaves(tree):
        try:
            arr = np.asarray(leaf)
        except Exception:
            continue
        if np.issubdtype(arr.dtype, np.inexact) and arr.size > 0:
            if not np.all(np.isfinite(arr)):
                return False
    return True


def evaluate_stable_steps(
    model,
    properties,
    valid_data: Dataset,
    eval_max_dlambda_values: tuple[float, ...],
    *,
    iters: int,
    ls_steps: int,
    abs_tol: float,
    rel_tol: float,
    early_stop: bool,
) -> tuple[Optional[float], list[dict]]:
    base, aux = get_slinky(properties)
    records = []
    largest_stable = None

    sorted_values = sorted({float(v) for v in eval_max_dlambda_values}, reverse=True)
    for max_dlambda in sorted_values:
        started = time.time()
        try:
            pred = predict(
                model,
                base,
                aux,
                valid_data.idx_b,
                valid_data.xb,
                valid_data.lambdas,
                max_dlambda=float(max_dlambda),
                iters=iters,
                ls_steps=ls_steps,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                fail_on_nonconvergence=True,
                early_stop=early_stop,
            )
            pred_np = np.asarray(pred, dtype=float)
            if not np.all(np.isfinite(pred_np)):
                raise FloatingPointError("nonfinite_prediction")
            mse = _masked_displacement_mse(pred_np, valid_data.qs, valid_data.valid)
            success = True
            failure_reason = ""
            largest_stable = float(max_dlambda)
        except Exception as exc:
            mse = float("nan")
            success = False
            failure_reason = repr(exc)

        records.append(
            {
                "max_dlambda": float(max_dlambda),
                "success": bool(success),
                "valid_displacement_mse": float(mse),
                "failure_reason": failure_reason,
                "seconds": round(time.time() - started, 3),
            }
        )

        if success:
            break

    return largest_stable, records


def compute_hessian_summary(
    model,
    properties,
    valid_data: Dataset,
    cfg: SweepConfig,
    *,
    max_trajectories: Optional[int],
    stride: int,
) -> dict:
    base, aux = get_slinky(properties)
    diag_data = _sample_dataset(valid_data, max_trajectories=max_trajectories, stride=stride)
    diag = _hessian_path_diagnostics(
        model,
        base,
        aux,
        diag_data,
        use_predicted=False,
        max_dlambda=cfg.max_dlambda,
        iters=cfg.iters,
        ls_steps=cfg.ls_steps,
        abs_tol=cfg.abs_tol,
        rel_tol=cfg.rel_tol,
        fail_on_nonconvergence=False,
        early_stop=cfg.early_stop,
    )
    scalar_keys = (
        "M_hat",
        "kappa_hat",
        "L_adj_hat",
        "sigma_min_min",
        "sigma_max_max",
        "lambda_min_min",
        "lambda_max_max",
        "negative_min_eig_fraction",
        "n_states",
        "n_adjacent_pairs",
    )
    return {key: _jsonable(diag[key]) for key in scalar_keys}


def write_summary_csv(records: list[dict], csv_path: str) -> None:
    fieldnames = [
        "arch_name",
        "seed",
        "hessian_reg_strength",
        "train_success",
        "failure_reason",
        "train_max_dlambda",
        "largest_stable_eval_max_dlambda",
        "best_eval_valid_displacement_mse",
        "final_train_loss",
        "final_valid_loss",
        "M_hat",
        "kappa_hat",
        "L_adj_hat",
        "sigma_min_min",
        "sigma_max_max",
        "lambda_min_min",
        "lambda_max_max",
        "negative_min_eig_fraction",
        "n_states",
        "n_adjacent_pairs",
        "exp_dir",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            eval_successes = [
                r["valid_displacement_mse"]
                for r in record.get("eval_by_max_dlambda", [])
                if r.get("success") and np.isfinite(r.get("valid_displacement_mse", np.nan))
            ]
            row = {
                "arch_name": record.get("arch_name", ""),
                "seed": record.get("seed", ""),
                "hessian_reg_strength": record.get("hessian_reg_strength", ""),
                "train_success": record.get("train_success", ""),
                "failure_reason": record.get("failure_reason", ""),
                "train_max_dlambda": record.get("train_max_dlambda", ""),
                "largest_stable_eval_max_dlambda": record.get(
                    "largest_stable_eval_max_dlambda", ""
                ),
                "best_eval_valid_displacement_mse": (
                    min(eval_successes) if eval_successes else np.nan
                ),
                "final_train_loss": record.get("final_train_loss", np.nan),
                "final_valid_loss": record.get("final_valid_loss", np.nan),
                "exp_dir": record.get("exp_dir", ""),
            }
            row.update(record.get("hessian_summary", {}))
            writer.writerow(row)


def append_jsonl(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_jsonable(payload)) + "\n")


def make_cfg(args, *, seed: int, hessian_reg_strength: float) -> SweepConfig:
    return SweepConfig(
        der_K_diag=args.der_k_diag,
        der_K_chol=args.der_k_chol,
        hidden=args.hidden,
        corr_factor=args.corr_factor,
        input_mode=args.input_mode,
        only_stretching_NN=args.only_stretching_nn,
        only_bending_NN=args.only_bending_nn,
        zero_reference=not args.no_zero_reference,
        activation=args.activation,
        mode=args.mode,
        n_epochs=args.n_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=int(seed),
        valid_every=args.valid_every,
        max_dlambda=args.train_max_dlambda,
        iters=args.iters,
        ls_steps=args.ls_steps,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        early_stop=not args.no_early_stop,
        train_fail_on_nonconvergence=not args.train_allow_nonconvergence,
        prediction_fail_on_nonconvergence=False,
        hessian_reg_strength=float(hessian_reg_strength),
        hessian_reg_probes=args.hessian_reg_probes,
        hessian_reg_seed=args.hessian_reg_seed,
        force_key=args.force_key,
        force_loss_strength=args.force_loss_strength,
        force_components=args.force_components,
        force_sign=args.force_sign,
        return_loss_components=args.return_loss_components,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        restore_best_model=not args.no_restore_best_model,
        save_force_predictions=False,
        plot_force_predictions=False,
        output_dir=args.output_dir,
        save_npz=True,
        save_model=True,
        save_plots=not args.no_plots,
        verbose=not args.quiet,
        save_energy_landscapes=False,
    )


def run_experiment(args) -> list[dict]:
    registry = build_architecture_registry()
    unknown = [name for name in args.architectures if name not in registry]
    if unknown:
        raise ValueError(f"Unknown architectures: {unknown}")

    properties = _properties_from_name(args.properties_class, args.mass)
    valid_data = Dataset.load(args.valid_file, force_key=args.force_key)
    os.makedirs(args.output_dir, exist_ok=True)

    jsonl_path = os.path.join(args.output_dir, "progress.jsonl")
    csv_path = os.path.join(args.output_dir, "summary.csv")
    records = []

    sorted_hregs = sorted(args.hessian_reg_strengths)
    eval_max_value = (
        max(args.eval_max_dlambda_values) if args.eval_max_dlambda_values else None
    )

    for arch_name in args.architectures:
        spec = registry[arch_name]
        arch_saturated = False
        for seed in args.seeds:
            if arch_saturated:
                print(
                    f"[skip] arch={arch_name} already saturated eval grid; "
                    f"skipping seed={seed}.",
                    flush=True,
                )
                break
            for hreg in sorted_hregs:
                cfg = make_cfg(args, seed=seed, hessian_reg_strength=hreg)
                started = time.time()
                print(
                    f"\n[train] arch={arch_name} seed={seed} "
                    f"hreg={hreg:g} train_max_dlambda={cfg.max_dlambda:g}",
                    flush=True,
                )

                result = run_one_architecture(
                    properties=properties,
                    train_file=args.train_file,
                    valid_file=args.valid_file,
                    spec=spec,
                    cfg=cfg,
                )

                record = {
                    "arch_name": arch_name,
                    "seed": int(seed),
                    "hessian_reg_strength": float(hreg),
                    "train_max_dlambda": float(cfg.max_dlambda),
                    "train_success": bool(result["success"]),
                    "failure_reason": result.get("failure_reason", ""),
                    "exp_dir": result["exp_dir"],
                    "exp_name": experiment_name(spec, cfg),
                    "config": asdict(cfg),
                    "seconds_train_plus_eval": None,
                    "largest_stable_eval_max_dlambda": None,
                    "eval_by_max_dlambda": [],
                    "hessian_summary": {},
                }

                if result["success"] and result["model"] is not None:
                    model = result["model"]
                    if not _is_finite_tree(model):
                        record["train_success"] = False
                        record["failure_reason"] = "nonfinite_model_parameters"
                    else:
                        print(
                            f"[eval] arch={arch_name} seed={seed} hreg={hreg:g} "
                            f"eval steps={args.eval_max_dlambda_values}",
                            flush=True,
                        )
                        largest, eval_records = evaluate_stable_steps(
                            model,
                            properties,
                            valid_data,
                            args.eval_max_dlambda_values,
                            iters=args.eval_iters,
                            ls_steps=args.ls_steps,
                            abs_tol=args.abs_tol,
                            rel_tol=args.rel_tol,
                            early_stop=cfg.early_stop,
                        )
                        record["largest_stable_eval_max_dlambda"] = largest
                        record["eval_by_max_dlambda"] = eval_records

                        if args.save_hessian_summary:
                            try:
                                record["hessian_summary"] = compute_hessian_summary(
                                    model,
                                    properties,
                                    valid_data,
                                    cfg,
                                    max_trajectories=args.hessian_max_trajectories,
                                    stride=args.hessian_stride,
                                )
                            except Exception as exc:
                                record["hessian_summary_error"] = repr(exc)

                    train_hist = np.asarray(result["train_hist"], dtype=float)
                    valid_hist = np.asarray(result["valid_hist"], dtype=float)
                    record["final_train_loss"] = (
                        float(train_hist[-1]) if train_hist.size else np.nan
                    )
                    record["final_valid_loss"] = (
                        float(valid_hist[-1]) if valid_hist.size else np.nan
                    )

                    summary_path = os.path.join(result["exp_dir"], "post_train_step_eval.json")
                    with open(summary_path, "w") as f:
                        json.dump(_jsonable(record), f, indent=2)

                record["seconds_train_plus_eval"] = round(time.time() - started, 3)
                records.append(record)
                append_jsonl(jsonl_path, record)
                write_summary_csv(records, csv_path)

                largest = record["largest_stable_eval_max_dlambda"]
                print(
                    f"[done] arch={arch_name} seed={seed} hreg={hreg:g} "
                    f"train_success={record['train_success']} largest_eval={largest}",
                    flush=True,
                )

                if (
                    eval_max_value is not None
                    and largest is not None
                    and float(largest) >= float(eval_max_value)
                ):
                    arch_saturated = True
                    print(
                        f"[skip] arch={arch_name} reached largest eval max_dlambda="
                        f"{eval_max_value:g} at hreg={hreg:g} (seed={seed}); "
                        f"skipping higher hregs and remaining seeds for this arch.",
                        flush=True,
                    )
                    break

    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(_jsonable(records), f, indent=2)
    write_summary_csv(records, csv_path)
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train with Hessian regularization, then measure largest stable inference max_dlambda."
    )
    parser.add_argument(
        "--properties-class",
        default="StripN9Properties",
        help="Class name in properties.py, e.g. StripN9Properties, SlinkyN3Properties, TapeN11Properties.",
    )
    parser.add_argument("--mass", type=float, default=None, help="Optional mass override for properties.")
    parser.add_argument(
        "--train-file",
        default="examples/slinky/experiment_data/n9_strip_train_dataset.npz",
    )
    parser.add_argument(
        "--valid-file",
        default="examples/slinky/experiment_data/n9_strip_test_dataset.npz",
    )
    parser.add_argument(
        "--architectures",
        type=_parse_csv_strings,
        default=DEFAULT_ARCHITECTURES,
        help="Comma-separated architecture names from run_architectures.py.",
    )
    parser.add_argument("--seeds", type=_parse_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--hessian-reg-strengths",
        type=_parse_csv_floats,
        default=DEFAULT_HESSIAN_REG_STRENGTHS,
    )
    parser.add_argument(
        "--eval-max-dlambda-values",
        type=_parse_csv_floats,
        default=DEFAULT_EVAL_MAX_DLAMBDAS,
    )
    parser.add_argument("--train-max-dlambda", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="hessian_reg_max_dlambda_outputs")

    parser.add_argument("--der-k-diag", type=_parse_csv_floats, default=(2.0, 0.002))
    parser.add_argument("--der-k-chol", type=_parse_csv_floats, default=(2.0, 0.0, 0.002))
    parser.add_argument("--hidden", type=_parse_csv_ints, default=(10,))
    parser.add_argument("--corr-factor", type=float, default=0.01)
    parser.add_argument("--input-mode", choices=("raw", "invariant"), default="raw")
    parser.add_argument("--only-stretching-nn", action="store_true")
    parser.add_argument("--only-bending-nn", action="store_true")
    parser.add_argument("--no-zero-reference", action="store_true")
    parser.add_argument("--activation", choices=("softplus", "tanh"), default="tanh")
    parser.add_argument(
        "--mode",
        choices=("anisotropic", "isotropic"),
        default=None,
        help="Structured Brazier mode; leave unset for the model default.",
    )

    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--valid-every", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=8,
        help="Newton iterations during the post-training max_dlambda eval sweep (separate from training iters).",
    )
    parser.add_argument("--ls-steps", type=int, default=10)
    parser.add_argument("--abs-tol", type=float, default=1e-4)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--train-allow-nonconvergence", action="store_true")

    parser.add_argument("--hessian-reg-probes", type=int, default=1)
    parser.add_argument("--hessian-reg-seed", type=int, default=0)
    parser.add_argument("--save-hessian-summary", action="store_true", default=True)
    parser.add_argument("--no-hessian-summary", dest="save_hessian_summary", action="store_false")
    parser.add_argument("--hessian-max-trajectories", type=int, default=1)
    parser.add_argument("--hessian-stride", type=int, default=10)

    parser.add_argument("--force-key", default=None)
    parser.add_argument("--force-loss-strength", type=float, default=0.0)
    parser.add_argument("--force-components", type=_parse_csv_ints, default=(0, 1, 2))
    parser.add_argument("--force-sign", type=float, default=1.0)
    parser.add_argument("--return-loss-components", action="store_true")

    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--no-restore-best-model", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
