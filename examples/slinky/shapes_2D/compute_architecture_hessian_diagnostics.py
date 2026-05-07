"""Post-process saved architecture runs with Hessian path diagnostics.

Example:
    uv run python compute_architecture_hessian_diagnostics.py arch_sweep_outputs \
        --use-predicted --stride 10

New runs save dataset paths and properties in config.json, so the output
directory is usually the only required argument. For older runs, pass
--train-file, --valid-file, and --properties-class.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import fields
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import dismech_jax as djx
from dismech_jax.solver import update_aux_state

from properties import Properties
import properties as properties_module
from util import Dataset, get_slinky
from run_architectures import (
    SweepConfig,
    build_architecture_registry,
    make_model_params,
)


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

    kwargs.pop("energy_landscape_spec", None)
    return SweepConfig(**kwargs)


def _properties_from_payload(
    payload: dict,
    fallback_class_name: Optional[str] = None,
) -> Properties:
    prop_payload = payload.get("properties")
    class_name = fallback_class_name
    values = {}
    if prop_payload is not None:
        class_name = prop_payload.get("class_name", class_name)
        values = {k: v for k, v in prop_payload.items() if k != "class_name"}

    if class_name is None:
        raise ValueError(
            "No properties metadata found in config.json. Pass --properties-class for older runs."
        )

    prop_cls = getattr(properties_module, class_name)
    valid_fields = {f.name for f in fields(prop_cls)}
    kwargs = {k: v for k, v in values.items() if k in valid_fields}
    return prop_cls(**kwargs)


def _find_experiment_dirs(output_dir: str) -> list[str]:
    output_dir = os.path.abspath(output_dir)
    if os.path.isfile(os.path.join(output_dir, "config.json")):
        return [output_dir]

    exp_dirs = []
    for root, dirs, files in os.walk(output_dir):
        if "config.json" in files:
            exp_dirs.append(root)
            dirs[:] = []
    return sorted(exp_dirs)


def _resolve_data_path(
    payload: dict,
    key: str,
    cli_value: Optional[str],
    exp_dir: str,
) -> str:
    path = cli_value if cli_value is not None else payload.get(key)
    if path is None:
        raise ValueError(
            f"No {key} stored in config.json. Pass --{key.replace('_', '-')} for older runs."
        )
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


def _append_hessian_to_results(exp_dir: str, diagnostics: dict[str, dict]):
    results_path = os.path.join(exp_dir, "results.npz")
    if not os.path.isfile(results_path):
        return

    with np.load(results_path, allow_pickle=False) as data:
        save_dict = {key: np.asarray(data[key]) for key in data.files}

    for split, diag in diagnostics.items():
        for key, value in diag.items():
            save_dict[f"{split}_hessian_{key}"] = np.asarray(value)

    np.savez(results_path, **save_dict)


def _split_label(split: str) -> str:
    if split == "valid":
        return "test"
    return split


def _hessian_summary_rows_for_experiment(exp_dir: str) -> list[dict]:
    summary_path = os.path.join(exp_dir, "hessian_diagnostics_summary.json")
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.isfile(summary_path):
        return []

    with open(summary_path) as f:
        summary = json.load(f)

    payload = {}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            payload = json.load(f)

    rows = []
    for split, split_summary in summary.items():
        rows.append(
            {
                "arch_name": payload.get("arch_name", os.path.basename(exp_dir)),
                "model_cls": payload.get("model_cls", ""),
                "source_split": split,
                "split": _split_label(split),
                "M_hat": split_summary.get("M_hat", np.nan),
                "kappa_hat": split_summary.get("kappa_hat", np.nan),
                "L_adj_hat": split_summary.get("L_adj_hat", np.nan),
                "sigma_min_min": split_summary.get("sigma_min_min", np.nan),
                "sigma_max_max": split_summary.get("sigma_max_max", np.nan),
                "lambda_min_min": split_summary.get("lambda_min_min", np.nan),
                "lambda_max_max": split_summary.get("lambda_max_max", np.nan),
                "negative_min_eig_fraction": split_summary.get(
                    "negative_min_eig_fraction", np.nan
                ),
                "n_states": split_summary.get("n_states", np.nan),
                "n_adjacent_pairs": split_summary.get("n_adjacent_pairs", np.nan),
                "sampled_n_trajectories": split_summary.get(
                    "sampled_n_trajectories", np.nan
                ),
                "sampled_stride": split_summary.get("sampled_stride", np.nan),
                "use_predicted": split_summary.get("use_predicted", np.nan),
                "seed": payload.get("seed", ""),
                "max_dlambda": payload.get("max_dlambda", ""),
                "exp_dir": exp_dir,
            }
        )
    return rows


def write_hessian_summary_csv(
    output_dir: str,
    exp_dirs: Optional[list[str]] = None,
    csv_name: str = "hessian_diagnostics_table.csv",
) -> str:
    if exp_dirs is None:
        exp_dirs = _find_experiment_dirs(output_dir)

    rows = []
    for exp_dir in exp_dirs:
        rows.extend(_hessian_summary_rows_for_experiment(exp_dir))

    output_dir = os.path.abspath(output_dir)
    if os.path.isfile(os.path.join(output_dir, "config.json")):
        csv_path = os.path.join(output_dir, csv_name)
    else:
        csv_path = os.path.join(output_dir, csv_name)

    fieldnames = [
        "arch_name",
        "model_cls",
        "source_split",
        "split",
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
        "sampled_n_trajectories",
        "sampled_stride",
        "use_predicted",
        "seed",
        "max_dlambda",
        "exp_dir",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote Hessian diagnostics table with {len(rows)} rows to {csv_path}")
    return csv_path


def _diag_save_dict(diag: dict) -> dict:
    return {key: np.asarray(value) for key, value in diag.items()}


def _diag_summary(diag: dict) -> dict:
    summary = {}
    for key, value in diag.items():
        arr = np.asarray(value)
        if arr.shape == ():
            summary[key] = arr.item()
    return summary


def _sample_dataset(data: Dataset, max_trajectories: Optional[int], stride: int) -> Dataset:
    n_traj = data.qs.shape[0]
    if max_trajectories is None:
        traj_slice = slice(None)
    else:
        traj_slice = slice(0, min(int(max_trajectories), n_traj))

    time_slice = slice(None, None, max(1, int(stride)))
    idx_b = data.idx_b if data.idx_b.ndim == 1 else data.idx_b[traj_slice]
    lambdas = data.lambdas[time_slice] if data.lambdas.ndim == 1 else data.lambdas[traj_slice, time_slice]
    forces = None if data.forces is None else data.forces[traj_slice, time_slice]

    return Dataset(
        qs=data.qs[traj_slice, time_slice],
        xb=data.xb[traj_slice, time_slice],
        idx_b=idx_b,
        lambdas=lambdas,
        valid=data.valid[traj_slice, time_slice],
        forces=forces,
    )


def _compact_trajectory(qs, xb, lambdas, valid):
    T = valid.shape[0]
    steps = jnp.arange(T)
    n_valid = jnp.sum(valid)
    first_valid = jnp.argmax(valid)
    src_idx = jnp.minimum(first_valid + steps, T - 1)
    compact_valid = steps < n_valid
    return (
        np.asarray(qs[src_idx]),
        np.asarray(xb[src_idx]),
        np.asarray(lambdas[src_idx]),
        np.asarray(compact_valid, dtype=bool),
    )


def _free_hessian_block(model, rod, aux, _lambda, q):
    aux_q = update_aux_state(aux, q, rod)
    H = rod.get_H(_lambda, q, model, aux_q)
    H = 0.5 * (H + H.T)
    free = np.asarray(rod.bc.mask(q), dtype=bool)
    H_np = np.asarray(H)
    return H_np[np.ix_(free, free)]


def _hessian_path_diagnostics(
    model,
    base,
    aux,
    data: Dataset,
    use_predicted=False,
    max_dlambda=5e-3,
    iters=5,
    ls_steps=10,
    abs_tol=1e-8,
    rel_tol=1e-6,
    fail_on_nonconvergence=False,
    early_stop=True,
    min_distance=1e-12,
    eps=1e-12,
):
    n_traj = data.qs.shape[0]

    if data.idx_b.ndim == 1:
        idx_all = jnp.broadcast_to(data.idx_b, (n_traj, data.idx_b.shape[0]))
    else:
        idx_all = data.idx_b

    if data.lambdas.ndim == 1:
        lam_all = jnp.broadcast_to(data.lambdas, (n_traj, data.lambdas.shape[0]))
    else:
        lam_all = data.lambdas

    min_abs_eigs = []
    min_eigs = []
    max_eigs = []
    max_abs_eigs = []
    condition_numbers = []
    l_adj_values = []
    n_negative = 0
    n_states = 0

    for traj_idx in range(n_traj):
        idx_b = idx_all[traj_idx]
        qs_true, xb, lambdas, valid = _compact_trajectory(
            data.qs[traj_idx],
            data.xb[traj_idx],
            lam_all[traj_idx],
            data.valid[traj_idx],
        )
        rod = base.with_bc(
            djx.DirectBC(idx_b=idx_b, xb=jnp.asarray(xb), lambdas=jnp.asarray(lambdas))
        )

        if use_predicted:
            qs = rod.solve(
                model,
                jnp.asarray(lambdas),
                aux,
                max_dlambda=max_dlambda,
                iters=iters,
                ls_steps=ls_steps,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                fail_on_nonconvergence=fail_on_nonconvergence,
                early_stop=early_stop,
            )
            qs = np.asarray(qs)
        else:
            qs = qs_true

        prev_H = None
        prev_q = None
        for _lambda, q, is_valid in zip(lambdas, qs, valid):
            if not is_valid:
                continue

            H_free = _free_hessian_block(
                model,
                rod,
                aux,
                jnp.asarray(_lambda),
                jnp.asarray(q),
            )
            eigs = np.linalg.eigvalsh(H_free)
            abs_eigs = np.abs(eigs)
            min_abs = float(np.min(abs_eigs))
            max_abs = float(np.max(abs_eigs))

            min_abs_eigs.append(min_abs)
            min_eigs.append(float(np.min(eigs)))
            max_eigs.append(float(np.max(eigs)))
            max_abs_eigs.append(max_abs)
            condition_numbers.append(max_abs / max(min_abs, eps))
            n_negative += int(np.min(eigs) < 0.0)
            n_states += 1

            if prev_H is not None:
                dq = float(np.linalg.norm(q - prev_q))
                if dq > min_distance:
                    dH_norm = float(np.linalg.norm(H_free - prev_H, ord=2))
                    l_adj_values.append(dH_norm / dq)

            prev_H = H_free
            prev_q = q

    if n_states == 0:
        raise ValueError("No valid states found for Hessian diagnostics.")

    min_abs_eigs = np.asarray(min_abs_eigs)
    min_eigs = np.asarray(min_eigs)
    max_eigs = np.asarray(max_eigs)
    max_abs_eigs = np.asarray(max_abs_eigs)
    condition_numbers = np.asarray(condition_numbers)
    l_adj_values = np.asarray(l_adj_values)

    sigma_min = float(np.min(min_abs_eigs))
    l_adj_hat = float(np.max(l_adj_values)) if l_adj_values.size else np.nan

    return {
        "M_hat": 1.0 / max(sigma_min, eps),
        "kappa_hat": float(np.max(condition_numbers)),
        "L_adj_hat": l_adj_hat,
        "sigma_min_min": sigma_min,
        "sigma_max_max": float(np.max(max_abs_eigs)),
        "lambda_min_min": float(np.min(min_eigs)),
        "lambda_max_max": float(np.max(max_eigs)),
        "negative_min_eig_fraction": n_negative / n_states,
        "n_states": n_states,
        "n_adjacent_pairs": int(l_adj_values.size),
        "per_state_sigma_min": min_abs_eigs,
        "per_state_lambda_min": min_eigs,
        "per_state_lambda_max": max_eigs,
        "per_state_kappa": condition_numbers,
        "per_pair_L_adj": l_adj_values,
    }


def _compute_and_save_hessian_diagnostics(
    model,
    base,
    aux,
    train_data: Dataset,
    valid_data: Dataset,
    cfg: SweepConfig,
    exp_dir: str,
) -> dict[str, dict]:
    data_by_split = {"train": train_data, "valid": valid_data}
    diagnostics = {}
    summary = {}
    for split in cfg.hessian_diagnostics_splits:
        if split not in data_by_split:
            raise ValueError(
                f"Unknown hessian diagnostics split '{split}'. Expected one of {tuple(data_by_split)}."
            )

        diag_data = _sample_dataset(
            data_by_split[split],
            max_trajectories=cfg.hessian_diagnostics_max_trajectories,
            stride=cfg.hessian_diagnostics_stride,
        )
        diag = _hessian_path_diagnostics(
            model,
            base,
            aux,
            diag_data,
            use_predicted=cfg.hessian_diagnostics_use_predicted,
            max_dlambda=cfg.max_dlambda,
            iters=cfg.iters,
            ls_steps=cfg.ls_steps,
            abs_tol=cfg.abs_tol,
            rel_tol=cfg.rel_tol,
            fail_on_nonconvergence=cfg.prediction_fail_on_nonconvergence,
            early_stop=cfg.early_stop,
        )
        diagnostics[split] = diag
        summary[split] = _diag_summary(diag)
        summary[split]["sampled_n_trajectories"] = int(diag_data.qs.shape[0])
        summary[split]["sampled_stride"] = int(max(1, cfg.hessian_diagnostics_stride))
        summary[split]["use_predicted"] = bool(cfg.hessian_diagnostics_use_predicted)
        np.savez(
            os.path.join(exp_dir, f"hessian_diagnostics_{split}.npz"),
            **_diag_save_dict(diag),
            sampled_n_trajectories=np.asarray(diag_data.qs.shape[0], dtype=int),
            sampled_stride=np.asarray(max(1, cfg.hessian_diagnostics_stride), dtype=int),
            use_predicted=np.asarray(cfg.hessian_diagnostics_use_predicted, dtype=bool),
        )

    with open(os.path.join(exp_dir, "hessian_diagnostics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return diagnostics


def compute_for_experiment(
    exp_dir: str,
    args,
) -> bool:
    config_path = os.path.join(exp_dir, "config.json")
    model_path = os.path.join(exp_dir, "model.eqx")
    if not os.path.isfile(config_path) or not os.path.isfile(model_path):
        print(f"[skip] {exp_dir}: missing config.json or model.eqx")
        return False

    with open(config_path) as f:
        payload = json.load(f)

    cfg = _sweep_config_from_payload(payload)
    cfg = cfg.__class__(
        **{
            **cfg.__dict__,
            "save_hessian_diagnostics": True,
            "hessian_diagnostics_use_predicted": bool(args.use_predicted),
            "hessian_diagnostics_splits": tuple(args.splits),
            "hessian_diagnostics_stride": int(args.stride),
            "hessian_diagnostics_max_trajectories": (
                None if args.all_trajectories else args.max_trajectories
            ),
            "prediction_fail_on_nonconvergence": bool(args.fail_on_nonconvergence),
        }
    )

    registry = build_architecture_registry()
    arch_name = payload.get("arch_name")
    if arch_name not in registry:
        raise ValueError(f"Unknown or missing arch_name in {config_path}: {arch_name}")
    spec = registry[arch_name]

    properties = _properties_from_payload(payload, fallback_class_name=args.properties_class)
    train_file = _resolve_data_path(payload, "train_file", args.train_file, exp_dir)
    valid_file = _resolve_data_path(payload, "valid_file", args.valid_file, exp_dir)

    params = make_model_params(cfg, spec)
    model = eqx.tree_deserialise_leaves(model_path, spec.model_cls(params))
    base, aux = get_slinky(properties)

    force_key = args.force_key if args.force_key is not None else cfg.force_key
    train_data = Dataset.load(train_file, force_key=force_key)
    valid_data = Dataset.load(valid_file, force_key=force_key)

    diagnostics = _compute_and_save_hessian_diagnostics(
        model=model,
        base=base,
        aux=aux,
        train_data=train_data,
        valid_data=valid_data,
        cfg=cfg,
        exp_dir=exp_dir,
    )
    _append_hessian_to_results(exp_dir, diagnostics)

    split_bits = []
    for split, diag in diagnostics.items():
        split_bits.append(
            f"{split}: M={float(diag['M_hat']):.3e}, "
            f"kappa={float(diag['kappa_hat']):.3e}, states={int(diag['n_states'])}"
        )
    print(f"[ok] {exp_dir} | " + " | ".join(split_bits))
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compute Hessian diagnostics for saved architecture sweep outputs."
    )
    parser.add_argument("output_dir", help="Directory containing saved architecture run folders.")
    parser.add_argument(
        "--use-predicted",
        action="store_true",
        help="Compute Hessians along model-predicted paths q_pred instead of saved data q.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        choices=["train", "valid"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=1,
        help="Number of trajectories per split to sample. Ignored with --all-trajectories.",
    )
    parser.add_argument(
        "--all-trajectories",
        action="store_true",
        help="Use every trajectory in each split.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Time-step stride for Hessian diagnostics.",
    )
    parser.add_argument("--train-file", default=None, help="Train dataset path for older configs.")
    parser.add_argument("--valid-file", default=None, help="Valid/test dataset path for older configs.")
    parser.add_argument(
        "--properties-class",
        default=None,
        help="Properties class name from properties.py for older configs, e.g. SlinkyN3Properties.",
    )
    parser.add_argument("--force-key", default=None, help="Optional force key to use while loading data.")
    parser.add_argument(
        "--fail-on-nonconvergence",
        action="store_true",
        help="Raise if q_pred solves do not converge when --use-predicted is set.",
    )
    parser.add_argument(
        "--summary-csv",
        default="hessian_diagnostics_table.csv",
        help=(
            "CSV filename to write under output_dir with one row per architecture/split. "
            "Use an empty string to skip."
        ),
    )
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)

    exp_dirs = _find_experiment_dirs(args.output_dir)
    if len(exp_dirs) == 0:
        raise FileNotFoundError(f"No config.json files found under {args.output_dir}")

    n_ok = 0
    for exp_dir in exp_dirs:
        try:
            n_ok += int(compute_for_experiment(exp_dir, args))
        except Exception as exc:
            print(f"[failed] {exp_dir}: {exc!r}")

    if args.summary_csv:
        write_hessian_summary_csv(args.output_dir, exp_dirs=exp_dirs, csv_name=args.summary_csv)

    print(f"Finished Hessian diagnostics for {n_ok}/{len(exp_dirs)} runs.")


if __name__ == "__main__":
    main()
