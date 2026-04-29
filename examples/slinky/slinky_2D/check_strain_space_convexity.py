"""Check learned-energy convexity in strain space for a saved architecture run.

This script loads a saved ``model.eqx`` and ``config.json``, gathers reduced
strains from all valid train/test trajectory states, computes

    eigvals(d^2 model(eps) / d eps^2),

and prints the same convexity report used in the accompanying analysis.

By default it targets the N11 tape Brazier-Cholesky ICNN run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_EXPERIMENT_DIR = (
    SCRIPT_DIR
    / "arch_sweep_outputs_n11_tape_all_architectures_train_fail_on_nonconvergence_True"
    / (
        "brazier_chol_stiffness_icnn__hid_10__inp_raw__stretchNN_0__bendNN_0"
        "__act_tanh__corr_0.01__zr1__seed_42__mdl_0.05__it_10__hreg_1e-06"
        "__hprobe_1__hseed_0__wd_1e-05__mode_anisotropic"
    )
)


def _prepare_imports() -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _sweep_config_from_payload(payload):
    from run_architectures import SweepConfig

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


def _properties_from_payload(payload):
    import properties as properties_module

    prop_payload = payload["properties"]
    prop_cls = getattr(properties_module, prop_payload["class_name"])
    valid_fields = {f.name for f in fields(prop_cls)}
    kwargs = {k: v for k, v in prop_payload.items() if k in valid_fields}
    return prop_cls(**kwargs)


def _load_model_and_context(exp_dir: Path):
    from run_architectures import build_architecture_registry, make_model_params
    from util import get_slinky

    config_path = exp_dir / "config.json"
    model_path = exp_dir / "model.eqx"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model.eqx: {model_path}")

    with config_path.open() as f:
        payload = json.load(f)

    cfg = _sweep_config_from_payload(payload)
    registry = build_architecture_registry()
    arch_name = payload.get("arch_name")
    if arch_name not in registry:
        raise ValueError(f"Unknown arch_name in {config_path}: {arch_name!r}")

    spec = registry[arch_name]
    params = make_model_params(cfg, spec)
    model = eqx.tree_deserialise_leaves(model_path, spec.model_cls(params))
    base, aux0 = get_slinky(_properties_from_payload(payload))
    return payload, cfg, spec, model, base, aux0


def _strain_component_indices(payload: dict, full_strain_dim: int) -> tuple[list[int], list[str]]:
    """Return the strain components the model actually sees in reduced space."""
    model_cls = payload.get("model_cls", "")
    if model_cls in {
        "StructuredBrazierCholeskyEnergyNN",
        "StructuredBrazierDiagonalEnergyNN",
    }:
        if full_strain_dim >= 4:
            return [0, 1, 3], ["eps0", "eps1", "kappa2"]
        return [0, 1, 2], ["eps0", "eps1", "kappa1"]

    return list(range(full_strain_dim)), [f"strain[{i}]" for i in range(full_strain_dim)]


def _collect_reduced_strains(data, split, source, cfg, model, base, aux0, component_indices):
    import dismech_jax as djx
    from dismech_jax.solver import update_aux_state

    qs_all = np.asarray(data.qs)
    xb_all = np.asarray(data.xb)
    valid_all = np.asarray(data.valid, dtype=bool)
    if data.lambdas.ndim == 1:
        lam_all = np.broadcast_to(np.asarray(data.lambdas), valid_all.shape)
    else:
        lam_all = np.asarray(data.lambdas)

    if data.idx_b.ndim == 1:
        idx_all = np.broadcast_to(
            np.asarray(data.idx_b), (qs_all.shape[0], data.idx_b.shape[0])
        )
    else:
        idx_all = np.asarray(data.idx_b)

    strains = []
    meta = []
    component_indices = np.asarray(component_indices, dtype=int)

    for traj_idx in range(qs_all.shape[0]):
        valid_idx = np.where(valid_all[traj_idx])[0]
        if valid_idx.size == 0:
            continue

        qs = qs_all[traj_idx, valid_idx]
        xb = xb_all[traj_idx, valid_idx]
        lambdas = lam_all[traj_idx, valid_idx]
        rod = base.with_bc(
            djx.DirectBC(
                idx_b=jnp.asarray(idx_all[traj_idx]),
                xb=jnp.asarray(xb),
                lambdas=jnp.asarray(lambdas),
            )
        )

        if source == "pred":
            qs = np.asarray(
                rod.solve(
                    model,
                    jnp.asarray(lambdas),
                    aux0,
                    max_dlambda=cfg.max_dlambda,
                    iters=cfg.iters,
                    ls_steps=cfg.ls_steps,
                    abs_tol=cfg.abs_tol,
                    rel_tol=cfg.rel_tol,
                    fail_on_nonconvergence=False,
                    early_stop=cfg.early_stop,
                )
            )

        for state_idx, (_lambda, q) in enumerate(zip(lambdas, qs)):
            aux_q = update_aux_state(aux0, jnp.asarray(q), rod)
            full_strains = np.asarray(rod.get_del_strains(jnp.asarray(q), aux_q))
            reduced = full_strains[:, component_indices]
            for triplet_idx, eps in enumerate(reduced):
                strains.append(eps)
                meta.append(
                    (
                        split,
                        source,
                        traj_idx,
                        state_idx,
                        triplet_idx,
                        float(_lambda),
                    )
                )

    return np.asarray(strains), meta


def _format_vector(values) -> str:
    return "[" + ", ".join(f"{float(v):.10f}" for v in values) + "]"


def _print_split_report(split, source, strains, meta, min_eigs):
    loc = int(np.argmin(min_eigs))
    neg = min_eigs < -1e-9
    print(
        f"{split}/{source}: n={strains.shape[0]} "
        f"range_min={_format_vector(strains.min(axis=0))} "
        f"range_max={_format_vector(strains.max(axis=0))} "
        f"min_eig={float(min_eigs[loc]):.10f} "
        f"neg_count={int(np.sum(neg))} "
        f"neg_fraction={float(np.mean(neg)):.10f} "
        f"worst_eps={_format_vector(strains[loc])} "
        f"worst_meta={meta[loc]}"
    )


def run_report(exp_dir: Path, grid_n: int, include_predicted: bool) -> None:
    from util import Dataset

    payload, cfg, _spec, model, base, aux0 = _load_model_and_context(exp_dir)

    # Infer full strain dimensionality from the first valid train state.
    train_probe = Dataset.load(payload["train_file"], force_key=cfg.force_key)
    full_dim = int(train_probe.qs.shape[-1])
    del full_dim

    # The rod strain dimension is not the q dimension; inspect one strain sample.
    import dismech_jax as djx
    from dismech_jax.solver import update_aux_state

    valid_idx = np.where(np.asarray(train_probe.valid[0], dtype=bool))[0][0]
    lambdas = (
        np.asarray(train_probe.lambdas)
        if train_probe.lambdas.ndim == 1
        else np.asarray(train_probe.lambdas[0])
    )
    rod = base.with_bc(
        djx.DirectBC(
            idx_b=jnp.asarray(train_probe.idx_b),
            xb=jnp.asarray(train_probe.xb[0, np.asarray(train_probe.valid[0], dtype=bool)]),
            lambdas=jnp.asarray(lambdas[np.asarray(train_probe.valid[0], dtype=bool)]),
        )
    )
    q_probe = jnp.asarray(train_probe.qs[0, valid_idx])
    full_strain_dim = int(rod.get_del_strains(q_probe, update_aux_state(aux0, q_probe, rod)).shape[-1])
    component_indices, component_labels = _strain_component_indices(payload, full_strain_dim)

    @jax.jit
    def hessian_eigs(eps):
        hessian = jax.hessian(lambda x: model(x))(eps)
        hessian = 0.5 * (hessian + hessian.T)
        return jnp.linalg.eigvalsh(hessian)

    batched_hessian_eigs = jax.jit(jax.vmap(hessian_eigs))

    print(f"experiment_dir = {exp_dir}")
    print(f"arch_name = {payload.get('arch_name')}")
    print(
        "reduced_strain_components = "
        f"{component_labels} from full strain indices {component_indices}"
    )
    print()

    blocks = []
    metas = []
    sources = ["true", "pred"] if include_predicted else ["true"]
    for split, path_key in [("train", "train_file"), ("valid", "valid_file")]:
        data = Dataset.load(payload[path_key], force_key=cfg.force_key)
        print(
            f"{split}: qs={tuple(data.qs.shape)} "
            f"valid_count={int(np.asarray(data.valid).sum())} "
            f"trajectories={data.qs.shape[0]} T={data.qs.shape[1]}"
        )
        for source in sources:
            strains, meta = _collect_reduced_strains(
                data,
                split,
                source,
                cfg,
                model,
                base,
                aux0,
                component_indices,
            )
            eigs = np.asarray(batched_hessian_eigs(jnp.asarray(strains)))
            min_eigs = eigs[:, 0]
            _print_split_report(split, source, strains, meta, min_eigs)
            blocks.append(strains)
            metas.extend(meta)

    combined = np.concatenate(blocks, axis=0)
    combined_eigs = np.asarray(batched_hessian_eigs(jnp.asarray(combined)))
    combined_min_eigs = combined_eigs[:, 0]
    loc = int(np.argmin(combined_min_eigs))
    neg = combined_min_eigs < -1e-9

    print()
    print(f"combined: n={combined.shape[0]}")
    print(
        f"combined_range_min {component_labels} = "
        f"{_format_vector(combined.min(axis=0))}"
    )
    print(
        f"combined_range_max {component_labels} = "
        f"{_format_vector(combined.max(axis=0))}"
    )
    print(
        "combined_trajectory_min_eig = "
        f"{float(combined_min_eigs[loc]):.10f} "
        f"worst_eps={_format_vector(combined[loc])} "
        f"worst_meta={metas[loc]}"
    )
    print(
        "combined_trajectory_negative_count = "
        f"{int(np.sum(neg))} fraction={float(np.mean(neg)):.10f}"
    )

    mins = combined.min(axis=0)
    maxs = combined.max(axis=0)
    grid = np.stack(
        np.meshgrid(
            *[np.linspace(mins[i], maxs[i], grid_n) for i in range(combined.shape[1])],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, combined.shape[1])

    best = np.inf
    best_eps = None
    n_negative = 0
    batch_size = 8192
    for start in range(0, grid.shape[0], batch_size):
        eigs = np.asarray(batched_hessian_eigs(jnp.asarray(grid[start : start + batch_size])))
        min_eigs = eigs[:, 0]
        batch_loc = int(np.argmin(min_eigs))
        if float(min_eigs[batch_loc]) < best:
            best = float(min_eigs[batch_loc])
            best_eps = grid[start + batch_loc]
        n_negative += int(np.sum(min_eigs < -1e-9))

    print(
        f"grid{grid_n}: total={grid.shape[0]} "
        f"min_eig={best:.10f} "
        f"worst_eps={_format_vector(best_eps)} "
        f"neg_count={n_negative} fraction={n_negative / grid.shape[0]:.10f}"
    )

    if hasattr(model, "get_K0"):
        print(f"K0 = {_format_vector(np.asarray(model.get_K0()))}")
    if hasattr(model, "get_brazier_params"):
        params = [float(x) for x in model.get_brazier_params()]
        print(f"brazier_params = {_format_vector(params)}")

    print()
    if np.any(neg):
        print("Conclusion: not convex in strain space over the checked trajectory range.")
    else:
        print("Conclusion: no negative strain-space Hessian eigenvalues found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report strain-space Hessian convexity for a saved learned-energy model."
    )
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        default=str(DEFAULT_EXPERIMENT_DIR),
        help="Directory containing config.json and model.eqx.",
    )
    parser.add_argument(
        "--grid-n",
        type=int,
        default=51,
        help="Number of grid points per reduced strain dimension for the bounding-box sweep.",
    )
    parser.add_argument(
        "--true-only",
        action="store_true",
        help="Skip predicted trajectories and only check saved dataset trajectories.",
    )
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    _prepare_imports()
    run_report(
        exp_dir=Path(args.experiment_dir).expanduser().resolve(),
        grid_n=args.grid_n,
        include_predicted=not args.true_only,
    )


if __name__ == "__main__":
    main()
