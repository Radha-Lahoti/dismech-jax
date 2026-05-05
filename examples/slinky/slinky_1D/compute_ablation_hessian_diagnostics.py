"""Post-process saved slinky_1D runs with Hessian path diagnostics.

This is the 1D analog of
``examples/slinky/slinky_2D/compute_architecture_hessian_diagnostics.py``.

Two on-disk layouts are recognised:

1. ``<output_dir>/cases/`` (e.g. ``ablation_outputs_new``).  Every case is
   stored as three files:

       <case_name>.json   summary + case config (family, hidden_sizes,
                          which_case, K_initial, weight_scale) and ``meta``
                          (rest_length, initial_last_node_x).
       <case_name>.eqx    trained ``equinox`` model leaves.
       <case_name>.npz    ``disps``, ``strains``, ``train_mask``,
                          ``test_mask``, ``force_truth``, etc.

2. ``<output_dir>/runs/`` (e.g. ``seed_envelope_two_models``).  Each run is
   stored as a pair of files; there is no per-run JSON:

       <case_name>__seed<NNN>.eqx
       <case_name>__seed<NNN>.npz   includes ``pulled_node_x``, ``strains``,
                                    ``train_mask``, ``test_mask``, ``seed``.

   The ``CaseConfig`` template is looked up by ``<case_name>`` from the
   per-file ablation registries (init-sensitivity list first, then the full
   ablation grid).  Geometry (``rest_length``, ``initial_last_node_x``) is
   recovered from the npz arrays via ``l_k = pulled_node_x / (1 + strain)``.

For every run we rebuild the model, reconstruct the ``Slinky1D`` system, and
walk along the prescribed-displacement path computing the Hessian of the
elastic energy with respect to the only free DOF (the right node).  Because
the free Hessian is 1x1 here, the eigenvalue is just the scalar value itself;
we still report the full set of metrics (``M_hat``, ``kappa_hat``,
``L_adj_hat``, ``sigma_*``, ``lambda_*``, ``negative_min_eig_fraction``) so
the output schema matches the 2D pipeline.

Examples:
    uv run python compute_ablation_hessian_diagnostics.py ablation_outputs_new
    uv run python compute_ablation_hessian_diagnostics.py seed_envelope_two_models
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ablation_config import CaseConfig, build_case_list, init_sensitivity_cases
from ablation_models import make_model
from slinky_1d_system import Slinky1D


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    """Everything needed to evaluate one trained model from disk."""
    display_name: str       # used for log lines and output filenames
    eqx_path: str
    npz_path: str
    case: CaseConfig
    meta: dict              # {"rest_length": ..., "initial_last_node_x": ...}
    out_dir: str            # where to save *_hessian_diagnostics.{json,npz}
    seed: Optional[int] = None


def _case_template_lookup() -> dict[str, CaseConfig]:
    """Map case_name -> CaseConfig; init-sensitivity entries win on conflict."""
    out: dict[str, CaseConfig] = {c.name: c for c in build_case_list()}
    out.update({c.name: c for c in init_sensitivity_cases()})
    return out


def _meta_from_npz_arrays(pulled_node_x: np.ndarray, strains: np.ndarray) -> dict:
    """Recover (rest_length, initial_last_node_x) from saved arrays.

    For ``Slinky1D`` with ``x_left = 0``: ``strain = q[1] / l_k - 1``.  Pick the
    sample whose denominator ``1 + strain`` is largest in magnitude (most
    numerically stable inversion) to estimate ``l_k``.
    """
    denom = 1.0 + np.asarray(strains, dtype=np.float64)
    if not np.any(np.abs(denom) > 1e-6):
        raise ValueError(
            "Cannot recover rest_length: every strain sample has 1+strain ~= 0."
        )
    i = int(np.argmax(np.abs(denom)))
    l_k = float(np.asarray(pulled_node_x, dtype=np.float64)[i] / denom[i])
    return {"rest_length": l_k, "initial_last_node_x": l_k}


def _meta_from_data_path(data_path: str) -> dict:
    """Recover (rest_length, initial_last_node_x) from the source experiment data.

    Used as a fallback when the per-run npz omits ``strains`` (e.g. runs
    produced by ``run_init_sensitivity_sweep.py``).
    """
    with np.load(data_path, allow_pickle=False) as data:
        if "initial_last_node_x" not in data.files:
            raise KeyError(
                f"{data_path}: missing 'initial_last_node_x' needed to recover rest_length"
            )
        l_k = float(np.asarray(data["initial_last_node_x"]))
    return {"rest_length": l_k, "initial_last_node_x": l_k}


def _case_from_payload(payload: dict) -> CaseConfig:
    case = payload["case"]
    return CaseConfig(
        name=case["name"],
        family=case["family"],
        hidden_sizes=tuple(case.get("hidden_sizes", ())),
        which_case=case["which_case"],
        K_initial=float(case.get("K_initial", 0.1)),
        weight_scale=float(case.get("weight_scale", 1.0)),
    )


def _discover_cases_layout(output_dir: str) -> list[RunRecord]:
    """Discover runs under ``<output_dir>/cases/`` (.json + .eqx + .npz)."""
    cases_dir = os.path.join(os.path.abspath(output_dir), "cases")
    if not os.path.isdir(cases_dir):
        return []

    json_paths = sorted(
        os.path.join(cases_dir, f)
        for f in os.listdir(cases_dir)
        if f.endswith(".json") and not f.endswith("_hessian_diagnostics.json")
    )

    records: list[RunRecord] = []
    for jp in json_paths:
        base = jp[:-len(".json")]
        eqx_path = base + ".eqx"
        npz_path = base + ".npz"
        if not (os.path.isfile(eqx_path) and os.path.isfile(npz_path)):
            print(f"[skip] {jp}: missing matching .eqx or .npz")
            continue
        with open(jp) as f:
            payload = json.load(f)
        case = _case_from_payload(payload)
        records.append(
            RunRecord(
                display_name=case.name,
                eqx_path=eqx_path,
                npz_path=npz_path,
                case=case,
                meta=payload["meta"],
                out_dir=cases_dir,
            )
        )
    return records


_RUN_FNAME_RE = re.compile(r"^(?P<name>.+?)__seed(?P<seed>\d+)\.eqx$")


def _discover_runs_layout(output_dir: str, data_path: Optional[str] = None) -> list[RunRecord]:
    """Discover runs under ``<output_dir>/runs/`` (.eqx + .npz, no .json).

    If a per-run npz is missing ``strains``, ``data_path`` is consulted to
    recover ``initial_last_node_x`` directly from the source experiment data.
    """
    runs_dir = os.path.join(os.path.abspath(output_dir), "runs")
    if not os.path.isdir(runs_dir):
        return []

    template_lookup = _case_template_lookup()
    out_dir = os.path.join(os.path.abspath(output_dir), "hessian_diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    fallback_meta: Optional[dict] = None

    records: list[RunRecord] = []
    eqx_files = sorted(f for f in os.listdir(runs_dir) if f.endswith(".eqx"))
    for fname in eqx_files:
        match = _RUN_FNAME_RE.match(fname)
        if not match:
            print(f"[skip] {fname}: does not match <case>__seed<NNN>.eqx pattern")
            continue
        case_name = match.group("name")
        seed_int = int(match.group("seed"))
        base = os.path.join(runs_dir, fname[:-len(".eqx")])
        eqx_path = base + ".eqx"
        npz_path = base + ".npz"
        if not os.path.isfile(npz_path):
            print(f"[skip] {fname}: missing matching .npz")
            continue
        if case_name not in template_lookup:
            print(
                f"[skip] {fname}: unknown case_name {case_name!r} "
                f"(not in build_case_list / init_sensitivity_cases)"
            )
            continue

        with np.load(npz_path, allow_pickle=False) as data:
            if "pulled_node_x" not in data.files:
                print(f"[skip] {fname}: npz missing pulled_node_x")
                continue
            if "strains" in data.files:
                meta = _meta_from_npz_arrays(
                    np.asarray(data["pulled_node_x"]),
                    np.asarray(data["strains"]),
                )
            elif data_path is not None and os.path.isfile(data_path):
                if fallback_meta is None:
                    fallback_meta = _meta_from_data_path(data_path)
                    print(
                        f"[info] {runs_dir}: npz files lack 'strains'; "
                        f"using rest_length={fallback_meta['rest_length']:.6f} "
                        f"from {data_path}"
                    )
                meta = fallback_meta
            else:
                print(
                    f"[skip] {fname}: npz missing 'strains' and no usable "
                    f"--data-path fallback (tried {data_path!r})"
                )
                continue

        records.append(
            RunRecord(
                display_name=f"{case_name}__seed{seed_int:03d}",
                eqx_path=eqx_path,
                npz_path=npz_path,
                case=template_lookup[case_name],
                meta=meta,
                out_dir=out_dir,
                seed=seed_int,
            )
        )
    return records


def discover_runs(output_dir: str, layout: str, data_path: Optional[str] = None) -> list[RunRecord]:
    if layout == "cases":
        return _discover_cases_layout(output_dir)
    if layout == "runs":
        return _discover_runs_layout(output_dir, data_path=data_path)
    if layout == "auto":
        cases = _discover_cases_layout(output_dir)
        if cases:
            return cases
        return _discover_runs_layout(output_dir, data_path=data_path)
    raise ValueError(f"Unknown layout {layout!r}")


# ---------------------------------------------------------------------------
# Slinky / Hessian
# ---------------------------------------------------------------------------


def _slinky_from_meta(meta: dict) -> tuple[Slinky1D, jnp.ndarray]:
    initial_last_node_x = float(meta["initial_last_node_x"])
    rest_length = float(meta["rest_length"])
    x_left = 0.0
    q0 = jnp.array([x_left, initial_last_node_x])
    slinky = Slinky1D(l_k=jnp.array(rest_length), x_left=jnp.array(x_left))
    return slinky, q0


def _free_hessian_block(slinky: Slinky1D, model, disp: jnp.ndarray, q: jnp.ndarray) -> np.ndarray:
    """Hessian of E wrt the free DOF (the right node).

    For ``Slinky1D`` the left node is fixed at ``x_left`` and the right node is
    prescribed by ``disp`` via :meth:`get_q`; so along this path there is one
    free DOF (``q[1]``).  We differentiate :meth:`get_E` wrt ``q[1]`` only.
    """
    def E_of_q1(q1):
        q_full = q.at[1].set(q1)
        return slinky.get_E(disp, q_full, model, None)

    H_scalar = jax.grad(jax.grad(E_of_q1))(q[1])
    return np.asarray(H_scalar).reshape(1, 1)


def _hessian_path_diagnostics(
    slinky: Slinky1D,
    q0: jnp.ndarray,
    model,
    disps: np.ndarray,
    sort_by_disp: bool = True,
    eps: float = 1e-12,
    min_distance: float = 1e-12,
) -> dict:
    if sort_by_disp:
        order = np.argsort(np.asarray(disps))
        disps = np.asarray(disps)[order]

    min_abs_eigs = []
    min_eigs = []
    max_eigs = []
    max_abs_eigs = []
    condition_numbers = []
    l_adj_values = []
    n_negative = 0
    n_states = 0

    prev_H = None
    prev_q = None
    for d in disps:
        d_jax = jnp.asarray(d)
        q = slinky.get_q(d_jax, q0)
        H_free = _free_hessian_block(slinky, model, d_jax, q)
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

        q_np = np.asarray(q)
        if prev_H is not None:
            dq = float(np.linalg.norm(q_np[1:] - prev_q[1:]))  # free DOF only
            if dq > min_distance:
                dH_norm = float(np.linalg.norm(H_free - prev_H, ord=2))
                l_adj_values.append(dH_norm / dq)

        prev_H = H_free
        prev_q = q_np

    if n_states == 0:
        raise ValueError("No states found for Hessian diagnostics.")

    min_abs_eigs = np.asarray(min_abs_eigs)
    min_eigs = np.asarray(min_eigs)
    max_eigs = np.asarray(max_eigs)
    max_abs_eigs = np.asarray(max_abs_eigs)
    condition_numbers = np.asarray(condition_numbers)
    l_adj_values = np.asarray(l_adj_values)

    sigma_min = float(np.min(min_abs_eigs))
    l_adj_hat = float(np.max(l_adj_values)) if l_adj_values.size else float("nan")

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


def _diag_summary(diag: dict) -> dict:
    return {
        k: (v.item() if isinstance(v, np.ndarray) and v.shape == () else v)
        for k, v in diag.items()
        if not isinstance(v, np.ndarray) or v.ndim == 0
    }


def _diag_save_dict(diag: dict) -> dict:
    return {k: np.asarray(v) for k, v in diag.items()}


def _disps_from_npz(npz_path: str) -> np.ndarray:
    """Return per-sample displacement, regardless of which key the file uses."""
    with np.load(npz_path, allow_pickle=False) as data:
        if "disps" in data.files:
            return np.asarray(data["disps"])
        if "pulled_node_x" in data.files:
            return np.asarray(data["pulled_node_x"])
    raise KeyError(f"{npz_path}: neither 'disps' nor 'pulled_node_x' present")


# ---------------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------------


def compute_for_run(
    record: RunRecord,
    splits: list[str],
    template_seed: int,
    sort_by_disp: bool,
) -> dict[str, dict]:
    slinky, q0 = _slinky_from_meta(record.meta)
    template = make_model(record.case, jax.random.PRNGKey(template_seed))
    model = eqx.tree_deserialise_leaves(record.eqx_path, template)

    with np.load(record.npz_path, allow_pickle=False) as data:
        train_mask = np.asarray(data["train_mask"], dtype=bool)
        test_mask = np.asarray(data["test_mask"], dtype=bool)
    disps = _disps_from_npz(record.npz_path)

    split_indices = {
        "train": train_mask,
        "test": test_mask,
        "all": np.ones_like(train_mask, dtype=bool),
    }

    diagnostics: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    for split in splits:
        if split not in split_indices:
            raise ValueError(
                f"Unknown split {split!r}; expected one of {tuple(split_indices)}."
            )
        mask = split_indices[split]
        if not bool(mask.any()):
            print(f"[warn] {record.display_name}: split={split!r} is empty, skipping")
            continue
        diag = _hessian_path_diagnostics(
            slinky=slinky,
            q0=q0,
            model=model,
            disps=disps[mask],
            sort_by_disp=sort_by_disp,
        )
        diagnostics[split] = diag
        summary[split] = _diag_summary(diag)
        summary[split]["n_path_points"] = int(mask.sum())

    diag_npz_path = os.path.join(
        record.out_dir, f"{record.display_name}_hessian_diagnostics.npz"
    )
    save_dict = {}
    for split, diag in diagnostics.items():
        for key, value in _diag_save_dict(diag).items():
            save_dict[f"{split}_{key}"] = value
    np.savez(diag_npz_path, **save_dict)

    summary_path = os.path.join(
        record.out_dir, f"{record.display_name}_hessian_diagnostics.json"
    )
    payload = {"case": asdict(record.case), "splits": summary}
    if record.seed is not None:
        payload["seed"] = int(record.seed)
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    parts = [
        f"{split}: M={float(d['M_hat']):.3e}, "
        f"kappa={float(d['kappa_hat']):.3e}, "
        f"L_adj={float(d['L_adj_hat']):.3e}, "
        f"states={int(d['n_states'])}"
        for split, d in diagnostics.items()
    ]
    print(f"[ok] {record.display_name} | " + " | ".join(parts))
    return diagnostics


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------


def write_summary_csv(
    output_dir: str,
    records: list[RunRecord],
    csv_name: str,
) -> str:
    rows = []
    for rec in records:
        summary_path = os.path.join(
            rec.out_dir, f"{rec.display_name}_hessian_diagnostics.json"
        )
        if not os.path.isfile(summary_path):
            continue
        with open(summary_path) as f:
            payload = json.load(f)
        case = payload.get("case", {})
        for split, split_summary in payload.get("splits", {}).items():
            rows.append(
                {
                    "display_name": rec.display_name,
                    "case_name": case.get("name", ""),
                    "seed": payload.get("seed", "") if rec.seed is not None else "",
                    "family": case.get("family", ""),
                    "which_case": case.get("which_case", ""),
                    "hidden_sizes": "x".join(str(h) for h in case.get("hidden_sizes", []))
                                    or "0",
                    "split": split,
                    "M_hat": split_summary.get("M_hat", float("nan")),
                    "kappa_hat": split_summary.get("kappa_hat", float("nan")),
                    "L_adj_hat": split_summary.get("L_adj_hat", float("nan")),
                    "sigma_min_min": split_summary.get("sigma_min_min", float("nan")),
                    "sigma_max_max": split_summary.get("sigma_max_max", float("nan")),
                    "lambda_min_min": split_summary.get("lambda_min_min", float("nan")),
                    "lambda_max_max": split_summary.get("lambda_max_max", float("nan")),
                    "negative_min_eig_fraction": split_summary.get(
                        "negative_min_eig_fraction", float("nan")
                    ),
                    "n_states": split_summary.get("n_states", 0),
                    "n_adjacent_pairs": split_summary.get("n_adjacent_pairs", 0),
                    "n_path_points": split_summary.get("n_path_points", 0),
                    "summary_path": summary_path,
                }
            )

    csv_path = os.path.join(os.path.abspath(output_dir), csv_name)
    fieldnames = [
        "display_name",
        "case_name",
        "seed",
        "family",
        "which_case",
        "hidden_sizes",
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
        "n_path_points",
        "summary_path",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote Hessian diagnostics table with {len(rows)} rows to {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute Hessian path diagnostics for slinky_1D outputs."
    )
    parser.add_argument(
        "output_dir",
        help="Directory containing 'cases/' or 'runs/' subfolder.",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "cases", "runs"],
        default="auto",
        help="Disk layout to scan. 'auto' tries cases/ first, then runs/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test", "all"],
        help="Which subsets of the prescribed-displacement path to evaluate.",
    )
    parser.add_argument(
        "--template-seed",
        type=int,
        default=42,
        help="PRNG seed used to build the model template before deserialising weights. "
             "Architecture topology depends only on the case config, so any seed works.",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Walk the path in the order disps are saved instead of sorting by disp value.",
    )
    parser.add_argument(
        "--summary-csv",
        default="hessian_diagnostics_table.csv",
        help="CSV filename written under output_dir. Pass an empty string to skip.",
    )
    parser.add_argument(
        "--enable-x64",
        action="store_true",
        help="Enable JAX float64. Off by default to match the float32 dtype the 1D "
             "ablation models were trained with; turning it on upcasts everything to "
             "float64 and will fail to deserialise float32 .eqx files.",
    )
    parser.add_argument(
        "--data-path",
        default="experiment_data/pulling_phase_data.npz",
        help="Source experiment data npz used to recover rest_length when per-run "
             "npz files lack a 'strains' array (e.g. runs from "
             "run_init_sensitivity_sweep.py). Only consulted in the runs/ layout.",
    )
    args = parser.parse_args()

    if args.enable_x64:
        jax.config.update("jax_enable_x64", True)

    records = discover_runs(args.output_dir, args.layout, data_path=args.data_path)
    if not records:
        raise FileNotFoundError(
            f"No usable runs found under {args.output_dir} for layout={args.layout}"
        )
    print(f"Discovered {len(records)} runs (layout resolved from {args.layout!r}).")

    n_ok = 0
    for rec in records:
        try:
            compute_for_run(
                record=rec,
                splits=args.splits,
                template_seed=args.template_seed,
                sort_by_disp=not args.no_sort,
            )
            n_ok += 1
        except Exception as exc:
            print(f"[failed] {rec.display_name}: {exc!r}")

    if args.summary_csv:
        write_summary_csv(args.output_dir, records, args.summary_csv)

    print(f"Finished Hessian diagnostics for {n_ok}/{len(records)} runs.")


if __name__ == "__main__":
    main()
