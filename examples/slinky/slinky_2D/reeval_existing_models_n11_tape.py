"""Re-evaluate models trained by run_hessian_reg_max_dlambda_experiment.py
on a new (smaller-resolution, but extended) max_dlambda grid.

This script does NOT retrain. It walks an existing output directory, finds
each (arch, seed, hreg) combo that already has a saved model.eqx and matches
the requested filter, deserialises the model, and runs only the post-training
`evaluate_stable_steps` pass on the new grid. Records are written into a new
output folder using the same progress.jsonl / summary.csv schema produced by
the original experiment runner, so cell 4 of the notebook can read them.

Usage:
    uv run python examples/slinky/slinky_2D/reeval_existing_models_n11_tape.py

Run after stopping the training notebook (to avoid GPU/CPU contention).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from run_architectures import (  # noqa: E402
    _properties_from_payload,
    _resolve_saved_data_path,
    _sweep_config_from_payload,
    build_architecture_registry,
    experiment_name,
    make_model_params,
)
from run_hessian_reg_max_dlambda_experiment import (  # noqa: E402
    _is_finite_tree,
    _jsonable,
    append_jsonl,
    evaluate_stable_steps,
    write_summary_csv,
)
from util import Dataset  # noqa: E402


# ---------------- user config ----------------
OLD_OUTPUT_DIR = Path(
    "/Users/radha/GitRepos/dismech-jax/examples/slinky/slinky_2D/"
    "hessian_reg_max_dlambda_outputs_n11_tape"
)
NEW_OUTPUT_DIR = Path(
    "/Users/radha/GitRepos/dismech-jax/examples/slinky/slinky_2D/"
    "hessian_reg_max_dlambda_outputs_n11_tape_reeval"
)

WANTED_ARCHS = (
    "diag_energy_baseline",
    "diag_energy_mlp",
    "diag_energy_icnn",
    "icnn_energy",
)
WANTED_HREGS = (0.0, 1e-8, 1e-6, 1e-4)
NEW_EVAL_MAX_DLAMBDA_VALUES = (1e-3, 1e-2, 1e-1, 1.0)

EVAL_ITERS = 8  # matches the CLI default in run_hessian_reg_max_dlambda_experiment
# Hessian summary (kappa_hat, M_hat, ...) is reused from the original
# post_train_step_eval.json — it depends only on the trained model and
# validation data, not on the eval dlambda grid, so recomputing is redundant.
# ---------------------------------------------


def _hreg_match(actual: float, wanted: tuple[float, ...]) -> bool:
    for w in wanted:
        if w == 0.0 and actual == 0.0:
            return True
        if w != 0.0 and actual != 0.0 and abs(actual - w) / w < 1e-6:
            return True
    return False


def discover_runs(old_dir: Path) -> list[dict]:
    runs = []
    for subdir in sorted(old_dir.iterdir()):
        if not subdir.is_dir():
            continue
        config_path = subdir / "config.json"
        model_path = subdir / "model.eqx"
        if not config_path.is_file() or not model_path.is_file():
            continue
        with open(config_path) as f:
            payload = json.load(f)
        arch_name = payload.get("arch_name")
        hreg = float(payload.get("hessian_reg_strength", 0.0))
        seed = int(payload.get("seed", 0))
        if arch_name not in WANTED_ARCHS:
            continue
        if not _hreg_match(hreg, WANTED_HREGS):
            continue
        runs.append(
            {
                "exp_dir": str(subdir),
                "model_path": str(model_path),
                "payload": payload,
                "arch_name": arch_name,
                "seed": seed,
                "hreg": hreg,
            }
        )
    return runs


def reevaluate(run: dict, valid_data_cache: dict, registry: dict) -> dict:
    payload = run["payload"]
    cfg = _sweep_config_from_payload(payload)
    spec = registry[run["arch_name"]]
    properties = _properties_from_payload(payload)

    valid_file = _resolve_saved_data_path(payload, "valid_file", run["exp_dir"])
    if valid_file not in valid_data_cache:
        valid_data_cache[valid_file] = Dataset.load(
            valid_file, force_key=payload.get("force_key")
        )
    valid_data = valid_data_cache[valid_file]

    params = make_model_params(cfg, spec)
    model = eqx.tree_deserialise_leaves(run["model_path"], spec.model_cls(params))

    prior_record = {}
    prior_path = Path(run["exp_dir"]) / "post_train_step_eval.json"
    if prior_path.is_file():
        with open(prior_path) as f:
            prior_record = json.load(f)

    record = {
        "arch_name": run["arch_name"],
        "seed": int(run["seed"]),
        "hessian_reg_strength": float(run["hreg"]),
        "train_max_dlambda": float(cfg.max_dlambda),
        "train_success": True,
        "failure_reason": "",
        "exp_dir": run["exp_dir"],
        "exp_name": experiment_name(spec, cfg),
        "config": asdict(cfg),
        "seconds_train_plus_eval": 0.0,
        "largest_stable_eval_max_dlambda": None,
        "eval_by_max_dlambda": [],
        "hessian_summary": prior_record.get("hessian_summary", {}),
        "inferred_by_monotonicity": False,
        "final_train_loss": prior_record.get("final_train_loss", float("nan")),
        "final_valid_loss": prior_record.get("final_valid_loss", float("nan")),
        "reevaluated_from_saved_model": True,
        "source_exp_dir": run["exp_dir"],
    }

    if not _is_finite_tree(model):
        record["train_success"] = False
        record["failure_reason"] = "nonfinite_model_parameters_at_load"
        return record

    started = time.time()
    largest, eval_records = evaluate_stable_steps(
        model,
        properties,
        valid_data,
        NEW_EVAL_MAX_DLAMBDA_VALUES,
        iters=EVAL_ITERS,
        ls_steps=int(cfg.ls_steps),
        abs_tol=float(cfg.abs_tol),
        rel_tol=float(cfg.rel_tol),
        early_stop=bool(cfg.early_stop),
    )
    record["largest_stable_eval_max_dlambda"] = largest
    record["eval_by_max_dlambda"] = eval_records
    record["seconds_train_plus_eval"] = round(time.time() - started, 3)

    return record


def main() -> None:
    if not OLD_OUTPUT_DIR.is_dir():
        raise SystemExit(f"OLD_OUTPUT_DIR not found: {OLD_OUTPUT_DIR}")

    NEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    new_jsonl = NEW_OUTPUT_DIR / "progress.jsonl"
    new_csv = NEW_OUTPUT_DIR / "summary.csv"

    if new_jsonl.exists():
        raise SystemExit(
            f"Output already exists at {new_jsonl}. Move or remove it first "
            f"to avoid mixing reruns."
        )

    runs = discover_runs(OLD_OUTPUT_DIR)
    print(f"Found {len(runs)} (arch, seed, hreg) combos to re-evaluate:")
    for r in runs:
        print(f"  {r['arch_name']:24s} seed={r['seed']} hreg={r['hreg']:g}")
    if not runs:
        print("Nothing to do.")
        return

    registry = build_architecture_registry()
    valid_data_cache: dict = {}
    records: list[dict] = []
    for i, run in enumerate(runs, 1):
        print(
            f"\n[{i}/{len(runs)}] arch={run['arch_name']} seed={run['seed']} "
            f"hreg={run['hreg']:g}",
            flush=True,
        )
        record = reevaluate(run, valid_data_cache, registry)
        records.append(record)
        append_jsonl(str(new_jsonl), record)
        write_summary_csv(records, str(new_csv))
        print(
            f"  -> largest_stable_eval_max_dlambda="
            f"{record['largest_stable_eval_max_dlambda']} "
            f"({record['seconds_train_plus_eval']:.1f}s)"
        )

    with open(NEW_OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(_jsonable(records), f, indent=2)
    write_summary_csv(records, str(new_csv))
    print(f"\nDone. Wrote {len(records)} records to {NEW_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
