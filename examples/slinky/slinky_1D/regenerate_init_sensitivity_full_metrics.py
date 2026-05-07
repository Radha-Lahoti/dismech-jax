"""
Regenerate full-metric data and plots for initialization-sensitivity runs.

The init-sensitivity training scripts save trained ``.eqx`` models plus force
prediction/loss arrays, but older outputs may not include strain-based energy
and stiffness arrays. This script loads the saved models, computes the missing
strain metrics, writes them back into each run ``.npz``, and regenerates the
figures without retraining.

Example:
    uv run python regenerate_init_sensitivity_full_metrics.py seed_envelope_two_models
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import equinox as eqx
import jax
import numpy as np

from ablation_config import CaseConfig, build_case_list, init_sensitivity_cases
from ablation_data import load_problem
from ablation_io import ensure_dir
from ablation_models import make_model
from ablation_plots_init import plot_all_init_figures
from ablation_predict import predict_effective_stiffness, predict_energy, summary_sharpness


SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_output_root(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _runs_dir(output_root: Path) -> Path:
    if output_root.name == "runs":
        return output_root
    return output_root / "runs"


def _case_by_name() -> Dict[str, CaseConfig]:
    out: Dict[str, CaseConfig] = {case.name: case for case in build_case_list()}
    out.update({case.name: case for case in init_sensitivity_cases()})
    return out


def _load_npz_dict(path: Path) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _seed_from_run_data(run_data: dict, npz_path: Path) -> int:
    if "seed" in run_data:
        return int(np.asarray(run_data["seed"]).item())

    stem = npz_path.stem
    if "__seed" not in stem:
        raise ValueError(f"Cannot infer seed from {npz_path.name}")
    return int(stem.rsplit("__seed", 1)[1])


def _case_name_from_run_path(npz_path: Path) -> str:
    stem = npz_path.stem
    if "__seed" not in stem:
        raise ValueError(f"Cannot infer case name from {npz_path.name}")
    return stem.rsplit("__seed", 1)[0]


def _load_model(eqx_path: Path, case: CaseConfig, seed: int):
    template = make_model(case, jax.random.PRNGKey(seed))
    return eqx.tree_deserialise_leaves(str(eqx_path), template)


def _save_enriched_npz(path: Path, run_data: dict, enriched: dict) -> None:
    merged = dict(run_data)
    merged.update(enriched)
    np.savez(path, **merged)


def _result_for_plotting(case: CaseConfig, run_data: dict, problem: dict, seed: int) -> dict:
    train_mse = float(np.asarray(run_data["train_mse"]).item())
    test_mse = float(np.asarray(run_data["test_mse"]).item())
    result = {
        "case": asdict(case),
        "seed": seed,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "generalization_gap": float(np.asarray(run_data.get("generalization_gap", test_mse - train_mse)).item()),
        "stiffness_sharpness": float(np.asarray(run_data["stiffness_sharpness"]).item()),
        "pulled_node_x": np.asarray(run_data["pulled_node_x"]),
        "force_truth": np.asarray(run_data["force_truth"]),
        "pred_force": np.asarray(run_data["pred_force"]),
        "train_mask": np.asarray(run_data["train_mask"]).astype(bool),
        "test_mask": np.asarray(run_data["test_mask"]).astype(bool),
        "train_hist": np.asarray(run_data["train_hist"]),
        "test_hist": np.asarray(run_data["test_hist"]),
        "strains": np.asarray(run_data["strains"]),
        "pred_energy": np.asarray(run_data["pred_energy"]),
        "pred_stiffness": np.asarray(run_data["pred_stiffness"]),
        "meta": problem["meta"],
    }
    return result


def regenerate_full_metrics(output_root: Path, data_path: Path) -> None:
    runs_dir = _runs_dir(output_root)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Could not find runs directory: {runs_dir}")

    if runs_dir.name == "runs":
        root_dir = runs_dir.parent
    else:
        root_dir = output_root
    fig_dir = root_dir / "figures"
    ensure_dir(str(fig_dir))

    problem = load_problem(data_path=str(data_path), test_range=(0.2, 0.8))
    strains = problem["strains"]
    cases = _case_by_name()
    results_by_case: Dict[str, List[dict]] = {name: [] for name in cases}

    npz_paths = sorted(runs_dir.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz run files found in {runs_dir}")

    for npz_path in npz_paths:
        case_name = _case_name_from_run_path(npz_path)
        if case_name not in cases:
            print(f"Skipping {npz_path.name}: case is not in init_sensitivity_cases()")
            continue

        run_data = _load_npz_dict(npz_path)
        seed = _seed_from_run_data(run_data, npz_path)
        eqx_path = npz_path.with_suffix(".eqx")
        if not eqx_path.exists():
            raise FileNotFoundError(f"Missing model file for {npz_path.name}: {eqx_path}")

        case = cases[case_name]
        model = _load_model(eqx_path, case, seed)
        pred_energy = np.asarray(predict_energy(model, strains))
        pred_stiffness = np.asarray(predict_effective_stiffness(model, strains))
        train_mse = float(np.asarray(run_data["train_mse"]).item())
        test_mse = float(np.asarray(run_data["test_mse"]).item())

        enriched = {
            "strains": np.asarray(strains),
            "pred_energy": pred_energy,
            "pred_stiffness": pred_stiffness,
            "stiffness_sharpness": summary_sharpness(pred_stiffness),
            "generalization_gap": test_mse - train_mse,
        }
        _save_enriched_npz(npz_path, run_data, enriched)

        updated_run_data = dict(run_data)
        updated_run_data.update(enriched)
        results_by_case[case_name].append(_result_for_plotting(case, updated_run_data, problem, seed))
        print(f"Enriched {npz_path.name}")

    for case_name, case_results in results_by_case.items():
        if not case_results:
            continue
        plot_all_init_figures(case_results, str(fig_dir), case_name)
        print(f"Regenerated plots for {case_name}")

    print("\nDone.")
    print(f"Updated runs:   {runs_dir}")
    print(f"Saved figures:  {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate strain energy/stiffness data and init-sensitivity plots from saved models."
    )
    parser.add_argument(
        "output_root",
        help="Output folder from run_ablations_init_sensitivity.py, or its runs/ subfolder.",
    )
    parser.add_argument(
        "--data-path",
        default=str(SCRIPT_DIR / "experiment_data" / "pulling_phase_data.npz"),
        help="Path to pulling_phase_data.npz. Defaults to the experiment_data folder next to this script.",
    )
    args = parser.parse_args()

    regenerate_full_metrics(
        output_root=_resolve_output_root(args.output_root),
        data_path=_resolve_output_root(args.data_path),
    )


if __name__ == "__main__":
    main()
