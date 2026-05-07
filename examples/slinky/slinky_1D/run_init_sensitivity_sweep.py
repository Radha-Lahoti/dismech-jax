"""
Multi-seed initialization sensitivity sweep across architectures and test ranges.

Runs ICNN_energy, MLP_energy, and MLP_Stiffness for three test_range splits
and writes outputs into separate per-test-range directories.
"""

import json
import os

import equinox as eqx
import numpy as np

from ablation_config import CaseConfig
from ablation_data import load_problem
from ablation_io import ensure_dir
from ablation_plots_init import plot_all_init_figures
from ablation_training import train_one_case


ARCHITECTURES = [
    CaseConfig("energy_mlp_L2", "energy_mlp", (10, 10), "mlp"),
    CaseConfig("energy_icnn_L2", "energy_icnn", (10, 10), "icnn"),
    CaseConfig("stiffness_baseline_plus_mlp_L2", "stiffness_mlp", (10, 10), "combined"),
]

TEST_RANGES = [
    (0.2, 0.8),
    (0.0, 0.5),
    (0.5, 1.0),
]

SEEDS = list(range(50))


def test_range_dirname(test_range):
    lo, hi = test_range
    return f"seed_envelope_test_{lo:.1f}_{hi:.1f}".replace(".", "p")


def run_for_test_range(test_range):
    out_root = test_range_dirname(test_range)
    run_dir = os.path.join(out_root, "runs")
    fig_dir = os.path.join(out_root, "figures")
    ensure_dir(run_dir)
    ensure_dir(fig_dir)

    print("#" * 90)
    print(f"# test_range = {test_range}  ->  {out_root}")
    print("#" * 90)

    problem = load_problem(
        data_path="experiment_data/pulling_phase_data.npz",
        test_range=test_range,
    )

    all_results = {case.name: [] for case in ARCHITECTURES}
    summary_rows = []

    for case in ARCHITECTURES:
        print("=" * 90)
        print(f"Running case: {case.name}")

        for seed in SEEDS:
            print(f"  seed = {seed}")
            model, result = train_one_case(
                case=case,
                problem=problem,
                seed=seed,
                lr=1e-3,
                num_epochs=10000,
                log_freq=500,
                gradient_clip_norm=1.0,
                full_metrics=True,
            )

            all_results[case.name].append(result)
            summary_rows.append(
                {
                    "name": case.name,
                    "seed": seed,
                    "test_range": list(test_range),
                    "train_mse": result["train_mse"],
                    "test_mse": result["test_mse"],
                }
            )

            stem = f"{case.name}__seed{seed:03d}"
            np.savez(
                os.path.join(run_dir, f"{stem}.npz"),
                pulled_node_x=result["pulled_node_x"],
                force_truth=result["force_truth"],
                pred_force=result["pred_force"],
                train_mask=result["train_mask"],
                test_mask=result["test_mask"],
                train_hist=result["train_hist"],
                test_hist=result["test_hist"],
                train_mse=result["train_mse"],
                test_mse=result["test_mse"],
                seed=result["seed"],
            )
            eqx.tree_serialise_leaves(os.path.join(run_dir, f"{stem}.eqx"), model)

        plot_all_init_figures(all_results[case.name], fig_dir, case.name)

    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\n[done] test_range={test_range}")
    print(f"  runs:    {run_dir}")
    print(f"  figures: {fig_dir}")


def main():
    for test_range in TEST_RANGES:
        run_for_test_range(test_range)
    print("\nAll test ranges complete.")


if __name__ == "__main__":
    main()
