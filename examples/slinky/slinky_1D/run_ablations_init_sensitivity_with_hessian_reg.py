"""
Multi-seed initialization sensitivity with Hessian regularization.

Shared logic lives in ``ablation_*`` modules; this script only wires cases,
hyperparameters, saving, and figures.
"""

import json
import os

import equinox as eqx
import numpy as np

from ablation_config import CaseConfig
from ablation_data import load_problem
from ablation_io import ensure_dir
from ablation_plots_init import plot_all_init_figures
from ablation_training_with_reg import train_one_case


def main():
    out_root = "seed_envelope_mlp_energy_hessian_reg_1e-4"
    run_dir = os.path.join(out_root, "runs")
    fig_dir = os.path.join(out_root, "figures")
    ensure_dir(run_dir)
    ensure_dir(fig_dir)

    problem = load_problem(
        data_path="experiment_data/pulling_phase_data.npz",
        test_range=(0.2, 0.8),
    )

    cases = [
        CaseConfig("energy_mlp_L2", "energy_mlp", (10, 10), "mlp"),
    ]
    seeds = list(range(50))

    all_results = {case.name: [] for case in cases}
    summary_rows = []

    for case in cases:
        print("=" * 90)
        print(f"Running case: {case.name}")

        for seed in seeds:
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
                hessian_reg_strength=1e-4,
                hessian_reg_probes=1,
                hessian_reg_seed=seed,
            )

            all_results[case.name].append(result)
            summary_rows.append(
                {
                    "name": case.name,
                    "seed": seed,
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

    print("\nDone.")
    print(f"Saved runs to:    {run_dir}")
    print(f"Saved figures to: {fig_dir}")


if __name__ == "__main__":
    main()
