"""
Full ablation grid on experimental pulling-phase data with Hessian regularization.

Shared logic lives in ``ablation_*`` modules; this script selects overlay groups
and writes summary tables and figures.
"""

import json
import os

import numpy as np

from ablation_config import build_case_list
from ablation_data import load_problem
from ablation_io import ensure_dir, save_case_outputs
from ablation_plots_experiment import (
    plot_energy_strain_overlay,
    plot_force_displacement_overlay,
    plot_loss_histories_overlay,
    plot_stiffness_strain_overlay,
    plot_summary_bars,
)
from ablation_training_with_reg import train_one_case


def main():
    out_root = "ablation_outputs_new_hessian_reg_1e-6"
    case_outdir = os.path.join(out_root, "cases")
    fig_outdir = os.path.join(out_root, "figures")
    ensure_dir(case_outdir)
    ensure_dir(fig_outdir)

    problem = load_problem(data_path="experiment_data/pulling_phase_data.npz", test_range=(0.2, 0.8))
    cases = build_case_list()

    results_by_name = {}
    summary_rows = []

    for i, case in enumerate(cases):
        print("=" * 80)
        print(f"Running case {i + 1}/{len(cases)}: {case.name}")
        final_model, result = train_one_case(
            case=case,
            problem=problem,
            seed=0,
            lr=1e-3,
            num_epochs=10000,
            log_freq=500,
            gradient_clip_norm=1.0,
            full_metrics=True,
            hessian_reg_strength=1e-6,
            hessian_reg_probes=1,
            hessian_reg_seed=0,
        )
        save_case_outputs(case_outdir, final_model, result)
        results_by_name[case.name] = result
        summary_rows.append(
            {
                "name": case.name,
                "family": case.family,
                "hidden_sizes": case.hidden_sizes,
                "which_case": case.which_case,
                "train_mse": result["train_mse"],
                "test_mse": result["test_mse"],
                "generalization_gap": result["generalization_gap"],
                "stiffness_sharpness": result["stiffness_sharpness"],
            }
        )

    summary_rows = sorted(summary_rows, key=lambda x: x["test_mse"])
    np.savez(os.path.join(out_root, "summary_table.npz"), rows=np.array(summary_rows, dtype=object))

    with open(os.path.join(out_root, "summary_table.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    ablation_A = [
        "energy_baseline",
        "energy_mlp_L2",
        "energy_icnn_L2",
        "stiffness_baseline_plus_mlp_L2",
    ]
    ablation_A = [name for name in ablation_A if name in results_by_name]

    plot_force_displacement_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_A_force_displacement_overlay.png"),
        selected_names=ablation_A,
        title="Ablation A: parameterization and architecture",
    )
    plot_stiffness_strain_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_A_stiffness_strain_overlay.png"),
        selected_names=ablation_A,
        title="Ablation A: parameterization and architecture",
    )
    plot_energy_strain_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_A_energy_strain_overlay.png"),
        selected_names=ablation_A,
        title="Ablation A: parameterization and architecture",
    )
    plot_loss_histories_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_A_loss_histories_overlay.png"),
        selected_names=ablation_A,
        title="Ablation A: training dynamics",
    )
    plot_summary_bars(
        [row for row in summary_rows if row["name"] in ablation_A],
        save_path=os.path.join(fig_outdir, "ablation_A_summary_bars.png"),
        title="Ablation A: summary metrics",
    )

    ablation_B = [
        "energy_icnn_L1",
        "energy_icnn_L2",
        "stiffness_only_mlp_L1",
        "stiffness_only_mlp_L2",
    ]
    ablation_B = [name for name in ablation_B if name in results_by_name]

    plot_force_displacement_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_B_force_displacement_overlay.png"),
        selected_names=ablation_B,
        title="Ablation B: depth effect",
    )
    plot_stiffness_strain_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_B_stiffness_strain_overlay.png"),
        selected_names=ablation_B,
        title="Ablation B: depth effect",
    )
    plot_energy_strain_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_B_energy_strain_overlay.png"),
        selected_names=ablation_B,
        title="Ablation B: depth effect",
    )
    plot_loss_histories_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_B_loss_histories_overlay.png"),
        selected_names=ablation_B,
        title="Ablation B: training dynamics",
    )
    plot_summary_bars(
        [row for row in summary_rows if row["name"] in ablation_B],
        save_path=os.path.join(fig_outdir, "ablation_B_summary_bars.png"),
        title="Ablation B: summary metrics",
    )

    all_names = list(results_by_name.keys())
    plot_force_displacement_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "supp_force_displacement_all.png"),
        selected_names=all_names,
        title="Supplement: all cases",
    )
    plot_energy_strain_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "supp_energy_strain_all.png"),
        selected_names=all_names,
        title="Supplement: all cases",
    )
    plot_stiffness_strain_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "supp_stiffness_strain_all.png"),
        selected_names=all_names,
        title="Supplement: all cases",
    )
    plot_loss_histories_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "supp_loss_histories_all.png"),
        selected_names=all_names,
        title="Supplement: all cases",
    )

    print("\nDone. Outputs written to:")
    print(f"  {out_root}/summary_table.json")
    print(f"  {fig_outdir}/")
    print(f"  {case_outdir}/")


if __name__ == "__main__":
    main()
