"""Resume the N3 slinky-sim seed ablation after the kernel crashed mid-run.

Runs only the work that's missing from
`seed_ablation_outputs_n3_slinky_sim/`:
  - chol_energy_icnn for seeds (23, 24, 25)
  - diag_stiffness_mlp for seeds 0..25
  - chol_stiffness_mlp for seeds 0..25

Then reloads every results.npz under the output dir to regenerate the seed
ablation summary across all 6 selected architectures.

Run from examples/slinky/slinky_2D/ so the relative dataset paths resolve.
"""

import os
import sys
from dataclasses import replace
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from properties import SlinkyN3Properties
from run_architectures import SweepConfig
from seed_ablation_utils import (
    load_seed_ablation_results,
    make_seed_ablation_plots,
    print_seed_summary,
    run_seed_ablation,
    save_seed_summary_json,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "seed_ablation_outputs_n3_slinky_sim"
SUMMARY_DIR = OUTPUT_DIR / "seed_ablation_summary"

train_file = "../simulation_data_2D/3_noded/n3_slinky_sim_train_dataset_7_trajs.npz"
valid_file = "../simulation_data_2D/3_noded/n3_slinky_sim_test_dataset_6_trajs.npz"

properties = SlinkyN3Properties(mass=0.2)
K_init_chol = (0.02, 0.0, 0.05)
K_init_diag = (0.02, 0.05)

FULL_SEED_LIST = tuple(range(26))

base_cfg = SweepConfig(
    der_K_diag=K_init_diag,
    der_K_chol=K_init_chol,
    hidden=(10, 10),
    corr_factor=0.05,
    input_mode="invariant",
    only_stretching_NN=True,
    only_bending_NN=False,
    zero_reference=True,
    activation="tanh",
    n_epochs=500,
    lr=1e-3,
    seed=42,
    seed_list=FULL_SEED_LIST,
    valid_every=10,
    max_dlambda=1e-2,
    iters=20,
    ls_steps=10,
    abs_tol=1e-4,
    rel_tol=1e-4,
    early_stop=True,
    train_fail_on_nonconvergence=False,
    prediction_fail_on_nonconvergence=False,
    hessian_reg_strength=1e-4,
    hessian_reg_probes=1,
    hessian_reg_seed=0,
    force_key="F",
    force_loss_strength=0.1,
    force_components=(0,),
    force_sign=1.0,
    return_loss_components=True,
    early_stopping=True,
    early_stopping_patience=200,
    early_stopping_min_delta=1e-5,
    early_stopping_warmup_epochs=200,
    restore_best_model=True,
    output_dir=str(OUTPUT_DIR),
    save_npz=True,
    save_model=True,
    save_plots=True,
    save_force_predictions=True,
    plot_force_predictions=True,
    save_hessian_diagnostics=False,
    save_energy_landscapes=True,
    energy_snapshot_initial=True,
    energy_snapshot_final=True,
    energy_snapshot_epochs=(),
    energy_snapshot_every=None,
    energy_snapshot_use_valid=True,
    energy_snapshot_dpi=180,
    energy_snapshot_n_grid=None,
    verbose=True,
    continue_on_failure=True,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[resume] writing into: {OUTPUT_DIR.resolve()}", flush=True)


SELECTED_ARCHITECTURES = [
    "diag_energy_mlp",
    "diag_energy_icnn",
    "chol_energy_mlp",
    "chol_energy_icnn",
    "diag_stiffness_mlp",
    "chol_stiffness_mlp",
]


def run_phase(label: str, seed_list: tuple[int, ...], archs: list[str]) -> None:
    print("\n" + "#" * 90, flush=True)
    print(f"[resume] phase: {label}", flush=True)
    print(f"[resume] architectures: {archs}", flush=True)
    print(f"[resume] seeds: {seed_list}", flush=True)
    print("#" * 90, flush=True)

    cfg = replace(base_cfg, seed_list=seed_list)
    run_seed_ablation(
        properties=properties,
        train_file=train_file,
        valid_file=valid_file,
        base_cfg=cfg,
        selected_architectures=archs,
    )


def main() -> None:
    # Phase 1: finish the chol_energy_icnn seeds that were missing/partial.
    run_phase(
        label="chol_energy_icnn tail (seeds 23, 24, 25)",
        seed_list=(23, 24, 25),
        archs=["chol_energy_icnn"],
    )

    # Phase 2: stiffness architectures that were never started.
    run_phase(
        label="stiffness architectures (full seed list)",
        seed_list=FULL_SEED_LIST,
        archs=["diag_stiffness_mlp", "chol_stiffness_mlp"],
    )

    # Phase 3: rebuild the summary dict from disk so it covers ALL 6
    # architectures × 26 seeds (not just the ones rerun in this script).
    print("\n" + "#" * 90, flush=True)
    print("[resume] rebuilding seed ablation summary from disk", flush=True)
    print("#" * 90, flush=True)
    all_seed_results = load_seed_ablation_results(
        str(OUTPUT_DIR), architectures=SELECTED_ARCHITECTURES
    )
    for arch, runs in all_seed_results.items():
        seeds = sorted(int(r["cfg"].seed) for r in runs)
        print(f"  {arch}: {len(runs)} seeds -> {seeds}", flush=True)

    print_seed_summary(all_seed_results)
    save_seed_summary_json(all_seed_results, output_dir=str(SUMMARY_DIR))
    make_seed_ablation_plots(
        all_seed_results=all_seed_results,
        output_dir=str(SUMMARY_DIR),
        traj_idx=0,
        x_idx=4,
        z_idx=6,
    )
    print(f"[resume] summary written to: {SUMMARY_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
