"""Post-training Hessian diagnostics for the N3 slinky-sim seed ablation.

Mirrors cell 5 of run_seed_ablations_N3_slinky_sim.ipynb. The training cells
are skipped — this assumes results.npz already exists for every seed run
under seed_ablation_outputs_n3_slinky_sim/.

Run from examples/slinky/slinky_2D/ (relative dataset paths inside saved
config.json files resolve from there).
"""

import os
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from seed_ablation_utils import run_seed_hessian_diagnostics


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "seed_ablation_outputs_n3_slinky_sim"


def main() -> int:
    print(f"[hessian] output_dir = {OUTPUT_DIR.resolve()}", flush=True)

    HESSIAN_USE_PREDICTED = True
    HESSIAN_STRIDE = 10
    HESSIAN_MAX_TRAJECTORIES = 1

    n_ok = run_seed_hessian_diagnostics(
        str(OUTPUT_DIR),
        use_predicted=HESSIAN_USE_PREDICTED,
        splits=("train", "valid"),
        stride=HESSIAN_STRIDE,
        max_trajectories=HESSIAN_MAX_TRAJECTORIES,
        all_trajectories=False,
        fail_on_nonconvergence=False,
    )
    print(f"[hessian] done: {n_ok} runs processed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
