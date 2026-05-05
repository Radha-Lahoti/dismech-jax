"""Load experiment pulling-phase data and build train/test masks."""

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from slinky_1d_system import Slinky1D


def load_problem(
    data_path: str = "experiment_data/pulling_phase_data.npz",
    test_range: Tuple[float, float] = (0.2, 0.8),
) -> Dict[str, Any]:
    data = np.load(data_path)

    force_truth = np.abs(jnp.array(data["F"]))
    disps = jnp.array(data["disps"])
    initial_last_node_x = float(data["initial_last_node_x"])
    num_steps = force_truth.shape[0]

    x_left = 0.0
    q0 = jnp.array([x_left, initial_last_node_x])
    l_k0 = q0[1] - q0[0]

    slinky = Slinky1D(
        l_k=jnp.array(l_k0),
        x_left=jnp.array(x_left),
    )

    disp_min = disps.min()
    disp_max = disps.max()
    split_lo = disp_min + test_range[0] * (disp_max - disp_min)
    split_hi = disp_min + test_range[1] * (disp_max - disp_min)

    include_hi = test_range[1] >= 1.0
    if include_hi:
        train_mask = disps < split_lo
        test_mask = disps >= split_lo
    else:
        train_mask = (disps < split_lo) | (disps >= split_hi)
        test_mask = (disps >= split_lo) & (disps < split_hi)

    qs = jax.vmap(lambda d: slinky.get_q(d, q0))(disps)
    strains = jax.vmap(slinky.get_eps)(qs)

    return {
        "slinky": slinky,
        "q0": q0,
        "force_truth": force_truth,
        "disps": disps,
        "pulled_node_x": disps,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "strains": strains,
        "meta": {
            "initial_last_node_x": initial_last_node_x,
            "rest_length": float(l_k0),
            "num_steps": int(num_steps),
            "test_range_fraction": list(test_range),
            "disp_split_lo": float(split_lo),
            "disp_split_hi": float(split_hi),
        },
    }
