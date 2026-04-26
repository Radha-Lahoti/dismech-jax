"""
Utilities for adding linear interpolation points to direct-BC datasets.

The datasets produced by ``extract_directbc_dataset`` are rectangular ``.npz``
files. Inserting timesteps for one trajectory therefore extends the shared time
axis for every trajectory. For synthetic points that were not requested for a
trajectory, this module fills numerically sensible linear values but marks the
corresponding ``valid`` entries as False.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np


TIME_DEPENDENT_KEYS = ("qs", "xb", "lambdas", "valid")


def add_interpolated_regions(
    input_npz_path,
    interpolation_specs,
    output_npz_path=None,
    *,
    mark_unrequested_valid=False,
    compressed=False,
):
    """
    Add linearly interpolated timesteps to an extracted direct-BC dataset.

    Parameters
    ----------
    input_npz_path : str or pathlib.Path
        Path to a ``.npz`` file produced by ``extract_directbc_dataset``.

    interpolation_specs : list
        Each item describes one region to augment and may be either a dict:

            {"traj_idx": 0, "timesteps": (10, 11), "num_interp_points": 5}

        or a tuple/list:

            (traj_idx, timestep_start, timestep_end, num_interp_points)

        ``timestep_end`` must be exactly ``timestep_start + 1``. Timesteps are
        interpreted in the original, pre-interpolation dataset.

    output_npz_path : str or pathlib.Path or None
        If provided, the augmented dataset is written here.

    mark_unrequested_valid : bool
        If False, inserted points are valid only for trajectories explicitly
        listed in ``interpolation_specs`` for that interval. If True, inserted
        points are marked valid for all trajectories whose bracketing endpoints
        are valid.

    compressed : bool
        If True, save with ``np.savez_compressed`` instead of ``np.savez``.

    Returns
    -------
    dict
        Augmented dataset arrays keyed by the original ``.npz`` keys.
    """
    input_npz_path = Path(input_npz_path)
    with np.load(input_npz_path) as loaded:
        data = {key: loaded[key] for key in loaded.files}

    _validate_directbc_dataset(data)
    requests_by_interval = _normalize_specs(interpolation_specs, data["qs"].shape)

    if not requests_by_interval:
        augmented = {key: value.copy() for key, value in data.items()}
    else:
        augmented = _insert_intervals(
            data,
            requests_by_interval,
            mark_unrequested_valid=mark_unrequested_valid,
        )

    if output_npz_path is not None:
        output_npz_path = Path(output_npz_path)
        save = np.savez_compressed if compressed else np.savez
        save(output_npz_path, **augmented)

    return augmented


def _validate_directbc_dataset(data):
    required = {"qs", "xb", "idx_b", "lambdas", "valid"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Dataset is missing required keys: {missing}")

    qs = np.asarray(data["qs"])
    xb = np.asarray(data["xb"])
    valid = np.asarray(data["valid"])

    if qs.ndim != 3:
        raise ValueError(f"qs must have shape (n_traj, T, dof), got {qs.shape}")
    if xb.ndim != 3:
        raise ValueError(f"xb must have shape (n_traj, T, n_b), got {xb.shape}")
    if valid.ndim != 2:
        raise ValueError(f"valid must have shape (n_traj, T), got {valid.shape}")

    n_traj, T = qs.shape[:2]
    if xb.shape[:2] != (n_traj, T):
        raise ValueError(f"xb shape {xb.shape} incompatible with qs shape {qs.shape}")
    if valid.shape != (n_traj, T):
        raise ValueError(f"valid shape {valid.shape} incompatible with qs shape {qs.shape}")

    lambdas = np.asarray(data["lambdas"])
    if lambdas.ndim < 2 or lambdas.shape[1] != T:
        raise ValueError(
            "lambdas must have time as axis 1 and match qs.shape[1], "
            f"got lambdas shape {lambdas.shape} and T={T}"
        )
    if lambdas.shape[0] not in (1, n_traj):
        raise ValueError(
            "lambdas first axis must be either 1 or n_traj, "
            f"got lambdas shape {lambdas.shape} and n_traj={n_traj}"
        )


def _normalize_specs(interpolation_specs, qs_shape):
    n_traj, T = qs_shape[:2]
    requests_by_interval = defaultdict(list)
    seen = set()

    for raw_spec in interpolation_specs:
        traj_idx, t0, t1, num_points = _parse_spec(raw_spec)

        if not 0 <= traj_idx < n_traj:
            raise ValueError(f"traj_idx={traj_idx} is out of range for n_traj={n_traj}")
        if not 0 <= t0 < T or not 0 <= t1 < T:
            raise ValueError(f"timesteps {(t0, t1)} out of range for T={T}")
        if t1 != t0 + 1:
            raise ValueError(
                "Interpolation timesteps must be consecutive; "
                f"got {(t0, t1)}"
            )
        if num_points < 0:
            raise ValueError(f"num_interp_points must be >= 0, got {num_points}")
        if num_points == 0:
            continue

        request_id = (traj_idx, t0, t1)
        if request_id in seen:
            raise ValueError(f"Duplicate interpolation request for {request_id}")
        seen.add(request_id)

        requests_by_interval[(t0, t1)].append((traj_idx, num_points))

    normalized = {}
    for interval, requests in requests_by_interval.items():
        counts = {num_points for _, num_points in requests}
        if len(counts) != 1:
            raise ValueError(
                "Requests for the same timestep interval must use the same "
                "num_interp_points because the dataset has one shared time axis. "
                f"Interval {interval} got counts {sorted(counts)}."
            )
        normalized[interval] = {
            "traj_indices": np.array([traj_idx for traj_idx, _ in requests], dtype=int),
            "num_points": counts.pop(),
        }

    return normalized


def _parse_spec(raw_spec):
    if isinstance(raw_spec, dict):
        traj_idx = raw_spec["traj_idx"]
        t0, t1 = raw_spec["timesteps"]
        num_points = raw_spec["num_interp_points"]
    else:
        if len(raw_spec) != 4:
            raise ValueError(
                "Tuple/list interpolation specs must be "
                "(traj_idx, timestep_start, timestep_end, num_interp_points)"
            )
        traj_idx, t0, t1, num_points = raw_spec

    return int(traj_idx), int(t0), int(t1), int(num_points)


def _insert_intervals(data, requests_by_interval, *, mark_unrequested_valid):
    augmented = {key: value.copy() for key, value in data.items()}
    current_T = augmented["qs"].shape[1]

    for (t0, t1), request in sorted(requests_by_interval.items(), reverse=True):
        if current_T != augmented["qs"].shape[1]:
            raise RuntimeError("Internal time-axis bookkeeping error.")

        num_points = request["num_points"]
        traj_indices = request["traj_indices"]
        fractions = np.arange(1, num_points + 1, dtype=float) / (num_points + 1)

        for key, arr in list(augmented.items()):
            if not _has_matching_time_axis(arr, current_T):
                continue

            if key == "valid":
                insert_block = _make_valid_insert_block(
                    arr,
                    t0,
                    t1,
                    num_points,
                    traj_indices,
                    mark_unrequested_valid=mark_unrequested_valid,
                )
            else:
                insert_block = _linear_insert_block(arr, t0, t1, fractions)

            augmented[key] = np.concatenate(
                [arr[:, :t1, ...], insert_block, arr[:, t1:, ...]],
                axis=1,
            )

        current_T += num_points

    return augmented


def _has_matching_time_axis(arr, T):
    return isinstance(arr, np.ndarray) and arr.ndim >= 2 and arr.shape[1] == T


def _linear_insert_block(arr, t0, t1, fractions):
    start = arr[:, t0 : t0 + 1, ...]
    end = arr[:, t1 : t1 + 1, ...]
    reshape = (1, len(fractions)) + (1,) * (arr.ndim - 2)
    weights = fractions.reshape(reshape)
    return start + weights * (end - start)


def _make_valid_insert_block(
    valid,
    t0,
    t1,
    num_points,
    traj_indices,
    *,
    mark_unrequested_valid,
):
    endpoint_valid = valid[:, t0] & valid[:, t1]
    if mark_unrequested_valid:
        inserted_valid = endpoint_valid
    else:
        inserted_valid = np.zeros(valid.shape[0], dtype=bool)
        inserted_valid[traj_indices] = endpoint_valid[traj_indices]

    return np.repeat(inserted_valid[:, None], num_points, axis=1)
