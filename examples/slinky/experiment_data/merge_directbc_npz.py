"""
Merge two direct-BC trajectory datasets exported by extract_data.py.

The exported .npz files are expected to contain at least:
    qs      : (n_traj, T, dof)
    xb      : (n_traj, T, n_b)
    idx_b   : (n_b,)
    lambdas : (1, T) or (n_traj, T)
    valid   : (n_traj, T)

The merge is controlled by the final-node x coordinate, qs[:, :, -3].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


MERGE_MODES = ("trim_traj1", "trim_traj2", "append_offset")


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _require_directbc_format(data, path):
    required = ("qs", "xb", "idx_b", "lambdas", "valid")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required key(s): {missing}")

    qs = data["qs"]
    xb = data["xb"]
    idx_b = data["idx_b"]
    lambdas = data["lambdas"]
    valid = data["valid"]

    if qs.ndim != 3:
        raise ValueError(f"{path}: qs must have shape (n_traj, T, dof), got {qs.shape}")
    if xb.ndim != 3:
        raise ValueError(f"{path}: xb must have shape (n_traj, T, n_b), got {xb.shape}")
    if idx_b.ndim != 1:
        raise ValueError(f"{path}: idx_b must have shape (n_b,), got {idx_b.shape}")
    n_traj, t_steps, dof = qs.shape
    if lambdas.ndim != 2 or lambdas.shape[0] not in (1, n_traj):
        raise ValueError(
            f"{path}: lambdas must have shape (1, T) or (n_traj, T), got {lambdas.shape}"
        )
    if valid.ndim != 2:
        raise ValueError(f"{path}: valid must have shape (n_traj, T), got {valid.shape}")

    if (dof + 1) % 4 != 0:
        raise ValueError(f"{path}: qs dof={dof} is not compatible with 4*n_nodes - 1")
    if xb.shape[:2] != (n_traj, t_steps):
        raise ValueError(f"{path}: xb shape {xb.shape} is incompatible with qs shape {qs.shape}")
    if xb.shape[2] != len(idx_b):
        raise ValueError(f"{path}: xb.shape[2]={xb.shape[2]} must equal len(idx_b)={len(idx_b)}")
    if lambdas.shape[1] != t_steps:
        raise ValueError(f"{path}: lambdas shape {lambdas.shape} is incompatible with T={t_steps}")
    if valid.shape != (n_traj, t_steps):
        raise ValueError(f"{path}: valid shape {valid.shape} is incompatible with qs shape {qs.shape}")


def _check_pair_compatible(data1, data2, path1, path2):
    _require_directbc_format(data1, path1)
    _require_directbc_format(data2, path2)

    qs1, qs2 = data1["qs"], data2["qs"]
    if qs1.shape[0] != qs2.shape[0]:
        raise ValueError(f"n_traj mismatch: {path1} has {qs1.shape[0]}, {path2} has {qs2.shape[0]}")
    if qs1.shape[2] != qs2.shape[2]:
        raise ValueError(f"dof mismatch: {path1} has {qs1.shape[2]}, {path2} has {qs2.shape[2]}")
    if not np.array_equal(data1["idx_b"], data2["idx_b"]):
        raise ValueError("idx_b arrays differ; cannot merge moving BC columns safely")
    if data1["xb"].shape[2] != data2["xb"].shape[2]:
        raise ValueError("xb boundary-column counts differ")


def _monotonic_direction(x, tol):
    dx = np.diff(x)
    dx = dx[np.abs(dx) > tol]
    if dx.size == 0:
        raise ValueError("Cannot infer x direction because final-node x is constant")
    sign = np.sign(np.median(dx))
    if sign > 0 and np.any(np.diff(x) < -tol):
        raise ValueError("final-node x is not monotonic increasing")
    if sign < 0 and np.any(np.diff(x) > tol):
        raise ValueError("final-node x is not monotonic decreasing")
    return int(sign)


def _merge_direction(x1, x2, tol):
    direction1 = _monotonic_direction(x1, tol)
    direction2 = _monotonic_direction(x2, tol)
    if direction1 != direction2:
        raise ValueError(
            "The two final-node x histories move in opposite directions; "
            "reverse one dataset before merging or use append_offset if appropriate"
        )
    return direction1


def _time_slice(arr, mask):
    return arr[:, mask, ...]


def _slice_dataset(data, mask):
    sliced = {}
    t_steps = data["qs"].shape[1]
    n_traj = data["qs"].shape[0]

    for key, arr in data.items():
        if key == "idx_b":
            sliced[key] = arr.copy()
        elif key == "lambdas":
            sliced[key] = arr[:, mask].copy()
        elif arr.ndim >= 2 and arr.shape[1] == t_steps and arr.shape[0] in (1, n_traj):
            sliced[key] = _time_slice(arr, mask).copy()
        else:
            sliced[key] = arr.copy()

    return sliced


def _shift_dof_coordinate(data, dof_mod, offset, skip_dof_idx=()):
    shifted = {key: arr.copy() for key, arr in data.items()}
    if offset == 0:
        return shifted

    dof = shifted["qs"].shape[2]
    coord_dof_idx = np.arange(dof_mod, dof, 4)
    if skip_dof_idx:
        skip_dof_idx = set(skip_dof_idx)
        coord_dof_idx = np.array(
            [idx for idx in coord_dof_idx if idx not in skip_dof_idx],
            dtype=int,
        )
    shifted["qs"][:, :, coord_dof_idx] += offset

    idx_b = shifted["idx_b"]
    coord_bc_cols = np.flatnonzero(idx_b % 4 == dof_mod)
    if skip_dof_idx:
        coord_bc_cols = np.array(
            [col for col in coord_bc_cols if int(idx_b[col]) not in skip_dof_idx],
            dtype=int,
        )
    if coord_bc_cols.size:
        shifted["xb"][:, :, coord_bc_cols] += offset

    return shifted


def _shift_x(data, offset):
    return _shift_dof_coordinate(data, dof_mod=0, offset=offset)


def _shift_z(data, offset, skip_first_node=False, skip_second_node=False):
    skip_dof_idx = []
    if skip_first_node:
        skip_dof_idx.append(2)
    if skip_second_node:
        skip_dof_idx.append(6)
    return _shift_dof_coordinate(
        data,
        dof_mod=2,
        offset=offset,
        skip_dof_idx=skip_dof_idx,
    )


def _freeze_node1_x(data):
    frozen = {key: arr.copy() for key, arr in data.items()}
    if frozen["qs"].shape[2] <= 4:
        raise ValueError("Cannot freeze node 1 x because qs does not contain dof index 4")

    node1_x = frozen["qs"][:, 0:1, 4]
    frozen["qs"][:, :, 4] = node1_x

    idx_b = frozen["idx_b"]
    node1_x_bc_cols = np.flatnonzero(idx_b == 4)
    if node1_x_bc_cols.size:
        frozen["xb"][:, :, node1_x_bc_cols] = node1_x[:, :, None]

    return frozen


def _concat_value(key, arr1, arr2, t1, t2, n_traj, recompute_lambdas):
    if key == "idx_b":
        return arr1.copy()
    if key == "lambdas" and recompute_lambdas:
        base = np.linspace(0.0, 1.0, t1 + t2, dtype=np.result_type(arr1, arr2))[None, :]
        if arr1.shape[0] == 1:
            return base
        return np.repeat(base, arr1.shape[0], axis=0)
    if key == "lambdas":
        return np.concatenate([arr1, arr2], axis=1)
    if arr1.ndim >= 2 and arr2.ndim >= 2:
        if arr1.shape[1] == t1 and arr2.shape[1] == t2 and arr1.shape[0] in (1, n_traj):
            if arr1.shape[0] != arr2.shape[0] or arr1.shape[2:] != arr2.shape[2:]:
                raise ValueError(f"Cannot concatenate key '{key}' with shapes {arr1.shape} and {arr2.shape}")
            return np.concatenate([arr1, arr2], axis=1)
    if arr1.shape != arr2.shape or not np.array_equal(arr1, arr2):
        raise ValueError(
            f"Non-time-dependent key '{key}' differs between files; "
            "merge would be ambiguous"
        )
    return arr1.copy()


def merge_directbc_npz(
    data1_npz,
    data2_npz,
    output_npz,
    mode="trim_traj2",
    overlap_tol=1e-9,
    reference_traj=0,
    z_offset1=0.0,
    z_offset2=0.0,
    skip_first_z_offset1=False,
    skip_first_z_offset2=False,
    skip_second_z_offset1=False,
    skip_second_z_offset2=False,
    freeze_node1_x=True,
    recompute_lambdas=True,
    compressed=False,
):
    """
    Merge two direct-BC .npz files into one continuous trajectory.

    Parameters
    ----------
    data1_npz, data2_npz : str or Path
        Input files exported by extract_directbc_dataset().
    output_npz : str or Path
        Output merged .npz path.
    mode : {"trim_traj1", "trim_traj2", "append_offset"}
        - "trim_traj1": remove the suffix of traj1 whose final-node x overlaps traj2.
        - "trim_traj2": remove the prefix of traj2 whose final-node x overlaps traj1.
        - "append_offset": keep both histories and shift traj2 x values so
          traj2's first final-node x equals traj1's last final-node x.
    overlap_tol : float
        Tolerance used when comparing final-node x values.
    reference_traj : int
        Batch trajectory used to choose trim points.
    z_offset1, z_offset2 : float
        Offsets added to every node z coordinate in data1 and data2,
        respectively. Matching z columns in xb are shifted too.
    skip_first_z_offset1, skip_first_z_offset2 : bool
        If True, do not apply that trajectory's z offset to the first node
        z coordinate. Matching z columns in xb are skipped too.
    skip_second_z_offset1, skip_second_z_offset2 : bool
        If True, do not apply that trajectory's z offset to the second node
        z coordinate. Matching z columns in xb are skipped too.
    freeze_node1_x : bool
        If True, set qs[:, :, 4] to qs[:, 0, 4] throughout the final merged
        trajectory. If idx_b contains dof 4, the matching xb column is frozen too.
    recompute_lambdas : bool
        If True, write lambdas as linspace(0, 1, merged_T).
    compressed : bool
        If True, use np.savez_compressed instead of np.savez.

    Returns
    -------
    dict
        The merged arrays that were written to output_npz.
    """
    if mode not in MERGE_MODES:
        raise ValueError(f"mode must be one of {MERGE_MODES}, got {mode!r}")

    data1_npz = Path(data1_npz)
    data2_npz = Path(data2_npz)
    output_npz = Path(output_npz)

    data1 = _load_npz(data1_npz)
    data2 = _load_npz(data2_npz)
    _check_pair_compatible(data1, data2, data1_npz, data2_npz)

    n_traj, t1_orig, _ = data1["qs"].shape
    _, t2_orig, _ = data2["qs"].shape
    if not 0 <= reference_traj < n_traj:
        raise ValueError(f"reference_traj must be in [0, {n_traj}), got {reference_traj}")

    x1 = data1["qs"][reference_traj, :, -3]
    x2 = data2["qs"][reference_traj, :, -3]

    if mode == "append_offset":
        offset = x1[-1] - x2[0]
        data2 = _shift_x(data2, offset)
        mask1 = np.ones(t1_orig, dtype=bool)
        mask2 = np.ones(t2_orig, dtype=bool)
    else:
        direction = _merge_direction(x1, x2, overlap_tol)
        if mode == "trim_traj1":
            if direction > 0:
                mask1 = x1 < x2[0] - overlap_tol
            else:
                mask1 = x1 > x2[0] + overlap_tol
            mask2 = np.ones(t2_orig, dtype=bool)
        else:
            mask1 = np.ones(t1_orig, dtype=bool)
            if direction > 0:
                mask2 = x2 > x1[-1] + overlap_tol
            else:
                mask2 = x2 < x1[-1] - overlap_tol

    if not np.any(mask1):
        raise ValueError("All traj1 samples were removed; check mode, x ordering, or overlap_tol")
    if not np.any(mask2):
        raise ValueError("All traj2 samples were removed; check mode, x ordering, or overlap_tol")

    data1 = _slice_dataset(data1, mask1)
    data2 = _slice_dataset(data2, mask2)
    data1 = _shift_z(
        data1,
        z_offset1,
        skip_first_node=skip_first_z_offset1,
        skip_second_node=skip_second_z_offset1,
    )
    data2 = _shift_z(
        data2,
        z_offset2,
        skip_first_node=skip_first_z_offset2,
        skip_second_node=skip_second_z_offset2,
    )

    t1 = data1["qs"].shape[1]
    t2 = data2["qs"].shape[1]
    merged = {}
    for key in data1:
        if key not in data2:
            raise ValueError(f"Key '{key}' exists in {data1_npz} but not in {data2_npz}")
        merged[key] = _concat_value(
            key,
            data1[key],
            data2[key],
            t1,
            t2,
            n_traj,
            recompute_lambdas,
        )

    if freeze_node1_x:
        merged = _freeze_node1_x(merged)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(output_npz, **merged)

    return merged


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Merge two direct-BC .npz trajectory datasets along the time axis."
    )
    parser.add_argument("data1_npz", help="First input .npz file")
    parser.add_argument("data2_npz", help="Second input .npz file")
    parser.add_argument("output_npz", help="Output merged .npz file")
    parser.add_argument(
        "--mode",
        choices=MERGE_MODES,
        default="trim_traj2",
        help="How to handle overlap in final-node x (default: trim_traj2)",
    )
    parser.add_argument(
        "--overlap-tol",
        type=float,
        default=1e-9,
        help="Tolerance for x overlap comparisons",
    )
    parser.add_argument(
        "--reference-traj",
        type=int,
        default=0,
        help="Batch trajectory index used to choose trim points",
    )
    parser.add_argument(
        "--z-offset1",
        type=float,
        default=0.0,
        help="Offset added to data1 node z coordinates and z moving-BC values",
    )
    parser.add_argument(
        "--z-offset2",
        type=float,
        default=0.0,
        help="Offset added to data2 node z coordinates and z moving-BC values",
    )
    parser.add_argument(
        "--skip-first-z-offset1",
        action="store_true",
        help="Do not apply --z-offset1 to the first node z coordinate",
    )
    parser.add_argument(
        "--skip-first-z-offset2",
        action="store_true",
        help="Do not apply --z-offset2 to the first node z coordinate",
    )
    parser.add_argument(
        "--skip-second-z-offset1",
        action="store_true",
        help="Do not apply --z-offset1 to the second node z coordinate",
    )
    parser.add_argument(
        "--skip-second-z-offset2",
        action="store_true",
        help="Do not apply --z-offset2 to the second node z coordinate",
    )
    parser.add_argument(
        "--no-freeze-node1-x",
        action="store_true",
        help="Do not force qs[:, :, 4] to remain at qs[:, 0, 4]",
    )
    parser.add_argument(
        "--keep-lambdas",
        action="store_true",
        help="Concatenate original lambdas instead of regenerating linspace(0, 1, T)",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Write output with np.savez_compressed",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    merged = merge_directbc_npz(
        args.data1_npz,
        args.data2_npz,
        args.output_npz,
        mode=args.mode,
        overlap_tol=args.overlap_tol,
        reference_traj=args.reference_traj,
        z_offset1=args.z_offset1,
        z_offset2=args.z_offset2,
        skip_first_z_offset1=args.skip_first_z_offset1,
        skip_first_z_offset2=args.skip_first_z_offset2,
        skip_second_z_offset1=args.skip_second_z_offset1,
        skip_second_z_offset2=args.skip_second_z_offset2,
        freeze_node1_x=not args.no_freeze_node1_x,
        recompute_lambdas=not args.keep_lambdas,
        compressed=args.compressed,
    )
    print(f"Wrote {args.output_npz}")
    print("qs shape:", merged["qs"].shape)
    print("xb shape:", merged["xb"].shape)
    print("idx_b:", merged["idx_b"])
    print("lambdas shape:", merged["lambdas"].shape)
    print("valid shape:", merged["valid"].shape)


if __name__ == "__main__":
    main()
