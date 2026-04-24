import json
import re
import warnings
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def _plot_marker_pair(marker_data, coord1="x", coord2="y", title="", xlabel=None, ylabel=None):
    plt.figure(figsize=(7.5, 6))
    for name in sorted(marker_data.keys()):
        a = np.asarray(marker_data[name][coord1], dtype=float)
        b = np.asarray(marker_data[name][coord2], dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if np.any(mask):
            plt.plot(a[mask], b[mask], linewidth=1.8, label=name)

    plt.xlabel(xlabel if xlabel is not None else coord1)
    plt.ylabel(ylabel if ylabel is not None else coord2)
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_ee_pair(a, b, title="", xlabel="x", ylabel="z"):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)

    plt.figure(figsize=(7.0, 5.5))
    if np.any(mask):
        plt.plot(a[mask], b[mask], linewidth=2.0, label="end_effector")
        plt.scatter(a[mask][0], b[mask][0], s=40, label="start")
        plt.scatter(a[mask][-1], b[mask][-1], s=40, label="end")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_markers_and_ee_final(marker_data, ee_x, ee_z, title="Final transformed trajectories"):
    plt.figure(figsize=(8.0, 6.5))

    for name in sorted(marker_data.keys()):
        x = np.asarray(marker_data[name]["x"], dtype=float)
        z = np.asarray(marker_data[name]["z"], dtype=float)
        mask = np.isfinite(x) & np.isfinite(z)
        if np.any(mask):
            plt.plot(x[mask], z[mask], linewidth=1.8, label=name)
            plt.text(x[mask][-1], z[mask][-1], name, fontsize=9)

    ee_x = np.asarray(ee_x, dtype=float)
    ee_z = np.asarray(ee_z, dtype=float)
    mask = np.isfinite(ee_x) & np.isfinite(ee_z)
    if np.any(mask):
        plt.plot(ee_x[mask], ee_z[mask], linewidth=2.5, label="EE")
        plt.scatter(ee_x[mask][0], ee_z[mask][0], s=45, marker="o")
        plt.scatter(ee_x[mask][-1], ee_z[mask][-1], s=45, marker="s")
        plt.text(ee_x[mask][-1], ee_z[mask][-1], "EE", fontsize=10)

    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_markers_and_lastnode_final(
    marker_data,
    last_x,
    last_z,
    last_node_label="EE",
    title="Final transformed trajectories",
):
    """
    Combined final plot where the last node can be either EE or marker_0.
    """
    plt.figure(figsize=(8.0, 6.5))

    for name in sorted(marker_data.keys()):
        x = np.asarray(marker_data[name]["x"], dtype=float)
        z = np.asarray(marker_data[name]["z"], dtype=float)
        mask = np.isfinite(x) & np.isfinite(z)
        if np.any(mask):
            plt.plot(x[mask], z[mask], linewidth=1.8, label=name)
            plt.text(x[mask][-1], z[mask][-1], name, fontsize=9)

    last_x = np.asarray(last_x, dtype=float)
    last_z = np.asarray(last_z, dtype=float)
    mask = np.isfinite(last_x) & np.isfinite(last_z)
    if np.any(mask):
        plt.plot(last_x[mask], last_z[mask], linewidth=2.5, label=last_node_label)
        plt.scatter(last_x[mask][0], last_z[mask][0], s=45, marker="o")
        plt.scatter(last_x[mask][-1], last_z[mask][-1], s=45, marker="s")
        plt.text(last_x[mask][-1], last_z[mask][-1], last_node_label, fontsize=10)

    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _marker_index(name):
    """
    Extract integer index from marker name like 'marker_4'.
    """
    m = re.match(r"marker_(\d+)$", name)
    if m is None:
        raise ValueError(f"Unexpected marker name format: {name}")
    return int(m.group(1))


# ---------------------------------------------------------
# Utility to mimic clamped BC on the left
# ---------------------------------------------------------
def prepend_clamped_node(
    qs,
    xb,
    idx_b,
    lambdas,
    valid=None,
    first_edge_length=0.0,
    check_consistency=True,
):
    """
    Prepend a clamped node to an already-constructed direct-BC dataset.

    Assumed state layout:
        [x0,y0,z0,th0, x1,y1,z1,th1, ..., x_{n-1},y_{n-1},z_{n-1}]
    with total dof = 4*n_nodes - 1.

    What this does
    --------------
    1. Prepends a new node/edge block [0,0,0,0] to the LEFT of qs.
    2. Shifts the entire old trajectory by +first_edge_length in x.
    3. Prepends clamp BCs for the new node:
           new node 0 fixed in position  -> indices [0,1,2]
           new theta0 fixed              -> index  [3]
    4. Shifts all old BC indices by +4.
    5. Shifts x-type BC values by +first_edge_length.
    """
    qs = np.asarray(qs)
    xb = np.asarray(xb)
    idx_b = np.asarray(idx_b, dtype=int)
    lambdas = np.asarray(lambdas)
    valid_new = None if valid is None else np.asarray(valid).copy()

    if qs.ndim != 3:
        raise ValueError(f"qs must have shape (n_traj, T, dof), got {qs.shape}")
    if xb.ndim != 3:
        raise ValueError(f"xb must have shape (n_traj, T, n_b), got {xb.shape}")
    if idx_b.ndim != 1:
        raise ValueError(f"idx_b must have shape (n_b,), got {idx_b.shape}")

    n_traj, T, dof_old = qs.shape
    if xb.shape[:2] != (n_traj, T):
        raise ValueError(f"xb shape {xb.shape} incompatible with qs shape {qs.shape}")
    if xb.shape[2] != len(idx_b):
        raise ValueError(
            f"xb.shape[2]={xb.shape[2]} must match len(idx_b)={len(idx_b)}"
        )

    if (dof_old + 1) % 4 != 0:
        raise ValueError(
            f"dof_old={dof_old} is not compatible with dof = 4*n_nodes - 1"
        )

    n_nodes_old = (dof_old + 1) // 4
    n_nodes_new = n_nodes_old + 1
    dof_new = 4 * n_nodes_new - 1

    qs_new = np.zeros((n_traj, T, dof_new), dtype=qs.dtype)
    qs_new[:, :, 4:] = qs

    # Shift all old node x coordinates by +first_edge_length
    old_node_x_idx_new = 4 + 4 * np.arange(n_nodes_old)
    qs_new[:, :, old_node_x_idx_new] += first_edge_length

    idx_b_new_list = []
    xb_new_cols = []

    def append_bc(idx, values):
        idx_b_new_list.append(int(idx))
        xb_new_cols.append(np.asarray(values))

    zeros = np.zeros((n_traj, T), dtype=xb.dtype)

    # New prepended clamped node
    append_bc(0, zeros)  # x_new0 = 0
    append_bc(1, zeros)  # y_new0 = 0
    append_bc(2, zeros)  # z_new0 = 0
    append_bc(3, zeros)  # theta_new0 = 0

    # Shift all original BCs by +4
    for j, old_idx in enumerate(idx_b):
        new_idx = int(old_idx + 4)
        new_vals = xb[:, :, j].copy()

        # x-position DOFs are 0,4,8,...
        if old_idx % 4 == 0:
            new_vals = new_vals + first_edge_length

        append_bc(new_idx, new_vals)

    idx_b_new = np.asarray(idx_b_new_list, dtype=int)
    xb_new = np.stack(xb_new_cols, axis=2)
    lambdas_new = lambdas.copy()

    if check_consistency:
        if qs_new.shape != (n_traj, T, dof_new):
            raise RuntimeError(f"Unexpected qs_new shape: {qs_new.shape}")
        if xb_new.shape[:2] != (n_traj, T):
            raise RuntimeError(f"Unexpected xb_new shape: {xb_new.shape}")
        if xb_new.shape[2] != len(idx_b_new):
            raise RuntimeError(
                f"xb_new.shape[2]={xb_new.shape[2]} != len(idx_b_new)={len(idx_b_new)}"
            )

        if not np.all(np.isfinite(qs_new)):
            raise ValueError("qs_new contains NaN/Inf")
        if not np.all(np.isfinite(xb_new)):
            raise ValueError("xb_new contains NaN/Inf")
        if not np.all(np.isfinite(lambdas_new)):
            raise ValueError("lambdas_new contains NaN/Inf")
        if valid_new is not None and not np.all(np.isfinite(valid_new.astype(float))):
            raise ValueError("valid_new contains NaN/Inf")

    return qs_new, xb_new, idx_b_new, lambdas_new, valid_new


# ---------------------------------------------------------
# Utility to add a right-end ghost clamp
# ---------------------------------------------------------
def append_right_ghost_clamp(
    qs,
    xb,
    idx_b,
    lambdas,
    valid=None,
    ghost_edge_length=0.0,
    check_consistency=True,
):
    """
    Append a ghost node to the RIGHT end of an already-constructed direct-BC dataset.

    Ghost node rule
    ---------------
    If the current last node is at (x_last, y_last, z_last), the appended ghost node is:
        x_ghost = x_last + ghost_edge_length
        y_ghost = 0
        z_ghost = z_last

    and this node is added to idx_b / xb as a prescribed boundary node.

    Notes
    -----
    - Assumes the current last node is already the rightmost physical node in qs.
    - Adds one new edge scalar DOF (set to 0 in qs and constrained to 0 in xb).
    - Adds one new node at the end, increasing dof by 4.
    """
    qs = np.asarray(qs)
    xb = np.asarray(xb)
    idx_b = np.asarray(idx_b, dtype=int)
    lambdas = np.asarray(lambdas)
    valid_new = None if valid is None else np.asarray(valid).copy()

    if qs.ndim != 3:
        raise ValueError(f"qs must have shape (n_traj, T, dof), got {qs.shape}")
    if xb.ndim != 3:
        raise ValueError(f"xb must have shape (n_traj, T, n_b), got {xb.shape}")
    if idx_b.ndim != 1:
        raise ValueError(f"idx_b must have shape (n_b,), got {idx_b.shape}")

    n_traj, T, dof_old = qs.shape
    if xb.shape[:2] != (n_traj, T):
        raise ValueError(f"xb shape {xb.shape} incompatible with qs shape {qs.shape}")
    if xb.shape[2] != len(idx_b):
        raise ValueError(
            f"xb.shape[2]={xb.shape[2]} must match len(idx_b)={len(idx_b)}"
        )

    if (dof_old + 1) % 4 != 0:
        raise ValueError(
            f"dof_old={dof_old} is not compatible with dof = 4*n_nodes - 1"
        )

    n_nodes_old = (dof_old + 1) // 4
    n_nodes_new = n_nodes_old + 1
    dof_new = 4 * n_nodes_new - 1

    qs_new = np.zeros((n_traj, T, dof_new), dtype=qs.dtype)
    qs_new[:, :, :dof_old] = qs

    # old last node location
    old_last_node_start = 4 * (n_nodes_old - 1)
    x_last = qs[:, :, old_last_node_start + 0]
    z_last = qs[:, :, old_last_node_start + 2]

    # appended edge scalar DOF sits at old dof_old index, already zero
    # appended new last node starts at dof_old + 1
    new_last_node_start = dof_old + 1
    qs_new[:, :, new_last_node_start + 0] = x_last + ghost_edge_length
    qs_new[:, :, new_last_node_start + 1] = 0.0
    qs_new[:, :, new_last_node_start + 2] = z_last

    idx_b_new_list = list(idx_b.tolist())
    xb_new_cols = [xb[:, :, j].copy() for j in range(xb.shape[2])]

    def add_or_replace_bc(idx, values):
        values = np.asarray(values, dtype=xb.dtype)
        if values.shape != (n_traj, T):
            raise ValueError(
                f"BC values for dof {idx} must have shape {(n_traj, T)}, got {values.shape}"
            )
        if idx in idx_b_new_list:
            j = idx_b_new_list.index(idx)
            xb_new_cols[j] = values
        else:
            idx_b_new_list.append(int(idx))
            xb_new_cols.append(values)

    zeros = np.zeros((n_traj, T), dtype=xb.dtype)

    # Constrain the newly added edge scalar DOF to zero
    add_or_replace_bc(dof_old, zeros)

    # Constrain new ghost node position
    add_or_replace_bc(new_last_node_start + 0, x_last + ghost_edge_length)
    add_or_replace_bc(new_last_node_start + 1, zeros)
    add_or_replace_bc(new_last_node_start + 2, z_last)

    idx_b_new = np.asarray(idx_b_new_list, dtype=int)
    xb_new = np.stack(xb_new_cols, axis=2)
    lambdas_new = lambdas.copy()

    if check_consistency:
        if qs_new.shape != (n_traj, T, dof_new):
            raise RuntimeError(f"Unexpected qs_new shape: {qs_new.shape}")
        if xb_new.shape[:2] != (n_traj, T):
            raise RuntimeError(f"Unexpected xb_new shape: {xb_new.shape}")
        if xb_new.shape[2] != len(idx_b_new):
            raise RuntimeError(
                f"xb_new.shape[2]={xb_new.shape[2]} != len(idx_b_new)={len(idx_b_new)}"
            )

        if not np.all(np.isfinite(qs_new)):
            raise ValueError("qs_new contains NaN/Inf")
        if not np.all(np.isfinite(xb_new)):
            raise ValueError("xb_new contains NaN/Inf")
        if not np.all(np.isfinite(lambdas_new)):
            raise ValueError("lambdas_new contains NaN/Inf")
        if valid_new is not None and not np.all(np.isfinite(valid_new.astype(float))):
            raise ValueError("valid_new contains NaN/Inf")

    return qs_new, xb_new, idx_b_new, lambdas_new, valid_new


def prescribe_last_node_positions_from_qs(
    qs,
    xb,
    idx_b,
    n_last_nodes=2,
    check_consistency=True,
):
    """
    Add direct BCs for the translational DOFs of the last `n_last_nodes` nodes.

    Values are copied from the corresponding qs columns. Existing idx_b entries
    are left in place, with their xb columns replaced by qs values.
    """
    qs = np.asarray(qs)
    xb = np.asarray(xb)
    idx_b = np.asarray(idx_b, dtype=int)

    if qs.ndim != 3:
        raise ValueError(f"qs must have shape (n_traj, T, dof), got {qs.shape}")
    if xb.ndim != 3:
        raise ValueError(f"xb must have shape (n_traj, T, n_b), got {xb.shape}")
    if idx_b.ndim != 1:
        raise ValueError(f"idx_b must have shape (n_b,), got {idx_b.shape}")
    if xb.shape[:2] != qs.shape[:2]:
        raise ValueError(f"xb shape {xb.shape} incompatible with qs shape {qs.shape}")
    if xb.shape[2] != len(idx_b):
        raise ValueError(
            f"xb.shape[2]={xb.shape[2]} must match len(idx_b)={len(idx_b)}"
        )
    if (qs.shape[2] + 1) % 4 != 0:
        raise ValueError(
            f"dof={qs.shape[2]} is not compatible with dof = 4*n_nodes - 1"
        )

    n_nodes = (qs.shape[2] + 1) // 4
    if n_last_nodes < 1:
        raise ValueError(f"n_last_nodes must be >= 1, got {n_last_nodes}")
    if n_last_nodes > n_nodes:
        raise ValueError(
            f"n_last_nodes={n_last_nodes} cannot exceed n_nodes={n_nodes}"
        )

    idx_b_new_list = list(idx_b.tolist())
    xb_new_cols = [xb[:, :, j].copy() for j in range(xb.shape[2])]
    added_idx = []
    replaced_idx = []

    for node_id in range(n_nodes - n_last_nodes, n_nodes):
        for component in range(3):
            idx = 4 * node_id + component
            values = qs[:, :, idx].copy()
            if idx in idx_b_new_list:
                j = idx_b_new_list.index(idx)
                xb_new_cols[j] = values
                replaced_idx.append(int(idx))
            else:
                idx_b_new_list.append(int(idx))
                xb_new_cols.append(values)
                added_idx.append(int(idx))

    idx_b_new = np.asarray(idx_b_new_list, dtype=int)
    xb_new = np.stack(xb_new_cols, axis=2)

    if not added_idx:
        warnings.warn(
            "prescribe_last_node_positions_from_qs did not add any new idx_b "
            f"entries because the final {n_last_nodes} nodes' translational DOFs "
            f"were already prescribed: {replaced_idx}. Existing xb columns were "
            "refreshed from qs.",
            UserWarning,
            stacklevel=2,
        )

    if check_consistency:
        if xb_new.shape[2] != len(idx_b_new):
            raise RuntimeError(
                f"xb_new.shape[2]={xb_new.shape[2]} != len(idx_b_new)={len(idx_b_new)}"
            )
        if not np.all(np.isfinite(xb_new)):
            raise ValueError("xb_new contains NaN/Inf")

    return xb_new, idx_b_new


def extract_directbc_dataset(
    input_json_path,
    output_npz_path,
    n_nodes=3,
    use_ee_as_last_node=True,
    drop_first_points=1,
    reverse_trajectory=False,
    make_plots=False,
    prepend_clamped_node_flag=False,
    append_right_ghost_clamp_flag=False,
    prescribe_last_two_nodes_flag=False,
    min_marker_presence_ratio=0.8,
):
    """
    Build direct-BC dataset with format:
        qs      : (n_traj, T, dof)
        xb      : (n_traj, T, n_b)
        idx_b   : (n_b,)
        lambdas : (1, T)
        valid   : (n_traj, T)

    Supported modes
    ---------------
    1) n_nodes = 3
       Use:
         - node 0 = fixed origin (from the highest-index persistent marker)
         - node 1 = center marker among the remaining persistent markers
         - node 2 = either:
             * end-effector        if use_ee_as_last_node=True
             * marker_0            if use_ee_as_last_node=False

    2) n_nodes = "all" OR n_nodes = number of persistent markers in data
       Use:
         - node 0 = fixed origin (highest-index persistent marker)
         - intermediate nodes depend on last-node choice
         - final node = either:
             * end-effector        if use_ee_as_last_node=True
             * marker_0            if use_ee_as_last_node=False

    Notes
    -----
    - Rare spurious markers are discarded before selecting the fixed marker.
    - A marker is kept only if it appears in at least
      `min_marker_presence_ratio * n_samples` frames.

    Optional post-processing
    ------------------------
    - prepend_clamped_node_flag:
        prepend a fixed node to the left end

    - append_right_ghost_clamp_flag:
        append a ghost node to the right end, where
            x_ghost = x_last + edge_len
            z_ghost = z_last
        with edge_len computed from the 0th frame as the norm of the
        distance between marker_0 and marker_1 in transformed x-z coordinates.

    - prescribe_last_two_nodes_flag:
        add x/y/z direct BCs for the final two nodes, with xb values copied
        from the corresponding qs columns. Existing BC columns are reused.
    """

    # =========================================================
    # Load JSON lines
    # =========================================================
    data = []
    with open(input_json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(data) == 0:
        raise ValueError("No valid JSON samples found.")

    print(f"Loaded {len(data)} samples")

    # =========================================================
    # Parse timestamps
    # =========================================================
    timestamps = []
    for d in data:
        try:
            timestamps.append(datetime.fromisoformat(d["timestamp"]))
        except Exception:
            timestamps.append(None)
    timestamps = np.array(timestamps)

    # =========================================================
    # Extract raw EE pose (in mm)
    # =========================================================
    ee_x_raw, ee_y_raw, ee_z_raw = [], [], []
    for d in data:
        ee = d.get("ee_pose", {})
        ee_x_raw.append(ee.get("x_mm", np.nan))
        ee_y_raw.append(ee.get("y_mm", np.nan))
        ee_z_raw.append(ee.get("z_mm", np.nan))

    ee_x_raw = np.asarray(ee_x_raw, dtype=float)
    ee_y_raw = np.asarray(ee_y_raw, dtype=float)
    ee_z_raw = np.asarray(ee_z_raw, dtype=float)

    # =========================================================
    # Extract marker names and remove rare outliers
    # =========================================================
    marker_names_all = set()
    for d in data:
        marker_names_all.update(d.get("markers", {}).keys())
    marker_names_all = sorted(marker_names_all, key=_marker_index)

    if len(marker_names_all) == 0:
        raise ValueError("No markers found in the data.")

    marker_counts = {
        name: sum(name in d.get("markers", {}) for d in data)
        for name in marker_names_all
    }

    marker_names = [
        name for name in marker_names_all
        if marker_counts[name] / len(data) >= min_marker_presence_ratio
    ]
    marker_names = sorted(marker_names, key=_marker_index)

    discarded_markers = [
        name for name in marker_names_all
        if name not in marker_names
    ]

    if len(marker_names) == 0:
        raise ValueError(
            "No markers remain after filtering rare/outlier markers. "
            f"Try lowering min_marker_presence_ratio (current value: {min_marker_presence_ratio})."
        )

    print("Persistent markers kept:", marker_names)
    if len(discarded_markers) > 0:
        print("Discarded rare/outlier markers:", discarded_markers)

    n_markers = len(marker_names)

    if "marker_0" not in marker_names:
        raise ValueError(
            "marker_0 must survive persistence filtering because EE alignment uses marker_0. "
            f"Try lowering min_marker_presence_ratio (current value: {min_marker_presence_ratio})."
        )
    if append_right_ghost_clamp_flag and "marker_1" not in marker_names:
        raise ValueError(
            "marker_1 must survive persistence filtering when append_right_ghost_clamp_flag=True. "
            f"Try lowering min_marker_presence_ratio (current value: {min_marker_presence_ratio})."
        )

    # Highest-index persistent marker = fixed marker defining the origin
    fixed_marker = marker_names[-1]

    # Resolve n_nodes mode
    if n_nodes == "all":
        target_n_nodes = n_markers
    elif isinstance(n_nodes, int):
        if n_nodes == 3:
            target_n_nodes = 3
        elif n_nodes == n_markers:
            target_n_nodes = n_markers
        else:
            raise ValueError(
                f"Unsupported n_nodes={n_nodes}. After filtering persistent markers, "
                f"supported values are 3, 'all', or {n_markers}."
            )
    else:
        raise ValueError(
            f"n_nodes must be 3, 'all', or equal to the number of persistent markers in the data. Got {n_nodes}."
        )

    # =========================================================
    # Choose markers used in the state
    # =========================================================
    dynamic_marker_candidates = [
        m for m in marker_names
        if (m != fixed_marker and m != "marker_0")
    ]
    dynamic_marker_candidates = sorted(
        dynamic_marker_candidates,
        key=_marker_index,
        reverse=True,
    )

    if target_n_nodes == 3:
        if len(dynamic_marker_candidates) == 0:
            raise ValueError("Not enough persistent markers to build a 3-node model.")
        center_marker = dynamic_marker_candidates[len(dynamic_marker_candidates) // 2]
        state_markers = [center_marker]
    else:
        state_markers = dynamic_marker_candidates

    required_markers = sorted(
        set([fixed_marker, "marker_0"] + state_markers),
        key=_marker_index,
    )

    print("Fixed marker used for origin:", fixed_marker)
    print("Markers used in state:", state_markers)
    if use_ee_as_last_node:
        print("Final node = end-effector (marker_0 replaced)")
    else:
        print("Final node = marker_0 (no end-effector used in state/BCs)")

    for m in required_markers:
        if m not in marker_names:
            raise ValueError(f"Required marker '{m}' not found in filtered marker set.")

    # =========================================================
    # Collect marker time series
    # =========================================================
    marker_data = {
        name: {"x": [], "y": [], "z": []}
        for name in marker_names
    }

    for d in data:
        markers = d.get("markers", {})
        for name in marker_names:
            if name in markers:
                marker_data[name]["x"].append(markers[name].get("x", np.nan))
                marker_data[name]["y"].append(markers[name].get("y", np.nan))
                marker_data[name]["z"].append(markers[name].get("z", np.nan))
            else:
                marker_data[name]["x"].append(np.nan)
                marker_data[name]["y"].append(np.nan)
                marker_data[name]["z"].append(np.nan)

    for name in marker_names:
        for c in ["x", "y", "z"]:
            marker_data[name][c] = np.asarray(marker_data[name][c], dtype=float)

    # =========================================================
    # Initial raw plots
    # =========================================================
    if make_plots:
        _plot_marker_pair(
            marker_data,
            coord1="x",
            coord2="y",
            title="Raw marker trajectories (x vs y)",
            xlabel="x",
            ylabel="y",
        )
        _plot_ee_pair(
            ee_y_raw,
            ee_z_raw,
            title="Raw end-effector trajectory (y vs z)",
            xlabel="y_raw [mm]",
            ylabel="z_raw [mm]",
        )

    # =========================================================
    # Define origin from fixed marker mean position
    # =========================================================
    origin_x = np.nanmean(marker_data[fixed_marker]["x"])
    origin_y = np.nanmean(marker_data[fixed_marker]["y"])
    origin_z = np.nanmean(marker_data[fixed_marker]["z"])

    print("Origin from fixed marker mean:")
    print("x0 =", origin_x)
    print("y0 =", origin_y)
    print("z0 =", origin_z)

    # =========================================================
    # Remove global offset from all markers
    # =========================================================
    marker_data_centered = {}
    for name in marker_names:
        marker_data_centered[name] = {
            "x": marker_data[name]["x"] - origin_x,
            "y": marker_data[name]["y"] - origin_y,
            "z": marker_data[name]["z"] - origin_z,
        }

    # =========================================================
    # Coordinate transform for markers
    #   new_x = -old_x
    #   new_y =  old_z
    #   new_z = -old_y
    # =========================================================
    for name in marker_names:
        old_x = marker_data_centered[name]["x"].copy()
        old_y = marker_data_centered[name]["y"].copy()
        old_z = marker_data_centered[name]["z"].copy()

        marker_data_centered[name]["x"] = -old_x
        marker_data_centered[name]["y"] = old_z
        marker_data_centered[name]["z"] = -old_y

    # =========================================================
    # EE transform
    #   ee_x = -ee_y_raw
    #   ee_y =  ee_x_raw
    #   ee_z =  ee_z_raw
    #   convert mm -> m
    # =========================================================
    ee_x = -ee_y_raw / 1000.0
    ee_y = ee_x_raw / 1000.0
    ee_z = ee_z_raw / 1000.0

    # =========================================================
    # Align EE trajectory to transformed marker_0 first point
    # =========================================================
    ee_x = ee_x - ee_x[0] + marker_data_centered["marker_0"]["x"][0]
    ee_y = ee_y - ee_y[0] + marker_data_centered["marker_0"]["y"][0]
    ee_z = ee_z - ee_z[0] + marker_data_centered["marker_0"]["z"][0]

    # =========================================================
    # First transformed planar plots: x-z
    # =========================================================
    if make_plots:
        _plot_marker_pair(
            marker_data_centered,
            coord1="x",
            coord2="z",
            title="Markers after transform and centering (x vs z)",
            xlabel="x",
            ylabel="z",
        )
        _plot_ee_pair(
            ee_x,
            ee_z,
            title="EE after transform and alignment (x vs z)",
            xlabel="x [m]",
            ylabel="z [m]",
        )

    # =========================================================
    # Remove first point
    # =========================================================
    if drop_first_points > 0:
        for name in marker_names:
            marker_data_centered[name]["x"] = marker_data_centered[name]["x"][drop_first_points:]
            marker_data_centered[name]["y"] = marker_data_centered[name]["y"][drop_first_points:]
            marker_data_centered[name]["z"] = marker_data_centered[name]["z"][drop_first_points:]

        ee_x = ee_x[drop_first_points:]
        ee_y = ee_y[drop_first_points:]
        ee_z = ee_z[drop_first_points:]

        if make_plots:
            _plot_marker_pair(
                marker_data_centered,
                coord1="x",
                coord2="z",
                title="Markers after dropping first point (x vs z)",
                xlabel="x",
                ylabel="z",
            )
            _plot_ee_pair(
                ee_x,
                ee_z,
                title="EE after dropping first point (x vs z)",
                xlabel="x [m]",
                ylabel="z [m]",
            )

    # =========================================================
    # Reverse trajectory
    # =========================================================
    if reverse_trajectory:
        for name in marker_names:
            marker_data_centered[name]["x"] = marker_data_centered[name]["x"][::-1]
            marker_data_centered[name]["y"] = marker_data_centered[name]["y"][::-1]
            marker_data_centered[name]["z"] = marker_data_centered[name]["z"][::-1]

        ee_x = ee_x[::-1]
        ee_y = ee_y[::-1]
        ee_z = ee_z[::-1]

        if make_plots:
            _plot_marker_pair(
                marker_data_centered,
                coord1="x",
                coord2="z",
                title="Markers after trajectory reversal (x vs z)",
                xlabel="x",
                ylabel="z",
            )
            _plot_ee_pair(
                ee_x,
                ee_z,
                title="EE after trajectory reversal (x vs z)",
                xlabel="x [m]",
                ylabel="z [m]",
            )

    # =========================================================
    # Final clean combined plot: x-z
    # =========================================================
    if make_plots:
        if use_ee_as_last_node:
            last_x = ee_x
            last_z = ee_z
            last_node_label = "EE"
        else:
            last_x = marker_data_centered["marker_0"]["x"]
            last_z = marker_data_centered["marker_0"]["z"]
            last_node_label = "marker_0"

        _plot_markers_and_lastnode_final(
            marker_data_centered,
            last_x=last_x,
            last_z=last_z,
            last_node_label=last_node_label,
            title="Final transformed marker + last-node trajectories (x vs z)",
        )

    # =========================================================
    # Remove timesteps with NaNs
    # Check only quantities actually used in the final state
    # =========================================================
    if use_ee_as_last_node:
        mask = np.isfinite(ee_x) & np.isfinite(ee_z)
    else:
        mask = (
            np.isfinite(marker_data_centered["marker_0"]["x"]) &
            np.isfinite(marker_data_centered["marker_0"]["z"])
        )

    for m in [fixed_marker, "marker_0"] + state_markers:
        mask &= np.isfinite(marker_data_centered[m]["x"])
        mask &= np.isfinite(marker_data_centered[m]["z"])

    n_before = len(mask)
    n_after = np.sum(mask)
    print(f"Removing NaNs: kept {n_after}/{n_before} timesteps")

    if n_after == 0:
        raise ValueError("All data points are NaN after filtering.")

    ee_x = ee_x[mask]
    ee_y = ee_y[mask]
    ee_z = ee_z[mask]

    for name in marker_names:
        for c in ["x", "y", "z"]:
            marker_data_centered[name][c] = marker_data_centered[name][c][mask]

    # =========================================================
    # Sanity check lengths
    # =========================================================
    T = len(ee_x)
    for m in [fixed_marker, "marker_0"] + state_markers:
        for c in ["x", "y", "z"]:
            if len(marker_data_centered[m][c]) != T:
                raise ValueError(f"Length mismatch in {m}_{c}")

    # =========================================================
    # Compute right ghost edge length from 0th frame
    # using marker_0 and marker_1 in transformed x-z plane
    # =========================================================
    if append_right_ghost_clamp_flag:
        marker0_x0 = marker_data_centered["marker_0"]["x"][0]
        marker0_z0 = marker_data_centered["marker_0"]["z"][0]
        marker1_x0 = marker_data_centered["marker_1"]["x"][0]
        marker1_z0 = marker_data_centered["marker_1"]["z"][0]
        right_ghost_edge_length = np.sqrt(
            (marker0_x0 - marker1_x0) ** 2 + (marker0_z0 - marker1_z0) ** 2
        )
        print("Right ghost edge length from frame 0 (marker_0 to marker_1):", right_ghost_edge_length)

    # =========================================================
    # Build qs programmatically
    # =========================================================
    qs_cols = []

    # Node 0 = fixed origin
    qs_cols.extend([
        np.zeros(T),  # x0
        np.zeros(T),  # y0
        np.zeros(T),  # z0
    ])

    # Add edge DOF after node 0
    if target_n_nodes >= 2:
        qs_cols.append(np.zeros(T))  # th0

    # Intermediate dynamic markers
    for marker_name in state_markers:
        qs_cols.extend([
            marker_data_centered[marker_name]["x"],
            np.zeros(T),
            marker_data_centered[marker_name]["z"],
        ])
        qs_cols.append(np.zeros(T))

    # Final node = EE or marker_0
    if use_ee_as_last_node:
        final_x = ee_x
        final_z = ee_z
    else:
        final_x = marker_data_centered["marker_0"]["x"]
        final_z = marker_data_centered["marker_0"]["z"]

    qs_cols.extend([
        final_x,
        np.zeros(T),
        final_z,
    ])

    qs = np.column_stack(qs_cols)

    # =========================================================
    # Build idx_b and xb programmatically
    # =========================================================
    idx_b = [0, 1, 2, 3]

    for i in range(1, target_n_nodes - 1):
        idx_b.append(4 * i + 3)

    final_node_start = 4 * (target_n_nodes - 1)
    idx_b.extend([final_node_start, final_node_start + 1, final_node_start + 2])

    idx_b = np.array(idx_b, dtype=int)

    xb_cols = []
    for idx in idx_b:
        if idx == final_node_start:
            xb_cols.append(final_x)
        elif idx == final_node_start + 1:
            xb_cols.append(np.zeros(T))
        elif idx == final_node_start + 2:
            xb_cols.append(final_z)
        else:
            xb_cols.append(np.zeros(T))

    xb = np.column_stack(xb_cols)

    # =========================================================
    # Lambdas
    # =========================================================
    lambdas = np.linspace(0.0, 1.0, T)[None, ...]

    # =========================================================
    # Add trajectory dimension
    # =========================================================
    qs = qs[None, :, :]
    xb = xb[None, :, :]
    valid = np.full((qs.shape[0], qs.shape[1]), True)

    # =========================================================
    # Optional left clamp prepend
    # =========================================================
    if prepend_clamped_node_flag:
        first_edge_length = np.linalg.norm(qs[0, 0, 0:3] - qs[0, 0, 4:7])
        qs, xb, idx_b, lambdas, valid = prepend_clamped_node(
            qs=qs,
            xb=xb,
            idx_b=idx_b,
            lambdas=lambdas,
            valid=valid,
            first_edge_length=first_edge_length,
            check_consistency=True,
        )
        target_n_nodes += 1

    # =========================================================
    # Optional right ghost clamp append
    # =========================================================
    if append_right_ghost_clamp_flag:
        qs, xb, idx_b, lambdas, valid = append_right_ghost_clamp(
            qs=qs,
            xb=xb,
            idx_b=idx_b,
            lambdas=lambdas,
            valid=valid,
            ghost_edge_length=right_ghost_edge_length,
            check_consistency=True,
        )
        target_n_nodes += 1

    # =========================================================
    # Optional prescribed motion on the final two nodes
    # =========================================================
    if prescribe_last_two_nodes_flag:
        xb, idx_b = prescribe_last_node_positions_from_qs(
            qs=qs,
            xb=xb,
            idx_b=idx_b,
            n_last_nodes=2,
            check_consistency=True,
        )

    # =========================================================
    # Final checks
    # =========================================================
    def assert_no_nans(name, arr):
        if not np.all(np.isfinite(arr)):
            n_nan = np.isnan(arr).sum()
            n_inf = np.isinf(arr).sum()
            raise ValueError(
                f"{name} contains NaNs/Infs | NaNs: {n_nan}, Infs: {n_inf}, shape: {arr.shape}"
            )

    assert_no_nans("qs", qs)
    assert_no_nans("xb", xb)
    assert_no_nans("lambdas", lambdas)
    assert_no_nans("valid", valid.astype(float))

    expected_dof = 4 * target_n_nodes - 1
    if qs.shape[2] != expected_dof:
        raise ValueError(
            f"Constructed qs has dof={qs.shape[2]}, expected {expected_dof} for n_nodes={target_n_nodes}."
        )

    np.savez(
        output_npz_path,
        qs=qs,
        xb=xb,
        idx_b=idx_b,
        lambdas=lambdas,
        valid=valid,
    )

    print("target_n_nodes:", target_n_nodes)
    print("use_ee_as_last_node:", use_ee_as_last_node)
    print("prepend_clamped_node_flag:", prepend_clamped_node_flag)
    print("append_right_ghost_clamp_flag:", append_right_ghost_clamp_flag)
    print("prescribe_last_two_nodes_flag:", prescribe_last_two_nodes_flag)
    print("min_marker_presence_ratio:", min_marker_presence_ratio)
    print("qs shape:", qs.shape)
    print("xb shape:", xb.shape)
    print("idx_b:", idx_b)
    print("lambdas shape:", lambdas.shape)

    return qs, xb, idx_b, lambdas

def plot_qs_snapshots(
    qs,
    traj_idx=0,
    title=None,
    every_step=1,
    n_nodes=None,
    node_labels=None,
    show_node_labels=True,
    use_ee_as_last_node=True,
):
    """
    Plot rod configurations at multiple time steps.

    Parameters
    ----------
    qs : array-like
        Shape (T, dof) or (n_traj, T, dof)

    traj_idx : int
        Which trajectory to plot if qs has shape (n_traj, T, dof)

    title : str or None
        Plot title

    every_step : int
        Plot every `every_step`-th timestep

    n_nodes : int or None
        Number of nodes. If None, infer from dof using:
            dof = 4*n_nodes - 1

    node_labels : list[str] or None
        Optional labels for the nodes, length must equal n_nodes.

    show_node_labels : bool
        Whether to annotate the start and end configurations with node labels.

    use_ee_as_last_node : bool
        Used only for default node labels.
    """
    qs = np.asarray(qs)

    if qs.ndim == 2:
        q = qs
    elif qs.ndim == 3:
        q = qs[traj_idx]
    else:
        raise ValueError("qs must have shape (T, dof) or (n_traj, T, dof)")

    T, dof = q.shape

    if n_nodes is None:
        if (dof + 1) % 4 != 0:
            raise ValueError(
                f"Cannot infer n_nodes from dof={dof}. Expected dof = 4*n_nodes - 1."
            )
        n_nodes = (dof + 1) // 4

    expected_dof = 4 * n_nodes - 1
    if dof != expected_dof:
        raise ValueError(
            f"Inconsistent n_nodes={n_nodes} for dof={dof}. Expected dof={expected_dof}."
        )

    if node_labels is None:
        node_labels = [f"node_{i}" for i in range(n_nodes)]
        node_labels[0] = "fixed"
        if n_nodes >= 2:
            node_labels[-1] = "EE" if use_ee_as_last_node else "marker_0"

    if len(node_labels) != n_nodes:
        raise ValueError(
            f"node_labels must have length {n_nodes}, got {len(node_labels)}."
        )

    x_idx = [4 * i for i in range(n_nodes)]
    z_idx = [4 * i + 2 for i in range(n_nodes)]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for t in range(0, T, every_step):
        x = q[t, x_idx]
        z = q[t, z_idx]
        ax.plot(x, z, "-o", alpha=0.25)

    x_start = q[0, x_idx]
    z_start = q[0, z_idx]
    x_end = q[-1, x_idx]
    z_end = q[-1, z_idx]

    ax.plot(x_start, z_start, "-o", linewidth=2.5, label="Start")
    ax.plot(x_end, z_end, "-o", linewidth=2.5, label="End")

    if show_node_labels:
        for i, label in enumerate(node_labels):
            ax.text(
                x_start[i],
                z_start[i],
                f"{label}",
                fontsize=9,
                ha="right",
                va="bottom",
            )
            ax.text(
                x_end[i],
                z_end[i],
                f"{label}",
                fontsize=9,
                ha="left",
                va="top",
            )

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if title is None:
        last_name = "EE" if use_ee_as_last_node else "marker_0"
        title = f"Rod snapshots from qs (trajectory {traj_idx}, n_nodes={n_nodes}, last={last_name})"
    ax.set_title(title)

    ax.legend()
    plt.tight_layout()
    plt.show()
