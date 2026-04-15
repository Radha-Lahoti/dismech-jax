import json
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


def extract_directbc_dataset(
    input_json_path,
    output_npz_path,
    drop_first_point=True,
    reverse_trajectory=True,
    make_plots=False,
):
    """
    Build direct-BC dataset with format:
        qs      : (n_traj, T, dof)
        xb      : (n_traj, T, n_b)
        idx_b   : (n_b,)
        lambdas : (T,)

    Plotting behavior when make_plots=True
    --------------------------------------
    Initial raw plots:
      1. Raw markers: x vs y
      2. Raw EE: y vs z

    After transformation and every later processing step:
      - markers plotted separately in x-z plane
      - EE plotted separately in x-z plane

    Final:
      - one combined clean x-z plot with all markers + EE
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
    # Extract marker data
    # =========================================================
    marker_names = set()
    for d in data:
        marker_names.update(d.get("markers", {}).keys())
    marker_names = sorted(marker_names)

    required_markers = ["marker_4", "marker_2", "marker_0"]
    for m in required_markers:
        if m not in marker_names:
            raise ValueError(f"Required marker '{m}' not found in data.")

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
    # markers: x-y
    # EE: y-z
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
    # Define origin from marker_4 mean position
    # =========================================================
    ref_name = "marker_4"
    origin_x = np.nanmean(marker_data[ref_name]["x"])
    origin_y = np.nanmean(marker_data[ref_name]["y"])
    origin_z = np.nanmean(marker_data[ref_name]["z"])

    print("Origin from marker_4 mean:")
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
    # convert mm -> m
    # =========================================================
    ee_x = -ee_y_raw / 1000.0
    ee_y =  ee_x_raw / 1000.0
    ee_z =  ee_z_raw / 1000.0

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
    if drop_first_point:
        for name in marker_names:
            marker_data_centered[name]["x"] = marker_data_centered[name]["x"][1:]
            marker_data_centered[name]["y"] = marker_data_centered[name]["y"][1:]
            marker_data_centered[name]["z"] = marker_data_centered[name]["z"][1:]

        ee_x = ee_x[1:]
        ee_y = ee_y[1:]
        ee_z = ee_z[1:]

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
        _plot_markers_and_ee_final(
            marker_data_centered,
            ee_x=ee_x,
            ee_z=ee_z,
            title="Final transformed marker + EE trajectories (x vs z)",
        )

    # =========================================================
    # Sanity check lengths
    # =========================================================
    T = len(ee_x)
    for m in required_markers:
        for c in ["x", "y", "z"]:
            if len(marker_data_centered[m][c]) != T:
                raise ValueError(f"Length mismatch in {m}_{c}")

    # =========================================================
    # Build qs
    # [x0,y0,z0,th0, x1,y1,z1,th1, x2,y2,z2]
    # =========================================================
    qs = np.column_stack([
        np.zeros(T),                             # x0
        np.zeros(T),                             # y0
        np.zeros(T),                             # z0
        np.zeros(T),                             # th0

        marker_data_centered["marker_2"]["x"],  # x1
        np.zeros(T),                             # y1
        marker_data_centered["marker_2"]["z"],  # z1
        np.zeros(T),                             # th1

        ee_x,                                    # x2
        np.zeros(T),                             # y2
        ee_z,                                    # z2
    ])

    # =========================================================
    # Direct BCs
    # =========================================================
    xb = np.column_stack([
        np.zeros(T),  # x0
        np.zeros(T),  # y0
        np.zeros(T),  # z0
        np.zeros(T),  # th0
        np.zeros(T),  # th1
        ee_x,         # x2
        np.zeros(T),  # y2
        ee_z,         # z2
    ])

    idx_b = np.array([0, 1, 2, 3, 7, 8, 9, 10], dtype=int)

    # =========================================================
    # Lambdas
    # =========================================================
    lambdas = np.linspace(0.0, 1.0, T)[None, ...]

    # =========================================================
    # Add trajectory dimension
    # =========================================================
    qs = qs[None, :, :]
    xb = xb[None, :, :]

    # All valid for single dataset
    valid = np.full((qs.shape[0], qs.shape[1]), True)
    # =========================================================
    # Save
    # =========================================================
    np.savez(
        output_npz_path,
        qs=qs,
        xb=xb,
        idx_b=idx_b,
        lambdas=lambdas,
        valid=valid,
    )

    print("qs shape:", qs.shape)
    print("xb shape:", xb.shape)
    print("idx_b:", idx_b)
    print("lambdas shape:", lambdas.shape)

    return qs, xb, idx_b, lambdas
def plot_qs_snapshots(qs, traj_idx=0, title=None, every_step=1):
    """
    Plot the rod configuration at multiple time steps.
    """
    qs = np.asarray(qs)

    if qs.ndim == 2:
        q = qs
    elif qs.ndim == 3:
        q = qs[traj_idx]
    else:
        raise ValueError("qs must have shape (T,11) or (n_traj,T,11)")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for t in range(0, q.shape[0], every_step):
        x = [q[t, 0], q[t, 4], q[t, 8]]
        z = [q[t, 2], q[t, 6], q[t, 10]]
        ax.plot(x, z, "-o", alpha=0.25)

    # Highlight first and last
    ax.plot([q[0, 0], q[0, 4], q[0, 8]],
            [q[0, 2], q[0, 6], q[0, 10]],
            "-o", linewidth=2.5, label="Start")

    ax.plot([q[-1, 0], q[-1, 4], q[-1, 8]],
            [q[-1, 2], q[-1, 6], q[-1, 10]],
            "-o", linewidth=2.5, label="End")

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if title is None:
        title = f"Rod snapshots from qs (trajectory {traj_idx})"
    ax.set_title(title)

    ax.legend()
    plt.tight_layout()
    plt.show()