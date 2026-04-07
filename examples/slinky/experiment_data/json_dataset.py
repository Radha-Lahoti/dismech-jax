"""
Convert slinky motion-capture JSON (step labels + marker dicts) into train/valid NPZ files.

The logic previously lived in duplicate cells inside ``create_train_data.ipynb`` and
``create_train_valid_data.ipynb``.  This module centralizes parsing, QC, and export.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# Matches labels like "y+ step 3/40" used to group time series in the experiment logs.
STEP_RE = re.compile(r"^(y\+|y\-) step (\d+)/(\d+)$")

# Three markers define the reduced configuration used in the 3-node slinky models.
DEFAULT_REQUIRED_MARKERS = ("marker_4", "marker_2", "marker_0")


def has_required_markers(entry: dict[str, Any], *, required: tuple[str, ...]) -> bool:
    markers = entry.get("markers", {})
    return all(name in markers for name in required)


def get_marker_xyz_raw(entry: dict[str, Any], marker_name: str) -> np.ndarray:
    markers = entry.get("markers", {})
    if marker_name not in markers:
        raise ValueError(
            f"{marker_name} not found in entry markers. "
            f"label={entry.get('label', '<no label>')}, "
            f"available={list(markers.keys())}"
        )
    m = markers[marker_name]
    return np.asarray([m["x"], m["y"], m["z"]], dtype=np.float64)


def remap_coordinates(p_raw: np.ndarray) -> np.ndarray:
    """
    Map camera/world coordinates into the simulation frame used in the notebooks:

    - ``x_new``: axis along the undeformed slinky (flipped raw ``x``)
    - ``y_new``: out-of-plane depth (set to 0 here)
    - ``z_new``: gravity direction (``-raw_y``)
    """
    return np.asarray([-p_raw[0], 0.0, -p_raw[1]], dtype=np.float64)


def get_marker_xyz(entry: dict[str, Any], marker_name: str) -> np.ndarray:
    return remap_coordinates(get_marker_xyz_raw(entry, marker_name))


def entry_to_q(entry: dict[str, Any], *, required_markers: tuple[str, ...]) -> np.ndarray:
    """
    Assemble the 11-DOF configuration vector for the 3-marker reduced model.

    Layout: ``[x4,y4,z4,0, x2,y2,z2,0, x0,y0,z0]`` with positions expressed
    relative to ``marker_4`` (first triple is therefore zeros except twist slot).
    """
    p4 = get_marker_xyz(entry, required_markers[0])
    p2 = get_marker_xyz(entry, required_markers[1])
    p0 = get_marker_xyz(entry, required_markers[2])

    p4_rel = p4 - p4
    p2_rel = p2 - p4
    p0_rel = p0 - p4

    return np.array(
        [
            p4_rel[0],
            p4_rel[1],
            p4_rel[2],
            0.0,
            p2_rel[0],
            p2_rel[1],
            p2_rel[2],
            0.0,
            p0_rel[0],
            p0_rel[1],
            p0_rel[2],
        ],
        dtype=np.float64,
    )


def extract_y_sweeps(data: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group JSON entries into complete ``y+`` / ``y-`` step sequences."""
    sweeps: list[tuple[str, list[dict[str, Any]]]] = []

    current: list[dict[str, Any]] = []
    current_dir: str | None = None
    expected_step: int | None = None
    expected_total: int | None = None

    for entry in data:
        label = entry.get("label", "")
        m = STEP_RE.match(label)
        if m is None:
            continue

        direction = m.group(1)
        step_idx = int(m.group(2))
        total = int(m.group(3))

        if step_idx == 1:
            current = [entry]
            current_dir = direction
            expected_step = 2
            expected_total = total
            continue

        if (
            current
            and current_dir == direction
            and expected_step is not None
            and expected_total is not None
            and step_idx == expected_step
            and total == expected_total
        ):
            current.append(entry)
            expected_step += 1

            if step_idx == total:
                sweeps.append((direction, current))
                current = []
                current_dir = None
                expected_step = None
                expected_total = None

    return sweeps


def first_local_min_index(signal: np.ndarray) -> int:
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")
    n = signal.shape[0]
    if n < 3:
        return 0

    for i in range(1, n - 1):
        if signal[i] <= signal[i - 1] and signal[i] <= signal[i + 1]:
            return i

    return int(np.argmin(signal))


def trim_first_yplus_initial_dip(
    sweeps: list[tuple[str, list[dict[str, Any]]]],
    *,
    required_markers: tuple[str, ...],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """
    Drop samples before the first local minimum of ``marker_0``'s ``z`` coordinate
    on the first ``y+`` sweep (removes transient settling in some captures).
    """
    out: list[tuple[str, list[dict[str, Any]]]] = []
    trimmed = False

    for direction, sweep in sweeps:
        if (not trimmed) and direction == "y+":
            qs = np.stack([entry_to_q(entry, required_markers=required_markers) for entry in sweep], axis=0)
            z0 = qs[:, 10]
            dip_idx = first_local_min_index(z0)
            sweep = sweep[dip_idx:]
            trimmed = True
        out.append((direction, sweep))

    return out


def build_direct_bc_from_qs(qs: np.ndarray) -> np.ndarray:
    """
    Pack prescribed DOFs for ``BatchedDirectBC``.

    ``idx_b = [0,1,2,3,7,8,9,10]``; each row of ``xb`` is the prescribed values at one load step.
    """
    xb = np.zeros((qs.shape[0], 8), dtype=np.float64)

    xb[:, 0] = qs[:, 0]
    xb[:, 1] = qs[:, 1]
    xb[:, 2] = qs[:, 2]
    xb[:, 3] = qs[:, 3]
    xb[:, 4] = qs[:, 7]
    xb[:, 5] = qs[:, 8]
    xb[:, 6] = qs[:, 9]
    xb[:, 7] = qs[:, 10]

    return xb


@dataclass
class SlinkyJsonDatasetConfig:
    """User-tunable knobs for JSON → NPZ conversion."""

    required_markers: tuple[str, ...] = field(default_factory=lambda: DEFAULT_REQUIRED_MARKERS)
    trim_first_yplus_dip: bool = False
    filter_incomplete_marker_entries: bool = True
    train_valid_split: str = "alternate"
    # Even-index sweeps → train, odd → valid when split == "alternate".


class SlinkyJsonDatasetBuilder:
    """
    End-to-end builder: load JSON list, validate sweeps, write ``train.npz`` / ``valid.npz``.

    Example
    -------
    >>> builder = SlinkyJsonDatasetBuilder()
    >>> builder.write_train_valid_npz("capture.json", Path("train.npz"), Path("valid.npz"))
    """

    def __init__(self, config: SlinkyJsonDatasetConfig | None = None) -> None:
        self.config = config or SlinkyJsonDatasetConfig()

    def load_json_list(self, json_file: str | Path) -> list[dict[str, Any]]:
        json_file = Path(json_file)
        with open(json_file, "r") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Expected JSON file to contain a nonempty list of entries.")
        return data

    def sweeps_from_data(self, data: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
        sweeps = extract_y_sweeps(data)
        if len(sweeps) == 0:
            raise ValueError("No complete y+ / y- sweeps found.")
        print(f"Found {len(sweeps)} raw y+/y- sweeps.")

        if not self.config.filter_incomplete_marker_entries:
            return sweeps

        req = self.config.required_markers
        filtered: list[tuple[str, list[dict[str, Any]]]] = []
        for i, (direction, sweep) in enumerate(sweeps):
            bad = [j for j, entry in enumerate(sweep) if not has_required_markers(entry, required=req)]
            if bad:
                print(
                    f"Skipping sweep {i} ({direction}) because entries "
                    f"{[b + 1 for b in bad]} are missing required markers."
                )
                continue
            filtered.append((direction, sweep))

        if len(filtered) == 0:
            raise ValueError("No sweeps remain after filtering missing-marker entries.")

        print(f"Remaining valid sweeps: {len(filtered)}")
        return filtered

    def write_train_valid_npz(
        self,
        json_file: str | Path,
        train_file: str | Path = "train.npz",
        valid_file: str | Path = "valid.npz",
    ) -> None:
        """Parse ``json_file`` and write train/validation NPZ archives."""
        data = self.load_json_list(json_file)
        sweeps = self.sweeps_from_data(data)

        if self.config.trim_first_yplus_dip:
            sweeps = trim_first_yplus_initial_dip(sweeps, required_markers=self.config.required_markers)

        lengths = [len(sweep) for _, sweep in sweeps]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Sweep lengths are not all equal after extraction/trimming: {lengths}. "
                "Either disable trimming or crop manually."
            )

        req = self.config.required_markers
        qs_list: list[np.ndarray] = []
        xb_list: list[np.ndarray] = []
        dirs: list[str] = []

        for i, (direction, sweep) in enumerate(sweeps):
            qs = np.stack([entry_to_q(entry, required_markers=req) for entry in sweep], axis=0)
            xb = build_direct_bc_from_qs(qs)
            qs_list.append(qs)
            xb_list.append(xb)
            dirs.append(direction)
            print(f"Sweep {i}: direction={direction}, T={qs.shape[0]}, q_dim={qs.shape[1]}")

        qs_arr = np.stack(qs_list, axis=0)
        xb_arr = np.stack(xb_list, axis=0)
        print(f"qs_arr.shape = {qs_arr.shape}")
        print(f"xb_arr.shape = {xb_arr.shape}")

        idx_b = np.array([0, 1, 2, 3, 7, 8, 9, 10], dtype=np.int32)
        lambdas = np.linspace(0.0, 1.0, qs_arr.shape[1])

        if self.config.train_valid_split != "alternate":
            raise ValueError(f"Unsupported train_valid_split: {self.config.train_valid_split!r}")

        train_idx = list(range(0, len(sweeps), 2))
        valid_idx = list(range(1, len(sweeps), 2))

        train_qs = qs_arr[train_idx]
        train_xb = xb_arr[train_idx]
        train_dirs = [dirs[i] for i in train_idx]

        valid_qs = qs_arr[valid_idx]
        valid_xb = xb_arr[valid_idx]
        valid_dirs = [dirs[i] for i in valid_idx]

        train_file = Path(train_file)
        valid_file = Path(valid_file)

        np.savez(
            train_file,
            qs=train_qs,
            idx_b=idx_b,
            xb=train_xb,
            lambdas=lambdas,
            sweep_directions=np.array(train_dirs, dtype=object),
        )
        np.savez(
            valid_file,
            qs=valid_qs,
            idx_b=idx_b,
            xb=valid_xb,
            lambdas=lambdas,
            sweep_directions=np.array(valid_dirs, dtype=object),
        )

        print("\nSaved train.npz and valid.npz")
        print(f"train.qs.shape = {train_qs.shape}")
        print(f"valid.qs.shape = {valid_qs.shape}")

    def write_single_npz(
        self,
        json_file: str | Path,
        out_file: str | Path = "train_from_experiment.npz",
    ) -> None:
        """
        Write one NPZ containing every sweep (no train/valid split).

        Matches the workflow in ``create_train_data.ipynb`` where all sequences
        are kept in a single archive for exploratory training.
        """
        data = self.load_json_list(json_file)
        sweeps = self.sweeps_from_data(data)

        if self.config.trim_first_yplus_dip:
            sweeps = trim_first_yplus_initial_dip(sweeps, required_markers=self.config.required_markers)

        lengths = [len(sweep) for _, sweep in sweeps]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Sweep lengths are not all equal after extraction/trimming: {lengths}. "
                "Either disable trimming or crop manually."
            )

        req = self.config.required_markers
        qs_list: list[np.ndarray] = []
        xb_list: list[np.ndarray] = []
        dirs: list[str] = []

        for i, (direction, sweep) in enumerate(sweeps):
            qs = np.stack([entry_to_q(entry, required_markers=req) for entry in sweep], axis=0)
            xb = build_direct_bc_from_qs(qs)
            qs_list.append(qs)
            xb_list.append(xb)
            dirs.append(direction)
            print(f"Sweep {i}: direction={direction}, T={qs.shape[0]}, q_dim={qs.shape[1]}")

        qs_arr = np.stack(qs_list, axis=0)
        xb_arr = np.stack(xb_list, axis=0)
        idx_b = np.array([0, 1, 2, 3, 7, 8, 9, 10], dtype=np.int32)

        out_file = Path(out_file)
        np.savez(
            out_file,
            qs=qs_arr,
            idx_b=idx_b,
            xb=xb_arr,
            lambdas=np.linspace(0.0, 1.0, qs_arr.shape[1]),
            sweep_directions=np.array(dirs, dtype=object),
        )

        print(f"\nSaved {out_file}")
        print(f"qs.shape   = {qs_arr.shape}")
        print(f"idx_b.shape= {idx_b.shape}")
        print(f"xb.shape   = {xb_arr.shape}")
        print(f"dirs       = {dirs}")
        print(f"idx_b      = {idx_b}")


def json_to_train_npz(
    json_file: str | Path,
    train_file: str | Path = "train.npz",
    valid_file: str | Path = "valid.npz",
    *,
    trim_first_yplus_dip: bool = False,
) -> None:
    """
    Backward-compatible function API matching the old notebook top-level helper.

    For finer control, instantiate :class:`SlinkyJsonDatasetBuilder` with a custom
    :class:`SlinkyJsonDatasetConfig`.
    """
    cfg = SlinkyJsonDatasetConfig(trim_first_yplus_dip=trim_first_yplus_dip)
    SlinkyJsonDatasetBuilder(cfg).write_train_valid_npz(json_file, train_file, valid_file)


def json_to_single_npz(
    json_file: str | Path,
    out_file: str | Path = "train_from_experiment.npz",
    *,
    trim_first_yplus_dip: bool = False,
) -> None:
    """Single-file export (all sweeps), matching the old ``create_train_data`` helper."""
    cfg = SlinkyJsonDatasetConfig(
        trim_first_yplus_dip=trim_first_yplus_dip,
        filter_incomplete_marker_entries=False,
    )
    SlinkyJsonDatasetBuilder(cfg).write_single_npz(json_file, out_file)
