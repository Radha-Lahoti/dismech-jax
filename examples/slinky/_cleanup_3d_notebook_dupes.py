#!/usr/bin/env python3
"""Remove leftover ``class Example`` cells and fix a few arch / unpack mistakes."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ARCH_BY_NAME: dict[str, str] = {
    "3d_slinky_K(eps)_matrix_multiset_spectral_loss.ipynb": "Triplet3DCholeskyMLP",
    "3d_slinky_10_loop_K(eps)_multiset.ipynb": "Triplet2DStiffnessMLP",
    "3d_slinky_10_loop_K(eps).ipynb": "Triplet2DKPairSoftplusMLP",
}


def _cell_src(cell: dict) -> str:
    return "".join(cell.get("source") or [])


def _should_drop_cell(src: str) -> bool:
    if "class Example(TripletModel)" in src:
        return True
    lines = [ln.strip() for ln in src.splitlines()]
    nonempty = [ln for ln in lines if ln]
    if not nonempty:
        return False
    if all(ln.startswith("#") for ln in nonempty) and any(
        "class Example" in ln for ln in nonempty
    ):
        return True
    return False


def _fix_arch_cell(src: str, arch: str) -> str:
    src = re.sub(
        r"from slinky_triplet_architectures import \w+",
        f"from slinky_triplet_architectures import {arch}",
        src,
        count=1,
    )
    src = re.sub(
        r"Example = \w+\s*# notebook alias",
        f"Example = {arch}  # notebook alias",
        src,
        count=1,
    )
    return src


def _to_source_list(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return [line + "\n" for line in text.splitlines()]


def _fix_spectral_loss_unpack(src: str) -> str:
    # Second return value is stiffness trace at init, not init_K.
    old = (
        "final_model, init_K, train_total_history, train_data_history, train_spec_history, "
        "valid_total_history, valid_data_history, valid_spec_history, ="
    )
    new = (
        "final_model, init_stiff_trace, train_total_history, train_data_history, train_spec_history, "
        "valid_total_history, valid_data_history, valid_spec_history ="
    )
    if old in src:
        src = src.replace(old, new)
    return src


def process_notebook(path: Path) -> bool:
    data = json.loads(path.read_text())
    cells = data.get("cells", [])
    name = path.name
    changed = False

    new_cells = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            new_cells.append(cell)
            continue
        src = _cell_src(cell)
        if _should_drop_cell(src):
            changed = True
            continue
        if name in ARCH_BY_NAME and "slinky_triplet_architectures import" in src:
            fixed = _fix_arch_cell(src, ARCH_BY_NAME[name])
            if fixed != src:
                cell = {**cell, "source": _to_source_list(fixed)}
                changed = True
        if name == "3d_slinky_K(eps)_matrix_multiset_spectral_loss.ipynb":
            src2 = _cell_src(cell)
            fixed2 = _fix_spectral_loss_unpack(src2)
            if fixed2 != src2:
                cell = {**cell, "source": _to_source_list(fixed2)}
                changed = True
        new_cells.append(cell)

    if changed:
        data["cells"] = new_cells
        path.write_text(json.dumps(data, indent=1) + "\n")
    return changed


def main() -> None:
    n = 0
    for p in sorted(ROOT.glob("3d*.ipynb")):
        if process_notebook(p):
            print("updated", p.name)
            n += 1
    print("done,", n, "files changed")


if __name__ == "__main__":
    main()
