#!/usr/bin/env python3
"""Strip duplicated MLP/Example cells; standardize on ``util`` + ``slinky_triplet_architectures``."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def pick_arch(name: str) -> str:
    n = name.lower()
    if "spectral" in n:
        return "Triplet3DBoundedSpectral"
    # ``matrix`` / ``matrix_multiset`` notebooks use the Cholesky+MLP 3×3 model.
    if "matrix" in n:
        return "Triplet3DCholeskyMLP"
    if "k_is_3" in n:
        return "Triplet3DCholeskyConstant"
    if "10_loop_slinky" in n:
        return "TripletSlinkyPureMLP"
    if n == "3d_slinky.ipynb" or "3d_slinky_original" in n:
        return "Triplet2DQuadraticPlusMLP"
    return "Triplet2DStiffnessMLP"


def rod_preset_hint(first_cell: str) -> str | None:
    if "util_multiset_copy" in first_cell:
        return "ribbon_experiment"
    if "util_multiset_ribbon" in first_cell:
        return "ribbon_strip"
    return None


def strip_models_keep_tail(src: str) -> str | None:
    """Remove ``class MLP`` / ``class Example`` blocks; keep trailing validate/train code."""
    if "class Example" not in src or "TripletModel" not in src:
        return None
    tail_markers = (
        "\nvalidate_model(",
        "\nfinal_model,",
        "validate_model(",
        "final_model,",
    )
    cut = None
    for m in tail_markers:
        idx = src.find(m)
        if idx != -1 and (cut is None or idx < cut):
            cut = idx
    if cut is None:
        return None
    head = src[:cut].rstrip()
    tail = src[cut:].lstrip("\n")
    head = re.sub(r"\nfrom slinky_common\.jax_math import inv_softplus\n", "\n", head)
    head = re.sub(r"\nimport equinox as eqx\n", "\n", head)
    head = re.sub(r"\nimport jax\n", "\n", head)
    head = re.sub(r"\nimport jax\.numpy as jnp\n", "\n", head)
    head = head.rstrip()
    while head.endswith("\n\n"):
        head = head[:-1]
    if "class MLP" in head or "class Example" in head:
        head = re.sub(
            r"\nclass MLP\b[\s\S]*",
            "",
            head,
            count=1,
        )
    if "class Example" in head:
        head = re.sub(r"\nclass Example\b[\s\S]*", "", head, count=1)
    head = head.rstrip() + "\n\n"
    if "validate_model(Example)" in tail and "der_K" not in tail.split("validate_model", 1)[1][:80]:
        tail = tail.replace("validate_model(Example)", "validate_model(Example, init_K)", 1)
        if "init_K" not in head and "init_K=" not in tail:
            head += "init_K = jnp.array([2.0, 0.01])\n\n"
    out = head + tail
    out = re.sub(r"\n\n\n+", "\n\n", out)
    return out


def patch_train_rod(src: str, preset: str | None) -> str:
    if not preset or f'rod_preset="{preset}"' in src:
        return src
    pos = 0
    parts = []
    for m in re.finditer(r"train_model\(\s*", src):
        parts.append(src[pos : m.end()])
        parts.append(f'rod_preset="{preset}", ')
        pos = m.end()
    parts.append(src[pos:])
    return "".join(parts)


def patch_unpack_3_to_4(src: str) -> str:
    """Old mistake: three return values from train_model."""
    return re.sub(
        r"final_model,\s*train_history,\s*valid_history\s*=\s*train_model",
        "final_model, _k0, train_history, valid_history = train_model",
        src,
    )


def patch_notebook(path: Path) -> bool:
    arch = pick_arch(path.name)
    nb = json.loads(path.read_text())
    first = ""
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code" and c.get("source"):
            first = "".join(c["source"])
            break
    preset = rod_preset_hint(first)

    changed = False
    arch_cell_inserted = False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        orig = src

        src = re.sub(
            r"from util_multiset_copy import \*",
            "from util import TripletModel, validate_model, train_model, get_base_rod, TestCase",
            src,
        )
        src = re.sub(
            r"from util_multiset_ribbon import \*",
            "from util import TripletModel, validate_model, train_model, get_base_rod, TestCase",
            src,
        )
        src = re.sub(
            r"from util_multiset import ([^\n]+)",
            r"from util import \1",
            src,
        )
        src = re.sub(
            r"from util_reg_loss import ([^\n]+)",
            r"from util import \1",
            src,
        )
        src = re.sub(
            r"from util_copy import ([^\n]+)",
            r"from util import \1",
            src,
        )
        src = re.sub(r"from util_reg_loss import TestCase\n", "from util import TestCase\n", src)
        src = re.sub(r"from util import TestCase\n", "from util import TestCase\n", src)

        new_src = strip_models_keep_tail(src)
        if new_src is not None:
            src = new_src
            changed = True

        src = patch_train_rod(src, preset)
        src = patch_unpack_3_to_4(src)

        if src != orig:
            changed = True
        cell["source"] = [ln + "\n" for ln in src.splitlines()] if src else []

    arch_src = (
        f"from slinky_triplet_architectures import {arch}\n"
        f"Example = {arch}  # notebook alias\n"
    )
    if preset:
        arch_src += f'\n# Match prior rod geometry: train_model(..., rod_preset="{preset}")\n'

    new_cells = []
    for i, cell in enumerate(nb["cells"]):
        new_cells.append(cell)
        if (
            not arch_cell_inserted
            and cell.get("cell_type") == "code"
            and ("import dismech_jax" in "".join(cell.get("source", [])) or "import jax" in "".join(cell.get("source", [])))
        ):
            new_cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [arch_src],
                    "outputs": [],
                    "execution_count": None,
                }
            )
            arch_cell_inserted = True
            changed = True

    nb["cells"] = new_cells
    if changed:
        path.write_text(json.dumps(nb, indent=1) + "\n")
    return changed


def main() -> None:
    for p in sorted(ROOT.glob("3d*.ipynb")):
        if patch_notebook(p):
            print("updated", p.name)


if __name__ == "__main__":
    main()
