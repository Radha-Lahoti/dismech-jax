#!/usr/bin/env python3
"""Fix invalid train_model(...) argument order and wrong unpack variable names."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _to_source_list(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return [line + "\n" for line in text.splitlines()]


def fix_text(path: Path, src: str) -> str:
    s = src
    s = s.replace(
        'train_model(rod_preset="ribbon_experiment", Example,',
        'train_model(Example, rod_preset="ribbon_experiment",',
    )
    s = s.replace(
        'train_model(rod_preset="ribbon_strip", Example,',
        'train_model(Example, rod_preset="ribbon_strip",',
    )
    s = s.replace(
        "final_model, init_K, train_history, valid_history",
        "final_model, init_stiff_trace, train_history, valid_history",
    )
    s = s.replace("validate_model(Example, init_K)", "validate_model(Example, der_K=init_K)")

    name = path.name
    if "validate_model(Example)\n" in s:
        if name == "3d_slinky_10_loop_slinky.ipynb":
            s = s.replace(
                "validate_model(Example)\n",
                "validate_model(Example, der_K=jnp.array([2.0, 0.01, 0.02]))\n",
            )
        elif "K_is_2_dim" in name and "multiset" not in name:
            s = s.replace(
                "validate_model(Example)\n",
                "validate_model(Example, der_K=jnp.array([2.0, 0.02]))\n",
            )
    return s


def main() -> None:
    n = 0
    for path in sorted(ROOT.glob("3d*.ipynb")):
        data = json.loads(path.read_text())
        changed = False
        for cell in data.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            lines = cell.get("source") or []
            if isinstance(lines, str):
                old = lines
            else:
                old = "".join(lines)
            new = fix_text(path, old)
            if new != old:
                cell["source"] = _to_source_list(new)
                changed = True
        if changed:
            path.write_text(json.dumps(data, indent=1) + "\n")
            print("updated", path.name)
            n += 1
    print("done,", n, "files")


if __name__ == "__main__":
    main()
