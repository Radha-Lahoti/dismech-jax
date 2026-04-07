#!/usr/bin/env python3
"""
Execute all notebooks under ``examples/slinky`` with tiny epoch counts (in-memory only).

This is meant as a smoke test: imports, JAX jitting, and short optimization scans.
Full ``n_epochs`` / ``num_epochs`` literals are rewritten to :data:`FAST_N_EPOCHS` before
execution. Notebooks still fail if required NPZ/JSON inputs are absent.
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

import nbformat
from nbclient import execute
from nbclient.exceptions import CellExecutionError

FAST_N_EPOCHS = 3
SLINKY_DIR = Path(__file__).resolve().parent
RE_N_EPOCHS = re.compile(r"\bn_epochs\s*=\s*\d+")
RE_NUM_EPOCHS = re.compile(r"^(\s*)num_epochs\s*=\s*\d+\s*$", re.MULTILINE)


def cap_source(text: str) -> str:
    text = RE_N_EPOCHS.sub(f"n_epochs={FAST_N_EPOCHS}", text)
    text = RE_NUM_EPOCHS.sub(lambda m: f"{m.group(1)}num_epochs = {FAST_N_EPOCHS}", text)
    return text


def prepare_notebook(nb_path: Path) -> nbformat.NotebookNode:
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        cell["source"] = cap_source(src)
    return nb


def run_one(nb_path: Path, timeout: int = 600) -> tuple[bool, str]:
    nb = prepare_notebook(nb_path)
    cwd = str(nb_path.parent)
    try:
        execute(
            nb,
            cwd=cwd,
            timeout=timeout,
            kernel_name="python3",
            allow_errors=False,
        )
    except CellExecutionError as e:
        tb = e.traceback or ""
        return False, f"{e.ename}: {e.evalue}\n{tb}"
    except Exception:
        return False, traceback.format_exc()
    return True, "ok"


def main() -> int:
    notebooks = sorted(
        p
        for p in SLINKY_DIR.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in p.parts
    )
    failures: list[tuple[str, str]] = []
    for p in notebooks:
        rel = p.relative_to(SLINKY_DIR)
        ok, msg = run_one(p)
        status = "PASS" if ok else "FAIL"
        print(f"{status}\t{rel}")
        if not ok:
            failures.append((str(rel), msg))
            for line in msg.splitlines()[:40]:
                print(f"  | {line}")
            if msg.count("\n") > 40:
                print("  | ...")
    n_ok = len(notebooks) - len(failures)
    print(f"\nSummary: {n_ok}/{len(notebooks)} passed (epochs capped to {FAST_N_EPOCHS}).")
    if failures:
        print("Failures (missing data paths are expected until you generate NPZ/JSON):")
        for rel, _ in failures:
            print(f"  - {rel}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
