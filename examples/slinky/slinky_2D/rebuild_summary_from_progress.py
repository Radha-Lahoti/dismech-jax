"""Rebuild summary.csv from progress.jsonl after a multi-stage sweep.

run_experiment rewrites summary.csv from the in-memory records of its current
call only, so a second sweep into the same output dir clobbers any rows from a
prior call (e.g. the reeval pass). progress.jsonl is append-only and retains
every record, so we rebuild the union here by deduplicating on
(arch_name, seed, hessian_reg_strength) and keeping the latest record.

Usage:
    uv run python examples/slinky/slinky_2D/rebuild_summary_from_progress.py \
        /path/to/hessian_reg_max_dlambda_outputs_n11_tape_reeval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from run_hessian_reg_max_dlambda_experiment import write_summary_csv


def rebuild(output_dir: Path) -> None:
    progress_path = output_dir / "progress.jsonl"
    summary_path = output_dir / "summary.csv"
    if not progress_path.is_file():
        raise SystemExit(f"progress.jsonl not found at {progress_path}")

    latest: dict[tuple, dict] = {}
    n_total = 0
    with open(progress_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            rec = json.loads(line)
            key = (
                rec.get("arch_name"),
                rec.get("seed"),
                rec.get("hessian_reg_strength"),
            )
            latest[key] = rec

    records = list(latest.values())
    write_summary_csv(records, str(summary_path))
    print(
        f"Read {n_total} records from {progress_path.name}; "
        f"wrote {len(records)} deduplicated rows to {summary_path}."
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: rebuild_summary_from_progress.py <output_dir>")
    rebuild(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    main()
