#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Allow running as a script without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qp_gurobi.solve import solve_from_paths


_PEN_RE = re.compile(r"preconditioned_problem_pen=(?P<pen>[0-9]+\.[0-9]+)\.dat$")


def _discover_preconditioned(layer_dir: Path) -> List[Tuple[float, Path]]:
    items: List[Tuple[float, Path]] = []
    for p in sorted(layer_dir.glob("preconditioned_problem_pen=*.dat")):
        m = _PEN_RE.search(p.name)
        if not m:
            continue
        pen = float(m.group("pen"))
        items.append((pen, p))
    if not items:
        raise FileNotFoundError(f"No preconditioned_problem_pen=*.dat found under {layer_dir}")
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="Solve baseline and preconditioned instances with hard bisection constraint.")
    ap.add_argument("--data-root", type=Path, default=Path("Data"), help="Root Data folder (default: Data)")
    ap.add_argument("--n", type=int, default=8, help="Problem size N (default: 8)")
    ap.add_argument("--seed", type=int, default=0, help="Seed folder (default: 0)")
    ap.add_argument("--layers", type=int, default=1, help="n_qaoa_layers folder (default: 1)")
    ap.add_argument("--time-limit", type=float, default=None, help="Gurobi TimeLimit seconds")
    ap.add_argument("--mip-gap", type=float, default=None, help="Gurobi MIPGap")
    ap.add_argument("--threads", type=int, default=None, help="Gurobi Threads")
    ap.add_argument("--gurobi-seed", type=int, default=None, help="Gurobi Seed")
    ap.add_argument("--show-logs", action="store_true", help="Stream Gurobi log output to the console")
    ap.add_argument("--log-dir", type=Path, default=None, help="If set, write per-run Gurobi logs into this directory")
    ap.add_argument("--out", type=Path, default=None, help="Output CSV path (default: results/N=..._seed=..._layers=....csv)")

    args = ap.parse_args()

    instance_dir = args.data_root / f"N={args.n}" / f"seed={args.seed}"
    baseline_path = instance_dir / "problem.dat"
    layer_dir = instance_dir / f"n_qaoa_layers={args.layers}"

    precond = _discover_preconditioned(layer_dir)

    results = solve_from_paths(
        baseline_path=baseline_path,
        preconditioned_paths=precond,
        time_limit_sec=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        seed=args.gurobi_seed,
        output_flag=1 if args.show_logs else 0,
        log_dir=args.log_dir,
    )

    out = args.out
    if out is None:
        out = Path("results") / f"N={args.n}_seed={args.seed}_layers={args.layers}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [r.to_row() for r in results]
    fieldnames = list(rows[0].keys())

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
