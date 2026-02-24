#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running as a script without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qp_gurobi.solve import solve_from_paths


_PEN_RE = re.compile(r"preconditioned_problem_pen=(?P<pen>[0-9]+\.[0-9]+)\.dat$")


def _discover_seeds(n_dir: Path) -> List[int]:
    seeds: List[int] = []
    for p in sorted(n_dir.glob("seed=*")):
        if not p.is_dir():
            continue
        try:
            seeds.append(int(p.name.split("=", 1)[1]))
        except Exception:
            continue
    return sorted(set(seeds))


def _parse_seed_list(seeds_csv: str) -> List[int]:
    seeds: List[int] = []
    for tok in seeds_csv.split(","):
        tok = tok.strip()
        if not tok:
            continue
        seeds.append(int(tok))
    return sorted(set(seeds))


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
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("one_equality") / "complete_random",
        help="Root data folder (default: one_equality/complete_random)",
    )
    ap.add_argument("--n", type=int, default=8, help="Problem size N (default: 8)")
    ap.add_argument("--seed", type=int, default=None, help="Single seed to run (legacy; prefer --seeds or --all-seeds)")
    ap.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list (e.g. 0,1,2,5)")
    ap.add_argument("--all-seeds", action="store_true", help="Discover and run all available seed=* folders")
    ap.add_argument("--layers", type=int, default=1, help="n_qaoa_layers folder (default: 1)")
    ap.add_argument("--time-limit", type=float, default=None, help="Gurobi TimeLimit seconds")
    ap.add_argument("--mip-gap", type=float, default=None, help="Gurobi MIPGap")
    ap.add_argument("--threads", type=int, default=None, help="Gurobi Threads")
    ap.add_argument("--gurobi-seed", type=int, default=None, help="Gurobi Seed")
    ap.add_argument("--show-logs", action="store_true", help="Stream Gurobi log output to the console")
    ap.add_argument("--log-dir", type=Path, default=None, help="If set, write per-run Gurobi logs into this directory")
    ap.add_argument("--no-presolve", action="store_true", help="Disable Gurobi presolve (pure branch-and-bound)")
    ap.add_argument("--out", type=Path, default=None, help="Output CSV path (default: results/N=..._seed=..._layers=....csv)")

    args = ap.parse_args()

    data_root = args.data_root
    # If invoked from a subdirectory (e.g. `cd scripts`), make relative paths
    # resolve from the repo root rather than the current working directory.
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()

    n_dir = data_root / f"N={args.n}"
    if args.all_seeds:
        seeds = _discover_seeds(n_dir)
        if not seeds:
            raise FileNotFoundError(f"No seed=* directories found under {n_dir}")
    elif args.seeds is not None:
        seeds = _parse_seed_list(args.seeds)
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = [0]

    all_rows: List[Dict[str, object]] = []

    for seed in seeds:
        instance_dir = n_dir / f"seed={seed}"
        baseline_path = instance_dir / "problem.dat"
        layer_dir = instance_dir / f"n_qaoa_layers={args.layers}"

        if not baseline_path.exists():
            print(f"[WARN] Missing baseline file: {baseline_path} (skipping)")
            continue
        if not layer_dir.exists():
            print(f"[WARN] Missing layer dir: {layer_dir} (skipping)")
            continue

        try:
            precond = _discover_preconditioned(layer_dir)
        except FileNotFoundError as e:
            print(f"[WARN] {e} (skipping seed={seed})")
            continue

        seed_log_dir = None
        if args.log_dir is not None:
            seed_log_dir = Path(args.log_dir) / f"N={args.n}" / f"seed={seed}" / f"n_qaoa_layers={args.layers}"

        results = solve_from_paths(
            baseline_path=baseline_path,
            preconditioned_paths=precond,
            time_limit_sec=args.time_limit,
            mip_gap=args.mip_gap,
            threads=args.threads,
            seed=args.gurobi_seed,
            output_flag=1 if args.show_logs else 0,
            log_dir=seed_log_dir,
            presolve=not args.no_presolve,
        )

        for r in results:
            row = r.to_row()
            row["seed"] = int(seed)
            row["layers"] = int(args.layers)
            row["data_root"] = str(data_root)
            row["presolve"] = not args.no_presolve
            all_rows.append(row)

    out = args.out
    if out is None:
        if args.all_seeds:
            seed_tag = "all"
        elif args.seeds is not None:
            compact = re.sub(r"\s+", "", str(args.seeds))
            compact = compact.replace(",", "_")
            seed_tag = compact if len(compact) <= 40 else f"listN{len(seeds)}"
        else:
            seed_tag = str(args.seed) if args.seed is not None else "0"
        out = Path("results") / f"N={args.n}_seeds={seed_tag}_layers={args.layers}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not all_rows:
        raise RuntimeError("No rows produced (all seeds skipped?).")

    fieldnames = list(all_rows[0].keys())

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
