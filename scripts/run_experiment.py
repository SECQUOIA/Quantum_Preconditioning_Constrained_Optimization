#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running as a script without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qp_gurobi.instance import resolve_data_root, trajectory_to_running_best
from qp_gurobi.solve import solve_from_paths


_PEN_RE = re.compile(r"preconditioned_problem_pen=(?P<pen>[0-9]+\.[0-9]+)\.dat$")


def _discover_seeds(n_dir: Path) -> List[int]:
    """Discover available integer seed directories under an ``N`` folder."""
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
    """Parse a comma-separated seed list."""
    seeds: List[int] = []
    for tok in seeds_csv.split(","):
        tok = tok.strip()
        if not tok:
            continue
        seeds.append(int(tok))
    return sorted(set(seeds))


def _discover_preconditioned(layer_dir: Path) -> List[Tuple[float, Path]]:
    """Discover preconditioned instance files and their penalties."""
    items: List[Tuple[float, Path]] = []
    for p in sorted(layer_dir.glob("preconditioned_problem_pen=*.dat")):
        m = _PEN_RE.search(p.name)
        if not m:
            continue
        pen = float(m.group("pen"))
        if not p.read_text().strip():
            print(f"[WARN] Empty preconditioned file, skipping penalty={pen:.3f}: {p}")
            continue
        items.append((pen, p))
    if not items:
        raise FileNotFoundError(f"No preconditioned_problem_pen=*.dat found under {layer_dir}")
    return items


def _setting_dir(instance_dir: Path, family: str, layers: int | None, sweeps: int | None) -> Path:
    """Return the data directory for one preconditioner family setting."""
    if family == "quantum":
        if layers is None:
            raise ValueError("--layers is required when --preconditioner=quantum")
        return instance_dir / f"n_qaoa_layers={layers}"
    if sweeps is None:
        raise ValueError("--sweeps is required when --preconditioner=classical")
    return instance_dir / f"n_sweeps={sweeps}"


def _family_tag(family: str, layers: int | None, sweeps: int | None) -> str:
    """Return the filename tag for one preconditioner family setting."""
    if family == "quantum":
        if layers is None:
            raise ValueError("layers is required for quantum outputs")
        return f"quantum_layers={layers}"
    if sweeps is None:
        raise ValueError("sweeps is required for classical outputs")
    return f"classical_sweeps={sweeps}"


def _seed_tag(seeds: List[int]) -> str:
    """Return a compact, collision-resistant filename tag for seed subsets."""
    compact = "_".join(str(seed) for seed in seeds)
    if len(compact) <= 40:
        return compact
    digest = hashlib.sha1(",".join(str(seed) for seed in seeds).encode("ascii")).hexdigest()[:10]
    return f"listN{len(seeds)}_{digest}"


def main() -> int:
    """Run the experiment CLI and write summary/trajectory outputs."""
    ap = argparse.ArgumentParser(description="Solve baseline and preconditioned instances with hard bisection constraint.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=resolve_data_root(REPO_ROOT),
        help="Root data folder (default: auto-detected download location, see README's Data section)",
    )
    ap.add_argument("--n", type=int, default=8, help="Problem size N (default: 8)")
    ap.add_argument("--seed", type=int, default=None, help="Single seed to run (legacy; prefer --seeds or --all-seeds)")
    ap.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list (e.g. 0,1,2,5)")
    ap.add_argument("--all-seeds", action="store_true", help="Discover and run all available seed=* folders")
    ap.add_argument(
        "--preconditioner",
        choices=["quantum", "classical"],
        default="quantum",
        help="Preconditioning family to run (default: quantum)",
    )
    ap.add_argument("--layers", type=int, default=None, help="Quantum n_qaoa_layers folder (required when --preconditioner=quantum)")
    ap.add_argument("--sweeps", type=int, default=None, help="Classical n_sweeps folder (required when --preconditioner=classical)")
    ap.add_argument("--time-limit", type=float, default=None, help="Gurobi TimeLimit seconds")
    ap.add_argument("--mip-gap", type=float, default=None, help="Gurobi MIPGap")
    ap.add_argument("--threads", type=int, default=None, help="Gurobi Threads")
    ap.add_argument("--gurobi-seed", type=int, default=None, help="Gurobi Seed")
    ap.add_argument("--show-logs", action="store_true", help="Stream Gurobi log output to the console")
    ap.add_argument("--log-dir", type=Path, default=None, help="If set, write per-run Gurobi logs into this directory")
    ap.add_argument("--no-presolve", action="store_true", help="Disable Gurobi presolve (pure branch-and-bound)")
    ap.add_argument("--pen-min", type=float, default=None, help="Only run penalties >= this value (inclusive)")
    ap.add_argument("--pen-max", type=float, default=None, help="Only run penalties <= this value (inclusive)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output summary CSV path (default includes N, family parameter, and presolve mode).",
    )
    ap.add_argument(
        "--trajectory-out",
        type=Path,
        default=None,
        help=(
            "Path for the long-format trajectory CSV (one row per MIPSol callback event). "
            "Columns: seed, layers, sweeps, preconditioner_family, name, penalty, presolve, event_idx, "
            "time_sec, orig_obj, running_best_orig_obj. "
            "Defaults to a results/ CSV with matching N, family, seed, and presolve tags."
        ),
    )

    args = ap.parse_args()

    if args.preconditioner == "quantum":
        if args.layers is None:
            raise SystemExit("--layers is required when --preconditioner=quantum")
        if args.sweeps is not None:
            raise SystemExit("--sweeps cannot be used when --preconditioner=quantum")
    else:
        if args.sweeps is None:
            raise SystemExit("--sweeps is required when --preconditioner=classical")
        if args.layers is not None:
            raise SystemExit("--layers cannot be used when --preconditioner=classical")

    seed_selectors = [args.seed is not None, args.seeds is not None, args.all_seeds]
    if sum(seed_selectors) > 1:
        raise SystemExit("Use only one of --seed, --seeds, or --all-seeds")

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
    family_tag = _family_tag(args.preconditioner, args.layers, args.sweeps)
    seed_tag = "all" if args.all_seeds else _seed_tag(seeds)

    for seed in seeds:
        instance_dir = n_dir / f"seed={seed}"
        baseline_path = instance_dir / "problem.dat"
        precond_dir = _setting_dir(instance_dir, args.preconditioner, args.layers, args.sweeps)

        if not baseline_path.exists():
            print(f"[WARN] Missing baseline file: {baseline_path} (skipping)")
            continue
        if not precond_dir.exists():
            print(f"[WARN] Missing preconditioner dir: {precond_dir} (skipping)")
            continue

        try:
            precond = _discover_preconditioned(precond_dir)
        except FileNotFoundError as e:
            print(f"[WARN] {e} (skipping seed={seed})")
            continue

        if args.pen_min is not None:
            precond = [(p, f) for p, f in precond if p >= args.pen_min - 1e-9]
        if args.pen_max is not None:
            precond = [(p, f) for p, f in precond if p <= args.pen_max + 1e-9]
        if not precond:
            print(f"[WARN] No penalties in range after filtering (skipping seed={seed})")
            continue

        seed_log_dir = None
        if args.log_dir is not None:
            seed_log_dir = Path(args.log_dir) / f"N={args.n}" / f"seed={seed}" / family_tag

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
            row["preconditioner_family"] = args.preconditioner
            row["layers"] = int(args.layers) if args.layers is not None else None
            row["sweeps"] = int(args.sweeps) if args.sweeps is not None else None
            all_rows.append(row)

    out = args.out
    if out is None:
        mode = "off" if args.no_presolve else "on"
        if args.all_seeds:
            # All seeds aggregated into one file; seed_tag not needed in name.
            out = Path("results") / f"N={args.n}_{family_tag}_presolve_{mode}.csv"
        else:
            # Include seed_tag so partial-seed runs never silently overwrite each other.
            out = Path("results") / f"N={args.n}_{family_tag}_seeds={seed_tag}_presolve_{mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not all_rows:
        raise RuntimeError("No rows produced (all seeds skipped?).")

    fieldnames = list(all_rows[0].keys())

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {out}")

    traj_out = args.trajectory_out
    if traj_out is None:
        mode = "off" if args.no_presolve else "on"
        if args.all_seeds:
            traj_out = Path("results") / f"N={args.n}_{family_tag}_presolve_{mode}_trajectories.csv"
        else:
            traj_out = Path("results") / f"N={args.n}_{family_tag}_seeds={seed_tag}_presolve_{mode}_trajectories.csv"
    if traj_out:
        traj_out.parent.mkdir(parents=True, exist_ok=True)
        traj_rows: List[Dict[str, object]] = []
        for row in all_rows:
            raw = row.get("trajectory", "[]")
            events: List[List[float]] = json.loads(raw) if raw else []
            trajectory = [(float(t), float(orig_obj)) for t, orig_obj in events]
            running_best = trajectory_to_running_best(trajectory)
            for idx, ((t, orig_obj), (_, best_orig_obj)) in enumerate(zip(trajectory, running_best)):
                traj_rows.append({
                    "seed": row["seed"],
                    "layers": row["layers"],
                    "sweeps": row.get("sweeps"),
                    "preconditioner_family": row.get("preconditioner_family"),
                    "name": row["name"],
                    "penalty": row.get("penalty"),
                    "presolve": row.get("presolve"),
                    "event_idx": idx,
                    "time_sec": t,
                    "orig_obj": orig_obj,
                    "running_best_orig_obj": best_orig_obj,
                })
        if traj_rows:
            traj_fieldnames = list(traj_rows[0].keys())
            with traj_out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=traj_fieldnames)
                w.writeheader()
                w.writerows(traj_rows)
            print(f"Wrote {len(traj_rows)} trajectory rows to {traj_out}")
        else:
            print("[WARN] No trajectory events were recorded.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
