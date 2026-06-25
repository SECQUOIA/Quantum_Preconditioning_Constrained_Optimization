#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
import pandas as pd

# Allow running as a script without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qp_gurobi.instance import eval_ising, read_dat, z_from_x


def _parse_x_bits(x_bits: str, n: int) -> list[int]:
    """Parse a compact binary solution string into bits."""
    s = str(x_bits).strip()
    if len(s) != n:
        # tolerate missing leading zeros
        s = s.zfill(n)
    if len(s) != n or any(ch not in "01" for ch in s):
        raise ValueError(f"Invalid x_bits {x_bits!r} for n={n}")
    return [int(ch) for ch in s]


def main() -> int:
    """Run the result verification CLI."""
    ap = argparse.ArgumentParser(
        description=(
            "Verify a results CSV by recomputing baseline objective from stored bitstrings "
            "and checking the bisection constraint."
        )
    )
    ap.add_argument("--csv", type=Path, required=True, help="Results CSV to verify")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("one_equality_new") / "complete_random",
        help="Root data folder (default: one_equality_new/complete_random)",
    )
    ap.add_argument("--n", type=int, required=True, help="Problem size N")
    ap.add_argument("--seed", type=int, required=True, help="Seed folder")
    ap.add_argument("--tol", type=float, default=1e-6, help="Tolerance for objective comparison")

    args = ap.parse_args()

    data_root = args.data_root
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()

    if args.n % 2 != 0:
        raise SystemExit(f"N must be even for bisection, got {args.n}")

    baseline_path = data_root / f"N={args.n}" / f"seed={args.seed}" / "problem.dat"
    baseline = read_dat(baseline_path)

    df = pd.read_csv(
        args.csv,
        dtype={"name": "string", "x_bits": "string", "z_bits": "string"},
    )

    required_cols = {"name", "n", "objective_baseline", "x_bits", "sum_z"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {sorted(missing)}")

    # Basic checks
    if int(df["n"].max()) != args.n:
        print(f"WARNING: CSV n.max()={int(df['n'].max())} but --n={args.n}")

    failures: list[str] = []
    max_abs_err = 0.0

    for idx, row in df.iterrows():
        name = str(row["name"])
        n_row = int(row["n"])
        if n_row != args.n:
            failures.append(f"Row {idx}: {name}: n={n_row} != {args.n}")
            continue

        x_bits = _parse_x_bits(row["x_bits"], args.n)
        z_bits = z_from_x(x_bits)

        sum_z = int(sum(z_bits))
        if "sum_z" in row and not pd.isna(row["sum_z"]):
            if int(row["sum_z"]) != sum_z:
                failures.append(f"Row {idx}: {name}: sum_z column {int(row['sum_z'])} != recomputed {sum_z}")
        if sum_z != 0:
            failures.append(f"Row {idx}: {name}: infeasible bisection (sum_z={sum_z})")

        obj_re = float(eval_ising(baseline, z_bits))
        obj_csv = float(row["objective_baseline"]) if not pd.isna(row["objective_baseline"]) else float("nan")

        if not (math.isfinite(obj_re) and math.isfinite(obj_csv)):
            failures.append(f"Row {idx}: {name}: non-finite objective (csv={obj_csv}, recomputed={obj_re})")
            continue

        err = abs(obj_re - obj_csv)
        max_abs_err = max(max_abs_err, err)
        if err > args.tol:
            failures.append(
                f"Row {idx}: {name}: objective_baseline mismatch |recomputed-csv|={err:.3e} (tol={args.tol})"
            )

    # Baseline row consistency check
    base_rows = df[df["name"] == "baseline"]
    if len(base_rows) != 1:
        failures.append(f"Expected exactly 1 baseline row, found {len(base_rows)}")
    else:
        b = base_rows.iloc[0]
        if "objective_model" in df.columns and not pd.isna(b.get("objective_model")):
            if abs(float(b["objective_model"]) - float(b["objective_baseline"])) > args.tol:
                failures.append(
                    f"Baseline row: objective_model != objective_baseline (|diff|>{args.tol})"
                )

    if failures:
        print("FAILED verification:")
        for msg in failures[:50]:
            print("-", msg)
        if len(failures) > 50:
            print(f"... and {len(failures)-50} more")
        print(f"Max |objective error| seen: {max_abs_err:.3e}")
        return 2

    print("OK: verification passed")
    print(f"- Baseline instance: {baseline_path}")
    print(f"- Rows checked: {len(df)}")
    print(f"- Max |objective error|: {max_abs_err:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
