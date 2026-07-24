#!/usr/bin/env python3
"""Merge penalty-restricted cluster runs (e.g. results/pen1.1-2.0/) with the
original full-range sweeps into results/merged/, so every existing loader
(load_summary_data, load_family_comparison_data, etc.) can point at one
complete directory without duplicating any (seed, penalty) solved twice.

results/merged/ is a full drop-in replacement for results/: every file from
results/ is copied over first (classical sweeps, quantum_layers=inf, anything
else untouched by this merge), then any (N, layers, presolve) combo that also
appears under the extra directory is re-written as the deduplicated union of
both sources. Safe to re-run whenever more penalty-restricted data lands.

Usage:
    python3 scripts/merge_penalty_results.py
    python3 scripts/merge_penalty_results.py --extra-dir results/pen1.1-2.0
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SUMMARY_RE = re.compile(r"^N=(?P<n>\d+)_layers=(?P<layers>\d+)_presolve_(?P<mode>on|off)\.csv$")


def _discover_combos(a_dir: Path) -> set[tuple[int, int, str]]:
    """Find (n, layers, presolve_mode) combos with a quantum finite-depth summary CSV."""
    combos: set[tuple[int, int, str]] = set()
    if not a_dir.exists():
        return combos
    for p in a_dir.glob("N=*_layers=*_presolve_*.csv"):
        if "trajector" in p.name:
            continue
        m = _SUMMARY_RE.match(p.name)
        if m:
            combos.add((int(m.group("n")), int(m.group("layers")), m.group("mode")))
    return combos


def _merge_frames(frames: list[pd.DataFrame], dedup_cols: list[str]) -> pd.DataFrame:
    """Concat frames and drop duplicate (seed, penalty, ...) rows, preferring the
    last-listed source (the extra/penalty-restricted directory) on exact overlap.
    """
    combined = pd.concat(frames, ignore_index=True)
    combined["_pen_key"] = combined["penalty"].round(3)
    dedup_subset = [c.replace("penalty", "_pen_key") for c in dedup_cols]
    combined = combined.drop_duplicates(subset=dedup_subset, keep="last")
    return combined.drop(columns=["_pen_key"])


def _merge_one(base_dir: Path, extra_dir: Path, out_dir: Path, n: int, layers: int, mode: str) -> None:
    tag = f"N={n}_layers={layers}_presolve_{mode}"
    summary_name = f"{tag}.csv"
    traj_name = f"{tag}_trajectories.csv"

    summary_frames = [pd.read_csv(d / summary_name) for d in (base_dir, extra_dir) if (d / summary_name).exists()]
    if not summary_frames:
        return
    summary = _merge_frames(summary_frames, ["name", "seed", "penalty", "presolve"])
    summary.to_csv(out_dir / summary_name, index=False)

    traj_frames = [pd.read_csv(d / traj_name) for d in (base_dir, extra_dir) if (d / traj_name).exists()]
    traj_rows = 0
    if traj_frames:
        traj = _merge_frames(traj_frames, ["name", "seed", "penalty", "presolve", "event_idx"])
        traj.to_csv(out_dir / traj_name, index=False)
        traj_rows = len(traj)

    print(f"{tag}: {len(summary)} summary rows" + (f", {traj_rows} trajectory rows" if traj_frames else ", no trajectory data"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results", help="Original full-range results directory")
    ap.add_argument("--extra-dir", type=Path, default=REPO_ROOT / "results" / "pen1.1-2.0", help="Penalty-restricted results directory to merge in")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "merged", help="Output directory (fully overwritten on each run)")
    args = ap.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(f"Missing --results-dir: {args.results_dir}")
    if not args.extra_dir.exists():
        raise SystemExit(f"Missing --extra-dir: {args.extra_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Start from a complete copy of the original directory (classical sweeps,
    # quantum_layers=inf, anything not touched by this merge) so out_dir is a
    # full drop-in replacement, not just the merged subset.
    for p in args.results_dir.glob("*.csv"):
        shutil.copy2(p, args.out_dir / p.name)
    copied = sum(1 for _ in args.results_dir.glob("*.csv"))
    print(f"Copied {copied} untouched CSVs from {args.results_dir} into {args.out_dir}")

    combos = _discover_combos(args.results_dir) | _discover_combos(args.extra_dir)
    if not combos:
        print("No N=..._layers=..._presolve_....csv combos found in either directory; nothing to merge.")
        return 0

    print(f"\nMerging {len(combos)} (N, layers, presolve) combo(s):")
    for n, layers, mode in sorted(combos):
        _merge_one(args.results_dir, args.extra_dir, args.out_dir, n, layers, mode)

    print(f"\nDone. {args.out_dir} is ready to use as RESULTS_DIR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
