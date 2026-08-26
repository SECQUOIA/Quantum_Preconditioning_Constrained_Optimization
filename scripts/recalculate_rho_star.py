#!/usr/bin/env python3
"""Recalculate the empirical critical spin-penalty table.

The experiment uses complete graphs with independent U[0, 2] edge weights and
computes, for every instance,

    rho_star = max(0, max_z_infeasible
                   (H_star - H(z)) / sum(z)^2),

where H is the cut weight and H_star is its minimum over balanced spins.

For reproducibility, instance ``seed`` is generated with
``numpy.random.default_rng(seed)``.  One vector of n(n-1)/2 independent edge
weights is drawn and assigned to the strict upper triangle in NumPy index
order.  The default run uses the 50 manuscript instances, seeds 0,...,49, at
every problem size.

Requirements are Python 3.10 or later, NumPy, and Numba.  Matplotlib is needed
only when ``--figure`` is requested.  For example:

    python recalculate_rho_star.py --threads 8 \
        --output rho_star_summary.csv \
        --samples-output rho_star_samples.csv \
        --figure penalties_per_instance.pdf
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads


@njit(cache=True)
def _rho_star_exact(weights: np.ndarray) -> float:
    """Compute rho_star by exact Gray-code enumeration of all cuts."""

    n = weights.shape[0]
    degree = np.empty(n, dtype=np.float64)
    for i in range(n):
        total = 0.0
        for j in range(n):
            total += weights[i, j]
        degree[i] = total

    min_cut = np.full(n + 1, np.inf, dtype=np.float64)
    min_cut[0] = 0.0
    min_cut[n] = 0.0
    weight_to_subset = np.zeros(n, dtype=np.float64)
    in_subset = np.zeros(n, dtype=np.uint8)

    # A cut and its complement have equal weight.  Fixing vertex n-1 outside
    # enumerates one representative from every complementary pair.
    subset_count = 1 << (n - 1)
    cut = 0.0
    cardinality = 0

    # Consecutive reflected Gray codes differ in bit ctz(step).
    for step in range(1, subset_count):
        shifted = step
        vertex = 0
        while (shifted & 1) == 0:
            shifted >>= 1
            vertex += 1

        if in_subset[vertex] == 0:
            cut += degree[vertex] - 2.0 * weight_to_subset[vertex]
            for j in range(n):
                weight_to_subset[j] += weights[j, vertex]
            in_subset[vertex] = 1
            cardinality += 1
        else:
            cut += -degree[vertex] + 2.0 * weight_to_subset[vertex]
            for j in range(n):
                weight_to_subset[j] -= weights[j, vertex]
            in_subset[vertex] = 0
            cardinality -= 1

        if cut < min_cut[cardinality]:
            min_cut[cardinality] = cut
        complement_cardinality = n - cardinality
        if cut < min_cut[complement_cardinality]:
            min_cut[complement_cardinality] = cut

    balanced_optimum = min_cut[n // 2]
    rho_star = 0.0
    for cardinality in range(n + 1):
        if cardinality == n // 2:
            continue
        imbalance = 2 * cardinality - n
        candidate = (
            (balanced_optimum - min_cut[cardinality])
            / (imbalance * imbalance)
        )
        if candidate > rho_star:
            rho_star = candidate
    return rho_star


@njit(parallel=True, cache=True)
def _rho_star_batch(weight_matrices: np.ndarray) -> np.ndarray:
    values = np.empty(weight_matrices.shape[0], dtype=np.float64)
    for instance in prange(weight_matrices.shape[0]):
        values[instance] = _rho_star_exact(weight_matrices[instance])
    return values


def _generate_instances(n: int, seeds: range) -> np.ndarray:
    matrices = np.zeros((len(seeds), n, n), dtype=np.float64)
    upper_indices = np.triu_indices(n, 1)
    for instance, seed in enumerate(seeds):
        upper_weights = np.random.default_rng(seed).uniform(
            0.0,
            2.0,
            size=n * (n - 1) // 2,
        )
        matrices[instance][upper_indices] = upper_weights
        matrices[instance][(upper_indices[1], upper_indices[0])] = upper_weights
    return matrices


def _summary(n: int, values: np.ndarray) -> dict[str, float | int]:
    median, percentile_90, percentile_99 = np.percentile(
        values, [50.0, 90.0, 99.0]
    )
    return {
        "n": n,
        "mean": float(np.mean(values)),
        "median": float(median),
        "percentile_90": float(percentile_90),
        "percentile_99": float(percentile_99),
        "max": float(np.max(values)),
    }


def _print_markdown(rows: list[dict[str, float | int]]) -> None:
    print("| n | Mean | Median | 90th pct | 99th pct | Max |")
    print("|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['n']} | {row['mean']:.2f} | {row['median']:.2f} | "
            f"{row['percentile_90']:.2f} | {row['percentile_99']:.2f} | "
            f"{row['max']:.2f} |"
        )


def _write_csv(
    path: Path,
    rows: list[dict[str, float | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_samples_csv(
    path: Path,
    sizes: list[int],
    seeds: range,
    values_by_size: list[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["n", "seed", "rho_star"])
        for n, values in zip(sizes, values_by_size):
            for seed, value in zip(seeds, values):
                writer.writerow([n, seed, float(value)])


def _write_figure(
    path: Path,
    sizes: list[int],
    values_by_size: list[np.ndarray],
) -> None:
    """Single-panel Mean/Median/90th/99th/Max vs n, matching the manuscript's
    Figure 2 layout exactly (one line per summary statistic, not a per-instance
    scatter or a fixed 50/75/99 quantile triple)."""
    import matplotlib.pyplot as plt

    rows = [_summary(n, values) for n, values in zip(sizes, values_by_size)]

    figure, ax = plt.subplots(figsize=(6.0, 4.2))
    series = (
        ("mean", "Mean", "#3B82C4", "o"),
        ("median", "Median", "#42B24A", "s"),
        ("percentile_90", "90th pct", "#F2994A", "^"),
        ("percentile_99", "99th pct", "#D6455C", "D"),
        ("max", "Max", "#6B6B6B", "v"),
    )
    for key, label, color, marker in series:
        ax.plot(
            sizes,
            [row[key] for row in rows],
            marker=marker,
            color=color,
            label=label,
            linewidth=1.8,
            markersize=6,
        )
    ax.set_xlabel(r"Number of variables $n$")
    ax.set_ylabel(r"$\rho^\star$")
    ax.set_xticks(sizes)
    ax.set_ylim(0.0, max(1.0, 1.05 * max(row["max"] for row in rows)))
    ax.legend(frameon=False)
    ax.tick_params(direction="in", top=True, right=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[8, 12, 16, 20],
        help="Even graph sizes (default: 8 12 16 20).",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=50,
        help="Number of instances per size (default: 50).",
    )
    parser.add_argument(
        "--first-seed",
        type=int,
        default=0,
        help="First integer instance seed (default: 0).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Numba worker threads (default: min(8, available CPUs)).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the unrounded summary CSV.",
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        help="Optional path for all unrounded per-instance values.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        help="Optional path for a PDF or raster figure of the samples.",
    )
    args = parser.parse_args()

    if args.instances <= 0:
        parser.error("--instances must be positive")
    if any(n <= 0 or n % 2 for n in args.sizes):
        parser.error("--sizes must contain positive even integers")
    if args.threads <= 0:
        parser.error("--threads must be positive")

    set_num_threads(args.threads)
    seeds = range(args.first_seed, args.first_seed + args.instances)
    rows: list[dict[str, float | int]] = []
    values_by_size: list[np.ndarray] = []
    for n in args.sizes:
        matrices = _generate_instances(n, seeds)
        values = _rho_star_batch(matrices)
        values_by_size.append(values)
        rows.append(_summary(n, values))

    _print_markdown(rows)
    if args.output is not None:
        _write_csv(args.output, rows)
        print(f"\nWrote unrounded results to {args.output}")
    if args.samples_output is not None:
        _write_samples_csv(
            args.samples_output,
            args.sizes,
            seeds,
            values_by_size,
        )
        print(f"Wrote per-instance results to {args.samples_output}")
    if args.figure is not None:
        _write_figure(args.figure, args.sizes, values_by_size)
        print(f"Wrote figure to {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
