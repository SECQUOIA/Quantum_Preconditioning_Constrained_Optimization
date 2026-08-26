#!/usr/bin/env python3
"""
Validate QAOA parameter transfer by plotting the cost landscape at N=20
alongside each larger N that received a transfer from it, for a single
instance (seed=0 by default).

Why seed=0, and why a single instance rather than an average: seed=0 is the
only seed with a genuine N=20 reference for every layer/penalty in this
dataset (including p=3 penalties patched in from a collaborator -- see
seed_transfer.py), so it's the cleanest case to check. Averaging landscapes
across many random instances would wash out the exact basin structure this
is meant to verify lines up -- this is a per-instance sanity check ("does
this transferred point land in the same place"), not a statistical claim.

p=1 (layer=1): the full landscape is 2D (gamma, beta) -- plotted directly via
the closed-form analytical formula (cheap). gamma is shown in "N=20-
equivalent" units (gamma_plot = gamma * sqrt(N/20)), matching the transfer
relation gamma(N) = gamma(20) * sqrt(20/N): if transfer is exact, every N
should overlay on the same axes.

p=2,3 (layer>1): the angle space is 2p-dimensional, so a full landscape can't
be plotted. Instead this takes a 2D slice: vary ONE layer's (gamma, beta)
over the same rescaled grid, holding every other layer's angle fixed at its
actual (optimized or transferred) value, and marks where the real angles
land. Requires a full state-vector build per grid point (O(2^N)) -- slow for
N=28, so grid cells are parallelized with -j.

Usage:
    # p=1, N=20 vs all transfer targets
    python plot_transfer_landscape.py --data-root /path/to/complete_random \\
        --seed 0 --penalty 1.5 --layer 1 --n-values 20,24,28,32,36,40

    # p=3, N=20 vs N=24,28, slicing layer 1's angles, 8 parallel workers
    python plot_transfer_landscape.py --data-root /path/to/complete_random \\
        --seed 0 --penalty 1.5 --layer 3 --n-values 20,24,28 \\
        --vary-layer 1 --resolution 41 -j 8

External dependency: this script imports the internal, proprietary
`quantum_preconditioning` package (Rigetti's QAOA state-vector tooling), but
only for recomputing a landscape from scratch (--recompute, or a cache miss).
Source: https://github.com/anurag-r20/Quantum_Preconditioning_Rigetti,
commit 211426d23680b0a2b3e782c982699718c04aa68d (quantum_preconditioning_src,
v0.27.0). That package cannot be redistributed here, so the imports below are
local to the functions that need them rather than top-level, which lets the
common case, replotting from the cached grid data already checked into
`scripts/landscape_cache/`, run with no proprietary dependency at all. Only
`--recompute` (or a cache-cold run) needs the package installed and importable.
"""
from __future__ import annotations

import argparse
import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless -- this runs on the cluster, no display
import matplotlib.pyplot as plt
import numpy as np

N_REF = 20  # reference size that everything transfers from

# Sized for the 6-panel (p=1) case at the existing figsize=(4.5*n, 4.2) -- verified to
# fit without title/label overlap at n=6; leave figsize alone, only these changed.
TITLE_FS = 19
LABEL_FS = 21
TICK_FS = 16
CBAR_LABEL_FS = 21
CBAR_TICK_FS = 16
MARKER_SIZE = 14


# ---------------------------------------------------------------------------
# Problem loading (same convention as generate_high_penalties.py)
# ---------------------------------------------------------------------------

def _load_problem(data_root: Path, N: int, seed: int, pen: float) -> QuadraticSpinProblem:
    from quantum_preconditioning.problem.quadratic import QuadraticSpinProblem
    from quantum_preconditioning.problem.disk import from_termlist

    prob = from_termlist(str(data_root / f"N={N}" / f"seed={seed}" / "problem.dat"), True)
    penal = from_termlist(str(data_root / f"N={N}" / "penalty.dat"), True)
    terms: dict = defaultdict(float)
    for t, v in prob.items():
        if abs(v) > 1e-10:
            terms[t] += v
    for t, v in penal.items():
        if abs(v) > 1e-10:
            terms[t] += pen * v
    return QuadraticSpinProblem(dict(sorted(
        {t: v for t, v in terms.items() if abs(v) > 1e-10}.items()
    )))


def _load_angles(data_root: Path, N: int, seed: int, pen: float, layer: int) -> np.ndarray:
    path = data_root / f"N={N}" / f"seed={seed}" / f"n_qaoa_layers={layer}" / f"qaoa_angles_pen={pen:.3f}.dat"
    if not path.is_file():
        raise FileNotFoundError(f"No angles file for N={N} seed={seed} layer={layer} pen={pen:.3f}: {path}")
    return np.genfromtxt(str(path))[1:]  # strip stored cost, keep [γ1, β1, γ2, β2, ...]


# ---------------------------------------------------------------------------
# p=1: closed-form landscape
# ---------------------------------------------------------------------------

def _precompute_p1(problem: QuadraticSpinProblem) -> dict:
    N = problem.n_variables
    constant = 0.0
    linear: dict[int, float] = {}
    quadratic: dict[tuple, float] = {}
    for variables, coefficient in problem:
        if len(variables) == 0:
            constant += coefficient
        elif len(variables) == 1:
            linear[variables[0]] = coefficient
        elif len(variables) == 2:
            quadratic[variables] = coefficient

    connected: dict[int, np.ndarray] = {}
    for var in range(N):
        rows = [
            (variables[0] if var == variables[1] else variables[1], coeff)
            for variables, coeff in problem
            if len(variables) == 2 and var in variables
        ]
        connected[var] = np.array(rows, dtype=float) if rows else np.empty((0, 2), dtype=float)

    return dict(constant=constant, linear=linear, quadratic=quadratic, connected=connected)


def _eval_cost_p1(p1_data: dict, gamma: float, beta: float) -> float:
    """Reference (slow) scalar evaluator -- calls the library's numba functions directly.
    Used only as a correctness oracle for the vectorized landscape below, not for the
    actual grid sweep (each call costs ~0.3-0.6ms of numba parallel-dispatch overhead,
    which is fine once but would take hours over a 40000-point grid)."""
    from quantum_preconditioning.utils.qaoa_quadratic_p1_formula import compute_Zi, compute_ZiZj

    cost = p1_data["constant"]
    for i, h in p1_data["linear"].items():
        cost += h * compute_Zi(gamma, beta, p1_data["connected"][i], h)
    for (i, j), J in p1_data["quadratic"].items():
        cost += J * compute_ZiZj(
            gamma, beta,
            i, p1_data["connected"][i],
            j, p1_data["connected"][j],
            J,
            p1_data["linear"].get(i, 0.0),
            p1_data["linear"].get(j, 0.0),
        )
    return cost


def _masked_cos_prod(gamma_vals: np.ndarray, terms: np.ndarray, exclude: Optional[int]) -> np.ndarray:
    if len(terms) == 0:
        return np.ones_like(gamma_vals)
    if exclude is not None:
        keep = terms[:, 0].astype(int) != exclude
        terms = terms[keep]
    if len(terms) == 0:
        return np.ones_like(gamma_vals)
    return np.prod(np.cos(2.0 * np.outer(gamma_vals, terms[:, 1])), axis=1)


def _get_weight(terms: np.ndarray, k: int) -> float:
    if len(terms) == 0:
        return 0.0
    match = terms[:, 0].astype(int) == k
    return float(terms[match, 1][0]) if match.any() else 0.0


def _zi_gamma_part(gamma_vals: np.ndarray, terms_i: np.ndarray, term_i: float) -> np.ndarray:
    """f_i(gamma) such that compute_Zi(gamma, beta) == -sin(2*beta) * f_i(gamma)."""
    if term_i == 0.0:
        return np.zeros_like(gamma_vals)
    return np.sin(2.0 * gamma_vals * term_i) * _masked_cos_prod(gamma_vals, terms_i, None)


def _zizj_gamma_parts(
    gamma_vals: np.ndarray, i: int, terms_i: np.ndarray, j: int, terms_j: np.ndarray,
    term_ij: float, term_i: float, term_j: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(A(gamma), B(gamma)) such that
    compute_ZiZj(gamma, beta) == -0.5*sin(4*beta)*A(gamma) - 0.5*sin(2*beta)**2*B(gamma)."""
    termAi = np.cos(2.0 * gamma_vals * term_i) * _masked_cos_prod(gamma_vals, terms_i, j)
    termAj = np.cos(2.0 * gamma_vals * term_j) * _masked_cos_prod(gamma_vals, terms_j, i)
    A = np.sin(2.0 * gamma_vals * term_ij) * (termAi + termAj)

    idx_i = set(terms_i[:, 0].astype(int)) if len(terms_i) else set()
    idx_j = set(terms_j[:, 0].astype(int)) if len(terms_j) else set()
    union_idx = (idx_i | idx_j) - {i, j}

    termBp = np.cos(2.0 * gamma_vals * (term_i + term_j))
    termBm = np.cos(2.0 * gamma_vals * (term_i - term_j))
    for k in union_idx:
        Wki, Wkj = _get_weight(terms_i, k), _get_weight(terms_j, k)
        termBp = termBp * np.cos(2.0 * gamma_vals * (Wki + Wkj))
        termBm = termBm * np.cos(2.0 * gamma_vals * (Wki - Wkj))

    return A, termBp - termBm


def _selftest_p1(p1_data: dict, n_points: int = 8, atol: float = 1e-9) -> None:
    """Cross-check the vectorized landscape against the reference scalar formula
    (the library's own numba functions) at random points -- raises if they disagree."""
    rng = np.random.default_rng(12345)
    for _ in range(n_points):
        gamma = rng.uniform(0.0, 1.5)
        beta = rng.uniform(0.0, np.pi / 2)
        gamma_arr = np.array([gamma])
        vec_cost = p1_data["constant"]
        for i, h in p1_data["linear"].items():
            vec_cost += h * (-np.sin(2 * beta) * _zi_gamma_part(gamma_arr, p1_data["connected"][i], h)[0])
        for (i, j), J in p1_data["quadratic"].items():
            A, B = _zizj_gamma_parts(
                gamma_arr, i, p1_data["connected"][i], j, p1_data["connected"][j],
                J, p1_data["linear"].get(i, 0.0), p1_data["linear"].get(j, 0.0),
            )
            vec_cost += J * (-0.5 * np.sin(4 * beta) * A[0] - 0.5 * np.sin(2 * beta) ** 2 * B[0])
        ref_cost = _eval_cost_p1(p1_data, gamma, beta)
        if not np.isclose(vec_cost, ref_cost, atol=atol):
            raise AssertionError(
                f"Vectorized p=1 landscape disagrees with reference formula at "
                f"gamma={gamma}, beta={beta}: vectorized={vec_cost}, reference={ref_cost}"
            )


def _landscape_p1(
    problem: QuadraticSpinProblem, N: int, resolution: int,
    u_range: tuple[float, float], beta_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (u_grid, beta_grid, Z) where u = gamma * sqrt(N/N_REF).

    Vectorized over the whole grid: compute_Zi/compute_ZiZj are separable into a
    beta-only prefactor times a gamma-only function (standard p=1 QAOA structure), so
    each term contributes one outer product over the grid instead of one numba call
    per grid point -- avoids the ~0.3-0.6ms per-call dispatch cost of numba's
    parallel=True kernels, which otherwise turns a 40000-point grid into hours.

    u_range/beta_range are passed in (not fixed to [0, pi/sqrt(20)] x [0, pi/2]) because
    BFGS is unconstrained after initialization and QAOA cost is periodic -- the actual
    optimized/transferred angle can legitimately land outside the initialization box.
    """
    p1_data = _precompute_p1(problem)
    _selftest_p1(p1_data)  # verify against the library's own formula before trusting this

    u_vals = np.linspace(u_range[0], u_range[1], resolution)
    beta_vals = np.linspace(beta_range[0], beta_range[1], resolution)
    scale = np.sqrt(N_REF / N)              # gamma(N) = u * sqrt(N_REF/N)
    gamma_vals = u_vals * scale

    Z = np.full((resolution, resolution), p1_data["constant"])
    sin_2beta = np.sin(2.0 * beta_vals)
    sin_4beta = np.sin(4.0 * beta_vals)
    sin2_2beta = np.sin(2.0 * beta_vals) ** 2

    for i, h in p1_data["linear"].items():
        f_i = _zi_gamma_part(gamma_vals, p1_data["connected"][i], h)
        Z += h * np.outer(-sin_2beta, f_i)
    for (i, j), J in p1_data["quadratic"].items():
        A, B = _zizj_gamma_parts(
            gamma_vals, i, p1_data["connected"][i], j, p1_data["connected"][j],
            J, p1_data["linear"].get(i, 0.0), p1_data["linear"].get(j, 0.0),
        )
        Z += J * (np.outer(-0.5 * sin_4beta, A) + np.outer(-0.5 * sin2_2beta, B))

    return u_vals, beta_vals, Z


# ---------------------------------------------------------------------------
# p=2,3: state-vector slice landscape (one layer varied, rest held fixed)
# ---------------------------------------------------------------------------

_WORKER_GRAPH_COST = None
_WORKER_BASE_ANGLES = None
_WORKER_VARY_IDX = None


def _setup_worker_state(data_root: Path, N: int, seed: int, pen: float, base_angles: np.ndarray, vary_idx: int) -> None:
    from quantum_preconditioning.solver.qaoa_state_vector import (
        QAOAStateVector as SolverSV,
        get_graph_cost,
    )

    global _WORKER_GRAPH_COST, _WORKER_BASE_ANGLES, _WORKER_VARY_IDX
    problem = _load_problem(data_root, N, seed, pen)
    _WORKER_GRAPH_COST = get_graph_cost(N, *SolverSV(angles=())._format_problem(problem))
    _WORKER_BASE_ANGLES = base_angles
    _WORKER_VARY_IDX = vary_idx


def _init_pool_worker(data_root: Path, N: int, seed: int, pen: float, base_angles: np.ndarray, vary_idx: int) -> None:
    """Pool initializer (only used when -j > 1): pin to 1 numba thread per process, since
    parallelism comes from multiple worker processes here, not per-call numba threading."""
    import numba as nb
    nb.set_num_threads(1)
    _setup_worker_state(data_root, N, seed, pen, base_angles, vary_idx)


def _worker_eval(task: tuple) -> tuple:
    from quantum_preconditioning.solver.qaoa_state_vector import get_average_cost, get_qaoa_psi

    bi, ui, gamma, beta = task
    angles = _WORKER_BASE_ANGLES.copy()
    angles[_WORKER_VARY_IDX] = gamma
    angles[_WORKER_VARY_IDX + 1] = beta
    cost = float(get_average_cost(get_qaoa_psi(_WORKER_GRAPH_COST, angles), _WORKER_GRAPH_COST))
    return bi, ui, cost


def _landscape_slice(
    data_root: Path, N: int, seed: int, pen: float, base_angles: np.ndarray,
    vary_layer: int, resolution: int, workers: int,
    u_range: tuple[float, float], beta_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (u_grid, beta_grid, Z) for the (gamma, beta) pair of `vary_layer` (1-indexed).
    u_range/beta_range passed in for the same reason as _landscape_p1 -- see its docstring."""
    vary_idx = 2 * (vary_layer - 1)
    u_vals = np.linspace(u_range[0], u_range[1], resolution)
    beta_vals = np.linspace(beta_range[0], beta_range[1], resolution)
    scale = np.sqrt(N_REF / N)

    tasks = [
        (bi, ui, u * scale, beta)
        for bi, beta in enumerate(beta_vals)
        for ui, u in enumerate(u_vals)
    ]

    Z = np.empty((resolution, resolution))
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_pool_worker, initargs=(data_root, N, seed, pen, base_angles, vary_idx)) as pool:
            for bi, ui, cost in pool.imap_unordered(_worker_eval, tasks, chunksize=4):
                Z[bi, ui] = cost
    else:
        # Single seed, one process: let numba use however many threads NUMBA_NUM_THREADS
        # specifies (or its default) for each state-vector call, instead of forcing 1.
        _setup_worker_state(data_root, N, seed, pen, base_angles, vary_idx)
        for task in tasks:
            bi, ui, cost = _worker_eval(task)
            Z[bi, ui] = cost
    return u_vals, beta_vals, Z


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _operating_point(angles: np.ndarray, vary_layer: int, N: int) -> tuple[float, float]:
    idx = 2 * (vary_layer - 1)
    gamma, beta = angles[idx], angles[idx + 1]
    return gamma * np.sqrt(N / N_REF), beta  # convert to shared u-axis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True, help="Path to complete_random/ data directory.")
    ap.add_argument("--seed", type=int, default=0, help="Instance to check (default: 0 -- see module docstring).")
    ap.add_argument("--penalty", type=float, required=True)
    ap.add_argument("--layer", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument(
        "--n-values", type=str, default=None,
        help="Comma-separated N's to compare against N=20 (default: 20,24,28,32,36,40 for layer=1; 20,24,28 otherwise).",
    )
    ap.add_argument(
        "--vary-layer", type=int, default=1,
        help="For layer=2,3: which layer's (gamma,beta) to scan, others held fixed (default: 1).",
    )
    ap.add_argument(
        "--resolution", type=int, default=None,
        help="Grid points per axis (default: 200 for layer=1, 41 for layer=2,3).",
    )
    ap.add_argument("-j", "--workers", type=int, default=1, help="Parallel workers for layer=2,3 (default: 1).")
    ap.add_argument("--output-dir", type=Path, default=Path("landscape_plots"))
    ap.add_argument(
        "--recompute", action="store_true",
        help="Ignore any cached grid data and recompute from scratch (e.g. if angle files changed on disk).",
    )
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    n_values = (
        [int(n) for n in args.n_values.split(",")] if args.n_values
        else ([20, 24, 28, 32, 36, 40] if args.layer == 1 else [20, 24, 28])
    )
    if N_REF not in n_values:
        n_values = [N_REF] + n_values
    resolution = args.resolution or (200 if args.layer == 1 else 41)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Grid computation (esp. layer=2,3 state-vector sweeps) is the expensive part;
    # plot styling (colors, markers, labels) is not. Cache the former so cosmetic
    # replots don't repeat the latter.
    n_tag = "-".join(map(str, n_values))
    vary_tag = f"_vary{args.vary_layer}" if args.layer > 1 else ""
    cache_path = args.output_dir / (
        f"cache_p{args.layer}_pen{args.penalty:.3f}_seed{args.seed}"
        f"_res{resolution}_N{n_tag}{vary_tag}.npz"
    )

    if args.layer == 1:
        # compute_Zi/compute_ZiZj are numba parallel=True kernels. For arrays this small,
        # per-call thread-spawn/sync overhead dominates and swamps any parallel speedup
        # (>>10x slower than serial) -- same fix as generate_high_penalties.py's p=1 path.
        import numba as nb
        nb.set_num_threads(1)

    print(f"Data root   : {data_root}")
    print(f"Seed        : {args.seed}")
    print(f"Penalty     : {args.penalty}")
    print(f"Layer       : {args.layer}")
    print(f"N values    : {n_values}")
    print(f"Resolution  : {resolution}")

    if cache_path.is_file() and not args.recompute:
        print(f"Loading cached grid data: {cache_path}")
        cached = np.load(cache_path)
        points = {N: (float(cached[f"point_{N}"][0]), float(cached[f"point_{N}"][1])) for N in n_values}
        grids = {N: (cached[f"u_{N}"], cached[f"beta_{N}"], cached[f"Z_{N}"]) for N in n_values}
    else:
        # First pass: load each N's actual angles to find where the operating point sits.
        # BFGS is unconstrained after initialization and QAOA cost is periodic, so a found
        # angle can legitimately land outside the [0, pi/sqrt(20)] x [0, pi/2] init box --
        # the plotted range has to be widened to actually contain every point, or markers
        # end up stranded off-grid (as happened before this fix).
        problems: dict[int, QuadraticSpinProblem] = {}
        points = {}
        for N in n_values:
            problems[N] = _load_problem(data_root, N, args.seed, args.penalty)
            angles = _load_angles(data_root, N, args.seed, args.penalty, args.layer)
            points[N] = (
                (angles[0] * np.sqrt(N / N_REF), angles[1]) if args.layer == 1
                else _operating_point(angles, args.vary_layer, N)
            )

        def _padded_range(default_lo: float, default_hi: float, values) -> tuple[float, float]:
            # Margin as a fraction of the span rather than of the value itself -- a
            # value-relative margin (e.g. value * 1.1) shrinks to ~nothing when the point
            # sits near 0, which left marker glyphs visually clipped against the axis
            # frame even though their center coordinate was technically in-range.
            lo = min([default_lo] + list(values))
            hi = max([default_hi] + list(values))
            margin = 0.08 * (hi - lo)
            return (lo - margin, hi + margin)

        default_u = np.pi / np.sqrt(N_REF)
        default_beta = np.pi / 2
        u_range = _padded_range(0.0, default_u, [p[0] for p in points.values()])
        beta_range = _padded_range(0.0, default_beta, [p[1] for p in points.values()])

        grids = {}
        for N in n_values:
            print(f"  computing landscape for N={N} ...")
            angles = _load_angles(data_root, N, args.seed, args.penalty, args.layer)
            if args.layer == 1:
                grids[N] = _landscape_p1(problems[N], N, resolution, u_range, beta_range)
            else:
                grids[N] = _landscape_slice(
                    data_root, N, args.seed, args.penalty, angles, args.vary_layer, resolution, args.workers,
                    u_range, beta_range,
                )

        save_dict = {}
        for N in n_values:
            u_vals, beta_vals, Z = grids[N]
            save_dict[f"u_{N}"], save_dict[f"beta_{N}"], save_dict[f"Z_{N}"] = u_vals, beta_vals, Z
            save_dict[f"point_{N}"] = np.array(points[N])
        np.savez(cache_path, **save_dict)
        print(f"Cached grid data: {cache_path}")

    # Raw cost magnitude naturally grows with N (more terms in the Hamiltonian), so a
    # shared absolute color scale would squeeze small-N landscapes into a sliver of the
    # range and make shape comparison meaningless. Normalize each panel to its own
    # [0,1] range instead -- what matters here is shape and operating-point location,
    # not absolute magnitude (which trivially differs by problem size).
    fig, axes = plt.subplots(1, len(n_values), figsize=(4.5 * len(n_values), 4.2), squeeze=False)
    axes = axes[0]
    for ax, N in zip(axes, n_values):
        u_vals, beta_vals, Z = grids[N]
        Z_norm = (Z - Z.min()) / (Z.max() - Z.min())
        im = ax.pcolormesh(u_vals, beta_vals, Z_norm, shading="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        gx, gy = points[N]
        marker = "*" if N == N_REF else "x"
        # 'x' has no separate fill -- it's drawn via markeredgecolor, not color, so both
        # must be set to red or the 'x' renders in whatever markeredgecolor says (was
        # hardcoded black, which is the bug that made it invisible against dark viridis).
        ax.plot(gx, gy, marker=marker, color="red", markersize=MARKER_SIZE, markeredgecolor="red", markeredgewidth=1.5)
        ax.set_title(f"N={N}" + (" (reference)" if N == N_REF else " (transferred)"), fontsize=TITLE_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.set_xlabel(
            r"$\gamma \cdot \sqrt{N/20}$" if args.layer == 1 else rf"$\gamma_{args.vary_layer} \cdot \sqrt{{N/20}}$",
            fontsize=LABEL_FS,
        )
        if ax is axes[0]:
            ax.set_ylabel(r"$\beta$" if args.layer == 1 else rf"$\beta_{args.vary_layer}$", fontsize=LABEL_FS)

    cbar = fig.colorbar(im, ax=axes, shrink=0.95, aspect=15)
    cbar.set_label("normalized ⟨C(γ,β)⟩", fontsize=CBAR_LABEL_FS)
    cbar.ax.tick_params(labelsize=CBAR_TICK_FS)

    out_path = args.output_dir / f"landscape_p{args.layer}_pen{args.penalty:.3f}_seed{args.seed}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
