from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlotConfig:
    """Plot configuration for one analysis family/setting.
    
    Parameters
    ----------
    n : int
        Problem size.
    layers : int, optional
        Quantum/QAOA depth when ``family='quantum'``.
    family : {'quantum', 'classical'}, default='quantum'
        Preconditioner family.
    sweeps : int, optional
        Classical preconditioner sweep count when ``family='classical'``.
    """
    n: int
    layers: Optional[int] = None
    family: str = "quantum"
    sweeps: Optional[int] = None

    @property
    def family_param_name(self) -> str:
        """Return the short parameter name for the configured family."""
        return "p" if self.family == "quantum" else "s"

    @property
    def family_value(self) -> Optional[int]:
        """Return the active family parameter value, if configured."""
        return self.layers if self.family == "quantum" else self.sweeps

    @property
    def family_label(self) -> str:
        """Return a compact label for the configured family value."""
        value = self.family_value
        if value is None:
            return self.family
        return f"{self.family_param_name}={value}"

    @property
    def context_label(self) -> str:
        """Return a compact label containing problem size and family setting."""
        value = self.family_value
        if value is None:
            return f"N={self.n}"
        field = "layers" if self.family == "quantum" else "sweeps"
        return f"N={self.n}, {field}={value}"


def apply_plot_style() -> None:
    """Set consistent plotting defaults for notebook figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (9.2, 5.8),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "font.size": 12,
        }
    )


def _require_columns(df: pd.DataFrame, cols: Iterable[str], source: str) -> None:
    """Validate that a dataframe contains required columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to validate.
    cols : iterable of str
        Required column names.
    source : str
        Human-readable source label used in error messages.
    
    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def _parse_penalty_from_name(name: str) -> float:
    """Extract the penalty value encoded in a run name.
    
    Parameters
    ----------
    name : str
        Run name such as ``baseline`` or ``precond_pen=0.300``.
    
    Returns
    -------
    float
        Parsed penalty, ``-1`` for baseline, or infinity when parsing fails.
    """
    if name == "baseline":
        return -1.0
    try:
        return float(str(name).split("pen=")[1])
    except (IndexError, ValueError):
        return float("inf")


def _sorted_variant_names(names: Iterable[str], include_baseline: bool) -> List[str]:
    """Sort run names by penalty and optionally include the baseline."""
    ordered = sorted(names, key=lambda n: (_parse_penalty_from_name(str(n)), str(n)))
    if include_baseline:
        return ordered
    return [n for n in ordered if str(n) != "baseline"]


def _normalize_penalty_filter(penalty_filter: Optional[Iterable[float]]) -> Optional[set[float]]:
    """Normalize an optional penalty filter to rounded penalty values."""
    if penalty_filter is None:
        return None
    return {round(float(x), 3) for x in penalty_filter}


def _penalty_allowed(name: str, penalty_filter: Optional[set[float]]) -> bool:
    """Return whether a run name passes an optional penalty filter."""
    if name == "baseline" or penalty_filter is None:
        return True
    pen = _parse_penalty_from_name(name)
    if np.isinf(pen):
        return False
    return round(pen, 3) in penalty_filter


def _mean_sem(series: pd.Series) -> tuple[float, float, int]:
    """Compute mean, standard error of the mean, and count.
    
    Parameters
    ----------
    series : pandas.Series
        Numeric values with possible missing entries.
    
    Returns
    -------
    tuple of float, float, int
        Mean, SEM, and non-missing count.
    """
    vals = series.dropna().astype(float)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(vals.mean())
    sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, sem, n


def _fit_exponential_order(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    """Fit ``y = scale * base**x`` on positive finite observations.
    
    Returns
    -------
    tuple of float, float
        Exponential base and scale. Returns NaNs when fewer than two valid points exist.
    """
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (y_arr > 0)
    if np.sum(valid) < 2:
        return float("nan"), float("nan")

    slope, intercept = np.polyfit(x_arr[valid], np.log(y_arr[valid]), 1)
    base = float(np.exp(slope))
    scale = float(np.exp(intercept))
    return base, scale


def _filter_rows_reaching_original_optimum(
    df: pd.DataFrame,
    tol: float = 1e-9,
) -> pd.DataFrame:
    """Keep rows whose final or trajectory objective reaches the original optimum.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Summary dataframe containing baseline and preconditioned runs.
    tol : float, default=1e-9
        Objective tolerance for matching the baseline optimum.
    
    Returns
    -------
    pandas.DataFrame
        Filtered dataframe with ``time_to_best_sec`` recomputed from trajectories when available.
    """
    required = {"n", "seed", "presolve", "name", "objective_baseline"}
    if not required.issubset(df.columns):
        return df.copy()

    baseline = (
        df[df["name"] == "baseline"][["n", "seed", "presolve", "objective_baseline"]]
        .drop_duplicates(subset=["n", "seed", "presolve"])
        .rename(columns={"objective_baseline": "baseline_opt"})
    )
    out = df.merge(baseline, on=["n", "seed", "presolve"], how="left")
    keep = (out["name"] == "baseline") | (
        out["baseline_opt"].notna()
        & out["objective_baseline"].notna()
        & (out["objective_baseline"] - out["baseline_opt"]).abs().le(tol)
    )
    out = out.loc[keep].copy()

    # For "time to best solution", use the first incumbent that actually
    # reaches the original baseline optimum. The stored time_to_best_sec is the
    # last incumbent time for the solved model objective, which can differ for
    # preconditioned objectives.
    if {"trajectory", "time_to_best_sec"}.issubset(out.columns):
        def _first_original_optimum_hit(row: pd.Series) -> float:
            """Return the first trajectory time that reaches the baseline optimum."""
            target = row.get("baseline_opt")
            raw = row.get("trajectory")
            if pd.isna(target) or pd.isna(raw):
                return float("nan")
            try:
                events = json.loads(str(raw))
            except (TypeError, ValueError, json.JSONDecodeError):
                return float("nan")
            for event in events:
                if len(event) < 2:
                    continue
                t, obj = event[0], event[1]
                try:
                    if float(obj) <= float(target) + tol:
                        return float(t)
                except (TypeError, ValueError):
                    continue
            return float("nan")

        out["time_to_best_sec"] = out.apply(_first_original_optimum_hit, axis=1)
        out = out[out["time_to_best_sec"].notna()].copy()

    return out.drop(columns=["baseline_opt"])


def _format_quantum_depth_label(layer: float | int) -> str:
    """Format a quantum depth value for plot labels."""
    value = float(layer)
    if np.isinf(value):
        return "∞"
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:g}"


def _format_epsilon_label(eps_value: float) -> str:
    """Format an approximation threshold for plot titles."""
    return "0" if abs(float(eps_value)) <= 1e-12 else rf"{100 * float(eps_value):g}\%"


def _sort_quantum_depths(values: Iterable[float | int]) -> list[float]:
    """Sort finite quantum depths before infinite depth."""
    unique = sorted({float(v) for v in values}, key=lambda v: (np.isinf(v), v))
    return unique


def _parse_z_bits_string(z_bits: str) -> list[int]:
    """Parse compact ``+``/``-`` spin strings into integer spins."""
    mapping = {"+": 1, "-": -1}
    return [mapping[ch] for ch in str(z_bits).strip()]


def _build_infinite_depth_instance(z_opt: list[int]):
    """Construct an idealized preconditioner instance centered on an optimum spin vector."""
    from .instance import IsingInstance

    n = len(z_opt)
    quadratic = {
        (i, j): float(-z_opt[i] * z_opt[j])
        for i in range(n)
        for j in range(i + 1, n)
    }
    return IsingInstance(n=n, constant=0.0, linear={}, quadratic=quadratic)


def _baseline_summary_candidates(results_dir: Path, n: int, presolve: bool) -> list[Path]:
    """Return candidate baseline summary CSV paths for a problem size."""
    mode = "on" if presolve else "off"
    candidates: list[Path] = []
    for layer in [1, 2, 3]:
        candidates.extend(
            [
                results_dir / f"N={n}_quantum_layers={layer}_presolve_{mode}.csv",
                results_dir / f"N={n}_layers={layer}_presolve_{mode}.csv",
            ]
        )
    candidates.append(results_dir / f"N={n}_presolve_{mode}.csv")
    return candidates


def _resolve_existing_data_root(
    stored_data_root: str | Path | None,
    fallback_data_root: str | Path,
    results_dir: Path,
) -> Path:
    """Resolve a stored or fallback data root against common repo-relative locations."""
    repo_root = Path(__file__).resolve().parents[1]
    candidates: list[Path] = []

    for value in [stored_data_root, fallback_data_root]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        path = Path(str(value))
        candidates.append(path)
        if not path.is_absolute():
            candidates.append((repo_root / path).resolve())
            candidates.append((results_dir.parent / path).resolve())
            candidates.append((Path.cwd() / path).resolve())
        else:
            tail_parts = path.parts[-2:] if len(path.parts) >= 2 else path.parts
            if tail_parts:
                tail = Path(*tail_parts)
                candidates.append((repo_root / tail).resolve())
                candidates.append((results_dir.parent / tail).resolve())

    # Standard local repo location.
    candidates.append((repo_root / "one_equality_new" / "complete_random").resolve())
    candidates.append((results_dir.parent / "one_equality_new" / "complete_random").resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    tried = ", ".join(str(p) for p in seen)
    raise FileNotFoundError(f"Could not resolve a valid data_root. Tried: {tried}")


def _family_param_value(family: str, layers: Optional[int], sweeps: Optional[int]) -> int:
    """Return and validate the active family parameter value."""
    if family == "quantum":
        if layers is None:
            raise ValueError("layers is required for quantum data")
        return int(layers)
    if sweeps is None:
        raise ValueError("sweeps is required for classical data")
    return int(sweeps)


def _family_filter_column(family: str) -> str:
    """Return the dataframe column used for a family parameter."""
    return "layers" if family == "quantum" else "sweeps"


def _family_display_field(family: str) -> str:
    """Return the display name for a family parameter."""
    return "layers" if family == "quantum" else "sweeps"


def _summary_csv_candidates(
    results_dir: Path,
    n: int,
    presolve: bool,
    family: str,
    layers: Optional[int] = None,
    sweeps: Optional[int] = None,
) -> list[Path]:
    """Return candidate summary CSV paths for one family setting."""
    mode = "on" if presolve else "off"
    value = _family_param_value(family, layers, sweeps)
    if family == "quantum":
        return [
            results_dir / f"N={n}_quantum_layers={value}_presolve_{mode}.csv",
            results_dir / f"N={n}_layers={value}_presolve_{mode}.csv",
            results_dir / f"N={n}_presolve_{mode}.csv",
        ]
    return [
        results_dir / f"N={n}_classical_sweeps={value}_presolve_{mode}.csv",
    ]


def _trajectory_csv_candidates(
    results_dir: Path,
    n: int,
    presolve: bool,
    family: str,
    layers: Optional[int] = None,
    sweeps: Optional[int] = None,
) -> list[Path]:
    """Return candidate trajectory CSV paths for one family setting."""
    mode = "on" if presolve else "off"
    value = _family_param_value(family, layers, sweeps)
    if family == "quantum":
        return [
            results_dir / f"N={n}_quantum_layers={value}_presolve_{mode}_trajectories.csv",
            results_dir / f"N={n}_layers={value}_presolve_{mode}_trajectories.csv",
            results_dir / f"N={n}_presolve_{mode}_trajectories.csv",
        ]
    return [
        results_dir / f"N={n}_classical_sweeps={value}_presolve_{mode}_trajectories.csv",
    ]


def _first_existing_path(paths: Iterable[Path], missing_label: str) -> Path:
    """Return the first path that exists from a candidate list.
    
    Raises
    ------
    FileNotFoundError
        If none of the candidates exists.
    """
    candidates = list(paths)
    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Missing {missing_label}. Tried: {tried}")


def _inject_family_metadata(
    df: pd.DataFrame,
    family: str,
    layers: Optional[int] = None,
    sweeps: Optional[int] = None,
) -> pd.DataFrame:
    """Ensure a dataframe carries family, layer, and sweep metadata."""
    out = df.copy()
    if "preconditioner_family" not in out.columns:
        out["preconditioner_family"] = family
    else:
        out["preconditioner_family"] = out["preconditioner_family"].fillna(family)
    if "layers" not in out.columns:
        out["layers"] = layers
    if "sweeps" not in out.columns:
        out["sweeps"] = sweeps
    return out


def _filter_df_to_family_param(
    df: pd.DataFrame,
    family: str,
    layers: Optional[int] = None,
    sweeps: Optional[int] = None,
) -> pd.DataFrame:
    """Filter a dataframe to one preconditioner family parameter value."""
    out = df.copy()
    if "preconditioner_family" in out.columns:
        out = out[out["preconditioner_family"].fillna(family) == family].copy()
    column = _family_filter_column(family)
    value = _family_param_value(family, layers, sweeps)
    if column in out.columns:
        out = out[out[column] == value].copy()
    return out


def _filter_df_to_cfg(df: pd.DataFrame, cfg: PlotConfig) -> pd.DataFrame:
    """Filter a dataframe according to a :class:`PlotConfig`."""
    return _filter_df_to_family_param(df, cfg.family, cfg.layers, cfg.sweeps)


def _cfg_setting_suffix(cfg: PlotConfig) -> str:
    """Return a text suffix describing a plot configuration."""
    field = _family_display_field(cfg.family)
    value = cfg.family_value
    if value is None:
        return f"N={cfg.n}"
    return f"N={cfg.n}, {field}={value}"


def _filter_baseline_rows_to_cfg(df: pd.DataFrame, cfg: PlotConfig) -> pd.DataFrame:
    """Filter baseline rows to a plot configuration and problem size."""
    out = df.copy()
    if "n" in out.columns:
        out = out[out["n"] == cfg.n].copy()
    return _filter_df_to_cfg(out, cfg)


def load_summary_data(
    n: int,
    layers: Optional[int] = None,
    results_dir: Path | str = "../results",
    family: str = "quantum",
    sweeps: Optional[int] = None,
) -> pd.DataFrame:
    """Load presolve-on and presolve-off summary CSVs for one setting.
    
    Parameters
    ----------
    n : int
        Problem size.
    layers : int, optional
        Quantum depth.
    results_dir : path-like, default='../results'
        Directory containing summary CSV files.
    family : {'quantum', 'classical'}, default='quantum'
        Preconditioner family.
    sweeps : int, optional
        Classical sweep count.
    
    Returns
    -------
    pandas.DataFrame
        Summary rows filtered to the requested family setting.
    """
    results_dir = Path(results_dir)
    paths = [
        _first_existing_path(
            _summary_csv_candidates(results_dir, n, presolve=True, family=family, layers=layers, sweeps=sweeps),
            f"summary CSV for N={n}, family={family}, presolve=on",
        ),
        _first_existing_path(
            _summary_csv_candidates(results_dir, n, presolve=False, family=family, layers=layers, sweeps=sweeps),
            f"summary CSV for N={n}, family={family}, presolve=off",
        ),
    ]

    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing summary CSV: {path}")
        frames.append(pd.read_csv(path))

    df = pd.concat(frames, ignore_index=True)
    _require_columns(
        df,
        [
            "name",
            "objective_baseline",
            "penalty",
            "runtime_sec",
            "time_to_best_sec",
            "presolve",
            "seed",
        ],
        "summary CSVs",
    )
    df = _inject_family_metadata(df, family=family, layers=layers, sweeps=sweeps)
    return _filter_df_to_family_param(df, family=family, layers=layers, sweeps=sweeps)


def load_family_comparison_data(
    problem_sizes: Iterable[int],
    family: str,
    param_values: Iterable[int],
    presolve: bool,
    results_dir: Path | str = "../results",
) -> pd.DataFrame:
    """Load summary data across problem sizes and family parameter values."""
    results_dir = Path(results_dir)
    frames: list[pd.DataFrame] = []
    for param in param_values:
        for n_val in problem_sizes:
            try:
                summary_path = _first_existing_path(
                    _summary_csv_candidates(
                        results_dir,
                        int(n_val),
                        presolve,
                        family=family,
                        layers=param if family == "quantum" else None,
                        sweeps=param if family == "classical" else None,
                    ),
                    f"summary CSV for N={n_val}, family={family}, param={param}, presolve={'on' if presolve else 'off'}",
                )
            except FileNotFoundError:
                continue

            df = pd.read_csv(summary_path)
            df = _inject_family_metadata(
                df,
                family=family,
                layers=param if family == "quantum" else None,
                sweeps=param if family == "classical" else None,
            )
            df = _filter_df_to_family_param(
                df,
                family=family,
                layers=param if family == "quantum" else None,
                sweeps=param if family == "classical" else None,
            )
            df = df[df["presolve"] == bool(presolve)].copy()
            if df.empty:
                continue
            df["n"] = int(n_val)
            df["family_param"] = int(param)
            frames.append(df)

    if not frames:
        raise ValueError(f"No {family} comparison summary data could be loaded.")
    return pd.concat(frames, ignore_index=True)


def _expand_summary_trajectories(
    summary_df: pd.DataFrame,
    n_val: int,
    layers_val: float,
    presolve: bool,
) -> pd.DataFrame:
    """Expand the 'trajectory' JSON column of a summary dataframe into long-format rows.

    Used for p=inf, which stores trajectories inline in the summary CSV rather
    than in a separate trajectories file.
    """
    rows: list[dict] = []
    for row in summary_df.itertuples(index=False):
        raw = getattr(row, "trajectory", None)
        if pd.isna(raw) or not raw:
            continue
        try:
            events = json.loads(str(raw))
        except (TypeError, ValueError):
            continue
        running_best = float("inf")
        for idx, event in enumerate(events):
            t, v = float(event[0]), float(event[1])
            running_best = min(running_best, v)
            rows.append({
                "seed": int(row.seed),
                "layers": layers_val,
                "name": str(row.name),
                "penalty": getattr(row, "penalty", None),
                "presolve": bool(presolve),
                "event_idx": idx,
                "time_sec": t,
                "orig_obj": v,
                "running_best_orig_obj": running_best,
                "n": n_val,
                "preconditioner_family": "quantum",
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_family_comparison_trajectory_data(
    problem_sizes: Iterable[int],
    family: str,
    param_values: Iterable[int],
    presolve: bool,
    results_dir: Path | str = "../results",
) -> pd.DataFrame:
    """Load trajectory data across problem sizes and family parameter values.

    For quantum p=inf, trajectories are expanded from the summary CSV since
    no separate trajectory file exists for that case.
    """
    results_dir = Path(results_dir)
    mode = "on" if presolve else "off"
    frames: list[pd.DataFrame] = []
    for param in param_values:
        for n_val in problem_sizes:
            if family == "quantum" and np.isinf(float(param)):
                summary_path = results_dir / f"N={int(n_val)}_quantum_layers=inf_presolve_{mode}.csv"
                if not summary_path.exists():
                    continue
                expanded = _expand_summary_trajectories(
                    pd.read_csv(summary_path), int(n_val), float("inf"), presolve
                )
                if not expanded.empty:
                    frames.append(expanded)
                continue

            try:
                traj_path = _first_existing_path(
                    _trajectory_csv_candidates(
                        results_dir,
                        int(n_val),
                        presolve,
                        family=family,
                        layers=param if family == "quantum" else None,
                        sweeps=param if family == "classical" else None,
                    ),
                    f"trajectory CSV for N={n_val}, family={family}, param={param}, presolve={mode}",
                )
            except FileNotFoundError:
                continue
            df = pd.read_csv(traj_path)
            df = _inject_family_metadata(
                df,
                family=family,
                layers=param if family == "quantum" else None,
                sweeps=param if family == "classical" else None,
            )
            df = _filter_df_to_family_param(
                df,
                family=family,
                layers=param if family == "quantum" else None,
                sweeps=param if family == "classical" else None,
            )
            df = df[df["presolve"] == bool(presolve)].copy()
            df["n"] = int(n_val)
            frames.append(df)
    if not frames:
        raise ValueError(f"No {family} comparison trajectory data could be loaded.")
    return pd.concat(frames, ignore_index=True)


def load_quantum_infinite_depth_comparison_data(
    problem_sizes: Iterable[int],
    presolve: bool,
    results_dir: Path | str = "../results",
    data_root: Path | str = "one_equality_new/complete_random",
    threads: Optional[int] = None,
    mip_gap: Optional[float] = None,
    gurobi_seed: Optional[int] = None,
    output_flag: int = 0,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load or generate idealized infinite-depth quantum comparison data."""
    from .instance import read_dat
    from .solve import solve_bisection_ip

    results_dir = Path(results_dir)
    data_root = Path(data_root)
    mode = "on" if presolve else "off"
    frames: list[pd.DataFrame] = []

    for n_val in problem_sizes:
        cache_path = results_dir / f"N={int(n_val)}_quantum_layers=inf_presolve_{mode}.csv"
        if cache_path.exists() and not refresh:
            frames.append(pd.read_csv(cache_path))
            continue

        summary_path = _first_existing_path(
            _baseline_summary_candidates(results_dir, int(n_val), presolve),
            f"baseline summary CSV for N={n_val}, presolve={mode}",
        )
        summary_df = pd.read_csv(summary_path)
        baseline_rows = summary_df[
            (summary_df["name"] == "baseline") & (summary_df["presolve"] == bool(presolve))
        ].copy()
        if baseline_rows.empty:
            continue

        rows: list[dict[str, object]] = []
        for row in baseline_rows.itertuples(index=False):
            seed = int(row.seed)
            z_opt = _parse_z_bits_string(row.z_bits)

            baseline_root = _resolve_existing_data_root(
                getattr(row, "data_root", None),
                data_root,
                results_dir=results_dir,
            )
            baseline_path = baseline_root / f"N={int(n_val)}" / f"seed={seed}" / "problem.dat"
            if not baseline_path.exists():
                raise FileNotFoundError(f"Missing baseline problem for N={n_val}, seed={seed}: {baseline_path}")

            baseline_instance = read_dat(baseline_path)
            inf_instance = _build_infinite_depth_instance(z_opt)
            result = solve_bisection_ip(
                inf_instance,
                baseline_instance,
                name="precond_p=inf",
                penalty=0.0,
                mip_gap=mip_gap,
                threads=threads,
                seed=gurobi_seed,
                output_flag=output_flag,
                presolve=presolve,
            )
            out = result.to_row()
            out["seed"] = seed
            out["n"] = int(n_val)
            out["layers"] = float("inf")
            out["preconditioner_family"] = "quantum"
            out["data_root"] = str(baseline_root)
            rows.append(out)

        if not rows:
            continue

        cache_df = pd.DataFrame(rows)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df.to_csv(cache_path, index=False)
        frames.append(cache_df)

    if not frames:
        raise ValueError("No p=inf quantum comparison data could be loaded or generated.")
    return pd.concat(frames, ignore_index=True)


def load_layer_comparison_data(
    problem_sizes: Iterable[int],
    layers_list: Iterable[int],
    presolve: bool,
    results_dir: Path | str = "../results",
) -> pd.DataFrame:
    """Load quantum comparison data across QAOA layer counts."""
    return load_family_comparison_data(
        problem_sizes=problem_sizes,
        family="quantum",
        param_values=layers_list,
        presolve=presolve,
        results_dir=results_dir,
    )


def load_trajectory_data(
    n: int,
    layers: Optional[int] = None,
    results_dir: Path | str = "../results",
    family: str = "quantum",
    sweeps: Optional[int] = None,
) -> Dict[bool, pd.DataFrame]:
    """Load presolve-on and presolve-off trajectory CSVs for one setting."""
    results_dir = Path(results_dir)
    paths = {
        True: _first_existing_path(
            _trajectory_csv_candidates(results_dir, n, presolve=True, family=family, layers=layers, sweeps=sweeps),
            f"trajectory CSV for N={n}, family={family}, presolve=on",
        ),
        False: _first_existing_path(
            _trajectory_csv_candidates(results_dir, n, presolve=False, family=family, layers=layers, sweeps=sweeps),
            f"trajectory CSV for N={n}, family={family}, presolve=off",
        ),
    }

    out: Dict[bool, pd.DataFrame] = {}
    for presolve, path in paths.items():
        df = pd.read_csv(path)
        _require_columns(
            df,
            [
                "seed",
                "name",
                "presolve",
                "event_idx",
                "time_sec",
                "orig_obj",
                "running_best_orig_obj",
            ],
            str(path),
        )
        df = _inject_family_metadata(df, family=family, layers=layers, sweeps=sweeps)
        df = _filter_df_to_family_param(df, family=family, layers=layers, sweeps=sweeps)
        df = df[df["presolve"] == presolve].copy()
        out[presolve] = df

    return out


def prepare_penalty_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute objective-quality metrics for preconditioned penalty runs."""
    merge_keys = ["seed", "presolve", "n"] if "n" in df.columns else ["seed", "presolve"]
    baseline_cols = merge_keys + ["objective_baseline", "runtime_sec", "time_to_best_sec"]
    baseline = df[df["name"] == "baseline"][baseline_cols].copy()
    baseline = baseline.rename(
        columns={
            "objective_baseline": "baseline_obj",
            "runtime_sec": "baseline_runtime_sec",
            "time_to_best_sec": "baseline_time_to_best_sec",
        }
    )

    pre = df[df["penalty"].notna()].copy()
    pre["penalty_round"] = pre["penalty"].astype(float).round(3)
    pre = pre.merge(baseline, on=merge_keys, how="left")
    pre = pre[pre["baseline_obj"].notna()].copy()
    pre = pre[pre["baseline_obj"] != 0].copy()
    pre["approx_ratio_minus_1"] = pre["objective_baseline"] / pre["baseline_obj"] - 1.0
    return pre


def plot_ratio_minus_one(pre: pd.DataFrame, cfg: PlotConfig) -> plt.Figure:
    """Plot approximation ratio minus one against penalty for one setting."""
    fig, ax = plt.subplots(figsize=(7.1, 5.5))
    palette = {True: "#42C6C6", False: "#F08AA2"}

    for presolve in [True, False]:
        sub = pre[pre["presolve"] == presolve]
        if sub.empty:
            continue
        agg = (
            sub.groupby("penalty_round")["approx_ratio_minus_1"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("penalty_round")
        )
        agg["sem"] = agg["std"].fillna(0.0) / np.sqrt(agg["count"])
        ax.errorbar(
            agg["penalty_round"],
            agg["mean"],
            yerr=agg["sem"],
            fmt="o-",
            capsize=4,
            capthick=1.2,
            elinewidth=1.2,
            linewidth=2.3,
            markersize=7.0,
            markeredgewidth=1.1,
            markeredgecolor=palette[presolve],
            color=palette[presolve],
            label="Presolve on" if presolve else "Presolve off",
        )

    ax.axhline(0.0, color="#353B55", linestyle="--", linewidth=1.4, alpha=0.65)
    ax.set_xlabel("Penalty", fontsize=14)
    ax.set_ylabel("Approx Ratio - 1", fontsize=14)
    ax.set_title(f"Approx Ratio - 1 vs Penalty ({_cfg_setting_suffix(cfg)})", fontsize=15)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_alpha(0.85)
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=7, width=1.1, labelsize=12)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=3.5, width=0.9)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.legend(ncol=1, frameon=True, framealpha=0.88, edgecolor="#CCCCCC", fontsize=10, loc="upper right", handlelength=2.4, handletextpad=0.6)
    fig.tight_layout()
    return fig


def prepare_hit_time_metrics(
    summary_df: pd.DataFrame,
    traj_by_presolve: Dict[bool, pd.DataFrame],
    cfg: PlotConfig,
    thresholds: Iterable[float],
    penalty_factor: float = 1.0,
) -> pd.DataFrame:
    """Compute epsilon-hit and penalized hit-time records from trajectories."""
    eps = [float(x) for x in thresholds]
    if not eps:
        raise ValueError("thresholds cannot be empty")

    baseline_rows = summary_df[(summary_df["name"] == "baseline")].copy()
    baseline_rows = _filter_baseline_rows_to_cfg(baseline_rows, cfg)
    baseline_rows = baseline_rows.sort_values(["presolve", "seed"]).drop_duplicates(
        subset=["presolve", "seed"], keep="last"
    )
    baseline_opt_lookup = {
        (bool(row.presolve), int(row.seed)): float(row.objective_baseline)
        for row in baseline_rows.itertuples()
        if pd.notna(row.objective_baseline)
    }
    baseline_runtime_lookup = {
        (bool(row.presolve), int(row.seed)): float(row.runtime_sec)
        for row in baseline_rows.itertuples()
        if pd.notna(row.runtime_sec)
    }
    runtime_lookup = {
        (bool(row.presolve), str(row.name), int(row.seed)): float(row.runtime_sec)
        for row in summary_df.itertuples()
        if pd.notna(row.runtime_sec)
    }

    records: list[dict[str, object]] = []
    for presolve, traj_df in traj_by_presolve.items():
        per_seed = _build_per_seed_data(traj_df, penalty_filter=None, value_col="running_best_orig_obj")
        for name, seed_map in per_seed.items():
            penalty = np.nan if name == "baseline" else _parse_penalty_from_name(name)
            penalty_round = np.nan if name == "baseline" else round(float(penalty), 3)
            for seed, (times, vals) in seed_map.items():
                base_opt = baseline_opt_lookup.get((presolve, int(seed)))
                baseline_runtime_sec = baseline_runtime_lookup.get((presolve, int(seed)))
                runtime_sec = runtime_lookup.get((presolve, name, int(seed)))
                if base_opt is None:
                    continue
                for eps_value in eps:
                    threshold_value = base_opt + abs(base_opt) * eps_value
                    hit_time = next(
                        (float(t) for t, v in zip(times, vals) if float(v) <= threshold_value + 1e-8),
                        np.nan,
                    )
                    penalized_hit_time = (
                        hit_time
                        if not np.isnan(hit_time)
                        else (
                            float(runtime_sec) + float(penalty_factor) * float(baseline_runtime_sec)
                            if runtime_sec is not None and baseline_runtime_sec is not None
                            else np.nan
                        )
                    )
                    records.append(
                        {
                            "presolve": presolve,
                            "name": name,
                            "seed": int(seed),
                            "penalty": penalty,
                            "penalty_round": penalty_round,
                            "threshold": eps_value,
                            "threshold_label": f"{100 * eps_value:g}%",
                            "baseline_opt": base_opt,
                            "target_obj": threshold_value,
                            "runtime_sec": runtime_sec,
                            "baseline_runtime_sec": baseline_runtime_sec,
                            "hit_time_sec": hit_time,
                            "penalty_factor": float(penalty_factor),
                            "penalized_hit_time_sec": penalized_hit_time,
                            "hit_found": not np.isnan(hit_time),
                            "is_baseline": name == "baseline",
                        }
                    )

    hit_df = pd.DataFrame.from_records(records)
    if hit_df.empty:
        raise ValueError("No hit-time records could be constructed from the trajectory data.")
    return hit_df


def _aggregate_hit_time_stats(eps_variants: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hit-time and penalized hit-time statistics per penalty value.

    Both SEMs are divided by sqrt(total_count) — the full seed population — so
    that error bars for hit_time and penalized_hit_time are computed on the same
    denominator and remain comparable when some seeds miss the threshold.
    """
    grouped = (
        eps_variants.groupby("penalty_round")
        .agg(
            mean_hit_time=("hit_time_sec", "mean"),
            std_hit_time=("hit_time_sec", "std"),
            hit_count=("hit_time_sec", "count"),
            penalized_mean=("penalized_hit_time_sec", "mean"),
            penalized_std=("penalized_hit_time_sec", "std"),
            total_count=("seed", "size"),
        )
        .reset_index()
        .sort_values("penalty_round")
    )
    grouped["sem_hit_time"] = grouped["std_hit_time"].fillna(0.0) / np.sqrt(grouped["total_count"])
    grouped["sem_penalized"] = grouped["penalized_std"].fillna(0.0) / np.sqrt(grouped["total_count"])
    grouped["miss_count"] = grouped["total_count"] - grouped["hit_count"]
    return grouped


def compute_hit_time_plot_limits(
    hit_df: pd.DataFrame,
    presolve: bool,
    thresholds: Iterable[float],
    penalty_filter: Optional[Iterable[float]] = None,
) -> tuple[float, float]:
    """Compute shared positive y-axis limits for epsilon-hit bar plots."""
    eps = [float(x) for x in thresholds]
    if not eps:
        raise ValueError("thresholds cannot be empty")

    pfilter = _normalize_penalty_filter(penalty_filter)
    sub = hit_df[hit_df["presolve"] == presolve].copy()
    if sub.empty:
        raise ValueError(f"No hit-time data for presolve={presolve}")

    baseline_sub = sub[sub["is_baseline"]].copy()
    variant_sub = sub[~sub["is_baseline"]].copy()
    if pfilter is not None:
        variant_sub = variant_sub[variant_sub["penalty_round"].round(3).isin(pfilter)].copy()

    positive_vals: list[float] = []
    upper_vals: list[float] = []

    for eps_value in eps:
        eps_variants = variant_sub[variant_sub["threshold"] == eps_value].copy()
        if not eps_variants.empty:
            grouped = _aggregate_hit_time_stats(eps_variants)
            for row in grouped.itertuples():
                if row.hit_count > 0 and not np.isnan(row.mean_hit_time):
                    mean = float(row.mean_hit_time)
                    sem = float(row.sem_hit_time) if not np.isnan(row.sem_hit_time) else 0.0
                    positive_vals.append(max(mean * 0.8, 1e-4))
                    upper_vals.append(mean + sem + max(0.007, 0.06 * max(mean + sem, 0.03)))
                if not np.isnan(row.penalized_mean):
                    penalized_mean = float(row.penalized_mean)
                    penalized_sem = float(row.sem_penalized) if not np.isnan(row.sem_penalized) else 0.0
                    positive_vals.append(max(min(penalized_mean, max(penalized_mean * 0.8, 1e-4)), 1e-4))
                    upper_vals.append(
                        penalized_mean + penalized_sem + max(0.007, 0.06 * max(penalized_mean + penalized_sem, 0.03))
                    )

        eps_baseline = baseline_sub[baseline_sub["threshold"] == eps_value]
        bl_mean, bl_sem, bl_n = _mean_sem(eps_baseline["hit_time_sec"])
        if bl_n > 0 and not np.isnan(bl_mean):
            positive_vals.append(max(bl_mean * 0.8, 1e-4))
            upper_vals.append(bl_mean + max(bl_sem, 0.0))

    if not positive_vals or not upper_vals:
        raise ValueError("Could not derive positive y-axis limits for hit-time plots.")

    y_min = max(min(positive_vals), 1e-4)
    y_max = max(upper_vals) * 1.15
    if y_max <= y_min:
        y_max = y_min * 10.0
    return y_min, y_max


def plot_layer_comparison(
    summary_df: pd.DataFrame,
    layers_list: Iterable[int],
    presolve: bool,
    metric: str = "runtime_sec",
    y_log: bool = True,
    common_converged_only: bool = False,
    annotate_counts: bool = False,
    traj_df: Optional[pd.DataFrame] = None,
    eps_threshold: Optional[float] = None,
) -> plt.Figure:
    """Plot quantum scaling across QAOA layer counts and problem sizes."""
    if eps_threshold is None:
        if metric not in {"runtime_sec", "time_to_best_sec"}:
            raise ValueError("metric must be one of: runtime_sec, time_to_best_sec")
        _require_columns(
            summary_df,
            [
                "name",
                "penalty",
                "presolve",
                "seed",
                "layers",
                "n",
                "objective_baseline",
                metric,
            ],
            "layer comparison summary data",
        )
    else:
        _require_columns(
            summary_df,
            ["name", "penalty", "presolve", "seed", "layers", "n", "objective_baseline"],
            "layer comparison summary data",
        )

    sub = summary_df[summary_df["presolve"] == bool(presolve)].copy()
    if sub.empty:
        raise ValueError(f"No summary data for presolve={presolve}")

    layers_sorted = _sort_quantum_depths(layers_list)
    baseline_rows = sub[sub["name"] == "baseline"].copy()
    if baseline_rows.empty:
        raise ValueError("No baseline rows found for layer comparison.")

    # Baseline rows are duplicated across layer-specific files; collapse by (n, seed).
    baseline_opt = (
        baseline_rows[["n", "seed", "presolve", "objective_baseline"]]
        .drop_duplicates(subset=["n", "seed", "presolve"])
        .rename(columns={"objective_baseline": "baseline_opt"})
    )
    total_seed_counts = baseline_opt.groupby("n")["seed"].nunique().to_dict()

    if traj_df is not None and eps_threshold is not None:
        value_col = "eps_hit_time"
        _bl_lookup = {
            (int(r.n), int(r.seed)): float(r.baseline_opt)
            for r in baseline_opt.itertuples()
            if pd.notna(getattr(r, "baseline_opt", None))
        }
        _t = traj_df.copy()
        _t["_bopt"] = [_bl_lookup.get((int(n), int(s))) for n, s in zip(_t["n"], _t["seed"])]
        _t = _t[_t["_bopt"].notna()].copy()
        _t["_thr"] = _t["_bopt"] + _t["_bopt"].abs() * eps_threshold
        _t["_pk"] = _t["penalty"].fillna(-1e9)
        sub["_pk"] = sub["penalty"].fillna(-1e9)
        _gcols = ["n", "name", "seed", "_pk"] + (["layers"] if "layers" in _t.columns else [])
        _eps_rows: list[dict] = []
        for _keys, _grp in _t.groupby(_gcols):
            _thr_val = float(_grp["_thr"].iloc[0])
            _sg = _grp.sort_values("event_idx")
            _hit = _sg.loc[_sg["running_best_orig_obj"] <= _thr_val + 1e-8, "time_sec"]
            _kd = dict(zip(_gcols, _keys if isinstance(_keys, tuple) else (_keys,)))
            _kd["eps_hit_time"] = float(_hit.iloc[0]) if not _hit.empty else float("nan")
            _eps_rows.append(_kd)
        _eps_df = pd.DataFrame(_eps_rows)
        _mcols = [c for c in _gcols if c in sub.columns]
        sub = sub.merge(_eps_df[_mcols + ["eps_hit_time"]], on=_mcols, how="left")
        sub = sub.drop(columns=["_pk"])
        baseline_rows = sub[sub["name"] == "baseline"].copy()
    else:
        value_col = metric

    raw_pre_rows = sub[sub["name"] != "baseline"].copy()
    raw_pre_rows = raw_pre_rows.merge(baseline_opt, on=["n", "seed", "presolve"], how="left")
    raw_pre_rows["reached_baseline_opt"] = (
        raw_pre_rows["baseline_opt"].notna()
        & raw_pre_rows["objective_baseline"].notna()
        & (raw_pre_rows["objective_baseline"] <= raw_pre_rows["baseline_opt"] + 1e-8)
    )
    pre_rows = (
        raw_pre_rows[raw_pre_rows["reached_baseline_opt"]].copy()
        if common_converged_only and eps_threshold is None
        else raw_pre_rows.copy()
    )

    fig, ax = plt.subplots(figsize=(7.1, 5.5))
    if eps_threshold is not None:
        y_label = f"Time to ε = {eps_threshold * 100:g}%-optimal (s)"
    elif metric == "runtime_sec":
        y_label = "Runtime"
    elif metric == "time_to_best_sec":
        y_label = "Time to best solution"
    else:
        y_label = "Runtime"

    series_styles = {
        "baseline": {"color": "#353B55", "marker": "s"},
        1.0: {"color": "#42C6C6", "marker": "o"},
        2.0: {"color": "#F08AA2", "marker": "o"},
        3.0: {"color": "#F2C94C", "marker": "o"},
        float("inf"): {"color": "#6C4AB6", "marker": "^"},
    }

    selected_by_layer: list[pd.DataFrame] = []
    fallback_cmap = plt.get_cmap("tab10")

    for idx, layer in enumerate(layers_sorted):
        layer_rows = pre_rows[pre_rows["layers"] == layer].copy()
        if layer_rows.empty:
            continue

        if common_converged_only and eps_threshold is None:
            seed_best = (
                layer_rows.sort_values(["n", "seed", value_col, "penalty"])
                .groupby(["n", "seed"], as_index=False)
                .first()
            )
            grouped = (
                seed_best.groupby("n")
                .agg(
                    mean=(value_col, "mean"),
                    std=(value_col, "std"),
                    count=("seed", "count"),
                )
                .reset_index()
                .sort_values("n")
            )
            grouped["penalty"] = np.nan
            best = grouped
        else:
            # per-seed oracle: for each (N, seed) pick the ρ with the lowest hit time
            valid = layer_rows[layer_rows[value_col].notna()].copy()
            if valid.empty:
                continue
            seed_best = (
                valid.sort_values([value_col, "penalty"])
                .groupby(["n", "seed"], as_index=False)
                .first()
            )
            best = (
                seed_best.groupby("n")[value_col]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("n")
            )
            best["penalty"] = np.nan

        if best.empty:
            continue

        best["total_count"] = best["n"].map(total_seed_counts).fillna(best["count"]).astype(int)
        selected_by_layer.append(best.assign(layers=layer))

    baseline_seed = (
        baseline_rows.groupby(["n", "seed"], as_index=False)[value_col]
        .mean()
        .sort_values(["n", "seed"])
    )

    baseline_grouped = (
        baseline_seed.groupby("n")[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("n")
    )
    baseline_grouped["total_count"] = baseline_grouped["n"].map(total_seed_counts).fillna(baseline_grouped["count"]).astype(int)
    baseline_grouped["sem"] = baseline_grouped["std"].fillna(0.0) / np.sqrt(baseline_grouped["count"])

    base_order, base_scale = _fit_exponential_order(baseline_grouped["n"], baseline_grouped["mean"])
    baseline_label = (
        f"Original Gurobi - O({base_order:.2f}^N)"
        if np.isfinite(base_order)
        else "Original Gurobi"
    )
    baseline_style = series_styles["baseline"]
    ax.errorbar(
        baseline_grouped["n"],
        baseline_grouped["mean"],
        yerr=baseline_grouped["sem"],
        fmt=baseline_style["marker"],
        linestyle="none",
        markersize=7.0,
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        markeredgewidth=1.1,
        markeredgecolor=baseline_style["color"],
        ecolor=baseline_style["color"],
        color=baseline_style["color"],
        label=baseline_label,
    )
    if common_converged_only and annotate_counts:
        for point in baseline_grouped.itertuples(index=False):
            ax.annotate(
                f"{int(point.count)}/{int(point.total_count)}",
                (float(point.n), float(point.mean)),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#C62828",
            )
    if np.isfinite(base_order) and np.isfinite(base_scale):
        x_fit = np.linspace(float(baseline_grouped["n"].min()), float(baseline_grouped["n"].max()), 300)
        y_fit = base_scale * (base_order ** x_fit)
        ax.plot(x_fit, y_fit, color=baseline_style["color"], linewidth=2.8, alpha=0.8, zorder=1)

    if eps_threshold is None and metric == "time_to_best_sec":
        baseline_runtime_seed = (
            baseline_rows.groupby(["n", "seed"], as_index=False)["runtime_sec"]
            .mean()
            .sort_values(["n", "seed"])
        )
        baseline_runtime_grouped = (
            baseline_runtime_seed.groupby("n")["runtime_sec"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("n")
        )
        baseline_runtime_grouped["sem"] = (
            baseline_runtime_grouped["std"].fillna(0.0) / np.sqrt(baseline_runtime_grouped["count"])
        )
        proof_order, proof_scale = _fit_exponential_order(
            baseline_runtime_grouped["n"],
            baseline_runtime_grouped["mean"],
        )
        proof_label = "Original Gurobi optimality guarantee"
        ax.errorbar(
            baseline_runtime_grouped["n"],
            baseline_runtime_grouped["mean"],
            yerr=baseline_runtime_grouped["sem"],
            fmt=baseline_style["marker"],
            linestyle="none",
            markersize=7.8,
            capsize=5,
            capthick=1.4,
            elinewidth=1.4,
            markeredgewidth=1.2,
            markerfacecolor="none",
            markeredgecolor=baseline_style["color"],
            ecolor=baseline_style["color"],
            color=baseline_style["color"],
            alpha=0.9,
            label=proof_label,
        )

    for idx, layer in enumerate(layers_sorted):
        matching = [df for df in selected_by_layer if float(df["layers"].iloc[0]) == float(layer)]
        if not matching:
            continue
        best = matching[0].sort_values("n")
        best["sem"] = best["std"].fillna(0.0) / np.sqrt(best["count"])
        style = series_styles.get(layer, {"color": fallback_cmap(idx % fallback_cmap.N), "marker": "o"})
        fit_order, fit_scale = _fit_exponential_order(best["n"], best["mean"])
        layer_label = _format_quantum_depth_label(layer)
        show_fit = not (
            common_converged_only
            and metric == "time_to_best_sec"
            and float(layer) in {2.0, 3.0}
        )
        hit_range = ""
        if common_converged_only and "count" in best.columns and "total_count" in best.columns:
            min_hits = int(best["count"].min())
            max_hits = int(best["count"].max())
            total_hits = int(best["total_count"].max())
            hit_range = (
                f", hits {min_hits}/{total_hits}"
                if min_hits == max_hits
                else f", hits {min_hits}-{max_hits}/{total_hits}"
            )
        label = (
            f"p={layer_label} - O({fit_order:.2f}^N){hit_range}"
            if show_fit and np.isfinite(fit_order)
            else f"p={layer_label}{hit_range}"
        )
        ax.errorbar(
            best["n"],
            best["mean"],
            yerr=best["sem"],
            fmt=style["marker"],
            linestyle="none",
            markersize=7.4,
            capsize=5,
            capthick=1.3,
            elinewidth=1.3,
            markeredgewidth=1.1,
            markeredgecolor=style["color"],
            ecolor=style["color"],
            color=style["color"],
            label=label,
        )
        if common_converged_only and annotate_counts:
            for point in best.itertuples(index=False):
                ax.annotate(
                    f"{int(point.count)}/{int(point.total_count)}",
                    (float(point.n), float(point.mean)),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="#C62828",
                )
        if show_fit and np.isfinite(fit_order) and np.isfinite(fit_scale):
            x_fit = np.linspace(float(best["n"].min()), float(best["n"].max()), 300)
            y_fit = fit_scale * (fit_order ** x_fit)
            ax.plot(x_fit, y_fit, color=style["color"], linewidth=2.6, alpha=0.75, zorder=1)

    ax.set_xlabel("Number of variables N")
    ax.set_ylabel(y_label)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.8, alpha=0.45)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_alpha(0.85)
        ax.spines[side].set_linewidth(1.1)

    if y_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=7, width=1.1, labelsize=12)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=3.5, width=0.9)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, framealpha=0.88, edgecolor="#CCCCCC", fontsize=10, loc="upper left", handlelength=2.4, handletextpad=0.6)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Penalty-strategy panel comparison
# ---------------------------------------------------------------------------

_PANEL_STRATEGIES: list[tuple[str, str]] = [
    ("oracle", "Oracle\n(best ρ per instance — upper bound)"),
    ("fixed",  "Fixed global ρ\n(single ρ, mean-optimal across all N & seeds)"),
    ("loo",    "Cross-validated ρ\n(LOO over seeds within each N)"),
]


def _apply_penalty_strategy(
    layer_rows: pd.DataFrame,
    value_col: str,
    strategy: str,
) -> pd.DataFrame:
    """Return one row per (n, seed) chosen by the given penalty strategy.

    strategy="oracle": retrospective best ρ per instance (upper bound).
    strategy="fixed":  single ρ minimising mean metric across all (n, seed).
    strategy="loo":    for each seed, ρ chosen by mean over other seeds at same N.
    """
    all_rows = layer_rows.copy()
    all_rows["_pen_r"] = all_rows["penalty"].round(3)

    valid = layer_rows[layer_rows[value_col].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["_pen_r"] = valid["penalty"].round(3)

    if strategy == "oracle":
        return (
            valid.sort_values([value_col, "_pen_r"])
            .groupby(["n", "seed"], as_index=False)
            .first()
        )

    if strategy == "fixed":
        # Rank ρ by N-normalised hit rate: for each ρ, average the per-N hit fraction
        # across ALL N values in the dataset, treating N values where ρ was not run as
        # 0% hit rate.  This prevents a ρ that was only run at easy small N from winning
        # just because it looks good on those few instances.
        all_n = sorted(all_rows["n"].unique())
        pen_avg_rate: dict[float, float] = {}
        pen_mean_hit: dict[float, float] = {}
        for p in all_rows["_pen_r"].unique():
            rates = []
            for n_val in all_n:
                total_at_n = int((all_rows[(all_rows["n"] == n_val) & (all_rows["_pen_r"] == p)]).shape[0])
                if total_at_n == 0:
                    rates.append(0.0)
                else:
                    hits_at_n = int((valid[(valid["n"] == n_val) & (valid["_pen_r"] == p)]).shape[0])
                    rates.append(hits_at_n / total_at_n)
            pen_avg_rate[p] = float(np.mean(rates))
            pen_valid = valid[valid["_pen_r"] == p]
            pen_mean_hit[p] = float(pen_valid[value_col].mean()) if not pen_valid.empty else float("inf")
        score_df = pd.DataFrame({"avg_rate": pen_avg_rate, "mean_hit": pen_mean_hit})
        score_df = score_df.sort_values(["avg_rate", "mean_hit"], ascending=[False, True])
        best_pen = round(float(score_df.index[0]), 3)
        fixed = valid[valid["_pen_r"] == best_pen].copy()
        return fixed.groupby(["n", "seed"], as_index=False).first()

    if strategy == "loo":
        chosen = []
        for (n_val, seed_val), grp in valid.groupby(["n", "seed"]):
            train = valid[(valid["n"] == n_val) & (valid["seed"] != seed_val)]
            if train.empty:
                best_pen = round(float(grp.sort_values(value_col)["_pen_r"].iloc[0]), 3)
            else:
                best_pen = round(float(train.groupby("_pen_r")[value_col].mean().idxmin()), 3)
            row = grp[grp["_pen_r"] == best_pen]
            if not row.empty:
                chosen.append(row.iloc[0])
        return pd.DataFrame(chosen) if chosen else pd.DataFrame()

    raise ValueError(f"Unknown penalty strategy {strategy!r}. Choose 'oracle', 'fixed', or 'loo'.")


def plot_layer_comparison_panel_strategies(
    summary_df: pd.DataFrame,
    layers_list: Iterable[int],
    presolve: bool,
    traj_df: Optional[pd.DataFrame] = None,
    eps_threshold: Optional[float] = None,
    metric: str = "runtime_sec",
    y_log: bool = True,
    strategies: Optional[list[tuple[str, str]]] = None,
) -> plt.Figure:
    """Three-panel scaling plot comparing oracle, fixed-global, and LOO penalty selection.

    Produces panels side-by-side with a shared y-axis so the performance gap between
    strategies is immediately visible without axis rescaling tricks.
    """
    if strategies is None:
        strategies = _PANEL_STRATEGIES

    sub = summary_df[summary_df["presolve"] == bool(presolve)].copy()
    if sub.empty:
        raise ValueError(f"No summary data for presolve={presolve}")

    layers_sorted = _sort_quantum_depths(layers_list)
    baseline_rows = sub[sub["name"] == "baseline"].copy()
    if baseline_rows.empty:
        raise ValueError("No baseline rows found.")

    baseline_opt = (
        baseline_rows[["n", "seed", "presolve", "objective_baseline"]]
        .drop_duplicates(subset=["n", "seed", "presolve"])
        .rename(columns={"objective_baseline": "baseline_opt"})
    )
    total_seed_counts = baseline_opt.groupby("n")["seed"].nunique().to_dict()

    if traj_df is not None and eps_threshold is not None:
        value_col = "eps_hit_time"
        _bl_lookup = {
            (int(r.n), int(r.seed)): float(r.baseline_opt)
            for r in baseline_opt.itertuples()
            if pd.notna(getattr(r, "baseline_opt", None))
        }
        _t = traj_df.copy()
        _t["_bopt"] = [_bl_lookup.get((int(n), int(s))) for n, s in zip(_t["n"], _t["seed"])]
        _t = _t[_t["_bopt"].notna()].copy()
        _t["_thr"] = _t["_bopt"] + _t["_bopt"].abs() * eps_threshold
        _t["_pk"] = _t["penalty"].fillna(-1e9)
        sub["_pk"] = sub["penalty"].fillna(-1e9)
        _gcols = ["n", "name", "seed", "_pk"] + (["layers"] if "layers" in _t.columns else [])
        _eps_rows: list[dict] = []
        for _keys, _grp in _t.groupby(_gcols):
            _thr_val = float(_grp["_thr"].iloc[0])
            _sg = _grp.sort_values("event_idx")
            _hit = _sg.loc[_sg["running_best_orig_obj"] <= _thr_val + 1e-8, "time_sec"]
            _kd = dict(zip(_gcols, _keys if isinstance(_keys, tuple) else (_keys,)))
            _kd["eps_hit_time"] = float(_hit.iloc[0]) if not _hit.empty else float("nan")
            _eps_rows.append(_kd)
        _eps_df = pd.DataFrame(_eps_rows)
        _mcols = [c for c in _gcols if c in sub.columns]
        sub = sub.merge(_eps_df[_mcols + ["eps_hit_time"]], on=_mcols, how="left")
        sub = sub.drop(columns=["_pk"])
        baseline_rows = sub[sub["name"] == "baseline"].copy()
        y_label = f"Time to ε = {eps_threshold * 100:g}%-optimal (s)"
    else:
        value_col = metric
        y_label = "Runtime (s)" if metric == "runtime_sec" else "Time to best solution (s)"

    raw_pre_rows = sub[sub["name"] != "baseline"].copy()
    raw_pre_rows = raw_pre_rows.merge(baseline_opt, on=["n", "seed", "presolve"], how="left")
    pre_rows = raw_pre_rows.copy()

    baseline_seed = (
        baseline_rows.groupby(["n", "seed"], as_index=False)[value_col]
        .mean()
        .sort_values(["n", "seed"])
    )
    baseline_grouped = (
        baseline_seed.groupby("n")[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("n")
    )
    baseline_grouped["total_count"] = (
        baseline_grouped["n"].map(total_seed_counts).fillna(baseline_grouped["count"]).astype(int)
    )
    baseline_grouped["sem"] = baseline_grouped["std"].fillna(0.0) / np.sqrt(baseline_grouped["count"])

    base_order, base_scale = _fit_exponential_order(baseline_grouped["n"], baseline_grouped["mean"])
    baseline_style = {"color": "#353B55", "marker": "s"}
    series_styles = {
        1.0: {"color": "#42C6C6", "marker": "o"},
        2.0: {"color": "#F08AA2", "marker": "o"},
        3.0: {"color": "#F2C94C", "marker": "o"},
        float("inf"): {"color": "#6C4AB6", "marker": "^"},
    }
    fallback_cmap = plt.get_cmap("tab10")

    n_panels = len(strategies)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(7.1 * n_panels, 5.5),
        sharey=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (strategy, title) in zip(axes, strategies):
        baseline_label = (
            f"Original Gurobi - O({base_order:.2f}^N)" if np.isfinite(base_order) else "Original Gurobi"
        )
        ax.errorbar(
            baseline_grouped["n"],
            baseline_grouped["mean"],
            yerr=baseline_grouped["sem"],
            fmt=baseline_style["marker"],
            linestyle="none",
            markersize=7.0,
            capsize=4,
            capthick=1.2,
            elinewidth=1.2,
            markeredgewidth=1.1,
            markeredgecolor=baseline_style["color"],
            ecolor=baseline_style["color"],
            color=baseline_style["color"],
            label=baseline_label,
        )
        if np.isfinite(base_order) and np.isfinite(base_scale):
            x_fit = np.linspace(float(baseline_grouped["n"].min()), float(baseline_grouped["n"].max()), 300)
            ax.plot(x_fit, base_scale * (base_order ** x_fit),
                    color=baseline_style["color"], linewidth=2.8, alpha=0.8, zorder=1)

        chosen_pen_note: Optional[str] = None

        for idx, layer in enumerate(layers_sorted):
            layer_rows = pre_rows[pre_rows["layers"] == layer].copy()
            if layer_rows.empty:
                continue

            seed_best = _apply_penalty_strategy(layer_rows, value_col, strategy)
            if seed_best.empty:
                continue

            if strategy == "fixed" and "_pen_r" in seed_best.columns:
                unique_pens = seed_best["_pen_r"].dropna().unique()
                if len(unique_pens) == 1 and chosen_pen_note is None:
                    hit_n = int(layer_rows[value_col].notna().sum()) if value_col in layer_rows.columns else 0
                    total_n = len(layer_rows)
                    chosen_pen_note = (
                        f"ρ = {unique_pens[0]:.2g}  (selected by hit rate: {hit_n}/{total_n})"
                    )

            best = (
                seed_best.groupby("n")[value_col]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("n")
            )
            best["sem"] = best["std"].fillna(0.0) / np.sqrt(best["count"])

            style = series_styles.get(float(layer), {"color": fallback_cmap(idx % fallback_cmap.N), "marker": "o"})
            fit_order, fit_scale = _fit_exponential_order(best["n"], best["mean"])
            layer_label = _format_quantum_depth_label(layer)
            label = (
                f"p={layer_label} - O({fit_order:.2f}^N)" if np.isfinite(fit_order) else f"p={layer_label}"
            )
            ax.errorbar(
                best["n"], best["mean"], yerr=best["sem"],
                fmt=style["marker"], linestyle="none", markersize=7.4,
                capsize=5, capthick=1.3, elinewidth=1.3, markeredgewidth=1.1,
                markeredgecolor=style["color"], ecolor=style["color"], color=style["color"],
                label=label,
            )
            if np.isfinite(fit_order) and np.isfinite(fit_scale):
                x_fit = np.linspace(float(best["n"].min()), float(best["n"].max()), 300)
                ax.plot(x_fit, fit_scale * (fit_order ** x_fit),
                        color=style["color"], linewidth=2.6, alpha=0.75, zorder=1)

        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("Number of variables N", fontsize=12)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.8, alpha=0.45)
        for side in ["left", "bottom", "top", "right"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_alpha(0.85)
            ax.spines[side].set_linewidth(1.1)
        ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=7, width=1.1, labelsize=11)
        ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=3.5, width=0.9)
        handles, labels_leg = ax.get_legend_handles_labels()
        ax.legend(handles, labels_leg, frameon=True, framealpha=0.88, edgecolor="#CCCCCC",
                  fontsize=9, loc="upper left", handlelength=2.4, handletextpad=0.6)
        if strategy == "fixed" and chosen_pen_note:
            ax.annotate(chosen_pen_note, xy=(0.98, 0.05), xycoords="axes fraction",
                        fontsize=8.5, ha="right", va="bottom", color="#555555")

    axes[0].set_ylabel(y_label, fontsize=12)
    if y_log:
        for ax in axes:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
            ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    suptitle = (
        f"Quantum preconditioning scaling — penalty selection strategies  (ε = {eps_threshold * 100:g}%)"
        if eps_threshold is not None
        else "Quantum preconditioning scaling — penalty selection strategies"
    )
    fig.suptitle(suptitle, fontsize=13)
    return fig


def plot_family_param_comparison(
    summary_df: pd.DataFrame,
    family: str,
    param_values: Iterable[int],
    presolve: bool,
    metric: str = "runtime_sec",
    y_log: bool = True,
    common_converged_only: bool = False,
    annotate_counts: bool = False,
    traj_df: Optional[pd.DataFrame] = None,
    eps_threshold: Optional[float] = None,
) -> plt.Figure:
    """Dispatch family scaling plots for quantum layers or classical sweeps."""
    if family == "quantum":
        return plot_layer_comparison(
            summary_df=summary_df,
            layers_list=param_values,
            presolve=presolve,
            metric=metric,
            y_log=y_log,
            common_converged_only=common_converged_only,
            annotate_counts=annotate_counts,
            traj_df=traj_df,
            eps_threshold=eps_threshold,
        )
    if family == "classical":
        return plot_sweep_comparison(
            summary_df=summary_df,
            sweeps_list=param_values,
            presolve=presolve,
            metric=metric,
            y_log=y_log,
        )
    raise ValueError("family must be 'quantum' or 'classical'")


def plot_sweep_comparison(
    summary_df: pd.DataFrame,
    sweeps_list: Iterable[int],
    presolve: bool,
    metric: str = "runtime_sec",
    y_log: bool = True,
) -> plt.Figure:
    """Plot classical scaling across sweep counts and problem sizes."""
    if metric not in {"runtime_sec", "time_to_best_sec"}:
        raise ValueError("metric must be one of: runtime_sec, time_to_best_sec")

    _require_columns(
        summary_df,
        [
            "name",
            "penalty",
            "presolve",
            "seed",
            "sweeps",
            "n",
            metric,
        ],
        "sweep comparison summary data",
    )

    sub = summary_df[summary_df["presolve"] == bool(presolve)].copy()
    if "preconditioner_family" in sub.columns:
        sub = sub[sub["preconditioner_family"].fillna("classical") == "classical"].copy()
    if sub.empty:
        raise ValueError(f"No classical summary data for presolve={presolve}")

    sweeps_sorted = sorted({int(v) for v in sweeps_list})
    baseline_rows = sub[sub["name"] == "baseline"].copy()
    value_col = metric

    pre_rows = sub[sub["name"] != "baseline"].copy()
    fig, ax = plt.subplots(figsize=(7.1, 5.5))
    if metric == "runtime_sec":
        y_label = "Runtime"
    elif metric == "time_to_best_sec":
        y_label = "Time to best solution"
    else:
        y_label = "Runtime"

    baseline_style = {"color": "#353B55", "marker": "s"}
    style_map = {
        sweeps_sorted[0] if len(sweeps_sorted) > 0 else 10: {"color": "#2CA58D", "marker": "D"},
        sweeps_sorted[1] if len(sweeps_sorted) > 1 else 100: {"color": "#3FA7D6", "marker": "D"},
        sweeps_sorted[2] if len(sweeps_sorted) > 2 else 1000: {"color": "#7A77B9", "marker": "D"},
        sweeps_sorted[3] if len(sweeps_sorted) > 3 else 10000: {"color": "#A9714B", "marker": "D"},
    }
    fallback_cmap = plt.get_cmap("tab20")
    selected_by_sweep: list[pd.DataFrame] = []
    for sweeps in sweeps_sorted:
        sweep_rows = pre_rows[pre_rows["sweeps"] == int(sweeps)].copy()
        if sweep_rows.empty:
            continue
        penalty_grouped = (
            sweep_rows.groupby(["n", "penalty"])[value_col]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values(["n", "mean", "penalty"])
        )
        if penalty_grouped.empty:
            continue
        selected_by_sweep.append(penalty_grouped.groupby("n", as_index=False).first().assign(sweeps=int(sweeps)))

    baseline_seed = (
        baseline_rows.groupby(["n", "seed"], as_index=False)[value_col]
        .mean()
        .sort_values(["n", "seed"])
    )

    baseline_grouped = (
        baseline_seed.groupby("n")[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("n")
    )
    baseline_grouped["sem"] = baseline_grouped["std"].fillna(0.0) / np.sqrt(baseline_grouped["count"])

    base_order, base_scale = _fit_exponential_order(baseline_grouped["n"], baseline_grouped["mean"])
    baseline_label = (
        f"Original Gurobi - O({base_order:.2f}^N)"
        if np.isfinite(base_order)
        else "Original Gurobi"
    )
    ax.errorbar(
        baseline_grouped["n"],
        baseline_grouped["mean"],
        yerr=baseline_grouped["sem"],
        fmt=baseline_style["marker"],
        linestyle="none",
        markersize=7.0,
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        markeredgewidth=1.1,
        markeredgecolor=baseline_style["color"],
        ecolor=baseline_style["color"],
        color=baseline_style["color"],
        label=baseline_label,
    )
    if np.isfinite(base_order) and np.isfinite(base_scale):
        x_fit = np.linspace(float(baseline_grouped["n"].min()), float(baseline_grouped["n"].max()), 300)
        ax.plot(x_fit, base_scale * (base_order ** x_fit), color=baseline_style["color"], linewidth=2.2, alpha=0.8)

    for idx, sweeps in enumerate(sweeps_sorted):
        matching = [df for df in selected_by_sweep if int(df["sweeps"].iloc[0]) == int(sweeps)]
        if not matching:
            continue
        best = matching[0].sort_values("n")
        best["sem"] = best["std"].fillna(0.0) / np.sqrt(best["count"])
        style = style_map.get(sweeps, {"color": fallback_cmap(idx % fallback_cmap.N), "marker": "D"})
        fit_order, fit_scale = _fit_exponential_order(best["n"], best["mean"])
        label = f"s={sweeps} - O({fit_order:.2f}^N)" if np.isfinite(fit_order) else f"s={sweeps}"

        ax.errorbar(
            best["n"],
            best["mean"],
            yerr=best["sem"],
            fmt=style["marker"],
            linestyle="none",
            markersize=6.0,
            capsize=4,
            capthick=1.1,
            elinewidth=1.1,
            markeredgewidth=1.0,
            markeredgecolor=style["color"],
            ecolor=style["color"],
            color=style["color"],
            label=label,
        )
        if np.isfinite(fit_order) and np.isfinite(fit_scale):
            x_fit = np.linspace(float(best["n"].min()), float(best["n"].max()), 300)
            ax.plot(x_fit, fit_scale * (fit_order ** x_fit), color=style["color"], linewidth=2.0, alpha=0.75)

    ax.set_xlabel("Number of variables N")
    ax.set_ylabel(y_label)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.8, alpha=0.45)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_alpha(0.85)
        ax.spines[side].set_linewidth(1.1)
    if y_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=7, width=1.1, labelsize=12)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=3.5, width=0.9)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, framealpha=0.88, edgecolor="#CCCCCC", fontsize=10, loc="upper left", handlelength=2.4, handletextpad=0.6)
    fig.tight_layout()
    return fig


def plot_quantum_vs_classical_comparison(
    quantum_df: pd.DataFrame,
    classical_df: pd.DataFrame,
    quantum_layers: Iterable[int],
    classical_sweeps: Iterable[int],
    presolve: bool,
    metric: str = "runtime_sec",
    y_log: bool = True,
) -> plt.Figure:
    """Plot a combined scaling comparison for quantum and classical preconditioners."""
    if metric not in {"runtime_sec", "time_to_best_sec"}:
        raise ValueError("metric must be one of: runtime_sec, time_to_best_sec")

    quantum_sub = quantum_df[quantum_df["presolve"] == bool(presolve)].copy()
    classical_sub = classical_df[classical_df["presolve"] == bool(presolve)].copy()
    if quantum_sub.empty and classical_sub.empty:
        raise ValueError("No quantum or classical comparison data available.")

    baseline_rows = pd.concat(
        [
            quantum_sub[quantum_sub["name"] == "baseline"],
            classical_sub[classical_sub["name"] == "baseline"],
        ],
        ignore_index=True,
    )
    baseline_seed = baseline_rows.groupby(["n", "seed"], as_index=False)[metric].mean().sort_values(["n", "seed"])
    baseline_grouped = baseline_seed.groupby("n")[metric].agg(["mean", "std", "count"]).reset_index().sort_values("n")
    baseline_grouped["sem"] = baseline_grouped["std"].fillna(0.0) / np.sqrt(baseline_grouped["count"])

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    y_label = (
        "Runtime"
        if metric == "runtime_sec"
        else "Time to best solution"
    )

    baseline_style = {"color": "#353B55", "marker": "s"}
    base_order, base_scale = _fit_exponential_order(baseline_grouped["n"], baseline_grouped["mean"])
    baseline_label = f"Original Gurobi - O({base_order:.2f}^N)" if np.isfinite(base_order) else "Original Gurobi"
    ax.errorbar(
        baseline_grouped["n"],
        baseline_grouped["mean"],
        yerr=baseline_grouped["sem"],
        fmt=baseline_style["marker"],
        linestyle="none",
        markersize=7.0,
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        markeredgewidth=1.1,
        markeredgecolor=baseline_style["color"],
        ecolor=baseline_style["color"],
        color=baseline_style["color"],
        label=baseline_label,
    )
    if np.isfinite(base_order) and np.isfinite(base_scale):
        x_fit = np.linspace(float(baseline_grouped["n"].min()), float(baseline_grouped["n"].max()), 300)
        ax.plot(x_fit, base_scale * (base_order ** x_fit), color=baseline_style["color"], linewidth=2.2, alpha=0.8)

    quantum_styles = {
        1.0: {"color": "#42C6C6", "marker": "o"},
        2.0: {"color": "#F08AA2", "marker": "o"},
        3.0: {"color": "#F2C94C", "marker": "o"},
        float("inf"): {"color": "#6C4AB6", "marker": "^"},
    }
    classical_palette = ["#2CA58D", "#3FA7D6", "#7A77B9", "#A9714B"]

    for layer in _sort_quantum_depths(quantum_layers):
        layer_rows = quantum_sub[(quantum_sub["name"] != "baseline") & np.isclose(quantum_sub["layers"].astype(float), layer)].copy()
        if layer_rows.empty:
            continue
        penalty_grouped = (
            layer_rows.groupby(["n", "penalty"])[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values(["n", "mean", "penalty"])
        )
        if penalty_grouped.empty:
            continue
        best = penalty_grouped.groupby("n", as_index=False).first().sort_values("n")
        best["sem"] = best["std"].fillna(0.0) / np.sqrt(best["count"])
        style = quantum_styles.get(layer, {"color": "#999999", "marker": "o"})
        fit_order, fit_scale = _fit_exponential_order(best["n"], best["mean"])
        layer_label = _format_quantum_depth_label(layer)
        label = f"p={layer_label} - O({fit_order:.2f}^N)" if np.isfinite(fit_order) else f"p={layer_label}"
        ax.errorbar(
            best["n"], best["mean"], yerr=best["sem"], fmt=style["marker"], linestyle="none",
            markersize=6.0, capsize=4, capthick=1.1, elinewidth=1.1,
            markeredgewidth=1.0, markeredgecolor=style["color"], ecolor=style["color"],
            color=style["color"], label=label,
        )
        if np.isfinite(fit_order) and np.isfinite(fit_scale):
            x_fit = np.linspace(float(best["n"].min()), float(best["n"].max()), 300)
            ax.plot(x_fit, fit_scale * (fit_order ** x_fit), color=style["color"], linewidth=2.0, alpha=0.75)

    for idx, sweeps in enumerate(sorted({int(v) for v in classical_sweeps})):
        sweep_rows = classical_sub[(classical_sub["name"] != "baseline") & (classical_sub["sweeps"] == sweeps)].copy()
        if sweep_rows.empty:
            continue
        penalty_grouped = (
            sweep_rows.groupby(["n", "penalty"])[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values(["n", "mean", "penalty"])
        )
        if penalty_grouped.empty:
            continue
        best = penalty_grouped.groupby("n", as_index=False).first().sort_values("n")
        best["sem"] = best["std"].fillna(0.0) / np.sqrt(best["count"])
        color = classical_palette[idx % len(classical_palette)]
        fit_order, fit_scale = _fit_exponential_order(best["n"], best["mean"])
        label = f"s={sweeps} - O({fit_order:.2f}^N)" if np.isfinite(fit_order) else f"s={sweeps}"
        ax.errorbar(
            best["n"], best["mean"], yerr=best["sem"], fmt="D", linestyle="none",
            markersize=6.0, capsize=4, capthick=1.1, elinewidth=1.1,
            markeredgewidth=1.0, markeredgecolor=color, ecolor=color,
            color=color, label=label,
        )
        if np.isfinite(fit_order) and np.isfinite(fit_scale):
            x_fit = np.linspace(float(best["n"].min()), float(best["n"].max()), 300)
            ax.plot(x_fit, fit_scale * (fit_order ** x_fit), color=color, linewidth=2.0, alpha=0.75)

    ax.set_xlabel("Number of variables N")
    ax.set_ylabel(y_label)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.8, alpha=0.45)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_alpha(0.85)
        ax.spines[side].set_linewidth(1.1)
    if y_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=7, width=1.1, labelsize=12)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=3.5, width=0.9)
    ax.legend(frameon=True, framealpha=0.88, edgecolor="#CCCCCC", fontsize=10, loc="upper left", handlelength=2.4, handletextpad=0.6)
    fig.tight_layout()
    return fig


def _build_per_seed_data(
    df_t: pd.DataFrame,
    penalty_filter: Optional[set[float]],
    value_col: str = "running_best_orig_obj",
) -> Dict[str, Dict[int, tuple[List[float], List[float]]]]:
    """Group trajectory events into per-run, per-seed time series."""
    if value_col not in df_t.columns:
        raise ValueError(f"Column '{value_col}' not found in trajectory data.")

    per_seed: Dict[str, Dict[int, tuple[List[float], List[float]]]] = {}

    for (name, seed), grp in df_t.groupby(["name", "seed"]):
        name = str(name)
        if not _penalty_allowed(name, penalty_filter):
            continue

        grp = grp.sort_values("event_idx")
        times = grp["time_sec"].astype(float).tolist()
        vals = grp[value_col].astype(float).tolist()
        if times:
            per_seed.setdefault(name, {})[int(seed)] = (times, vals)

    return per_seed


def _interpolate_step(times: List[float], vals: List[float], t_grid: np.ndarray) -> np.ndarray:
    """Evaluate a right-continuous step trajectory on a time grid."""
    result = np.full(len(t_grid), np.nan)
    t_arr = np.asarray(times)
    v_arr = np.asarray(vals)

    for i, t in enumerate(t_grid):
        idx = int(np.searchsorted(t_arr, t, side="right")) - 1
        if idx >= 0:
            result[i] = v_arr[idx]

    return result


def plot_mipsol_ratio_minus_one(
    traj_df: pd.DataFrame,
    cfg: PlotConfig,
    presolve: bool,
    mode: str = "aggregate",
    seed: int = 0,
    penalty_filter: Optional[Iterable[float]] = None,
    value_col: str = "running_best_orig_obj",
) -> plt.Figure:
    """Plot MIP incumbent quality trajectories against discovery time."""
    if mode not in {"aggregate", "single"}:
        raise ValueError("mode must be 'aggregate' or 'single'")
    if value_col not in {"orig_obj", "running_best_orig_obj"}:
        raise ValueError("value_col must be one of: orig_obj, running_best_orig_obj")

    pfilter = _normalize_penalty_filter(penalty_filter)
    per_seed = _build_per_seed_data(traj_df, pfilter, value_col=value_col)
    if "baseline" not in per_seed:
        raise RuntimeError("Baseline trajectory missing.")

    presolve_text = "presolve on" if presolve else "presolve off"
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    variant_names = _sorted_variant_names(per_seed.keys(), include_baseline=True)
    _plt_palette = ["#353B55", "#42C6C6", "#F08AA2", "#F2C94C", "#6C4AB6",
                    "#2CA58D", "#3FA7D6", "#7A77B9", "#A9714B", "#E67E22"]
    _color_map = {nm: _plt_palette[i % len(_plt_palette)] for i, nm in enumerate(variant_names)}
    positive_times = [
        float(t)
        for variants in per_seed.values()
        for times, _ in variants.values()
        for t in times
        if float(t) > 0
    ]
    x_floor = max(min(positive_times) * 0.5, 1e-6) if positive_times else 1e-6

    if mode == "aggregate":
        seeds = sorted(per_seed["baseline"].keys())
        baseline_final = {s: vals[-1] for s, (_, vals) in per_seed["baseline"].items() if vals}

        all_end_times = [times[-1] for variants in per_seed.values() for times, _ in variants.values() if times]
        t_max = float(np.percentile(all_end_times, 95)) if all_end_times else 1.0
        t_max = max(t_max, x_floor * 10.0)
        t_grid = np.geomspace(x_floor, t_max, 400)

        for name in variant_names:
            curves = []
            for s in seeds:
                data = per_seed.get(name, {}).get(s)
                base = baseline_final.get(s)
                if data is None or base is None:
                    continue
                times, vals = data
                curves.append(_interpolate_step(times, [v / base - 1.0 for v in vals], t_grid))

            if not curves:
                continue
            arr = np.asarray(curves)
            valid = np.sum(~np.isnan(arr), axis=0) > 0
            mean = np.full(len(t_grid), np.nan)
            q25 = np.full(len(t_grid), np.nan)
            q75 = np.full(len(t_grid), np.nan)
            mean[valid] = np.nanmean(arr[:, valid], axis=0)
            q25[valid] = np.nanpercentile(arr[:, valid], 25, axis=0)
            q75[valid] = np.nanpercentile(arr[:, valid], 75, axis=0)

            is_baseline = name == "baseline"
            label = "baseline (original problem)" if is_baseline else name.replace("precond_", "")
            _clr = _color_map[name]
            ax.plot(
                t_grid[valid],
                mean[valid],
                linewidth=2.8 if is_baseline else 2.3,
                color=_clr,
                label=label,
            )
            if not is_baseline:
                ax.fill_between(t_grid[valid], q25[valid], q75[valid], alpha=0.16, linewidth=0, color=_clr)

        title_extra = f"n_seeds={len(seeds)}"
    else:
        baseline_seed = per_seed["baseline"].get(int(seed))
        if baseline_seed is None:
            raise RuntimeError(f"No baseline trajectory for seed={seed}")

        baseline_final = baseline_seed[1][-1]
        for name in variant_names:
            data = per_seed.get(name, {}).get(int(seed))
            if data is None:
                continue
            times, vals = data
            times_plot = [max(float(t), x_floor) for t in times]
            ratio_gap = [v / baseline_final - 1.0 for v in vals]
            is_baseline = name == "baseline"
            label = "baseline (original problem)" if is_baseline else name.replace("precond_", "")
            _clr = _color_map[name]
            if value_col == "running_best_orig_obj":
                ax.step(
                    times_plot,
                    ratio_gap,
                    where="post",
                    linewidth=2.8 if is_baseline else 2.3,
                    color=_clr,
                    label=label,
                )
            else:
                ax.plot(
                    times_plot,
                    ratio_gap,
                    linewidth=2.0 if is_baseline else 1.8,
                    marker="o",
                    markersize=4.6,
                    color=_clr,
                    label=label,
                )

        title_extra = f"seed={seed}"

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.4, label="Baseline optimum")
    ax.set_xscale("log")
    ax.set_xlabel("Incumbent discovery time (s)", fontsize=14)
    ax.set_ylabel("Approx Ratio - 1", fontsize=14)
    title_setting = f"N={cfg.n}, p={_format_quantum_depth_label(cfg.layers)}" if cfg.family == "quantum" else f"N={cfg.n}, s={cfg.sweeps}"
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_alpha(0.85)
        ax.spines[side].set_linewidth(1.1)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=7, width=1.1, labelsize=12)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=3.5, width=0.9)
    ax.legend(
        fontsize=10,
        ncol=2,
        loc="upper right",
        frameon=True,
        framealpha=0.88,
        edgecolor="#CCCCCC",
        handlelength=2.4,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.tight_layout()
    return fig


def plot_hit_time_vs_penalty(
    hit_df: pd.DataFrame,
    cfg: PlotConfig,
    presolve: bool,
    thresholds: Iterable[float],
    penalty_filter: Optional[Iterable[float]] = None,
    y_log: bool = False,
    y_limits: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Plot PAR and reached-threshold epsilon-hit times by penalty."""
    eps = [float(x) for x in thresholds]
    if not eps:
        raise ValueError("thresholds cannot be empty")

    pfilter = _normalize_penalty_filter(penalty_filter)
    sub = hit_df[hit_df["presolve"] == presolve].copy()
    if sub.empty:
        raise ValueError(f"No hit-time data for presolve={presolve}")

    baseline_sub = sub[sub["is_baseline"]].copy()
    variant_sub = sub[~sub["is_baseline"]].copy()
    if pfilter is not None:
        variant_sub = variant_sub[variant_sub["penalty_round"].round(3).isin(pfilter)].copy()

    _fig_w = max(6.2 * len(eps), 9.5)
    _fs = 1.0  # fixed: match font sizes of scaling / performance-distribution plots
    fig, axes = plt.subplots(1, len(eps), figsize=(_fig_w, 5.5), sharey=True)
    if len(eps) == 1:
        axes = [axes]

    presolve_text = "presolve on" if presolve else "presolve off"
    outer_bar_color = "#9ECFD8"
    inner_bar_color = "#42C6C6"
    miss_color = "#A23B3B"
    n_seeds = int(baseline_sub["seed"].nunique())
    penalty_factor = (
        float(sub["penalty_factor"].dropna().iloc[0])
        if "penalty_factor" in sub.columns and sub["penalty_factor"].notna().any()
        else 10.0
    )

    baseline_handle = plt.Line2D([0], [0], color="#353B55", linewidth=2.0)
    outer_patch = plt.Rectangle((0, 0), 1, 1, facecolor=outer_bar_color, edgecolor="none", alpha=0.95)
    inner_patch = plt.Rectangle((0, 0), 1, 1, facecolor=inner_bar_color, edgecolor="none", alpha=0.98)
    miss_patch = plt.Rectangle((0, 0), 1, 1, facecolor=miss_color, edgecolor="none", alpha=0.85)
    legend_handles = [baseline_handle, outer_patch, inner_patch, miss_patch]
    legend_labels = [
        "Baseline time to reach threshold",
        f"Penalized average runtime ({int(penalty_factor)}x timeout penalty)",
        "Instances that reached threshold",
        "Instances that did not reach threshold",
    ]

    for ax, eps_value in zip(axes, eps):
        eps_variants = variant_sub[variant_sub["threshold"] == eps_value].copy()
        y_max = 0.0
        text_tops: list[float] = []
        if not eps_variants.empty:
            grouped = _aggregate_hit_time_stats(eps_variants)

            x = np.arange(len(grouped))
            heights = grouped["mean_hit_time"].to_numpy()
            sems = grouped["sem_hit_time"].fillna(0.0).to_numpy()
            penalized_heights = grouped["penalized_mean"].to_numpy()
            penalized_sems = grouped["sem_penalized"].fillna(0.0).to_numpy()

            ax.bar(
                x,
                penalized_heights,
                width=0.78,
                color=outer_bar_color,
                alpha=0.95,
                edgecolor="none",
                zorder=2,
            )
            ax.errorbar(
                x,
                penalized_heights,
                yerr=penalized_sems,
                fmt="none",
                capsize=4,
                capthick=1.2,
                elinewidth=1.2,
                color="black",
                zorder=3,
            )
            solved_mask = ~np.isnan(heights)
            if solved_mask.any():
                ax.bar(
                    x[solved_mask],
                    heights[solved_mask],
                    width=0.46,
                    color=inner_bar_color,
                    alpha=0.98,
                    edgecolor="none",
                    zorder=4,
                )
                ax.errorbar(
                    x[solved_mask],
                    heights[solved_mask],
                    yerr=sems[solved_mask],
                    fmt="none",
                    capsize=4,
                    capthick=1.2,
                    elinewidth=1.2,
                    color="black",
                    zorder=5,
                )

            for idx, row in enumerate(grouped.itertuples()):
                if row.hit_count > 0 and not np.isnan(row.mean_hit_time):
                    y_max = max(y_max, float(row.mean_hit_time + row.sem_hit_time))
                if not np.isnan(row.penalized_mean):
                    y_max = max(y_max, float(row.penalized_mean + row.sem_penalized))

            ax.set_xticks(x)
            ax.set_xticklabels([f"ρ={pen:.2f}" for pen in grouped["penalty_round"]], rotation=0, ha="center")

            # red "(N never)" annotation below each tick in axes-fraction y coords
            from matplotlib.transforms import blended_transform_factory
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            for idx, row in enumerate(grouped.itertuples()):
                if row.miss_count > 0:
                    ax.text(
                        idx, -0.055, f"({int(row.miss_count)} never)",
                        transform=trans, ha="center", va="top",
                        color=miss_color, fontsize=round(12 * _fs),
                        clip_on=False,
                    )
        else:
            grouped = pd.DataFrame()

        eps_baseline = baseline_sub[baseline_sub["threshold"] == eps_value]
        bl_mean, bl_sem, bl_n = _mean_sem(eps_baseline["hit_time_sec"])
        if bl_n > 0 and not np.isnan(bl_mean):
            ax.axhline(
                bl_mean,
                color="#353B55",
                linestyle="-",
                linewidth=round(2.0 * _fs, 1),
                label="Baseline time to reach threshold",
                zorder=4,
            )
            if bl_sem > 0:
                ax.axhspan(bl_mean - bl_sem, bl_mean + bl_sem, color="#353B55", alpha=0.12, zorder=1)
            y_max = max(y_max, bl_mean + bl_sem)

        ax.set_title(rf"$\epsilon = {_format_epsilon_label(eps_value)}$", fontsize=round(15 * _fs))
        if ax is axes[0]:
            ax.set_ylabel(r"$\epsilon$-hit time (s)", fontsize=round(14 * _fs))
        for side in ["left", "bottom", "top", "right"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_alpha(0.85)
            ax.spines[side].set_linewidth(round(1.1 * _fs, 1))
        ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=round(7 * _fs), width=round(1.1 * _fs, 1), labelsize=round(12 * _fs))
        ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=round(3.5 * _fs), width=round(0.9 * _fs, 1))
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.8, alpha=0.45)
        if y_log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
            ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            if y_limits is not None:
                ax.set_ylim(*y_limits)
        elif y_max > 0:
            y_top = max([y_max] + text_tops) if text_tops else y_max
            ax.set_ylim(bottom=0.0, top=y_top + max(0.01, 0.05 * y_top))

    fig.tight_layout(w_pad=1.1)
    if legend_handles:
        legend_labels_titled = legend_labels  # presolve goes in legend title
        fig.legend(
            legend_handles,
            legend_labels_titled,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            frameon=True,
            framealpha=0.90,
            edgecolor="#CCCCCC",
            fontsize=round(10 * _fs),
            handlelength=2.4,
            handletextpad=0.6,
            title=presolve_text.capitalize(),
            title_fontsize=round(10 * _fs),
        )
        fig.subplots_adjust(bottom=min(0.20 * _fs, 0.24))
    return fig


def plot_performance_distribution(
    summary_dfs: Dict[str, pd.DataFrame],
    presolve: bool = True,
    reference_lines: Optional[List[float]] = None,
) -> plt.Figure:
    """Fig-1b analog: performance distribution across seeds per N, density-coloured.

    Performance (%) = (baseline_obj / precond_obj) × 100.
    Top strip shows instances at exactly 100% with fraction labels; main panel
    shows suboptimal instances on a log-scale complement axis (gap = 100 − perf%).
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.transforms import blended_transform_factory
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker

    if reference_lines is None:
        reference_lines = [99.0, 99.9]

    # ── per-label density colourmaps (light → saturated) ──────────────────
    _families: Dict[str, tuple] = {
        "p=1":   ("#B3ECEC", "#0D7A7A"),
        "p=2":   ("#FAD0D0", "#B02040"),
        "p=3":   ("#FFF0B0", "#997000"),
        "p=∞":   ("#D5C5F5", "#3E1580"),
        "p=inf": ("#D5C5F5", "#3E1580"),
    }
    _fb_families = [
        ("#C0E8D0", "#1A6040"), ("#C0C8F0", "#203090"),
        ("#F0D8C0", "#804020"), ("#F0D0F0", "#702870"),
    ]

    # ── gather per-seed performance for each (label, N) ───────────────────
    strip_data: Dict[tuple, dict] = {}
    for i, (label, df_raw) in enumerate(summary_dfs.items()):
        df = df_raw[df_raw["presolve"] == presolve].copy()
        if df.empty:
            continue
        bl = (
            df[df["name"] == "baseline"][["n", "seed", "objective_baseline"]]
            .drop_duplicates(subset=["n", "seed"])
            .rename(columns={"objective_baseline": "bl_opt"})
        )
        pre = df[df["name"] != "baseline"].copy()
        if pre.empty:
            continue
        pre = pre.merge(bl, on=["n", "seed"], how="left")
        pre = pre[pre["bl_opt"].notna()].copy()
        pre["alpha"] = pre["objective_baseline"] / pre["bl_opt"]
        pre["perf"] = 100.0 / pre["alpha"]   # 100% = exact match

        ends = _families.get(label, _fb_families[i % len(_fb_families)])
        cmap_i = LinearSegmentedColormap.from_list(f"d_{i}", [ends[0], ends[1]])

        for n_val, grp in pre.groupby("n"):
            if "penalty" in grp.columns and grp["penalty"].notna().any():
                sub = grp.loc[grp.groupby("seed")["alpha"].idxmin()]
            else:
                sub = grp
            perfs = sub["perf"].to_numpy()

            # histogram-based local density in perf space, normalised to [0, 1]
            if len(perfs) > 1 and perfs.max() > perfs.min():
                n_bins = max(4, min(len(perfs) // 4, 14))
                counts, edges = np.histogram(perfs, bins=n_bins)
                idx = np.clip(np.digitize(perfs, edges[:-1]) - 1, 0, len(counts) - 1)
                density = counts[idx].astype(float) / counts.max()
            else:
                density = np.ones(len(perfs))

            strip_data[(label, int(n_val))] = {
                "perfs": perfs, "density": density,
                "cmap": cmap_i, "ends": ends,
            }

    if not strip_data:
        raise ValueError("No data found — check presolve setting and DataFrame contents.")

    labels_ordered = list(summary_dfs.keys())
    Ns_sorted = sorted({n for (_, n) in strip_data})
    n_labels = len(labels_ordered)

    AT_100_THRESH = 99.999  # perf >= this → "solved optimally"

    # ── y-range for main panel (compute before drawing) ───────────────────
    all_sub_gaps = [
        np.clip(100.0 - strip_data[k]["perfs"][strip_data[k]["perfs"] < AT_100_THRESH], 1e-4, None)
        for k in strip_data if np.any(strip_data[k]["perfs"] < AT_100_THRESH)
    ]
    if all_sub_gaps:
        flat = np.concatenate(all_sub_gaps)
        gap_y_hi = flat.max() * 3.0
        gap_y_lo = max(flat.min() * 0.3, 1e-4)
    else:
        gap_y_hi, gap_y_lo = 10.0, 0.001

    # ── figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.1, 5.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.06)
    ax_top = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])
    rng = np.random.default_rng(42)

    cluster_w = 0.82
    strip_w = cluster_w / max(n_labels, 1)
    xlim = (-0.72, len(Ns_sorted) - 0.28)

    # stagger each label to its own y-row in the top strip
    _top_ys = np.linspace(0.72, 1.28, max(n_labels, 1))
    _label_top_y = dict(zip(labels_ordered, _top_ys))

    # ── scatter strips ────────────────────────────────────────────────────
    for ci, n_val in enumerate(Ns_sorted):
        for li, label in enumerate(labels_ordered):
            key = (label, n_val)
            if key not in strip_data:
                continue
            d = strip_data[key]
            perfs, density, cmap_i = d["perfs"], d["density"], d["cmap"]
            strip_cx = ci + (li - (n_labels - 1) / 2.0) * strip_w

            mask_100 = perfs >= AT_100_THRESH
            top_perfs = perfs[mask_100]
            sub_perfs = perfs[~mask_100]

            # top strip: each label at its own y-row, jitter across full cluster width
            if len(top_perfs):
                jitter = rng.uniform(-cluster_w * 0.40, cluster_w * 0.40, size=len(top_perfs))
                ax_top.scatter(
                    ci + jitter, np.full(len(top_perfs), _label_top_y[label]),
                    c=[cmap_i(0.75)] * len(top_perfs),
                    s=20, alpha=0.85, linewidths=0, edgecolors="none", zorder=3,
                )

            # main panel: suboptimal instances on log gap scale
            if len(sub_perfs):
                gap = np.clip(100.0 - sub_perfs, 1e-4, None)
                log_gap = np.log10(gap)
                if len(log_gap) > 1 and log_gap.max() > log_gap.min():
                    n_bins = max(4, min(len(log_gap) // 4, 14))
                    cts, eds = np.histogram(log_gap, bins=n_bins)
                    idx = np.clip(np.digitize(log_gap, eds[:-1]) - 1, 0, len(cts) - 1)
                    sub_dens = cts[idx].astype(float) / cts.max()
                else:
                    sub_dens = np.ones(len(sub_perfs))

                jitter = rng.uniform(-strip_w * 0.30, strip_w * 0.30, size=len(sub_perfs))
                ax_main.scatter(
                    strip_cx + jitter, gap,
                    c=cmap_i(sub_dens),
                    s=32, alpha=0.90, linewidths=0.4, edgecolors="white", zorder=3,
                )

            # mean bar: mean gap over ALL instances (incl. 100% ones)
            mean_gap = float(np.mean(100.0 - perfs))
            if mean_gap > 0:
                ax_main.plot(
                    [strip_cx - strip_w * 0.44, strip_cx + strip_w * 0.44],
                    [mean_gap, mean_gap],
                    color="#1a1a2e", linewidth=3.0, solid_capstyle="butt", zorder=5,
                )

    # ── fraction-solved-optimally: ONE aggregate label per N column ───────
    trans_top = blended_transform_factory(ax_top.transData, ax_top.transAxes)
    for ci, n_val in enumerate(Ns_sorted):
        total = sum(
            len(strip_data[(l, n_val)]["perfs"])
            for l in labels_ordered if (l, n_val) in strip_data
        )
        n_100 = sum(
            int(np.sum(strip_data[(l, n_val)]["perfs"] >= AT_100_THRESH))
            for l in labels_ordered if (l, n_val) in strip_data
        )
        if total > 0:
            frac = 100.0 * n_100 / total
            ax_top.text(
                ci, 1.08, f"{frac:.0f}%",
                transform=trans_top, ha="center", va="bottom",
                fontsize=9, color="#555555", clip_on=False,
            )

    ax_top.text(
        0.5, 1.55, "Fraction solved optimally",
        ha="center", va="bottom", fontsize=9, color="#888888",
        transform=ax_top.transAxes, clip_on=False,
    )

    # ── style top strip ───────────────────────────────────────────────────
    ax_top.set_xlim(*xlim)
    ax_top.set_ylim(0.45, 1.55)
    ax_top.set_yticks([1.0])
    ax_top.set_yticklabels(["100%"], fontsize=12)
    ax_top.set_xticks([])
    ax_top.axhline(1.0, color="#353B55", linestyle="--", linewidth=1.3, alpha=0.60, zorder=2)
    ax_top.spines["bottom"].set_visible(False)
    for side in ["top", "left", "right"]:
        ax_top.spines[side].set_linewidth(1.1)
    ax_top.tick_params(axis="y", direction="in", left=True, right=True,
                       length=7, width=1.1, labelsize=12)
    ax_top.tick_params(axis="x", bottom=False, top=False)

    # ── style main panel ──────────────────────────────────────────────────
    ax_main.set_yscale("log")
    ax_main.set_ylim(gap_y_hi, gap_y_lo)  # large gap (bad) at bottom, small (good) at top

    # y-ticks: pick at most 4 clean candidates within data range
    cand_gaps   = [10,    1,    0.1,    0.01,    0.001]
    cand_labels = ["90%", "99%", "99.9%", "99.99%", "99.999%"]
    valid = [(g, l) for g, l in zip(cand_gaps, cand_labels) if gap_y_lo <= g <= gap_y_hi]
    valid = valid[:4]  # cap at 4 ticks
    if valid:
        vg, vl = zip(*valid)
        ax_main.set_yticks(list(vg))
        ax_main.set_yticklabels(list(vl), fontsize=12)
    ax_main.yaxis.set_minor_locator(mticker.LogLocator(subs="all", numticks=10))

    # reference lines (converted from perf% to gap)
    for ref in reference_lines:
        ref_gap = 100.0 - ref
        if gap_y_lo <= ref_gap <= gap_y_hi:
            ax_main.axhline(ref_gap, color="#888888", linestyle=":",
                            linewidth=1.1, alpha=0.75, zorder=1)

    ax_main.set_xticks(range(len(Ns_sorted)))
    ax_main.set_xticklabels([str(n) for n in Ns_sorted], fontsize=12)
    ax_main.set_xlim(*xlim)
    ax_main.set_xlabel("Number of variables N", fontsize=14)
    ax_main.set_ylabel("")
    ax_main.grid(axis="y", which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax_main.grid(axis="x", visible=False)
    for side in ["top", "right"]:
        ax_main.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax_main.spines[side].set_alpha(0.85)
        ax_main.spines[side].set_linewidth(1.1)
    ax_main.tick_params(axis="both", which="major", direction="in",
                        top=False, right=True, length=7, width=1.1, labelsize=12)
    ax_main.tick_params(axis="both", which="minor", direction="in",
                        top=False, right=True, length=3.5, width=0.9)

    # ── broken-axis diagonal marks (left edge only, cleaner) ──────────────
    d_b = 0.008
    for ax_, y_pos in [(ax_top, 0.0), (ax_main, 1.0)]:
        ax_.plot(
            [-d_b, +d_b], [y_pos - 2 * d_b, y_pos + 2 * d_b],
            transform=ax_.transAxes, color="k", clip_on=False, linewidth=1.1,
        )

    # ── combined gradient legend below x-axis ────────────────────────────
    from matplotlib.legend_handler import HandlerBase

    class _GradHandler(HandlerBase):
        """Legend handler that draws a horizontal gradient rectangle."""
        def __init__(self, cmap, n_seg=32):
            self._cmap = cmap
            self._n = n_seg
            super().__init__()

        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            patches = []
            seg_w = width / self._n
            for j in range(self._n):
                patches.append(plt.Rectangle(
                    (xdescent + j * seg_w, ydescent), seg_w + 0.5, height,
                    facecolor=self._cmap(j / max(self._n - 1, 1)),
                    edgecolor="none", transform=trans,
                ))
            patches.append(plt.Rectangle(
                (xdescent, ydescent), width, height,
                facecolor="none", edgecolor="#aaaaaa",
                linewidth=0.5, transform=trans,
            ))
            return patches

    leg_handles = []
    handler_map = {}
    for i, label in enumerate(labels_ordered):
        ends = _families.get(label, _fb_families[i % len(_fb_families)])
        cmap_h = LinearSegmentedColormap.from_list(f"_lh{i}", [ends[0], ends[1]])
        h = mpatches.Patch(label=label)
        leg_handles.append(h)
        handler_map[h] = _GradHandler(cmap_h)

    mean_h = plt.Line2D([0], [0], color="#1a1a2e", linewidth=3.0, label="Mean")
    leg_handles.append(mean_h)

    presolve_label = "Presolve on" if presolve else "Presolve off"
    ax_main.legend(
        handles=leg_handles,
        handler_map=handler_map,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=len(leg_handles),
        fontsize=11,
        title=f"{presolve_label}   ·   Color: instance density (Low → High)",
        title_fontsize=11,
        frameon=True, framealpha=0.92, edgecolor="#CCCCCC",
        handlelength=3.5, handletextpad=0.5, columnspacing=1.2,
    )

    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.30, top=0.86, hspace=0.08)
    # y-label centred across both panels, clear of the widest tick label
    fig.text(0.045, (0.30 + 0.86) / 2, "Performance distribution (%)",
             va="center", ha="center", rotation="vertical", fontsize=14)
    return fig
