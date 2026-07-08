"""Shared pytest fixtures for the Gurobi_QP test suite."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from qp_gurobi.instance import IsingInstance


# ---------------------------------------------------------------------------
# IsingInstance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_instance() -> IsingInstance:
    """4-node instance with one quadratic coupling: w_{01}=2.0."""
    return IsingInstance(
        n=4,
        constant=1.0,
        linear={},
        quadratic={(0, 1): 2.0},
    )


@pytest.fixture
def write_dat(tmp_path):
    """Factory: write a .dat file and return its Path."""
    def _write(lines: list[str]) -> Path:
        p = tmp_path / "instance.dat"
        p.write_text("\n".join(lines))
        return p
    return _write


# ---------------------------------------------------------------------------
# DataFrame fixtures for utils analysis tests
# ---------------------------------------------------------------------------

def _make_trajectory(events: list[tuple[float, float]]) -> str:
    return json.dumps([[round(t, 6), round(v, 8)] for t, v in events])


def _summary_row(
    *,
    name: str,
    n: int,
    seed: int,
    presolve: bool,
    objective_baseline: float,
    penalty: float | None = None,
    runtime_sec: float = 1.0,
    time_to_best_sec: float = 0.5,
    trajectory: list[tuple[float, float]] | None = None,
) -> dict:
    return {
        "name": name,
        "n": n,
        "seed": seed,
        "presolve": presolve,
        "objective_baseline": objective_baseline,
        "penalty": penalty,
        "runtime_sec": runtime_sec,
        "time_to_best_sec": time_to_best_sec,
        "trajectory": _make_trajectory(trajectory or []),
        "layers": 1,
        "preconditioner_family": "quantum",
    }


@pytest.fixture
def single_n_summary_df() -> pd.DataFrame:
    """Summary CSV rows for N=8, seeds 0 and 1, two penalty variants."""
    rows = [
        _summary_row(name="baseline", n=8, seed=0, presolve=True, objective_baseline=-10.0),
        _summary_row(name="baseline", n=8, seed=1, presolve=True, objective_baseline=-12.0),
        _summary_row(name="precond_pen=0.300", n=8, seed=0, presolve=True,
                     objective_baseline=-9.5, penalty=0.3),
        _summary_row(name="precond_pen=0.300", n=8, seed=1, presolve=True,
                     objective_baseline=-11.0, penalty=0.3),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def multi_n_summary_df() -> pd.DataFrame:
    """Summary rows spanning N=8 and N=16 — same seed numbers, different N."""
    rows = [
        # N=8
        _summary_row(name="baseline", n=8, seed=0, presolve=True, objective_baseline=-10.0),
        _summary_row(name="precond_pen=0.300", n=8, seed=0, presolve=True,
                     objective_baseline=-9.5, penalty=0.3),
        # N=16
        _summary_row(name="baseline", n=16, seed=0, presolve=True, objective_baseline=-50.0),
        _summary_row(name="precond_pen=0.300", n=16, seed=0, presolve=True,
                     objective_baseline=-48.0, penalty=0.3),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def trajectory_df_two_seeds() -> pd.DataFrame:
    """Long-format trajectory data: 2 seeds, baseline + one preconditioned run.

    seed=0: hits threshold at t=0.2
    seed=1: never hits threshold (stays at -9.0 which is above -9.9 for eps=0.01)
    """
    rows = []
    # threshold formula: base_opt + abs(base_opt)*eps = -10.0 + 1.0*0.01 = -9.9
    # seed 0 reaches -10.2 at t=0.2 (-10.2 ≤ -9.9) → hits
    # seed 1 best is -9.5 (-9.5 > -9.9) → misses
    for seed, events in [
        (0, [(0.1, -9.0), (0.2, -10.2)]),
        (1, [(0.1, -9.0), (0.5, -9.5)]),
    ]:
        best = float("inf")
        for idx, (t, v) in enumerate(events):
            best = min(best, v)
            rows.append({
                "name": "precond_pen=0.300",
                "seed": seed,
                "penalty": 0.3,
                "presolve": True,
                "event_idx": idx,
                "time_sec": t,
                "orig_obj": v,
                "running_best_orig_obj": best,
                "layers": 1,
                "preconditioner_family": "quantum",
            })
    return pd.DataFrame(rows)
