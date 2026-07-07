"""Unit tests for the results verifier CLI."""
from __future__ import annotations

import sys

import pandas as pd
import pytest

from scripts import verify_results


def _write_constant_instance(path, constant: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"4 1\n{constant}\n")


def test__verify_results__given_multiseed_csv__filters_to_requested_seed(tmp_path, monkeypatch):
    # ARRANGE
    data_root = tmp_path / "data"
    _write_constant_instance(data_root / "N=4" / "seed=0" / "problem.dat", -10.0)
    _write_constant_instance(data_root / "N=4" / "seed=1" / "problem.dat", -20.0)

    csv_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {"name": "baseline", "n": 4, "seed": 0, "objective_model": -10.0, "objective_baseline": -10.0, "x_bits": "0011", "sum_z": 0},
            {"name": "precond", "n": 4, "seed": 0, "objective_model": -9.0, "objective_baseline": -10.0, "x_bits": "0011", "sum_z": 0},
            {"name": "baseline", "n": 4, "seed": 1, "objective_model": -20.0, "objective_baseline": -20.0, "x_bits": "0011", "sum_z": 0},
            {"name": "precond", "n": 4, "seed": 1, "objective_model": -19.0, "objective_baseline": -20.0, "x_bits": "0011", "sum_z": 0},
        ]
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_results.py",
            "--csv",
            str(csv_path),
            "--data-root",
            str(data_root),
            "--n",
            "4",
            "--seed",
            "0",
        ],
    )

    # ACT / ASSERT
    assert verify_results.main() == 0


def test__verify_results__given_missing_seed_column__fails_clearly(tmp_path, monkeypatch):
    # ARRANGE
    data_root = tmp_path / "data"
    _write_constant_instance(data_root / "N=4" / "seed=0" / "problem.dat", -10.0)

    csv_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {"name": "baseline", "n": 4, "objective_model": -10.0, "objective_baseline": -10.0, "x_bits": "0011", "sum_z": 0},
        ]
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_results.py",
            "--csv",
            str(csv_path),
            "--data-root",
            str(data_root),
            "--n",
            "4",
            "--seed",
            "0",
        ],
    )

    # ACT / ASSERT
    with pytest.raises(SystemExit, match="CSV missing required columns: \\['seed'\\]"):
        verify_results.main()
