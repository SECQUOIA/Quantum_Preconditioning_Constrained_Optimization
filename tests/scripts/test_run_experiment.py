"""Unit tests for the experiment runner CLI helpers."""
from __future__ import annotations

import sys

import pytest

from scripts import run_experiment


def test__seed_tag__given_long_equal_size_lists__uses_hash_to_avoid_collisions():
    # ARRANGE
    seeds_a = list(range(25))
    seeds_b = list(range(25, 50))

    # ACT
    tag_a = run_experiment._seed_tag(seeds_a)
    tag_b = run_experiment._seed_tag(seeds_b)

    # ASSERT
    assert tag_a.startswith("listN25_")
    assert tag_b.startswith("listN25_")
    assert tag_a != tag_b


def test__main__given_conflicting_seed_selectors__exits_before_running(monkeypatch):
    # ARRANGE
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--n",
            "8",
            "--layers",
            "1",
            "--seed",
            "3",
            "--seeds",
            "0,1",
        ],
    )

    # ACT / ASSERT
    with pytest.raises(SystemExit, match="Use only one of --seed, --seeds, or --all-seeds"):
        run_experiment.main()
