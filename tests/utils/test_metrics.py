"""Unit tests for analysis metrics in qp_gurobi/utils.py.

Covers the four confirmed bugs from BUGS_AND_OPTIMIZATIONS.md:
  Bug #1 — time_to_best_sec  (tested in tests/solve/)
  Bug #2 — epsilon threshold formula  (documented here; see note in class)
  Bug #3 — cross-join on missing 'n' key in prepare_penalty_metrics
  Bug #4 — SEM denominator: hit_count vs total_count

No Gurobi dependency; all tests are pure pandas/numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qp_gurobi.utils import (
    PlotConfig,
    _aggregate_hit_time_stats,
    compute_fixed_rho_by_depth,
    plot_hit_time_vs_penalty,
    plot_performance_distribution,
    prepare_hit_time_metrics,
    prepare_penalty_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hit_df_fixture(
    *,
    n_seeds: int,
    n_hitting: int,
    base_opt: float,
    eps: float,
    hit_times: list[float],
) -> tuple[pd.DataFrame, dict[bool, pd.DataFrame], PlotConfig]:
    """Build minimal summary_df and traj_by_presolve for prepare_hit_time_metrics."""
    assert len(hit_times) == n_hitting

    summary_rows = []
    traj_rows = []

    for seed in range(n_seeds):
        summary_rows.append({
            "name": "baseline",
            "n": 8,
            "seed": seed,
            "presolve": True,
            "objective_baseline": base_opt,
            "penalty": None,
            "runtime_sec": 5.0,
            "time_to_best_sec": 1.0,
            "layers": 1,
            "preconditioner_family": "quantum",
        })
        summary_rows.append({
            "name": "precond_pen=0.300",
            "n": 8,
            "seed": seed,
            "presolve": True,
            "objective_baseline": base_opt - 0.5 if seed < n_hitting else base_opt * 1.1,
            "penalty": 0.3,
            "runtime_sec": 4.0,
            "time_to_best_sec": 0.5,
            "layers": 1,
            "preconditioner_family": "quantum",
        })
        # Build trajectory: seed 0..n_hitting-1 hit; remaining seeds miss
        if seed < n_hitting:
            t_hit = hit_times[seed]
            events = [
                (0.01, float(base_opt) * 1.5),   # initial bad solution
                (t_hit, float(base_opt) - 0.5),   # solution better than threshold
            ]
        else:
            events = [(0.01, float(base_opt) * 1.5)]  # never hits

        best = float("inf")
        for idx, (t, v) in enumerate(events):
            best = min(best, v)
            traj_rows.append({
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

    summary_df = pd.DataFrame(summary_rows)
    traj_df = pd.DataFrame(traj_rows)
    cfg = PlotConfig(n=8, layers=1, family="quantum")
    return summary_df, {True: traj_df}, cfg


def _make_disagreeing_penalty_df() -> pd.DataFrame:
    """Build two seeds whose individually best penalties disagree."""
    return pd.DataFrame(
        [
            {"name": "baseline", "n": 8, "seed": 0, "presolve": True, "objective_baseline": -10.0, "penalty": None},
            {"name": "baseline", "n": 8, "seed": 1, "presolve": True, "objective_baseline": -10.0, "penalty": None},
            {"name": "precond_pen=0.100", "n": 8, "seed": 0, "presolve": True, "objective_baseline": -9.9, "penalty": 0.1},
            {"name": "precond_pen=0.200", "n": 8, "seed": 0, "presolve": True, "objective_baseline": -9.8, "penalty": 0.2},
            {"name": "precond_pen=0.300", "n": 8, "seed": 0, "presolve": True, "objective_baseline": -9.0, "penalty": 0.3},
            {"name": "precond_pen=0.100", "n": 8, "seed": 1, "presolve": True, "objective_baseline": -9.1, "penalty": 0.1},
            {"name": "precond_pen=0.200", "n": 8, "seed": 1, "presolve": True, "objective_baseline": -9.75, "penalty": 0.2},
            {"name": "precond_pen=0.300", "n": 8, "seed": 1, "presolve": True, "objective_baseline": -9.9, "penalty": 0.3},
        ]
    )


# ---------------------------------------------------------------------------
# Bug #3 — prepare_penalty_metrics cross-join (CONFIRMED BUG)
# ---------------------------------------------------------------------------

class TestPreparePenaltyMetrics:
    def test__prepare_penalty_metrics__given_single_n__approx_ratio_correct(
        self, single_n_summary_df
    ):
        # ARRANGE / ACT
        result = prepare_penalty_metrics(single_n_summary_df)

        # ASSERT — seed 0: 9.5 / 10.0 - 1 = -0.05; seed 1: 11.0 / 12.0 - 1 ≈ -0.0833
        seed0 = result[result["seed"] == 0]
        assert seed0["approx_ratio_minus_1"].iloc[0] == pytest.approx(-0.05)

    def test__prepare_penalty_metrics__given_single_n__no_extra_rows(
        self, single_n_summary_df
    ):
        # ARRANGE / ACT
        result = prepare_penalty_metrics(single_n_summary_df)

        # ASSERT — 2 seeds × 1 penalty = 2 preconditioned rows (no Cartesian product)
        assert len(result) == 2

    def test__prepare_penalty_metrics__given_multiple_n__no_cross_join_across_problem_sizes(
        self, multi_n_summary_df
    ):
        """BUG #3: without 'n' in merge keys, baseline rows for N=8 attach to
        preconditioned rows for N=16 (same seed, different problem size).

        This test FAILS before the fix (merge on ['seed', 'presolve'] only) and
        PASSES after adding 'n' to the merge keys.
        """
        # ARRANGE / ACT
        result = prepare_penalty_metrics(multi_n_summary_df)

        # ASSERT — each preconditioned row must have the baseline from its OWN N
        n8_rows = result[result["n"] == 8]
        n16_rows = result[result["n"] == 16]

        # N=8 baseline_obj must be -10.0, not -50.0 (the N=16 baseline)
        assert np.allclose(n8_rows["baseline_obj"].to_numpy(), -10.0), (
            "N=8 preconditioned rows got the wrong baseline — cross-join bug"
        )
        # N=16 baseline_obj must be -50.0, not -10.0
        assert np.allclose(n16_rows["baseline_obj"].to_numpy(), -50.0), (
            "N=16 preconditioned rows got the wrong baseline — cross-join bug"
        )

    def test__prepare_penalty_metrics__given_multiple_n__result_has_one_row_per_preconditioned_seed(
        self, multi_n_summary_df
    ):
        # ARRANGE / ACT
        result = prepare_penalty_metrics(multi_n_summary_df)

        # ASSERT — 1 N=8 precond row + 1 N=16 precond row = 2 total (no Cartesian product)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# prepare_hit_time_metrics — coverage tests
# ---------------------------------------------------------------------------

class TestPrepareHitTimeMetrics:
    def test__prepare_hit_time_metrics__given_trajectory_never_crosses__hit_time_is_nan(self):
        """Seeds whose trajectory never reaches the threshold produce NaN hit_time_sec."""
        # ARRANGE — base_opt=10.0, eps=0.01 → threshold = 10.0 + 0.1 = 10.1
        # Initial (only) event is base_opt*1.5 = 15.0 > 10.1, so no hit.
        summary_df, traj_by_presolve, cfg = _make_hit_df_fixture(
            n_seeds=1, n_hitting=0, base_opt=10.0, eps=0.01, hit_times=[]
        )

        # ACT
        hit_df = prepare_hit_time_metrics(
            summary_df, traj_by_presolve, cfg, thresholds=[0.01]
        )

        # ASSERT
        variant_rows = hit_df[~hit_df["is_baseline"]]
        assert not variant_rows["hit_found"].any(), "No seed should have hit the threshold"
        assert variant_rows["hit_time_sec"].isna().all(), "hit_time_sec must be NaN for all misses"

    def test__prepare_hit_time_metrics__given_early_crossing__hit_time_matches_first_event(self):
        """The recorded hit_time_sec must be the timestamp of the first threshold crossing."""
        # ARRANGE — seed hits at t=0.2 (second event; first is too high)
        summary_df, traj_by_presolve, cfg = _make_hit_df_fixture(
            n_seeds=1, n_hitting=1, base_opt=10.0, eps=0.01, hit_times=[0.2]
        )

        # ACT
        hit_df = prepare_hit_time_metrics(
            summary_df, traj_by_presolve, cfg, thresholds=[0.01]
        )

        # ASSERT
        variant_rows = hit_df[~hit_df["is_baseline"]]
        assert variant_rows["hit_found"].all()
        # hit_time must be ≤ the runtime (0.2 is the crossing event time)
        assert variant_rows["hit_time_sec"].iloc[0] == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# plot_performance_distribution
# ---------------------------------------------------------------------------

class TestPlotPerformanceDistribution:
    def test__plot_performance_distribution__given_negative_objective__worse_solution_has_positive_gap(self):
        # ARRANGE
        df = pd.DataFrame(
            [
                {"name": "baseline", "n": 8, "seed": 0, "presolve": True, "objective_baseline": -10.0, "penalty": None},
                {"name": "precond_pen=0.100", "n": 8, "seed": 0, "presolve": True, "objective_baseline": -9.5, "penalty": 0.1},
            ]
        )

        # ACT
        fig = plot_performance_distribution({"p=1": df}, presolve=True, reference_lines=[])

        # ASSERT
        offsets = [
            collection.get_offsets()
            for collection in fig.axes[1].collections
            if len(collection.get_offsets())
        ]
        assert len(offsets) == 1
        assert offsets[0][0, 1] == pytest.approx(5.0)

    def test__plot_performance_distribution__given_positive_objective__worse_solution_has_positive_gap(self):
        # ARRANGE
        df = pd.DataFrame(
            [
                {"name": "baseline", "n": 8, "seed": 0, "presolve": True, "objective_baseline": 10.0, "penalty": None},
                {"name": "precond_pen=0.100", "n": 8, "seed": 0, "presolve": True, "objective_baseline": 10.5, "penalty": 0.1},
            ]
        )

        # ACT
        fig = plot_performance_distribution({"p=1": df}, presolve=True, reference_lines=[])

        # ASSERT
        offsets = [
            collection.get_offsets()
            for collection in fig.axes[1].collections
            if len(collection.get_offsets())
        ]
        assert len(offsets) == 1
        assert offsets[0][0, 1] == pytest.approx(5.0)

    def test__plot_performance_distribution__given_multiple_penalties__plots_best_ratio_for_seed(self):
        # ARRANGE
        df = pd.DataFrame(
            [
                {"name": "baseline", "n": 8, "seed": 0, "presolve": True, "objective_baseline": -10.0, "penalty": None},
                {"name": "precond_pen=0.100", "n": 8, "seed": 0, "presolve": True, "objective_baseline": -9.5, "penalty": 0.1},
                {"name": "precond_pen=0.200", "n": 8, "seed": 0, "presolve": True, "objective_baseline": -9.8, "penalty": 0.2},
            ]
        )

        # ACT
        fig = plot_performance_distribution({"p=1": df}, presolve=True, reference_lines=[])

        # ASSERT
        offsets = [
            collection.get_offsets()
            for collection in fig.axes[1].collections
            if len(collection.get_offsets())
        ]
        assert len(offsets) == 1
        assert offsets[0][0, 1] == pytest.approx(2.0)

    def test__plot_performance_distribution__given_multiple_seeds__uses_fixed_penalty_not_per_seed_oracle(self):
        # A single seed can't distinguish "oracle per instance" from "fixed
        # per depth" (both degenerate to the same pick). With two seeds whose
        # individual best penalties disagree, the fixed strategy must hold one
        # penalty fixed across both -- picking the penalty with the best
        # *average* gap (0.2, avg gap 2.25) over the oracle-per-seed pick
        # (which would use 0.1 for seed 0 and 0.3 for seed 1, both gap 1.0).
        df = _make_disagreeing_penalty_df()

        # ACT
        fig = plot_performance_distribution({"p=1": df}, presolve=True, reference_lines=[])

        # ASSERT
        offsets = [
            collection.get_offsets()
            for collection in fig.axes[1].collections
            if len(collection.get_offsets())
        ]
        assert len(offsets) == 1
        gaps = sorted(offsets[0][:, 1])
        assert gaps == pytest.approx([2.0, 2.5])

    def test__plot_performance_distribution__given_oracle_strategy__uses_best_penalty_per_seed(self):
        # ARRANGE
        df = _make_disagreeing_penalty_df()

        # ACT
        fig = plot_performance_distribution(
            {"p=1": df}, presolve=True, reference_lines=[], strategy="oracle"
        )

        # ASSERT
        offsets = [
            collection.get_offsets()
            for collection in fig.axes[1].collections
            if len(collection.get_offsets())
        ]
        assert len(offsets) == 1
        gaps = sorted(offsets[0][:, 1])
        assert gaps == pytest.approx([1.0, 1.0])

    def test__plot_performance_distribution__given_fixed_rho_by_label__overrides_internal_selection(self):
        # The internal "fixed" strategy selects rho against quality gap and
        # would pick 0.2 here (see the sibling test above). A caller wanting
        # this plot to report quality under the scaling grid's own Global-rho
        # choice (selected against hit-time, a different quantity) needs an
        # explicit override, since the two selections are not guaranteed to
        # agree and silently falling back to the internal pick would
        # reintroduce the exact cross-figure mismatch this parameter fixes.
        df = _make_disagreeing_penalty_df()

        # ACT: force rho=0.3, which the quality-based "fixed" strategy would
        # not have chosen on its own.
        fig = plot_performance_distribution(
            {"p=1": df}, presolve=True, reference_lines=[],
            fixed_rho_by_label={"p=1": 0.3},
        )

        # ASSERT
        offsets = [
            collection.get_offsets()
            for collection in fig.axes[1].collections
            if len(collection.get_offsets())
        ]
        assert len(offsets) == 1
        gaps = sorted(offsets[0][:, 1])
        assert gaps == pytest.approx([1.0, 10.0])

    def test__plot_performance_distribution__given_fixed_rho_by_label_missing_a_label__raises(self):
        df = _make_disagreeing_penalty_df()

        with pytest.raises(KeyError):
            plot_performance_distribution(
                {"p=1": df}, presolve=True, reference_lines=[],
                fixed_rho_by_label={"p=2": 0.3},
            )


# ---------------------------------------------------------------------------
# compute_fixed_rho_by_depth
# ---------------------------------------------------------------------------

class TestComputeFixedRhoByDepth:
    def test__compute_fixed_rho_by_depth__matches_grid_selected_rho(self):
        # ARRANGE: the hit-time fixture already carries a "layers" column,
        # so it doubles as a minimal stand-in for the scaling grid's own
        # summary_df/traj_df.
        summary_df, traj_by_presolve, _cfg = _make_hit_df_fixture(
            n_seeds=2, n_hitting=2, base_opt=10.0, eps=0.01, hit_times=[0.2, 0.5],
        )
        # The fixture's trajectory rows omit "n" (fine for prepare_hit_time_metrics,
        # which is always called for one fixed size), but _compute_eps_hit_col
        # groups across "n" too, so add it here to match summary_df.
        traj_df = traj_by_presolve[True].copy()
        traj_df["n"] = 8

        # ACT
        result = compute_fixed_rho_by_depth(
            summary_df=summary_df,
            traj_df=traj_df,
            layers_list=[1],
            presolve=True,
            eps_threshold=0.01,
        )

        # ASSERT: a single tested penalty degenerates to itself being picked.
        assert result == pytest.approx({1.0: 0.3})


# ---------------------------------------------------------------------------
# plot_hit_time_vs_penalty
# ---------------------------------------------------------------------------

def test__plot_hit_time_vs_penalty__given_missing_penalty_factor_column__uses_prepare_default():
    # ARRANGE
    summary_df, traj_by_presolve, cfg = _make_hit_df_fixture(
        n_seeds=1, n_hitting=1, base_opt=10.0, eps=0.01, hit_times=[0.2]
    )
    hit_df = prepare_hit_time_metrics(
        summary_df, traj_by_presolve, cfg, thresholds=[0.01]
    ).drop(columns=["penalty_factor"])

    # ACT
    fig = plot_hit_time_vs_penalty(hit_df, cfg, presolve=True, thresholds=[0.01])

    # ASSERT
    legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert "Penalized average runtime (1x timeout penalty)" in legend_labels


# ---------------------------------------------------------------------------
# Bug #2 — epsilon threshold formula (documentation test)
#
# NOTE: After careful analysis, the formula `base_opt + abs(base_opt) * eps`
# is CORRECT for "within eps% worse than baseline" on a minimisation problem,
# for BOTH positive and negative objectives.  The semantics:
#
#   threshold = base_opt + abs(base_opt) * eps
#
# means: accept solutions with objective ≤ threshold, i.e. solutions that are
# at most eps% worse than what the baseline achieved.  This is symmetric:
#
#   positive base_opt=100, eps=0.01 → threshold=101  (≤101 counts as hitting)
#   negative base_opt=-100, eps=0.01 → threshold=-99  (≤-99 counts as hitting)
#
# In both cases, solutions exactly eps% WORSE than baseline are on the boundary,
# and the OPTIMAL baseline solution always hits (base_opt ≤ threshold).
#
# The bug report claimed the formula was wrong for negative objectives and
# proposed `base_opt * (1 - eps)` as a fix. But:
#   -100 * (1 - 0.01) = -99   (same as current formula — no change!)
# and the alternative `base_opt + base_opt * eps`:
#   -100 + (-100 * 0.01) = -101  (stricter than baseline — requires beating it)
#
# Neither variant improves on the current formula for the stated use-case.
# The tests below document the CORRECT expected behavior of the current formula.
# ---------------------------------------------------------------------------

class TestEpsilonThreshold:
    def test__hit_time_metrics__positive_baseline__threshold_above_baseline(
        self,
    ):
        """For positive base_opt, threshold > base_opt (allows eps% WORSE)."""
        # ARRANGE
        base_opt = 10.0
        eps = 0.01
        summary_df, traj_by_presolve, cfg = _make_hit_df_fixture(
            n_seeds=1, n_hitting=1, base_opt=base_opt, eps=eps, hit_times=[0.2]
        )

        # ACT
        hit_df = prepare_hit_time_metrics(
            summary_df, traj_by_presolve, cfg, thresholds=[eps]
        )

        # ASSERT — threshold must be above base_opt (allows 1% worse solutions)
        variant_row = hit_df[~hit_df["is_baseline"]].iloc[0]
        expected_threshold = base_opt + abs(base_opt) * eps
        assert variant_row["target_obj"] == pytest.approx(expected_threshold)
        assert variant_row["target_obj"] > base_opt, (
            "For positive baseline, threshold should be > base_opt (allows eps% worse)"
        )

    def test__hit_time_metrics__at_eps_zero__threshold_equals_baseline(self):
        """eps=0 means 'match baseline exactly': threshold == base_opt."""
        # ARRANGE
        base_opt = 10.0
        summary_df, traj_by_presolve, cfg = _make_hit_df_fixture(
            n_seeds=1, n_hitting=1, base_opt=base_opt, eps=0.0, hit_times=[0.1]
        )

        # ACT
        hit_df = prepare_hit_time_metrics(
            summary_df, traj_by_presolve, cfg, thresholds=[0.0]
        )

        # ASSERT
        variant_row = hit_df[~hit_df["is_baseline"]].iloc[0]
        assert variant_row["target_obj"] == pytest.approx(base_opt, abs=1e-8)

    def test__hit_time_metrics__baseline_solution_always_hits_its_own_threshold(self):
        """The baseline optimum must always satisfy (baseline_opt ≤ threshold)."""
        # ARRANGE
        for base_opt in [5.0, 10.0, 100.0]:
            for eps in [0.0, 0.01, 0.05]:
                threshold = base_opt + abs(base_opt) * eps
                # ACT / ASSERT
                assert base_opt <= threshold + 1e-9, (
                    f"base_opt={base_opt} does not satisfy its own threshold={threshold} at eps={eps}"
                )


# ---------------------------------------------------------------------------
# Bug #4 — SEM denominator consistency  (CONFIRMED INCONSISTENCY)
# ---------------------------------------------------------------------------

class TestSemConsistency:
    def test__hit_time_metrics__hit_count_and_total_count_differ_when_seeds_miss(self):
        """Demonstrates the asymmetry: with partial hits, hit_count < total_count.

        This test confirms the fixture data exposes the inconsistency (hit_count=1,
        total_count=2) so that the fix (using total_count for sem_hit_time) is
        verifiable via plot_bar_hit_time / compute_hit_time_plot_limits.
        """
        # ARRANGE — 2 seeds, only seed 0 hits
        summary_df, traj_by_presolve, cfg = _make_hit_df_fixture(
            n_seeds=2, n_hitting=1, base_opt=10.0, eps=0.01, hit_times=[0.2]
        )

        # ACT
        hit_df = prepare_hit_time_metrics(
            summary_df, traj_by_presolve, cfg, thresholds=[0.01]
        )

        # ASSERT — one seed hit, one missed
        variant_rows = hit_df[~hit_df["is_baseline"]]
        assert variant_rows["hit_found"].sum() == 1, "Exactly one seed should hit"
        total_seeds = len(variant_rows)
        hitting_seeds = int(variant_rows["hit_found"].sum())
        assert hitting_seeds < total_seeds, (
            "hit_count < total_count required to expose Bug #4"
        )

    def test__aggregate_hit_time_stats__sem_uses_total_count_not_hit_count(self):
        """BUG #4: sem_hit_time must use total_count (all seeds), not hit_count
        (only seeds that hit), so it is consistent with sem_penalized.

        With 3 seeds and 2 hitting (hit times 1.0 and 3.0):
          std ≈ 1.414
          sem_hit   = std / sqrt(hit_count=2)  ≈ 1.0   — WRONG (old code)
          sem_total = std / sqrt(total_count=3) ≈ 0.816 — CORRECT (fixed)
        """
        # ARRANGE — build a minimal hit_df with 3 seeds, 2 hitting
        hit_times = [1.0, 3.0, np.nan]  # seed 2 misses
        penalized_times = [1.0, 3.0, 9.0]
        rows = [
            {
                "penalty_round": 0.3,
                "hit_time_sec": ht,
                "penalized_hit_time_sec": pt,
                "seed": i,
            }
            for i, (ht, pt) in enumerate(zip(hit_times, penalized_times))
        ]
        eps_variants = pd.DataFrame(rows)

        # ACT
        grouped = _aggregate_hit_time_stats(eps_variants)

        # ASSERT
        row = grouped.iloc[0]
        std_val = np.std([1.0, 3.0], ddof=1)  # pandas std uses ddof=1
        expected_sem = std_val / np.sqrt(3)    # total_count = 3
        wrong_sem    = std_val / np.sqrt(2)    # hit_count = 2

        assert row["total_count"] == 3
        assert row["hit_count"] == 2
        assert row["sem_hit_time"] == pytest.approx(expected_sem, rel=1e-6), (
            f"sem_hit_time should be std/sqrt(total_count)={expected_sem:.4f}, "
            f"not std/sqrt(hit_count)={wrong_sem:.4f}"
        )
        # sem_penalized must use the same denominator
        assert row["sem_penalized"] == pytest.approx(
            np.std([1.0, 3.0, 9.0], ddof=1) / np.sqrt(3), rel=1e-6
        )
