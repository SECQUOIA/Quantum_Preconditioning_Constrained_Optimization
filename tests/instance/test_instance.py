"""Unit tests for qp_gurobi/instance.py — no external dependencies."""
from __future__ import annotations

import math

import pytest

from qp_gurobi.instance import (
    IsingInstance,
    bisection_violation,
    eval_ising,
    read_dat,
    trajectory_to_running_best,
    x_from_z,
    z_from_x,
)


# ---------------------------------------------------------------------------
# read_dat
# ---------------------------------------------------------------------------

class TestReadDat:
    def test__read_dat__given_constant_and_quadratic__parses_all_terms(self, write_dat):
        # ARRANGE — 4-node file: constant + one quadratic coupling
        path = write_dat([
            "4 2",
            "3.5",
            "0 1 2.0",
        ])

        # ACT
        inst = read_dat(path)

        # ASSERT
        assert inst.n == 4
        assert inst.constant == pytest.approx(3.5)
        assert inst.quadratic[(0, 1)] == pytest.approx(2.0)
        assert inst.linear == {}

    def test__read_dat__given_linear_terms__parses_correctly(self, write_dat):
        # ARRANGE
        path = write_dat([
            "3 2",
            "0 1.5",
            "1 -0.5",
        ])

        # ACT
        inst = read_dat(path)

        # ASSERT
        assert inst.n == 3
        assert inst.linear[0] == pytest.approx(1.5)
        assert inst.linear[1] == pytest.approx(-0.5)
        assert inst.quadratic == {}
        assert inst.constant == pytest.approx(0.0)

    def test__read_dat__given_diagonal_quadratic__folds_into_constant(self, write_dat):
        # ARRANGE — z_i^2 = 1 so w_ii should fold into constant
        path = write_dat([
            "2 1",
            "0 0 5.0",
        ])

        # ACT
        inst = read_dat(path)

        # ASSERT
        assert inst.constant == pytest.approx(5.0)
        assert inst.quadratic == {}

    def test__read_dat__given_reversed_quadratic_indices__normalises_to_i_lt_j(self, write_dat):
        # ARRANGE — file uses (j, i) with j > i; should be stored as (i, j)
        path = write_dat([
            "4 1",
            "3 1 -1.0",
        ])

        # ACT
        inst = read_dat(path)

        # ASSERT — key must be (1, 3) not (3, 1)
        assert (1, 3) in inst.quadratic
        assert (3, 1) not in inst.quadratic

    def test__read_dat__given_empty_file__raises_value_error(self, write_dat):
        # ARRANGE
        path = write_dat([])

        # ACT / ASSERT
        with pytest.raises(ValueError, match="Empty file"):
            read_dat(path)

    def test__read_dat__given_term_count_mismatch__raises_value_error(self, write_dat):
        # ARRANGE — header declares 3 terms, only 1 provided
        path = write_dat([
            "4 3",
            "0 1 2.0",
        ])

        # ACT / ASSERT
        with pytest.raises(ValueError, match="Term count mismatch"):
            read_dat(path)

    def test__read_dat__given_out_of_range_index__raises_value_error(self, write_dat):
        # ARRANGE — index 9 is out of range for n=4
        path = write_dat([
            "4 1",
            "0 9 1.0",
        ])

        # ACT / ASSERT
        with pytest.raises(ValueError, match="out of range"):
            read_dat(path)

    def test__read_dat__given_no_constant_line__defaults_to_zero(self, write_dat):
        # ARRANGE — preconditioned files omit the constant
        path = write_dat([
            "2 1",
            "0 1 1.0",
        ])

        # ACT
        inst = read_dat(path)

        # ASSERT
        assert inst.constant == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# eval_ising
# ---------------------------------------------------------------------------

class TestEvalIsing:
    def test__eval_ising__given_pure_constant__returns_constant(self):
        # ARRANGE
        inst = IsingInstance(n=2, constant=7.0, linear={}, quadratic={})
        z = [1, -1]

        # ACT
        val = eval_ising(inst, z)

        # ASSERT
        assert val == pytest.approx(7.0)

    def test__eval_ising__given_quadratic_coupling__computes_correctly(self):
        # ARRANGE — E = 0 + w_{01} z_0 z_1 = 3.0 * (1)*(-1) = -3.0
        inst = IsingInstance(n=2, constant=0.0, linear={}, quadratic={(0, 1): 3.0})
        z = [1, -1]

        # ACT
        val = eval_ising(inst, z)

        # ASSERT
        assert val == pytest.approx(-3.0)

    def test__eval_ising__given_linear_term__computes_correctly(self):
        # ARRANGE — E = h_0 z_0 = 2.0 * (-1) = -2.0
        inst = IsingInstance(n=2, constant=0.0, linear={0: 2.0}, quadratic={})
        z = [-1, 1]

        # ACT
        val = eval_ising(inst, z)

        # ASSERT
        assert val == pytest.approx(-2.0)

    def test__eval_ising__given_wrong_z_length__raises_value_error(self):
        # ARRANGE
        inst = IsingInstance(n=4, constant=0.0, linear={}, quadratic={})

        # ACT / ASSERT
        with pytest.raises(ValueError, match="z length"):
            eval_ising(inst, [1, -1])

    def test__eval_ising__given_all_terms_combined__returns_correct_sum(self):
        # ARRANGE — E = 1.0 + 2.0*z_0 + 3.0*z_0*z_1 with z=[1,1]
        #           = 1.0 + 2.0 + 3.0 = 6.0
        inst = IsingInstance(n=2, constant=1.0, linear={0: 2.0}, quadratic={(0, 1): 3.0})

        # ACT
        val = eval_ising(inst, [1, 1])

        # ASSERT
        assert val == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# z_from_x / x_from_z
# ---------------------------------------------------------------------------

class TestSpinConversions:
    @pytest.mark.parametrize("x_bit, expected_z", [(0, -1), (1, 1)])
    def test__z_from_x__maps_each_bit_correctly(self, x_bit, expected_z):
        # ACT / ASSERT
        assert z_from_x([x_bit]) == [expected_z]

    def test__z_from_x__given_mixed_bits__maps_all(self):
        # ARRANGE / ACT
        result = z_from_x([0, 1, 0, 1])

        # ASSERT
        assert result == [-1, 1, -1, 1]

    @pytest.mark.parametrize("z_spin, expected_x", [(-1, 0), (1, 1)])
    def test__x_from_z__maps_each_spin_correctly(self, z_spin, expected_x):
        # ACT / ASSERT
        assert x_from_z([z_spin]) == [expected_x]

    def test__x_from_z__given_invalid_spin__raises_value_error(self):
        # ACT / ASSERT
        with pytest.raises(ValueError, match="±1"):
            x_from_z([0])

    def test__z_from_x_and_x_from_z__are_inverses(self):
        # ARRANGE
        original = [0, 1, 1, 0, 1]

        # ACT
        result = x_from_z(z_from_x(original))

        # ASSERT
        assert result == original


# ---------------------------------------------------------------------------
# bisection_violation
# ---------------------------------------------------------------------------

class TestBisectionViolation:
    def test__bisection_violation__given_balanced_partition__returns_zero(self):
        # ARRANGE — equal +1 and -1
        z = [1, 1, -1, -1]

        # ACT / ASSERT
        assert bisection_violation(z) == 0

    def test__bisection_violation__given_unbalanced__returns_nonzero_sum(self):
        # ARRANGE
        z = [1, 1, 1, -1]

        # ACT / ASSERT
        assert bisection_violation(z) == 2


# ---------------------------------------------------------------------------
# trajectory_to_running_best
# ---------------------------------------------------------------------------

class TestTrajectoryToRunningBest:
    def test__trajectory_to_running_best__given_monotone_improving__unchanged(self):
        # ARRANGE — objectives strictly decrease (improve)
        traj = [(0.1, 5.0), (0.5, 3.0), (1.0, 1.0)]

        # ACT
        result = trajectory_to_running_best(traj)

        # ASSERT
        assert [v for _, v in result] == pytest.approx([5.0, 3.0, 1.0])

    def test__trajectory_to_running_best__given_non_monotone__tracks_minimum(self):
        # ARRANGE — third event is worse than second
        traj = [(0.1, 5.0), (0.3, 2.0), (0.8, 4.0)]

        # ACT
        result = trajectory_to_running_best(traj)

        # ASSERT — running best stays at 2.0 after the third event
        assert [v for _, v in result] == pytest.approx([5.0, 2.0, 2.0])

    def test__trajectory_to_running_best__given_empty__returns_empty(self):
        # ACT / ASSERT
        assert trajectory_to_running_best([]) == []

    def test__trajectory_to_running_best__preserves_timestamps(self):
        # ARRANGE
        traj = [(0.1, 3.0), (0.9, 1.0)]

        # ACT
        result = trajectory_to_running_best(traj)

        # ASSERT
        times = [t for t, _ in result]
        assert times == pytest.approx([0.1, 0.9])
