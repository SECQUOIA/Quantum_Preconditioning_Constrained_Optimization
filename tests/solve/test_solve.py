"""Unit tests for qp_gurobi/solve.py.

Pure-logic tests (no Gurobi) run unconditionally.
Tests marked `integration` require a live Gurobi license and are skipped otherwise.
"""
from __future__ import annotations

import pytest

from qp_gurobi.instance import IsingInstance, eval_ising, z_from_x
from qp_gurobi.solve import _ising_to_qubo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inst(n: int, constant: float = 0.0, linear=None, quadratic=None) -> IsingInstance:
    return IsingInstance(
        n=n,
        constant=constant,
        linear=linear or {},
        quadratic=quadratic or {},
    )


# ---------------------------------------------------------------------------
# _ising_to_qubo  (pure, no Gurobi)
# ---------------------------------------------------------------------------

class TestIsingToQubo:
    def test__ising_to_qubo__given_constant_only__no_linear_or_quadratic(self):
        # ARRANGE
        inst = _inst(n=2, constant=5.0)

        # ACT
        const, lin, quad = _ising_to_qubo(inst)

        # ASSERT
        assert const == pytest.approx(5.0)
        assert lin == {}
        assert quad == {}

    def test__ising_to_qubo__given_linear_term__produces_correct_qubo_linear(self):
        # ARRANGE — Ising: h_0 * z_0, substitute z_0 = 2x_0 - 1
        #   => h * (2x-1) = -h (constant shift) + 2h * x
        inst = _inst(n=2, linear={0: 3.0})

        # ACT
        const, lin, quad = _ising_to_qubo(inst)

        # ASSERT
        assert const == pytest.approx(-3.0)    # -h
        assert lin[0] == pytest.approx(6.0)    # 2h
        assert quad == {}

    def test__ising_to_qubo__given_quadratic_coupling__produces_correct_qubo(self):
        # ARRANGE — Ising: w * z_i z_j, substitute z = 2x-1
        #   => w*(2x_i-1)*(2x_j-1) = w - 2w x_i - 2w x_j + 4w x_i x_j
        inst = _inst(n=2, quadratic={(0, 1): 2.0})

        # ACT
        const, lin, quad = _ising_to_qubo(inst)

        # ASSERT
        assert const == pytest.approx(2.0)           # +w
        assert lin[0] == pytest.approx(-4.0)         # -2w
        assert lin[1] == pytest.approx(-4.0)         # -2w
        assert quad[(0, 1)] == pytest.approx(8.0)    # 4w

    def test__ising_to_qubo__zero_linear_coefficients_are_filtered(self):
        # ARRANGE — two couplings cancel their linear contributions at node 0
        inst = _inst(n=3, quadratic={(0, 1): 1.0, (0, 2): -1.0})

        # ACT
        _, lin, _ = _ising_to_qubo(inst)

        # ASSERT — coefficient at 0 is -2+2=0, should be absent
        assert 0 not in lin

    def test__ising_to_qubo__qubo_and_ising_agree_at_all_binary_assignments(self):
        """QUBO objective evaluated at x must equal Ising at z=2x-1."""
        # ARRANGE
        inst = _inst(n=3, constant=1.0, linear={1: -0.5},
                     quadratic={(0, 1): 2.0, (1, 2): -1.0})
        const, lin, quad = _ising_to_qubo(inst)

        for mask in range(8):  # all 2^3 assignments
            x = [(mask >> i) & 1 for i in range(3)]
            z = z_from_x(x)

            # ACT — evaluate QUBO at x
            qubo_val = const
            for i, c in lin.items():
                qubo_val += c * x[i]
            for (i, j), c in quad.items():
                qubo_val += c * x[i] * x[j]

            # ASSERT — must match Ising at z
            ising_val = eval_ising(inst, z)
            assert qubo_val == pytest.approx(ising_val, abs=1e-9), (
                f"Mismatch at x={x}: QUBO={qubo_val}, Ising={ising_val}"
            )


# ---------------------------------------------------------------------------
# Happy-path integration test (integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test__solve_bisection_ip__given_small_feasible_instance__returns_feasible_result():
    """End-to-end: a tiny 4-node instance should solve to a balanced partition."""
    pytest.importorskip("gurobipy", reason="Gurobi not available")

    from qp_gurobi.solve import solve_bisection_ip

    # ARRANGE — 4-node instance; optimal cut separates {0,1} from {2,3}
    inst = IsingInstance(n=4, constant=0.0, linear={}, quadratic={(0, 2): 1.0, (1, 3): 1.0})

    # ACT
    result = solve_bisection_ip(inst, inst, name="happy-path", output_flag=0)

    # ASSERT — feasible solution found with correct structural properties
    assert result.n == 4
    assert not any(v != v for v in [result.objective_model, result.objective_baseline])  # no NaN
    assert result.sum_z == 0, "bisection constraint: sum of spins must be 0"
    assert len(result.x_bits) == 4
    assert sum(result.x_bits) == 2, "exactly half the bits should be 1"
    assert result.runtime_sec >= 0.0


# ---------------------------------------------------------------------------
# Bug #1 — time_to_best_sec tracks last incumbent, not best (integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test__solve_bisection_ip__time_to_best_reflects_original_obj_minimum_not_last_incumbent():
    """
    BUG #1: the callback overwrites best_t[0] on every incumbent, so
    time_to_best_sec ends up pointing to the *last* solution event rather than
    the one with the minimum original-objective value.

    This test constructs a tiny instance where we can verify (via trajectory)
    that the reported time_to_best_sec matches the event whose orig_obj is
    lowest, not necessarily the final callback event.

    Gurobi is required; skip if not available.
    """
    pytest.importorskip("gurobipy", reason="Gurobi not available")

    import json
    from qp_gurobi.solve import solve_bisection_ip

    # ARRANGE — 4-node balanced bisection problem with known structure
    baseline = IsingInstance(n=4, constant=0.0, linear={}, quadratic={(0, 1): 1.0, (2, 3): 1.0})

    # ACT
    result = solve_bisection_ip(baseline, baseline, name="test", output_flag=0)

    # ASSERT — if multiple incumbents, time_to_best_sec must match the event
    # in the trajectory with the minimum orig_obj value
    if len(result.trajectory) > 1:
        best_event_time = min(result.trajectory, key=lambda e: e[1])[0]
        assert result.time_to_best_sec == pytest.approx(best_event_time, abs=1e-6), (
            "time_to_best_sec should point to the best orig_obj event, not the last"
        )
