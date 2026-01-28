from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .instance import IsingInstance, bisection_violation, eval_ising, read_dat, x_from_z, z_from_x


@dataclass(frozen=True)
class SolveResult:
    name: str
    n: int
    objective_model: float
    objective_baseline: float
    penalty: Optional[float]
    mip_status: int
    runtime_sec: float
    mip_gap: Optional[float]
    x_bits: List[int]
    z_bits: List[int]
    sum_z: int

    def to_row(self) -> Dict[str, object]:
        d = asdict(self)
        d["x_bits"] = "".join(str(b) for b in self.x_bits)
        d["z_bits"] = "".join("+" if z == 1 else "-" for z in self.z_bits)
        return d


def _build_quadratic_objective_over_x(instance: IsingInstance) -> Tuple[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Convert Ising objective to QUBO-style objective in x (0/1) for Gurobi.

    Given z=2x-1, we can expand:
    w_ij z_i z_j = w_ij (2x_i-1)(2x_j-1) = 4 w_ij x_i x_j - 2 w_ij x_i - 2 w_ij x_j + w_ij
    h_i z_i = h_i (2x_i-1) = 2 h_i x_i - h_i

    Returns (constant, linear, quadratic) for objective:
      constant + sum_i lin[i]*x_i + sum_{i<j} quad[(i,j)]*x_i*x_j
    """

    constant = float(instance.constant)
    linear: Dict[int, float] = {i: 0.0 for i in range(instance.n)}
    quadratic: Dict[Tuple[int, int], float] = {}

    # linear Ising terms
    for i, h in instance.linear.items():
        constant += -h
        linear[i] += 2.0 * h

    # quadratic Ising terms
    for (i, j), w in instance.quadratic.items():
        constant += w
        linear[i] += -2.0 * w
        linear[j] += -2.0 * w
        a, b = (i, j) if i < j else (j, i)
        quadratic[(a, b)] = quadratic.get((a, b), 0.0) + 4.0 * w

    # remove near-zeros to keep model tidy
    linear = {i: v for i, v in linear.items() if abs(v) > 0.0}
    quadratic = {k: v for k, v in quadratic.items() if abs(v) > 0.0}

    return constant, linear, quadratic


def solve_bisection_ip(
    instance: IsingInstance,
    baseline_instance: IsingInstance,
    *,
    name: str,
    penalty: Optional[float] = None,
    time_limit_sec: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    seed: Optional[int] = None,
    output_flag: int = 0,
    log_file: Optional[str | Path] = None,
) -> SolveResult:
    """Solve hard bisection using Gurobi.

    Constraint: sum(x_i) == n/2, where z_i = 2x_i-1.

    objective_model: objective value for `instance` (with its own coefficients)
    objective_baseline: evaluation of returned bitstring on `baseline_instance`
    """

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "gurobipy is required to solve. Install and ensure a Gurobi license is available."
        ) from e

    if instance.n != baseline_instance.n:
        raise ValueError("instance and baseline_instance must have same n")
    n = instance.n
    if n % 2 != 0:
        raise ValueError(f"Bisection requires even n, got {n}")

    const, lin, quad = _build_quadratic_objective_over_x(instance)

    model = gp.Model(name)
    model.Params.OutputFlag = int(output_flag)
    # Ensure console output is enabled when requested.
    if int(output_flag) != 0:
        try:
            model.Params.LogToConsole = 1
        except Exception:
            pass
    if log_file is not None:
        model.Params.LogFile = str(log_file)
    if time_limit_sec is not None:
        model.Params.TimeLimit = float(time_limit_sec)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)
    if seed is not None:
        model.Params.Seed = int(seed)

    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    obj = gp.LinExpr(const)
    for i, c in lin.items():
        obj += c * x[i]
    for (i, j), c in quad.items():
        obj += c * x[i] * x[j]

    model.setObjective(obj, GRB.MINIMIZE)
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == n / 2, name="bisection")

    model.optimize()

    status = int(model.Status)
    runtime = float(model.Runtime)

    x_bits: List[int]
    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        x_bits = [int(round(x[i].X)) for i in range(n)]
    else:
        x_bits = [0] * n

    z_bits = z_from_x(x_bits)

    objective_model = float(model.ObjVal) if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) else float("nan")
    objective_baseline = eval_ising(baseline_instance, z_bits) if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) else float("nan")

    gap: Optional[float] = None
    try:
        gap = float(model.MIPGap) if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) else None
    except Exception:
        gap = None

    return SolveResult(
        name=name,
        n=n,
        objective_model=objective_model,
        objective_baseline=objective_baseline,
        penalty=penalty,
        mip_status=status,
        runtime_sec=runtime,
        mip_gap=gap,
        x_bits=x_bits,
        z_bits=z_bits,
        sum_z=bisection_violation(z_bits),
    )


def solve_from_paths(
    *,
    baseline_path: str | Path,
    preconditioned_paths: List[Tuple[float, str | Path]],
    time_limit_sec: Optional[float] = None,
    mip_gap: Optional[float] = None,
    threads: Optional[int] = None,
    seed: Optional[int] = None,
    output_flag: int = 0,
    log_dir: Optional[str | Path] = None,
) -> List[SolveResult]:
    baseline = read_dat(baseline_path)

    log_dir_path: Optional[Path] = None
    if log_dir is not None:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

    results: List[SolveResult] = []
    results.append(
        solve_bisection_ip(
            baseline,
            baseline,
            name="baseline",
            penalty=None,
            time_limit_sec=time_limit_sec,
            mip_gap=mip_gap,
            threads=threads,
            seed=seed,
            output_flag=output_flag,
            log_file=(log_dir_path / "baseline.log") if log_dir_path is not None else None,
        )
    )

    for pen, pth in preconditioned_paths:
        inst = read_dat(pth)
        results.append(
            solve_bisection_ip(
                inst,
                baseline,
                name=f"precond_pen={pen:.3f}",
                penalty=float(pen),
                time_limit_sec=time_limit_sec,
                mip_gap=mip_gap,
                threads=threads,
                seed=seed,
                output_flag=output_flag,
                log_file=(log_dir_path / f"precond_pen={pen:.3f}.log") if log_dir_path is not None else None,
            )
        )

    return results
