from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .instance import IsingInstance, bisection_violation, eval_ising, read_dat, z_from_x


@dataclass
class SolveResult:
    """Container for one Gurobi solve result.
    
    Attributes
    ----------
    name : str
        Run name.
    n : int
        Problem size.
    objective_model : float
        Objective value in the solved model.
    objective_baseline : float
        Solution value evaluated with the original objective.
    runtime_sec : float
        Total solver runtime in seconds.
    time_to_best_sec : float
        Callback time of the final incumbent for the solved model.
    trajectory : list of tuple
        Incumbent callback events as ``(time_sec, original_objective)``.
    """
    name: str
    n: int
    objective_model: float
    objective_baseline: float
    penalty: Optional[float]
    mip_status: int
    runtime_sec: float
    time_to_best_sec: float
    mip_gap: Optional[float]
    x_bits: List[int]
    z_bits: List[int]
    sum_z: int
    presolve: bool
    # Each entry is (wall_time_sec, original_obj_value) for every incumbent
    # found by the solver (evaluated against the *original* baseline objective).
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    def to_row(self) -> Dict[str, object]:
        """Serialize the solve result to a CSV-friendly row dictionary."""
        d = asdict(self)
        d["x_bits"] = "".join(str(b) for b in self.x_bits)
        d["z_bits"] = "".join("+" if z == 1 else "-" for z in self.z_bits)
        d["trajectory"] = json.dumps(
            [[round(t, 6), round(v, 8)] for t, v in self.trajectory]
        )
        return d


def _ising_to_qubo(instance: IsingInstance):
    """Convert an Ising objective in spins to a binary QUBO expression."""
    constant = float(instance.constant)
    linear: Dict[int, float] = {}
    quadratic: Dict[Tuple[int, int], float] = {}
    for i, h in instance.linear.items():
        constant -= h
        linear[i] = linear.get(i, 0.0) + 2.0 * h
    for (i, j), w in instance.quadratic.items():
        constant += w
        linear[i] = linear.get(i, 0.0) - 2.0 * w
        linear[j] = linear.get(j, 0.0) - 2.0 * w
        a, b = (i, j) if i < j else (j, i)
        quadratic[(a, b)] = quadratic.get((a, b), 0.0) + 4.0 * w
    linear = {k: v for k, v in linear.items() if abs(v) > 0}
    quadratic = {k: v for k, v in quadratic.items() if abs(v) > 0}
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
    presolve: bool = True,
) -> SolveResult:
    """Solve a balanced bisection integer program with Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError("gurobipy not available") from e

    if instance.n % 2 != 0:
        raise ValueError(f"n must be even for balanced bisection; got n={instance.n}")
    if instance.n != baseline_instance.n:
        raise ValueError(
            f"instance.n={instance.n} != baseline_instance.n={baseline_instance.n}"
        )

    n = instance.n
    const, lin, quad = _ising_to_qubo(instance)

    model = gp.Model(name)
    # Enable Gurobi output whenever file logging or console output is requested.
    # OutputFlag=0 suppresses file logging too, so decouple the two controls.
    model.Params.OutputFlag = 1 if (log_file or output_flag) else 0
    model.Params.LogToConsole = 1 if output_flag else 0
    if log_file:
        model.Params.LogFile = str(log_file)
    if time_limit_sec is not None:
        model.Params.TimeLimit = float(time_limit_sec)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)
    if seed is not None:
        model.Params.Seed = int(seed)
    if not presolve:
        model.Params.Presolve = 0

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    obj = gp.LinExpr(const)
    for i, c in lin.items():
        obj += c * x[i]
    for (i, j), c in quad.items():
        obj += c * x[i] * x[j]
    model.setObjective(obj, GRB.MINIMIZE)
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == n / 2, name="bisection")

    best_t: List[float] = []
    trajectory: List[Tuple[float, float]] = []

    def _cb(model, where):
        """Record incumbent callback events during Gurobi optimization."""
        if where == GRB.Callback.MIPSOL:
            t = float(model.cbGet(GRB.Callback.RUNTIME))
            # Evaluate each preconditioned problem's feasible solution against the original objective.
            x_sol = [model.cbGetSolution(x[i]) for i in range(n)]
            z_sol = z_from_x([int(round(v)) for v in x_sol])
            orig_obj = eval_ising(baseline_instance, z_sol)
            trajectory.append((t, orig_obj))
            # Track time of last preconditioned-objective incumbent (time-to-best).
            if best_t:
                best_t[0] = t
            else:
                best_t.append(t)

    model.optimize(_cb)

    status = int(model.Status)
    feasible = model.SolCount > 0  # safe under TIME_LIMIT with no incumbent
    x_bits = [int(round(x[i].X)) for i in range(n)] if feasible else [0] * n
    z_bits = z_from_x(x_bits)

    return SolveResult(
        name=name,
        n=n,
        objective_model=float(model.ObjVal) if feasible else float("nan"),
        objective_baseline=eval_ising(baseline_instance, z_bits) if feasible else float("nan"),
        penalty=penalty,
        mip_status=status,
        runtime_sec=float(model.Runtime),
        time_to_best_sec=best_t[0] if best_t else float("nan"),
        mip_gap=float(model.MIPGap) if feasible else None,
        x_bits=x_bits,
        z_bits=z_bits,
        sum_z=bisection_violation(z_bits),
        presolve=presolve,
        trajectory=trajectory,
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
    presolve: bool = True,
) -> List[SolveResult]:
    """Solve baseline and preconditioned instances loaded from filesystem paths."""
    baseline = read_dat(baseline_path)
    log_path = Path(log_dir) if log_dir else None
    if log_path:
        log_path.mkdir(parents=True, exist_ok=True)

    def _solve(inst, name, penalty):
        """Solve one instance path using shared baseline and solver settings."""
        return solve_bisection_ip(
            inst, baseline, name=name, penalty=penalty,
            time_limit_sec=time_limit_sec, mip_gap=mip_gap,
            threads=threads, seed=seed, output_flag=output_flag,
            log_file=(log_path / f"{name}.log") if log_path else None,
            presolve=presolve,
        )

    results = [_solve(baseline, "baseline", None)]
    for pen, pth in preconditioned_paths:
        results.append(_solve(read_dat(pth), f"precond_pen={pen:.3f}", float(pen)))
    return results
