from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .instance import IsingInstance, bisection_violation, eval_ising, read_dat, z_from_x


@dataclass(frozen=True)
class SolveResult:
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

    def to_row(self) -> Dict[str, object]:
        d = asdict(self)
        d["x_bits"] = "".join(str(b) for b in self.x_bits)
        d["z_bits"] = "".join("+" if z == 1 else "-" for z in self.z_bits)
        return d


def _ising_to_qubo(instance: IsingInstance):
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
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError("gurobipy not available") from e

    n = instance.n
    const, lin, quad = _ising_to_qubo(instance)

    model = gp.Model(name)
    model.Params.OutputFlag = int(output_flag)
    if output_flag:
        model.Params.LogToConsole = 1
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

    def _cb(model, where):
        if where == GRB.Callback.MIPSOL:
            t = float(model.cbGet(GRB.Callback.RUNTIME))
            if best_t:
                best_t[0] = t
            else:
                best_t.append(t)

    model.optimize(_cb)

    status = int(model.Status)
    feasible = status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL)
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
    baseline = read_dat(baseline_path)
    log_path = Path(log_dir) if log_dir else None
    if log_path:
        log_path.mkdir(parents=True, exist_ok=True)

    def _solve(inst, name, penalty):
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
