from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class IsingInstance:
    """Ising objective over z in {+1,-1}:  c + sum_{i<j} w_ij z_i z_j.

    Notes:
    - This matches the observed .dat format in this repo.
    - Linear terms are supported in parsing but not present in current data.
    """

    n: int
    constant: float
    linear: Dict[int, float]
    quadratic: Dict[Tuple[int, int], float]  # keys are (i,j) with i<j

    def quadratic_items(self) -> Iterable[Tuple[int, int, float]]:
        """Yield quadratic coefficients as ``(i, j, weight)`` triples."""
        for (i, j), w in self.quadratic.items():
            yield i, j, w


def _parse_numbers(line: str) -> List[float]:
    """Parse whitespace-separated numbers from a line."""
    return [float(x) for x in line.strip().split()]


def read_dat(path: str | Path) -> IsingInstance:
    """Read a .dat file of the form described in the dataset README.

    Supported term lines:
    - constant: <c>
    - linear:   <i> <w_i>
    - quad:     <i> <j> <w_ij>

    The file may omit the constant term (observed for preconditioned files).
    """

    path = Path(path)
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty file: {path}")

    header = lines[0].split()
    if len(header) != 2:
        raise ValueError(f"Invalid header in {path}: {lines[0]!r}")
    n = int(header[0])
    declared_terms = int(header[1])

    term_lines = lines[1:]
    if len(term_lines) != declared_terms:
        # Some datasets include blank lines; we already stripped them.
        raise ValueError(
            f"Term count mismatch in {path}: header says {declared_terms}, got {len(term_lines)}"
        )

    constant: float | None = None
    linear: Dict[int, float] = {}
    quadratic: Dict[Tuple[int, int], float] = {}

    for idx, ln in enumerate(term_lines):
        parts = ln.split()
        if len(parts) == 1:
            if constant is not None:
                raise ValueError(f"Multiple constant terms in {path} at term line {idx+1}")
            constant = float(parts[0])
            continue

        if len(parts) == 2:
            i = int(parts[0])
            w = float(parts[1])
            if not (0 <= i < n):
                raise ValueError(f"Linear index out of range in {path}: {i}")
            linear[i] = linear.get(i, 0.0) + w
            continue

        if len(parts) == 3:
            i = int(parts[0])
            j = int(parts[1])
            w = float(parts[2])
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"Quadratic index out of range in {path}: {i},{j}")
            if i == j:
                # z_i^2 = 1; fold into constant.
                if constant is None:
                    constant = 0.0
                constant += w
                continue
            a, b = (i, j) if i < j else (j, i)
            quadratic[(a, b)] = quadratic.get((a, b), 0.0) + w
            continue

        raise ValueError(f"Invalid term line in {path}: {ln!r}")

    if constant is None:
        constant = 0.0

    return IsingInstance(n=n, constant=constant, linear=linear, quadratic=quadratic)


def z_from_x(x_bits: List[int]) -> List[int]:
    """Map x in {0,1} to z in {-1,+1} via z=2x-1."""

    return [2 * int(b) - 1 for b in x_bits]


def x_from_z(z_bits: List[int]) -> List[int]:
    """Map z in {-1,+1} to x in {0,1} via x=(z+1)/2."""

    out: List[int] = []
    for z in z_bits:
        if z not in (-1, 1):
            raise ValueError(f"z must be ±1, got {z}")
        out.append((z + 1) // 2)
    return out


def eval_ising(instance: IsingInstance, z_bits: List[int]) -> float:
    """Compute c + sum_i h_i z_i + sum_{i<j} w_ij z_i z_j."""

    if len(z_bits) != instance.n:
        raise ValueError(f"z length {len(z_bits)} != n {instance.n}")

    val = instance.constant
    for i, h in instance.linear.items():
        val += h * z_bits[i]
    for (i, j), w in instance.quadratic.items():
        val += w * z_bits[i] * z_bits[j]
    return float(val)


def bisection_violation(z_bits: List[int]) -> int:
    """Return sum(z_i); feasibility requires 0."""

    return int(sum(z_bits))


def trajectory_to_running_best(
    trajectory: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Convert raw (time, obj) incumbent events to a running-minimum trajectory.

    Returns a list of (time_sec, best_orig_obj_so_far) where each entry reflects
    the best original-objective value seen up to and including that time point.
    Events are assumed to already be sorted by time (as recorded by the callback).
    """
    result: List[Tuple[float, float]] = []
    best = float("inf")
    for t, v in trajectory:
        if v < best:
            best = v
        result.append((t, best))
    return result
