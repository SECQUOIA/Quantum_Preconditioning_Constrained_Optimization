# Gurobi_QP: baseline vs quantum-preconditioned instances

This repo contains a tiny workflow to compare:

- **Baseline** graph-bisection objective from `Data/N=.../seed=.../problem.dat`
- **Preconditioned** objectives from `Data/N=.../seed=.../n_qaoa_layers=.../preconditioned_problem_pen=...dat`

Both are solved with the **same hard bisection constraint** (equal-size partition):

- variables are represented internally as `x_i ∈ {0,1}`
- with `z_i = 2x_i - 1 ∈ {−1,+1}`
- constraint: `sum_i z_i = 0`  ⇔  `sum_i x_i = N/2`

The key comparison is done by **re-evaluating** each preconditioned solution under the **baseline** objective.

## Requirements

- Python 3
- `gurobipy` and a working Gurobi license (the solver code imports `gurobipy`)
- For plotting in the notebook: `pandas`, `matplotlib`

## Run the experiment

From the repo root:

```bash
python3 scripts/run_experiment.py --n 8 --seed 0 --layers 1 --out results/demo.csv
```

If `gurobipy` is installed in a specific conda env (e.g. `benchmark`), run with that env’s Python:

This writes a CSV with one row for the baseline solve and one row per `penalty` file.

If you want to limit runtime:

```bash
python3 scripts/run_experiment.py --time-limit 60 --mip-gap 1e-6
```

## Analyze

Open and run the notebook:

- `notebooks/analysis.ipynb`

It expects `results/demo.csv` by default.

## Output columns

- `objective_model`: objective value for the model that was solved (baseline or preconditioned coefficients)
- `objective_baseline`: **baseline objective value** evaluated at the returned solution (this is what you compare across penalties)
- `x_bits`: bitstring of `x ∈ {0,1}`
- `z_bits`: visualization of `z ∈ {−1,+1}` as `-` and `+`
- `sum_z`: should be `0` due to hard constraint

## Code layout

- `qp_gurobi/instance.py`: .dat parser + objective evaluation
- `qp_gurobi/solve.py`: Gurobi model and solve routines
- `scripts/run_experiment.py`: CLI runner
- `notebooks/analysis.ipynb`: plots
