# Gurobi_QP: baseline vs quantum-preconditioned instances

This repo contains a tiny workflow to compare:

- **Baseline** graph bipartitioning objective from `one_equality_new/complete_random/N=.../seed=.../problem.dat`
- **Preconditioned** objectives from `one_equality_new/complete_random/N=.../seed=.../n_qaoa_layers=.../preconditioned_problem_pen=<value>.dat`

Both are solved with the **same hard bipartitioning constraint** (equal-size partition):

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
python3 scripts/run_experiment.py --n 8 --seed 0 --layers 1 --out results/test.csv
```

By default, the runner looks under `one_equality_new/complete_random`.

Multi-seed run (flexible: discovers whatever penalty files exist per seed):

```bash
python3 scripts/run_experiment.py --n 20 --layers 1 --seeds 0,1,2 --out results/N=20_seeds=0_1_2_layers=1.csv
```

Or run all available `seed=*` folders under a given `N=...`:

```bash
python3 scripts/run_experiment.py --n 20 --layers 1 --all-seeds --out results/N=20_seeds=all_layers=1.csv
```

Disable Gurobi presolve for a pure branch-and-bound comparison:

```bash
python3 scripts/run_experiment.py --n 20 --layers 1 --all-seeds --no-presolve --out results/N=20_nopresolve_layers=1.csv
```

The runner also supports solver controls and diagnostics:

```bash
python3 scripts/run_experiment.py --n 20 --layers 1 --all-seeds --threads 8 --gurobi-seed 0 --log-dir logs
```

Use `--show-logs` to stream Gurobi logs to the console. A trajectory CSV is written by default next to the summary CSV; pass `--trajectory-out path/to/file.csv` to choose its path.

This writes a CSV with one row for the baseline solve and one row per `penalty` file.

If you want to limit runtime:

```bash
python3 scripts/run_experiment.py --time-limit 60 --mip-gap 1e-6
```

## Analyze

Open and run the notebook:

- `notebooks/analysis.ipynb`

It reads from the committed CSVs under `results/`.

## Output columns

- `objective_model`: objective value for the model that was solved (baseline or preconditioned coefficients)
- `objective_baseline`: **baseline objective value** evaluated at the returned solution (this is what you compare across penalties)
- `x_bits`: bitstring of `x ∈ {0,1}`
- `z_bits`: visualization of `z ∈ {−1,+1}` as `-` and `+`
- `sum_z`: should be `0` due to hard constraint
- `time_to_best_sec`: first time when the best baseline-objective incumbent was found
- `trajectory`: raw incumbent events as `(time_sec, objective_baseline)` pairs
- `presolve`: whether Gurobi presolve was enabled for the solve

## Code layout

- `qp_gurobi/instance.py`: .dat parser + objective evaluation
- `qp_gurobi/solve.py`: Gurobi model and solve routines
- `scripts/run_experiment.py`: CLI runner
- `notebooks/analysis.ipynb`: plots
