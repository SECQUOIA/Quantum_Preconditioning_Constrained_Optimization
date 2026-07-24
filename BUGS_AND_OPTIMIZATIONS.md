# Bugs and Optimization Opportunities

Identified via full codebase review on 2026-06-14.

---

## Bugs (Confirmed / Plausible)

### 1. `solve.py` ~line 140 — `time_to_best_sec` always reflects the *last* incumbent, not the *best* one `[CONFIRMED]`

`best_t[0]` is overwritten unconditionally on every MIPSOL callback. If the solver finds a worse original-objective incumbent after a good one (common with preconditioned objectives where model-objective and original-objective orderings diverge), `time_to_best_sec` gets the later, wrong timestamp.

**Fix:** Only update `best_t[0]` when `orig_obj` strictly improves over the running best.

---

### 2. `utils.py` ~line 976 — Epsilon threshold formula wrong for negative objectives `[CONFIRMED]`

`threshold = base_opt + abs(base_opt) * eps` gives `-99` when `base_opt = -100, eps = 0.01`. But 1%-within-optimum on a minimisation problem requires `threshold = base_opt * (1 - eps) = -101` (a stricter, more negative value). Right now all ε-hit times are reported optimistically — solutions 2% from optimal count as "hitting" the 1% threshold.

**Fix:** Replace `base_opt + abs(base_opt) * eps` with `base_opt * (1 - eps)` (or equivalently `base_opt + base_opt * eps` without the `abs`).

---

### 3. `utils.py` ~line 874 — `prepare_penalty_metrics` cross-join bug when multiple N values present `[CONFIRMED]`

Baseline rows are merged on `['seed', 'presolve']` with no `'n'` key. Any DataFrame spanning multiple problem sizes (e.g. loaded across N=12 and N=16) produces a Cartesian product: each baseline row from N=12 attaches to preconditioned rows from N=16 sharing the same seed number.

**Fix:** Include `'n'` in the merge keys: `on=['seed', 'presolve', 'n']`.

---

### 4. `utils.py` ~line 1057 — SEM for hit-time divides by `sqrt(hit_count)` not `sqrt(total_count)` `[CONFIRMED]`

`hit_count` is the pandas `.count()` of non-NaN values — it excludes seeds that never hit the threshold. The SEM is therefore computed over only the hitting subset and is artificially tight, producing error bars that are too narrow when many seeds miss the threshold. Line 2019 repeats the same bug. Line 1058 (penalised metric) correctly uses `total_count`, making the asymmetry obvious.

**Fix:** Use `sqrt(total_count)` (i.e. `sqrt(n_seeds)`) as the denominator in both places.

---

### 5. `solve.py` ~line 148 — `GRB.SUBOPTIMAL` treated as feasible without guarding `SolCount > 0` `[PLAUSIBLE]`

Gurobi can return `SUBOPTIMAL` (status 11) with no incumbent stored. Accessing `x[i].X` or `model.ObjVal` in that state raises `GurobiError`.

**Fix:** Gate solution extraction on `model.SolCount > 0` rather than status code alone.

---

### 6. `solve.py` ~line 161 — `model.MIPGap` accessed under `TIME_LIMIT` with `SolCount == 0` `[PLAUSIBLE]`

When a time-limited run terminates before finding any solution, Gurobi sets `MIPGap = GRB.INFINITY (1e100)`. The `feasible` flag is `True` for `TIME_LIMIT` regardless of `SolCount`, so this sentinel value gets written to the CSV silently rather than `None`, corrupting downstream gap analysis.

**Fix:** Guard `model.MIPGap` access with `model.SolCount > 0`; write `None` otherwise.

---

## Performance and Efficiency

### 7. `utils.py` ~line 1786 — `_interpolate_step` runs `np.searchsorted` inside a Python loop over 400 grid points

Called V×S times in `plot_mipsol_ratio_minus_one` (V variants × S seeds). The loop applies `searchsorted` one grid point at a time.

**Fix:** Replace the loop with a single vectorised call: `np.searchsorted(t_arr, t_grid, side='right') - 1`.

---

### 8. `utils.py` ~line 255 — `_filter_rows_reaching_original_optimum` calls `json.loads` + Python event scan per row via `.apply(axis=1)`

O(rows) Python dispatch with JSON parsing on every row. With many seeds and penalty variants this is the dominant cost of several analysis functions.

**Fix:** Explode the trajectory column once with `pd.json_normalize` / `explode`, compute threshold crossings vectorised with NumPy, then aggregate back — eliminating the per-row Python dispatch entirely.

---

### 9. `scripts/run_experiment.py` ~line 238 — Full trajectory expansion held in memory alongside raw `all_rows`

`traj_rows` is built by re-iterating `all_rows` after all seeds complete, so the JSON-string version and the expanded-dict version are both live simultaneously. For many seeds with long trajectories this can be 10–100× the size of the final CSV.

**Fix:** Stream `traj_rows` directly into the CSV writer inside the per-seed loop, so expanded rows are never all live at once.

---

### 10. `utils.py` ~line 930 — `prepare_hit_time_metrics` runs a V×S×E nested Python loop with generator scans over event lists

For V variants × S seeds × E epsilon thresholds the code executes a Python `next()` generator scan over zipped lists for each cell.

**Fix:** Pre-compute first-crossing times for all thresholds at once using `np.searchsorted` per `(name, seed)` group, then merge results back — replacing the triple nested loop with vectorised NumPy operations.

---

## Priority Order

| Priority | Item | Impact |
|---|---|---|
| 1 | Bug #1 — wrong `time_to_best_sec` | Corrupts all timing results |
| 2 | Bug #2 — wrong ε-threshold direction | Corrupts all hit-time and scaling plots |
| 3 | Bug #3 — cross-join on missing `n` key | Silent data corruption for multi-N loads |
| 4 | Bug #4 — SEM inflated when seeds miss threshold | Misleading error bars in all bar charts |
| 5 | Bug #5 & #6 — SUBOPTIMAL/TIME_LIMIT solution access | Rare but causes crashes or corrupt CSVs |
| 6 | Perf #7 — vectorise `_interpolate_step` | Slow trajectory plots |
| 7 | Perf #8 — vectorise row-wise JSON scan | Slow analysis functions |
| 8 | Perf #9 — stream trajectory CSV output | High peak memory for large runs |
| 9 | Perf #10 — vectorise hit-time metrics loop | Slow `prepare_hit_time_metrics` |
