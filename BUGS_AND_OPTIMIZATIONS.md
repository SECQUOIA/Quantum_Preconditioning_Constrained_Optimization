# Bugs and Optimization Opportunities

Originally identified via full codebase review on 2026-06-14. Re-verified against
the current code on 2026-07-28 — status of each item updated below.

---

## Bugs

### 1. `solve.py` — `time_to_best_sec` always reflects the *last* incumbent, not the *best* one `[FIXED]`

Fixed in `qp_gurobi/solve.py`'s `_cb` callback: `best_t[0]` is now only advanced
when `orig_obj` strictly improves over the running best (`best_orig[0]`), guarded
explicitly before the timestamp is recorded.

---

### 2. `utils.py` — Epsilon threshold formula for negative objectives `[NOT A BUG]`

Re-derived: for a minimization problem, "within ε of optimal" should mean
`threshold = base_opt + ε·|base_opt|` — a value proportionally *worse* (higher)
than `base_opt`, regardless of its sign. This is exactly what the current code
computes (`threshold_value = base_opt + abs(base_opt) * eps_value` in
`prepare_hit_time_metrics`, and the equivalent inline form used throughout the
scaling-plot functions in `utils.py`). The originally-proposed fix
(`base_opt * (1 - eps)`) only agrees with this when `base_opt < 0`; for
`base_opt > 0` it would produce a *stricter*, unreachable threshold. The current
formula is correct as written — no change needed.

---

### 3. `utils.py` — `prepare_penalty_metrics` cross-join bug when multiple N values present `[FIXED]`

Fixed: `merge_keys = ["seed", "presolve", "n"] if "n" in df.columns else ["seed", "presolve"]`
now includes `"n"` whenever the column is present, preventing the cross-N join.

---

### 4. `utils.py` — SEM for hit-time divided by `sqrt(hit_count)` instead of `sqrt(total_count)` `[FIXED]`

Fixed: `_aggregate_hit_time_stats` now divides both `sem_hit_time` and
`sem_penalized` by `sqrt(total_count)` (the full seed population), and separately
tracks `miss_count = total_count - hit_count`.

---

### 5. `solve.py` — `GRB.SUBOPTIMAL` treated as feasible without guarding `SolCount > 0` `[FIXED]`

Fixed: `feasible = model.SolCount > 0` gates all solution extraction
(`x_bits`, `objective_model`, `objective_baseline`), independent of status code.

---

### 6. `solve.py` — `model.MIPGap` accessed under `TIME_LIMIT` with `SolCount == 0` `[FIXED]`

Fixed: `mip_gap=float(model.MIPGap) if feasible else None` — guarded by the same
`feasible` flag as #5.

---

## Performance and Efficiency (still open — not required for correctness)

### 7. `utils.py` — `_interpolate_step` runs `np.searchsorted` inside a Python loop `[OPEN]`

Still a per-grid-point Python loop (`qp_gurobi/utils.py`, `_interpolate_step`).
Fix remains: vectorize with a single `np.searchsorted(t_arr, t_grid, side='right') - 1` call.

### 8. `utils.py` — `_filter_rows_reaching_original_optimum` row-wise JSON parsing `[REMOVED]`

This function was dead code (defined but never called anywhere in the repo)
and has been deleted as part of the 2026-07-28 dead-code cleanup. No longer
applicable.

### 9. `scripts/run_experiment.py` — trajectory rows held fully in memory `[OPEN]`

Still builds `traj_rows` by re-iterating `all_rows` after every seed completes,
rather than streaming per-seed. Only matters for very long trajectories /
many-seed runs; not an issue at current N/seed scales.

### 10. `utils.py` — `prepare_hit_time_metrics` nested V×S×E Python loop `[OPEN]`

Still a nested loop with a generator scan per `(name, seed, eps)` cell. Fix
remains: vectorize first-crossing computation per `(name, seed)` group.

---

## Priority Order (remaining)

| Priority | Item | Impact |
|---|---|---|
| 1 | Perf #7 — vectorize `_interpolate_step` | Slow trajectory plots |
| 2 | Perf #10 — vectorize hit-time metrics loop | Slow `prepare_hit_time_metrics` |
| 3 | Perf #9 — stream trajectory CSV output | High peak memory for large runs only |

All correctness bugs (#1, #3–#6) are fixed; #2 was a false positive; #8 was
dead code and has been removed. Only performance optimizations remain open,
and none are currently a practical bottleneck at the dataset sizes this repo
runs (N ≤ 40, ≤50 seeds).

## Dead code removed (2026-07-28)

Also identified and deleted during the same pass — defined but never called
anywhere in the repo (notebook, scripts, tests, or internally within
`utils.py`):

- `qp_gurobi/utils.py`: `load_layer_comparison_data`,
  `_filter_rows_reaching_original_optimum`,
  `plot_layer_comparison_panel_strategies`, `plot_layer_comparison_grid`,
  and the now-orphaned `_PANEL_STRATEGIES` constant.
- `notebooks/analysis.ipynb`: unused imports of the two removed plotting
  functions above, plus `plot_hit_time_lines` (only its side-by-side wrapper,
  `plot_hit_time_lines_side_by_side_presolve`, was ever called directly).
