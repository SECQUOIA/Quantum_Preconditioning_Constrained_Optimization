#!/bin/bash
set -uo pipefail
# Fill in the missing low-penalty range for N=8 (N=8 had zero quantum data in
# the base results/ before the pen1.1-2.0 cluster batch, which only covered
# rho 1.1-2.0). Each depth's missing range differs based on which
# preconditioned_problem_pen=*.dat files actually exist:
#   p=1: rho 0.0-1.0   p=2: rho 0.2-0.8   p=3: rho 0.2-1.0
# Writes straight into results/ and logs/ using the same plain naming as every
# other N (N=8_layers=<p>_presolve_<on|off>.csv) -- no separate tag/directory,
# no merge step needed. Confirmed no existing N=8_layers=* files there to
# collide with.

REPO_DIR="/local/scratch/a/rames102/QP_Rigetti/Gurobi_QP"
ENV_NAME="gurobi_qp"
GUROBI_MODULE="gurobi/13.0"

N=8
THREADS=8
MIP_GAP="1e-6"

cd "$REPO_DIR"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
module load "$GUROBI_MODULE"

mkdir -p results logs

FAILED=()

run_one() {
  local layers="$1" pen_min="$2" pen_max="$3" mode="$4"
  local tag="N=${N}_layers=${layers}_presolve_${mode}"
  local args=(
    --n "$N"
    --layers "$layers"
    --all-seeds
    --threads "$THREADS"
    --mip-gap "$MIP_GAP"
    --pen-min "$pen_min"
    --pen-max "$pen_max"
    --show-logs
    --log-dir "logs/${tag}"
    --out "results/${tag}.csv"
    --trajectory-out "results/${tag}_trajectories.csv"
  )
  if [ "$mode" = "off" ]; then
    args+=(--no-presolve)
  fi

  echo "=== ${tag} (rho ${pen_min}-${pen_max}) ==="
  if python3 scripts/run_experiment.py "${args[@]}"; then
    echo "--- OK: ${tag} ---"
  else
    echo "!!! FAILED: ${tag} !!!"
    FAILED+=("${tag}")
  fi
}

# layers, pen_min, pen_max
COMBOS=(
  "1 0.0 1.0"
  "2 0.2 0.8"
  "3 0.2 1.0"
)

for combo in "${COMBOS[@]}"; do
  read -r layers pen_min pen_max <<< "$combo"
  for MODE in on off; do
    run_one "$layers" "$pen_min" "$pen_max" "$MODE"
  done
done

echo ""
echo "Done."
if [ "${#FAILED[@]}" -eq 0 ]; then
  echo "All combos succeeded."
else
  echo "${#FAILED[@]} combo(s) failed:"
  printf '  %s\n' "${FAILED[@]}"
fi
