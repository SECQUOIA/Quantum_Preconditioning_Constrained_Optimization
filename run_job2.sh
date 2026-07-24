#!/bin/bash
set -uo pipefail
# Note: no -e here on purpose. With 42 separate solves below, one N/layers
# combo having zero penalties in range must not kill every combo after it.
#
# Independent copy of run_job.sh, for running other problem sizes while
# run_job.sh is still busy on a different N. Same logic, same flags, same
# output layout (results/<PEN_TAG>/, logs/<PEN_TAG>/) — just a separate file
# so editing/transferring this one never touches whatever run_job.sh is
# mid-execution on.

REPO_DIR="/local/scratch/a/rames102/QP_Rigetti/Gurobi_QP"
ENV_NAME="gurobi_qp"
GUROBI_MODULE="gurobi/13.0"

THREADS=8
MIP_GAP="1e-6"
PEN_MIN="1.1"
PEN_MAX="2.0"
PEN_TAG="pen1.1-2.0"

# Pick which N's and p's (layers) to run either by editing these two lines,
# or (safer for repeated nohup launches) by passing them as arguments:
#   bash run_job2.sh <N> <layers>
# Passing args overrides both lists down to that single (N, layers) pair, so
# you never have to edit this file while another launch might still be
# reading it from disk.
N_LIST=(8 12 16 20 24)
LAYERS_LIST=(1 2)

if [ "$#" -ge 1 ]; then
  N_LIST=("$1")
fi
if [ "$#" -ge 2 ]; then
  LAYERS_LIST=("$2")
fi

cd "$REPO_DIR"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
module load "$GUROBI_MODULE"

mkdir -p "results/${PEN_TAG}" "logs/${PEN_TAG}"

FAILED=()

run_one() {
  local n="$1" layers="$2" mode="$3"
  # Same tag/naming convention as the old full-sweep runs, just nested one
  # level deeper under results/<PEN_TAG>/ and logs/<PEN_TAG>/ so this never
  # collides with (or overwrites) the existing full 0.0-2.0 sweep data.
  local tag="N=${n}_layers=${layers}_presolve_${mode}"
  # Build args as one array (never empty) so `set -u` is safe on older bash
  # (bash <4.4 treats expanding a zero-element array under nounset as an error).
  local args=(
    --n "$n"
    --layers "$layers"
    --all-seeds
    --threads "$THREADS"
    --mip-gap "$MIP_GAP"
    --pen-min "$PEN_MIN"
    --pen-max "$PEN_MAX"
    --show-logs
    --log-dir "logs/${PEN_TAG}/${tag}"
    --out "results/${PEN_TAG}/${tag}.csv"
    --trajectory-out "results/${PEN_TAG}/${tag}_trajectories.csv"
  )
  if [ "$mode" = "off" ]; then
    args+=(--no-presolve)
  fi

  echo "=== ${PEN_TAG}/${tag} ==="
  if python3 scripts/run_experiment.py "${args[@]}"; then
    echo "--- OK: ${PEN_TAG}/${tag} ---"
  else
    echo "!!! FAILED: ${PEN_TAG}/${tag} !!!"
    FAILED+=("${PEN_TAG}/${tag}")
  fi
}

for N in "${N_LIST[@]}"; do
  for LAYERS in "${LAYERS_LIST[@]}"; do
    for MODE in on off; do
      run_one "$N" "$LAYERS" "$MODE"
    done
  done
done

echo ""
echo "Done."
if [ "${#FAILED[@]}" -eq 0 ]; then
  echo "All combos succeeded."
else
  echo "${#FAILED[@]} combo(s) failed (likely no penalties in [${PEN_MIN}, ${PEN_MAX}] for that N/layers):"
  printf '  %s\n' "${FAILED[@]}"
fi
