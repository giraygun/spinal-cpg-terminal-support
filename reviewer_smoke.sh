#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_ROOT="${1:-derived/reviewer_smoke}"
DRYRUN_DIR="${OUTPUT_ROOT}_matrix"
NUMERIC_DIR="${OUTPUT_ROOT}_numeric"

for target in "$DRYRUN_DIR" "$NUMERIC_DIR"; do
  if [[ -e "$target" ]]; then
    echo "Refusing existing reviewer output: $target" >&2
    exit 73
  fi
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cpg-v2-6-2-matplotlib}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 reviewer_verify.py
python3 -B -m unittest -v test_single_realization_v2_6_2.py
python3 -B test_numerics_architecture_v2_6_1.py

python3 run_single_realization_v2_6_2.py \
  --dry-run \
  --outdir "$DRYRUN_DIR" \
  --workers 1

python3 run_ah_experiments_accelerated_v2_6_1.py \
  --stage A \
  --seeds 601 \
  --outdir "$NUMERIC_DIR" \
  --workers 1 \
  --smoke \
  --limit 1 \
  --progress-every 1

echo "reviewer_smoke=PASS"
echo "matrix_dryrun=$DRYRUN_DIR"
echo "numeric_smoke=$NUMERIC_DIR"
