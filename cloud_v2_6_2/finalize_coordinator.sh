#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 MANIFEST_PATH" >&2
  exit 2
fi

manifest_path="$1"
release_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$release_dir"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0

.venv/bin/python distributed_single_realization_v2_6_2.py merge \
  --manifest "$manifest_path"

echo "verified_merge_preflight_single_realization_analysis=PASS"
