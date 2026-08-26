#!/usr/bin/env bash
set -euo pipefail

release_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$release_dir"

python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$python_version" != "3.12" ]]; then
  echo "ERROR: CPython 3.12.x required; found $python_version" >&2
  exit 1
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3-venv python3-pip tmux pigz

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

.venv/bin/python - <<'PY'
import run_ah_experiments_accelerated_v2_6_1 as frozen
import run_single_realization_v2_6_2 as single

frozen.assert_frozen_model_identity()
single.assert_release_contract()
tasks, representatives, _, _ = single.build_matrix()
single.assert_matrix_contract(tasks, representatives)
print("frozen_model_single_realization_release_and_matrix=PASS")
PY

echo "bootstrap=PASS"
