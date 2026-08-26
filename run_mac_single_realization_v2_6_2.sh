#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if (( $# > 1 )); then
  echo "Kullanim: $0 [sonuc_dizini]" >&2
  exit 64
fi

RESULT_DIRECTORY="${1:-single_realization_results_v2_6_2}"
WORKER_COUNT="${CPG_WORKERS:-6}"

if [[ ! "$WORKER_COUNT" =~ ^[1-9][0-9]*$ ]] || (( WORKER_COUNT > 8 )); then
  echo "CPG_WORKERS 1 ile 8 arasinda bir tamsayi olmalidir." >&2
  exit 64
fi

REQUIRED_FILES=(
  SINGLE_REALIZATION_RELEASE_v2_6_2.json
  requirements-lock.txt
  dual_timescale_spinal_cpg_v2_6_1_candidate.py
  run_ah_experiments_v2_6_1.py
  run_ah_experiments_accelerated_v2_6_1.py
  run_single_realization_v2_6_2.py
  preflight_single_realization_v2_6_2.py
  analyze_primary_v2_6_1.py
  analyze_single_realization_v2_6_2.py
)

for required_file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Uretim kapali: zorunlu dosya eksik: $required_file" >&2
    exit 66
  fi
done

mkdir -p "$RESULT_DIRECTORY"
PID_FILE="$RESULT_DIRECTORY/mac_single_realization.pid"
if [[ -f "$PID_FILE" ]]; then
  previous_pid="$(tr -dc '0-9' < "$PID_FILE")"
  if [[ -n "$previous_pid" ]] && kill -0 "$previous_pid" 2>/dev/null; then
    echo "Kosucu zaten calisiyor (PID=$previous_pid). Ikinci kopya baslatilmadi." >&2
    exit 73
  fi
  rm -f "$PID_FILE"
fi
printf '%s\n' "$$" > "$PID_FILE"
cleanup_pid() {
  rm -f "$PID_FILE"
}
trap cleanup_pid EXIT INT TERM

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0

python3 - <<'PY'
import sys
import numpy
import scipy
import run_ah_experiments_accelerated_v2_6_1 as frozen
import run_single_realization_v2_6_2 as single

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"CPython 3.12.x gerekli; bulunan: {sys.version.split()[0]}")
if numpy.__version__ != "2.3.5" or scipy.__version__ != "1.17.0":
    raise SystemExit(
        "Bagimlilik surumu uyusmuyor: "
        f"numpy={numpy.__version__}, scipy={scipy.__version__}"
    )
frozen.assert_frozen_model_identity()
single.assert_release_contract()
tasks, representatives, _, _ = single.build_matrix()
single.assert_matrix_contract(tasks, representatives)
print("model_release_task_matrix_environment=PASS", flush=True)
PY

RUN_COMMAND=(
  python3 run_single_realization_v2_6_2.py
  --outdir "$RESULT_DIRECTORY"
  --workers "$WORKER_COUNT"
  --progress-every 10
)

echo "single_realization_seed=601"
echo "analysis_tasks=11686"
echo "unique_simulations=3610"
echo "workers=$WORKER_COUNT"
echo "results=$RESULT_DIRECTORY"

if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu "${RUN_COMMAND[@]}"
else
  "${RUN_COMMAND[@]}"
fi

python3 preflight_single_realization_v2_6_2.py "$RESULT_DIRECTORY"
python3 analyze_single_realization_v2_6_2.py "$RESULT_DIRECTORY"

echo "single_realization_run_preflight_analysis=PASS"
