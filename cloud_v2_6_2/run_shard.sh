#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: $0 SHARD_INDEX MANIFEST_PATH [WORKERS]" >&2
  exit 2
fi

shard_index="$1"
manifest_path="$2"
workers="${3:-7}"
release_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$release_dir"

if [[ ! "$shard_index" =~ ^[0-2]$ ]]; then
  echo "ERROR: SHARD_INDEX must be 0, 1, or 2" >&2
  exit 2
fi
if [[ ! "$workers" =~ ^[1-8]$ ]]; then
  echo "ERROR: WORKERS must be between 1 and 8" >&2
  exit 2
fi
if [[ ! -f "$manifest_path" ]]; then
  echo "ERROR: manifest not found: $manifest_path" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0

production_root="$(.venv/bin/python - "$manifest_path" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["production_root"])
PY
)"
mkdir -p "$production_root/logs"
log_path="$production_root/logs/shard-${shard_index}.log"
pid_path="$production_root/logs/shard-${shard_index}.pid"

nohup .venv/bin/python distributed_single_realization_v2_6_2.py run-shard \
  --manifest "$manifest_path" \
  --index "$shard_index" \
  --workers "$workers" \
  --progress-every 25 \
  >"$log_path" 2>&1 &
pid="$!"
printf '%s\n' "$pid" >"$pid_path"

echo "shard=$shard_index pid=$pid workers=$workers"
echo "log=$log_path"
echo "monitor: tail -f $log_path"
