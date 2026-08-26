#!/usr/bin/env python3
"""Fail-closed postrun audit for the v2.6.2 single-realization design."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Mapping

import run_ah_experiments_accelerated_v2_6_1 as accelerated
import run_single_realization_v2_6_2 as single


REPORT_NAME = "postrun_preflight_single_realization_v2_6_2.json"


def _check_task_index(
    output_dir: Path,
    tasks,
    task_to_sim: Mapping[str, str],
    multiplicity: Mapping[str, int],
) -> None:
    path = output_dir / "analysis_task_index.csv"
    if not path.is_file():
        raise RuntimeError("analysis_task_index.csv is missing")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != single.EXPECTED_ANALYSIS_TASK_COUNT:
        raise RuntimeError("analysis task index row count is not 11,686")
    expected = {task.task_id: task for task in tasks}
    if len({row.get("task_id") for row in rows}) != len(rows):
        raise RuntimeError("analysis task index has duplicate task IDs")
    for row in rows:
        task_id = row.get("task_id")
        if task_id not in expected:
            raise RuntimeError(f"unknown task ID in analysis index: {task_id}")
        task = expected[task_id]
        sid = task_to_sim[task_id]
        if row.get("simulation_id") != sid:
            raise RuntimeError(f"task-to-simulation mapping drifted: {task_id}")
        if row.get("seed") != str(single.FIXED_SEED):
            raise RuntimeError(f"non-601 seed in analysis index: {task_id}")
        if row.get("stage") != task.stage:
            raise RuntimeError(f"stage mismatch in analysis index: {task_id}")
        if row.get("reuse_count") != str(multiplicity[sid]):
            raise RuntimeError(f"reuse count mismatch in analysis index: {task_id}")


def _check_plan(output_dir: Path, matrix: Mapping[str, object]) -> None:
    path = output_dir / "experiment_plan_single_realization_v2_6_2.json"
    if not path.is_file():
        raise RuntimeError("immutable single-realization experiment plan is missing")
    actual = accelerated.load_json_contract(path, "single-realization plan")
    expected = single.execution_plan(matrix)
    if actual != expected:
        raise RuntimeError("single-realization experiment plan drifted")


def _check_checkpoints(
    output_dir: Path,
    representatives: Mapping[str, object],
) -> None:
    simulation_dir = output_dir / "simulations"
    if not simulation_dir.is_dir():
        raise RuntimeError("simulations directory is missing")
    expected = set(representatives)
    actual = {path.stem for path in simulation_dir.glob("sim-*.json")}
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise RuntimeError(
            f"checkpoint set mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    for sid in sorted(expected):
        path = simulation_dir / f"{sid}.json"
        accelerated.validate_checkpoint(path, sid)
        record = json.loads(path.read_text(encoding="utf-8"))
        identity = record.get("simulation_identity")
        if not isinstance(identity, dict):
            raise RuntimeError(f"checkpoint identity missing: {sid}")
        if identity.get("seed") != single.FIXED_SEED:
            raise RuntimeError(f"checkpoint seed is not 601: {sid}")
        if identity.get("structural_seed") != single.FIXED_STRUCTURAL_SEED:
            raise RuntimeError(f"checkpoint structural seed is not 160601: {sid}")
        if record.get("scientific_valid") is not True:
            raise RuntimeError(f"checkpoint is not full-resolution scientific: {sid}")


def run_preflight(output_dir: Path) -> Dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    checks: Dict[str, Dict[str, object]] = {}

    def execute(name: str, function) -> None:
        try:
            function()
        except Exception as exc:  # report all independent gates in one artifact
            checks[name] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            checks[name] = {"pass": True}

    tasks, representatives, task_to_sim, multiplicity = single.build_matrix()
    matrix: Dict[str, object] = {}

    def model_and_release() -> None:
        accelerated.assert_frozen_model_identity()
        single.assert_release_contract()

    def matrix_check() -> None:
        nonlocal matrix
        matrix = single.assert_matrix_contract(tasks, representatives)

    execute("frozen_model_and_release_contract", model_and_release)
    execute("task_matrix_identity", matrix_check)
    if not matrix:
        matrix = single.matrix_contract(tasks, representatives)
    execute("immutable_experiment_plan", lambda: _check_plan(output_dir, matrix))
    execute(
        "analysis_task_index",
        lambda: _check_task_index(output_dir, tasks, task_to_sim, multiplicity),
    )
    execute(
        "complete_exact_checkpoint_set",
        lambda: _check_checkpoints(output_dir, representatives),
    )
    result = {
        "preflight_version": "single-realization-preflight-2.6.2",
        "design_type": single.DESIGN_TYPE,
        "inferential_scope": "conditional_on_one_frozen_network_realization",
        "stochastic_population_inference_authorized": False,
        **matrix,
        "checks": checks,
        "all_checks_pass": all(item["pass"] for item in checks.values()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerated.write_json_atomic(output_dir / REPORT_NAME, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    args = parser.parse_args()
    result = run_preflight(Path(args.results_dir))
    print(json.dumps(result, indent=2))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
