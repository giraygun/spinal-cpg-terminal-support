#!/usr/bin/env python3
"""Run the complete A-H design in one frozen stochastic realization.

This execution overlay changes only the seed registry.  It imports the
SHA-locked v2.6.1 biological model and normative task generator unchanged.
All stages, classes, routes, factorial arms, contexts and mechanistic controls
remain present.  Results are conditional on seed 601 and cannot estimate
between-realization uncertainty.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import time
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

import dual_timescale_spinal_cpg_v2_6_1_candidate as model
import run_ah_experiments_v2_6_1 as reference
import run_ah_experiments_accelerated_v2_6_1 as accelerated


ROOT = Path(__file__).resolve().parent
RUNNER_VERSION = "single-realization-runner-2.6.2"
DESIGN_TYPE = "frozen_single_realization_paired_mechanism_test"
FIXED_SEED = 601
FIXED_STRUCTURAL_SEED = 160_601
EXPECTED_ANALYSIS_TASK_COUNT = 11_686
EXPECTED_UNIQUE_SIMULATION_COUNT = 3_610
EXPECTED_AVOIDED_RECOMPUTATIONS = 8_076
EXPECTED_TASK_IDENTITY_SHA256 = (
    "30857ac18a7944a18cc022270428758a91fff49721ffec152f12edc78fdff4bc"
)
EXPECTED_TASKS_BY_STAGE = {
    "A": 27,
    "B": 270,
    "C": 162,
    "D": 54,
    "E": 270,
    "F": 10_800,
    "G": 31,
    "H": 72,
}
EXPECTED_UNIQUE_WITHIN_STAGE = {
    "A": 27,
    "B": 270,
    "C": 162,
    "D": 54,
    "E": 270,
    "F": 3_267,
    "G": 31,
    "H": 72,
}
RELEASE_PATH = ROOT / "SINGLE_REALIZATION_RELEASE_v2_6_2.json"


def build_matrix() -> tuple[
    list[reference.Task],
    Dict[str, reference.Task],
    Dict[str, str],
    Counter,
]:
    tasks = reference.build_tasks(reference.STAGES, (FIXED_SEED,))
    representatives, task_to_sim, multiplicity = accelerated.unique_simulations(
        tasks, smoke=False
    )
    return tasks, representatives, task_to_sim, multiplicity


def matrix_contract(
    tasks: Sequence[reference.Task],
    representatives: Mapping[str, reference.Task],
) -> Dict[str, object]:
    return {
        "fixed_seed": FIXED_SEED,
        "fixed_structural_seed": FIXED_STRUCTURAL_SEED,
        "analysis_task_count": len(tasks),
        "unique_simulation_count": len(representatives),
        "avoided_identical_recomputations": len(tasks) - len(representatives),
        "task_identity_sha256": accelerated.task_identity_sha256(tasks),
        "task_count_by_stage": {
            stage: sum(task.stage == stage for task in tasks)
            for stage in reference.STAGES
        },
        "unique_simulations_within_stage": {
            stage: len({
                accelerated.simulation_id(task, smoke=False)
                for task in tasks if task.stage == stage
            })
            for stage in reference.STAGES
        },
    }


def expected_matrix_contract() -> Dict[str, object]:
    return {
        "fixed_seed": FIXED_SEED,
        "fixed_structural_seed": FIXED_STRUCTURAL_SEED,
        "analysis_task_count": EXPECTED_ANALYSIS_TASK_COUNT,
        "unique_simulation_count": EXPECTED_UNIQUE_SIMULATION_COUNT,
        "avoided_identical_recomputations": EXPECTED_AVOIDED_RECOMPUTATIONS,
        "task_identity_sha256": EXPECTED_TASK_IDENTITY_SHA256,
        "task_count_by_stage": EXPECTED_TASKS_BY_STAGE,
        "unique_simulations_within_stage": EXPECTED_UNIQUE_WITHIN_STAGE,
    }


def assert_matrix_contract(
    tasks: Sequence[reference.Task],
    representatives: Mapping[str, reference.Task],
) -> Dict[str, object]:
    actual = matrix_contract(tasks, representatives)
    expected = expected_matrix_contract()
    if actual != expected:
        raise RuntimeError(
            "single-realization task matrix drifted:\n"
            + json.dumps({"actual": actual, "expected": expected}, indent=2)
        )
    if any(task.seed != FIXED_SEED for task in tasks):
        raise RuntimeError("task matrix contains a seed other than 601")
    return actual


def assert_release_contract() -> Dict[str, object]:
    release = accelerated.load_json_contract(
        RELEASE_PATH, "single-realization release contract"
    )
    expected = {
        "release_status": "GO_SINGLE_REALIZATION_V2_6_2",
        "design_type": DESIGN_TYPE,
        "model_version": accelerated.EXPECTED_MODEL_VERSION,
        "model_sha256": accelerated.EXPECTED_MODEL_SHA256,
        **expected_matrix_contract(),
        "model_equations_or_parameters_changed": False,
        "stochastic_population_inference_authorized": False,
    }
    if release != expected:
        raise RuntimeError("single-realization release contract is not exact")
    return release


def execution_plan(matrix: Mapping[str, object]) -> Dict[str, object]:
    return {
        "schema": "single-realization-experiment-plan-1.0",
        "runner_version": RUNNER_VERSION,
        "design_type": DESIGN_TYPE,
        "scientific_valid": True,
        "inferential_scope": "conditional_on_one_frozen_network_realization",
        "stochastic_population_inference_authorized": False,
        "model_version": accelerated.EXPECTED_MODEL_VERSION,
        "model_sha256": accelerated.EXPECTED_MODEL_SHA256,
        "reference_runner_sha256": accelerated.sha256_file(
            Path(reference.__file__).resolve()
        ),
        "execution_engine_sha256": accelerated.EXPECTED_EXECUTION_ENGINE_SHA256,
        "single_realization_runner_sha256": accelerated.sha256_file(
            Path(__file__).resolve()
        ),
        "requirements_lock_sha256": accelerated.sha256_file(
            ROOT / "requirements-lock.txt"
        ),
        "stages": {
            stage: reference.STAGE_DESCRIPTIONS[stage]
            for stage in reference.STAGES
        },
        "seeds_by_stage": {
            stage: [FIXED_SEED] for stage in reference.STAGES
        },
        **dict(matrix),
        "preserved_axes": {
            "neuron_or_afferent_classes": list(model.CLASSES),
            "presynaptic_mt_routes": list(model.MT_ROUTES),
            "factorial_arms_per_class_route_pair": 4,
            "contexts_per_factorial_arm": 27,
        },
        "pairing_contract": (
            "Every compared condition uses seed 601, structural seed 160601, "
            "and the simulator's deterministic yoked/random-stream rules."
        ),
        "statistical_unit_warning": (
            "Task rows, routes, cells and contexts are not independent stochastic "
            "replicates. This run supplies one conditional mechanistic realization."
        ),
        "model_equations_or_parameters_changed": False,
        "task_axes_other_than_seed_changed": False,
    }


def write_or_verify_plan(output_dir: Path, plan: Mapping[str, object]) -> None:
    plan_path = output_dir / "experiment_plan_single_realization_v2_6_2.json"
    if plan_path.exists():
        existing = accelerated.load_json_contract(plan_path, "experiment plan")
        if existing != plan:
            raise RuntimeError(
                "existing output directory belongs to a different immutable plan"
            )
        return
    accelerated.write_json_atomic(plan_path, plan)


def automatic_worker_count() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(8, cpu - 1 if cpu > 2 else cpu))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="single_realization_results_v2_6_2")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.dry_run and args.compile_only:
        parser.error("--dry-run and --compile-only cannot be combined")

    accelerated.assert_frozen_model_identity()
    assert_release_contract()
    tasks, representatives, task_to_sim, multiplicity = build_matrix()
    matrix = assert_matrix_contract(tasks, representatives)
    plan = execution_plan(matrix)

    output_dir = Path(args.outdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_or_verify_plan(output_dir, plan)
    accelerated.write_task_index(output_dir, tasks, task_to_sim, multiplicity)

    completed = accelerated.existing_simulations(output_dir)
    expected_sids = set(representatives)
    extra = completed - expected_sids
    if extra:
        raise RuntimeError(
            f"output directory contains {len(extra)} unrelated valid checkpoints"
        )
    pending = [
        (sid, task) for sid, task in representatives.items()
        if sid not in completed
    ]
    workers = args.workers or automatic_worker_count()
    status = {
        **matrix,
        "already_completed": len(expected_sids) - len(pending),
        "pending": len(pending),
        "workers": workers,
        "outdir": str(output_dir),
        "inferential_scope": "conditional_on_one_frozen_network_realization",
    }
    print(json.dumps(status, indent=2), flush=True)
    if args.dry_run:
        return
    if args.compile_only:
        if pending:
            raise RuntimeError(
                f"cannot compile: {len(pending)} canonical simulations are missing"
            )
        accelerated.compile_outputs(output_dir, tasks, task_to_sim, multiplicity)
        return

    started = time.perf_counter()
    durations = accelerated.run_pending_parallel(
        pending,
        output_dir,
        False,
        args.save_trace,
        workers,
        args.progress_every,
    )
    if not args.no_compile:
        accelerated.compile_outputs(output_dir, tasks, task_to_sim, multiplicity)
    completion = {
        "runner_version": RUNNER_VERSION,
        "design_type": DESIGN_TYPE,
        "scientific_valid": True,
        "inferential_scope": "conditional_on_one_frozen_network_realization",
        "stochastic_population_inference_authorized": False,
        **matrix,
        "workers": workers,
        "wall_elapsed_s_this_invocation": time.perf_counter() - started,
        "worker_task_elapsed_s_median": (
            float(np.median(durations)) if durations else None
        ),
        "completed_checkpoint_count": len(
            accelerated.existing_simulations(output_dir)
        ),
    }
    accelerated.write_json_atomic(
        output_dir / "completion_single_realization_v2_6_2.json", completion
    )


if __name__ == "__main__":
    main()
