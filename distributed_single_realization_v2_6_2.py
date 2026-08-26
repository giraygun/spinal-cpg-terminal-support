#!/usr/bin/env python3
"""Deterministic 3-VM execution and verified merge for seed 601 only."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
from typing import Dict, Mapping

import analyze_single_realization_v2_6_2 as analysis
import distributed_shards_v2_6_1 as shard_tools
import preflight_single_realization_v2_6_2 as preflight
import run_ah_experiments_accelerated_v2_6_1 as accelerated
import run_single_realization_v2_6_2 as single


SCHEMA = "weighted-lpt-single-realization-shards-v2.6.2"
DEFAULT_SHARD_COUNT = 3


def build_manifest(shard_count: int, production_root: str) -> Dict[str, object]:
    accelerated.assert_frozen_model_identity()
    single.assert_release_contract()
    tasks, representatives, _, _ = single.build_matrix()
    matrix = single.assert_matrix_contract(tasks, representatives)
    assignments, loads = shard_tools.deterministic_lpt_assignment(
        representatives, shard_count, smoke=False
    )
    counts = Counter(int(row["shard_index"]) for row in assignments)
    return {
        "schema": SCHEMA,
        "design_type": single.DESIGN_TYPE,
        "model_version": accelerated.EXPECTED_MODEL_VERSION,
        "model_sha256": accelerated.EXPECTED_MODEL_SHA256,
        "single_realization_runner_sha256": accelerated.sha256_file(
            Path(single.__file__).resolve()
        ),
        "reference_runner_sha256": accelerated.sha256_file(
            Path(single.reference.__file__).resolve()
        ),
        "production_root": str(Path(production_root).expanduser().resolve()),
        **matrix,
        "shard_count": shard_count,
        "assignment_algorithm": (
            "deterministic_lpt_by_integration_steps_times_required_model_calls"
        ),
        "shard_estimated_work_units": list(loads),
        "shard_unique_simulation_counts": [
            counts.get(index, 0) for index in range(shard_count)
        ],
        "assignments": list(assignments),
        "assignment_sha256": shard_tools.canonical_json_sha256(assignments),
        "stochastic_population_inference_authorized": False,
    }


def write_manifest(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    accelerated.write_json_atomic(path, value)


def load_and_verify(path: Path) -> Dict[str, object]:
    manifest = accelerated.load_json_contract(path, "single-realization shard manifest")
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError("single-realization shard schema mismatch")
    if manifest.get("design_type") != single.DESIGN_TYPE:
        raise RuntimeError("shard design type mismatch")
    if manifest.get("model_sha256") != accelerated.EXPECTED_MODEL_SHA256:
        raise RuntimeError("shard model SHA mismatch")
    if manifest.get("single_realization_runner_sha256") != accelerated.sha256_file(
        Path(single.__file__).resolve()
    ):
        raise RuntimeError("single-realization runner SHA mismatch")
    assignments = manifest.get("assignments")
    if not isinstance(assignments, list):
        raise RuntimeError("shard assignments are missing")
    if shard_tools.canonical_json_sha256(assignments) != manifest.get(
        "assignment_sha256"
    ):
        raise RuntimeError("shard assignment SHA mismatch")

    tasks, representatives, _, _ = single.build_matrix()
    matrix = single.assert_matrix_contract(tasks, representatives)
    for key, expected in matrix.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"shard matrix field drifted: {key}")
    by_id = {
        str(row["simulation_id"]): row
        for row in assignments if isinstance(row, dict)
    }
    if len(by_id) != len(assignments) or set(by_id) != set(representatives):
        raise RuntimeError("shard assignments do not exactly cover the workload")
    shard_count = manifest.get("shard_count")
    if not isinstance(shard_count, int) or shard_count <= 0:
        raise RuntimeError("invalid shard count")
    for sid, task in representatives.items():
        row = by_id[sid]
        if row.get("representative_task_id") != task.task_id:
            raise RuntimeError(f"representative task drift: {sid}")
        if not 0 <= int(row.get("shard_index", -1)) < shard_count:
            raise RuntimeError(f"invalid shard index: {sid}")
        if int(row.get("estimated_work_units", -1)) != (
            shard_tools.estimated_work_units(task, smoke=False)
        ):
            raise RuntimeError(f"work weight drift: {sid}")
    return manifest


def run_shard(
    manifest_path: Path,
    shard_index: int,
    workers: int,
    progress_every: int,
) -> Dict[str, object]:
    accelerated.assert_frozen_model_identity()
    single.assert_release_contract()
    manifest = load_and_verify(manifest_path)
    _, representatives, _, _ = single.build_matrix()
    shard_count = int(manifest["shard_count"])
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard index is outside the manifest")
    root = Path(str(manifest["production_root"]))
    output = root / "shards" / f"shard-{shard_index:03d}"
    output.mkdir(parents=True, exist_ok=True)
    assigned = {
        str(row["simulation_id"])
        for row in manifest["assignments"]
        if int(row["shard_index"]) == shard_index
    }
    completed = accelerated.existing_simulations(output)
    if completed - assigned:
        raise RuntimeError("shard directory contains unrelated checkpoints")
    pending = [
        (sid, representatives[sid])
        for sid in sorted(assigned - completed)
    ]
    durations = accelerated.run_pending_parallel(
        pending, output, False, False, workers, progress_every
    )
    final_ids = accelerated.existing_simulations(output)
    if final_ids != assigned:
        raise RuntimeError("shard completion set is not exact")
    result = {
        "schema": "single-realization-shard-completion-v2.6.2",
        "manifest_sha256": accelerated.sha256_file(manifest_path),
        "assignment_sha256": manifest["assignment_sha256"],
        "shard_index": shard_index,
        "assigned_simulation_count": len(assigned),
        "completed_simulation_count": len(final_ids),
        "elapsed_worker_seconds_sum_this_invocation": float(sum(durations)),
        "passes": True,
    }
    write_manifest(output / "shard_completion.json", result)
    return result


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError(f"different checkpoint at merge target: {destination.name}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def merge(manifest_path: Path) -> Dict[str, object]:
    accelerated.assert_frozen_model_identity()
    single.assert_release_contract()
    manifest = load_and_verify(manifest_path)
    tasks, representatives, task_to_sim, multiplicity = single.build_matrix()
    matrix = single.assert_matrix_contract(tasks, representatives)
    root = Path(str(manifest["production_root"]))
    merged = root / "simulations"
    merged.mkdir(parents=True, exist_ok=True)
    observed: set[str] = set()
    for index in range(int(manifest["shard_count"])):
        shard_dir = root / "shards" / f"shard-{index:03d}"
        completion = accelerated.load_json_contract(
            shard_dir / "shard_completion.json", f"shard {index} completion"
        )
        if (
            completion.get("passes") is not True
            or completion.get("assignment_sha256") != manifest["assignment_sha256"]
            or completion.get("manifest_sha256") != accelerated.sha256_file(manifest_path)
        ):
            raise RuntimeError(f"shard {index} completion contract failed")
        assigned = {
            str(row["simulation_id"])
            for row in manifest["assignments"]
            if int(row["shard_index"]) == index
        }
        actual = accelerated.existing_simulations(shard_dir)
        if actual != assigned or observed & actual:
            raise RuntimeError(f"shard {index} is incomplete or overlaps another shard")
        observed.update(actual)
        for sid in sorted(actual):
            source = shard_dir / "simulations" / f"{sid}.json"
            target = merged / source.name
            accelerated.validate_checkpoint(source, sid)
            _link_or_copy(source, target)
            accelerated.validate_checkpoint(target, sid)
    if observed != set(representatives):
        raise RuntimeError("merged checkpoint set does not cover the workload")

    single.write_or_verify_plan(root, single.execution_plan(matrix))
    accelerated.write_task_index(root, tasks, task_to_sim, multiplicity)
    accelerated.compile_outputs(root, tasks, task_to_sim, multiplicity)
    audit = preflight.run_preflight(root)
    if not audit["all_checks_pass"]:
        raise RuntimeError("post-merge single-realization preflight failed")
    result_payload = analysis.analyze(root)
    report = {
        "schema": "single-realization-verified-shard-merge-v2.6.2",
        **matrix,
        "assignment_sha256": manifest["assignment_sha256"],
        "preflight_all_checks_pass": True,
        "analysis_file": "single_realization_results_v2_6_2.json",
        "overall_interpretation": result_payload["overall_interpretation"],
        "passes": True,
    }
    write_manifest(root / "shard_merge_single_realization_v2_6_2.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--out", required=True)
    plan.add_argument("--production-root", required=True)
    plan.add_argument("--shards", type=int, default=DEFAULT_SHARD_COUNT)
    run = sub.add_parser("run-shard")
    run.add_argument("--manifest", required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--workers", type=int, required=True)
    run.add_argument("--progress-every", type=int, default=25)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    if args.command == "plan":
        value = build_manifest(args.shards, args.production_root)
        path = Path(args.out).expanduser().resolve()
        write_manifest(path, value)
        print(json.dumps({
            "manifest": str(path),
            "manifest_sha256": accelerated.sha256_file(path),
            "shard_counts": value["shard_unique_simulation_counts"],
            "shard_work_units": value["shard_estimated_work_units"],
            "assignment_sha256": value["assignment_sha256"],
        }, indent=2))
    elif args.command == "run-shard":
        print(json.dumps(run_shard(
            Path(args.manifest).expanduser().resolve(),
            args.index,
            args.workers,
            args.progress_every,
        ), indent=2))
    else:
        print(json.dumps(merge(Path(args.manifest).expanduser().resolve()), indent=2))


if __name__ == "__main__":
    main()
