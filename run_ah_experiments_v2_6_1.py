#!/usr/bin/env python3
"""A-H reference task generator for the SHA-locked v2.6.1 candidate.

The default path uses the full 384-neuron model. ``--smoke`` is explicitly
non-scientific and exists only to exercise serialization and analysis paths.
Every task is independently checkpointed, seed-yoked across conditions, and
safe to resume.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import platform
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import dual_timescale_spinal_cpg_v2_6_1_candidate as model


STAGES = tuple("ABCDEFGH")
PREREGISTERED_SEEDS = tuple(range(601, 733))
VALIDATION_SEEDS = tuple(range(401, 421))
PRODUCTION_STAGE_SEEDS = {
    **{stage: VALIDATION_SEEDS for stage in "ABCDEF"},
    **{stage: PREREGISTERED_SEEDS for stage in "GH"},
}
PULSE_TARGET_SIDE = "R"
PULSE_TARGET_PHASE = "F"
SINGLE_ABLATIONS = (
    "V1Ia", "V1Ren", "V2b", "V2a", "V0D", "V0V", "V3",
    "Ia", "Ib", "groupI",
)
DOUBLE_ABLATIONS = (
    ("V0D", "V0V"),
    ("V1Ia", "V2b"),
    ("V2a", "V0V"),
    ("V1Ren", "V2b"),
    ("V3", "Ia"),
    ("V3", "Ib"),
)
RECRUITMENT_ABLATIONS = ((), ("V0D",), ("V0V",), ("V2a",), ("V1Ia",), ("V0D", "V0V"))
ABLATION_ROUTE_PAIRS = tuple(product(model.CLASSES, model.MT_ROUTES))

STAGE_DESCRIPTIONS = {
    "A": "intact multi-context robustness",
    "B": "single population/afferent ablations",
    "C": "prespecified double ablations for route degeneracy",
    "D": "external construct check of speed-dependent interneuron recruitment",
    "E": "selective impairment of each MT slow-replenishment route",
    "F": "full population-by-presynaptic-MT-route 2x2 specificity matrix",
    "G": "long demand to exogenous vesicle-depletion challenge and functional recovery",
    "H": "fast-KCa and MT mechanism falsification controls",
}


@dataclass(frozen=True)
class Task:
    stage: str
    seed: int
    protocol: str = "pulse"
    speed: str = "medium"
    load: str = "normal"
    load_side: str = "L"
    pulse: str = "excitatory"
    ablations: Tuple[str, ...] = ()
    mt_mode: str = "dynamic"
    impaired_mt_routes: Tuple[str, ...] = ()
    challenged_routes: Tuple[str, ...] = model.MT_ROUTES
    fast_mode: str = "dynamic"
    label: str = ""

    def identity(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def task_id(self) -> str:
        payload = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))
        return f"{self.stage}-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def contexts() -> Iterable[Tuple[str, str, str]]:
    for speed in model.SPEED_LEVELS:
        for load in model.LOAD_CONTEXTS:
            for pulse in model.PULSE_DIRECTIONS:
                yield speed, load, pulse


def tasks_for_seed(stage: str, seed: int) -> List[Task]:
    tasks: List[Task] = []
    if stage == "A":
        for speed, load, pulse in contexts():
            tasks.append(Task(stage, seed, speed=speed, load=load, pulse=pulse, label="intact"))
    elif stage == "B":
        for ablation in SINGLE_ABLATIONS:
            for speed, load, pulse in contexts():
                tasks.append(Task(
                    stage, seed, speed=speed, load=load, pulse=pulse,
                    ablations=(ablation,), label=f"ablate_{ablation}",
                ))
    elif stage == "C":
        for pair in DOUBLE_ABLATIONS:
            for speed, load, pulse in contexts():
                tasks.append(Task(
                    stage, seed, speed=speed, load=load, pulse=pulse,
                    ablations=pair, label=f"ablate_{'+'.join(pair)}",
                ))
    elif stage == "D":
        for ablations in RECRUITMENT_ABLATIONS:
            for speed in model.SPEED_LEVELS:
                for pulse in model.PULSE_DIRECTIONS:
                    name = "intact" if not ablations else "+".join(ablations)
                    tasks.append(Task(
                        stage, seed, speed=speed, pulse=pulse,
                        ablations=ablations, label=f"recruitment_{name}",
                    ))
    elif stage == "E":
        for route in model.MT_ROUTES:
            for speed, load, pulse in contexts():
                tasks.append(Task(
                    stage, seed, speed=speed, load=load, pulse=pulse,
                    impaired_mt_routes=(route,), label=f"impair_M_{route}",
                ))
    elif stage == "F":
        for ablation, route in ABLATION_ROUTE_PAIRS:
            for ablation_on in (False, True):
                for mt_on in (False, True):
                    for speed, load, pulse in contexts():
                        tasks.append(Task(
                            stage, seed, speed=speed, load=load, pulse=pulse,
                            ablations=(ablation,) if ablation_on else (),
                            impaired_mt_routes=(route,) if mt_on else (),
                            label=(f"factorial_{ablation}_M_{route}_"
                                   f"A{int(ablation_on)}M{int(mt_on)}"),
                        ))
    elif stage == "G":
        tasks.append(Task(
            stage, seed, protocol="long", pulse="none",
            challenged_routes=(), label="long_no_challenge",
        ))
        for route in model.MT_ROUTES:
            tasks.append(Task(
                stage, seed, protocol="long", pulse="none",
                challenged_routes=(), impaired_mt_routes=(route,),
                label=f"long_no_challenge_impaired_{route}",
            ))
            tasks.append(Task(
                stage, seed, protocol="long", pulse="none",
                challenged_routes=(route,), label=f"long_challenge_{route}",
            ))
            tasks.append(Task(
                stage, seed, protocol="long", pulse="none",
                challenged_routes=(route,), impaired_mt_routes=(route,),
                label=f"long_challenge_impaired_{route}",
            ))
    elif stage == "H":
        for fast_mode in ("dynamic", "static_mean", "yoked", "off"):
            for mt_mode in model.MT_MODES:
                for pulse in model.PULSE_DIRECTIONS:
                    tasks.append(Task(
                        stage, seed, pulse=pulse, fast_mode=fast_mode,
                        mt_mode=mt_mode,
                        label=f"control_KCa_{fast_mode}_MT_{mt_mode}",
                    ))
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return tasks


def build_tasks(stages: Sequence[str], seeds: Sequence[int]) -> List[Task]:
    return [task for seed in seeds for stage in stages for task in tasks_for_seed(stage, seed)]


def build_production_tasks(stages: Sequence[str]) -> List[Task]:
    """Build the frozen split-seed production design.

    A--F are construct/architecture validation and use 20 held-out validation
    seeds. G--H contain the ten primary families and retain all 132 powered
    seed pairs. The split makes the full 10x10 specificity matrix feasible
    without weakening or reusing the primary inferential units.
    """
    return [
        task
        for stage in stages
        for seed in PRODUCTION_STAGE_SEEDS[stage]
        for task in tasks_for_seed(stage, seed)
    ]


def config_for_task(task: Task, smoke: bool) -> model.Config:
    base = model.Config()
    if task.protocol == "long":
        base = replace(
            base,
            duration_s=base.long_n_epochs * base.long_epoch_duration_s,
            burn_in_s=base.long_epoch_duration_s,
        )
    if not smoke:
        return base
    if task.protocol == "long":
        return replace(
            base,
            dt_ms=0.5,
            long_epoch_duration_s=0.20,
            duration_s=base.long_n_epochs * 0.20,
            burn_in_s=0.20,
            rg_neurons=4,
            pf_neurons=3,
            relay_neurons=3,
            mn_neurons=3,
        )
    return replace(
        base,
        dt_ms=0.5,
        duration_s=5.2,
        burn_in_s=0.8,
        perturbation_start_s=4.0,
        perturbation_end_s=4.75,
        pulse_arm_after_s=1.6,
        rg_neurons=4,
        pf_neurons=3,
        relay_neurons=3,
        mn_neurons=3,
    )


def run_task(task: Task, output_dir: Path, smoke: bool, save_trace: bool) -> Mapping[str, object]:
    cfg = config_for_task(task, smoke)
    structural_seed = 160000 + task.seed
    external_kca_event_times_s = None
    external_kca_event_neurons = None
    external_mt_event_times_s = None
    external_mt_event_edges = None
    common_simulation_arguments = dict(
        structural_seed=structural_seed,
        speed_level=task.speed,
        load_context=task.load,
        load_side=task.load_side,
        pulse_direction=task.pulse,
        pulse_target_side=PULSE_TARGET_SIDE,
        pulse_target_phase=PULSE_TARGET_PHASE,
        ablated_populations=task.ablations,
        impaired_mt_routes=task.impaired_mt_routes,
        challenged_routes=task.challenged_routes,
    )
    if task.mt_mode == "time_yoked":
        mt_reference_fast_mode = (
            "dynamic" if task.fast_mode == "yoked" else task.fast_mode
        )
        mt_reference_trace = model.simulate(
            cfg,
            task.seed,
            task.protocol,
            "dynamic",
            fast_mode=mt_reference_fast_mode,
            **common_simulation_arguments,
        )
        external_mt_event_times_s, external_mt_event_edges = (
            model.build_yoked_mt_event_times(mt_reference_trace, cfg, task.seed)
        )
    if task.fast_mode == "yoked":
        reference_trace = model.simulate(
            cfg,
            task.seed,
            task.protocol,
            task.mt_mode,
            fast_mode="dynamic",
            external_mt_event_times_s=external_mt_event_times_s,
            external_mt_event_edges=external_mt_event_edges,
            **common_simulation_arguments,
        )
        external_kca_event_times_s, external_kca_event_neurons = (
            model.build_yoked_kca_event_times(reference_trace, cfg, task.seed)
        )
    trace = model.simulate(
        cfg,
        task.seed,
        task.protocol,
        task.mt_mode,
        fast_mode=task.fast_mode,
        external_kca_event_times_s=external_kca_event_times_s,
        external_kca_event_neurons=external_kca_event_neurons,
        external_mt_event_times_s=external_mt_event_times_s,
        external_mt_event_edges=external_mt_event_edges,
        **common_simulation_arguments,
    )
    summary = model.summarize(trace, cfg)
    if summary.get("technical_valid") != 1:
        raise RuntimeError(
            "simulation failed technical validity; no canonical task checkpoint "
            f"will be written ({summary.get('technical_exclusion_reason')})"
        )
    record: Dict[str, object] = {
        "task_id": task.task_id,
        "scientific_valid": not smoke,
        "task": task.identity(),
        "summary": summary,
        "analysis_events": model.analysis_event_payload(trace, cfg),
        "analysis_observables": model.analysis_observable_payload(trace, cfg),
        "intervention_log": model.intervention_log_payload(trace, cfg),
        "manifest": model.simulation_manifest(trace, cfg),
    }
    if task.protocol == "long":
        record["epochs"] = model.summarize_long_epochs(trace, cfg)
    task_dir = output_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    final_path = task_dir / f"{task.task_id}.json"
    temp_path = task_dir / f".{task.task_id}.{os.getpid()}.tmp"
    temp_path.write_text(
        json.dumps(model.json_safe(record), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp_path.replace(final_path)
    if save_trace:
        trace_dir = output_dir / "traces"
        trace_dir.mkdir(exist_ok=True)
        np.savez_compressed(trace_dir / f"{task.task_id}.npz", **trace)
    return record


def flatten_record(record: Mapping[str, object]) -> Dict[str, object]:
    task = dict(record["task"])
    summary = dict(record["summary"])
    row: Dict[str, object] = {
        "task_id": record["task_id"],
        "scientific_valid": record["scientific_valid"],
        **task,
        **summary,
    }
    for key in ("ablations", "impaired_mt_routes", "challenged_routes"):
        row[key] = "+".join(row[key]) if row[key] else "none"
    return row


def compile_outputs(output_dir: Path) -> None:
    records = []
    for path in sorted((output_dir / "tasks").glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    rows = [flatten_record(record) for record in records]
    if rows:
        fieldnames: List[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    epochs = []
    for record in records:
        for epoch in record.get("epochs", []):
            epochs.append({"task_id": record["task_id"], **epoch})
    if epochs:
        fieldnames = list(dict.fromkeys(key for row in epochs for key in row))
        with (output_dir / "long_epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(epochs)


def parse_csv(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def write_plan(
    output_dir: Path,
    tasks: Sequence[Task],
    stages: Sequence[str],
    seeds: Sequence[int],
    smoke: bool,
    seed_profile: str = "explicit",
) -> None:
    counts = {stage: sum(task.stage == stage for task in tasks) for stage in stages}
    identity_hash = hashlib.sha256()
    identity_rows = [] if len(tasks) <= 5000 else None
    for task in tasks:
        row = {"task_id": task.task_id, **task.identity()}
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"))
        identity_hash.update(encoded.encode())
        identity_hash.update(b"\n")
        if identity_rows is not None:
            identity_rows.append(row)
    seeds_by_stage = {
        stage: sorted({task.seed for task in tasks if task.stage == stage})
        for stage in stages
    }
    scientific_valid = not smoke and seed_profile == "production_split"
    plan = {
        "model_version": model.MODEL_VERSION,
        "scientific_valid": scientific_valid,
        "smoke_warning": "Technical execution only; never analyze as scientific data." if smoke else None,
        "stages": {stage: STAGE_DESCRIPTIONS[stage] for stage in stages},
        "seeds": list(seeds),
        "seeds_by_stage": seeds_by_stage,
        "seed_profile": seed_profile,
        "primary_seed_range_inclusive": [601, 732],
        "construct_validation_seed_range_inclusive": [401, 420],
        "paired_structural_seed_rule": "160000 + noise_seed",
        "task_count_by_stage": counts,
        "total_task_count": len(tasks),
        "task_identity_sha256": identity_hash.hexdigest(),
        "task_identities": identity_rows,
        "task_identity_note": (
            "omitted from plan JSON above 5000 tasks; deterministically generated "
            "by this frozen runner and committed by task_identity_sha256"
            if len(tasks) > 5000 else "included"
        ),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_plan.json").write_text(
        json.dumps(model.json_safe(plan), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="A", help="A-H, comma separated, or all")
    parser.add_argument(
        "--seeds",
        default="production",
        help=(
            "'production' uses validation seeds 401-420 for A-F and primary "
            "seeds 601-732 for G-H; explicit comma-separated seeds are "
            "development-only"
        ),
    )
    parser.add_argument("--outdir", default="ah_results")
    parser.add_argument("--dry-run", action="store_true", help="write the plan only")
    parser.add_argument("--smoke", action="store_true", help="non-scientific reduced-size execution")
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="development-only task cap")
    args = parser.parse_args()

    requested = STAGES if args.stage.lower() == "all" else tuple(part.upper() for part in parse_csv(args.stage))
    if not requested or any(stage not in STAGES for stage in requested):
        parser.error("--stage must contain A-H or 'all'")
    production_profile = args.seeds.lower() == "production"
    if production_profile:
        seeds = tuple(sorted({
            seed for stage in requested for seed in PRODUCTION_STAGE_SEEDS[stage]
        }))
    else:
        try:
            seeds = tuple(int(value) for value in parse_csv(args.seeds))
        except ValueError as exc:
            parser.error(f"invalid seed: {exc}")
        if not seeds or len(set(seeds)) != len(seeds):
            parser.error("--seeds must be 'production' or a non-empty unique integer list")
    if not args.smoke and not args.dry_run:
        parser.error(
            "reference CLI cannot execute scientific production; use "
            "run_ah_experiments_accelerated.py with the frozen single-run GO guard"
        )

    output_dir = Path(args.outdir).expanduser().resolve()
    tasks = (
        build_production_tasks(requested)
        if production_profile else build_tasks(requested, seeds)
    )
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        tasks = tasks[:args.limit]
    write_plan(
        output_dir, tasks, requested, seeds, args.smoke,
        "production_split" if production_profile else "explicit",
    )
    counts = {stage: sum(task.stage == stage for task in tasks) for stage in requested}
    print(json.dumps({"tasks": len(tasks), "by_stage": counts, "outdir": str(output_dir)}, indent=2))
    if args.dry_run:
        return

    completed = {path.stem for path in (output_dir / "tasks").glob("*.json")} if (output_dir / "tasks").exists() else set()
    for index, task in enumerate(tasks, start=1):
        if task.task_id in completed:
            print(f"[{index}/{len(tasks)}] resume-skip {task.task_id}")
            continue
        print(f"[{index}/{len(tasks)}] run {task.task_id} {task.label}", flush=True)
        run_task(task, output_dir, args.smoke, args.save_trace)
    compile_outputs(output_dir)


if __name__ == "__main__":
    main()
