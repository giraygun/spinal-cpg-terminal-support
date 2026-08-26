#!/usr/bin/env python3
"""Deduplicated, parallel A-H runner for the SHA-locked v2.6.1 model.

This runner does not alter model equations, parameters, seeds, durations,
integration step, neuron counts, perturbations, or the preregistered task
matrix. It only avoids recomputing tasks whose complete simulator inputs are
identical. One canonical simulation is stored once and referenced by every
preregistered analysis task that uses it.

``run_ah_experiments_v2_6_1.py`` remains the normative task generator
and single-task reference implementation. This file imports it rather than
copying the A-H design, preventing drift between the reference and accelerated
execution paths.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

import dual_timescale_spinal_cpg_v2_6_1_candidate as frozen_model
import dual_timescale_spinal_cpg_v2_6_1_candidate as model
import run_ah_experiments_v2_6_1 as reference


RUNNER_VERSION = "deduplicated-parallel-runner-2.6.1-candidate"
EXPECTED_MODEL_VERSION = "distributed-local-terminal-mt-cpg-2.6.1-candidate"
EXPECTED_MODEL_SHA256 = "a0dc8a7338ab1619874135b1a3e8809f4eaa22394cb65dfd951544df5b62f47a"
EXPECTED_EXECUTION_ENGINE_SHA256 = EXPECTED_MODEL_SHA256
ROOT = Path(__file__).resolve().parent
REQUIREMENTS_LOCK_PATH = ROOT / "requirements-lock.txt"
PRODUCTION_FREEZE_MANIFEST_PATH = ROOT / "FREEZE_MANIFEST_v2_6_1.json"
PRODUCTION_RUN_CLAIM_PATH = ROOT / "SINGLE_PRODUCTION_RUN_CLAIM_v2_6_1.json"
EXPECTED_RELEASE_STATUS = "GO_FOR_SINGLE_PRODUCTION_RUN_V2_6_1"
EXPECTED_PRODUCTION_DT_MS = 0.025
EXPECTED_PRODUCTION_ANALYSIS_TASK_COUNT = 245_256
EXPECTED_PRODUCTION_UNIQUE_SIMULATION_COUNT = 83_796
EXPECTED_PRODUCTION_TASK_IDENTITY_SHA256 = (
    "cbc3322ed317bab95e891e3a35804ff0c0e2f1365dfcc772273dd893815d5214"
)
REQUIRED_SUMMARY_FIELDS = {
    "frequency_hz", "lr_phase_error_mean_abs_deg",
    "fe_phase_error_mean_abs_deg", "lr_phase_slip_count",
    "lr_phase_cycle_count", "fe_phase_slip_count",
    "fe_phase_cycle_count", "rg_cycle_interval_cv_mean",
    "bilateral_amplitude_imbalance", "pf_transfer_anchor_count",
    "pf_transfer_missed_count", "pf_transfer_matched_count",
    "pf_transfer_reliability",
    "mn_transfer_anchor_count", "mn_transfer_missed_count",
    "mn_transfer_matched_count", "mn_transfer_reliability",
    "recovery_time_s", "recovery_event_observed",
    "recovery_time_or_censor_s", "recovery_endpoint_eligible",
    "recovery_ineligibility_reason", "recovery_composite_eligible",
    "recovery_composite_event", "recovery_composite_time_s",
    "recovery_censor_time_s", "pulse_start_s", "pulse_end_s",
    "pulse_required", "pulse_delivered",
    "rhythmic_failure", "technical_valid", "technical_exclusion_reason",
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest and fail if provenance is absent."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> object:
    raise ValueError(f"non-finite JSON constant: {token}")


def load_json_contract(path: Path, label: str) -> Dict[str, object]:
    """Load an auditable JSON object, rejecting corruption and duplicate keys."""
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable or corrupt: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object: {path}")
    return value


def _json_exact_equal(left: object, right: object) -> bool:
    """JSON-type-sensitive equality (unlike Python's ``True == 1``)."""
    try:
        return json.dumps(
            left, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) == json.dumps(
            right, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        return False


def require_exact_contract_fields(
    record: Mapping[str, object],
    expected: Mapping[Tuple[str, ...], object],
    label: str,
) -> None:
    """Require selected immutable JSON fields without rejecting annotations."""
    missing = object()
    for path, expected_value in expected.items():
        current: object = record
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = missing
                break
            current = current[key]
        dotted = ".".join(path)
        if current is missing:
            raise RuntimeError(f"{label} immutable field is missing: {dotted}")
        if not _json_exact_equal(current, expected_value):
            raise RuntimeError(
                f"{label} immutable field mismatch: {dotted}; "
                f"found={current!r}, expected={expected_value!r}"
            )


def write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    """Create/replace a JSON file atomically after callers perform guards."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(model.json_safe(dict(value)), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


REQUIRED_EVENT_FIELDS = {
    f"{cell_class}_{side}_{phase}_onset_s"
    for cell_class in ("RG", "PF", "MN")
    for side in ("L", "R")
    for phase in ("F", "E")
}
REQUIRED_ANALYSIS_OBSERVABLE_FIELDS = {
    "mn_left_rate_sum_hz_samples",
    "mn_right_rate_sum_hz_samples",
    "mn_rate_sample_count",
}
REQUIRED_INTERVENTION_LOG_FIELDS = {
    "pulse_required",
    "pulse_delivered",
    "pulse_start_s",
    "pulse_end_s",
    "pulse_noneligibility_reason",
}


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def assert_frozen_model_identity() -> str:
    """Refuse execution unless the reviewed v2.6.1 core identity is exact."""
    frozen_path = Path(frozen_model.__file__).resolve()
    frozen_digest = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    engine_path = Path(model.__file__).resolve()
    engine_digest = hashlib.sha256(engine_path.read_bytes()).hexdigest()
    if frozen_model.MODEL_VERSION != EXPECTED_MODEL_VERSION:
        raise RuntimeError(
            f"model version mismatch: {frozen_model.MODEL_VERSION!r} != {EXPECTED_MODEL_VERSION!r}"
        )
    if model.MODEL_VERSION != EXPECTED_MODEL_VERSION:
        raise RuntimeError("optimized engine declares a different biological model version")
    if frozen_digest != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "frozen model SHA-256 mismatch; accelerated execution is blocked: "
            f"{frozen_digest} != {EXPECTED_MODEL_SHA256}"
        )
    if engine_digest != EXPECTED_EXECUTION_ENGINE_SHA256:
        raise RuntimeError(
            "optimized engine SHA-256 mismatch; accelerated execution is blocked: "
            f"{engine_digest} != {EXPECTED_EXECUTION_ENGINE_SHA256}"
        )
    return frozen_digest


def production_release_contract() -> Dict[Tuple[str, ...], object]:
    """Exact, independently recomputed authorization fields for one full run."""
    return {
        ("release_status",): EXPECTED_RELEASE_STATUS,
        ("scientific_results_included",): False,
        ("model_version",): EXPECTED_MODEL_VERSION,
        ("model_sha256",): EXPECTED_MODEL_SHA256,
        ("execution_engine_sha256",): EXPECTED_EXECUTION_ENGINE_SHA256,
        ("accelerated_runner_sha256",): sha256_file(Path(__file__).resolve()),
        ("reference_runner_sha256",): sha256_file(Path(reference.__file__).resolve()),
        ("requirements_lock_sha256",): sha256_file(REQUIREMENTS_LOCK_PATH),
        ("primary_analysis_sha256",): sha256_file(ROOT / "analyze_primary_v2_6_1.py"),
        ("preflight_sha256",): sha256_file(ROOT / "preflight_results_v2_6_1.py"),
        ("dt_ms",): EXPECTED_PRODUCTION_DT_MS,
        ("analysis_task_count",): EXPECTED_PRODUCTION_ANALYSIS_TASK_COUNT,
        ("unique_simulation_count",): EXPECTED_PRODUCTION_UNIQUE_SIMULATION_COUNT,
        ("task_identity_sha256",): EXPECTED_PRODUCTION_TASK_IDENTITY_SHA256,
        ("numerical_validation_amended_passed",): True,
        ("acceleration_equivalence_passed",): True,
        ("calibration_values_match_frozen_config",): True,
        ("pending_sentinel_count",): 0,
        ("historical_failure_preserved",): True,
        ("old_failure_reclassified",): False,
    }


def assert_production_release_authorized(
    *,
    smoke: bool,
    dry_run: bool,
    freeze_path: Path | None = None,
) -> None:
    """Fail before task construction/work unless the final GO artifact is exact.

    Smoke and dry-run invocations are explicit development-only exceptions. A
    compile-only full-resolution invocation is intentionally *not* exempt: it
    can materialize scientific output and therefore belongs to the same frozen
    contract as worker execution.
    """
    if smoke or dry_run:
        return
    path = PRODUCTION_FREEZE_MANIFEST_PATH if freeze_path is None else Path(freeze_path)
    if not path.is_file():
        raise RuntimeError(
            "full scientific execution is blocked: missing "
            f"{path.name}; smoke and dry-run modes remain available"
        )
    manifest = load_json_contract(path, "production freeze manifest")
    require_exact_contract_fields(
        manifest, production_release_contract(), "production freeze manifest"
    )
    if frozen_model.Config().dt_ms != EXPECTED_PRODUCTION_DT_MS:
        raise RuntimeError(
            "full scientific execution is blocked: runtime Config.dt_ms is not 0.025"
        )
    # This repeats the CLI-level check deliberately so direct callers cannot
    # bypass the reviewed core/facade identity contract.
    assert_frozen_model_identity()


def production_claim_binding(
    output_dir: Path,
    freeze_path: Path | None = None,
) -> Dict[str, object]:
    """Build the deterministic authorization bound to one canonical outdir."""
    path = (
        PRODUCTION_FREEZE_MANIFEST_PATH
        if freeze_path is None else Path(freeze_path)
    ).expanduser().resolve()
    manifest = load_json_contract(path, "production freeze manifest")
    require_exact_contract_fields(
        manifest, production_release_contract(), "production freeze manifest"
    )
    binding: Dict[str, object] = {
        "claim_schema_version": "single-production-run-claim-v2.6.1",
        "release_status": EXPECTED_RELEASE_STATUS,
        "freeze_manifest_sha256": sha256_file(path),
        "production_output_dir": str(Path(output_dir).expanduser().resolve()),
        "model_version": EXPECTED_MODEL_VERSION,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "execution_engine_sha256": EXPECTED_EXECUTION_ENGINE_SHA256,
        "accelerated_runner_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_task_count": EXPECTED_PRODUCTION_ANALYSIS_TASK_COUNT,
        "unique_simulation_count": EXPECTED_PRODUCTION_UNIQUE_SIMULATION_COUNT,
        "task_identity_sha256": EXPECTED_PRODUCTION_TASK_IDENTITY_SHA256,
    }
    binding["claim_id"] = sha256_json(binding)
    return binding


def _write_claim_exclusive(path: Path, claim: Mapping[str, object]) -> bool:
    """Atomically publish a complete immutable claim; return False if claimed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.claim.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        payload = json.dumps(
            model.json_safe(dict(claim)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            return False
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def claim_single_production_run(
    output_dir: Path,
    *,
    claim_path: Path | None = None,
    freeze_path: Path | None = None,
) -> Dict[str, object]:
    """Claim the sole production outdir, allowing only exact same-dir resume."""
    path = (
        PRODUCTION_RUN_CLAIM_PATH if claim_path is None else Path(claim_path)
    ).expanduser().resolve()
    expected = production_claim_binding(output_dir, freeze_path)
    if path.exists():
        existing = load_json_contract(path, "single-production-run claim")
        require_exact_contract_fields(
            existing, _top_level_contract(expected), "single-production-run claim"
        )
        if set(existing) != set(expected):
            raise RuntimeError("single-production-run claim has unexpected fields")
        return existing
    if not _write_claim_exclusive(path, expected):
        existing = load_json_contract(path, "single-production-run claim")
        require_exact_contract_fields(
            existing, _top_level_contract(expected), "single-production-run claim"
        )
        if set(existing) != set(expected):
            raise RuntimeError("single-production-run claim has unexpected fields")
        return existing
    return expected


def config_profile(task: reference.Task, smoke: bool) -> Dict[str, object]:
    return asdict(reference.config_for_task(task, smoke))


@lru_cache(maxsize=4)
def config_sha256(protocol: str, smoke: bool) -> str:
    """Config depends only on protocol and smoke/full-resolution profile."""
    probe = reference.Task("A", 0, protocol=protocol)
    return sha256_json(config_profile(probe, smoke))


def simulation_identity(task: reference.Task, smoke: bool) -> Dict[str, object]:
    """Complete identity of values that can affect ``model.simulate``.

    Stage and label are intentionally absent: the reference runner never
    passes them to the simulator. They remain in the analysis-task index.
    """
    return {
        "model_version": model.MODEL_VERSION,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "execution_engine_sha256": EXPECTED_EXECUTION_ENGINE_SHA256,
        "config_sha256": config_sha256(task.protocol, smoke),
        "execution_profile": "smoke" if smoke else "scientific_full_resolution",
        "seed": task.seed,
        "structural_seed": 160000 + task.seed,
        "protocol": task.protocol,
        "speed_level": task.speed,
        "load_context": task.load,
        "load_side": task.load_side,
        "pulse_direction": task.pulse,
        "ablated_populations": sorted(task.ablations),
        "mt_mode": task.mt_mode,
        "impaired_mt_routes": sorted(task.impaired_mt_routes),
        "challenged_routes": sorted(task.challenged_routes),
        "fast_mode": task.fast_mode,
        "simulate_defaults": {
            "static_scale": 1.0,
            "fast_activation_scale": 1.0,
            "pulse_cycle_fraction_override": None,
            "pulse_target_side": reference.PULSE_TARGET_SIDE,
            "pulse_target_phase": reference.PULSE_TARGET_PHASE,
            "external_kca_events": (
                "deterministic_physical_time_dynamic_reference_replay_v2"
                if task.fast_mode == "yoked" else None
            ),
            "external_mt_events": (
                "deterministic_physical_time_dynamic_terminal_replay_v2"
                if task.mt_mode == "time_yoked" else None
            ),
        },
    }


def simulation_id(task: reference.Task, smoke: bool) -> str:
    return f"sim-{sha256_json(simulation_identity(task, smoke))}"


def unique_simulations(
    tasks: Sequence[reference.Task], smoke: bool
) -> Tuple[Dict[str, reference.Task], Dict[str, str], Counter]:
    representatives: Dict[str, reference.Task] = {}
    task_to_sim: Dict[str, str] = {}
    multiplicity: Counter = Counter()
    for task in tasks:
        sid = simulation_id(task, smoke)
        representatives.setdefault(sid, task)
        task_to_sim[task.task_id] = sid
        multiplicity[sid] += 1
    return representatives, task_to_sim, multiplicity


def compact_manifest(trace: Mapping[str, np.ndarray], cfg: model.Config) -> Dict[str, object]:
    """Keep run-varying provenance per simulation; store full configs once."""
    manifest = dict(model.simulation_manifest(trace, cfg))
    manifest.pop("parameters", None)
    # v2.6.1 executes the reviewed core directly rather than through the old
    # delegating facade.  Record that single-file execution identity explicitly
    # so checkpoint validation remains fail-closed and unambiguous.
    manifest["execution_engine_sha256"] = EXPECTED_EXECUTION_ENGINE_SHA256
    return manifest


def run_unique_simulation(
    task: reference.Task,
    sid: str,
    output_dir_text: str,
    smoke: bool,
    save_trace: bool,
) -> Tuple[str, float]:
    """Worker entry point; writes one atomic canonical checkpoint."""
    started = time.perf_counter()
    output_dir = Path(output_dir_text)
    cfg = reference.config_for_task(task, smoke)
    identity = simulation_identity(task, smoke)
    if sid != f"sim-{sha256_json(identity)}":
        raise RuntimeError(f"simulation identity mismatch for {sid}")

    external_kca_event_times_s = None
    external_kca_event_neurons = None
    external_mt_event_times_s = None
    external_mt_event_edges = None
    common_simulation_arguments = dict(
        structural_seed=160000 + task.seed,
        speed_level=task.speed,
        load_context=task.load,
        load_side=task.load_side,
        pulse_direction=task.pulse,
        pulse_target_side=reference.PULSE_TARGET_SIDE,
        pulse_target_phase=reference.PULSE_TARGET_PHASE,
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
            "simulation failed technical validity; no canonical checkpoint "
            f"will be written ({summary.get('technical_exclusion_reason')})"
        )
    record: Dict[str, object] = {
        "simulation_id": sid,
        "scientific_valid": not smoke,
        "simulation_identity": identity,
        "representative_task_id": task.task_id,
        "summary": summary,
        "analysis_events": model.analysis_event_payload(trace, cfg),
        "analysis_observables": model.analysis_observable_payload(trace, cfg),
        "intervention_log": model.intervention_log_payload(trace, cfg),
        "manifest": compact_manifest(trace, cfg),
    }
    if task.protocol == "long":
        record["epochs"] = model.summarize_long_epochs(trace, cfg)

    simulation_dir = output_dir / "simulations"
    simulation_dir.mkdir(parents=True, exist_ok=True)
    final_path = simulation_dir / f"{sid}.json"
    temp_path = simulation_dir / f".{sid}.{os.getpid()}.tmp"
    safe_record = model.json_safe(record)
    safe_record["checkpoint_payload_sha256"] = sha256_json(safe_record)
    temp_path.write_text(
        json.dumps(safe_record, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    temp_path.replace(final_path)
    validate_checkpoint(final_path, sid)

    if save_trace:
        trace_dir = output_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_dir / f"{sid}.npz", **trace)
    return sid, time.perf_counter() - started


def existing_simulations(output_dir: Path) -> set[str]:
    """Return only complete, hash-consistent checkpoints."""
    simulation_dir = output_dir / "simulations"
    if not simulation_dir.exists():
        return set()
    valid: set[str] = set()
    for path in simulation_dir.glob("sim-*.json"):
        try:
            validate_checkpoint(path, path.stem)
        except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError):
            continue
        valid.add(path.stem)
    return valid


def validate_checkpoint(path: Path, expected_sid: str) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    recorded_payload_sha = record.get("checkpoint_payload_sha256")
    payload = dict(record)
    payload.pop("checkpoint_payload_sha256", None)
    if (
        not isinstance(recorded_payload_sha, str)
        or sha256_json(payload) != recorded_payload_sha
    ):
        raise RuntimeError(f"checkpoint payload hash mismatch: {path}")
    if record.get("simulation_id") != expected_sid:
        raise RuntimeError(f"checkpoint identity mismatch: {path}")
    identity = record.get("simulation_identity")
    if not isinstance(identity, dict) or f"sim-{sha256_json(identity)}" != expected_sid:
        raise RuntimeError(f"checkpoint hash mismatch: {path}")
    summary = record.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError(f"checkpoint summary missing: {path}")
    missing_summary = REQUIRED_SUMMARY_FIELDS - set(summary)
    if missing_summary:
        raise RuntimeError(
            f"checkpoint summary schema incomplete ({sorted(missing_summary)}): {path}"
        )
    for slips, cycles in (
        ("lr_phase_slip_count", "lr_phase_cycle_count"),
        ("fe_phase_slip_count", "fe_phase_cycle_count"),
    ):
        if not (
            isinstance(summary[slips], int)
            and isinstance(summary[cycles], int)
            and 0 <= summary[slips] <= summary[cycles]
        ):
            raise RuntimeError(f"checkpoint phase counts invalid: {path}")
    for prefix in ("pf", "mn"):
        anchor = summary[f"{prefix}_transfer_anchor_count"]
        missed = summary[f"{prefix}_transfer_missed_count"]
        matched = summary[f"{prefix}_transfer_matched_count"]
        if not (
            all(isinstance(value, int) and value >= 0 for value in (
                anchor, missed, matched
            ))
            and missed + matched == anchor
        ):
            raise RuntimeError(f"checkpoint transfer counts invalid: {path}")
    for flag in (
        "recovery_composite_eligible", "recovery_composite_event",
        "pulse_required", "pulse_delivered", "rhythmic_failure",
        "technical_valid",
    ):
        if summary[flag] not in (0, 1):
            raise RuntimeError(f"checkpoint binary flag invalid ({flag}): {path}")
    if summary["technical_valid"] != 1:
        raise RuntimeError(f"checkpoint is technically invalid and must be rerun: {path}")
    events = record.get("analysis_events")
    if not isinstance(events, dict):
        raise RuntimeError(f"checkpoint event payload missing: {path}")
    if set(events) != REQUIRED_EVENT_FIELDS:
        raise RuntimeError(f"checkpoint event payload schema invalid: {path}")
    for key, values in events.items():
        if not isinstance(values, list):
            raise RuntimeError(f"checkpoint event list invalid ({key}): {path}")
        array = np.asarray(values, dtype=float)
        if (
            not np.all(np.isfinite(array))
            or np.any(array < 0.0)
            or (len(array) > 1 and np.any(np.diff(array) <= 0.0))
        ):
            raise RuntimeError(f"checkpoint event values invalid ({key}): {path}")
    observables = record.get("analysis_observables")
    if (
        not isinstance(observables, dict)
        or set(observables) != REQUIRED_ANALYSIS_OBSERVABLE_FIELDS
    ):
        raise RuntimeError(f"checkpoint analysis-observable schema invalid: {path}")
    sample_count = observables["mn_rate_sample_count"]
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
        raise RuntimeError(f"checkpoint MN sample count invalid: {path}")
    for key in (
        "mn_left_rate_sum_hz_samples", "mn_right_rate_sum_hz_samples"
    ):
        value = observables[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise RuntimeError(f"checkpoint analysis observable invalid ({key}): {path}")
    intervention_log = record.get("intervention_log")
    if (
        not isinstance(intervention_log, dict)
        or set(intervention_log) != REQUIRED_INTERVENTION_LOG_FIELDS
    ):
        raise RuntimeError(f"checkpoint intervention-log schema invalid: {path}")
    for key in ("pulse_required", "pulse_delivered"):
        if intervention_log[key] not in (0, 1):
            raise RuntimeError(f"checkpoint intervention flag invalid ({key}): {path}")
    required = int(intervention_log["pulse_required"])
    delivered = int(intervention_log["pulse_delivered"])
    start = intervention_log["pulse_start_s"]
    end = intervention_log["pulse_end_s"]
    reason = intervention_log["pulse_noneligibility_reason"]
    if not isinstance(reason, str):
        raise RuntimeError(f"checkpoint intervention reason invalid: {path}")
    if delivered:
        if (
            required != 1
            or reason != "none"
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not (np.isfinite(float(start)) and np.isfinite(float(end)))
            or not (0.0 <= float(start) < float(end))
        ):
            raise RuntimeError(f"checkpoint delivered-pulse log invalid: {path}")
    elif required:
        if (
            start is not None
            or end is not None
            or reason != "biological_no_phase_eligible_cycle"
        ):
            raise RuntimeError(f"checkpoint biological non-delivery log invalid: {path}")
    elif reason != "no_pulse_condition" or start is not None or end is not None:
        raise RuntimeError(f"checkpoint no-pulse log invalid: {path}")
    manifest = record.get("manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError(f"checkpoint manifest missing: {path}")
    if (
        manifest.get("model_version") != EXPECTED_MODEL_VERSION
        or manifest.get("model_sha256") != EXPECTED_MODEL_SHA256
        or manifest.get("execution_engine_sha256")
            != EXPECTED_EXECUTION_ENGINE_SHA256
    ):
        raise RuntimeError(f"checkpoint model/engine identity mismatch: {path}")
    for manifest_key, identity_key in (
        ("seed", "seed"), ("structural_seed", "structural_seed"),
        ("protocol", "protocol"), ("fast_mode", "fast_mode"),
        ("mt_mode", "mt_mode"), ("speed_level", "speed_level"),
        ("load_context", "load_context"), ("load_side", "load_side"),
        ("pulse_direction", "pulse_direction"),
    ):
        if manifest.get(manifest_key) != identity.get(identity_key):
            raise RuntimeError(
                f"checkpoint manifest/identity mismatch ({manifest_key}): {path}"
            )
    expected_target = (
        f"{identity['simulate_defaults']['pulse_target_side']}-"
        f"{identity['simulate_defaults']['pulse_target_phase']}"
    )
    if manifest.get("pulse_target") != expected_target:
        raise RuntimeError(f"checkpoint pulse target mismatch: {path}")
    if sorted(manifest.get("challenged_routes", [])) != sorted(
        identity.get("challenged_routes", [])
    ):
        raise RuntimeError(f"checkpoint challenged-route mismatch: {path}")
    if identity.get("protocol") == "long":
        epochs = record.get("epochs")
        if (
            not isinstance(epochs, list)
            or len(epochs) != model.Config().long_n_epochs
            or [row.get("epoch") for row in epochs]
                != list(range(1, model.Config().long_n_epochs + 1))
        ):
            raise RuntimeError(f"checkpoint long-epoch payload invalid: {path}")


def write_task_index(
    output_dir: Path,
    tasks: Sequence[reference.Task],
    task_to_sim: Mapping[str, str],
    multiplicity: Mapping[str, int],
) -> None:
    path = output_dir / "analysis_task_index.csv"
    fieldnames = [
        "task_id", "simulation_id", "reuse_count", "stage", "seed", "protocol",
        "speed", "load", "load_side", "pulse", "ablations", "mt_mode",
        "impaired_mt_routes", "challenged_routes", "fast_mode", "label",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in tasks:
            sid = task_to_sim[task.task_id]
            row = task.identity()
            for key in ("ablations", "impaired_mt_routes", "challenged_routes"):
                row[key] = "+".join(row[key]) if row[key] else "none"
            writer.writerow({
                "task_id": task.task_id,
                "simulation_id": sid,
                "reuse_count": multiplicity[sid],
                **row,
            })


def task_identity_sha256(tasks: Sequence[reference.Task]) -> str:
    """Match the normative reference runner's ordered task commitment."""
    digest = hashlib.sha256()
    for task in tasks:
        row = {"task_id": task.task_id, **task.identity()}
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def assert_production_task_matrix(
    tasks: Sequence[reference.Task],
    representatives: Mapping[str, reference.Task],
) -> None:
    """Independently bind the materialized production matrix to the GO counts."""
    if len(tasks) != EXPECTED_PRODUCTION_ANALYSIS_TASK_COUNT:
        raise RuntimeError(
            "production analysis-task count drifted: "
            f"{len(tasks)} != {EXPECTED_PRODUCTION_ANALYSIS_TASK_COUNT}"
        )
    if len(representatives) != EXPECTED_PRODUCTION_UNIQUE_SIMULATION_COUNT:
        raise RuntimeError(
            "production unique-simulation count drifted: "
            f"{len(representatives)} != {EXPECTED_PRODUCTION_UNIQUE_SIMULATION_COUNT}"
        )
    actual_identity = task_identity_sha256(tasks)
    if actual_identity != EXPECTED_PRODUCTION_TASK_IDENTITY_SHA256:
        raise RuntimeError(
            "production task identity drifted: "
            f"{actual_identity} != {EXPECTED_PRODUCTION_TASK_IDENTITY_SHA256}"
        )


def execution_environment_manifest(
    *,
    seed_profile: str,
    smoke: bool,
    production_authorization: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Build immutable code/config/environment provenance for resume checks."""
    config_examples = {
        "pulse": config_profile(reference.Task("A", 0), smoke),
        "long": config_profile(
            reference.Task("G", 0, protocol="long", pulse="none"), smoke
        ),
    }
    return {
        "runner_version": RUNNER_VERSION,
        "seed_profile": seed_profile,
        "model_version": model.MODEL_VERSION,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "execution_engine_sha256": EXPECTED_EXECUTION_ENGINE_SHA256,
        "reference_runner_sha256": sha256_file(Path(reference.__file__).resolve()),
        "accelerated_runner_sha256": sha256_file(Path(__file__).resolve()),
        "requirements_lock_sha256": sha256_file(REQUIREMENTS_LOCK_PATH),
        "config_profiles": config_examples,
        "config_profile_sha256": {
            key: sha256_json(value) for key, value in config_examples.items()
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
        "production_authorization": (
            dict(production_authorization)
            if production_authorization is not None else None
        ),
    }


def _top_level_contract(value: Mapping[str, object]) -> Dict[Tuple[str, ...], object]:
    return {(key,): item for key, item in value.items()}


def plan_resume_contract(
    tasks: Sequence[reference.Task],
    stages: Sequence[str],
    seed_profile: str,
    smoke: bool,
    representatives: Mapping[str, reference.Task],
) -> Dict[Tuple[str, ...], object]:
    """Fields that make an existing plan scientifically resume-compatible."""
    counts = {
        stage: sum(task.stage == stage for task in tasks) for stage in stages
    }
    seeds_by_stage = {
        stage: sorted({task.seed for task in tasks if task.stage == stage})
        for stage in stages
    }
    return {
        ("model_version",): model.MODEL_VERSION,
        ("scientific_valid",): not smoke and seed_profile == "production_split",
        ("stages",): {
            stage: reference.STAGE_DESCRIPTIONS[stage] for stage in stages
        },
        ("seeds_by_stage",): seeds_by_stage,
        ("seed_profile",): seed_profile,
        ("task_count_by_stage",): counts,
        ("total_task_count",): len(tasks),
        ("task_identity_sha256",): task_identity_sha256(tasks),
        ("accelerated_execution", "runner_version"): RUNNER_VERSION,
        ("accelerated_execution", "seed_profile"): seed_profile,
        ("accelerated_execution", "model_equations_or_parameters_changed"): False,
        ("accelerated_execution", "task_matrix_changed"): False,
        ("accelerated_execution", "seed_plan_changed"): False,
        ("accelerated_execution", "deterministic_reuse_only"): True,
        ("accelerated_execution", "analysis_task_count"): len(tasks),
        ("accelerated_execution", "unique_simulation_count"): len(representatives),
    }


def write_execution_plan(
    output_dir: Path,
    tasks: Sequence[reference.Task],
    stages: Sequence[str],
    seeds: Sequence[int],
    seed_profile: str,
    smoke: bool,
    workers: int,
    representatives: Mapping[str, reference.Task],
    task_to_sim: Mapping[str, str],
    multiplicity: Mapping[str, int],
    production_authorization: Mapping[str, object] | None = None,
) -> None:
    plan_path = output_dir / "experiment_plan.json"
    manifest_path = output_dir / "execution_manifest.json"
    plan_contract = plan_resume_contract(
        tasks, stages, seed_profile, smoke, representatives
    )
    expected_execution_manifest = execution_environment_manifest(
        seed_profile=seed_profile,
        smoke=smoke,
        production_authorization=production_authorization,
    )

    # Validate every pre-existing immutable artifact before writing either one.
    # Thus a changed environment/task matrix cannot partially rewrite a resume
    # directory and cannot reach worker creation.
    plan_exists = plan_path.exists()
    manifest_exists = manifest_path.exists()
    if plan_exists != manifest_exists:
        raise RuntimeError(
            "incomplete immutable resume directory: experiment_plan.json and "
            "execution_manifest.json must either both be absent or both exist"
        )
    if plan_exists:
        existing_plan = load_json_contract(plan_path, "experiment plan")
        require_exact_contract_fields(
            existing_plan, plan_contract, "experiment plan"
        )
    if manifest_exists:
        existing_manifest = load_json_contract(
            manifest_path, "execution manifest"
        )
        require_exact_contract_fields(
            existing_manifest,
            _top_level_contract(expected_execution_manifest),
            "execution manifest",
        )

    if plan_exists:
        return

    if not plan_exists:
        reference.write_plan(
            output_dir, tasks, stages, seeds, smoke, seed_profile
        )
        plan = load_json_contract(plan_path, "new experiment plan")
        unique_by_stage: Dict[str, int] = {}
        seen: set[str] = set()
        novel_by_stage: Dict[str, int] = {}
        for stage in stages:
            stage_ids = {
                task_to_sim[task.task_id] for task in tasks if task.stage == stage
            }
            unique_by_stage[stage] = len(stage_ids)
            novel_by_stage[stage] = len(stage_ids - seen)
            seen.update(stage_ids)
        multiplicity_histogram = Counter(multiplicity.values())
        plan["accelerated_execution"] = {
            "runner_version": RUNNER_VERSION,
            "seed_profile": seed_profile,
            "model_equations_or_parameters_changed": False,
            "task_matrix_changed": False,
            "seed_plan_changed": False,
            "deterministic_reuse_only": True,
            "analysis_task_count": len(tasks),
            "unique_simulation_count": len(representatives),
            "avoided_identical_recomputations": len(tasks) - len(representatives),
            "reduction_fraction": 1.0 - len(representatives) / max(1, len(tasks)),
            "unique_simulations_within_stage": unique_by_stage,
            "novel_simulations_after_prior_stages": novel_by_stage,
            "reuse_multiplicity_histogram": {
                str(key): value
                for key, value in sorted(multiplicity_histogram.items())
            },
            "workers": workers,
            "statistical_unit_warning": (
                "Rows sharing simulation_id are one deterministic seed-condition result, "
                "not independent observations. Use them only in their preregistered paired contrasts."
            ),
        }
        plan["runtime"]["accelerated_runner"] = RUNNER_VERSION
        write_json_atomic(plan_path, plan)
        require_exact_contract_fields(plan, plan_contract, "new experiment plan")

    write_json_atomic(manifest_path, expected_execution_manifest)


def flatten_simulation_record(record: Mapping[str, object]) -> Dict[str, object]:
    identity = dict(record["simulation_identity"])
    for key in ("ablated_populations", "impaired_mt_routes", "challenged_routes"):
        values = identity.get(key, [])
        identity[key] = "+".join(values) if values else "none"
    identity.pop("simulate_defaults", None)
    return {
        "simulation_id": record["simulation_id"],
        "scientific_valid": record["scientific_valid"],
        **identity,
        **dict(record["summary"]),
    }


def compile_outputs(
    output_dir: Path,
    tasks: Sequence[reference.Task],
    task_to_sim: Mapping[str, str],
    multiplicity: Mapping[str, int],
) -> None:
    """Create unique and task-mapped tables without loading all manifests."""
    by_sim: MutableMapping[str, List[reference.Task]] = defaultdict(list)
    for task in tasks:
        by_sim[task_to_sim[task.task_id]].append(task)

    unique_path = output_dir / "unique_simulation_metrics.csv"
    mapped_path = output_dir / "metrics.csv"
    epoch_path = output_dir / "long_epoch_metrics.csv"
    unique_handle = unique_path.open("w", newline="", encoding="utf-8")
    mapped_handle = mapped_path.open("w", newline="", encoding="utf-8")
    epoch_handle = epoch_path.open("w", newline="", encoding="utf-8")
    unique_writer = None
    mapped_writer = None
    epoch_writer = None
    try:
        for sid, mapped_tasks in by_sim.items():
            path = output_dir / "simulations" / f"{sid}.json"
            validate_checkpoint(path, sid)
            record = json.loads(path.read_text(encoding="utf-8"))
            unique_row = flatten_simulation_record(record)
            if unique_writer is None:
                unique_writer = csv.DictWriter(unique_handle, fieldnames=list(unique_row))
                unique_writer.writeheader()
            unique_writer.writerow(unique_row)

            for task in mapped_tasks:
                task_row = task.identity()
                for key in ("ablations", "impaired_mt_routes", "challenged_routes"):
                    task_row[key] = "+".join(task_row[key]) if task_row[key] else "none"
                row = {
                    "task_id": task.task_id,
                    "simulation_id": sid,
                    "reuse_count": multiplicity[sid],
                    "scientific_valid": record["scientific_valid"],
                    **task_row,
                    **dict(record["summary"]),
                }
                if mapped_writer is None:
                    mapped_writer = csv.DictWriter(mapped_handle, fieldnames=list(row))
                    mapped_writer.writeheader()
                mapped_writer.writerow(row)

            for epoch in record.get("epochs", []):
                row = {"simulation_id": sid, **epoch}
                if epoch_writer is None:
                    epoch_writer = csv.DictWriter(epoch_handle, fieldnames=list(row))
                    epoch_writer.writeheader()
                epoch_writer.writerow(row)
    finally:
        unique_handle.close()
        mapped_handle.close()
        epoch_handle.close()
    if epoch_writer is None:
        epoch_path.unlink(missing_ok=True)


def run_pending_parallel(
    pending_items: Sequence[Tuple[str, reference.Task]],
    output_dir: Path,
    smoke: bool,
    save_trace: bool,
    workers: int,
    progress_every: int,
) -> List[float]:
    if not pending_items:
        return []
    durations: List[float] = []
    total = len(pending_items)
    if workers == 1:
        for index, (sid, task) in enumerate(pending_items, start=1):
            _, elapsed = run_unique_simulation(
                task, sid, str(output_dir), smoke, save_trace
            )
            durations.append(elapsed)
            if index == 1 or index % progress_every == 0 or index == total:
                print(f"[{index}/{total}] completed {sid} ({elapsed:.2f} s)", flush=True)
        return durations

    iterator = iter(pending_items)
    submitted = 0
    completed = 0
    max_in_flight = max(workers * 2, workers + 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        while submitted < min(max_in_flight, total):
            sid, task = next(iterator)
            future = executor.submit(
                run_unique_simulation, task, sid, str(output_dir), smoke, save_trace
            )
            futures[future] = sid
            submitted += 1
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                sid = futures.pop(future)
                result_sid, elapsed = future.result()
                if result_sid != sid:
                    raise RuntimeError(f"worker returned wrong simulation id: {result_sid}")
                completed += 1
                durations.append(elapsed)
                if completed == 1 or completed % progress_every == 0 or completed == total:
                    print(
                        f"[{completed}/{total}] completed {sid} ({elapsed:.2f} s; "
                        f"workers={workers})",
                        flush=True,
                    )
                if submitted < total:
                    next_sid, next_task = next(iterator)
                    next_future = executor.submit(
                        run_unique_simulation,
                        next_task,
                        next_sid,
                        str(output_dir),
                        smoke,
                        save_trace,
                    )
                    futures[next_future] = next_sid
                    submitted += 1
    return durations


def parse_csv(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def automatic_worker_count() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(8, cpu - 1 if cpu > 2 else cpu))


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
    parser.add_argument("--outdir", default="ah_results_accelerated")
    parser.add_argument("--workers", type=int, default=0, help="0 selects a conservative automatic value")
    parser.add_argument("--dry-run", action="store_true", help="write plan and task index only")
    parser.add_argument("--smoke", action="store_true", help="non-scientific reduced-size execution")
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="development-only analysis-task cap")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    requested = (
        reference.STAGES
        if args.stage.lower() == "all"
        else tuple(part.upper() for part in parse_csv(args.stage))
    )
    if not requested or any(stage not in reference.STAGES for stage in requested):
        parser.error("--stage must contain A-H or 'all'")
    production_profile = args.seeds.lower() == "production"
    if production_profile:
        seeds = tuple(sorted({
            seed
            for stage in requested
            for seed in reference.PRODUCTION_STAGE_SEEDS[stage]
        }))
    else:
        try:
            seeds = tuple(int(value) for value in parse_csv(args.seeds))
        except ValueError as exc:
            parser.error(f"invalid seed: {exc}")
        if not seeds or len(set(seeds)) != len(seeds):
            parser.error(
                "--seeds must be 'production' or a non-empty unique integer list"
            )
    if not args.smoke and not args.dry_run:
        if not production_profile:
            parser.error("scientific execution requires --seeds production")
        if requested != reference.STAGES:
            parser.error("scientific execution requires --stage all")
        if args.limit is not None:
            parser.error("--limit is allowed only with --smoke or --dry-run")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    workers = args.workers or automatic_worker_count()

    # All parser/profile errors and both release guards precede output
    # directory creation, task construction, checkpoint inspection and worker
    # pool creation. Smoke/dry runs bypass only the final GO-manifest gate.
    assert_frozen_model_identity()
    assert_production_release_authorized(
        smoke=args.smoke,
        dry_run=args.dry_run,
    )

    output_dir = Path(args.outdir).expanduser().resolve()
    tasks = (
        reference.build_production_tasks(requested)
        if production_profile else reference.build_tasks(requested, seeds)
    )
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        tasks = tasks[:args.limit]
    representatives, task_to_sim, multiplicity = unique_simulations(tasks, args.smoke)
    if not args.smoke and not args.dry_run:
        assert_production_task_matrix(tasks, representatives)
    production_authorization = None
    if not args.smoke and not args.dry_run:
        production_authorization = claim_single_production_run(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_execution_plan(
        output_dir,
        tasks,
        requested,
        seeds,
        "production_split" if production_profile else "explicit",
        args.smoke,
        workers,
        representatives,
        task_to_sim,
        multiplicity,
        production_authorization,
    )
    write_task_index(output_dir, tasks, task_to_sim, multiplicity)

    completed = existing_simulations(output_dir)
    pending_items = [
        (sid, task) for sid, task in representatives.items() if sid not in completed
    ]
    print(json.dumps({
        "analysis_tasks": len(tasks),
        "unique_simulations": len(representatives),
        "identical_recomputations_avoided": len(tasks) - len(representatives),
        "already_completed": len(representatives) - len(pending_items),
        "pending": len(pending_items),
        "workers": workers,
        "outdir": str(output_dir),
    }, indent=2), flush=True)
    if args.dry_run:
        return
    if args.compile_only:
        if pending_items:
            raise RuntimeError(
                f"cannot compile: {len(pending_items)} canonical simulations are missing"
            )
        compile_outputs(output_dir, tasks, task_to_sim, multiplicity)
        return

    wall_start = time.perf_counter()
    durations = run_pending_parallel(
        pending_items,
        output_dir,
        args.smoke,
        args.save_trace,
        workers,
        args.progress_every,
    )
    wall_elapsed = time.perf_counter() - wall_start
    if not args.no_compile:
        compile_outputs(output_dir, tasks, task_to_sim, multiplicity)
    completion = {
        "runner_version": RUNNER_VERSION,
        "scientific_valid": not args.smoke and production_profile,
        "analysis_tasks": len(tasks),
        "unique_simulations": len(representatives),
        "identical_recomputations_avoided": len(tasks) - len(representatives),
        "workers": workers,
        "wall_elapsed_s_this_invocation": wall_elapsed,
        "worker_task_elapsed_s_median": float(np.median(durations)) if durations else None,
        "worker_task_elapsed_s_min": float(np.min(durations)) if durations else None,
        "worker_task_elapsed_s_max": float(np.max(durations)) if durations else None,
        "completed_checkpoint_count": len(existing_simulations(output_dir)),
        "production_claim_id": (
            production_authorization["claim_id"]
            if production_authorization is not None else None
        ),
        "freeze_manifest_sha256": (
            production_authorization["freeze_manifest_sha256"]
            if production_authorization is not None else None
        ),
    }
    (output_dir / "completion.json").write_text(
        json.dumps(model.json_safe(completion), indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
