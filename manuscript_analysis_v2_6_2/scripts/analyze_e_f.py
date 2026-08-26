#!/usr/bin/env python3
"""Locked descriptive Stage E--F analysis for the v2.6.2 manuscript.

This script implements only the contracts in
ANALYSIS_PROTOCOL_LOCKED_2026-08-26.md and PROTOCOL_SPEC.json.  It produces
paired, conditional point estimates for one frozen realization.  It never
produces p-values, confidence intervals, standard errors, degrees of freedom,
or pseudo-replicate counts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_VERSION = "cpg-v2.6.2-manuscript-e-f-1.0"
NEUTRAL_TOLERANCE = 1.0e-12
ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
SPEC_PATH = ROOT / "PROTOCOL_SPEC.json"
DERIVED = REPOSITORY_ROOT / "derived" / "manuscript_analysis_v2_6_2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_selected(path: Path, keep: set[str] | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        if keep is not None:
            missing = keep - set(reader.fieldnames)
            if missing:
                raise RuntimeError(f"{path.name} lacks fields: {sorted(missing)}")
        rows = []
        for source in reader:
            rows.append(
                dict(source) if keep is None
                else {key: source[key] for key in keep}
            )
    return rows


def finite_float(row: Mapping[str, str], field: str) -> float | None:
    raw = row.get(field, "")
    if raw is None or str(raw).strip() == "":
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def integer(row: Mapping[str, str], field: str) -> int | None:
    value = finite_float(row, field)
    if value is None:
        return None
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"non-integer value in {field}: {value}")
    return rounded


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def split_members(value: str) -> tuple[str, ...]:
    value = str(value).strip()
    if not value or value == "none":
        return ()
    return tuple(part for part in value.split("+") if part)


def rhythmic(row: Mapping[str, str]) -> bool:
    return integer(row, "rhythmic_failure") == 0


def failure_transition(reference: Mapping[str, str], intervention: Mapping[str, str]) -> str:
    before = rhythmic(reference)
    after = rhythmic(intervention)
    if before and after:
        return "rhythmic_to_rhythmic"
    if before and not after:
        return "rhythmic_to_failure"
    if not before and after:
        return "failure_to_rhythmic"
    return "failure_to_failure"


def ratio(row: Mapping[str, str], numerator: str, denominator: str) -> float | None:
    events = integer(row, numerator)
    trials = integer(row, denominator)
    if events is None or trials is None or trials <= 0:
        return None
    if not 0 <= events <= trials:
        raise RuntimeError(f"invalid event/trial pair: {numerator}/{denominator}")
    return events / trials


def context_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        row[field]
        for field in ("seed", "protocol", "speed", "load", "load_side", "pulse")
    )


def context_id(row: Mapping[str, str]) -> str:
    return "|".join(context_key(row))


def quantile_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "median": None,
            "q1": None,
            "q3": None,
            "minimum": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise RuntimeError("non-finite value reached descriptive summary")
    return {
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25, method="linear")),
        "q3": float(np.quantile(array, 0.75, method="linear")),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def direction_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "negative_count": sum(value < -NEUTRAL_TOLERANCE for value in values),
        "neutral_count": sum(abs(value) <= NEUTRAL_TOLERANCE for value in values),
        "positive_count": sum(value > NEUTRAL_TOLERANCE for value in values),
    }


def metadata_columns(
    *, generated_utc: str, script_sha256: str, source_hashes: Mapping[str, str], row_count: int
) -> dict[str, Any]:
    return {
        "analysis_version": SCRIPT_VERSION,
        "generated_utc": generated_utc,
        "script_sha256": script_sha256,
        "source_task_index_sha256": source_hashes["analysis_task_index.csv"],
        "source_metrics_sha256": source_hashes["metrics.csv"],
        "protocol_spec_sha256": sha256_file(SPEC_PATH),
        "output_row_count": row_count,
        "independent_stochastic_realization_count": 1,
        "inferential_scope": "conditional_on_one_frozen_realization_no_population_inference",
    }


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    generated_utc: str,
    script_sha256: str,
    source_hashes: Mapping[str, str],
) -> int:
    row_count = len(rows)
    meta = metadata_columns(
        generated_utc=generated_utc,
        script_sha256=script_sha256,
        source_hashes=source_hashes,
        row_count=row_count,
    )
    enriched = [{**meta, **dict(row)} for row in rows]
    fields: list[str] = list(meta)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(enriched)
    # Re-open the physical artifact: a manifest must never certify an assumed
    # in-memory length if a write was interrupted or truncated.
    with path.open(newline="", encoding="utf-8") as handle:
        actual_rows = sum(1 for _ in csv.DictReader(handle))
    if actual_rows != row_count:
        raise RuntimeError(
            f"derived CSV row-count mismatch for {path.name}: "
            f"{actual_rows} != {row_count}"
        )
    return row_count


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


Getter = Callable[[Mapping[str, str], str], float | None]
Eligible = Callable[[Mapping[str, str], str], bool]


class Endpoint:
    def __init__(
        self,
        name: str,
        role: str,
        direction: str,
        getter: Getter,
        eligible: Eligible,
    ) -> None:
        self.name = name
        self.role = role
        self.direction = direction
        self.getter = getter
        self.eligible = eligible


def field_getter(field: str) -> Getter:
    return lambda row, _route: finite_float(row, field)


def always_rhythmic(row: Mapping[str, str], _route: str) -> bool:
    return rhythmic(row)


def slip_getter(prefix: str) -> Getter:
    return lambda row, _route: ratio(
        row, f"{prefix}_phase_slip_count", f"{prefix}_phase_cycle_count"
    )


def transfer_getter(prefix: str) -> Getter:
    return lambda row, _route: ratio(
        row, f"{prefix}_transfer_missed_count", f"{prefix}_transfer_anchor_count"
    )


def recovery_event_getter(row: Mapping[str, str], _route: str) -> float | None:
    if integer(row, "recovery_endpoint_eligible") != 1:
        return None
    event = integer(row, "recovery_event_observed")
    return None if event is None else float(event)


def recovery_event_eligible(row: Mapping[str, str], _route: str) -> bool:
    return row["pulse"] != "none" and integer(row, "recovery_endpoint_eligible") == 1


def recovery_observed_getter(row: Mapping[str, str], _route: str) -> float | None:
    if integer(row, "recovery_event_observed") != 1:
        return None
    return finite_float(row, "recovery_time_s")


def recovery_observed_eligible(row: Mapping[str, str], _route: str) -> bool:
    return recovery_event_eligible(row, _route) and integer(
        row, "recovery_event_observed"
    ) == 1 and finite_float(row, "recovery_time_s") is not None


def state_getter(prefix: str) -> Getter:
    def getter(row: Mapping[str, str], route: str) -> float | None:
        return finite_float(row, f"{prefix}_{route}_mean")
    return getter


PHASE_RECOVERY_ENDPOINTS = (
    Endpoint(
        "lr_phase_error_mean_abs_deg", "primary_phase", "higher_is_worse",
        field_getter("lr_phase_error_mean_abs_deg"), always_rhythmic,
    ),
    Endpoint(
        "fe_phase_error_mean_abs_deg", "primary_phase", "higher_is_worse",
        field_getter("fe_phase_error_mean_abs_deg"), always_rhythmic,
    ),
    Endpoint(
        "lr_phase_slip_rate", "primary_phase", "higher_is_worse",
        slip_getter("lr"),
        lambda row, route: always_rhythmic(row, route)
        and integer(row, "lr_phase_cycle_count") not in (None, 0),
    ),
    Endpoint(
        "fe_phase_slip_rate", "primary_phase", "higher_is_worse",
        slip_getter("fe"),
        lambda row, route: always_rhythmic(row, route)
        and integer(row, "fe_phase_cycle_count") not in (None, 0),
    ),
    Endpoint(
        "recovery_event_observed", "primary_recovery", "higher_is_better",
        recovery_event_getter, recovery_event_eligible,
    ),
    Endpoint(
        "recovery_observed_time_s", "primary_recovery", "higher_is_worse",
        recovery_observed_getter, recovery_observed_eligible,
    ),
)


PROPAGATION_ENDPOINTS = (
    Endpoint(
        "frequency_hz", "secondary_rhythm", "raw_change_no_favorable_direction",
        field_getter("frequency_hz"), always_rhythmic,
    ),
    Endpoint(
        "rg_cycle_interval_cv_mean", "secondary_rhythm", "higher_is_worse",
        field_getter("rg_cycle_interval_cv_mean"), always_rhythmic,
    ),
    Endpoint(
        "bilateral_amplitude_imbalance", "secondary_motor", "higher_is_worse",
        field_getter("bilateral_amplitude_imbalance"), always_rhythmic,
    ),
    Endpoint(
        "pf_missed_transfer_rate", "secondary_propagation", "higher_is_worse",
        transfer_getter("pf"),
        lambda row, route: always_rhythmic(row, route)
        and integer(row, "pf_transfer_anchor_count") not in (None, 0),
    ),
    Endpoint(
        "mn_missed_transfer_rate", "secondary_propagation", "higher_is_worse",
        transfer_getter("mn"),
        lambda row, route: always_rhythmic(row, route)
        and integer(row, "mn_transfer_anchor_count") not in (None, 0),
    ),
    Endpoint(
        "rg_pf_latency_mean_ms", "secondary_propagation", "higher_is_longer",
        field_getter("rg_pf_latency_mean_ms"), always_rhythmic,
    ),
    Endpoint(
        "rg_mn_latency_mean_ms", "secondary_propagation", "higher_is_longer",
        field_getter("rg_mn_latency_mean_ms"), always_rhythmic,
    ),
)


TERMINAL_ENDPOINTS = (
    Endpoint(
        "intended_route_mt_support", "mechanistic_terminal_state",
        "mechanistic_no_favorable_direction", state_getter("mt"), always_rhythmic,
    ),
    Endpoint(
        "intended_route_rrp", "mechanistic_terminal_state",
        "mechanistic_no_favorable_direction", state_getter("rrp"), always_rhythmic,
    ),
    Endpoint(
        "intended_route_replenishment_resource", "mechanistic_terminal_state",
        "mechanistic_no_favorable_direction",
        state_getter("replenishment_resource"), always_rhythmic,
    ),
)


def validate_sources(spec: Mapping[str, Any]) -> tuple[Path, dict[str, str]]:
    source_dir = (ROOT / spec["source_directory"]).resolve()
    expected = dict(spec["source_sha256"])
    actual: dict[str, str] = {}
    for name, expected_hash in expected.items():
        path = source_dir / name
        if not path.is_file():
            raise RuntimeError(f"missing frozen source: {path}")
        actual_hash = sha256_file(path)
        actual[name] = actual_hash
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"frozen source hash mismatch for {name}: {actual_hash} != {expected_hash}"
            )
    return source_dir, actual


def validate_global_matrix(
    spec: Mapping[str, Any],
    task_rows: Sequence[Mapping[str, str]],
    metric_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    expected = spec["expected_counts"]
    if len(task_rows) != expected["analysis_tasks"]:
        raise RuntimeError("analysis task count mismatch")
    if len(metric_rows) != expected["analysis_tasks"]:
        raise RuntimeError("metrics task count mismatch")
    task_ids = [row["task_id"] for row in task_rows]
    metric_ids = [row["task_id"] for row in metric_rows]
    if len(set(task_ids)) != len(task_ids) or len(set(metric_ids)) != len(metric_ids):
        raise RuntimeError("task_id is not unique")
    if set(task_ids) != set(metric_ids):
        raise RuntimeError("task index and metrics task IDs differ")
    unique_simulations = {row["simulation_id"] for row in task_rows}
    if len(unique_simulations) != expected["unique_simulations"]:
        raise RuntimeError("global unique simulation count mismatch")
    stage_tasks = Counter(row["stage"] for row in task_rows)
    if dict(stage_tasks) != expected["stage_tasks"]:
        raise RuntimeError(f"stage task counts mismatch: {dict(stage_tasks)}")
    stage_unique = {
        stage: len({row["simulation_id"] for row in task_rows if row["stage"] == stage})
        for stage in expected["stage_unique_simulations"]
    }
    if stage_unique != expected["stage_unique_simulations"]:
        raise RuntimeError(f"stage unique simulation counts mismatch: {stage_unique}")
    return {
        "analysis_task_count": len(task_rows),
        "unique_simulation_count": len(unique_simulations),
        "stage_task_counts": dict(stage_tasks),
        "stage_unique_simulation_counts": stage_unique,
    }


def validate_metrics_contract(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    invalid_scientific = 0
    invalid_technical = 0
    transfer_checks = 0
    for row in rows:
        if not truthy(row["scientific_valid"]):
            invalid_scientific += 1
        if integer(row, "technical_valid") != 1:
            invalid_technical += 1
        for prefix in ("pf", "mn"):
            anchor = integer(row, f"{prefix}_transfer_anchor_count")
            missed = integer(row, f"{prefix}_transfer_missed_count")
            matched = integer(row, f"{prefix}_transfer_matched_count")
            if anchor is None and missed is None and matched is None:
                continue
            if None in (anchor, missed, matched) or matched + missed != anchor:
                raise RuntimeError(
                    f"transfer identity failed for {row['task_id']} {prefix}"
                )
            transfer_checks += 1
    if invalid_scientific or invalid_technical:
        raise RuntimeError(
            f"invalid canonical rows: scientific={invalid_scientific}, technical={invalid_technical}"
        )
    return {
        "scientific_invalid_count": invalid_scientific,
        "technical_invalid_count": invalid_technical,
        "matched_plus_missed_equals_anchor_checks": transfer_checks,
    }


def join_design_and_metrics(
    task_rows: Sequence[Mapping[str, str]],
    metric_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    by_task = {row["task_id"]: row for row in task_rows}
    joined: list[dict[str, str]] = []
    design_fields = (
        "simulation_id", "stage", "seed", "protocol", "speed", "load",
        "load_side", "pulse", "ablations", "mt_mode", "impaired_mt_routes",
        "challenged_routes", "fast_mode", "label",
    )
    for metric in metric_rows:
        task = by_task[metric["task_id"]]
        for field in design_fields:
            if metric[field] != task[field]:
                raise RuntimeError(
                    f"design mismatch for task {metric['task_id']} field {field}"
                )
        joined.append(dict(metric))
    return joined


def base_atomic_fields(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "seed": int(row["seed"]),
        "structural_seed": 160601,
        "protocol": row["protocol"],
        "speed": row["speed"],
        "load": row["load"],
        "load_side": row["load_side"],
        "pulse": row["pulse"],
        "context_id": context_id(row),
    }


def recovery_raw_fields(prefix: str, row: Mapping[str, str]) -> dict[str, Any]:
    """Carry the full-follow-up recovery event and censor record unchanged."""
    return {
        f"{prefix}_pulse_required": integer(row, "pulse_required"),
        f"{prefix}_pulse_delivered": integer(row, "pulse_delivered"),
        f"{prefix}_pulse_response_eligible": integer(row, "pulse_response_eligible"),
        f"{prefix}_pulse_noneligibility_reason": row["pulse_noneligibility_reason"],
        f"{prefix}_recovery_endpoint_eligible": integer(
            row, "recovery_endpoint_eligible"
        ),
        f"{prefix}_recovery_ineligibility_reason": row[
            "recovery_ineligibility_reason"
        ],
        f"{prefix}_recovery_event_observed": integer(row, "recovery_event_observed"),
        f"{prefix}_recovery_time_s": finite_float(row, "recovery_time_s"),
        f"{prefix}_recovery_time_or_censor_s": finite_float(
            row, "recovery_time_or_censor_s"
        ),
        f"{prefix}_recovery_censor_time_s": finite_float(
            row, "recovery_censor_time_s"
        ),
    }


def event_count_raw_fields(prefix: str, row: Mapping[str, str]) -> dict[str, Any]:
    """Carry raw numerators and denominators beside all derived rates."""
    return {
        f"{prefix}_lr_phase_slip_count": integer(row, "lr_phase_slip_count"),
        f"{prefix}_lr_phase_cycle_count": integer(row, "lr_phase_cycle_count"),
        f"{prefix}_fe_phase_slip_count": integer(row, "fe_phase_slip_count"),
        f"{prefix}_fe_phase_cycle_count": integer(row, "fe_phase_cycle_count"),
        f"{prefix}_pf_transfer_anchor_count": integer(row, "pf_transfer_anchor_count"),
        f"{prefix}_pf_transfer_missed_count": integer(row, "pf_transfer_missed_count"),
        f"{prefix}_pf_transfer_matched_count": integer(row, "pf_transfer_matched_count"),
        f"{prefix}_mn_transfer_anchor_count": integer(row, "mn_transfer_anchor_count"),
        f"{prefix}_mn_transfer_missed_count": integer(row, "mn_transfer_missed_count"),
        f"{prefix}_mn_transfer_matched_count": integer(row, "mn_transfer_matched_count"),
    }


def build_e_atomic(
    a_rows: Sequence[Mapping[str, str]],
    e_rows: Sequence[Mapping[str, str]],
    routes: Sequence[str],
    endpoints: Sequence[Endpoint],
    domain: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    a_by_context = {context_key(row): row for row in a_rows}
    if len(a_by_context) != len(a_rows):
        raise RuntimeError("Stage A context key is not unique")
    atomic: list[dict[str, Any]] = []
    pair_inventory: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for intervention in e_rows:
        route_members = split_members(intervention["impaired_mt_routes"])
        if len(route_members) != 1 or route_members[0] not in routes:
            raise RuntimeError(f"invalid Stage E impaired route: {route_members}")
        route = route_members[0]
        key = context_key(intervention)
        if (route, key) in seen:
            raise RuntimeError("duplicate Stage E route/context pair")
        seen.add((route, key))
        reference = a_by_context.get(key)
        if reference is None:
            raise RuntimeError(f"Stage E lacks matched A reference: {route} {key}")
        transition = failure_transition(reference, intervention)
        common = {
            **base_atomic_fields(intervention),
            "analysis_stage": "E",
            "analysis_domain": domain,
            "intended_route": route,
            "reference_task_id": reference["task_id"],
            "reference_simulation_id": reference["simulation_id"],
            "intervention_task_id": intervention["task_id"],
            "intervention_simulation_id": intervention["simulation_id"],
            "reference_rhythmic_failure": integer(reference, "rhythmic_failure"),
            "intervention_rhythmic_failure": integer(intervention, "rhythmic_failure"),
            "failure_transition": transition,
            **recovery_raw_fields("reference", reference),
            **recovery_raw_fields("intervention", intervention),
            **event_count_raw_fields("reference", reference),
            **event_count_raw_fields("intervention", intervention),
        }
        pair_inventory.append(common)
        for endpoint in endpoints:
            reference_value = endpoint.getter(reference, route)
            intervention_value = endpoint.getter(intervention, route)
            complete = (
                endpoint.eligible(reference, route)
                and endpoint.eligible(intervention, route)
                and reference_value is not None
                and intervention_value is not None
            )
            atomic.append({
                **common,
                "endpoint": endpoint.name,
                "endpoint_role": endpoint.role,
                "direction_semantics": endpoint.direction,
                "reference_value": reference_value,
                "intervention_value": intervention_value,
                "complete_pair": int(complete),
                "paired_delta_intervention_minus_reference": (
                    intervention_value - reference_value if complete else None
                ),
            })
    expected_pairs = len(routes) * 27
    if len(seen) != expected_pairs:
        raise RuntimeError(f"Stage E pair count {len(seen)} != {expected_pairs}")
    return atomic, pair_inventory


def summarize_e_atomic(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row["analysis_domain"]), str(row["intended_route"]),
            str(row["pulse"]), str(row["endpoint"]), str(row["direction_semantics"]),
        )].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        values = [
            float(row["paired_delta_intervention_minus_reference"])
            for row in group if int(row["complete_pair"]) == 1
        ]
        domain, route, pulse, endpoint, semantics = key
        output.append({
            "analysis_stage": "E",
            "analysis_domain": domain,
            "intended_route": route,
            "pulse": pulse,
            "endpoint": endpoint,
            "direction_semantics": semantics,
            "design_cell_count": len(group),
            "complete_pair_count": len(values),
            **quantile_summary(values),
            **direction_counts(values),
            "summary_is_design_grid_not_sampling_uncertainty": 1,
        })
    return output


def summarize_e_failures(
    pair_inventory: Sequence[Mapping[str, Any]],
    routes: Sequence[str],
    pulses: Sequence[str],
    transition_order: Sequence[str],
) -> list[dict[str, Any]]:
    counts = Counter(
        (row["intended_route"], row["pulse"], row["failure_transition"])
        for row in pair_inventory
    )
    return [
        {
            "analysis_stage": "E",
            "intended_route": route,
            "pulse": pulse,
            "failure_transition": transition,
            "design_cell_count": counts[(route, pulse, transition)],
            "denominator_speed_by_load_cells": 9,
        }
        for route in routes
        for pulse in pulses
        for transition in transition_order
    ]


FACTORIAL_LABEL = re.compile(r"^factorial_(.+)_M_(.+)_A([01])M([01])$")


def build_f_groups(
    f_rows: Sequence[Mapping[str, str]],
    classes: Sequence[str],
    routes: Sequence[str],
) -> dict[tuple[str, str, tuple[str, ...]], dict[str, Mapping[str, str]]]:
    groups: dict[
        tuple[str, str, tuple[str, ...]], dict[str, Mapping[str, str]]
    ] = defaultdict(dict)
    for row in f_rows:
        match = FACTORIAL_LABEL.fullmatch(row["label"])
        if match is None:
            raise RuntimeError(f"invalid Stage F label: {row['label']}")
        intended_class, route, ablation_on, mt_on = match.groups()
        if intended_class not in classes or route not in routes:
            raise RuntimeError(f"unknown Stage F class/route: {intended_class}/{route}")
        expected_ablations = (intended_class,) if ablation_on == "1" else ()
        expected_routes = (route,) if mt_on == "1" else ()
        if split_members(row["ablations"]) != expected_ablations:
            raise RuntimeError(f"Stage F ablation arm mismatch: {row['task_id']}")
        if split_members(row["impaired_mt_routes"]) != expected_routes:
            raise RuntimeError(f"Stage F route arm mismatch: {row['task_id']}")
        arm = f"A{ablation_on}M{mt_on}"
        key = (intended_class, route, context_key(row))
        if arm in groups[key]:
            raise RuntimeError(f"duplicate Stage F arm: {key} {arm}")
        groups[key][arm] = row
    expected_arms = {"A0M0", "A1M0", "A0M1", "A1M1"}
    bad = {key: set(arms) for key, arms in groups.items() if set(arms) != expected_arms}
    if bad:
        first = next(iter(bad.items()))
        raise RuntimeError(f"incomplete Stage F cell: {first}")
    if len(groups) != len(classes) * len(routes) * 27:
        raise RuntimeError(f"Stage F cell count {len(groups)} is not 2700")
    return dict(groups)


def f_failure_pattern(arms: Mapping[str, Mapping[str, str]]) -> str:
    return "|".join(
        f"{arm}:{'failure' if not rhythmic(arms[arm]) else 'rhythmic'}"
        for arm in ("A0M0", "A1M0", "A0M1", "A1M1")
    )


def f_recovery_eligibility_pattern(arms: Mapping[str, Mapping[str, str]]) -> str:
    if arms["A0M0"]["pulse"] == "none":
        return "not_applicable_no_pulse"
    return "|".join(
        f"{arm}:{'eligible' if integer(arms[arm], 'recovery_endpoint_eligible') == 1 else 'ineligible'}"
        for arm in ("A0M0", "A1M0", "A0M1", "A1M1")
    )


def build_f_atomic(
    groups: Mapping[
        tuple[str, str, tuple[str, ...]], Mapping[str, Mapping[str, str]]
    ],
    endpoints: Sequence[Endpoint],
    domain: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    atomic: list[dict[str, Any]] = []
    cell_inventory: list[dict[str, Any]] = []
    arm_order = ("A0M0", "A1M0", "A0M1", "A1M1")
    for (intended_class, route, _key), arms in sorted(groups.items()):
        representative = arms["A0M0"]
        pattern = f_failure_pattern(arms)
        recovery_eligibility_pattern = f_recovery_eligibility_pattern(arms)
        common: dict[str, Any] = {
            **base_atomic_fields(representative),
            "analysis_stage": "F",
            "analysis_domain": domain,
            "intended_class": intended_class,
            "intended_route": route,
            "matrix_position": "diagonal" if intended_class == route else "off_diagonal",
            "failure_pattern": pattern,
            "recovery_eligibility_pattern": recovery_eligibility_pattern,
            "all_four_arms_rhythmic": int(all(rhythmic(arms[arm]) for arm in arm_order)),
            "unique_simulation_ids_within_factorial_cell": len({
                arms[arm]["simulation_id"] for arm in arm_order
            }),
        }
        for arm in arm_order:
            common[f"{arm.lower()}_task_id"] = arms[arm]["task_id"]
            common[f"{arm.lower()}_simulation_id"] = arms[arm]["simulation_id"]
            common[f"{arm.lower()}_rhythmic_failure"] = integer(
                arms[arm], "rhythmic_failure"
            )
            common.update(recovery_raw_fields(arm.lower(), arms[arm]))
            common.update(event_count_raw_fields(arm.lower(), arms[arm]))
        cell_inventory.append(common)
        for endpoint in endpoints:
            values = {arm: endpoint.getter(arms[arm], route) for arm in arm_order}
            complete = all(
                endpoint.eligible(arms[arm], route) and values[arm] is not None
                for arm in arm_order
            )
            interaction = None
            if complete:
                interaction = (
                    values["A1M1"] - values["A1M0"]
                    - values["A0M1"] + values["A0M0"]
                )
            atomic.append({
                **common,
                "endpoint": endpoint.name,
                "endpoint_role": endpoint.role,
                "direction_semantics": endpoint.direction,
                "a0m0_value": values["A0M0"],
                "a1m0_value": values["A1M0"],
                "a0m1_value": values["A0M1"],
                "a1m1_value": values["A1M1"],
                "complete_four_arm_cell": int(complete),
                "nonadditivity_a1m1_minus_a1m0_minus_a0m1_plus_a0m0": interaction,
            })
    return atomic, cell_inventory


def summarize_f_atomic(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row["analysis_domain"]), str(row["intended_class"]),
            str(row["intended_route"]), str(row["matrix_position"]),
            str(row["pulse"]), str(row["endpoint"]),
            str(row["direction_semantics"]),
        )].append(row)
    output: list[dict[str, Any]] = []
    interaction_field = "nonadditivity_a1m1_minus_a1m0_minus_a0m1_plus_a0m0"
    for key, group in sorted(grouped.items()):
        values = [
            float(row[interaction_field])
            for row in group if int(row["complete_four_arm_cell"]) == 1
        ]
        domain, intended_class, route, position, pulse, endpoint, semantics = key
        output.append({
            "analysis_stage": "F",
            "analysis_domain": domain,
            "intended_class": intended_class,
            "intended_route": route,
            "matrix_position": position,
            "pulse": pulse,
            "endpoint": endpoint,
            "direction_semantics": semantics,
            "design_cell_count": len(group),
            "complete_four_arm_cell_count": len(values),
            **quantile_summary(values),
            **direction_counts(values),
            "summary_is_design_grid_not_sampling_uncertainty": 1,
        })
    return output


def summarize_f_diagonal(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row["analysis_domain"]), str(row["matrix_position"]),
            str(row["pulse"]), str(row["endpoint"]),
            str(row["direction_semantics"]),
        )].append(row)
    interaction_field = "nonadditivity_a1m1_minus_a1m0_minus_a0m1_plus_a0m0"
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        values = [
            float(row[interaction_field])
            for row in group if int(row["complete_four_arm_cell"]) == 1
        ]
        domain, position, pulse, endpoint, semantics = key
        output.append({
            "analysis_stage": "F",
            "analysis_domain": domain,
            "matrix_position": position,
            "pulse": pulse,
            "endpoint": endpoint,
            "direction_semantics": semantics,
            "design_cell_count": len(group),
            "complete_four_arm_cell_count": len(values),
            **quantile_summary(values),
            **direction_counts(values),
            "summary_is_design_grid_not_sampling_uncertainty": 1,
        })
    return output


def summarize_f_failure_patterns(
    cell_inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    counts = Counter(
        (
            row["intended_class"], row["intended_route"], row["matrix_position"],
            row["pulse"], row["failure_pattern"],
            row["recovery_eligibility_pattern"],
        )
        for row in cell_inventory
    )
    return [
        {
            "analysis_stage": "F",
            "intended_class": key[0],
            "intended_route": key[1],
            "matrix_position": key[2],
            "pulse": key[3],
            "failure_pattern": key[4],
            "recovery_eligibility_pattern": key[5],
            "design_cell_count": count,
            "denominator_speed_by_load_cells": 9,
        }
        for key, count in sorted(counts.items())
    ]


def forbidden_headers(paths: Iterable[Path], forbidden: Sequence[str]) -> list[dict[str, str]]:
    forbidden_set = {name.lower() for name in forbidden}
    findings: list[dict[str, str]] = []
    for path in paths:
        if path.suffix != ".csv":
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        for field in header:
            if field.lower() in forbidden_set:
                findings.append({"file": path.name, "field": field})
    return findings


def main() -> None:
    spec = read_json(SPEC_PATH)
    if spec.get("status") != "locked":
        raise RuntimeError("analysis protocol is not locked")
    source_dir, source_hashes = validate_sources(spec)
    generated_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    script_sha256 = sha256_file(Path(__file__).resolve())

    design_fields = {
        "task_id", "simulation_id", "reuse_count", "stage", "seed", "protocol",
        "speed", "load", "load_side", "pulse", "ablations", "mt_mode",
        "impaired_mt_routes", "challenged_routes", "fast_mode", "label",
    }
    metric_fields = set(design_fields) | {
        "scientific_valid", "technical_valid", "technical_exclusion_reason",
        "rhythmic_failure", "lr_phase_error_mean_abs_deg",
        "fe_phase_error_mean_abs_deg", "lr_phase_slip_count",
        "lr_phase_cycle_count", "fe_phase_slip_count", "fe_phase_cycle_count",
        "pulse_required", "pulse_delivered", "pulse_response_eligible",
        "pulse_noneligibility_reason", "recovery_endpoint_eligible",
        "recovery_ineligibility_reason",
        "recovery_event_observed", "recovery_time_s",
        "recovery_time_or_censor_s", "recovery_censor_time_s", "frequency_hz",
        "rg_cycle_interval_cv_mean", "bilateral_amplitude_imbalance",
        "pf_transfer_anchor_count", "pf_transfer_missed_count",
        "pf_transfer_matched_count", "mn_transfer_anchor_count",
        "mn_transfer_missed_count", "mn_transfer_matched_count",
        "rg_pf_latency_mean_ms", "rg_mn_latency_mean_ms",
    }
    for route in spec["axes"]["routes"]:
        metric_fields.update({
            f"mt_{route}_mean", f"rrp_{route}_mean",
            f"replenishment_resource_{route}_mean",
        })

    task_rows = read_csv_selected(source_dir / "analysis_task_index.csv", design_fields)
    metric_rows = read_csv_selected(source_dir / "metrics.csv", metric_fields)
    matrix_qc = validate_global_matrix(spec, task_rows, metric_rows)
    metric_qc = validate_metrics_contract(metric_rows)
    rows = join_design_and_metrics(task_rows, metric_rows)

    a_rows = [row for row in rows if row["stage"] == "A"]
    e_rows = [row for row in rows if row["stage"] == "E"]
    f_rows = [row for row in rows if row["stage"] == "F"]
    routes = tuple(spec["axes"]["routes"])
    classes = tuple(spec["axes"]["classes"])
    pulses = tuple(spec["axes"]["pulse"])

    e_phase, e_inventory = build_e_atomic(
        a_rows, e_rows, routes, PHASE_RECOVERY_ENDPOINTS, "phase_recovery"
    )
    e_propagation, e_inventory_2 = build_e_atomic(
        a_rows, e_rows, routes, PROPAGATION_ENDPOINTS, "propagation_performance"
    )
    e_terminal, e_inventory_3 = build_e_atomic(
        a_rows, e_rows, routes, TERMINAL_ENDPOINTS, "terminal_state"
    )
    inventory_ids = lambda inventory: {
        (row["intended_route"], row["context_id"], row["intervention_task_id"])
        for row in inventory
    }
    if not (
        inventory_ids(e_inventory)
        == inventory_ids(e_inventory_2)
        == inventory_ids(e_inventory_3)
    ):
        raise RuntimeError("Stage E domain inventories diverged")
    e_phase_summary = summarize_e_atomic(e_phase)
    e_propagation_summary = summarize_e_atomic(e_propagation)
    e_terminal_summary = summarize_e_atomic(e_terminal)
    e_failures = summarize_e_failures(
        e_inventory, routes, pulses, spec["failure_policy"]["failure_transition_order"]
    )

    f_groups = build_f_groups(f_rows, classes, routes)
    f_phase, f_inventory = build_f_atomic(
        f_groups, PHASE_RECOVERY_ENDPOINTS, "phase_recovery"
    )
    f_propagation, f_inventory_2 = build_f_atomic(
        f_groups, PROPAGATION_ENDPOINTS, "propagation_performance"
    )
    f_terminal, f_inventory_3 = build_f_atomic(
        f_groups, TERMINAL_ENDPOINTS, "terminal_state"
    )
    f_inventory_key = lambda inventory: {
        (row["intended_class"], row["intended_route"], row["context_id"])
        for row in inventory
    }
    if not (
        f_inventory_key(f_inventory)
        == f_inventory_key(f_inventory_2)
        == f_inventory_key(f_inventory_3)
    ):
        raise RuntimeError("Stage F domain inventories diverged")
    f_phase_summary = summarize_f_atomic(f_phase)
    f_propagation_summary = summarize_f_atomic(f_propagation)
    f_terminal_summary = summarize_f_atomic(f_terminal)
    f_diagonal = summarize_f_diagonal(f_phase + f_propagation + f_terminal)
    f_failures = summarize_f_failure_patterns(f_inventory)

    outputs: list[tuple[str, Sequence[Mapping[str, Any]]]] = [
        ("e_f_e_phase_recovery_atomic.csv", e_phase),
        ("e_f_e_phase_recovery_summary.csv", e_phase_summary),
        ("e_f_e_propagation_atomic.csv", e_propagation),
        ("e_f_e_propagation_summary.csv", e_propagation_summary),
        ("e_f_e_terminal_state_atomic.csv", e_terminal),
        ("e_f_e_terminal_state_summary.csv", e_terminal_summary),
        ("e_f_e_failure_transitions.csv", e_failures),
        ("e_f_f_phase_recovery_atomic.csv", f_phase),
        ("e_f_f_phase_recovery_summary.csv", f_phase_summary),
        ("e_f_f_propagation_atomic.csv", f_propagation),
        ("e_f_f_propagation_summary.csv", f_propagation_summary),
        ("e_f_f_terminal_state_atomic.csv", f_terminal),
        ("e_f_f_terminal_state_summary.csv", f_terminal_summary),
        ("e_f_f_diagonal_off_diagonal_summary.csv", f_diagonal),
        ("e_f_f_failure_patterns.csv", f_failures),
    ]
    DERIVED.mkdir(parents=True, exist_ok=True)
    output_rows: dict[str, int] = {}
    output_paths: list[Path] = []
    for name, data in outputs:
        path = DERIVED / name
        output_rows[name] = write_csv(
            path,
            data,
            generated_utc=generated_utc,
            script_sha256=script_sha256,
            source_hashes=source_hashes,
        )
        output_paths.append(path)

    f_class_route_contexts = {
        (key[0], key[1], key[2]) for key in f_groups
    }
    f_unique_simulations = {row["simulation_id"] for row in f_rows}
    f_diagonal_cells = sum(key[0] == key[1] for key in f_groups)
    f_off_diagonal_cells = len(f_groups) - f_diagonal_cells
    forbidden = forbidden_headers(output_paths, spec["forbidden_inference_fields"])
    qc_checks = {
        "source_hashes_match_locked_spec": True,
        "global_matrix_contract_pass": True,
        "metrics_validity_contract_pass": True,
        "stage_e_task_count": len(e_rows),
        "stage_e_exact_context_matched_pairs": len(e_inventory),
        "stage_e_route_by_context_expected": len(routes) * 27,
        "stage_f_task_count": len(f_rows),
        "stage_f_unique_simulation_count": len(f_unique_simulations),
        "stage_f_factorial_cells": len(f_groups),
        "stage_f_class_route_context_key_count": len(f_class_route_contexts),
        "stage_f_four_arm_complete_cell_count": sum(
            set(arms) == {"A0M0", "A1M0", "A0M1", "A1M1"}
            for arms in f_groups.values()
        ),
        "stage_f_diagonal_context_cells": f_diagonal_cells,
        "stage_f_off_diagonal_context_cells": f_off_diagonal_cells,
        "stage_f_expected_diagonal_context_cells": 10 * 27,
        "stage_f_expected_off_diagonal_context_cells": 90 * 27,
        "forbidden_inference_header_findings": forbidden,
        "simulation_id_treated_as_replication": False,
    }
    qc_pass = (
        len(e_rows) == 270
        and len(e_inventory) == 270
        and len(f_rows) == 10800
        and len(f_unique_simulations) == 3267
        and len(f_groups) == 2700
        and qc_checks["stage_f_four_arm_complete_cell_count"] == 2700
        and f_diagonal_cells == 270
        and f_off_diagonal_cells == 2430
        and not forbidden
    )
    qc_payload = {
        "analysis_version": SCRIPT_VERSION,
        "generated_utc": generated_utc,
        "script_sha256": script_sha256,
        "source_sha256": source_hashes,
        "protocol_spec_sha256": sha256_file(SPEC_PATH),
        "output_row_count": 1,
        "independent_stochastic_realization_count": 1,
        "inferential_scope": "conditional_on_one_frozen_realization_no_population_inference",
        "qc_pass": qc_pass,
        "matrix_qc": matrix_qc,
        "metric_qc": metric_qc,
        "checks": qc_checks,
        "derived_csv_row_counts": output_rows,
    }
    qc_path = DERIVED / "e_f_qc.json"
    write_json(qc_path, qc_payload)
    output_paths.append(qc_path)
    output_rows[qc_path.name] = 1
    if not qc_pass:
        raise RuntimeError("Stage E--F QC failed; see e_f_qc.json")

    manifest_files = [
        {
            "name": path.name,
            "sha256": sha256_file(path),
            "row_count": output_rows[path.name],
        }
        for path in output_paths
    ]
    manifest_payload = {
        "analysis_version": SCRIPT_VERSION,
        "generated_utc": generated_utc,
        "script_sha256": script_sha256,
        "source_sha256": source_hashes,
        "protocol_spec_sha256": sha256_file(SPEC_PATH),
        "output_row_count": 1,
        "independent_stochastic_realization_count": 1,
        "inferential_scope": "conditional_on_one_frozen_realization_no_population_inference",
        "files": manifest_files,
        "qc_file": qc_path.name,
        "qc_pass": True,
    }
    manifest_path = DERIVED / "e_f_manifest.json"
    write_json(manifest_path, manifest_payload)

    print(json.dumps({
        "analysis_version": SCRIPT_VERSION,
        "qc_pass": True,
        "stage_e_pairs": len(e_inventory),
        "stage_f_factorial_cells": len(f_groups),
        "stage_f_unique_simulations": len(f_unique_simulations),
        "derived_files": len(output_paths) + 1,
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
