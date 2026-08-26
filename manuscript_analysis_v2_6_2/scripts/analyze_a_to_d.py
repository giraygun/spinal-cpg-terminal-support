#!/usr/bin/env python3
"""Locked, descriptive A--D analysis for the v2.6.2 frozen realization.

This script never mutates the frozen simulator outputs.  It creates only
``derived/manuscript_analysis_v2_6_2/a_to_d_*`` files.  All comparisons are
exact, context-matched contrasts within the one frozen network realization;
no sampling-inference quantities are calculated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_VERSION = "1.0.0"
EPSILON = 1e-12

SCRIPT_PATH = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT_PATH.parent.parent
WORKSPACE_ROOT = ANALYSIS_DIR.parent
DERIVED_DIR = WORKSPACE_ROOT / "derived" / "manuscript_analysis_v2_6_2"
SPEC_PATH = ANALYSIS_DIR / "PROTOCOL_SPEC.json"
LOCKED_PROTOCOL_PATH = ANALYSIS_DIR / "ANALYSIS_PROTOCOL_LOCKED_2026-08-26.md"

FORBIDDEN_INFERENCE_FIELDS = {
    "p_value",
    "pvalue",
    "confidence_interval",
    "ci_lower",
    "ci_upper",
    "standard_error",
    "se",
    "degrees_of_freedom",
    "df",
    "cohen_d",
}

CORE_TASK_FIELDS = (
    "task_id",
    "simulation_id",
    "reuse_count",
    "stage",
    "seed",
    "protocol",
    "speed",
    "load",
    "load_side",
    "pulse",
    "ablations",
    "mt_mode",
    "impaired_mt_routes",
    "challenged_routes",
    "fast_mode",
    "label",
)


@dataclass(frozen=True)
class Endpoint:
    name: str
    family: str
    window: str
    unit: str
    direction_rule: str
    value_field: str | None = None
    numerator_field: str | None = None
    denominator_field: str | None = None
    applicability: str = "all"
    eligibility_field: str | None = None
    event_required: bool = False


ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint(
        "baseline_lr_phase_error_deg",
        "primary_phase",
        "whole_run",
        "deg",
        "higher_is_worse",
        value_field="lr_phase_error_mean_abs_deg",
    ),
    Endpoint(
        "baseline_fe_phase_error_deg",
        "primary_phase",
        "whole_run",
        "deg",
        "higher_is_worse",
        value_field="fe_phase_error_mean_abs_deg",
    ),
    Endpoint(
        "baseline_lr_phase_slip_rate",
        "primary_phase_slip",
        "whole_run",
        "proportion",
        "higher_is_worse",
        numerator_field="lr_phase_slip_count",
        denominator_field="lr_phase_cycle_count",
    ),
    Endpoint(
        "baseline_fe_phase_slip_rate",
        "primary_phase_slip",
        "whole_run",
        "proportion",
        "higher_is_worse",
        numerator_field="fe_phase_slip_count",
        denominator_field="fe_phase_cycle_count",
    ),
    Endpoint(
        "post_pulse_lr_phase_error_deg",
        "primary_phase",
        "post_pulse",
        "deg",
        "higher_is_worse",
        value_field="post_pulse_lr_phase_error_mean_abs_deg",
        applicability="active_pulse",
        eligibility_field="pulse_response_eligible",
    ),
    Endpoint(
        "post_pulse_fe_phase_error_deg",
        "primary_phase",
        "post_pulse",
        "deg",
        "higher_is_worse",
        value_field="post_pulse_fe_phase_error_mean_abs_deg",
        applicability="active_pulse",
        eligibility_field="pulse_response_eligible",
    ),
    Endpoint(
        "post_pulse_lr_phase_slip_rate",
        "primary_phase_slip",
        "post_pulse",
        "proportion",
        "higher_is_worse",
        numerator_field="post_pulse_lr_phase_slip_count",
        denominator_field="post_pulse_lr_phase_cycle_count",
        applicability="active_pulse",
        eligibility_field="pulse_response_eligible",
    ),
    Endpoint(
        "post_pulse_fe_phase_slip_rate",
        "primary_phase_slip",
        "post_pulse",
        "proportion",
        "higher_is_worse",
        numerator_field="post_pulse_fe_phase_slip_count",
        denominator_field="post_pulse_fe_phase_cycle_count",
        applicability="active_pulse",
        eligibility_field="pulse_response_eligible",
    ),
    Endpoint(
        "frequency_hz",
        "secondary_rhythm",
        "whole_run",
        "Hz",
        "not_ordered",
        value_field="frequency_hz",
    ),
    Endpoint(
        "rg_cycle_interval_cv",
        "secondary_rhythm",
        "whole_run",
        "ratio",
        "higher_is_worse",
        value_field="rg_cycle_interval_cv_mean",
    ),
    Endpoint(
        "bilateral_amplitude_imbalance",
        "secondary_motor_balance",
        "whole_run",
        "ratio",
        "higher_is_worse",
        value_field="bilateral_amplitude_imbalance",
    ),
    Endpoint(
        "pf_missed_propagation_fraction",
        "secondary_network_propagation",
        "whole_run",
        "proportion",
        "higher_is_worse",
        numerator_field="pf_transfer_missed_count",
        denominator_field="pf_transfer_anchor_count",
    ),
    Endpoint(
        "mn_missed_propagation_fraction",
        "secondary_network_propagation",
        "whole_run",
        "proportion",
        "higher_is_worse",
        numerator_field="mn_transfer_missed_count",
        denominator_field="mn_transfer_anchor_count",
    ),
    Endpoint(
        "rg_pf_latency_ms",
        "secondary_network_propagation",
        "whole_run",
        "ms",
        "not_ordered",
        value_field="rg_pf_latency_mean_ms",
    ),
    Endpoint(
        "rg_mn_latency_ms",
        "secondary_network_propagation",
        "whole_run",
        "ms",
        "not_ordered",
        value_field="rg_mn_latency_mean_ms",
    ),
    Endpoint(
        "recovery_observed_time_s",
        "primary_recovery",
        "post_pulse",
        "s",
        "higher_is_worse",
        value_field="recovery_time_s",
        applicability="active_pulse",
        eligibility_field="recovery_endpoint_eligible",
        event_required=True,
    ),
)

PARTICIPATION_FIELDS: tuple[str, ...] = (
    "RG_mean_rate_hz",
    "PF_mean_rate_hz",
    "MN_mean_rate_hz",
    "V0D_mean_rate_hz",
    "V0V_mean_rate_hz",
    "V2a_mean_rate_hz",
    "V3_mean_rate_hz",
    "V1Ia_mean_rate_hz",
    "V1Ren_mean_rate_hz",
    "V2b_mean_rate_hz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    raise ValueError(f"Unrecognized Boolean value: {value!r}")


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    result = float(text)
    if not math.isfinite(result):
        return None
    return result


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    rounded = round(number)
    if abs(number - rounded) > EPSILON:
        raise ValueError(f"Expected integer-valued count, got {value!r}")
    return int(rounded)


def analysis_valid(row: Mapping[str, Any]) -> bool:
    return as_bool(row.get("scientific_valid")) and as_bool(row.get("technical_valid"))


def rhythmic(row: Mapping[str, Any]) -> bool:
    return analysis_valid(row) and not as_bool(row.get("rhythmic_failure"))


def failure_transition(reference: Mapping[str, Any], intervention: Mapping[str, Any]) -> str:
    if not analysis_valid(reference) or not analysis_valid(intervention):
        return "excluded_invalid"
    ref_failure = as_bool(reference.get("rhythmic_failure"))
    int_failure = as_bool(intervention.get("rhythmic_failure"))
    if not ref_failure and not int_failure:
        return "rhythmic_to_rhythmic"
    if not ref_failure and int_failure:
        return "rhythmic_to_failure"
    if ref_failure and not int_failure:
        return "failure_to_rhythmic"
    return "failure_to_failure"


def recovery_event_transition(reference: Mapping[str, Any], intervention: Mapping[str, Any]) -> str:
    if str(reference.get("pulse", "")) == "none" or str(intervention.get("pulse", "")) == "none":
        return "not_applicable_without_active_pulse"
    if not analysis_valid(reference) or not analysis_valid(intervention):
        return "excluded_invalid"
    if not rhythmic(reference) or not rhythmic(intervention):
        return "incomplete_due_to_rhythmic_failure"
    if not as_bool(reference.get("recovery_endpoint_eligible")):
        return "reference_recovery_ineligible"
    if not as_bool(intervention.get("recovery_endpoint_eligible")):
        return "intervention_recovery_ineligible"
    reference_event = as_bool(reference.get("recovery_event_observed"))
    intervention_event = as_bool(intervention.get("recovery_event_observed"))
    if reference_event and intervention_event:
        return "event_to_event"
    if reference_event and not intervention_event:
        return "event_to_no_event"
    if not reference_event and intervention_event:
        return "no_event_to_event"
    return "no_event_to_no_event"


@dataclass(frozen=True)
class Measurement:
    applicable: bool
    measurement_eligible: bool
    value: float | None
    numerator: int | None
    denominator: int | None
    reason: str


def endpoint_measurement(row: Mapping[str, Any], endpoint: Endpoint) -> Measurement:
    active_pulse = str(row.get("pulse", "")) in {"excitatory", "inhibitory"}
    if endpoint.applicability == "active_pulse" and not active_pulse:
        return Measurement(False, False, None, None, None, "not_applicable_without_active_pulse")

    if endpoint.eligibility_field and not as_bool(row.get(endpoint.eligibility_field)):
        return Measurement(True, False, None, None, None, f"{endpoint.eligibility_field}_false")
    if endpoint.event_required and not as_bool(row.get("recovery_event_observed")):
        return Measurement(True, False, None, None, None, "recovery_event_not_observed")

    if endpoint.value_field is not None:
        value = as_float(row.get(endpoint.value_field))
        if value is None:
            return Measurement(True, False, None, None, None, "missing_value")
        return Measurement(True, True, value, None, None, "complete")

    numerator = as_int(row.get(endpoint.numerator_field or ""))
    denominator = as_int(row.get(endpoint.denominator_field or ""))
    if numerator is None or denominator is None:
        return Measurement(True, False, None, numerator, denominator, "missing_count")
    if denominator <= 0:
        return Measurement(True, False, None, numerator, denominator, "zero_denominator")
    return Measurement(True, True, numerator / denominator, numerator, denominator, "complete")


def pair_complete(
    intervention: Mapping[str, Any],
    reference: Mapping[str, Any],
    int_measurement: Measurement,
    ref_measurement: Measurement,
) -> tuple[bool, str]:
    if not int_measurement.applicable or not ref_measurement.applicable:
        return False, "not_applicable"
    if not analysis_valid(intervention) or not analysis_valid(reference):
        return False, "invalid_arm"
    if not rhythmic(intervention) or not rhythmic(reference):
        return False, "rhythmic_failure_in_pair"
    if not int_measurement.measurement_eligible:
        return False, f"intervention_{int_measurement.reason}"
    if not ref_measurement.measurement_eligible:
        return False, f"reference_{ref_measurement.reason}"
    return True, "complete"


def direction_label(value: float | None, rule: str, *, did: bool = False) -> str:
    if value is None:
        return "not_calculated"
    if abs(value) <= EPSILON:
        return "neutral"
    if rule != "higher_is_worse":
        return "positive" if value > 0 else "negative"
    if did:
        return "higher_burden_at_high_speed" if value > 0 else "lower_burden_at_high_speed"
    return "deterioration" if value > 0 else "improvement"


def linear_percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def ordered_rows(rows: Iterable[dict[str, str]], spec: Mapping[str, Any]) -> list[dict[str, str]]:
    axes = spec["axes"]
    order_maps = {
        "speed": {value: index for index, value in enumerate(axes["speed"])},
        "load": {value: index for index, value in enumerate(axes["load"])},
        "pulse": {value: index for index, value in enumerate(axes["pulse"])},
        "ablations": {
            value: index
            for index, value in enumerate(
                ["none", *axes["single_interventions"], *axes["paired_interventions"]]
            )
        },
    }
    return sorted(
        rows,
        key=lambda row: (
            order_maps["ablations"].get(row.get("ablations", ""), 999),
            order_maps["pulse"].get(row.get("pulse", ""), 999),
            order_maps["speed"].get(row.get("speed", ""), 999),
            order_maps["load"].get(row.get("load", ""), 999),
            row.get("task_id", ""),
        ),
    )


def key_for(row: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in fields)


def unique_map(
    rows: Iterable[dict[str, str]], fields: Sequence[str], description: str
) -> dict[tuple[str, ...], dict[str, str]]:
    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = key_for(row, fields)
        if key in result:
            raise AssertionError(f"Duplicate {description} key: {key}")
        result[key] = row
    return result


def output_columns_are_allowed(fieldnames: Iterable[str]) -> None:
    normalized = {field.strip().lower() for field in fieldnames}
    forbidden = sorted(normalized & FORBIDDEN_INFERENCE_FIELDS)
    if forbidden:
        raise AssertionError(f"Forbidden inference fields in output: {forbidden}")


def main() -> None:
    with SPEC_PATH.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    source_dir = (ANALYSIS_DIR / spec["source_directory"]).resolve()
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    script_sha256 = sha256_file(SCRIPT_PATH)
    source_hashes: dict[str, dict[str, Any]] = {}
    for name, expected in spec["source_sha256"].items():
        path = source_dir / name
        actual = sha256_file(path)
        verified = actual == expected
        if not verified:
            raise AssertionError(f"Frozen source hash mismatch for {name}: {actual} != {expected}")
        source_hashes[name] = {"expected": expected, "actual": actual, "verified": verified}

    protocol_hashes = {
        "PROTOCOL_SPEC.json": sha256_file(SPEC_PATH),
        "ANALYSIS_PROTOCOL_LOCKED_2026-08-26.md": sha256_file(LOCKED_PROTOCOL_PATH),
    }

    index_rows, index_fields = read_csv(source_dir / "analysis_task_index.csv")
    metric_rows, metric_fields = read_csv(source_dir / "metrics.csv")
    unique_rows, _ = read_csv(source_dir / "unique_simulation_metrics.csv")
    epoch_rows, _ = read_csv(source_dir / "long_epoch_metrics.csv")

    expected = spec["expected_counts"]
    if len(index_rows) != expected["analysis_tasks"]:
        raise AssertionError("analysis_task_index.csv task count mismatch")
    if len(metric_rows) != expected["analysis_tasks"]:
        raise AssertionError("metrics.csv task count mismatch")
    if len(unique_rows) != expected["unique_simulations"]:
        raise AssertionError("unique_simulation_metrics.csv count mismatch")
    if len(epoch_rows) != expected["long_epoch_rows"]:
        raise AssertionError("long_epoch_metrics.csv row count mismatch")
    if len({row["task_id"] for row in index_rows}) != len(index_rows):
        raise AssertionError("analysis_task_index.csv task_id is not unique")
    if len({row["task_id"] for row in metric_rows}) != len(metric_rows):
        raise AssertionError("metrics.csv task_id is not unique")
    if len({row["simulation_id"] for row in unique_rows}) != len(unique_rows):
        raise AssertionError("unique_simulation_metrics.csv simulation_id is not unique")
    if len({(row["simulation_id"], row["epoch"]) for row in epoch_rows}) != len(epoch_rows):
        raise AssertionError("long_epoch_metrics.csv (simulation_id, epoch) is not unique")

    index_by_task = {row["task_id"]: row for row in index_rows}
    metrics_by_task = {row["task_id"]: row for row in metric_rows}
    if set(index_by_task) != set(metrics_by_task):
        raise AssertionError("Task identities differ between index and metrics")
    for task_id, index_row in index_by_task.items():
        metric_row = metrics_by_task[task_id]
        for field in CORE_TASK_FIELDS:
            if index_row[field] != metric_row[field]:
                raise AssertionError(f"Task field mismatch for {task_id}, {field}")

    simulation_ids = {row["simulation_id"] for row in index_rows}
    unique_simulation_ids = {row["simulation_id"] for row in unique_rows}
    if simulation_ids != unique_simulation_ids:
        raise AssertionError("Index and unique-simulation identity sets differ")

    transfer_identity_checks = 0
    for row in metric_rows:
        for prefix in ("pf", "mn"):
            anchor = as_int(row[f"{prefix}_transfer_anchor_count"])
            matched = as_int(row[f"{prefix}_transfer_matched_count"])
            missed = as_int(row[f"{prefix}_transfer_missed_count"])
            if anchor is None and matched is None and missed is None:
                continue
            if anchor is None or matched is None or missed is None:
                raise AssertionError(f"Incomplete transfer counts for task {row['task_id']}, {prefix}")
            if matched + missed != anchor:
                raise AssertionError(f"matched + missed != anchor for task {row['task_id']}, {prefix}")
            transfer_identity_checks += 1

    stage_rows: dict[str, list[dict[str, str]]] = {
        stage: [row for row in metric_rows if row["stage"] == stage] for stage in "ABCD"
    }
    for stage, rows in stage_rows.items():
        if len(rows) != expected["stage_tasks"][stage]:
            raise AssertionError(f"Stage {stage} task count mismatch")
        if len({row["simulation_id"] for row in rows}) != expected["stage_unique_simulations"][stage]:
            raise AssertionError(f"Stage {stage} unique simulation count mismatch")

    # Provenance columns are repeated deliberately so every CSV is self-describing.
    metadata = {
        "generated_at_utc": generated_at,
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_sha256,
        "source_analysis_task_index_sha256": source_hashes["analysis_task_index.csv"]["actual"],
        "source_metrics_sha256": source_hashes["metrics.csv"]["actual"],
        "source_unique_simulation_metrics_sha256": source_hashes["unique_simulation_metrics.csv"]["actual"],
        "source_long_epoch_metrics_sha256": source_hashes["long_epoch_metrics.csv"]["actual"],
        "source_experiment_plan_sha256": source_hashes["experiment_plan_single_realization_v2_6_2.json"]["actual"],
        "source_locked_contrasts_sha256": source_hashes["single_realization_contrasts_v2_6_2.csv"]["actual"],
        "protocol_spec_sha256": protocol_hashes["PROTOCOL_SPEC.json"],
        "locked_protocol_sha256": protocol_hashes["ANALYSIS_PROTOCOL_LOCKED_2026-08-26.md"],
    }
    output_manifest: dict[str, dict[str, Any]] = {}

    def write_output(filename: str, rows: list[dict[str, Any]]) -> None:
        if not filename.startswith("a_to_d_") or not filename.endswith(".csv"):
            raise AssertionError(f"Unexpected output name: {filename}")
        if not rows:
            raise AssertionError(f"Refusing to create empty output: {filename}")
        output_row_count = len(rows)
        fieldnames = list(rows[0].keys()) + list(metadata.keys()) + ["output_row_count"]
        output_columns_are_allowed(fieldnames)
        for row in rows:
            if list(row.keys()) != list(rows[0].keys()):
                raise AssertionError(f"Inconsistent row schema in {filename}")
        path = DERIVED_DIR / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, **metadata, "output_row_count": output_row_count})
        output_manifest[filename] = {
            "row_count": output_row_count,
            "sha256": sha256_file(path),
            "field_count": len(fieldnames),
            "forbidden_inference_fields_present": False,
        }

    # Stage inventory.
    inventory_rows: list[dict[str, Any]] = []
    for stage in "ABCD":
        rows = stage_rows[stage]
        inventory_rows.append(
            {
                "stage": stage,
                "task_count": len(rows),
                "expected_task_count": expected["stage_tasks"][stage],
                "unique_simulation_count": len({row["simulation_id"] for row in rows}),
                "expected_unique_simulation_count": expected["stage_unique_simulations"][stage],
                "analysis_valid_task_count": sum(analysis_valid(row) for row in rows),
                "scientific_invalid_task_count": sum(not as_bool(row["scientific_valid"]) for row in rows),
                "technical_invalid_task_count": sum(not as_bool(row["technical_valid"]) for row in rows),
                "rhythmic_failure_task_count": sum(as_bool(row["rhythmic_failure"]) for row in rows),
                "pulse_required_task_count": sum(as_bool(row["pulse_required"]) for row in rows),
                "pulse_delivered_task_count": sum(as_bool(row["pulse_delivered"]) for row in rows),
            }
        )
    write_output("a_to_d_stage_inventory.csv", inventory_rows)

    context_fields = ("seed", "protocol", "speed", "load", "load_side", "pulse")
    no_pulse_fields = ("seed", "protocol", "speed", "load", "load_side")
    a_rows = stage_rows["A"]
    a_map = unique_map(a_rows, context_fields, "Stage A context")
    a_none_rows = [row for row in a_rows if row["pulse"] == "none"]
    a_none_map = unique_map(a_none_rows, no_pulse_fields, "Stage A no-pulse context")
    if len(a_none_rows) != 9:
        raise AssertionError("Stage A intact no-pulse grid is not 3 x 3")

    a_grid_rows: list[dict[str, Any]] = []
    for row in ordered_rows(a_none_rows, spec):
        pf_anchor = as_int(row["pf_transfer_anchor_count"])
        pf_missed = as_int(row["pf_transfer_missed_count"])
        mn_anchor = as_int(row["mn_transfer_anchor_count"])
        mn_missed = as_int(row["mn_transfer_missed_count"])
        a_grid_rows.append(
            {
                "task_id": row["task_id"],
                "simulation_id": row["simulation_id"],
                "seed": row["seed"],
                "speed": row["speed"],
                "load": row["load"],
                "load_side": row["load_side"],
                "analysis_valid": int(analysis_valid(row)),
                "rhythmic_failure": int(as_bool(row["rhythmic_failure"])),
                "frequency_hz": as_float(row["frequency_hz"]),
                "lr_phase_error_mean_abs_deg": as_float(row["lr_phase_error_mean_abs_deg"]),
                "fe_phase_error_mean_abs_deg": as_float(row["fe_phase_error_mean_abs_deg"]),
                "lr_phase_slip_count": as_int(row["lr_phase_slip_count"]),
                "lr_phase_cycle_count": as_int(row["lr_phase_cycle_count"]),
                "lr_phase_slip_rate": endpoint_measurement(row, ENDPOINTS[2]).value,
                "fe_phase_slip_count": as_int(row["fe_phase_slip_count"]),
                "fe_phase_cycle_count": as_int(row["fe_phase_cycle_count"]),
                "fe_phase_slip_rate": endpoint_measurement(row, ENDPOINTS[3]).value,
                "rg_cycle_interval_cv_mean": as_float(row["rg_cycle_interval_cv_mean"]),
                "bilateral_amplitude_imbalance": as_float(row["bilateral_amplitude_imbalance"]),
                "pf_transfer_anchor_count": pf_anchor,
                "pf_transfer_matched_count": as_int(row["pf_transfer_matched_count"]),
                "pf_transfer_missed_count": pf_missed,
                "pf_missed_propagation_fraction": (pf_missed / pf_anchor) if pf_anchor else None,
                "mn_transfer_anchor_count": mn_anchor,
                "mn_transfer_matched_count": as_int(row["mn_transfer_matched_count"]),
                "mn_transfer_missed_count": mn_missed,
                "mn_missed_propagation_fraction": (mn_missed / mn_anchor) if mn_anchor else None,
                "rg_pf_latency_mean_ms": as_float(row["rg_pf_latency_mean_ms"]),
                "rg_mn_latency_mean_ms": as_float(row["rg_mn_latency_mean_ms"]),
            }
        )
    write_output("a_to_d_a_intact_no_pulse_grid.csv", a_grid_rows)

    # Stage A active pulse versus the direction-matched sham window in the
    # context-matched no-pulse simulation.
    a_pulse_rows: list[dict[str, Any]] = []
    a_pulse_summary_atoms: list[dict[str, Any]] = []
    for active in ordered_rows([row for row in a_rows if row["pulse"] != "none"], spec):
        reference = a_none_map[key_for(active, no_pulse_fields)]
        pulse = active["pulse"]
        sham_prefix = f"sham_{pulse}"
        active_eligible = as_bool(active["pulse_response_eligible"])
        sham_eligible = as_bool(reference[f"{sham_prefix}_endpoint_eligible"])
        common_complete = rhythmic(active) and rhythmic(reference) and active_eligible and sham_eligible

        active_lr = as_float(active["post_pulse_lr_phase_error_mean_abs_deg"])
        active_fe = as_float(active["post_pulse_fe_phase_error_mean_abs_deg"])
        sham_lr = as_float(reference[f"{sham_prefix}_lr_phase_error_mean_abs_deg"])
        sham_fe = as_float(reference[f"{sham_prefix}_fe_phase_error_mean_abs_deg"])

        active_lr_num = as_int(active["post_pulse_lr_phase_slip_count"])
        active_lr_den = as_int(active["post_pulse_lr_phase_cycle_count"])
        active_fe_num = as_int(active["post_pulse_fe_phase_slip_count"])
        active_fe_den = as_int(active["post_pulse_fe_phase_cycle_count"])
        sham_lr_num = as_int(reference[f"{sham_prefix}_lr_phase_slip_count"])
        sham_lr_den = as_int(reference[f"{sham_prefix}_lr_phase_cycle_count"])
        sham_fe_num = as_int(reference[f"{sham_prefix}_fe_phase_slip_count"])
        sham_fe_den = as_int(reference[f"{sham_prefix}_fe_phase_cycle_count"])

        lr_phase_complete = common_complete and active_lr is not None and sham_lr is not None
        fe_phase_complete = common_complete and active_fe is not None and sham_fe is not None
        lr_slip_complete = (
            common_complete
            and active_lr_num is not None
            and active_lr_den is not None
            and active_lr_den > 0
            and sham_lr_num is not None
            and sham_lr_den is not None
            and sham_lr_den > 0
        )
        fe_slip_complete = (
            common_complete
            and active_fe_num is not None
            and active_fe_den is not None
            and active_fe_den > 0
            and sham_fe_num is not None
            and sham_fe_den is not None
            and sham_fe_den > 0
        )
        active_lr_rate = (active_lr_num / active_lr_den) if active_lr_den else None
        sham_lr_rate = (sham_lr_num / sham_lr_den) if sham_lr_den else None
        active_fe_rate = (active_fe_num / active_fe_den) if active_fe_den else None
        sham_fe_rate = (sham_fe_num / sham_fe_den) if sham_fe_den else None
        lr_phase_delta = (active_lr - sham_lr) if lr_phase_complete else None
        fe_phase_delta = (active_fe - sham_fe) if fe_phase_complete else None
        lr_slip_delta = (active_lr_rate - sham_lr_rate) if lr_slip_complete else None
        fe_slip_delta = (active_fe_rate - sham_fe_rate) if fe_slip_complete else None

        pulse_row = {
            "active_task_id": active["task_id"],
            "active_simulation_id": active["simulation_id"],
            "sham_reference_task_id": reference["task_id"],
            "sham_reference_simulation_id": reference["simulation_id"],
            "seed": active["seed"],
            "speed": active["speed"],
            "load": active["load"],
            "load_side": active["load_side"],
            "pulse": pulse,
            "failure_transition": failure_transition(reference, active),
            "active_pulse_delivered": int(as_bool(active["pulse_delivered"])),
            "active_pulse_response_eligible": int(active_eligible),
            "sham_endpoint_eligible": int(sham_eligible),
            "active_post_lr_phase_error_deg": active_lr,
            "sham_lr_phase_error_deg": sham_lr,
            "lr_phase_complete_pair": int(lr_phase_complete),
            "lr_phase_delta_deg": lr_phase_delta,
            "lr_phase_direction": direction_label(lr_phase_delta, "higher_is_worse"),
            "active_post_fe_phase_error_deg": active_fe,
            "sham_fe_phase_error_deg": sham_fe,
            "fe_phase_complete_pair": int(fe_phase_complete),
            "fe_phase_delta_deg": fe_phase_delta,
            "fe_phase_direction": direction_label(fe_phase_delta, "higher_is_worse"),
            "active_post_lr_phase_slip_count": active_lr_num,
            "active_post_lr_phase_cycle_count": active_lr_den,
            "active_post_lr_phase_slip_rate": active_lr_rate,
            "sham_lr_phase_slip_count": sham_lr_num,
            "sham_lr_phase_cycle_count": sham_lr_den,
            "sham_lr_phase_slip_rate": sham_lr_rate,
            "lr_slip_complete_pair": int(lr_slip_complete),
            "lr_slip_rate_delta": lr_slip_delta,
            "lr_slip_direction": direction_label(lr_slip_delta, "higher_is_worse"),
            "active_post_fe_phase_slip_count": active_fe_num,
            "active_post_fe_phase_cycle_count": active_fe_den,
            "active_post_fe_phase_slip_rate": active_fe_rate,
            "sham_fe_phase_slip_count": sham_fe_num,
            "sham_fe_phase_cycle_count": sham_fe_den,
            "sham_fe_phase_slip_rate": sham_fe_rate,
            "fe_slip_complete_pair": int(fe_slip_complete),
            "fe_slip_rate_delta": fe_slip_delta,
            "fe_slip_direction": direction_label(fe_slip_delta, "higher_is_worse"),
            "recovery_eligible": int(as_bool(active["recovery_endpoint_eligible"])),
            "recovery_event_observed": int(as_bool(active["recovery_event_observed"])),
            "recovery_observed_time_s": (
                as_float(active["recovery_time_s"])
                if as_bool(active["recovery_event_observed"])
                else None
            ),
            "recovery_time_or_censor_s": as_float(active["recovery_time_or_censor_s"]),
            "recovery_censor_time_s": as_float(active["recovery_censor_time_s"]),
        }
        a_pulse_rows.append(pulse_row)
        for endpoint_name, value, complete in (
            ("post_pulse_lr_phase_error_deg", lr_phase_delta, lr_phase_complete),
            ("post_pulse_fe_phase_error_deg", fe_phase_delta, fe_phase_complete),
            ("post_pulse_lr_phase_slip_rate", lr_slip_delta, lr_slip_complete),
            ("post_pulse_fe_phase_slip_rate", fe_slip_delta, fe_slip_complete),
        ):
            a_pulse_summary_atoms.append(
                {
                    "analysis_block": "A_pulse_vs_matched_sham",
                    "effect_type": "active_pulse_minus_direction_matched_sham",
                    "intervention": "intact",
                    "pulse": pulse,
                    "endpoint": endpoint_name,
                    "direction_rule": "higher_is_worse",
                    "applicable": 1,
                    "complete": int(complete),
                    "value": value,
                }
            )
    if len(a_pulse_rows) != 18:
        raise AssertionError("Stage A active pulse comparison count is not 18")
    write_output("a_to_d_a_pulse_vs_matched_sham.csv", a_pulse_rows)

    # Generic exact context-matched B effects.
    b_effect_rows: list[dict[str, Any]] = []
    b_state_rows: list[dict[str, Any]] = []
    b_summary_atoms: list[dict[str, Any]] = []
    for intervention in ordered_rows(stage_rows["B"], spec):
        reference = a_map[key_for(intervention, context_fields)]
        b_state_rows.append(
            {
                "intervention_task_id": intervention["task_id"],
                "intervention_simulation_id": intervention["simulation_id"],
                "reference_task_id": reference["task_id"],
                "reference_simulation_id": reference["simulation_id"],
                "intervention": intervention["ablations"],
                "seed": intervention["seed"],
                "speed": intervention["speed"],
                "load": intervention["load"],
                "load_side": intervention["load_side"],
                "pulse": intervention["pulse"],
                "failure_transition": failure_transition(reference, intervention),
                "recovery_event_transition": recovery_event_transition(reference, intervention),
                "intervention_analysis_valid": int(analysis_valid(intervention)),
                "reference_analysis_valid": int(analysis_valid(reference)),
                "intervention_rhythmic_failure": int(as_bool(intervention["rhythmic_failure"])),
                "reference_rhythmic_failure": int(as_bool(reference["rhythmic_failure"])),
                "intervention_pulse_delivered": int(as_bool(intervention["pulse_delivered"])),
                "reference_pulse_delivered": int(as_bool(reference["pulse_delivered"])),
                "intervention_recovery_eligible": int(as_bool(intervention["recovery_endpoint_eligible"])),
                "reference_recovery_eligible": int(as_bool(reference["recovery_endpoint_eligible"])),
                "intervention_recovery_event": int(as_bool(intervention["recovery_event_observed"])),
                "reference_recovery_event": int(as_bool(reference["recovery_event_observed"])),
                "intervention_recovery_observed_time_s": (
                    as_float(intervention["recovery_time_s"])
                    if as_bool(intervention["recovery_event_observed"])
                    else None
                ),
                "reference_recovery_observed_time_s": (
                    as_float(reference["recovery_time_s"])
                    if as_bool(reference["recovery_event_observed"])
                    else None
                ),
                "intervention_recovery_time_or_censor_s": as_float(intervention["recovery_time_or_censor_s"]),
                "reference_recovery_time_or_censor_s": as_float(reference["recovery_time_or_censor_s"]),
                "intervention_recovery_censor_time_s": as_float(intervention["recovery_censor_time_s"]),
                "reference_recovery_censor_time_s": as_float(reference["recovery_censor_time_s"]),
            }
        )
        for endpoint in ENDPOINTS:
            int_measurement = endpoint_measurement(intervention, endpoint)
            ref_measurement = endpoint_measurement(reference, endpoint)
            complete, reason = pair_complete(intervention, reference, int_measurement, ref_measurement)
            delta = (
                int_measurement.value - ref_measurement.value
                if complete and int_measurement.value is not None and ref_measurement.value is not None
                else None
            )
            b_effect_rows.append(
                {
                    "intervention_task_id": intervention["task_id"],
                    "intervention_simulation_id": intervention["simulation_id"],
                    "reference_task_id": reference["task_id"],
                    "reference_simulation_id": reference["simulation_id"],
                    "intervention": intervention["ablations"],
                    "seed": intervention["seed"],
                    "speed": intervention["speed"],
                    "load": intervention["load"],
                    "load_side": intervention["load_side"],
                    "pulse": intervention["pulse"],
                    "endpoint": endpoint.name,
                    "endpoint_family": endpoint.family,
                    "analysis_window": endpoint.window,
                    "unit": endpoint.unit,
                    "direction_rule": endpoint.direction_rule,
                    "failure_transition": failure_transition(reference, intervention),
                    "intervention_value": int_measurement.value,
                    "reference_value": ref_measurement.value,
                    "intervention_numerator": int_measurement.numerator,
                    "intervention_denominator": int_measurement.denominator,
                    "reference_numerator": ref_measurement.numerator,
                    "reference_denominator": ref_measurement.denominator,
                    "applicable": int(int_measurement.applicable and ref_measurement.applicable),
                    "complete_pair": int(complete),
                    "incomplete_reason": reason,
                    "delta_intervention_minus_reference": delta,
                    "direction": direction_label(delta, endpoint.direction_rule),
                }
            )
            b_summary_atoms.append(
                {
                    "analysis_block": "B_single",
                    "effect_type": "single_minus_intact",
                    "intervention": intervention["ablations"],
                    "pulse": intervention["pulse"],
                    "endpoint": endpoint.name,
                    "direction_rule": endpoint.direction_rule,
                    "applicable": int(int_measurement.applicable and ref_measurement.applicable),
                    "complete": int(complete),
                    "value": delta,
                }
            )
    if len(b_state_rows) != 270 or len(b_effect_rows) != 270 * len(ENDPOINTS):
        raise AssertionError("Stage B derived row count mismatch")
    write_output("a_to_d_b_single_states.csv", b_state_rows)
    write_output("a_to_d_b_single_continuous_effects.csv", b_effect_rows)

    # Stage C: pair versus intact, both component singles, and exact
    # four-arm nonadditivity.
    b_component_map = unique_map(
        stage_rows["B"], ("ablations", *context_fields), "Stage B intervention context"
    )
    c_effect_rows: list[dict[str, Any]] = []
    c_state_rows: list[dict[str, Any]] = []
    c_summary_atoms: list[dict[str, Any]] = []
    for pair_row in ordered_rows(stage_rows["C"], spec):
        components = pair_row["ablations"].split("+")
        if len(components) != 2:
            raise AssertionError(f"Invalid Stage C pair: {pair_row['ablations']}")
        component_1, component_2 = components
        context = key_for(pair_row, context_fields)
        intact = a_map[context]
        single_1 = b_component_map[(component_1, *context)]
        single_2 = b_component_map[(component_2, *context)]
        c_state_rows.append(
            {
                "pair_task_id": pair_row["task_id"],
                "pair_simulation_id": pair_row["simulation_id"],
                "intact_task_id": intact["task_id"],
                "intact_simulation_id": intact["simulation_id"],
                "single_1_task_id": single_1["task_id"],
                "single_1_simulation_id": single_1["simulation_id"],
                "single_2_task_id": single_2["task_id"],
                "single_2_simulation_id": single_2["simulation_id"],
                "pair": pair_row["ablations"],
                "component_1": component_1,
                "component_2": component_2,
                "seed": pair_row["seed"],
                "speed": pair_row["speed"],
                "load": pair_row["load"],
                "load_side": pair_row["load_side"],
                "pulse": pair_row["pulse"],
                "pair_vs_intact_failure_transition": failure_transition(intact, pair_row),
                "pair_vs_single_1_failure_transition": failure_transition(single_1, pair_row),
                "pair_vs_single_2_failure_transition": failure_transition(single_2, pair_row),
                "pair_vs_intact_recovery_event_transition": recovery_event_transition(intact, pair_row),
                "pair_vs_single_1_recovery_event_transition": recovery_event_transition(single_1, pair_row),
                "pair_vs_single_2_recovery_event_transition": recovery_event_transition(single_2, pair_row),
                "intact_rhythmic_failure": int(as_bool(intact["rhythmic_failure"])),
                "single_1_rhythmic_failure": int(as_bool(single_1["rhythmic_failure"])),
                "single_2_rhythmic_failure": int(as_bool(single_2["rhythmic_failure"])),
                "pair_rhythmic_failure": int(as_bool(pair_row["rhythmic_failure"])),
                "intact_pulse_response_eligible": int(as_bool(intact["pulse_response_eligible"])),
                "single_1_pulse_response_eligible": int(as_bool(single_1["pulse_response_eligible"])),
                "single_2_pulse_response_eligible": int(as_bool(single_2["pulse_response_eligible"])),
                "pair_pulse_response_eligible": int(as_bool(pair_row["pulse_response_eligible"])),
                "intact_recovery_eligible": int(as_bool(intact["recovery_endpoint_eligible"])),
                "single_1_recovery_eligible": int(as_bool(single_1["recovery_endpoint_eligible"])),
                "single_2_recovery_eligible": int(as_bool(single_2["recovery_endpoint_eligible"])),
                "pair_recovery_eligible": int(as_bool(pair_row["recovery_endpoint_eligible"])),
                "intact_recovery_event": int(as_bool(intact["recovery_event_observed"])),
                "single_1_recovery_event": int(as_bool(single_1["recovery_event_observed"])),
                "single_2_recovery_event": int(as_bool(single_2["recovery_event_observed"])),
                "pair_recovery_event": int(as_bool(pair_row["recovery_event_observed"])),
                "intact_recovery_observed_time_s": (
                    as_float(intact["recovery_time_s"])
                    if as_bool(intact["recovery_event_observed"])
                    else None
                ),
                "single_1_recovery_observed_time_s": (
                    as_float(single_1["recovery_time_s"])
                    if as_bool(single_1["recovery_event_observed"])
                    else None
                ),
                "single_2_recovery_observed_time_s": (
                    as_float(single_2["recovery_time_s"])
                    if as_bool(single_2["recovery_event_observed"])
                    else None
                ),
                "pair_recovery_observed_time_s": (
                    as_float(pair_row["recovery_time_s"])
                    if as_bool(pair_row["recovery_event_observed"])
                    else None
                ),
                "intact_recovery_time_or_censor_s": as_float(intact["recovery_time_or_censor_s"]),
                "single_1_recovery_time_or_censor_s": as_float(single_1["recovery_time_or_censor_s"]),
                "single_2_recovery_time_or_censor_s": as_float(single_2["recovery_time_or_censor_s"]),
                "pair_recovery_time_or_censor_s": as_float(pair_row["recovery_time_or_censor_s"]),
                "intact_recovery_censor_time_s": as_float(intact["recovery_censor_time_s"]),
                "single_1_recovery_censor_time_s": as_float(single_1["recovery_censor_time_s"]),
                "single_2_recovery_censor_time_s": as_float(single_2["recovery_censor_time_s"]),
                "pair_recovery_censor_time_s": as_float(pair_row["recovery_censor_time_s"]),
            }
        )
        for endpoint in ENDPOINTS:
            measurements = {
                "pair": endpoint_measurement(pair_row, endpoint),
                "intact": endpoint_measurement(intact, endpoint),
                "single_1": endpoint_measurement(single_1, endpoint),
                "single_2": endpoint_measurement(single_2, endpoint),
            }
            pair_intact_complete, pair_intact_reason = pair_complete(
                pair_row, intact, measurements["pair"], measurements["intact"]
            )
            pair_single_1_complete, pair_single_1_reason = pair_complete(
                pair_row, single_1, measurements["pair"], measurements["single_1"]
            )
            pair_single_2_complete, pair_single_2_reason = pair_complete(
                pair_row, single_2, measurements["pair"], measurements["single_2"]
            )
            all_applicable = all(item.applicable for item in measurements.values())
            all_valid_rhythmic = all(rhythmic(row) for row in (pair_row, intact, single_1, single_2))
            all_measurable = all(item.measurement_eligible for item in measurements.values())
            complete_quad = all_applicable and all_valid_rhythmic and all_measurable
            if not all_applicable:
                quad_reason = "not_applicable"
            elif not all_valid_rhythmic:
                quad_reason = "invalid_or_rhythmic_failure_arm"
            elif not all_measurable:
                quad_reason = "incomplete_measurement_arm"
            else:
                quad_reason = "complete"
            values = {name: measurement.value for name, measurement in measurements.items()}
            pair_minus_intact = (
                values["pair"] - values["intact"]
                if pair_intact_complete and values["pair"] is not None and values["intact"] is not None
                else None
            )
            pair_minus_single_1 = (
                values["pair"] - values["single_1"]
                if pair_single_1_complete
                and values["pair"] is not None
                and values["single_1"] is not None
                else None
            )
            pair_minus_single_2 = (
                values["pair"] - values["single_2"]
                if pair_single_2_complete
                and values["pair"] is not None
                and values["single_2"] is not None
                else None
            )
            nonadditivity = (
                values["pair"] - values["single_1"] - values["single_2"] + values["intact"]
                if complete_quad and all(value is not None for value in values.values())
                else None
            )
            c_effect_rows.append(
                {
                    "pair_task_id": pair_row["task_id"],
                    "pair_simulation_id": pair_row["simulation_id"],
                    "intact_task_id": intact["task_id"],
                    "intact_simulation_id": intact["simulation_id"],
                    "single_1_task_id": single_1["task_id"],
                    "single_1_simulation_id": single_1["simulation_id"],
                    "single_2_task_id": single_2["task_id"],
                    "single_2_simulation_id": single_2["simulation_id"],
                    "pair": pair_row["ablations"],
                    "component_1": component_1,
                    "component_2": component_2,
                    "seed": pair_row["seed"],
                    "speed": pair_row["speed"],
                    "load": pair_row["load"],
                    "load_side": pair_row["load_side"],
                    "pulse": pair_row["pulse"],
                    "endpoint": endpoint.name,
                    "endpoint_family": endpoint.family,
                    "analysis_window": endpoint.window,
                    "unit": endpoint.unit,
                    "direction_rule": endpoint.direction_rule,
                    "pair_value": values["pair"],
                    "pair_numerator": measurements["pair"].numerator,
                    "pair_denominator": measurements["pair"].denominator,
                    "single_1_value": values["single_1"],
                    "single_1_numerator": measurements["single_1"].numerator,
                    "single_1_denominator": measurements["single_1"].denominator,
                    "single_2_value": values["single_2"],
                    "single_2_numerator": measurements["single_2"].numerator,
                    "single_2_denominator": measurements["single_2"].denominator,
                    "intact_value": values["intact"],
                    "intact_numerator": measurements["intact"].numerator,
                    "intact_denominator": measurements["intact"].denominator,
                    "pair_vs_intact_complete_pair": int(pair_intact_complete),
                    "pair_vs_intact_incomplete_reason": pair_intact_reason,
                    "pair_minus_intact": pair_minus_intact,
                    "pair_vs_intact_direction": direction_label(pair_minus_intact, endpoint.direction_rule),
                    "pair_vs_single_1_complete_pair": int(pair_single_1_complete),
                    "pair_vs_single_1_incomplete_reason": pair_single_1_reason,
                    "pair_minus_single_1": pair_minus_single_1,
                    "pair_vs_single_2_complete_pair": int(pair_single_2_complete),
                    "pair_vs_single_2_incomplete_reason": pair_single_2_reason,
                    "pair_minus_single_2": pair_minus_single_2,
                    "complete_quad": int(complete_quad),
                    "quad_incomplete_reason": quad_reason,
                    "nonadditivity_pair_minus_singles_plus_intact": nonadditivity,
                    "nonadditivity_direction": direction_label(nonadditivity, endpoint.direction_rule),
                }
            )
            for effect_type, complete, value in (
                ("pair_minus_intact", pair_intact_complete, pair_minus_intact),
                ("nonadditivity", complete_quad, nonadditivity),
            ):
                c_summary_atoms.append(
                    {
                        "analysis_block": "C_pair",
                        "effect_type": effect_type,
                        "intervention": pair_row["ablations"],
                        "pulse": pair_row["pulse"],
                        "endpoint": endpoint.name,
                        "direction_rule": endpoint.direction_rule,
                        "applicable": int(all_applicable),
                        "complete": int(complete),
                        "value": value,
                    }
                )
    if len(c_state_rows) != 162 or len(c_effect_rows) != 162 * len(ENDPOINTS):
        raise AssertionError("Stage C derived row count mismatch")
    write_output("a_to_d_c_pair_states.csv", c_state_rows)
    write_output("a_to_d_c_pair_continuous_effects.csv", c_effect_rows)

    # Stage D: raw recruitment profiles and exact high-vs-low modification of
    # intervention effects, using the Stage D intact arm as its locked match.
    d_rows = stage_rows["D"]
    d_key_fields = ("ablations", "seed", "protocol", "speed", "load", "load_side", "pulse")
    d_map = unique_map(d_rows, d_key_fields, "Stage D arm context")
    if set(row["load"] for row in d_rows) != {"normal"}:
        raise AssertionError("Stage D is not restricted to normal load")
    d_arms = sorted({row["ablations"] for row in d_rows})
    expected_d_arms = {"none", "V0D", "V0V", "V2a", "V1Ia", "V0D+V0V"}
    if set(d_arms) != expected_d_arms:
        raise AssertionError(f"Unexpected Stage D arms: {d_arms}")

    d_condition_rows: list[dict[str, Any]] = []
    d_recruitment_rows: list[dict[str, Any]] = []
    for row in ordered_rows(d_rows, spec):
        d_condition_rows.append(
            {
                "task_id": row["task_id"],
                "simulation_id": row["simulation_id"],
                "arm": row["ablations"],
                "seed": row["seed"],
                "speed": row["speed"],
                "load": row["load"],
                "load_side": row["load_side"],
                "pulse": row["pulse"],
                "analysis_valid": int(analysis_valid(row)),
                "rhythmic_failure": int(as_bool(row["rhythmic_failure"])),
                "pulse_delivered": int(as_bool(row["pulse_delivered"])),
                "pulse_response_eligible": int(as_bool(row["pulse_response_eligible"])),
                "recovery_eligible": int(as_bool(row["recovery_endpoint_eligible"])),
                "recovery_event": int(as_bool(row["recovery_event_observed"])),
                "recovery_observed_time_s": (
                    as_float(row["recovery_time_s"])
                    if as_bool(row["recovery_event_observed"])
                    else None
                ),
                "recovery_time_or_censor_s": as_float(row["recovery_time_or_censor_s"]),
                "recovery_censor_time_s": as_float(row["recovery_censor_time_s"]),
            }
        )
        for field in PARTICIPATION_FIELDS:
            d_recruitment_rows.append(
                {
                    "task_id": row["task_id"],
                    "simulation_id": row["simulation_id"],
                    "arm": row["ablations"],
                    "seed": row["seed"],
                    "speed": row["speed"],
                    "load": row["load"],
                    "load_side": row["load_side"],
                    "pulse": row["pulse"],
                    "model_population": field.removesuffix("_mean_rate_hz"),
                    "mean_rate_hz": as_float(row[field]),
                    "analysis_valid": int(analysis_valid(row)),
                    "rhythmic_failure": int(as_bool(row["rhythmic_failure"])),
                }
            )
    if len(d_condition_rows) != 54 or len(d_recruitment_rows) != 54 * len(PARTICIPATION_FIELDS):
        raise AssertionError("Stage D recruitment row count mismatch")
    write_output("a_to_d_d_conditions.csv", d_condition_rows)
    write_output("a_to_d_d_recruitment_rates.csv", d_recruitment_rows)

    d_did_rows: list[dict[str, Any]] = []
    d_summary_atoms: list[dict[str, Any]] = []
    d_interventions = [arm for arm in d_arms if arm != "none"]
    for intervention_name in d_interventions:
        for pulse in spec["axes"]["pulse"]:
            common = (str(spec["seed"]), "pulse")
            # Seed/protocol are asserted from the frozen design rather than
            # treated as sampling axes.
            low_context = (common[0], common[1], "low", "normal", "L", pulse)
            high_context = (common[0], common[1], "high", "normal", "L", pulse)
            int_low = d_map[(intervention_name, *low_context)]
            ref_low = d_map[("none", *low_context)]
            int_high = d_map[(intervention_name, *high_context)]
            ref_high = d_map[("none", *high_context)]
            for endpoint in ENDPOINTS:
                measurements = {
                    "int_low": endpoint_measurement(int_low, endpoint),
                    "ref_low": endpoint_measurement(ref_low, endpoint),
                    "int_high": endpoint_measurement(int_high, endpoint),
                    "ref_high": endpoint_measurement(ref_high, endpoint),
                }
                all_applicable = all(item.applicable for item in measurements.values())
                all_valid_rhythmic = all(rhythmic(row) for row in (int_low, ref_low, int_high, ref_high))
                all_measurable = all(item.measurement_eligible for item in measurements.values())
                complete_quad = all_applicable and all_valid_rhythmic and all_measurable
                if not all_applicable:
                    reason = "not_applicable"
                elif not all_valid_rhythmic:
                    reason = "invalid_or_rhythmic_failure_arm"
                elif not all_measurable:
                    reason = "incomplete_measurement_arm"
                else:
                    reason = "complete"
                values = {name: measurement.value for name, measurement in measurements.items()}
                low_effect = (
                    values["int_low"] - values["ref_low"]
                    if complete_quad and values["int_low"] is not None and values["ref_low"] is not None
                    else None
                )
                high_effect = (
                    values["int_high"] - values["ref_high"]
                    if complete_quad and values["int_high"] is not None and values["ref_high"] is not None
                    else None
                )
                did = (
                    high_effect - low_effect
                    if high_effect is not None and low_effect is not None
                    else None
                )
                d_did_rows.append(
                    {
                        "intervention": intervention_name,
                        "seed": int_low["seed"],
                        "load": "normal",
                        "load_side": "L",
                        "pulse": pulse,
                        "endpoint": endpoint.name,
                        "endpoint_family": endpoint.family,
                        "analysis_window": endpoint.window,
                        "unit": endpoint.unit,
                        "direction_rule": endpoint.direction_rule,
                        "low_intervention_task_id": int_low["task_id"],
                        "low_intact_task_id": ref_low["task_id"],
                        "high_intervention_task_id": int_high["task_id"],
                        "high_intact_task_id": ref_high["task_id"],
                        "low_failure_transition": failure_transition(ref_low, int_low),
                        "high_failure_transition": failure_transition(ref_high, int_high),
                        "low_recovery_event_transition": recovery_event_transition(ref_low, int_low),
                        "high_recovery_event_transition": recovery_event_transition(ref_high, int_high),
                        "low_intervention_value": values["int_low"],
                        "low_intact_value": values["ref_low"],
                        "high_intervention_value": values["int_high"],
                        "high_intact_value": values["ref_high"],
                        "applicable": int(all_applicable),
                        "complete_quad": int(complete_quad),
                        "incomplete_reason": reason,
                        "low_speed_intervention_minus_intact": low_effect,
                        "high_speed_intervention_minus_intact": high_effect,
                        "high_minus_low_effect_difference": did,
                        "speed_modification_direction": direction_label(did, endpoint.direction_rule, did=True),
                    }
                )
                d_summary_atoms.append(
                    {
                        "analysis_block": "D_speed_modification",
                        "effect_type": "high_minus_low_intervention_effect",
                        "intervention": intervention_name,
                        "pulse": pulse,
                        "endpoint": endpoint.name,
                        "direction_rule": endpoint.direction_rule,
                        "applicable": int(all_applicable),
                        "complete": int(complete_quad),
                        "value": did,
                    }
                )
    expected_did_rows = 5 * 3 * len(ENDPOINTS)
    if len(d_did_rows) != expected_did_rows:
        raise AssertionError("Stage D difference-in-differences row count mismatch")
    write_output("a_to_d_d_speed_effect_differences.csv", d_did_rows)

    # Equal-cell descriptive summaries.  Medians and IQRs describe only the
    # prespecified design grid and are not uncertainty estimates.
    summary_atoms = a_pulse_summary_atoms + b_summary_atoms + c_summary_atoms + d_summary_atoms
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for atom in summary_atoms:
        group_key = (
            atom["analysis_block"],
            atom["effect_type"],
            atom["intervention"],
            atom["pulse"],
            atom["endpoint"],
            atom["direction_rule"],
        )
        grouped[group_key].append(atom)

    summary_rows: list[dict[str, Any]] = []
    for group_key in sorted(grouped):
        atoms = grouped[group_key]
        values = [float(atom["value"]) for atom in atoms if atom["complete"] and atom["value"] is not None]
        positive = sum(value > EPSILON for value in values)
        negative = sum(value < -EPSILON for value in values)
        neutral = len(values) - positive - negative
        analysis_block, effect_type, intervention, pulse, endpoint_name, direction_rule = group_key
        summary_rows.append(
            {
                "analysis_block": analysis_block,
                "effect_type": effect_type,
                "intervention": intervention,
                "pulse": pulse,
                "endpoint": endpoint_name,
                "direction_rule": direction_rule,
                "design_cell_count": len(atoms),
                "applicable_cell_count": sum(atom["applicable"] for atom in atoms),
                "complete_cell_count": len(values),
                "incomplete_or_not_applicable_cell_count": len(atoms) - len(values),
                "median": median(values) if values else None,
                "q1_linear": linear_percentile(values, 0.25),
                "q3_linear": linear_percentile(values, 0.75),
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
                "positive_cell_count": positive,
                "neutral_cell_count": neutral,
                "negative_cell_count": negative,
                "deterioration_cell_count": positive if direction_rule == "higher_is_worse" else None,
                "improvement_cell_count": negative if direction_rule == "higher_is_worse" else None,
                "summary_interpretation": "prespecified_design_grid_not_sampling_uncertainty",
            }
        )
    write_output("a_to_d_descriptive_summaries.csv", summary_rows)

    # Re-open all CSV headers and counts as an independent final check.
    for filename, record in output_manifest.items():
        path = DERIVED_DIR / filename
        rows, fields = read_csv(path)
        if len(rows) != record["row_count"]:
            raise AssertionError(f"Post-write row-count mismatch: {filename}")
        output_columns_are_allowed(fields)
        if any(as_int(row["output_row_count"]) != record["row_count"] for row in rows):
            raise AssertionError(f"Embedded row-count mismatch: {filename}")
        if any(row["script_sha256"] != script_sha256 for row in rows):
            raise AssertionError(f"Embedded script hash mismatch: {filename}")

    qc = {
        "schema": "cpg-manuscript-a-to-d-analysis-qc-1.0",
        "record_count": 1,
        "analysis_scope": "Stages A-D; one frozen stochastic realization; descriptive exact contrasts",
        "generated_at_utc": generated_at,
        "script": {
            "path": str(SCRIPT_PATH.relative_to(WORKSPACE_ROOT)),
            "version": SCRIPT_VERSION,
            "sha256": script_sha256,
        },
        "locked_protocol_hashes": protocol_hashes,
        "sources": source_hashes,
        "global_source_checks": {
            "analysis_task_count": len(index_rows),
            "unique_simulation_count": len(unique_rows),
            "avoided_recomputation_count": len(index_rows) - len(unique_rows),
            "long_epoch_row_count": len(epoch_rows),
            "task_identity_and_design_fields_match": True,
            "unique_simulation_identity_sets_match": True,
            "transfer_identity_checks": transfer_identity_checks,
            "all_transfer_identities_hold": True,
        },
        "stage_checks": {
            stage: {
                "task_count": len(rows),
                "unique_simulation_count": len({row["simulation_id"] for row in rows}),
                "analysis_valid_task_count": sum(analysis_valid(row) for row in rows),
                "rhythmic_failure_task_count": sum(as_bool(row["rhythmic_failure"]) for row in rows),
            }
            for stage, rows in stage_rows.items()
        },
        "design_checks": {
            "A_intact_no_pulse_cells": len(a_grid_rows),
            "A_active_pulse_vs_matched_sham_pairs": len(a_pulse_rows),
            "B_single_context_pairs": len(b_state_rows),
            "C_pair_context_quads": len(c_state_rows),
            "D_condition_rows": len(d_condition_rows),
            "D_recruitment_rows": len(d_recruitment_rows),
            "D_speed_effect_difference_rows": len(d_did_rows),
            "C_component_single_arms_complete": True,
            "failure_transitions_preserved": True,
            "continuous_values_require_complete_pairs_or_quads": True,
            "blank_values_not_coerced_to_zero": True,
            "simulation_ids_not_counted_as_replicates": True,
            "descriptive_quantile_method": "linear interpolation at (n-1)*p",
            "neutral_tolerance": EPSILON,
        },
        "inference_boundary": {
            "independent_stochastic_realizations": 1,
            "sampling_inference_calculated": False,
            "forbidden_inference_fields": sorted(FORBIDDEN_INFERENCE_FIELDS),
            "forbidden_inference_fields_present_in_outputs": False,
        },
        "outputs": output_manifest,
        "status": "PASS",
    }
    qc_path = DERIVED_DIR / "a_to_d_qc.json"
    with qc_path.open("w", encoding="utf-8") as handle:
        json.dump(qc, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    print(json.dumps({"status": "PASS", "outputs": output_manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
