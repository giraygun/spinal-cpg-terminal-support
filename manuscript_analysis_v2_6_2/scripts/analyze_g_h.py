#!/usr/bin/env python3
"""Locked manuscript analysis for stages G and H of CPG v2.6.2.

This script implements only the post-run analysis contract in
ANALYSIS_PROTOCOL_LOCKED_2026-08-26.md and PROTOCOL_SPEC.json.  It never
modifies frozen inputs.  All outputs are descriptive and conditional on the
single frozen stochastic realization (seed 601; structural seed 160601).

No p-values, confidence intervals, standard errors, degrees of freedom, or
between-seed effect sizes are computed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


SCRIPT_VERSION = "g-h-analysis-1.0.0"
NEUTRAL_TOLERANCE = 1e-12

HERE = Path(__file__).resolve()
ANALYSIS_ROOT = HERE.parents[1]
REPOSITORY_ROOT = ANALYSIS_ROOT.parent
SPEC_PATH = ANALYSIS_ROOT / "PROTOCOL_SPEC.json"
OUT_DIR = REPOSITORY_ROOT / "derived" / "manuscript_analysis_v2_6_2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean: {value!r}")


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value!r}")
    return result


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    rounded = int(round(number))
    if abs(number - rounded) > NEUTRAL_TOLERANCE:
        raise ValueError(f"Expected integer-valued field, observed {value!r}")
    return rounded


def int_flag(value: bool | None) -> int | str:
    return "" if value is None else int(value)


def mean_defined(values: Iterable[Any]) -> float | None:
    defined = [as_float(value) for value in values]
    usable = [value for value in defined if value is not None]
    return None if not usable else sum(usable) / len(usable)


def sum_int_defined(values: Iterable[Any]) -> int | None:
    parsed = [as_int(value) for value in values]
    if any(value is None for value in parsed):
        return None
    return sum(value for value in parsed if value is not None)


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    num = as_float(numerator)
    den = as_float(denominator)
    if num is None or den is None or den == 0:
        return None
    return num / den


def numeric_difference(target: Any, reference: Any) -> float | None:
    target_value = as_float(target)
    reference_value = as_float(reference)
    if target_value is None or reference_value is None:
        return None
    return target_value - reference_value


def failure_transition(reference_failed: bool, target_failed: bool) -> str:
    if not reference_failed and not target_failed:
        return "rhythmic_to_rhythmic"
    if not reference_failed and target_failed:
        return "rhythmic_to_failure"
    if reference_failed and not target_failed:
        return "failure_to_rhythmic"
    return "failure_to_failure"


def direction_label(value: float | None, higher_is_worse: bool | None) -> str:
    if value is None:
        return "not_computed"
    if abs(value) <= NEUTRAL_TOLERANCE:
        return "numeric_neutral"
    if higher_is_worse is True:
        return "degradation" if value > 0 else "improvement"
    if higher_is_worse is False:
        return "increase" if value > 0 else "decrease"
    return "descriptive_increase" if value > 0 else "descriptive_decrease"


def require(condition: bool, label: str, checks: dict[str, bool]) -> None:
    checks[label] = bool(condition)
    if not condition:
        raise RuntimeError(f"QC failed: {label}")


def source_field_name(filename: str) -> str:
    stem = Path(filename).stem.replace("-", "_")
    return f"source_{stem}_sha256"


def write_csv_with_provenance(
    path: Path,
    rows: list[dict[str, Any]],
    data_fields: list[str],
    source_names: list[str],
    generated_at: str,
    script_sha256: str,
    protocol_spec_sha256: str,
    source_hashes: dict[str, str],
) -> list[str]:
    provenance_fields = [
        "generated_at_utc",
        "script_version",
        "script_sha256",
        "protocol_spec_sha256",
        *[source_field_name(name) for name in source_names],
        "output_row_count",
    ]
    fieldnames = provenance_fields + data_fields
    row_count = len(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            payload: dict[str, Any] = {
                "generated_at_utc": generated_at,
                "script_version": SCRIPT_VERSION,
                "script_sha256": script_sha256,
                "protocol_spec_sha256": protocol_spec_sha256,
                "output_row_count": row_count,
            }
            for name in source_names:
                payload[source_field_name(name)] = source_hashes[name]
            payload.update({field: row.get(field, "") for field in data_fields})
            writer.writerow({key: "" if value is None else value for key, value in payload.items()})
    return fieldnames


def window_for_epoch(epoch: int, windows: dict[str, list[int]]) -> tuple[str, bool]:
    for name in ("baseline", "stress_prechallenge", "stress_challenge", "recovery"):
        if epoch in windows[name]:
            return name, True
    if epoch == 1:
        return "display_only_early", False
    if epoch == 19:
        return "display_only_transition", False
    raise RuntimeError(f"Epoch {epoch} is absent from the locked window contract")


def classify_g_arm(index_row: dict[str, str]) -> tuple[str, str, int, int]:
    impaired = index_row["impaired_mt_routes"] != "none"
    challenged = index_row["challenged_routes"] != "none"
    if impaired and challenged and index_row["impaired_mt_routes"] != index_row["challenged_routes"]:
        raise RuntimeError(f"G route mismatch in {index_row['task_id']}")
    route = (
        index_row["impaired_mt_routes"]
        if impaired
        else index_row["challenged_routes"] if challenged else "shared_intact"
    )
    arm = {
        (False, False): "no_challenge_intact",
        (True, False): "no_challenge_impaired",
        (False, True): "challenge_intact",
        (True, True): "challenge_impaired",
    }[(impaired, challenged)]
    return route, arm, int(challenged), int(impaired)


def aggregate_g_window(
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
    route_names: list[str],
) -> dict[str, Any]:
    epochs = sorted(int(row["epoch"]) for row in rows)
    technical = [as_bool(row["technical_valid"]) is True for row in rows]
    scientific = as_bool(meta["scientific_valid"]) is True
    failures = [as_bool(row["rhythmic_failure"]) is True for row in rows]

    result: dict[str, Any] = {
        "task_id": meta["task_id"],
        "simulation_id": meta["simulation_id"],
        "route_scope": meta["route_scope"],
        "arm": meta["arm"],
        "challenge_flag": meta["challenge_flag"],
        "impairment_flag": meta["impairment_flag"],
        "window": meta["window"],
        "epochs": "+".join(str(epoch) for epoch in epochs),
        "epoch_count": len(rows),
        "all_epochs_technical_valid": int(all(technical)),
        "scientific_valid": int(scientific),
        "failure_epoch_count": sum(failures),
        "rhythmic_failure_any": int(any(failures)),
        "rhythmic_failure_epoch_fraction": sum(failures) / len(failures),
        "continuous_window_status_complete": int(
            all(technical) and scientific and not any(failures)
        ),
    }

    mean_fields = [
        "frequency_hz",
        "rg_cycle_interval_cv_mean",
        "lr_phase_error_mean_abs_deg",
        "fe_phase_error_mean_abs_deg",
        "bilateral_amplitude_balance",
        "challenged_rrp_mean_secondary",
        "challenged_replenishment_resource_mean_secondary",
    ]
    for field in mean_fields:
        result[field] = mean_defined(row[field] for row in rows)
    balance = as_float(result["bilateral_amplitude_balance"])
    result["bilateral_amplitude_imbalance"] = None if balance is None else 1.0 - balance

    for prefix in ("lr_phase", "fe_phase"):
        slips = sum_int_defined(row[f"{prefix}_slip_count"] for row in rows)
        cycles = sum_int_defined(row[f"{prefix}_cycle_count"] for row in rows)
        result[f"{prefix}_slip_count_sum"] = slips
        result[f"{prefix}_cycle_count_sum"] = cycles
        result[f"{prefix}_slip_fraction"] = mean_defined(
            safe_ratio(row[f"{prefix}_slip_count"], row[f"{prefix}_cycle_count"])
            for row in rows
        )
        result[f"{prefix}_slip_fraction_cycle_pooled_sensitivity"] = safe_ratio(
            slips, cycles
        )

    for prefix in ("pf", "mn"):
        anchor = sum_int_defined(row[f"{prefix}_transfer_anchor_count"] for row in rows)
        missed = sum_int_defined(row[f"{prefix}_transfer_missed_count"] for row in rows)
        matched = sum_int_defined(row[f"{prefix}_transfer_matched_count"] for row in rows)
        result[f"{prefix}_transfer_anchor_count_sum"] = anchor
        result[f"{prefix}_transfer_missed_count_sum"] = missed
        result[f"{prefix}_transfer_matched_count_sum"] = matched
        result[f"{prefix}_network_propagation_gap"] = mean_defined(
            safe_ratio(
                row[f"{prefix}_transfer_missed_count"],
                row[f"{prefix}_transfer_anchor_count"],
            )
            for row in rows
        )
        result[f"{prefix}_network_propagation_gap_anchor_pooled_sensitivity"] = safe_ratio(
            missed, anchor
        )
        result[f"{prefix}_transfer_identity_ok"] = int(
            anchor is not None and missed is not None and matched is not None
            and missed + matched == anchor
        )

    for route in route_names:
        for prefix in ("mt", "rrp", "replenishment_resource"):
            suffix = "_mean" if prefix == "mt" else "_mean_secondary"
            field = f"{prefix}_{route}{suffix}"
            result[field] = mean_defined(row[field] for row in rows)
    return result


def add_g_baseline_changes(window_rows: list[dict[str, Any]]) -> None:
    baseline_by_sim = {
        row["simulation_id"]: row for row in window_rows if row["window"] == "baseline"
    }
    delta_fields = [
        "lr_phase_error_mean_abs_deg",
        "fe_phase_error_mean_abs_deg",
        "lr_phase_slip_fraction",
        "fe_phase_slip_fraction",
        "frequency_hz",
        "rg_cycle_interval_cv_mean",
        "bilateral_amplitude_imbalance",
        "pf_network_propagation_gap",
        "mn_network_propagation_gap",
    ]
    for row in window_rows:
        baseline = baseline_by_sim[row["simulation_id"]]
        row["failure_transition_from_baseline"] = failure_transition(
            bool(baseline["rhythmic_failure_any"]), bool(row["rhythmic_failure_any"])
        )
        status_complete = bool(
            baseline["continuous_window_status_complete"]
            and row["continuous_window_status_complete"]
        )
        row["complete_status_vs_baseline"] = int(status_complete)
        for field in delta_fields:
            key = f"delta_{field}_from_baseline"
            row[key] = numeric_difference(row[field], baseline[field]) if status_complete else None


def component_values(row: dict[str, Any], endpoint: str) -> tuple[Any, Any, Any]:
    if endpoint == "lr_phase_slip_fraction":
        return (
            row["lr_phase_slip_count_sum"], row["lr_phase_cycle_count_sum"], None
        )
    if endpoint == "fe_phase_slip_fraction":
        return (
            row["fe_phase_slip_count_sum"], row["fe_phase_cycle_count_sum"], None
        )
    if endpoint == "pf_network_propagation_gap":
        return (
            row["pf_transfer_missed_count_sum"],
            row["pf_transfer_anchor_count_sum"],
            row["pf_transfer_matched_count_sum"],
        )
    if endpoint == "mn_network_propagation_gap":
        return (
            row["mn_transfer_missed_count_sum"],
            row["mn_transfer_anchor_count_sum"],
            row["mn_transfer_matched_count_sum"],
        )
    if endpoint == "rhythmic_failure_epoch_fraction":
        return row["failure_epoch_count"], row["epoch_count"], None
    return None, None, None


def g_sensitivity_value(row: dict[str, Any], endpoint: str) -> float | None:
    sensitivity_fields = {
        "lr_phase_slip_fraction": "lr_phase_slip_fraction_cycle_pooled_sensitivity",
        "fe_phase_slip_fraction": "fe_phase_slip_fraction_cycle_pooled_sensitivity",
        "pf_network_propagation_gap": "pf_network_propagation_gap_anchor_pooled_sensitivity",
        "mn_network_propagation_gap": "mn_network_propagation_gap_anchor_pooled_sensitivity",
    }
    field = sensitivity_fields.get(endpoint)
    return None if field is None else as_float(row[field])


def build_g_did_rows(
    window_rows: list[dict[str, Any]],
    routes: list[str],
    windows: list[str],
) -> list[dict[str, Any]]:
    shared = {
        row["window"]: row
        for row in window_rows
        if row["arm"] == "no_challenge_intact"
    }
    specific = {
        (row["route_scope"], row["arm"], row["window"]): row
        for row in window_rows
        if row["arm"] != "no_challenge_intact"
    }

    endpoint_specs: list[dict[str, Any]] = [
        {"endpoint": "lr_phase_error_mean_abs_deg", "family": "phase", "unit": "deg", "higher_is_worse": True, "aggregation": "equal_epoch_mean"},
        {"endpoint": "fe_phase_error_mean_abs_deg", "family": "phase", "unit": "deg", "higher_is_worse": True, "aggregation": "equal_epoch_mean"},
        {"endpoint": "lr_phase_slip_fraction", "family": "phase_slip", "unit": "fraction", "higher_is_worse": True, "aggregation": "equal_epoch_mean_of_raw_count_ratios"},
        {"endpoint": "fe_phase_slip_fraction", "family": "phase_slip", "unit": "fraction", "higher_is_worse": True, "aggregation": "equal_epoch_mean_of_raw_count_ratios"},
        {"endpoint": "rhythmic_failure_epoch_fraction", "family": "failure", "unit": "fraction", "higher_is_worse": True, "failure_endpoint": True, "aggregation": "equal_epoch_mean_of_binary_status"},
        {"endpoint": "frequency_hz", "family": "rhythm", "unit": "Hz", "higher_is_worse": None, "aggregation": "equal_epoch_mean"},
        {"endpoint": "rg_cycle_interval_cv_mean", "family": "rhythm", "unit": "fraction", "higher_is_worse": True, "aggregation": "equal_epoch_mean"},
        {"endpoint": "bilateral_amplitude_imbalance", "family": "motor_balance", "unit": "fraction", "higher_is_worse": True, "aggregation": "equal_epoch_mean"},
        {"endpoint": "pf_network_propagation_gap", "family": "network_propagation", "unit": "fraction", "higher_is_worse": True, "aggregation": "equal_epoch_mean_of_raw_count_ratios"},
        {"endpoint": "mn_network_propagation_gap", "family": "network_propagation", "unit": "fraction", "higher_is_worse": True, "aggregation": "equal_epoch_mean_of_raw_count_ratios"},
        {"endpoint": "challenged_rrp_mean_secondary", "family": "terminal_state", "unit": "normalized", "higher_is_worse": None, "aggregation": "equal_epoch_mean"},
        {"endpoint": "challenged_replenishment_resource_mean_secondary", "family": "terminal_state", "unit": "normalized", "higher_is_worse": None, "aggregation": "equal_epoch_mean"},
        {"endpoint": "route_mt_mean", "family": "terminal_state", "unit": "normalized", "higher_is_worse": None, "route_field": "mt_{route}_mean", "aggregation": "equal_epoch_mean"},
        {"endpoint": "route_rrp_mean_secondary", "family": "terminal_state", "unit": "normalized", "higher_is_worse": None, "route_field": "rrp_{route}_mean_secondary", "aggregation": "equal_epoch_mean"},
        {"endpoint": "route_replenishment_resource_mean_secondary", "family": "terminal_state", "unit": "normalized", "higher_is_worse": None, "route_field": "replenishment_resource_{route}_mean_secondary", "aggregation": "equal_epoch_mean"},
    ]

    arms_order = [
        "no_challenge_intact",
        "no_challenge_impaired",
        "challenge_intact",
        "challenge_impaired",
    ]
    output: list[dict[str, Any]] = []
    for route in routes:
        for window in windows:
            arms = {
                "no_challenge_intact": shared[window],
                **{
                    arm: specific[(route, arm, window)]
                    for arm in arms_order[1:]
                },
            }
            for spec in endpoint_specs:
                endpoint = spec["endpoint"]
                source_field = spec.get("route_field", endpoint).format(route=route)
                values = {arm: as_float(arms[arm][source_field]) for arm in arms_order}
                all_status_valid = all(
                    arms[arm]["all_epochs_technical_valid"]
                    and arms[arm]["scientific_valid"]
                    for arm in arms_order
                )
                failure_endpoint = bool(spec.get("failure_endpoint", False))
                rhythm_complete = all(
                    not arms[arm]["rhythmic_failure_any"] for arm in arms_order
                )
                complete = all_status_valid and all(
                    values[arm] is not None for arm in arms_order
                ) and (failure_endpoint or rhythm_complete)
                contrast = None
                if complete:
                    contrast = (
                        values["challenge_impaired"]
                        - values["no_challenge_impaired"]
                        - values["challenge_intact"]
                        + values["no_challenge_intact"]
                    )
                sensitivity_values = {
                    arm: g_sensitivity_value(arms[arm], endpoint) for arm in arms_order
                }
                sensitivity_complete = all(
                    sensitivity_values[arm] is not None for arm in arms_order
                ) and all_status_valid and (failure_endpoint or rhythm_complete)
                sensitivity_contrast = None
                if sensitivity_complete:
                    sensitivity_contrast = (
                        sensitivity_values["challenge_impaired"]
                        - sensitivity_values["no_challenge_impaired"]
                        - sensitivity_values["challenge_intact"]
                        + sensitivity_values["no_challenge_intact"]
                    )
                row: dict[str, Any] = {
                    "route": route,
                    "window": window,
                    "endpoint": endpoint,
                    "endpoint_family": spec["family"],
                    "unit": spec["unit"],
                    "higher_is_worse": int_flag(spec["higher_is_worse"]),
                    "aggregation_rule": spec["aggregation"],
                    "formula": "(challenge_impaired - no_challenge_impaired) - (challenge_intact - no_challenge_intact)",
                    "neutral_tolerance": NEUTRAL_TOLERANCE,
                    "all_arms_status_valid": int(all_status_valid),
                    "all_arms_rhythmic": int(rhythm_complete),
                    "complete_pair": int(complete),
                    "contrast_value": contrast,
                    "contrast_direction": direction_label(contrast, spec["higher_is_worse"]),
                    "sensitivity_complete_pair": int(sensitivity_complete),
                    "sensitivity_contrast_value": sensitivity_contrast,
                    "challenge_effect_transition_intact": failure_transition(
                        bool(arms["no_challenge_intact"]["rhythmic_failure_any"]),
                        bool(arms["challenge_intact"]["rhythmic_failure_any"]),
                    ),
                    "challenge_effect_transition_impaired": failure_transition(
                        bool(arms["no_challenge_impaired"]["rhythmic_failure_any"]),
                        bool(arms["challenge_impaired"]["rhythmic_failure_any"]),
                    ),
                    "impairment_effect_transition_no_challenge": failure_transition(
                        bool(arms["no_challenge_intact"]["rhythmic_failure_any"]),
                        bool(arms["no_challenge_impaired"]["rhythmic_failure_any"]),
                    ),
                    "impairment_effect_transition_challenge": failure_transition(
                        bool(arms["challenge_intact"]["rhythmic_failure_any"]),
                        bool(arms["challenge_impaired"]["rhythmic_failure_any"]),
                    ),
                }
                for arm in arms_order:
                    row[f"task_id_{arm}"] = arms[arm]["task_id"]
                    row[f"simulation_id_{arm}"] = arms[arm]["simulation_id"]
                    row[f"value_{arm}"] = values[arm]
                    row[f"rhythmic_failure_{arm}"] = arms[arm]["rhythmic_failure_any"]
                    numerator, denominator, matched = component_values(arms[arm], endpoint)
                    row[f"raw_numerator_{arm}"] = numerator
                    row[f"raw_denominator_{arm}"] = denominator
                    row[f"raw_matched_{arm}"] = matched
                    row[f"sensitivity_value_{arm}"] = sensitivity_values[arm]
                output.append(row)
    return output


def build_h_condition_rows(
    h_metrics: list[dict[str, str]],
    route_names: list[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in h_metrics:
        pulse = source["pulse"]
        active_pulse = pulse != "none"
        row: dict[str, Any] = {
            key: source[key]
            for key in [
                "task_id", "simulation_id", "seed", "protocol", "speed", "load",
                "load_side", "pulse", "fast_mode", "mt_mode", "label",
                "scientific_valid", "technical_valid", "technical_exclusion_reason",
                "rhythmic_failure", "pulse_required", "pulse_delivered",
                "pulse_response_eligible", "pulse_noneligibility_reason",
                "recovery_endpoint_eligible", "recovery_event_observed",
                "recovery_time_s", "recovery_time_or_censor_s",
                "recovery_censor_time_s", "recovery_ineligibility_reason",
            ]
        }
        row["active_pulse"] = int(active_pulse)
        row["analysis_lr_phase_error_mean_abs_deg"] = source[
            "post_pulse_lr_phase_error_mean_abs_deg" if active_pulse
            else "lr_phase_error_mean_abs_deg"
        ]
        row["analysis_fe_phase_error_mean_abs_deg"] = source[
            "post_pulse_fe_phase_error_mean_abs_deg" if active_pulse
            else "fe_phase_error_mean_abs_deg"
        ]
        for side in ("lr", "fe"):
            prefix = "post_pulse_" if active_pulse else ""
            numerator = source[f"{prefix}{side}_phase_slip_count"]
            denominator = source[f"{prefix}{side}_phase_cycle_count"]
            row[f"analysis_{side}_phase_slip_count"] = numerator
            row[f"analysis_{side}_phase_cycle_count"] = denominator
            row[f"analysis_{side}_phase_slip_fraction"] = safe_ratio(numerator, denominator)

        common_fields = [
            "frequency_hz", "rg_cycle_interval_cv_mean",
            "bilateral_amplitude_balance", "bilateral_amplitude_imbalance",
            "pf_transfer_anchor_count", "pf_transfer_missed_count",
            "pf_transfer_matched_count", "mn_transfer_anchor_count",
            "mn_transfer_missed_count", "mn_transfer_matched_count",
            "rg_pf_latency_mean_ms", "rg_mn_latency_mean_ms",
            "mean_mt_support_left", "mean_mt_support_right",
            "ia_signal_mean", "ib_signal_mean", "muscle_force_mean",
        ]
        common_fields.extend(f"{route}_mean_rate_hz" for route in route_names)
        for route in route_names:
            common_fields.extend(
                [f"mt_{route}_mean", f"rrp_{route}_mean", f"replenishment_resource_{route}_mean"]
            )
        for field in common_fields:
            row[field] = source[field]

        for prefix in ("pf", "mn"):
            row[f"{prefix}_network_propagation_gap"] = safe_ratio(
                source[f"{prefix}_transfer_missed_count"],
                source[f"{prefix}_transfer_anchor_count"],
            )
            anchor = as_int(source[f"{prefix}_transfer_anchor_count"])
            missed = as_int(source[f"{prefix}_transfer_missed_count"])
            matched = as_int(source[f"{prefix}_transfer_matched_count"])
            row[f"{prefix}_transfer_identity_ok"] = int(
                anchor is not None and missed is not None and matched is not None
                and missed + matched == anchor
            )

        eligible = as_bool(source["recovery_endpoint_eligible"]) is True
        event = as_bool(source["recovery_event_observed"]) is True
        row["recovery_failure"] = int(not event) if active_pulse and eligible else None
        valid = (
            as_bool(source["scientific_valid"]) is True
            and as_bool(source["technical_valid"]) is True
            and as_bool(source["rhythmic_failure"]) is False
        )
        pulse_valid = (
            not active_pulse
            or (
                as_bool(source["pulse_delivered"]) is True
                and as_bool(source["pulse_response_eligible"]) is True
                and eligible
            )
        )
        selected_defined = all(
            as_float(row[field]) is not None
            for field in [
                "analysis_lr_phase_error_mean_abs_deg",
                "analysis_fe_phase_error_mean_abs_deg",
                "analysis_lr_phase_slip_fraction",
                "analysis_fe_phase_slip_fraction",
            ]
        )
        row["condition_status_complete"] = int(valid and pulse_valid and selected_defined)
        output.append(row)
    return output


def h_endpoint_specs(route_names: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"endpoint": "analysis_lr_phase_error_mean_abs_deg", "family": "phase", "unit": "deg", "higher_is_worse": True},
        {"endpoint": "analysis_fe_phase_error_mean_abs_deg", "family": "phase", "unit": "deg", "higher_is_worse": True},
        {"endpoint": "analysis_lr_phase_slip_fraction", "family": "phase_slip", "unit": "fraction", "higher_is_worse": True, "components": ("analysis_lr_phase_slip_count", "analysis_lr_phase_cycle_count", None)},
        {"endpoint": "analysis_fe_phase_slip_fraction", "family": "phase_slip", "unit": "fraction", "higher_is_worse": True, "components": ("analysis_fe_phase_slip_count", "analysis_fe_phase_cycle_count", None)},
        {"endpoint": "rhythmic_failure", "family": "failure", "unit": "binary", "higher_is_worse": True, "failure_endpoint": True},
        {"endpoint": "recovery_failure", "family": "recovery", "unit": "binary", "higher_is_worse": True, "recovery_event_endpoint": True},
        {"endpoint": "recovery_time_s", "family": "recovery", "unit": "s", "higher_is_worse": True, "recovery_time_endpoint": True},
        {"endpoint": "frequency_hz", "family": "rhythm", "unit": "Hz", "higher_is_worse": None},
        {"endpoint": "rg_cycle_interval_cv_mean", "family": "rhythm", "unit": "fraction", "higher_is_worse": True},
        {"endpoint": "bilateral_amplitude_imbalance", "family": "motor_balance", "unit": "fraction", "higher_is_worse": True},
        {"endpoint": "pf_network_propagation_gap", "family": "network_propagation", "unit": "fraction", "higher_is_worse": True, "components": ("pf_transfer_missed_count", "pf_transfer_anchor_count", "pf_transfer_matched_count")},
        {"endpoint": "mn_network_propagation_gap", "family": "network_propagation", "unit": "fraction", "higher_is_worse": True, "components": ("mn_transfer_missed_count", "mn_transfer_anchor_count", "mn_transfer_matched_count")},
        {"endpoint": "rg_pf_latency_mean_ms", "family": "network_propagation", "unit": "ms", "higher_is_worse": None},
        {"endpoint": "rg_mn_latency_mean_ms", "family": "network_propagation", "unit": "ms", "higher_is_worse": None},
        {"endpoint": "mean_mt_support_left", "family": "terminal_state", "unit": "normalized", "higher_is_worse": None},
        {"endpoint": "mean_mt_support_right", "family": "terminal_state", "unit": "normalized", "higher_is_worse": None},
    ]
    for route in route_names:
        specs.append({"endpoint": f"{route}_mean_rate_hz", "family": "population_rate", "unit": "Hz", "higher_is_worse": None})
    for route in route_names:
        specs.extend(
            [
                {"endpoint": f"mt_{route}_mean", "family": "terminal_state", "unit": "normalized", "higher_is_worse": None},
                {"endpoint": f"rrp_{route}_mean", "family": "terminal_state", "unit": "normalized", "higher_is_worse": None},
                {"endpoint": f"replenishment_resource_{route}_mean", "family": "terminal_state", "unit": "normalized", "higher_is_worse": None},
            ]
        )
    return specs


def h_endpoint_value(row: dict[str, Any], spec: dict[str, Any]) -> float | None:
    endpoint = spec["endpoint"]
    if endpoint == "rhythmic_failure":
        value = as_bool(row[endpoint])
        return None if value is None else float(value)
    return as_float(row.get(endpoint))


def h_endpoint_complete(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    status_valid = (
        as_bool(row["scientific_valid"]) is True
        and as_bool(row["technical_valid"]) is True
    )
    if not status_valid:
        return False
    active_pulse = bool(row["active_pulse"])
    endpoint = spec["endpoint"]
    if spec.get("failure_endpoint"):
        return h_endpoint_value(row, spec) is not None
    if spec.get("recovery_event_endpoint"):
        return (
            active_pulse
            and as_bool(row["recovery_endpoint_eligible"]) is True
            and h_endpoint_value(row, spec) is not None
        )
    if spec.get("recovery_time_endpoint"):
        return (
            active_pulse
            and as_bool(row["recovery_endpoint_eligible"]) is True
            and as_bool(row["recovery_event_observed"]) is True
            and h_endpoint_value(row, spec) is not None
        )
    if as_bool(row["rhythmic_failure"]) is True:
        return False
    if endpoint.startswith("analysis_") and active_pulse:
        if not (
            as_bool(row["pulse_delivered"]) is True
            and as_bool(row["pulse_response_eligible"]) is True
        ):
            return False
    return h_endpoint_value(row, spec) is not None


def h_components(row: dict[str, Any], spec: dict[str, Any]) -> tuple[Any, Any, Any]:
    component_fields = spec.get("components")
    if component_fields is None:
        if spec.get("recovery_event_endpoint"):
            return row["recovery_event_observed"], row["recovery_endpoint_eligible"], None
        return None, None, None
    numerator, denominator, matched = component_fields
    return (
        row.get(numerator), row.get(denominator), row.get(matched) if matched else None
    )


def h_pair_row(
    reference: dict[str, Any],
    target: dict[str, Any],
    spec: dict[str, Any],
    identity: dict[str, Any],
    formula: str,
) -> dict[str, Any]:
    reference_value = h_endpoint_value(reference, spec)
    target_value = h_endpoint_value(target, spec)
    complete = h_endpoint_complete(reference, spec) and h_endpoint_complete(target, spec)
    contrast = target_value - reference_value if complete else None
    ref_num, ref_den, ref_matched = h_components(reference, spec)
    tar_num, tar_den, tar_matched = h_components(target, spec)
    row: dict[str, Any] = {
        **identity,
        "endpoint": spec["endpoint"],
        "endpoint_family": spec["family"],
        "unit": spec["unit"],
        "higher_is_worse": int_flag(spec["higher_is_worse"]),
        "formula": formula,
        "neutral_tolerance": NEUTRAL_TOLERANCE,
        "task_id_reference": reference["task_id"],
        "task_id_target": target["task_id"],
        "simulation_id_reference": reference["simulation_id"],
        "simulation_id_target": target["simulation_id"],
        "value_reference": reference_value,
        "value_target": target_value,
        "raw_numerator_reference": ref_num,
        "raw_denominator_reference": ref_den,
        "raw_matched_reference": ref_matched,
        "raw_numerator_target": tar_num,
        "raw_denominator_target": tar_den,
        "raw_matched_target": tar_matched,
        "rhythmic_failure_reference": int_flag(as_bool(reference["rhythmic_failure"])),
        "rhythmic_failure_target": int_flag(as_bool(target["rhythmic_failure"])),
        "failure_transition_reference_to_target": failure_transition(
            as_bool(reference["rhythmic_failure"]) is True,
            as_bool(target["rhythmic_failure"]) is True,
        ),
        "recovery_eligible_reference": int_flag(as_bool(reference["recovery_endpoint_eligible"])),
        "recovery_eligible_target": int_flag(as_bool(target["recovery_endpoint_eligible"])),
        "recovery_event_reference": int_flag(as_bool(reference["recovery_event_observed"])),
        "recovery_event_target": int_flag(as_bool(target["recovery_event_observed"])),
        "recovery_time_or_censor_reference": reference["recovery_time_or_censor_s"],
        "recovery_time_or_censor_target": target["recovery_time_or_censor_s"],
        "recovery_censor_time_reference": reference["recovery_censor_time_s"],
        "recovery_censor_time_target": target["recovery_censor_time_s"],
        "complete_pair": int(complete),
        "contrast_value": contrast,
        "contrast_direction": direction_label(contrast, spec["higher_is_worse"]),
    }
    return row


def build_h_contrasts(
    conditions: list[dict[str, Any]],
    spec: dict[str, Any],
    endpoint_specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_cell = {
        (row["fast_mode"], row["mt_mode"], row["pulse"]): row for row in conditions
    }
    mt_modes = spec["axes"]["mt_modes"]
    fast_modes = spec["axes"]["fast_modes"]
    pulses = spec["axes"]["pulse"]

    mt_rows: list[dict[str, Any]] = []
    for fast_mode in fast_modes:
        for pulse in pulses:
            dynamic = by_cell[(fast_mode, "dynamic", pulse)]
            for comparator in mt_modes:
                if comparator == "dynamic":
                    continue
                reference = by_cell[(fast_mode, comparator, pulse)]
                for endpoint in endpoint_specs:
                    mt_rows.append(
                        h_pair_row(
                            reference,
                            dynamic,
                            endpoint,
                            {
                                "contrast_family": "mt_dynamic_minus_comparator",
                                "fast_mode": fast_mode,
                                "pulse": pulse,
                                "mt_comparator": comparator,
                                "reference_arm": comparator,
                                "target_arm": "dynamic",
                            },
                            "MT_dynamic - MT_comparator within fast_mode and pulse",
                        )
                    )

    fast_rows: list[dict[str, Any]] = []
    for mt_mode in mt_modes:
        for pulse in pulses:
            reference = by_cell[("dynamic", mt_mode, pulse)]
            for fast_mode in fast_modes:
                if fast_mode == "dynamic":
                    continue
                target = by_cell[(fast_mode, mt_mode, pulse)]
                for endpoint in endpoint_specs:
                    fast_rows.append(
                        h_pair_row(
                            reference,
                            target,
                            endpoint,
                            {
                                "contrast_family": "fast_mode_minus_dynamic_fast",
                                "mt_mode": mt_mode,
                                "pulse": pulse,
                                "fast_comparator": fast_mode,
                                "reference_arm": "dynamic",
                                "target_arm": fast_mode,
                            },
                            "fast_mode - dynamic_fast within MT mode and pulse",
                        )
                    )

    did_rows: list[dict[str, Any]] = []
    for fast_mode in fast_modes:
        if fast_mode == "dynamic":
            continue
        for pulse in pulses:
            for mt_comparator in mt_modes:
                if mt_comparator == "dynamic":
                    continue
                four = {
                    "dynamic_fast_mt_dynamic": by_cell[("dynamic", "dynamic", pulse)],
                    "dynamic_fast_mt_comparator": by_cell[("dynamic", mt_comparator, pulse)],
                    "fast_mode_mt_dynamic": by_cell[(fast_mode, "dynamic", pulse)],
                    "fast_mode_mt_comparator": by_cell[(fast_mode, mt_comparator, pulse)],
                }
                for endpoint in endpoint_specs:
                    values = {key: h_endpoint_value(row, endpoint) for key, row in four.items()}
                    complete = all(h_endpoint_complete(row, endpoint) for row in four.values())
                    contrast_dynamic_fast = None
                    contrast_fast_mode = None
                    exact_did = None
                    if complete:
                        contrast_dynamic_fast = (
                            values["dynamic_fast_mt_dynamic"]
                            - values["dynamic_fast_mt_comparator"]
                        )
                        contrast_fast_mode = (
                            values["fast_mode_mt_dynamic"]
                            - values["fast_mode_mt_comparator"]
                        )
                        exact_did = contrast_fast_mode - contrast_dynamic_fast
                    row: dict[str, Any] = {
                        "contrast_family": "mt_contrast_by_fast_mode_exact_did",
                        "fast_comparator": fast_mode,
                        "pulse": pulse,
                        "mt_comparator": mt_comparator,
                        "endpoint": endpoint["endpoint"],
                        "endpoint_family": endpoint["family"],
                        "unit": endpoint["unit"],
                        "higher_is_worse": int_flag(endpoint["higher_is_worse"]),
                        "formula": "(MT_dynamic - MT_comparator at fast_mode) - (MT_dynamic - MT_comparator at dynamic_fast)",
                        "neutral_tolerance": NEUTRAL_TOLERANCE,
                        "complete_pair": int(complete),
                        "mt_contrast_dynamic_fast": contrast_dynamic_fast,
                        "mt_contrast_fast_mode": contrast_fast_mode,
                        "exact_difference_in_differences": exact_did,
                        "contrast_direction": direction_label(exact_did, endpoint["higher_is_worse"]),
                        "failure_transition_dynamic_fast_mt_comparator_to_dynamic": failure_transition(
                            as_bool(four["dynamic_fast_mt_comparator"]["rhythmic_failure"]) is True,
                            as_bool(four["dynamic_fast_mt_dynamic"]["rhythmic_failure"]) is True,
                        ),
                        "failure_transition_fast_mode_mt_comparator_to_dynamic": failure_transition(
                            as_bool(four["fast_mode_mt_comparator"]["rhythmic_failure"]) is True,
                            as_bool(four["fast_mode_mt_dynamic"]["rhythmic_failure"]) is True,
                        ),
                    }
                    for key, condition in four.items():
                        row[f"task_id_{key}"] = condition["task_id"]
                        row[f"simulation_id_{key}"] = condition["simulation_id"]
                        row[f"value_{key}"] = values[key]
                        row[f"rhythmic_failure_{key}"] = int_flag(as_bool(condition["rhythmic_failure"]))
                        num, den, matched = h_components(condition, endpoint)
                        row[f"raw_numerator_{key}"] = num
                        row[f"raw_denominator_{key}"] = den
                        row[f"raw_matched_{key}"] = matched
                        row[f"recovery_eligible_{key}"] = int_flag(as_bool(condition["recovery_endpoint_eligible"]))
                        row[f"recovery_event_{key}"] = int_flag(as_bool(condition["recovery_event_observed"]))
                        row[f"recovery_time_or_censor_{key}"] = condition["recovery_time_or_censor_s"]
                        row[f"recovery_censor_time_{key}"] = condition["recovery_censor_time_s"]
                    did_rows.append(row)
    return mt_rows, fast_rows, did_rows


def main() -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    script_sha256 = sha256_file(HERE)
    protocol_spec_sha256 = sha256_file(SPEC_PATH)
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    source_dir = (ANALYSIS_ROOT / spec["source_directory"]).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    checks: dict[str, bool] = {}
    observed_hashes: dict[str, str] = {}
    for name, expected_hash in spec["source_sha256"].items():
        path = source_dir / name
        actual_hash = sha256_file(path)
        observed_hashes[name] = actual_hash
        require(actual_hash == expected_hash, f"source_sha256::{name}", checks)

    index_rows = read_csv(source_dir / "analysis_task_index.csv")
    metric_rows = read_csv(source_dir / "metrics.csv")
    epoch_rows = read_csv(source_dir / "long_epoch_metrics.csv")
    expected = spec["expected_counts"]
    require(len(index_rows) == expected["analysis_tasks"], "analysis_task_count", checks)
    require(len(metric_rows) == expected["analysis_tasks"], "metrics_row_count", checks)
    require(len(epoch_rows) == expected["long_epoch_rows"], "G_epoch_source_row_count", checks)

    metric_by_task = {row["task_id"]: row for row in metric_rows}
    index_by_sim = {row["simulation_id"]: row for row in index_rows}
    g_index = [row for row in index_rows if row["stage"] == "G"]
    h_metrics = [row for row in metric_rows if row["stage"] == "H"]
    require(len(g_index) == 31, "G_simulation_count_31", checks)
    require(len({row["simulation_id"] for row in g_index}) == 31, "G_simulation_ids_unique", checks)
    require(len(h_metrics) == 72, "H_condition_count_72", checks)

    routes = spec["axes"]["routes"]
    locked_windows = spec["long_windows"]
    epoch_keys = [(row["simulation_id"], int(row["epoch"])) for row in epoch_rows]
    require(len(epoch_keys) == 744 and len(set(epoch_keys)) == 744, "G_744_unique_epoch_keys", checks)
    require(
        {sim for sim, _ in epoch_keys} == {row["simulation_id"] for row in g_index},
        "G_epoch_simulation_set_exact",
        checks,
    )
    require(
        all(count == 24 for count in Counter(sim for sim, _ in epoch_keys).values()),
        "G_24_epochs_per_simulation",
        checks,
    )
    require({epoch for _, epoch in epoch_keys} == set(range(1, 25)), "G_epoch_domain_1_to_24", checks)

    g_meta_by_sim: dict[str, dict[str, Any]] = {}
    for index_row in g_index:
        route, arm, challenge_flag, impairment_flag = classify_g_arm(index_row)
        metric = metric_by_task[index_row["task_id"]]
        require(metric["simulation_id"] == index_row["simulation_id"], f"G_task_sim::{index_row['task_id']}", checks)
        g_meta_by_sim[index_row["simulation_id"]] = {
            "task_id": index_row["task_id"],
            "simulation_id": index_row["simulation_id"],
            "seed": index_row["seed"],
            "protocol": index_row["protocol"],
            "route_scope": route,
            "arm": arm,
            "challenge_flag": challenge_flag,
            "impairment_flag": impairment_flag,
            "scientific_valid": metric["scientific_valid"],
        }

    g_epoch_output: list[dict[str, Any]] = []
    epoch_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    epoch_source_fields = list(epoch_rows[0].keys())
    for source in sorted(epoch_rows, key=lambda row: (row["simulation_id"], int(row["epoch"]))):
        meta = g_meta_by_sim[source["simulation_id"]]
        epoch = int(source["epoch"])
        locked_window, included = window_for_epoch(epoch, locked_windows)
        epoch_groups[(source["simulation_id"], locked_window)].append(source)
        pf_anchor = as_int(source["pf_transfer_anchor_count"])
        pf_missed = as_int(source["pf_transfer_missed_count"])
        pf_matched = as_int(source["pf_transfer_matched_count"])
        mn_anchor = as_int(source["mn_transfer_anchor_count"])
        mn_missed = as_int(source["mn_transfer_missed_count"])
        mn_matched = as_int(source["mn_transfer_matched_count"])
        balance = as_float(source["bilateral_amplitude_balance"])
        payload: dict[str, Any] = {
            "task_id": meta["task_id"],
            "seed": meta["seed"],
            "protocol": meta["protocol"],
            "route_scope": meta["route_scope"],
            "arm": meta["arm"],
            "challenge_flag": meta["challenge_flag"],
            "impairment_flag": meta["impairment_flag"],
            "locked_window": locked_window,
            "included_in_locked_summary": int(included),
            "scientific_valid": int_flag(as_bool(meta["scientific_valid"])),
            "epoch_continuous_status_complete": int(
                as_bool(meta["scientific_valid"]) is True
                and as_bool(source["technical_valid"]) is True
                and as_bool(source["rhythmic_failure"]) is False
            ),
            "bilateral_amplitude_imbalance": None if balance is None else 1.0 - balance,
            "lr_phase_slip_fraction_derived": safe_ratio(source["lr_phase_slip_count"], source["lr_phase_cycle_count"]),
            "fe_phase_slip_fraction_derived": safe_ratio(source["fe_phase_slip_count"], source["fe_phase_cycle_count"]),
            "pf_network_propagation_gap": safe_ratio(pf_missed, pf_anchor),
            "mn_network_propagation_gap": safe_ratio(mn_missed, mn_anchor),
            "pf_transfer_identity_ok": int(
                pf_anchor is not None and pf_missed is not None and pf_matched is not None
                and pf_missed + pf_matched == pf_anchor
            ),
            "mn_transfer_identity_ok": int(
                mn_anchor is not None and mn_missed is not None and mn_matched is not None
                and mn_missed + mn_matched == mn_anchor
            ),
        }
        for field in epoch_source_fields:
            renamed = "source_epoch_stage" if field == "stage" else field
            payload[renamed] = source[field]
        g_epoch_output.append(payload)

    require(sum(row["included_in_locked_summary"] == 0 for row in g_epoch_output) == 62, "G_epoch_1_19_display_only", checks)
    require(
        all(row["pf_transfer_identity_ok"] and row["mn_transfer_identity_ok"] for row in g_epoch_output),
        "G_matched_plus_missed_equals_anchor",
        checks,
    )

    g_window_rows: list[dict[str, Any]] = []
    summary_windows = ["baseline", "stress_prechallenge", "stress_challenge", "recovery"]
    for sim in sorted(g_meta_by_sim):
        for window in summary_windows:
            rows = epoch_groups[(sim, window)]
            expected_epochs = locked_windows[window]
            require(
                sorted(int(row["epoch"]) for row in rows) == expected_epochs,
                f"G_window_epochs::{sim}::{window}",
                checks,
            )
            meta = {**g_meta_by_sim[sim], "window": window}
            g_window_rows.append(aggregate_g_window(rows, meta, routes))
    add_g_baseline_changes(g_window_rows)
    require(len(g_window_rows) == 31 * 4, "G_window_arm_rows_124", checks)

    arm_counts = Counter((row["route_scope"], row["arm"]) for row in g_window_rows if row["window"] == "baseline")
    require(arm_counts[("shared_intact", "no_challenge_intact")] == 1, "G_shared_intact_arm", checks)
    for route in routes:
        for arm in ("no_challenge_impaired", "challenge_intact", "challenge_impaired"):
            require(arm_counts[(route, arm)] == 1, f"G_arm_complete::{route}::{arm}", checks)

    g_did_rows = build_g_did_rows(g_window_rows, routes, summary_windows)
    require(len(g_did_rows) == 10 * 4 * 15, "G_route_window_endpoint_DiD_rows_600", checks)

    h_conditions = build_h_condition_rows(h_metrics, routes)
    h_cells = {(row["fast_mode"], row["mt_mode"], row["pulse"]) for row in h_conditions}
    expected_h_cells = {
        (fast, mt, pulse)
        for fast in spec["axes"]["fast_modes"]
        for mt in spec["axes"]["mt_modes"]
        for pulse in spec["axes"]["pulse"]
    }
    require(h_cells == expected_h_cells, "H_4x6x3_complete", checks)
    require(all(row["pf_transfer_identity_ok"] and row["mn_transfer_identity_ok"] for row in h_conditions), "H_matched_plus_missed_equals_anchor", checks)
    endpoint_specs = h_endpoint_specs(routes)
    h_mt_rows, h_fast_rows, h_did_rows = build_h_contrasts(h_conditions, spec, endpoint_specs)
    endpoint_count = len(endpoint_specs)
    require(len(h_mt_rows) == 4 * 3 * 5 * endpoint_count, "H_MT_contrast_row_count", checks)
    require(len(h_fast_rows) == 6 * 3 * 3 * endpoint_count, "H_fast_contrast_row_count", checks)
    require(len(h_did_rows) == 3 * 3 * 5 * endpoint_count, "H_MT_by_fast_DiD_row_count", checks)

    common_sources_g = ["analysis_task_index.csv", "metrics.csv", "long_epoch_metrics.csv"]
    common_sources_h = ["analysis_task_index.csv", "metrics.csv"]

    g_epoch_data_fields = [
        "task_id", "seed", "protocol", "route_scope", "arm", "challenge_flag",
        "impairment_flag", "locked_window", "included_in_locked_summary",
        "scientific_valid", "epoch_continuous_status_complete",
        "bilateral_amplitude_imbalance", "lr_phase_slip_fraction_derived",
        "fe_phase_slip_fraction_derived", "pf_network_propagation_gap",
        "mn_network_propagation_gap", "pf_transfer_identity_ok",
        "mn_transfer_identity_ok",
        *[("source_epoch_stage" if field == "stage" else field) for field in epoch_source_fields],
    ]
    g_window_base_fields = [
        "task_id", "simulation_id", "route_scope", "arm", "challenge_flag",
        "impairment_flag", "window", "epochs", "epoch_count",
        "all_epochs_technical_valid", "scientific_valid", "failure_epoch_count",
        "rhythmic_failure_any", "rhythmic_failure_epoch_fraction",
        "continuous_window_status_complete", "frequency_hz",
        "rg_cycle_interval_cv_mean", "lr_phase_error_mean_abs_deg",
        "fe_phase_error_mean_abs_deg", "bilateral_amplitude_balance",
        "bilateral_amplitude_imbalance", "lr_phase_slip_count_sum",
        "lr_phase_cycle_count_sum", "lr_phase_slip_fraction",
        "lr_phase_slip_fraction_cycle_pooled_sensitivity",
        "fe_phase_slip_count_sum", "fe_phase_cycle_count_sum",
        "fe_phase_slip_fraction", "fe_phase_slip_fraction_cycle_pooled_sensitivity",
        "pf_transfer_anchor_count_sum",
        "pf_transfer_missed_count_sum", "pf_transfer_matched_count_sum",
        "pf_network_propagation_gap",
        "pf_network_propagation_gap_anchor_pooled_sensitivity",
        "pf_transfer_identity_ok",
        "mn_transfer_anchor_count_sum", "mn_transfer_missed_count_sum",
        "mn_transfer_matched_count_sum", "mn_network_propagation_gap",
        "mn_network_propagation_gap_anchor_pooled_sensitivity",
        "mn_transfer_identity_ok", "challenged_rrp_mean_secondary",
        "challenged_replenishment_resource_mean_secondary",
    ]
    g_route_state_fields: list[str] = []
    for route in routes:
        g_route_state_fields.extend(
            [f"mt_{route}_mean", f"rrp_{route}_mean_secondary", f"replenishment_resource_{route}_mean_secondary"]
        )
    g_baseline_change_fields = ["failure_transition_from_baseline", "complete_status_vs_baseline"] + [
        f"delta_{field}_from_baseline"
        for field in [
            "lr_phase_error_mean_abs_deg", "fe_phase_error_mean_abs_deg",
            "lr_phase_slip_fraction", "fe_phase_slip_fraction", "frequency_hz",
            "rg_cycle_interval_cv_mean", "bilateral_amplitude_imbalance",
            "pf_network_propagation_gap", "mn_network_propagation_gap",
        ]
    ]
    g_did_fields = [
        "route", "window", "endpoint", "endpoint_family", "unit",
        "higher_is_worse", "aggregation_rule", "formula", "neutral_tolerance",
        "all_arms_status_valid",
        "all_arms_rhythmic", "complete_pair", "contrast_value", "contrast_direction",
        "sensitivity_complete_pair", "sensitivity_contrast_value",
        "challenge_effect_transition_intact", "challenge_effect_transition_impaired",
        "impairment_effect_transition_no_challenge", "impairment_effect_transition_challenge",
    ]
    g_arms = ["no_challenge_intact", "no_challenge_impaired", "challenge_intact", "challenge_impaired"]
    for arm in g_arms:
        g_did_fields.extend(
            [
                f"task_id_{arm}", f"simulation_id_{arm}", f"value_{arm}",
                f"rhythmic_failure_{arm}", f"raw_numerator_{arm}",
                f"raw_denominator_{arm}", f"raw_matched_{arm}",
                f"sensitivity_value_{arm}",
            ]
        )

    h_condition_id_fields = [
        "task_id", "simulation_id", "seed", "protocol", "speed", "load",
        "load_side", "pulse", "fast_mode", "mt_mode", "label",
        "scientific_valid", "technical_valid", "technical_exclusion_reason",
        "rhythmic_failure", "active_pulse", "pulse_required", "pulse_delivered",
        "pulse_response_eligible", "pulse_noneligibility_reason",
        "recovery_endpoint_eligible", "recovery_event_observed",
        "recovery_time_s", "recovery_time_or_censor_s", "recovery_censor_time_s",
        "recovery_ineligibility_reason", "recovery_failure",
        "condition_status_complete",
    ]
    h_condition_analysis_fields = [
        "analysis_lr_phase_error_mean_abs_deg", "analysis_fe_phase_error_mean_abs_deg",
        "analysis_lr_phase_slip_count", "analysis_lr_phase_cycle_count",
        "analysis_lr_phase_slip_fraction", "analysis_fe_phase_slip_count",
        "analysis_fe_phase_cycle_count", "analysis_fe_phase_slip_fraction",
        "frequency_hz", "rg_cycle_interval_cv_mean", "bilateral_amplitude_balance",
        "bilateral_amplitude_imbalance", "pf_transfer_anchor_count",
        "pf_transfer_missed_count", "pf_transfer_matched_count",
        "pf_network_propagation_gap", "pf_transfer_identity_ok",
        "mn_transfer_anchor_count", "mn_transfer_missed_count",
        "mn_transfer_matched_count", "mn_network_propagation_gap",
        "mn_transfer_identity_ok", "rg_pf_latency_mean_ms", "rg_mn_latency_mean_ms",
        "mean_mt_support_left", "mean_mt_support_right", "ia_signal_mean",
        "ib_signal_mean", "muscle_force_mean",
    ]
    h_mechanism_fields = [f"{route}_mean_rate_hz" for route in routes]
    for route in routes:
        h_mechanism_fields.extend(
            [f"mt_{route}_mean", f"rrp_{route}_mean", f"replenishment_resource_{route}_mean"]
        )

    h_pair_base_fields = [
        "contrast_family", "fast_mode", "mt_mode", "pulse", "mt_comparator",
        "fast_comparator", "reference_arm", "target_arm", "endpoint",
        "endpoint_family", "unit", "higher_is_worse", "formula",
        "neutral_tolerance", "task_id_reference", "task_id_target",
        "simulation_id_reference", "simulation_id_target", "value_reference",
        "value_target", "raw_numerator_reference", "raw_denominator_reference",
        "raw_matched_reference", "raw_numerator_target", "raw_denominator_target",
        "raw_matched_target", "rhythmic_failure_reference", "rhythmic_failure_target",
        "failure_transition_reference_to_target", "recovery_eligible_reference",
        "recovery_eligible_target", "recovery_event_reference", "recovery_event_target",
        "recovery_time_or_censor_reference", "recovery_time_or_censor_target",
        "recovery_censor_time_reference", "recovery_censor_time_target",
        "complete_pair", "contrast_value", "contrast_direction",
    ]
    h_did_base_fields = [
        "contrast_family", "fast_comparator", "pulse", "mt_comparator", "endpoint",
        "endpoint_family", "unit", "higher_is_worse", "formula", "neutral_tolerance",
        "complete_pair", "mt_contrast_dynamic_fast", "mt_contrast_fast_mode",
        "exact_difference_in_differences", "contrast_direction",
        "failure_transition_dynamic_fast_mt_comparator_to_dynamic",
        "failure_transition_fast_mode_mt_comparator_to_dynamic",
    ]
    h_did_arms = [
        "dynamic_fast_mt_dynamic", "dynamic_fast_mt_comparator",
        "fast_mode_mt_dynamic", "fast_mode_mt_comparator",
    ]
    for arm in h_did_arms:
        h_did_base_fields.extend(
            [
                f"task_id_{arm}", f"simulation_id_{arm}", f"value_{arm}",
                f"rhythmic_failure_{arm}", f"raw_numerator_{arm}",
                f"raw_denominator_{arm}", f"raw_matched_{arm}",
                f"recovery_eligible_{arm}", f"recovery_event_{arm}",
                f"recovery_time_or_censor_{arm}", f"recovery_censor_time_{arm}",
            ]
        )

    output_headers: dict[str, list[str]] = {}
    output_paths = {
        "g_epoch": OUT_DIR / "g_h_g_epoch_atomic.csv",
        "g_window": OUT_DIR / "g_h_g_window_arms.csv",
        "g_did": OUT_DIR / "g_h_g_route_did.csv",
        "h_conditions": OUT_DIR / "g_h_h_conditions.csv",
        "h_mt": OUT_DIR / "g_h_h_mt_contrasts.csv",
        "h_fast": OUT_DIR / "g_h_h_fast_contrasts.csv",
        "h_did": OUT_DIR / "g_h_h_mt_by_fast_did.csv",
    }
    output_headers["g_epoch"] = write_csv_with_provenance(
        output_paths["g_epoch"], g_epoch_output, g_epoch_data_fields, common_sources_g,
        generated_at, script_sha256, protocol_spec_sha256, observed_hashes,
    )
    output_headers["g_window"] = write_csv_with_provenance(
        output_paths["g_window"], g_window_rows,
        g_window_base_fields + g_route_state_fields + g_baseline_change_fields,
        common_sources_g, generated_at, script_sha256, protocol_spec_sha256, observed_hashes,
    )
    output_headers["g_did"] = write_csv_with_provenance(
        output_paths["g_did"], g_did_rows, g_did_fields, common_sources_g,
        generated_at, script_sha256, protocol_spec_sha256, observed_hashes,
    )
    output_headers["h_conditions"] = write_csv_with_provenance(
        output_paths["h_conditions"], h_conditions,
        h_condition_id_fields + h_condition_analysis_fields + h_mechanism_fields,
        common_sources_h, generated_at, script_sha256, protocol_spec_sha256, observed_hashes,
    )
    output_headers["h_mt"] = write_csv_with_provenance(
        output_paths["h_mt"], h_mt_rows, h_pair_base_fields, common_sources_h,
        generated_at, script_sha256, protocol_spec_sha256, observed_hashes,
    )
    output_headers["h_fast"] = write_csv_with_provenance(
        output_paths["h_fast"], h_fast_rows, h_pair_base_fields, common_sources_h,
        generated_at, script_sha256, protocol_spec_sha256, observed_hashes,
    )
    output_headers["h_did"] = write_csv_with_provenance(
        output_paths["h_did"], h_did_rows, h_did_base_fields, common_sources_h,
        generated_at, script_sha256, protocol_spec_sha256, observed_hashes,
    )

    capsule_source = source_dir / "single_realization_contrasts_v2_6_2.csv"
    capsule_output = OUT_DIR / "g_h_locked_10_contrast_capsule.csv"
    shutil.copyfile(capsule_source, capsule_output)
    capsule_rows = read_csv(capsule_output)
    capsule_identical = capsule_source.read_bytes() == capsule_output.read_bytes()
    require(capsule_identical, "locked_capsule_byte_identical", checks)
    require(len(capsule_rows) == 10, "locked_capsule_ten_rows", checks)
    capsule_metadata_path = OUT_DIR / "g_h_locked_10_contrast_capsule.metadata.json"
    capsule_metadata = {
        "schema": "cpg-g-h-locked-capsule-metadata-1.0",
        "generated_at_utc": generated_at,
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_sha256,
        "protocol_spec_sha256": protocol_spec_sha256,
        "source_single_realization_contrasts_v2_6_2_sha256": observed_hashes[
            "single_realization_contrasts_v2_6_2.csv"
        ],
        "output_sha256": sha256_file(capsule_output),
        "output_row_count": len(capsule_rows),
        "byte_identical_to_frozen_source": capsule_identical,
        "analysis_role": "separate locked preregistration capsule; not a global pass/fail rule",
        "independent_stochastic_realization_count": 1,
    }
    capsule_metadata_path.write_text(
        json.dumps(capsule_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    forbidden = {field.lower() for field in spec["forbidden_inference_fields"]}
    forbidden_hits: dict[str, list[str]] = {}
    for key, headers in output_headers.items():
        hits = sorted(field for field in headers if field.lower() in forbidden)
        if hits:
            forbidden_hits[output_paths[key].name] = hits
    capsule_header = list(capsule_rows[0].keys()) if capsule_rows else []
    capsule_hits = sorted(field for field in capsule_header if field.lower() in forbidden)
    if capsule_hits:
        forbidden_hits[capsule_output.name] = capsule_hits
    require(not forbidden_hits, "no_forbidden_inference_fields", checks)

    output_row_counts = {
        output_paths["g_epoch"].name: len(g_epoch_output),
        output_paths["g_window"].name: len(g_window_rows),
        output_paths["g_did"].name: len(g_did_rows),
        output_paths["h_conditions"].name: len(h_conditions),
        output_paths["h_mt"].name: len(h_mt_rows),
        output_paths["h_fast"].name: len(h_fast_rows),
        output_paths["h_did"].name: len(h_did_rows),
        capsule_output.name: len(capsule_rows),
        capsule_metadata_path.name: 1,
    }
    output_files = [*output_paths.values(), capsule_output, capsule_metadata_path]
    output_manifest = {
        path.name: {
            "row_count": output_row_counts[path.name],
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in output_files
    }

    g_complete_counts = Counter(
        (row["window"], row["complete_pair"]) for row in g_did_rows
    )
    h_complete_counts = {
        "mt": dict(Counter(row["complete_pair"] for row in h_mt_rows)),
        "fast": dict(Counter(row["complete_pair"] for row in h_fast_rows)),
        "mt_by_fast": dict(Counter(row["complete_pair"] for row in h_did_rows)),
    }
    capsule_directions = Counter(row["interpretation"] for row in capsule_rows)

    qc_path = OUT_DIR / "g_h_qc.json"
    qc_payload = {
        "schema": "cpg-g-h-analysis-qc-1.0",
        "generated_at_utc": generated_at,
        "script_version": SCRIPT_VERSION,
        "script": str(HERE.relative_to(REPOSITORY_ROOT)),
        "script_sha256": script_sha256,
        "protocol_spec": str(SPEC_PATH.relative_to(REPOSITORY_ROOT)),
        "protocol_spec_sha256": protocol_spec_sha256,
        "source_directory": str(source_dir.relative_to(REPOSITORY_ROOT)),
        "source_hashes": {
            name: {
                "expected": spec["source_sha256"][name],
                "actual": observed_hashes[name],
                "verified": observed_hashes[name] == spec["source_sha256"][name],
            }
            for name in sorted(observed_hashes)
        },
        "counts": {
            "G_simulations": len(g_index),
            "G_epoch_rows": len(g_epoch_output),
            "G_display_only_epoch_rows": sum(
                row["included_in_locked_summary"] == 0 for row in g_epoch_output
            ),
            "G_window_arm_rows": len(g_window_rows),
            "G_route_window_endpoint_DiD_rows": len(g_did_rows),
            "G_rhythmic_failure_epochs": sum(
                as_bool(row["rhythmic_failure"]) is True for row in epoch_rows
            ),
            "H_conditions": len(h_conditions),
            "H_endpoint_count": endpoint_count,
            "H_MT_contrast_rows": len(h_mt_rows),
            "H_fast_contrast_rows": len(h_fast_rows),
            "H_MT_by_fast_DiD_rows": len(h_did_rows),
            "H_rhythmic_failure_conditions": sum(
                as_bool(row["rhythmic_failure"]) is True for row in h_conditions
            ),
            "locked_capsule_rows": len(capsule_rows),
        },
        "locked_windows": locked_windows,
        "G_complete_pair_counts_by_window": {
            f"{window}::{complete}": count
            for (window, complete), count in sorted(g_complete_counts.items())
        },
        "H_complete_pair_counts": h_complete_counts,
        "locked_capsule_interpretation_counts": dict(sorted(capsule_directions.items())),
        "forbidden_inference_field_hits": forbidden_hits,
        "outputs": output_manifest,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "independent_stochastic_realization_count": 1,
        "inference_scope": "descriptive paired mechanism analysis conditional on one frozen realization",
        "blank_handling": "blank values were not converted to zero",
    }
    qc_path.write_text(
        json.dumps(qc_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qc_payload["counts"], ensure_ascii=False, indent=2))
    print(f"all_checks_pass={qc_payload['all_checks_pass']}")
    print(qc_path)


if __name__ == "__main__":
    main()
