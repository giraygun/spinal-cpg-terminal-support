#!/usr/bin/env python3
"""Strict causal and structural preflight for scientific v2.6.1 results.

The preflight runs before any primary result is opened.  It verifies the exact
A--H task matrix, canonical checkpoints, event-derived summaries, the
dynamic-MT versus static-mean-matched primary contrast, and the observable
four-cell challenge x MT-impairment design used by primary family 10.

Primary endpoint code is inspected as data-flow code: hidden mechanistic states
(MT, RRP, reserve, resource, damage, support, or calcium) may be retained as
secondary diagnostics, but they may not be read as primary outcomes.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import re
from typing import Dict, Mapping, Sequence

import numpy as np

import dual_timescale_spinal_cpg_v2_6_1_candidate as model
import run_ah_experiments_v2_6_1 as reference
import run_ah_experiments_accelerated_v2_6_1 as accelerated


EXPECTED_MODEL_VERSION_PREFIX = "distributed-local-terminal-mt-cpg-2.6.1-"
EXPECTED_STAGE_TASKS_PER_SEED = {
    "A": 27,
    "B": 270,
    "C": 162,
    "D": 54,
    "E": 270,
    "F": 10800,
    "G": 31,
    "H": 72,
}
EXPECTED_STAGE_TASK_TOTALS = {
    "A": 540,
    "B": 5_400,
    "C": 3_240,
    "D": 1_080,
    "E": 5_400,
    "F": 216_000,
    "G": 4_092,
    "H": 9_504,
}
EXPECTED_ANALYSIS_TASKS = 245_256
EXPECTED_UNIQUE_SIMULATIONS = 83_796
EXPECTED_TASK_IDENTITY_SHA256 = (
    "cbc3322ed317bab95e891e3a35804ff0c0e2f1365dfcc772273dd893815d5214"
)
EXPECTED_VALIDATION_SEEDS = tuple(range(401, 421))
EXPECTED_PRIMARY_SEEDS = tuple(range(601, 733))
EXPECTED_PRODUCTION_SEEDS_BY_STAGE = {
    **{stage: EXPECTED_VALIDATION_SEEDS for stage in "ABCDEF"},
    **{stage: EXPECTED_PRIMARY_SEEDS for stage in "GH"},
}
EXPECTED_PLAN_SEEDS = tuple(sorted({
    seed
    for seeds in EXPECTED_PRODUCTION_SEEDS_BY_STAGE.values()
    for seed in seeds
}))
EXPECTED_CLASSES = (
    "RG", "PF", "MN", "V0D", "V0V", "V2a", "V3", "V1Ia", "V1Ren", "V2b",
)
EXPECTED_MT_ROUTES = EXPECTED_CLASSES
EXPECTED_PRIMARY_FAMILIES = tuple(range(1, 11))
EXPECTED_PRIMARY_FAMILY_REGISTRY = {
    1: "lr_absolute_phase_error_burden",
    2: "fe_absolute_phase_error_burden",
    3: "lr_phase_slip_probability_burden",
    4: "fe_phase_slip_probability_burden",
    5: "bounded_cycle_interval_cv_burden",
    6: "bilateral_mn_amplitude_imbalance_burden",
    7: "rg_to_pf_missed_transfer_probability_burden",
    8: "rg_to_mn_missed_transfer_probability_burden",
    9: "four_second_nonrecovery_burden_s",
    10: "postchallenge_rg_to_mn_transfer_deficit_difference_in_differences",
}
EXPECTED_H_MT_MODES = {
    "dynamic", "static_matched", "time_yoked", "spatial_shuffled",
    "impaired", "off",
}
EXPECTED_H_FAST_MODES = {"dynamic", "static_mean", "yoked", "off"}
EXPECTED_PULSES = {"none", "excitatory", "inhibitory"}
EXPECTED_PRIMARY_ANALYSIS_CONTRACT = {
    "contract_schema": "v2.5-primary-analysis-contract-1.0",
    "alpha_per_family_two_sided": 0.005,
    "paired_seed_count": 132,
    "paired_seed_range_inclusive": [601, 732],
    "favorable_direction": "negative",
    "families_1_to_9_comparator": "static_matched",
    "families_1_to_8_pulse_contexts": [
        "none", "excitatory", "inhibitory",
    ],
    "family_9_pulse_contexts": ["excitatory", "inhibitory"],
    "family_9_recovery_horizon_s": 4.0,
    "primary_contexts": {
        "families_1_to_9": {
            "speed": "medium", "load": "normal", "fast_mode": "dynamic",
        },
        "family_10": {
            "speed": "medium", "load": "normal", "protocol": "long",
            "pulse": "none",
        },
    },
    "probability_burden": {
        "transform": "jeffreys_corrected_arcsine_square_root",
        "formula": "asin(sqrt((events+0.5)/(trials+1.0)))",
        "zero_denominator_biological_failure": "pi_over_2",
    },
    "worst_burden_policy": {
        "absolute_phase_error_deg": 180.0,
        "bounded_cycle_interval_cv": 1.0,
        "bilateral_mn_amplitude_imbalance": 1.0,
        "probability_burden": "pi_over_2",
        "recovery_time_s": 4.0,
    },
    "family_6": {
        "endpoint": "bilateral_mn_amplitude_imbalance",
        "formula": "abs(mean_left_mn_rate-mean_right_mn_rate)/(mean_left_mn_rate+mean_right_mn_rate)",
        "sufficient_observables": [
            "mn_left_rate_sum_hz_samples",
            "mn_right_rate_sum_hz_samples",
            "mn_rate_sample_count",
        ],
    },
    "burst_detection": {
        "rate_tau_ms": 20.0,
        "RG_on_hz": 16.0,
        "RG_off_hz": 7.0,
        "PF_on_hz": 14.0,
        "PF_off_hz": 6.0,
        "MN_on_hz": 10.0,
        "MN_off_hz": 4.0,
        "minimum_interburst_s": 0.22,
    },
    "phase_slip_threshold_deg": 45.0,
    "transfer_windows_s": {
        "RG_to_PF": {"pre": 0.0, "post": 0.25},
        "RG_to_MN": {"pre": 0.0, "post": 0.25},
    },
    "recovery_definition": {
        "consecutive_cycles": 3,
        "frequency_tolerance_fraction": 0.25,
        "phase_tolerance_deg": 45.0,
    },
    "family_9_observable_contract": {
        "event_streams": [
            "RG_L_F_onset_s", "RG_L_E_onset_s",
            "RG_R_F_onset_s", "RG_R_E_onset_s",
        ],
        "intervention_log_fields": [
            "pulse_required", "pulse_delivered", "pulse_start_s",
            "pulse_end_s", "pulse_noneligibility_reason",
        ],
        "biological_nondelivery_reason": (
            "biological_no_phase_eligible_cycle"
        ),
    },
    "family_10": {
        "baseline_epochs_inclusive": [2, 6],
        "recovery_epochs_inclusive": [20, 24],
        "route_reduction": "unweighted_mean_within_seed",
        "contrast": "(challenge_intact-no_challenge_intact)-(challenge_impaired-no_challenge_impaired)",
    },
}
EXPECTED_FAMILY6_OBSERVABLE_KEYS = tuple(
    EXPECTED_PRIMARY_ANALYSIS_CONTRACT["family_6"]["sufficient_observables"]
)
EXPECTED_FAMILY9_INTERVENTION_KEYS = (
    "pulse_required", "pulse_delivered", "pulse_start_s", "pulse_end_s",
    "pulse_noneligibility_reason",
)
EXPECTED_FAMILY9_RG_EVENT_KEYS = (
    "RG_L_F_onset_s", "RG_L_E_onset_s",
    "RG_R_F_onset_s", "RG_R_E_onset_s",
)

# These terms identify simulator-internal mediators, not independent observable
# outcomes.  The check is deliberately applied to literal data keys read by the
# primary-analysis source, rather than to comments, variable names, or the
# legitimate intervention name ``mt_mode``.
FORBIDDEN_PRIMARY_ENDPOINT_TERMS = (
    "microtubule", "rrp", "reserve", "resource", "damage",
    "calcium", "capacity", "support",
)
ALLOWED_INTERVENTION_KEYS = {
    "mt_mode", "impaired_mt_routes", "challenged_routes", "fast_mode",
}
# Fail closed: every literal mapping key currently read by the frozen primary
# analysis is registered here.  A newly introduced, innocuously named proxy for
# a hidden mediator is therefore blocked even when it evades the term filter.
ALLOWED_PRIMARY_DATA_KEYS = {
    "all_checks_pass",
    "analysis_events",
    "analysis_observables",
    "family_10_contrast",
    "fe_phase_cycle_count",
    "fe_phase_error_mean_abs_deg",
    "fe_phase_slip_count",
    "long_no_challenge",
    "lr_phase_cycle_count",
    "lr_phase_error_mean_abs_deg",
    "lr_phase_slip_count",
    "mn_transfer_anchor_count",
    "mn_transfer_missed_count",
    "pf_transfer_anchor_count",
    "pf_transfer_missed_count",
    "intervention_log",
    "mn_left_rate_sum_hz_samples",
    "mn_right_rate_sum_hz_samples",
    "mn_rate_sample_count",
    "pulse_delivered",
    "pulse_end_s",
    "pulse_noneligibility_reason",
    "pulse_required",
    "pulse_start_s",
    "recovery_endpoint_eligible",
    "recovery_time_or_censor_s",
    "rg_cycle_interval_cv_mean",
    "rhythmic_failure",
    "scientific_valid",
    "summary",
    "technical_valid",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_number(observed: object, expected: float, atol: float = 1e-12) -> bool:
    """Compare a strict-JSON number/None with a finite or NaN expectation."""
    if observed is None and not np.isfinite(expected):
        return True
    if observed is None or not np.isfinite(expected):
        return False
    return bool(np.isclose(float(observed), expected, rtol=0.0, atol=atol))


def probability_burden(events: int, trials: int, failure: int) -> float:
    """Independent copy of the frozen primary probability transformation."""
    if trials > 0:
        if not 0 <= events <= trials:
            raise ValueError("invalid event/trial counts")
        return float(math.asin(math.sqrt((events + 0.5) / (trials + 1.0))))
    if failure == 1:
        return math.pi / 2.0
    raise ValueError("zero denominator without rhythmic failure")


def primary_runtime_contract_violations() -> list[str]:
    """Compare every outcome-defining runtime constant to the preregistration."""
    cfg = model.Config()
    expected = EXPECTED_PRIMARY_ANALYSIS_CONTRACT
    observed = {
        "paired_seed_count": len(reference.PREREGISTERED_SEEDS),
        "paired_seed_range_inclusive": [
            min(reference.PREREGISTERED_SEEDS),
            max(reference.PREREGISTERED_SEEDS),
        ],
        "burst_detection": {
            "rate_tau_ms": cfg.rate_tau_ms,
            "RG_on_hz": cfg.burst_on_threshold_hz,
            "RG_off_hz": cfg.burst_off_threshold_hz,
            "PF_on_hz": cfg.pf_burst_on_threshold_hz,
            "PF_off_hz": cfg.pf_burst_off_threshold_hz,
            "MN_on_hz": cfg.mn_burst_on_threshold_hz,
            "MN_off_hz": cfg.mn_burst_off_threshold_hz,
            "minimum_interburst_s": cfg.minimum_interburst_s,
        },
        "phase_slip_threshold_deg": cfg.phase_slip_threshold_deg,
        "transfer_windows_s": {
            "RG_to_PF": {"pre": 0.0, "post": cfg.rg_pf_match_window_s},
            "RG_to_MN": {
                "pre": cfg.rg_mn_match_pre_window_s,
                "post": cfg.rg_mn_match_post_window_s,
            },
        },
        "recovery_definition": {
            "consecutive_cycles": cfg.recovery_consecutive_cycles,
            "frequency_tolerance_fraction": (
                cfg.recovery_frequency_tolerance_fraction
            ),
            "phase_tolerance_deg": cfg.phase_slip_threshold_deg,
        },
    }
    violations: list[str] = []
    for key, value in observed.items():
        if not _json_contract_equal(value, expected[key]):
            violations.append(f"runtime_primary_contract:{key}")
    if tuple(reference.PREREGISTERED_SEEDS) != EXPECTED_PRIMARY_SEEDS:
        violations.append("runtime_primary_contract:paired_seed_registry")
    return violations


def _json_contract_equal(left: object, right: object) -> bool:
    try:
        return json.dumps(
            left, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) == json.dumps(
            right, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        return False


def bursts_from_payload(record: Mapping[str, object]) -> Dict[str, np.ndarray]:
    payload = record.get("analysis_events")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint lacks an analysis_events mapping")
    bursts: Dict[str, np.ndarray] = {}
    for key, values in payload.items():
        event_key = str(key)
        if not event_key.endswith("_onset_s"):
            raise ValueError(f"invalid analysis event key: {event_key}")
        array = np.asarray(values, dtype=float)
        if (
            array.ndim != 1
            or not np.all(np.isfinite(array))
            or np.any(np.diff(array) < 0.0)
        ):
            raise ValueError(f"invalid analysis event stream: {event_key}")
        bursts[event_key.removesuffix("_onset_s")] = array
    required = {
        f"{cell_class}_{side}_{phase}"
        for cell_class in ("RG", "PF", "MN")
        for side, phase in model.SIDE_PHASES
    }
    missing = sorted(required - set(bursts))
    if missing:
        raise ValueError("missing analysis event streams: " + ",".join(missing))
    return bursts


def family6_amplitude_imbalance_preflight(
    record: Mapping[str, object], _biological_failure: int,
) -> float:
    """Independent Family-6 reconstruction from exact sufficient sums."""
    payload = record.get("analysis_observables")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint lacks analysis_observables")
    if any(key not in payload for key in EXPECTED_FAMILY6_OBSERVABLE_KEYS):
        raise ValueError("checkpoint lacks Family-6 sufficient observables")
    sample_count = payload["mn_rate_sample_count"]
    if (
        isinstance(sample_count, bool) or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise ValueError("mn_rate_sample_count must be a positive integer")
    sums = []
    for key in (
        "mn_left_rate_sum_hz_samples", "mn_right_rate_sum_hz_samples",
    ):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a JSON number")
        number = float(value)
        if not np.isfinite(number) or number < 0.0:
            raise ValueError(f"{key} must be finite and nonnegative")
        sums.append(number)
    left_mean, right_mean = (value / sample_count for value in sums)
    denominator = left_mean + right_mean
    if denominator == 0.0:
        return 1.0
    return float(abs(left_mean - right_mean) / denominator)


def _phase_relation_events_preflight(
    anchor: np.ndarray, counterpart: np.ndarray, start_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    errors: list[float] = []
    for t0, t1 in zip(anchor[:-1], anchor[1:]):
        if t0 < start_s:
            continue
        inside = counterpart[(counterpart > t0) & (counterpart < t1)]
        times.append(float(t0))
        if len(inside) == 1:
            fraction = float((inside[0] - t0) / (t1 - t0))
            errors.append((fraction - 0.5) * 360.0)
        else:
            errors.append(180.0)
    return np.asarray(times), np.asarray(errors)


def recovery_outcome_preflight(
    bursts: Mapping[str, np.ndarray],
    cfg: model.Config,
    pulse_start_s: float,
    pulse_end_s: float,
) -> Dict[str, object]:
    """Independent raw-event implementation of the frozen recovery rule."""
    empty = {
        "recovery_time_s": float("nan"),
        "recovery_event_observed": 0,
        "recovery_time_or_censor_s": float("nan"),
        "recovery_censor_time_s": float("nan"),
        "recovery_endpoint_eligible": 0,
        "recovery_ineligibility_reason": "pulse_not_delivered",
    }
    if not np.isfinite(pulse_start_s) or not np.isfinite(pulse_end_s):
        return empty
    censor_time = max(0.0, cfg.duration_s - pulse_end_s)
    if pulse_end_s >= cfg.duration_s:
        return {
            **empty,
            "recovery_censor_time_s": censor_time,
            "recovery_ineligibility_reason": (
                "pulse_ended_at_or_after_simulation_end"
            ),
        }
    populations = tuple(
        bursts[f"RG_{side}_{phase}"] for side, phase in model.SIDE_PHASES
    )
    lf, le, rf, re = populations
    relations = (
        (lf, rf), (le, re), (lf, le), (rf, re),
    )
    relation_data = [
        _phase_relation_events_preflight(anchor, counterpart, cfg.burn_in_s)
        for anchor, counterpart in relations
    ]
    pre_errors = [
        np.abs(errors[times < pulse_start_s])
        for times, errors in relation_data
    ]
    pre_events = [
        events[(events >= cfg.burn_in_s) & (events < pulse_start_s)]
        for events in populations
    ]
    if any(len(values) < 2 for values in pre_errors) or any(
        len(events) < 3 for events in pre_events
    ):
        return {
            **empty,
            "recovery_censor_time_s": censor_time,
            "recovery_ineligibility_reason": "insufficient_prepulse_cycles",
        }
    periods = [float(np.median(np.diff(events))) for events in pre_events]
    consecutive = cfg.recovery_consecutive_cycles
    for candidate in lf[lf >= pulse_end_s]:
        horizon = candidate + (consecutive + 1.5) * max(periods)
        relation_ok = True
        for times, errors in relation_data:
            selected = np.abs(
                errors[(times >= candidate) & (times <= horizon)]
            )[:consecutive]
            if len(selected) < consecutive or np.any(
                selected > cfg.phase_slip_threshold_deg
            ):
                relation_ok = False
                break
        if not relation_ok:
            continue
        period_ok = True
        for events, period in zip(populations, periods):
            selected = events[events >= candidate][:consecutive + 1]
            intervals = np.diff(selected)
            if len(intervals) < consecutive or np.any(
                np.abs(intervals - period)
                > cfg.recovery_frequency_tolerance_fraction * period
            ):
                period_ok = False
                break
        if period_ok:
            observed = float(candidate - pulse_end_s)
            return {
                "recovery_time_s": observed,
                "recovery_event_observed": 1,
                "recovery_time_or_censor_s": observed,
                "recovery_censor_time_s": censor_time,
                "recovery_endpoint_eligible": 1,
                "recovery_ineligibility_reason": "none",
            }
    return {
        "recovery_time_s": float("nan"),
        "recovery_event_observed": 0,
        "recovery_time_or_censor_s": censor_time,
        "recovery_censor_time_s": censor_time,
        "recovery_endpoint_eligible": 1,
        "recovery_ineligibility_reason": "right_censored_no_recovery",
    }


def family9_burden_and_mismatches_preflight(
    record: Mapping[str, object], task: reference.Task,
) -> tuple[float, list[str]]:
    """Rebuild the recovery composite from RG events and intervention log."""
    cfg = reference.config_for_task(task, smoke=False)
    summary = record.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("checkpoint lacks summary")
    if summary.get("technical_valid") != 1:
        raise ValueError("technical-invalid Family-9 checkpoint")
    log = record.get("intervention_log")
    if not isinstance(log, Mapping):
        raise ValueError("checkpoint lacks intervention_log")
    if any(key not in log for key in EXPECTED_FAMILY9_INTERVENTION_KEYS):
        raise ValueError("checkpoint lacks Family-9 intervention fields")
    required = log["pulse_required"]
    delivered = log["pulse_delivered"]
    if required not in (0, 1) or delivered not in (0, 1):
        raise ValueError("intervention-log flags must be binary")
    expected_required = int(task.protocol == "pulse" and task.pulse != "none")
    if required != expected_required or required != 1:
        raise ValueError("Family 9 requires an active pulse arm")

    def parse_time(value: object, label: str) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be null or numeric")
        number = float(value)
        if not np.isfinite(number) or number < 0.0:
            raise ValueError(f"{label} must be null or finite and nonnegative")
        return number

    pulse_start = parse_time(log["pulse_start_s"], "pulse_start_s")
    pulse_end = parse_time(log["pulse_end_s"], "pulse_end_s")
    reason = log["pulse_noneligibility_reason"]
    if not isinstance(reason, str):
        raise ValueError("pulse_noneligibility_reason must be a string")
    bursts = bursts_from_payload(record)
    mismatches: list[str] = []
    if delivered == 1:
        if (
            not np.isfinite(pulse_start) or not np.isfinite(pulse_end)
            or pulse_end <= pulse_start or pulse_end > cfg.duration_s
        ):
            raise ValueError("delivered pulse interval is invalid")
        baseline_cycles = min(
            max(0, int(np.sum(
                (bursts[f"RG_{side}_{phase}"] >= cfg.burn_in_s)
                & (bursts[f"RG_{side}_{phase}"] < pulse_start)
            )) - 1)
            for side, phase in model.SIDE_PHASES
        )
        expected_summary_reason = (
            "none" if baseline_cycles >= 2 else "insufficient_prepulse_cycles"
        )
        if reason != "none":
            raise ValueError("delivered pulse must have intervention reason 'none'")
        recovery = recovery_outcome_preflight(
            bursts, cfg, pulse_start, pulse_end
        )
        censor_time = float(cfg.duration_s - pulse_end)
        if censor_time < 4.0:
            raise ValueError("less than four seconds of post-pulse follow-up")
        if recovery["recovery_endpoint_eligible"] == 1:
            composite_time = float(recovery["recovery_time_or_censor_s"])
            composite_event = int(recovery["recovery_event_observed"])
        else:
            composite_time = max(0.0, cfg.duration_s - cfg.pulse_arm_after_s)
            composite_event = 0
    else:
        if np.isfinite(pulse_start) or np.isfinite(pulse_end):
            raise ValueError("undelivered pulse has finite timing")
        if reason != "biological_no_phase_eligible_cycle":
            raise ValueError("undelivered pulse lacks biological-failure reason")
        recovery = recovery_outcome_preflight(
            bursts, cfg, float("nan"), float("nan")
        )
        composite_time = max(0.0, cfg.duration_s - cfg.pulse_arm_after_s)
        composite_event = 0
        expected_summary_reason = "biological_no_phase_eligible_cycle"
    expected_summary = {
        **recovery,
        "pulse_required": 1,
        "pulse_delivered": delivered,
        "pulse_start_s": None if not np.isfinite(pulse_start) else pulse_start,
        "pulse_end_s": None if not np.isfinite(pulse_end) else pulse_end,
        "pulse_noneligibility_reason": expected_summary_reason,
        "recovery_composite_eligible": 1,
        "recovery_composite_event": composite_event,
        "recovery_composite_time_s": composite_time,
    }
    for key, expected_value in expected_summary.items():
        observed = summary.get(key)
        if isinstance(expected_value, float):
            numeric_expected = expected_value
            if not same_number(observed, numeric_expected):
                mismatches.append(key)
        elif observed != expected_value:
            mismatches.append(key)
    return min(composite_time, 4.0), mismatches


def match_output_bursts_preflight(
    rg_events: np.ndarray,
    output_events: np.ndarray,
    start_s: float,
    end_s: float,
    pre_window_s: float,
    post_window_s: float,
) -> tuple[np.ndarray, int, int]:
    """Independent one-to-one matcher with complete observation support."""
    rg = np.asarray(rg_events, dtype=float)
    output = np.asarray(output_events, dtype=float)
    if (
        rg.ndim != 1 or output.ndim != 1
        or not np.all(np.isfinite(rg)) or not np.all(np.isfinite(output))
        or np.any(np.diff(rg) < 0.0) or np.any(np.diff(output) < 0.0)
    ):
        raise ValueError("burst event arrays must be finite, one-dimensional, and sorted")
    if (
        not np.isfinite(start_s) or not np.isfinite(end_s)
        or not np.isfinite(pre_window_s) or not np.isfinite(post_window_s)
        or end_s <= start_s or pre_window_s < 0.0 or post_window_s < 0.0
    ):
        raise ValueError("invalid transfer-observation interval")
    anchors = rg[
        (rg >= start_s + pre_window_s)
        & (rg < end_s - post_window_s)
    ]
    latencies: list[float] = []
    missed = 0
    output_cursor = 0
    for onset in anchors:
        lower = onset - pre_window_s
        upper = onset + post_window_s
        while output_cursor < len(output) and output[output_cursor] < lower:
            output_cursor += 1
        if output_cursor < len(output) and output[output_cursor] <= upper:
            latencies.append(float(output[output_cursor] - onset))
            output_cursor += 1
        else:
            missed += 1
    return np.asarray(latencies), int(missed), int(len(anchors))


def recompute_event_endpoints(
    record: Mapping[str, object], task: reference.Task
) -> Sequence[str]:
    """Recompute whole-trace observable endpoints from persisted burst events."""
    cfg = reference.config_for_task(task, smoke=False)
    bursts = bursts_from_payload(record)
    summary = record["summary"]
    errors: list[str] = []
    clean = {
        name: values[values >= cfg.burn_in_s]
        for name, values in bursts.items() if name.startswith("RG_")
    }
    frequencies = [
        1.0 / np.mean(np.diff(values)) for values in clean.values()
        if len(values) >= 2 and np.mean(np.diff(values)) > 0
    ]
    expected_frequency = (
        float(np.mean(frequencies)) if frequencies else float("nan")
    )
    if not same_number(summary["frequency_hz"], expected_frequency):
        errors.append("frequency_hz")

    lf = clean["RG_L_F"]
    le = clean["RG_L_E"]
    rf = clean["RG_R_F"]
    re = clean["RG_R_E"]
    lr = model._safe_concat((
        model.cycle_phase_errors_deg(lf, rf, cfg.burn_in_s),
        model.cycle_phase_errors_deg(le, re, cfg.burn_in_s),
    ))
    fe = model._safe_concat((
        model.cycle_phase_errors_deg(lf, le, cfg.burn_in_s),
        model.cycle_phase_errors_deg(rf, re, cfg.burn_in_s),
    ))
    phase_checks = {
        "lr_phase_error_mean_abs_deg": (
            float(np.mean(np.abs(lr))) if len(lr) else float("nan")
        ),
        "fe_phase_error_mean_abs_deg": (
            float(np.mean(np.abs(fe))) if len(fe) else float("nan")
        ),
        "lr_phase_slip_count": int(np.sum(
            np.abs(lr) > cfg.phase_slip_threshold_deg
        )),
        "lr_phase_cycle_count": int(len(lr)),
        "fe_phase_slip_count": int(np.sum(
            np.abs(fe) > cfg.phase_slip_threshold_deg
        )),
        "fe_phase_cycle_count": int(len(fe)),
    }
    for key, expected in phase_checks.items():
        if isinstance(expected, int):
            if summary[key] != expected:
                errors.append(key)
        elif not same_number(summary[key], expected):
            errors.append(key)

    cvs = np.asarray([
        model.cycle_interval_cv(values, cfg.burn_in_s)
        for values in clean.values()
    ])
    cvs = cvs[np.isfinite(cvs)]
    expected_cv = float(np.mean(cvs)) if len(cvs) else float("nan")
    if not same_number(summary["rg_cycle_interval_cv_mean"], expected_cv):
        errors.append("rg_cycle_interval_cv_mean")

    for cell_class, prefix, pre_window, post_window in (
        ("PF", "pf", 0.0, cfg.rg_pf_match_window_s),
        (
            "MN", "mn", cfg.rg_mn_match_pre_window_s,
            cfg.rg_mn_match_post_window_s,
        ),
    ):
        total = missed = 0
        for side, phase in model.SIDE_PHASES:
            _, local_missed, local_total = match_output_bursts_preflight(
                bursts[f"RG_{side}_{phase}"],
                bursts[f"{cell_class}_{side}_{phase}"],
                cfg.burn_in_s, cfg.duration_s, pre_window, post_window,
            )
            total += local_total
            missed += local_missed
        expected_counts = {
            f"{prefix}_transfer_anchor_count": total,
            f"{prefix}_transfer_missed_count": missed,
            f"{prefix}_transfer_matched_count": total - missed,
        }
        for key, expected in expected_counts.items():
            if summary[key] != expected:
                errors.append(key)
        reliability = 1.0 - missed / total if total else float("nan")
        if not same_number(summary[f"{prefix}_transfer_reliability"], reliability):
            errors.append(f"{prefix}_transfer_reliability")

    pulse_start = summary["pulse_start_s"]
    pulse_end = summary["pulse_end_s"]
    recovery = recovery_outcome_preflight(
        bursts,
        cfg,
        float("nan") if pulse_start is None else float(pulse_start),
        float("nan") if pulse_end is None else float(pulse_end),
    )
    for key in (
        "recovery_time_s", "recovery_event_observed",
        "recovery_time_or_censor_s", "recovery_censor_time_s",
        "recovery_endpoint_eligible", "recovery_ineligibility_reason",
    ):
        expected = recovery[key]
        observed = summary[key]
        if isinstance(expected, float):
            if not same_number(observed, expected):
                errors.append(key)
        elif observed != expected:
            errors.append(key)

    expected_rhythmic_failure = int(
        min(len(values) for values in clean.values()) < 2
    )
    if summary.get("rhythmic_failure") != expected_rhythmic_failure:
        errors.append("rhythmic_failure")
    expected_imbalance = family6_amplitude_imbalance_preflight(
        record, expected_rhythmic_failure
    )
    observable_payload = record["analysis_observables"]
    if (
        float(observable_payload["mn_left_rate_sum_hz_samples"])
        + float(observable_payload["mn_right_rate_sum_hz_samples"])
        == 0.0
    ):
        # The descriptive ratio is undefined under complete bilateral silence;
        # the primary burden is nevertheless preregistered as the worst value.
        expected_summary_imbalance = float("nan")
    else:
        expected_summary_imbalance = expected_imbalance
    if not same_number(
        summary.get("bilateral_amplitude_imbalance"),
        expected_summary_imbalance,
    ):
        errors.append("bilateral_amplitude_imbalance")
    return errors


def epoch_transfer_observation(
    bursts: Mapping[str, np.ndarray],
    cfg: model.Config,
    epoch: int,
    cell_class: str,
) -> Dict[str, object]:
    """Reconstruct one epoch's RG-to-PF/MN transfer from raw event times."""
    if cell_class not in {"PF", "MN"}:
        raise ValueError("epoch transfer is defined only for PF or MN")
    start = (epoch - 1) * cfg.long_epoch_duration_s
    end = epoch * cfg.long_epoch_duration_s
    if cell_class == "PF":
        pre_window, post_window = 0.0, cfg.rg_pf_match_window_s
    else:
        pre_window = cfg.rg_mn_match_pre_window_s
        post_window = cfg.rg_mn_match_post_window_s
    total = missed = 0
    rg_counts = []
    for side, phase in model.SIDE_PHASES:
        rg = bursts[f"RG_{side}_{phase}"]
        output = bursts[f"{cell_class}_{side}_{phase}"]
        _, local_missed, local_total = match_output_bursts_preflight(
            rg, output, start, end, pre_window, post_window,
        )
        total += local_total
        missed += local_missed
        rg_counts.append(int(np.sum((rg >= start) & (rg < end))))
    return {
        "anchor_count": int(total),
        "missed_count": int(missed),
        "matched_count": int(total - missed),
        "reliability": 1.0 - missed / total if total else float("nan"),
        "rhythmic_failure": int(min(rg_counts) < 2),
    }


def recompute_long_epoch_transfer_endpoints(
    record: Mapping[str, object], task: reference.Task
) -> Sequence[str]:
    """Verify every persisted long-epoch transfer field from raw events."""
    if task.protocol != "long":
        return ()
    cfg = reference.config_for_task(task, smoke=False)
    bursts = bursts_from_payload(record)
    epochs = record.get("epochs")
    if not isinstance(epochs, Sequence) or isinstance(epochs, (str, bytes)):
        return ("epochs_missing",)
    errors: list[str] = []
    if len(epochs) != cfg.long_n_epochs:
        errors.append("epoch_count")
        return errors
    for expected_epoch, row in enumerate(epochs, start=1):
        if not isinstance(row, Mapping) or row.get("epoch") != expected_epoch:
            errors.append(f"epoch_{expected_epoch}.identity")
            continue
        for cell_class, prefix in (("PF", "pf"), ("MN", "mn")):
            expected = epoch_transfer_observation(
                bursts, cfg, expected_epoch, cell_class
            )
            field_values = {
                f"{prefix}_transfer_anchor_count": expected["anchor_count"],
                f"{prefix}_transfer_missed_count": expected["missed_count"],
                f"{prefix}_transfer_matched_count": expected["matched_count"],
            }
            for key, value in field_values.items():
                if row.get(key) != value:
                    errors.append(f"epoch_{expected_epoch}.{key}")
            reliability_key = f"{prefix}_transfer_reliability"
            if not same_number(row.get(reliability_key), expected["reliability"]):
                errors.append(f"epoch_{expected_epoch}.{reliability_key}")
        # Family 10's zero-denominator handling depends on this independently
        # reproducible biological-failure flag.
        if row.get("rhythmic_failure") != epoch_transfer_observation(
            bursts, cfg, expected_epoch, "MN"
        )["rhythmic_failure"]:
            errors.append(f"epoch_{expected_epoch}.rhythmic_failure")
    return errors


def raw_union_mn_observation(
    record: Mapping[str, object],
    task: reference.Task,
    start_epoch: int,
    end_epoch: int,
) -> Dict[str, object]:
    """Match one contiguous epoch union with one used-output set per stream."""
    cfg = reference.config_for_task(task, smoke=False)
    if task.protocol != "long":
        raise ValueError("Family 10 requires a long-protocol task")
    if not 1 <= start_epoch <= end_epoch <= cfg.long_n_epochs:
        raise ValueError("invalid long-protocol epoch union")
    bursts = bursts_from_payload(record)
    start_s = (start_epoch - 1) * cfg.long_epoch_duration_s
    end_s = end_epoch * cfg.long_epoch_duration_s
    anchors = missed = 0
    rg_counts: list[int] = []
    for side, phase in model.SIDE_PHASES:
        rg = bursts[f"RG_{side}_{phase}"]
        output = bursts[f"MN_{side}_{phase}"]
        _, local_missed, local_anchors = match_output_bursts_preflight(
            rg,
            output,
            start_s,
            end_s,
            cfg.rg_mn_match_pre_window_s,
            cfg.rg_mn_match_post_window_s,
        )
        anchors += local_anchors
        missed += local_missed
        rg_counts.append(int(np.sum((rg >= start_s) & (rg < end_s))))
    return {
        "anchor_count": int(anchors),
        "missed_count": int(missed),
        "matched_count": int(anchors - missed),
        "rhythmic_failure": int(min(rg_counts) < 2),
    }


def raw_union_mn_burden(
    record: Mapping[str, object],
    task: reference.Task,
    start_epoch: int,
    end_epoch: int,
) -> float:
    """Compute one Family-10 burden from a single raw-event union match."""
    observation = raw_union_mn_observation(
        record, task, start_epoch, end_epoch
    )
    return probability_burden(
        int(observation["missed_count"]),
        int(observation["anchor_count"]),
        int(observation["rhythmic_failure"]),
    )


def raw_postchallenge_mn_deficit(
    record: Mapping[str, object], task: reference.Task
) -> float:
    """Observable recovery-minus-baseline RG-to-MN transfer burden."""
    baseline = raw_union_mn_burden(record, task, 2, 6)
    recovery = raw_union_mn_burden(record, task, 20, 24)
    return recovery - baseline


def expected_task_identity_sha(tasks: Sequence[reference.Task]) -> str:
    digest = hashlib.sha256()
    for task in tasks:
        row = {"task_id": task.task_id, **task.identity()}
        digest.update(json.dumps(
            row, sort_keys=True, separators=(",", ":")
        ).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def task_matrix_violations(tasks: Sequence[reference.Task]) -> list[str]:
    """Check the fixed preregistered matrix and two causal primary designs."""
    violations: list[str] = []
    if tuple(reference.STAGES) != tuple("ABCDEFGH"):
        violations.append("stage_order")
    if tuple(reference.VALIDATION_SEEDS) != EXPECTED_VALIDATION_SEEDS:
        violations.append("construct_validation_seed_set")
    if tuple(reference.PREREGISTERED_SEEDS) != EXPECTED_PRIMARY_SEEDS:
        violations.append("primary_seed_set")
    observed_stage_seeds = {
        stage: tuple(reference.PRODUCTION_STAGE_SEEDS.get(stage, ()))
        for stage in reference.STAGES
    }
    if observed_stage_seeds != EXPECTED_PRODUCTION_SEEDS_BY_STAGE:
        violations.append("production_stage_seed_partition")
    if tuple(model.CLASSES) != EXPECTED_CLASSES:
        violations.append("population_class_registry")
    if tuple(model.MT_ROUTES) != EXPECTED_MT_ROUTES:
        violations.append("presynaptic_mt_route_registry")
    if len(tasks) != EXPECTED_ANALYSIS_TASKS:
        violations.append("total_task_count")
    if len({task.task_id for task in tasks}) != len(tasks):
        violations.append("duplicate_task_id")
    if expected_task_identity_sha(tasks) != EXPECTED_TASK_IDENTITY_SHA256:
        violations.append("frozen_task_identity_sha256")

    observed_stage_totals = {
        stage: sum(task.stage == stage for task in tasks)
        for stage in reference.STAGES
    }
    if observed_stage_totals != EXPECTED_STAGE_TASK_TOTALS:
        violations.append("stage_task_totals")

    by_seed_stage: Dict[tuple[int, str], list[reference.Task]] = {}
    for task in tasks:
        by_seed_stage.setdefault((task.seed, task.stage), []).append(task)

    expected_seed_stage_cells = {
        (seed, stage)
        for stage, seeds in EXPECTED_PRODUCTION_SEEDS_BY_STAGE.items()
        for seed in seeds
    }
    if set(by_seed_stage) != expected_seed_stage_cells:
        violations.append("seed_stage_cell_partition")
    for stage, seeds in EXPECTED_PRODUCTION_SEEDS_BY_STAGE.items():
        for seed in seeds:
            expected = EXPECTED_STAGE_TASKS_PER_SEED[stage]
            if len(by_seed_stage.get((seed, stage), ())) != expected:
                violations.append(f"seed_{seed}.stage_{stage}_count")

    # Stage F is not a route proxy or a reduced subset: it is the complete
    # 10-population x 10-presynaptic-route x ablation(0/1) x impairment(0/1)
    # factorial in every speed/load/pulse context.
    expected_f_signatures = {
        (
            speed,
            load,
            pulse,
            (population,) if ablation_on else (),
            (route,) if impairment_on else (),
            (
                f"factorial_{population}_M_{route}_"
                f"A{int(ablation_on)}M{int(impairment_on)}"
            ),
        )
        for population in EXPECTED_CLASSES
        for route in EXPECTED_MT_ROUTES
        for ablation_on in (False, True)
        for impairment_on in (False, True)
        for speed in ("low", "medium", "high")
        for load in ("normal", "unilateral", "bilateral_high")
        for pulse in ("none", "excitatory", "inhibitory")
    }
    for seed in EXPECTED_VALIDATION_SEEDS:
        f_tasks = by_seed_stage.get((seed, "F"), ())
        observed_f_signatures = {
            (
                task.speed,
                task.load,
                task.pulse,
                task.ablations,
                task.impaired_mt_routes,
                task.label,
            )
            for task in f_tasks
        }
        fixed_fields_valid = all(
            task.protocol == "pulse"
            and task.load_side == "L"
            and task.mt_mode == "dynamic"
            and task.fast_mode == "dynamic"
            and tuple(task.challenged_routes) == EXPECTED_MT_ROUTES
            for task in f_tasks
        )
        if (
            observed_f_signatures != expected_f_signatures
            or not fixed_fields_valid
        ):
            violations.append(f"seed_{seed}.stage_F_full_10x10_2x2")

    for seed in EXPECTED_PRIMARY_SEEDS:
        # Family 10 must have the complete challenge x route-impairment 2x2.
        g_tasks = by_seed_stage.get((seed, "G"), ())
        by_label = {task.label: task for task in g_tasks}
        expected_g_labels = {"long_no_challenge"}
        for route in model.MT_ROUTES:
            expected_g_labels.update({
                f"long_no_challenge_impaired_{route}",
                f"long_challenge_{route}",
                f"long_challenge_impaired_{route}",
            })
        if set(by_label) != expected_g_labels:
            violations.append(f"seed_{seed}.family10_label_set")
        else:
            base = by_label["long_no_challenge"]
            if not (
                base.protocol == "long" and base.pulse == "none"
                and not base.challenged_routes
                and not base.impaired_mt_routes
            ):
                violations.append(f"seed_{seed}.family10_base_cell")
            for route in model.MT_ROUTES:
                cells = {
                    "no_challenge_impaired": by_label[
                        f"long_no_challenge_impaired_{route}"
                    ],
                    "challenge": by_label[f"long_challenge_{route}"],
                    "challenge_impaired": by_label[
                        f"long_challenge_impaired_{route}"
                    ],
                }
                valid = (
                    cells["no_challenge_impaired"].protocol == "long"
                    and cells["no_challenge_impaired"].pulse == "none"
                    and not cells["no_challenge_impaired"].challenged_routes
                    and cells["no_challenge_impaired"].impaired_mt_routes
                        == (route,)
                    and cells["challenge"].challenged_routes == (route,)
                    and not cells["challenge"].impaired_mt_routes
                    and cells["challenge_impaired"].challenged_routes
                        == (route,)
                    and cells["challenge_impaired"].impaired_mt_routes
                        == (route,)
                )
                if not valid:
                    violations.append(
                        f"seed_{seed}.family10_2x2_{route}"
                    )

        # Families 1--9 require both dynamic and only the frozen matched primary
        # comparator in each pulse context at dynamic KCa.
        h_tasks = by_seed_stage.get((seed, "H"), ())
        observed_h = {
            (task.fast_mode, task.mt_mode, task.pulse) for task in h_tasks
        }
        expected_h = {
            (fast, mt, pulse)
            for fast in EXPECTED_H_FAST_MODES
            for mt in EXPECTED_H_MT_MODES
            for pulse in EXPECTED_PULSES
        }
        if observed_h != expected_h:
            violations.append(f"seed_{seed}.stage_H_factorial")
        for pulse in EXPECTED_PULSES:
            if not {
                ("dynamic", "dynamic", pulse),
                ("dynamic", "static_matched", pulse),
            } <= observed_h:
                violations.append(
                    f"seed_{seed}.families1_9_primary_pair_{pulse}"
                )
    return violations


def _literal_assignments(tree: ast.Module) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                result[target.id] = literal
    return result


def _function_source(source: str, tree: ast.Module, name: str) -> str:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(source, node)
            return "" if segment is None else segment
    return ""


def _literal_call_windows(
    tree: ast.Module,
    function_name: str,
    called_name: str,
    first_window_arg: int,
) -> set[tuple[int, int]]:
    """Extract two integer window arguments from calls inside one function."""
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function_name:
            continue
        windows: set[tuple[int, int]] = set()
        for child in ast.walk(node):
            if not (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == called_name
                and len(child.args) >= first_window_arg + 2
            ):
                continue
            start, end = child.args[
                first_window_arg:first_window_arg + 2
            ]
            if (
                isinstance(start, ast.Constant)
                and isinstance(start.value, int)
                and isinstance(end, ast.Constant)
                and isinstance(end.value, int)
            ):
                windows.add((start.value, end.value))
        return windows
    return set()


def _literal_data_keys(tree: ast.Module) -> set[str]:
    """Find literal mapping keys, including ``get`` and simple key aliases."""
    assigned_strings: Dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                assigned_strings.setdefault(target.id, set()).add(value.value)

    unambiguous_aliases = {
        name: next(iter(values))
        for name, values in assigned_strings.items()
        if len(values) == 1
    }

    def resolve(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return unambiguous_aliases.get(node.id)
        return None

    keys: set[str] = set()
    for node in ast.walk(tree):
        key: str | None = None
        if isinstance(node, ast.Subscript):
            root = node.value
            while isinstance(root, ast.Subscript):
                root = root.value
            if isinstance(root, ast.Name) and root.id == "PRIMARY_ANALYSIS_CONTRACT":
                continue
            key = resolve(node.slice)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get", "pop", "setdefault", "__getitem__"}
            and node.args
        ):
            key = resolve(node.args[0])
        if key is not None:
            keys.add(key)
    return keys


def _is_forbidden_primary_endpoint_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in ALLOWED_INTERVENTION_KEYS:
        return False
    if any(term in lowered for term in FORBIDDEN_PRIMARY_ENDPOINT_TERMS):
        return True
    # Match MT as a complete snake/kebab/space-delimited token so ordinary
    # words containing the letters "mt" are not rejected.
    return "mt" in {
        token for token in re.split(r"[^a-z0-9]+", lowered) if token
    }


def primary_analysis_contract(source: str) -> Dict[str, object]:
    """Audit primary code without importing it or evaluating any result."""
    tree = ast.parse(source)
    assignments = _literal_assignments(tree)
    violations: list[str] = []
    if "preflight_results_v2_6_1" not in source:
        violations.append("primary_analysis_not_guarded_by_v2_6_1_preflight")
    declared_contract = assignments.get("PRIMARY_ANALYSIS_CONTRACT")
    if not _json_contract_equal(
        declared_contract, EXPECTED_PRIMARY_ANALYSIS_CONTRACT
    ):
        violations.append("primary_analysis_declarative_contract")
        declared_contract = {}
    comparator = (
        declared_contract.get("families_1_to_9_comparator")
        if isinstance(declared_contract, dict) else None
    )
    if comparator != "static_matched":
        violations.append("families_1_9_comparator_not_static_matched")
    if "MT_CONTROLS" in assignments:
        violations.append("legacy_multi_control_primary_present")

    families = assignments.get("FAMILIES")
    if not _json_contract_equal(families, EXPECTED_PRIMARY_FAMILY_REGISTRY):
        violations.append("primary_family_registry")
        families = {}
    family_10 = str(families.get(10, ""))
    if not (
        "rg_to_mn" in family_10
        and "difference_in_differences" in family_10
    ):
        violations.append("family10_not_observable_transfer_did")

    literal_data_keys = _literal_data_keys(tree)
    forbidden_keys = sorted(
        key for key in literal_data_keys
        if _is_forbidden_primary_endpoint_key(key)
    )
    if forbidden_keys:
        violations.append("hidden_state_primary_keys:" + "+".join(forbidden_keys))
    unregistered_keys = sorted(
        literal_data_keys
        - ALLOWED_PRIMARY_DATA_KEYS
        - ALLOWED_INTERVENTION_KEYS
    )
    if unregistered_keys:
        violations.append(
            "unregistered_primary_data_keys:" + "+".join(unregistered_keys)
        )
    forbidden_family_labels = sorted(
        str(value) for value in families.values()
        if _is_forbidden_primary_endpoint_key(str(value))
    )
    if forbidden_family_labels:
        violations.append(
            "hidden_state_primary_family_labels:" + "+".join(
                forbidden_family_labels
            )
        )

    h_source = _function_source(source, tree, "h_family_contrast")
    if not h_source:
        violations.append("h_family_contrast_missing")
    else:
        if 'h_task(seed, "dynamic", pulse)' not in h_source:
            violations.append("h_dynamic_arm_missing")
        if "PRIMARY_MT_COMPARATOR" not in h_source:
            violations.append("h_static_matched_arm_missing")
        if "MT_CONTROLS" in h_source or "controls =" in h_source:
            violations.append("h_legacy_averaged_controls")
        if "pulse_differences.append(dynamic - comparator)" not in h_source:
            violations.append("h_primary_contrast_sign_or_formula")
        if "return float(np.mean(pulse_differences))" not in h_source:
            violations.append("h_pulse_context_reduction")

    exact_source_requirements = {
        "alpha_bound_to_declarative_contract": (
            'PRIMARY_ANALYSIS_CONTRACT["alpha_per_family_two_sided"]'
        ),
        "paired_seed_count_bound_to_declarative_contract": (
            'PRIMARY_ANALYSIS_CONTRACT["paired_seed_count"]'
        ),
        "recovery_horizon_bound_to_declarative_contract": (
            'PRIMARY_ANALYSIS_CONTRACT["family_9_recovery_horizon_s"]'
        ),
        "pulse_contexts_bound_to_declarative_contract": (
            'PRIMARY_ANALYSIS_CONTRACT["families_1_to_8_pulse_contexts"]'
        ),
        "probability_burden_transform": (
            "math.asin(math.sqrt((events + 0.5) / (trials + 1.0)))"
        ),
        "probability_burden_worst_policy": "return math.pi / 2.0",
        "paired_seed_count_guard": (
            "len(array) != EXPECTED_PAIRED_SEED_COUNT"
        ),
        "two_sided_alpha_decision": "p_value < ALPHA_PRIMARY",
        "favorable_negative_direction": 'favorable_direction = "negative"',
        "favorable_negative_decision": "favorable = mean < 0.0",
        "frozen_pulse_context_selection": (
            "pulses = PULSES_9 if family == 9 else PULSES_1_TO_8"
        ),
    }
    for label, fragment in exact_source_requirements.items():
        if fragment not in source:
            violations.append(label)

    endpoint_source = _function_source(source, tree, "endpoint_burden")
    if (
        'summary["bilateral_amplitude_imbalance"]' in endpoint_source
        or 'summary["recovery_composite_time_s"]' in endpoint_source
        or 'summary["recovery_composite_eligible"]' in endpoint_source
    ):
        violations.append("family6_or_9_uses_opaque_summary_outcome")
    family6_source = _function_source(
        source, tree, "bilateral_mn_amplitude_imbalance_burden"
    )
    for required in (
        "analysis_observables", "mn_left_rate_sum_hz_samples",
        "mn_right_rate_sum_hz_samples", "mn_rate_sample_count",
        "abs(left_mean - right_mean) / denominator",
    ):
        if required not in family6_source:
            violations.append(f"family6_contract_missing:{required}")
    family9_source = _function_source(source, tree, "family9_recovery_burden")
    for required in (
        "analysis_events", "intervention_log", "FAMILY9_RG_EVENT_KEYS",
        "pulse_required", "pulse_delivered", "pulse_start_s", "pulse_end_s",
        "pulse_noneligibility_reason", "RECOVERY_HORIZON_S",
        "model.recovery_outcome_from_phase_and_period",
    ):
        if required not in family9_source and required not in _function_source(
            source, tree, "_event_arrays"
        ):
            violations.append(f"family9_contract_missing:{required}")
    h_task_source = _function_source(source, tree, "h_task")
    for required in (
        'speed="medium"', 'load="normal"', 'fast_mode="dynamic"',
    ):
        if required not in h_task_source:
            violations.append(f"primary_h_context_missing:{required}")

    family10_sources = "\n".join(
        _function_source(source, tree, name)
        for name in (
            "_event_arrays",
            "family10_bursts_from_record",
            "match_rg_mn_bursts_primary",
            "raw_window_mn_transfer_burden",
            "postchallenge_mn_transfer_deficit",
            "g_family_contrast",
        )
    )
    for required in (
        "analysis_events", "FAMILY10_EVENT_KEYS",
        "match_rg_mn_bursts_primary", "raw_window_mn_transfer_burden",
        "long_no_challenge", "long_no_challenge_impaired_",
        "long_challenge_", "long_challenge_impaired_",
    ):
        if required not in family10_sources:
            violations.append(f"family10_contract_missing:{required}")
    if any(
        forbidden in family10_sources
        for forbidden in (
            'record["epochs"]', "record['epochs']",
            "mn_transfer_missed_count", "mn_transfer_anchor_count",
            "model.match_output_bursts",
        )
    ):
        violations.append("family10_not_raw_union_event_matching")
    expected_family10_event_keys = (
        "RG_L_F_onset_s", "RG_L_E_onset_s",
        "RG_R_F_onset_s", "RG_R_E_onset_s",
        "MN_L_F_onset_s", "MN_L_E_onset_s",
        "MN_R_F_onset_s", "MN_R_E_onset_s",
    )
    if assignments.get("FAMILY10_EVENT_KEYS") != expected_family10_event_keys:
        violations.append("family10_raw_event_key_registry")
    if (
        assignments.get("FAMILY9_INTERVENTION_KEYS")
        != EXPECTED_FAMILY9_INTERVENTION_KEYS
    ):
        violations.append("family9_intervention_log_key_registry")
    if not (
        "normal_challenge_effect" in family10_sources
        and "impaired_challenge_effect" in family10_sources
        and "normal_challenge_effect - impaired_challenge_effect"
            in family10_sources
    ):
        violations.append("family10_difference_in_differences_formula")
    family10_epoch_windows = _literal_call_windows(
        tree,
        "postchallenge_mn_transfer_deficit",
        "raw_window_mn_transfer_burden",
        2,
    )
    if family10_epoch_windows != {(2, 6), (20, 24)}:
        violations.append("family10_frozen_baseline_recovery_epochs")

    return {
        "declarative_contract": declared_contract,
        "comparator": comparator,
        "family_10": family_10,
        "literal_primary_data_keys": sorted(literal_data_keys),
        "forbidden_primary_data_keys": forbidden_keys,
        "unregistered_primary_data_keys": unregistered_keys,
        "family10_epoch_windows": sorted(family10_epoch_windows),
        "runtime_contract_violations": primary_runtime_contract_violations(),
        "violations": violations,
        "passes": not violations and not primary_runtime_contract_violations(),
    }


def locate_primary_analysis() -> Path:
    root = Path(__file__).resolve().parent
    path = root / "analyze_primary_v2_6_1.py"
    if not path.is_file():
        raise FileNotFoundError("frozen v2.6.1 primary analysis module is missing")
    return path


def raw_family10_difference_in_differences(
    seed: int,
    task_to_sim: Mapping[str, str],
    records: Mapping[str, Mapping[str, object]],
) -> float:
    """Rebuild the true 2x2 Family-10 contrast solely from RG/MN events."""
    by_label = {
        task.label: task for task in reference.tasks_for_seed("G", seed)
    }
    base_task = by_label["long_no_challenge"]
    base_record = records[task_to_sim[base_task.task_id]]
    no_challenge_intact = raw_postchallenge_mn_deficit(
        base_record, base_task
    )
    route_differences = []
    for route in model.MT_ROUTES:
        no_challenge_impaired_task = by_label[
            f"long_no_challenge_impaired_{route}"
        ]
        challenge_intact_task = by_label[f"long_challenge_{route}"]
        challenge_impaired_task = by_label[
            f"long_challenge_impaired_{route}"
        ]
        no_challenge_impaired = raw_postchallenge_mn_deficit(
            records[task_to_sim[no_challenge_impaired_task.task_id]],
            no_challenge_impaired_task,
        )
        challenge_intact = raw_postchallenge_mn_deficit(
            records[task_to_sim[challenge_intact_task.task_id]],
            challenge_intact_task,
        )
        challenge_impaired = raw_postchallenge_mn_deficit(
            records[task_to_sim[challenge_impaired_task.task_id]],
            challenge_impaired_task,
        )
        intact_challenge_effect = challenge_intact - no_challenge_intact
        impaired_challenge_effect = (
            challenge_impaired - no_challenge_impaired
        )
        route_differences.append(
            intact_challenge_effect - impaired_challenge_effect
        )
    return float(np.mean(route_differences))


def family10_cells_are_event_computable(
    seed: int,
    task_to_sim: Mapping[str, str],
    records: Mapping[str, Mapping[str, object]],
) -> bool:
    """Check all four cells per route without retaining or revealing results."""
    try:
        value = raw_family10_difference_in_differences(
            seed, task_to_sim, records
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    return bool(np.isfinite(value))


def _primary_summary_burden_preflight(
    record: Mapping[str, object], family: int,
) -> float:
    summary = record.get("summary")
    if not isinstance(summary, Mapping) or summary.get("technical_valid") != 1:
        raise ValueError("technical-invalid checkpoint")
    failure = int(summary["rhythmic_failure"])
    if family == 1:
        value = summary["lr_phase_error_mean_abs_deg"]
        return 180.0 if value is None and failure else float(value)
    if family == 2:
        value = summary["fe_phase_error_mean_abs_deg"]
        return 180.0 if value is None and failure else float(value)
    if family == 3:
        return probability_burden(
            int(summary["lr_phase_slip_count"]),
            int(summary["lr_phase_cycle_count"]), failure,
        )
    if family == 4:
        return probability_burden(
            int(summary["fe_phase_slip_count"]),
            int(summary["fe_phase_cycle_count"]), failure,
        )
    if family == 5:
        value = summary["rg_cycle_interval_cv_mean"]
        if value is None:
            if failure:
                return 1.0
            raise ValueError("missing CV without rhythmic failure")
        return min(float(value), 1.0)
    if family == 6:
        return family6_amplitude_imbalance_preflight(record, failure)
    if family == 7:
        return probability_burden(
            int(summary["pf_transfer_missed_count"]),
            int(summary["pf_transfer_anchor_count"]), failure,
        )
    if family == 8:
        return probability_burden(
            int(summary["mn_transfer_missed_count"]),
            int(summary["mn_transfer_anchor_count"]), failure,
        )
    raise ValueError(f"unsupported preflight family: {family}")


def _primary_h_task(seed: int, mt_mode: str, pulse: str) -> reference.Task:
    return reference.Task(
        "H", seed, speed="medium", load="normal", pulse=pulse,
        fast_mode="dynamic", mt_mode=mt_mode,
        label=f"control_KCa_dynamic_MT_{mt_mode}",
    )


def primary_family_computability(
    task_to_sim: Mapping[str, str],
    records: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    """Check 132 finite contrasts per family without retaining their values."""
    counts = {family: 0 for family in EXPECTED_PRIMARY_FAMILIES}
    failures: list[Dict[str, object]] = []
    for seed in EXPECTED_PRIMARY_SEEDS:
        for family in range(1, 10):
            try:
                pulses = (
                    ("excitatory", "inhibitory")
                    if family == 9 else
                    ("none", "excitatory", "inhibitory")
                )
                differences: list[float] = []
                for pulse in pulses:
                    task_dynamic = _primary_h_task(seed, "dynamic", pulse)
                    task_static = _primary_h_task(
                        seed, "static_matched", pulse
                    )
                    dynamic = records[task_to_sim[task_dynamic.task_id]]
                    static = records[task_to_sim[task_static.task_id]]
                    if family == 9:
                        dynamic_value, dynamic_mismatches = (
                            family9_burden_and_mismatches_preflight(
                                dynamic, task_dynamic
                            )
                        )
                        static_value, static_mismatches = (
                            family9_burden_and_mismatches_preflight(
                                static, task_static
                            )
                        )
                        if dynamic_mismatches or static_mismatches:
                            raise ValueError(
                                "Family-9 summary/raw mismatch: "
                                + ",".join(sorted(set(
                                    dynamic_mismatches + static_mismatches
                                )))
                            )
                    else:
                        dynamic_value = _primary_summary_burden_preflight(
                            dynamic, family
                        )
                        static_value = _primary_summary_burden_preflight(
                            static, family
                        )
                    differences.append(dynamic_value - static_value)
                contrast = float(np.mean(differences))
                if not np.isfinite(contrast):
                    raise ValueError("non-finite H contrast")
                counts[family] += 1
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                failures.append({
                    "seed": seed,
                    "family": family,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        try:
            value = raw_family10_difference_in_differences(
                seed, task_to_sim, records
            )
            if not np.isfinite(value):
                raise ValueError("non-finite Family-10 contrast")
            counts[10] += 1
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            failures.append({
                "seed": seed,
                "family": 10,
                "error": f"{type(exc).__name__}: {exc}",
            })
    expected_count = len(EXPECTED_PRIMARY_SEEDS)
    return {
        "finite_seed_contrast_count_by_family": counts,
        "expected_per_family": expected_count,
        "failures": failures,
        "passes": not failures and all(
            value == expected_count for value in counts.values()
        ),
    }


def run_preflight(
    output_dir: Path, report_path: Path | None = None
) -> Dict[str, object]:
    output_dir = output_dir.resolve()
    checks: Dict[str, bool] = {}
    accelerated.assert_frozen_model_identity()
    checks["v2_6_1_model_version"] = model.MODEL_VERSION.startswith(
        EXPECTED_MODEL_VERSION_PREFIX
    )

    tasks = reference.build_production_tasks(reference.STAGES)
    task_violations = task_matrix_violations(tasks)
    checks["frozen_v2_6_1_task_matrix"] = not task_violations
    representatives, task_to_sim, multiplicity = accelerated.unique_simulations(
        tasks, smoke=False
    )
    expected_unique_simulations = len(representatives)
    checks["frozen_unique_simulation_count"] = (
        expected_unique_simulations == EXPECTED_UNIQUE_SIMULATIONS
    )

    primary_path = locate_primary_analysis()
    primary_contract = primary_analysis_contract(
        primary_path.read_text(encoding="utf-8")
    )
    checks["primary_analysis_causal_contract"] = bool(
        primary_contract["passes"]
    )
    runtime_contract_violations = primary_runtime_contract_violations()
    checks["primary_runtime_threshold_contract"] = not runtime_contract_violations
    checks["primary_endpoints_observable_only"] = not bool(
        primary_contract["forbidden_primary_data_keys"]
        or primary_contract["unregistered_primary_data_keys"]
    )

    plan_path = output_dir / "experiment_plan.json"
    manifest_path = output_dir / "execution_manifest.json"
    completion_path = output_dir / "completion.json"
    index_path = output_dir / "analysis_task_index.csv"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    execution_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    checks["experiment_plan"] = bool(
        plan.get("scientific_valid") is True
        and plan.get("seed_profile") == "production_split"
        and plan.get("seeds") == list(EXPECTED_PLAN_SEEDS)
        and plan.get("seeds_by_stage") == {
            stage: list(seeds)
            for stage, seeds in EXPECTED_PRODUCTION_SEEDS_BY_STAGE.items()
        }
        and list(plan.get("stages", {})) == list(reference.STAGES)
        and plan.get("task_count_by_stage") == EXPECTED_STAGE_TASK_TOTALS
        and plan.get("total_task_count") == EXPECTED_ANALYSIS_TASKS
        and plan.get("task_identity_sha256") == EXPECTED_TASK_IDENTITY_SHA256
        and plan.get("accelerated_execution", {}).get("seed_profile")
            == "production_split"
        and plan.get("accelerated_execution", {}).get("analysis_task_count")
            == EXPECTED_ANALYSIS_TASKS
        and plan.get("accelerated_execution", {}).get("unique_simulation_count")
            == expected_unique_simulations
    )
    checks["execution_manifest"] = bool(
        execution_manifest.get("seed_profile") == "production_split"
        and execution_manifest.get("model_sha256")
            == accelerated.EXPECTED_MODEL_SHA256
        and execution_manifest.get("execution_engine_sha256")
            == accelerated.EXPECTED_EXECUTION_ENGINE_SHA256
        and execution_manifest.get("runner_version")
            == accelerated.RUNNER_VERSION
    )
    checks["completion_record"] = bool(
        completion.get("scientific_valid") is True
        and completion.get("analysis_tasks") == EXPECTED_ANALYSIS_TASKS
        and completion.get("unique_simulations")
            == expected_unique_simulations
        and completion.get("completed_checkpoint_count")
            == expected_unique_simulations
    )

    index_ok = True
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        for task, row in zip_longest(tasks, rows):
            if task is None or row is None:
                index_ok = False
                break
            sid = task_to_sim[task.task_id]
            if not (
                row.get("task_id") == task.task_id
                and row.get("simulation_id") == sid
                and row.get("reuse_count") == str(multiplicity[sid])
            ):
                index_ok = False
                break
    checks["task_index_exact"] = index_ok

    simulation_dir = output_dir / "simulations"
    actual_paths = {
        path.stem: path for path in simulation_dir.glob("sim-*.json")
    }
    expected_ids = set(representatives)
    checks["checkpoint_name_set_exact"] = set(actual_paths) == expected_ids
    technical_invalid: list[str] = []
    structural_invalid: list[Dict[str, str]] = []
    event_mismatches: list[Dict[str, object]] = []
    long_epoch_mismatches: list[Dict[str, object]] = []
    validated = 0
    primary_records: Dict[str, Mapping[str, object]] = {}
    if checks["checkpoint_name_set_exact"]:
        primary_simulation_ids = {
            task_to_sim[task.task_id]
            for task in tasks
            if (
                task.stage == "G"
                or (
                    task.stage == "H"
                    and task.fast_mode == "dynamic"
                    and task.mt_mode in {"dynamic", "static_matched"}
                )
            )
        }
        for sid, task in representatives.items():
            path = actual_paths[sid]
            try:
                accelerated.validate_checkpoint(path, sid)
                record = json.loads(path.read_text(encoding="utf-8"))
                validated += 1
                if record["summary"]["technical_valid"] != 1:
                    technical_invalid.append(sid)
                mismatched_fields = recompute_event_endpoints(record, task)
                if mismatched_fields:
                    event_mismatches.append({
                        "simulation_id": sid,
                        "fields": sorted(set(mismatched_fields)),
                    })
                epoch_fields = recompute_long_epoch_transfer_endpoints(
                    record, task
                )
                if epoch_fields:
                    long_epoch_mismatches.append({
                        "simulation_id": sid,
                        "fields": sorted(set(epoch_fields)),
                    })
                if sid in primary_simulation_ids:
                    primary_records[sid] = record
            except Exception as exc:  # report every bad checkpoint, do not hide it
                structural_invalid.append({
                    "simulation_id": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    checks["all_checkpoints_structurally_valid"] = (
        validated == expected_unique_simulations and not structural_invalid
    )
    checks["zero_technical_invalid_simulations"] = not technical_invalid
    checks["raw_events_reproduce_whole_trace_endpoints"] = not event_mismatches
    checks["raw_events_reproduce_long_epoch_transfer"] = (
        not long_epoch_mismatches
    )

    family10_computable = 0
    primary_computability: Dict[str, object] = {
        "finite_seed_contrast_count_by_family": {
            family: 0 for family in EXPECTED_PRIMARY_FAMILIES
        },
        "expected_per_family": 132,
        "failures": [{"error": "upstream structural/event validation failed"}],
        "passes": False,
    }
    if (
        not structural_invalid and not event_mismatches
        and not long_epoch_mismatches and not technical_invalid
    ):
        primary_computability = primary_family_computability(
            task_to_sim, primary_records
        )
        family10_computable = int(
            primary_computability["finite_seed_contrast_count_by_family"][10]
        )
    checks["family10_true_2x2_observable_cells_complete"] = (
        family10_computable == len(EXPECTED_PRIMARY_SEEDS)
    )
    checks["all_ten_families_132_finite_computable"] = bool(
        primary_computability["passes"]
    )

    derivative_files = (
        output_dir / "metrics.csv",
        output_dir / "unique_simulation_metrics.csv",
        output_dir / "long_epoch_metrics.csv",
    )
    checks["compiled_tables_present"] = all(
        path.is_file() and path.stat().st_size > 0 for path in derivative_files
    )
    inventory_paths = (
        plan_path, manifest_path, completion_path, index_path, *derivative_files
    )
    result = {
        "preflight_version": "v2.6.1-causal-1.0",
        "preflight_kind": "strict_postrun_before_scientific_analysis",
        "output_dir": str(output_dir),
        "expected_analysis_tasks": EXPECTED_ANALYSIS_TASKS,
        "expected_task_identity_sha256": EXPECTED_TASK_IDENTITY_SHA256,
        "expected_stage_tasks_per_seed": EXPECTED_STAGE_TASKS_PER_SEED,
        "expected_stage_task_totals": EXPECTED_STAGE_TASK_TOTALS,
        "expected_production_seeds_by_stage": {
            stage: list(seeds)
            for stage, seeds in EXPECTED_PRODUCTION_SEEDS_BY_STAGE.items()
        },
        "expected_unique_simulations": EXPECTED_UNIQUE_SIMULATIONS,
        "regenerated_unique_simulations": expected_unique_simulations,
        "validated_checkpoint_count": validated,
        "technical_invalid_count": len(technical_invalid),
        "technical_invalid_simulation_ids": technical_invalid[:100],
        "structural_invalid_count": len(structural_invalid),
        "structural_invalid_checkpoints": structural_invalid[:100],
        "event_recomputation_mismatch_count": len(event_mismatches),
        "event_recomputation_mismatches": event_mismatches[:100],
        "long_epoch_transfer_mismatch_count": len(long_epoch_mismatches),
        "long_epoch_transfer_mismatches": long_epoch_mismatches[:100],
        "family10_event_computable_seed_count": family10_computable,
        "primary_family_computability": primary_computability,
        "primary_runtime_contract_violations": runtime_contract_violations,
        "task_matrix_violations": task_violations[:100],
        "primary_analysis_file": primary_path.name,
        "primary_analysis_contract": primary_contract,
        "derivative_file_sha256": {
            path.name: sha256(path) for path in inventory_paths if path.is_file()
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    if report_path is None:
        report_path = output_dir / "postrun_preflight_report_v2_6_1.json"
    report_path.write_text(
        json.dumps(model.json_safe(result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    result = run_preflight(
        Path(args.results_dir),
        None if args.report is None else Path(args.report).resolve(),
    )
    print(json.dumps(model.json_safe(result), indent=2, allow_nan=False))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
