#!/usr/bin/env python3
"""Execute the preregistered ten outcome-independent contrasts for v2.6.1."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy import stats

import dual_timescale_spinal_cpg_v2_6_1_candidate as model
import preflight_results_v2_6_1 as preflight
import run_ah_experiments_v2_6_1 as reference
import run_ah_experiments_accelerated_v2_6_1 as accelerated


PRIMARY_ANALYSIS_CONTRACT = {
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

ALPHA_PRIMARY = float(PRIMARY_ANALYSIS_CONTRACT["alpha_per_family_two_sided"])
EXPECTED_PAIRED_SEED_COUNT = int(PRIMARY_ANALYSIS_CONTRACT["paired_seed_count"])
RECOVERY_HORIZON_S = float(
    PRIMARY_ANALYSIS_CONTRACT["family_9_recovery_horizon_s"]
)
PRIMARY_MT_COMPARATOR = str(
    PRIMARY_ANALYSIS_CONTRACT["families_1_to_9_comparator"]
)
PULSES_1_TO_8 = tuple(
    PRIMARY_ANALYSIS_CONTRACT["families_1_to_8_pulse_contexts"]
)
PULSES_9 = tuple(PRIMARY_ANALYSIS_CONTRACT["family_9_pulse_contexts"])
FAMILIES = {
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
FAMILY10_EVENT_KEYS = (
    "RG_L_F_onset_s", "RG_L_E_onset_s",
    "RG_R_F_onset_s", "RG_R_E_onset_s",
    "MN_L_F_onset_s", "MN_L_E_onset_s",
    "MN_R_F_onset_s", "MN_R_E_onset_s",
)
FAMILY6_OBSERVABLE_KEYS = tuple(
    PRIMARY_ANALYSIS_CONTRACT["family_6"]["sufficient_observables"]
)
FAMILY9_RG_EVENT_KEYS = tuple(
    key for key in FAMILY10_EVENT_KEYS if key.startswith("RG_")
)
FAMILY9_INTERVENTION_KEYS = (
    "pulse_required", "pulse_delivered", "pulse_start_s", "pulse_end_s",
    "pulse_noneligibility_reason",
)


def probability_burden(events: int, trials: int, failure: int) -> float:
    if trials > 0:
        if not 0 <= events <= trials:
            raise ValueError("invalid event/trial counts")
        return float(math.asin(math.sqrt((events + 0.5) / (trials + 1.0))))
    if failure == 1:
        return math.pi / 2.0
    raise ValueError("zero denominator without rhythmic failure")


def _finite_nonnegative_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def bilateral_mn_amplitude_imbalance_burden(
    record: Mapping[str, object], _biological_failure: int,
) -> float:
    """Rebuild Family 6 from its three persisted sufficient observables."""
    payload = record.get("analysis_observables")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint lacks analysis_observables")
    if any(key not in payload for key in FAMILY6_OBSERVABLE_KEYS):
        raise ValueError("checkpoint lacks Family-6 sufficient observables")
    sample_count = payload["mn_rate_sample_count"]
    if (
        isinstance(sample_count, bool) or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise ValueError("mn_rate_sample_count must be a positive integer")
    left_sum = _finite_nonnegative_number(
        payload["mn_left_rate_sum_hz_samples"],
        "mn_left_rate_sum_hz_samples",
    )
    right_sum = _finite_nonnegative_number(
        payload["mn_right_rate_sum_hz_samples"],
        "mn_right_rate_sum_hz_samples",
    )
    left_mean = left_sum / sample_count
    right_mean = right_sum / sample_count
    denominator = left_mean + right_mean
    if denominator == 0.0:
        # Complete bilateral MN silence is itself a biological motor-output
        # failure, irrespective of whether the upstream RG still oscillates.
        return 1.0
    return float(abs(left_mean - right_mean) / denominator)


def _event_arrays(
    record: Mapping[str, object], keys: Sequence[str],
) -> Dict[str, np.ndarray]:
    payload = record.get("analysis_events")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint lacks an analysis_events mapping")
    result: Dict[str, np.ndarray] = {}
    for key in keys:
        if key not in payload:
            raise ValueError(f"checkpoint lacks required event stream: {key}")
        values = np.asarray(payload[key], dtype=float)
        if (
            values.ndim != 1
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(np.diff(values) <= 0.0)
        ):
            raise ValueError(f"invalid event stream: {key}")
        result[key] = values
    return result


def _nullable_finite_time(value: object, label: str) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be null or a JSON number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be null or finite and nonnegative")
    return result


def family9_recovery_burden(
    record: Mapping[str, object], task: reference.Task,
) -> float:
    """Rebuild Family 9 only from RG events and the intervention log."""
    summary = record.get("summary")
    if not isinstance(summary, Mapping) or summary.get("technical_valid") != 1:
        raise ValueError("technical-invalid simulation reached primary analysis")
    log = record.get("intervention_log")
    if not isinstance(log, Mapping):
        raise ValueError("checkpoint lacks intervention_log")
    if any(key not in log for key in FAMILY9_INTERVENTION_KEYS):
        raise ValueError("checkpoint lacks Family-9 intervention-log fields")
    required = log["pulse_required"]
    delivered = log["pulse_delivered"]
    if required not in (0, 1) or delivered not in (0, 1):
        raise ValueError("pulse log flags must be binary integers")
    expected_required = int(task.protocol == "pulse" and task.pulse != "none")
    if required != expected_required or required != 1:
        raise ValueError("Family 9 requires a logged active pulse arm")
    pulse_start = _nullable_finite_time(log["pulse_start_s"], "pulse_start_s")
    pulse_end = _nullable_finite_time(log["pulse_end_s"], "pulse_end_s")
    reason = log["pulse_noneligibility_reason"]
    if not isinstance(reason, str):
        raise ValueError("pulse_noneligibility_reason must be a string")
    cfg = reference.config_for_task(task, smoke=False)
    rg_events = _event_arrays(record, FAMILY9_RG_EVENT_KEYS)
    bursts = {
        key.removesuffix("_onset_s"): values
        for key, values in rg_events.items()
    }
    if delivered == 1:
        if not np.isfinite(pulse_start) or not np.isfinite(pulse_end):
            raise ValueError("delivered pulse lacks finite start/end times")
        if pulse_end <= pulse_start or pulse_end > cfg.duration_s:
            raise ValueError("delivered pulse interval is invalid")
        if reason != "none":
            raise ValueError("delivered pulse must have intervention reason 'none'")
        censor_time = float(cfg.duration_s - pulse_end)
        if censor_time < RECOVERY_HORIZON_S:
            raise ValueError("less than four seconds of post-pulse follow-up")
        recovery = model.recovery_outcome_from_phase_and_period(
            bursts, cfg, pulse_start, pulse_end
        )
        if recovery["recovery_endpoint_eligible"] == 1:
            value = recovery["recovery_time_or_censor_s"]
        else:
            # Insufficient pre-pulse biological rhythm is retained as the
            # preregistered worst non-recovery burden, never excluded.
            value = max(0.0, cfg.duration_s - cfg.pulse_arm_after_s)
    else:
        if np.isfinite(pulse_start) or np.isfinite(pulse_end):
            raise ValueError("undelivered pulse cannot have finite start/end")
        if reason != "biological_no_phase_eligible_cycle":
            raise ValueError("undelivered pulse lacks the biological-failure reason")
        value = max(0.0, cfg.duration_s - cfg.pulse_arm_after_s)
    if not np.isfinite(float(value)):
        raise ValueError("Family-9 recovery burden is non-finite")
    return min(float(value), RECOVERY_HORIZON_S)


def endpoint_burden(
    summary: Mapping[str, object],
    family: int,
    *,
    record: Mapping[str, object] | None = None,
    task: reference.Task | None = None,
) -> float:
    if summary["technical_valid"] != 1:
        raise ValueError("technical-invalid simulation reached primary analysis")
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
        if record is None:
            raise ValueError("Family 6 requires record-level sufficient observables")
        return bilateral_mn_amplitude_imbalance_burden(record, failure)
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
    if family == 9:
        if record is None or task is None:
            raise ValueError("Family 9 requires raw events, intervention log, and task")
        return family9_recovery_burden(record, task)
    raise ValueError(f"unsupported H endpoint family: {family}")


def huber_location(values: np.ndarray, tuning: float = 1.345) -> float:
    location = float(np.median(values))
    mad = float(np.median(np.abs(values - location)))
    scale = max(1.4826 * mad, float(np.std(values, ddof=1)) * 1e-6, 1e-12)
    for _ in range(100):
        residual = (values - location) / scale
        weights = np.ones_like(values)
        large = np.abs(residual) > tuning
        weights[large] = tuning / np.abs(residual[large])
        updated = float(np.sum(weights * values) / np.sum(weights))
        if abs(updated - location) <= 1e-12 * max(1.0, abs(location)):
            return updated
        location = updated
    return location


def one_sample_result(values: Sequence[float], family: int) -> Dict[str, object]:
    array = np.asarray(values, dtype=float)
    if len(array) != EXPECTED_PAIRED_SEED_COUNT or not np.all(
        np.isfinite(array)
    ):
        raise ValueError(f"family {family} lacks 132 finite seed contrasts")
    n = len(array)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    se = sd / math.sqrt(n)
    if sd == 0.0:
        t_statistic = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
        p_value = 1.0 if mean == 0.0 else 0.0
        dz = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
    else:
        t_statistic = mean / se
        p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=n - 1))
        dz = mean / sd
    t95 = float(stats.t.ppf(0.975, df=n - 1))
    t995 = float(stats.t.ppf(1.0 - ALPHA_PRIMARY / 2.0, df=n - 1))
    try:
        wilcoxon = stats.wilcoxon(
            array, zero_method="wilcox", alternative="two-sided",
            method="auto",
        )
        wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    favorable_direction = "negative"
    favorable = mean < 0.0
    return {
        "family": family,
        "endpoint": FAMILIES[family],
        "n_seed_pairs": n,
        "contrast_definition": (
            "dynamic_MT_minus_static_matched_MT"
            if family <= 9 else
            "challenge_by_route_impairment_difference_in_differences_on_RG_to_MN_transfer"
        ),
        "favorable_direction": favorable_direction,
        "mean_contrast": mean,
        "sd_contrast": sd,
        "standard_error": se,
        "ci95": [mean - t95 * se, mean + t95 * se],
        "ci99_5_primary_decision": [mean - t995 * se, mean + t995 * se],
        "paired_effect_dz": dz,
        "t_statistic": t_statistic,
        "degrees_freedom": n - 1,
        "p_two_sided": p_value,
        "alpha_two_sided": ALPHA_PRIMARY,
        "reject_primary_null": p_value < ALPHA_PRIMARY,
        "direction_is_favorable": favorable,
        "huber_location_sensitivity": huber_location(array),
        "wilcoxon_two_sided_p_sensitivity": wilcoxon_p,
    }


def load_record(
    output_dir: Path,
    task: reference.Task,
    cache: Dict[str, Mapping[str, object]],
) -> Mapping[str, object]:
    sid = accelerated.simulation_id(task, smoke=False)
    if sid not in cache:
        path = output_dir / "simulations" / f"{sid}.json"
        accelerated.validate_checkpoint(path, sid)
        cache[sid] = json.loads(path.read_text(encoding="utf-8"))
    record = cache[sid]
    if record["scientific_valid"] is not True:
        raise ValueError(f"non-scientific checkpoint reached analysis: {sid}")
    return record


def h_task(seed: int, mt_mode: str, pulse: str) -> reference.Task:
    return reference.Task(
        "H", seed, speed="medium", load="normal", pulse=pulse,
        fast_mode="dynamic", mt_mode=mt_mode,
        label=f"control_KCa_dynamic_MT_{mt_mode}",
    )


def h_family_contrast(
    output_dir: Path,
    seed: int,
    family: int,
    cache: Dict[str, Mapping[str, object]],
) -> float:
    pulses = PULSES_9 if family == 9 else PULSES_1_TO_8
    pulse_differences = []
    for pulse in pulses:
        dynamic_task = h_task(seed, "dynamic", pulse)
        dynamic_record = load_record(output_dir, dynamic_task, cache)
        comparator_task = h_task(seed, PRIMARY_MT_COMPARATOR, pulse)
        comparator_record = load_record(output_dir, comparator_task, cache)
        dynamic = endpoint_burden(
            dynamic_record["summary"], family,
            record=dynamic_record, task=dynamic_task,
        )
        comparator = endpoint_burden(
            comparator_record["summary"], family,
            record=comparator_record, task=comparator_task,
        )
        pulse_differences.append(dynamic - comparator)
    return float(np.mean(pulse_differences))


def family10_bursts_from_record(
    record: Mapping[str, object],
) -> Dict[str, np.ndarray]:
    """Read only the preregistered observable RG/MN event streams."""
    return _event_arrays(record, FAMILY10_EVENT_KEYS)


def match_rg_mn_bursts_primary(
    rg_events: np.ndarray,
    mn_events: np.ndarray,
    start_s: float,
    end_s: float,
    pre_window_s: float,
    post_window_s: float,
) -> tuple[np.ndarray, int, int]:
    """Primary-only one-to-one matcher with complete observation support.

    Eligible RG anchors have their entire matching interval inside the
    half-open analysis interval.  MN candidates at either end of an eligible
    anchor's closed matching interval are admissible, and each MN burst may be
    consumed at most once across the complete contiguous interval.
    """
    rg = np.asarray(rg_events, dtype=float)
    mn = np.asarray(mn_events, dtype=float)
    if (
        rg.ndim != 1 or mn.ndim != 1
        or not np.all(np.isfinite(rg)) or not np.all(np.isfinite(mn))
        or np.any(np.diff(rg) < 0.0) or np.any(np.diff(mn) < 0.0)
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
        while output_cursor < len(mn) and mn[output_cursor] < lower:
            output_cursor += 1
        if output_cursor < len(mn) and mn[output_cursor] <= upper:
            latencies.append(float(mn[output_cursor] - onset))
            output_cursor += 1
        else:
            missed += 1
    return np.asarray(latencies), int(missed), int(len(anchors))


def raw_window_mn_transfer_burden(
    record: Mapping[str, object],
    task: reference.Task,
    start_epoch: int,
    end_epoch: int,
) -> float:
    """Match one contiguous epoch union directly from raw RG/MN events."""
    cfg = reference.config_for_task(task, smoke=False)
    if task.protocol != "long":
        raise ValueError("Family 10 requires a long-protocol task")
    if not 1 <= start_epoch <= end_epoch <= cfg.long_n_epochs:
        raise ValueError("invalid long-protocol epoch union")
    start_s = (start_epoch - 1) * cfg.long_epoch_duration_s
    end_s = end_epoch * cfg.long_epoch_duration_s
    bursts = family10_bursts_from_record(record)
    missed = anchors = 0
    rg_counts: list[int] = []
    for side, phase in model.SIDE_PHASES:
        rg = bursts[f"RG_{side}_{phase}_onset_s"]
        mn = bursts[f"MN_{side}_{phase}_onset_s"]
        _, local_missed, local_anchors = match_rg_mn_bursts_primary(
            rg,
            mn,
            start_s,
            end_s,
            cfg.rg_mn_match_pre_window_s,
            cfg.rg_mn_match_post_window_s,
        )
        missed += local_missed
        anchors += local_anchors
        rg_counts.append(int(np.sum((rg >= start_s) & (rg < end_s))))
    rhythmic_failure = int(min(rg_counts) < 2)
    return probability_burden(missed, anchors, rhythmic_failure)


def postchallenge_mn_transfer_deficit(
    record: Mapping[str, object], task: reference.Task,
) -> float:
    """Postchallenge minus baseline burden from two contiguous raw windows."""
    # Epoch 1 is the numerical/network burn-in and epoch 19 is the immediate
    # demand-off transition. Equal five-epoch windows prevent either transient
    # from entering the preregistered downstream endpoint.
    baseline = raw_window_mn_transfer_burden(record, task, 2, 6)
    recovery = raw_window_mn_transfer_burden(record, task, 20, 24)
    return recovery - baseline


def g_family_contrast(
    output_dir: Path,
    seed: int,
    cache: Dict[str, Mapping[str, object]],
) -> float:
    by_label = {task.label: task for task in reference.tasks_for_seed("G", seed)}
    no_challenge = load_record(
        output_dir, by_label["long_no_challenge"], cache
    )
    no_challenge_task = by_label["long_no_challenge"]
    no_challenge_deficit = postchallenge_mn_transfer_deficit(
        no_challenge, no_challenge_task
    )
    route_differences = []
    for route in model.MT_ROUTES:
        challenge_normal = load_record(
            output_dir, by_label[f"long_challenge_{route}"], cache
        )
        no_challenge_impaired = load_record(
            output_dir,
            by_label[f"long_no_challenge_impaired_{route}"], cache,
        )
        challenge_impaired = load_record(
            output_dir,
            by_label[f"long_challenge_impaired_{route}"], cache,
        )
        normal_challenge_effect = (
            postchallenge_mn_transfer_deficit(
                challenge_normal, by_label[f"long_challenge_{route}"]
            )
            - no_challenge_deficit
        )
        impaired_challenge_effect = (
            postchallenge_mn_transfer_deficit(
                challenge_impaired,
                by_label[f"long_challenge_impaired_{route}"],
            )
            - postchallenge_mn_transfer_deficit(
                no_challenge_impaired,
                by_label[f"long_no_challenge_impaired_{route}"],
            )
        )
        route_differences.append(
            normal_challenge_effect - impaired_challenge_effect
        )
    return float(np.mean(route_differences))


def analyze(output_dir: Path) -> Dict[str, object]:
    output_dir = output_dir.resolve()
    if tuple(reference.PREREGISTERED_SEEDS) != tuple(range(601, 733)):
        raise RuntimeError("paired primary seed registry drifted from 601--732")
    preflight_result = preflight.run_preflight(output_dir)
    if not preflight_result["all_checks_pass"]:
        raise RuntimeError("strict postrun preflight failed; analysis is blocked")
    cache: Dict[str, Mapping[str, object]] = {}
    seed_rows = []
    by_family: Dict[int, list[float]] = {family: [] for family in FAMILIES}
    for seed in reference.PREREGISTERED_SEEDS:
        row: Dict[str, object] = {"seed": seed}
        for family in range(1, 10):
            value = h_family_contrast(output_dir, seed, family, cache)
            by_family[family].append(value)
            row[f"family_{family}_contrast"] = value
        value = g_family_contrast(output_dir, seed, cache)
        by_family[10].append(value)
        row["family_10_contrast"] = value
        seed_rows.append(row)
    family_results = [
        one_sample_result(by_family[family], family)
        for family in sorted(FAMILIES)
    ]
    seed_path = output_dir / "primary_seed_contrasts_v2_6_1.csv"
    with seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    result = {
        "analysis_version": "v2.6.1-causal-primary-1.0",
        "primary_analysis_contract": PRIMARY_ANALYSIS_CONTRACT,
        "scientific_seed_range_inclusive": [601, 732],
        "independent_unit": "paired noise/connectome seed",
        "primary_family_count": 10,
        "alpha_per_family_two_sided": ALPHA_PRIMARY,
        "familywise_bonferroni_bound": 0.05,
        "recovery_horizon_s": RECOVERY_HORIZON_S,
        "preflight_report": "postrun_preflight_report_v2_6_1.json",
        "seed_contrast_file": seed_path.name,
        "family_results": family_results,
    }
    out = output_dir / "primary_results_v2_6_1.json"
    out.write_text(
        json.dumps(model.json_safe(result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    args = parser.parse_args()
    result = analyze(Path(args.results_dir))
    print(json.dumps(model.json_safe(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
