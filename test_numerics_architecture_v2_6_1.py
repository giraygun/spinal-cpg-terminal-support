#!/usr/bin/env python3
"""Focused numerical and executable-microcircuit tests for v2.6.1."""

from __future__ import annotations

from dataclasses import replace
import traceback

import numpy as np

import dual_timescale_spinal_cpg_v2_6_1_candidate as model


def test_exponential_rosenbrock_matches_linear_closed_form() -> None:
    rhs = np.asarray([1.0, 1.0, 1.0])
    jacobian = np.asarray([0.0, -1.0, 1.0])
    interval = np.asarray([0.1, 0.1, 0.1])
    observed, effective = model.exponential_rosenbrock_increment(
        rhs, jacobian, interval
    )
    expected = np.asarray([
        0.1,
        1.0 - np.exp(-0.1),
        np.exp(0.1) - 1.0,
    ])
    assert np.allclose(observed, expected, rtol=0.0, atol=1.0e-14)
    assert np.array_equal(effective, jacobian)


def test_threshold_inversion_matches_exponential_endpoint() -> None:
    initial = np.asarray([-60.0])
    rhs = np.asarray([20.0])
    jacobian = np.asarray([1.0])
    interval = np.asarray([1.0])
    increment, effective = model.exponential_rosenbrock_increment(
        rhs, jacobian, interval
    )
    endpoint = initial + increment
    threshold = -50.0
    fraction = model.locally_linearized_threshold_fraction(
        initial,
        threshold,
        rhs,
        effective,
        interval,
        np.asarray([0.0]),
        1.0,
        endpoint,
    )
    expected = np.log1p((threshold - initial[0]) / rhs[0])
    assert np.allclose(fraction, expected, rtol=0.0, atol=1.0e-14)


def test_default_short_trace_is_technical_and_delay_exact() -> None:
    cfg = replace(model.Config(), duration_s=0.2, burn_in_s=0.05)
    trace = model.simulate(
        cfg,
        101,
        "steady_state",
        "dynamic",
        structural_seed=160101,
        fast_mode="dynamic",
        speed_level="medium",
        load_context="normal",
        pulse_direction="none",
    )
    assert model.technical_trace_quality(trace)["technical_valid"] == 1
    assert np.max(trace["central_delay_reconstruction_max_abs_error_ms"]) <= 1e-12
    assert np.max(trace["nmj_delay_reconstruction_max_abs_error_ms"]) <= 1e-12


def test_v3_vlat_mn_microcircuit_is_bidirectionally_executable() -> None:
    cfg = replace(
        model.Config(),
        duration_s=1.2,
        burn_in_s=0.2,
        pulse_arm_after_s=0.3,
        rg_neurons=3,
        pf_neurons=2,
        relay_neurons=3,
        mn_neurons=2,
    )
    trace = model.simulate(
        cfg, 118, "steady_state", "off", structural_seed=218
    )
    assert np.array_equal(
        trace["v3_microcircuit_pathway_names"],
        np.asarray(["V3_VLat_to_ipsilateral_MN", "MN_to_V3_VLat_GluR"]),
    )
    assert np.all(trace["v3_microcircuit_edge_counts"] > 0)
    assert np.all(trace["v3_microcircuit_scheduled_edge_event_counts"] > 0)
    assert model.technical_trace_quality(trace)["technical_valid"] == 1


def main() -> None:
    tests = sorted(
        (name, function) for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    failed = []
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception:
            failed.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - len(failed)}/{len(tests)} tests passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
