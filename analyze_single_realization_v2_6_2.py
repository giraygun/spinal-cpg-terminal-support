#!/usr/bin/env python3
"""Descriptive mechanistic contrasts for one frozen network realization.

No p-values, confidence intervals, standard errors or pseudo-replicate counts
are produced.  The ten endpoint definitions are imported unchanged from the
locked v2.6.1 primary analyzer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Mapping

import dual_timescale_spinal_cpg_v2_6_1_candidate as model
import analyze_primary_v2_6_1 as locked_endpoints
import preflight_single_realization_v2_6_2 as preflight
import run_ah_experiments_accelerated_v2_6_1 as accelerated
import run_single_realization_v2_6_2 as single


def contrast_record(family: int, value: float) -> Dict[str, object]:
    return {
        "family": family,
        "endpoint": locked_endpoints.FAMILIES[family],
        "contrast_definition": (
            "dynamic_MT_minus_static_matched_MT"
            if family <= 9
            else "challenge_by_route_impairment_difference_in_differences_on_RG_to_MN_transfer"
        ),
        "contrast_value": float(value),
        "favorable_direction": "negative",
        "direction_is_favorable": bool(value < 0.0),
        "zero_is_neutral": bool(value == 0.0),
        "independent_stochastic_realization_count": 1,
        "interpretation": (
            "favorable_within_frozen_realization"
            if value < 0.0 else
            "neutral_within_frozen_realization"
            if value == 0.0 else
            "unfavorable_within_frozen_realization"
        ),
    }


def analyze(output_dir: Path) -> Dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    audit = preflight.run_preflight(output_dir)
    if not audit["all_checks_pass"]:
        raise RuntimeError("strict single-realization preflight failed")

    cache: Dict[str, Mapping[str, object]] = {}
    family_results = []
    for family in range(1, 10):
        value = locked_endpoints.h_family_contrast(
            output_dir, single.FIXED_SEED, family, cache
        )
        family_results.append(contrast_record(family, value))
    family_results.append(contrast_record(
        10,
        locked_endpoints.g_family_contrast(
            output_dir, single.FIXED_SEED, cache
        ),
    ))

    all_favorable = all(
        row["direction_is_favorable"] for row in family_results
    )
    overall = (
        "mechanistic_support_within_frozen_realization"
        if all_favorable else
        "not_all_preregistered_contrasts_favorable_within_frozen_realization"
    )
    csv_path = output_dir / "single_realization_contrasts_v2_6_2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(family_results[0]))
        writer.writeheader()
        writer.writerows(family_results)

    result = {
        "analysis_version": "single-realization-mechanistic-analysis-2.6.2",
        "design_type": single.DESIGN_TYPE,
        "model_version": model.MODEL_VERSION,
        "fixed_seed": single.FIXED_SEED,
        "fixed_structural_seed": single.FIXED_STRUCTURAL_SEED,
        "independent_stochastic_realization_count": 1,
        "inferential_scope": "conditional_on_one_frozen_network_realization",
        "stochastic_population_inference_authorized": False,
        "routes_cells_contexts_are_independent_replicates": False,
        "primary_family_count": 10,
        "overall_interpretation": overall,
        "general_hypothesis_statistically_confirmed": False,
        "general_hypothesis_statistically_rejected": False,
        "interpretation_limit": (
            "The result is a conditional mechanistic demonstration in seed 601; "
            "it does not quantify robustness across alternative network/noise realizations."
        ),
        "endpoint_contract_source": "analyze_primary_v2_6_1.py",
        "preflight_report": preflight.REPORT_NAME,
        "contrast_file": csv_path.name,
        "family_results": family_results,
    }
    accelerated.write_json_atomic(
        output_dir / "single_realization_results_v2_6_2.json", result
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    args = parser.parse_args()
    print(json.dumps(analyze(Path(args.results_dir)), indent=2))


if __name__ == "__main__":
    main()
