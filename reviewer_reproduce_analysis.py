#!/usr/bin/env python3
"""Reproduce locked analysis in a writable copy of the frozen result tree."""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

import analyze_single_realization_v2_6_2 as analysis
import reviewer_verify


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "single_realization_results_v2_6_2"
COMPARE_FILES = (
    "postrun_preflight_single_realization_v2_6_2.json",
    "single_realization_contrasts_v2_6_2.csv",
    "single_realization_results_v2_6_2.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        nargs="?",
        default="derived/reanalysis_v2_6_2",
        help="new directory that must not already exist",
    )
    args = parser.parse_args()

    destination = Path(args.destination).expanduser().resolve()
    if destination.exists():
        raise SystemExit(f"refusing existing destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    reviewer_verify.verify_manifest(
        reviewer_verify.CORE_MANIFEST,
        reviewer_verify.EXPECTED_CORE_MANIFEST_SHA256,
    )
    reviewer_verify.verify_manifest(
        reviewer_verify.RESULT_MANIFEST,
        reviewer_verify.EXPECTED_RESULT_MANIFEST_SHA256,
    )
    reviewer_verify.verify_metadata()
    reviewer_verify.verify_environment()

    working_results = destination / SOURCE.name
    shutil.copytree(SOURCE, working_results)
    analysis.analyze(working_results)

    mismatches = []
    for name in COMPARE_FILES:
        if not filecmp.cmp(SOURCE / name, working_results / name, shallow=False):
            mismatches.append(name)
    if mismatches:
        raise SystemExit(f"reanalysis mismatch: {', '.join(mismatches)}")

    print("reviewer_reanalysis=PASS")
    print(f"derived_results={working_results}")
    for name in COMPARE_FILES:
        print(f"byte_identical={name}")


if __name__ == "__main__":
    main()
