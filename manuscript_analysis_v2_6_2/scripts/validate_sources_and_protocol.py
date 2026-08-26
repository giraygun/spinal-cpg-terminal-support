#!/usr/bin/env python3
"""Validate the frozen v2.6.2 sources against the locked manuscript protocol.

This script performs structural and provenance checks only. It does not select
or summarize scientific effects.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
ANALYSIS_ROOT = HERE.parents[1]
REPOSITORY_ROOT = ANALYSIS_ROOT.parent
SPEC_PATH = ANALYSIS_ROOT / "PROTOCOL_SPEC.json"
PROTOCOL_PATH = ANALYSIS_ROOT / "ANALYSIS_PROTOCOL_LOCKED_2026-08-26.md"
DERIVED_ROOT = REPOSITORY_ROOT / "derived" / "manuscript_analysis_v2_6_2"
OUT_PATH = DERIVED_ROOT / "source_protocol_qc.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, label: str, checks: dict[str, Any]) -> None:
    checks[label] = bool(condition)
    if not condition:
        raise RuntimeError(f"QC failed: {label}")


def comparable_value(field: str, value: str) -> str:
    """Canonicalize unordered list-valued design fields before comparison."""
    if field in {"ablations", "impaired_mt_routes", "challenged_routes"}:
        if value in {"", "none"}:
            return value
        return "+".join(sorted(value.split("+")))
    return value


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    source_dir = (ANALYSIS_ROOT / spec["source_directory"]).resolve()
    checks: dict[str, Any] = {}

    observed_hashes: dict[str, str] = {}
    for name, expected in spec["source_sha256"].items():
        path = source_dir / name
        observed = sha256(path)
        observed_hashes[name] = observed
        require(observed == expected, f"sha256::{name}", checks)

    index_rows = read_csv(source_dir / "analysis_task_index.csv")
    metric_rows = read_csv(source_dir / "metrics.csv")
    unique_rows = read_csv(source_dir / "unique_simulation_metrics.csv")
    epoch_rows = read_csv(source_dir / "long_epoch_metrics.csv")

    expected = spec["expected_counts"]
    require(len(index_rows) == expected["analysis_tasks"], "analysis_task_count", checks)
    require(len(metric_rows) == expected["analysis_tasks"], "metrics_row_count", checks)
    require(len(unique_rows) == expected["unique_simulations"], "unique_simulation_count", checks)
    require(len(epoch_rows) == expected["long_epoch_rows"], "long_epoch_row_count", checks)

    index_task_ids = [row["task_id"] for row in index_rows]
    metric_task_ids = [row["task_id"] for row in metric_rows]
    unique_sim_ids = [row["simulation_id"] for row in unique_rows]
    require(len(set(index_task_ids)) == len(index_task_ids), "index_task_id_unique", checks)
    require(len(set(metric_task_ids)) == len(metric_task_ids), "metrics_task_id_unique", checks)
    require(len(set(unique_sim_ids)) == len(unique_sim_ids), "unique_simulation_id_unique", checks)
    require(set(index_task_ids) == set(metric_task_ids), "index_metrics_task_sets_equal", checks)
    require({row["simulation_id"] for row in index_rows} == set(unique_sim_ids), "index_unique_sim_sets_equal", checks)

    metric_by_task = {row["task_id"]: row for row in metric_rows}
    mapping_mismatches = sum(
        metric_by_task[row["task_id"]]["simulation_id"] != row["simulation_id"]
        for row in index_rows
    )
    require(mapping_mismatches == 0, "task_to_simulation_mapping_exact", checks)

    task_frequency = Counter(row["simulation_id"] for row in index_rows)
    reuse_mismatches = sum(
        int(row["reuse_count"]) != task_frequency[row["simulation_id"]]
        for row in index_rows
    )
    require(reuse_mismatches == 0, "reuse_count_exact", checks)
    require(
        len(index_rows) - len(unique_rows) == expected["avoided_recomputations"],
        "avoided_recomputations_exact",
        checks,
    )

    stage_task_counts = Counter(row["stage"] for row in index_rows)
    stage_unique: dict[str, int] = {}
    for stage in sorted(stage_task_counts):
        stage_unique[stage] = len({
            row["simulation_id"] for row in index_rows if row["stage"] == stage
        })
    require(stage_task_counts == Counter(expected["stage_tasks"]), "stage_task_counts_exact", checks)
    require(stage_unique == expected["stage_unique_simulations"], "stage_unique_counts_exact", checks)

    index_fields = set(index_rows[0])
    metric_fields = set(metric_rows[0])
    index_common = index_fields & metric_fields
    index_mismatches = 0
    for index_row in index_rows:
        metric_row = metric_by_task[index_row["task_id"]]
        for field in index_common:
            if index_row[field] != metric_row[field]:
                index_mismatches += 1
    require(index_mismatches == 0, "index_fields_match_metrics", checks)

    unique_by_sim = {row["simulation_id"]: row for row in unique_rows}
    nonphysical_fields = {"task_id", "stage", "label", "reuse_count"}
    common_physical = (metric_fields & set(unique_rows[0])) - nonphysical_fields
    physical_mismatches = 0
    for metric_row in metric_rows:
        unique_row = unique_by_sim[metric_row["simulation_id"]]
        for field in common_physical:
            if comparable_value(field, metric_row[field]) != comparable_value(
                field, unique_row[field]
            ):
                physical_mismatches += 1
    require(physical_mismatches == 0, "metrics_match_unique_physical_fields", checks)

    transfer_identity_errors = 0
    for row in unique_rows:
        for prefix in ("pf", "mn"):
            anchor = row[f"{prefix}_transfer_anchor_count"]
            matched = row[f"{prefix}_transfer_matched_count"]
            missed = row[f"{prefix}_transfer_missed_count"]
            if anchor == matched == missed == "":
                continue
            if int(matched) + int(missed) != int(anchor):
                transfer_identity_errors += 1
    require(transfer_identity_errors == 0, "matched_plus_missed_equals_anchor", checks)

    epoch_keys = [(row["simulation_id"], int(row["epoch"])) for row in epoch_rows]
    require(len(set(epoch_keys)) == len(epoch_keys), "long_epoch_key_unique", checks)
    g_simulations = {
        row["simulation_id"] for row in index_rows if row["stage"] == "G"
    }
    require({key[0] for key in epoch_keys} == g_simulations, "long_epoch_G_simulation_set_exact", checks)
    epoch_counts = Counter(key[0] for key in epoch_keys)
    require(set(epoch_counts.values()) == {24}, "twenty_four_epochs_per_G_simulation", checks)
    require({key[1] for key in epoch_keys} == set(range(1, 25)), "epoch_domain_1_to_24", checks)

    f_rows = [row for row in index_rows if row["stage"] == "F"]
    f_pattern = re.compile(
        r"^factorial_(?P<class>RG|PF|MN|V0D|V0V|V2a|V3|V1Ia|V1Ren|V2b)"
        r"_M_(?P<route>RG|PF|MN|V0D|V0V|V2a|V3|V1Ia|V1Ren|V2b)"
        r"_(?P<arm>A[01]M[01])$"
    )
    f_cells: dict[tuple[str, ...], set[str]] = defaultdict(set)
    f_parse_errors = 0
    for row in f_rows:
        match = f_pattern.fullmatch(row["label"])
        if match is None:
            f_parse_errors += 1
            continue
        key = (
            match.group("class"), match.group("route"), row["speed"],
            row["load"], row["load_side"], row["pulse"],
        )
        f_cells[key].add(match.group("arm"))
    require(f_parse_errors == 0, "F_labels_parse", checks)
    require(len(f_cells) == 10 * 10 * 27, "F_factorial_cell_count", checks)
    require(
        all(arms == {"A0M0", "A1M0", "A0M1", "A1M1"} for arms in f_cells.values()),
        "F_four_arms_complete",
        checks,
    )

    h_rows = [row for row in index_rows if row["stage"] == "H"]
    h_cells = {
        (row["fast_mode"], row["mt_mode"], row["pulse"]) for row in h_rows
    }
    expected_h = {
        (fast, mt, pulse)
        for fast in spec["axes"]["fast_modes"]
        for mt in spec["axes"]["mt_modes"]
        for pulse in spec["axes"]["pulse"]
    }
    require(h_cells == expected_h, "H_4x6x3_complete", checks)

    technical_invalid = sum(row["technical_valid"] != "1" for row in unique_rows)
    scientific_invalid = sum(row["scientific_valid"].lower() not in {"true", "1"} for row in unique_rows)

    payload = {
        "schema": "cpg-source-protocol-qc-1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(HERE.relative_to(REPOSITORY_ROOT)),
        "script_sha256": sha256(HERE),
        "locked_protocol_sha256": sha256(PROTOCOL_PATH),
        "protocol_spec_sha256": sha256(SPEC_PATH),
        "source_directory": str(source_dir.relative_to(REPOSITORY_ROOT)),
        "observed_source_sha256": observed_hashes,
        "counts": {
            "analysis_tasks": len(index_rows),
            "unique_simulations": len(unique_rows),
            "avoided_recomputations": len(index_rows) - len(unique_rows),
            "long_epoch_rows": len(epoch_rows),
            "stage_tasks": dict(sorted(stage_task_counts.items())),
            "stage_unique_simulations": stage_unique,
            "technical_invalid_unique_simulations": technical_invalid,
            "scientific_invalid_unique_simulations": scientific_invalid,
            "F_factorial_cells": len(f_cells),
            "H_cells": len(h_cells)
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "inference_scope": "one frozen stochastic realization; structural QC only"
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))
    print(f"all_checks_pass={payload['all_checks_pass']}")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
