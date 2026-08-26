#!/usr/bin/env python3
"""Cross-audit the complete locked A--H manuscript analysis layer."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = ROOT.parent
DERIVED = REPOSITORY_ROOT / "derived" / "manuscript_analysis_v2_6_2"
OUT = DERIVED / "master_analysis_qc.json"
PROTOCOL = ROOT / "ANALYSIS_PROTOCOL_LOCKED_2026-08-26.md"
SPEC = ROOT / "PROTOCOL_SPEC.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, int, list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return rows, len(header), header


def require(condition: bool, name: str, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    if not condition:
        raise RuntimeError(f"Master analysis QC failed: {name}")


def main() -> None:
    checks: dict[str, bool] = {}
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    source = (ROOT / spec["source_directory"]).resolve()
    source.relative_to(REPOSITORY_ROOT)
    source_qc = json.loads((DERIVED / "source_protocol_qc.json").read_text(encoding="utf-8"))
    a_qc = json.loads((DERIVED / "a_to_d_qc.json").read_text(encoding="utf-8"))
    ef_qc = json.loads((DERIVED / "e_f_qc.json").read_text(encoding="utf-8"))
    ef_manifest = json.loads((DERIVED / "e_f_manifest.json").read_text(encoding="utf-8"))
    gh_qc = json.loads((DERIVED / "g_h_qc.json").read_text(encoding="utf-8"))

    current_protocol_hash = sha256(PROTOCOL)
    current_spec_hash = sha256(SPEC)
    require(source_qc["all_checks_pass"] is True, "source_protocol_qc_pass", checks)
    require(a_qc["status"] == "PASS", "a_to_d_qc_pass", checks)
    require(ef_qc["qc_pass"] is True, "e_f_qc_pass", checks)
    require(gh_qc["all_checks_pass"] is True, "g_h_qc_pass", checks)

    script_paths = {
        "validate_sources_and_protocol.py": ROOT / "scripts/validate_sources_and_protocol.py",
        "analyze_a_to_d.py": ROOT / "scripts/analyze_a_to_d.py",
        "analyze_e_f.py": ROOT / "scripts/analyze_e_f.py",
        "analyze_g_h.py": ROOT / "scripts/analyze_g_h.py",
    }
    script_hashes = {name: sha256(path) for name, path in script_paths.items()}
    require(
        a_qc["script"]["sha256"] == script_hashes["analyze_a_to_d.py"],
        "a_to_d_script_hash_current",
        checks,
    )
    require(
        ef_qc["script_sha256"] == script_hashes["analyze_e_f.py"],
        "e_f_script_hash_current",
        checks,
    )
    require(
        gh_qc["script_sha256"] == script_hashes["analyze_g_h.py"],
        "g_h_script_hash_current",
        checks,
    )
    require(
        a_qc["locked_protocol_hashes"]["ANALYSIS_PROTOCOL_LOCKED_2026-08-26.md"]
        == current_protocol_hash,
        "a_to_d_locked_protocol_hash_current",
        checks,
    )
    for name, payload_hash in (
        ("source", source_qc["protocol_spec_sha256"]),
        ("a_to_d", a_qc["locked_protocol_hashes"]["PROTOCOL_SPEC.json"]),
        ("e_f", ef_qc["protocol_spec_sha256"]),
        ("g_h", gh_qc["protocol_spec_sha256"]),
    ):
        require(payload_hash == current_spec_hash, f"{name}_protocol_spec_hash_current", checks)

    expected = spec["expected_counts"]
    require(source_qc["counts"]["analysis_tasks"] == expected["analysis_tasks"], "master_11686_tasks", checks)
    require(source_qc["counts"]["unique_simulations"] == expected["unique_simulations"], "master_3610_simulations", checks)
    require(source_qc["counts"]["long_epoch_rows"] == expected["long_epoch_rows"], "master_744_epochs", checks)
    require(ef_qc["checks"]["stage_f_factorial_cells"] == 2700, "master_F_2700_cells", checks)
    require(gh_qc["counts"]["H_conditions"] == 72, "master_H_72_conditions", checks)

    audited_csv: dict[str, dict[str, Any]] = {}
    forbidden = set(spec["forbidden_inference_fields"])
    forbidden_hits: dict[str, list[str]] = {}
    composite_hits: list[str] = []
    for path in sorted(DERIVED.glob("*.csv")):
        rows, fields, header = csv_shape(path)
        audited_csv[path.name] = {
            "rows": rows,
            "fields": fields,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        hits = sorted(forbidden & set(header))
        if hits:
            forbidden_hits[path.name] = hits
        if any("recovery_composite" in field for field in header):
            composite_hits.append(path.name)
    require(not forbidden_hits, "no_forbidden_inference_csv_fields", checks)
    require(not composite_hits, "no_composite_recovery_csv_fields", checks)

    for name, info in a_qc["outputs"].items():
        require(name in audited_csv, f"a_to_d_output_exists::{name}", checks)
        require(audited_csv[name]["rows"] == info["row_count"], f"a_to_d_row_count::{name}", checks)
        require(audited_csv[name]["sha256"] == info["sha256"], f"a_to_d_hash::{name}", checks)

    for info in ef_manifest["files"]:
        name = info["name"]
        path = DERIVED / name
        require(path.exists(), f"e_f_output_exists::{name}", checks)
        if path.suffix == ".csv":
            require(audited_csv[name]["rows"] == info["row_count"], f"e_f_row_count::{name}", checks)
        require(sha256(path) == info["sha256"], f"e_f_hash::{name}", checks)

    for name, info in gh_qc["outputs"].items():
        path = DERIVED / name
        require(path.exists(), f"g_h_output_exists::{name}", checks)
        if path.suffix == ".csv":
            require(audited_csv[name]["rows"] == info["row_count"], f"g_h_row_count::{name}", checks)
        require(sha256(path) == info["sha256"], f"g_h_hash::{name}", checks)

    capsule = DERIVED / "g_h_locked_10_contrast_capsule.csv"
    frozen_capsule = source / "single_realization_contrasts_v2_6_2.csv"
    require(capsule.read_bytes() == frozen_capsule.read_bytes(), "locked_capsule_byte_identical", checks)

    code_composite_hits = []
    for path in script_paths.values():
        if "recovery_composite" in path.read_text(encoding="utf-8"):
            code_composite_hits.append(path.name)
    require(not code_composite_hits, "no_composite_recovery_in_analysis_code", checks)

    payload = {
        "schema": "cpg-master-analysis-qc-1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(SCRIPT.relative_to(REPOSITORY_ROOT)),
        "script_sha256": sha256(SCRIPT),
        "locked_protocol_sha256": current_protocol_hash,
        "protocol_spec_sha256": current_spec_hash,
        "analysis_script_sha256": script_hashes,
        "counts": {
            "analysis_tasks": expected["analysis_tasks"],
            "unique_simulations": expected["unique_simulations"],
            "long_epoch_rows": expected["long_epoch_rows"],
            "F_factorial_cells": 2700,
            "H_conditions": 72,
            "derived_csv_files": len(audited_csv),
            "derived_csv_rows_total": sum(item["rows"] for item in audited_csv.values()),
            "technical_invalid_unique_simulations": source_qc["counts"]["technical_invalid_unique_simulations"],
            "scientific_invalid_unique_simulations": source_qc["counts"]["scientific_invalid_unique_simulations"],
        },
        "audited_csv": audited_csv,
        "forbidden_inference_field_hits": forbidden_hits,
        "composite_recovery_hits": composite_hits + code_composite_hits,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "inference_scope": "conditional on one frozen realization; no sampling inference"
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))
    print(f"all_checks_pass={payload['all_checks_pass']}")
    print(OUT)


if __name__ == "__main__":
    main()
