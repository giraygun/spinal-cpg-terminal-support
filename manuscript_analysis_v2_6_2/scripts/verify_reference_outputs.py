#!/usr/bin/env python3
"""Compare regenerated manuscript outputs with the curated reference set.

Timestamps and PDF container metadata are intentionally excluded from the
comparison. Scientific table values, captions, decoded PNG pixels, PDF page
geometry/text, the locked ten-contrast capsule, and QC PASS states are checked.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image
from pypdf import PdfReader


SCRIPT = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
GENERATED_ROOT = REPOSITORY_ROOT / "derived" / "manuscript_analysis_v2_6_2"
GENERATED_PUBLICATION = GENERATED_ROOT / "publication_outputs"
REFERENCE_ROOT = PACKAGE_ROOT / "reference_outputs"
VOLATILE_KEYS = {"generated_at_utc", "generated_utc"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def csv_without_timestamps(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"CSV has no header: {path}")
        fields = [name for name in reader.fieldnames if name not in VOLATILE_KEYS]
        rows = [
            {name: row[name] for name in fields}
            for row in reader
        ]
    return fields, rows


def compare_csv(reference: Path, generated: Path) -> None:
    require(generated.is_file(), f"missing regenerated CSV: {generated}")
    expected_fields, expected_rows = csv_without_timestamps(reference)
    actual_fields, actual_rows = csv_without_timestamps(generated)
    require(actual_fields == expected_fields, f"CSV field mismatch: {reference.name}")
    require(actual_rows == expected_rows, f"CSV scientific-content mismatch: {reference.name}")


def strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_volatile(item)
            for key, item in value.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [strip_volatile(item) for item in value]
    return value


def compare_json(reference: Path, generated: Path) -> None:
    require(generated.is_file(), f"missing regenerated JSON: {generated}")
    expected = strip_volatile(json.loads(reference.read_text(encoding="utf-8")))
    actual = strip_volatile(json.loads(generated.read_text(encoding="utf-8")))
    require(actual == expected, f"JSON scientific-content mismatch: {reference.name}")


def compare_png(reference: Path, generated: Path) -> None:
    require(generated.is_file(), f"missing regenerated PNG: {generated}")
    with Image.open(reference) as expected_image, Image.open(generated) as actual_image:
        expected = expected_image.convert("RGBA")
        actual = actual_image.convert("RGBA")
        require(actual.size == expected.size, f"PNG dimension mismatch: {reference.name}")
        require(actual.tobytes() == expected.tobytes(), f"PNG pixel mismatch: {reference.name}")


def pdf_signature(path: Path) -> list[tuple[str, str]]:
    reader = PdfReader(path)
    return [
        (
            f"{float(page.mediabox.width):.6f}x{float(page.mediabox.height):.6f}",
            page.extract_text() or "",
        )
        for page in reader.pages
    ]


def compare_pdf(reference: Path, generated: Path) -> None:
    require(generated.is_file(), f"missing regenerated PDF: {generated}")
    require(pdf_signature(generated) == pdf_signature(reference), f"PDF content mismatch: {reference.name}")


def compare_directory_file_sets(reference: Path, generated: Path) -> None:
    expected = {path.name for path in reference.iterdir() if path.is_file()}
    actual = {path.name for path in generated.iterdir() if path.is_file()}
    require(actual == expected, f"file-set mismatch: {reference.name}; missing={sorted(expected - actual)} extra={sorted(actual - expected)}")


def verify_qc() -> None:
    checks = {
        GENERATED_ROOT / "source_protocol_qc.json": ("all_checks_pass", True),
        GENERATED_ROOT / "a_to_d_qc.json": ("status", "PASS"),
        GENERATED_ROOT / "e_f_qc.json": ("qc_pass", True),
        GENERATED_ROOT / "g_h_qc.json": ("all_checks_pass", True),
        GENERATED_ROOT / "master_analysis_qc.json": ("all_checks_pass", True),
        GENERATED_PUBLICATION / "qa" / "publication_outputs_qc.json": ("all_checks_pass", True),
    }
    for path, (field, expected) in checks.items():
        require(path.is_file(), f"missing QC file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload.get(field) == expected, f"QC did not pass: {path.name}")


def main() -> None:
    table_reference = REFERENCE_ROOT / "tables"
    table_generated = GENERATED_PUBLICATION / "tables"
    figure_reference = REFERENCE_ROOT / "figures"
    figure_generated = GENERATED_PUBLICATION / "figures"
    require(table_generated.is_dir(), f"missing generated table directory: {table_generated}")
    require(figure_generated.is_dir(), f"missing generated figure directory: {figure_generated}")
    compare_directory_file_sets(table_reference, table_generated)
    compare_directory_file_sets(figure_reference, figure_generated)

    for reference in sorted(table_reference.iterdir()):
        generated = table_generated / reference.name
        if reference.suffix == ".csv":
            compare_csv(reference, generated)
        elif reference.suffix == ".json":
            compare_json(reference, generated)
        else:
            require(generated.read_bytes() == reference.read_bytes(), f"text mismatch: {reference.name}")

    for reference in sorted(figure_reference.iterdir()):
        generated = figure_generated / reference.name
        if reference.suffix == ".png":
            compare_png(reference, generated)
        elif reference.suffix == ".pdf":
            compare_pdf(reference, generated)
        elif reference.suffix == ".json":
            compare_json(reference, generated)
        else:
            require(generated.read_bytes() == reference.read_bytes(), f"text mismatch: {reference.name}")

    provenance_pairs = {
        "source_protocol_qc.json": GENERATED_ROOT / "source_protocol_qc.json",
        "g_h_locked_10_contrast_capsule.csv": GENERATED_ROOT / "g_h_locked_10_contrast_capsule.csv",
        "g_h_locked_10_contrast_capsule.metadata.json": GENERATED_ROOT / "g_h_locked_10_contrast_capsule.metadata.json",
    }
    for name, generated in provenance_pairs.items():
        reference = REFERENCE_ROOT / "provenance" / name
        if reference.suffix == ".csv":
            compare_csv(reference, generated)
        else:
            compare_json(reference, generated)

    verify_qc()
    print("manuscript_reference_output_verification=PASS")
    print("reference_tables_and_panel_csv=32")
    print("reference_figures=5_png_plus_5_pdf")
    print("independent_stochastic_realizations=1")


if __name__ == "__main__":
    main()
