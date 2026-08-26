#!/usr/bin/env python3
"""Read-only integrity verification for the v2.6.2 reviewer package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "single_realization_results_v2_6_2"
CORE_MANIFEST = ROOT / "RELEASE_CONTENTS_v2_6_2.sha256"
RESULT_MANIFEST = ROOT / "RESULTS_CONTENTS_v2_6_2.sha256"
PACKAGE_MANIFEST = ROOT / "REVIEWER_PACKAGE_CONTENTS_v2_6_2.sha256"

EXPECTED_CORE_MANIFEST_SHA256 = (
    "30c1dcc4035ed337a33f5db9edde3b78913974618f77710ea17bb37dbd248bf8"
)
EXPECTED_RESULT_MANIFEST_SHA256 = (
    "6f18eb1a78da7001d53c3dcb406a9d2faf67e29264dabcae4a6a01496c001711"
)
EXPECTED_MODEL_SHA256 = (
    "a0dc8a7338ab1619874135b1a3e8809f4eaa22394cb65dfd951544df5b62f47a"
)
EXPECTED_TASK_IDENTITY_SHA256 = (
    "30857ac18a7944a18cc022270428758a91fff49721ffec152f12edc78fdff4bc"
)
EXPECTED_STAGE_COUNTS = {
    "A": 27,
    "B": 270,
    "C": 162,
    "D": 54,
    "E": 270,
    "F": 10800,
    "G": 31,
    "H": 72,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_entries(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise RuntimeError(f"malformed manifest line {line_number}: {path.name}")
        expected, relative_text = parts
        relative_text = relative_text.lstrip("*")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe manifest path: {relative_text}")
        candidate = (ROOT / relative).resolve()
        candidate.relative_to(ROOT)
        entries.append((expected, candidate))
    return entries


def verify_manifest(path: Path, expected_manifest_sha: str) -> int:
    if sha256(path) != expected_manifest_sha:
        raise RuntimeError(f"manifest identity mismatch: {path.name}")
    entries = manifest_entries(path)
    for expected, candidate in entries:
        if not candidate.is_file():
            raise RuntimeError(f"missing file: {candidate.relative_to(ROOT)}")
        actual = sha256(candidate)
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch: {candidate.relative_to(ROOT)} "
                f"expected={expected} actual={actual}"
            )
    return len(entries)


def verify_curated_package() -> int:
    """Verify every curated file without making the manifest hash itself."""
    entries = manifest_entries(PACKAGE_MANIFEST)
    manifest_paths = [candidate.relative_to(ROOT) for _, candidate in entries]
    if len(manifest_paths) != len(set(manifest_paths)):
        raise RuntimeError("duplicate path in curated package manifest")

    for expected, candidate in entries:
        if not candidate.is_file():
            raise RuntimeError(f"missing curated file: {candidate.relative_to(ROOT)}")
        actual = sha256(candidate)
        if actual != expected:
            raise RuntimeError(
                f"curated SHA-256 mismatch: {candidate.relative_to(ROOT)} "
                f"expected={expected} actual={actual}"
            )

    actual_paths: set[Path] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative == PACKAGE_MANIFEST.relative_to(ROOT):
            continue
        if relative.parts[0] in {
            ".git",
            ".venv",
            "derived",
            "__pycache__",
            "reviewer_full_run",
            "reviewer_matrix_dryrun",
            "reviewer_numeric_smoke",
        }:
            continue
        if (
            path.name == ".DS_Store"
            or path.name.startswith("._")
            or path.suffix in {".pyc", ".pyo"}
            or relative == Path("reviewer_full_run.log")
        ):
            continue
        actual_paths.add(relative)

    manifest_path_set = set(manifest_paths)
    if manifest_path_set != actual_paths:
        missing = sorted(actual_paths - manifest_path_set)
        extra = sorted(manifest_path_set - actual_paths)
        raise RuntimeError(
            f"curated package manifest coverage mismatch: "
            f"missing={missing} extra={extra}"
        )

    required_release_metadata = {
        Path("LICENSE"),
        Path("LICENSE-DATA"),
        Path("CITATION.cff"),
        Path(".zenodo.json"),
        Path("README.md"),
        Path("REPRODUCIBILITY.md"),
    }
    if not required_release_metadata.issubset(manifest_path_set):
        absent = sorted(required_release_metadata - manifest_path_set)
        raise RuntimeError(f"release metadata absent from manifest: {absent}")
    return len(entries)


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path.name}")
    return value


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def verify_metadata() -> None:
    release = load_json(ROOT / "SINGLE_REALIZATION_RELEASE_v2_6_2.json")
    completion = load_json(RESULTS / "completion_single_realization_v2_6_2.json")
    plan = load_json(RESULTS / "experiment_plan_single_realization_v2_6_2.json")
    preflight = load_json(
        RESULTS / "postrun_preflight_single_realization_v2_6_2.json"
    )
    result = load_json(RESULTS / "single_realization_results_v2_6_2.json")

    records = (release, completion, plan, preflight, result)
    for record in records:
        if record.get("fixed_seed") != 601:
            raise RuntimeError("fixed seed mismatch")
        if record.get("fixed_structural_seed") != 160601:
            raise RuntimeError("fixed structural seed mismatch")

    for record in (release, completion, plan, preflight):
        if record.get("analysis_task_count") != 11686:
            raise RuntimeError("analysis task count mismatch")
        if record.get("unique_simulation_count") != 3610:
            raise RuntimeError("unique simulation count mismatch")
        if record.get("task_identity_sha256") != EXPECTED_TASK_IDENTITY_SHA256:
            raise RuntimeError("task identity mismatch")

    if release.get("model_sha256") != EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA-256 mismatch in release")
    if completion.get("completed_checkpoint_count") != 3610:
        raise RuntimeError("completion checkpoint count mismatch")
    if completion.get("scientific_valid") is not True:
        raise RuntimeError("completion is not scientifically valid")
    if preflight.get("all_checks_pass") is not True:
        raise RuntimeError("archived post-run preflight did not pass")
    if result.get("independent_stochastic_realization_count") != 1:
        raise RuntimeError("independent realization count mismatch")
    if result.get("primary_family_count") != 10:
        raise RuntimeError("primary family count mismatch")
    if result.get("general_hypothesis_statistically_confirmed") is not False:
        raise RuntimeError("invalid inferential claim in result metadata")
    if result.get("general_hypothesis_statistically_rejected") is not False:
        raise RuntimeError("invalid inferential claim in result metadata")

    expected_rows = {
        "analysis_task_index.csv": 11686,
        "metrics.csv": 11686,
        "unique_simulation_metrics.csv": 3610,
        "long_epoch_metrics.csv": 744,
        "single_realization_contrasts_v2_6_2.csv": 10,
    }
    for name, expected in expected_rows.items():
        actual = csv_row_count(RESULTS / name)
        if actual != expected:
            raise RuntimeError(f"row count mismatch: {name}={actual}, expected={expected}")

    stage_counts: Counter[str] = Counter()
    with (RESULTS / "analysis_task_index.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            stage_counts[row["stage"]] += 1
    if dict(stage_counts) != EXPECTED_STAGE_COUNTS:
        raise RuntimeError(f"stage count mismatch: {dict(stage_counts)}")

    simulation_files = sorted((RESULTS / "simulations").glob("sim-*.json"))
    if len(simulation_files) != 3610:
        raise RuntimeError(f"checkpoint file count mismatch: {len(simulation_files)}")

    manifest_result_paths = {
        candidate.relative_to(ROOT) for _, candidate in manifest_entries(RESULT_MANIFEST)
    }
    actual_result_paths = {
        path.relative_to(ROOT) for path in RESULTS.rglob("*") if path.is_file()
    }
    if manifest_result_paths != actual_result_paths:
        missing = sorted(actual_result_paths - manifest_result_paths)
        extra = sorted(manifest_result_paths - actual_result_paths)
        raise RuntimeError(f"result manifest coverage mismatch: missing={missing} extra={extra}")


def verify_environment() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"CPython 3.12.x required; found {sys.version.split()[0]}")
    expected = {
        "numpy": "2.3.5",
        "scipy": "1.17.0",
        "matplotlib": "3.10.8",
    }
    for package, expected_version in expected.items():
        try:
            actual = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"missing dependency: {package}") from exc
        if actual != expected_version:
            raise RuntimeError(
                f"dependency mismatch: {package}={actual}, expected={expected_version}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help="verify bytes and metadata without checking installed dependencies",
    )
    args = parser.parse_args()

    core_count = verify_manifest(CORE_MANIFEST, EXPECTED_CORE_MANIFEST_SHA256)
    result_count = verify_manifest(RESULT_MANIFEST, EXPECTED_RESULT_MANIFEST_SHA256)
    package_count = verify_curated_package()
    verify_metadata()
    if not args.artifact_only:
        verify_environment()

    print("reviewer_artifact_verification=PASS")
    print(f"frozen_release_files={core_count}")
    print(f"frozen_result_files={result_count}")
    print(f"curated_package_files={package_count}")
    print("analysis_tasks=11686")
    print("unique_simulations=3610")
    print("fixed_seed=601")
    print("fixed_structural_seed=160601")
    print("independent_stochastic_realizations=1")


if __name__ == "__main__":
    main()
