# Changelog

## v2.6.2 reviewer release - 2026-08-26

- Preserves the completed single-realization A-H experiment at seed 601 and
  structural seed 160601.
- Includes 11,686 analysis tasks represented by 3,610 unique simulations.
- Includes complete result CSV/JSON files, completion record, experiment plan,
  post-run preflight PASS record, and historical production log.
- Retains the byte-identical v2.6.1 frozen model core and endpoint definitions.
- Adds reviewer-facing integrity, smoke-test, safe-reanalysis, environment,
  version-map, citation, and data-dictionary layers without modifying the
  frozen scientific core or outputs.
- Adds the locked, read-only manuscript-analysis layer, including A-H analysis
  scripts, Tables R1-R6, panel-data CSVs, Figures 4-8, captions, provenance,
  and semantic reference-output verification.
- Removes AppleDouble files, bytecode caches, and macOS quarantine metadata from
  the curated reviewer copy only. The original archive is preserved separately.
- Redacts one local workstation path from the reviewer-facing production log;
  the exact unredacted log remains in the preserved original archive.
- Applies the author-approved BSD 3-Clause license to software and CC BY 4.0
  to frozen and explicitly published derived scientific data.
- Defers the immutable GitHub `v2.6.2` release and Zenodo deposit until the
  updated analysis layer passes clean remote CI and the final manuscript and
  administrative metadata are fixed; the frozen model equations, parameters,
  and outputs remain unchanged.
- Extends the reviewer verifier and CI to check full curated-package coverage,
  reproduce both the archived and manuscript analyses, compare publication
  outputs semantically, and build/run the pinned reviewer container.
