# Reproducibility workflow

## Supported reviewer levels

1. **Artifact verification** checks frozen source and result hashes, complete
   curated-package coverage, release contracts, counts, licenses/metadata, and
   the recorded post-run PASS state without simulation.
2. **Technical execution** runs the seven contract tests, four focused numerical
   tests, a matrix dry run, and one reduced non-scientific simulation.
3. **Archived-analysis reproduction** copies the frozen output tree to a new
   directory, re-runs strict preflight and the original locked-contrast
   analyzer, and compares regenerated outputs byte-for-byte with the archived
   originals.
4. **Manuscript-analysis reproduction** rebuilds the descriptive A-H analyses,
   Tables R1-R6, panel data, and Figures 4-8 without rerunning the simulator.
5. **Full production rerun** is optional and historically required about 101.39
   hours with six workers.

The first four levels are the normal reviewer workflow.

## Integrity commands

The cross-platform verifier is preferred:

```bash
python3 reviewer_verify.py
```

Manual GNU commands are:

```bash
sha256sum -c RELEASE_CONTENTS_v2_6_2.sha256
sha256sum -c RESULTS_CONTENTS_v2_6_2.sha256
sha256sum -c REVIEWER_PACKAGE_CONTENTS_v2_6_2.sha256
```

On macOS, use `shasum -a 256 -c` for each manifest.

Expected frozen completion values are:

```text
scientific_valid: true
fixed_seed: 601
fixed_structural_seed: 160601
analysis_task_count: 11686
unique_simulation_count: 3610
completed_checkpoint_count: 3610
stochastic_population_inference_authorized: false
postrun all_checks_pass: true
```

## Safe analysis reproduction

Never invoke the original analyzer directly on
`single_realization_results_v2_6_2/`, because it writes its reports into the
supplied directory. Use:

```bash
python3 reviewer_reproduce_analysis.py derived/reanalysis_v2_6_2
```

The wrapper refuses an existing destination, preserves the source result tree,
and compares these regenerated files with their archived counterparts:

- `postrun_preflight_single_realization_v2_6_2.json`
- `single_realization_contrasts_v2_6_2.csv`
- `single_realization_results_v2_6_2.json`

The resulting ten-contrast capsule is retained only as its original
preregistration record. It is conditional on the frozen realization and is not
a global model score or a statistical acceptance/rejection rule for the
biological hypothesis.

## Manuscript-analysis reproduction

The manuscript layer reads `single_realization_results_v2_6_2/` in place and
writes only to the ignored `derived/manuscript_analysis_v2_6_2/` directory:

```bash
python3 manuscript_analysis_v2_6_2/scripts/validate_sources_and_protocol.py
python3 manuscript_analysis_v2_6_2/scripts/analyze_a_to_d.py
python3 manuscript_analysis_v2_6_2/scripts/analyze_e_f.py
python3 manuscript_analysis_v2_6_2/scripts/analyze_g_h.py
python3 manuscript_analysis_v2_6_2/scripts/audit_derived_outputs.py
python3 manuscript_analysis_v2_6_2/scripts/build_publication_outputs.py
python3 manuscript_analysis_v2_6_2/scripts/verify_reference_outputs.py
```

The analysis is descriptive and conditional on one frozen realization. The
source field `speed` denotes descending-drive command, not measured locomotor
speed. Rhythmic failures and the distinct recovery eligibility/event/time
fields are never collapsed into a single score. The final verifier compares
scientific CSV content, captions, decoded PNG pixels, PDF page content, compact
provenance, and all QC PASS states while excluding generation timestamps and
PDF container metadata.

## Full rerun safeguards

The archived default result directory is already complete. A full rerun must use
a new, empty result directory and a separate log:

```bash
nohup env CPG_WORKERS=6 \
  ./run_mac_single_realization_v2_6_2.sh reviewer_full_run \
  > reviewer_full_run.log 2>&1 &
```

The output is valid only if the terminal log ends with
`single_realization_run_preflight_analysis=PASS` and the new post-run report has
`all_checks_pass: true`.

## Public release architecture

The curated repository will be published on GitHub. The working `main` branch
may be made public while manuscript metadata are completed, but the immutable
`v2.6.2` tag and release must be created only after the manuscript analysis and
the final reference/administrative fields pass the clean CI workflow. All
release assets should be attached to a draft before publication.
That exact release will then be archived in Zenodo to obtain a version DOI.

The persistent record should contain both:

1. the original production archive, byte-for-byte; and
2. the curated reviewer package with its full checksum manifest.

The Git tag, Git commit, container digest, Zenodo version DOI, and Zenodo concept
DOI must be recorded together after publication. Any correction must receive a
new version; the v2.6.2 tag and assets must not be replaced.

The author approved `BSD-3-Clause` for code and `CC-BY-4.0` for frozen and
derived scientific data on 2026-08-26. The manuscript-specific table/figure
analysis layer is now included and must remain CI-validated. Final references,
institutional/funding/conflict/contribution fields, and publication identifiers
remain release gates; identifiers must not be guessed or added as placeholders.
