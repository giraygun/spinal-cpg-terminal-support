# Reproducibility workflow

## Supported reviewer levels

1. **Artifact verification** checks frozen source and result hashes, complete
   curated-package coverage, release contracts, counts, licenses/metadata, and
   the recorded post-run PASS state without simulation.
2. **Technical execution** runs the seven contract tests, four focused numerical
   tests, a matrix dry run, and one reduced non-scientific simulation.
3. **Analysis reproduction** copies the frozen output tree to a new directory,
   re-runs strict preflight and the locked ten-contrast analyzer, and compares
   regenerated outputs byte-for-byte with the archived originals.
4. **Full production rerun** is optional and historically required about 101.39
   hours with six workers.

The first three levels are the normal reviewer workflow.

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

The expected descriptive classification is 5 favorable, 3 unfavorable, and
2 neutral contrasts. This classification is conditional on the frozen
realization and is not a statistical acceptance or rejection of the biological
hypothesis.

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
may be made public while reviewer-facing analysis files are completed, but the
immutable `v2.6.2` tag and release must be created only after all manuscript
table/figure builders and statistical-summary scripts pass the clean CI
workflow. All release assets should be attached to a draft before publication.
That exact release will then be archived in Zenodo to obtain a version DOI.

The persistent record should contain both:

1. the original production archive, byte-for-byte; and
2. the curated reviewer package with its full checksum manifest.

The Git tag, Git commit, container digest, Zenodo version DOI, and Zenodo concept
DOI must be recorded together after publication. Any correction must receive a
new version; the v2.6.2 tag and assets must not be replaced.

The author approved `BSD-3-Clause` for code and `CC-BY-4.0` for frozen and
derived scientific data on 2026-08-26. The remaining hard block on the
immutable GitHub release and Zenodo publication is completion and clean
validation of the manuscript-specific table/figure analysis layer. Publication
identifiers must not be guessed or added as placeholders.
