# CPG v2.6.2 frozen single-realization reviewer package

This repository contains the frozen spinal locomotor-network model, the full
A-H experimental design, the v2.6.2 single-realization execution overlay, and
the complete outputs used for the accompanying manuscript on terminal-local,
MT-associated slow vesicle replenishment.

**Status:** this is a validated prepublication package. The frozen simulator and
completed outputs are final. The read-only manuscript analysis, tables, panel
data, and figures are included under `manuscript_analysis_v2_6_2/`; they do not
alter the frozen core or outputs. No final public tag or archival release has
been issued yet.

Development and reviewer repository:
<https://github.com/giraygun/spinal-cpg-terminal-support>. The repository may
remain private until the public release is approved; an immutable tagged
release and archival DOI will supersede the moving `main` branch for citation.

## Scientific identity

The version labels identify different layers:

- **v2.6.1 frozen scientific core**: model equations, biological parameters,
  A-H task definitions, numerical engine, and the locked endpoint definitions.
- **v2.6.2 release overlay**: fixes the production design to noise seed `601`
  and structural seed `160601`, deduplicates identical simulator inputs, checks
  the completed checkpoint set, and reports the locked descriptive contrasts.

The v2.6.2 overlay did not change the model equations or biological parameters.
See [VERSION_MAP.md](VERSION_MAP.md) for the exact file-level mapping.

Do not edit any file listed in `RELEASE_CONTENTS_v2_6_2.sha256`. Additional
reviewer analyses must write to `derived/` or another new directory.

## Experimental matrix

| Stage | Scope | Tasks |
|---|---|---:|
| A | Intact multi-context scan | 27 |
| B | Single circuit interruptions | 270 |
| C | Prespecified paired interruptions | 162 |
| D | Descending-drive recruitment control | 54 |
| E | Route-specific phenomenological terminal-support impairment | 270 |
| F | Model-population x route x mechanism factorial | 10,800 |
| G | Composite neural, synaptic, and mechanical stress | 31 |
| H | KCa x phenomenological terminal-support controls | 72 |
| **Total** |  | **11,686** |

Tasks with identical complete simulator inputs are stored once. This leaves
3,610 unique simulations and avoids 8,076 identical recomputations without
changing any comparison.

## Inferential scope

This is one frozen stochastic network realization. Stages, routes, model
populations, contexts, pulses, and task rows are not independent replicates.
The package therefore does not authorize p-values, confidence intervals,
standard errors, degrees of freedom, or across-seed population inference.
Results are conditional mechanistic observations within seed `601` and
structural seed `160601`.

## Retained-data scope

The repository contains the complete dataset retained by the production
workflow: the task matrix, one checkpoint per unique simulation, event and
intervention records, epoch summaries, derived metrics, analysis tables, and
figure data. Full continuous membrane-potential, spike-train, or other
timestep-level time series were not retained during the production run and are
therefore not part of this archive. References to "complete outputs" in this
repository mean the complete retained analysis record, not unrecorded
continuous trajectories.

## Quick start

CPython 3.12.x is required. The frozen simulator uses NumPy 2.3.5, SciPy
1.17.0, and Matplotlib 3.10.8. The reviewer lock also includes the read-only
publication-analysis dependencies.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-reviewer-lock.txt
```

### 1. Verify the artifact

```bash
python3 reviewer_verify.py
```

This checks the 31 frozen release files, all 3,619 frozen result files, every
file in the curated-package manifest, the complete 3,610-checkpoint set, the
task counts, seeds, licenses/metadata, and the recorded post-run PASS state.

### 2. Run contract and numerical tests

```bash
python3 -B -m unittest -v test_single_realization_v2_6_2.py
python3 -B test_numerics_architecture_v2_6_1.py
```

Expected results are 7/7 release-contract tests and 4/4 focused numerical and
executable-microcircuit tests passing.

### 3. Run a reduced technical smoke test

```bash
bash reviewer_smoke.sh derived/reviewer_smoke
```

The numerical part executes one reduced, non-scientific Stage A simulation.
It completed in 11.23 seconds in the validated Linux environment. Its output
must never be interpreted as scientific evidence.

### 4. Regenerate the locked analysis safely

The original analyzers write into the supplied result directory. Use the
reviewer wrapper, which first verifies the frozen artifact, copies the result
tree to a new writable directory, re-runs preflight and analysis there, and
compares the regenerated files with the archived originals:

```bash
python3 reviewer_reproduce_analysis.py derived/reanalysis_v2_6_2
```

### 5. Rebuild the manuscript analysis and figures

This pipeline reads the frozen results in place and writes only below
`derived/manuscript_analysis_v2_6_2/`:

```bash
python3 manuscript_analysis_v2_6_2/scripts/validate_sources_and_protocol.py
python3 manuscript_analysis_v2_6_2/scripts/analyze_a_to_d.py
python3 manuscript_analysis_v2_6_2/scripts/analyze_e_f.py
python3 manuscript_analysis_v2_6_2/scripts/analyze_g_h.py
python3 manuscript_analysis_v2_6_2/scripts/audit_derived_outputs.py
python3 manuscript_analysis_v2_6_2/scripts/build_publication_outputs.py
python3 manuscript_analysis_v2_6_2/scripts/verify_reference_outputs.py
```

The final check compares regenerated scientific content with the curated
Tables R1-R6, panel-data CSVs, and Figures 4-8 while ignoring generation times
and PDF container metadata. See
[`manuscript_analysis_v2_6_2/README.md`](manuscript_analysis_v2_6_2/README.md)
for the scientific scope and interpretation limits.

## Full production rerun

Reviewers do not need to repeat the complete calculation. The completed
six-worker production run took 365,008.99 seconds, approximately 101.39 hours.
If an independent full rerun is specifically required, use a new empty result
directory and a new log file:

```bash
nohup env CPG_WORKERS=6 \
  ./run_mac_single_realization_v2_6_2.sh reviewer_full_run \
  > reviewer_full_run.log 2>&1 &
```

Do not use the default result or log names: those names already contain the
archived production evidence and would trigger resume-skip behavior or overwrite
the historical log.

## Contents

- `single_realization_results_v2_6_2/`: all frozen scientific outputs.
- `manuscript_analysis_v2_6_2/`: locked read-only analysis scripts, protocol,
  curated tables, panel data, figures, captions, and compact provenance.
- `RESULTS_CONTENTS_v2_6_2.sha256`: hashes for all 3,619 result files.
- `RELEASE_CONTENTS_v2_6_2.sha256`: hashes for the 31 frozen source, release,
  validation, and execution files.
- `REVIEWER_PACKAGE_CONTENTS_v2_6_2.sha256`: pre-publication checksum manifest
  for the complete curated reviewer tree.
- `DATA_DICTIONARY.md`: file and field semantics.
- `REPRODUCIBILITY.md`: complete reviewer and publication workflow.
- `SINGLE_REALIZATION_PREREGISTRATION_v2_6_2_TR.md`: original preregistration.
- `run_single_realization_v2_6_2.log`: historical run log with only the local
  workstation path redacted; the byte-identical raw log remains in the preserved
  original archive.

The original production archive is preserved separately, byte-for-byte, with
SHA-256 `2c90e852304fed10ea37702ad39f2555569ea4b2cc033a67c449fddf01bc4b7f`.

## Licensing

Copyright (c) 2026 Giray Güneş.

- Source code, shell scripts, workflow files, container definitions, and
  software documentation in this repository are licensed under the
  **BSD 3-Clause License** (`BSD-3-Clause`); see [LICENSE](LICENSE).
- Frozen CSV/JSON scientific outputs under
  `single_realization_results_v2_6_2/`, the historical production log, outputs
  written under `derived/`, and any subsequently published tables, figures, or
  derived data explicitly identified as part of this release are
  licensed under the **Creative Commons Attribution 4.0 International
  License** (`CC-BY-4.0`); see [LICENSE-DATA](LICENSE-DATA).

These grants apply only to material for which Giray Güneş holds the necessary
rights. Names and licenses of third-party dependencies are not changed by this
repository's licenses.
