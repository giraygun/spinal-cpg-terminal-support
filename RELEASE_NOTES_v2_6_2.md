# v2.6.2 frozen single-realization spinal CPG release

This release preserves the frozen v2.6.1 spinal locomotor CPG model core and
the v2.6.2 single-realization execution and analysis layer used for the
accompanying manuscript.

## Scientific scope

- Fixed noise seed: `601`
- Fixed structural seed: `160601`
- Prespecified analysis tasks: `11,686`
- Unique simulations: `3,610`
- Independent stochastic network realizations: `1`
- Full production runtime: approximately `101.39` hours with six workers

The results are conditional mechanistic observations within one frozen network
realization. Task rows, contexts, routes, populations, cycles, and transfer
events are not independent replicates; this release does not authorize
across-seed population inference.

## Included material

- Frozen source, release contracts, task definitions, and execution scripts
- Complete retained checkpoint, event, intervention, epoch-summary, and
  derived-metric outputs
- Read-only A-H manuscript analysis scripts
- Tables R1-R6, panel-data CSVs, Figures 4-8, captions, and provenance records
- Full package, frozen-source, and frozen-result SHA-256 manifests
- Reviewer verification, safe reanalysis, numerical tests, smoke test, and
  pinned-container workflow

Full continuous membrane-potential, spike-train, and other timestep-level time
series were not retained during production and are not part of this release.

## Provenance and privacy

The byte-identical original workstation archive is privately retained under
SHA-256
`2c90e852304fed10ea37702ad39f2555569ea4b2cc033a67c449fddf01bc4b7f`.
It is not published because it contains local tar ownership, path, AppleDouble,
and compiled-bytecode metadata. The curated public release retains every
scientifically relevant original file; the only textual sanitization replaces
the workstation path in the historical log with `<LOCAL_PATH>`.

## Licenses

- Software and software documentation: BSD 3-Clause (`BSD-3-Clause`)
- Frozen and explicitly published derived scientific data: Creative Commons
  Attribution 4.0 International (`CC-BY-4.0`)

See `README.md`, `REPRODUCIBILITY.md`, `LICENSE`, and `LICENSE-DATA` for the
complete reviewer workflow, interpretation limits, and license boundaries.
