# Manuscript analysis layer for spinal CPG v2.6.2

This directory contains the read-only post-run analysis for the frozen
single-realization experiment. The scripts read the archived result files;
they neither modify nor rerun the simulator.

## Protocol provenance and reviewer interface

`ANALYSIS_PROTOCOL_LOCKED_2026-08-26.md` is the original Turkish-language
analysis-lock record. Its reference to manuscript snapshot `v1_8` identifies
the document used when the analysis contract was frozen; later language and
layout revisions do not redefine endpoints, contrasts, pairing rules, or
figures. `PROTOCOL_SPEC.json` and this README provide the machine-readable and
English reviewer interfaces. Scientific identity is enforced by the frozen
source hashes, expected counts, and semantic reference-output checks rather
than by the current manuscript filename.

## Scientific scope

The central analysis asks whether the detailed closed-loop locomotor network
maintains left-right and flexor-extensor phase organization across:

1. descending-drive command and mechanical load;
2. excitatory and inhibitory perturbations and subsequent rhythmic recovery;
3. single and paired interventions in the tested model populations/pathways;
4. presynaptic route disruptions and population-by-route dependencies; and
5. terminal-activity-dependent phenomenological replenishment support,
   including temporal, spatial, and KCa controls.

The source field named `speed` is the low/medium/high **descending-drive
command**, not measured locomotor speed. Measured network frequency is a
separate endpoint and need not vary monotonically with that command.

The MT-associated variable is a phenomenological terminal-support mechanism.
The experiment tests its network consequences; it does not establish
microtubule biology. The ten previously locked contrasts are retained as a
separate preregistration capsule and are not a global pass/fail score for the
model.

All results are conditional on one stochastic network realization
(`seed=601`, `structural_seed=160601`). Contexts, tasks, routes, model
populations, cycles, and transfer events are not independent replicates. No
p-values, confidence intervals, standard errors, degrees of freedom, or
across-seed population claims are produced. Rhythmic failure, pulse delivery,
recovery eligibility, recovery events, and observed/censored times remain
separate fields.

## Contents

- `ANALYSIS_PROTOCOL_LOCKED_2026-08-26.md`: human-readable locked protocol.
- `PROTOCOL_SPEC.json`: machine-readable axes, endpoints, pairing keys,
  source hashes, and expected counts.
- `scripts/validate_sources_and_protocol.py`: frozen-source and design checks.
- `scripts/analyze_a_to_d.py`: intact network, perturbations, circuit
  interventions, and descending-drive recruitment.
- `scripts/analyze_e_f.py`: route impairment and population-by-route analyses.
- `scripts/analyze_g_h.py`: composite-stress and terminal-support/KCa controls.
- `scripts/audit_derived_outputs.py`: cross-stage provenance and content audit.
- `scripts/build_publication_outputs.py`: Tables R1-R6, panel data, and
  Figures 4-8.
- `scripts/verify_reference_outputs.py`: semantic regression against the
  curated publication snapshot.
- `reference_outputs/`: tracked tables, panel data, figures, captions, and
  compact provenance. These files are a reviewer-facing snapshot, not inputs
  to the scientific calculations.
- `MANIFEST.sha256`: checksum list for this curated analysis package.

## Complete rerun

Run from the repository root after installing
`requirements-reviewer-lock.txt`:

```bash
python3 manuscript_analysis_v2_6_2/scripts/validate_sources_and_protocol.py
python3 manuscript_analysis_v2_6_2/scripts/analyze_a_to_d.py
python3 manuscript_analysis_v2_6_2/scripts/analyze_e_f.py
python3 manuscript_analysis_v2_6_2/scripts/analyze_g_h.py
python3 manuscript_analysis_v2_6_2/scripts/audit_derived_outputs.py
python3 manuscript_analysis_v2_6_2/scripts/build_publication_outputs.py
python3 manuscript_analysis_v2_6_2/scripts/verify_reference_outputs.py
```

All regenerated files are written below the ignored directory
`derived/manuscript_analysis_v2_6_2/`; tracked references are never
overwritten. The final command ignores only generation timestamps and PDF
container metadata. It compares all scientific CSV fields and rows, captions,
decoded PNG pixels, PDF page geometry/text, the locked contrast capsule, and
all QC PASS states.

Expected terminal markers include:

```text
all_checks_pass=True
manuscript_reference_output_verification=PASS
independent_stochastic_realizations=1
```
