# Data dictionary

The frozen CSV/JSON outputs described here are licensed under CC BY 4.0; see
`LICENSE-DATA`. This license statement does not turn task rows, contexts,
routes, or model populations into independent biological replicates.

The production workflow retained task definitions, per-simulation checkpoints,
event and intervention records, epoch summaries, and derived metrics. It did
not retain full continuous membrane-potential, spike-train, or other
timestep-level time series. Accordingly, this dictionary describes the
complete retained analysis dataset rather than continuous trajectories that
were never stored.

## General conventions

- Empty CSV fields mean not applicable or not observed for that task; they are
  not automatically zero.
- Angles and phase errors are in degrees; frequencies are in hertz; latency is
  in milliseconds; recovery and protocol times are in seconds.
- Boolean validity and failure fields must be analyzed separately from
  continuous performance fields.
- `seed=601` and `structural_seed=160601` identify the single frozen realization.
- Route or population lists in CSV files use `+` as the separator; `none` means
  that no member of that axis was intervened upon.
- `mt_*` and `mean_mt_support_*` are terminal-local, MT-associated
  phenomenological support variables. They are not direct measurements or a
  claim that the model proves microtubule biology.
- `rrp_*` summarizes the readily releasable-pool state. Fields containing
  `replenishment_resource` summarize the modeled slow terminal replenishment
  resource.

## Files

### `analysis_task_index.csv`

One row per prespecified analysis task (11,686 rows). `task_id` is the analysis
identity; `simulation_id` links the task to one of 3,610 unique simulator calls;
`reuse_count` records how many analysis tasks use that identical simulation.
The remaining fields encode stage, protocol, speed, load, pulse, ablation,
terminal-support mode, impaired route, challenged route, KCa/fast mode, and
human-readable label.

### `unique_simulation_metrics.csv`

One row per unique simulation (3,610 rows). It contains source/config hashes,
execution identity, the full intervention context, primary observable metrics,
validity/failure fields, population rates, burst counts, and secondary
terminal-state summaries.

### `metrics.csv`

One row per analysis task (11,686 rows). Simulation metrics are joined back to
the task matrix, so rows sharing a `simulation_id` are deterministic reuse, not
independent replicates. Major field groups are:

- rhythm: `frequency_*`, `rg_cycle_interval_cv_mean`, burst counts;
- coordination: `lr_*`, `fe_*`, phase-slip counts/rates;
- motor balance: `bilateral_amplitude_balance` and `_imbalance`;
- propagation: PF/MN anchor, matched, missed, reliability, and latency fields;
- perturbation/recovery: pulse/sham timing, eligibility, event, censoring, and
  recovery-time fields;
- validity/failure: `scientific_valid`, `technical_valid`, exclusion reason,
  and `rhythmic_failure`;
- network participation: class-specific mean firing-rate fields;
- secondary state indicators: class-specific MT, RRP, and replenishment fields.

Legacy convenience fields at the end of the file are retained for provenance.
Manuscript analyses should use the explicitly named L-R and F-E endpoints rather
than replacing them with a single global phase metric.

### `long_epoch_metrics.csv`

Stage G long-protocol summaries (744 rows) indexed by `simulation_id` and epoch.
It contains epoch-specific rhythm, phase, transfer counts, failure/validity, and
route-level terminal-state summaries. Raw anchor/missed/matched counts must be
kept distinct from transformed burden measures.

### `single_realization_contrasts_v2_6_2.csv`

The ten prespecified descriptive contrast families. Negative values were the
locked favorable direction. `zero_is_neutral` distinguishes exact neutral
contrasts. These ten rows are not independent statistical observations.

### `single_realization_results_v2_6_2.json`

Machine-readable metadata and the same ten family results, including explicit
single-realization and non-inferential interpretation limits.

### `simulations/sim-*.json`

One checkpoint per unique simulation. Each record contains:

- `simulation_id` and `simulation_identity`;
- `representative_task_id`;
- release/config `manifest`;
- `summary` and `analysis_observables`;
- `analysis_events` and `intervention_log`;
- `scientific_valid`;
- an internal `checkpoint_payload_sha256` integrity value.

The post-run preflight checks the exact expected checkpoint set and its internal
payload identities.
