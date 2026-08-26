# Version map

The version labels refer to distinct layers and do not indicate post hoc changes
to the biological model.

| Layer | Identity | Primary files | Meaning |
|---|---|---|---|
| Biological model core | `distributed-local-terminal-mt-cpg-2.6.1-candidate` | `dual_timescale_spinal_cpg_v2_6_1_candidate.py` | Frozen equations and parameters used in production. |
| Model SHA-256 | `a0dc8a7338ab1619874135b1a3e8809f4eaa22394cb65dfd951544df5b62f47a` | Release and experiment-plan JSON files | Immutable model identity. |
| Pre-production validation | v2.6.1 | `FREEZE_MANIFEST_v2_6_1.json` and validation JSON files | Calibration, construct, convergence, stress, reproducibility, and shard checks. |
| Historical full-plan design | v2.6.1 | `FREEZE_MANIFEST_v2_6_1.json` | Earlier 245,256-task/83,796-simulation production plan; not the completed dataset. |
| Single-realization contract | v2.6.2 | `SINGLE_REALIZATION_RELEASE_v2_6_2.json` and preregistration | Reduces only the independent seed axis while retaining A-H. |
| Execution overlay | `single-realization-runner-2.6.2` | `run_single_realization_v2_6_2.py` | Runs seed `601` and structural seed `160601`. |
| Endpoint contract | v2.6.1 locked analyzer | `analyze_primary_v2_6_1.py` | Defines the ten prespecified endpoint contrasts used unchanged in v2.6.2. |
| Post-run audit | `single-realization-preflight-2.6.2` | preflight code and JSON | Verifies release, task, plan, index, and checkpoint completeness. |
| Result analyzer | `single-realization-mechanistic-analysis-2.6.2` | analysis code and result JSON | Produces ten descriptive contrasts without inferential statistics. |
| Frozen outputs | v2.6.2 | `single_realization_results_v2_6_2/` | 11,686 task rows represented by 3,610 unique simulations. |

`FREEZE_MANIFEST_v2_6_1.json` is a historical pre-production authorization
record; its `scientific_results_included: false` field describes that manifest,
not the later complete-run archive. The completed v2.6.2 output identity is
defined by `SINGLE_REALIZATION_RELEASE_v2_6_2.json`, the experiment plan,
completion record, post-run preflight, and `RESULTS_CONTENTS_v2_6_2.sha256`.

The software license is `BSD-3-Clause`; frozen and explicitly published
derived scientific data use `CC-BY-4.0`. Publication identifiers will be added
only after the manuscript analysis layer, reviewed repository, container, and
persistent archive are finalized. No placeholder DOI or repository URL is part
of the frozen identity.
