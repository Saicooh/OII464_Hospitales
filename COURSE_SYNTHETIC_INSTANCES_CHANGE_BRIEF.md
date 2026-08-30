# Course-ready synthetic instances change brief

_Pre-SDD scope capture for removing obsolete execution modes, checkpoint behavior, selected outputs, and dependencies on non-shareable real data._

---

> 📌 **Status:** This document is planning input only. No SDD phase has started, and no implementation decision beyond the user's stated requirements is considered final.

## 🎯 Objective

Prepare the repository for use in a course by removing features and outputs that are no longer part of the intended workflow, eliminating dependencies on real hospital data, and replacing those inputs with readable synthetic instances stored in YAML.

The future design must be derived from what the repository actually consumes. It must not reproduce the previous JSON schema because that schema was incomplete.

## 📋 Confirmed scope

| Area | Required outcome |
| --- | --- |
| **Sweep mode** | Remove sweep mode from the repository, including its executable paths and dedicated support surfaces |
| **Analysis mode distinction** | Remove `analysis mode` as a separate mode, flag, configuration branch, or workflow distinction; promote its current behavior to the repository's normal and single execution behavior while preserving the analysis capabilities themselves |
| **Algorithm implementations** | Rename `dmshoa_old` to `dmshoa`, retain it as the sole implementation, and remove the other implementations, including their selectors, configuration branches, tests, fixtures, and documentation |
| **Iteration CSV files** | Stop generating files matching `analysis_algorithm_iterations_run*.csv` |
| **Checkpoints** | Remove checkpoint creation, persistence, resume, recovery, and checkpoint-specific orchestration logic |
| **Replay workflows** | Remove replay execution logic, `replay days`, replay builders and loaders, replay-specific artifacts, configuration, tests, and documentation |
| **Retained plots** | Move plots that remain useful after checkpoint removal into the run-scoped `plots/` output area |
| **Run separation** | Preserve the current behavior that separates plot outputs by run |
| **Convergence plots** | Remove generation of `convergence_by_time_run*.png` and `convergence_combined_run*.png` rather than moving them |
| **Real data** | Remove runtime logic and repository dependencies that require the non-shareable real dataset |
| **Real-data PKL** | Remove the real-data `.pkl` artifact from the repository and remove every associated loader, path, parser, transformation, fixture, test, configuration entry, and documentation reference |
| **Synthetic instances** | Replace real-data inputs with course-ready YAML instances and metadata |
| **Static resource configuration** | Remove repository-wide or hardcoded resource and personnel values such as `resources.num_pabellones`, `personnel.num_anesthesiologists`, and `personnel.num_surgeons`; operational values must be defined by the selected YAML instance |
| **Static job and timing configuration** | Remove repository-wide or hardcoded `jobs.types` and `times.setup`, `times.cleanup`, and `times.max_wait` values from global configuration; retain them as per-instance data when required by `dmshoa` or other supported consumers |
| **Obsolete YAML configuration** | Audit and remove any obsolete section, field, default, schema entry, or documentation reference from configuration YAML files after the synthetic-instance migration; retain only settings still required by the general instance-driven runtime |
| **Migration order** | Before deleting the real-data `.pkl`, derive the required input structure from it, build and validate the synthetic YAML instances, and ensure they use different, intentionally varied configurations; delete the PKL only after the replacement is usable |

Examples of outputs that the new workflow must no longer generate:

- `results/20260830_011709/analysis_algorithm_iterations_run4.csv`
- `results/20260830_011709/convergence/convergence_by_time_run3.png`
- `results/20260830_011709/convergence/convergence_combined_run3.png`

## 📦 Target instance layout

The intended starting layout is:

```text
instances/
├── metadata.yaml
├── didactic/
│   ├── HOSP-DIDACT-03-01.yaml
│   ├── HOSP-DIDACT-05-01.yaml
│   ├── HOSP-DIDACT-08-01.yaml
│   └── HOSP-DIDACT-10-01.yaml
├── standard/
│   ├── HOSP-STD-15-01.yaml
│   ├── HOSP-STD-20-01.yaml
│   ├── HOSP-STD-25-01.yaml
│   └── HOSP-STD-30-01.yaml
└── hospital_12rooms/
    ├── HOSP-12R-15-01.yaml
    ├── HOSP-12R-20-01.yaml
    ├── HOSP-12R-25-01.yaml
    ├── HOSP-12R-30-01.yaml
    ├── HOSP-12R-40-01.yaml
    ├── HOSP-12R-50-01.yaml
    └── HOSP-12R-60-01.yaml
```

The `-01` suffix identifies the first replica position within an instance family. It does not, by itself, establish that the file is a replica of a real case. The final number of replicas for each size remains an investigation outcome because multi-seed and multi-instance comparisons may require more than one.

## ⚙️ YAML and metadata requirements

The YAML instance schema must be derived from the complete set of fields, constraints, and relationships consumed by the current supported runtime after obsolete paths are removed.

The real-data PKL may be used temporarily as a source for understanding the consumed structure and constraints. It must not be copied into the synthetic instances. Before the PKL is deleted, the replacement YAML instances must be generated and validated against the supported runtime using different configurations from the original data, while preserving the valid structural and scheduling requirements needed by the course.

Resource and personnel concepts are not removed from the domain. Their operational values must move from repository-wide or hardcoded configuration into each instance YAML, so the selected instance determines values such as the number of operating rooms, anesthesiologists, and surgeons. `metadata.yaml` may describe the resource metadata required by the catalog, but it must not reintroduce one global operational configuration for all instances.

Job and timing concepts are treated the same way. The global configuration must no longer own `jobs.types` or the `times.setup`, `times.cleanup`, and `times.max_wait` mappings. If those fields are required by the retained `dmshoa` implementation or by the complete runtime input contract, they must be represented in the selected instance YAML and allowed to vary between synthetic configurations. Removing them from global configuration must not remove required per-instance input data.

Once the instance-driven workflow is established, all remaining configuration must be audited for dead or obsolete entries, not only settings related to `dmshoa`. This includes YAML sections and fields made unnecessary by removing sweep, analysis-mode branching, checkpoints, replay, real-data loading, alternative implementations, and global resource, job, or timing values. The cleanup must preserve only configuration still needed by the general runtime to select, load, validate, and execute a synthetic instance.

`instances/metadata.yaml` must record, at minimum:

- Schema version
- Provenance
- Generation method
- Seed
- Dimensions
- Resources
- Instance classification
- Verified bounds

A bound must not be recorded as fact while its validity remains pending. Pending, heuristic, estimated, or unverified values must be represented honestly or omitted according to the schema established during design.

The instance format must prioritize readability and safe modification by students. The future specification must define validation behavior for malformed YAML, missing required fields, unsupported schema versions, and inconsistent dimensions or resources.

## 🔍 Academic and reproducibility basis

The project will investigate how to construct reproducible and academically defensible instances for this specific scheduling model. The investigation must resolve:

- Duration distributions and their correlations
- Assignment of specialties, surgeons, and anesthesiologists
- Room eligibility rules
- Congestion levels
- Number of replicas required for each size
- Seed policy
- Statistical and structural validations
- Conditions for classifying an instance as a `replica`, `calibrated`, or `fully synthetic` instance

An instance must not be described as a replica of the real case until sufficient evidence supports that classification. When the evidence is insufficient, the required designation is `calibrated synthetic instance`. The metadata and documentation must make the classification and its supporting evidence explicit.

## 🔍 Required investigation before design

The SDD exploration phase must answer these questions before a proposal, specification, or implementation is treated as ready:

1. Where sweep mode enters the system through CLI options, configuration, orchestration, tests, documentation, and output handling
2. Where `analysis mode` is selected or distinguished through CLI options, configuration, orchestration, code branches, tests, and documentation
3. Which behavior is currently exclusive to `analysis mode` and must become the normal execution path without removing the underlying analysis capability
4. Which implementation is currently identified as `dmshoa_old`, how it must be renamed to `dmshoa`, what its complete input and output contract is, and where the other implementations are referenced
5. Which producers and consumers create or depend on iteration CSV files
6. Which behaviors are true checkpoint semantics and which are ordinary reporting or plotting behavior that must remain
7. Which replay entry points, `replay days` builders, loaders, artifacts, configuration, tests, and documentation must be removed
8. Which checkpoint-associated plots must move to `plots/`, excluding the two convergence plot families marked for removal
9. Which loaders, transformations, schemas, fixtures, tests, and documentation depend on real data
10. Which real-data `.pkl` file is present, where it is referenced, and which associated logic must be removed with it
11. Which structural fields and constraints must be extracted from the PKL before its removal, without retaining private values
12. Which synthetic configurations should differ from the real instances while remaining valid for the supported runtime
13. Which resource and personnel values are currently global or hardcoded, where they are consumed, and how they must be represented per instance
14. Which job and timing values are currently global or hardcoded, where they are consumed, and which must move into each instance YAML
15. Which configuration YAML sections, fields, defaults, schema entries, and documentation references become obsolete after the instance migration and have no remaining consumers, independently of any particular implementation
16. What complete data shape the supported algorithms and simulation runtime consume
17. Which synthetic generation rules preserve valid scheduling constraints without encoding or reconstructing private source data
18. How many replicas each size requires for meaningful multi-seed and multi-instance teaching experiments
19. Which bounds can be verified, how they are verified, and how pending bounds are represented
20. What evidence is sufficient for the labels `replica`, `calibrated`, and `fully synthetic`, and how the fallback `calibrated synthetic instance` designation is recorded
21. Whether already-generated historical files under `results/` should be deleted or only prevented from being generated again

## ✅ Acceptance criteria for the future change

- Sweep mode has no executable, configurable, documented, or tested product path remaining
- `analysis mode` is no longer a distinct selectable or branching mode, and the former analysis-mode behavior is the normal repository behavior
- Analysis functionality itself remains available through that normal behavior, except for outputs and logic explicitly removed by this brief
- `dmshoa` is the sole retained algorithm implementation, the former `dmshoa_old` name is absent from supported code and documentation, and no supported selector or residual implementation path exists for the removed alternatives
- A fresh supported run does not produce `analysis_algorithm_iterations_run*.csv`
- No supported execution path creates, loads, resumes from, or manages checkpoints
- No supported execution path creates, loads, runs, or manages replay workflows or `replay days`
- Retained plots previously coupled to checkpoints are written under the run-scoped `plots/` area while preserving run separation
- A fresh supported run does not produce `convergence_by_time_run*.png` or `convergence_combined_run*.png`
- Supported workflows require no private or real hospital dataset
- The real-data `.pkl` artifact is absent from the repository, and no associated executable, configurable, tested, or documented logic remains
- Synthetic YAML instances contain every field required by supported consumers and no fields copied merely from the incomplete legacy JSON schema
- Synthetic YAML instances were constructed and validated before PKL deletion, are loadable without the PKL, and cover intentionally different configurations
- Instance generation is supported by documented duration distributions and correlations, assignment rules, room eligibility, congestion levels, replica counts, seed policy, and statistical or structural validation evidence
- Instance labels follow the evidence policy: `replica` is used only when sufficient evidence supports similarity to the real case; otherwise the instance is documented as a `calibrated synthetic instance` or `fully synthetic` according to the established criteria
- Operational resource and personnel values are supplied by the selected instance YAML, with no repository-wide hardcoded values such as `resources.num_pabellones` or the quoted personnel counts
- Operational job and timing values are supplied by the selected instance YAML when required, with no global `jobs.types` or `times` configuration controlling all instances
- No obsolete configuration entry, default, schema field, or documentation reference remains without a consumer in the general instance-driven workflow
- Instance validation fails clearly when required structure or constraints are invalid
- `metadata.yaml` records the required provenance and generation information without presenting unverified bounds as established facts
- Tests, examples, and documentation use synthetic instances only
- No private values, source records, paths, or reconstructable real-data artifacts remain in the course distribution

## 🚫 Explicit non-actions at this stage

- Do not implement repository changes yet
- Do not start SDD exploration, proposal, specification, design, tasks, apply, verify, or archive phases yet
- Do not invent the YAML schema before tracing current consumers
- Do not remove analysis capabilities when removing the separate `analysis mode` distinction
- Do not remove the `dmshoa` implementation or required job and timing fields while removing alternative implementations and global configuration; rename the former `dmshoa_old` implementation without losing its behavior and preserve it in the final instance-driven contract
- Do not remove resource or personnel concepts; remove only their global or hardcoded configuration and represent them per instance
- Do not delete configuration merely because it is not visibly used; trace consumers first and preserve settings required by the general instance-driven runtime for instance selection, loading, validation, execution, or output routing
- Do not preserve replay or `replay days` as a supported compatibility path
- Do not delete the real-data `.pkl` before the synthetic YAML replacement has been constructed, validated, and shown to work without it
- Do not fix the final replica count before investigation
- Do not mark any bound as verified without evidence
- Do not remove unrelated `.pkl` artifacts unless exploration confirms that they are part of the real-data path
- Do not delete historical generated results until their cleanup scope is explicitly decided

## ✍️ Next step

When authorized, begin with the SDD session preflight and repository exploration. Use this document as the scope input, not as a substitute for the proposal, specifications, design, or task plan.
