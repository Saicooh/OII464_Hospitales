# Course synthetic instances

The repository runs the historical four-algorithm comparison—GA, dPSO, SBOA,
and dMShOA—against one selected synthetic YAML instance. Operational rooms,
personnel, timings, eligibility, and job count are owned by that YAML instance;
the global configuration selects the input, solver budgets, simulation count,
parallelism, and output locations.

An optional `MH` slot is available for student metaheuristics. It is disabled by
default and uses the same typed solver contract and report outputs as the base
algorithms when enabled.

## Run

```text
python main.py
```

The default input is `instances/hospital_12rooms/HOSP-12R-15-01.yaml`. To select
a different configuration without editing source code, set
`HOSPITAL_CONFIG_PATH` to a supported YAML configuration, or set
`HOSPITAL_INSTANCE_PATH` for a one-off instance override.

Each execution writes the historical report contract under `results/elective/`:
summary, room overtime, statistical, schedule, sequencing, and routing CSV
tables plus comparison, histogram, Gantt, personnel, KPI, and convergence
plots. The typed `results/<timestamp>/run<N>/` writer remains available for
integrations. Existing local result directories are not removed by repository
maintenance.

## Catalog

The catalog contains 37 validated synthetic instances across didactic, standard,
and twelve-room teaching families. Every entry has a canonical digest, explicit
resource eligibility, generation seed, validation evidence, and intentionally
varied timing/load parameters. Entries are classified as fully synthetic; bounds
remain marked pending or heuristic until independent verification is available.

The loader rejects malformed documents, unsupported schema versions, inconsistent
dimensions, invalid resource references, incomplete evidence, and digest
mismatches before any parallel work begins.
