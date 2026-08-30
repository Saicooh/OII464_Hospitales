# Reusable validation and replay tooling

Scripts use paths relative to the repository, so they can be run from any
working directory with the project virtual environment:

```
.venv/Scripts/python.exe reproducibility/<script>.py
```

## Retained components

| Component | Purpose | Inputs and outputs |
|---|---|---|
| `instance_reconstruction.py` | Read-only reconstruction of persisted run instances, resource bounds, and provenance records | `results/<run>/analysis.db` → validated instance data and JSON records |
| `diagnostics/check_fixed_instance.py` | Validate fixed per-run data and report drift | An analysis database; no output files are modified |
| `diagnostics/validate_pairing.py` | Validate aligned algorithm observations and workload consistency | Run summary CSVs; no output files are modified |
| `build_replay_days.py` | Build deterministic real-day inputs from the raw hospital dataset | Raw PKL → `datasets/replay_days/replay_days.pkl` and its summary CSV |
| `../replay_run.py` | Execute the optimizer on the persisted real-day inputs | `datasets/replay_days/replay_days.pkl` → replay result CSVs |
| `analyze_replay.py` | Analyze replay results by day, scale, algorithm, and manual reference | `datasets/replay_days/replay_results.csv` → `reproducibility/output/replay_per_day.csv` and console diagnostics |
| `../offline/analysis_exporter.py` | Export persisted iteration schedules and strategies | Analysis database → CSV exports beside the database |
| `../offline/convergence_analysis.py` | Inspect convergence of completed runs | Analysis database → convergence plots beside the database |
| `../offline/generate_analysis_plots.py` | Generate standard makespan comparison plots | Analysis database → plots beside the database |
| `../offline/generate_advanced_plots.py` | Generate operational and scalability plots | Analysis database → plots beside the database |

## Diagnostics

Use `--strict` when a validation failure should produce a non-zero exit code:

```
.venv/Scripts/python.exe reproducibility/diagnostics/check_fixed_instance.py --strict
.venv/Scripts/python.exe reproducibility/diagnostics/validate_pairing.py --strict
```

Both validators open persisted data read-only and do not regenerate or rewrite
results.

## Real-day replay workflow

Phase A builds the deterministic day set. Phase B runs the optimizer; `--smoke`
checks one day and one seed before a full run. Phase C performs the retained
paired analysis.

Importing `build_replay_days.py` only exposes its builder and constants; it does
not build or write replay outputs. Run it explicitly to execute Phase A.

```
.venv/Scripts/python.exe reproducibility/build_replay_days.py
.venv/Scripts/python.exe replay_run.py --smoke
.venv/Scripts/python.exe replay_run.py --seeds 5
.venv/Scripts/python.exe reproducibility/analyze_replay.py
```

Replay instances and result CSVs remain under `datasets/replay_days/`.
