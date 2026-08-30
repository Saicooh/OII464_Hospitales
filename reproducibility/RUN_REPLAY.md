# Running the real-day replay on another machine

The replay only needs the **code** and the **raw PKL**
(`datasets/2_dataset_procesado_actualizado.pkl`, which is tracked in git).
The `results/` folder is git-ignored and is **not** required for the replay.

Phase A is deterministic (`seed=42`), so building the day set on any machine from
the same raw PKL produces the identical 120 days.

---

## 0. Get the code onto the other machine

On **this** machine, commit and push the new files:

```powershell
git add replay_run.py reproducibility/ datasets/replay_days/
git commit -m "Add real-day replay experiment (phases A/B) and reproducibility scripts"
git push
```

On the **other** machine:

```powershell
git clone <repo-url>        # or: git pull
cd Hospitales
```

## 1. Environment (Python 3.10+; 3.11–3.13 all fine)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # Linux/macOS
pip install -r requirements.txt
```

## 2. Verify the raw data is present

```powershell
python -c "import os; print(os.path.exists('datasets/2_dataset_procesado_actualizado.pkl'))"
# must print True
```

## 3. Phase A — build the 120 real-day instances (deterministic)

```powershell
.venv\Scripts\python.exe reproducibility\build_replay_days.py
```

Creates `datasets/replay_days/replay_days.pkl` (+ summary CSV). Skip this step if
that file already came over with the repo.

## 4. Smoke test (fast — validates the pipeline end to end)

```powershell
.venv\Scripts\python.exe replay_run.py --smoke
```

Should finish in well under a minute and print a small table. If it errors, stop
and fix the environment before the full run.

## 5. Full run

```powershell
.venv\Scripts\python.exe replay_run.py --seeds 5     # ~1 hour (recommended)
# .venv\Scripts\python.exe replay_run.py --seeds 10  # ~1.5-3 hours
```

- Runs one subprocess per distinct room count (rooms are matched to each real day).
- Parallelism auto-scales to the machine (`min(18, cpu_count-2)` workers).
- Output: `datasets/replay_days/replay_results.csv` with one row per
  `(day, algorithm, seed)`: `opt_makespan`, `patients_with_extra_wait`,
  `avg_extra_wait_min`, `realized_makespan`, `room_count`, `bucket`, `n`, `cpu_s`.

## 6. Bring the result back

Commit/copy **`datasets/replay_days/replay_results.csv`** back so Phase C
(paired analysis: optimized vs. manual, and algorithm-vs-algorithm across the
120 real days) can be run:

```powershell
git add datasets/replay_days/replay_results.csv
git commit -m "Real-day replay results (Phase B)"
git push
```

## Validation boundaries

The retained files under `reproducibility/` provide replay entry points,
read-only instance reconstruction, and diagnostics. The scheduling source of
truth is in `simulation/`, while shared statistical helpers are in `utils/`.
The diagnostics support an explicit `--strict` mode when a CI or audit run must
fail rather than merely print a warning:

```powershell
.venv\Scripts\python.exe reproducibility\diagnostics\check_fixed_instance.py --strict
.venv\Scripts\python.exe reproducibility\diagnostics\validate_pairing.py --strict
```

The replay workflow does not modify the source dataset or the persisted
analysis database. Its generated instances and result CSVs are kept under
`datasets/replay_days/`.

## Notes / knobs

- `--seeds N` controls seeds per (day, algorithm). 5 is enough to stabilise the
  per-day mean; the statistical power comes from the 120 paired days.
- To resume after interruption, re-run; it recomputes from scratch (no partial
  checkpointing yet). For very slow machines, reduce `--seeds`.
- The metaheuristic iteration budget (1000) comes from `config/config.yaml`,
  unchanged from the main experiment.
