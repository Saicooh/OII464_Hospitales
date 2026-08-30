"""
Real-day replay experiment (standalone, non-invasive).

Parent mode: groups the 120 pre-built real days by their actual room count and
launches one subprocess per room-count value with HOSPITAL_CONFIG_PATH pointing
to a temp config whose resources.num_pabellones equals that room count. This is
required because room lists are bind-by-value module globals.

Worker mode: for its assigned room count, runs the 4 metaheuristics x N seeds on
each day, reusing the static scheduler exactly like ElectiveWorker, and records the
optimized makespan and patient-waiting alongside the day's realized (manual)
makespan.

Usage:
  python replay_run.py                      # full run (all room counts, 10 seeds)
  python replay_run.py --smoke              # 1 room-count, 1 day, 1 seed (validate)
  python replay_run.py --worker --days <pkl> --rc <int> --out <csv> --seeds <int>
"""
import argparse, os, sys, pickle, subprocess, tempfile, time
import yaml
import pandas as pd

PROJ = os.path.dirname(os.path.abspath(__file__))
REPLAY_DIR = os.path.join(PROJ, "datasets", "replay_days")
DAYS_PKL = os.path.join(REPLAY_DIR, "replay_days.pkl")


def run_worker(days_pkl, room_count, out_csv, n_seeds, n_jobs):
    sys.path.insert(0, PROJ)
    from joblib import Parallel, delayed
    from simulation.scheduler import run_static_schedule
    from utils.statistics import calculate_patient_wait_metrics
    from algorithms.ga import run as run_ga
    from algorithms.dpso import run as run_dpso
    from algorithms.sboa import run as run_sboa
    from algorithms.dmshoa_old import run as run_dmshoa_old
    from config.config import PABELLONES

    print(f"[worker rc={room_count}] loaded {len(PABELLONES)} rooms: {PABELLONES[:3]}...", flush=True)
    specs = [("GA", run_ga), ("dPSO", run_dpso), ("SBOA", run_sboa), ("dMShOA", run_dmshoa_old)]

    with open(days_pkl, "rb") as f:
        days = pickle.load(f)

    tasks = []
    for d in days:
        for algo_name, runner in specs:
            for seed in range(n_seeds):
                tasks.append((d, algo_name, runner, seed))

    def one(task):
        d, algo_name, runner, seed = task
        job_ids = list(range(1, d["n"] + 1))
        t0 = time.time()
        try:
            schedule_details, makespan, _bh, _ah = run_static_schedule(
                algorithm_runner=runner,
                surgeries_data=d["surgeries_data"],
                job_ids=job_ids,
                seed=seed,
            )
            wsum = calculate_patient_wait_metrics(schedule_details)["summary"]
            mk = makespan if schedule_details else float("inf")
            pw = wsum["patients_with_extra_wait"]; aw = wsum["avg_extra_wait_min"]
        except Exception as e:
            mk, pw, aw = float("inf"), -1, -1.0
            print(f"  ERR {algo_name} day={d['day']} seed={seed}: {e}", flush=True)
        return {"day": d["day"], "bucket": d["bucket"], "n": d["n"],
                "room_count": d["room_count"], "realized_makespan": d["realized_makespan"],
                "algo": algo_name, "seed": seed, "opt_makespan": mk,
                "patients_with_extra_wait": pw, "avg_extra_wait_min": aw,
                "cpu_s": round(time.time() - t0, 2)}

    resolved_jobs = n_jobs if (n_jobs and n_jobs > 0) else (os.cpu_count() or 4)
    print(f"[worker rc={room_count}] {len(tasks)} tasks on {resolved_jobs} workers", flush=True)
    rows = Parallel(n_jobs=resolved_jobs)(delayed(one)(t) for t in tasks)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[worker rc={room_count}] wrote {out_csv}", flush=True)


def run_parent(smoke, n_seeds, n_jobs):
    with open(DAYS_PKL, "rb") as f:
        days = pickle.load(f)
    base_cfg_path = os.path.join(PROJ, "config", "config.yaml")
    with open(base_cfg_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    by_rc = {}
    for d in days:
        by_rc.setdefault(d["room_count"], []).append(d)

    rcs = sorted(by_rc)
    if smoke:
        rc0 = rcs[len(rcs) // 2]  # a mid room count
        by_rc = {rc0: by_rc[rc0][:1]}
        rcs = [rc0]
        n_seeds = 1
        print(f"[SMOKE] rc={rc0}, 1 day, 1 seed")

    tmpdir = tempfile.mkdtemp(prefix="replay_")
    partials = []
    for rc in rcs:
        grp = by_rc[rc]
        cfg = dict(base_cfg)
        cfg["resources"] = dict(base_cfg["resources"]); cfg["resources"]["num_pabellones"] = rc
        cfg_path = os.path.join(tmpdir, f"config_rc{rc}.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
        days_path = os.path.join(tmpdir, f"days_rc{rc}.pkl")
        with open(days_path, "wb") as f:
            pickle.dump(grp, f)
        out_csv = os.path.join(tmpdir, f"partial_rc{rc}.csv")
        env = dict(os.environ); env["HOSPITAL_CONFIG_PATH"] = cfg_path
        print(f"\n=== rc={rc}: {len(grp)} days x 4 algos x {n_seeds} seeds ===", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, os.path.abspath(__file__), "--worker",
                        "--days", days_path, "--rc", str(rc), "--out", out_csv,
                        "--seeds", str(n_seeds), "--jobs", str(n_jobs)], env=env, check=True)
        print(f"=== rc={rc} done in {time.time()-t0:.0f}s ===", flush=True)
        partials.append(out_csv)

    allrows = pd.concat([pd.read_csv(p) for p in partials], ignore_index=True)
    out = os.path.join(REPLAY_DIR, "replay_results_smoke.csv" if smoke else "replay_results.csv")
    allrows.to_csv(out, index=False)
    print(f"\nWROTE {out}  ({len(allrows)} rows)")
    print(allrows.groupby("algo")["opt_makespan"].agg(["count", "mean"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--days"); ap.add_argument("--rc", type=int)
    ap.add_argument("--out"); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers; -1 = all logical cores")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.worker:
        run_worker(a.days, a.rc, a.out, a.seeds, a.jobs)
    else:
        run_parent(a.smoke, a.seeds, a.jobs)
