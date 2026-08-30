"""
Phase C of the real-day replay experiment: paired analysis.

Consumes datasets/replay_days/replay_results.csv (produced by ../replay_run.py)
and answers two questions on the 120 real days, each algorithm evaluated on the
same day with the day's actual room count:

  1) OPTIMIZATION vs. MANUAL: per day, compare the optimized makespan (mean over
     seeds) against the realized (manual) makespan. Reported per algorithm and
     for the best-of-4 (a planner running all methods and keeping the best),
     with a paired Wilcoxon signed-rank test and a 95% bootstrap CI of the
     median reduction. This is the head-to-head baseline (rooms matched, so the
     room confounder is removed).

  2) ALGORITHM vs. ALGORITHM: paired across the 120 real days (per-day mean over
     seeds), Friedman + Wilcoxon/Holm + matched-pairs rank-biserial, on makespan
     and on patient waiting -- the clean multi-instance head-to-head that also
     removes the single-instance-per-size limitation of the synthetic study.

Outputs a console report and reproducibility/output/replay_per_day.csv.

Run (after the full replay finishes):
  python reproducibility/analyze_replay.py
  python reproducibility/analyze_replay.py --file datasets/replay_days/replay_results.csv
"""
from pathlib import Path
import sys, argparse
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
from utils.statistics import perform_paired_statistical_test

ALGOS = ["GA", "SBOA", "dPSO", "dMShOA"]
rng = np.random.default_rng(42)


def boot_ci_median(x, B=2000):
    x = np.asarray(x, float); n = len(x)
    bs = np.array([np.median(x[rng.integers(0, n, n)]) for _ in range(B)])
    return np.median(x), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main(path):
    if not Path(path).exists():
        print(f"[!] {path} not found. Run the replay first (see reproducibility/RUN_REPLAY.md):")
        print("    python reproducibility/build_replay_days.py")
        print("    python replay_run.py --seeds 5")
        return

    df = pd.read_csv(path)
    df = df[np.isfinite(df["opt_makespan"])].copy()
    df["patients_with_extra_wait"] = pd.to_numeric(df["patients_with_extra_wait"], errors="coerce")

    # per (day, algo): mean optimized makespan and mean waiting over seeds
    g = df.groupby(["day", "bucket", "room_count", "realized_makespan", "algo"])
    per = g.agg(opt_makespan=("opt_makespan", "mean"),
                wait=("patients_with_extra_wait", "mean")).reset_index()
    mk = per.pivot_table(index=["day", "bucket", "realized_makespan"], columns="algo", values="opt_makespan")
    wt = per.pivot_table(index=["day", "bucket", "realized_makespan"], columns="algo", values="wait")
    mk = mk.dropna(subset=[a for a in ALGOS if a in mk.columns])
    algos = [a for a in ALGOS if a in mk.columns]
    ndays = len(mk)
    print(f"Loaded {path}: {ndays} days, algorithms={algos}\n")
    if ndays < 6:
        print("[!] Fewer than 6 usable days -> paired tests need the full replay. "
              "Structure OK; rerun after the full run.")

    realized = mk.index.get_level_values("realized_makespan").to_numpy(float)
    buckets = mk.index.get_level_values("bucket").to_numpy(int)
    best = mk[algos].min(axis=1).to_numpy(float)

    # save tidy per-day table
    out = mk.reset_index().copy(); out["best_opt"] = best
    outdir = REPO / "reproducibility" / "output"; outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "replay_per_day.csv", index=False)

    # ---- 1) optimization vs manual ----
    print("=" * 70)
    print("1) OPTIMIZATION vs MANUAL (per day, rooms matched)")
    def report_vs_manual(opt, label):
        red = realized - opt                      # minutes saved
        redpct = 100 * red / realized
        med, lo, hi = boot_ci_median(redpct)
        try:
            _, p = wilcoxon(realized, opt)
        except ValueError:
            p = float("nan")
        print(f"  {label:<10} median reduction = {med:5.1f}% "
              f"[95% CI {lo:.1f}, {hi:.1f}] | median {np.median(red):5.1f} min | "
              f"Wilcoxon p={p:.2e}")
    for a in algos:
        report_vs_manual(mk[a].to_numpy(float), a)
    report_vs_manual(best, "best-of-4")
    print("\n  By size bucket (best-of-4):")
    for size in [15, 20, 25, 30]:
        m = buckets == size
        if m.sum() == 0:
            continue
        redpct = 100 * (realized[m] - best[m]) / realized[m]
        print(f"    N~{size}: days={m.sum():2d} | realized median={np.median(realized[m]):6.1f} "
              f"| best-opt median={np.median(best[m]):6.1f} | reduction={np.median(redpct):5.1f}%")

    if ndays < 6:
        return

    # ---- 2) algorithm vs algorithm (paired across days) ----
    print("\n" + "=" * 70)
    print("2) ALGORITHM vs ALGORITHM (paired across the real days)")
    for metric_name, piv in [("makespan", mk), ("patient waiting", wt)]:
        piv = piv.dropna(subset=algos)
        allr = {a: {"v": piv[a].tolist()} for a in algos}
        res = perform_paired_statistical_test(allr, 0.05, verbose=False, metric="v")
        fr = res["friedman"]
        print(f"\n  [{metric_name}] Friedman chi2={fr['chi2']:.1f} p={fr['p_value']:.2e} (n={fr['n']} days)")
        means = {a: np.mean(piv[a]) for a in algos}
        print("   mean per day: " + " | ".join(f"{a}={means[a]:.1f}" for a in algos))
        for r in res["pairwise"]:
            if not r["better_algo"]:
                continue
            print(f"    {r['algo_a']:<7} vs {r['algo_b']:<7} p_holm={r.get('p_adjusted', r['p_value']):.4f} "
                  f"r={r['effect_r']:.3f}({r['effect_magnitude'][0].upper()}) better={r['better_algo']}")
    print(f"\nWrote {outdir / 'replay_per_day.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(REPO / "datasets" / "replay_days" / "replay_results.csv"))
    a = ap.parse_args()
    main(a.file)
