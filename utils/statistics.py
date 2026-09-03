# mypy: ignore-errors
# /utils/statistics.py
"""
Module for statistical analysis of simulation results.
"""
import numpy as np
from scipy import stats
from itertools import combinations
from collections import defaultdict


def calculate_patient_wait_metrics(schedule_details):
    """Calcula la espera extra de cada paciente entre Op1 (anestesia) y Op2 (cirugía).

    La espera extra se define como el tiempo adicional que el paciente espera
    entre el fin de su anestesia y el inicio de su cirugía, descontando el
    tiempo de transición simulado (TransitionUsed). Un valor > 0 indica que
    el paciente quedó esperando por disponibilidad de sala o personal.

    Args:
        schedule_details: Lista de dicts con claves Job, Operation, Start,
                          Finish, TransitionUsed, etc.

    Returns:
        dict con:
            - ``per_patient``: lista de dicts con job_id, op1_room, op2_room,
              op1_finish, op2_start, transition_used, extra_wait_min
            - ``summary``: dict con total_patients, patients_with_extra_wait,
              avg_extra_wait_min, max_extra_wait_min
    """
    if not schedule_details:
        return {"per_patient": [], "summary": _empty_wait_summary()}

    ops_by_job = defaultdict(dict)
    for t in schedule_details:
        job = t.get("Job")
        op = t.get("Operation")
        if job is not None and op is not None:
            ops_by_job[job][op] = t

    per_patient = []
    for job_id, ops in sorted(ops_by_job.items()):
        if 1 not in ops or 2 not in ops:
            continue
        op1 = ops[1]
        op2 = ops[2]
        op1_finish = op1.get("Finish", 0)
        op2_start = op2.get("Start", 0)
        
        # El tiempo de transición real ocurre antes de la Op1 (Anestesia)
        transition_pre_op = op1.get("TransitionUsed", 0.0) or 0.0
        # No hay transición de paciente entre Op1 y Op2
        transition_inter_op = op2.get("TransitionUsed", 0.0) or 0.0
        
        extra_wait = max(0, (op2_start - op1_finish) - transition_inter_op)

        per_patient.append({
            "job_id": job_id,
            "op1_room": op1.get("Resource", ""),
            "op2_room": op2.get("Resource", ""),
            "op1_finish": round(op1_finish, 4),
            "op2_start": round(op2_start, 4),
            "transition_used": round(transition_pre_op, 4),
            "extra_wait_min": round(extra_wait, 4),
        })

    extra_waits = [p["extra_wait_min"] for p in per_patient if p["extra_wait_min"] > 0.01]
    summary = {
        "total_patients": len(per_patient),
        "patients_with_extra_wait": len(extra_waits),
        "avg_extra_wait_min": round(float(np.mean(extra_waits)), 4) if extra_waits else 0.0,
        "max_extra_wait_min": round(float(np.max(extra_waits)), 4) if extra_waits else 0.0,
    }

    return {"per_patient": per_patient, "summary": summary}


def _empty_wait_summary():
    return {
        "total_patients": 0,
        "patients_with_extra_wait": 0,
        "avg_extra_wait_min": 0.0,
        "max_extra_wait_min": 0.0,
    }


# ------------------------------------------------------------------
# Schedule quality metrics (aggregate per simulation)
# ------------------------------------------------------------------

STANDARD_SHIFT_MIN = 480.0  # 8-hour shift


def holm(pvals):
    """Return Holm step-down adjusted p-values in the original order."""
    pvals = np.asarray(pvals, dtype=float)
    idx = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, original_index in enumerate(idx):
        running = max(running, (m - rank) * pvals[original_index])
        adjusted[original_index] = min(running, 1.0)
    return adjusted


def rank_biserial(x, y):
    """Return the signed matched-pairs rank-biserial correlation.

    Zero differences are omitted and ties receive their average rank. Positive
    differences contribute to the positive rank sum, preserving the direction
    used by the published paired tables.
    """
    differences = np.asarray(x) - np.asarray(y)
    differences = differences[differences != 0]
    if len(differences) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(differences), method="average")
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    total = positive + negative
    return (positive - negative) / total if total > 0 else 0.0


def bootstrap_ci(x, y, stat_fn=rank_biserial, n_boot=2000, seed=42):
    """Return the percentile bootstrap confidence interval for a paired stat."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    values = []
    for _ in range(n_boot):
        indices = rng.choice(n, size=n, replace=True)
        values.append(stat_fn(x[indices], y[indices]))
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


def calculate_schedule_quality_metrics(schedule_details):
    """Calcula métricas de calidad operativa y clínica de un schedule.

    Métricas calculadas:
        1. **Room Overtime**: Minutos totales que los pabellones operan más allá
           de un turno estándar de 8 horas (480 min).
        2. **Workload Balance**: Desviación estándar del tiempo trabajado por
           cada cirujano/anestesiólogo activo. Valores altos → carga desigual.
        3. **Idle Gaps**: Huecos > 5 min en la agenda de un pabellón donde no
           se está operando. Cantidad y duración total.
        4. **Value-Added Ratio**: Porcentaje del tiempo total que se dedica a
           la cirugía real (proc_time) vs preparación (setup+transition+cleanup).

    Args:
        schedule_details: Lista de dicts con claves Job, Operation, Resource,
                          Personnel, Start, Finish, SetupUsed, TransitionUsed,
                          CleanupUsed, ProcessingEnd, etc.

    Returns:
        dict con claves:
            - rooms_used (int)
            - total_overtime_min (float): suma de overtime de todos los rooms
            - max_room_overtime_min (float): overtime del room más sobrecargado
            - personnel_count (int): personas únicas activas
            - workload_std_min (float): desviación estándar de carga
            - workload_max_min (float): máx minutos trabajados por una persona
            - workload_min_min (float): mín minutos trabajados por una persona
            - idle_gap_count (int): número de huecos > 5 min
            - idle_gap_total_min (float): suma de tiempo en huecos
            - avg_idle_gap_min (float): promedio de un hueco
            - value_added_ratio (float): 0.0 – 1.0
    """
    if not schedule_details:
        return _empty_quality_metrics()

    # ---- Agrupar operaciones por room y por personnel ----
    ops_by_room = defaultdict(list)
    ops_by_person = defaultdict(list)

    total_proc = 0.0
    total_setup = 0.0
    total_transition = 0.0
    total_cleanup = 0.0

    for t in schedule_details:
        resource = t.get("Resource", "")
        person = t.get("Personnel", "")
        start = t.get("Start", 0)
        finish = t.get("Finish", 0)

        if resource:
            ops_by_room[resource].append((start, finish))
        if person:
            ops_by_person[person].append((start, finish))

        # Tiempos para value-added ratio
        setup = t.get("SetupUsed", 0) or 0
        trans = t.get("TransitionUsed", 0) or 0
        clean = t.get("CleanupUsed", 0) or 0
        # proc_time = duración total - setup - transition - cleanup
        duration = finish - start
        proc_time = max(0, duration - setup - trans - clean)

        total_proc += proc_time
        total_setup += setup
        total_transition += trans
        total_cleanup += clean

    # ---- 1. Room Overtime ----
    room_overtimes = []
    for room, ops in ops_by_room.items():
        if not ops:
            continue
        room_end = max(f for _, f in ops)
        overtime = max(0, room_end - STANDARD_SHIFT_MIN)
        room_overtimes.append(overtime)

    total_overtime = sum(room_overtimes)
    max_room_overtime = max(room_overtimes) if room_overtimes else 0

    # ---- 2. Workload Balance ----
    person_workloads = []
    for person, ops in ops_by_person.items():
        total_work = sum(f - s for s, f in ops)
        person_workloads.append(total_work)

    if len(person_workloads) >= 2:
        workload_std = float(np.std(person_workloads, ddof=1))
    else:
        workload_std = 0.0
    workload_max = max(person_workloads) if person_workloads else 0
    workload_min = min(person_workloads) if person_workloads else 0

    # ---- 3. Idle Gaps ----
    IDLE_THRESHOLD = 5.0  # gaps > 5 min
    gap_count = 0
    gap_total = 0.0

    for room, ops in ops_by_room.items():
        sorted_ops = sorted(ops, key=lambda x: x[0])
        for i in range(1, len(sorted_ops)):
            gap = sorted_ops[i][0] - sorted_ops[i - 1][1]
            if gap > IDLE_THRESHOLD:
                gap_count += 1
                gap_total += gap

    avg_gap = gap_total / gap_count if gap_count > 0 else 0

    # ---- 4. Value-Added Ratio ----
    total_all = total_proc + total_setup + total_transition + total_cleanup
    va_ratio = total_proc / total_all if total_all > 0 else 0

    return {
        "rooms_used": len(ops_by_room),
        "total_overtime_min": round(total_overtime, 4),
        "max_room_overtime_min": round(max_room_overtime, 4),
        "personnel_count": len(ops_by_person),
        "workload_std_min": round(workload_std, 4),
        "workload_max_min": round(workload_max, 4),
        "workload_min_min": round(workload_min, 4),
        "idle_gap_count": gap_count,
        "idle_gap_total_min": round(gap_total, 4),
        "avg_idle_gap_min": round(avg_gap, 4),
        "value_added_ratio": round(va_ratio, 4),
    }


def _empty_quality_metrics():
    return {
        "rooms_used": 0,
        "total_overtime_min": 0.0,
        "max_room_overtime_min": 0.0,
        "personnel_count": 0,
        "workload_std_min": 0.0,
        "workload_max_min": 0.0,
        "workload_min_min": 0.0,
        "idle_gap_count": 0,
        "idle_gap_total_min": 0.0,
        "avg_idle_gap_min": 0.0,
        "value_added_ratio": 0.0,
    }

def calculate_room_kpis(schedule_details):
    """
    Calculates occupancy rate and related KPIs for each room.
    Occupancy = (Total processing time / Makespan) * 100
    """
    room_times = {}
    max_time = 0.0
    for task in schedule_details:
        room = task.get('Resource')
        if not room:
            continue
        
        start = task.get('Start', 0.0)
        finish = task.get('Finish', start)
        dur = max(0.0, finish - start)
        
        if room not in room_times:
            room_times[room] = 0.0
        room_times[room] += dur
        
        if finish > max_time:
            max_time = finish

    kpis = {}
    for room, utilized in room_times.items():
        kpis[room] = {
            'utilized_time': utilized,
            'occupancy_rate': (utilized / max_time * 100) if max_time > 0 else 0.0
        }
    
    if room_times:
        avg_rate = sum(r['occupancy_rate'] for r in kpis.values()) / len(kpis)
    else:
        avg_rate = 0.0
    
    kpis['Average'] = {'occupancy_rate': avg_rate}
    return kpis

def analyze_personnel_workload(schedule_details):
    """
    Analyzes workload for personnel dynamically extracted from schedule details.
    """
    personnel_stats = {}
    max_time = 0.0
    
    for task in schedule_details:
        person = task.get('Personnel')
        if not person:
            continue
        
        start = task.get('Start', 0.0)
        finish = task.get('Finish', start)
        dur = max(0.0, finish - start)
        
        if person not in personnel_stats:
            personnel_stats[person] = {
                'operations': 0,
                'total_time': 0.0,
                'first_task': start,
                'last_task': finish,
                'jobs': set()
            }
            
        stats = personnel_stats[person]
        if task.get('Job') not in stats['jobs']:
            stats['operations'] += 1
            stats['jobs'].add(task.get('Job'))
            
        stats['total_time'] += dur
        stats['first_task'] = min(stats['first_task'], start)
        stats['last_task'] = max(stats['last_task'], finish)
        
        if finish > max_time:
            max_time = finish
            
    # Calculate idle time and utilization
    for person, stats in personnel_stats.items():
        time_span = stats['last_task'] - stats['first_task']
        stats['idle_time'] = max(0.0, time_span - stats['total_time'])
        # Utilization relative to total simulation makespan
        stats['utilization'] = (stats['total_time'] / max_time * 100) if max_time > 0 else 0.0
        
    return personnel_stats

def _effect_magnitude_r(r):
    """Classifies effect size magnitude using Cohen (1988) thresholds for correlations."""
    ar = abs(r)
    if ar >= 0.5:
        return 'large'
    if ar >= 0.3:
        return 'medium'
    if ar >= 0.1:
        return 'small'
    return 'trivial'


def _holm_bonferroni(results, alpha):
    """Apply Holm-Bonferroni step-down correction to a list of result dicts.

    Mutates each result dict in-place:
      - adds 'p_adjusted' (Holm-corrected p-value)
      - updates 'is_significant' based on adjusted p
    """
    valid = [(i, r) for i, r in enumerate(results) if 'p_value' in r]
    if not valid:
        return

    # Sort by raw p_value ascending, then apply the shared vector correction.
    valid.sort(key=lambda x: x[1]['p_value'])
    adjusted = holm([res['p_value'] for _, res in valid])

    for (orig_i, res), p_adjusted in zip(valid, adjusted):
        res['p_adjusted'] = p_adjusted
        res['is_significant'] = p_adjusted < alpha


def perform_paired_statistical_test(all_results, alpha, verbose=True, metric='makespan'):
    """Correct statistical analysis for PAIRED samples.

    Since all algorithms are evaluated on the same simulation instances
    (same day_data per sim_i), the observations are paired. This function:

    1. Runs Friedman test (non-parametric omnibus for k >= 3 paired groups)
    2. Runs pairwise Wilcoxon signed-rank tests with Holm-Bonferroni
       correction for multiple comparisons
    3. Only declares 'better_algo' when the corrected p-value is significant

    The returned ``pairwise`` list has the same dict structure consumed by
    ``reporting.export_statistical_analysis``, so callers don't need changes
    beyond swapping the function name.

    Args:
        all_results: dict {algo_name: {metric: [float, ...]}}
            Each metric list must be aligned by sim index (same order).
        alpha: significance level (e.g. 0.05)
        verbose: print results to console
        metric: name of the per-sim series to test inside each algo dict.
            Defaults to 'makespan' for backward compatibility; pass e.g.
            'patients_with_extra_wait' to run the same paired test on a
            different operational metric. Lower is treated as better for the
            'better_algo' direction (matches makespan/waiting semantics).

    Returns:
        dict with:
          - 'friedman': dict with chi2, p_value, is_significant (or None)
          - 'pairwise': list of dicts compatible with export_statistical_analysis
          - 'n_valid': int, number of paired observations used
          - 'n_excluded': int, observations dropped (inf in any algo)
    """
    algo_keys = list(all_results.keys())

    # Build aligned arrays (same sim_i across all algorithms)
    arrays = {}
    for key in algo_keys:
        arrays[key] = np.asarray(all_results[key][metric], dtype=float)

    n_sims = min(len(a) for a in arrays.values())

    # Create mask: exclude sim_i where any algo has inf
    valid_mask = np.ones(n_sims, dtype=bool)
    for key in algo_keys:
        valid_mask &= np.isfinite(arrays[key][:n_sims])

    aligned = {key: arrays[key][:n_sims][valid_mask] for key in algo_keys}
    n_valid = int(valid_mask.sum())

    output = {
        'friedman': None,
        'pairwise': [],
        'n_valid': n_valid,
        'n_excluded': int(n_sims - n_valid),
    }

    if n_valid < 6:
        if verbose:
            print("Not enough valid paired observations for statistical analysis.")
        return output

    # --- Step 1: Friedman omnibus test (k >= 3 groups) ---
    if len(algo_keys) >= 3:
        aligned_arrays = [aligned[k] for k in algo_keys]
        chi2, p_friedman = stats.friedmanchisquare(*aligned_arrays)
        k = len(algo_keys)
        df_friedman = k - 1
        kendall_w = float(chi2 / (n_valid * df_friedman)) if (n_valid > 0 and df_friedman > 0) else 0.0
        output['friedman'] = {
            'chi2': float(chi2),
            'df': df_friedman,
            'p_value': float(p_friedman),
            'kendall_w': kendall_w,
            'is_significant': p_friedman < alpha,
            'k': k,
            'n': n_valid,
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Friedman Test (k={k}, n={n_valid}, df={df_friedman})")
            print(f"  Chi2 = {chi2:.4f}, df = {df_friedman}, p = {p_friedman:.6e}, Kendall W = {kendall_w:.4f}")
            if p_friedman < alpha:
                print(f"  -> Significant (p < {alpha}): proceeding to pairwise comparisons.")
            else:
                print(f"  -> Not significant (p >= {alpha}): no global differences detected.")
            print(f"{'=' * 60}")

    # --- Step 2: Pairwise Wilcoxon signed-rank tests ---
    pairwise_results = []

    for key_a, key_b in combinations(algo_keys, 2):
        data_a = aligned[key_a]
        data_b = aligned[key_b]

        res = {
            'algo_a': key_a, 'algo_b': key_b,
            'n_a': len(data_a), 'n_b': len(data_b),
            'mean_a': float(np.mean(data_a)),
            'std_a': float(np.std(data_a, ddof=1)),
            'mean_b': float(np.mean(data_b)),
            'std_b': float(np.std(data_b, ddof=1)),
        }

        diff = data_a - data_b

        # Wilcoxon can't handle all-zero differences
        if np.all(diff == 0):
            res.update({
                'w_stat': 0.0, 'p_value': 1.0, 'z_stat': 0.0,
                'effect_r': 0.0, 'effect_magnitude': 'trivial',
                'is_significant': False, 'better_algo': '',
            })
            pairwise_results.append(res)
            continue

        w_stat, p_value = stats.wilcoxon(data_a, data_b, alternative='two-sided')

        # Direction and effect size are derived from the signed ranks of the
        # paired differences. We report the matched-pairs rank-biserial
        # correlation (Kerby, 2014): r = |R+ - R-| / (R+ + R-). It is bounded
        # in [0, 1], is the recommended effect size for the Wilcoxon
        # signed-rank test, and---unlike the r = Z / sqrt(n) approximation
        # derived from the p-value---does not saturate when p is astronomically
        # small (which previously capped every strong effect at r = 0.410).
        # Positive differences mean A > B (makespan A > makespan B, so B is better).
        non_zero_diff = diff[diff != 0.0]
        if len(non_zero_diff) == 0:
            r_effect = 0.0
            z_stat = 0.0
            better = ''
        else:
            ranks = stats.rankdata(np.abs(non_zero_diff))
            pos_rank_sum = float(np.sum(ranks[non_zero_diff > 0.0]))
            neg_rank_sum = float(np.sum(ranks[non_zero_diff < 0.0]))
            total_rank = pos_rank_sum + neg_rank_sum
            r_effect = abs(pos_rank_sum - neg_rank_sum) / total_rank if total_rank > 0 else 0.0
            # Normal approximation of the Wilcoxon Z from the signed-rank
            # statistic, retained only as a reference column in the exports.
            n_nz = len(non_zero_diff)
            mean_w = n_nz * (n_nz + 1) / 4.0
            sd_w = np.sqrt(n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24.0)
            t_stat = min(pos_rank_sum, neg_rank_sum)
            z_stat = float(abs((t_stat - mean_w) / sd_w)) if sd_w > 0 else 0.0
            better = key_b if pos_rank_sum > neg_rank_sum else key_a

        res.update({
            'w_stat': float(w_stat),
            'p_value': float(p_value),
            'z_stat': z_stat,
            'effect_r': float(r_effect),
            'effect_magnitude': _effect_magnitude_r(r_effect),
            'is_significant': p_value < alpha,  # updated by Holm below
            'better_algo': better,
        })
        pairwise_results.append(res)

    # --- Step 3: Holm-Bonferroni correction ---
    _holm_bonferroni(pairwise_results, alpha)

    # Clear better_algo when not significant after correction
    for res in pairwise_results:
        if not res['is_significant']:
            res['better_algo'] = ''

    output['pairwise'] = pairwise_results

    if verbose:
        for res in pairwise_results:
            p_adj = res.get('p_adjusted', res['p_value'])
            print(f"\n--- {res['algo_a']} vs {res['algo_b']} ---")
            print(f"  Mean: {res['mean_a']:.2f} ± {res['std_a']:.2f}  vs  {res['mean_b']:.2f} ± {res['std_b']:.2f}")
            print(f"  Wilcoxon W = {res['w_stat']:.1f}, p_raw = {res['p_value']:.6f}, p_adj(Holm) = {p_adj:.6f}")
            print(f"  Effect: r = {res['effect_r']:.3f} ({res['effect_magnitude']})")
            if res['is_significant']:
                print(f"  -> {res['better_algo']} is significantly better (after Holm-Bonferroni)")
            else:
                print("  -> No significant difference (after Holm-Bonferroni)")

    return output


def compute_operational_summary(
    all_results,
    metrics=('patients_with_extra_wait', 'avg_extra_wait_min', 'final_makespan'),
):
    """Representative PAIRED aggregates of operational metrics per algorithm.

    Unlike ``best_runs_by_mh`` (which cherry-picks each algorithm's single
    best simulation by ``combined_obj``), this computes representative paired
    aggregates across ALL simulations, so paper tables reflect typical
    performance rather than a best-case run.

    Args:
        all_results: dict {algo_name: {metric_name: [values aligned by sim index]}}.
            All series for a given algo (and across algos) must share the same
            sim-index ordering so the makespan ranking below is paired.
        metrics: iterable of metric names to aggregate (mean and sample std).

    Returns:
        dict {algo_name: {
            '<metric>_mean': float, '<metric>_sd': float, ...,
            'win_rate_pct': float,   # paired makespan win-rate (%)
            'mean_rank': float,      # mean paired makespan rank (1 = best)
            'n': int,                # number of sims aggregated for this algo
        }}
        Pure function, no I/O.
    """
    algo_keys = list(all_results.keys())
    summary = {key: {} for key in algo_keys}

    # --- Per-algo mean and sample std (ddof=1) for each requested metric ---
    for key in algo_keys:
        for metric in metrics:
            values = np.asarray(all_results[key].get(metric, []), dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                summary[key][f'{metric}_mean'] = float('nan')
                summary[key][f'{metric}_sd'] = float('nan')
            else:
                summary[key][f'{metric}_mean'] = float(np.mean(finite))
                summary[key][f'{metric}_sd'] = (
                    float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                )
        summary[key]['n'] = int(
            np.asarray(all_results[key].get('final_makespan', []), dtype=float).size
        )

    # --- Paired makespan win-rate and mean rank ---
    makespan_arrays = {
        key: np.asarray(all_results[key].get('final_makespan', []), dtype=float)
        for key in algo_keys
    }
    if algo_keys and all(a.size for a in makespan_arrays.values()):
        n_sims = min(a.size for a in makespan_arrays.values())
        # Only rank sims where every algo has a finite makespan (paired).
        valid_mask = np.ones(n_sims, dtype=bool)
        for key in algo_keys:
            valid_mask &= np.isfinite(makespan_arrays[key][:n_sims])

        rank_sums = {key: 0.0 for key in algo_keys}
        win_counts = {key: 0 for key in algo_keys}
        n_ranked = 0

        for sim_i in range(n_sims):
            if not valid_mask[sim_i]:
                continue
            values = np.array(
                [makespan_arrays[key][sim_i] for key in algo_keys], dtype=float
            )
            # Lower is better: rank ascending, ties share average rank.
            ranks = stats.rankdata(values, method='average')
            for pos, key in enumerate(algo_keys):
                rank_sums[key] += ranks[pos]
                if ranks[pos] == 1:  # ties count as wins for all tied
                    win_counts[key] += 1
            n_ranked += 1

        for key in algo_keys:
            if n_ranked > 0:
                summary[key]['win_rate_pct'] = 100.0 * win_counts[key] / n_ranked
                summary[key]['mean_rank'] = rank_sums[key] / n_ranked
            else:
                summary[key]['win_rate_pct'] = float('nan')
                summary[key]['mean_rank'] = float('nan')
    else:
        for key in algo_keys:
            summary[key]['win_rate_pct'] = float('nan')
            summary[key]['mean_rank'] = float('nan')

    return summary


# ---------------------------------------------------------------------------
# Legacy wrapper — keeps old API surface for backward compatibility
# ---------------------------------------------------------------------------

def perform_u_test_mannwhitney(all_results, alpha, verbose=True):
    """Delegates to the corrected paired test (Friedman + Wilcoxon).

    Returns only the ``pairwise`` list so existing callers that expect a flat
    list of dicts continue to work unchanged.
    """
    result = perform_paired_statistical_test(all_results, alpha, verbose=verbose)
    return result['pairwise']
