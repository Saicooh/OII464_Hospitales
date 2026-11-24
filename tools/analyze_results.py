"""Análisis estadístico y figuras de la comparación final.

Uso (desde la raíz del repositorio):

    python tools/analyze_results.py --runs results/final --out results/final/analisis

Sigue el protocolo del informe 2: mediana por variante e instancia, prueba de
Friedman sobre las 25 instancias y, si se rechaza la igualdad, Wilcoxon con
corrección de Holm en dos familias de contrastes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

LABELS = {
    "base": "A1",
    "fija": "A2",
    "aleatoria": "A3",
    "reglas": "A4",
    "aprende_p": "A5",
    "aprende_mov": "A6",
    "dos_niveles": "A7",
}
ORDER = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
FAMILY_1 = [("A7", other) for other in ORDER if other != "A7"]
FAMILY_2 = [("A2", "A1"), ("A5", "A2"), ("A5", "A3"), ("A5", "A4"), ("A6", "A2")]


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def contrasts(medians: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        x, y = medians[a].to_numpy(), medians[b].to_numpy()
        diff = x - y
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)
        rel = (x - y) / np.abs(y) * 100
        rows.append(
            {
                "contraste": f"{a} vs {b}",
                "dif_rel_mediana_pct": float(np.median(rel)),
                "gana": int((diff < -1e-9).sum()),
                "empata": int((np.abs(diff) <= 1e-9).sum()),
                "pierde": int((diff > 1e-9).sum()),
                "p": p,
            }
        )
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].tolist())
    return table


def analyse(runs: pd.DataFrame, metric: str) -> dict:
    medians = runs.pivot_table(index="instance", columns="A", values=metric, aggfunc="median")[ORDER]
    ranks = medians.rank(axis=1)
    friedman = stats.friedmanchisquare(*[medians[c] for c in ORDER])
    return {
        "medians": medians,
        "mean_rank": ranks.mean(),
        "friedman_stat": float(friedman.statistic),
        "friedman_p": float(friedman.pvalue),
        "family1": contrasts(medians, FAMILY_1),
        "family2": contrasts(medians, FAMILY_2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/final")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    source = Path(args.runs)
    out = Path(args.out) if args.out else source / "analisis"
    out.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(source / "runs.csv")
    runs["A"] = runs["method"].map(LABELS)
    runs = runs[runs["A"].notna()]
    runs["escala"] = np.where(runs["jobs"] >= 40, "40-60", "15-30")
    failed = int((~np.isfinite(runs["fitness"])).sum())

    report: dict = {"corridas": len(runs), "fallidas": failed, "instancias": runs["instance"].nunique()}
    lines: list[str] = []

    for metric in ("fitness", "makespan"):
        result = analyse(runs, metric)
        lines.append(f"\n===== {metric} =====")
        lines.append(f"Friedman chi2={result['friedman_stat']:.3f} p={result['friedman_p']:.3g}")
        lines.append("Rango promedio:\n" + result["mean_rank"].round(2).to_string())
        lines.append("Familia 1 (A7 frente a las demás):\n" + result["family1"].round(4).to_string(index=False))
        lines.append("Familia 2 (preguntas P1 a P3):\n" + result["family2"].round(4).to_string(index=False))
        report[metric] = {
            "friedman_stat": result["friedman_stat"],
            "friedman_p": result["friedman_p"],
            "mean_rank": result["mean_rank"].to_dict(),
            "family1": result["family1"].to_dict(orient="records"),
            "family2": result["family2"].to_dict(orient="records"),
        }
        result["medians"].to_csv(out / f"medianas_{metric}.csv")

        for scale, group in runs.groupby("escala"):
            if group["instance"].nunique() < 3:
                continue
            sub = analyse(group, metric)
            lines.append(f"-- {metric}, escala {scale} ({group['instance'].nunique()} instancias): "
                         f"Friedman p={sub['friedman_p']:.3g}")
            lines.append("   rango promedio: " + ", ".join(f"{k}={v:.2f}" for k, v in sub["mean_rank"].items()))
            fam = pd.concat([sub["family1"], sub["family2"]])
            lines.append(fam[["contraste", "dif_rel_mediana_pct", "gana", "empata", "pierde", "p"]].round(4).to_string(index=False))
            report[metric][f"escala_{scale}"] = {
                "friedman_p": sub["friedman_p"],
                "mean_rank": sub["mean_rank"].to_dict(),
            }

    # Estabilidad: rango intercuartílico relativo por variante
    grouped = runs.groupby(["instance", "A"])["fitness"]
    iqr = ((grouped.quantile(0.75) - grouped.quantile(0.25)) / grouped.median()).unstack()[ORDER]
    lines.append("\nIQR relativo medio por variante:\n" + (iqr.mean() * 100).round(2).to_string())
    report["iqr_rel_pct"] = (iqr.mean() * 100).to_dict()

    # Tiempo, esperas y mutaciones
    summary = runs.groupby("A")[["seconds", "mutations", "total_wait", "max_wait", "makespan"]].median().loc[ORDER]
    lines.append("\nMedianas globales por variante:\n" + summary.round(2).to_string())
    report["resumen"] = summary.to_dict()
    time_rel = runs.pivot_table(index=["instance", "seed"], columns="A", values="seconds")
    time_ratio = (time_rel[ORDER].div(time_rel["A1"], axis=0)).median()
    lines.append("\nTiempo relativo a A1 (mediana de cocientes):\n" + time_ratio.round(3).to_string())
    report["tiempo_relativo"] = time_ratio.to_dict()

    # Uso de movimientos en A6 y A7
    moves = ["intercambio", "insercion", "sala"]
    for name in ("A2", "A6", "A7"):
        group = runs[runs["A"] == name]
        counts = group[[f"n_{m}" for m in moves]].sum()
        share = (counts / counts.sum() * 100).round(1)
        lines.append(f"Uso de movimientos {name} (%): " + ", ".join(f"{m}={s}" for m, s in zip(moves, share)))
        report[f"movimientos_{name}"] = dict(zip(moves, share.tolist()))

    # Intensidad elegida por A5, A7 y A4 a lo largo de la corrida
    blocks_path = source / "blocks.csv"
    if blocks_path.exists():
        blocks = pd.read_csv(blocks_path)
        blocks["A"] = blocks["method"].map(LABELS)
        blocks["bloque"] = blocks.groupby(["instance", "A", "seed"]).cumcount()
        blocks["tercio"] = pd.cut(blocks["bloque"], [-1, 22, 44, 1000], labels=["inicio", "medio", "final"])
        share = (
            blocks[blocks["A"].isin(["A4", "A5", "A7"])]
            .groupby(["A", "tercio"], observed=True)["p"]
            .value_counts(normalize=True)
            .unstack()
            .mul(100)
            .round(1)
        )
        lines.append("\nFrecuencia de p por tercio de la corrida (%):\n" + share.to_string())
        report["p_por_tercio"] = {f"{a}|{t}": row.to_dict() for (a, t), row in share.iterrows()}
        improving = blocks.groupby("A")["reward"].apply(lambda r: (r > 1e-6).mean() * 100)
        lines.append("Bloques con mejora (%):\n" + improving.round(1).to_string())
        report["bloques_con_mejora_pct"] = improving.to_dict()

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
        colors = {0.05: "#c6dbef", 0.15: "#6baed6", 0.3: "#08519c"}
        for ax, name in zip(axes, ("A5", "A7")):
            sub = blocks[(blocks["A"] == name) & (blocks["bloque"] < 66)]
            freq = sub.groupby("bloque")["p"].value_counts(normalize=True).unstack().fillna(0)
            freq = freq.rolling(5, min_periods=1).mean()
            bottom = np.zeros(len(freq))
            for p in sorted(freq.columns):
                ax.fill_between(freq.index, bottom, bottom + freq[p], color=colors.get(round(p, 2), "grey"),
                                label=f"$p={str(p).replace('.', ',')}$", linewidth=0)
                bottom += freq[p].to_numpy()
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("bloque", fontsize=8)
            ax.set_xlim(0, 65)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("fracción de corridas", fontsize=8)
        axes[1].legend(fontsize=7, loc="lower right", frameon=True)
        fig.tight_layout()
        fig.savefig(out / "intensidad.pdf"); fig.savefig(out / "intensidad.png", dpi=130)
        plt.close(fig)

    # Curvas de convergencia normalizadas
    curves_path = source / "curves.csv"
    if curves_path.exists():
        curves = pd.read_csv(curves_path)
        curves["A"] = curves["method"].map(LABELS)
        reference = runs.groupby("instance")["fitness"].min()
        curves["gap"] = (curves["best"] / curves["instance"].map(reference) - 1) * 100
        curves["escala"] = np.where(curves["instance"].str.contains(r"-(?:40|50|60)-"), "40-60", "15-30")
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        styles = {"A1": ("#000000", "-"), "A2": ("#7f7f7f", "--"), "A3": ("#bcbd22", ":"),
                  "A4": ("#9467bd", "-."), "A5": ("#1f77b4", "-"), "A6": ("#2ca02c", "--"),
                  "A7": ("#d62728", "-")}
        for ax, scale in zip(axes, ("15-30", "40-60")):
            sub = curves[curves["escala"] == scale]
            sub = sub[sub["evaluations"] >= 2000]
            for name in ORDER:
                line = sub[sub["A"] == name].groupby("evaluations")["gap"].median()
                color, style = styles[name]
                ax.plot(line.index, line.values, color=color, linestyle=style, linewidth=1.1, label=name)
            ax.set_title(f"{scale} trabajos ({sub['instance'].nunique()} instancias)", fontsize=9)
            ax.set_xlabel("evaluaciones", fontsize=8)
            ax.set_xlim(2000, 10000)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{int(v):,}".replace(",", " ") if v >= 10000 else f"{int(v)}"))
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, linewidth=0.5)
        axes[0].set_ylabel("distancia mediana a la mejor (%)", fontsize=8)
        axes[0].legend(fontsize=7, ncol=2, frameon=True)
        fig.tight_layout()
        fig.savefig(out / "convergencia.pdf"); fig.savefig(out / "convergencia.png", dpi=130)
        plt.close(fig)

    text = "\n".join(lines)
    (out / "resumen.txt").write_text(text, encoding="utf-8")
    (out / "resumen.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=lambda v: None if (isinstance(v, float) and math.isnan(v)) else str(v)),
        encoding="utf-8",
    )
    print(text)


if __name__ == "__main__":
    main()
