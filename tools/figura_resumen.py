"""Figura de tres paneles que resume P1, P2 y P3 para el informe.

Uso (desde la raíz del repositorio):

    python tools/figura_resumen.py --runs results/final --out docs/informes/figuras/resumen.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {
    "base": "A1", "fija": "A2", "aleatoria": "A3", "reglas": "A4",
    "aprende_p": "A5", "aprende_mov": "A6", "dos_niveles": "A7",
}
INK, MUTED, GRID, BAR = "#222222", "#8a8a8a", "#d9d9d9", "#2f6f9f"


def _coma(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(labelsize=7, colors=INK, width=0.6, length=2.5)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/final")
    parser.add_argument("--out", default="docs/informes/figuras/resumen.pdf")
    args = parser.parse_args()
    source, out = Path(args.runs), Path(args.out)

    runs = pd.read_csv(source / "runs.csv")
    runs["A"] = runs["method"].map(LABELS)
    blocks = pd.read_csv(source / "blocks.csv")
    blocks["A"] = blocks["method"].map(LABELS)

    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.1), gridspec_kw={"width_ratios": [2.3, 1, 1]}
    )

    # (a) Diferencia relativa de cada variante frente a A1, por instancia
    ax = axes[0]
    med = runs.pivot_table(index="instance", columns="A", values="fitness", aggfunc="median")
    rel = med.sub(med["A1"], axis=0).div(med["A1"], axis=0) * 100
    names = ["A2", "A3", "A4", "A5", "A6", "A7"]
    rng = np.random.default_rng(0)
    for x, name in enumerate(names):
        values = rel[name].to_numpy()
        jitter = rng.uniform(-0.18, 0.18, len(values))
        ax.scatter(x + jitter, values, s=9, color=MUTED, alpha=0.75, linewidths=0, zorder=2)
        ax.hlines(np.median(values), x - 0.3, x + 0.3, color=INK, linewidth=2, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("$\\Delta f$ frente a A1 (%)", fontsize=7.5)
    ax.set_title("(a) Diferencia por instancia", fontsize=8, loc="left")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _coma(v)))
    _style(ax)

    # (b) A3 elige p al azar: bloques que mejoran la mejor solución, por p
    ax = axes[1]
    a3 = blocks[blocks["A"] == "A3"]
    share = a3.groupby("p")["reward"].apply(lambda r: (r > 1e-6).mean() * 100)
    xs = np.arange(len(share))
    ax.bar(xs, share.to_numpy(), width=0.6, color=BAR)
    for x, value in zip(xs, share.to_numpy()):
        ax.text(x, value + 0.2, _coma(value, 1), ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(xs, [_coma(p, 2) for p in share.index])
    ax.set_xlabel("$p$", fontsize=7.5)
    ax.set_ylabel("bloques con mejora (%)", fontsize=7.5)
    ax.set_ylim(0, max(share) * 1.25)
    ax.set_title("(b) Intensidad (A3)", fontsize=8, loc="left")
    _style(ax)

    # (c) Uso del cambio de sala con movimiento uniforme o aprendido
    ax = axes[2]
    use = []
    for name in ("A2", "A6", "A7"):
        counts = runs.loc[runs["A"] == name, ["n_intercambio", "n_insercion", "n_sala"]].sum()
        use.append(counts["n_sala"] / counts.sum() * 100)
    xs = np.arange(3)
    ax.bar(xs, use, width=0.6, color=BAR)
    for x, value in zip(xs, use):
        ax.text(x, value + 0.5, _coma(value, 1), ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(xs, ["A2", "A6", "A7"])
    ax.set_ylabel("mutaciones de sala (%)", fontsize=7.5)
    ax.set_ylim(0, max(use) * 1.25)
    ax.set_title("(c) Movimiento", fontsize=8, loc="left")
    _style(ax)

    fig.tight_layout(w_pad=1.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"figura en {out}")


if __name__ == "__main__":
    main()
