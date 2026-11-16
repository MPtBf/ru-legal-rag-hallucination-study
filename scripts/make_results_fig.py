"""Рисунок 1 для раздела «Результаты»: исправленные и новые ошибки при переходе к RAG.

Читает `results/annotations/annotations_v2.xlsx` и сохраняет
`results/annotations/results_fig1.png`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = PROJECT_ROOT / "results" / "annotations" / "annotations_v2.xlsx"
OUTPUT = PROJECT_ROOT / "results" / "annotations" / "results_fig1.png"


def main() -> int:
    data = pd.read_excel(ANNOTATIONS)
    label = "Была галлюцинация (да/нет)"
    data["err"] = data[label].astype(str).str.strip().eq("да")

    pairs = {
        "GigaChat": (
            "Zero-Shot GigaChat",
            "Naive RAG GigaChat",
            "Hybrid RAG GigaChat",
        ),
        "DeepSeek": (
            "Zero-Shot DeepSeek",
            "Naive RAG DeepSeek",
            "Hybrid RAG DeepSeek",
        ),
    }

    stats: dict[str, dict[str, int]] = {}
    for model, (zero_run, naive_run, hybrid_run) in pairs.items():
        zero = data[data["Прогон"] == zero_run].set_index("№")["err"]
        naive = data[data["Прогон"] == naive_run].set_index("№")["err"]
        hybrid = data[data["Прогон"] == hybrid_run].set_index("№")["err"]
        stats[model] = {
            "zs": int(zero.sum()),
            "fixed_naive": int((zero & ~naive).sum()),
            "fixed_hybrid": int((zero & ~hybrid).sum()),
            "new_naive": int((~zero & naive).sum()),
            "new_hybrid": int((~zero & hybrid).sum()),
            "naive": int(naive.sum()),
            "hybrid": int(hybrid.sum()),
        }

    for model, values in stats.items():
        print(model, values)

    available = {font.name for font in font_manager.fontManager.ttflist}
    family = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
        }
    )

    categories = [
        "Исправлено\nNaive RAG",
        "Исправлено\nHybrid RAG",
        "Новые ошибки\nв Naive RAG",
        "Новые ошибки\nв Hybrid RAG",
    ]
    keys = ["fixed_naive", "fixed_hybrid", "new_naive", "new_hybrid"]

    figure, axes = plt.subplots(figsize=(6.3, 3.0), dpi=300)
    models = ["GigaChat", "DeepSeek"]
    colors = {"GigaChat": "#4d4d4d", "DeepSeek": "#a6a6a6"}
    hatches = {"GigaChat": "", "DeepSeek": "//"}

    height = 0.36
    positions = list(range(len(categories)))

    for index, model in enumerate(models):
        values = [stats[model][key] for key in keys]
        offset = (index - 0.5) * height
        bars = axes.barh(
            [position + offset for position in positions],
            values,
            height=height,
            color=colors[model],
            hatch=hatches[model],
            edgecolor="black",
            linewidth=0.6,
            label=model,
        )
        for bar, value in zip(bars, values):
            axes.text(
                bar.get_width() + 1.2,
                bar.get_y() + bar.get_height() / 2,
                str(value),
                va="center",
                ha="left",
                fontsize=8.5,
            )

    axes.set_yticks(positions)
    axes.set_yticklabels(categories)
    axes.invert_yaxis()
    axes.set_xlabel("Число ответов (из 211)")
    axes.set_xlim(0, 95)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(axis="x", linestyle=":", linewidth=0.6, color="#999999", alpha=0.7)
    axes.set_axisbelow(True)
    axes.legend(loc="lower right", frameon=False)

    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    print("saved", OUTPUT)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
