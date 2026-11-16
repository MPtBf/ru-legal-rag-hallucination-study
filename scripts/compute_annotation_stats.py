"""Пересчёт чисел для раздела «Результаты» по финальной разметке.

Читает `results/annotations/annotations_v2.xlsx`, печатает отчёт и
перезаписывает `results/annotations/_итоги.tsv`. Скрипт не меняет саму
разметку и не пересобирает ответы моделей.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = PROJECT_ROOT / "results" / "annotations" / "annotations_v2.xlsx"
SUMMARY_TSV = PROJECT_ROOT / "results" / "annotations" / "_итоги.tsv"

RUN = "Прогон"
ID = "№"
LABEL = "Была галлюцинация (да/нет)"
TAG = "Тип ошибки"
CTX = "Эталонная статья в контексте"
REF = "Отказ"

RUN_ORDER = [
    "Zero-Shot GigaChat",
    "Zero-Shot DeepSeek",
    "Naive RAG GigaChat",
    "Naive RAG DeepSeek",
    "Hybrid RAG GigaChat",
    "Hybrid RAG DeepSeek",
]
TAGS = ["ВЫДУМАННАЯ СТАТЬЯ", "НЕВЕРНАЯ СТАТЬЯ", "ВЫДУМАННЫЙ ФАКТ", "УПУЩЕНИЕ"]


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float, float]:
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return p, center - half, center + half


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> int:
    data = pd.read_excel(ANNOTATIONS)
    data["hall"] = data[LABEL].astype(str).str.strip().eq("да")
    data["ref"] = data[REF].astype(str).str.strip().eq("да")
    data["ctx"] = data[CTX].astype(str).str.strip().eq("да")
    data["tag"] = data[TAG].fillna("").astype(str).str.strip()

    print("== Доли и 95 % ДИ Уилсона ==")
    for run in RUN_ORDER:
        sub = data[data[RUN] == run]
        k = int(sub["hall"].sum())
        n = len(sub)
        p, low, high = wilson(k, n)
        print(f"{run:22} {k:3}/{n}  {p * 100:5.1f} %  CI95 {low * 100:4.1f}-{high * 100:4.1f}")

    print("\n== МакНемар (точный) ==")
    for model in ("GigaChat", "DeepSeek"):
        zero = data[data[RUN] == f"Zero-Shot {model}"].set_index(ID)["hall"]
        naive = data[data[RUN] == f"Naive RAG {model}"].set_index(ID)["hall"]
        hybrid = data[data[RUN] == f"Hybrid RAG {model}"].set_index(ID)["hall"]
        pairs = {
            "Zero-Naive": (zero, naive),
            "Zero-Hybrid": (zero, hybrid),
            "Naive-Hybrid": (naive, hybrid),
        }
        for name, (left, right) in pairs.items():
            b01 = int((left & ~right).sum())
            b10 = int((~left & right).sum())
            print(
                f"{model:9} {name:12} discordant {b01}+{b10:2}  "
                f"p={mcnemar_exact(b01, b10):.4f}"
            )
            if name in {"Zero-Naive", "Zero-Hybrid"}:
                print(f"          fixed {b01:3}  added {b10:3}")

    print("\n== Теги по прогонам ==")
    total_tags = data.loc[data["hall"], "tag"].value_counts()
    print("Всего:", {tag: int(total_tags.get(tag, 0)) for tag in TAGS})
    for run in RUN_ORDER:
        sub = data[data[RUN] == run]
        counts = sub.loc[sub["hall"], "tag"].value_counts()
        total = int(sub["hall"].sum())
        cells = ", ".join(f"{tag}:{int(counts.get(tag, 0))}" for tag in TAGS)
        print(f"{run:22} total {total:3} | {cells}")

    print("\n== Исправлено / новые ошибки ==")
    for model in ("GigaChat", "DeepSeek"):
        zero = data[data[RUN] == f"Zero-Shot {model}"].set_index(ID)["hall"]
        for label, run in (
            ("Naive", f"Naive RAG {model}"),
            ("Hybrid", f"Hybrid RAG {model}"),
        ):
            current = data[data[RUN] == run].set_index(ID)["hall"]
            fixed = int((zero & ~current).sum())
            new = int((~zero & current).sum())
            new_mask = (
                (data[RUN] == run)
                & (~data[ID].map(zero).fillna(False).astype(bool))
                & data["hall"]
            )
            new_tags = data.loc[new_mask, "tag"].value_counts().to_dict()
            print(
                f"{model:9} {label:6} fixed={fixed:3} new={new:3} "
                f"| new tags {new_tags}"
            )

    print("\n== Ложные предпосылки (201-211) ==")
    for run in RUN_ORDER:
        sub = data[(data[RUN] == run) & data[ID].between(201, 211)]
        print(f"{run:22} errors {int(sub['hall'].sum()):2} / {len(sub)}")

    print("\n== Отказы ==")
    for run in RUN_ORDER:
        sub = data[data[RUN] == run]
        refusals = sub[sub["ref"]]
        context = refusals["ctx"]
        print(
            f"{run:22} refusals {len(refusals):2} | "
            f"ctx {int(context.sum()):2} | no-ctx {int((~context).sum()):2}"
        )
    total_refusals = int(data["ref"].sum())
    total_refusals_context = int((data["ref"] & data["ctx"]).sum())
    print(
        f"Всего: {total_refusals} | ctx {total_refusals_context} | "
        f"no-ctx {total_refusals - total_refusals_context}"
    )

    rows = []
    for run in RUN_ORDER:
        sub = data[data[RUN] == run]
        context_strings = sub[CTX].astype(str).str.strip()
        tag_counts = sub.loc[sub["hall"], "tag"].value_counts().to_dict()
        rows.append(
            {
                "прогон": run,
                "строк": len(sub),
                "статья_в_контексте_да": int((context_strings == "да").sum()),
                "статья_в_контексте_нет": int((context_strings == "нет").sum()),
                "отказов": int(sub["ref"].sum()),
                "отказ_и_статья_в_контексте": int(
                    (sub["ref"] & (context_strings == "да")).sum()
                ),
                "отказ_без_статьи": int(
                    (sub["ref"] & (context_strings == "нет")).sum()
                ),
                "галлюцинаций_да": int(sub["hall"].sum()),
                "галлюцинаций_нет": int((~sub["hall"]).sum()),
                "теги": str(tag_counts),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_TSV, sep="\t", index=False, encoding="utf-8")
    print(f"\nЗаписан {SUMMARY_TSV}")

    total_hall = int(data["hall"].sum())
    print(
        f"Итого: {total_hall} / {len(data)} "
        f"({100 * total_hall / len(data):.1f} %)"
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
