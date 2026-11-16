"""Проверка финальной разметки и её соответствия results/*_v2.xlsx.

Скрипт только читает данные. Код возврата 0 — все проверки пройдены,
1 — есть расхождения. Ожидаемые итоговые числа зафиксированы ниже:
если разметка меняется, их нужно обновить вместе с документацией.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = PROJECT_ROOT / "results" / "annotations" / "annotations_v2.xlsx"

RUNS = {
    "Zero-Shot GigaChat": PROJECT_ROOT / "results/zero_shot/gigachat/zero_shot_gigachat_v2.xlsx",
    "Zero-Shot DeepSeek": PROJECT_ROOT / "results/zero_shot/deepseek/zero_shot_deepseek_v2.xlsx",
    "Naive RAG GigaChat": PROJECT_ROOT / "results/naive_rag/gigachat/naive_rag_gigachat_v2.xlsx",
    "Naive RAG DeepSeek": PROJECT_ROOT / "results/naive_rag/deepseek/naive_rag_deepseek_v2.xlsx",
    "Hybrid RAG GigaChat": PROJECT_ROOT / "results/hybrid_rag/gigachat/hybrid_rag_gigachat_v2.xlsx",
    "Hybrid RAG DeepSeek": PROJECT_ROOT / "results/hybrid_rag/deepseek/hybrid_rag_deepseek_v2.xlsx",
}

RUN = "Прогон"
ID = "№"
ANSWER = "Ответ модели"
FOUND = "Найденные статьи"
ARTICLE = "Статья ТК РФ"
LABEL = "Была галлюцинация (да/нет)"
TAG = "Тип ошибки"
CTX = "Эталонная статья в контексте"
REF = "Отказ"

TAGS = ("ВЫДУМАННЫЙ ФАКТ", "НЕВЕРНАЯ СТАТЬЯ", "УПУЩЕНИЕ", "ВЫДУМАННАЯ СТАТЬЯ")

EXPECTED_HALLUCINATIONS = {
    "Zero-Shot GigaChat": 116,
    "Zero-Shot DeepSeek": 69,
    "Naive RAG GigaChat": 53,
    "Naive RAG DeepSeek": 28,
    "Hybrid RAG GigaChat": 60,
    "Hybrid RAG DeepSeek": 15,
}
EXPECTED_TAGS = {
    "ВЫДУМАННЫЙ ФАКТ": 135,
    "НЕВЕРНАЯ СТАТЬЯ": 123,
    "УПУЩЕНИЕ": 80,
    "ВЫДУМАННАЯ СТАТЬЯ": 3,
}
EXPECTED_CONTEXT_YES = {
    "Naive RAG GigaChat": 191,
    "Naive RAG DeepSeek": 191,
    "Hybrid RAG GigaChat": 181,
    "Hybrid RAG DeepSeek": 181,
}
EXPECTED_REFUSALS = {
    "Naive RAG DeepSeek": 10,
    "Hybrid RAG DeepSeek": 19,
}


def article_number(value: object) -> str | None:
    match = re.search(r"(\d+(?:\.\d+)*(?:-\d+)?)", str(value))
    return match.group(1) if match else None


def retrieved_articles(value: object) -> set[str]:
    return set(re.findall(r"tk-(\d+(?:\.\d+)*(?:-\d+)?)", str(value)))


def normalize(value: object) -> str:
    return (
        str(value)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("_x000D_", "")
    )


def main() -> int:
    if not ANNOTATIONS.exists():
        print(f"Не найден файл разметки: {ANNOTATIONS}")
        return 1

    data = pd.read_excel(ANNOTATIONS)
    errors: list[str] = []
    warnings: list[str] = []

    required = [RUN, ID, ANSWER, LABEL, TAG, CTX, REF, ARTICLE, FOUND]
    missing = [column for column in required if column not in data.columns]
    if missing:
        print("В таблице нет обязательных колонок: " + ", ".join(missing))
        return 1

    if len(data) != 1266:
        errors.append(f"строк {len(data)}, ожидалось 1266")
    if data[RUN].nunique() != 6:
        errors.append(f"прогонов {data[RUN].nunique()}, ожидалось 6")

    for run, raw_path in RUNS.items():
        sub = data[data[RUN] == run]
        if len(sub) != 211:
            errors.append(f"{run}: строк {len(sub)}, ожидалось 211")
            continue

        ids = sorted(int(value) for value in sub[ID])
        if ids != list(range(1, 212)):
            errors.append(f"{run}: набор ID не 1..211")

        if not raw_path.exists():
            errors.append(f"{run}: не найден исходный файл {raw_path}")
            continue

        raw = pd.read_excel(raw_path).set_index(ID)
        answer_column = next(
            (column for column in raw.columns if column.startswith("Ответ_")),
            None,
        )
        if answer_column is None:
            errors.append(f"{run}: в исходном файле нет колонки Ответ_*")
            continue

        answer_mismatches: list[int] = []
        for _, row in sub.iterrows():
            rid = row[ID]
            if rid not in raw.index:
                answer_mismatches.append(int(rid))
                continue
            if normalize(row[ANSWER]) != normalize(raw.at[rid, answer_column]):
                answer_mismatches.append(int(rid))
        if answer_mismatches:
            errors.append(
                f"{run}: ответы модели расходятся с results для ID "
                + ", ".join(str(value) for value in answer_mismatches[:10])
                + ("..." if len(answer_mismatches) > 10 else "")
            )

        if run.startswith("Zero-Shot"):
            if not sub[FOUND].isna().all():
                errors.append(f"{run}: у zero-shot заполнена колонка «Найденные статьи»")
        else:
            found_column = next(
                (column for column in raw.columns if column.startswith("Найденные")),
                None,
            )
            if found_column is None:
                errors.append(f"{run}: в исходном файле нет колонки «Найденные статьи_*»")
            else:
                found_mismatches: list[int] = []
                missing_raw: list[int] = []
                for _, row in sub.iterrows():
                    rid = row[ID]
                    raw_value = raw.at[rid, found_column]
                    if pd.isna(raw_value) or str(raw_value).strip() == "":
                        missing_raw.append(int(rid))
                    elif normalize(row[FOUND]) != normalize(raw_value):
                        found_mismatches.append(int(rid))
                if missing_raw:
                    warnings.append(
                        f"{run}: в results не заполнены «Найденные статьи» для ID "
                        + ", ".join(str(value) for value in missing_raw)
                        + "; флаг контекста взят из размеченного источника"
                    )
                if found_mismatches:
                    errors.append(
                        f"{run}: «Найденные статьи» расходятся с results для ID "
                        + ", ".join(str(value) for value in found_mismatches[:10])
                        + ("..." if len(found_mismatches) > 10 else "")
                    )

        context_mismatches: list[int] = []
        for _, row in sub.iterrows():
            if run.startswith("Zero-Shot"):
                expected = "н/п"
            else:
                expected = (
                    "да"
                    if article_number(row[ARTICLE]) in retrieved_articles(row[FOUND])
                    else "нет"
                )
            if str(row[CTX]).strip() != expected:
                context_mismatches.append(int(row[ID]))
        if context_mismatches:
            errors.append(
                f"{run}: флаг «Эталонная статья в контексте» не совпадает с retrieval для ID "
                + ", ".join(str(value) for value in context_mismatches[:10])
                + ("..." if len(context_mismatches) > 10 else "")
            )

    labels = data[LABEL].astype(str).str.strip()
    tags = data[TAG].fillna("").astype(str).str.strip()
    contexts = data[CTX].astype(str).str.strip()
    refusals = data[REF].astype(str).str.strip()

    invalid_labels = sorted(set(labels) - {"да", "нет"})
    if invalid_labels:
        errors.append("недопустимые значения «Была галлюцинация»: " + ", ".join(invalid_labels))

    invalid_contexts = sorted(set(contexts) - {"да", "нет", "н/п"})
    if invalid_contexts:
        errors.append("недопустимые значения «Эталонная статья в контексте»: " + ", ".join(invalid_contexts))

    invalid_refusals = sorted(set(refusals) - {"да", "нет"})
    if invalid_refusals:
        errors.append("недопустимые значения «Отказ»: " + ", ".join(invalid_refusals))

    unknown_tags = sorted(set(tags[tags != ""]) - set(TAGS))
    if unknown_tags:
        errors.append("неизвестные теги: " + ", ".join(unknown_tags))

    yes = labels.eq("да")
    if (yes & tags.eq("")).any():
        errors.append(f"«да» без тега: {int((yes & tags.eq('')).sum())} строк")
    if (~yes & tags.ne("")).any():
        errors.append(f"тег без «да»: {int((~yes & tags.ne('')).sum())} строк")

    refusal_yes = refusals.eq("да")
    context_yes = contexts.eq("да")
    context_no = contexts.eq("нет")

    bad_refusal_context = data[
        refusal_yes
        & context_yes
        & ~(yes & tags.eq("ВЫДУМАННЫЙ ФАКТ"))
    ]
    if len(bad_refusal_context):
        errors.append(
            "отказ при статье в контексте не помечен как да/ВЫДУМАННЫЙ ФАКТ: "
            + ", ".join(str(int(value)) for value in bad_refusal_context[ID])
        )

    bad_refusal_no_context = data[
        refusal_yes & context_no & ~((~yes) & tags.eq(""))
    ]
    if len(bad_refusal_no_context):
        errors.append(
            "отказ без статьи в контексте не помечен как нет/без тега: "
            + ", ".join(str(int(value)) for value in bad_refusal_no_context[ID])
        )

    zero_shot_refusals = data[data[RUN].str.startswith("Zero-Shot") & refusal_yes]
    if len(zero_shot_refusals):
        errors.append(f"отказы в zero-shot: {len(zero_shot_refusals)} строк")

    hall_counts = (
        data.assign(_hall=yes)
        .groupby(RUN)["_hall"]
        .sum()
        .astype(int)
        .to_dict()
    )
    for run, expected in EXPECTED_HALLUCINATIONS.items():
        actual = int(hall_counts.get(run, 0))
        if actual != expected:
            errors.append(f"{run}: галлюцинаций {actual}, ожидалось {expected}")

    tag_counts = tags[tags != ""].value_counts().to_dict()
    for tag, expected in EXPECTED_TAGS.items():
        actual = int(tag_counts.get(tag, 0))
        if actual != expected:
            errors.append(f"тег {tag}: {actual}, ожидалось {expected}")

    context_counts = (
        data.assign(_ctx=context_yes)
        .groupby(RUN)["_ctx"]
        .sum()
        .astype(int)
        .to_dict()
    )
    for run, expected in EXPECTED_CONTEXT_YES.items():
        actual = int(context_counts.get(run, 0))
        if actual != expected:
            errors.append(f"{run}: статья в контексте {actual}, ожидалось {expected}")

    refusal_counts = (
        data.assign(_ref=refusal_yes)
        .groupby(RUN)["_ref"]
        .sum()
        .astype(int)
        .to_dict()
    )
    for run, expected in EXPECTED_REFUSALS.items():
        actual = int(refusal_counts.get(run, 0))
        if actual != expected:
            errors.append(f"{run}: отказов {actual}, ожидалось {expected}")

    total_hall = int(yes.sum())
    total_refusals = int(refusal_yes.sum())
    total_refusals_context = int((refusal_yes & context_yes).sum())
    total_refusals_no_context = int((refusal_yes & context_no).sum())
    if total_hall != 341:
        errors.append(f"всего галлюцинаций {total_hall}, ожидалось 341")
    if total_refusals != 29:
        errors.append(f"всего отказов {total_refusals}, ожидалось 29")
    if total_refusals_context != 11:
        errors.append(f"отказов со статьёй в контексте {total_refusals_context}, ожидалось 11")
    if total_refusals_no_context != 18:
        errors.append(f"отказов без статьи в контексте {total_refusals_no_context}, ожидалось 18")

    print(f"Строк: {len(data)}; прогонов: {data[RUN].nunique()}")
    print("Галлюцинации по прогонам:", hall_counts, "| всего", total_hall)
    print("Теги:", tag_counts)
    print(
        "Отказы:",
        refusal_counts,
        f"| всего {total_refusals}",
        f"| со статьёй {total_refusals_context}",
        f"| без статьи {total_refusals_no_context}",
    )
    print("Статья в контексте (да):", context_counts)

    if warnings:
        print("\nПРЕДУПРЕЖДЕНИЯ:")
        for warning in warnings:
            print(" -", warning)

    if errors:
        print("\nОШИБКИ:")
        for error in errors:
            print(" -", error)
        return 1

    print("\nOK: структура, ответы, retrieval и правила разметки согласованы.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
