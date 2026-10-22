import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict

import pandas as pd

import config
import prompts
from experiments import PIPELINES

from evaluate import EvaluateModel

ID_COL = "№"
ETALON_COL = "Ответ"
QUESTION_COL = "Вопрос"
MARKUP_COLUMNS = ["Была галлюцинация (да/нет)", "Комментарий"]
ERROR_PREFIX = "Ошибка"


def norm_question(value: Any) -> str:
    """Нормализация текста вопроса для сопоставления (нумерация № неустойчива)."""
    return str(value).strip().lower()


def is_filled(value: Any) -> bool:
    """Ответ считается готовым, если он непустой и не является ошибкой API."""
    text = "" if pd.isna(value) else str(value).strip()
    return text != "" and not text.startswith(ERROR_PREFIX)


def read_or_create(source_df: pd.DataFrame, out_path: Path, answer_col: str) -> pd.DataFrame:
    if out_path.exists():
        return pd.read_excel(out_path)
    results_df = source_df.copy()
    results_df[answer_col] = ""
    for col in MARKUP_COLUMNS:
        results_df[col] = ""
    return results_df


def save_metadata(out_path: Path, meta: Dict[str, Any]) -> None:
    # results/{mode}/{model}/{stem}_metadata.json рядом с таблицей результатов.
    meta_path = out_path.with_name(out_path.stem + "_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    allowed_models = config.allowed_giga_models + config.allowed_deepseek_models
    # print(allowed_models)

    parser = argparse.ArgumentParser(description="Единый раннер RAG-экспериментов")
    parser.add_argument("--mode", required=True, choices=sorted(PIPELINES), help="режим эксперимента")
    parser.add_argument("--model", required=True, choices=allowed_models, help="модель")
    parser.add_argument("--limit", type=int, default=None, help="обработать только первые N вопросов (для тестов)")
    parser.add_argument("--output-path", type=Path, default=None, help="свой путь к файлу результатов (для тестов)")
    parser.add_argument("--evaluate", action="store_true", help="проводить оценку результатов (добавьте флаг, чтобы включить)")
    parser.add_argument("--evaluate_model", default="deepseek-reasoner", choices=allowed_models, help="модель для проведения оценки")
    args = parser.parse_args()

    if args.model == "deepseek":
        model = config.DEEPSEEK_MODEL
    elif args.model == "gigachat":
        model = config.GIGACHAT_MODEL
    elif args.model in allowed_models:
        model = args.model
    else:
        raise ValueError(
            f"Выбранная модель '{args.model}' не поддерживается. "
            f"Доступные модели: {', '.join(allowed_models)}"
        )

    date_str = date.today().isoformat()
    pipeline_cls = PIPELINES[args.mode]
    pipeline = pipeline_cls(model=model)
    answer_col = pipeline.answer_col

    # Выбор модели оценщика
    evaluate_client = None
    evaluate_model = ""
    if args.evaluate:
        if args.evaluate_model == "deepseek":
            evaluate_model = config.DEEPSEEK_MODEL
        elif args.evaluate_model == "gigachat":
            evaluate_model = config.GIGACHAT_MODEL
        elif args.evaluate_model in allowed_models:
            evaluate_model = args.evaluate_model
        else:
            raise ValueError(
                f"Выбранная модель '{args.evaluate_model}' не поддерживается. "
                f"Доступные модели: {', '.join(allowed_models)}"
            )

        # eva_auth_key = ""
        # if evaluate_model in config.allowed_deepseek_models:
        #     eva_auth_key = config.DEEPSEEK_API_KEY
        # elif evaluate_model in config.allowed_giga_models:
        #     eva_auth_key = config.GIGACHAT_AUTH_KEY

        evaluate_client = EvaluateModel(model=evaluate_model)

    # results/{mode}/{model}/{mode}_{model}_{date}_v{prompt_version}.xlsx - без автоматической оценки
    # results/{mode}/{model}/{mode}_{model}_{date}_v{prompt_version}_eva_{evaluate_model}.xlsx - а оценкой
    if args.output_path is not None:
        out_path = args.output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = config.RESULTS_DIR / args.mode / args.model
        out_dir.mkdir(parents=True, exist_ok=True)

        # PROMPT_VERSION уже содержит префикс "v" (например "v1")
        fname = f"{args.mode}_{args.model}_{date_str}_{prompts.PROMPT_VERSION}"

        # Если оценка включена, добавляем имя оценщика в название файла
        if evaluate_client is not None:
            fname += f"_eva_{evaluate_model}"

        fname += ".xlsx"
        out_path = out_dir / fname

    source_df = config.load_dataset()
    if args.limit is not None:
        source_df = source_df.head(args.limit).copy()
    results_df = read_or_create(source_df, out_path, answer_col)

    # Гарантируем, что нужные колонки есть даже у существующего файла.
    if answer_col not in results_df.columns:
        results_df[answer_col] = ""
    for col in MARKUP_COLUMNS:
        if col not in results_df.columns:
            results_df[col] = ""

    index_by_question = {}
    for pos, q_text in enumerate(results_df[QUESTION_COL]):
        index_by_question.setdefault(norm_question(q_text), pos)

    processed = 0
    skipped = 0
    total = len(source_df)
    print(f"Запуск: mode={args.mode}, model={args.model}")
    print(f"Выходной файл: {out_path}")

    for idx, row in source_df.iterrows():
        q_id = row[ID_COL]
        q_text = row[QUESTION_COL]

        pos = index_by_question.get(norm_question(q_text))
        if pos is not None and is_filled(results_df.at[pos, answer_col]):
            print(f"Обработка: строка {idx + 1}/{total} - уже заполнена")
            skipped += 1
            continue

        print(f"Обработка: строка {idx + 1}/{total} (Вопрос №{q_id})...")
        answer = pipeline.answer(q_text)
        text = answer.text

        # Значения по умолчанию для колонок разметки
        eval_hallucinated = ""
        eval_comment = ""

        # Оценка, если включена
        if evaluate_client is not None:
            eval_result = evaluate_client.evaluate_row(etalon=str(row[ETALON_COL]), ai_answer=text)
            eval_hallucinated = eval_result.get("hallucinated", "")
            eval_comment = eval_result.get("comment", "")

        if pos is None:
            # Вопроса ещё нет в файле результатов — добавляем строку.
            new_row = row.to_dict()
            new_row[answer_col] = text
            new_row[MARKUP_COLUMNS[0]] = eval_hallucinated
            new_row[MARKUP_COLUMNS[1]] = eval_comment

            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            pos = results_df.index[-1]
            index_by_question.setdefault(norm_question(q_text), pos)
        else:
            results_df.at[pos, answer_col] = text
            results_df.at[pos, MARKUP_COLUMNS[0]] = eval_hallucinated
            results_df.at[pos, MARKUP_COLUMNS[1]] = eval_comment

        results_df.to_excel(out_path, index=False)
        processed += 1

    # Метаданные прогона для отслеживаемости эксперимента.
    meta_data = {
        "mode": args.mode,
        "model": args.model,
        "prompt_version": prompts.PROMPT_VERSION,
        "date": date_str,
        "system_prompt": prompts.SYSTEM_PROMPT,
        "answer_column": answer_col,
        "total_questions": total,
        "limit": args.limit,
        "processed": processed,
        "skipped_existing": skipped,
    }

    # Добавляем информацию об оценщике в метаданные
    if evaluate_client is not None:
        meta_data["evaluate_model"] = evaluate_model
        meta_data["evaluate_prompt"] = prompts.EVALUATE_PROMPT

    save_metadata(out_path, meta_data)

    skip_msg = f", пропущено (уже есть) {skipped}" if skipped > 0 else ""
    print(f"Готово: обработано {processed}{skip_msg}.")
    print(f"Результаты: {out_path}")


if __name__ == "__main__":
    main()
