import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict

import pandas as pd

import config
import prompts
from experiments import PIPELINES
from experiments.errors import LLMRequestError

logger = logging.getLogger(__name__)

ID_COL = "№"
QUESTION_COL = "Вопрос"
MARKUP_COLUMNS = ["Была галлюцинация (да/нет)", "Комментарий"]


def norm_question(value: Any) -> str:
    """Нормализация текста вопроса для сопоставления (нумерация № неустойчива)."""
    return str(value).strip().lower()


def is_filled(value: Any) -> bool:
    """Ответ считается готовым, если он непустой."""
    text = "" if pd.isna(value) else str(value).strip()
    return text != ""


def read_or_create(source_df: pd.DataFrame, out_path: Path, answer_col: str) -> pd.DataFrame:
    if out_path.exists():
        return pd.read_excel(out_path)
    results_df = source_df.copy()
    results_df[answer_col] = ""
    for col in MARKUP_COLUMNS:
        results_df[col] = ""
    return results_df


def align_ids(source_df: pd.DataFrame, results_df: pd.DataFrame) -> int:
    """Проставляет № из датасета строкам результатов по тексту вопроса.

    Нумерация в датасете менялась (часть номеров сдвинулась, часть добавилась),
    поэтому № в старых файлах результатов мог разойтись с актуальным датасетом.
    """
    if ID_COL not in results_df.columns:
        return 0

    ids_by_question = {}
    for _, row in source_df.iterrows():
        ids_by_question.setdefault(norm_question(row[QUESTION_COL]), row[ID_COL])

    changed = 0
    for pos, q_text in enumerate(results_df[QUESTION_COL]):
        new_id = ids_by_question.get(norm_question(q_text))
        if new_id is not None and str(results_df.at[pos, ID_COL]).strip() != str(new_id).strip():
            results_df.at[pos, ID_COL] = new_id
            changed += 1
    return changed


def save_metadata(out_path: Path, meta: Dict[str, Any]) -> None:
    # results/{mode}/{model}/{stem}_metadata.json рядом с таблицей результатов.
    meta_path = out_path.with_name(out_path.stem + "_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def setup_logging(mode: str, model: str, date_str: str) -> Path:
    """Логи прогона: файл logs/{mode}_{model}_{дата}.log и консоль."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOGS_DIR / f"{mode}_{model}_{date_str}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_path


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="Единый раннер RAG-экспериментов")
    parser.add_argument("--mode", required=True, choices=sorted(PIPELINES), help="режим эксперимента")
    parser.add_argument("--model", required=True, choices=sorted(config.MODEL_LABELS), help="модель")
    parser.add_argument("--limit", type=positive_int, default=None, help="обработать только первые N вопросов (для тестов)")
    parser.add_argument("--output-path", type=Path, default=None, help="свой путь к файлу результатов (для тестов)")
    args = parser.parse_args()

    if args.output_path is not None:
        out_path = args.output_path.resolve()
        if out_path.is_dir():
            parser.error("--output-path должен указывать на файл, а не на каталог")
        data_dir = (config.BASE_DIR / "data").resolve()
        if out_path == data_dir or data_dir in out_path.parents:
            parser.error("--output-path не может указывать в каталог data/")

    date_str = date.today().isoformat()
    log_path = setup_logging(args.mode, args.model, date_str)
    pipeline_cls = PIPELINES[args.mode]
    pipeline = pipeline_cls(model=args.model)
    answer_col = pipeline.answer_col

    # --output-path (для тестов) или results/{mode}/{model}/{mode}_{model}_{prompt_version}.xlsx
    if args.output_path is not None:
        out_path = args.output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # results/{mode}/{model}/{mode}_{model}_{prompt_version}.xlsx
        out_dir = config.RESULTS_DIR / args.mode / args.model
        out_dir.mkdir(parents=True, exist_ok=True)
        # PROMPT_VERSION уже содержит префикс "v" (например "v1").
        # Даты в имени нет: прогон продолжается в том же файле и на следующий день.
        fname = f"{args.mode}_{args.model}_{prompts.PROMPT_VERSION}.xlsx"
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

    # № мог измениться в датасете - синхронизируем и сразу сохраняем правки.
    fixed_ids = align_ids(source_df, results_df)
    if fixed_ids:
        results_df.to_excel(out_path, index=False)
        logger.info("Синхронизировано № по датасету: %d", fixed_ids)
        print(f"Синхронизировано № по датасету: {fixed_ids}")

    index_by_question = {}
    for pos, q_text in enumerate(results_df[QUESTION_COL]):
        index_by_question.setdefault(norm_question(q_text), pos)

    processed = 0
    skipped = 0
    response_models = set()
    system_fingerprints = set()
    total = len(source_df)
    print(f"Запуск: mode={args.mode}, model={args.model}")
    print(f"Выходной файл: {out_path}")
    logger.info("Запуск: mode=%s, model=%s, файл=%s, лог=%s", args.mode, args.model, out_path, log_path)

    for idx, row in source_df.iterrows():
        q_id = row[ID_COL]
        q_text = row[QUESTION_COL]

        pos = index_by_question.get(norm_question(q_text))
        if pos is not None and is_filled(results_df.at[pos, answer_col]):
            skipped += 1
            continue

        print(f"Обработка: строка {idx + 1}/{total} (Вопрос №{q_id})...")
        try:
            answer = pipeline.answer(q_text)
        except LLMRequestError as error:
            logger.error(
                "Остановка прогона на вопросе №%s: %s. Сохранено ответов: %d; "
                "при перезапуске заполненные строки будут пропущены.",
                q_id, error, processed,
            )
            sys.exit(1)
        text = answer.text
        response_model = answer.meta.get("response_model")
        if response_model:
            if response_models and response_model not in response_models:
                logger.warning("Модель ответа изменилась посреди прогона: %s", response_model)
            response_models.add(response_model)
        fingerprint = answer.meta.get("system_fingerprint")
        if fingerprint:
            if system_fingerprints and fingerprint not in system_fingerprints:
                logger.warning("Изменился system_fingerprint посреди прогона: %s", fingerprint)
            system_fingerprints.add(fingerprint)

        if pos is None:
            # Вопроса ещё нет в файле результатов — добавляем строку.
            new_row = row.to_dict()
            new_row[answer_col] = text
            for col in MARKUP_COLUMNS:
                new_row[col] = ""
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            pos = results_df.index[-1]
            index_by_question.setdefault(norm_question(q_text), pos)
        else:
            results_df.at[pos, answer_col] = text

        results_df.to_excel(out_path, index=False)
        processed += 1

    # Метаданные прогона для отслеживаемости эксперимента.
    save_metadata(out_path, {
        "mode": args.mode,
        "model": args.model,
        "response_models": sorted(response_models),
        "system_fingerprints": sorted(system_fingerprints),
        "prompt_version": prompts.PROMPT_VERSION,
        "date": date_str,
        "system_prompt": prompts.SYSTEM_PROMPT,
        "answer_column": answer_col,
        "total_questions": total,
        "limit": args.limit,
        "processed": processed,
        "skipped_existing": skipped,
    })

    print(f"Готово: обработано {processed}, пропущено (уже есть) {skipped}.")
    print(f"Результаты: {out_path}")
    logger.info("Готово: обработано %d, пропущено %d", processed, skipped)


if __name__ == "__main__":
    main()