import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

import config
import prompts
from experiments import PIPELINES
from llm import LLMRequestError
from evaluate import EvaluateModel

logger = logging.getLogger(__name__)

ID_COL = "№"
ETALON_COL = "Ответ"
QUESTION_COL = "Вопрос"
MARKUP_COLUMNS = ["Была галлюцинация (да/нет)", "Комментарий"]

# Условия, при расхождении с метаданными прогон останавливается:
# ответы в одном файле должны быть получены в одинаковых условиях.
CONDITION_FIELDS = [
    "requested_model",
    "system_prompt",
    "prompt_version",
    "temperature",
    "answer_column",
    "index",
    "evaluate_model",
    "evaluate_prompt",
]


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


def metadata_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + "_metadata.json")


def retrieval_path(out_path: Path) -> Path:
    """Полные тексты найденных чанков: в xlsx они не помещаются."""
    return out_path.with_name(out_path.stem + "_retrieval.jsonl")


def append_retrieval(out_path: Path, q_id: Any, question: str, docs: list) -> None:
    """Одна строка JSON на вопрос: что нашёл ретривер и с каким счётом."""
    record = {"id": q_id, "question": question, "fragments": docs}
    with open(retrieval_path(out_path), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metadata(out_path: Path, meta: Dict[str, Any]) -> None:
    """Атомарная запись: обрыв не оставит битый JSON."""
    meta_path = metadata_path(out_path)
    tmp_path = meta_path.with_name(meta_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, meta_path)


def read_metadata(out_path: Path) -> Dict[str, Any] | None:
    meta_path = metadata_path(out_path)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        # Битый JSON считаем отсутствием метаданных: проверять условия нельзя.
        return None


def dataset_sha256() -> str:
    digest = hashlib.sha256()
    with open(config.DATASET_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> tuple[str | None, bool | None]:
    """Коммит и признак незакоммиченных правок: без них git_commit врёт."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _preview(value: Any, limit: int = 80) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def check_conditions(existing: Dict[str, Any], current: Dict[str, Any]) -> list[str]:
    """Расхождения в условиях, при которых ответы в файле несравнимы."""
    diffs = []
    for key in CONDITION_FIELDS:
        old = existing.get(key)
        new = current.get(key)
        if old != new:
            diffs.append(f"{key}: {_preview(old)} -> {_preview(new)}")
    return diffs


def add_usage(total: Dict[str, int], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def clear_answers(results_df: pd.DataFrame, answer_col: str) -> None:
    results_df[answer_col] = ""
    for col in MARKUP_COLUMNS:
        results_df[col] = ""


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
    parser.add_argument(
        "--force",
        action="store_true",
        help="очистить ответы и перегенерировать всё заново, игнорируя прошлые условия",
    )
    parser.add_argument("--evaluate", action="store_true", help="оценивать ответы LLM-судьёй")
    parser.add_argument(
        "--evaluate-model", "--evaluate_model", dest="evaluate_model",
        default=config.EVALUATE_MODEL, choices=sorted(config.MODEL_LABELS),
        help="алиас модели для оценки (по умолчанию deepseek)",
    )
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
    evaluate_client = EvaluateModel(args.evaluate_model) if args.evaluate else None
    answer_col = pipeline.answer_col
    # RAG-режимы определяют свой системный промпт и версию; zero-shot берёт общие.
    system_prompt = getattr(pipeline, "system_prompt", prompts.SYSTEM_PROMPT)
    prompt_version = getattr(pipeline, "prompt_version", prompts.PROMPT_VERSION)
    # Колонка с найденными статьями и JSONL с фрагментами — только у ретривера.
    retrieval_col = getattr(pipeline, "retrieval_col", None)

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
        fname = f"{args.mode}_{args.model}_{prompt_version}.xlsx"
        out_path = out_dir / fname

    source_df = config.load_dataset()
    if args.limit is not None:
        source_df = source_df.head(args.limit).copy()
    results_df = read_or_create(source_df, out_path, answer_col)

    # Гарантируем, что нужные колонки есть даже у существующего файла.
    if answer_col not in results_df.columns:
        results_df[answer_col] = ""
    if retrieval_col and retrieval_col not in results_df.columns:
        results_df[retrieval_col] = ""
    for col in MARKUP_COLUMNS:
        if col not in results_df.columns:
            results_df[col] = ""

    # № мог измениться в датасете - синхронизируем и сразу сохраняем правки.
    fixed_ids = align_ids(source_df, results_df)
    if fixed_ids:
        results_df.to_excel(out_path, index=False)
        logger.info("Синхронизировано № по датасету: %d", fixed_ids)
        print(f"Синхронизировано № по датасету: {fixed_ids}")

    # Метаданные: условия прогона пишем до первого запроса к API,
    # итоги (модель, токены, счётчики) дописываем в тот же файл в конце.
    commit, dirty = git_state()
    meta = {
        "mode": args.mode,
        "requested_model": pipeline.requested_model,
        "response_model": None,
        "system_fingerprint": None,
        "prompt_version": prompt_version,
        "system_prompt": system_prompt,
        "temperature": config.TEMPERATURE,
        "answer_column": answer_col,
        # Параметры индекса: условия ретривера, None у zero-shot.
        "index": getattr(pipeline, "index_conditions", None),
        "evaluate_model": args.evaluate_model if evaluate_client else None,
        "evaluate_prompt": prompts.EVALUATE_PROMPT if evaluate_client else None,
        "dataset_sha256": dataset_sha256(),
        "git_commit": commit,
        "git_dirty": dirty,
        "run_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_finished_at": None,
        "limit": args.limit,
        "processed": 0,
        "skipped_existing": 0,
        "usage": {},
    }

    existing_meta = read_metadata(out_path) if out_path.exists() else None

    if out_path.exists() and existing_meta is None and not args.force:
        logger.error(
            "Нет метаданных для %s: нельзя проверить условия прогона. "
            "Запустите с --force, чтобы перегенерировать всё заново.",
            out_path,
        )
        sys.exit(2)

    if existing_meta is not None and not args.force:
        # Метаданные прежнего формата (без temperature) не позволяют проверить
        # условия: предупреждаем и перезаписываем их в новом формате.
        if "temperature" not in existing_meta:
            logger.warning(
                "Метаданные старого формата: условия прогона не проверены, "
                "файл будет перезаписан в новом формате"
            )
            diffs = []
        else:
            diffs = check_conditions(existing_meta, meta)
        if diffs:
            logger.error(
                "Условия прогона не совпадают с сохранёнными:\n  %s\n"
                "Верните прежние параметры или запустите с --force, "
                "чтобы перегенерировать всё заново.",
                "\n  ".join(diffs),
            )
            sys.exit(2)
        old_hash = existing_meta.get("dataset_sha256")
        if old_hash and old_hash != meta["dataset_sha256"]:
            logger.warning("Датасет изменился с прошлого запуска")
        old_commit = existing_meta.get("git_commit")
        if old_commit and old_commit != meta["git_commit"]:
            logger.warning("Код изменился с прошлого запуска: %s -> %s", old_commit, commit)

    if args.force and out_path.exists():
        clear_answers(results_df, answer_col)
        results_df.to_excel(out_path, index=False)
        # JSONL с фрагментами перезапишем заново — старые строки не должны
        # остаться рядом с ответами, сгенерированными заново.
        r_path = retrieval_path(out_path)
        if r_path.exists():
            r_path.unlink()
        logger.info("--force: ответы очищены, прогон начнётся заново")
    elif existing_meta is not None:
        # Продолжение прогона: сохраняем накопленный провенанс и токены,
        # иначе запуск без новых вопросов обнулил бы их.
        meta["response_model"] = existing_meta.get("response_model")
        meta["system_fingerprint"] = existing_meta.get("system_fingerprint")
        if isinstance(existing_meta.get("usage"), dict):
            meta["usage"] = dict(existing_meta["usage"])

    write_metadata(out_path, meta)

    index_by_question = {}
    for pos, q_text in enumerate(results_df[QUESTION_COL]):
        index_by_question.setdefault(norm_question(q_text), pos)

    processed = 0
    skipped = 0
    response_model = meta["response_model"]
    system_fingerprint = meta["system_fingerprint"]
    usage_totals: Dict[str, int] = dict(meta["usage"])
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
        eval_hallucinated = ""
        eval_comment = ""
        if evaluate_client is not None:
            evaluation = evaluate_client.evaluate_row(
                etalon=str(row[ETALON_COL]), ai_answer=text
            )
            eval_hallucinated = evaluation.get("hallucinated", "")
            eval_comment = evaluation.get("comment", "")
        model = answer.meta.get("response_model")
        if model:
            if response_model and model != response_model:
                logger.warning(
                    "Модель ответа изменилась посреди прогона: %s -> %s", response_model, model
                )
            response_model = model
        fingerprint = answer.meta.get("system_fingerprint")
        if fingerprint:
            if system_fingerprint and fingerprint != system_fingerprint:
                logger.warning("Изменился system_fingerprint посреди прогона: %s", fingerprint)
            system_fingerprint = fingerprint
        add_usage(usage_totals, answer.meta.get("usage"))

        if pos is None:
            # Вопроса ещё нет в файле результатов — добавляем строку.
            new_row = row.to_dict()
            new_row[answer_col] = text
            if retrieval_col:
                new_row[retrieval_col] = answer.meta.get("retrieval", "")
            new_row[MARKUP_COLUMNS[0]] = eval_hallucinated
            new_row[MARKUP_COLUMNS[1]] = eval_comment
            for col in MARKUP_COLUMNS:
                new_row[col] = ""
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            pos = results_df.index[-1]
            index_by_question.setdefault(norm_question(q_text), pos)
        else:
            results_df.at[pos, answer_col] = text
            if retrieval_col:
                results_df.at[pos, retrieval_col] = answer.meta.get("retrieval", "")
            if evaluate_client is not None:
                results_df.at[pos, MARKUP_COLUMNS[0]] = eval_hallucinated
                results_df.at[pos, MARKUP_COLUMNS[1]] = eval_comment

        if answer.docs:
            append_retrieval(out_path, q_id, q_text, answer.docs)

        results_df.to_excel(out_path, index=False)
        processed += 1

    # Дописываем в метаданные то, что стало известно после ответов.
    meta.update({
        "response_model": response_model,
        "system_fingerprint": system_fingerprint,
        "run_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "processed": processed,
        "skipped_existing": skipped,
        "usage": usage_totals,
    })
    write_metadata(out_path, meta)

    print(f"Готово: обработано {processed}, пропущено (уже есть) {skipped}.")
    print(f"Результаты: {out_path}")
    logger.info("Готово: обработано %d, пропущено %d", processed, skipped)


if __name__ == "__main__":
    main()
