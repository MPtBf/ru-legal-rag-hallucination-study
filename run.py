import argparse
import hashlib
import json
import logging
import os
import platform
import signal
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
DATASET_ID_COL = "№"
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
    "dataset_sha256",
    "corpus_sha256",
    "python_version",
    "limit",
    "evaluate_model",
    "evaluate_prompt",
]


def stable_id(value: Any) -> str:
    """Canonical form for the dataset's immutable, unique case identifier."""
    if pd.isna(value):
        raise ValueError("В датасете найден пустой ID в колонке №")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


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


def validate_dataset(source_df: pd.DataFrame) -> None:
    required = {DATASET_ID_COL, QUESTION_COL}
    missing = required.difference(source_df.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют обязательные колонки: {sorted(missing)}")
    ids = source_df[DATASET_ID_COL].map(stable_id)
    if ids.duplicated().any():
        duplicate = ids[ids.duplicated(keep=False)].tolist()
        raise ValueError(f"ID в колонке № должны быть уникальны; повторы: {duplicate[:10]}")


def metadata_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + "_metadata.json")


def retrieval_path(out_path: Path) -> Path:
    """Полные тексты найденных чанков: в xlsx они не помещаются."""
    return out_path.with_name(out_path.stem + "_retrieval.jsonl")


def write_retrieval(out_path: Path, provenance: Dict[str, Any], source_df: pd.DataFrame) -> None:
    """Atomically materialize retrieval records from authoritative row provenance."""
    path = retrieval_path(out_path)
    tmp = path.with_name(path.name + ".tmp")
    questions = {
        stable_id(row[DATASET_ID_COL]): row[QUESTION_COL]
        for _, row in source_df.iterrows()
    }
    with open(tmp, "w", encoding="utf-8") as f:
        for case_id, record in provenance.items():
            if record.get("retrieval"):
                f.write(json.dumps({
                    "id": case_id,
                    "question": questions.get(case_id),
                    "fragments": record["retrieval"],
                }, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


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


def validate_resume_provenance(results_df: pd.DataFrame, answer_col: str, provenance: Dict[str, Any]) -> None:
    """Refuse resume if a completed XLSX answer has no matching provenance."""
    for _, row in results_df.iterrows():
        case_id = stable_id(row[DATASET_ID_COL])
        if is_filled(row[answer_col]) and case_id not in provenance:
            raise ValueError(
                f"Для уже заполненного ответа ID {case_id} нет построчных метаданных; "
                "используйте новый --output-path или --force."
            )


def dataset_sha256() -> str:
    return file_sha256(config.DATASET_PATH)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_excel(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    df.to_excel(tmp, index=False)
    os.replace(tmp, path)


def current_python_version() -> str:
    return platform.python_version()


def mark_interrupted(signum, _frame) -> None:
    raise KeyboardInterrupt(f"Прогон прерван сигналом {signum}")


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

    signal.signal(signal.SIGINT, mark_interrupted)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, mark_interrupted)

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
    validate_dataset(source_df)
    results_df = read_or_create(source_df, out_path, answer_col)

    # Гарантируем, что нужные колонки есть даже у существующего файла.
    if answer_col not in results_df.columns:
        results_df[answer_col] = ""
    if retrieval_col and retrieval_col not in results_df.columns:
        results_df[retrieval_col] = ""
    for col in MARKUP_COLUMNS:
        if col not in results_df.columns:
            results_df[col] = ""

    if out_path.exists() and DATASET_ID_COL not in results_df.columns and not args.force:
        logger.error("В старом файле результатов нет стабильных ID; используйте новый файл или --force")
        sys.exit(2)

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
        "corpus_sha256": file_sha256(config.CORPUS_PATH),
        "python_version": current_python_version(),
        "git_commit": commit,
        "git_dirty": dirty,
        "run_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_finished_at": None,
        "run_status": "running",
        "completed_ids": [],
        "answer_provenance": {},
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
        # Legacy metadata cannot prove dataset/corpus identity or stable IDs.
        if any(key not in existing_meta for key in CONDITION_FIELDS):
            logger.warning(
                "Старые метаданные неполные; безопасное продолжение невозможно. "
                "Используйте новый --output-path или --force."
            )
            sys.exit(2)
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
        if DATASET_ID_COL not in results_df.columns:
            logger.error("В файле результатов отсутствует колонка стабильного ID №")
            sys.exit(2)
        source_ids = {stable_id(x) for x in source_df[DATASET_ID_COL]}
        result_ids = {stable_id(x) for x in results_df[DATASET_ID_COL]}
        if not source_ids.issubset(result_ids):
            logger.error("Набор ID в XLSX не соответствует датасету; используйте новый файл или --force")
            sys.exit(2)
        old_commit = existing_meta.get("git_commit")
        if old_commit and old_commit != meta["git_commit"]:
            logger.warning("Код изменился с прошлого запуска: %s -> %s", old_commit, commit)

    if args.force and out_path.exists():
        clear_answers(results_df, answer_col)
        atomic_excel(results_df, out_path)
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
        meta["answer_provenance"] = dict(existing_meta.get("answer_provenance", {}))
        meta["completed_ids"] = list(existing_meta.get("completed_ids", []))
        meta["run_history"] = list(existing_meta.get("run_history", []))
        try:
            validate_resume_provenance(results_df, answer_col, meta["answer_provenance"])
        except ValueError as error:
            logger.error("%s", error)
            sys.exit(2)

    run_started = meta["run_started_at"]
    meta.setdefault("run_history", [])
    meta["run_history"].append({"started_at": run_started, "status": "running"})

    write_metadata(out_path, meta)

    index_by_id = {}
    for pos, row in results_df.iterrows():
        case_id = stable_id(row[DATASET_ID_COL])
        if case_id in index_by_id:
            logger.error("В результатах повторяется ID %s", case_id)
            sys.exit(2)
        index_by_id[case_id] = pos

    processed = 0
    skipped = 0
    response_model = meta["response_model"]
    system_fingerprint = meta["system_fingerprint"]
    usage_totals: Dict[str, int] = dict(meta["usage"])
    total = len(source_df)
    print(f"Запуск: mode={args.mode}, model={args.model}")
    print(f"Выходной файл: {out_path}")
    logger.info("Запуск: mode=%s, model=%s, файл=%s, лог=%s", args.mode, args.model, out_path, log_path)

    try:
        for idx, row in source_df.iterrows():
            q_id = row[DATASET_ID_COL]
            q_text = row[QUESTION_COL]
            case_id = stable_id(q_id)
            pos = index_by_id.get(case_id)
            prior = meta["answer_provenance"].get(case_id)
            if pos is not None and not is_filled(results_df.at[pos, answer_col]) and prior and is_filled(prior.get("answer")):
                results_df.at[pos, answer_col] = prior["answer"]
                atomic_excel(results_df, out_path)
            if pos is not None and is_filled(results_df.at[pos, answer_col]):
                skipped += 1
                continue
            print(f"Обработка: строка {idx + 1}/{total} (вопрос №{q_id})...")
            try:
                answer = pipeline.answer(q_text)
            except LLMRequestError as error:
                logger.error("Остановка на вопросе №%s: %s", q_id, error)
                meta["run_status"] = "failed"
                meta["run_finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                meta["last_error"] = str(error)
                meta["processed"] = processed
                meta["skipped_existing"] = skipped
                meta["response_model"] = response_model
                meta["system_fingerprint"] = system_fingerprint
                meta["usage"] = usage_totals
                meta["run_history"][-1].update({"status": "failed", "finished_at": meta["run_finished_at"]})
                write_retrieval(out_path, meta["answer_provenance"], source_df)
                write_metadata(out_path, meta)
                sys.exit(1)
            text = answer.text
            eval_hallucinated = ""
            eval_comment = ""
            if evaluate_client is not None:
                evaluation = evaluate_client.evaluate_row(etalon=str(row[ETALON_COL]), ai_answer=text)
                eval_hallucinated = evaluation.get("hallucinated", "")
                eval_comment = evaluation.get("comment", "")
            model = answer.meta.get("response_model")
            if model:
                if response_model and model != response_model:
                    logger.warning("Response model changed: %s -> %s", response_model, model)
                response_model = model
            fingerprint = answer.meta.get("system_fingerprint")
            if fingerprint:
                if system_fingerprint and fingerprint != system_fingerprint:
                    logger.warning("system_fingerprint changed")
                system_fingerprint = fingerprint
            usage = answer.meta.get("usage")
            add_usage(usage_totals, usage)
            meta["answer_provenance"][case_id] = {
                "dataset_id": case_id, "answer": text, "mode": args.mode,
                "requested_model": pipeline.requested_model, "response_model": model,
                "system_fingerprint": fingerprint, "usage": usage,
                "index": getattr(pipeline, "index_conditions", None),
                "retrieval": answer.docs or None,
                "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            if pos is None:
                new_row = row.to_dict()
                new_row[answer_col] = text
                if retrieval_col:
                    new_row[retrieval_col] = answer.meta.get("retrieval", "")
                new_row[MARKUP_COLUMNS[0]] = eval_hallucinated
                new_row[MARKUP_COLUMNS[1]] = eval_comment
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                pos = results_df.index[-1]
                index_by_id[case_id] = pos
            else:
                results_df.at[pos, answer_col] = text
                if retrieval_col:
                    results_df.at[pos, retrieval_col] = answer.meta.get("retrieval", "")
                if evaluate_client is not None:
                    results_df.at[pos, MARKUP_COLUMNS[0]] = eval_hallucinated
                    results_df.at[pos, MARKUP_COLUMNS[1]] = eval_comment
            processed += 1
            meta["completed_ids"] = sorted(set(meta.get("completed_ids", [])) | {case_id})
            meta["processed"] = processed
            meta["skipped_existing"] = skipped
            meta["response_model"] = response_model
            meta["system_fingerprint"] = system_fingerprint
            meta["usage"] = usage_totals
            write_retrieval(out_path, meta["answer_provenance"], source_df)
            write_metadata(out_path, meta)
            atomic_excel(results_df, out_path)
    except KeyboardInterrupt as error:
        meta["run_status"] = "interrupted"
        meta["run_finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta["last_error"] = str(error)
        meta["processed"] = processed
        meta["skipped_existing"] = skipped
        meta["response_model"] = response_model
        meta["system_fingerprint"] = system_fingerprint
        meta["usage"] = usage_totals
        meta["run_history"][-1].update({"status": "interrupted", "finished_at": meta["run_finished_at"]})
        write_metadata(out_path, meta)
        raise
    except Exception as error:
        meta["run_status"] = "failed"
        meta["run_finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta["last_error"] = f"{type(error).__name__}: {error}"
        meta["processed"] = processed
        meta["skipped_existing"] = skipped
        meta["response_model"] = response_model
        meta["system_fingerprint"] = system_fingerprint
        meta["usage"] = usage_totals
        meta["run_history"][-1].update({"status": "failed", "finished_at": meta["run_finished_at"]})
        write_retrieval(out_path, meta["answer_provenance"], source_df)
        write_metadata(out_path, meta)
        raise

    # Сохраняем итоговое состояние после завершения всех вопросов.
    meta.update({
        "response_model": response_model,
        "system_fingerprint": system_fingerprint,
        "run_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "processed": processed,
        "skipped_existing": skipped,
        "usage": usage_totals,
        "run_status": "succeeded",
    })
    meta["run_history"][-1].update({"status": "succeeded", "finished_at": meta["run_finished_at"]})
    write_metadata(out_path, meta)

    print(f"Готово: обработано {processed}, пропущено (уже есть) {skipped}.")
    print(f"Результаты: {out_path}")
    logger.info("Готово: обработано %d, пропущено %d", processed, skipped)


if __name__ == "__main__":
    main()
