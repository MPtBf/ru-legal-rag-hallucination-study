"""Оценка сохранённых ответов на галлюцинации отдельным проходом."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

import config
from llm import create_llm
from prompts import EVALUATE_PROMPT

ID_COL = "№"
ETALON_COL = "Ответ"
QUESTION_COL = "Вопрос"
ANSWER_PREFIX = "Ответ_"
HALLUCINATED_COL = "Была галлюцинация (да/нет)"
COMMENT_COL = "Комментарий"


class EvaluateModel:
    """Оценщик, использующий общий слой моделей и ретраев проекта."""

    def __init__(self, model: str = config.EVALUATE_MODEL):
        if model not in config.MODEL_LABELS:
            raise ValueError(f"Неизвестный алиас модели-оценщика: {model}")
        self.model = model
        self.client = create_llm(model)

    def evaluate_row(self, question: str, etalon: str, ai_answer: str) -> dict[str, str]:
        context = (
            f"Вопрос: {question}\n\n"
            f"Эталонный ответ: {etalon}\n\n"
            f"Ответ ИИ-модели: {ai_answer}"
        )
        raw_response = self.client.generate_response(prompt=context, system_prompt=EVALUATE_PROMPT)
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                hallucinated = "да" if data.get("hallucinated") else "нет"
                comment = data.get("comment", "Нет комментария")
                quote = data.get("quote", "")
                if quote and hallucinated == "да":
                    comment = f"[{quote}] {comment}"
                return {"hallucinated": hallucinated, "comment": str(comment)}
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
        return {"hallucinated": "ошибка парсинга", "comment": raw_response}


def find_answer_column(file_path: Path, data: pd.DataFrame) -> str:
    meta_path = file_path.with_name(f"{file_path.stem}_metadata.json")
    if meta_path.exists():
        try:
            with meta_path.open(encoding="utf-8") as f:
                answer_col = json.load(f).get("answer_column")
            if answer_col in data.columns:
                return answer_col
        except (OSError, json.JSONDecodeError):
            pass
    candidates = [str(col) for col in data.columns if str(col).startswith(ANSWER_PREFIX)]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Не удалось однозначно определить колонку ответа. Кандидаты: {candidates}; "
        f"все колонки: {list(data.columns)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка ответов LLM-судьёй")
    parser.add_argument("--model", required=True, choices=sorted(config.MODEL_LABELS))
    parser.add_argument("--file_path", "--file-path", dest="file_path", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="перегенерировать существующие оценки")
    args = parser.parse_args()
    file_path = args.file_path.resolve()
    if not file_path.is_file():
        parser.error(f"Файл результатов не найден: {file_path}")
    data = pd.read_excel(file_path)
    missing = {ID_COL, ETALON_COL, QUESTION_COL}.difference(data.columns)
    if missing:
        parser.error(f"В файле не хватает обязательных колонок: {sorted(missing)}")

    answer_col = find_answer_column(file_path, data)
    print(f"Колонка ответа модели: {answer_col!r}")
    for col in (HALLUCINATED_COL, COMMENT_COL):
        if col not in data.columns:
            data[col] = ""
        data[col] = data[col].astype("object").where(data[col].notna(), "")

    evaluator = EvaluateModel(args.model)
    total = len(data)
    for idx, row in data.iterrows():
        existing = "" if pd.isna(row[HALLUCINATED_COL]) else str(row[HALLUCINATED_COL]).strip()
        if existing and not args.force:
            continue
        print(f"Оценка: строка {idx + 1}/{total} (вопрос №{row[ID_COL]})...")
        result = evaluator.evaluate_row(
            question=str(row[QUESTION_COL]), etalon=str(row[ETALON_COL]),
            ai_answer="" if pd.isna(row[answer_col]) else str(row[answer_col]),
        )
        data.at[idx, HALLUCINATED_COL] = result["hallucinated"]
        data.at[idx, COMMENT_COL] = result["comment"]
        data.to_excel(file_path, index=False)
    print(f"Готово. Оценки сохранены: {file_path}")


if __name__ == "__main__":
    main()
