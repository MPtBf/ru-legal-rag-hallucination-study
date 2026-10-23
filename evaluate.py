import json
import re
import config
from llms import GigaChatModel, DeepSeekModel, _check_model
from prompts import EVALUATE_PROMPT
from config import EVALUATE_MODEL, allowed_models, load_dataset, GIGACHAT_MODEL, DEEPSEEK_MODEL
import pandas as pd
import argparse
from pathlib import Path

ID_COL = "№"
ETALON_COL = "Ответ"
QUESTION_COL = "Вопрос"
AI_ANSWER_COL_PREFIX = "Ответ"
HALLUCINATED_COL = "Была галлюцинация (да/нет)"
COMMENT_COL = "Комментарий"

class EvaluateModel():
    """Класс оценщика, создает клиент выбранной модели и проверяет ответ ИИ на наличие галлюцинации"""
    def __init__(self, model: str = EVALUATE_MODEL):

        _check_model(model)
        if model=="gigachat":
            model = GIGACHAT_MODEL
        if model == "deepseek":
            model = DEEPSEEK_MODEL
        self.model = model

        if self.model in config.allowed_giga_models:
            self.client = GigaChatModel(auth_key=config.GIGACHAT_AUTH_KEY, model=self.model)
        else: #deepseek
            self.client = DeepSeekModel(api_key=config.DEEPSEEK_API_KEY, model=self.model)

    def evaluate_row(self, question: str, etalon: str, ai_answer: str) -> dict:
        context = f"""
        Вопрос: {question}\n\n
        Эталонный ответ: {etalon} \n\n
        Ответ ИИ-модели: {ai_answer}
        """
        raw_response = self.client.generate_response(prompt=context, system_prompt=EVALUATE_PROMPT)

        try: # пробуем запарсить джейсон
            match = re.search(r'{.*}', raw_response, re.DOTALL) # от { до }
            if match:
                data = json.loads(match.group(0))
                hallucinated = "да" if data.get("hallucinated") else "нет"
                comment = data.get("comment", "Нет комментария")
                quote = data.get("quote", "")
                if quote and hallucinated == "да":
                    comment = f"[{quote}] {comment}"
                return {"hallucinated": hallucinated, "comment": comment}
        except Exception:
            pass

        # Если не удалось распарсить, пишем ошибку в комментарий
        return {"hallucinated": "ошибка парсинга", "comment": raw_response}


def main():
    parser = argparse.ArgumentParser(description="Единый ранер Оценщика")
    parser.add_argument("--model", required=True, choices=sorted(config.allowed_models), help="модель")
    parser.add_argument("--file_path", required=True, type=Path, help="путь к файлу результатов для оценки")
    parser.add_argument("--force", action="store_true", help="очистить ответы и перегенерировать всё заново, игнорируя прошлые условия",
    )
    args = parser.parse_args()

    evaluate_model = EvaluateModel(args.model)

    file_path = ""
    data = None

    if args.file_path is None:
        ValueError("Укажите путь к файлу, который нужно анализировать")
    else:
        try:
            file_path = args.file_path
            data = load_dataset(file_path)
        except FileNotFoundError:
            raise ValueError("Указанный путь не существует")


    # file_name_chunks = file_path.stem.split("_")
    # answer_model = file_name_chunks[-2]
    # ai_answer_mode = "_".join(file_name_chunks[:-2])
    #
    # ai_answer_col = AI_ANSWER_COL_PREFIX + "_" + answer_model + "_" + ai_answer_mode
    # ai_answer_col = ai_answer_col.strip().lower().replace("zero_shot", "zeroshot")
    #
    # columns = [i.strip().lower() for i in data.columns]
    #
    # if ai_answer_col != columns[6]:
    #     raise ValueError("Имя модели в колонке ответа не совпадает с именем модели в названии файла:\n"
    #                      f"{ai_answer_col} вместо {columns[6]}")
    # else:
    #     ai_answer_col = list(data.columns)[6]

    import json

    def find_answer_column(file_path: Path, data: pd.DataFrame) -> str:
        # 1) основной путь: берём из метаданных, которые пишет run.py
        meta_path = file_path.with_name(file_path.stem + "_metadata.json")
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                col = meta.get("answer_column")
                if col in data.columns:
                    return col
                # если колонки нет — возможно, файл пересохранили вручную
                print(f"⚠ В метаданных answer_column={col!r}, но такой колонки нет")
            except (OSError, json.JSONDecodeError):
                pass

        # 2) fallback: единственная колонка с префиксом "Ответ_" (эталон — просто "Ответ")
        candidates = [
            c for c in data.columns
            if str(c).startswith(AI_ANSWER_COL_PREFIX + "_")
        ]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f"Не удалось однозначно определить колонку ответа модели.\n"
            f"Кандидаты: {candidates}\n"
            f"Все колонки: {list(data.columns)}"
        )

    ai_answer_col = find_answer_column(file_path, data)
    print(f"Колонка ответа модели: {ai_answer_col!r}")

    #  ===========================================

    # создаем колонки, если их нет
    for col in (HALLUCINATED_COL, COMMENT_COL):
        if col not in data.columns:
            data[col] = ""
        data[col] = data[col].astype("object").where(data[col].notna(), "")

    total = len(data)

    for idx, row in data.iterrows():
        # пропуск уже оценённых, если не --force
        if not args.force and pd.notna(row[HALLUCINATED_COL]) and str(row[HALLUCINATED_COL]).strip():
            continue
        q_id = row[ID_COL]
        q_text = row[QUESTION_COL]
        etalon = row[ETALON_COL]
        ai_answer = row[ai_answer_col]

        print(f"\nОценка: строка {q_id}/{total} (Ответ №{q_id})...")
        try:
            answer = evaluate_model.evaluate_row(question=q_text, etalon=etalon, ai_answer=ai_answer)
        except:
            raise Exception(
                "Ошибка оценщика"
            )

        # answer = {"hallucinated": hallucinated, "comment": comment}

        data.at[idx, HALLUCINATED_COL] = answer["hallucinated"]
        data.at[idx, COMMENT_COL] = answer["comment"]

        # атомарное сохранение после каждой строки
        data.to_excel(file_path, index=False)

if __name__ == "__main__":
    main()