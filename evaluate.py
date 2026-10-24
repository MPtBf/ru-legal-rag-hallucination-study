import json
import re
from llm import create_llm
from prompts import EVALUATE_PROMPT
import config

class EvaluateModel:
    """Класс оценщика, создает клиент выбранной модели и проверяет ответ ИИ на наличие галлюцинации"""
    def __init__(self, model: str = config.EVALUATE_MODEL):
        self.model = model
        self.client = create_llm(model)

    def evaluate_row(self, etalon: str, ai_answer: str) -> dict:
        context = f"""
        Эталонный ответ: {etalon} \n\n
        Ответ ИИ-модели: {ai_answer}
        """
        raw_response = self.client.generate_response(prompt=context, system_prompt=EVALUATE_PROMPT)

        try: # пробуем запарсить джейсон
            match = re.search(r'\{.*\}', raw_response, re.DOTALL) # от { до }
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
