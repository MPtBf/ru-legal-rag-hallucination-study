import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Пути к данным (согласованная структура)
DATASET_PATH = BASE_DIR / "data" / "dataset.xlsx"
CORPUS_PATH = BASE_DIR / "data" / "corpus" / "tk_rf_articles.jsonl"
INDEX_DIR = BASE_DIR / "data" / "index"
RESULTS_DIR = BASE_DIR / "results"

load_dotenv(BASE_DIR / ".env")

GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Основные модели
GIGACHAT_MODEL = "GigaChat-2-Pro"
DEEPSEEK_MODEL = "deepseek-reasoner"
EVALUATE_MODEL = "deepseek-reasoner"

# Короткое имя для CLI и подпись в колонках результатов
# MODEL_LABELS = {"gigachat": "GigaChat", "deepseek": "DeepSeek"}

# Параметры генерации
TEMPERATURE = 0.1

# Таймаут одного HTTP-запроса к модели, секунды
REQUEST_TIMEOUT = 30

# Повторы запросов: 5 попыток с паузой 2, 4, 8, 16 секунд
RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 2

# Каталог логов прогонов
LOGS_DIR = BASE_DIR / "logs"

# Endpoints
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def load_dataset() -> pd.DataFrame:
    return pd.read_excel(DATASET_PATH)


# Допустимые для работы модели
allowed_giga_models = [
    "gigachat",
    "GigaChat",
    "GigaChat-2",
    "GigaChat-2-Pro",
    "GigaChat-2-Max",
    "GigaChat-Pro",
    "GigaChat-Max",
    "GigaChat-3-Ultra",
]

allowed_deepseek_models = [
    "deepseek",
    "deepseek-chat",
    "deepseek-reasoner",
]

allowed_models = allowed_giga_models + allowed_deepseek_models