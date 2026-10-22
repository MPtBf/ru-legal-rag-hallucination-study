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
GIGACHAT_MODEL = "GigaChat-Pro"
DEEPSEEK_MODEL = "deepseek-reasoner"
EVALUATE_MODEL = "deepseek-reasoner"

# Параметры генерации
TEMPERATURE = 0.1

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
