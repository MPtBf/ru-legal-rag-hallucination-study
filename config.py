import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.xlsx"

# Загружаем ключи из .env (см. .env.example)
load_dotenv(BASE_DIR / ".env")

# API-ключи
GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Имена моделей
GIGACHAT_MODEL = "GigaChat-2"
DEEPSEEK_MODEL = "deepseek-chat"

# Параметры генерации
TEMPERATURE = 0.1

# Endpoints
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def load_dataset() -> pd.DataFrame:
    return pd.read_excel(DATASET_PATH)
