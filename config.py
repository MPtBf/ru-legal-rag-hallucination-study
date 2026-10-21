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


def load_dataset() -> pd.DataFrame:
    return pd.read_excel(DATASET_PATH)
