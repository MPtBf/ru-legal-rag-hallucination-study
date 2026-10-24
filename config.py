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

# Реестр моделей: ключ CLI -> провайдер, имя модели в API и подпись в колонках.
# Новую модель добавляем здесь; пайплайн и CLI берут данные отсюда.
MODELS = {
    "gigachat": {
        "provider": "gigachat",
        "api_model": "GigaChat-2",
        "label": "GigaChat",
    },
    "deepseek": {
        "provider": "deepseek",
        "api_model": "deepseek-chat",
        "label": "DeepSeek",
    },
}

# Короткое имя для CLI и подпись в колонках результатов
MODEL_LABELS = {alias: item["label"] for alias, item in MODELS.items()}

# Параметры генерации
TEMPERATURE = 0.1

# Таймаут одного HTTP-запроса к модели, секунды
REQUEST_TIMEOUT = 30

# Повторы запросов: 5 попыток с паузой 2, 4, 8, 16 секунд
RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 2

# Каталог логов прогонов
LOGS_DIR = BASE_DIR / "logs"

# Наивный RAG: модель эмбеддингов, чанкинг корпуса и глубина поиска.
# Ревизия весов на HuggingFace фиксирует индекс: без неё пересборка
# могла бы молча дать другой результат при обновлении модели.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
INDEX_VERSION = "v1"

# Модель-судья для необязательной автоматической оценки ответов.
EVALUATE_MODEL = "deepseek"


def load_dataset(dataset_path: str | Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_excel(dataset_path)


# Совместимость для скриптов и исторических файлов результатов.
allowed_giga_models = ["gigachat", "GigaChat", "GigaChat-2", "GigaChat-2-Pro", "GigaChat-2-Max", "GigaChat-Pro", "GigaChat-Max", "GigaChat-3-Ultra"]
allowed_deepseek_models = ["deepseek", "deepseek-chat", "deepseek-reasoner"]
allowed_models = allowed_giga_models + allowed_deepseek_models
