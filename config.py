import pandas as pd

# API-ключи
GIGACHAT_AUTH_KEY = "MDFhMTRlMzItNmVmOS03YzRmLTkxYTctYjg0ZTVhN2M0NjI4OjA1MGEwMTJlLTEwNTEtNDU5YS05NDg1LTQ5NDcyZmQwZWIzNg=="
DEEPSEEK_API_KEY = "sk-41c285c8fe27498ab0c19bb2904dd7f3"

def load_dataset(): return pd.read_excel("dataset.xlsx")