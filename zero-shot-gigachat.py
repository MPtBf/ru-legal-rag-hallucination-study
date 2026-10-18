import os
import pandas as pd
import config
from models import GigaChatModel


def main():
    print("Запуск генерации ответов GigaChat-2 в режиме Zero-Shot")

    OUTPUT_FILE = "results_GigaChat_ZeroShot.xlsx"

    # 1. Загружаем исходный датасет
    try:
        source_df = config.load_dataset()
        print(f"dataset.xlsx загружен")
    except Exception as e:
        print(f"Ошибка загрузки dataset.xlsx: {e}")
        return

    # Автоматически определяем имена ключевых колонок в исходнике (№ или id)
    id_col = "№" if "№" in source_df.columns else source_df.columns[0]
    question_col = "Вопрос"

    # 2. Инициализируем или загружаем таблицу результатов
    if os.path.exists(OUTPUT_FILE):
        print(f"Таблица результатов существует: {OUTPUT_FILE}. Процесс продолжится с последней записи")
        results_df = pd.read_excel(OUTPUT_FILE)
    else:
        print(f"Таблица результатов создана")
        base_columns = list(source_df.columns)
        new_columns = ["Ответ_GigaChat_ZeroShot", "Была галлюцинация (да/нет)", "Комментарий"]
        results_df = pd.DataFrame(columns=base_columns + new_columns)

    # 3. Инициализируем модель Сбера
    llm = GigaChatModel(auth_key=config.GIGACHAT_AUTH_KEY)

    system_prompt = (
        "Ты - высококвалифицированный юрист по трудовому праву РФ. "
        "Отвечай строго на основе Трудового кодекса РФ. "
        "Обязательно указывай номера статей ТК РФ, на которые опираешься."
    )

    print("=" * 10 + "\nОбработка запущена\n")

    # 4. Основной цикл по всем строкам исходного датасета
    for index, row in source_df.iterrows():
        q_number = row[id_col]
        q_text = row[question_col]

        # ПРОВЕРКА НАЛИЧИЯ ОТВЕТА
        is_already_processed = False
        if not results_df.empty and q_number in results_df[id_col].values:
            existing_row = results_df[results_df[id_col] == q_number]
            if pd.notna(existing_row["Ответ_GigaChat_ZeroShot"].values[0]) and \
                    existing_row["Ответ_GigaChat_ZeroShot"].values[0] != "":
                is_already_processed = True

        if is_already_processed:
            print(f"Пропуск: Вопрос №{q_number} уже был обработан ранее.")
            continue

        # Если ответа нет, то запрашиваем его у GigaChat-2
        print(f"Обработка GigaChat: Строка {index + 1}/{len(source_df)} (Вопрос №{q_number})...")
        ai_response = llm.generate_response(prompt=q_text, system_prompt=system_prompt)

        # Собираем оставшиеся элементы строки таблицы
        new_row_data = row.to_dict()
        new_row_data["Ответ_GigaChat_ZeroShot"] = ai_response
        new_row_data["Была галлюцинация (да/нет)"] = ""
        new_row_data["Комментарий"] = ""

        # Добавляем новую запись в общую таблицу результатов
        results_df = pd.concat([results_df, pd.DataFrame([new_row_data])], ignore_index=True)

        # Сохраняем изменения на жесткий диск после каждого запроса
        results_df.to_excel(OUTPUT_FILE, index=False)

    print(f"Обработка GigaChat полностью завершена")
    print(f"Все доступные ответы сохранены в файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()