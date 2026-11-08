# ADR-0005: Чистый прогон zero-shot на промпте v2

**Дата:** 08.11.2026
**Статус:** принято (решение участника проекта).

## Контекст

По [ADR-0004](ADR-0004-return-to-prompt-v2.md) промпт возвращён к состоянию `b462385`, на котором созданы сохранённые результаты RAG v2. `SYSTEM_PROMPT` версии v2 (510 символов) является точной базовой частью `RAG_SYSTEM_PROMPT` (870 символов), поэтому новый zero-shot прогон на текущем v2 сопоставим с RAG v2 по промпту.

Сохранённые zero-shot артефакты неполные и неоднородные:

- `results/zero_shot/gigachat/zero_shot_gigachat_v1.xlsx` — 211 ответов на промпте v1, без автоматической разметки;
- `results/zero_shot/gigachat/zero_shot_gigachat_v2_metadata.json` — только metadata v2, без xlsx; запрошена модель `GigaChat-2-Pro`, ответов 0, `run_status` не зафиксирован;
- `results/zero_shot/deepseek/zero_shot_deepseek_v2.xlsx` — 5 ответов на модели `deepseek-reasoner`, разметка 3 «нет» / 2 «да»;
- `results/zero_shot/deepseek/zero_shot_deepseek_v2_metadata.json` — metadata к этим 5 ответам, имя колонки ответа не совпадает с xlsx, `run_status` не зафиксирован.

Историческая metadata zero-shot v2 содержит `system_prompt` длиной 509 символов (без перевода строки между правилами 4 и 5) и не проходит текущие проверки условий `run.py`. Поэтому продолжить или перезаписать эти файлы без `--force` нельзя, а `--force` смешал бы старые и новые условия.

## Решение

1. Удалить из рабочего дерева все четыре перечисленных zero-shot файла. Они остаются доступны в истории git.
2. Выполнить zero-shot заново для обеих моделей на текущем промпте v2:
   - `venv\Scripts\python run.py --mode zero_shot --model gigachat --evaluate`
   - `venv\Scripts\python run.py --mode zero_shot --model deepseek --evaluate`
3. Сохранить результаты по штатным путям:
   - `results/zero_shot/gigachat/zero_shot_gigachat_v2.xlsx` + metadata + retrieval;
   - `results/zero_shot/deepseek/zero_shot_deepseek_v2.xlsx` + metadata + retrieval.
4. Использовать те же условия, что в RAG v2: модели `GigaChat-2` и `deepseek-chat`, temperature 0.1, автоматический судья DeepSeek, текущие датасет и корпус.
5. Коммитить результаты каждого прогона отдельно, чтобы metadata следующего запуска фиксировала чистое рабочее дерево.

## Обоснование

- Единая версия промпта v2 для zero-shot и RAG делает сравнение режимов корректным по базовой инструкции.
- Чистые файлы с каноническими именами `*_v2.*` и полной metadata устраняют неоднозначность старых артефактов.
- Удаление только из рабочего дерева сохраняет историю и не уничтожает старые данные безвозвратно.

## Последствия и ограничения

- Прежний zero-shot v1 исчезает из рабочего дерева; при необходимости его можно восстановить из git.
- Zero-shot создаётся 08.11.2026, а RAG v2 — 31.10.2026. Даты и фактические `response_model` нужно указывать при описании сравнения.
- Автоматическая разметка DeepSeek не является человеческой разметкой.
- Строгая сопоставимость «в одной сессии» потребовала бы повторной генерации RAG v2; это решение её не выполняет.
- Результаты RAG v2 и v3 не изменяются.

## Фактические результаты

Оба прогона выполнены 08.11.2026. Команды: `venv\Scripts\python run.py --mode zero_shot --model gigachat --evaluate` и `venv\Scripts\python run.py --mode zero_shot --model deepseek --evaluate`.

| Модель | Файлы | Запуск (UTC) | Длительность | Ответов | Разметка «да/нет» | usage |
|---|---|---|---|---|---|---|
| GigaChat | `results/zero_shot/gigachat/zero_shot_gigachat_v2.xlsx` + `_metadata.json` + `_retrieval.jsonl` | 08.11.2026 11:58:54 → 12:07:05 | ~8 мин | 211/211 | 94 «да» / 117 «нет» | см. metadata |
| DeepSeek | `results/zero_shot/deepseek/zero_shot_deepseek_v2.xlsx` + `_metadata.json` + `_retrieval.jsonl` | 08.11.2026 12:15:26 → 12:28:36 | ~13 мин | 211/211 | 25 «да» / 186 «нет» | prompt 44036 / completion 56539 / total 100575 |

Проверки для обоих прогонов:

- `run_status=succeeded`, `processed=211`, `skipped_existing=0`;
- `prompt_version=v2`, `system_prompt` (510 символов) побайтово совпадает с текущим `prompts.SYSTEM_PROMPT`;
- `evaluate_prompt` совпадает с `prompts.EVALUATE_PROMPT`, `evaluate_model=deepseek`, `temperature=0.1`;
- хеши датасета `8e3eaa9b…` и корпуса `5dc526ad…` совпадают с RAG v2-прогонами;
- `completed_ids` и `answer_provenance` — по 211 записей;
- `response_model`: GigaChat `GigaChat-2:2.0.30.01`, DeepSeek `deepseek-flash`;
- `git_dirty=false` (промпт v2 закоммичен до запусков): GigaChat — `a016cd1`, DeepSeek — `492a8b7`;
- `_retrieval.jsonl` пуст (ожидаемо для zero-shot: retrieval-этапа нет).

Замеченное отклонение: перед удалением старых файлов `api.giga.chat` был недоступен (TLS-handshake timeout / `CERTIFICATE_VERIFY_FAILED` у `requests` без `verify_ssl_certs=False`). В `llm/factory.py` для GigaChat уже задано `verify_ssl_certs=False`, поэтому сам эксперимент на это не влияет; инцидент внешний и в результаты не попал. Результаты RAG v2 и v3 не изменялись.

Файлы закоммичены: GigaChat — `492a8b7`, DeepSeek — `ed7acb5`.
