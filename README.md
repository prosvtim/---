# A/B Test Pet-Project (Synthetic Data)

## Описание
Проект демонстрирует end-to-end пайплайн A/B-теста:
- генерация синтетических данных (`data_generation.py`)
- разведочный анализ и визуализации (`eda.py`)
- статистический анализ (z-test, bootstrap, CUPED) и выводы (`analysis.py`)

Датасет: `ab_test_data.csv`, 1_000_000 пользователей, искусственный uplift в группе B (+3% к CR).

## Файлы
- `data_generation.py` — генерация и сохранение `ab_test_data.csv`
- `eda.py` — EDA, базовые агрегаты, графики (в `plots/`)
- `analysis.py` — z-test, оптимизированный bootstrap для долей и revenue, CUPED, сохранение результатов
- `plots/` — автоматически генерируемая папка с графиками
- `stat_results.json` — ключевые численные результаты анализа

## Как запустить (локально)
1. Убедись, что установлен Python 3.8+ и пакеты:
