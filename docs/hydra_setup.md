# Настройка и использование Hydra в PolitAgent

## Обзор

Hydra - это фреймворк для управления конфигурациями экспериментов, который позволяет:
- Организовать конфигурации в иерархическую структуру
- Легко переопределять параметры через командную строку
- Отслеживать эксперименты с автоматическим созданием директорий
- Запускать многократные эксперименты (sweeps)

## Структура конфигураций

```
configs/
├── config.yaml              # Основная конфигурация
├── experiment/              # Конфигурации экспериментов
│   ├── default.yaml
│   ├── full_benchmark.yaml
│   └── model_comparison.yaml
├── model/                   # Конфигурации моделей
│   ├── ollama.yaml
│   └── openai.yaml
└── game/                    # Конфигурации игр
    ├── askguess.yaml
    └── spyfall.yaml
```

## Базовое использование

### 1. Запуск с базовой конфигурацией

```bash
poetry run python -m core.benchmark_hydra
```

### 2. Переопределение параметров

```bash
# Изменить модель
poetry run python -m core.benchmark_hydra model.default_model=llama3.1:latest

# Изменить эксперимент и параметры
poetry run python -m core.benchmark_hydra experiment=full_benchmark experiment.runs_per_game=2

# Указать конкретные игры
poetry run python -m core.benchmark_hydra "experiment.games=[askguess]"
```

### 3. Использование удобного скрипта

```bash
# Базовый запуск
python scripts/run_experiments.py --model ollama --model-name gemma3:latest

# С конкретными играми
python scripts/run_experiments.py --games askguess spyfall --runs 2

# Сравнение моделей
python scripts/run_experiments.py --compare-models

# Полный бенчмарк
python scripts/run_experiments.py --comprehensive
```

## Конфигурации экспериментов

### default.yaml
- Быстрый единичный запуск
- Одна игра (askguess)
- Базовые настройки

### full_benchmark.yaml
- Полное тестирование всех игр
- Множественные запуски
- Параллельное выполнение

### model_comparison.yaml
- Для сравнения разных моделей
- Настроен для sweep-ов
- Ограниченное количество фраз

## Результаты экспериментов

Hydra автоматически создает структуру директорий:

```
outputs/
└── politagent_benchmark/
    └── 2025-05-26_14-19-52/
        ├── experiment_config.yaml    # Полная конфигурация
        ├── experiment_summary.json   # Сводка результатов
        ├── benchmark_hydra.log       # Логи выполнения
        ├── results/                  # Детальные результаты
        │   └── all_results.jsonl
        ├── logs/                     # Логи игр
        ├── artifacts/                # Артефакты
        └── .hydra/                   # Служебные файлы Hydra
```

## Примеры использования

### Сравнение моделей Ollama

```bash
# Gemma3
poetry run python -m core.benchmark_hydra model.default_model=gemma3:latest

# Llama3.1
poetry run python -m core.benchmark_hydra model.default_model=llama3.1:latest

# Gemma 2B
poetry run python -m core.benchmark_hydra model.default_model=gemma:2b
```

### Тестирование конкретной игры

```bash
# Только AskGuess с ограниченными фразами
poetry run python -m core.benchmark_hydra \
  "experiment.games=[askguess]" \
  experiment.max_phrases=5 \
  experiment.runs_per_game=3
```

### Параллельное выполнение

```bash
# С 2 воркерами
poetry run python -m core.benchmark_hydra \
  experiment.workers=2 \
  experiment.runs_per_game=4
```

## Продвинутые возможности

### Sweeps (множественные эксперименты)

```bash
# Автоматическое тестирование разных моделей
poetry run python -m core.benchmark_hydra \
  --multirun \
  model.default_model=gemma3:latest,llama3.1:latest,gemma:2b
```

### Кастомные теги и описания

```bash
poetry run python -m core.benchmark_hydra \
  experiment.name=my_custom_test \
  experiment.description="Testing new approach" \
  "experiment.tags=[custom,test,new]"
```

### Отладка и логирование

```bash
# Подробные логи
poetry run python -m core.benchmark_hydra \
  output.log_level=DEBUG \
  experiment.debug=true

# Отключение определенных компонентов
poetry run python -m core.benchmark_hydra \
  benchmark.use_llm_evaluation=false
```

## Мониторинг результатов

### Просмотр сводки эксперимента

```bash
# Последний эксперимент
cat outputs/politagent_benchmark/$(ls -t outputs/politagent_benchmark/ | head -1)/experiment_summary.json | jq .

# Конкретный эксперимент
cat outputs/politagent_benchmark/2025-05-26_14-19-52/experiment_summary.json | jq .results_summary
```

### Анализ результатов

```python
import json
from pathlib import Path

# Загрузка результатов
results_dir = Path("outputs/politagent_benchmark/2025-05-26_14-19-52")
with open(results_dir / "experiment_summary.json") as f:
    summary = json.load(f)

print(f"Успешность: {summary['results_summary']['success_rate']:.2%}")
print(f"Протестированные игры: {summary['results_summary']['games_tested']}")
```

## Интеграция с MLflow (опционально)

```bash
# Установка MLflow
poetry add mlflow

# Запуск с отслеживанием
poetry run python -m core.benchmark_hydra \
  experiment.tracking.enabled=true \
  experiment.tracking.log_metrics=true
```

## Troubleshooting

### Общие проблемы

1. **Модель не найдена**: Проверьте `ollama list`
2. **Ошибки импорта**: Используйте `poetry run python -m core.benchmark_hydra`
3. **Проблемы с многопроцессностью**: Уменьшите `experiment.workers=1`

### Полезные команды

```bash
# Проверка конфигурации без запуска
poetry run python -m core.benchmark_hydra --cfg job

# Показать все возможные переопределения
poetry run python -m core.benchmark_hydra --help

# Валидация конфигурации
poetry run python -m core.benchmark_hydra --config-path=configs --config-name=config --cfg job
```

## Заключение

Hydra значительно упрощает управление экспериментами, обеспечивая:
- **Воспроизводимость**: Все конфигурации сохраняются
- **Гибкость**: Легкое переопределение параметров
- **Организованность**: Структурированное хранение результатов
- **Масштабируемость**: Поддержка sweep-ов и параллельности

Используйте готовые конфигурации как отправную точку и адаптируйте их под свои нужды! 