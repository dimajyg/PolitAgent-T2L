#!/usr/bin/env python3
"""
Простой скрипт для запуска бенчмарка PolitAgent.

Использование:
    python run_benchmark.py                    # Запуск с настройками по умолчанию
    python run_benchmark.py --profile quick    # Быстрый тест
    python run_benchmark.py --profile full     # Полный тест
    python run_benchmark.py --list-profiles    # Показать доступные профили
"""

import asyncio
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent))

from scripts.benchmark_config_runner import main

if __name__ == "__main__":
    asyncio.run(main()) 