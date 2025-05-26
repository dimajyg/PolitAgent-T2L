#!/usr/bin/env python
"""
Удобный скрипт для запуска экспериментов PolitAgent с различными конфигурациями.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

def run_experiment(
    experiment: str = "default",
    model: str = "ollama", 
    model_name: str = "gemma3:latest",
    games: Optional[List[str]] = None,
    runs: int = 1,
    workers: int = 1,
    extra_args: Optional[List[str]] = None
) -> None:
    """
    Запускает эксперимент с заданными параметрами.
    
    Args:
        experiment: Название эксперимента (default, full_benchmark, model_comparison)
        model: Провайдер модели (ollama, openai, mistral)
        model_name: Конкретная модель
        games: Список игр для запуска
        runs: Количество запусков на игру
        workers: Количество воркеров
        extra_args: Дополнительные аргументы для Hydra
    """
    
    # Базовая команда
    cmd = [
        "poetry", "run", "python", "-m", "core.benchmark_hydra",
        f"experiment={experiment}",
        f"model={model}",
        f"model.default_model={model_name}",
        f"experiment.runs_per_game={runs}",
        f"experiment.workers={workers}"
    ]
    
    # Добавляем игры если указаны
    if games:
        games_str = ",".join(games)
        cmd.append(f"experiment.games=[{games_str}]")
    
    # Добавляем дополнительные аргументы
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"🚀 Запуск эксперимента: {experiment}")
    print(f"📊 Модель: {model} ({model_name})")
    print(f"🎮 Игры: {games or 'из конфигурации'}")
    print(f"🔄 Запуски: {runs}")
    print(f"⚡ Воркеры: {workers}")
    print(f"📝 Команда: {' '.join(cmd)}")
    print("-" * 60)
    
    # Запускаем команду
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode == 0:
        print("✅ Эксперимент завершен успешно!")
    else:
        print("❌ Эксперимент завершен с ошибкой!")
        sys.exit(result.returncode)

def run_model_comparison():
    """Запускает сравнение моделей."""
    print("🔬 Запуск сравнения моделей Ollama...")
    
    models = [
        ("ollama", "gemma3:latest"),
        ("ollama", "llama3.1:latest"),
        ("ollama", "gemma:2b")
    ]
    
    for provider, model_name in models:
        print(f"\n📊 Тестирование {provider}: {model_name}")
        run_experiment(
            experiment="default",
            model=provider,
            model_name=model_name,
            games=["askguess"],
            runs=2,
            workers=1
        )

def run_comprehensive_benchmark():
    """Запускает полный бенчмарк."""
    print("🎯 Запуск полного бенчмарка...")
    
    run_experiment(
        experiment="full_benchmark",
        model="ollama",
        model_name="gemma3:latest",
        runs=3,
        workers=2
    )

def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Запуск экспериментов PolitAgent")
    parser.add_argument("--experiment", "-e", default="default", 
                       choices=["default", "full_benchmark", "model_comparison"],
                       help="Тип эксперимента")
    parser.add_argument("--model", "-m", default="ollama",
                       choices=["ollama", "openai", "mistral"],
                       help="Провайдер модели")
    parser.add_argument("--model-name", "-n", default="gemma3:latest",
                       help="Название модели")
    parser.add_argument("--games", "-g", nargs="+", 
                       choices=["askguess", "spyfall", "beast", "tofukingdom"],
                       help="Игры для запуска")
    parser.add_argument("--runs", "-r", type=int, default=1,
                       help="Количество запусков на игру")
    parser.add_argument("--workers", "-w", type=int, default=1,
                       help="Количество воркеров")
    parser.add_argument("--compare-models", action="store_true",
                       help="Запустить сравнение моделей")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Запустить полный бенчмарк")
    
    args = parser.parse_args()
    
    if args.compare_models:
        run_model_comparison()
    elif args.comprehensive:
        run_comprehensive_benchmark()
    else:
        run_experiment(
            experiment=args.experiment,
            model=args.model,
            model_name=args.model_name,
            games=args.games,
            runs=args.runs,
            workers=args.workers
        )

if __name__ == "__main__":
    main() 