#!/usr/bin/env python
"""
PolitAgent Benchmark - единый интерфейс для запуска и оценки языковых моделей
на различных игровых средах.
"""

import argparse
import json
import os
import multiprocessing
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import importlib
import random

# Заменяем импорт на унифицированный интерфейс моделей
from llm.models import get_model, get_available_models

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark")

# Настройки игровых сред
GAME_ENVIRONMENTS = {
    "spyfall": {
        "module": "environments.spyfall.game",
        "class": "SpyfallGame",
        "default_args": {
            "label_path": "environments/spyfall/prompts/labels.txt", 
            "spy_model_name": "openai",
            "villager_model_name": "openai",
            "n": 10,
            "debug": True,
            "openai_api_key": None,
            "embedding_model": "auto",
            "embedding_model_name": "text-embedding-3-large",
            "perplexity_model": "auto"
        },
        "requires_phrases": True,
        "model_args": ["spy_model_name", "villager_model_name"]
    },
    "beast": {
        "module": "environments.beast.game",
        "class": "BeastGame",
        "default_args": {
            "max_rounds": 5,
            "debug": True,
            "output_dir": "./results/beast",
            "model_name": "openai"
        },
        "requires_phrases": False,
        "model_args": ["model_name"]
    },
    "askguess": {
        "module": "environments.askguess.game",
        "class": "AskGuessGame",
        "default_args": {
            "label_path": "environments/askguess/test_labels.json",
            "mode": "hard",
            "n": 1,
            "max_rounds": 10,
            "debug": True,
            "model_name": "openai"
        },
        "requires_phrases": True,
        "model_args": ["model_name"]
    },
    "tofukingdom": {
        "module": "environments.tofukingdom.game",
        "class": "TofuKingdomGame",
        "default_args": {
            "debug": True,
            "n_players": 5,
            "model_name": "openai",
            "prince_model_name": "openai",
            "princess_model_name": "openai",
            "queen_model_name": "openai",
            "neutral_model_name": "openai"
        },
        "requires_phrases": False,
        "model_args": ["prince_model_name", "princess_model_name", "queen_model_name", "neutral_model_name"]
    },
    "diplomacy": {
        "module": "environments.diplomacy_game.game",
        "class": "DiplomacyGame",
        "default_args": {
            "max_rounds": 5,
            "debug": True,
            "diplomacy_model_name": "openai"
        },
        "requires_phrases": False,
        "model_args": ["diplomacy_model_name"]
    }
}

# Доступные модели из унифицированного интерфейса
AVAILABLE_MODELS = list(get_available_models().keys())

def setup_results_dir() -> str:
    """Создает и возвращает директорию для сохранения результатов."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"benchmark_results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def load_phrases(game_type: str, args: argparse.Namespace) -> List[Any]:
    """Загружает фразы/слова для игр, требующих внешние данные."""
    if game_type == "spyfall":
        with open(args.label_path, 'r') as f:
            phrases = [line.strip().split(",") for line in f.readlines()]
            # Если указан параметр max_phrases, ограничиваем количество фраз
            if hasattr(args, 'max_phrases') and args.max_phrases is not None:
                return phrases[:args.max_phrases]
            return phrases
    elif game_type == "askguess":
        with open(args.label_path, 'r') as f:
            try:
                # Пытаемся загрузить как JSON
                all_phrases = json.load(f)
                # Проверяем, является ли результат списком или словарем
                if isinstance(all_phrases, list):
                    # Если список, ограничиваем количество элементов при необходимости
                    if hasattr(args, 'max_phrases') and args.max_phrases is not None:
                        return all_phrases[:args.max_phrases]
                    return all_phrases
                else:
                    # Если словарь, обрабатываем как раньше
                    if hasattr(args, 'max_phrases') and args.max_phrases is not None:
                        keys = list(all_phrases.keys())[:args.max_phrases]
                        return {k: all_phrases[k] for k in keys}
                    return all_phrases
            except json.JSONDecodeError:
                # В случае ошибки декодирования JSON, пытаемся прочитать как простой текстовый файл
                f.seek(0)
                phrases = [line.strip() for line in f.readlines()]
                if hasattr(args, 'max_phrases') and args.max_phrases is not None:
                    return phrases[:args.max_phrases]
                return phrases
    return [None]  # Для игр без фраз

def run_game(game_config: Tuple[str, Dict, Any, int, str]) -> Dict[str, Any]:
    """
    Запускает одну игровую сессию.
    
    Args:
        game_config: (тип_игры, аргументы, фраза, номер_запуска, директория_результатов)
        
    Returns:
        Dict с результатами игры
    """
    game_type, args_dict, phrase, run_id, results_dir = game_config
    
    # Преобразуем словарь аргументов обратно в Namespace
    args = argparse.Namespace(**args_dict)
    
    # Получаем информацию о среде
    game_info = GAME_ENVIRONMENTS[game_type]
    
    # Инициализируем модели
    models = {}
    for model_arg in game_info["model_args"]:
        model_name = args_dict.get(model_arg, "openai")
        if model_name not in models:
            # Добавляем поддержку specific_model и ollama_base_url
            model_kwargs = {}
            if args_dict.get("specific_model"):
                model_kwargs["specific_model"] = args_dict.get("specific_model")
            if model_name == "ollama" and args_dict.get("ollama_base_url"):
                model_kwargs["base_url"] = args_dict.get("ollama_base_url")
            
            models[model_name] = get_model(model_name, **model_kwargs)
    
    # Инициализируем модель для LLM-оценки, если она задана
    if args_dict.get("use_llm_evaluation", False) and args_dict.get("evaluation_model") is not None:
        evaluation_model_name = args_dict.get("evaluation_model")
        if evaluation_model_name not in models:
            # Добавляем поддержку specific_model и ollama_base_url для модели оценки
            model_kwargs = {}
            if args_dict.get("specific_model"):
                model_kwargs["specific_model"] = args_dict.get("specific_model")
            if evaluation_model_name == "ollama" and args_dict.get("ollama_base_url"):
                model_kwargs["base_url"] = args_dict.get("ollama_base_url")
                
            models["evaluation_model"] = get_model(evaluation_model_name, **model_kwargs)
            args.evaluation_model = models["evaluation_model"]
    
    # Импортируем нужный модуль и класс игры
    game_module = importlib.import_module(game_info["module"])
    game_class = getattr(game_module, game_info["class"])
    
    # Создаем директорию для логов
    model_names = "_".join([args_dict.get(arg, "default") for arg in game_info["model_args"]])
    log_dir = f"{results_dir}/{game_type}/{model_names}"
    
    if phrase:
        if isinstance(phrase, list):
            phrase_str = "&".join(phrase)
        else:
            phrase_str = phrase
        log_dir = f"{log_dir}/{phrase_str}"
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Устанавливаем переменную окружения с путем к директории результатов
    os.environ["BENCHMARK_RESULTS_DIR"] = log_dir
    
    # Создаем экземпляр игры в зависимости от типа
    if game_type == "spyfall":
        spy_model = models[args.spy_model_name]
        villager_model = models[args.villager_model_name]
        game = game_class(args, spy_model, villager_model)
        settings = game.init_game(phrase)
    
    elif game_type == "beast":
        game = game_class(args, models[args.model_name])
        settings = game.init_game()
    
    elif game_type == "askguess":
        game = game_class(args, models[args.model_name])
        game.init_game(phrase)
        settings = f"Word: {phrase}"
    
    elif game_type == "tofukingdom":
        prince_model = models[args.prince_model_name]
        
        # Get other models, defaulting to prince model if not specified
        princess_model = models.get(args.princess_model_name, prince_model)
        queen_model = models.get(args.queen_model_name, prince_model)
        neutral_model = models.get(args.neutral_model_name, prince_model)
        
        game = game_class(args, prince_model, princess_model, queen_model, neutral_model)
        settings = game.init_game()
    
    elif game_type == "diplomacy":
        game = game_class(args, models[args.diplomacy_model_name])
        settings = game.init_game()
    
    else:
        logger.error(f"Неизвестный тип игры: {game_type}")
        return {"error": f"Неизвестный тип игры: {game_type}"}
    
    # Запускаем игру и записываем логи
    with open(f"{log_dir}/{run_id}.log", "w") as f:
        f.write(f"Game: {game_type}\n")
        f.write(f"Models: {model_names}\n")
        f.write(f"Settings: {settings}\n")
        f.write("-" * 80 + "\n")
        
        result = game.game_loop(f)
        
        if result is not None:
            result["log"] = f"{run_id}.log"
            result["game_type"] = game_type
            result["model"] = model_names
            result["timestamp"] = datetime.now().isoformat()
            
            # Записываем результат в отдельный файл
            with open(f"{log_dir}/{run_id}_result.json", "w") as rf:
                json.dump(result, rf, indent=2)
                
            # Добавляем в общий файл результатов
            with open(f"{results_dir}/all_results.jsonl", "a") as arf:
                arf.write(json.dumps(result) + "\n")
    
    logger.info(f"Завершена игра {game_type} (run {run_id}) с {'фразой ' + str(phrase) if phrase else 'без фразы'}")
    return result

def run_benchmark(args: argparse.Namespace) -> None:
    """Запускает бенчмарк игр на основе аргументов командной строки."""
    results_dir = setup_results_dir()
    logger.info(f"Результаты будут сохранены в {results_dir}")
    
    # Сохраняем конфигурацию запуска
    with open(f"{results_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Определяем, какие игры запускать
    games_to_run = args.games.split(",") if args.games else list(GAME_ENVIRONMENTS.keys())
    
    # Проверяем валидность игр
    for game in games_to_run:
        if game not in GAME_ENVIRONMENTS:
            logger.error(f"Неизвестная игра: {game}. Доступные игры: {', '.join(GAME_ENVIRONMENTS.keys())}")
            return
    
    # Проверяем валидность моделей
    if args.models:
        models = args.models.split(",")
        for model in models:
            if model not in AVAILABLE_MODELS:
                logger.error(f"Неизвестная модель: {model}. Доступные модели: {', '.join(AVAILABLE_MODELS)}")
                return
    
    # Формируем задания для запуска
    tasks = []
    
    for game_type in games_to_run:
        # Получаем информацию о среде
        game_info = GAME_ENVIRONMENTS[game_type]
        
        # Создаем копию аргументов для конкретной игры
        game_args = vars(args).copy()
        
        # Обновляем значения по умолчанию из конфигурации игры
        for key, value in game_info["default_args"].items():
            if key not in game_args or game_args[key] is None:
                game_args[key] = value
            elif key == "label_path" and game_type == "askguess" and game_args[key] == "environments/spyfall/prompts/labels.txt":
                # Исправление: не используем путь от spyfall для askguess
                game_args[key] = game_info["default_args"][key]
        
        # Если установлены общие модели, применяем их для всех модельных аргументов
        if args.models:
            models_list = args.models.split(",")
            for model_arg in game_info["model_args"]:
                if len(models_list) > 1:
                    # Используем разные модели для разных ролей, если их указано несколько
                    game_args[model_arg] = random.choice(models_list)
                else:
                    # Иначе используем одну модель для всех ролей
                    game_args[model_arg] = models_list[0]
        
        # Загружаем фразы, если требуется
        if game_info["requires_phrases"]:
            phrases = load_phrases(game_type, argparse.Namespace(**game_args))
        else:
            phrases = [None]
        
        # Формируем задания для каждой фразы и прогона
        for phrase in phrases:
            for run_id in range(args.runs_per_game):
                tasks.append((game_type, game_args, phrase, run_id, results_dir))
    
    # Запускаем задания
    logger.info(f"Запуск {len(tasks)} игровых сессий с использованием {args.workers} рабочих процессов")
    
    if args.workers > 1:
        # Многопроцессный режим
        with multiprocessing.Pool(args.workers) as pool:
            results = pool.map(run_game, tasks)
    else:
        # Последовательный режим
        results = [run_game(task) for task in tasks]
    
    # Сохраняем сводный отчет
    summary = {
        "total_games": len(tasks),
        "completed_games": sum(1 for r in results if r is not None and "error" not in r),
        "games_by_type": {},
        "timestamp": datetime.now().isoformat()
    }
    
    for game_type in games_to_run:
        game_results = [r for r in results if r is not None and r.get("game_type") == game_type]
        summary["games_by_type"][game_type] = {
            "total": len(game_results),
            "successful": sum(1 for r in game_results if "error" not in r)
        }
    
    with open(f"{results_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Бенчмарк завершен. Всего запущено {len(tasks)} игр, успешно завершено {summary['completed_games']}.")
    logger.info(f"Результаты доступны в {results_dir}")

def main():
    """Точка входа программы."""
    parser = argparse.ArgumentParser(description="PolitAgent Benchmark - оценка языковых моделей в игровых средах")
    
    # Общие аргументы
    parser.add_argument('--models', type=str, default=None, 
                        help="Общий список моделей через запятую (openai,mistral)")
    parser.add_argument('--games', type=str, default=None,
                        help="Игры для запуска через запятую (spyfall,beast,askguess,tofukingdom). По умолчанию - все.")
    parser.add_argument('--workers', type=int, default=1,
                        help="Количество параллельных процессов")
    parser.add_argument('--runs_per_game', type=int, default=1,
                        help="Количество запусков каждой комбинации игра/фраза")
    parser.add_argument('--debug', type=bool, default=False,
                        help="Режим отладки с подробным выводом")
    
    # Добавляем параметр max_phrases для ограничения количества фраз
    parser.add_argument('--max_phrases', type=int, default=None,
                        help="Максимальное количество фраз для игр с фразами (spyfall, askguess)")
    
    # Параметры для LLM-оценки
    parser.add_argument('--use_llm_evaluation', type=bool, default=False,
                        help="Включить оценку игрового процесса с помощью LLM")
    parser.add_argument('--evaluation_model', type=str, default=None,
                        help="Модель для оценки игрового процесса. По умолчанию используется основная модель игры.")
    
    # Параметры для моделей
    parser.add_argument('--specific_model', type=str, default=None,
                        help="Конкретная модель провайдера (например, 'gpt-4' для OpenAI или 'llama2' для Ollama)")
    parser.add_argument('--ollama_base_url', type=str, default="http://localhost:11434",
                        help="URL для доступа к Ollama API (по умолчанию http://localhost:11434)")
    
    # Аргументы для Spyfall
    parser.add_argument('--label_path', type=str, default="environments/spyfall/prompts/labels.txt",
                        help="Путь к файлу с фразами для Spyfall")
    parser.add_argument('--spy_model_name', type=str, default=None,
                        help="Модель для шпиона (используется в Spyfall и TofuKingdom)")
    parser.add_argument('--villager_model_name', type=str, default=None,
                        help="Модель для жителей в Spyfall")
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help="OpenAI API ключ (переопределяет переменную окружения)")
    parser.add_argument('--embedding_model', type=str, default=None,
                        help="Тип модели эмбеддингов (local, openai, auto)")
    parser.add_argument('--embedding_model_name', type=str, default=None,
                        help="Название модели эмбеддингов")
    parser.add_argument('--perplexity_model', type=str, default=None,
                        help="Модель для расчета перплексии (auto/local/model name)")
    
    # Аргументы для Beast
    parser.add_argument('--model_name', type=str, default=None,
                        help="Название модели для Beast и AskGuess")
    
    # Аргументы для AskGuess
    parser.add_argument('--mode', type=str, default=None,
                        help="Режим игры (используется в askguess)")
    parser.add_argument('--max_rounds', type=int, default=None,
                        help="Максимальное количество раундов в игре AskGuess")
    
    # Аргументы для TofuKingdom
    parser.add_argument('--prince_model_name', type=str, default=None,
                        help="Модель для принца в TofuKingdom")
    parser.add_argument('--princess_model_name', type=str, default=None,
                        help="Модель для принцессы в TofuKingdom")
    parser.add_argument('--queen_model_name', type=str, default=None,
                        help="Модель для королевы в TofuKingdom")
    parser.add_argument('--neutral_model_name', type=str, default=None,
                        help="Модель для нейтрального персонажа в TofuKingdom")
    
    # Аргументы для Diplomacy
    parser.add_argument('--diplomacy_model_name', type=str, default=None,
                        help="Модель для игроков в Diplomacy")
    
    args = parser.parse_args()
    
    # Установка уровня логгирования
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    # Запуск бенчмарка
    run_benchmark(args)

if __name__ == "__main__":
    main() 