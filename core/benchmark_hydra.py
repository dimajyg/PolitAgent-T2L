#!/usr/bin/env python
"""
PolitAgent Benchmark с поддержкой Hydra - улучшенный интерфейс для управления экспериментами.
"""

import logging
import os
import json
import multiprocessing
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import importlib
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import traceback

from llm.models import get_model, get_available_models

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark_hydra")

# Импорт игровых сред из оригинального benchmark.py
from core.benchmark import GAME_ENVIRONMENTS, load_phrases, run_game

class HydraExperimentTracker:
    """Класс для отслеживания экспериментов с Hydra."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.experiment_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.results: List[Dict[str, Any]] = []
        
        # Создаем директории для результатов
        self.results_dir = self.experiment_dir / "results"
        self.logs_dir = self.experiment_dir / "logs"
        self.artifacts_dir = self.experiment_dir / "artifacts"
        
        for dir_path in [self.results_dir, self.logs_dir, self.artifacts_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def log_experiment_config(self):
        """Сохраняет конфигурацию эксперимента."""
        config_path = self.experiment_dir / "experiment_config.yaml"
        with open(config_path, 'w') as f:
            OmegaConf.save(self.cfg, f)
        logger.info(f"Конфигурация сохранена в {config_path}")
    
    def log_result(self, result: Dict[str, Any]):
        """Добавляет результат в отслеживание."""
        result["timestamp"] = datetime.now().isoformat()
        result["experiment_name"] = self.cfg.experiment.name
        self.results.append(result)
        
        # Сохраняем результат сразу в JSONL файл
        with open(self.results_dir / "all_results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
    
    def save_summary(self):
        """Сохраняет сводку эксперимента."""
        summary = {
            "experiment": {
                "name": self.cfg.experiment.name,
                "description": self.cfg.experiment.description,
                "total_results": len(self.results),
                "start_time": datetime.now().isoformat(),
                "config": OmegaConf.to_container(self.cfg)
            },
            "results_summary": self._compute_summary_stats(),
            "games_performance": self._compute_game_performance()
        }
        
        summary_path = self.experiment_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Сводка эксперимента сохранена в {summary_path}")
    
    def _compute_summary_stats(self) -> Dict[str, Any]:
        """Вычисляет общую статистику."""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if "error" not in r]
        return {
            "total_runs": len(self.results),
            "successful_runs": len(successful),
            "success_rate": len(successful) / len(self.results) if self.results else 0,
            "games_tested": list(set(r.get("game_type") for r in self.results if r.get("game_type")))
        }
    
    def _compute_game_performance(self) -> Dict[str, Any]:
        """Вычисляет производительность по играм."""
        game_stats = {}
        for result in self.results:
            game_type = result.get("game_type")
            if not game_type:
                continue
                
            if game_type not in game_stats:
                game_stats[game_type] = {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "results": []
                }
            
            game_stats[game_type]["total_runs"] += 1
            if "error" not in result:
                game_stats[game_type]["successful_runs"] += 1
            game_stats[game_type]["results"].append(result)
        
        # Добавляем success rate для каждой игры
        for game_type, stats in game_stats.items():
            stats["success_rate"] = stats["successful_runs"] / stats["total_runs"] if stats["total_runs"] > 0 else 0
        
        return game_stats

def setup_model_from_config(cfg: DictConfig) -> Dict[str, Any]:
    """Создает модель на основе конфигурации Hydra."""
    model_config = cfg.model
    
    # Основные параметры модели
    model_kwargs = {
        "temperature": model_config.settings.temperature,
    }
    
    # Добавляем специфичные параметры для провайдера
    if model_config.provider == "ollama":
        model_kwargs.update({
            "base_url": model_config.connection.base_url,
            "specific_model": model_config.default_model
        })
    elif model_config.provider == "openai":
        if "api" in model_config and "key" in model_config.api:
            model_kwargs["api_key"] = model_config.api.key
        model_kwargs["specific_model"] = model_config.default_model
    
    return get_model(model_config.provider, **model_kwargs)

def prepare_game_configs_from_hydra(cfg: DictConfig, tracker: HydraExperimentTracker) -> List[tuple]:
    """Подготавливает конфигурации игр на основе Hydra config."""
    tasks = []
    
    # Получаем список игр из конфигурации эксперимента
    games_to_run = cfg.experiment.games if hasattr(cfg.experiment, 'games') else ["askguess"]
    
    for game_type in games_to_run:
        if game_type not in GAME_ENVIRONMENTS:
            logger.error(f"Неизвестная игра: {game_type}")
            continue
        
        game_info = GAME_ENVIRONMENTS[game_type]
        
        # Создаем аргументы для игры
        game_args = {}
        
        # Основные параметры из конфигурации
        game_args.update({
            "debug": cfg.experiment.get("debug", False),
            "workers": cfg.experiment.get("workers", 1),
            "runs_per_game": cfg.experiment.get("runs_per_game", 1),
            "max_phrases": cfg.experiment.get("max_phrases", None),
        })
        
        # Добавляем дефолтные параметры игры
        game_args.update(game_info["default_args"])
        
        # Переопределяем специфичными параметрами модели
        for model_arg in game_info["model_args"]:
            game_args[model_arg] = cfg.model.provider
        
        # Специфичные параметры модели
        if cfg.model.provider == "ollama":
            game_args["specific_model"] = cfg.model.default_model
            game_args["ollama_base_url"] = cfg.model.connection.base_url
        
        # Загружаем фразы для игр, которым они нужны
        if game_info["requires_phrases"]:
            try:
                # Используем путь из конфигурации игры, если доступен
                if hasattr(cfg, 'game') and hasattr(cfg.game, 'settings') and hasattr(cfg.game.settings, 'label_path'):
                    game_args["label_path"] = cfg.game.settings.label_path
                
                phrases = load_phrases(game_type, type('Args', (), game_args)())
                if cfg.experiment.get("max_phrases"):
                    phrases = phrases[:cfg.experiment.max_phrases]
            except Exception as e:
                logger.warning(f"Не удалось загрузить фразы для {game_type}: {e}")
                phrases = [None]
        else:
            phrases = [None]
        
        # Создаем задачи для каждой фразы и каждого запуска
        for phrase in phrases:
            for run_id in range(cfg.experiment.runs_per_game):
                tasks.append((game_type, game_args, phrase, run_id, str(tracker.results_dir)))
    
    return tasks

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def run_hydra_benchmark(cfg: DictConfig) -> None:
    """Основная функция для запуска бенчмарка с Hydra."""
    
    # Настройка логирования
    if cfg.output.log_level:
        logging.getLogger().setLevel(getattr(logging, cfg.output.log_level))
    
    logger.info("🚀 Запуск PolitAgent Benchmark с Hydra")
    logger.info(f"Эксперимент: {cfg.experiment.name}")
    logger.info(f"Описание: {cfg.experiment.description}")
    logger.info(f"Модель: {cfg.model.provider} ({cfg.model.default_model})")
    
    # Создаем трекер эксперимента
    tracker = HydraExperimentTracker(cfg)
    tracker.log_experiment_config()
    
    # Подготавливаем задачи
    tasks = prepare_game_configs_from_hydra(cfg, tracker)
    logger.info(f"Подготовлено {len(tasks)} задач для выполнения")
    
    # Устанавливаем seed для воспроизводимости
    if cfg.get("seed"):
        random.seed(cfg.seed)
        import numpy as np
        np.random.seed(cfg.seed)
    
    # Запуск задач
    if cfg.experiment.workers > 1:
        logger.info(f"Запуск в многопроцессном режиме с {cfg.experiment.workers} воркерами")
        with multiprocessing.Pool(cfg.experiment.workers) as pool:
            results = pool.map(run_game, tasks)
    else:
        logger.info("Запуск в последовательном режиме")
        results = [run_game(task) for task in tasks]
    
    # Обработка результатов
    for result in results:
        if result:
            tracker.log_result(result)
    
    # Сохранение итоговой статистики
    tracker.save_summary()
    
    # Финальный отчет
    summary_stats = tracker._compute_summary_stats()
    logger.info("=" * 60)
    logger.info("🎯 РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА")
    logger.info("=" * 60)
    logger.info(f"Всего запусков: {summary_stats.get('total_runs', 0)}")
    logger.info(f"Успешных запусков: {summary_stats.get('successful_runs', 0)}")
    logger.info(f"Успешность: {summary_stats.get('success_rate', 0):.2%}")
    logger.info(f"Протестированные игры: {', '.join(summary_stats.get('games_tested', []))}")
    logger.info(f"Результаты сохранены в: {tracker.experiment_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_hydra_benchmark() 