#!/usr/bin/env python3
"""
Система бенчмаркинга LLM моделей на всех играх PolitAgent environments.

Этот модуль позволяет:
- Запускать множество моделей параллельно на всех играх
- Собирать метрики по каждой модели и игре
- Вычислять общий рейтинг моделей
- Генерировать сравнительные отчеты
"""

import asyncio
import concurrent.futures
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import traceback

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from llm.models import get_model, AVAILABLE_MODELS
from metrics import METRICS_MAP
from environments.diplomacy_game.game import DiplomacyGame
from environments.beast.game import BeastGame
from environments.spyfall.game import SpyfallGame
from environments.askguess.game import AskGuessGame
from environments.tofukingdom.game import TofuKingdomGame

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Конфигурация модели для бенчмарка."""
    provider: str  # 'openai', 'mistral', 'ollama'
    model_name: str  # конкретная модель
    display_name: str  # отображаемое имя
    temperature: float = 0.7
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None
    enabled: bool = True

@dataclass
class GameConfig:
    """Конфигурация игры для бенчмарка."""
    name: str
    game_class: Any
    num_games: int = 3  # количество игр для усреднения
    max_rounds: int = 10
    enabled: bool = True

@dataclass
class BenchmarkResult:
    """Результат бенчмарка одной модели на одной игре."""
    model_name: str
    game_name: str
    game_results: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    execution_time: float
    success_rate: float
    error_count: int
    timestamp: str

@dataclass
class ModelRating:
    """Рейтинг модели."""
    model_name: str
    overall_score: float
    game_scores: Dict[str, float]
    total_games: int
    success_rate: float
    avg_execution_time: float
    rank: int = 0

class BenchmarkRunner:
    """Основной класс для запуска бенчмарков."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Инициализация игр
        self.games = {
            "diplomacy": GameConfig("diplomacy", DiplomacyGame, num_games=2, max_rounds=5),
            "beast": GameConfig("beast", BeastGame, num_games=3, max_rounds=8),
            "spyfall": GameConfig("spyfall", SpyfallGame, num_games=3, max_rounds=6),
            "askguess": GameConfig("askguess", AskGuessGame, num_games=4, max_rounds=10),
            "tofukingdom": GameConfig("tofukingdom", TofuKingdomGame, num_games=3, max_rounds=8)
        }
        
        # Дефолтные модели для тестирования
        self.models = self._get_default_models()
        
        # Веса для расчета общего рейтинга
        self.game_weights = {
            "diplomacy": 0.25,    # Сложная стратегическая игра
            "beast": 0.20,        # Социальная дедукция
            "spyfall": 0.20,      # Быстрая дедукция
            "askguess": 0.15,     # Логические вопросы
            "tofukingdom": 0.20   # Ролевая дедукция
        }
    
    def _get_default_models(self) -> List[ModelConfig]:
        """Получает список моделей по умолчанию для тестирования."""
        models = []
        
        # OpenAI модели
        if os.getenv("OPENAI_API_KEY"):
            models.extend([
                ModelConfig("openai", "gpt-3.5-turbo", "GPT-3.5 Turbo"),
                ModelConfig("openai", "gpt-4-turbo", "GPT-4 Turbo", temperature=0.7),
            ])
        
        # Mistral модели
        if os.getenv("MISTRAL_API_KEY"):
            models.extend([
                ModelConfig("mistral", "mistral-tiny", "Mistral Tiny"),
                ModelConfig("mistral", "mistral-small", "Mistral Small"),
            ])
        
        # Ollama модели (если локально запущен)
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                ollama_models = response.json().get('models', [])
                available_names = [m['name'].split(':')[0] for m in ollama_models]
                
                for model_name in ['llama2', 'mistral', 'phi2', 'gemma']:
                    if model_name in available_names:
                        models.append(ModelConfig("ollama", model_name, f"Ollama {model_name.title()}"))
        except:
            logger.warning("Ollama не доступен, пропускаем локальные модели")
        
        if not models:
            logger.warning("Не найдено доступных моделей! Добавьте API ключи или запустите Ollama")
            # Добавляем тестовую заглушку
            models.append(ModelConfig("test", "dummy", "Test Dummy Model", enabled=False))
        
        return models
    
    def add_model(self, provider: str, model_name: str, display_name: str, **kwargs) -> None:
        """Добавляет модель в бенчмарк."""
        self.models.append(ModelConfig(provider, model_name, display_name, **kwargs))
    
    def configure_game(self, game_name: str, **kwargs) -> None:
        """Настраивает параметры игры."""
        if game_name in self.games:
            for key, value in kwargs.items():
                if hasattr(self.games[game_name], key):
                    setattr(self.games[game_name], key, value)
    
    async def run_single_game(self, model_config: ModelConfig, game_config: GameConfig, 
                             game_index: int) -> Tuple[Dict[str, Any], float]:
        """Запускает одну игру для одной модели."""
        start_time = time.time()
        
        try:
            # Создаем модель
            model = get_model(
                model_config.provider,
                specific_model=model_config.model_name,
                temperature=model_config.temperature,
                api_key=model_config.api_key
            )
            
            # Создаем аргументы для игры
            class Args:
                def __init__(self):
                    self.max_rounds = game_config.max_rounds
                    self.debug = False
                    self.num_players = 4  # дефолт для большинства игр
            
            args = Args()
            
            # Запускаем игру
            game = game_config.game_class(args, model)
            
            # Инициализируем игру
            game.init_game()
            
            # Запускаем игровой цикл
            result = game.game_loop()
            
            execution_time = time.time() - start_time
            
            # Добавляем метаданные
            result.update({
                "model_provider": model_config.provider,
                "model_name": model_config.model_name,
                "game_name": game_config.name,
                "game_index": game_index,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            })
            
            return result, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Ошибка в игре {game_config.name} с моделью {model_config.display_name}: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "error": str(e),
                "model_provider": model_config.provider,
                "model_name": model_config.model_name,
                "game_name": game_config.name,
                "game_index": game_index,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }, execution_time
    
    async def benchmark_model_on_game(self, model_config: ModelConfig, 
                                    game_config: GameConfig) -> BenchmarkResult:
        """Бенчмарк одной модели на одной игре."""
        logger.info(f"Запуск бенчмарка: {model_config.display_name} на {game_config.name}")
        
        game_results = []
        total_time = 0
        error_count = 0
        
        # Запускаем несколько игр для усреднения
        for i in range(game_config.num_games):
            result, exec_time = await self.run_single_game(model_config, game_config, i)
            game_results.append(result)
            total_time += exec_time
            
            if "error" in result:
                error_count += 1
        
        # Сохраняем результаты игр
        game_results_file = (self.results_dir / 
                           f"{model_config.provider}_{model_config.model_name}_{game_config.name}_games.json")
        with open(game_results_file, 'w', encoding='utf-8') as f:
            json.dump(game_results, f, indent=2, ensure_ascii=False)
        
        # Вычисляем метрики
        try:
            metrics_class = METRICS_MAP[game_config.name]
            metrics = metrics_class()
            metrics_data = metrics.calculate_metrics(str(game_results_file.parent))
            
            # Сохраняем метрики
            metrics_file = (self.results_dir / 
                          f"{model_config.provider}_{model_config.model_name}_{game_config.name}_metrics.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Ошибка в расчете метрик для {game_config.name}: {e}")
            metrics_data = {"error": str(e)}
        
        # Вычисляем success rate
        successful_games = len([r for r in game_results if "error" not in r])
        success_rate = successful_games / len(game_results)
        
        return BenchmarkResult(
            model_name=model_config.display_name,
            game_name=game_config.name,
            game_results=game_results,
            metrics=metrics_data,
            execution_time=total_time,
            success_rate=success_rate,
            error_count=error_count,
            timestamp=datetime.now().isoformat()
        )
    
    async def run_benchmark(self, max_parallel_models: int = 3, 
                          max_parallel_games: int = 2) -> List[BenchmarkResult]:
        """Запускает полный бенчмарк всех моделей на всех играх."""
        logger.info("Запуск полного бенчмарка")
        
        # Фильтруем включенные модели и игры
        active_models = [m for m in self.models if m.enabled]
        active_games = [g for g in self.games.values() if g.enabled]
        
        logger.info(f"Тестируем {len(active_models)} моделей на {len(active_games)} играх")
        
        all_results = []
        
        # Семафоры для ограничения параллельности
        model_semaphore = asyncio.Semaphore(max_parallel_models)
        game_semaphore = asyncio.Semaphore(max_parallel_games)
        
        async def run_model_benchmark(model: ModelConfig) -> List[BenchmarkResult]:
            async with model_semaphore:
                model_results = []
                
                # Запускаем все игры для одной модели
                tasks = []
                for game in active_games:
                    async def run_game_with_semaphore(m=model, g=game):
                        async with game_semaphore:
                            return await self.benchmark_model_on_game(m, g)
                    tasks.append(run_game_with_semaphore())
                
                game_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in game_results:
                    if isinstance(result, Exception):
                        logger.error(f"Ошибка в бенчмарке модели {model.display_name}: {result}")
                    else:
                        model_results.append(result)
                
                return model_results
        
        # Запускаем бенчмарки всех моделей
        model_tasks = [run_model_benchmark(model) for model in active_models]
        all_model_results = await asyncio.gather(*model_tasks, return_exceptions=True)
        
        # Собираем все результаты
        for model_results in all_model_results:
            if isinstance(model_results, Exception):
                logger.error(f"Ошибка в бенчмарке модели: {model_results}")
            else:
                all_results.extend(model_results)
        
        return all_results
    
    def calculate_model_ratings(self, benchmark_results: List[BenchmarkResult]) -> List[ModelRating]:
        """Вычисляет рейтинг моделей на основе результатов бенчмарка."""
        logger.info("Вычисление рейтингов моделей")
        
        # Группируем результаты по моделям
        model_results = {}
        for result in benchmark_results:
            if result.model_name not in model_results:
                model_results[result.model_name] = []
            model_results[result.model_name].append(result)
        
        ratings = []
        
        for model_name, results in model_results.items():
            game_scores = {}
            total_success_rate = 0
            total_execution_time = 0
            total_games = 0
            
            for result in results:
                game_name = result.game_name
                
                # Базовый скор на основе success rate
                base_score = result.success_rate * 100
                
                # Бонусы за качество из метрик
                quality_bonus = 0
                if "model_performance" in result.metrics:
                    perf = result.metrics["model_performance"]
                    if "average_quality_score" in perf:
                        quality_bonus += perf["average_quality_score"] * 20  # до 20 очков
                    if "decision_consistency" in perf:
                        quality_bonus += perf["decision_consistency"] * 10  # до 10 очков
                
                # Бонус за эффективность (быстрое выполнение)
                efficiency_bonus = 0
                if result.execution_time > 0:
                    # Нормализуем время (меньше времени = больше бонус)
                    max_time = 300  # 5 минут максимум
                    efficiency_bonus = max(0, (max_time - result.execution_time) / max_time * 10)
                
                # Штраф за ошибки
                error_penalty = result.error_count * 10
                
                # Итоговый скор для игры
                game_score = max(0, base_score + quality_bonus + efficiency_bonus - error_penalty)
                game_scores[game_name] = game_score
                
                total_success_rate += result.success_rate
                total_execution_time += result.execution_time
                total_games += 1
            
            # Взвешенный общий скор
            overall_score = sum(game_scores.get(game, 0) * weight 
                              for game, weight in self.game_weights.items())
            
            avg_success_rate = total_success_rate / len(results) if results else 0
            avg_execution_time = total_execution_time / len(results) if results else 0
            
            ratings.append(ModelRating(
                model_name=model_name,
                overall_score=overall_score,
                game_scores=game_scores,
                total_games=total_games,
                success_rate=avg_success_rate,
                avg_execution_time=avg_execution_time
            ))
        
        # Сортируем по общему скору и присваиваем ранки
        ratings.sort(key=lambda r: r.overall_score, reverse=True)
        for i, rating in enumerate(ratings):
            rating.rank = i + 1
        
        return ratings
    
    def generate_benchmark_report(self, benchmark_results: List[BenchmarkResult], 
                                ratings: List[ModelRating]) -> str:
        """Генерирует итоговый отчет по бенчмарку."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# PolitAgent Benchmark Report
Generated: {timestamp}

## Executive Summary

Протестировано **{len(set(r.model_name for r in benchmark_results))}** моделей на **{len(set(r.game_name for r in benchmark_results))}** играх.

### Top 3 Models:
"""
        
        for i, rating in enumerate(ratings[:3]):
            report += f"{i+1}. **{rating.model_name}** - {rating.overall_score:.1f} очков\n"
        
        report += f"\n## Detailed Rankings\n\n"
        report += "| Rank | Model | Overall Score | Success Rate | Avg Time |\n"
        report += "|------|-------|---------------|--------------|----------|\n"
        
        for rating in ratings:
            report += (f"| {rating.rank} | {rating.model_name} | "
                      f"{rating.overall_score:.1f} | {rating.success_rate:.1%} | "
                      f"{rating.avg_execution_time:.1f}s |\n")
        
        # Детальные результаты по играм
        report += f"\n## Performance by Game\n\n"
        
        for game_name in self.game_weights.keys():
            report += f"### {game_name.title()}\n\n"
            report += "| Rank | Model | Score | Success Rate |\n"
            report += "|------|-------|-------|-------------|\n"
            
            # Сортируем модели по скору в этой игре
            game_ratings = [(r.model_name, r.game_scores.get(game_name, 0), 
                           next((br.success_rate for br in benchmark_results 
                                if br.model_name == r.model_name and br.game_name == game_name), 0))
                          for r in ratings]
            game_ratings.sort(key=lambda x: x[1], reverse=True)
            
            for i, (model, score, success_rate) in enumerate(game_ratings):
                report += f"| {i+1} | {model} | {score:.1f} | {success_rate:.1%} |\n"
            
            report += "\n"
        
        # Анализ результатов
        report += "## Analysis\n\n"
        
        # Лучшая модель в целом
        if ratings:
            best_model = ratings[0]
            report += f"**Best Overall Model**: {best_model.model_name} with {best_model.overall_score:.1f} points\n"
            
            # Лучшая модель по играм
            for game_name in self.game_weights.keys():
                game_ratings = [(r.model_name, r.game_scores.get(game_name, 0)) for r in ratings]
                game_ratings.sort(key=lambda x: x[1], reverse=True)
                if game_ratings:
                    report += f"**Best at {game_name.title()}**: {game_ratings[0][0]} ({game_ratings[0][1]:.1f} points)\n"
        
        return report
    
    def save_results(self, benchmark_results: List[BenchmarkResult], 
                    ratings: List[ModelRating]) -> Tuple[str, str]:
        """Сохраняет результаты бенчмарка."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем JSON с результатами
        results_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            data = {
                "timestamp": timestamp,
                "benchmark_results": [asdict(r) for r in benchmark_results],
                "model_ratings": [asdict(r) for r in ratings],
                "game_weights": self.game_weights
            }
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Генерируем и сохраняем отчет
        report = self.generate_benchmark_report(benchmark_results, ratings)
        report_file = self.results_dir / f"benchmark_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return str(results_file), str(report_file)

# Основная функция для CLI
async def main():
    """Основная функция для запуска бенчмарка."""
    runner = BenchmarkRunner()
    
    # Настройки бенчмарка (можно адаптировать)
    runner.configure_game("diplomacy", num_games=1, max_rounds=3)  # Быстрый тест
    runner.configure_game("beast", num_games=2, max_rounds=5)
    runner.configure_game("spyfall", num_games=2, max_rounds=4)
    runner.configure_game("askguess", num_games=2, max_rounds=8)
    runner.configure_game("tofukingdom", num_games=2, max_rounds=6)
    
    print("🚀 Запуск PolitAgent Benchmark")
    print(f"📊 Моделей для тестирования: {len([m for m in runner.models if m.enabled])}")
    print(f"🎮 Игр для тестирования: {len([g for g in runner.games.values() if g.enabled])}")
    
    # Запускаем бенчмарк
    start_time = time.time()
    benchmark_results = await runner.run_benchmark(max_parallel_models=2, max_parallel_games=1)
    total_time = time.time() - start_time
    
    # Вычисляем рейтинги
    ratings = runner.calculate_model_ratings(benchmark_results)
    
    # Сохраняем результаты
    results_file, report_file = runner.save_results(benchmark_results, ratings)
    
    print(f"\n✅ Бенчмарк завершен за {total_time:.1f} секунд")
    print(f"📄 Результаты сохранены: {results_file}")
    print(f"📋 Отчет сохранен: {report_file}")
    
    # Показываем топ-3
    print("\n🏆 ТОП-3 МОДЕЛИ:")
    for i, rating in enumerate(ratings[:3]):
        print(f"{i+1}. {rating.model_name} - {rating.overall_score:.1f} очков")

if __name__ == "__main__":
    asyncio.run(main()) 