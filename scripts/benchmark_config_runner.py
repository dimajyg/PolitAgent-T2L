#!/usr/bin/env python3
"""
Конфигурируемый бенчмарк-раннер с поддержкой YAML конфигурации.
"""

import asyncio
import yaml
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from scripts.benchmark_runner import BenchmarkRunner, ModelConfig, GameConfig

class ConfigurableBenchmarkRunner(BenchmarkRunner):
    """Бенчмарк-раннер с поддержкой YAML конфигурации."""
    
    def __init__(self, config_path: str = "configs/benchmark_config.yaml"):
        # Загружаем конфигурацию
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Инициализируем базовый класс
        results_dir = self.config['benchmark']['results_dir']
        super().__init__(results_dir)
        
        # Переопределяем настройки из конфига
        self._load_config()
    
    def _load_config(self):
        """Загружает настройки из конфигурации."""
        # Веса игр
        self.game_weights = self.config['benchmark']['game_weights']
        
        # Настройки игр
        self.games = {}
        for game_name, game_config in self.config['games'].items():
            if game_config['enabled']:
                # Получаем класс игры
                game_class = self._get_game_class(game_name)
                self.games[game_name] = GameConfig(
                    name=game_name,
                    game_class=game_class,
                    num_games=game_config['num_games'],
                    max_rounds=game_config['max_rounds'],
                    enabled=True
                )
        
        # Настройки моделей
        self.models = []
        for model_id, model_config in self.config['models'].items():
            if model_config['enabled']:
                self.models.append(ModelConfig(
                    provider=model_config['provider'],
                    model_name=model_config['model_name'],
                    display_name=model_config['display_name'],
                    temperature=model_config['temperature'],
                    enabled=True
                ))
    
    def _get_game_class(self, game_name: str):
        """Получает класс игры по имени."""
        from environments.diplomacy_game.game import DiplomacyGame
        from environments.beast.game import BeastGame
        from environments.spyfall.game import SpyfallGame
        from environments.askguess.game import AskGuessGame
        from environments.tofukingdom.game import TofuKingdomGame
        
        game_classes = {
            "diplomacy": DiplomacyGame,
            "beast": BeastGame,
            "spyfall": SpyfallGame,
            "askguess": AskGuessGame,
            "tofukingdom": TofuKingdomGame
        }
        
        return game_classes[game_name]
    
    def apply_profile(self, profile_name: str):
        """Применяет профиль конфигурации."""
        if profile_name not in self.config['profiles']:
            raise ValueError(f"Профиль '{profile_name}' не найден")
        
        profile = self.config['profiles'][profile_name]
        
        # Обновляем настройки игр
        if 'games' in profile:
            for game_name, game_settings in profile['games'].items():
                if game_name in self.games:
                    for key, value in game_settings.items():
                        setattr(self.games[game_name], key, value)
        
        # Обновляем список моделей
        if 'models' in profile:
            enabled_models = set(profile['models'])
            for model in self.models:
                # Находим ID модели в конфиге
                model_id = None
                for mid, mconfig in self.config['models'].items():
                    if (mconfig['provider'] == model.provider and 
                        mconfig['model_name'] == model.model_name):
                        model_id = mid
                        break
                
                model.enabled = model_id in enabled_models
    
    async def run_benchmark_with_config(self, profile: str = None) -> None:
        """Запускает бенчмарк с настройками из конфига."""
        if profile:
            print(f"📋 Применяем профиль: {profile}")
            self.apply_profile(profile)
        
        # Получаем настройки параллельности
        max_parallel_models = self.config['benchmark']['max_parallel_models']
        max_parallel_games = self.config['benchmark']['max_parallel_games']
        
        print("🚀 Запуск конфигурируемого PolitAgent Benchmark")
        print(f"📊 Моделей для тестирования: {len([m for m in self.models if m.enabled])}")
        print(f"🎮 Игр для тестирования: {len([g for g in self.games.values() if g.enabled])}")
        
        # Запускаем бенчмарк
        import time
        start_time = time.time()
        benchmark_results = await self.run_benchmark(max_parallel_models, max_parallel_games)
        total_time = time.time() - start_time
        
        # Вычисляем рейтинги
        ratings = self.calculate_model_ratings(benchmark_results)
        
        # Сохраняем результаты
        results_file, report_file = self.save_results(benchmark_results, ratings)
        
        print(f"\n✅ Бенчмарк завершен за {total_time:.1f} секунд")
        print(f"📄 Результаты сохранены: {results_file}")
        print(f"📋 Отчет сохранен: {report_file}")
        
        # Показываем топ-3
        print("\n🏆 ТОП-3 МОДЕЛИ:")
        for i, rating in enumerate(ratings[:3]):
            print(f"{i+1}. {rating.model_name} - {rating.overall_score:.1f} очков")

async def main():
    """Основная функция CLI."""
    parser = argparse.ArgumentParser(description="PolitAgent Benchmark Runner")
    parser.add_argument("--config", "-c", default="configs/benchmark_config.yaml",
                       help="Путь к файлу конфигурации")
    parser.add_argument("--profile", "-p", default=None,
                       help="Профиль конфигурации для использования")
    parser.add_argument("--list-profiles", action="store_true",
                       help="Показать доступные профили")
    
    args = parser.parse_args()
    
    # Показываем доступные профили
    if args.list_profiles:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("📋 Доступные профили:")
        for profile_name, profile_data in config['profiles'].items():
            models = profile_data.get('models', [])
            print(f"  • {profile_name}: {len(models)} моделей")
        return
    
    # Запускаем бенчмарк
    runner = ConfigurableBenchmarkRunner(args.config)
    await runner.run_benchmark_with_config(args.profile)

if __name__ == "__main__":
    asyncio.run(main()) 