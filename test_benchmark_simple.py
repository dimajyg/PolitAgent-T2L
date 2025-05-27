#!/usr/bin/env python3
"""
Простой тест системы бенчмаркинга без проблемных игр.
"""

import asyncio
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent))

# Простой тест конфигурации
def test_config_loading():
    """Тестируем загрузку конфигурации."""
    try:
        import yaml
        with open("configs/benchmark_config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ Конфигурация загружена успешно")
        print(f"📊 Найдено профилей: {len(config.get('profiles', {}))}")
        print(f"🎮 Найдено игр: {len(config.get('games', {}))}")
        print(f"🤖 Найдено моделей: {len(config.get('models', {}))}")
        
        # Показываем профили
        print("\n📋 Доступные профили:")
        for profile_name, profile_data in config.get('profiles', {}).items():
            models = profile_data.get('models', [])
            print(f"  • {profile_name}: {len(models)} моделей")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        return False

def test_metrics_import():
    """Тестируем импорт метрик."""
    try:
        from metrics import METRICS_MAP
        print(f"✅ Метрики импортированы: {list(METRICS_MAP.keys())}")
        return True
    except Exception as e:
        print(f"❌ Ошибка импорта метрик: {e}")
        return False

def test_llm_models():
    """Тестируем доступность LLM моделей."""
    try:
        from llm.models import AVAILABLE_MODELS
        print(f"✅ Доступные модели: {list(AVAILABLE_MODELS.keys())}")
        return True
    except Exception as e:
        print(f"❌ Ошибка импорта моделей: {e}")
        return False

async def main():
    """Основная функция тестирования."""
    print("🧪 Тестирование системы бенчмаркинга PolitAgent")
    print("=" * 50)
    
    tests = [
        ("Загрузка конфигурации", test_config_loading),
        ("Импорт метрик", test_metrics_import),
        ("Доступность моделей", test_llm_models),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ Тест '{test_name}' провален")
    
    print(f"\n📊 Результаты тестирования: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 Все тесты пройдены! Система готова к использованию.")
        print("\n💡 Для запуска бенчмарка используйте:")
        print("   python run_benchmark.py --profile quick")
    else:
        print("⚠️ Некоторые тесты провалены. Проверьте зависимости и конфигурацию.")

if __name__ == "__main__":
    asyncio.run(main()) 