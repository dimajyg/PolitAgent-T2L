#!/usr/bin/env python3
"""
Модуль для управления vLLM моделями с ограниченными ресурсами.
Поддерживает последовательное тестирование моделей с автоматическим
управлением памятью GPU.
"""

import gc
import os
import psutil
import subprocess
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import asyncio
from dataclasses import dataclass
import GPUtil

logger = logging.getLogger(__name__)

@dataclass
class VLLMModelConfig:
    """Конфигурация модели для vLLM."""
    model_path: str  # Путь к модели или HuggingFace ID
    display_name: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 4096
    temperature: float = 0.7
    trust_remote_code: bool = True
    port: int = 8000
    host: str = "127.0.0.1"

class VLLMResourceManager:
    """Менеджер ресурсов для vLLM."""
    
    def __init__(self, max_gpu_memory_mb: int = None):
        self.max_gpu_memory_mb = max_gpu_memory_mb
        self.current_process = None
        self.current_model = None
        
    def get_gpu_memory_usage(self) -> float:
        """Получает использование памяти GPU в MB."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
            return 0.0
        except:
            return 0.0
    
    def get_system_memory_usage(self) -> float:
        """Получает использование системной памяти в MB."""
        return psutil.virtual_memory().used / 1024 / 1024
    
    def check_memory_available(self, required_memory_mb: int) -> bool:
        """Проверяет, достаточно ли памяти для загрузки модели."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                available_gpu = gpus[0].memoryFree
                return available_gpu >= required_memory_mb
            return True  # Если нет GPU, предполагаем CPU
        except:
            return True
    
    def cleanup_current_model(self):
        """Очищает текущую модель из памяти."""
        if self.current_process:
            logger.info(f"Останавливаем текущую модель: {self.current_model}")
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
                self.current_process.wait()
            except Exception as e:
                logger.error(f"Ошибка при остановке модели: {e}")
            
            self.current_process = None
            self.current_model = None
            
            # Принудительная очистка памяти
            gc.collect()
            time.sleep(5)  # Ждем освобождения памяти
    
    def launch_vllm_model(self, config: VLLMModelConfig) -> bool:
        """Запускает vLLM модель как отдельный процесс."""
        try:
            # Очищаем предыдущую модель
            self.cleanup_current_model()
            
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", config.model_path,
                "--host", config.host,
                "--port", str(config.port),
                "--tensor-parallel-size", str(config.tensor_parallel_size),
                "--gpu-memory-utilization", str(config.gpu_memory_utilization),
                "--max-model-len", str(config.max_model_len),
            ]
            
            if config.trust_remote_code:
                cmd.append("--trust-remote-code")
            
            logger.info(f"Запускаем vLLM модель: {config.display_name}")
            logger.info(f"Команда: {' '.join(cmd)}")
            
            # Запускаем процесс
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Ждем загрузки модели
            max_wait_time = 300  # 5 минут
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                # Проверяем, что процесс еще жив
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    logger.error(f"vLLM процесс завершился с ошибкой:")
                    logger.error(f"STDOUT: {stdout}")
                    logger.error(f"STDERR: {stderr}")
                    return False
                
                # Проверяем доступность API
                try:
                    import requests
                    response = requests.get(
                        f"http://{config.host}:{config.port}/health",
                        timeout=5
                    )
                    if response.status_code == 200:
                        logger.info(f"vLLM модель {config.display_name} успешно запущена")
                        self.current_process = process
                        self.current_model = config.display_name
                        return True
                except:
                    pass
                
                time.sleep(10)
            
            # Таймаут
            logger.error(f"Таймаут запуска модели {config.display_name}")
            process.terminate()
            return False
            
        except Exception as e:
            logger.error(f"Ошибка запуска vLLM модели: {e}")
            return False

class SequentialVLLMBenchmark:
    """Класс для последовательного бенчмаркинга vLLM моделей."""
    
    def __init__(self, results_dir: str = "vllm_benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.resource_manager = VLLMResourceManager()
        self.models_to_test = []
        self.all_results = []
        
        # Импортируем бенчмарк-раннер
        from scripts.benchmark_runner import BenchmarkRunner, ModelConfig
        self.BenchmarkRunner = BenchmarkRunner
        self.ModelConfig = ModelConfig
        
    def add_vllm_model(self, model_config: VLLMModelConfig):
        """Добавляет vLLM модель для тестирования."""
        self.models_to_test.append(model_config)
    
    def add_models_from_config(self, models_config: List[Dict[str, Any]]):
        """Добавляет модели из конфигурации."""
        for config in models_config:
            vllm_config = VLLMModelConfig(**config)
            self.add_vllm_model(vllm_config)
    
    async def test_single_vllm_model(self, vllm_config: VLLMModelConfig) -> Dict[str, Any]:
        """Тестирует одну vLLM модель на всех играх."""
        logger.info(f"🧪 Начинаем тестирование модели: {vllm_config.display_name}")
        
        try:
            # Запускаем vLLM модель
            if not self.resource_manager.launch_vllm_model(vllm_config):
                return {
                    "model_name": vllm_config.display_name,
                    "error": "Не удалось запустить vLLM модель",
                    "status": "failed"
                }
            
            # Создаем бенчмарк-раннер
            runner = self.BenchmarkRunner(str(self.results_dir / vllm_config.display_name.replace(" ", "_")))
            
            # Настраиваем модель как OpenAI-совместимую через vLLM endpoint
            runner.models = [self.ModelConfig(
                provider="openai",  # vLLM совместим с OpenAI API
                model_name=vllm_config.model_path,
                display_name=vllm_config.display_name,
                temperature=vllm_config.temperature,
                api_key="EMPTY",  # vLLM не требует ключ
                enabled=True
            )]
            
            # Переопределяем базовый URL для OpenAI клиента
            import openai
            original_base_url = getattr(openai, 'base_url', None)
            openai.api_base = f"http://{vllm_config.host}:{vllm_config.port}/v1"
            
            # Настраиваем игры для быстрого тестирования
            runner.configure_game("diplomacy", num_games=1, max_rounds=3)
            runner.configure_game("beast", num_games=1, max_rounds=4)
            runner.configure_game("spyfall", num_games=1, max_rounds=4)
            runner.configure_game("askguess", num_games=1, max_rounds=6)
            runner.configure_game("tofukingdom", num_games=1, max_rounds=5)
            
            logger.info(f"🎮 Запускаем бенчмарк для {vllm_config.display_name}")
            
            # Запускаем бенчмарк
            start_time = time.time()
            benchmark_results = await runner.run_benchmark(max_parallel_models=1, max_parallel_games=1)
            total_time = time.time() - start_time
            
            # Вычисляем рейтинги
            ratings = runner.calculate_model_ratings(benchmark_results)
            
            # Сохраняем результаты
            results_file, report_file = runner.save_results(benchmark_results, ratings)
            
            # Восстанавливаем исходный базовый URL
            if original_base_url:
                openai.api_base = original_base_url
            
            result = {
                "model_name": vllm_config.display_name,
                "model_path": vllm_config.model_path,
                "status": "completed",
                "total_time": total_time,
                "results_file": str(results_file),
                "report_file": str(report_file),
                "ratings": [asdict(r) for r in ratings] if ratings else [],
                "benchmark_results": [asdict(r) for r in benchmark_results],
                "memory_usage": {
                    "gpu_memory_mb": self.resource_manager.get_gpu_memory_usage(),
                    "system_memory_mb": self.resource_manager.get_system_memory_usage()
                }
            }
            
            logger.info(f"✅ Модель {vllm_config.display_name} протестирована за {total_time:.1f}с")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования модели {vllm_config.display_name}: {e}")
            return {
                "model_name": vllm_config.display_name,
                "error": str(e),
                "status": "error"
            }
        
        finally:
            # Всегда очищаем модель после тестирования
            self.resource_manager.cleanup_current_model()
    
    async def run_sequential_benchmark(self) -> Dict[str, Any]:
        """Запускает последовательный бенчмарк всех моделей."""
        logger.info(f"🚀 Запуск последовательного vLLM бенчмарка")
        logger.info(f"📊 Моделей к тестированию: {len(self.models_to_test)}")
        
        start_time = time.time()
        
        for i, vllm_config in enumerate(self.models_to_test, 1):
            logger.info(f"📈 Модель {i}/{len(self.models_to_test)}: {vllm_config.display_name}")
            
            result = await self.test_single_vllm_model(vllm_config)
            self.all_results.append(result)
            
            # Небольшая пауза между моделями для стабилизации системы
            if i < len(self.models_to_test):
                logger.info("⏳ Пауза между моделями...")
                time.sleep(10)
        
        total_time = time.time() - start_time
        
        # Сохраняем сводные результаты
        summary = {
            "total_models": len(self.models_to_test),
            "completed_models": len([r for r in self.all_results if r.get("status") == "completed"]),
            "failed_models": len([r for r in self.all_results if r.get("status") in ["failed", "error"]]),
            "total_time": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": self.all_results
        }
        
        # Сохраняем результаты
        summary_file = self.results_dir / f"vllm_benchmark_summary_{int(time.time())}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Последовательный бенчмарк завершен за {total_time:.1f}с")
        logger.info(f"📄 Сводка сохранена: {summary_file}")
        
        return summary
    
    def generate_comparison_report(self) -> str:
        """Генерирует сравнительный отчет по всем протестированным моделям."""
        if not self.all_results:
            return "Нет результатов для сравнения"
        
        successful_results = [r for r in self.all_results if r.get("status") == "completed"]
        
        report = f"""# Сравнительный отчет vLLM бенчмарка

## Общая статистика
- **Протестировано моделей**: {len(self.all_results)}
- **Успешно завершено**: {len(successful_results)}
- **Провалено**: {len(self.all_results) - len(successful_results)}

## Результаты по моделям

"""
        
        for result in successful_results:
            if result.get("ratings"):
                rating = result["ratings"][0]  # Первый (единственный) рейтинг
                report += f"""### {result['model_name']}
- **Путь модели**: `{result['model_path']}`
- **Общий балл**: {rating.get('overall_score', 0):.1f}
- **Success Rate**: {rating.get('success_rate', 0):.1%}
- **Время выполнения**: {result['total_time']:.1f}с
- **Результаты файлы**: {result['results_file']}

"""
        
        return report

# Пример конфигурации моделей
EXAMPLE_VLLM_MODELS = [
    {
        "model_path": "microsoft/DialoGPT-medium",
        "display_name": "DialoGPT Medium",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.6,
        "max_model_len": 2048,
        "port": 8000
    },
    {
        "model_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "display_name": "TinyLlama 1.1B",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.4,
        "max_model_len": 2048,
        "port": 8001
    },
    {
        "model_path": "microsoft/DialoGPT-small",
        "display_name": "DialoGPT Small",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.3,
        "max_model_len": 1024,
        "port": 8002
    }
]

async def main():
    """Пример использования."""
    # Создаем бенчмарк
    benchmark = SequentialVLLMBenchmark()
    
    # Добавляем модели из примера
    benchmark.add_models_from_config(EXAMPLE_VLLM_MODELS)
    
    # Запускаем бенчмарк
    summary = await benchmark.run_sequential_benchmark()
    
    # Генерируем отчет
    report = benchmark.generate_comparison_report()
    print(report)
    
    return summary

if __name__ == "__main__":
    asyncio.run(main())
