#!/bin/bash

# vLLM Benchmark Runner Script
# Автоматизированный скрипт для запуска бенчмарка моделей через vLLM

set -e  # Выходить при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода цветного текста
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверка зависимостей
check_dependencies() {
    print_status "Проверка зависимостей..."
    
    # Проверка Python
    if ! command -v python &> /dev/null; then
        print_error "Python не найден. Установите Python 3.8+"
        exit 1
    fi
    
    # Проверка pip
    if ! command -v pip &> /dev/null; then
        print_error "pip не найден. Установите pip"
        exit 1
    fi
    
    # Проверка vLLM
    if ! python -c "import vllm" &> /dev/null; then
        print_warning "vLLM не установлен. Устанавливаем зависимости..."
        pip install -r requirements_vllm.txt
    fi
    
    print_success "Все зависимости в порядке"
}

# Создание необходимых директорий
setup_directories() {
    print_status "Создание директорий..."
    mkdir -p models
    mkdir -p vllm_benchmark_results
    print_success "Директории созданы"
}

# Проверка GPU
check_gpu() {
    print_status "Проверка GPU..."
    if python -c "import torch; print(f'CUDA доступна: {torch.cuda.is_available()}')" | grep -q "True"; then
        print_success "GPU доступна"
        if command -v nvidia-smi &> /dev/null; then
            print_status "Информация о GPU:"
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        fi
    else
        print_warning "GPU недоступна, будет использоваться CPU (медленно)"
    fi
}

# Показать справку
show_help() {
    echo "vLLM Benchmark Runner"
    echo ""
    echo "Использование:"
    echo "  $0 [команда] [параметры]"
    echo ""
    echo "Команды:"
    echo "  setup                    - Установка зависимостей и настройка"
    echo "  example                  - Запуск легких моделей"
    echo "  advanced                 - Запуск продвинутых моделей"
    echo "  model <model_id>         - Тестирование одной модели"
    echo "  custom <config_file>     - Запуск с кастомной конфигурацией"
    echo "  monitor                  - Мониторинг ресурсов во время работы"
    echo "  clean                    - Очистка временных файлов"
    echo "  help                     - Показать эту справку"
    echo ""
    echo "Примеры:"
    echo "  $0 example                                    # Легкие модели"
    echo "  $0 model TinyLlama/TinyLlama-1.1B-Chat-v1.0  # Одна модель"
    echo "  $0 advanced                                   # Продвинутые модели"
    echo ""
}

# Установка и настройка
setup() {
    print_status "🚀 Начинаем установку и настройку..."
    check_dependencies
    setup_directories
    check_gpu
    print_success "✅ Установка завершена!"
}

# Запуск примера моделей
run_example() {
    print_status "🧪 Запуск бенчмарка с легкими моделями..."
    python scripts/vllm_benchmark_cli.py --preset example --log-level INFO
    print_success "✅ Бенчмарк завершен!"
}

# Запуск продвинутых моделей
run_advanced() {
    print_status "🚀 Запуск бенчмарка с продвинутыми моделями..."
    print_warning "Внимание: Требует много ресурсов GPU!"
    python scripts/vllm_benchmark_cli.py --preset advanced --log-level INFO
    print_success "✅ Бенчмарк завершен!"
}

# Запуск одной модели
run_single_model() {
    if [ -z "$1" ]; then
        print_error "Укажите ID модели. Пример: TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        exit 1
    fi
    
    print_status "🧪 Тестирование модели: $1"
    python scripts/vllm_benchmark_cli.py --model "$1" --log-level INFO
    print_success "✅ Тестирование завершено!"
}

# Мониторинг ресурсов
monitor_resources() {
    print_status "📊 Запуск мониторинга ресурсов..."
    print_status "Нажмите Ctrl+C для выхода"
    
    if command -v nvidia-smi &> /dev/null; then
        print_status "Мониторинг GPU..."
        watch -n 2 'echo "=== GPU Status ===" && nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv && echo "" && echo "=== System Memory ===" && free -h'
    else
        print_status "Мониторинг системных ресурсов..."
        watch -n 2 'echo "=== System Resources ===" && free -h && echo "" && echo "=== Top Processes ===" && top -bn1 | head -20'
    fi
}

# Очистка временных файлов
clean() {
    print_status "🧹 Очистка временных файлов..."
    
    # Удаляем логи старше 7 дней
    find . -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # Очищаем кэш Python
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # Спрашиваем про удаление загруженных моделей
    if [ -d "models" ]; then
        echo -n "Удалить загруженные модели? [y/N]: "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -rf models/*
            print_success "Модели удалены"
        fi
    fi
    
    print_success "✅ Очистка завершена!"
}

# Главная логика
main() {
    case "${1:-help}" in
        setup)
            setup
            ;;
        example)
            check_dependencies
            setup_directories
            run_example
            ;;
        advanced)
            check_dependencies
            setup_directories
            run_advanced
            ;;
        model)
            check_dependencies
            setup_directories
            run_single_model "$2"
            ;;
        monitor)
            monitor_resources
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Неизвестная команда: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Проверка аргументов
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Запуск главной функции
main "$@" 