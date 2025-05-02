#!/usr/bin/env python
"""
PolitAgent Benchmark Visualizer - модуль для анализа результатов бенчмарка
и визуализации метрик производительности моделей.
"""

import argparse
import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark_visualizer")

def load_results(results_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Загружает результаты бенчмарка из указанной директории.
    
    Args:
        results_dir: Путь к директории с результатами
        
    Returns:
        Кортеж (summary, results) с сводкой и результатами
    """
    # Загружаем общую сводку
    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(summary_path):
        logger.error(f"Файл сводки не найден: {summary_path}")
        return {}, []
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Загружаем все результаты
    results_path = os.path.join(results_dir, "all_results.jsonl")
    results = []
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Ошибка при разборе строки JSON: {line}")
    else:
        # Если нет общего файла, ищем результаты рекурсивно
        logger.info("Файл all_results.jsonl не найден, поиск отдельных файлов результатов...")
        result_files = glob.glob(os.path.join(results_dir, "**", "*_result.json"), recursive=True)
        
        for file_path in result_files:
            with open(file_path, 'r') as f:
                try:
                    result = json.load(f)
                    results.append(result)
                except json.JSONDecodeError:
                    logger.warning(f"Ошибка при разборе файла JSON: {file_path}")
    
    logger.info(f"Загружено {len(results)} результатов игр")
    return summary, results

def create_performance_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Создает DataFrame с данными о производительности моделей.
    
    Args:
        results: Список результатов игр
        
    Returns:
        pandas.DataFrame с данными о производительности
    """
    performance_data = []
    
    for result in results:
        # Базовая информация
        game_type = result.get("game_type", "unknown")
        model = result.get("model", "unknown")
        
        # Извлекаем ключевые метрики в зависимости от типа игры
        if game_type == "spyfall":
            # Метрики для Spyfall
            performance_data.append({
                "game_type": game_type,
                "model": model,
                "success": result.get("spy_win", False) if result.get("spy_index", 0) > 0 else not result.get("spy_win", False),
                "spy_detected": result.get("spy_detected", False),
                "elimination_speed": result.get("rounds", 0),
                "spy_index": result.get("spy_index", 0)
            })
        elif game_type == "beast":
            # Метрики для Beast
            performance_data.append({
                "game_type": game_type,
                "model": model,
                "success": result.get("winner", "") == "beast",
                "rounds": result.get("rounds", 0)
            })
        elif game_type == "askguess":
            # Метрики для AskGuess
            performance_data.append({
                "game_type": game_type,
                "model": model,
                "success": result.get("correct", False),
                "questions": result.get("questions", 0)
            })
        elif game_type == "tofukingdom":
            # Метрики для TofuKingdom
            performance_data.append({
                "game_type": game_type,
                "model": model,
                "success": result.get("winner", "none") not in ["none", "timeout"],
                "spy_won": result.get("winner", "") == "spy",
                "prince_won": result.get("winner", "") == "prince",
                "queen_won": result.get("winner", "") == "queen",
                "rounds": result.get("rounds", 0)
            })
    
    return pd.DataFrame(performance_data)

def visualize_model_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """
    Создает визуализации для сравнения моделей.
    
    Args:
        df: DataFrame с данными о производительности
        output_dir: Директория для сохранения визуализаций
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Настройка стиля графиков
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Успех по играм и моделям
    plt.figure(figsize=(10, 6))
    success_by_game_model = df.groupby(['game_type', 'model'])['success'].mean().reset_index()
    success_plot = sns.barplot(x='game_type', y='success', hue='model', data=success_by_game_model)
    plt.title('Успех моделей по типам игр')
    plt.xlabel('Тип игры')
    plt.ylabel('Доля успешных игр')
    plt.savefig(os.path.join(output_dir, 'success_by_game_model.png'))
    plt.close()
    
    # Специфичные для spyfall метрики
    spyfall_df = df[df['game_type'] == 'spyfall']
    if len(spyfall_df) > 0:
        plt.figure(figsize=(10, 6))
        spy_detection = spyfall_df.groupby('model')['spy_detected'].mean().reset_index()
        spy_detection_plot = sns.barplot(x='model', y='spy_detected', data=spy_detection)
        plt.title('Вероятность обнаружения шпиона по моделям')
        plt.xlabel('Модель')
        plt.ylabel('Доля игр с обнаруженным шпионом')
        plt.savefig(os.path.join(output_dir, 'spy_detection.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        rounds_by_model = spyfall_df.groupby('model')['elimination_speed'].mean().reset_index()
        rounds_plot = sns.barplot(x='model', y='elimination_speed', data=rounds_by_model)
        plt.title('Среднее количество раундов до завершения игры')
        plt.xlabel('Модель')
        plt.ylabel('Количество раундов')
        plt.savefig(os.path.join(output_dir, 'rounds_by_model_spyfall.png'))
        plt.close()
    
    # Специфичные для askguess метрики
    askguess_df = df[df['game_type'] == 'askguess']
    if len(askguess_df) > 0:
        plt.figure(figsize=(10, 6))
        questions_by_model = askguess_df.groupby('model')['questions'].mean().reset_index()
        questions_plot = sns.barplot(x='model', y='questions', data=questions_by_model)
        plt.title('Среднее количество вопросов до отгадки')
        plt.xlabel('Модель')
        plt.ylabel('Количество вопросов')
        plt.savefig(os.path.join(output_dir, 'questions_by_model.png'))
        plt.close()
    
    # Специфичные для tofukingdom метрики
    tofukingdom_df = df[df['game_type'] == 'tofukingdom']
    if len(tofukingdom_df) > 0:
        # Победы по ролям
        role_columns = ['spy_won', 'prince_won', 'queen_won']
        if all(col in tofukingdom_df.columns for col in role_columns):
            plt.figure(figsize=(10, 6))
            
            # Собираем данные о победах по ролям
            role_wins = pd.DataFrame({
                'Роль': ['Шпион', 'Принц', 'Королева'],
                'Доля побед': [
                    tofukingdom_df['spy_won'].mean(),
                    tofukingdom_df['prince_won'].mean(),
                    tofukingdom_df['queen_won'].mean()
                ]
            })
            
            role_plot = sns.barplot(x='Роль', y='Доля побед', data=role_wins)
            plt.title('Доля побед по ролям в TofuKingdom')
            plt.ylabel('Доля побед')
            plt.savefig(os.path.join(output_dir, 'tofukingdom_role_wins.png'))
            plt.close()
            
            # Сравнение моделей по количеству раундов
            plt.figure(figsize=(10, 6))
            rounds_by_model = tofukingdom_df.groupby('model')['rounds'].mean().reset_index()
            rounds_plot = sns.barplot(x='model', y='rounds', data=rounds_by_model)
            plt.title('Среднее количество раундов в игре TofuKingdom')
            plt.xlabel('Модель')
            plt.ylabel('Количество раундов')
            plt.savefig(os.path.join(output_dir, 'rounds_by_model_tofukingdom.png'))
            plt.close()
    
    # Общий успех по моделям
    plt.figure(figsize=(10, 6))
    success_by_model = df.groupby('model')['success'].mean().reset_index()
    overall_plot = sns.barplot(x='model', y='success', data=success_by_model)
    plt.title('Общий успех по моделям')
    plt.xlabel('Модель')
    plt.ylabel('Доля успешных игр')
    plt.savefig(os.path.join(output_dir, 'overall_success.png'))
    plt.close()
    
    logger.info(f"Визуализации сохранены в {output_dir}")

def generate_report(summary: Dict[str, Any], df: pd.DataFrame, output_dir: str) -> None:
    """
    Генерирует текстовый отчет по результатам бенчмарка.
    
    Args:
        summary: Сводка по бенчмарку
        df: DataFrame с данными о производительности
        output_dir: Директория для сохранения отчета
    """
    report_lines = ["# Отчет по бенчмарку PolitAgent", ""]
    
    # Общая информация
    report_lines.append("## Общая информация")
    report_lines.append(f"* Всего запущено игр: {summary.get('total_games', 0)}")
    report_lines.append(f"* Успешно завершено игр: {summary.get('completed_games', 0)}")
    report_lines.append(f"* Дата запуска: {summary.get('timestamp', 'не указана')}")
    report_lines.append("")
    
    # Статистика по типам игр
    report_lines.append("## Статистика по типам игр")
    games_by_type = summary.get('games_by_type', {})
    for game_type, stats in games_by_type.items():
        report_lines.append(f"### {game_type.capitalize()}")
        report_lines.append(f"* Всего игр: {stats.get('total', 0)}")
        report_lines.append(f"* Успешно завершено: {stats.get('successful', 0)}")
        
        # Добавляем специфичную статистику по типу игры
        game_df = df[df['game_type'] == game_type]
        if len(game_df) > 0:
            success_rate = game_df['success'].mean() * 100
            report_lines.append(f"* Общий процент успеха: {success_rate:.2f}%")
            
            # Статистика по моделям
            report_lines.append("* Статистика по моделям:")
            for model in game_df['model'].unique():
                model_df = game_df[game_df['model'] == model]
                model_success = model_df['success'].mean() * 100
                report_lines.append(f"  * {model}: {model_success:.2f}% успешных игр")
                
                # Специфичные метрики для разных игр
                if game_type == "spyfall" and 'spy_detected' in model_df.columns:
                    spy_detected = model_df['spy_detected'].mean() * 100
                    report_lines.append(f"    * Процент обнаружения шпиона: {spy_detected:.2f}%")
                    avg_rounds = model_df['elimination_speed'].mean()
                    report_lines.append(f"    * Среднее количество раундов: {avg_rounds:.2f}")
                    
                elif game_type == "askguess" and 'questions' in model_df.columns:
                    avg_questions = model_df['questions'].mean()
                    report_lines.append(f"    * Среднее количество вопросов: {avg_questions:.2f}")
                    
                elif game_type == "tofukingdom":
                    if 'rounds' in model_df.columns:
                        avg_rounds = model_df['rounds'].mean()
                        report_lines.append(f"    * Среднее количество раундов: {avg_rounds:.2f}")
                    
                    # Добавляем статистику по ролям, если доступно
                    role_columns = ['spy_won', 'prince_won', 'queen_won']
                    if all(col in model_df.columns for col in role_columns):
                        spy_win_rate = model_df['spy_won'].mean() * 100
                        prince_win_rate = model_df['prince_won'].mean() * 100
                        queen_win_rate = model_df['queen_won'].mean() * 100
                        report_lines.append(f"    * Процент побед шпиона: {spy_win_rate:.2f}%")
                        report_lines.append(f"    * Процент побед принца: {prince_win_rate:.2f}%")
                        report_lines.append(f"    * Процент побед королевы: {queen_win_rate:.2f}%")
        
        report_lines.append("")
    
    # Сводный рейтинг моделей
    report_lines.append("## Сводный рейтинг моделей")
    model_success = df.groupby('model')['success'].mean().reset_index().sort_values('success', ascending=False)
    
    for i, (_, row) in enumerate(model_success.iterrows(), 1):
        model_name = row['model']
        success_rate = row['success'] * 100
        report_lines.append(f"{i}. **{model_name}**: {success_rate:.2f}% успешных игр")
    
    report_lines.append("")
    report_lines.append("## Визуализации")
    report_lines.append("Визуализации доступны в директории `visualizations/`.")
    
    # Записываем отчет в файл
    report_path = os.path.join(output_dir, "benchmark_report.md")
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Отчет сохранен в {report_path}")

def main():
    """Точка входа программы."""
    parser = argparse.ArgumentParser(description="PolitAgent Benchmark Visualizer - визуализация результатов бенчмарка")
    parser.add_argument('--results_dir', type=str, required=True,
                      help="Директория с результатами бенчмарка")
    parser.add_argument('--output_dir', type=str, default=None,
                      help="Директория для сохранения визуализаций и отчета (по умолчанию внутри директории с результатами)")
    args = parser.parse_args()
    
    # Проверяем наличие директории с результатами
    if not os.path.isdir(args.results_dir):
        logger.error(f"Директория с результатами не найдена: {args.results_dir}")
        return
    
    # Если output_dir не указан, создаем его внутри директории с результатами
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "analysis")
    
    # Создаем директории для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Загружаем результаты
    summary, results = load_results(args.results_dir)
    
    if not results:
        logger.error("Результаты не найдены или формат неверный.")
        return
    
    # Создаем DataFrame
    performance_df = create_performance_dataframe(results)
    
    # Сохраняем DataFrame для возможного дальнейшего анализа
    performance_df.to_csv(os.path.join(args.output_dir, "performance_data.csv"), index=False)
    
    # Создаем визуализации
    visualize_model_comparison(performance_df, vis_dir)
    
    # Генерируем отчет
    generate_report(summary, performance_df, args.output_dir)
    
    logger.info(f"Анализ завершен. Результаты доступны в {args.output_dir}")

if __name__ == "__main__":
    main() 