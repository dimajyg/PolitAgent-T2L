#!/usr/bin/env python3
"""
Анализатор результатов бенчмарков PolitAgent.

Позволяет:
- Сравнивать результаты разных запусков
- Генерировать сводные отчеты
- Создавать графики производительности
- Экспортировать данные в различные форматы
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime

class BenchmarkAnalyzer:
    """Анализатор результатов бенчмарков."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_files = list(self.results_dir.glob("benchmark_results_*.json"))
        
    def load_results(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Загружает результаты бенчмарка."""
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Загружаем последний файл результатов
        if not self.results_files:
            raise FileNotFoundError("Не найдено файлов с результатами бенчмарков")
        
        latest_file = max(self.results_files, key=lambda f: f.stat().st_mtime)
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compare_models(self, results_data: Dict[str, Any]) -> pd.DataFrame:
        """Создает DataFrame для сравнения моделей."""
        ratings = results_data['model_ratings']
        
        df_data = []
        for rating in ratings:
            row = {
                'Model': rating['model_name'],
                'Overall Score': rating['overall_score'],
                'Success Rate': rating['success_rate'],
                'Avg Time (s)': rating['avg_execution_time'],
                'Total Games': rating['total_games'],
                'Rank': rating['rank']
            }
            
            # Добавляем скоры по играм
            for game, score in rating['game_scores'].items():
                row[f'{game.title()} Score'] = score
            
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def generate_performance_plots(self, df: pd.DataFrame, output_dir: str = "plots"):
        """Генерирует графики производительности."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Общий рейтинг моделей
        plt.figure(figsize=(12, 8))
        df_sorted = df.sort_values('Overall Score', ascending=True)
        plt.barh(df_sorted['Model'], df_sorted['Overall Score'])
        plt.title('Общий рейтинг моделей', fontsize=16, fontweight='bold')
        plt.xlabel('Общий балл')
        plt.tight_layout()
        plt.savefig(output_path / 'overall_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Success Rate vs Execution Time
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df['Success Rate'], df['Avg Time (s)'], 
                            s=df['Overall Score']*2, alpha=0.7)
        
        for i, model in enumerate(df['Model']):
            plt.annotate(model, (df.iloc[i]['Success Rate'], df.iloc[i]['Avg Time (s)']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Success Rate')
        plt.ylabel('Average Execution Time (seconds)')
        plt.title('Success Rate vs Execution Time\n(размер точки = общий балл)')
        plt.tight_layout()
        plt.savefig(output_path / 'success_vs_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Производительность по играм
        game_columns = [col for col in df.columns if col.endswith(' Score')]
        if game_columns:
            plt.figure(figsize=(14, 8))
            
            # Подготавливаем данные для heatmap
            heatmap_data = df.set_index('Model')[game_columns]
            
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd')
            plt.title('Производительность моделей по играм', fontsize=16, fontweight='bold')
            plt.ylabel('Модели')
            plt.xlabel('Игры')
            plt.tight_layout()
            plt.savefig(output_path / 'games_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Графики сохранены в {output_path}")
    
    def export_to_csv(self, df: pd.DataFrame, filename: str = None):
        """Экспортирует результаты в CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_comparison_{timestamp}.csv"
        
        filepath = self.results_dir / filename
        df.to_csv(filepath, index=False)
        print(f"📄 CSV экспортирован: {filepath}")
        return filepath
    
    def generate_summary_report(self, results_data: Dict[str, Any]) -> str:
        """Генерирует сводный отчет."""
        df = self.compare_models(results_data)
        
        report = f"""# Сводный отчет по бенчмарку PolitAgent

## Общая статистика
- **Дата анализа**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Протестировано моделей**: {len(df)}
- **Общее количество игр**: {df['Total Games'].sum()}

## Топ-3 модели по общему рейтингу:
"""
        
        top_3 = df.nlargest(3, 'Overall Score')
        for i, (_, row) in enumerate(top_3.iterrows()):
            report += f"{i+1}. **{row['Model']}** - {row['Overall Score']:.1f} очков\n"
        
        report += f"""
## Лучшие модели по категориям:

### 🏆 Самая высокая точность
**{df.loc[df['Success Rate'].idxmax(), 'Model']}** - {df['Success Rate'].max():.1%} success rate

### ⚡ Самая быстрая
**{df.loc[df['Avg Time (s)'].idxmin(), 'Model']}** - {df['Avg Time (s)'].min():.1f} секунд в среднем

### 🎯 Лучший баланс скорости и точности
"""
        
        # Вычисляем баланс как (success_rate / normalized_time)
        df_temp = df.copy()
        df_temp['normalized_time'] = df_temp['Avg Time (s)'] / df_temp['Avg Time (s)'].max()
        df_temp['balance_score'] = df_temp['Success Rate'] / (df_temp['normalized_time'] + 0.1)
        best_balance = df_temp.loc[df_temp['balance_score'].idxmax()]
        
        report += f"**{best_balance['Model']}** - оптимальное соотношение скорости и точности\n"
        
        # Анализ по играм
        game_columns = [col for col in df.columns if col.endswith(' Score')]
        if game_columns:
            report += f"\n## Лучшие модели по играм:\n"
            for game_col in game_columns:
                game_name = game_col.replace(' Score', '')
                best_model = df.loc[df[game_col].idxmax(), 'Model']
                best_score = df[game_col].max()
                report += f"- **{game_name}**: {best_model} ({best_score:.1f} очков)\n"
        
        # Статистика
        report += f"""
## Детальная статистика:

| Метрика | Среднее | Медиана | Мин | Макс |
|---------|---------|---------|-----|------|
| Общий балл | {df['Overall Score'].mean():.1f} | {df['Overall Score'].median():.1f} | {df['Overall Score'].min():.1f} | {df['Overall Score'].max():.1f} |
| Success Rate | {df['Success Rate'].mean():.1%} | {df['Success Rate'].median():.1%} | {df['Success Rate'].min():.1%} | {df['Success Rate'].max():.1%} |
| Время выполнения | {df['Avg Time (s)'].mean():.1f}s | {df['Avg Time (s)'].median():.1f}s | {df['Avg Time (s)'].min():.1f}s | {df['Avg Time (s)'].max():.1f}s |

## Рекомендации:

"""
        
        # Генерируем рекомендации
        best_overall = df.loc[df['Overall Score'].idxmax()]
        fastest = df.loc[df['Avg Time (s)'].idxmin()]
        most_accurate = df.loc[df['Success Rate'].idxmax()]
        
        if best_overall['Model'] == most_accurate['Model']:
            report += f"- **{best_overall['Model']}** показывает лучшие результаты как по общему рейтингу, так и по точности. Рекомендуется для продакшена.\n"
        else:
            report += f"- **{best_overall['Model']}** лучше всего подходит для общего использования.\n"
            report += f"- **{most_accurate['Model']}** рекомендуется, когда критична точность.\n"
        
        if fastest['Model'] != best_overall['Model']:
            report += f"- **{fastest['Model']}** подходит для случаев, когда важна скорость выполнения.\n"
        
        return report
    
    def compare_multiple_runs(self, file_paths: List[str]) -> pd.DataFrame:
        """Сравнивает результаты нескольких запусков."""
        all_data = []
        
        for file_path in file_paths:
            results = self.load_results(file_path)
            timestamp = results.get('timestamp', Path(file_path).stem)
            
            for rating in results['model_ratings']:
                row = {
                    'Run': timestamp,
                    'Model': rating['model_name'],
                    'Overall Score': rating['overall_score'],
                    'Success Rate': rating['success_rate'],
                    'Avg Time': rating['avg_execution_time']
                }
                all_data.append(row)
        
        return pd.DataFrame(all_data)

def main():
    """Основная функция CLI."""
    parser = argparse.ArgumentParser(description="PolitAgent Benchmark Analyzer")
    parser.add_argument("--results-dir", "-d", default="benchmark_results",
                       help="Директория с результатами")
    parser.add_argument("--file", "-f", default=None,
                       help="Конкретный файл результатов для анализа")
    parser.add_argument("--export-csv", action="store_true",
                       help="Экспортировать в CSV")
    parser.add_argument("--generate-plots", action="store_true",
                       help="Генерировать графики")
    parser.add_argument("--output-dir", "-o", default="analysis_output",
                       help="Директория для сохранения результатов анализа")
    
    args = parser.parse_args()
    
    # Создаем анализатор
    analyzer = BenchmarkAnalyzer(args.results_dir)
    
    try:
        # Загружаем результаты
        results_data = analyzer.load_results(args.file)
        print(f"📊 Загружены результаты бенчмарка от {results_data.get('timestamp', 'неизвестно')}")
        
        # Создаем DataFrame для анализа
        df = analyzer.compare_models(results_data)
        print(f"📈 Проанализировано {len(df)} моделей")
        
        # Создаем директорию для вывода
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Генерируем сводный отчет
        summary_report = analyzer.generate_summary_report(results_data)
        report_file = output_path / "summary_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        print(f"📋 Сводный отчет сохранен: {report_file}")
        
        # Экспортируем в CSV
        if args.export_csv:
            csv_file = analyzer.export_to_csv(df, "model_comparison.csv")
        
        # Генерируем графики
        if args.generate_plots:
            analyzer.generate_performance_plots(df, str(output_path / "plots"))
        
        # Показываем краткую сводку
        print("\n🏆 ТОП-3 МОДЕЛИ:")
        top_3 = df.nlargest(3, 'Overall Score')
        for i, (_, row) in enumerate(top_3.iterrows()):
            print(f"{i+1}. {row['Model']} - {row['Overall Score']:.1f} очков")
        
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        print("Убедитесь, что вы запустили бенчмарк и файлы результатов существуют.")
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")

if __name__ == "__main__":
    main() 