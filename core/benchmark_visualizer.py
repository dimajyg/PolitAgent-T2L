#!/usr/bin/env python
"""
PolitAgent Benchmark Visualizer - analyzes benchmark results and visualizes
model performance metrics across different game environments.
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
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("benchmark_visualizer")

def load_results(results_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Loads benchmark results from directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Tuple (summary, results) with summary and results
    """
    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(summary_path):
        logger.error(f"Summary file not found: {summary_path}")
        return {}, []
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    results_path = os.path.join(results_dir, "all_results.jsonl")
    results = []
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing JSON line: {line}")
    else:
        logger.info("all_results.jsonl not found, searching for individual result files...")
        result_files = glob.glob(os.path.join(results_dir, "**", "*_result.json"), recursive=True)
        
        for file_path in result_files:
            with open(file_path, 'r') as f:
                try:
                    result = json.load(f)
                    results.append(result)
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing JSON file: {file_path}")
    
    logger.info(f"Loaded {len(results)} game results")
    return summary, results

def create_performance_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Creates DataFrame with model performance data.
    
    Args:
        results: List of game results
        
    Returns:
        pandas.DataFrame with performance data
    """
    performance_data = []
    
    for result in results:
        game_type = result.get("game_type", "unknown")
        model = result.get("model", "unknown")
        
        if game_type == "spyfall":
            performance_data.append({
                "game_type": game_type,
                "model": model,
                "success": result.get("spy_win", False) if result.get("spy_index", 0) > 0 else not result.get("spy_win", False),
                "spy_detected": result.get("spy_detected", False),
                "elimination_speed": result.get("rounds", 0),
                "spy_index": result.get("spy_index", 0)
            })
        elif game_type == "beast":
            performance_data.append({
                "game_type": game_type,
                "model": model,
                "success": result.get("winner", "") == "beast",
                "rounds": result.get("rounds", 0)
            })
        elif game_type == "askguess":
            performance_data.append({
                "game_type": game_type,
                "model": model,
                "success": result.get("correct", False),
                "questions": result.get("questions", 0)
            })
        elif game_type == "tofukingdom":
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
    Creates visualizations comparing model performance.
    
    Args:
        df: DataFrame with performance data
        output_dir: Directory for saving visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    plt.figure(figsize=(10, 6))
    success_by_game_model = df.groupby(['game_type', 'model'])['success'].mean().reset_index()
    success_plot = sns.barplot(x='game_type', y='success', hue='model', data=success_by_game_model)
    plt.title('Model Success by Game Type')
    plt.xlabel('Game Type')
    plt.ylabel('Success Rate')
    plt.savefig(os.path.join(output_dir, 'success_by_game_model.png'))
    plt.close()
    
    spyfall_df = df[df['game_type'] == 'spyfall']
    if len(spyfall_df) > 0:
        plt.figure(figsize=(10, 6))
        spy_detection = spyfall_df.groupby('model')['spy_detected'].mean().reset_index()
        spy_detection_plot = sns.barplot(x='model', y='spy_detected', data=spy_detection)
        plt.title('Spy Detection Rate by Model')
        plt.xlabel('Model')
        plt.ylabel('Spy Detection Rate')
        plt.savefig(os.path.join(output_dir, 'spy_detection.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        rounds_by_model = spyfall_df.groupby('model')['elimination_speed'].mean().reset_index()
        rounds_plot = sns.barplot(x='model', y='elimination_speed', data=rounds_by_model)
        plt.title('Average Rounds to Game Completion')
        plt.xlabel('Model')
        plt.ylabel('Number of Rounds')
        plt.savefig(os.path.join(output_dir, 'rounds_by_model_spyfall.png'))
        plt.close()
    
    askguess_df = df[df['game_type'] == 'askguess']
    if len(askguess_df) > 0:
        plt.figure(figsize=(10, 6))
        questions_by_model = askguess_df.groupby('model')['questions'].mean().reset_index()
        questions_plot = sns.barplot(x='model', y='questions', data=questions_by_model)
        plt.title('Average Questions to Solution')
        plt.xlabel('Model')
        plt.ylabel('Number of Questions')
        plt.savefig(os.path.join(output_dir, 'questions_by_model.png'))
        plt.close()
    
    tofukingdom_df = df[df['game_type'] == 'tofukingdom']
    if len(tofukingdom_df) > 0:
        role_columns = ['spy_won', 'prince_won', 'queen_won']
        if all(col in tofukingdom_df.columns for col in role_columns):
            plt.figure(figsize=(10, 6))
            
            role_wins = pd.DataFrame({
                'Role': ['Spy', 'Prince', 'Queen'],
                'Win Rate': [
                    tofukingdom_df['spy_won'].mean(),
                    tofukingdom_df['prince_won'].mean(),
                    tofukingdom_df['queen_won'].mean()
                ]
            })
            
            role_plot = sns.barplot(x='Role', y='Win Rate', data=role_wins)
            plt.title('Win Rate by Role in TofuKingdom')
            plt.ylabel('Win Rate')
            plt.savefig(os.path.join(output_dir, 'tofukingdom_role_wins.png'))
            plt.close()
            
            plt.figure(figsize=(10, 6))
            rounds_by_model = tofukingdom_df.groupby('model')['rounds'].mean().reset_index()
            rounds_plot = sns.barplot(x='model', y='rounds', data=rounds_by_model)
            plt.title('Average Rounds in TofuKingdom')
            plt.xlabel('Model')
            plt.ylabel('Number of Rounds')
            plt.savefig(os.path.join(output_dir, 'rounds_by_model_tofukingdom.png'))
            plt.close()
    
    plt.figure(figsize=(10, 6))
    success_by_model = df.groupby('model')['success'].mean().reset_index()
    overall_plot = sns.barplot(x='model', y='success', data=success_by_model)
    plt.title('Overall Success by Model')
    plt.xlabel('Model')
    plt.ylabel('Success Rate')
    plt.savefig(os.path.join(output_dir, 'overall_success.png'))
    plt.close()
    
    logger.info(f"Visualizations saved in {output_dir}")

def generate_report(summary: Dict[str, Any], df: pd.DataFrame, output_dir: str) -> None:
    """
    Generates textual report of benchmark results.
    
    Args:
        summary: Benchmark summary
        df: DataFrame with performance data
        output_dir: Directory for saving report
    """
    report_lines = ["# PolitAgent Benchmark Report", ""]
    
    report_lines.append("## Overview")
    report_lines.append(f"* Total Games: {summary.get('total_games', 0)}")
    report_lines.append(f"* Successfully Completed: {summary.get('completed_games', 0)}")
    report_lines.append(f"* Run Date: {summary.get('timestamp', 'not specified')}")
    report_lines.append("")
    
    report_lines.append("## Game Type Statistics")
    games_by_type = summary.get('games_by_type', {})
    for game_type, stats in games_by_type.items():
        report_lines.append(f"### {game_type.capitalize()}")
        report_lines.append(f"* Total Games: {stats.get('total', 0)}")
        report_lines.append(f"* Successfully Completed: {stats.get('successful', 0)}")
        
        game_df = df[df['game_type'] == game_type]
        if len(game_df) > 0:
            success_rate = game_df['success'].mean() * 100
            report_lines.append(f"* Overall Success Rate: {success_rate:.2f}%")
            
            report_lines.append("* Model Statistics:")
            for model in game_df['model'].unique():
                model_df = game_df[game_df['model'] == model]
                model_success = model_df['success'].mean() * 100
                report_lines.append(f"  * {model}: {model_success:.2f}% success rate")
                
                if game_type == "spyfall" and 'spy_detected' in model_df.columns:
                    spy_detected = model_df['spy_detected'].mean() * 100
                    report_lines.append(f"    * Spy Detection Rate: {spy_detected:.2f}%")
                    avg_rounds = model_df['elimination_speed'].mean()
                    report_lines.append(f"    * Average Rounds: {avg_rounds:.2f}")
                    
                elif game_type == "askguess" and 'questions' in model_df.columns:
                    avg_questions = model_df['questions'].mean()
                    report_lines.append(f"    * Average Questions: {avg_questions:.2f}")
                    
                elif game_type == "tofukingdom":
                    if 'rounds' in model_df.columns:
                        avg_rounds = model_df['rounds'].mean()
                        report_lines.append(f"    * Average Rounds: {avg_rounds:.2f}")
                    
                    role_columns = ['spy_won', 'prince_won', 'queen_won']
                    if all(col in model_df.columns for col in role_columns):
                        spy_win_rate = model_df['spy_won'].mean() * 100
                        prince_win_rate = model_df['prince_won'].mean() * 100
                        queen_win_rate = model_df['queen_won'].mean() * 100
                        report_lines.append(f"    * Spy Win Rate: {spy_win_rate:.2f}%")
                        report_lines.append(f"    * Prince Win Rate: {prince_win_rate:.2f}%")
                        report_lines.append(f"    * Queen Win Rate: {queen_win_rate:.2f}%")
        
        report_lines.append("")
    
    report_lines.append("## Model Rankings")
    model_success = df.groupby('model')['success'].mean().reset_index().sort_values('success', ascending=False)
    
    for i, (_, row) in enumerate(model_success.iterrows(), 1):
        model_name = row['model']
        success_rate = row['success'] * 100
        report_lines.append(f"{i}. **{model_name}**: {success_rate:.2f}% success rate")
    
    report_lines.append("")
    report_lines.append("## Visualizations")
    report_lines.append("Visualizations available in `visualizations/` directory.")
    
    report_path = os.path.join(output_dir, "benchmark_report.md")
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Report saved to {report_path}")

def main():
    """Main entry point for benchmark visualization."""
    parser = argparse.ArgumentParser(description="PolitAgent Benchmark Visualizer - visualize benchmark results")
    parser.add_argument('--results_dir', type=str, required=True,
                      help="Directory containing benchmark results")
    parser.add_argument('--output_dir', type=str, default=None,
                      help="Directory for saving visualizations and report (default: inside results directory)")
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        logger.error(f"Results directory not found: {args.results_dir}")
        return
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "analysis")
    
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    summary, results = load_results(args.results_dir)
    
    if not results:
        logger.error("No results found or invalid format.")
        return
    
    performance_df = create_performance_dataframe(results)
    
    performance_df.to_csv(os.path.join(args.output_dir, "performance_data.csv"), index=False)
    
    visualize_model_comparison(performance_df, vis_dir)
    
    generate_report(summary, performance_df, args.output_dir)
    
    logger.info(f"Analysis complete. Results available in {args.output_dir}")

if __name__ == "__main__":
    main() 