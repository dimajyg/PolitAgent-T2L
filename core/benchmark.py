import argparse
import json
import os
import multiprocessing
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import importlib
import random
import traceback
import statistics
from pathlib import Path

from llm.models import get_model, get_available_models

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark")

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
            "max_rounds": 8,
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
            "max_rounds": 3,
            "debug": True,
            "diplomacy_model_name": "openai"
        },
        "requires_phrases": False,
        "model_args": ["diplomacy_model_name"]
    }
}

AVAILABLE_MODELS = list(get_available_models().keys())

def setup_results_dir() -> str:
    """Creates and returns a directory for saving results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"benchmark_results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def load_phrases(game_type: str, args: argparse.Namespace) -> List[Any]:
    """Loads phrases/words for games requiring external data."""
    if game_type == "spyfall":
        with open(args.label_path, 'r') as f:
            phrases = [line.strip().split(",") for line in f.readlines()]
            if hasattr(args, 'max_phrases') and args.max_phrases is not None:
                return phrases[:args.max_phrases]
            return phrases
    elif game_type == "askguess":
        with open(args.label_path, 'r') as f:
            try:
                all_phrases = json.load(f)
                if isinstance(all_phrases, list):
                    if hasattr(args, 'max_phrases') and args.max_phrases is not None:
                        return all_phrases[:args.max_phrases]
                    return all_phrases
                else:
                    if hasattr(args, 'max_phrases') and args.max_phrases is not None:
                        keys = list(all_phrases.keys())[:args.max_phrases]
                        return {k: all_phrases[k] for k in keys}
                    return all_phrases
            except json.JSONDecodeError:
                f.seek(0)
                phrases = [line.strip() for line in f.readlines()]
                if hasattr(args, 'max_phrases') and args.max_phrases is not None:
                    return phrases[:args.max_phrases]
                return phrases
    return [None] 

def run_game(game_config: Tuple[str, Dict, Any, int, str]) -> Dict[str, Any]:
    """
    Runs a single game session.
    
    Args:
        game_config: (game_type, arguments, phrase, run_id, results_dir)
        
    Returns:
        Dict with game results
    """
    game_type, args_dict, phrase, run_id, results_dir = game_config
    args = argparse.Namespace(**args_dict)

    game_info = GAME_ENVIRONMENTS[game_type]
    
    models = {}
    for model_arg in game_info["model_args"]:
        model_name = args_dict.get(model_arg, "openai")
        if model_name not in models:
            model_kwargs = {}
            if args_dict.get("specific_model"):
                model_kwargs["specific_model"] = args_dict.get("specific_model")
            if model_name == "ollama" and args_dict.get("ollama_base_url"):
                model_kwargs["base_url"] = args_dict.get("ollama_base_url")
            
            models[model_name] = get_model(model_name, **model_kwargs)
    
    if args_dict.get("use_llm_evaluation", False) and args_dict.get("evaluation_model") is not None:
        evaluation_model_name = args_dict.get("evaluation_model")
        if evaluation_model_name not in models:
            model_kwargs = {}
            if args_dict.get("specific_model"):
                model_kwargs["specific_model"] = args_dict.get("specific_model")
            if evaluation_model_name == "ollama" and args_dict.get("ollama_base_url"):
                model_kwargs["base_url"] = args_dict.get("ollama_base_url")
                
            models["evaluation_model"] = get_model(evaluation_model_name, **model_kwargs)
            args.evaluation_model = models["evaluation_model"]
    
    game_module = importlib.import_module(game_info["module"])
    game_class = getattr(game_module, game_info["class"])
    
    model_names = "_".join([args_dict.get(arg, "default") for arg in game_info["model_args"]])
    log_dir = f"{results_dir}/{game_type}/{model_names}"
    
    if phrase:
        if isinstance(phrase, list):
            phrase_str = "&".join(phrase)
        else:
            phrase_str = phrase
        log_dir = f"{log_dir}/{phrase_str}"
    
    os.makedirs(log_dir, exist_ok=True)
    
    os.environ["BENCHMARK_RESULTS_DIR"] = log_dir
    
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
        
        princess_model = models.get(args.princess_model_name, prince_model)
        queen_model = models.get(args.queen_model_name, prince_model)
        neutral_model = models.get(args.neutral_model_name, prince_model)
        
        game = game_class(args, prince_model, princess_model, queen_model, neutral_model)
        settings = game.init_game()
    
    elif game_type == "diplomacy":
        game = game_class(args, models[args.diplomacy_model_name])
        settings = game.init_game()
    
    else:
        logger.error(f"Unknown game type: {game_type}")
        return {"error": f"Unknown game type: {game_type}"}
    
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
            
            with open(f"{log_dir}/{run_id}_result.json", "w") as rf:
                json.dump(result, rf, indent=2)
                
            with open(f"{results_dir}/all_results.jsonl", "a") as arf:
                arf.write(json.dumps(result) + "\n")
    
    logger.info(f"Completed game {game_type} (run {run_id}) with {'phrase ' + str(phrase) if phrase else 'no phrase'}")
    return result

def run_benchmark(args: argparse.Namespace) -> None:
    """Runs benchmark games based on command line arguments."""
    start_time = datetime.now()
    results_dir = setup_results_dir()
    logger.info(f"Results will be saved in {results_dir}")
    
    if args.full_benchmark:
        logger.info("🚀 Running full benchmark on all games...")
        args.games = ",".join(GAME_ENVIRONMENTS.keys())
        args.runs_per_game = max(args.runs_per_game, 3)
        args.max_phrases = args.max_phrases or 5
        logger.info(f"Full benchmark settings: {args.runs_per_game} runs per game, up to {args.max_phrases} phrases")
    
    with open(f"{results_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    games_to_run = args.games.split(",") if args.games else list(GAME_ENVIRONMENTS.keys())
    
    for game in games_to_run:
        if game not in GAME_ENVIRONMENTS:
            logger.error(f"Unknown game: {game}. Available games: {', '.join(GAME_ENVIRONMENTS.keys())}")
            return
    
    if args.models:
        models = args.models.split(",")
        for model in models:
            if model not in AVAILABLE_MODELS:
                logger.error(f"Unknown model: {model}. Available models: {', '.join(AVAILABLE_MODELS)}")
                return
    
    tasks = []
    
    for game_type in games_to_run:
        game_info = GAME_ENVIRONMENTS[game_type]
        
        game_args = vars(args).copy()
        
        for key, value in game_info["default_args"].items():
            if key not in game_args or game_args[key] is None:
                game_args[key] = value
            elif key == "label_path" and game_type == "askguess" and game_args[key] == "environments/spyfall/prompts/labels.txt":
                game_args[key] = game_info["default_args"][key]
        
        if args.models:
            models_list = args.models.split(",")
            for model_arg in game_info["model_args"]:
                if len(models_list) > 1:
                    game_args[model_arg] = random.choice(models_list)
                else:
                    game_args[model_arg] = models_list[0]
                    
        if args.model_args:
            game_args.update(args.model_args)
        
        if game_info["requires_phrases"]:
            phrases = load_phrases(game_type, argparse.Namespace(**game_args))
        else:
            phrases = [None]
        
        for phrase in phrases:
            for run_id in range(args.runs_per_game):
                tasks.append((game_type, game_args, phrase, run_id, results_dir))
    
    logger.info(f"Running {len(tasks)} game sessions using {args.workers} worker processes")
    
    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            results = pool.map(run_game, tasks)
    else:
        results = [run_game(task) for task in tasks]
    
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
    
    if args.full_benchmark:
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("📊 Generating full benchmark report...")
        full_report = generate_full_benchmark_report(results_dir, results, args)
        
        full_report["benchmark_metadata"]["benchmark_duration"] = str(duration)
        
        report_path = Path(results_dir) / "full_benchmark_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        
        overall_score = full_report["overall_performance"]["overall_score"]
        model_rating = full_report["overall_performance"]["model_rating"]
        
        print("\n" + "="*80)
        print("🏆 FULL BENCHMARK COMPLETED")
        print("="*80)
        print(f"📊 Overall model score: {overall_score:.2f}/100")
        print(f"🎯 Rating: {model_rating}")
        print(f"⏱️ Execution time: {duration}")
        print(f"🎮 Games played: {len(tasks)}")
        print(f"✅ Successful completions: {summary['completed_games']}")
        print(f"📁 Full report: {report_path}")
        print("="*80)
        
        if full_report["overall_performance"]["strengths"]:
            print("\n🌟 Strengths:")
            for strength in full_report["overall_performance"]["strengths"]:
                print(f"  • {strength}")
        
        if full_report["overall_performance"]["weaknesses"]:
            print("\n⚠️ Areas for improvement:")
            for weakness in full_report["overall_performance"]["weaknesses"]:
                print(f"  • {weakness}")
        
        if full_report["overall_performance"]["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in full_report["overall_performance"]["recommendations"]:
                print(f"  • {rec}")
        
        print("\n")
    
    logger.info(f"Benchmark completed. Total {len(tasks)} games, successfully completed {summary['completed_games']}.")
    logger.info(f"Results available in {results_dir}")

def calculate_game_metrics(results: List[Dict[str, Any]], game_type: str) -> Dict[str, Any]:
    """Calculates metrics for a specific game."""
    game_results = [r for r in results if r and r.get("game_type") == game_type and "error" not in r]
    
    if not game_results:
        return {
            "total_games": 0,
            "success_rate": 0.0,
            "average_score": 0.0,
            "metrics": {}
        }
    
    total_games = len(game_results)
    success_rate = len(game_results) / len([r for r in results if r and r.get("game_type") == game_type])
    
    metrics = {"total_games": total_games, "success_rate": success_rate}
    
    if game_type == "spyfall":
        spy_wins = sum(1 for r in game_results if r.get("winner") == "spy")
        villager_wins = sum(1 for r in game_results if r.get("winner") == "villager")
        metrics.update({
            "spy_win_rate": spy_wins / total_games if total_games > 0 else 0,
            "villager_win_rate": villager_wins / total_games if total_games > 0 else 0,
            "average_rounds": statistics.mean([r.get("rounds", 0) for r in game_results]) if game_results else 0
        })
        
    elif game_type == "beast":
        winners = [r.get("winner") for r in game_results if r.get("winner")]
        elimination_rounds = [r.get("elimination_round", 0) for r in game_results if r.get("elimination_round")]
        metrics.update({
            "average_elimination_round": statistics.mean(elimination_rounds) if elimination_rounds else 0,
            "survival_rate": len([r for r in game_results if r.get("survived", False)]) / total_games if total_games > 0 else 0,
            "winners": len(set(winners)) if winners else 0
        })
        
    elif game_type == "askguess":
        correct_guesses = [r for r in game_results if r.get("correct_guess", False)]
        rounds_to_solve = [r.get("rounds_to_solve", 0) for r in game_results if r.get("rounds_to_solve")]
        metrics.update({
            "correct_guess_rate": len(correct_guesses) / total_games if total_games > 0 else 0,
            "average_rounds_to_solve": statistics.mean(rounds_to_solve) if rounds_to_solve else 0,
            "efficiency_score": len(correct_guesses) / sum(rounds_to_solve) if rounds_to_solve else 0
        })
        
    elif game_type == "tofukingdom":
        princess_found = sum(1 for r in game_results if r.get("princess_found", False))
        metrics.update({
            "princess_detection_rate": princess_found / total_games if total_games > 0 else 0,
            "average_game_duration": statistics.mean([r.get("duration", 0) for r in game_results]) if game_results else 0
        })
        
    elif game_type == "diplomacy":
        country_wins = {}
        for r in game_results:
            winner = r.get("winner")
            if winner:
                country_wins[winner] = country_wins.get(winner, 0) + 1
        
        metrics.update({
            "country_distribution": country_wins,
            "diplomatic_success": len([r for r in game_results if r.get("diplomatic_actions", 0) > 0]) / total_games if total_games > 0 else 0,
            "average_rounds": statistics.mean([r.get("total_rounds", 0) for r in game_results]) if game_results else 0
        })
    
    base_score = success_rate * 100
    
    quality_bonus = 0
    if game_type == "spyfall":
        balance = abs(0.5 - metrics["spy_win_rate"])
        quality_bonus = (0.5 - balance) * 20
        
    elif game_type == "beast":
        quality_bonus = metrics["survival_rate"] * 15
        
    elif game_type == "askguess":
        quality_bonus = metrics["efficiency_score"] * 20
        
    elif game_type == "tofukingdom":
        quality_bonus = metrics["princess_detection_rate"] * 25
        
    elif game_type == "diplomacy":
        quality_bonus = metrics["diplomatic_success"] * 15
    
    average_score = min(100, base_score + quality_bonus)
    metrics["average_score"] = average_score
    
    return metrics

def generate_full_benchmark_report(results_dir: str, results: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    """Generates a full benchmark report with overall model score."""
    
    games_tested = list(set(r.get("game_type") for r in results if r and r.get("game_type")))
    
    game_metrics = {}
    game_scores = {}
    
    for game_type in games_tested:
        metrics = calculate_game_metrics(results, game_type)
        game_metrics[game_type] = metrics
        game_scores[game_type] = metrics.get("average_score", 0)
    
    game_weights = {
        "diplomacy": 0.25,
        "beast": 0.20,
        "spyfall": 0.20,
        "tofukingdom": 0.20,
        "askguess": 0.15
    }
    
    weighted_score = 0
    total_weight = 0
    
    for game_type, score in game_scores.items():
        weight = game_weights.get(game_type, 0.1)
        weighted_score += score * weight
        total_weight += weight
    
    overall_score = weighted_score / total_weight if total_weight > 0 else 0
    
    def get_model_rating(score: float) -> str:
        if score >= 90: return "🏆 Exceptional (A+)"
        elif score >= 80: return "🥇 Excellent (A)"
        elif score >= 70: return "🥈 Good (B+)"
        elif score >= 60: return "🥉 Above Average (B)"
        elif score >= 50: return "📈 Average (C+)"
        elif score >= 40: return "📉 Below Average (C)"
        elif score >= 30: return "⚠️ Poor (D)"
        else: return "❌ Critical (F)"
    
    model_rating = get_model_rating(overall_score)
    
    full_report = {
        "benchmark_metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_tested": args.specific_model or "default",
            "total_games_run": len([r for r in results if r]),
            "successful_games": len([r for r in results if r and "error" not in r]),
            "games_tested": games_tested,
            "benchmark_duration": "calculated_separately"
        },
        "overall_performance": {
            "overall_score": round(overall_score, 2),
            "model_rating": model_rating,
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        },
        "game_performance": game_metrics,
        "detailed_analysis": {
            "game_scores": game_scores,
            "game_weights": game_weights,
            "performance_breakdown": {}
        }
    }
    
    sorted_games = sorted(game_scores.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_games:
        for game, score in sorted_games[:2]:
            if score > 70:
                full_report["overall_performance"]["strengths"].append(f"Excellent performance in {game} ({score:.1f}/100)")
        
        for game, score in sorted_games[-2:]:
            if score < 60:
                full_report["overall_performance"]["weaknesses"].append(f"Needs improvement in {game} ({score:.1f}/100)")
    
    if overall_score < 50:
        full_report["overall_performance"]["recommendations"].append("Consider fine-tuning on strategic game scenarios")
    if any(score < 40 for score in game_scores.values()):
        full_report["overall_performance"]["recommendations"].append("Focus on games with lowest performance scores")
    if overall_score > 80:
        full_report["overall_performance"]["recommendations"].append("Excellent model - consider testing on more complex scenarios")
    
    report_path = Path(results_dir) / "full_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    md_report = generate_markdown_report(full_report)
    md_path = Path(results_dir) / "full_benchmark_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    
    logger.info(f"📊 Full benchmark report generated: {report_path}")
    logger.info(f"📋 Markdown report generated: {md_path}")
    logger.info(f"🎯 Overall Model Score: {overall_score:.2f}/100 ({model_rating})")
    
    return full_report

def generate_markdown_report(report: Dict[str, Any]) -> str:
    """Generates markdown report."""
    md = f"""# PolitAgent Full Benchmark Report

## Model Performance Summary

**Overall Score:** {report['overall_performance']['overall_score']}/100  
**Rating:** {report['overall_performance']['model_rating']}  
**Generated:** {report['benchmark_metadata']['timestamp']}  

---

## Executive Summary

This model was tested on {len(report['benchmark_metadata']['games_tested'])} different strategic games:
{', '.join(report['benchmark_metadata']['games_tested'])}.

**Total Games Played:** {report['benchmark_metadata']['total_games_run']}  
**Successful Completions:** {report['benchmark_metadata']['successful_games']}  
**Success Rate:** {report['benchmark_metadata']['successful_games']/report['benchmark_metadata']['total_games_run']*100:.1f}%

---

## Game Performance Breakdown

"""
    
    for game, metrics in report['game_performance'].items():
        md += f"""### {game.title()}
- **Score:** {metrics.get('average_score', 0):.1f}/100
- **Success Rate:** {metrics.get('success_rate', 0)*100:.1f}%
- **Games Played:** {metrics.get('total_games', 0)}

"""
        
        if game == "spyfall":
            md += f"- **Spy Win Rate:** {metrics.get('spy_win_rate', 0)*100:.1f}%\n"
            md += f"- **Villager Win Rate:** {metrics.get('villager_win_rate', 0)*100:.1f}%\n"
        elif game == "beast":
            md += f"- **Survival Rate:** {metrics.get('survival_rate', 0)*100:.1f}%\n"
            md += f"- **Avg Elimination Round:** {metrics.get('average_elimination_round', 0):.1f}\n"
        elif game == "askguess":
            md += f"- **Correct Guess Rate:** {metrics.get('correct_guess_rate', 0)*100:.1f}%\n"
            md += f"- **Efficiency Score:** {metrics.get('efficiency_score', 0):.2f}\n"
        elif game == "tofukingdom":
            md += f"- **Princess Detection Rate:** {metrics.get('princess_detection_rate', 0)*100:.1f}%\n"
        elif game == "diplomacy":
            md += f"- **Diplomatic Success:** {metrics.get('diplomatic_success', 0)*100:.1f}%\n"
            md += f"- **Average Rounds:** {metrics.get('average_rounds', 0):.1f}\n"
        
        md += "\n"
    
    if report['overall_performance']['strengths']:
        md += "## Strengths\n"
        for strength in report['overall_performance']['strengths']:
            md += f"- ✅ {strength}\n"
        md += "\n"
    
    if report['overall_performance']['weaknesses']:
        md += "## Areas for Improvement\n"
        for weakness in report['overall_performance']['weaknesses']:
            md += f"- ⚠️ {weakness}\n"
        md += "\n"
    
    if report['overall_performance']['recommendations']:
        md += "## Recommendations\n"
        for rec in report['overall_performance']['recommendations']:
            md += f"- 💡 {rec}\n"
        md += "\n"
    
    md += """---

## Scoring Methodology

- **Overall Score:** Weighted average across all games
- **Game Weights:** Diplomacy (25%), Beast/Spyfall/TofuKingdom (20% each), AskGuess (15%)
- **Rating Scale:** 90+ (A+), 80+ (A), 70+ (B+), 60+ (B), 50+ (C+), 40+ (C), 30+ (D), <30 (F)

"""
    
    return md

def add_benchmark_args(parser):
    """Add benchmark-specific arguments to the parser."""
    parser.add_argument(
        '--evaluation_model',
        type=str,
        default='',
        help='Model to use for evaluating game results (LLM as judge)'
    )

def main():
    """Entry point of the program."""
    parser = argparse.ArgumentParser(description="PolitAgent Benchmark - evaluate language models in game environments")
    
    parser.add_argument('--models', type=str, default=None, 
                        help="Common list of models separated by commas (openai,mistral)")
    parser.add_argument('--games', type=str, default=None,
                        help="Games to run separated by commas (spyfall,beast,askguess,tofukingdom). Default - all.")
    parser.add_argument('--workers', type=int, default=1,
                        help="Number of parallel processes")
    parser.add_argument('--runs_per_game', type=int, default=1,
                        help="Number of runs per game/phrase combination")
    parser.add_argument('--debug', type=bool, default=False,
                        help="Debug mode with detailed output")
    
    parser.add_argument('--full_benchmark', action='store_true',
                        help="Run full benchmark on all games with overall report and model score")
    
    parser.add_argument('--max_phrases', type=int, default=None,
                        help="Maximum number of phrases for games with phrases (spyfall, askguess)")
    
    parser.add_argument('--use_llm_evaluation', type=bool, default=False,
                        help="Enable game process evaluation using LLM")
    parser.add_argument('--evaluation_model', type=str, default=None,
                        help="Model for game process evaluation. Default uses game model.")
    
    parser.add_argument('--specific_model', type=str, default=None,
                        help="Specific provider model (e.g., 'gpt-4' for OpenAI or 'llama2' for Ollama)")
    parser.add_argument('--ollama_base_url', type=str, default="http://localhost:11434",
                        help="URL for accessing Ollama API (default http://localhost:11434)")
    
    parser.add_argument('--label_path', type=str, default="environments/spyfall/prompts/labels.txt",
                        help="Path to file with phrases for Spyfall")
    parser.add_argument('--spy_model_name', type=str, default=None,
                        help="Model for spy (used in Spyfall and TofuKingdom)")
    parser.add_argument('--villager_model_name', type=str, default=None,
                        help="Model for villagers in Spyfall")
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help="OpenAI API key (overrides environment variable)")
    parser.add_argument('--embedding_model', type=str, default=None,
                        help="Embedding model type (local, openai, auto)")
    parser.add_argument('--embedding_model_name', type=str, default=None,
                        help="Embedding model name")
    parser.add_argument('--perplexity_model', type=str, default=None,
                        help="Model for calculating perplexity (auto/local/model name)")
    
    parser.add_argument('--model_name', type=str, default=None,
                        help="Model name for Beast and AskGuess")
    
    parser.add_argument('--mode', type=str, default=None,
                        help="Game mode (used in askguess)")
    parser.add_argument('--max_rounds', type=int, default=None,
                        help="Maximum number of rounds in AskGuess game")
    
    parser.add_argument('--prince_model_name', type=str, default=None,
                        help="Model for prince in TofuKingdom")
    parser.add_argument('--princess_model_name', type=str, default=None,
                        help="Model for princess in TofuKingdom")
    parser.add_argument('--queen_model_name', type=str, default=None,
                        help="Model for queen in TofuKingdom")
    parser.add_argument('--neutral_model_name', type=str, default=None,
                        help="Model for neutral character in TofuKingdom")
    
    parser.add_argument('--diplomacy_model_name', type=str, default=None,
                        help="Model for players in Diplomacy")

    parser.add_argument('--model_args', type=json.loads, default=None,
                        help='JSON string with additional model arguments')

    
    args = parser.parse_args()
    
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    run_benchmark(args)

if __name__ == "__main__":
    main() 