#!/usr/bin/env python3
"""
Example script demonstrating enhanced Diplomacy metrics with comprehensive model evaluation.

This script shows how to:
1. Calculate detailed inference metrics
2. Perform LLM as judge evaluation  
3. Generate human-readable reports
4. Compare different models
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.diplomacy_metrics import DiplomacyMetrics
from llm.openai_chat import OpenAIChatModel
from llm import config as llm_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_game_data() -> Dict[str, Any]:
    """
    Create sample game data for demonstration purposes.
    In real usage, this would come from actual game results.
    """
    return {
        "rounds_played": 8,
        "winner": "FRANCE",
        "game_time": 1847.3,
        "supply_centers": {
            "AUSTRIA": 2,
            "ENGLAND": 4,
            "FRANCE": 18,  # Winner
            "GERMANY": 1,
            "ITALY": 3,
            "RUSSIA": 5,
            "TURKEY": 1
        },
        "strategic_decisions": {
            "FRANCE": "Focus on central Europe expansion",
            "ENGLAND": "Naval dominance strategy",
            "RUSSIA": "Eastern expansion with German alliance"
        },
        "rounds_data": [
            {
                "round": 1,
                "year": 1901,
                "phase": "Spring",
                "territories_before": {
                    "FRANCE": ["PAR", "MAR", "BRE"],
                    "GERMANY": ["BER", "MUN", "KIE"],
                    "ENGLAND": ["LON", "LVP", "EDI"]
                },
                "territories_after": {
                    "FRANCE": ["PAR", "MAR", "BRE", "BUR"],
                    "GERMANY": ["BER", "MUN", "KIE", "HOL"],
                    "ENGLAND": ["LON", "LVP", "EDI", "NWG"]
                },
                "negotiations": {
                    "FRANCE": {
                        "GERMANY": {
                            "0": "I propose we work together against England. I'll support your move to Holland if you help me take Burgundy."
                        },
                        "ENGLAND": {
                            "0": "Let's maintain peace in the Channel. I won't attack your fleet if you don't attack mine."
                        }
                    },
                    "GERMANY": {
                        "FRANCE": {
                            "0": "Agreed! Let's coordinate our spring moves. I'll move to Holland, you take Burgundy."
                        }
                    }
                },
                "orders": {
                    "FRANCE": ["A PAR - BUR", "A MAR - SPA", "F BRE - MAO"],
                    "GERMANY": ["A BER - KIE", "A MUN S A BER - KIE", "F KIE - HOL"],
                    "ENGLAND": ["F LON - NTH", "F EDI - NWG", "A LVP - YOR"]
                },
                "attacks_received": {
                    "FRANCE": [],
                    "GERMANY": [],
                    "ENGLAND": []
                }
            },
            {
                "round": 2,
                "year": 1901,
                "phase": "Fall",
                "territories_before": {
                    "FRANCE": ["PAR", "MAR", "BRE", "BUR"],
                    "GERMANY": ["BER", "MUN", "KIE", "HOL"],
                    "ENGLAND": ["LON", "LVP", "EDI", "NWG"]
                },
                "territories_after": {
                    "FRANCE": ["PAR", "MAR", "BRE", "BUR", "SPA"],
                    "GERMANY": ["BER", "MUN", "KIE", "HOL", "DEN"],
                    "ENGLAND": ["LON", "LVP", "EDI", "NWG", "NOR"]
                },
                "negotiations": {
                    "FRANCE": {
                        "ITALY": {
                            "0": "I see you're expanding north. Let's divide Austria between us - you take Trieste, I'll take Vienna."
                        }
                    },
                    "GERMANY": {
                        "RUSSIA": {
                            "0": "Russia, I propose a non-aggression pact. Let's focus our attention westward while you secure the south."
                        }
                    }
                },
                "orders": {
                    "FRANCE": ["A BUR - MUN", "A SPA - POR", "F MAO S A SPA - POR"],
                    "GERMANY": ["A KIE - DEN", "A MUN S A KIE - DEN", "F HOL - NTH"],
                    "ENGLAND": ["F NTH - NOR", "F NWG S F NTH - NOR", "A YOR - LON"]
                },
                "attacks_received": {
                    "FRANCE": [],
                    "GERMANY": [{"from": "FRANCE", "target": "MUN", "order": "A BUR - MUN"}],
                    "ENGLAND": []
                }
            }
        ]
    }

def evaluate_model_performance(results_dir: str, evaluator_model: Optional[Any] = None) -> Dict[str, Any]:
    """
    Evaluate model performance using enhanced Diplomacy metrics.
    
    Args:
        results_dir: Directory containing game result files
        evaluator_model: Optional LLM model for evaluation
        
    Returns:
        Dict[str, Any]: Comprehensive metrics and analysis
    """
    # Initialize metrics calculator
    metrics_calculator = DiplomacyMetrics(model=evaluator_model)
    
    # Calculate metrics
    logger.info("Calculating comprehensive Diplomacy metrics...")
    metrics = metrics_calculator.calculate_metrics(results_dir)
    
    return metrics

def demonstrate_with_sample_data():
    """Demonstrate metrics calculation with sample data."""
    logger.info("Demonstrating enhanced Diplomacy metrics with sample data")
    
    # Create sample data directory
    sample_dir = "sample_diplomacy_results"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Generate sample game data
    sample_games = [create_sample_game_data() for _ in range(3)]
    
    # Add some variation to the games
    sample_games[1]["winner"] = "GERMANY"
    sample_games[1]["supply_centers"]["GERMANY"] = 18
    sample_games[1]["supply_centers"]["FRANCE"] = 3
    
    sample_games[2]["winner"] = None  # Draw
    sample_games[2]["supply_centers"] = {
        "AUSTRIA": 4, "ENGLAND": 5, "FRANCE": 6,
        "GERMANY": 5, "ITALY": 4, "RUSSIA": 6, "TURKEY": 4
    }
    
    # Save sample games
    for i, game in enumerate(sample_games):
        with open(f"{sample_dir}/diplomacy_game_{i+1}.json", 'w') as f:
            json.dump(game, f, indent=2)
    
    # Initialize LLM evaluator (optional - comment out if no API key)
    try:
        if llm_config.openai_api_key:
            evaluator = OpenAIChatModel(
                model=llm_config.model_openai,
                temperature=llm_config.temperature_openai,
                api_key=llm_config.openai_api_key
            )
            logger.info("LLM evaluator initialized for comprehensive analysis")
        else:
            evaluator = None
            logger.warning("No OpenAI API key found, skipping LLM evaluation")
    except Exception as e:
        logger.warning(f"Could not initialize LLM evaluator: {e}")
        evaluator = None
    
    # Calculate metrics
    metrics_calculator = DiplomacyMetrics(model=evaluator)
    metrics = metrics_calculator.calculate_metrics(sample_dir)
    
    # Print key results
    print_metrics_summary(metrics)
    
    # Generate detailed report
    metrics_calculator.save_detailed_report("sample_diplomacy_analysis")
    logger.info("Detailed report saved to sample_diplomacy_analysis.md and sample_diplomacy_analysis.json")
    
    # Clean up sample data
    import shutil
    shutil.rmtree(sample_dir)
    logger.info("Sample data cleaned up")
    
    return metrics

def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """Print a summary of key metrics."""
    print("\n" + "="*60)
    print("DIPLOMACY MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    # Basic stats
    print(f"\nGames Analyzed: {metrics.get('games_total', 0)}")
    
    # Model performance
    model_perf = metrics.get('model_performance', {})
    print(f"\nMODEL INFERENCE METRICS:")
    print(f"  Total Inferences: {model_perf.get('total_inferences', 0):,}")
    print(f"  Negotiation Inferences: {model_perf.get('negotiation_inferences', 0):,}")
    print(f"  Action Inferences: {model_perf.get('action_inferences', 0):,}")
    print(f"  Total Errors: {model_perf.get('total_errors', 0)}")
    
    # Calculate average metrics
    def avg_metric(metric_dict):
        if not metric_dict:
            return 0.0
        values = [v for v in metric_dict.values() if isinstance(v, (int, float))]
        return sum(values) / len(values) if values else 0.0
    
    response_quality = avg_metric(model_perf.get('response_quality', {}))
    decision_consistency = avg_metric(model_perf.get('decision_consistency', {}))
    context_utilization = avg_metric(model_perf.get('context_utilization', {}))
    
    print(f"  Average Response Quality: {response_quality:.3f}")
    print(f"  Average Decision Consistency: {decision_consistency:.3f}")
    print(f"  Average Context Utilization: {context_utilization:.3f}")
    
    # Strategic performance
    strategic = metrics.get('strategic_metrics', {})
    win_rates = strategic.get('win_rate_by_power', {})
    
    print(f"\nSTRATEGIC PERFORMANCE:")
    if win_rates:
        best_power = max(win_rates, key=win_rates.get)
        worst_power = min(win_rates, key=win_rates.get)
        print(f"  Best Power: {best_power} ({win_rates[best_power]:.2%} win rate)")
        print(f"  Worst Power: {worst_power} ({win_rates[worst_power]:.2%} win rate)")
        
        print("\n  Win Rates by Power:")
        for power, rate in sorted(win_rates.items()):
            print(f"    {power}: {rate:.2%}")
    
    # LLM Evaluation (if available)
    llm_eval = metrics.get('llm_evaluation', {})
    if llm_eval:
        print(f"\nLLM JUDGE EVALUATION:")
        strategic_avg = llm_eval.get('strategic_avg', {})
        diplomatic_avg = llm_eval.get('diplomatic_avg', {})
        tactical_avg = llm_eval.get('tactical_avg', {})
        overall_avg = llm_eval.get('overall_avg', {})
        
        if strategic_avg:
            print("  Strategic Scores (1-10):")
            for power, score in sorted(strategic_avg.items()):
                print(f"    {power}: {score:.1f}")
        
        if overall_avg:
            best_overall = max(overall_avg, key=overall_avg.get)
            print(f"\n  Best Overall Performance: {best_overall} ({overall_avg[best_overall]:.1f}/10)")
    
    # Game outcomes
    outcomes = metrics.get('game_outcome_metrics', {})
    print(f"\nGAME OUTCOME ANALYSIS:")
    print(f"  Average Game Length: {outcomes.get('average_game_length', 0):.1f} rounds")
    print(f"  Decisive Victories: {outcomes.get('decisive_victories', 0)}")
    print(f"  Draws: {outcomes.get('draws', 0)}")
    print(f"  Fastest Victory: {outcomes.get('fastest_victory', 0)} rounds")
    
    print("\n" + "="*60)

def compare_models(results_dirs: Dict[str, str], evaluator_model: Optional[Any] = None) -> None:
    """
    Compare performance of different models across multiple result directories.
    
    Args:
        results_dirs: Dictionary mapping model names to result directories
        evaluator_model: Optional LLM for evaluation
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON ANALYSIS")
    print("="*60)
    
    all_metrics = {}
    
    for model_name, results_dir in results_dirs.items():
        logger.info(f"Analyzing {model_name}...")
        metrics = evaluate_model_performance(results_dir, evaluator_model)
        all_metrics[model_name] = metrics
    
    # Compare key metrics
    print(f"\nCOMPARISON ACROSS {len(results_dirs)} MODELS:")
    print("-" * 50)
    
    # Header
    print(f"{'Metric':<25} ", end="")
    for model_name in results_dirs.keys():
        print(f"{model_name:<15}", end="")
    print()
    
    # Win rate comparison
    print(f"{'Avg Win Rate':<25} ", end="")
    for model_name in results_dirs.keys():
        metrics = all_metrics[model_name]
        win_rates = metrics.get('strategic_metrics', {}).get('win_rate_by_power', {})
        avg_win_rate = sum(win_rates.values()) / len(win_rates) if win_rates else 0
        print(f"{avg_win_rate:.2%}{'':>8}", end="")
    print()
    
    # Error rate comparison
    print(f"{'Error Rate':<25} ", end="")
    for model_name in results_dirs.keys():
        metrics = all_metrics[model_name]
        model_perf = metrics.get('model_performance', {})
        error_rate = model_perf.get('total_errors', 0) / max(model_perf.get('total_inferences', 1), 1)
        print(f"{error_rate:.2%}{'':>8}", end="")
    print()
    
    # Response quality comparison
    print(f"{'Response Quality':<25} ", end="")
    for model_name in results_dirs.keys():
        metrics = all_metrics[model_name]
        model_perf = metrics.get('model_performance', {})
        response_quality = model_perf.get('response_quality', {})
        
        def avg_metric(metric_dict):
            if not metric_dict:
                return 0.0
            values = [v for v in metric_dict.values() if isinstance(v, (int, float))]
            return sum(values) / len(values) if values else 0.0
        
        avg_quality = avg_metric(response_quality)
        print(f"{avg_quality:.3f}{'':>9}", end="")
    print()
    
    print("-" * 50)

def main():
    """Main demonstration function."""
    print("Enhanced Diplomacy Metrics Demonstration")
    print("========================================")
    
    # Demonstrate with sample data
    demonstrate_with_sample_data()
    
    # Example of comparing multiple models (uncomment to use with real data)
    """
    # Compare different models
    model_results = {
        "GPT-4": "results/gpt4_diplomacy",
        "Claude-3": "results/claude3_diplomacy", 
        "Llama-3": "results/llama3_diplomacy"
    }
    
    # Initialize evaluator for comparison
    try:
        config = LLMConfig()
        evaluator = OpenAIChat(config, model_name="gpt-4o")
        compare_models(model_results, evaluator)
    except Exception as e:
        logger.warning(f"Could not perform model comparison: {e}")
    """

if __name__ == "__main__":
    main() 