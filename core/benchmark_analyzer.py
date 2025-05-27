#!/usr/bin/env python
"""
PolitAgent Benchmark Analyzer - comprehensive analysis of benchmark results
with proper handling of all game types and calculation of final scores.
"""

import argparse
import json
import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("benchmark_analyzer")

class BenchmarkAnalyzer:
    """Comprehensive analyzer for PolitAgent benchmark results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.summary = {}
        self.results = []
        self.metrics = {}
        
    def load_results(self) -> None:
        """Load all benchmark results from the directory."""
        summary_path = os.path.join(self.results_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
        
        # Load individual result files
        result_files = glob.glob(os.path.join(self.results_dir, "**", "*_result.json"), recursive=True)
        
        for file_path in result_files:
            with open(file_path, 'r') as f:
                try:
                    result = json.load(f)
                    result['file_path'] = file_path
                    self.results.append(result)
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing JSON file: {file_path}")
        
        logger.info(f"Loaded {len(self.results)} game results")
    
    def analyze_askguess_results(self) -> Dict[str, Any]:
        """Analyze AskGuess game results."""
        askguess_results = [r for r in self.results if r.get("game_type") == "askguess"]
        
        if not askguess_results:
            return {"total_games": 0}
        
        metrics = {
            "total_games": len(askguess_results),
            "successful_games": 0,
            "failed_games": 0,
            "avg_rounds": 0,
            "avg_questions": 0,
            "success_rate": 0,
            "efficiency": 0,
            "detailed_results": []
        }
        
        successful_rounds = []
        total_questions = []
        
        for result in askguess_results:
            # Determine success based on multiple criteria
            is_successful = (
                result.get("error_type") == "SuccessfulTrial" or
                result.get("error_type") == "EndingError" and 
                result.get("qa_history", [])[-1].get("role") == "questioner" and
                "apple" in result.get("qa_history", [])[-1].get("content", "").lower()
            )
            
            rounds = result.get("round", 0)
            qa_history = result.get("qa_history", [])
            num_questions = len([qa for qa in qa_history if qa.get("role") == "questioner"])
            
            if is_successful:
                metrics["successful_games"] += 1
                successful_rounds.append(rounds)
            else:
                metrics["failed_games"] += 1
            
            total_questions.append(num_questions)
            
            # Detailed analysis
            metrics["detailed_results"].append({
                "object": result.get("object"),
                "success": is_successful,
                "rounds": rounds,
                "questions": num_questions,
                "error_type": result.get("error_type"),
                "final_qa": qa_history[-2:] if len(qa_history) >= 2 else qa_history
            })
        
        metrics["success_rate"] = metrics["successful_games"] / metrics["total_games"]
        metrics["avg_rounds"] = np.mean([r["rounds"] for r in metrics["detailed_results"]])
        metrics["avg_questions"] = np.mean(total_questions)
        
        if successful_rounds:
            # Efficiency: earlier success = higher efficiency
            max_rounds = 10  # Default max rounds
            metrics["efficiency"] = np.mean([(max_rounds - r) / max_rounds for r in successful_rounds])
        
        return metrics
    
    def analyze_spyfall_results(self) -> Dict[str, Any]:
        """Analyze Spyfall game results."""
        spyfall_results = [r for r in self.results if r.get("game_type") == "spyfall"]
        
        if not spyfall_results:
            return {"total_games": 0}
        
        metrics = {
            "total_games": len(spyfall_results),
            "spy_wins": 0,
            "villager_wins": 0,
            "spy_detected": 0,
            "avg_rounds": 0,
            "success_rate": 0,
            "detailed_results": []
        }
        
        for result in spyfall_results:
            # Extract key game information
            spy_detected = result.get("spy_caught", False) or result.get("spy_detected", False)
            winner = result.get("winner", "")
            spy_win = (winner == "spy") or (not spy_detected and winner != "villager")
            villager_win = not spy_win
            
            if spy_win:
                metrics["spy_wins"] += 1
            else:
                metrics["villager_wins"] += 1
            
            if spy_detected:
                metrics["spy_detected"] += 1
            
            metrics["detailed_results"].append({
                "winner": winner,
                "spy_caught": spy_detected,
                "spy_win": spy_win,
                "rounds": result.get("round", 0),
                "spy_index": result.get("spy_index", -1),
                "players": result.get("players", [])
            })
        
        if spyfall_results:
            metrics["avg_rounds"] = np.mean([r.get("round", 0) for r in spyfall_results])
            # For spyfall, success rate could be from spy perspective or overall game completion
            metrics["success_rate"] = metrics["spy_wins"] / metrics["total_games"]
        
        return metrics
    
    def analyze_beast_results(self) -> Dict[str, Any]:
        """Analyze Beast game results."""
        beast_results = [r for r in self.results if r.get("game_type") == "beast"]
        
        if not beast_results:
            return {"total_games": 0}
        
        metrics = {
            "total_games": len(beast_results),
            "beast_wins": 0,
            "player_wins": 0,
            "avg_rounds": 0,
            "success_rate": 0,
            "detailed_results": []
        }
        
        for result in beast_results:
            winner = result.get("winner", "")
            rounds = result.get("total_rounds", 0)
            survivors = result.get("survivors", [])
            eliminated = result.get("eliminated_players", [])
            
            # In beast game, if there's a specific winner, that's success
            # Otherwise, check survivors vs eliminated
            if winner:
                if "beast" in winner.lower():
                    metrics["beast_wins"] += 1
                else:
                    metrics["player_wins"] += 1
            else:
                # Default: if there are survivors, players win
                if len(survivors) > 0:
                    metrics["player_wins"] += 1
                else:
                    metrics["beast_wins"] += 1
            
            metrics["detailed_results"].append({
                "winner": winner,
                "rounds": rounds,
                "survivors": len(survivors),
                "eliminated": len(eliminated),
                "final_rankings": result.get("final_rankings", []),
                "game_duration": result.get("game_duration", 0)
            })
        
        if beast_results:
            metrics["avg_rounds"] = np.mean([r.get("total_rounds", 0) for r in beast_results])
            metrics["success_rate"] = metrics["beast_wins"] / metrics["total_games"]
        
        return metrics
    
    def analyze_tofukingdom_results(self) -> Dict[str, Any]:
        """Analyze TofuKingdom game results."""
        tofukingdom_results = [r for r in self.results if r.get("game_type") == "tofukingdom"]
        
        if not tofukingdom_results:
            return {"total_games": 0}
        
        metrics = {
            "total_games": len(tofukingdom_results),
            "prince_wins": 0,
            "princess_wins": 0,
            "queen_wins": 0,
            "spy_wins": 0,
            "neutral_wins": 0,
            "timeouts": 0,
            "avg_rounds": 0,
            "success_rate": 0,
            "detailed_results": []
        }
        
        for result in tofukingdom_results:
            # Check for winner_team (new format) or winner (old format)
            winner_team = result.get("winner_team", "")
            winner = result.get("winner", "timeout")
            
            # Determine rounds from embedded metrics or timing
            rounds = 0
            if "metrics" in result and "timing" in result["metrics"]:
                rounds = result["metrics"]["timing"].get("rounds_count", 0)
            
            # Count wins by team
            if winner_team:
                if winner_team.lower() == "prince":
                    metrics["prince_wins"] += 1
                elif winner_team.lower() == "princess":
                    metrics["princess_wins"] += 1
                elif winner_team.lower() == "queen":
                    metrics["queen_wins"] += 1
                elif winner_team.lower() == "spy" or winner_team.lower() == "neutral":
                    metrics["spy_wins"] += 1
                else:
                    metrics["timeouts"] += 1
            elif winner:
                if winner == "prince":
                    metrics["prince_wins"] += 1
                elif winner == "princess":
                    metrics["princess_wins"] += 1
                elif winner == "queen":
                    metrics["queen_wins"] += 1
                elif winner == "spy":
                    metrics["spy_wins"] += 1
                else:
                    metrics["timeouts"] += 1
            else:
                metrics["timeouts"] += 1
            
            metrics["detailed_results"].append({
                "winner_team": winner_team or winner,
                "winners": result.get("winners", []),
                "rounds": rounds,
                "identities": result.get("identities", {}),
                "princess_guess": result.get("princess_guess", ""),
                "true_princess": result.get("true_princess", "")
            })
        
        if tofukingdom_results:
            rounds_list = [d["rounds"] for d in metrics["detailed_results"]]
            metrics["avg_rounds"] = np.mean(rounds_list) if rounds_list else 0
            # Success rate could be defined differently based on perspective
            metrics["success_rate"] = (metrics["prince_wins"] + metrics["princess_wins"]) / metrics["total_games"]
        
        return metrics
    
    def analyze_diplomacy_results(self) -> Dict[str, Any]:
        """Analyze Diplomacy game results."""
        diplomacy_results = [r for r in self.results if r.get("game_type") == "diplomacy"]
        
        if not diplomacy_results:
            return {"total_games": 0}
        
        metrics = {
            "total_games": len(diplomacy_results),
            "wins_by_country": {},
            "avg_game_length": 0,
            "avg_rounds": 0,
            "success_rate": 0,
            "detailed_results": []
        }
        
        for result in diplomacy_results:
            winner = result.get("winner", "")
            # Check both possible field names for game history
            game_history = result.get("game_history", [])
            if not game_history:
                game_history = result.get("rounds_data", [])
            rounds = len(game_history)
            territories_final = {}
            
            # Extract final territories if available
            if game_history:
                last_round = game_history[-1]
                territories_final = last_round.get("territories_after", {})
            
            if winner:
                metrics["wins_by_country"][winner] = metrics["wins_by_country"].get(winner, 0) + 1
            
            metrics["detailed_results"].append({
                "winner": winner,
                "rounds": rounds,
                "territories_final": territories_final,
                "game_duration": rounds  # Using rounds as proxy for duration
            })
        
        if diplomacy_results:
            # Check both possible field names for game history
            game_lengths = []
            for r in diplomacy_results:
                game_history = r.get("game_history", [])
                if not game_history:
                    game_history = r.get("rounds_data", [])
                game_lengths.append(len(game_history))
            
            metrics["avg_game_length"] = np.mean(game_lengths)
            metrics["avg_rounds"] = metrics["avg_game_length"]
            # Success rate could be defined as games that completed with a clear winner
            completed_games = len([r for r in diplomacy_results if r.get("winner")])
            metrics["success_rate"] = completed_games / metrics["total_games"] if metrics["total_games"] > 0 else 0
        
        return metrics
    
    def calculate_overall_score(self) -> Dict[str, Any]:
        """
        Calculate overall benchmark score across all games using a weighted scoring system.
        
        Scoring Methodology:
        1. Each game type has a different complexity weight
        2. Each game contributes: (success_rate * 0.7 + efficiency * 0.3) * weight
        3. Final score is weighted average across all played games
        4. Score range: 0.0 to 1.0 (higher is better)
        
        Game Weights (by complexity):
        - AskGuess: 1.0x (basic information gathering)
        - Spyfall: 1.0x (social deduction)
        - Beast: 1.0x (strategic negotiation)
        - TofuKingdom: 1.2x (logic + deception)
        - Diplomacy: 1.5x (complex multi-agent strategy)
        
        Returns:
            Dict containing overall_score, game_metrics, and detailed breakdown
        """
        game_metrics = {
            "askguess": self.analyze_askguess_results(),
            "spyfall": self.analyze_spyfall_results(), 
            "beast": self.analyze_beast_results(),
            "tofukingdom": self.analyze_tofukingdom_results(),
            "diplomacy": self.analyze_diplomacy_results()
        }
        
        # Calculate weighted overall score
        total_games = sum(m.get("total_games", 0) for m in game_metrics.values())
        if total_games == 0:
            return {
                "overall_score": 0.0,
                "total_games": 0,
                "game_metrics": game_metrics,
                "score_breakdown": {},
                "scoring_methodology": self._get_scoring_methodology_info()
            }
        
        # Game complexity weights - higher weights for more complex strategic environments
        complexity_weights = {
            "askguess": 1.0,      # Information gathering and deduction
            "spyfall": 1.0,       # Social deduction and role consistency  
            "beast": 1.0,         # Strategic negotiation and survival
            "tofukingdom": 1.2,   # Logic puzzles with deception
            "diplomacy": 1.5      # Complex multi-agent geopolitical strategy
        }
        
        weighted_score_sum = 0.0
        total_weight = 0.0
        game_contributions = {}
        
        for game_type, metrics in game_metrics.items():
            games_played = metrics.get("total_games", 0)
            if games_played > 0:
                weight = complexity_weights.get(game_type, 1.0)
                success_rate = metrics.get("success_rate", 0.0)
                efficiency = metrics.get("efficiency", success_rate)  # Fallback to success_rate
                
                # Combined score: 70% success rate + 30% efficiency
                game_score = (success_rate * 0.7 + efficiency * 0.3)
                weighted_game_score = game_score * weight
                
                weighted_score_sum += weighted_game_score
                total_weight += weight
                
                # Track each game's contribution to final score
                game_contributions[game_type] = {
                    "success_rate": success_rate,
                    "efficiency": efficiency,
                    "game_score": game_score,
                    "weight": weight,
                    "weighted_score": weighted_game_score,
                    "games_played": games_played,
                    "contribution_percentage": 0.0  # Will be calculated after total
                }
        
        # Calculate final overall score
        overall_score = weighted_score_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate each game's percentage contribution to final score
        for game_type in game_contributions:
            game_contributions[game_type]["contribution_percentage"] = (
                game_contributions[game_type]["weighted_score"] / weighted_score_sum * 100 
                if weighted_score_sum > 0 else 0.0
            )
        
        return {
            "overall_score": overall_score,
            "total_games": total_games,
            "total_weight": total_weight,
            "weighted_score_sum": weighted_score_sum,
            "game_metrics": game_metrics,
            "score_breakdown": game_contributions,
            "scoring_methodology": self._get_scoring_methodology_info(),
            "performance_classification": self._classify_performance(overall_score)
        }
    
    def _get_scoring_methodology_info(self) -> Dict[str, Any]:
        """Return detailed information about the scoring methodology."""
        return {
            "formula": "(success_rate * 0.7 + efficiency * 0.3) * complexity_weight",
            "success_rate_weight": 0.7,
            "efficiency_weight": 0.3,
            "complexity_weights": {
                "askguess": 1.0,
                "spyfall": 1.0,
                "beast": 1.0,
                "tofukingdom": 1.2,
                "diplomacy": 1.5
            },
            "score_range": "0.0 to 1.0 (higher is better)",
            "efficiency_fallback": "Uses success_rate if efficiency not available"
        }
    
    def _classify_performance(self, score: float) -> Dict[str, str]:
        """Classify performance based on overall score."""
        if score >= 0.8:
            return {
                "rating": "Excellent",
                "emoji": "🟢",
                "description": "Exceptional strategic thinking and game understanding across multiple environments"
            }
        elif score >= 0.6:
            return {
                "rating": "Good", 
                "emoji": "🟡",
                "description": "Solid performance with room for improvement in complex scenarios"
            }
        elif score >= 0.4:
            return {
                "rating": "Fair",
                "emoji": "🟠", 
                "description": "Basic competency but struggles with advanced strategic reasoning"
            }
        elif score >= 0.2:
            return {
                "rating": "Poor",
                "emoji": "🔴",
                "description": "Limited understanding of game mechanics and strategic thinking"
            }
        else:
            return {
                "rating": "Very Poor",
                "emoji": "⚫",
                "description": "Requires significant improvement in strategic reasoning and game comprehension"
            }
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive benchmark report."""
        overall_results = self.calculate_overall_score()
        
        report = f"""# PolitAgent Benchmark Analysis Report

## Executive Summary
- **Overall Score**: {overall_results['overall_score']:.3f} / 1.000
- **Total Games Analyzed**: {overall_results['total_games']}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Detailed Results by Game Type

"""
        
        for game_type, metrics in overall_results['game_metrics'].items():
            if metrics.get("total_games", 0) > 0:
                report += f"### {game_type.title()}\n"
                report += f"- **Total Games**: {metrics['total_games']}\n"
                report += f"- **Success Rate**: {metrics.get('success_rate', 0):.1%}\n"
                
                if game_type == "askguess":
                    report += f"- **Successful Games**: {metrics['successful_games']}\n"
                    report += f"- **Failed Games**: {metrics['failed_games']}\n"
                    report += f"- **Average Rounds**: {metrics['avg_rounds']:.1f}\n"
                    report += f"- **Average Questions**: {metrics['avg_questions']:.1f}\n"
                    report += f"- **Efficiency**: {metrics.get('efficiency', 0):.1%}\n"
                    
                    report += "\n**Detailed Game Results**:\n"
                    for i, detail in enumerate(metrics['detailed_results']):
                        status = "✅ SUCCESS" if detail['success'] else "❌ FAILED"
                        report += f"  {i+1}. Object: {detail['object']} - {status} - {detail['questions']} questions in {detail['rounds']} rounds\n"
                
                elif game_type == "spyfall":
                    report += f"- **Spy Wins**: {metrics['spy_wins']}\n"
                    report += f"- **Villager Wins**: {metrics['villager_wins']}\n"
                    report += f"- **Spy Detected**: {metrics['spy_detected']}\n"
                    report += f"- **Average Rounds**: {metrics['avg_rounds']:.1f}\n"
                    
                    if metrics.get('detailed_results'):
                        report += "\n**Detailed Game Results**:\n"
                        for i, detail in enumerate(metrics['detailed_results']):
                            status = "🕵️ SPY WIN" if detail['spy_win'] else "👥 VILLAGER WIN"
                            caught = " (Spy Caught)" if detail['spy_caught'] else ""
                            report += f"  {i+1}. {status}{caught} - {detail['rounds']} rounds\n"
                
                elif game_type == "beast":
                    report += f"- **Beast Wins**: {metrics['beast_wins']}\n"
                    report += f"- **Player Wins**: {metrics['player_wins']}\n"
                    report += f"- **Average Rounds**: {metrics['avg_rounds']:.1f}\n"
                    
                    if metrics.get('detailed_results'):
                        report += "\n**Detailed Game Results**:\n"
                        for i, detail in enumerate(metrics['detailed_results']):
                            winner_text = f"Winner: {detail['winner']}" if detail['winner'] else "No clear winner"
                            report += f"  {i+1}. {winner_text} - {detail['rounds']} rounds, {detail['survivors']} survivors\n"
                
                elif game_type == "tofukingdom":
                    report += f"- **Prince Wins**: {metrics['prince_wins']}\n"
                    report += f"- **Princess Wins**: {metrics['princess_wins']}\n"
                    report += f"- **Queen Wins**: {metrics['queen_wins']}\n"
                    report += f"- **Spy Wins**: {metrics['spy_wins']}\n"
                    report += f"- **Timeouts**: {metrics['timeouts']}\n"
                    report += f"- **Average Rounds**: {metrics['avg_rounds']:.1f}\n"
                
                elif game_type == "diplomacy":
                    report += f"- **Average Game Length**: {metrics['avg_game_length']:.1f}\n"
                    report += f"- **Average Rounds**: {metrics['avg_rounds']:.1f}\n"
                    report += f"- **Completion Rate**: {metrics['success_rate']:.1%}\n"
                    if metrics.get('wins_by_country'):
                        report += "- **Wins by Country**:\n"
                        for country, wins in metrics.get('wins_by_country', {}).items():
                            report += f"  - {country}: {wins}\n"
                    
                    if metrics.get('detailed_results'):
                        report += "\n**Detailed Game Results**:\n"
                        for i, detail in enumerate(metrics['detailed_results']):
                            winner_text = f"Winner: {detail['winner']}" if detail['winner'] else "No winner"
                            report += f"  {i+1}. {winner_text} - {detail['rounds']} rounds\n"
                
                report += "\n"
        
        ## Score Breakdown
        report += "## Score Breakdown\n\n"
        
        # Add scoring methodology explanation
        methodology = overall_results.get('scoring_methodology', {})
        report += f"**Scoring Formula**: `{methodology.get('formula', 'N/A')}`\n\n"
        
        for game, breakdown in overall_results['score_breakdown'].items():
            report += f"**{game.title()}** (Weight: {breakdown['weight']:.1f}x):\n"
            report += f"- Success Rate: {breakdown['success_rate']:.1%}\n"
            report += f"- Efficiency: {breakdown['efficiency']:.1%}\n"
            report += f"- Game Score: {breakdown['game_score']:.3f}\n"
            report += f"- Weighted Score: {breakdown['weighted_score']:.3f}\n"
            report += f"- Games Played: {breakdown['games_played']}\n"
            report += f"- Contribution: {breakdown['contribution_percentage']:.1f}%\n\n"
        
        report += f"\n## Model Performance Summary\n"
        report += f"Based on {overall_results['total_games']} games across multiple environments, "
        report += f"the model achieved an overall score of {overall_results['overall_score']:.3f}.\n\n"
        
        # Performance interpretation
        performance_info = overall_results['performance_classification']
        report += f"**Performance Rating**: {performance_info['rating']} ({performance_info['emoji']}) {performance_info['description']}\n"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Analyze PolitAgent benchmark results")
    parser.add_argument("--results_dir", required=True, help="Directory containing benchmark results")
    parser.add_argument("--output_file", help="Output file for the report")
    parser.add_argument("--json_output", help="JSON file for machine-readable results")
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer(args.results_dir)
    analyzer.load_results()
    
    # Generate report
    report = analyzer.generate_report(args.output_file)
    print(report)
    
    # Save JSON results if requested
    if args.json_output:
        overall_results = analyzer.calculate_overall_score()
        with open(args.json_output, 'w') as f:
            json.dump(overall_results, f, indent=2)
        logger.info(f"JSON results saved to {args.json_output}")


if __name__ == "__main__":
    main() 