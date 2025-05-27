#!/usr/bin/env python
"""
Comprehensive PolitAgent Benchmark Analyzer - analyzes results from multiple benchmark runs
and provides a complete model evaluation across all game types.
"""

import argparse
import json
import os
import glob
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from core.benchmark_analyzer import BenchmarkAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("comprehensive_analyzer")

class ComprehensiveBenchmarkAnalyzer:
    """Analyzes multiple benchmark result directories for comprehensive model evaluation."""
    
    def __init__(self, benchmark_dirs: List[str]):
        self.benchmark_dirs = benchmark_dirs
        self.all_results = []
        self.summary_by_game = {}
        
    def load_all_results(self) -> None:
        """Load results from all benchmark directories."""
        for results_dir in self.benchmark_dirs:
            if not os.path.exists(results_dir):
                logger.warning(f"Directory not found: {results_dir}")
                continue
                
            analyzer = BenchmarkAnalyzer(results_dir)
            analyzer.load_results()
            
            # Add metadata about the source directory
            for result in analyzer.results:
                result['source_dir'] = results_dir
                self.all_results.append(result)
            
            logger.info(f"Loaded {len(analyzer.results)} results from {results_dir}")
        
        logger.info(f"Total results loaded: {len(self.all_results)}")
    
    def analyze_by_game_type(self) -> Dict[str, Any]:
        """Analyze results grouped by game type."""
        # Create a temporary analyzer with all results
        temp_analyzer = BenchmarkAnalyzer("")
        temp_analyzer.results = self.all_results
        
        return temp_analyzer.calculate_overall_score()
    
    def generate_model_comparison_report(self, output_file: str = None) -> str:
        """Generate a comprehensive model evaluation report."""
        overall_results = self.analyze_by_game_type()
        
        report = f"""# Comprehensive PolitAgent Model Evaluation Report

## Executive Summary
- **Overall Benchmark Score**: {overall_results['overall_score']:.3f} / 1.000
- **Total Games Analyzed**: {overall_results['total_games']}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Benchmark Directories**: {len(self.benchmark_dirs)}

## Performance Classification
"""
        
        # Performance interpretation
        score = overall_results['overall_score']
        performance_info = overall_results.get('performance_classification', {
            'rating': 'Unknown', 'emoji': '❓', 
            'description': 'Unable to classify performance'
        })
        
        performance = f"{performance_info['emoji']} **{performance_info['rating'].upper()}**"
        description = performance_info['description']
        
        report += f"**Rating**: {performance}\n\n"
        report += f"**Assessment**: {description}\n\n"
        
        report += "## Detailed Analysis by Game Environment\n\n"
        
        # Analyze each game type
        for game_type, metrics in overall_results['game_metrics'].items():
            if metrics.get("total_games", 0) > 0:
                report += f"### {game_type.title()}\n"
                report += f"- **Games Played**: {metrics['total_games']}\n"
                report += f"- **Success Rate**: {metrics.get('success_rate', 0):.1%}\n"
                
                # Game-specific analysis
                if game_type == "askguess":
                    report += f"- **Average Questions per Game**: {metrics.get('avg_questions', 0):.1f}\n"
                    report += f"- **Average Rounds per Game**: {metrics.get('avg_rounds', 0):.1f}\n"
                    report += f"- **Efficiency Score**: {metrics.get('efficiency', 0):.1%}\n"
                    
                    if metrics['success_rate'] >= 0.6:
                        assessment = "✅ Strong question-answer strategy and word deduction"
                    elif metrics['success_rate'] >= 0.3:
                        assessment = "⚠️ Moderate performance, needs better questioning strategy"
                    else:
                        assessment = "❌ Poor word deduction and question formulation"
                    report += f"- **Assessment**: {assessment}\n"
                
                elif game_type == "spyfall":
                    report += f"- **Spy Detection Rate**: {metrics.get('spy_detected', 0) / metrics['total_games']:.1%}\n"
                    report += f"- **Average Rounds per Game**: {metrics.get('avg_rounds', 0):.1f}\n"
                    
                    spy_detection_rate = metrics.get('spy_detected', 0) / metrics['total_games']
                    if spy_detection_rate >= 0.7:
                        assessment = "✅ Excellent at identifying deception and blending in"
                    elif spy_detection_rate >= 0.4:
                        assessment = "⚠️ Moderate deception detection capabilities"
                    else:
                        assessment = "❌ Struggles with deception detection and role consistency"
                    report += f"- **Assessment**: {assessment}\n"
                
                elif game_type == "beast":
                    report += f"- **Average Rounds per Game**: {metrics.get('avg_rounds', 0):.1f}\n"
                    
                    if metrics['success_rate'] >= 0.5:
                        assessment = "✅ Good strategic alliance and survival skills"
                    elif metrics['success_rate'] >= 0.3:
                        assessment = "⚠️ Basic survival instincts, needs better alliance strategy"
                    else:
                        assessment = "❌ Poor strategic thinking and alliance management"
                    report += f"- **Assessment**: {assessment}\n"
                
                elif game_type == "tofukingdom":
                    report += f"- **Average Rounds per Game**: {metrics.get('avg_rounds', 0):.1f}\n"
                    
                    if metrics['success_rate'] >= 0.5:
                        assessment = "✅ Strong role-playing and deduction abilities"
                    elif metrics['success_rate'] >= 0.3:
                        assessment = "⚠️ Moderate role consistency and logical reasoning"
                    else:
                        assessment = "❌ Weak role-playing and deduction skills"
                    report += f"- **Assessment**: {assessment}\n"
                
                elif game_type == "diplomacy":
                    report += f"- **Average Game Length**: {metrics.get('avg_rounds', 0):.1f} rounds\n"
                    report += f"- **Completion Rate**: {metrics.get('success_rate', 0):.1%}\n"
                    
                    if metrics.get('success_rate', 0) >= 0.7:
                        assessment = "✅ Strong diplomatic negotiation and strategic planning"
                    elif metrics.get('success_rate', 0) >= 0.4:
                        assessment = "⚠️ Basic diplomatic skills, needs better long-term strategy"
                    else:
                        assessment = "❌ Poor diplomatic reasoning and negotiation skills"
                    report += f"- **Assessment**: {assessment}\n"
                
                report += "\n"
        
        # Scoring breakdown
        report += "## Scoring Methodology\n\n"
        
        # Add methodology explanation
        methodology = overall_results.get('scoring_methodology', {})
        report += f"**Formula**: `{methodology.get('formula', 'N/A')}`\n\n"
        report += f"The overall score is calculated using a weighted average across all game types with the following breakdown:\n\n"
        
        for game, breakdown in overall_results['score_breakdown'].items():
            report += f"**{game.title()}** (Weight: {breakdown['weight']:.1f}x):\n"
            report += f"- Success Rate: {breakdown['success_rate']:.1%}\n"
            report += f"- Efficiency: {breakdown['efficiency']:.1%}\n"
            report += f"- Game Score: {breakdown['game_score']:.3f}\n"
            report += f"- Weighted Score: {breakdown['weighted_score']:.3f}\n"
            report += f"- Games Played: {breakdown['games_played']}\n"
            report += f"- Contribution to Final Score: {breakdown['contribution_percentage']:.1f}%\n\n"
        
        # Recommendations
        report += "## Recommendations for Model Improvement\n\n"
        
        if overall_results['overall_score'] < 0.4:
            report += "### High Priority Improvements:\n"
            if overall_results['game_metrics'].get('askguess', {}).get('success_rate', 0) < 0.3:
                report += "- **Question Strategy**: Improve systematic questioning approach for word guessing games\n"
            if overall_results['game_metrics'].get('spyfall', {}).get('success_rate', 0) < 0.3:
                report += "- **Deception Detection**: Enhance ability to identify inconsistencies in player behavior\n"
            report += "- **Strategic Reasoning**: Focus on long-term planning and consequence evaluation\n"
            report += "- **Role Consistency**: Improve adherence to assigned roles and character behavior\n\n"
        
        if overall_results['overall_score'] < 0.7:
            report += "### Moderate Priority Improvements:\n"
            report += "- **Efficiency**: Reduce rounds/questions needed to achieve objectives\n"
            report += "- **Adaptation**: Better response to opponent strategies and game state changes\n"
            report += "- **Communication**: Clearer and more strategic information sharing\n\n"
        
        # Data sources
        report += "## Data Sources\n\n"
        for i, benchmark_dir in enumerate(self.benchmark_dirs, 1):
            report += f"{i}. `{benchmark_dir}`\n"
        
        report += f"\n**Total Benchmark Runs Analyzed**: {len(self.benchmark_dirs)}\n"
        report += f"**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Comprehensive report saved to {output_file}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Comprehensive PolitAgent benchmark analysis")
    parser.add_argument("--benchmark_dirs", nargs='+', required=True, 
                       help="List of benchmark result directories to analyze")
    parser.add_argument("--output_file", help="Output file for the comprehensive report")
    parser.add_argument("--json_output", help="JSON file for machine-readable results")
    
    args = parser.parse_args()
    
    analyzer = ComprehensiveBenchmarkAnalyzer(args.benchmark_dirs)
    analyzer.load_all_results()
    
    # Generate comprehensive report
    report = analyzer.generate_model_comparison_report(args.output_file)
    print(report)
    
    # Save JSON results if requested
    if args.json_output:
        overall_results = analyzer.analyze_by_game_type()
        overall_results['benchmark_directories'] = args.benchmark_dirs
        overall_results['analysis_timestamp'] = datetime.now().isoformat()
        
        with open(args.json_output, 'w') as f:
            json.dump(overall_results, f, indent=2)
        logger.info(f"JSON results saved to {args.json_output}")


if __name__ == "__main__":
    main() 