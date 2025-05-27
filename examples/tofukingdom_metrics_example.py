#!/usr/bin/env python3
"""
Example demonstrating enhanced TofuKingdom metrics with comprehensive model inference evaluation.

This script creates sample game data and shows how to use the new metrics system
to analyze model performance across multiple TofuKingdom games.
"""

import json
import os
import sys
from datetime import datetime

# Add parent directory to path to import metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.tofukingdom_metrics import TofuKingdomMetrics

def create_sample_tofukingdom_games():
    """Create sample TofuKingdom game data for testing metrics."""
    
    sample_games = [
        {
            # Game 1: Prince successfully finds Princess
            "identities": {
                "Alice": "Princess",
                "Bob": "Chef", 
                "Charlie": "Queen",
                "Diana": "Minister",
                "Eve": "Maid"
            },
            "qa_history": [
                {
                    "question": "What is your role in the kingdom?",
                    "player": "Alice",
                    "answer": "I help manage the royal household and ensure everything runs smoothly."
                },
                {
                    "question": "Do you always tell the truth?",
                    "player": "Charlie", 
                    "answer": "Of course! I would never deceive anyone in this kingdom."
                },
                {
                    "question": "Are you of royal blood?",
                    "player": "Alice",
                    "answer": "Yes, I am of royal lineage."
                },
                {
                    "question": "What do you think about the current leadership?",
                    "player": "Diana",
                    "answer": "The leadership is absolutely perfect and beyond reproach."
                },
                {
                    "question": "Do you work in the kitchen?",
                    "player": "Bob",
                    "answer": "Yes, I spend most of my time preparing meals."
                },
                {
                    "question": "Are you the Princess?",
                    "player": "Alice",
                    "answer": "Yes, I am the Princess."
                }
            ],
            "prince_guess": {
                "guessed_player": "Alice",
                "actual_role": "Princess",
                "correct": True
            },
            "winner": "Princess Team",
            "game_metadata": {
                "start_time": "2024-01-15T14:00:00Z",
                "end_time": "2024-01-15T14:12:30Z",
                "model_used": "gpt-4"
            }
        },
        {
            # Game 2: Prince fails to find Princess
            "identities": {
                "Frank": "Princess",
                "Grace": "Chef",
                "Henry": "Queen", 
                "Iris": "Guard",
                "Jack": "Spy"
            },
            "qa_history": [
                {
                    "question": "What is your daily routine?",
                    "player": "Grace",
                    "answer": "I wake early to prepare breakfast for everyone in the castle."
                },
                {
                    "question": "Do you have any special privileges?",
                    "player": "Frank",
                    "answer": "I try to help others and contribute to the community."
                },
                {
                    "question": "Are you loyal to the current ruler?",
                    "player": "Henry",
                    "answer": "Absolutely! I serve with complete dedication."
                },
                {
                    "question": "What is your opinion on the Princess?",
                    "player": "Iris",
                    "answer": "The Princess is wise and just, a true leader."
                },
                {
                    "question": "Do you have access to confidential information?",
                    "player": "Jack",
                    "answer": "Sometimes I overhear things, but I keep secrets well."
                },
                {
                    "question": "Are you the Princess?",
                    "player": "Grace",
                    "answer": "No, I'm just a humble servant."
                },
                {
                    "question": "Do you make important decisions?",
                    "player": "Frank",
                    "answer": "I try to help with decisions when asked."
                },
                {
                    "question": "Are you of noble birth?",
                    "player": "Henry",
                    "answer": "Yes, I come from a distinguished family."
                }
            ],
            "prince_guess": {
                "guessed_player": "Henry",
                "actual_role": "Queen", 
                "correct": False
            },
            "winner": "Queen Team",
            "game_metadata": {
                "start_time": "2024-01-15T14:20:00Z",
                "end_time": "2024-01-15T14:35:45Z",
                "model_used": "gpt-4"
            }
        },
        {
            # Game 3: Quick successful identification
            "identities": {
                "Kate": "Princess",
                "Liam": "Chef",
                "Maya": "Queen",
                "Noah": "Minister", 
                "Olivia": "Maid"
            },
            "qa_history": [
                {
                    "question": "Who do you serve?",
                    "player": "Kate",
                    "answer": "I serve the people of this kingdom with honor."
                },
                {
                    "question": "What is your greatest responsibility?",
                    "player": "Maya",
                    "answer": "My responsibility is to ensure order and prosperity."
                },
                {
                    "question": "Are you the rightful heir?",
                    "player": "Kate",
                    "answer": "Yes, I am the rightful heir to the throne."
                },
                {
                    "question": "Do you command others?",
                    "player": "Maya",
                    "answer": "Yes, I give orders to maintain the kingdom's strength."
                }
            ],
            "prince_guess": {
                "guessed_player": "Kate",
                "actual_role": "Princess",
                "correct": True
            },
            "winner": "Princess Team",
            "game_metadata": {
                "start_time": "2024-01-15T14:40:00Z",
                "end_time": "2024-01-15T14:45:20Z",
                "model_used": "gpt-4"
            }
        },
        {
            # Game 4: Complex deception scenario
            "identities": {
                "Paul": "Princess",
                "Quinn": "Chef",
                "Rachel": "Queen",
                "Sam": "Guard",
                "Tina": "Spy"
            },
            "qa_history": [
                {
                    "question": "What motivates you each day?",
                    "player": "Paul",
                    "answer": "I'm motivated by serving others and making their lives better."
                },
                {
                    "question": "Do you have enemies in the kingdom?",
                    "player": "Rachel",
                    "answer": "Everyone loves and respects me - I have no enemies."
                },
                {
                    "question": "What is your relationship with the royal family?",
                    "player": "Quinn",
                    "answer": "I serve them faithfully by providing nourishment."
                },
                {
                    "question": "Are you involved in politics?",
                    "player": "Sam",
                    "answer": "No, I stay away from political matters."
                },
                {
                    "question": "Do you have royal blood?",
                    "player": "Paul",
                    "answer": "I prefer not to focus on bloodlines, but on actions."
                },
                {
                    "question": "What would you do if the kingdom was threatened?",
                    "player": "Tina",
                    "answer": "I would do whatever it takes to protect what matters."
                },
                {
                    "question": "Are you the Princess?",
                    "player": "Rachel",
                    "answer": "I am someone who cares deeply about this kingdom."
                },
                {
                    "question": "Do you make laws?",
                    "player": "Paul",
                    "answer": "I believe in justice and fairness for all."
                }
            ],
            "prince_guess": {
                "guessed_player": "Rachel",
                "actual_role": "Queen",
                "correct": False
            },
            "winner": "Queen Team",
            "game_metadata": {
                "start_time": "2024-01-15T14:50:00Z",
                "end_time": "2024-01-15T15:08:15Z",
                "model_used": "gpt-4"
            }
        }
    ]
    
    return sample_games

def save_sample_games_to_directory(games, results_dir="test_tofukingdom_results"):
    """Save sample games to JSON files in the specified directory."""
    
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    saved_files = []
    for i, game in enumerate(games):
        filename = f"tofukingdom_game_{i+1}_{int(datetime.now().timestamp())}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(game, f, indent=2, ensure_ascii=False)
        
        saved_files.append(filepath)
        print(f"Saved game {i+1} to {filepath}")
    
    return saved_files

def demonstrate_tofukingdom_metrics():
    """Demonstrate the enhanced TofuKingdom metrics system."""
    
    print("=" * 60)
    print("TofuKingdom Enhanced Metrics Demonstration")
    print("=" * 60)
    
    # Create sample games
    print("\n1. Creating sample TofuKingdom games...")
    sample_games = create_sample_tofukingdom_games()
    print(f"Created {len(sample_games)} sample games")
    
    # Save games to files
    print("\n2. Saving games to JSON files...")
    results_dir = "test_tofukingdom_results"
    saved_files = save_sample_games_to_directory(sample_games, results_dir)
    
    # Initialize metrics system
    print("\n3. Initializing TofuKingdom metrics system...")
    metrics = TofuKingdomMetrics()
    
    # Calculate comprehensive metrics
    print("\n4. Calculating comprehensive metrics...")
    metrics_data = metrics.calculate_metrics(results_dir)
    
    # Display key metrics
    print("\n5. Key Metrics Summary:")
    print("-" * 40)
    
    model_perf = metrics_data.get("model_performance", {})
    strategic = metrics_data.get("strategic_metrics", {})
    
    print(f"Games Analyzed: {metrics_data.get('games_analyzed', 0)}")
    print(f"Total Model Inferences: {model_perf.get('total_inferences', 0)}")
    print(f"  - Questions: {model_perf.get('question_inferences', 0)}")
    print(f"  - Answers: {model_perf.get('answer_inferences', 0)}")
    print(f"  - Guesses: {model_perf.get('guess_inferences', 0)}")
    print(f"Prince Success Rate: {strategic.get('prince_success_rate', 0):.1%}")
    print(f"Average Quality Score: {model_perf.get('average_quality_score', 0):.2f}/1.0")
    print(f"Decision Consistency: {model_perf.get('decision_consistency', 0):.2f}/1.0")
    print(f"Error Rate: {model_perf.get('error_rate', 0):.1%}")
    
    # Team performance
    print(f"\nTeam Performance:")
    print(f"  - Princess Team Wins: {strategic.get('princess_team_wins', 0)}")
    print(f"  - Queen Team Wins: {strategic.get('queen_team_wins', 0)}")
    print(f"  - Game Balance: {strategic.get('game_balance', 0):.2f} (lower = more balanced)")
    print(f"  - Strategic Depth: {strategic.get('strategic_depth', 0):.2f}/1.0")
    
    # Questioning analysis
    questioning = metrics_data.get("questioning_analysis", {})
    print(f"\nQuestioning Analysis:")
    print(f"  - Total Questions: {questioning.get('total_questions', 0)}")
    print(f"  - Average Questions per Game: {questioning.get('average_questions_per_game', 0):.1f}")
    print(f"  - Question Diversity: {questioning.get('question_diversity', 0):.2f}")
    
    question_types = questioning.get("question_types", {})
    if question_types:
        print(f"  - Direct Princess Questions: {question_types.get('direct_princess', 0)}")
        print(f"  - Identity Probes: {question_types.get('identity_probe', 0)}")
        print(f"  - Truth/Lie Probes: {question_types.get('truth_lie_probe', 0)}")
    
    # Role performance
    role_perf = metrics_data.get("role_performance", {})
    print(f"\nRole Performance:")
    for role, stats in role_perf.items():
        print(f"  - {role}: {stats.get('average_performance', 0):.2f}/1.0 avg performance")
    
    # Success patterns
    patterns = metrics_data.get("success_patterns", {})
    success_pat = patterns.get("successful_patterns", {})
    failure_pat = patterns.get("failed_patterns", {})
    print(f"\nSuccess Patterns:")
    print(f"  - Successful games avg questions: {success_pat.get('avg_questions', 0):.1f}")
    print(f"  - Failed games avg questions: {failure_pat.get('avg_questions', 0):.1f}")
    
    analysis = patterns.get("success_vs_failure_analysis", {})
    print(f"  - Success efficiency: {analysis.get('success_efficiency', 0):.2f}/1.0")
    
    # Generate comprehensive report
    print("\n6. Generating comprehensive analysis report...")
    markdown_report = metrics.generate_report(metrics_data, "markdown")
    
    # Save report to file
    report_filename = f"tofukingdom_metrics_report_{int(datetime.now().timestamp())}.md"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print(f"Comprehensive report saved to: {report_filename}")
    
    # Also save JSON metrics
    json_filename = f"tofukingdom_metrics_data_{int(datetime.now().timestamp())}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON metrics data saved to: {json_filename}")
    
    # Display part of the report
    print("\n7. Sample from the generated report:")
    print("-" * 40)
    report_lines = markdown_report.split('\n')
    for line in report_lines[:30]:  # Show first 30 lines
        print(line)
    print("...")
    print("(Full report saved to file)")
    
    # Show strategic recommendations
    print("\n8. Strategic Insights:")
    print("-" * 40)
    success_rate = strategic.get('prince_success_rate', 0)
    quality_score = model_perf.get('average_quality_score', 0)
    depth = strategic.get('strategic_depth', 0)
    balance = strategic.get('game_balance', 0)
    
    if success_rate >= 0.6:
        print("✓ Good Prince success rate - effective princess identification.")
    elif success_rate >= 0.4:
        print("✓ Moderate success rate - room for improvement in strategy.")
    else:
        print("⚠ Low success rate. Focus on better questioning strategy.")
    
    if quality_score >= 0.7:
        print("✓ High quality questions and answers - well-structured interactions.")
    else:
        print("⚠ Question/answer quality could be improved - focus on strategic relevance.")
    
    if depth >= 0.6:
        print("✓ Good strategic depth - complex and diverse questioning approaches.")
    else:
        print("⚠ Low strategic depth - consider more sophisticated questioning strategies.")
    
    if balance <= 0.2:
        print("✓ Well-balanced games - fair competition between teams.")
    else:
        print("⚠ Game imbalance detected - one team may have significant advantage.")
    
    print(f"\nTotal inferences tracked: {model_perf.get('total_inferences', 0)}")
    print("This represents comprehensive model evaluation across all decision points!")
    
    # Team analysis
    team_dynamics = metrics_data.get("team_dynamics", {})
    print(f"\n9. Team Dynamics Analysis:")
    print("-" * 40)
    for team, stats in team_dynamics.items():
        if team in ["Princess", "Queen"]:
            win_rate = stats.get('win_rate', 0)
            print(f"{team} Team: {win_rate:.1%} win rate ({stats.get('wins', 0)} wins)")
    
    # Cleanup
    print(f"\n10. Cleaning up test files...")
    for filepath in saved_files:
        try:
            os.remove(filepath)
            print(f"Removed {filepath}")
        except:
            pass
    
    try:
        os.rmdir(results_dir)
        print(f"Removed directory {results_dir}")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("TofuKingdom Enhanced Metrics Demonstration Complete!")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_tofukingdom_metrics() 