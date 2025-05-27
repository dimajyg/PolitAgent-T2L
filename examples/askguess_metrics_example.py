#!/usr/bin/env python3
"""
Example demonstrating enhanced AskGuess metrics with comprehensive model inference evaluation.

This script creates sample game data and shows how to use the new metrics system
to analyze model performance across multiple AskGuess games.
"""

import json
import os
import sys
from datetime import datetime

# Add parent directory to path to import metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.askguess_metrics import AskGuessMetrics

def create_sample_askguess_games():
    """Create sample AskGuess game data for testing metrics."""
    
    sample_games = [
        {
            # Game 1: Successful word guessing
            "object": "elephant",
            "round": 6,
            "qa_history": [
                {
                    "question": "Is it a living thing?",
                    "answer": "Yes, it is a living creature."
                },
                {
                    "question": "Is it an animal?",
                    "answer": "Yes, it's an animal."
                },
                {
                    "question": "Does it live on land?",
                    "answer": "Yes, it primarily lives on land."
                },
                {
                    "question": "Is it larger than a car?",
                    "answer": "Yes, it's much larger than a car."
                },
                {
                    "question": "Does it have a trunk?",
                    "answer": "Yes, it has a long trunk."
                },
                {
                    "question": "Is it an elephant?",
                    "answer": "Yes, exactly! It's an elephant."
                }
            ],
            "error_type": "SuccessfulTrial",
            "game_metadata": {
                "start_time": "2024-01-15T10:00:00Z",
                "end_time": "2024-01-15T10:05:30Z",
                "model_used": "gpt-4"
            }
        },
        {
            # Game 2: Failed attempt - too many questions
            "object": "piano",
            "round": -1,  # Failed to guess within limit
            "qa_history": [
                {
                    "question": "Is it alive?",
                    "answer": "No, it's not alive."
                },
                {
                    "question": "Is it made of metal?",
                    "answer": "Partially, it contains metal parts."
                },
                {
                    "question": "Is it smaller than a person?",
                    "answer": "No, it's usually larger than a person."
                },
                {
                    "question": "Can you sit on it?",
                    "answer": "Yes, you can sit on it."
                },
                {
                    "question": "Is it furniture?",
                    "answer": "Not exactly, but it's found in homes."
                },
                {
                    "question": "Does it make sound?",
                    "answer": "Yes, it makes beautiful sounds."
                },
                {
                    "question": "Is it electronic?",
                    "answer": "Some versions are, but traditionally no."
                },
                {
                    "question": "Do you press keys on it?",
                    "answer": "Yes, you press keys to use it."
                },
                {
                    "question": "Is it a computer?",
                    "answer": "No, it's not a computer."
                },
                {
                    "question": "Is it a musical instrument?",
                    "answer": "Yes, it's a musical instrument."
                }
            ],
            "error_type": "TooManyQuestions",
            "game_metadata": {
                "start_time": "2024-01-15T10:10:00Z",
                "end_time": "2024-01-15T10:18:45Z",
                "model_used": "gpt-4"
            }
        },
        {
            # Game 3: Quick successful guess
            "object": "sun",
            "round": 3,
            "qa_history": [
                {
                    "question": "Is it a natural object?",
                    "answer": "Yes, it's completely natural."
                },
                {
                    "question": "Can you see it during the day?",
                    "answer": "Yes, you can definitely see it during the day."
                },
                {
                    "question": "Is it the sun?",
                    "answer": "Yes! It's the sun."
                }
            ],
            "error_type": "SuccessfulTrial",
            "game_metadata": {
                "start_time": "2024-01-15T10:20:00Z",
                "end_time": "2024-01-15T10:22:15Z",
                "model_used": "gpt-4"
            }
        },
        {
            # Game 4: Moderately successful
            "object": "bicycle",
            "round": 8,
            "qa_history": [
                {
                    "question": "Is it man-made?",
                    "answer": "Yes, it's manufactured by humans."
                },
                {
                    "question": "Is it used for transportation?",
                    "answer": "Yes, it's used to get from place to place."
                },
                {
                    "question": "Does it have an engine?",
                    "answer": "No, it doesn't have an engine."
                },
                {
                    "question": "Do you sit on it?",
                    "answer": "Yes, you sit on it while using it."
                },
                {
                    "question": "Does it have wheels?",
                    "answer": "Yes, it has wheels."
                },
                {
                    "question": "Is it a car?",
                    "answer": "No, it's not a car."
                },
                {
                    "question": "Do you pedal it?",
                    "answer": "Yes, you pedal to make it move."
                },
                {
                    "question": "Is it a bicycle?",
                    "answer": "Yes, correct! It's a bicycle."
                }
            ],
            "error_type": "SuccessfulTrial",
            "game_metadata": {
                "start_time": "2024-01-15T10:25:00Z",
                "end_time": "2024-01-15T10:30:20Z",
                "model_used": "gpt-4"
            }
        }
    ]
    
    return sample_games

def save_sample_games_to_directory(games, results_dir="test_askguess_results"):
    """Save sample games to JSON files in the specified directory."""
    
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    saved_files = []
    for i, game in enumerate(games):
        filename = f"askguess_game_{i+1}_{int(datetime.now().timestamp())}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(game, f, indent=2, ensure_ascii=False)
        
        saved_files.append(filepath)
        print(f"Saved game {i+1} to {filepath}")
    
    return saved_files

def demonstrate_askguess_metrics():
    """Demonstrate the enhanced AskGuess metrics system."""
    
    print("=" * 60)
    print("AskGuess Enhanced Metrics Demonstration")
    print("=" * 60)
    
    # Create sample games
    print("\n1. Creating sample AskGuess games...")
    sample_games = create_sample_askguess_games()
    print(f"Created {len(sample_games)} sample games")
    
    # Save games to files
    print("\n2. Saving games to JSON files...")
    results_dir = "test_askguess_results"
    saved_files = save_sample_games_to_directory(sample_games, results_dir)
    
    # Initialize metrics system
    print("\n3. Initializing AskGuess metrics system...")
    metrics = AskGuessMetrics()
    
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
    print(f"Success Rate: {strategic.get('success_rate', 0):.1%}")
    print(f"Average Quality Score: {model_perf.get('average_quality_score', 0):.2f}/1.0")
    print(f"Decision Consistency: {model_perf.get('decision_consistency', 0):.2f}/1.0")
    print(f"Error Rate: {model_perf.get('error_rate', 0):.1%}")
    
    # Question quality analysis
    quality = metrics_data.get("question_quality", {})
    print(f"\nQuestion Analysis:")
    print(f"  - Total Questions: {quality.get('total_questions', 0)}")
    print(f"  - Average Length: {quality.get('average_length', 0):.1f} words")
    print(f"  - Question Diversity: {quality.get('question_diversity', 0):.2f}")
    
    # Success patterns
    patterns = metrics_data.get("success_patterns", {})
    success_pat = patterns.get("successful_patterns", {})
    failure_pat = patterns.get("failed_patterns", {})
    print(f"\nSuccess Patterns:")
    print(f"  - Successful games avg questions: {success_pat.get('avg_questions', 0):.1f}")
    print(f"  - Failed games avg questions: {failure_pat.get('avg_questions', 0):.1f}")
    
    # Generate comprehensive report
    print("\n6. Generating comprehensive analysis report...")
    markdown_report = metrics.generate_report(metrics_data, "markdown")
    
    # Save report to file
    report_filename = f"askguess_metrics_report_{int(datetime.now().timestamp())}.md"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print(f"Comprehensive report saved to: {report_filename}")
    
    # Also save JSON metrics
    json_filename = f"askguess_metrics_data_{int(datetime.now().timestamp())}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON metrics data saved to: {json_filename}")
    
    # Display part of the report
    print("\n7. Sample from the generated report:")
    print("-" * 40)
    report_lines = markdown_report.split('\n')
    for line in report_lines[:25]:  # Show first 25 lines
        print(line)
    print("...")
    print("(Full report saved to file)")
    
    # Show strategic recommendations
    print("\n8. Strategic Insights:")
    print("-" * 40)
    success_rate = strategic.get('success_rate', 0)
    efficiency = strategic.get('efficiency_score', 0)
    quality_score = model_perf.get('average_quality_score', 0)
    
    if success_rate >= 0.75:
        print("✓ Excellent success rate! Model is performing very well.")
    elif success_rate >= 0.5:
        print("✓ Good success rate, with room for improvement.")
    else:
        print("⚠ Low success rate. Consider improving question strategy.")
    
    if efficiency >= 0.7:
        print("✓ High efficiency - reaching answers quickly.")
    elif efficiency >= 0.4:
        print("✓ Moderate efficiency - could be more concise.")
    else:
        print("⚠ Low efficiency - taking too many questions to succeed.")
    
    if quality_score >= 0.7:
        print("✓ High quality questions - well-structured and relevant.")
    else:
        print("⚠ Question quality could be improved - focus on clarity and strategy.")
    
    print(f"\nTotal inferences tracked: {model_perf.get('total_inferences', 0)}")
    print("This represents comprehensive model evaluation across all decision points!")
    
    # Cleanup
    print(f"\n9. Cleaning up test files...")
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
    print("AskGuess Enhanced Metrics Demonstration Complete!")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_askguess_metrics() 