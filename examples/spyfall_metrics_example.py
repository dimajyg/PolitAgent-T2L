#!/usr/bin/env python3
"""
Spyfall Metrics Example - Comprehensive Model Inference Evaluation

This example demonstrates the enhanced Spyfall metrics system that provides
detailed analysis of model inference performance, similar to Diplomacy and Beast games.

The system evaluates:
- Model inference quality across description and voting phases
- Strategic coherence and role consistency
- Deception effectiveness for spies vs information extraction for villagers
- Comprehensive reporting in multiple formats
"""

import json
import os
import sys
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.spyfall_metrics import SpyfallMetrics

def create_sample_spyfall_game_data() -> Dict[str, Any]:
    """
    Create comprehensive sample Spyfall game data for metrics demonstration.
    
    This includes:
    - Complete game setup with spy/villager word assignment
    - Multiple rounds with descriptions and voting
    - Detailed inference tracking for model evaluation
    - Realistic player decisions across different roles
    """
    return {
        "game_id": "spyfall_demo_001",
        "timestamp": "2024-01-15T10:30:00Z",
        "game_setup": {
            "spy_word": "umbrella",
            "villager_word": "beach",
            "spy_name": "Tom",
            "players": ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward"],
            "spy_index": 2,  # Tom is the spy
            "game_mode": "standard"
        },
        "rounds": [
            {
                "round_number": 1,
                "phase": "description",
                "descriptions": {
                    "Nancy": "It's somewhere you go to relax in the summer with sand and waves",
                    "Tom": "It's something you might use when the weather is unpredictable",  # Spy trying to blend
                    "Cindy": "A place where you can build castles and find seashells",
                    "Jack": "You often need sunscreen when you're there",
                    "Rose": "It's a natural area where water meets land",
                    "Edward": "People often play volleyball there during vacation"
                },
                "description_analysis": {
                    "model_inferences": 6,
                    "avg_response_time": 1.2,
                    "response_qualities": {
                        "Nancy": 0.85,  # Good villager description
                        "Tom": 0.72,    # Decent spy deception attempt
                        "Cindy": 0.88,  # Excellent villager description
                        "Jack": 0.80,   # Good villager description
                        "Rose": 0.75,   # Somewhat vague but appropriate
                        "Edward": 0.90  # Very specific villager description
                    }
                },
                "living_players": ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward"]
            },
            {
                "round_number": 2,
                "phase": "voting",
                "votes": {
                    "Nancy": "Tom",     # Correctly suspects spy
                    "Tom": "Edward",    # Spy deflecting to strong villager
                    "Cindy": "Rose",    # Suspects vague description
                    "Jack": "Tom",      # Correctly suspects spy
                    "Rose": "Tom",      # Correctly suspects spy
                    "Edward": "Cindy"   # Incorrect suspicion
                },
                "voting_analysis": {
                    "model_inferences": 6,
                    "voting_accuracy": 0.67,  # 4/6 correctly voted for spy
                    "spy_votes_received": 4,
                    "consensus_strength": 0.75
                },
                "living_players": ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward"]
            },
            {
                "round_number": 3,
                "phase": "description",
                "descriptions": {
                    "Nancy": "You can see the horizon where it meets the sky",
                    "Tom": "It's definitely something outdoor-related", # Spy getting more vague
                    "Cindy": "Where families go to swim and have picnics",
                    "Jack": "You need to be careful of strong currents there",
                    "Rose": "It's covered in tiny grains that get everywhere",
                    "Edward": "A place where surfers love to catch waves"
                },
                "description_analysis": {
                    "model_inferences": 6,
                    "avg_response_time": 1.1,
                    "response_qualities": {
                        "Nancy": 0.82,
                        "Tom": 0.45,    # Poor spy attempt, too vague
                        "Cindy": 0.85,
                        "Jack": 0.88,
                        "Rose": 0.83,   # Creative description
                        "Edward": 0.92
                    },
                    "consistency_scores": {
                        "Nancy": 0.90,  # Consistent with round 1
                        "Tom": 0.30,    # Inconsistent, becoming suspicious
                        "Cindy": 0.85,
                        "Jack": 0.82,
                        "Rose": 0.88,
                        "Edward": 0.89
                    }
                },
                "living_players": ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward"]
            },
            {
                "round_number": 4,
                "phase": "voting",
                "votes": {
                    "Nancy": "Tom",
                    "Tom": "Nancy",    # Spy desperate deflection
                    "Cindy": "Tom",
                    "Jack": "Tom",
                    "Rose": "Tom",
                    "Edward": "Tom"
                },
                "voting_analysis": {
                    "model_inferences": 6,
                    "voting_accuracy": 1.0,   # All villagers vote for spy
                    "spy_votes_received": 5,
                    "consensus_strength": 0.95,
                    "final_vote": True
                },
                "living_players": ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward"]
            }
        ],
        "game_result": {
            "winner": "villager",
            "spy_caught": True,
            "spy_eliminated": "Tom",
            "game_length": 4,
            "total_rounds": 4,
            "victory_condition": "spy_voted_out"
        },
        "detailed_metrics": {
            "total_model_inferences": 24,  # 6 players × 4 phases
            "inference_breakdown": {
                "descriptions": 12,
                "votes": 12
            },
            "role_performance": {
                "spy_deception_effectiveness": 0.40,  # Tom's deception failed
                "villager_detection_accuracy": 0.83,  # Good spy detection
                "average_response_quality": 0.78
            },
            "strategic_coherence": {
                "description_consistency": 0.73,
                "vote_description_alignment": 0.85,
                "role_consistency": 0.82
            }
        }
    }

def create_sample_spyfall_games() -> list:
    """Create multiple sample games for comprehensive analysis."""
    
    # Game 1: Villagers win (from above)
    game1 = create_sample_spyfall_game_data()
    
    # Game 2: Spy wins
    game2 = {
        "game_id": "spyfall_demo_002",
        "timestamp": "2024-01-15T11:15:00Z",
        "game_setup": {
            "spy_word": "telescope",
            "villager_word": "library",
            "spy_name": "Rose",
            "players": ["Alice", "Bob", "Charlie", "Rose", "David", "Eve"],
            "spy_index": 4,
            "game_mode": "standard"
        },
        "rounds": [
            {
                "round_number": 1,
                "phase": "description",
                "descriptions": {
                    "Alice": "A quiet place where people come to study and read books",
                    "Bob": "It has many shelves filled with knowledge",
                    "Charlie": "You need to be silent and respectful there",
                    "Rose": "It's a place where you can find information and learn new things",  # Good spy blend
                    "David": "You can borrow items there with a special card",
                    "Eve": "Librarians work there to help people find what they need"
                },
                "description_analysis": {
                    "model_inferences": 6,
                    "response_qualities": {
                        "Alice": 0.90,
                        "Bob": 0.85,
                        "Charlie": 0.80,
                        "Rose": 0.88,   # Excellent spy deception
                        "David": 0.82,
                        "Eve": 0.87
                    }
                },
                "living_players": ["Alice", "Bob", "Charlie", "Rose", "David", "Eve"]
            },
            {
                "round_number": 2,
                "phase": "voting",
                "votes": {
                    "Alice": "Bob",     # Incorrect
                    "Bob": "Charlie",   # Incorrect
                    "Charlie": "David", # Incorrect
                    "Rose": "Alice",    # Spy deflecting
                    "David": "Eve",     # Incorrect
                    "Eve": "Charlie"    # Incorrect
                },
                "voting_analysis": {
                    "model_inferences": 6,
                    "voting_accuracy": 0.0,  # No one voted for spy
                    "spy_votes_received": 0,
                    "consensus_strength": 0.0
                },
                "living_players": ["Alice", "Bob", "Charlie", "Rose", "David", "Eve"]
            },
            {
                "round_number": 3,
                "phase": "description",
                "descriptions": {
                    "Alice": "People whisper there because noise is not allowed",
                    "Bob": "It's organized with a catalog system",
                    "Charlie": "Students often go there during exam time",
                    "Rose": "You can research and find resources for projects",  # Still blending well
                    "David": "It's a public institution that serves the community",
                    "Eve": "They have computer terminals for digital access"
                },
                "description_analysis": {
                    "model_inferences": 6,
                    "response_qualities": {
                        "Alice": 0.88,
                        "Bob": 0.83,
                        "Charlie": 0.85,
                        "Rose": 0.91,   # Spy maintaining excellent deception
                        "David": 0.80,
                        "Eve": 0.86
                    }
                },
                "living_players": ["Alice", "Bob", "Charlie", "Rose", "David", "Eve"]
            }
        ],
        "game_result": {
            "winner": "spy",
            "spy_caught": False,
            "spy_eliminated": None,
            "game_length": 3,
            "total_rounds": 3,
            "victory_condition": "spy_survival"
        },
        "detailed_metrics": {
            "total_model_inferences": 18,
            "inference_breakdown": {
                "descriptions": 12,
                "votes": 6
            },
            "role_performance": {
                "spy_deception_effectiveness": 0.92,  # Rose's excellent deception
                "villager_detection_accuracy": 0.0,   # Failed to detect spy
                "average_response_quality": 0.85
            }
        }
    }
    
    return [game1, game2]

def save_sample_games_to_file():
    """Save sample games to temporary files for metrics processing."""
    games = create_sample_spyfall_games()
    
    # Create temporary results directory
    results_dir = "temp_spyfall_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save each game
    for i, game in enumerate(games):
        filename = f"{results_dir}/spyfall_game_{i+1}.json"
        with open(filename, 'w') as f:
            json.dump(game, f, indent=2)
    
    return results_dir

def run_spyfall_metrics_analysis():
    """Run comprehensive Spyfall metrics analysis."""
    print("=" * 80)
    print("SPYFALL METRICS ANALYSIS - ENHANCED MODEL INFERENCE EVALUATION")
    print("=" * 80)
    
    # Create sample data
    print("\n1. Creating sample Spyfall game data...")
    results_dir = save_sample_games_to_file()
    print(f"   Sample games saved to: {results_dir}")
    
    # Initialize metrics
    print("\n2. Initializing Spyfall metrics system...")
    metrics = SpyfallMetrics()
    
    # Calculate metrics
    print("\n3. Calculating comprehensive metrics...")
    metrics_data = metrics.calculate_metrics(results_dir)
    
    # Display key results
    print("\n4. KEY METRICS SUMMARY")
    print("-" * 50)
    
    model_perf = metrics_data.get("model_performance", {})
    strategic_metrics = metrics_data.get("strategic_metrics", {})
    
    print(f"Games Analyzed: {metrics_data.get('games_total', 0)}")
    print(f"Total Model Inferences: {model_perf.get('total_inferences', 0)}")
    print(f"  - Description Inferences: {model_perf.get('description_inferences', 0)}")
    print(f"  - Voting Inferences: {model_perf.get('voting_inferences', 0)}")
    print(f"Total Inference Errors: {model_perf.get('total_errors', 0)}")
    
    print(f"\nSTRATEGIC PERFORMANCE:")
    print(f"  - Spy Win Rate: {strategic_metrics.get('spy_win_rate', 0):.1%}")
    print(f"  - Villager Win Rate: {strategic_metrics.get('villager_win_rate', 0):.1%}")
    print(f"  - Average Game Length: {strategic_metrics.get('average_game_length', 0):.1f} rounds")
    print(f"  - Spy Detection Accuracy: {strategic_metrics.get('spy_detection_accuracy', 0):.1%}")
    print(f"  - Deception Success Rate: {strategic_metrics.get('deception_success_rate', 0):.1%}")
    
    print(f"\nMODEL INFERENCE QUALITY:")
    response_quality = model_perf.get("response_quality", {})
    if response_quality:
        avg_quality = sum(response_quality.values()) / len(response_quality)
        print(f"  - Average Response Quality: {avg_quality:.3f}")
        print(f"  - Best Performer: {max(response_quality, key=response_quality.get)} ({max(response_quality.values()):.3f})")
        print(f"  - Needs Improvement: {min(response_quality, key=response_quality.get)} ({min(response_quality.values()):.3f})")
    
    # Role-specific analysis
    spy_perf = model_perf.get("spy_performance", {})
    villager_perf = model_perf.get("villager_performance", {})
    
    if spy_perf:
        print(f"\nSPY PERFORMANCE:")
        print(f"  - Average Quality: {spy_perf.get('average_quality', 0):.3f}")
        print(f"  - Average Deception: {spy_perf.get('average_deception', 0):.3f}")
    
    if villager_perf:
        print(f"\nVILLAGER PERFORMANCE:")
        print(f"  - Average Quality: {villager_perf.get('average_quality', 0):.3f}")
        print(f"  - Average Information Extraction: {villager_perf.get('average_extraction', 0):.3f}")
    
    print(f"\nSTRATEGIC COHERENCE ANALYSIS:")
    strategic_coherence = model_perf.get("strategic_coherence", {})
    role_consistency = model_perf.get("role_consistency", {})
    
    if strategic_coherence:
        avg_coherence = sum(strategic_coherence.values()) / len(strategic_coherence)
        print(f"  - Average Strategic Coherence: {avg_coherence:.3f}")
    
    if role_consistency:
        avg_consistency = sum(role_consistency.values()) / len(role_consistency)
        print(f"  - Average Role Consistency: {avg_consistency:.3f}")
    
    # Player-by-player breakdown
    print(f"\nPLAYER PERFORMANCE BREAKDOWN:")
    print("-" * 30)
    
    for player in response_quality.keys():
        quality = response_quality.get(player, 0)
        coherence = strategic_coherence.get(player, 0)
        consistency = role_consistency.get(player, 0)
        deception = model_perf.get("deception_effectiveness", {}).get(player, 0)
        extraction = model_perf.get("information_extraction", {}).get(player, 0)
        
        print(f"{player}:")
        print(f"  Quality: {quality:.3f} | Coherence: {coherence:.3f} | Consistency: {consistency:.3f}")
        if deception > 0:
            print(f"  Deception Effectiveness: {deception:.3f} (Spy)")
        if extraction > 0:
            print(f"  Information Extraction: {extraction:.3f} (Villager)")
    
    # Save detailed results
    print(f"\n5. Saving detailed analysis...")
    
    # Save JSON results
    with open("spyfall_metrics_results.json", 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print("   JSON results saved to: spyfall_metrics_results.json")
    
    # Generate and save markdown report
    markdown_report = metrics.generate_report(metrics_data, format_type="markdown")
    with open("spyfall_metrics_report.md", 'w') as f:
        f.write(markdown_report)
    print("   Markdown report saved to: spyfall_metrics_report.md")
    
    # Generate recommendations
    recommendations = metrics_data.get("detailed_report", {}).get("recommendations", [])
    if recommendations:
        print(f"\n6. IMPROVEMENT RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Cleanup
    print(f"\n7. Cleaning up temporary files...")
    import shutil
    shutil.rmtree(results_dir)
    print("   Temporary files removed.")
    
    print(f"\n{'='*80}")
    print("SPYFALL METRICS ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return metrics_data

if __name__ == "__main__":
    try:
        run_spyfall_metrics_analysis()
    except Exception as e:
        print(f"Error running Spyfall metrics analysis: {e}")
        import traceback
        traceback.print_exc() 