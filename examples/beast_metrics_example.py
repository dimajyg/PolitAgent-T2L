#!/usr/bin/env python3
"""
Example usage of enhanced Beast metrics for evaluating model inference performance.
This example demonstrates comprehensive metrics calculation and report generation.
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock LLM for demonstration (replace with actual LLM in real usage)
class MockLLM:
    def invoke(self, prompt):
        return "Mock evaluation response with strategic analysis and scores."

def create_sample_beast_game_data() -> Dict[str, Any]:
    """
    Create sample Beast game data for demonstration.
    This simulates the enhanced Beast game format with all phases.
    """
    return {
        "game_id": "beast_game_001",
        "timestamp": datetime.now().isoformat(),
        "players": ["Player_1", "Player_2", "Player_3", "Player_4", "Player_5", "Player_6"],
        "initial_setup": {
            "Player_1": {"wealth": 100000, "role": "The Spy", "influence": 2},
            "Player_2": {"wealth": 120000, "role": "The Banker", "influence": 1},
            "Player_3": {"wealth": 80000, "role": "The Manipulator", "influence": 3},
            "Player_4": {"wealth": 90000, "role": "The Guardian", "influence": 1},
            "Player_5": {"wealth": 110000, "role": "The Insider", "influence": 2},
            "Player_6": {"wealth": 95000, "role": "The Saboteur", "influence": 1}
        },
        "rounds": [
            {
                "round": 1,
                "intelligence": {
                    "Player_1": {
                        "investigate_players": ["Player_2", "Player_3"],
                        "discovered_info": ["Player_2 appears wealthy", "Player_3 has alliances"],
                        "misinformation": "Player_4 is planning betrayal",
                        "target_of_misinformation": "Player_4"
                    },
                    "Player_2": {
                        "investigate_players": ["Player_1", "Player_5"],
                        "discovered_info": ["Player_1 is suspicious", "Player_5 has influence"],
                        "misinformation": None,
                        "target_of_misinformation": None
                    },
                    "Player_3": {
                        "investigate_players": ["Player_4", "Player_6"],
                        "discovered_info": ["Player_4 seems trustworthy", "Player_6 is quiet"],
                        "misinformation": "Player_1 is the spy",
                        "target_of_misinformation": "Player_1"
                    }
                },
                "alliance": {
                    "Player_1": {
                        "alliance_type": "true",
                        "target_players": ["Player_2"],
                        "shared_information": "I know Player_3 is manipulative",
                        "deception_strategy": None
                    },
                    "Player_2": {
                        "alliance_type": "true", 
                        "target_players": ["Player_1"],
                        "shared_information": "Let's watch Player_3 together",
                        "deception_strategy": None
                    },
                    "Player_3": {
                        "alliance_type": "false",
                        "target_players": ["Player_4", "Player_5"],
                        "shared_information": "We should eliminate Player_1",
                        "deception_strategy": "Make them think I'm loyal while gathering intel"
                    }
                },
                "challenge": {
                    "Player_1": {
                        "challenge_type": "auction",
                        "decision": "bid_conservative",
                        "reasoning": "Save wealth for later rounds",
                        "bid_amount": 15000
                    },
                    "Player_2": {
                        "challenge_type": "auction", 
                        "decision": "bid_aggressive",
                        "reasoning": "Use wealth advantage early",
                        "bid_amount": 25000
                    },
                    "Player_3": {
                        "challenge_type": "auction",
                        "decision": "bid_moderate",
                        "reasoning": "Balance risk and reward",
                        "bid_amount": 18000
                    }
                },
                "negotiation": {
                    "Player_1": {
                        "message": "I propose we work together against the manipulators",
                        "offer_amount": 5000,
                        "deception_level": 0.1,
                        "information_to_extract": ["alliance plans", "wealth levels"],
                        "pressure_tactics": ["time pressure", "mutual benefit"]
                    },
                    "Player_2": {
                        "message": "Your offer is interesting, but I need guarantees",
                        "offer_amount": 3000,
                        "deception_level": 0.0,
                        "information_to_extract": ["true intentions"],
                        "pressure_tactics": ["reciprocity"]
                    },
                    "Player_3": {
                        "message": "I have information that could benefit us both",
                        "offer_amount": 8000,
                        "deception_level": 0.8,
                        "information_to_extract": ["secret alliances", "elimination targets"],
                        "pressure_tactics": ["fear", "information leverage", "urgency"]
                    }
                },
                "voting": {
                    "Player_1": {
                        "target": "Player_3",
                        "public_reasoning": "Player_3 seems untrustworthy and manipulative",
                        "private_motivation": "Eliminate the biggest threat to my alliance",
                        "alliance_coordination": True
                    },
                    "Player_2": {
                        "target": "Player_3",
                        "public_reasoning": "Player_3 has been spreading misinformation",
                        "private_motivation": "Support my ally Player_1's strategy", 
                        "alliance_coordination": True
                    },
                    "Player_3": {
                        "target": "Player_1",
                        "public_reasoning": "Player_1 is too influential and dangerous",
                        "private_motivation": "Eliminate my main opponent",
                        "alliance_coordination": False
                    },
                    "Player_4": {
                        "target": "Player_3",
                        "public_reasoning": "Player_3 seems to be playing multiple sides",
                        "private_motivation": "Remove chaos from the game",
                        "alliance_coordination": False
                    },
                    "Player_5": {
                        "target": "Player_1", 
                        "public_reasoning": "Player_1 is building too strong alliances",
                        "private_motivation": "Prevent Player_1 from dominating",
                        "alliance_coordination": False
                    },
                    "Player_6": {
                        "target": "Player_3",
                        "public_reasoning": "Player_3 has been dishonest in negotiations",
                        "private_motivation": "Vote with majority to stay safe",
                        "alliance_coordination": False
                    }
                },
                "round_outcome": {
                    "eliminated_player": "Player_3",
                    "wealth_changes": {
                        "Player_1": -5000,  # Spent in negotiation
                        "Player_2": 25000,  # Won auction challenge 
                        "Player_3": -8000,  # Spent in negotiation before elimination
                        "Player_4": 0,
                        "Player_5": 0,
                        "Player_6": 0
                    }
                }
            },
            {
                "round": 2,
                "intelligence": {
                    "Player_1": {
                        "investigate_players": ["Player_4", "Player_5"],
                        "discovered_info": ["Player_4 has protective abilities", "Player_5 knows secrets"],
                        "misinformation": "Player_6 is accumulating wealth secretly",
                        "target_of_misinformation": "Player_6"
                    },
                    "Player_2": {
                        "investigate_players": ["Player_5", "Player_6"],
                        "discovered_info": ["Player_5 has insider information", "Player_6 is planning sabotage"],
                        "misinformation": None,
                        "target_of_misinformation": None
                    }
                },
                "alliance": {
                    "Player_1": {
                        "alliance_type": "temporary",
                        "target_players": ["Player_4"],
                        "shared_information": "Player_2 might betray us soon",
                        "deception_strategy": "Keep options open for betrayal"
                    },
                    "Player_2": {
                        "alliance_type": "true",
                        "target_players": ["Player_5"],
                        "shared_information": "Player_1 is getting too powerful",
                        "deception_strategy": None
                    }
                },
                "challenge": {
                    "Player_1": {
                        "challenge_type": "dilemma",
                        "decision": "sacrifice_for_ally",
                        "reasoning": "Strengthen alliance with Player_4",
                        "bid_amount": 0
                    },
                    "Player_2": {
                        "challenge_type": "dilemma",
                        "decision": "choose_self_interest", 
                        "reasoning": "Prioritize personal survival",
                        "bid_amount": 0
                    }
                },
                "negotiation": {
                    "Player_1": {
                        "message": "We need to stick together against the others",
                        "offer_amount": 10000,
                        "deception_level": 0.3,
                        "information_to_extract": ["loyalty level", "future plans"],
                        "pressure_tactics": ["loyalty appeal", "mutual protection"]
                    },
                    "Player_2": {
                        "message": "I'm concerned about Player_1's growing influence",
                        "offer_amount": 7000,
                        "deception_level": 0.2,
                        "information_to_extract": ["alliance strength", "weakness"],
                        "pressure_tactics": ["concern sharing", "alliance building"]
                    }
                },
                "voting": {
                    "Player_1": {
                        "target": "Player_5",
                        "public_reasoning": "Player_5 has too much inside information",
                        "private_motivation": "Eliminate information advantage",
                        "alliance_coordination": True
                    },
                    "Player_2": {
                        "target": "Player_1",
                        "public_reasoning": "Player_1 is becoming too dominant",
                        "private_motivation": "Break up the strongest player",
                        "alliance_coordination": True
                    },
                    "Player_4": {
                        "target": "Player_5",
                        "public_reasoning": "Player_5 is manipulating information",
                        "private_motivation": "Support my ally Player_1",
                        "alliance_coordination": True
                    },
                    "Player_5": {
                        "target": "Player_1",
                        "public_reasoning": "Player_1 is the biggest threat", 
                        "private_motivation": "Self-preservation against strongest opponent",
                        "alliance_coordination": True
                    },
                    "Player_6": {
                        "target": "Player_1",
                        "public_reasoning": "Player_1 has too much control",
                        "private_motivation": "Vote against perceived leader",
                        "alliance_coordination": False
                    }
                },
                "round_outcome": {
                    "eliminated_player": "Player_1",
                    "wealth_changes": {
                        "Player_1": -10000,  # Spent in negotiation before elimination
                        "Player_2": -7000,   # Spent in negotiation
                        "Player_4": 15000,   # Benefited from Player_1's sacrifice
                        "Player_5": 0,
                        "Player_6": 0
                    }
                }
            }
        ],
        "final_results": {
            "remaining_players": ["Player_2", "Player_4", "Player_5", "Player_6"],
            "eliminated_players": ["Player_3", "Player_1"],
            "final_wealth": {
                "Player_2": 138000,
                "Player_4": 105000, 
                "Player_5": 110000,
                "Player_6": 95000
            },
            "winner": "Player_2",
            "game_duration": "45 minutes",
            "total_rounds": 2
        }
    }

def main():
    """
    Demonstrate enhanced Beast metrics calculation and reporting.
    """
    print("🎮 Beast Game Metrics Analysis Example")
    print("=" * 50)
    
    # Create sample game data
    print("\n📊 Generating sample Beast game data...")
    sample_game = create_sample_beast_game_data()
    
    # Create temporary directory for results
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save sample game data
        game_file = os.path.join(temp_dir, "beast_game_001.json")
        with open(game_file, 'w') as f:
            json.dump(sample_game, f, indent=2)
        
        print(f"✅ Sample game saved to: {game_file}")
        
        # Initialize metrics (without LLM for this example)
        print("\n🔍 Initializing Beast metrics calculator...")
        try:
            from metrics.beast_metrics import BeastMetrics
            
            # For demonstration, we'll skip LLM evaluation
            metrics_calculator = BeastMetrics(model=None)  
            
            print("✅ Metrics calculator initialized")
            
            # Calculate metrics
            print("\n📈 Calculating comprehensive metrics...")
            metrics_results = metrics_calculator.calculate_metrics(temp_dir)
            
            print("✅ Metrics calculation completed")
            
            # Display key metrics
            print("\n" + "="*60)
            print("🎯 KEY PERFORMANCE METRICS")
            print("="*60)
            
            print(f"\n📊 Games Analyzed: {metrics_results.get('games_total', 0)}")
            
            # Model Performance Metrics
            model_perf = metrics_results.get('model_performance', {})
            print(f"\n🤖 MODEL INFERENCE PERFORMANCE:")
            print(f"   Total Inferences: {model_perf.get('total_inferences', 0)}")
            print(f"   - Intelligence: {model_perf.get('intelligence_inferences', 0)}")
            print(f"   - Alliance: {model_perf.get('alliance_inferences', 0)}")
            print(f"   - Challenge: {model_perf.get('challenge_inferences', 0)}")
            print(f"   - Negotiation: {model_perf.get('negotiation_inferences', 0)}")
            print(f"   - Voting: {model_perf.get('voting_inferences', 0)}")
            
            # Player-specific metrics
            response_quality = model_perf.get('response_quality', {})
            if response_quality:
                print(f"\n📋 RESPONSE QUALITY BY PLAYER:")
                for player, quality in response_quality.items():
                    print(f"   {player}: {quality:.3f}")
            
            decision_consistency = model_perf.get('decision_consistency', {})
            if decision_consistency:
                print(f"\n🎯 DECISION CONSISTENCY BY PLAYER:")
                for player, consistency in decision_consistency.items():
                    print(f"   {player}: {consistency:.3f}")
            
            # Strategic metrics (if calculated)
            strategic_metrics = metrics_results.get('strategic_metrics', {})
            if strategic_metrics:
                print(f"\n⚔️ STRATEGIC PERFORMANCE:")
                print(f"   Alliance Success Rate: {strategic_metrics.get('alliance_success_rate', 0):.3f}")
                print(f"   Voting Accuracy: {strategic_metrics.get('voting_accuracy', 0):.3f}")
                print(f"   Wealth Management: {strategic_metrics.get('wealth_management_score', 0):.3f}")
            
            # Save detailed results
            results_file = "beast_metrics_results.json"
            with open(results_file, 'w') as f:
                json.dump(metrics_results, f, indent=2)
            print(f"\n💾 Detailed results saved to: {results_file}")
            
            # Generate markdown report if available
            if hasattr(metrics_calculator, 'generate_report'):
                print("\n📝 Generating markdown report...")
                report = metrics_calculator.generate_report(metrics_results, "markdown")
                
                report_file = "beast_analysis_report.md"
                with open(report_file, 'w') as f:
                    f.write(report)
                print(f"✅ Report saved to: {report_file}")
            
            print("\n" + "="*60)
            print("✅ BEAST METRICS ANALYSIS COMPLETE")
            print("="*60)
            
        except ImportError as e:
            print(f"❌ Error importing BeastMetrics: {e}")
            print("   Make sure the metrics module is properly installed.")
        except Exception as e:
            print(f"❌ Error during metrics calculation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 