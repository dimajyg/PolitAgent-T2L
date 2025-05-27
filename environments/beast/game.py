"""
Enhanced Beast Game Environment - Strategic Survival with Advanced Mechanics

Inspired by MrBeast's strategic challenge designs, this version features:
- Secret roles with special abilities
- Multi-phase rounds with strategic challenges
- Information warfare and trust tracking
- Alliance formation and betrayal mechanics
- Time pressure and elimination dynamics
"""

import json
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from environments.beast.agents.base_agent import (
    BeastAgent, 
    SecretRole, 
    ChallengeType, 
    TrustLevel,
    IntelligenceAction,
    AllianceAction,
    ChallengeAction,
    NegotiationAction,
    VoteAction
)
from environments.beast.utils.utils import create_message
from environments.beast.utils.prompt import format_prompt, get_voting_results_prompt_template
try:
    from langchain_core.language_models.base import BaseLanguageModel
except ImportError:
    from langchain.llms.base import BaseLLM as BaseLanguageModel

class EnhancedBeastGame:
    """
    Enhanced Beast game with advanced strategic mechanics.
    
    Features:
    - 6-8 players with secret roles and hidden information
    - Multi-phase rounds: Intelligence → Alliances → Challenges → Negotiations → Voting
    - Trust tracking and relationship dynamics
    - Information warfare and misinformation campaigns
    - Strategic resource management (wealth, influence, information)
    - Escalating pressure and time limits
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        num_players: int = 6,
        max_rounds: int = 8,
        output_dir: str = "./results",
        debug: bool = False
    ):
        self.llm = llm
        self.num_players = min(max(num_players, 6), 8)  # Enforce 6-8 players
        self.max_rounds = max_rounds
        self.output_dir = output_dir
        self.debug = debug
        
        # Game state
        self.current_round = 1
        self.players: List[str] = []
        self.agents: Dict[str, BeastAgent] = {}
        self.eliminated_players: List[str] = []
        self.game_history: List[Dict[str, Any]] = []
        
        # Strategic elements
        self.trust_matrix: Dict[str, Dict[str, TrustLevel]] = {}
        self.public_information: Dict[str, Any] = {}
        self.alliance_registry: List[Dict[str, Any]] = []
        self.information_pool: Dict[str, List[str]] = {}
        
        # Game balance
        self.pressure_level = 1  # Escalates each round
        self.available_challenges = list(ChallengeType)
        
    def initialize_game(self) -> None:
        """Initialize players with secret roles and hidden information"""
        # Create player names
        self.players = [f"Player_{i+1}" for i in range(self.num_players)]
        
        # Assign secret roles (ensure variety)
        available_roles = list(SecretRole)
        assigned_roles = []
        for i in range(self.num_players):
            if i < len(available_roles):
                assigned_roles.append(available_roles[i])
            else:
                assigned_roles.append(random.choice(available_roles))
        random.shuffle(assigned_roles)
        
        # Initialize agents with enhanced capabilities
        for i, player_name in enumerate(self.players):
            initial_wealth = random.randint(50000, 150000)
            initial_influence = random.randint(0, 3)
            
            agent = BeastAgent(
                llm=self.llm,
                player_name=player_name,
                players=self.players.copy(),
                wealth=initial_wealth,
                secret_role=assigned_roles[i],
                influence_points=initial_influence
            )
            
            self.agents[player_name] = agent
            
        # Initialize trust matrix
        for player1 in self.players:
            self.trust_matrix[player1] = {}
            for player2 in self.players:
                if player1 != player2:
                    self.trust_matrix[player1][player2] = TrustLevel.MEDIUM
                    
        # Initialize information pools
        for player in self.players:
            self.information_pool[player] = []
        
        if self.debug:
            print(f"🎮 Enhanced Beast Game initialized with {self.num_players} players")
            for player, agent in self.agents.items():
                print(f"  {player}: {agent.secret_role.value}, Wealth: {agent.wealth}, Influence: {agent.influence_points}")
    
    def run_intelligence_gathering_phase(self) -> Dict[str, Any]:
        """Phase 1: Intelligence gathering and misinformation spread"""
        if self.debug:
            print(f"\n🔍 ROUND {self.current_round} - Intelligence Gathering Phase")
        
        phase_results = {
            "investigations": {},
            "misinformation": {},
            "intelligence_discovered": {}
        }
        
        active_players = [p for p in self.players if p not in self.eliminated_players]
        
        for player_name in active_players:
            agent = self.agents[player_name]
            available_targets = [p for p in active_players if p != player_name]
            
            try:
                action = agent.intelligence_gathering_phase(available_targets)
                phase_results["investigations"][player_name] = action.investigate_players
                
                # Process investigations
                discovered_info = []
                for target in action.investigate_players:
                    if target in active_players:
                        info = self._conduct_investigation(player_name, target)
                        discovered_info.extend(info)
                        agent.known_information[target].extend(info)
                
                phase_results["intelligence_discovered"][player_name] = discovered_info
                
                # Process misinformation
                if action.misinformation and action.target_of_misinformation:
                    self._spread_misinformation(player_name, action.target_of_misinformation, action.misinformation)
                    phase_results["misinformation"][player_name] = {
                        "target": action.target_of_misinformation,
                        "content": action.misinformation
                    }
                    
            except Exception as e:
                if self.debug:
                    print(f"  ❌ Error in intelligence phase for {player_name}: {e}")
                    
        return phase_results
    
    def _conduct_investigation(self, investigator: str, target: str) -> List[str]:
        """Conduct investigation and return discovered information"""
        target_agent = self.agents[target]
        discovered = []
        
        # Random chance to discover different types of information
        if random.random() < 0.6:  # 60% chance to learn wealth level
            if target_agent.wealth > 100000:
                discovered.append(f"{target} appears to be wealthy (high resources)")
            elif target_agent.wealth < 70000:
                discovered.append(f"{target} seems to have limited resources")
            else:
                discovered.append(f"{target} has moderate wealth levels")
        
        if random.random() < 0.4:  # 40% chance to learn about alliances
            if target_agent.current_alliances:
                ally = random.choice(target_agent.current_alliances[0]) if target_agent.current_alliances[0] else None
                if ally and ally != target:
                    discovered.append(f"{target} appears to be allied with {ally}")
        
        if random.random() < 0.3:  # 30% chance to learn about role abilities
            discovered.append(f"{target} seems to have special strategic capabilities")
            
        if random.random() < 0.2:  # 20% chance to learn secret information
            if target_agent.secret_information:
                secret = random.choice(target_agent.secret_information)
                discovered.append(f"Intelligence suggests: {secret}")
        
        return discovered
    
    def _spread_misinformation(self, spreader: str, target: str, misinformation: str) -> None:
        """Spread misinformation in the information pool"""
        # Add to information pool where other players might pick it up
        fake_info = f"RUMOR about {target}: {misinformation}"
        
        # Spread to random other players
        active_players = [p for p in self.players if p not in self.eliminated_players and p != spreader]
        recipients = random.sample(active_players, min(2, len(active_players)))
        
        for recipient in recipients:
            self.information_pool[recipient].append(fake_info)
            
        if self.debug:
            print(f"  📢 {spreader} spread misinformation about {target}: {misinformation}")
    
    def run_alliance_formation_phase(self) -> Dict[str, Any]:
        """Phase 2: Secret alliance formation"""
        if self.debug:
            print(f"\n🤝 ROUND {self.current_round} - Alliance Formation Phase")
        
        phase_results = {
            "alliances_formed": {},
            "alliance_registry": []
        }
        
        active_players = [p for p in self.players if p not in self.eliminated_players]
        
        for player_name in active_players:
            agent = self.agents[player_name]
            available_targets = [p for p in active_players if p != player_name]
            
            try:
                action = agent.alliance_formation_phase(available_targets)
                
                if action.target_players:
                    alliance_data = {
                        "initiator": player_name,
                        "members": [player_name] + action.target_players,
                        "type": action.alliance_type,
                        "round_formed": self.current_round,
                        "shared_info": action.shared_information,
                        "deception_strategy": action.deception_strategy
                    }
                    
                    # Update agent alliance tracking
                    agent.current_alliances.append([player_name] + action.target_players)
                    
                    # Update trust levels for alliance members
                    for ally in action.target_players:
                        agent.update_trust(ally, "increase")
                        if ally in self.agents:
                            self.agents[ally].update_trust(player_name, "increase")
                    
                    phase_results["alliances_formed"][player_name] = alliance_data
                    self.alliance_registry.append(alliance_data)
                    
                    if self.debug:
                        print(f"  🤝 {player_name} formed {action.alliance_type} alliance with {action.target_players}")
                        
            except Exception as e:
                if self.debug:
                    print(f"  ❌ Error in alliance phase for {player_name}: {e}")
        
        phase_results["alliance_registry"] = self.alliance_registry
        return phase_results
    
    def run_strategic_challenge_phase(self) -> Dict[str, Any]:
        """Phase 3: Strategic challenges (auctions, dilemmas, etc.)"""
        challenge_type = random.choice(self.available_challenges)
        
        if self.debug:
            print(f"\n⚔️ ROUND {self.current_round} - Strategic Challenge: {challenge_type.value}")
        
        phase_results = {
            "challenge_type": challenge_type.value,
            "participant_actions": {},
            "challenge_results": {}
        }
        
        # Generate challenge-specific details
        challenge_details = self._generate_challenge_details(challenge_type)
        phase_results["challenge_details"] = challenge_details
        
        active_players = [p for p in self.players if p not in self.eliminated_players]
        participant_actions = {}
        
        # Collect all players' actions
        for player_name in active_players:
            agent = self.agents[player_name]
            try:
                action = agent.strategic_challenge_phase(challenge_type, challenge_details)
                participant_actions[player_name] = action
                phase_results["participant_actions"][player_name] = {
                    "decision": action.decision,
                    "reasoning": action.reasoning,
                    "bid_amount": action.bid_amount
                }
            except Exception as e:
                if self.debug:
                    print(f"  ❌ Error in challenge phase for {player_name}: {e}")
        
        # Process challenge results
        challenge_results = self._process_challenge_results(challenge_type, participant_actions, challenge_details)
        phase_results["challenge_results"] = challenge_results
        
        # Apply challenge effects
        self._apply_challenge_effects(challenge_results)
        
        return phase_results
    
    def _generate_challenge_details(self, challenge_type: ChallengeType) -> Dict[str, Any]:
        """Generate specific details for each challenge type"""
        if challenge_type == ChallengeType.AUCTION:
            return {
                "items": {
                    "immunity_idol": {"min_bid": 10000, "protection": "elimination immunity"},
                    "intelligence_report": {"min_bid": 5000, "benefit": "learn 2 secrets"},
                    "wealth_boost": {"min_bid": 15000, "gain": 50000},
                    "influence_power": {"min_bid": 8000, "gain": "2 influence points"}
                }
            }
        elif challenge_type == ChallengeType.DILEMMA:
            return {
                "scenario": "prisoner_dilemma",
                "cooperate_reward": 30000,
                "defect_penalty": -20000,
                "mixed_defector_gain": 50000,
                "mixed_cooperator_loss": -30000
            }
        elif challenge_type == ChallengeType.TRUST:
            return {
                "truth_trust_gain": 2,
                "lie_risk": "credibility loss if caught",
                "detection_chance": 0.3
            }
        elif challenge_type == ChallengeType.SACRIFICE:
            return {
                "sacrifice_cost": -40000,
                "group_benefit": 20000,
                "trust_gain": 3,
                "elimination_protection": True
            }
        
        return {}
    
    def _process_challenge_results(self, challenge_type: ChallengeType, actions: Dict[str, ChallengeAction], details: Dict[str, Any]) -> Dict[str, Any]:
        """Process the results of strategic challenges"""
        results = {"winners": {}, "effects": {}, "summary": ""}
        
        if challenge_type == ChallengeType.AUCTION:
            # Process auction bids
            bids_by_item = {}
            for player, action in actions.items():
                if action.bid_amount > 0:
                    item = action.decision  # Which item they're bidding on
                    if item not in bids_by_item:
                        bids_by_item[item] = []
                    bids_by_item[item].append((player, action.bid_amount))
            
            # Determine winners
            for item, bids in bids_by_item.items():
                if bids:
                    winner, winning_bid = max(bids, key=lambda x: x[1])
                    results["winners"][item] = {"player": winner, "bid": winning_bid}
                    results["effects"][winner] = {"item_won": item, "cost": winning_bid}
                    
        elif challenge_type == ChallengeType.DILEMMA:
            # Count cooperators vs defectors
            cooperators = [p for p, a in actions.items() if "cooperate" in a.decision.lower()]
            defectors = [p for p, a in actions.items() if "defect" in a.decision.lower()]
            
            if len(defectors) == 0:  # All cooperate
                for player in cooperators:
                    results["effects"][player] = {"wealth_change": details["cooperate_reward"]}
            elif len(cooperators) == 0:  # All defect
                for player in defectors:
                    results["effects"][player] = {"wealth_change": details["defect_penalty"]}
            else:  # Mixed
                for player in defectors:
                    results["effects"][player] = {"wealth_change": details["mixed_defector_gain"]}
                for player in cooperators:
                    results["effects"][player] = {"wealth_change": details["mixed_cooperator_loss"]}
                    
        elif challenge_type == ChallengeType.TRUST:
            # Process truth vs lie decisions
            for player, action in actions.items():
                if "truth" in action.decision.lower():
                    results["effects"][player] = {"trust_change": details["truth_trust_gain"]}
                else:
                    # Risk of being caught lying
                    if random.random() < details["detection_chance"]:
                        results["effects"][player] = {"trust_change": -2, "caught_lying": True}
                    else:
                        results["effects"][player] = {"deception_success": True}
                        
        elif challenge_type == ChallengeType.SACRIFICE:
            # Find volunteers and group choice
            volunteers = [p for p, a in actions.items() if "volunteer" in a.decision.lower()]
            if volunteers:
                sacrificed = random.choice(volunteers)
                results["effects"][sacrificed] = {"wealth_change": details["sacrifice_cost"], "trust_gain": details["trust_gain"]}
                # Everyone else benefits
                for player in actions.keys():
                    if player != sacrificed:
                        if player not in results["effects"]:
                            results["effects"][player] = {}
                        results["effects"][player]["wealth_change"] = results["effects"][player].get("wealth_change", 0) + details["group_benefit"]
        
        return results
    
    def _apply_challenge_effects(self, results: Dict[str, Any]) -> None:
        """Apply the effects of challenge results to players"""
        for player, effects in results.get("effects", {}).items():
            if player in self.agents:
                agent = self.agents[player]
                
                # Apply wealth changes
                if "wealth_change" in effects:
                    agent.wealth += effects["wealth_change"]
                    agent.wealth = max(0, agent.wealth)  # Prevent negative wealth
                
                # Apply trust changes
                if "trust_change" in effects:
                    for other_player in self.players:
                        if other_player != player and other_player not in self.eliminated_players:
                            if effects["trust_change"] > 0:
                                agent.update_trust(other_player, "increase")
                            else:
                                agent.update_trust(other_player, "decrease")
                
                # Apply influence changes
                if "influence_change" in effects:
                    agent.influence_points += effects["influence_change"]
                    agent.influence_points = max(0, agent.influence_points)
    
    def run_negotiation_phase(self) -> Dict[str, Any]:
        """Phase 4: Intense negotiations with time pressure"""
        if self.debug:
            print(f"\n💬 ROUND {self.current_round} - Negotiation Phase")
        
        phase_results = {
            "conversations": [],
            "offers_made": {},
            "offers_accepted": {},
            "trust_changes": {}
        }
        
        active_players = [p for p in self.players if p not in self.eliminated_players]
        
        # Multiple rounds of negotiations
        for negotiation_round in range(2):  # 2 rounds of negotiations
            if self.debug:
                print(f"  💬 Negotiation Round {negotiation_round + 1}")
            
            # Randomly pair players for conversations
            random.shuffle(active_players)
            pairs = [(active_players[i], active_players[i+1]) for i in range(0, len(active_players)-1, 2)]
            
            for player1, player2 in pairs:
                try:
                    # Both players negotiate
                    context = {
                        "round": self.current_round,
                        "pressure_level": self.pressure_level,
                        "remaining_players": len(active_players)
                    }
                    
                    action1 = self.agents[player1].enhanced_negotiation(player2, context)
                    action2 = self.agents[player2].enhanced_negotiation(player1, context)
                    
                    conversation = {
                        "participants": [player1, player2],
                        "player1_message": action1.message,
                        "player2_message": action2.message,
                        "player1_offer": action1.offer_amount,
                        "player2_offer": action2.offer_amount,
                        "negotiation_round": negotiation_round + 1
                    }
                    
                    phase_results["conversations"].append(conversation)
                    
                    # Process offers
                    if action1.offer_amount > 0:
                        accepted = self.agents[player2].handle_offer(player1, action1.offer_amount)
                        if accepted:
                            self.agents[player1].wealth -= action1.offer_amount
                            self.agents[player2].wealth += action1.offer_amount
                            phase_results["offers_accepted"][f"{player1}_to_{player2}"] = action1.offer_amount
                            
                    if action2.offer_amount > 0:
                        accepted = self.agents[player1].handle_offer(player2, action2.offer_amount)
                        if accepted:
                            self.agents[player2].wealth -= action2.offer_amount
                            self.agents[player1].wealth += action2.offer_amount
                            phase_results["offers_accepted"][f"{player2}_to_{player1}"] = action2.offer_amount
                    
                    # Update trust based on deception levels
                    if action1.deception_level > 0.5:
                        self.agents[player2].update_trust(player1, "decrease")
                    if action2.deception_level > 0.5:
                        self.agents[player1].update_trust(player2, "decrease")
                        
                except Exception as e:
                    if self.debug:
                        print(f"    ❌ Error in negotiation between {player1} and {player2}: {e}")
        
        return phase_results
    
    def run_voting_phase(self) -> Dict[str, Any]:
        """Phase 5: Elimination voting with strategic considerations"""
        if self.debug:
            print(f"\n🗳️ ROUND {self.current_round} - Elimination Voting Phase")
        
        active_players = [p for p in self.players if p not in self.eliminated_players]
        
        if len(active_players) <= 2:
            return {"status": "game_ended", "reason": "insufficient_players"}
        
        votes = {}
        voting_context = {
            "round": self.current_round,
            "pressure_level": self.pressure_level,
            "remaining_players": len(active_players)
        }
        
        # Collect votes
        for player_name in active_players:
            agent = self.agents[player_name]
            available_targets = [p for p in active_players if p != player_name]
            
            try:
                vote_action = agent.enhanced_voting(available_targets, voting_context)
                votes[player_name] = vote_action
                
                if self.debug:
                    print(f"  🗳️ {player_name} votes for {vote_action.target}: {vote_action.public_reasoning}")
                    
            except Exception as e:
                if self.debug:
                    print(f"  ❌ Error in voting for {player_name}: {e}")
                # Random fallback vote
                votes[player_name] = VoteAction(
                    target=random.choice(available_targets),
                    public_reasoning="Strategic elimination",
                    private_motivation="Fallback vote",
                    alliance_coordination=False
                )
        
        # Count votes
        vote_counts = {}
        for voter, vote_action in votes.items():
            target = vote_action.target
            if target not in vote_counts:
                vote_counts[target] = 0
            vote_counts[target] += 1
        
        # Determine elimination (handle ties)
        if vote_counts:
            max_votes = max(vote_counts.values())
            tied_players = [player for player, count in vote_counts.items() if count == max_votes]
            
            if len(tied_players) > 1:
                # Tiebreaker - random among tied players
                eliminated_player = random.choice(tied_players)
            else:
                eliminated_player = tied_players[0]
        else:
            # No valid votes - random elimination
            eliminated_player = random.choice(active_players)
        
        # Apply elimination
        self.eliminated_players.append(eliminated_player)
        eliminated_wealth = self.agents[eliminated_player].wealth
        
        # Elimination twist - distribute some wealth
        remaining_active = [p for p in active_players if p != eliminated_player]
        if remaining_active:
            wealth_bonus = eliminated_wealth // len(remaining_active)
            for player in remaining_active:
                self.agents[player].wealth += wealth_bonus
        
        phase_results = {
            "eliminated_player": eliminated_player,
            "eliminated_wealth": eliminated_wealth,
            "vote_breakdown": vote_counts,
            "votes_cast": {voter: vote.target for voter, vote in votes.items()},
            "remaining_players": remaining_active,
            "wealth_redistribution": wealth_bonus if remaining_active else 0
        }
        
        if self.debug:
            print(f"  💀 {eliminated_player} has been eliminated!")
            print(f"  💰 Wealth redistributed: {wealth_bonus} to each survivor")
        
        return phase_results
    
    def run_single_round(self) -> Dict[str, Any]:
        """Execute a complete round with all phases"""
        if self.debug:
            print(f"\n🚀 === ROUND {self.current_round} START === ")
            print(f"🎮 Pressure Level: {self.pressure_level}")
        
        round_results = {
            "round": self.current_round,
            "pressure_level": self.pressure_level,
            "phase_results": {}
        }
        
        # Update all agents with current round
        for agent in self.agents.values():
            agent.current_round = self.current_round
        
        # Phase 1: Intelligence Gathering
        round_results["phase_results"]["intelligence"] = self.run_intelligence_gathering_phase()
        
        # Phase 2: Alliance Formation
        round_results["phase_results"]["alliances"] = self.run_alliance_formation_phase()
        
        # Phase 3: Strategic Challenge
        round_results["phase_results"]["challenge"] = self.run_strategic_challenge_phase()
        
        # Phase 4: Negotiations
        round_results["phase_results"]["negotiations"] = self.run_negotiation_phase()
        
        # Phase 5: Voting
        voting_results = self.run_voting_phase()
        round_results["phase_results"]["voting"] = voting_results
        
        # Check game end conditions
        active_players = [p for p in self.players if p not in self.eliminated_players]
        round_results["active_players_remaining"] = len(active_players)
        round_results["game_continues"] = len(active_players) > 2 and self.current_round < self.max_rounds
        
        # Escalate pressure for next round
        self.pressure_level = min(5, self.pressure_level + 0.5)
        
        # Save round state
        self.game_history.append(round_results)
        
        if self.debug:
            print(f"🚀 === ROUND {self.current_round} END ===")
            print(f"🎯 Players remaining: {len(active_players)}")
            if active_players:
                for player in active_players:
                    print(f"  {player}: {self.agents[player].wealth} wealth, {self.agents[player].influence_points} influence")
        
        return round_results
    
    def run_game(self) -> Dict[str, Any]:
        """Run the complete enhanced Beast game"""
        start_time = time.time()
        
        if self.debug:
            print("🎮 Starting Enhanced Beast Game...")
        
        self.initialize_game()
        
        # Main game loop
        while True:
            round_results = self.run_single_round()
            
            if not round_results["game_continues"]:
                break
                
            self.current_round += 1
        
        # Determine final results
        active_players = [p for p in self.players if p not in self.eliminated_players]
        
        # Final rankings by wealth
        final_rankings = []
        for player in active_players:
            agent = self.agents[player]
            final_rankings.append({
                "player": player,
                "wealth": agent.wealth,
                "role": agent.secret_role.value,
                "influence": agent.influence_points,
                "status": "survivor"
            })
        
        # Add eliminated players
        for player in self.eliminated_players:
            agent = self.agents[player]
            final_rankings.append({
                "player": player,
                "wealth": agent.wealth,
                "role": agent.secret_role.value,
                "influence": agent.influence_points,
                "status": "eliminated"
            })
        
        # Sort by wealth (winners)
        final_rankings.sort(key=lambda x: x["wealth"], reverse=True)
        
        game_results = {
            "game_type": "enhanced_beast",
            "total_rounds": self.current_round,
            "game_duration": time.time() - start_time,
            "final_rankings": final_rankings,
            "survivors": [p["player"] for p in final_rankings if p["status"] == "survivor"],
            "eliminated_players": self.eliminated_players,
            "winner": final_rankings[0]["player"] if final_rankings else None,
            "game_history": self.game_history,
            "alliance_registry": self.alliance_registry
        }
        
        if self.debug:
            print("\n🏆 === GAME COMPLETE ===")
            print(f"🎯 Winner: {game_results['winner']}")
            print(f"⏱️ Duration: {game_results['game_duration']:.2f} seconds")
            print("📊 Final Rankings:")
            for i, player_data in enumerate(final_rankings[:3], 1):
                print(f"  {i}. {player_data['player']} ({player_data['role']}): {player_data['wealth']} wealth")
        
        return game_results

def run_enhanced_beast_game(
    llm: BaseLanguageModel,
    num_players: int = 6,
    max_rounds: int = 8,
    output_dir: str = "./results",
    debug: bool = False
) -> Dict[str, Any]:
    """
    Run the enhanced Beast game.
    
    Args:
        llm: Language model for agents
        num_players: Number of players (6-8)
        max_rounds: Maximum rounds before game ends
        output_dir: Directory to save results
        debug: Enable debug output
        
    Returns:
        Dict containing game results and statistics
    """
    game = EnhancedBeastGame(llm, num_players, max_rounds, output_dir, debug)
    return game.run_game()

# Legacy compatibility function
def run_beast_game(llm: BaseLanguageModel, num_players: int = 10, max_rounds: int = 5, output_dir: str = "./results", debug: bool = False) -> Dict[str, Any]:
    """Legacy compatibility wrapper"""
    # Convert old parameters to new enhanced version
    enhanced_num_players = min(8, max(6, num_players))  # Clamp to 6-8 range
    enhanced_max_rounds = max(max_rounds, 8)  # Minimum 8 rounds for full strategic experience
    
    return run_enhanced_beast_game(llm, enhanced_num_players, enhanced_max_rounds, output_dir, debug)

class BeastGame:
    """
    Compatibility wrapper for benchmark integration.
    
    This class provides the interface expected by the benchmark system
    while delegating to the enhanced Beast game implementation.
    """
    
    def __init__(self, args, llm: BaseLanguageModel):
        """Initialize with benchmark-style arguments."""
        self.args = args
        self.llm = llm
        self.enhanced_game = None
        
    def init_game(self) -> str:
        """Initialize the game and return settings description."""
        # Extract parameters from args
        num_players = getattr(self.args, 'num_players', 6)
        max_rounds = getattr(self.args, 'max_rounds', 8)
        debug = getattr(self.args, 'debug', False)
        output_dir = getattr(self.args, 'output_dir', './results/beast')
        
        # Ensure valid parameters for enhanced game
        num_players = min(8, max(6, num_players))
        max_rounds = max(8, max_rounds)
        
        # Create the enhanced game
        self.enhanced_game = EnhancedBeastGame(
            llm=self.llm,
            num_players=num_players,
            max_rounds=max_rounds,
            output_dir=output_dir,
            debug=debug
        )
        
        # Initialize the enhanced game
        self.enhanced_game.initialize_game()
        
        return f"Enhanced Beast Game: {num_players} players, {max_rounds} max rounds"
    
    def game_loop(self, log_file) -> Dict[str, Any]:
        """Run the game loop and return results."""
        if self.enhanced_game is None:
            raise RuntimeError("Game not initialized. Call init_game() first.")
        
        # Redirect debug output to log file if needed
        original_debug = self.enhanced_game.debug
        if hasattr(log_file, 'write'):
            # For file-like objects, we'll capture the output differently
            self.enhanced_game.debug = False  # Disable console debug
            
        try:
            # Run the enhanced game
            results = self.enhanced_game.run_game()
            
            # Write game details to log file
            if hasattr(log_file, 'write'):
                log_file.write(f"Enhanced Beast Game Results:\n")
                log_file.write(f"Winner: {results.get('winner', 'Unknown')}\n")
                log_file.write(f"Total Rounds: {results.get('total_rounds', 0)}\n")
                log_file.write(f"Survivors: {results.get('survivors', [])}\n")
                log_file.write(f"Game Duration: {results.get('game_duration', 0):.2f}s\n")
                
                # Write final rankings
                log_file.write("\nFinal Rankings:\n")
                for i, player_data in enumerate(results.get('final_rankings', []), 1):
                    log_file.write(f"{i}. {player_data['player']} ({player_data['role']}): "
                                 f"{player_data['wealth']} wealth, {player_data['status']}\n")
                
                # Write round summaries
                log_file.write("\nRound Summaries:\n")
                for round_data in results.get('game_history', []):
                    round_num = round_data['round']
                    active_players = round_data['active_players_remaining']
                    eliminated = round_data['phase_results'].get('voting', {}).get('eliminated_player', 'None')
                    log_file.write(f"Round {round_num}: {active_players} players remaining, "
                                 f"eliminated: {eliminated}\n")
            
            return results
            
        finally:
            # Restore original debug setting
            self.enhanced_game.debug = original_debug