import json
import logging
import os
import re
import time
from typing import Dict, List, Any, Tuple, Optional, Union
from metrics.base_metrics import BaseMetrics
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class BeastMetrics(BaseMetrics):
    """
    Comprehensive metrics for evaluating model performance in Beast strategic wealth game.
    Includes inference metrics, strategic evaluation, and detailed reporting.
    """
    
    def __init__(self, model: Optional[BaseLanguageModel] = None):
        """
        Initialize Beast metrics with optional LLM evaluator.
        
        Args:
            model: LLM model for evaluation (LLM as judge)
        """
        super().__init__(game_type="beast")
        self.model = model
        self.metrics = {}
        self.inference_data = []  # Store detailed inference metrics
        self.strategic_analysis = {}  # Store strategic patterns
    
    def compute_all(self) -> Dict[str, Any]:
        """
        Implementation of abstract method from BaseMetrics.
        Computes all metrics from recorded events.
        
        Returns:
            Dict[str, Any]: Complete metrics suite
        """
        # For Beast metrics, we don't use the event-based approach
        # Instead, we calculate from game logs in calculate_metrics
        return self.computed_metrics
    
    def calculate_metrics(self, results_dir: str) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics from game results.
        
        Args:
            results_dir: Directory containing game results
            
        Returns:
            Dict[str, Any]: Complete metrics suite
        """
        game_logs = self._load_game_logs(results_dir)
        
        if not game_logs:
            logger.warning("No games found in results directory: %s", results_dir)
            return {"error": "No game logs found", "games_total": 0}
        
        # Core metrics
        self.metrics = {
            "games_total": len(game_logs),
            "timestamp": datetime.now().isoformat(),
            "model_performance": self._calculate_model_inference_metrics(game_logs),
            "strategic_metrics": self._calculate_strategic_metrics(game_logs),
            "social_metrics": self._calculate_social_metrics(game_logs),
            "economic_metrics": self._calculate_economic_metrics(game_logs),
            "game_outcome_metrics": self._calculate_game_outcome_metrics(game_logs),
            "behavioral_analysis": self._calculate_behavioral_analysis(game_logs)
        }
        
        # LLM as judge evaluation
        if self.model:
            self.metrics["llm_evaluation"] = self._calculate_llm_judge_metrics(game_logs)
        
        # Generate comprehensive report
        self.metrics["detailed_report"] = self._generate_detailed_report()
        
        # Convert numpy types to JSON-serializable types
        self.metrics = self._convert_numpy_types(self.metrics)
        
        # Store in computed_metrics for base class compatibility
        self.computed_metrics = self.metrics
        
        return self.metrics
    
    def _load_game_logs(self, results_dir: str) -> List[Dict[str, Any]]:
        """
        Load game logs from results directory.
        
        Args:
            results_dir: Directory with game results
            
        Returns:
            List[Dict[str, Any]]: List of game logs
        """
        game_logs = []
        for root, _, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.json') and 'beast' in file:
                    try:
                        with open(os.path.join(root, file), 'r') as f:
                            game_data = json.load(f)
                            game_logs.append(game_data)
                    except Exception as e:
                        logger.error(f"Error loading game log {file}: {e}")
        
        return game_logs
    
    def _calculate_model_inference_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate detailed model inference performance metrics.
        
        Args:
            game_logs: List of game results
            
        Returns:
            Dict[str, Any]: Inference performance metrics
        """
        inference_metrics = {
            "total_inferences": 0,
            "intelligence_inferences": 0,
            "alliance_inferences": 0,
            "challenge_inferences": 0,
            "negotiation_inferences": 0,
            "voting_inferences": 0,
            "response_quality": {},
            "decision_consistency": {},
            "strategic_coherence": {},
            "error_rate": {},
            "total_errors": 0,
            "context_utilization": {}
        }
        
        players = set()
        
        for game in game_logs:
            # Extract player data from different game formats
            if "players" in game:
                players.update(game["players"])
            
            game_rounds = game.get("rounds", [])
            if not game_rounds and "game_data" in game:
                game_rounds = game["game_data"].get("rounds", [])
            
            # Count inferences by phase
            for round_data in game_rounds:
                # Intelligence phase inferences
                intel_data = round_data.get("intelligence", {})
                if isinstance(intel_data, dict):
                    inference_metrics["intelligence_inferences"] += len(intel_data)
                
                # Alliance phase inferences
                alliance_data = round_data.get("alliance", {})
                if isinstance(alliance_data, dict):
                    inference_metrics["alliance_inferences"] += len(alliance_data)
                
                # Challenge phase inferences
                challenge_data = round_data.get("challenge", {})
                if isinstance(challenge_data, dict):
                    inference_metrics["challenge_inferences"] += len(challenge_data)
                
                # Negotiation phase inferences
                negotiation_data = round_data.get("negotiation", {})
                if isinstance(negotiation_data, dict):
                    inference_metrics["negotiation_inferences"] += len(negotiation_data)
                
                # Voting phase inferences
                voting_data = round_data.get("voting", {})
                if isinstance(voting_data, dict):
                    inference_metrics["voting_inferences"] += len(voting_data)
                
                # Collect players from round data
                for phase in ["intelligence", "alliance", "challenge", "negotiation", "voting"]:
                    phase_data = round_data.get(phase, {})
                    if isinstance(phase_data, dict):
                        players.update(phase_data.keys())
        
        # Initialize player-specific metrics
        for player in players:
            inference_metrics["response_quality"][player] = 0.0
            inference_metrics["decision_consistency"][player] = 0.0
            inference_metrics["strategic_coherence"][player] = 0.0
            inference_metrics["error_rate"][player] = 0.0
            inference_metrics["context_utilization"][player] = 0.0
        
        # Analyze response quality and consistency for each player
        for game in game_logs:
            game_rounds = game.get("rounds", [])
            if not game_rounds and "game_data" in game:
                game_rounds = game["game_data"].get("rounds", [])
            
            for player in players:
                # Extract player decisions across all phases
                player_decisions = self._extract_player_decisions_beast(player, game_rounds)
                
                if player_decisions:
                    # Quality analysis
                    quality_score = self._analyze_decision_quality_beast(player_decisions)
                    inference_metrics["response_quality"][player] = quality_score
                    
                    # Consistency analysis
                    consistency_score = self._analyze_decision_consistency_beast(player_decisions)
                    inference_metrics["decision_consistency"][player] = consistency_score
                    
                    # Strategic coherence
                    coherence_score = self._analyze_strategic_coherence_beast(player_decisions)
                    inference_metrics["strategic_coherence"][player] = coherence_score
                    
                    # Context utilization
                    context_score = self._analyze_context_utilization_beast(player_decisions)
                    inference_metrics["context_utilization"][player] = context_score
                    
                    # Error rate analysis
                    error_rate = self._calculate_error_rate_beast(player_decisions)
                    inference_metrics["error_rate"][player] = error_rate
                    inference_metrics["total_errors"] += int(error_rate * len(player_decisions))
        
        # Calculate totals
        inference_metrics["total_inferences"] = (
            inference_metrics["intelligence_inferences"] + 
            inference_metrics["alliance_inferences"] + 
            inference_metrics["challenge_inferences"] + 
            inference_metrics["negotiation_inferences"] + 
            inference_metrics["voting_inferences"]
        )
        
        return inference_metrics
    
    def _extract_player_decisions_beast(self, player: str, game_rounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract all decisions made by a player across game rounds."""
        decisions = []
        
        for round_num, round_data in enumerate(game_rounds):
            round_decisions = {
                "round": round_num + 1,
                "intelligence": round_data.get("intelligence", {}).get(player, {}),
                "alliance": round_data.get("alliance", {}).get(player, {}),
                "challenge": round_data.get("challenge", {}).get(player, {}),
                "negotiation": round_data.get("negotiation", {}).get(player, {}),
                "voting": round_data.get("voting", {}).get(player, {})
            }
            decisions.append(round_decisions)
        
        return decisions
    
    def _analyze_decision_quality_beast(self, player_decisions: List[Dict[str, Any]]) -> float:
        """Analyze the quality of a player's decisions."""
        quality_scores = []
        
        for decision in player_decisions:
            round_quality = 0.0
            valid_phases = 0
            
            # Intelligence phase quality
            intel = decision.get("intelligence", {})
            if intel:
                intel_quality = self._evaluate_intelligence_quality(intel)
                quality_scores.append(intel_quality)
                round_quality += intel_quality
                valid_phases += 1
            
            # Alliance phase quality
            alliance = decision.get("alliance", {})
            if alliance:
                alliance_quality = self._evaluate_alliance_quality(alliance)
                quality_scores.append(alliance_quality)
                round_quality += alliance_quality
                valid_phases += 1
            
            # Challenge phase quality
            challenge = decision.get("challenge", {})
            if challenge:
                challenge_quality = self._evaluate_challenge_quality(challenge)
                quality_scores.append(challenge_quality)
                round_quality += challenge_quality
                valid_phases += 1
            
            # Negotiation phase quality
            negotiation = decision.get("negotiation", {})
            if negotiation:
                negotiation_quality = self._evaluate_negotiation_quality(negotiation)
                quality_scores.append(negotiation_quality)
                round_quality += negotiation_quality
                valid_phases += 1
            
            # Voting phase quality
            voting = decision.get("voting", {})
            if voting:
                voting_quality = self._evaluate_voting_quality(voting)
                quality_scores.append(voting_quality)
                round_quality += voting_quality
                valid_phases += 1
            
            if valid_phases > 0:
                quality_scores.append(round_quality / valid_phases)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _analyze_decision_consistency_beast(self, player_decisions: List[Dict[str, Any]]) -> float:
        """Analyze consistency in player's decision making."""
        if len(player_decisions) < 2:
            return 1.0
        
        consistency_scores = []
        
        # Analyze alliance consistency
        alliance_decisions = [d.get("alliance", {}) for d in player_decisions if d.get("alliance")]
        if len(alliance_decisions) > 1:
            alliance_consistency = self._calculate_alliance_consistency(alliance_decisions)
            consistency_scores.append(alliance_consistency)
        
        # Analyze voting pattern consistency
        voting_decisions = [d.get("voting", {}) for d in player_decisions if d.get("voting")]
        if len(voting_decisions) > 1:
            voting_consistency = self._calculate_voting_consistency(voting_decisions)
            consistency_scores.append(voting_consistency)
        
        # Analyze strategic coherence across phases
        strategic_consistency = self._calculate_strategic_consistency(player_decisions)
        consistency_scores.append(strategic_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _analyze_strategic_coherence_beast(self, player_decisions: List[Dict[str, Any]]) -> float:
        """Analyze strategic coherence across all phases."""
        coherence_scores = []
        
        for decision in player_decisions:
            # Check alignment between intelligence and alliance actions
            intel = decision.get("intelligence", {})
            alliance = decision.get("alliance", {})
            
            if intel and alliance:
                intel_alliance_coherence = self._evaluate_intel_alliance_coherence(intel, alliance)
                coherence_scores.append(intel_alliance_coherence)
            
            # Check alignment between alliance and voting
            voting = decision.get("voting", {})
            if alliance and voting:
                alliance_voting_coherence = self._evaluate_alliance_voting_coherence(alliance, voting)
                coherence_scores.append(alliance_voting_coherence)
            
            # Check negotiation alignment with overall strategy
            negotiation = decision.get("negotiation", {})
            if negotiation:
                negotiation_coherence = self._evaluate_negotiation_coherence(negotiation, decision)
                coherence_scores.append(negotiation_coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _analyze_context_utilization_beast(self, player_decisions: List[Dict[str, Any]]) -> float:
        """Analyze how well player utilizes available context."""
        utilization_scores = []
        
        for decision in player_decisions:
            round_utilization = 0.0
            valid_phases = 0
            
            # Intelligence context utilization
            intel = decision.get("intelligence", {})
            if intel:
                intel_utilization = self._evaluate_intelligence_context_usage(intel)
                utilization_scores.append(intel_utilization)
                round_utilization += intel_utilization
                valid_phases += 1
            
            # Alliance context utilization
            alliance = decision.get("alliance", {})
            if alliance:
                alliance_utilization = self._evaluate_alliance_context_usage(alliance)
                utilization_scores.append(alliance_utilization)
                round_utilization += alliance_utilization
                valid_phases += 1
            
            # Voting context utilization
            voting = decision.get("voting", {})
            if voting:
                voting_utilization = self._evaluate_voting_context_usage(voting)
                utilization_scores.append(voting_utilization)
                round_utilization += voting_utilization
                valid_phases += 1
            
            if valid_phases > 0:
                utilization_scores.append(round_utilization / valid_phases)
        
        return np.mean(utilization_scores) if utilization_scores else 0.5
    
    def _calculate_error_rate_beast(self, player_decisions: List[Dict[str, Any]]) -> float:
        """Calculate error rate in player decisions."""
        total_decisions = 0
        total_errors = 0
        
        for decision in player_decisions:
            for phase in ["intelligence", "alliance", "challenge", "negotiation", "voting"]:
                phase_data = decision.get(phase, {})
                if phase_data:
                    total_decisions += 1
                    if self._is_decision_error(phase, phase_data):
                        total_errors += 1
        
        return total_errors / total_decisions if total_decisions > 0 else 0.0
    
    # Evaluation helper methods for different phases
    def _evaluate_intelligence_quality(self, intel_data: Dict[str, Any]) -> float:
        """Evaluate quality of intelligence gathering decisions."""
        score = 0.5  # Base score
        
        # Check if player investigated appropriate targets
        investigated = intel_data.get("investigate_players", [])
        if len(investigated) == 2:  # Full investigation capacity used
            score += 0.2
        elif len(investigated) == 1:
            score += 0.1
        
        # Check strategic use of misinformation
        misinformation = intel_data.get("misinformation")
        target_misinformation = intel_data.get("target_of_misinformation")
        
        if misinformation and target_misinformation:
            score += 0.2  # Bonus for using misinformation strategically
        
        # Check quality of discovered information
        discovered = intel_data.get("discovered_info", [])
        if discovered:
            score += min(len(discovered) * 0.1, 0.3)  # Up to 0.3 bonus
        
        return min(score, 1.0)
    
    def _evaluate_alliance_quality(self, alliance_data: Dict[str, Any]) -> float:
        """Evaluate quality of alliance formation decisions."""
        score = 0.5  # Base score
        
        alliance_type = alliance_data.get("alliance_type", "")
        target_players = alliance_data.get("target_players", [])
        shared_info = alliance_data.get("shared_information", "")
        deception_strategy = alliance_data.get("deception_strategy")
        
        # Evaluate alliance type appropriateness
        if alliance_type in ["true", "false", "temporary"]:
            score += 0.1
        
        # Evaluate target selection
        if len(target_players) > 0:
            score += 0.2
        
        # Evaluate information sharing strategy
        if shared_info:
            score += 0.1
        
        # Evaluate deception strategy (if false alliance)
        if alliance_type == "false" and deception_strategy:
            score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_challenge_quality(self, challenge_data: Dict[str, Any]) -> float:
        """Evaluate quality of challenge phase decisions."""
        score = 0.5  # Base score
        
        decision = challenge_data.get("decision", "")
        reasoning = challenge_data.get("reasoning", "")
        bid_amount = challenge_data.get("bid_amount", 0)
        
        # Evaluate decision clarity
        if decision:
            score += 0.2
        
        # Evaluate reasoning quality
        if reasoning and len(reasoning) > 10:  # Substantial reasoning
            score += 0.2
        
        # Evaluate bid appropriateness (for auction challenges)
        if challenge_data.get("challenge_type") == "auction":
            if 5000 <= bid_amount <= 50000:  # Reasonable bid range
                score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_negotiation_quality(self, negotiation_data: Dict[str, Any]) -> float:
        """Evaluate quality of negotiation decisions."""
        score = 0.5  # Base score
        
        message = negotiation_data.get("message", "")
        offer_amount = negotiation_data.get("offer_amount", 0)
        deception_level = negotiation_data.get("deception_level", 0)
        info_to_extract = negotiation_data.get("information_to_extract", [])
        pressure_tactics = negotiation_data.get("pressure_tactics", [])
        
        # Evaluate message quality
        if message and len(message) > 20:  # Substantial message
            score += 0.15
        
        # Evaluate strategic offer
        if offer_amount > 0:
            score += 0.1
        
        # Evaluate information extraction strategy
        if info_to_extract:
            score += min(len(info_to_extract) * 0.05, 0.15)
        
        # Evaluate pressure tactics
        if pressure_tactics:
            score += min(len(pressure_tactics) * 0.05, 0.1)
        
        return min(score, 1.0)
    
    def _evaluate_voting_quality(self, voting_data: Dict[str, Any]) -> float:
        """Evaluate quality of voting decisions."""
        score = 0.5  # Base score
        
        target = voting_data.get("target", "")
        public_reasoning = voting_data.get("public_reasoning", "")
        private_motivation = voting_data.get("private_motivation", "")
        alliance_coordination = voting_data.get("alliance_coordination", False)
        
        # Evaluate target selection
        if target:
            score += 0.2
        
        # Evaluate public reasoning
        if public_reasoning and len(public_reasoning) > 10:
            score += 0.15
        
        # Evaluate private motivation clarity
        if private_motivation and len(private_motivation) > 10:
            score += 0.1
        
        # Evaluate strategic coordination
        if alliance_coordination:
            score += 0.05
        
        return min(score, 1.0)
    
    # Consistency analysis methods
    def _calculate_alliance_consistency(self, alliance_decisions: List[Dict[str, Any]]) -> float:
        """Calculate consistency in alliance formation patterns."""
        if len(alliance_decisions) < 2:
            return 1.0
        
        # Analyze alliance type consistency
        alliance_types = [decision.get("alliance_type", "") for decision in alliance_decisions]
        type_consistency = len(set(alliance_types)) / len(alliance_types)  # Lower is more consistent
        
        # Analyze target consistency
        all_targets = []
        for decision in alliance_decisions:
            targets = decision.get("target_players", [])
            all_targets.extend(targets)
        
        target_consistency = 1.0
        if all_targets:
            unique_targets = len(set(all_targets))
            total_targets = len(all_targets)
            target_consistency = unique_targets / total_targets if total_targets > 0 else 1.0
        
        return 1.0 - ((1.0 - type_consistency) + (1.0 - target_consistency)) / 2
    
    def _calculate_voting_consistency(self, voting_decisions: List[Dict[str, Any]]) -> float:
        """Calculate consistency in voting patterns."""
        if len(voting_decisions) < 2:
            return 1.0
        
        # Analyze voting targets
        targets = [decision.get("target", "") for decision in voting_decisions]
        
        # Analyze alliance coordination consistency
        coordinated_votes = [decision.get("alliance_coordination", False) for decision in voting_decisions]
        coordination_consistency = 1.0 - (coordinated_votes.count(True) / len(coordinated_votes)) if coordinated_votes else 0.5
        
        # Overall consistency
        return coordination_consistency
    
    def _calculate_strategic_consistency(self, player_decisions: List[Dict[str, Any]]) -> float:
        """Calculate overall strategic consistency across phases."""
        consistency_scores = []
        
        for i in range(1, len(player_decisions)):
            prev_decision = player_decisions[i-1]
            curr_decision = player_decisions[i]
            
            # Compare alliance strategies
            prev_alliance = prev_decision.get("alliance", {})
            curr_alliance = curr_decision.get("alliance", {})
            
            if prev_alliance and curr_alliance:
                alliance_consistency = self._compare_alliance_strategies(prev_alliance, curr_alliance)
                consistency_scores.append(alliance_consistency)
            
            # Compare voting patterns
            prev_voting = prev_decision.get("voting", {})
            curr_voting = curr_decision.get("voting", {})
            
            if prev_voting and curr_voting:
                voting_consistency = self._compare_voting_strategies(prev_voting, curr_voting)
                consistency_scores.append(voting_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _compare_alliance_strategies(self, prev_alliance: Dict[str, Any], curr_alliance: Dict[str, Any]) -> float:
        """Compare alliance strategies between rounds."""
        score = 0.5
        
        # Check alliance type consistency
        prev_type = prev_alliance.get("alliance_type", "")
        curr_type = curr_alliance.get("alliance_type", "")
        
        if prev_type == curr_type:
            score += 0.2
        
        # Check target overlap
        prev_targets = set(prev_alliance.get("target_players", []))
        curr_targets = set(curr_alliance.get("target_players", []))
        
        if prev_targets and curr_targets:
            overlap = len(prev_targets.intersection(curr_targets))
            total_unique = len(prev_targets.union(curr_targets))
            if total_unique > 0:
                score += 0.3 * (overlap / total_unique)
        
        return min(score, 1.0)
    
    def _compare_voting_strategies(self, prev_voting: Dict[str, Any], curr_voting: Dict[str, Any]) -> float:
        """Compare voting strategies between rounds."""
        score = 0.5
        
        # Check coordination consistency
        prev_coord = prev_voting.get("alliance_coordination", False)
        curr_coord = curr_voting.get("alliance_coordination", False)
        
        if prev_coord == curr_coord:
            score += 0.3
        
        # Check reasoning consistency
        prev_reasoning = prev_voting.get("private_motivation", "")
        curr_reasoning = curr_voting.get("private_motivation", "")
        
        if prev_reasoning and curr_reasoning:
            # Simple keyword overlap analysis
            prev_keywords = set(prev_reasoning.lower().split())
            curr_keywords = set(curr_reasoning.lower().split())
            
            if prev_keywords and curr_keywords:
                overlap = len(prev_keywords.intersection(curr_keywords))
                total_unique = len(prev_keywords.union(curr_keywords))
                if total_unique > 0:
                    score += 0.2 * (overlap / total_unique)
        
        return min(score, 1.0)
    
    # Coherence analysis methods
    def _evaluate_intel_alliance_coherence(self, intel_data: Dict[str, Any], alliance_data: Dict[str, Any]) -> float:
        """Evaluate coherence between intelligence and alliance decisions."""
        score = 0.5
        
        investigated = intel_data.get("investigate_players", [])
        alliance_targets = alliance_data.get("target_players", [])
        
        # Check if alliance targets were investigated
        investigated_set = set(investigated)
        alliance_set = set(alliance_targets)
        
        if investigated_set and alliance_set:
            overlap = len(investigated_set.intersection(alliance_set))
            score += 0.3 * (overlap / len(alliance_set)) if alliance_set else 0
        
        # Check if misinformation aligns with alliance strategy
        misinformation_target = intel_data.get("target_of_misinformation")
        if misinformation_target and misinformation_target not in alliance_targets:
            score += 0.2  # Bonus for strategic misinformation
        
        return min(score, 1.0)
    
    def _evaluate_alliance_voting_coherence(self, alliance_data: Dict[str, Any], voting_data: Dict[str, Any]) -> float:
        """Evaluate coherence between alliance and voting decisions."""
        score = 0.5
        
        alliance_targets = alliance_data.get("target_players", [])
        voting_target = voting_data.get("target", "")
        alliance_coordination = voting_data.get("alliance_coordination", False)
        
        # Check if voting aligns with alliance
        if alliance_coordination and alliance_targets:
            score += 0.3
        
        # Check if voting target conflicts with alliance
        if voting_target and voting_target not in alliance_targets:
            score += 0.2  # Consistent with not voting against allies
        
        return min(score, 1.0)
    
    def _evaluate_negotiation_coherence(self, negotiation_data: Dict[str, Any], decision: Dict[str, Any]) -> float:
        """Evaluate coherence of negotiation with overall round strategy."""
        score = 0.5
        
        deception_level = negotiation_data.get("deception_level", 0)
        alliance_data = decision.get("alliance", {})
        alliance_type = alliance_data.get("alliance_type", "")
        
        # Check if deception level aligns with alliance type
        if alliance_type == "false" and deception_level > 0.5:
            score += 0.2  # High deception for false alliance
        elif alliance_type == "true" and deception_level < 0.3:
            score += 0.2  # Low deception for true alliance
        
        # Check information extraction alignment
        info_to_extract = negotiation_data.get("information_to_extract", [])
        if info_to_extract:
            score += 0.1  # Bonus for strategic information gathering
        
        return min(score, 1.0)
    
    # Context utilization methods
    def _evaluate_intelligence_context_usage(self, intel_data: Dict[str, Any]) -> float:
        """Evaluate how well intelligence phase uses available context."""
        score = 0.5
        
        # Check if full investigation capacity is used
        investigated = intel_data.get("investigate_players", [])
        if len(investigated) == 2:  # Maximum capacity
            score += 0.3
        
        # Check strategic misinformation use
        misinformation = intel_data.get("misinformation")
        if misinformation:
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_alliance_context_usage(self, alliance_data: Dict[str, Any]) -> float:
        """Evaluate how well alliance phase uses available context."""
        score = 0.5
        
        shared_info = alliance_data.get("shared_information", "")
        deception_strategy = alliance_data.get("deception_strategy")
        
        # Check information sharing
        if shared_info:
            score += 0.25
        
        # Check deception strategy detail
        if deception_strategy:
            score += 0.25
        
        return min(score, 1.0)
    
    def _evaluate_voting_context_usage(self, voting_data: Dict[str, Any]) -> float:
        """Evaluate how well voting phase uses available context."""
        score = 0.5
        
        public_reasoning = voting_data.get("public_reasoning", "")
        private_motivation = voting_data.get("private_motivation", "")
        
        # Check reasoning depth
        if public_reasoning and len(public_reasoning) > 20:
            score += 0.25
        
        if private_motivation and len(private_motivation) > 20:
            score += 0.25
        
        return min(score, 1.0)
    
    def _is_decision_error(self, phase: str, phase_data: Dict[str, Any]) -> bool:
        """Check if a decision contains obvious errors."""
        if phase == "intelligence":
            investigated = phase_data.get("investigate_players", [])
            # Error if investigating more than 2 players (game limit)
            return len(investigated) > 2
        
        elif phase == "alliance":
            alliance_type = phase_data.get("alliance_type", "")
            # Error if invalid alliance type
            return alliance_type not in ["true", "false", "temporary"]
        
        elif phase == "voting":
            target = phase_data.get("target", "")
            # Error if no voting target specified
            return not target
        
        return False  # No obvious errors detected
    
    def _calculate_strategic_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate strategic performance metrics."""
        strategic_metrics = {
            "alliance_success_rate": 0.0,
            "alliance_betrayal_rate": 0.0,
            "voting_accuracy": 0.0,
            "elimination_prediction_accuracy": 0.0,
            "strategic_coherence_score": 0.0,
            "information_warfare_effectiveness": 0.0
        }
        
        total_alliances = 0
        successful_alliances = 0
        betrayed_alliances = 0
        total_votes = 0
        accurate_votes = 0
        coherence_scores = []
        misinformation_instances = 0
        effective_misinformation = 0
        
        for game in game_logs:
            rounds = game.get("rounds", [])
            
            for round_data in rounds:
                # Alliance analysis
                alliance_data = round_data.get("alliance", {})
                for player, alliance_info in alliance_data.items():
                    if alliance_info:
                        total_alliances += 1
                        alliance_type = alliance_info.get("alliance_type", "")
                        if alliance_type == "true":
                            successful_alliances += 1
                        elif alliance_type == "false":
                            betrayed_alliances += 1
                
                # Voting analysis
                voting_data = round_data.get("voting", {})
                round_outcome = round_data.get("round_outcome", {})
                eliminated_player = round_outcome.get("eliminated_player")
                
                for player, vote_info in voting_data.items():
                    if vote_info:
                        total_votes += 1
                        voted_target = vote_info.get("target", "")
                        if voted_target == eliminated_player:
                            accurate_votes += 1
                
                # Intelligence/misinformation analysis
                intel_data = round_data.get("intelligence", {})
                for player, intel_info in intel_data.items():
                    if intel_info:
                        misinformation = intel_info.get("misinformation")
                        if misinformation:
                            misinformation_instances += 1
                            # Consider misinformation effective if it targets eventual elimination target
                            misinformation_target = intel_info.get("target_of_misinformation", "")
                            if misinformation_target == eliminated_player:
                                effective_misinformation += 1
                
                # Strategic coherence (simplified)
                for player in alliance_data.keys():
                    player_decisions = {
                        "intelligence": intel_data.get(player, {}),
                        "alliance": alliance_data.get(player, {}),
                        "voting": voting_data.get(player, {})
                    }
                    if any(player_decisions.values()):
                        coherence_score = self._calculate_round_coherence(player_decisions)
                        coherence_scores.append(coherence_score)
        
        # Calculate rates
        strategic_metrics["alliance_success_rate"] = successful_alliances / total_alliances if total_alliances > 0 else 0.0
        strategic_metrics["alliance_betrayal_rate"] = betrayed_alliances / total_alliances if total_alliances > 0 else 0.0
        strategic_metrics["voting_accuracy"] = accurate_votes / total_votes if total_votes > 0 else 0.0
        strategic_metrics["strategic_coherence_score"] = np.mean(coherence_scores) if coherence_scores else 0.0
        strategic_metrics["information_warfare_effectiveness"] = effective_misinformation / misinformation_instances if misinformation_instances > 0 else 0.0
        
        return strategic_metrics
    
    def _calculate_social_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate social dynamics and communication metrics."""
        social_metrics = {
            "trust_building_score": 0.0,
            "deception_usage_rate": 0.0,
            "alliance_formation_rate": 0.0,
            "communication_quality_score": 0.0,
            "pressure_tactics_effectiveness": 0.0,
            "information_sharing_propensity": 0.0
        }
        
        total_negotiations = 0
        high_deception_negotiations = 0
        alliance_formations = 0
        total_possible_alliances = 0
        communication_scores = []
        pressure_tactics_used = 0
        effective_pressure_tactics = 0
        information_sharing_instances = 0
        
        for game in game_logs:
            rounds = game.get("rounds", [])
            players = game.get("players", [])
            
            for round_data in rounds:
                # Negotiation analysis
                negotiation_data = round_data.get("negotiation", {})
                for player, negotiation_info in negotiation_data.items():
                    if negotiation_info:
                        total_negotiations += 1
                        
                        # Deception analysis
                        deception_level = negotiation_info.get("deception_level", 0)
                        if deception_level > 0.5:
                            high_deception_negotiations += 1
                        
                        # Communication quality
                        message = negotiation_info.get("message", "")
                        if message:
                            quality_score = self._assess_communication_quality(negotiation_info)
                            communication_scores.append(quality_score)
                        
                        # Pressure tactics
                        pressure_tactics = negotiation_info.get("pressure_tactics", [])
                        if pressure_tactics:
                            pressure_tactics_used += 1
                            # Consider effective if offer was made
                            offer_amount = negotiation_info.get("offer_amount", 0)
                            if offer_amount > 0:
                                effective_pressure_tactics += 1
                
                # Alliance formation analysis
                alliance_data = round_data.get("alliance", {})
                for player, alliance_info in alliance_data.items():
                    if alliance_info:
                        target_players = alliance_info.get("target_players", [])
                        if target_players:
                            alliance_formations += len(target_players)
                            
                        # Information sharing
                        shared_info = alliance_info.get("shared_information", "")
                        if shared_info:
                            information_sharing_instances += 1
                
                # Calculate possible alliances per round
                if players:
                    total_possible_alliances += len(players) * (len(players) - 1) // 2
        
        # Calculate metrics
        social_metrics["deception_usage_rate"] = high_deception_negotiations / total_negotiations if total_negotiations > 0 else 0.0
        social_metrics["alliance_formation_rate"] = alliance_formations / total_possible_alliances if total_possible_alliances > 0 else 0.0
        social_metrics["communication_quality_score"] = np.mean(communication_scores) if communication_scores else 0.0
        social_metrics["pressure_tactics_effectiveness"] = effective_pressure_tactics / pressure_tactics_used if pressure_tactics_used > 0 else 0.0
        social_metrics["information_sharing_propensity"] = information_sharing_instances / alliance_formations if alliance_formations > 0 else 0.0
        
        return social_metrics
    
    def _calculate_economic_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate wealth management and economic strategy metrics."""
        economic_metrics = {
            "wealth_management_score": 0.0,
            "investment_efficiency": 0.0,
            "wealth_volatility": 0.0,
            "final_wealth_distribution": {},
            "wealth_transfer_volume": 0.0,
            "economic_strategy_coherence": 0.0
        }
        
        all_wealth_changes = []
        total_investments = 0
        successful_investments = 0
        wealth_transfers = []
        player_wealth_histories = defaultdict(list)
        
        for game in game_logs:
            initial_setup = game.get("initial_setup", {})
            final_results = game.get("final_results", {})
            rounds = game.get("rounds", [])
            
            # Track wealth histories
            for player, setup in initial_setup.items():
                initial_wealth = setup.get("wealth", 0)
                player_wealth_histories[player].append(initial_wealth)
            
            # Analyze round-by-round changes
            for round_data in rounds:
                round_outcome = round_data.get("round_outcome", {})
                wealth_changes = round_outcome.get("wealth_changes", {})
                
                for player, change in wealth_changes.items():
                    all_wealth_changes.append(abs(change))
                    if change != 0:
                        # Update wealth history
                        prev_wealth = player_wealth_histories[player][-1] if player_wealth_histories[player] else 0
                        new_wealth = prev_wealth + change
                        player_wealth_histories[player].append(new_wealth)
                
                # Analyze investments (challenge participation)
                challenge_data = round_data.get("challenge", {})
                for player, challenge_info in challenge_data.items():
                    if challenge_info:
                        bid_amount = challenge_info.get("bid_amount", 0)
                        if bid_amount > 0:
                            total_investments += 1
                            # Consider successful if player gained wealth in this round
                            player_change = wealth_changes.get(player, 0)
                            if player_change > 0:
                                successful_investments += 1
                
                # Analyze wealth transfers (negotiations)
                negotiation_data = round_data.get("negotiation", {})
                for player, negotiation_info in negotiation_data.items():
                    if negotiation_info:
                        offer_amount = negotiation_info.get("offer_amount", 0)
                        if offer_amount > 0:
                            wealth_transfers.append(offer_amount)
            
            # Final wealth distribution
            final_wealth = final_results.get("final_wealth", {})
            for player, wealth in final_wealth.items():
                economic_metrics["final_wealth_distribution"][player] = wealth
        
        # Calculate metrics
        if all_wealth_changes:
            economic_metrics["wealth_management_score"] = 1.0 - (np.std(all_wealth_changes) / np.mean(all_wealth_changes)) if np.mean(all_wealth_changes) > 0 else 0.0
            economic_metrics["wealth_volatility"] = np.std(all_wealth_changes)
        
        economic_metrics["investment_efficiency"] = successful_investments / total_investments if total_investments > 0 else 0.0
        economic_metrics["wealth_transfer_volume"] = np.sum(wealth_transfers) if wealth_transfers else 0.0
        
        # Calculate wealth trajectory consistency per player
        trajectory_scores = []
        for player, wealth_history in player_wealth_histories.items():
            if len(wealth_history) > 1:
                # Simple trend consistency
                changes = [wealth_history[i+1] - wealth_history[i] for i in range(len(wealth_history)-1)]
                if changes:
                    consistency = 1.0 - (np.std(changes) / (np.mean(np.abs(changes)) + 1))
                    trajectory_scores.append(max(0, consistency))
        
        economic_metrics["economic_strategy_coherence"] = np.mean(trajectory_scores) if trajectory_scores else 0.0
        
        return economic_metrics
    
    def _calculate_game_outcome_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate game outcome and victory condition metrics."""
        outcome_metrics = {
            "average_game_length": 0.0,
            "elimination_patterns": {},
            "victory_types": {},
            "survival_rates": {},
            "player_performance_rankings": {}
        }
        
        game_lengths = []
        eliminations_by_round = defaultdict(int)
        victory_counts = defaultdict(int)
        player_survivals = defaultdict(list)
        player_final_positions = defaultdict(list)
        
        for game in game_logs:
            # Game length
            rounds = game.get("rounds", [])
            game_lengths.append(len(rounds))
            
            # Track eliminations
            for round_num, round_data in enumerate(rounds, 1):
                round_outcome = round_data.get("round_outcome", {})
                eliminated_player = round_outcome.get("eliminated_player")
                if eliminated_player:
                    eliminations_by_round[round_num] += 1
            
            # Victory analysis
            final_results = game.get("final_results", {})
            winner = final_results.get("winner")
            if winner:
                victory_counts[winner] += 1
            
            # Survival analysis
            remaining_players = final_results.get("remaining_players", [])
            eliminated_players = final_results.get("eliminated_players", [])
            
            for player in remaining_players:
                player_survivals[player].append(True)
            for player in eliminated_players:
                player_survivals[player].append(False)
            
            # Final rankings (based on final wealth)
            final_wealth = final_results.get("final_wealth", {})
            if final_wealth:
                ranked_players = sorted(final_wealth.items(), key=lambda x: x[1], reverse=True)
                for rank, (player, wealth) in enumerate(ranked_players, 1):
                    player_final_positions[player].append(rank)
        
        # Calculate metrics
        outcome_metrics["average_game_length"] = np.mean(game_lengths) if game_lengths else 0.0
        outcome_metrics["elimination_patterns"] = dict(eliminations_by_round)
        outcome_metrics["victory_types"] = dict(victory_counts)
        
        # Survival rates
        for player, survivals in player_survivals.items():
            outcome_metrics["survival_rates"][player] = np.mean(survivals) if survivals else 0.0
        
        # Performance rankings
        for player, positions in player_final_positions.items():
            outcome_metrics["player_performance_rankings"][player] = np.mean(positions) if positions else 0.0
        
        return outcome_metrics
    
    def _calculate_behavioral_analysis(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate behavioral patterns and psychological metrics."""
        behavioral_metrics = {
            "aggression_levels": {},
            "cooperation_tendencies": {},
            "risk_taking_patterns": {},
            "information_hoarding_vs_sharing": {},
            "trust_patterns": {},
            "adaptability_scores": {}
        }
        
        player_behaviors = defaultdict(lambda: {
            "aggressive_actions": 0,
            "cooperative_actions": 0,
            "high_risk_decisions": 0,
            "information_shared": 0,
            "information_hoarded": 0,
            "trust_given": 0,
            "trust_violated": 0,
            "strategy_changes": 0,
            "total_actions": 0
        })
        
        for game in game_logs:
            rounds = game.get("rounds", [])
            
            for round_data in rounds:
                # Analyze voting behavior (aggression)
                voting_data = round_data.get("voting", {})
                for player, vote_info in voting_data.items():
                    if vote_info:
                        player_behaviors[player]["total_actions"] += 1
                        
                        # Aggressive voting (targeting strong players)
                        private_motivation = vote_info.get("private_motivation", "")
                        if any(keyword in private_motivation.lower() for keyword in ["threat", "dominant", "powerful", "eliminate"]):
                            player_behaviors[player]["aggressive_actions"] += 1
                        
                        # Cooperative voting (alliance coordination)
                        alliance_coordination = vote_info.get("alliance_coordination", False)
                        if alliance_coordination:
                            player_behaviors[player]["cooperative_actions"] += 1
                
                # Analyze alliance behavior (cooperation/trust)
                alliance_data = round_data.get("alliance", {})
                for player, alliance_info in alliance_data.items():
                    if alliance_info:
                        player_behaviors[player]["total_actions"] += 1
                        
                        alliance_type = alliance_info.get("alliance_type", "")
                        shared_info = alliance_info.get("shared_information", "")
                        
                        if alliance_type == "true":
                            player_behaviors[player]["cooperative_actions"] += 1
                            player_behaviors[player]["trust_given"] += 1
                        elif alliance_type == "false":
                            player_behaviors[player]["trust_violated"] += 1
                        
                        # Information sharing vs hoarding
                        if shared_info:
                            player_behaviors[player]["information_shared"] += 1
                        else:
                            player_behaviors[player]["information_hoarded"] += 1
                
                # Analyze challenge behavior (risk taking)
                challenge_data = round_data.get("challenge", {})
                for player, challenge_info in challenge_data.items():
                    if challenge_info:
                        player_behaviors[player]["total_actions"] += 1
                        
                        bid_amount = challenge_info.get("bid_amount", 0)
                        decision = challenge_info.get("decision", "")
                        
                        # High risk decisions
                        if bid_amount > 20000 or "aggressive" in decision:
                            player_behaviors[player]["high_risk_decisions"] += 1
        
        # Calculate behavioral metrics
        for player, behaviors in player_behaviors.items():
            total = behaviors["total_actions"]
            if total > 0:
                behavioral_metrics["aggression_levels"][player] = behaviors["aggressive_actions"] / total
                behavioral_metrics["cooperation_tendencies"][player] = behaviors["cooperative_actions"] / total
                behavioral_metrics["risk_taking_patterns"][player] = behaviors["high_risk_decisions"] / total
                
                info_total = behaviors["information_shared"] + behaviors["information_hoarded"]
                if info_total > 0:
                    behavioral_metrics["information_hoarding_vs_sharing"][player] = behaviors["information_shared"] / info_total
                else:
                    behavioral_metrics["information_hoarding_vs_sharing"][player] = 0.5
                
                trust_total = behaviors["trust_given"] + behaviors["trust_violated"]
                if trust_total > 0:
                    behavioral_metrics["trust_patterns"][player] = behaviors["trust_given"] / trust_total
                else:
                    behavioral_metrics["trust_patterns"][player] = 0.5
        
        return behavioral_metrics
    
    def _calculate_llm_judge_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate LLM as judge evaluation metrics."""
        if not self.model:
            return {"error": "No LLM model provided for evaluation"}
        
        llm_metrics = {
            "strategic_performance": {},
            "social_performance": {},
            "overall_assessment": {},
            "improvement_recommendations": {}
        }
        
        # For now, return placeholder - would implement LLM evaluation calls here
        logger.info("LLM judge evaluation would be implemented here with actual model calls")
        
        return llm_metrics
    
    def _generate_detailed_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        if not self.metrics:
            return {"error": "No metrics calculated yet"}
        
        report = {
            "executive_summary": self._generate_executive_summary(),
            "detailed_analysis": self._generate_detailed_analysis(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of key findings."""
        model_perf = self.metrics.get("model_performance", {})
        strategic_metrics = self.metrics.get("strategic_metrics", {})
        
        summary = {
            "total_inferences": model_perf.get("total_inferences", 0),
            "average_response_quality": 0.0,
            "average_decision_consistency": 0.0,
            "strategic_effectiveness": strategic_metrics.get("alliance_success_rate", 0.0),
            "key_strengths": [],
            "key_weaknesses": []
        }
        
        # Calculate averages
        response_quality = model_perf.get("response_quality", {})
        if response_quality:
            summary["average_response_quality"] = np.mean(list(response_quality.values()))
        
        decision_consistency = model_perf.get("decision_consistency", {})
        if decision_consistency:
            summary["average_decision_consistency"] = np.mean(list(decision_consistency.values()))
        
        # Identify strengths and weaknesses
        if summary["average_response_quality"] > 0.7:
            summary["key_strengths"].append("High response quality")
        elif summary["average_response_quality"] < 0.4:
            summary["key_weaknesses"].append("Low response quality")
        
        if summary["strategic_effectiveness"] > 0.6:
            summary["key_strengths"].append("Effective strategic planning")
        elif summary["strategic_effectiveness"] < 0.3:
            summary["key_weaknesses"].append("Poor strategic execution")
        
        return summary
    
    def _generate_detailed_analysis(self) -> Dict[str, Any]:
        """Generate detailed analysis of all metrics."""
        return {
            "inference_analysis": self._analyze_inference_patterns(),
            "strategic_analysis": self._analyze_strategic_patterns(),
            "social_analysis": self._analyze_social_patterns(),
            "behavioral_analysis": self._analyze_behavioral_patterns()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations for improvement."""
        recommendations = []
        
        model_perf = self.metrics.get("model_performance", {})
        response_quality = model_perf.get("response_quality", {})
        
        if response_quality:
            avg_quality = np.mean(list(response_quality.values()))
            if avg_quality < 0.5:
                recommendations.append("Improve response quality by providing more detailed strategic reasoning")
        
        strategic_metrics = self.metrics.get("strategic_metrics", {})
        alliance_success = strategic_metrics.get("alliance_success_rate", 0)
        
        if alliance_success < 0.4:
            recommendations.append("Focus on building more genuine alliances rather than deceptive ones")
        
        voting_accuracy = strategic_metrics.get("voting_accuracy", 0)
        if voting_accuracy < 0.5:
            recommendations.append("Improve elimination prediction by better analyzing player threats")
        
        return recommendations
    
    # Helper methods for analysis
    def _calculate_round_coherence(self, player_decisions: Dict[str, Any]) -> float:
        """Calculate coherence score for a player's decisions in a round."""
        score = 0.5
        
        intelligence = player_decisions.get("intelligence", {})
        alliance = player_decisions.get("alliance", {})
        voting = player_decisions.get("voting", {})
        
        # Check intel-alliance coherence
        if intelligence and alliance:
            score += 0.2 * self._evaluate_intel_alliance_coherence(intelligence, alliance)
        
        # Check alliance-voting coherence
        if alliance and voting:
            score += 0.3 * self._evaluate_alliance_voting_coherence(alliance, voting)
        
        return min(score, 1.0)
    
    def _assess_communication_quality(self, negotiation_info: Dict[str, Any]) -> float:
        """Assess quality of communication in negotiation."""
        score = 0.5
        
        message = negotiation_info.get("message", "")
        if len(message) > 50:  # Substantial message
            score += 0.2
        
        info_to_extract = negotiation_info.get("information_to_extract", [])
        if info_to_extract:
            score += 0.15
        
        pressure_tactics = negotiation_info.get("pressure_tactics", [])
        if pressure_tactics:
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_inference_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in model inference behavior."""
        model_perf = self.metrics.get("model_performance", {})
        
        return {
            "inference_distribution": {
                "intelligence": model_perf.get("intelligence_inferences", 0),
                "alliance": model_perf.get("alliance_inferences", 0),
                "challenge": model_perf.get("challenge_inferences", 0),
                "negotiation": model_perf.get("negotiation_inferences", 0),
                "voting": model_perf.get("voting_inferences", 0)
            },
            "quality_analysis": model_perf.get("response_quality", {}),
            "consistency_analysis": model_perf.get("decision_consistency", {}),
            "error_analysis": model_perf.get("error_rate", {})
        }
    
    def _analyze_strategic_patterns(self) -> Dict[str, Any]:
        """Analyze strategic decision patterns."""
        return self.metrics.get("strategic_metrics", {})
    
    def _analyze_social_patterns(self) -> Dict[str, Any]:
        """Analyze social interaction patterns."""
        return self.metrics.get("social_metrics", {})
    
    def _analyze_behavioral_patterns(self) -> Dict[str, Any]:
        """Analyze behavioral patterns."""
        return self.metrics.get("behavioral_analysis", {})
    
    def generate_report(self, metrics_data: Dict[str, Any], format_type: str = "markdown") -> str:
        """
        Generate comprehensive analysis report in specified format.
        
        Args:
            metrics_data: Calculated metrics data
            format_type: Format type ("markdown", "json", "txt")
        
        Returns:
            str: Formatted report
        """
        if format_type == "markdown":
            return self._generate_markdown_report(metrics_data)
        elif format_type == "json":
            return json.dumps(metrics_data, indent=2)
        else:
            return self._generate_text_report(metrics_data)
    
    def _generate_markdown_report(self, metrics_data: Dict[str, Any]) -> str:
        """Generate markdown format report."""
        report = f"""# Beast Game Analysis Report

Generated: {metrics_data.get('timestamp', 'Unknown')}
Games Analyzed: {metrics_data.get('games_total', 0)}

## Executive Summary

"""
        
        executive_summary = metrics_data.get('detailed_report', {}).get('executive_summary', {})
        model_perf = metrics_data.get('model_performance', {})
        strategic_metrics = metrics_data.get('strategic_metrics', {})
        
        report += f"""### Key Performance Indicators
- **Total Inferences**: {executive_summary.get('total_inferences', 0)}
- **Average Response Quality**: {executive_summary.get('average_response_quality', 0):.3f}
- **Alliance Success Rate**: {strategic_metrics.get('alliance_success_rate', 0):.3f}
- **Voting Accuracy**: {strategic_metrics.get('voting_accuracy', 0):.3f}

## Model Inference Performance

### Overall Statistics
- **Intelligence Inferences**: {model_perf.get('intelligence_inferences', 0)}
- **Alliance Inferences**: {model_perf.get('alliance_inferences', 0)}
- **Challenge Inferences**: {model_perf.get('challenge_inferences', 0)}
- **Negotiation Inferences**: {model_perf.get('negotiation_inferences', 0)}
- **Voting Inferences**: {model_perf.get('voting_inferences', 0)}
- **Total Errors**: {model_perf.get('total_errors', 0)}

## Strategic Analysis

### Game Outcomes
- **Average Game Length**: {strategic_metrics.get('average_game_length', 0):.1f} rounds
- **Information Warfare Effectiveness**: {strategic_metrics.get('information_warfare_effectiveness', 0):.3f}
- **Strategic Coherence Score**: {strategic_metrics.get('strategic_coherence_score', 0):.3f}

## Player Performance Analysis

### Response Quality by Player
"""
        
        response_quality = model_perf.get('response_quality', {})
        for player, quality in sorted(response_quality.items(), key=lambda x: x[1], reverse=True):
            report += f"- **{player}**: {quality:.3f}\n"
        
        recommendations = metrics_data.get('detailed_report', {}).get('recommendations', [])
        if recommendations:
            report += """
## Recommendations for Improvement

"""
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
        
        report += f"""
---
*Report generated by PolitAgent Beast Metrics v2.0*
"""
        
        return report
    
    def _generate_text_report(self, metrics_data: Dict[str, Any]) -> str:
        """Generate plain text format report."""
        return f"Beast Analysis Report - {metrics_data.get('games_total', 0)} games analyzed"
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """
        Recursively convert numpy types to JSON-serializable Python types.
        
        Args:
            obj: Object that may contain numpy types
            
        Returns:
            Object with numpy types converted to standard Python types
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj 