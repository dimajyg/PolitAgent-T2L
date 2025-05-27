from typing import Dict, List, Tuple, Union, Any, Set, Optional
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import requests
import torch
import os
from datetime import datetime
import json
import logging
import time
from collections import defaultdict, Counter
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

from metrics.base_metrics import BaseMetrics

logger = logging.getLogger(__name__)

class SpyfallMetrics(BaseMetrics):
    """
    Comprehensive metrics for evaluating model performance in Spyfall deduction game.
    Includes inference metrics, strategic evaluation, and detailed reporting.
    """
    
    def __init__(self, model: Optional[BaseLanguageModel] = None):
        """
        Initialize Spyfall metrics with optional LLM evaluator.
        
        Args:
            model: LLM model for evaluation (LLM as judge)
        """
        super().__init__(game_type="spyfall")
        self.model = model
        self.metrics = {}
        self.inference_data = []  # Store detailed inference metrics
        self.strategic_analysis = {}  # Store strategic patterns
        self.game_state = {}  # Track current game state
    
    def compute_all(self) -> Dict[str, Any]:
        """
        Implementation of abstract method from BaseMetrics.
        Computes all metrics from recorded events.
        
        Returns:
            Dict[str, Any]: Complete metrics suite
        """
        # For Spyfall metrics, we don't use the event-based approach
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
            "deception_metrics": self._calculate_deception_metrics(game_logs),
            "communication_metrics": self._calculate_communication_metrics(game_logs),
            "voting_metrics": self._calculate_voting_metrics(game_logs),
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
                if file.endswith('.json') and 'spyfall' in file:
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
            "description_inferences": 0,
            "voting_inferences": 0,
            "response_quality": {},
            "strategic_coherence": {},
            "role_consistency": {},
            "error_rate": {},
            "total_errors": 0,
            "deception_effectiveness": {},
            "information_extraction": {}
        }
        
        players = set()
        spy_players = set()
        villager_players = set()
        
        for game in game_logs:
            # Extract game setup
            game_setup = game.get("game_setup", {})
            spy_name = game_setup.get("spy_name", "")
            spy_word = game_setup.get("spy_word", "")
            villager_word = game_setup.get("villager_word", "")
            
            if spy_name:
                spy_players.add(spy_name)
            
            # Extract players from game data
            if "players" in game_setup:
                all_players = game_setup["players"]
                players.update(all_players)
                for player in all_players:
                    if player != spy_name:
                        villager_players.add(player)
            
            # Count inferences from game rounds
            rounds = game.get("rounds", [])
            for round_data in rounds:
                # Description phase inferences
                descriptions = round_data.get("descriptions", {})
                inference_metrics["description_inferences"] += len(descriptions)
                
                # Voting phase inferences
                votes = round_data.get("votes", {})
                inference_metrics["voting_inferences"] += len(votes)
                
                # Extract players from round data
                players.update(descriptions.keys())
                players.update(votes.keys())
        
        # Initialize player-specific metrics
        for player in players:
            inference_metrics["response_quality"][player] = 0.0
            inference_metrics["strategic_coherence"][player] = 0.0
            inference_metrics["role_consistency"][player] = 0.0
            inference_metrics["error_rate"][player] = 0.0
            inference_metrics["deception_effectiveness"][player] = 0.0
            inference_metrics["information_extraction"][player] = 0.0
        
        # Analyze player performance across games
        for game in game_logs:
            game_setup = game.get("game_setup", {})
            spy_name = game_setup.get("spy_name", "")
            
            for player in players:
                player_decisions = self._extract_player_decisions_spyfall(player, game)
                
                if player_decisions:
                    # Quality analysis
                    quality_score = self._analyze_decision_quality_spyfall(player_decisions, player == spy_name)
                    inference_metrics["response_quality"][player] = quality_score
                    
                    # Strategic coherence
                    coherence_score = self._analyze_strategic_coherence_spyfall(player_decisions, player == spy_name)
                    inference_metrics["strategic_coherence"][player] = coherence_score
                    
                    # Role consistency
                    consistency_score = self._analyze_role_consistency_spyfall(player_decisions, player == spy_name)
                    inference_metrics["role_consistency"][player] = consistency_score
                    
                    # Deception effectiveness (mainly for spies)
                    if player == spy_name:
                        deception_score = self._analyze_deception_effectiveness_spyfall(player_decisions, game)
                        inference_metrics["deception_effectiveness"][player] = deception_score
                    
                    # Information extraction effectiveness (mainly for villagers)
                    if player != spy_name:
                        extraction_score = self._analyze_information_extraction_spyfall(player_decisions, game)
                        inference_metrics["information_extraction"][player] = extraction_score
                    
                    # Error rate analysis
                    error_rate = self._calculate_error_rate_spyfall(player_decisions)
                    inference_metrics["error_rate"][player] = error_rate
                    inference_metrics["total_errors"] += int(error_rate * len(player_decisions["descriptions"]) + len(player_decisions["votes"]))
        
        # Calculate totals
        inference_metrics["total_inferences"] = (
            inference_metrics["description_inferences"] + 
            inference_metrics["voting_inferences"]
        )
        
        # Add role-specific analysis
        inference_metrics["spy_performance"] = {
            "average_quality": np.mean([inference_metrics["response_quality"][spy] for spy in spy_players]) if spy_players else 0.0,
            "average_deception": np.mean([inference_metrics["deception_effectiveness"][spy] for spy in spy_players]) if spy_players else 0.0
        }
        
        inference_metrics["villager_performance"] = {
            "average_quality": np.mean([inference_metrics["response_quality"][villager] for villager in villager_players]) if villager_players else 0.0,
            "average_extraction": np.mean([inference_metrics["information_extraction"][villager] for villager in villager_players]) if villager_players else 0.0
        }
        
        return inference_metrics
    
    def _extract_player_decisions_spyfall(self, player: str, game: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all decisions made by a player across the game."""
        decisions = {
            "descriptions": [],
            "votes": [],
            "game_context": game.get("game_setup", {})
        }
        
        rounds = game.get("rounds", [])
        for round_num, round_data in enumerate(rounds):
            # Extract descriptions
            descriptions = round_data.get("descriptions", {})
            if player in descriptions:
                decisions["descriptions"].append({
                    "round": round_num + 1,
                    "content": descriptions[player],
                    "context": round_data
                })
            
            # Extract votes
            votes = round_data.get("votes", {})
            if player in votes:
                decisions["votes"].append({
                    "round": round_num + 1,
                    "target": votes[player],
                    "context": round_data
                })
        
        return decisions
    
    def _analyze_decision_quality_spyfall(self, player_decisions: Dict[str, Any], is_spy: bool) -> float:
        """Analyze the quality of a player's decisions."""
        quality_scores = []
        
        # Analyze description quality
        for desc_data in player_decisions["descriptions"]:
            desc_quality = self._evaluate_description_quality(desc_data, is_spy)
            quality_scores.append(desc_quality)
        
        # Analyze voting quality
        for vote_data in player_decisions["votes"]:
            vote_quality = self._evaluate_voting_quality(vote_data, is_spy)
            quality_scores.append(vote_quality)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _analyze_strategic_coherence_spyfall(self, player_decisions: Dict[str, Any], is_spy: bool) -> float:
        """Analyze strategic coherence across descriptions and votes."""
        if len(player_decisions["descriptions"]) < 2:
            return 1.0  # Can't measure coherence with less than 2 decisions
        
        coherence_scores = []
        
        # Analyze description coherence
        descriptions = player_decisions["descriptions"]
        for i in range(1, len(descriptions)):
            prev_desc = descriptions[i-1]
            curr_desc = descriptions[i]
            coherence = self._evaluate_description_coherence(prev_desc, curr_desc, is_spy)
            coherence_scores.append(coherence)
        
        # Analyze vote-description alignment
        if player_decisions["votes"] and player_decisions["descriptions"]:
            vote_desc_alignment = self._evaluate_vote_description_alignment(
                player_decisions["descriptions"], 
                player_decisions["votes"], 
                is_spy
            )
            coherence_scores.append(vote_desc_alignment)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _analyze_role_consistency_spyfall(self, player_decisions: Dict[str, Any], is_spy: bool) -> float:
        """Analyze how consistently player acts according to their role."""
        consistency_scores = []
        
        # Check if descriptions are appropriate for role
        for desc_data in player_decisions["descriptions"]:
            consistency = self._evaluate_role_consistency(desc_data, is_spy)
            consistency_scores.append(consistency)
        
        # Check if voting behavior is appropriate for role
        for vote_data in player_decisions["votes"]:
            consistency = self._evaluate_vote_role_consistency(vote_data, is_spy)
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _analyze_deception_effectiveness_spyfall(self, player_decisions: Dict[str, Any], game: Dict[str, Any]) -> float:
        """Analyze how effectively a spy deceives villagers."""
        if not player_decisions["descriptions"]:
            return 0.0
        
        deception_scores = []
        
        # Analyze how well spy descriptions blend in
        for desc_data in player_decisions["descriptions"]:
            deception_score = self._evaluate_spy_deception(desc_data, game)
            deception_scores.append(deception_score)
        
        # Check if spy was detected (outcome-based measure)
        game_result = game.get("game_result", {})
        if game_result.get("winner") == "spy":
            outcome_bonus = 0.3  # Bonus for successful deception
        else:
            outcome_bonus = 0.0
        
        base_score = np.mean(deception_scores) if deception_scores else 0.0
        return min(1.0, base_score + outcome_bonus)
    
    def _analyze_information_extraction_spyfall(self, player_decisions: Dict[str, Any], game: Dict[str, Any]) -> float:
        """Analyze how effectively a villager extracts information about the spy."""
        if not player_decisions["votes"]:
            return 0.0
        
        extraction_scores = []
        
        # Check voting accuracy
        spy_name = game.get("game_setup", {}).get("spy_name", "")
        for vote_data in player_decisions["votes"]:
            if vote_data["target"] == spy_name:
                extraction_scores.append(1.0)  # Correctly identified spy
            else:
                extraction_scores.append(0.0)  # Incorrect identification
        
        # Check if villagers won (outcome-based measure)
        game_result = game.get("game_result", {})
        if game_result.get("winner") == "villager":
            outcome_bonus = 0.2  # Bonus for successful spy detection
        else:
            outcome_bonus = 0.0
        
        base_score = np.mean(extraction_scores) if extraction_scores else 0.0
        return min(1.0, base_score + outcome_bonus)
    
    def _calculate_error_rate_spyfall(self, player_decisions: Dict[str, Any]) -> float:
        """Calculate error rate in player decisions."""
        total_decisions = len(player_decisions["descriptions"]) + len(player_decisions["votes"])
        total_errors = 0
        
        # Check for description errors
        for desc_data in player_decisions["descriptions"]:
            if self._is_description_error(desc_data):
                total_errors += 1
        
        # Check for voting errors
        for vote_data in player_decisions["votes"]:
            if self._is_voting_error(vote_data):
                total_errors += 1
        
        return total_errors / total_decisions if total_decisions > 0 else 0.0
    
    # Evaluation helper methods
    def _evaluate_description_quality(self, desc_data: Dict[str, Any], is_spy: bool) -> float:
        """Evaluate quality of a description."""
        content = desc_data.get("content", "")
        if not content:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length check (not too short, not too long)
        if 10 <= len(content) <= 200:
            score += 0.2
        
        # Specificity vs vagueness balance
        if is_spy:
            # Spy should be vague but not obviously so
            if self._is_appropriately_vague(content):
                score += 0.2
            if not self._contains_obvious_tells(content):
                score += 0.1
        else:
            # Villager should be specific but not too revealing
            if self._is_appropriately_specific(content):
                score += 0.2
            if not self._is_too_revealing(content):
                score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_voting_quality(self, vote_data: Dict[str, Any], is_spy: bool) -> float:
        """Evaluate quality of a voting decision."""
        target = vote_data.get("target", "")
        context = vote_data.get("context", {})
        
        if not target:
            return 0.0
        
        score = 0.5  # Base score
        
        # Check if vote target is valid
        living_players = context.get("living_players", [])
        if target in living_players:
            score += 0.2
        
        # Role-specific voting strategy
        if is_spy:
            # Spy should vote for villagers
            spy_name = vote_data.get("context", {}).get("spy_name", "")
            if target != spy_name:  # Not voting for self
                score += 0.3
        else:
            # Villager should try to vote for spy
            # This is harder to evaluate without knowing the outcome
            score += 0.3  # Assume reasonable attempt
        
        return min(score, 1.0)
    
    def _evaluate_description_coherence(self, prev_desc: Dict[str, Any], curr_desc: Dict[str, Any], is_spy: bool) -> float:
        """Evaluate coherence between consecutive descriptions."""
        prev_content = prev_desc.get("content", "")
        curr_content = curr_desc.get("content", "")
        
        if not prev_content or not curr_content:
            return 0.5
        
        # Simple keyword overlap analysis
        prev_words = set(prev_content.lower().split())
        curr_words = set(curr_content.lower().split())
        
        overlap = len(prev_words.intersection(curr_words))
        total_unique = len(prev_words.union(curr_words))
        
        if total_unique == 0:
            return 0.5
        
        # Coherence should be moderate - too much overlap is suspicious, too little is inconsistent
        overlap_ratio = overlap / total_unique
        if 0.2 <= overlap_ratio <= 0.6:
            return 0.8
        elif 0.1 <= overlap_ratio <= 0.8:
            return 0.6
        else:
            return 0.3
    
    def _evaluate_vote_description_alignment(self, descriptions: List[Dict[str, Any]], votes: List[Dict[str, Any]], is_spy: bool) -> float:
        """Evaluate alignment between descriptions and voting behavior."""
        if not descriptions or not votes:
            return 0.5
        
        # For simplicity, assume alignment if consistent strategy is maintained
        # This could be enhanced with more sophisticated analysis
        return 0.7  # Reasonable alignment assumed
    
    def _evaluate_role_consistency(self, desc_data: Dict[str, Any], is_spy: bool) -> float:
        """Evaluate if description is consistent with role."""
        content = desc_data.get("content", "")
        
        if is_spy:
            # Spy should avoid being too specific
            if self._is_appropriately_vague(content) and not self._contains_obvious_tells(content):
                return 0.8
            else:
                return 0.4
        else:
            # Villager should be reasonably specific
            if self._is_appropriately_specific(content) and not self._is_too_revealing(content):
                return 0.8
            else:
                return 0.4
    
    def _evaluate_vote_role_consistency(self, vote_data: Dict[str, Any], is_spy: bool) -> float:
        """Evaluate if voting behavior is consistent with role."""
        # Simplified evaluation - could be enhanced
        return 0.7
    
    def _evaluate_spy_deception(self, desc_data: Dict[str, Any], game: Dict[str, Any]) -> float:
        """Evaluate how well spy deceives with their description."""
        content = desc_data.get("content", "")
        villager_word = game.get("game_setup", {}).get("villager_word", "")
        
        score = 0.5
        
        # Spy should sound like they know the word without actually knowing it
        if self._sounds_knowledgeable(content, villager_word):
            score += 0.3
        
        if self._is_appropriately_vague(content):
            score += 0.2
        
        return min(score, 1.0)
    
    # Helper methods for content analysis
    def _is_appropriately_vague(self, content: str) -> bool:
        """Check if content is appropriately vague."""
        vague_indicators = ["something", "thing", "stuff", "kind of", "sort of", "maybe", "perhaps"]
        return any(indicator in content.lower() for indicator in vague_indicators)
    
    def _is_appropriately_specific(self, content: str) -> bool:
        """Check if content is appropriately specific."""
        # Not too vague, but not too specific
        return len(content.split()) >= 5 and not self._is_too_revealing(content)
    
    def _contains_obvious_tells(self, content: str) -> bool:
        """Check if content contains obvious spy tells."""
        tell_phrases = ["don't know", "not sure", "no idea", "never seen", "what is"]
        return any(phrase in content.lower() for phrase in tell_phrases)
    
    def _is_too_revealing(self, content: str) -> bool:
        """Check if content is too revealing of the actual word."""
        # This would need game-specific word analysis
        return len(content.split()) > 50  # Too long descriptions are suspicious
    
    def _sounds_knowledgeable(self, content: str, villager_word: str) -> bool:
        """Check if spy sounds like they know the villager word."""
        # Simple heuristic - could be enhanced with semantic analysis
        return len(content.split()) >= 3 and not self._contains_obvious_tells(content)
    
    def _is_description_error(self, desc_data: Dict[str, Any]) -> bool:
        """Check if description contains obvious errors."""
        content = desc_data.get("content", "")
        return not content or len(content.strip()) < 3
    
    def _is_voting_error(self, vote_data: Dict[str, Any]) -> bool:
        """Check if voting decision contains obvious errors."""
        target = vote_data.get("target", "")
        context = vote_data.get("context", {})
        living_players = context.get("living_players", [])
        
        return not target or (living_players and target not in living_players)
    
    def _calculate_strategic_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate strategic performance metrics."""
        strategic_metrics = {
            "spy_win_rate": 0.0,
            "villager_win_rate": 0.0,
            "average_game_length": 0.0,
            "spy_detection_accuracy": 0.0,
            "deception_success_rate": 0.0,
            "information_gathering_effectiveness": 0.0
        }
        
        total_games = len(game_logs)
        spy_wins = 0
        villager_wins = 0
        game_lengths = []
        correct_spy_votes = 0
        total_votes = 0
        successful_deceptions = 0
        total_spy_games = 0
        
        for game in game_logs:
            # Game outcome analysis
            game_result = game.get("game_result", {})
            winner = game_result.get("winner", "")
            
            if winner == "spy":
                spy_wins += 1
                successful_deceptions += 1
            elif winner == "villager":
                villager_wins += 1
            
            # Game length
            rounds = game.get("rounds", [])
            game_lengths.append(len(rounds))
            
            # Voting accuracy analysis
            spy_name = game.get("game_setup", {}).get("spy_name", "")
            if spy_name:
                total_spy_games += 1
                
                for round_data in rounds:
                    votes = round_data.get("votes", {})
                    for voter, target in votes.items():
                        total_votes += 1
                        if target == spy_name:
                            correct_spy_votes += 1
        
        # Calculate metrics
        if total_games > 0:
            strategic_metrics["spy_win_rate"] = spy_wins / total_games
            strategic_metrics["villager_win_rate"] = villager_wins / total_games
            strategic_metrics["average_game_length"] = np.mean(game_lengths) if game_lengths else 0.0
        
        if total_votes > 0:
            strategic_metrics["spy_detection_accuracy"] = correct_spy_votes / total_votes
        
        if total_spy_games > 0:
            strategic_metrics["deception_success_rate"] = successful_deceptions / total_spy_games
        
        # Information gathering effectiveness (simplified)
        strategic_metrics["information_gathering_effectiveness"] = strategic_metrics["spy_detection_accuracy"]
        
        return strategic_metrics
    
    def _calculate_deception_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate deception and detection metrics."""
        return {
            "spy_blending_effectiveness": 0.0,
            "villager_suspicion_accuracy": 0.0,
            "deception_complexity": 0.0,
            "tell_frequency": 0.0
        }
    
    def _calculate_communication_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate communication quality metrics."""
        return {
            "description_clarity": 0.0,
            "information_content": 0.0,
            "strategic_messaging": 0.0,
            "communication_consistency": 0.0
        }
    
    def _calculate_voting_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate voting behavior metrics."""
        return {
            "voting_accuracy": 0.0,
            "voting_confidence": 0.0,
            "strategic_voting": 0.0,
            "consensus_building": 0.0
        }
    
    def _calculate_game_outcome_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate game outcome metrics."""
        return {
            "win_rate_by_role": {},
            "elimination_patterns": {},
            "game_duration_analysis": {},
            "performance_rankings": {}
        }
    
    def _calculate_behavioral_analysis(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate behavioral patterns."""
        return {
            "aggression_levels": {},
            "deception_patterns": {},
            "cooperation_indicators": {},
            "risk_assessment": {}
        }
    
    def _calculate_llm_judge_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate LLM as judge evaluation metrics."""
        if not self.model:
            return {"error": "No LLM model provided for evaluation"}
        
        logger.info("LLM judge evaluation would be implemented here with actual model calls")
        return {}
    
    def _generate_detailed_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        if not self.metrics:
            return {"error": "No metrics calculated yet"}
        
        return {
            "executive_summary": self._generate_executive_summary(),
            "detailed_analysis": self._generate_detailed_analysis(),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary."""
        model_perf = self.metrics.get("model_performance", {})
        strategic_metrics = self.metrics.get("strategic_metrics", {})
        
        return {
            "total_inferences": model_perf.get("total_inferences", 0),
            "average_response_quality": np.mean(list(model_perf.get("response_quality", {}).values())) if model_perf.get("response_quality") else 0.0,
            "spy_win_rate": strategic_metrics.get("spy_win_rate", 0.0),
            "villager_win_rate": strategic_metrics.get("villager_win_rate", 0.0),
            "key_strengths": [],
            "key_weaknesses": []
        }
    
    def _generate_detailed_analysis(self) -> Dict[str, Any]:
        """Generate detailed analysis."""
        return {
            "inference_patterns": self.metrics.get("model_performance", {}),
            "strategic_patterns": self.metrics.get("strategic_metrics", {}),
            "deception_analysis": self.metrics.get("deception_metrics", {}),
            "communication_analysis": self.metrics.get("communication_metrics", {})
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations."""
        return [
            "Improve deception techniques for spy role",
            "Enhance information gathering for villager role",
            "Balance specificity in descriptions"
        ]
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """Recursively convert numpy types to JSON-serializable types."""
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
    
    # Additional methods for compatibility and report generation
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
        report = f"""# Spyfall Game Analysis Report

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
- **Spy Win Rate**: {executive_summary.get('spy_win_rate', 0):.3f}
- **Villager Win Rate**: {executive_summary.get('villager_win_rate', 0):.3f}

## Model Inference Performance

### Overall Statistics
- **Description Inferences**: {model_perf.get('description_inferences', 0)}
- **Voting Inferences**: {model_perf.get('voting_inferences', 0)}
- **Total Errors**: {model_perf.get('total_errors', 0)}

### Role-Specific Performance
"""
        
        spy_perf = model_perf.get('spy_performance', {})
        villager_perf = model_perf.get('villager_performance', {})
        
        if spy_perf:
            report += f"""
#### Spy Performance
- **Average Quality**: {spy_perf.get('average_quality', 0):.3f}
- **Average Deception**: {spy_perf.get('average_deception', 0):.3f}
"""
        
        if villager_perf:
            report += f"""
#### Villager Performance
- **Average Quality**: {villager_perf.get('average_quality', 0):.3f}
- **Average Information Extraction**: {villager_perf.get('average_extraction', 0):.3f}
"""
        
        report += f"""
## Strategic Analysis

### Game Outcomes
- **Average Game Length**: {strategic_metrics.get('average_game_length', 0):.1f} rounds
- **Spy Detection Accuracy**: {strategic_metrics.get('spy_detection_accuracy', 0):.3f}
- **Deception Success Rate**: {strategic_metrics.get('deception_success_rate', 0):.3f}

## Player Performance Analysis

### Response Quality by Player
"""
        
        response_quality = model_perf.get('response_quality', {})
        for player, quality in sorted(response_quality.items(), key=lambda x: x[1], reverse=True):
            report += f"- **{player}**: {quality:.3f}\n"
        
        report += """
### Strategic Coherence by Player
"""
        
        strategic_coherence = model_perf.get('strategic_coherence', {})
        for player, coherence in sorted(strategic_coherence.items(), key=lambda x: x[1], reverse=True):
            report += f"- **{player}**: {coherence:.3f}\n"
        
        recommendations = metrics_data.get('detailed_report', {}).get('recommendations', [])
        if recommendations:
            report += """
## Recommendations for Improvement

"""
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
        
        report += f"""
---
*Report generated by PolitAgent Spyfall Metrics v2.0*
"""
        
        return report
    
    def _generate_text_report(self, metrics_data: Dict[str, Any]) -> str:
        """Generate plain text format report."""
        return f"Spyfall Analysis Report - {metrics_data.get('games_total', 0)} games analyzed" 