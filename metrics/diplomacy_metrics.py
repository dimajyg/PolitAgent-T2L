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

class DiplomacyMetrics(BaseMetrics):
    """
    Comprehensive metrics for evaluating model performance in Diplomacy game.
    Includes inference metrics, strategic evaluation, and detailed reporting.
    """
    
    def __init__(self, model: Optional[BaseLanguageModel] = None):
        """
        Initialize Diplomacy metrics with optional LLM evaluator.
        
        Args:
            model: LLM model for evaluation (LLM as judge)
        """
        super().__init__(game_type="diplomacy")
        self.model = model
        self.metrics = {}
        self.powers = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
        self.inference_data = []  # Store detailed inference metrics
        self.strategic_analysis = {}  # Store strategic patterns
        
    def compute_all(self) -> Dict[str, Any]:
        """
        Implementation of abstract method from BaseMetrics.
        Computes all metrics from recorded events.
        
        Returns:
            Dict[str, Any]: Complete metrics suite
        """
        # For Diplomacy metrics, we don't use the event-based approach
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
            "tactical_metrics": self._calculate_tactical_metrics(game_logs),
            "diplomatic_metrics": self._calculate_diplomatic_metrics(game_logs),
            "game_outcome_metrics": self._calculate_game_outcome_metrics(game_logs),
            "behavioral_analysis": self._calculate_behavioral_analysis(game_logs)
        }
        
        # LLM as judge evaluation
        if self.model:
            self.metrics["llm_evaluation"] = self._calculate_llm_judge_metrics(game_logs)
        
        # Generate comprehensive report
        self.metrics["detailed_report"] = self._generate_detailed_report()
        
        # Store in computed_metrics for base class compatibility
        self.computed_metrics = self.metrics
        
        return self.metrics
    
    def _load_game_logs(self, results_dir: str) -> List[Dict[str, Any]]:
        """
        Загрузка логов игр из директории результатов.
        
        Args:
            results_dir: Директория с результатами игр
            
        Returns:
            List[Dict[str, Any]]: Список логов игр
        """
        game_logs = []
        for root, _, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.json') and 'diplomacy' in file:
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
            "negotiation_inferences": 0,
            "action_inferences": 0,
            "strategic_inferences": 0,
            "response_quality": {power: [] for power in self.powers},
            "decision_consistency": {power: [] for power in self.powers},
            "context_utilization": {power: [] for power in self.powers},
            "error_rate": {power: 0 for power in self.powers},
            "total_errors": 0
        }
        
        for game in game_logs:
            rounds_data = game.get("rounds_data", [])
            
            for round_data in rounds_data:
                # Count negotiation inferences
                negotiations = round_data.get("negotiations", {})
                for power, negotiation_partners in negotiations.items():
                    for partner, messages in negotiation_partners.items():
                        if isinstance(messages, dict):
                            inference_metrics["negotiation_inferences"] += len(messages)
                        else:
                            inference_metrics["negotiation_inferences"] += 1
                
                # Count action inferences
                orders = round_data.get("orders", {})
                for power, power_orders in orders.items():
                    if power_orders:  # If agent provided orders
                        inference_metrics["action_inferences"] += 1
                        
                        # Analyze order quality
                        quality_score = self._analyze_order_quality(power_orders, round_data)
                        inference_metrics["response_quality"][power].append(quality_score)
                        
                        # Check for errors in orders
                        if self._check_order_errors(power_orders):
                            inference_metrics["error_rate"][power] += 1
                            inference_metrics["total_errors"] += 1
            
            # Count strategic decision inferences
            strategic_decisions = game.get("strategic_decisions", {})
            inference_metrics["strategic_inferences"] += len(strategic_decisions)
            
            # Analyze decision consistency across rounds
            for power in self.powers:
                consistency_score = self._analyze_decision_consistency(power, rounds_data)
                if consistency_score is not None:
                    inference_metrics["decision_consistency"][power].append(consistency_score)
                
                # Analyze context utilization
                context_score = self._analyze_context_utilization(power, rounds_data)
                if context_score is not None:
                    inference_metrics["context_utilization"][power].append(context_score)
        
        # Calculate total inferences
        inference_metrics["total_inferences"] = (
            inference_metrics["negotiation_inferences"] + 
            inference_metrics["action_inferences"] + 
            inference_metrics["strategic_inferences"]
        )
        
        # Calculate averages
        for power in self.powers:
            if inference_metrics["response_quality"][power]:
                inference_metrics["response_quality"][power] = np.mean(inference_metrics["response_quality"][power])
            else:
                inference_metrics["response_quality"][power] = 0.0
                
            if inference_metrics["decision_consistency"][power]:
                inference_metrics["decision_consistency"][power] = np.mean(inference_metrics["decision_consistency"][power])
            else:
                inference_metrics["decision_consistency"][power] = 0.0
                
            if inference_metrics["context_utilization"][power]:
                inference_metrics["context_utilization"][power] = np.mean(inference_metrics["context_utilization"][power])
            else:
                inference_metrics["context_utilization"][power] = 0.0
        
        # Calculate error rates
        total_action_inferences = inference_metrics["action_inferences"]
        if total_action_inferences > 0:
            for power in self.powers:
                power_total_actions = sum(1 for game in game_logs 
                                        for round_data in game.get("rounds_data", []) 
                                        if round_data.get("orders", {}).get(power))
                if power_total_actions > 0:
                    inference_metrics["error_rate"][power] = inference_metrics["error_rate"][power] / power_total_actions
        
        return inference_metrics
    
    def _analyze_order_quality(self, orders: List[str], round_data: Dict[str, Any]) -> float:
        """
        Analyze the quality of orders given the game context.
        
        Args:
            orders: List of orders from a power
            round_data: Round context data
            
        Returns:
            float: Quality score from 0.0 to 1.0
        """
        if not orders:
            return 0.0
        
        quality_score = 0.0
        valid_orders = 0
        
        for order in orders:
            # Check if order follows proper format
            if self._is_valid_order_format(order):
                quality_score += 0.3
                valid_orders += 1
                
                # Check strategic value
                strategic_value = self._assess_order_strategic_value(order, round_data)
                quality_score += strategic_value * 0.7
        
        if valid_orders > 0:
            return quality_score / len(orders)
        return 0.0
    
    def _is_valid_order_format(self, order: str) -> bool:
        """Check if order follows valid Diplomacy syntax."""
        order = order.strip().upper()
        
        # Basic patterns for Diplomacy orders
        patterns = [
            r'^[AF] \w{3}$',  # Hold: A VIE, F LON
            r'^[AF] \w{3} - \w{3}$',  # Move: A VIE - BUD
            r'^[AF] \w{3} S [AF] \w{3}',  # Support hold
            r'^[AF] \w{3} S [AF] \w{3} - \w{3}',  # Support move
            r'^[AF] \w{3} C [AF] \w{3} - \w{3}',  # Convoy
        ]
        
        return any(re.match(pattern, order) for pattern in patterns)
    
    def _assess_order_strategic_value(self, order: str, round_data: Dict[str, Any]) -> float:
        """Assess strategic value of an order in context."""
        # Simple heuristic assessment
        if " S " in order.upper():  # Support orders generally good
            return 0.8
        elif " - " in order.upper():  # Movement orders
            return 0.6
        else:  # Hold orders
            return 0.4
    
    def _check_order_errors(self, orders: List[str]) -> bool:
        """Check if orders contain obvious errors."""
        for order in orders:
            if not order.strip():  # Empty order
                return True
            if not self._is_valid_order_format(order):
                return True
        return False
    
    def _analyze_decision_consistency(self, power: str, rounds_data: List[Dict[str, Any]]) -> Optional[float]:
        """Analyze consistency of decisions across rounds."""
        if len(rounds_data) < 2:
            return None
        
        # Extract patterns in negotiation and action choices
        negotiation_patterns = []
        action_patterns = []
        
        for round_data in rounds_data:
            # Negotiation patterns
            negotiations = round_data.get("negotiations", {}).get(power, {})
            negotiation_patterns.append(self._extract_negotiation_pattern(negotiations))
            
            # Action patterns
            orders = round_data.get("orders", {}).get(power, [])
            action_patterns.append(self._extract_action_pattern(orders))
        
        # Calculate consistency score
        negotiation_consistency = self._calculate_pattern_consistency(negotiation_patterns)
        action_consistency = self._calculate_pattern_consistency(action_patterns)
        
        return (negotiation_consistency + action_consistency) / 2
    
    def _extract_negotiation_pattern(self, negotiations: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pattern from negotiation messages."""
        pattern = {
            "cooperative_count": 0,
            "aggressive_count": 0,
            "alliance_mentions": 0,
            "threat_mentions": 0
        }
        
        cooperative_words = ["ally", "cooperation", "together", "help", "support"]
        aggressive_words = ["attack", "war", "enemy", "threat", "betray"]
        
        for partner, messages in negotiations.items():
            if isinstance(messages, dict):
                message_text = " ".join(str(msg) for msg in messages.values() if isinstance(msg, (str, int, float)))
            else:
                message_text = str(messages)
            
            message_lower = message_text.lower()
            
            pattern["cooperative_count"] += sum(1 for word in cooperative_words if word in message_lower)
            pattern["aggressive_count"] += sum(1 for word in aggressive_words if word in message_lower)
            pattern["alliance_mentions"] += message_lower.count("alliance")
            pattern["threat_mentions"] += message_lower.count("threat")
        
        return pattern
    
    def _extract_action_pattern(self, orders: List[str]) -> Dict[str, Any]:
        """Extract pattern from action orders."""
        pattern = {
            "aggressive_moves": 0,
            "defensive_moves": 0,
            "support_orders": 0,
            "hold_orders": 0
        }
        
        for order in orders:
            order_upper = order.upper()
            if " - " in order_upper:
                pattern["aggressive_moves"] += 1
            elif " S " in order_upper:
                pattern["support_orders"] += 1
            else:
                pattern["hold_orders"] += 1
        
        return pattern
    
    def _calculate_pattern_consistency(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate consistency score for a list of patterns."""
        if len(patterns) < 2:
            return 1.0
        
        # Calculate variance in pattern values
        pattern_keys = patterns[0].keys()
        total_variance = 0
        
        for key in pattern_keys:
            values = [pattern[key] for pattern in patterns]
            if max(values) > 0:  # Avoid division by zero
                normalized_variance = np.var(values) / max(values)
                total_variance += normalized_variance
        
        # Convert variance to consistency (lower variance = higher consistency)
        consistency = 1.0 / (1.0 + total_variance)
        return consistency
    
    def _analyze_context_utilization(self, power: str, rounds_data: List[Dict[str, Any]]) -> Optional[float]:
        """Analyze how well the model utilizes available context."""
        if not rounds_data:
            return None
        
        context_scores = []
        
        for round_data in rounds_data:
            # Check if decisions reflect game state
            territories_before = round_data.get("territories_before", {}).get(power, [])
            territories_after = round_data.get("territories_after", {}).get(power, [])
            orders = round_data.get("orders", {}).get(power, [])
            
            # Score based on territorial changes and order relevance
            if territories_before and orders:
                context_score = self._score_context_awareness(territories_before, territories_after, orders)
                context_scores.append(context_score)
        
        return np.mean(context_scores) if context_scores else 0.0
    
    def _score_context_awareness(self, territories_before: List[str], 
                                territories_after: List[str], orders: List[str]) -> float:
        """Score how well orders reflect territorial context."""
        if not orders:
            return 0.0
        
        # Simple heuristic: orders should reference territories the power controls
        territory_mentions = 0
        for order in orders:
            for territory in territories_before:
                if territory in order.upper():
                    territory_mentions += 1
                    break
        
        return territory_mentions / len(orders) if orders else 0.0
    
    def _calculate_strategic_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate strategic performance metrics."""
        return {
            "win_rate_by_power": self._calculate_win_rate_by_power(game_logs),
            "supply_centers_by_power": self._calculate_supply_centers_by_power(game_logs),
            "survival_rate_by_power": self._calculate_survival_rate_by_power(game_logs),
            "territorial_expansion": self._calculate_territorial_expansion(game_logs),
            "key_territory_control": self._calculate_key_territory_control(game_logs),
            "strategic_positioning": self._calculate_strategic_positioning(game_logs)
        }
    
    def _calculate_tactical_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate tactical performance metrics."""
        return {
            "attack_success_rate": self._calculate_attack_success_rate(game_logs),
            "defense_success_rate": self._calculate_defense_success_rate(game_logs),
            "support_coordination": self._calculate_support_coordination(game_logs),
            "unit_efficiency": self._calculate_unit_efficiency(game_logs),
            "order_complexity": self._calculate_order_complexity(game_logs)
        }
    
    def _calculate_diplomatic_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate diplomatic performance metrics."""
        return {
            "alliance_effectiveness": self._calculate_alliance_effectiveness(game_logs),
            "negotiation_success_rate": self._calculate_negotiation_success_rate(game_logs),
            "action_alignment": self._calculate_action_alignment(game_logs),
            "negotiation_honesty": self._calculate_negotiation_honesty(game_logs),
            "deception_detection": self._calculate_deception_detection(game_logs),
            "alliance_formation": self._calculate_alliance_formation(game_logs),
            "communication_quality": self._calculate_communication_quality(game_logs)
        }
    
    def _calculate_game_outcome_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate game outcome related metrics."""
        outcomes = {
            "total_games": len(game_logs),
            "decisive_victories": 0,
            "draws": 0,
            "early_eliminations": 0,
            "average_game_length": 0,
            "fastest_victory": float('inf'),
            "longest_game": 0
        }
        
        game_lengths = []
        
        for game in game_logs:
            rounds_played = game.get("rounds_played", 0)
            game_lengths.append(rounds_played)
            
            # Check for decisive victory (>=18 supply centers)
            winner = game.get("winner")
            if winner:
                outcomes["decisive_victories"] += 1
                outcomes["fastest_victory"] = min(outcomes["fastest_victory"], rounds_played)
            else:
                outcomes["draws"] += 1
            
            outcomes["longest_game"] = max(outcomes["longest_game"], rounds_played)
            
            # Check for early eliminations
            supply_centers = game.get("supply_centers", {})
            for power, centers in supply_centers.items():
                if centers == 0 and rounds_played < 5:
                    outcomes["early_eliminations"] += 1
        
        outcomes["average_game_length"] = np.mean(game_lengths) if game_lengths else 0
        if outcomes["fastest_victory"] == float('inf'):
            outcomes["fastest_victory"] = 0
        
        return outcomes
    
    def _calculate_behavioral_analysis(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze behavioral patterns of the model."""
        behavior = {
            "aggression_level": {power: [] for power in self.powers},
            "cooperation_tendency": {power: [] for power in self.powers},
            "risk_taking": {power: [] for power in self.powers},
            "adaptability": {power: [] for power in self.powers},
            "communication_style": {power: {} for power in self.powers}
        }
        
        for game in game_logs:
            rounds_data = game.get("rounds_data", [])
            
            for round_data in rounds_data:
                for power in self.powers:
                    # Analyze aggression
                    orders = round_data.get("orders", {}).get(power, [])
                    aggression_score = self._calculate_aggression_score(orders)
                    behavior["aggression_level"][power].append(aggression_score)
                    
                    # Analyze cooperation
                    negotiations = round_data.get("negotiations", {}).get(power, {})
                    cooperation_score = self._calculate_cooperation_score(negotiations)
                    behavior["cooperation_tendency"][power].append(cooperation_score)
                    
                    # Analyze risk taking
                    risk_score = self._calculate_risk_score(orders, round_data)
                    behavior["risk_taking"][power].append(risk_score)
        
        # Calculate averages
        for power in self.powers:
            for metric in ["aggression_level", "cooperation_tendency", "risk_taking"]:
                if behavior[metric][power]:
                    behavior[metric][power] = np.mean(behavior[metric][power])
                else:
                    behavior[metric][power] = 0.0
        
        return behavior
    
    def _calculate_aggression_score(self, orders: List[str]) -> float:
        """Calculate aggression score based on orders."""
        if not orders:
            return 0.0
        
        aggressive_moves = sum(1 for order in orders if " - " in order.upper())
        return aggressive_moves / len(orders)
    
    def _calculate_cooperation_score(self, negotiations: Dict[str, Any]) -> float:
        """Calculate cooperation score based on negotiations."""
        if not negotiations:
            return 0.0
        
        cooperative_keywords = ["ally", "help", "support", "together", "cooperation"]
        total_score = 0
        total_messages = 0
        
        for partner, messages in negotiations.items():
            if isinstance(messages, dict):
                for message in messages.values():
                    message_lower = str(message).lower()
                    score = sum(1 for keyword in cooperative_keywords if keyword in message_lower)
                    total_score += min(score / len(cooperative_keywords), 1.0)
                    total_messages += 1
            else:
                message_lower = str(messages).lower()
                score = sum(1 for keyword in cooperative_keywords if keyword in message_lower)
                total_score += min(score / len(cooperative_keywords), 1.0)
                total_messages += 1
        
        return total_score / total_messages if total_messages > 0 else 0.0
    
    def _calculate_risk_score(self, orders: List[str], round_data: Dict[str, Any]) -> float:
        """Calculate risk taking score."""
        if not orders:
            return 0.0
        
        # Simple heuristic: movement orders are riskier than holds
        risky_orders = sum(1 for order in orders if " - " in order.upper())
        return risky_orders / len(orders)
    
    def _calculate_llm_judge_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM as judge to evaluate strategic, diplomatic, and tactical performance.
        
        Args:
            game_logs: List of game results
            
        Returns:
            Dict[str, Any]: LLM evaluation scores
        """
        if not self.model:
            return {}
        
        llm_scores = {
            "strategic_scores": {power: [] for power in self.powers},
            "diplomatic_scores": {power: [] for power in self.powers},
            "tactical_scores": {power: [] for power in self.powers},
            "overall_scores": {power: [] for power in self.powers}
        }
        
        for game_idx, game in enumerate(game_logs):
            logger.info(f"Evaluating game {game_idx + 1}/{len(game_logs)} with LLM judge")
            
            # Strategic evaluation
            strategic_context = self._prepare_strategic_context(game)
            strategic_evaluation = self._llm_evaluate_strategic(strategic_context)
            
            # Diplomatic evaluation
            diplomatic_context = self._prepare_diplomatic_context(game)
            diplomatic_evaluation = self._llm_evaluate_diplomatic(diplomatic_context)
            
            # Tactical evaluation
            tactical_context = self._prepare_tactical_context(game)
            tactical_evaluation = self._llm_evaluate_tactical(tactical_context)
            
            # Overall evaluation
            overall_context = self._prepare_overall_context(game)
            overall_evaluation = self._llm_evaluate_overall(overall_context)
            
            # Parse and store scores
            for power in self.powers:
                llm_scores["strategic_scores"][power].append(
                    self._extract_score(strategic_evaluation, power, default=5.0)
                )
                llm_scores["diplomatic_scores"][power].append(
                    self._extract_score(diplomatic_evaluation, power, default=5.0)
                )
                llm_scores["tactical_scores"][power].append(
                    self._extract_score(tactical_evaluation, power, default=5.0)
                )
                llm_scores["overall_scores"][power].append(
                    self._extract_score(overall_evaluation, power, default=5.0)
                )
        
        # Calculate averages
        return {
            "strategic_avg": {power: np.mean(scores) for power, scores in llm_scores["strategic_scores"].items()},
            "diplomatic_avg": {power: np.mean(scores) for power, scores in llm_scores["diplomatic_scores"].items()},
            "tactical_avg": {power: np.mean(scores) for power, scores in llm_scores["tactical_scores"].items()},
            "overall_avg": {power: np.mean(scores) for power, scores in llm_scores["overall_scores"].items()},
            "raw_scores": llm_scores
        }
    
    def _prepare_strategic_context(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for strategic evaluation."""
        return {
            "supply_centers": game.get("supply_centers", {}),
            "winner": game.get("winner"),
            "rounds_played": game.get("rounds_played", 0),
            "territorial_changes": self._analyze_territorial_changes(game),
            "strategic_decisions": game.get("strategic_decisions", {})
        }
    
    def _prepare_diplomatic_context(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for diplomatic evaluation."""
        rounds_data = game.get("rounds_data", [])
        return {
            "negotiations_summary": self._summarize_negotiations(rounds_data),
            "alliance_patterns": self._analyze_alliance_patterns(rounds_data),
            "cooperation_instances": self._count_cooperation_instances(rounds_data),
            "betrayal_instances": self._count_betrayal_instances(rounds_data)
        }
    
    def _prepare_tactical_context(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for tactical evaluation."""
        rounds_data = game.get("rounds_data", [])
        return {
            "orders_quality": self._analyze_orders_quality(rounds_data),
            "attack_patterns": self._analyze_attack_patterns(rounds_data),
            "defense_patterns": self._analyze_defense_patterns(rounds_data),
            "coordination_examples": self._find_coordination_examples(rounds_data)
        }
    
    def _prepare_overall_context(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for overall evaluation."""
        return {
            "game_summary": {
                "winner": game.get("winner"),
                "rounds_played": game.get("rounds_played", 0),
                "final_centers": game.get("supply_centers", {}),
                "game_time": game.get("game_time", 0)
            },
            "key_moments": self._identify_key_moments(game),
            "power_performance": self._summarize_power_performance(game)
        }
    
    def _llm_evaluate_strategic(self, context: Dict[str, Any]) -> str:
        """Use LLM to evaluate strategic performance."""
        prompt = f"""
        As an expert Diplomacy analyst, evaluate the strategic performance of each power in this game:

        Game Context:
        - Supply Centers: {context['supply_centers']}
        - Winner: {context['winner']}
        - Rounds Played: {context['rounds_played']}
        - Territorial Changes: {context['territorial_changes']}

        Rate each power's strategic performance on a scale of 1-10, considering:
        1. Long-term planning effectiveness
        2. Adaptability to changing situations
        3. Resource management
        4. Positioning and expansion strategy

        Provide scores in format: "POWER: X/10 - reasoning"
        """
        
        return self._query_llm(prompt)
    
    def _llm_evaluate_diplomatic(self, context: Dict[str, Any]) -> str:
        """Use LLM to evaluate diplomatic performance."""
        prompt = f"""
        As an expert Diplomacy analyst, evaluate the diplomatic performance of each power:

        Diplomatic Context:
        - Negotiations Summary: {context['negotiations_summary']}
        - Alliance Patterns: {context['alliance_patterns']}
        - Cooperation Instances: {context['cooperation_instances']}

        Rate each power's diplomatic performance on a scale of 1-10, considering:
        1. Communication effectiveness
        2. Alliance building and maintenance
        3. Negotiation skills
        4. Trust and reputation management

        Provide scores in format: "POWER: X/10 - reasoning"
        """
        
        return self._query_llm(prompt)
    
    def _llm_evaluate_tactical(self, context: Dict[str, Any]) -> str:
        """Use LLM to evaluate tactical performance."""
        prompt = f"""
        As an expert Diplomacy analyst, evaluate the tactical performance of each power:

        Tactical Context:
        - Orders Quality: {context['orders_quality']}
        - Attack Patterns: {context['attack_patterns']}
        - Defense Patterns: {context['defense_patterns']}

        Rate each power's tactical performance on a scale of 1-10, considering:
        1. Order execution quality
        2. Attack timing and effectiveness
        3. Defensive coordination
        4. Support and convoy usage

        Provide scores in format: "POWER: X/10 - reasoning"
        """
        
        return self._query_llm(prompt)
    
    def _llm_evaluate_overall(self, context: Dict[str, Any]) -> str:
        """Use LLM to evaluate overall performance."""
        prompt = f"""
        As an expert Diplomacy analyst, provide an overall evaluation of each power's performance:

        Game Summary: {context['game_summary']}
        Key Moments: {context['key_moments']}
        Power Performance: {context['power_performance']}

        Rate each power's overall performance on a scale of 1-10, considering:
        1. Achievement of objectives
        2. Overall game impact
        3. Consistency across all aspects
        4. Adaptation to game flow

        Provide scores in format: "POWER: X/10 - reasoning"
        """
        
        return self._query_llm(prompt)
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with error handling."""
        try:
            messages = [
                SystemMessage(content="You are an expert Diplomacy game analyst."),
                HumanMessage(content=prompt)
            ]
            response = self.model.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return "Error in LLM evaluation"
    
    def _extract_score(self, evaluation_text: str, power: str, default: float = 5.0) -> float:
        """Extract numerical score for a power from LLM evaluation text."""
        try:
            # Look for pattern "POWER: X/10" or "POWER: X"
            pattern = rf"{power}:\s*(\d+(?:\.\d+)?)"
            match = re.search(pattern, evaluation_text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                return min(max(score, 0), 10)  # Clamp between 0 and 10
        except Exception as e:
            logger.warning(f"Could not extract score for {power}: {e}")
        
        return default
    
    def _generate_detailed_report(self) -> Dict[str, Any]:
        """Generate comprehensive human-readable report."""
        if not self.metrics:
            return {}
        
        report = {
            "executive_summary": self._generate_executive_summary(),
            "model_performance_analysis": self._generate_model_performance_report(),
            "strategic_analysis": self._generate_strategic_report(),
            "tactical_analysis": self._generate_tactical_report(),
            "diplomatic_analysis": self._generate_diplomatic_report(),
            "comparative_analysis": self._generate_comparative_analysis(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of model performance."""
        model_perf = self.metrics.get("model_performance", {})
        game_outcomes = self.metrics.get("game_outcome_metrics", {})
        
        # Find best and worst performing powers
        strategic_metrics = self.metrics.get("strategic_metrics", {})
        win_rates = strategic_metrics.get("win_rate_by_power", {})
        
        best_power = max(win_rates, key=win_rates.get) if win_rates else "N/A"
        worst_power = min(win_rates, key=win_rates.get) if win_rates else "N/A"
        
        return {
            "total_games_analyzed": self.metrics.get("games_total", 0),
            "total_model_inferences": model_perf.get("total_inferences", 0),
            "overall_error_rate": model_perf.get("total_errors", 0) / max(model_perf.get("total_inferences", 1), 1),
            "average_game_length": game_outcomes.get("average_game_length", 0),
            "decisive_victory_rate": game_outcomes.get("decisive_victories", 0) / max(game_outcomes.get("total_games", 1), 1),
            "best_performing_power": {
                "power": best_power,
                "win_rate": win_rates.get(best_power, 0)
            },
            "worst_performing_power": {
                "power": worst_power,
                "win_rate": win_rates.get(worst_power, 0)
            }
        }
    
    def _generate_model_performance_report(self) -> Dict[str, Any]:
        """Generate detailed model performance analysis."""
        model_perf = self.metrics.get("model_performance", {})
        
        return {
            "inference_breakdown": {
                "negotiation_inferences": model_perf.get("negotiation_inferences", 0),
                "action_inferences": model_perf.get("action_inferences", 0),
                "strategic_inferences": model_perf.get("strategic_inferences", 0)
            },
            "quality_metrics": {
                "average_response_quality": self._calculate_average_metric(model_perf.get("response_quality", {})),
                "average_decision_consistency": self._calculate_average_metric(model_perf.get("decision_consistency", {})),
                "average_context_utilization": self._calculate_average_metric(model_perf.get("context_utilization", {}))
            },
            "error_analysis": {
                "total_errors": model_perf.get("total_errors", 0),
                "error_rate_by_power": model_perf.get("error_rate", {}),
                "most_error_prone_power": self._find_most_error_prone_power(model_perf.get("error_rate", {}))
            },
            "performance_insights": self._generate_performance_insights(model_perf)
        }
    
    def _calculate_average_metric(self, metric_dict: Dict[str, float]) -> float:
        """Calculate average of a metric across all powers."""
        values = [v for v in metric_dict.values() if isinstance(v, (int, float))]
        return np.mean(values) if values else 0.0
    
    def _find_most_error_prone_power(self, error_rates: Dict[str, float]) -> Dict[str, Any]:
        """Find the power with highest error rate."""
        if not error_rates:
            return {"power": "N/A", "error_rate": 0}
        
        most_errors = max(error_rates, key=error_rates.get)
        return {
            "power": most_errors,
            "error_rate": error_rates[most_errors]
        }
    
    def _generate_performance_insights(self, model_perf: Dict[str, Any]) -> List[str]:
        """Generate insights about model performance."""
        insights = []
        
        total_inferences = model_perf.get("total_inferences", 0)
        total_errors = model_perf.get("total_errors", 0)
        
        if total_inferences > 0:
            error_rate = total_errors / total_inferences
            if error_rate < 0.1:
                insights.append("Model shows excellent error handling with <10% error rate")
            elif error_rate < 0.2:
                insights.append("Model shows good error handling with <20% error rate")
            else:
                insights.append(f"Model shows concerning error rate of {error_rate:.1%}")
        
        # Analyze response quality
        response_quality = model_perf.get("response_quality", {})
        avg_quality = self._calculate_average_metric(response_quality)
        if avg_quality > 0.8:
            insights.append("Model demonstrates high-quality responses across powers")
        elif avg_quality > 0.6:
            insights.append("Model shows moderate response quality")
        else:
            insights.append("Model response quality needs improvement")
        
        return insights
    
    def save_detailed_report(self, output_path: str) -> None:
        """
        Save detailed human-readable report to file.
        
        Args:
            output_path: Path to save the report
        """
        if not self.metrics:
            logger.warning("No metrics calculated yet. Call calculate_metrics() first.")
            return
        
        # Generate markdown report
        report_content = self._generate_markdown_report()
        
        # Save markdown report
        with open(f"{output_path}.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save JSON data for programmatic analysis
        with open(f"{output_path}.json", 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed report saved to {output_path}.md and {output_path}.json")
    
    def _generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report."""
        report = f"""# Diplomacy Model Performance Report

Generated: {self.metrics.get('timestamp', 'Unknown')}

## Executive Summary

{self._format_executive_summary()}

## Model Performance Analysis

{self._format_model_performance()}

## Strategic Performance

{self._format_strategic_performance()}

## Tactical Performance

{self._format_tactical_performance()}

## Diplomatic Performance

{self._format_diplomatic_performance()}

## LLM Judge Evaluation

{self._format_llm_evaluation()}

## Behavioral Analysis

{self._format_behavioral_analysis()}

## Recommendations

{self._format_recommendations()}

---
*Report generated by PolitAgent Diplomacy Metrics System*
"""
        return report
    
    def _format_executive_summary(self) -> str:
        """Format executive summary section."""
        summary = self.metrics.get("detailed_report", {}).get("executive_summary", {})
        
        return f"""
**Games Analyzed:** {summary.get('total_games_analyzed', 0)}
**Total Model Inferences:** {summary.get('total_model_inferences', 0):,}
**Overall Error Rate:** {summary.get('overall_error_rate', 0):.2%}
**Average Game Length:** {summary.get('average_game_length', 0):.1f} rounds
**Decisive Victory Rate:** {summary.get('decisive_victory_rate', 0):.2%}

**Best Performing Power:** {summary.get('best_performing_power', {}).get('power', 'N/A')} ({summary.get('best_performing_power', {}).get('win_rate', 0):.2%} win rate)
**Worst Performing Power:** {summary.get('worst_performing_power', {}).get('power', 'N/A')} ({summary.get('worst_performing_power', {}).get('win_rate', 0):.2%} win rate)
"""
    
    def _format_model_performance(self) -> str:
        """Format model performance section."""
        model_perf = self.metrics.get("model_performance", {})
        
        return f"""
### Inference Breakdown
- **Negotiation Inferences:** {model_perf.get('negotiation_inferences', 0):,}
- **Action Inferences:** {model_perf.get('action_inferences', 0):,}
- **Strategic Inferences:** {model_perf.get('strategic_inferences', 0):,}

### Quality Metrics
- **Average Response Quality:** {self._calculate_average_metric(model_perf.get('response_quality', {})):.3f}
- **Average Decision Consistency:** {self._calculate_average_metric(model_perf.get('decision_consistency', {})):.3f}
- **Average Context Utilization:** {self._calculate_average_metric(model_perf.get('context_utilization', {})):.3f}

### Error Analysis
- **Total Errors:** {model_perf.get('total_errors', 0)}
- **Error Rate by Power:**
{self._format_power_metrics(model_perf.get('error_rate', {}), format_type='percentage')}
"""
    
    def _format_power_metrics(self, metrics: Dict[str, float], format_type: str = 'float') -> str:
        """Format metrics by power in a readable table."""
        if not metrics:
            return "  No data available"
        
        lines = []
        for power, value in sorted(metrics.items()):
            if format_type == 'percentage':
                formatted_value = f"{value:.2%}"
            elif format_type == 'integer':
                formatted_value = f"{int(value)}"
            else:
                formatted_value = f"{value:.3f}"
            lines.append(f"  - **{power}:** {formatted_value}")
        
        return "\n".join(lines)
    
    def _format_strategic_performance(self) -> str:
        """Format strategic performance section."""
        strategic = self.metrics.get("strategic_metrics", {})
        
        return f"""
### Win Rates by Power
{self._format_power_metrics(strategic.get('win_rate_by_power', {}), 'percentage')}

### Average Supply Centers
{self._format_power_metrics(strategic.get('supply_centers_by_power', {}), 'float')}

### Survival Rates
{self._format_power_metrics(strategic.get('survival_rate_by_power', {}), 'percentage')}
"""
    
    def _format_tactical_performance(self) -> str:
        """Format tactical performance section."""
        tactical = self.metrics.get("tactical_metrics", {})
        
        return f"""
### Attack Success Rates
{self._format_power_metrics(tactical.get('attack_success_rate', {}), 'percentage')}

### Defense Success Rates
{self._format_power_metrics(tactical.get('defense_success_rate', {}), 'percentage')}
"""
    
    def _format_diplomatic_performance(self) -> str:
        """Format diplomatic performance section."""
        diplomatic = self.metrics.get("diplomatic_metrics", {})
        
        return f"""
### Alliance Effectiveness
{self._format_power_metrics(diplomatic.get('alliance_effectiveness', {}), 'float')}

### Negotiation Success Rates
{self._format_power_metrics(diplomatic.get('negotiation_success_rate', {}), 'percentage')}

### Negotiation Honesty
{self._format_power_metrics(diplomatic.get('negotiation_honesty', {}), 'float')}
"""
    
    def _format_llm_evaluation(self) -> str:
        """Format LLM judge evaluation section."""
        llm_eval = self.metrics.get("llm_evaluation", {})
        
        if not llm_eval:
            return "LLM evaluation not available (no evaluation model provided)."
        
        return f"""
### Strategic Scores (1-10)
{self._format_power_metrics(llm_eval.get('strategic_avg', {}), 'float')}

### Diplomatic Scores (1-10)
{self._format_power_metrics(llm_eval.get('diplomatic_avg', {}), 'float')}

### Tactical Scores (1-10)
{self._format_power_metrics(llm_eval.get('tactical_avg', {}), 'float')}

### Overall Scores (1-10)
{self._format_power_metrics(llm_eval.get('overall_avg', {}), 'float')}
"""
    
    def _format_behavioral_analysis(self) -> str:
        """Format behavioral analysis section."""
        behavior = self.metrics.get("behavioral_analysis", {})
        
        return f"""
### Aggression Levels (0-1)
{self._format_power_metrics(behavior.get('aggression_level', {}), 'float')}

### Cooperation Tendency (0-1)
{self._format_power_metrics(behavior.get('cooperation_tendency', {}), 'float')}

### Risk Taking (0-1)
{self._format_power_metrics(behavior.get('risk_taking', {}), 'float')}
"""
    
    def _format_recommendations(self) -> str:
        """Format recommendations section."""
        recommendations = self.metrics.get("detailed_report", {}).get("recommendations", [])
        
        if not recommendations:
            return "No specific recommendations generated."
        
        return "\n".join(f"- {rec}" for rec in recommendations)

    # Helper methods for LLM evaluation context preparation
    def _analyze_territorial_changes(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze territorial changes throughout the game."""
        rounds_data = game.get("rounds_data", [])
        if not rounds_data:
            return {}
        
        first_round = rounds_data[0] if rounds_data else {}
        last_round = rounds_data[-1] if rounds_data else {}
        
        initial_territories = first_round.get("territories_before", {})
        final_territories = last_round.get("territories_after", {})
        
        changes = {}
        for power in self.powers:
            initial = len(initial_territories.get(power, []))
            final = len(final_territories.get(power, []))
            changes[power] = final - initial
        
        return changes
    
    def _summarize_negotiations(self, rounds_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize negotiation patterns."""
        negotiation_counts = {power: 0 for power in self.powers}
        
        for round_data in rounds_data:
            negotiations = round_data.get("negotiations", {})
            for power, partners in negotiations.items():
                negotiation_counts[power] += len(partners)
        
        return negotiation_counts
    
    def _analyze_alliance_patterns(self, rounds_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze alliance formation patterns."""
        # This is a simplified analysis - could be expanded
        return {"analysis": "Alliance patterns analyzed"}
    
    def _count_cooperation_instances(self, rounds_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count cooperation instances."""
        return {power: 0 for power in self.powers}  # Placeholder
    
    def _count_betrayal_instances(self, rounds_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count betrayal instances."""
        return {power: 0 for power in self.powers}  # Placeholder
    
    def _analyze_orders_quality(self, rounds_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze quality of orders across rounds."""
        quality_scores = {power: [] for power in self.powers}
        
        for round_data in rounds_data:
            orders = round_data.get("orders", {})
            for power, power_orders in orders.items():
                if power in self.powers and power_orders:
                    quality = self._analyze_order_quality(power_orders, round_data)
                    quality_scores[power].append(quality)
        
        return {power: np.mean(scores) if scores else 0.0 
                for power, scores in quality_scores.items()}
    
    def _analyze_attack_patterns(self, rounds_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze attack patterns."""
        return {"analysis": "Attack patterns analyzed"}  # Placeholder
    
    def _analyze_defense_patterns(self, rounds_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze defense patterns."""
        return {"analysis": "Defense patterns analyzed"}  # Placeholder
    
    def _find_coordination_examples(self, rounds_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find examples of coordination."""
        return {"examples": "Coordination examples found"}  # Placeholder
    
    def _identify_key_moments(self, game: Dict[str, Any]) -> List[str]:
        """Identify key moments in the game."""
        return ["Game start", "Key battles", "Game end"]  # Placeholder
    
    def _summarize_power_performance(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize performance of each power."""
        return {power: "Performance summary" for power in self.powers}  # Placeholder

    def _calculate_win_rate_by_power(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет win rate для каждой страны.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Win rate для каждой страны
        """
        wins = {power: 0 for power in self.powers}
        games_played = {power: 0 for power in self.powers}
        
        for game in game_logs:
            if "winner" in game and game["winner"] in self.powers:
                wins[game["winner"]] += 1
            
            # Подсчет игр для каждой страны
            for power in self.powers:
                if power in game.get("supply_centers", {}):
                    games_played[power] += 1
        
        # Расчет win rate
        win_rates = {}
        for power in self.powers:
            win_rates[power] = (wins[power] / games_played[power]) if games_played[power] > 0 else 0.0
            
        return win_rates
    
    def _calculate_supply_centers_by_power(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет среднего количества центров снабжения для каждой страны.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Среднее количество центров снабжения
        """
        supply_centers = {power: [] for power in self.powers}
        
        for game in game_logs:
            if "supply_centers" in game:
                for power in self.powers:
                    if power in game["supply_centers"]:
                        supply_centers[power].append(game["supply_centers"][power])
        
        # Расчет среднего количества центров
        avg_supply_centers = {}
        for power in self.powers:
            avg_supply_centers[power] = np.mean(supply_centers[power]) if supply_centers[power] else 0.0
            
        return avg_supply_centers
    
    def _calculate_strategic_positioning(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate strategic positioning scores."""
        positioning_scores = {power: [] for power in self.powers}
        
        for game in game_logs:
            rounds_data = game.get("rounds_data", [])
            if len(rounds_data) >= 3:  # Need sufficient data
                for power in self.powers:
                    # Analyze territorial control progression
                    territories_progression = []
                    for round_data in rounds_data:
                        territories = round_data.get("territories_after", {}).get(power, [])
                        territories_progression.append(len(territories))
                    
                    # Score based on territorial stability and growth
                    if territories_progression:
                        growth_rate = (territories_progression[-1] - territories_progression[0]) / max(territories_progression[0], 1)
                        stability = 1.0 - (np.std(territories_progression) / max(np.mean(territories_progression), 1))
                        positioning_score = (growth_rate * 0.6 + stability * 0.4)
                        positioning_scores[power].append(max(0, min(1, positioning_score)))
        
        return {power: np.mean(scores) if scores else 0.0 
                for power, scores in positioning_scores.items()}
    
    def _calculate_support_coordination(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate support coordination effectiveness."""
        coordination_scores = {power: [] for power in self.powers}
        
        for game in game_logs:
            rounds_data = game.get("rounds_data", [])
            for round_data in rounds_data:
                orders = round_data.get("orders", {})
                for power, power_orders in orders.items():
                    if power in self.powers and power_orders:
                        support_orders = [order for order in power_orders if " S " in order.upper()]
                        total_orders = len(power_orders)
                        if total_orders > 0:
                            coordination_score = len(support_orders) / total_orders
                            coordination_scores[power].append(coordination_score)
        
        return {power: np.mean(scores) if scores else 0.0 
                for power, scores in coordination_scores.items()}
    
    def _calculate_unit_efficiency(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate unit utilization efficiency."""
        efficiency_scores = {power: [] for power in self.powers}
        
        for game in game_logs:
            rounds_data = game.get("rounds_data", [])
            for round_data in rounds_data:
                orders = round_data.get("orders", {})
                for power, power_orders in orders.items():
                    if power in self.powers:
                        # Get number of units from territories
                        territories = round_data.get("territories_before", {}).get(power, [])
                        expected_units = min(len(territories), 3)  # Simplified estimate
                        actual_orders = len(power_orders) if power_orders else 0
                        
                        if expected_units > 0:
                            efficiency = actual_orders / expected_units
                            efficiency_scores[power].append(min(efficiency, 1.0))
        
        return {power: np.mean(scores) if scores else 0.0 
                for power, scores in efficiency_scores.items()}
    
    def _calculate_order_complexity(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate complexity of orders (support, convoy vs simple moves)."""
        complexity_scores = {power: [] for power in self.powers}
        
        for game in game_logs:
            rounds_data = game.get("rounds_data", [])
            for round_data in rounds_data:
                orders = round_data.get("orders", {})
                for power, power_orders in orders.items():
                    if power in self.powers and power_orders:
                        complex_orders = 0
                        for order in power_orders:
                            order_upper = order.upper()
                            if " S " in order_upper or " C " in order_upper:
                                complex_orders += 1
                        
                        complexity = complex_orders / len(power_orders)
                        complexity_scores[power].append(complexity)
        
        return {power: np.mean(scores) if scores else 0.0 
                for power, scores in complexity_scores.items()}
    
    def _calculate_communication_quality(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate quality of diplomatic communications."""
        quality_scores = {power: [] for power in self.powers}
        
        for game in game_logs:
            rounds_data = game.get("rounds_data", [])
            for round_data in rounds_data:
                negotiations = round_data.get("negotiations", {})
                for power, partners in negotiations.items():
                    if power in self.powers:
                        total_quality = 0
                        message_count = 0
                        
                        for partner, messages in partners.items():
                            if isinstance(messages, dict):
                                for message in messages.values():
                                    quality = self._assess_message_quality(str(message))
                                    total_quality += quality
                                    message_count += 1
                            else:
                                quality = self._assess_message_quality(str(messages))
                                total_quality += quality
                                message_count += 1
                        
                        if message_count > 0:
                            avg_quality = total_quality / message_count
                            quality_scores[power].append(avg_quality)
        
        return {power: np.mean(scores) if scores else 0.0 
                for power, scores in quality_scores.items()}
    
    def _assess_message_quality(self, message: str) -> float:
        """Assess quality of a diplomatic message."""
        if not message or len(message) < 10:
            return 0.1
        
        quality_indicators = [
            "propose", "suggest", "alliance", "cooperation", "support",
            "attack", "defend", "territory", "strategy", "plan"
        ]
        
        message_lower = message.lower()
        quality_count = sum(1 for indicator in quality_indicators if indicator in message_lower)
        
        # Score based on length and keyword presence
        length_score = min(len(message) / 100, 1.0)  # Normalize to 100 chars
        keyword_score = min(quality_count / 3, 1.0)  # Up to 3 keywords for full score
        
        return (length_score * 0.3 + keyword_score * 0.7)
    
    def _generate_strategic_report(self) -> str:
        """Generate strategic analysis report."""
        return "Strategic analysis completed"
    
    def _generate_tactical_report(self) -> str:
        """Generate tactical analysis report."""
        return "Tactical analysis completed"
    
    def _generate_diplomatic_report(self) -> str:
        """Generate diplomatic analysis report."""
        return "Diplomatic analysis completed"
    
    def _generate_comparative_analysis(self) -> str:
        """Generate comparative analysis across powers."""
        return "Comparative analysis completed"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for model improvement."""
        recommendations = []
        
        model_perf = self.metrics.get("model_performance", {})
        error_rate = model_perf.get("total_errors", 0) / max(model_perf.get("total_inferences", 1), 1)
        
        if error_rate > 0.2:
            recommendations.append("Consider improving order validation and error handling")
        
        response_quality = self._calculate_average_metric(model_perf.get("response_quality", {}))
        if response_quality < 0.6:
            recommendations.append("Focus on improving response quality and strategic reasoning")
        
        consistency = self._calculate_average_metric(model_perf.get("decision_consistency", {}))
        if consistency < 0.5:
            recommendations.append("Work on maintaining consistent strategic approach across rounds")
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory across all metrics")
        
        return recommendations

    def _calculate_survival_rate_by_power(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет процента выживания для каждой страны.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Процент выживания
        """
        survivals = {power: 0 for power in self.powers}
        games_played = {power: 0 for power in self.powers}
        
        for game in game_logs:
            for power in self.powers:
                if power in game.get("supply_centers", {}) and game["supply_centers"].get(power, 0) > 0:
                    survivals[power] += 1
                    games_played[power] += 1
                elif power in game.get("supply_centers", {}):
                    games_played[power] += 1
        
        # Расчет процента выживания
        survival_rates = {}
        for power in self.powers:
            survival_rates[power] = (survivals[power] / games_played[power]) if games_played[power] > 0 else 0.0
            
        return survival_rates
    
    def _calculate_territorial_expansion(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет среднего территориального расширения для каждой страны.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Среднее территориальное расширение
        """
        # Начальное количество территорий
        initial_territories = {
            "AUSTRIA": 3, "ENGLAND": 3, "FRANCE": 3, 
            "GERMANY": 3, "ITALY": 3, "RUSSIA": 4, "TURKEY": 3
        }
        
        expansions = {power: [] for power in self.powers}
        
        for game in game_logs:
            if "supply_centers" in game:
                for power in self.powers:
                    if power in game["supply_centers"]:
                        expansion = game["supply_centers"][power] - initial_territories[power]
                        expansions[power].append(expansion)
        
        # Расчет среднего расширения
        avg_expansions = {}
        for power in self.powers:
            avg_expansions[power] = np.mean(expansions[power]) if expansions[power] else 0.0
            
        return avg_expansions
    
    def _calculate_key_territory_control(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет контроля ключевых территорий для каждой страны.
        Ключевые территории: Munich, Moscow, Vienna, Paris, London, Rome, Constantinople
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Процент контроля ключевых территорий
        """
        key_territories = ["MUN", "MOS", "VIE", "PAR", "LON", "ROM", "CON"]
        territory_control = {power: {terr: 0 for terr in key_territories} for power in self.powers}
        games_count = len(game_logs)
        
        if games_count == 0:
            return {power: 0.0 for power in self.powers}
        
        for game in game_logs:
            if "supply_centers" in game and isinstance(game["supply_centers"], dict):
                for territory in key_territories:
                    for power in self.powers:
                        # Предполагаем, что в игровых данных есть информация о контроле территорий
                        # Адаптировать под реальную структуру данных
                        if territory in game.get("territories", {}).get(power, []):
                            territory_control[power][territory] += 1
        
        # Расчет процента контроля
        control_rates = {}
        for power in self.powers:
            control_sum = sum(territory_control[power].values())
            control_rates[power] = control_sum / (len(key_territories) * games_count)
            
        return control_rates
    
    def _calculate_attack_success_rate(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет успешности атак для каждой страны.
        Атака считается успешной, если удалось захватить новую территорию.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Процент успешных атак
        """
        attack_attempts = {power: 0 for power in self.powers}
        attack_successes = {power: 0 for power in self.powers}
        
        for game in game_logs:
            # Анализируем логи игры поход за походом
            rounds_data = game.get("rounds_data", [])
            for round_data in rounds_data:
                orders = round_data.get("orders", {})
                territories_before = round_data.get("territories_before", {})
                territories_after = round_data.get("territories_after", {})
                
                for power in self.powers:
                    power_orders = orders.get(power, [])
                    # Находим все приказы атаки
                    attack_orders = [order for order in power_orders 
                                   if any(keyword in order.upper() for keyword in ["ATTACK", "SUPPORT", "MOVE TO"])]
                    
                    attack_attempts[power] += len(attack_orders)
                    
                    # Сравниваем территории до и после для определения успешности атак
                    territories_before_power = set(territories_before.get(power, []))
                    territories_after_power = set(territories_after.get(power, []))
                    
                    # Новые захваченные территории
                    new_territories = territories_after_power - territories_before_power
                    attack_successes[power] += len(new_territories)
        
        # Расчет процента успешных атак
        success_rates = {}
        for power in self.powers:
            success_rates[power] = (attack_successes[power] / attack_attempts[power]) if attack_attempts[power] > 0 else 0.0
            
        return success_rates
    
    def _calculate_defense_success_rate(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет успешности защиты для каждой страны.
        Защита считается успешной, если удалось сохранить территорию при атаке.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Процент успешной защиты
        """
        defense_attempts = {power: 0 for power in self.powers}
        defense_successes = {power: 0 for power in self.powers}
        
        for game in game_logs:
            rounds_data = game.get("rounds_data", [])
            for round_data in rounds_data:
                orders = round_data.get("orders", {})
                territories_before = round_data.get("territories_before", {})
                territories_after = round_data.get("territories_after", {})
                attacks_received = round_data.get("attacks_received", {})
                
                for power in self.powers:
                    # Количество атак, полученных державой
                    power_attacks_received = attacks_received.get(power, [])
                    defense_attempts[power] += len(power_attacks_received)
                    
                    # Сравниваем территории до и после для определения успешности защиты
                    territories_before_power = set(territories_before.get(power, []))
                    territories_after_power = set(territories_after.get(power, []))
                    
                    # Территории, которые сохранились после атак
                    maintained_territories = territories_before_power.intersection(territories_after_power)
                    
                    # Для каждой атакованной территории проверяем, сохранилась ли она
                    successful_defenses = 0
                    for attack in power_attacks_received:
                        # Handle both string and dict formats for attack data
                        if isinstance(attack, dict):
                            target_territory = attack.get("target", "")
                        else:
                            target_territory = str(attack)
                        
                        if target_territory in maintained_territories:
                            successful_defenses += 1
                    defense_successes[power] += successful_defenses
        
        # Расчет процента успешной защиты
        defense_rates = {}
        for power in self.powers:
            defense_rates[power] = (defense_successes[power] / defense_attempts[power]) if defense_attempts[power] > 0 else 1.0
            
        return defense_rates
    
    def _calculate_alliance_effectiveness(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет эффективности альянсов для каждой страны.
        Оценивается на основе координации действий и результатов.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Оценка эффективности альянсов (0-1)
        """
        alliance_scores = {power: [] for power in self.powers}
        
        for game in game_logs:
            negotiations = self._extract_negotiation_data(game)
            rounds_data = game.get("rounds_data", [])
            
            for power in self.powers:
                # Получаем все альянсы, объявленные державой
                power_alliances = {}
                
                for other_power, messages in negotiations.get(power, {}).items():
                    # Проверяем сообщения на наличие предложений альянса
                    alliance_mentioned = False
                    if isinstance(messages, dict):
                        for msg in messages.values():
                            if isinstance(msg, str) and ("alliance" in msg.lower() or "ally" in msg.lower()):
                                alliance_mentioned = True
                                break
                    elif isinstance(messages, str):
                        if "alliance" in messages.lower() or "ally" in messages.lower():
                            alliance_mentioned = True
                    
                    if alliance_mentioned:
                        power_alliances[other_power] = True
                
                # Оцениваем эффективность альянсов
                if power_alliances:
                    alliance_effectiveness_score = 0.0
                    
                    for round_data in rounds_data:
                        orders = round_data.get("orders", {})
                        power_orders = orders.get(power, [])
                        
                        # Проверяем, насколько действия соответствовали альянсам
                        coordinated_actions = 0
                        for ally in power_alliances:
                            ally_orders = orders.get(ally, [])
                            
                            # Проверяем координацию действий (поддержка, совместная атака и т.д.)
                            coordination = self._check_orders_coordination(power_orders, ally_orders)
                            if coordination > 0:
                                coordinated_actions += 1
                        
                        # Рассчитываем эффективность для текущего раунда
                        if power_alliances:
                            round_effectiveness = coordinated_actions / len(power_alliances)
                            alliance_effectiveness_score += round_effectiveness
                    
                    # Усредняем по количеству раундов
                    if rounds_data:
                        alliance_effectiveness_score /= len(rounds_data)
                        alliance_scores[power].append(alliance_effectiveness_score)
        
        # Расчет средней эффективности альянсов
        avg_alliance_effectiveness = {}
        for power in self.powers:
            avg_alliance_effectiveness[power] = np.mean(alliance_scores[power]) if alliance_scores[power] else 0.0
            
        return avg_alliance_effectiveness
    
    def _check_orders_coordination(self, orders1: List[str], orders2: List[str]) -> float:
        """
        Проверяет координацию между приказами двух держав.
        
        Args:
            orders1: Приказы первой державы
            orders2: Приказы второй державы
            
        Returns:
            float: Оценка координации (0-1)
        """
        # Проверяем наличие поддержки (support) между приказами
        support_count = 0
        for order1 in orders1:
            for order2 in orders2:
                if "SUPPORT" in order1.upper() and any(territory in order1 for territory in order2.split()):
                    support_count += 1
                if "SUPPORT" in order2.upper() and any(territory in order2 for territory in order1.split()):
                    support_count += 1
        
        # Проверяем атаки на общего противника
        common_targets = set()
        for order1 in orders1:
            for order2 in orders2:
                if "MOVE" in order1.upper() and "MOVE" in order2.upper():
                    target1 = order1.split()[-1] if order1.split() else ""
                    target2 = order2.split()[-1] if order2.split() else ""
                    if target1 == target2:
                        common_targets.add(target1)
        
        coordination_score = (support_count + len(common_targets)) / max(len(orders1) + len(orders2), 1)
        return min(coordination_score, 1.0)  # Ограничиваем максимальным значением 1.0
    
    def _calculate_negotiation_success_rate(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет успешности переговоров для каждой страны.
        Оценивается на основе выполнения договоренностей.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Процент успешных переговоров
        """
        proposal_counts = {power: 0 for power in self.powers}
        successful_proposals = {power: 0 for power in self.powers}
        
        for game in game_logs:
            negotiations = self._extract_negotiation_data(game)
            rounds_data = game.get("rounds_data", [])
            
            for power in self.powers:
                # Извлекаем все предложения, сделанные державой
                for other_power, messages in negotiations.get(power, {}).items():
                    for round_idx, message in messages.items():
                        proposals = self._extract_proposals_from_message(message)
                        proposal_counts[power] += len(proposals)
                        
                        # Проверяем выполнение каждого предложения
                        for proposal in proposals:
                            # Ищем соответствующий раунд данных
                            round_data = rounds_data[int(round_idx)] if int(round_idx) < len(rounds_data) else None
                            if round_data:
                                # Проверяем, было ли предложение выполнено
                                if self._check_proposal_fulfilled(proposal, power, other_power, round_data):
                                    successful_proposals[power] += 1
        
        # Расчет процента успешных переговоров
        success_rates = {}
        for power in self.powers:
            success_rates[power] = (successful_proposals[power] / proposal_counts[power]) if proposal_counts[power] > 0 else 0.0
            
        return success_rates
    
    def _extract_proposals_from_message(self, message: str) -> List[Dict[str, Any]]:
        """
        Извлекает предложения из сообщения.
        
        Args:
            message: Текст сообщения
            
        Returns:
            List[Dict[str, Any]]: Список предложений
        """
        proposals = []
        
        # Convert message to string if it's not already
        message_str = str(message) if message is not None else ""
        
        # Ищем предложения о демилитаризованной зоне (DMZ)
        dmz_matches = re.findall(r'DMZ in ([A-Z]{3})', message_str)
        for match in dmz_matches:
            proposals.append({"type": "DMZ", "territory": match})
        
        # Ищем предложения о поддержке
        support_matches = re.findall(r'support (?:your|my) (?:move|attack) (?:to|on) ([A-Z]{3})', message_str, re.IGNORECASE)
        for match in support_matches:
            proposals.append({"type": "SUPPORT", "territory": match})
        
        # Ищем предложения о ненападении
        nonaggression_matches = re.findall(r'not attack (?:you|your) (?:in|at) ([A-Z]{3})', message_str, re.IGNORECASE)
        for match in nonaggression_matches:
            proposals.append({"type": "NONAGGRESSION", "territory": match})
        
        return proposals
    
    def _check_proposal_fulfilled(self, proposal: Dict[str, Any], proposer: str, receiver: str, round_data: Dict[str, Any]) -> bool:
        """
        Проверяет, было ли выполнено предложение.
        
        Args:
            proposal: Предложение
            proposer: Держава, сделавшая предложение
            receiver: Держава, получившая предложение
            round_data: Данные раунда
            
        Returns:
            bool: True, если предложение было выполнено, иначе False
        """
        orders = round_data.get("orders", {})
        proposer_orders = orders.get(proposer, [])
        receiver_orders = orders.get(receiver, [])
        
        if proposal["type"] == "DMZ":
            # Проверяем, что ни одна из держав не вторглась в DMZ
            territory = proposal["territory"]
            for order in proposer_orders + receiver_orders:
                if territory in order and "MOVE" in order.upper():
                    return False
            return True
        
        elif proposal["type"] == "SUPPORT":
            # Проверяем наличие поддержки
            territory = proposal["territory"]
            for order in proposer_orders:
                if "SUPPORT" in order.upper() and territory in order:
                    return True
            return False
        
        elif proposal["type"] == "NONAGGRESSION":
            # Проверяем отсутствие атак
            territory = proposal["territory"]
            for order in proposer_orders:
                if territory in order and "MOVE" in order.upper():
                    return False
            return True
        
        return False
    
    def _calculate_action_alignment(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет соответствия действий заявленным намерениям для каждой страны.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Оценка соответствия действий намерениям (0-1)
        """
        alignment_scores = {power: [] for power in self.powers}
        
        for game in game_logs:
            negotiations = self._extract_negotiation_data(game)
            rounds_data = game.get("rounds_data", [])
            
            for power in self.powers:
                for round_idx, round_data in enumerate(rounds_data):
                    # Получаем намерения из переговоров
                    stated_intentions = []
                    for other_power, messages in negotiations.get(power, {}).items():
                        # Берем сообщение из предыдущего раунда
                        prev_round = str(round_idx - 1)
                        if prev_round in messages:
                            stated_intentions.extend(self._extract_intentions(messages[prev_round]))
                    
                    # Получаем фактические действия
                    orders = round_data.get("orders", {}).get(power, [])
                    
                    # Оцениваем соответствие
                    if stated_intentions:
                        alignment_score = self._calculate_intentions_actions_alignment(stated_intentions, orders)
                        alignment_scores[power].append(alignment_score)
        
        # Расчет среднего соответствия
        avg_alignment = {}
        for power in self.powers:
            avg_alignment[power] = np.mean(alignment_scores[power]) if alignment_scores[power] else 0.5
            
        return avg_alignment
    
    def _extract_intentions(self, message: str) -> List[Dict[str, Any]]:
        """
        Извлекает заявленные намерения из сообщения.
        
        Args:
            message: Текст сообщения
            
        Returns:
            List[Dict[str, Any]]: Список намерений
        """
        intentions = []
        
        # Convert message to string if it's not already
        message_str = str(message) if message is not None else ""
        
        # Ищем намерения о движении
        move_matches = re.findall(r'(?:move|attack) (?:to|on) ([A-Z]{3})', message_str, re.IGNORECASE)
        for match in move_matches:
            intentions.append({"type": "MOVE", "territory": match})
        
        # Ищем намерения о поддержке
        support_matches = re.findall(r'support (?:your|my) (?:move|attack) (?:to|on) ([A-Z]{3})', message_str, re.IGNORECASE)
        for match in support_matches:
            intentions.append({"type": "SUPPORT", "territory": match})
        
        # Ищем намерения о защите
        defend_matches = re.findall(r'(?:defend|protect|hold) ([A-Z]{3})', message_str, re.IGNORECASE)
        for match in defend_matches:
            intentions.append({"type": "HOLD", "territory": match})
        
        return intentions
    
    def _calculate_intentions_actions_alignment(self, intentions: List[Dict[str, Any]], orders: List[str]) -> float:
        """
        Рассчитывает соответствие между намерениями и фактическими действиями.
        
        Args:
            intentions: Список намерений
            orders: Список приказов
            
        Returns:
            float: Оценка соответствия (0-1)
        """
        if not intentions:
            return 0.5  # Нейтральная оценка, если намерения не заявлены
        
        fulfilled_intentions = 0
        
        for intention in intentions:
            intention_type = intention["type"]
            territory = intention["territory"]
            
            for order in orders:
                if intention_type == "MOVE" and "MOVE" in order.upper() and territory in order:
                    fulfilled_intentions += 1
                    break
                elif intention_type == "SUPPORT" and "SUPPORT" in order.upper() and territory in order:
                    fulfilled_intentions += 1
                    break
                elif intention_type == "HOLD" and "HOLD" in order.upper() and territory in order:
                    fulfilled_intentions += 1
                    break
        
        alignment_score = fulfilled_intentions / len(intentions)
        return alignment_score
    
    def _calculate_negotiation_honesty(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет честности в переговорах для каждой страны.
        Оценивается на основе соответствия действий обещаниям.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Оценка честности в переговорах (0-1)
        """
        honesty_scores = {power: [] for power in self.powers}
        
        for game in game_logs:
            negotiations = self._extract_negotiation_data(game)
            rounds_data = game.get("rounds_data", [])
            
            for power in self.powers:
                for round_idx, round_data in enumerate(rounds_data):
                    promises = []
                    # Собираем все обещания из переговоров
                    for other_power, messages in negotiations.get(power, {}).items():
                        prev_round = str(round_idx - 1)
                        if prev_round in messages:
                            message = messages[prev_round]
                            new_promises = self._extract_promises(message, other_power)
                            promises.extend(new_promises)
                    
                    if promises:
                        orders = round_data.get("orders", {}).get(power, [])
                        honesty_score = self._calculate_promises_fulfillment(promises, orders)
                        honesty_scores[power].append(honesty_score)
        
        # Расчет средней честности
        avg_honesty = {}
        for power in self.powers:
            avg_honesty[power] = np.mean(honesty_scores[power]) if honesty_scores[power] else 0.5
            
        return avg_honesty
    
    def _extract_promises(self, message: str, to_power: str) -> List[Dict[str, Any]]:
        """
        Извлекает обещания из сообщения.
        
        Args:
            message: Текст сообщения
            to_power: Держава, которой было адресовано сообщение
            
        Returns:
            List[Dict[str, Any]]: Список обещаний
        """
        promises = []
        
        # Convert message to string if it's not already
        message_str = str(message) if message is not None else ""
        
        # Ищем обещания о ненападении
        nonaggression_matches = re.findall(r'(?:promise|will) not (?:attack|move into) ([A-Z]{3})', message_str, re.IGNORECASE)
        for match in nonaggression_matches:
            promises.append({"type": "NONAGGRESSION", "territory": match, "to_power": to_power})
        
        # Ищем обещания о поддержке
        support_matches = re.findall(r'(?:promise|will) support (?:your|you) (?:in|at) ([A-Z]{3})', message_str, re.IGNORECASE)
        for match in support_matches:
            promises.append({"type": "SUPPORT", "territory": match, "to_power": to_power})
        
        # Ищем обещания о DMZ
        dmz_matches = re.findall(r'(?:promise|will) (?:respect|maintain) DMZ (?:in|at) ([A-Z]{3})', message_str, re.IGNORECASE)
        for match in dmz_matches:
            promises.append({"type": "DMZ", "territory": match, "to_power": to_power})
        
        return promises
    
    def _calculate_promises_fulfillment(self, promises: List[Dict[str, Any]], orders: List[str]) -> float:
        """
        Рассчитывает выполнение обещаний на основе фактических действий.
        
        Args:
            promises: Список обещаний
            orders: Список приказов
            
        Returns:
            float: Оценка выполнения обещаний (0-1)
        """
        if not promises:
            return 0.5  # Нейтральная оценка, если обещания не были даны
        
        kept_promises = 0
        
        for promise in promises:
            promise_type = promise["type"]
            territory = promise["territory"]
            
            if promise_type == "NONAGGRESSION":
                # Проверяем, что нет приказов на атаку этой территории
                if not any(territory in order and "MOVE" in order.upper() for order in orders):
                    kept_promises += 1
            
            elif promise_type == "SUPPORT":
                # Проверяем наличие приказа на поддержку
                if any("SUPPORT" in order.upper() and territory in order for order in orders):
                    kept_promises += 1
            
            elif promise_type == "DMZ":
                # Проверяем отсутствие приказов на вторжение в DMZ
                if not any(territory in order and "MOVE" in order.upper() for order in orders):
                    kept_promises += 1
        
        honesty_score = kept_promises / len(promises)
        return honesty_score
    
    def _calculate_deception_detection(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет способности обнаруживать обман для каждой страны.
        Оценивается на основе реакций на нарушенные обещания.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Оценка способности обнаруживать обман (0-1)
        """
        detection_scores = {power: [] for power in self.powers}
        
        for game in game_logs:
            negotiations = self._extract_negotiation_data(game)
            rounds_data = game.get("rounds_data", [])
            
            # Отслеживаем нарушенные обещания
            broken_promises = {}  # {(from_power, to_power): [territories]}
            
            for round_idx, round_data in enumerate(rounds_data):
                # Пропускаем первый раунд, так как нам нужны данные о предыдущих обещаниях
                if round_idx == 0:
                    continue
                
                # Проверяем обещания из предыдущего раунда
                for power in self.powers:
                    for other_power, messages in negotiations.get(power, {}).items():
                        prev_round = str(round_idx - 1)
                        if prev_round in messages:
                            message = messages[prev_round]
                            promises = self._extract_promises(message, other_power)
                            
                            orders = round_data.get("orders", {}).get(power, [])
                            
                            # Находим нарушенные обещания
                            for promise in promises:
                                if not self._is_promise_kept(promise, orders):
                                    key = (power, other_power)
                                    if key not in broken_promises:
                                        broken_promises[key] = []
                                    broken_promises[key].append(promise["territory"])
                
                # Проверяем реакцию на нарушенные обещания в следующем раунде
                if round_idx + 1 < len(rounds_data):
                    next_round = str(round_idx + 1)
                    for (from_power, to_power), territories in broken_promises.items():
                        if to_power in negotiations:
                            messages_to_betrayer = negotiations[to_power].get(from_power, {})
                            if next_round in messages_to_betrayer:
                                message = messages_to_betrayer[next_round]
                                # Оцениваем, обнаружен ли обман
                                detection_score = self._assess_deception_detection(message, territories)
                                detection_scores[to_power].append(detection_score)
        
        # Расчет средней способности обнаруживать обман
        avg_detection = {}
        for power in self.powers:
            avg_detection[power] = np.mean(detection_scores[power]) if detection_scores[power] else 0.5
            
        return avg_detection
    
    def _is_promise_kept(self, promise: Dict[str, Any], orders: List[str]) -> bool:
        """
        Проверяет, было ли обещание выполнено.
        
        Args:
            promise: Обещание
            orders: Список приказов
            
        Returns:
            bool: True, если обещание выполнено, иначе False
        """
        promise_type = promise["type"]
        territory = promise["territory"]
        
        if promise_type == "NONAGGRESSION":
            # Проверяем отсутствие атак на территорию
            return not any(territory in order and "MOVE" in order.upper() for order in orders)
        
        elif promise_type == "SUPPORT":
            # Проверяем наличие поддержки
            return any("SUPPORT" in order.upper() and territory in order for order in orders)
        
        elif promise_type == "DMZ":
            # Проверяем отсутствие вторжения в DMZ
            return not any(territory in order and "MOVE" in order.upper() for order in orders)
        
        return False
    
    def _assess_deception_detection(self, message: str, territories: List[str]) -> float:
        """
        Оценивает, насколько сообщение указывает на обнаружение обмана.
        
        Args:
            message: Текст сообщения
            territories: Список территорий, связанных с нарушенными обещаниями
            
        Returns:
            float: Оценка обнаружения обмана (0-1)
        """
        # Convert message to string if it's not already
        message_str = str(message) if message is not None else ""
        
        # Ключевые слова, указывающие на обнаружение обмана
        deception_keywords = ["betrayed", "lied", "broken promise", "deceived", "not trust", "violated"]
        
        # Подсчет упоминаний территорий из нарушенных обещаний
        territory_mentions = sum(1 for territory in territories if territory in message_str)
        
        # Подсчет ключевых слов, указывающих на обнаружение обмана
        keyword_count = sum(1 for keyword in deception_keywords if keyword in message_str.lower())
        
        # Комбинированная оценка
        detection_score = 0.0
        if territories:
            territory_score = territory_mentions / len(territories)
            detection_score = (territory_score + min(keyword_count / 3, 1.0)) / 2
        else:
            detection_score = min(keyword_count / 3, 1.0)
        
        return detection_score
    
    def _calculate_alliance_formation(self, game_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Расчет способности формировать союзы для каждой страны.
        
        Args:
            game_logs: Список логов игр
            
        Returns:
            Dict[str, float]: Оценка способности формировать союзы (0-1)
        """
        alliance_counts = {power: 0 for power in self.powers}
        stable_alliance_counts = {power: 0 for power in self.powers}
        
        for game in game_logs:
            negotiations = self._extract_negotiation_data(game)
            rounds_data = game.get("rounds_data", [])
            
            # Отслеживаем альянсы по раундам
            alliances_by_round = {power: {} for power in self.powers}  # {power: {round_idx: [allies]}}
            
            for round_idx, _ in enumerate(rounds_data):
                # Анализируем переговоры текущего раунда
                for power in self.powers:
                    for other_power, messages in negotiations.get(power, {}).items():
                        current_round = str(round_idx)
                        if current_round in messages:
                            message = messages[current_round]
                            # Проверяем предложение или подтверждение альянса
                            if self._is_alliance_proposal_or_confirmation(message):
                                if round_idx not in alliances_by_round[power]:
                                    alliances_by_round[power][round_idx] = []
                                alliances_by_round[power][round_idx].append(other_power)
                                
                                alliance_counts[power] += 1
            
            # Оцениваем стабильность альянсов
            for power in self.powers:
                ongoing_alliances = {}  # {ally: start_round}
                
                for round_idx in sorted(alliances_by_round[power].keys()):
                    allies = alliances_by_round[power][round_idx]
                    
                    # Добавляем новых союзников
                    for ally in allies:
                        if ally not in ongoing_alliances:
                            ongoing_alliances[ally] = round_idx
                    
                    # Проверяем продолжение альянсов
                    for ally, start_round in list(ongoing_alliances.items()):
                        # Если союзник отсутствует в текущем раунде
                        if ally not in allies:
                            # Если альянс продержался не менее 3 раундов
                            if round_idx - start_round >= 3:
                                stable_alliance_counts[power] += 1
                            
                            del ongoing_alliances[ally]
                
                # Проверяем альянсы, которые дожили до конца игры
                for ally, start_round in ongoing_alliances.items():
                    if len(rounds_data) - start_round >= 3:
                        stable_alliance_counts[power] += 1
        
        # Расчет способности формировать стабильные союзы
        alliance_formation_scores = {}
        for power in self.powers:
            # Если не было попыток формирования альянсов, ставим нейтральную оценку
            if alliance_counts[power] == 0:
                alliance_formation_scores[power] = 0.5
            else:
                alliance_formation_scores[power] = stable_alliance_counts[power] / alliance_counts[power]
        
        return alliance_formation_scores
    
    def _is_alliance_proposal_or_confirmation(self, message: str) -> bool:
        """
        Проверяет, является ли сообщение предложением или подтверждением альянса.
        
        Args:
            message: Текст сообщения
            
        Returns:
            bool: True, если сообщение содержит предложение или подтверждение альянса, иначе False
        """
        # Convert message to string if it's not already
        message_str = str(message) if message is not None else ""
        
        alliance_keywords = ["alliance", "ally", "join forces", "work together", "cooperate", "team up"]
        for keyword in alliance_keywords:
            if keyword in message_str.lower():
                return True
        return False
    
    def _extract_negotiation_data(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлечение данных о переговорах из игры.
        
        Args:
            game: Данные игры
            
        Returns:
            Dict[str, Any]: Данные о переговорах
        """
        negotiations = {}
        
        # Извлекаем данные из структуры игры
        rounds_data = game.get("rounds_data", [])
        
        for round_idx, round_data in enumerate(rounds_data):
            negotiation_messages = round_data.get("negotiations", {})
            
            for from_power, to_powers in negotiation_messages.items():
                if from_power not in negotiations:
                    negotiations[from_power] = {}
                
                for to_power, message in to_powers.items():
                    if to_power not in negotiations[from_power]:
                        negotiations[from_power][to_power] = {}
                    
                    negotiations[from_power][to_power][str(round_idx)] = message
        
        return negotiations
    
    def _extract_orders_data(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлечение данных о приказах из игры.
        
        Args:
            game: Данные игры
            
        Returns:
            Dict[str, Any]: Данные о приказах
        """
        orders_data = {power: [] for power in self.powers}
        
        # Извлекаем данные из структуры игры
        rounds_data = game.get("rounds_data", [])
        
        for round_data in rounds_data:
            round_orders = round_data.get("orders", {})
            
            for power, orders in round_orders.items():
                if power in self.powers:
                    orders_data[power].extend(orders)
        
        return orders_data
    
    def visualize_metrics(self, metrics: Dict[str, Any], output_file: str) -> None:
        """
        Визуализация метрик.
        
        Args:
            metrics: Метрики для визуализации
            output_file: Файл для сохранения визуализации
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Настройка стиля
        sns.set(style="whitegrid")
        
        # Win rate by power
        plt.figure(figsize=(12, 8))
        win_rates = metrics.get("win_rate_by_power", {})
        sns.barplot(x=list(win_rates.keys()), y=list(win_rates.values()))
        plt.title("Win Rate by Power")
        plt.ylabel("Win Rate")
        plt.xlabel("Power")
        plt.savefig(f"{output_file}_win_rate.png")
        
        # Supply centers by power
        plt.figure(figsize=(12, 8))
        supply_centers = metrics.get("supply_centers_by_power", {})
        sns.barplot(x=list(supply_centers.keys()), y=list(supply_centers.values()))
        plt.title("Average Supply Centers by Power")
        plt.ylabel("Average Supply Centers")
        plt.xlabel("Power")
        plt.savefig(f"{output_file}_supply_centers.png")
        
        # LLM judge metrics
        if "llm_judge_overall" in metrics:
            plt.figure(figsize=(12, 8))
            llm_scores = metrics.get("llm_judge_overall", {})
            sns.barplot(x=list(llm_scores.keys()), y=list(llm_scores.values()))
            plt.title("LLM Judge Overall Score by Power")
            plt.ylabel("Score (0-10)")
            plt.xlabel("Power")
            plt.savefig(f"{output_file}_llm_judge.png")
        
        # Radar chart for LLM judge categories
        if "llm_judge_strategic" in metrics:
            plt.figure(figsize=(12, 8))
            categories = ["Strategic", "Diplomatic", "Tactical", "Overall"]
            
            for power in self.powers:
                values = [
                    metrics.get("llm_judge_strategic", {}).get(power, 0),
                    metrics.get("llm_judge_diplomatic", {}).get(power, 0),
                    metrics.get("llm_judge_tactical", {}).get(power, 0),
                    metrics.get("llm_judge_overall", {}).get(power, 0)
                ]
                values += values[:1]  # Close the polygon
                
                # Angles for each category
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]  # Close the polygon
                
                plt.polar(angles, values, label=power)
            
            plt.title("LLM Judge Scores by Category")
            plt.legend(loc="upper right")
            plt.savefig(f"{output_file}_llm_judge_radar.png")
    
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
        report = f"""# Diplomacy Game Analysis Report

Generated: {metrics_data.get('timestamp', 'Unknown')}
Games Analyzed: {metrics_data.get('games_total', 0)}

## Executive Summary

"""
        
        detailed_report = metrics_data.get('detailed_report', {})
        executive_summary = detailed_report.get('executive_summary', {})
        model_perf = metrics_data.get('model_performance', {})
        strategic_metrics = metrics_data.get('strategic_metrics', {})
        
        report += f"""### Key Performance Indicators
- **Total Inferences**: {executive_summary.get('total_inferences', 0)}
- **Negotiation Inferences**: {model_perf.get('negotiation_inferences', 0)}
- **Action Inferences**: {model_perf.get('action_inferences', 0)}
- **Strategic Inferences**: {model_perf.get('strategic_inferences', 0)}
- **Total Errors**: {model_perf.get('total_errors', 0)}

## Model Inference Performance

### Overall Statistics
- **Average Response Quality**: {executive_summary.get('average_response_quality', 0):.3f}
- **Average Decision Consistency**: {executive_summary.get('average_decision_consistency', 0):.3f}
- **Average Context Utilization**: {executive_summary.get('average_context_utilization', 0):.3f}

### Power-Specific Performance
"""
        
        response_quality = model_perf.get('response_quality', {})
        decision_consistency = model_perf.get('decision_consistency', {})
        context_utilization = model_perf.get('context_utilization', {})
        
        for power in self.powers:
            if power in response_quality:
                report += f"""
#### {power}
- **Response Quality**: {response_quality.get(power, 0):.3f}
- **Decision Consistency**: {decision_consistency.get(power, 0):.3f}
- **Context Utilization**: {context_utilization.get(power, 0):.3f}
- **Error Rate**: {model_perf.get('error_rate', {}).get(power, 0):.3f}
"""
        
        report += f"""
## Strategic Analysis

### Game Outcomes
"""
        
        game_outcome_metrics = metrics_data.get('game_outcome_metrics', {})
        win_rates = game_outcome_metrics.get('win_rate_by_power', {})
        supply_centers = game_outcome_metrics.get('supply_centers_by_power', {})
        
        if win_rates:
            report += "#### Win Rates by Power\n"
            for power, rate in sorted(win_rates.items(), key=lambda x: x[1], reverse=True):
                report += f"- **{power}**: {rate:.1%}\n"
        
        if supply_centers:
            report += "\n#### Average Supply Centers by Power\n"
            for power, centers in sorted(supply_centers.items(), key=lambda x: x[1], reverse=True):
                report += f"- **{power}**: {centers:.1f}\n"
        
        # Strategic metrics
        if strategic_metrics:
            report += f"""
### Strategic Performance
"""
            for metric_name, metric_value in strategic_metrics.items():
                if isinstance(metric_value, dict):
                    report += f"\n#### {metric_name.replace('_', ' ').title()}\n"
                    for power, value in sorted(metric_value.items(), key=lambda x: x[1], reverse=True):
                        if isinstance(value, (int, float)):
                            report += f"- **{power}**: {value:.3f}\n"
                elif isinstance(metric_value, (int, float)):
                    report += f"- **{metric_name.replace('_', ' ').title()}**: {metric_value:.3f}\n"
        
        # Tactical and Diplomatic metrics
        tactical_metrics = metrics_data.get('tactical_metrics', {})
        diplomatic_metrics = metrics_data.get('diplomatic_metrics', {})
        
        if tactical_metrics:
            report += f"""
## Tactical Performance

"""
            for metric_name, metric_value in tactical_metrics.items():
                if isinstance(metric_value, dict):
                    report += f"### {metric_name.replace('_', ' ').title()}\n"
                    for power, value in sorted(metric_value.items(), key=lambda x: x[1], reverse=True):
                        if isinstance(value, (int, float)):
                            report += f"- **{power}**: {value:.3f}\n"
                    report += "\n"
        
        if diplomatic_metrics:
            report += f"""
## Diplomatic Performance

"""
            for metric_name, metric_value in diplomatic_metrics.items():
                if isinstance(metric_value, dict):
                    report += f"### {metric_name.replace('_', ' ').title()}\n"
                    for power, value in sorted(metric_value.items(), key=lambda x: x[1], reverse=True):
                        if isinstance(value, (int, float)):
                            report += f"- **{power}**: {value:.3f}\n"
                    report += "\n"
        
        # LLM evaluation if available
        llm_evaluation = metrics_data.get('llm_evaluation', {})
        if llm_evaluation and not llm_evaluation.get('error'):
            report += f"""
## LLM Judge Evaluation

"""
            for category in ['strategic', 'diplomatic', 'tactical', 'overall']:
                category_key = f'llm_judge_{category}'
                if category_key in llm_evaluation:
                    report += f"### {category.title()} Evaluation\n"
                    category_scores = llm_evaluation[category_key]
                    for power, score in sorted(category_scores.items(), key=lambda x: x[1], reverse=True):
                        report += f"- **{power}**: {score:.1f}/10\n"
                    report += "\n"
        
        # Behavioral analysis
        behavioral_analysis = metrics_data.get('behavioral_analysis', {})
        if behavioral_analysis:
            report += f"""
## Behavioral Analysis

"""
            for behavior_type, behavior_data in behavioral_analysis.items():
                if isinstance(behavior_data, dict):
                    report += f"### {behavior_type.replace('_', ' ').title()}\n"
                    for power, value in sorted(behavior_data.items(), key=lambda x: x[1], reverse=True):
                        if isinstance(value, (int, float)):
                            report += f"- **{power}**: {value:.3f}\n"
                    report += "\n"
        
        # Recommendations
        recommendations = detailed_report.get('recommendations', [])
        if recommendations:
            report += """
## Recommendations for Improvement

"""
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
        
        report += f"""
---
*Report generated by PolitAgent Diplomacy Metrics v2.0*
"""
        
        return report
    
    def _generate_text_report(self, metrics_data: Dict[str, Any]) -> str:
        """Generate plain text format report."""
        return f"Diplomacy Analysis Report - {metrics_data.get('games_total', 0)} games analyzed"