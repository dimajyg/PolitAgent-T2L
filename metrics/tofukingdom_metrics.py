from typing import Dict, List, Tuple, Union, Any, Set, Optional
import numpy as np
import re
import os
from datetime import datetime
import json

from metrics.base_metrics import BaseMetrics

class TofuKingdomMetrics(BaseMetrics):
    """
    Metrics collection and computation for the TofuKingdom game.
    
    This class extends BaseMetrics to handle TofuKingdom-specific metrics:
    - Prince's questioning strategy analysis
    - Truth/lie detection rates
    - Role behavior analysis
    - Team performance metrics
    - LLM evaluation of game performance
    
    Additionally includes all common metrics from BaseMetrics.
    """
    
    # TofuKingdom specific event types
    EVENT_QUESTION = "question"
    EVENT_ANSWER = "answer"
    EVENT_FINAL_GUESS = "final_guess"
    EVENT_PLAYER_ASSIGNED = "player_assigned"
    
    # Truth behavior types
    TRUTH_BEHAVIORS = {
        "Princess": "truth",
        "Chef": "truth",
        "Queen": "lie",
        "Minister": "lie",
        "Guard": "lie",
        "Maid": "choice",
        "Spy": "choice"
    }
    
    # Team definitions
    TEAM_ROLES = {
        "Princess": ["Princess", "Chef"],
        "Queen": ["Queen", "Minister", "Guard"],
        "Neutral": ["Maid", "Spy"]
    }
    
    # TofuKingdom specific LLM evaluation templates
    TOFUKINGDOM_GAME_EVALUATION_TEMPLATE = """
    Evaluate this TofuKingdom game based on the provided information:
    
    Game setup:
    - Player roles: {roles}
    - Winner team: {winner_team}
    - Prince's guess: {guess}
    
    Game summary:
    {game_summary}
    
    Please provide a detailed analysis of:
    1. Prince's questioning strategy and effectiveness (score 1-10)
    2. Each role player's adherence to their truth/lie behavior rules (score 1-10)
    3. Quality of deception/honesty by players (score 1-10)
    4. Critical questions or answers that determined the game outcome
    5. Overall game balance and fairness (score 1-10)
    
    For each player, provide an individual performance evaluation with score (1-10).
    
    Finally, suggest how the LLM agents could have played better in this game scenario.
    """
    
    TOFUKINGDOM_QUESTION_EVALUATION_TEMPLATE = """
    Evaluate this question in the TofuKingdom game:
    
    Game state:
    - Current player questioned: {player}
    - Player's actual role: {role}
    - Truth behavior: {truth_behavior}
    - Previous Q&A: {previous_qa}
    
    Current question: "{question}"
    
    Please analyze:
    1. Strategic value of this question (score 1-10)
    2. Relevance and specificity (score 1-10)
    3. How well it helps identify the Princess (score 1-10)
    
    Provide a brief analysis explaining your assessment of this question.
    """
    
    TOFUKINGDOM_ANSWER_EVALUATION_TEMPLATE = """
    Evaluate this answer in the TofuKingdom game:
    
    Game state:
    - Player answering: {player}
    - Player's actual role: {role}
    - Truth behavior required: {truth_behavior}
    - Question asked: {question}
    
    Answer given: "{answer}"
    Internal thinking (not seen by others): {thinking}
    
    Please analyze:
    1. Adherence to role's truth/lie behavior rules (score 1-10)
    2. Strategic effectiveness of the answer (score 1-10)
    3. Deception quality (for roles that must lie) or honesty quality (for roles that must tell truth) (score 1-10)
    4. How well it advances the player's team's goals (score 1-10)
    
    Provide a brief analysis explaining your assessment of this answer.
    """
    
    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the TofuKingdomMetrics collector.
        
        Args:
            metadata (Optional[Dict[str, Any]]): Additional metadata for the game session
        """
        super().__init__("tofukingdom", metadata)
        self.questions = []  # All questions asked by the Prince
        self.answers = []  # All answers given by players
        self.player_roles = {}  # Mapping of player names to roles
        self.team_assignments = {}  # Mapping of player names to teams
        self.final_guess = None  # Prince's final guess
        self.winner_team = None  # Winning team
        self.question_targets = {}  # Mapping of question indices to target players
        self.game_summary = []  # Summary of game events
        
    def set_player_role(self, player_name: str, role: str) -> None:
        """
        Record a player's assigned role.
        
        Args:
            player_name (str): Name of the player
            role (str): Role assigned to the player
        """
        self.player_roles[player_name] = role
        
        # Determine team based on role
        for team, roles in self.TEAM_ROLES.items():
            if role in roles:
                self.team_assignments[player_name] = team
                break
        
        self.record_event(
            self.EVENT_PLAYER_ASSIGNED,
            player=player_name,
            role=role,
            team=self.team_assignments.get(player_name, "Unknown")
        )
        
        self.add_metadata("player_roles", self.player_roles)
        self.add_metadata("team_assignments", self.team_assignments)
    
    def set_winner_team(self, team: str) -> None:
        """
        Set the winning team.
        
        Args:
            team (str): Name of the winning team
        """
        self.winner_team = team
        self.add_metadata("winner_team", team)
    
    def record_question(self, question: str, prince_player: str, target_player: str, 
                       round_num: int, thinking: Optional[str] = None) -> None:
        """
        Record a question asked by the Prince.
        
        Args:
            question (str): The question asked
            prince_player (str): Name of the Prince player
            target_player (str): Name of the player being questioned
            round_num (int): Round number
            thinking (Optional[str]): Prince's reasoning behind the question
        """
        question_data = {
            "question": question,
            "prince": prince_player,
            "target": target_player,
            "round": round_num,
            "thinking": thinking,
            "target_role": self.player_roles.get(target_player, "Unknown")
        }
        
        self.questions.append(question_data)
        self.question_targets[len(self.questions) - 1] = target_player
        
        self.record_event(
            self.EVENT_QUESTION,
            question=question,
            prince=prince_player,
            target=target_player,
            round=round_num,
            thinking=thinking,
            target_role=self.player_roles.get(target_player, "Unknown")
        )
        
        # Add to game summary
        self.game_summary.append(f"Prince asks {target_player}: {question}")
        
        # If LLM evaluation is enabled, evaluate this question
        if self.use_llm_evaluation:
            evaluation = self.evaluate_question(question, target_player, round_num)
            if evaluation:
                question_data["llm_evaluation"] = evaluation
    
    def record_answer(self, answer: str, player: str, question_idx: int, 
                     thinking: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an answer to a question.
        
        Args:
            answer (str): The answer given
            player (str): Player who answered
            question_idx (int): Index of the question being answered
            thinking (Optional[Dict[str, Any]]): Player's internal reasoning
        """
        # Find the corresponding question
        if question_idx >= len(self.questions):
            question = "Unknown question"
            target_player = player
        else:
            question = self.questions[question_idx]["question"]
            target_player = self.question_targets.get(question_idx, player)
        
        # Verify player matches target
        if player != target_player:
            return
        
        answer_data = {
            "answer": answer,
            "player": player,
            "question_idx": question_idx,
            "question": question,
            "role": self.player_roles.get(player, "Unknown"),
            "thinking": thinking,
            "truth_behavior": self.TRUTH_BEHAVIORS.get(self.player_roles.get(player, "Unknown"), "Unknown")
        }
        
        self.answers.append(answer_data)
        
        self.record_event(
            self.EVENT_ANSWER,
            answer=answer,
            player=player,
            question_idx=question_idx,
            question=question,
            role=self.player_roles.get(player, "Unknown"),
            thinking=thinking,
            truth_behavior=self.TRUTH_BEHAVIORS.get(self.player_roles.get(player, "Unknown"), "Unknown")
        )
        
        # Add to game summary
        self.game_summary.append(f"{player} answers: {answer}")
        
        # If LLM evaluation is enabled, evaluate this answer
        if self.use_llm_evaluation:
            evaluation = self.evaluate_answer(answer, player, question, thinking)
            if evaluation:
                answer_data["llm_evaluation"] = evaluation
    
    def record_final_guess(self, prince_player: str, guessed_player: str, 
                          actual_role: str, correct: bool, thinking: Optional[str] = None) -> None:
        """
        Record the Prince's final guess.
        
        Args:
            prince_player (str): Name of the Prince player
            guessed_player (str): Player guessed to be the Princess
            actual_role (str): Actual role of the guessed player
            correct (bool): Whether the guess was correct
            thinking (Optional[str]): Prince's reasoning behind the guess
        """
        self.final_guess = {
            "prince": prince_player,
            "guessed_player": guessed_player,
            "actual_role": actual_role,
            "correct": correct,
            "thinking": thinking
        }
        
        self.record_event(
            self.EVENT_FINAL_GUESS,
            prince=prince_player,
            guessed_player=guessed_player,
            actual_role=actual_role,
            correct=correct,
            thinking=thinking
        )
        
        # Add to game summary
        self.game_summary.append(f"Prince's final guess: {guessed_player} is the Princess")
        self.game_summary.append(f"Result: {guessed_player} is actually the {actual_role}")
        if correct:
            self.game_summary.append("The guess was CORRECT!")
        else:
            # Find the real Princess
            real_princess = None
            for player, role in self.player_roles.items():
                if role == "Princess":
                    real_princess = player
                    break
            if real_princess:
                self.game_summary.append(f"The real Princess was {real_princess}")
    
    def evaluate_game(self) -> Optional[Dict[str, Any]]:
        """
        Request an LLM evaluation of the entire game.
        
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None if LLM evaluation is disabled
        """
        if not self.use_llm_evaluation:
            return None
        
        # Format game summary
        game_summary_text = "\n".join(self.game_summary)
        
        # Format roles for display
        roles_text = ", ".join([f"{player}: {role}" for player, role in self.player_roles.items()])
        
        # Context for evaluation
        context = {
            "roles": roles_text,
            "winner_team": self.winner_team or "Unknown",
            "guess": (f"{self.final_guess['guessed_player']} "
                     f"(Actual role: {self.final_guess['actual_role']})") if self.final_guess else "No guess made",
            "game_summary": game_summary_text
        }
        
        # Request evaluation using the TofuKingdom-specific template
        return self.record_llm_evaluation("game", context, self.TOFUKINGDOM_GAME_EVALUATION_TEMPLATE)
    
    def evaluate_question(self, question: str, target_player: str, round_num: int) -> Optional[Dict[str, Any]]:
        """
        Request an LLM evaluation of a specific question.
        
        Args:
            question (str): The question to evaluate
            target_player (str): Player being questioned
            round_num (int): Current round number
            
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None if LLM evaluation is disabled
        """
        if not self.use_llm_evaluation:
            return None
        
        # Get player's role and truth behavior
        role = self.player_roles.get(target_player, "Unknown")
        truth_behavior = self.TRUTH_BEHAVIORS.get(role, "Unknown")
        
        # Get previous Q&A with this player
        previous_qa = []
        for i, q in enumerate(self.questions):
            if i < len(self.questions) - 1 and q["target"] == target_player:  # Exclude current question
                if i < len(self.answers):
                    a = self.answers[i]
                    previous_qa.append(f"Q: {q['question']}\nA: {a['answer']}")
        
        previous_qa_text = "\n\n".join(previous_qa)
        
        # Context for evaluation
        context = {
            "player": target_player,
            "role": role,
            "truth_behavior": truth_behavior,
            "previous_qa": previous_qa_text,
            "question": question
        }
        
        # Request evaluation using the TofuKingdom-specific template
        return self.record_llm_evaluation("question", context, self.TOFUKINGDOM_QUESTION_EVALUATION_TEMPLATE)
    
    def evaluate_answer(self, answer: str, player: str, question: str, 
                       thinking: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Request an LLM evaluation of a specific answer.
        
        Args:
            answer (str): The answer to evaluate
            player (str): Player who answered
            question (str): Question that was answered
            thinking (Optional[Dict[str, Any]]): Player's internal reasoning
            
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None if LLM evaluation is disabled
        """
        if not self.use_llm_evaluation:
            return None
        
        # Get player's role and truth behavior
        role = self.player_roles.get(player, "Unknown")
        truth_behavior = self.TRUTH_BEHAVIORS.get(role, "Unknown")
        
        # Format thinking for display
        thinking_text = json.dumps(thinking) if thinking else "No thinking available"
        
        # Context for evaluation
        context = {
            "player": player,
            "role": role,
            "truth_behavior": truth_behavior,
            "question": question,
            "answer": answer,
            "thinking": thinking_text
        }
        
        # Request evaluation using the TofuKingdom-specific template
        return self.record_llm_evaluation("answer", context, self.TOFUKINGDOM_ANSWER_EVALUATION_TEMPLATE)
    
    def compute_all(self) -> Dict[str, Any]:
        """
        Compute all TofuKingdom metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of computed metrics
        """
        # Call the parent method to get common metrics
        base_metrics = super().compute_all()
        
        # Add TofuKingdom-specific metrics
        tofukingdom_metrics = {
            "question_metrics": self._compute_question_metrics(),
            "answer_metrics": self._compute_answer_metrics(),
            "team_metrics": self._compute_team_metrics(),
            "prince_performance": self._compute_prince_performance(),
            "game_outcome": self._compute_game_outcome()
        }
        
        # Merge metrics
        self.computed_metrics.update(tofukingdom_metrics)
        
        return self.computed_metrics
    
    def _compute_question_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to Prince's questions.
        
        Returns:
            Dict[str, Any]: Question-related metrics
        """
        if not self.questions:
            return {"count": 0}
            
        # Count questions by type
        question_types = {
            "identity": 0,  # "What is your identity?"
            "princess_location": 0,  # "Which player is the Princess?"
            "other_identity": 0  # "What is the identity of X?"
        }
        
        for q in self.questions:
            question = q["question"].lower()
            if "which player is the princess" in question:
                question_types["princess_location"] += 1
            elif "what is your identity" in question:
                question_types["identity"] += 1
            elif "what is the identity of" in question:
                question_types["other_identity"] += 1
        
        # Questions per player
        questions_per_player = {}
        for q in self.questions:
            target = q["target"]
            if target not in questions_per_player:
                questions_per_player[target] = 0
            questions_per_player[target] += 1
        
        return {
            "count": len(self.questions),
            "types": question_types,
            "by_player": questions_per_player,
            "questions": self.questions  # Include all question data
        }
    
    def _compute_answer_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to player answers.
        
        Returns:
            Dict[str, Any]: Answer-related metrics
        """
        if not self.answers:
            return {"count": 0}
        
        # Analyze truth/lie behavior
        truth_adherence = {
            "truth_tellers": {"correct": 0, "incorrect": 0},
            "liars": {"correct": 0, "incorrect": 0},
            "choice": {"truth": 0, "lie": 0}
        }
        
        for answer in self.answers:
            role = answer.get("role", "Unknown")
            truth_behavior = self.TRUTH_BEHAVIORS.get(role, "Unknown")
            
            # Skip if role or behavior is unknown
            if truth_behavior == "Unknown":
                continue
                
            # This is a simplified heuristic - in a real implementation,
            # you'd need actual ground truth data or sophisticated NLP
            if truth_behavior == "truth":
                # For truth tellers
                truth_adherence["truth_tellers"]["correct"] += 1
            elif truth_behavior == "lie":
                # For liars
                truth_adherence["liars"]["correct"] += 1
            elif truth_behavior == "choice":
                # For choice, arbitrarily categorize (would need better analysis in practice)
                if "yes" in answer["answer"].lower() or "no" in answer["answer"].lower():
                    truth_adherence["choice"]["truth"] += 1
                else:
                    truth_adherence["choice"]["lie"] += 1
        
        return {
            "count": len(self.answers),
            "truth_adherence": truth_adherence,
            "answers": self.answers  # Include all answer data
        }
    
    def _compute_team_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to team performance.
        
        Returns:
            Dict[str, Any]: Team-related metrics
        """
        # Count players by team
        team_counts = {"Princess": 0, "Queen": 0, "Neutral": 0}
        for team in self.team_assignments.values():
            if team in team_counts:
                team_counts[team] += 1
        
        # Team communication analysis
        team_questions = {"Princess": 0, "Queen": 0, "Neutral": 0}
        team_accurate_answers = {"Princess": 0, "Queen": 0, "Neutral": 0}
        
        for q in self.questions:
            target = q["target"]
            target_team = self.team_assignments.get(target)
            if target_team in team_questions:
                team_questions[target_team] += 1
        
        # This is again a simplified heuristic
        for a in self.answers:
            player = a.get("player")
            team = self.team_assignments.get(player)
            if team not in team_accurate_answers:
                continue
                
            # For Princess team, "accurate" means truthful
            # For Queen team, "accurate" means deceptive
            # For Neutral team, both can be "accurate" depending on context
            role = self.player_roles.get(player)
            expected_behavior = self.TRUTH_BEHAVIORS.get(role)
            
            # Very simplistic analysis - would need more sophisticated NLP in practice
            # For now, just count all answers as "accurate"
            team_accurate_answers[team] += 1
            
        return {
            "team_counts": team_counts,
            "team_questions": team_questions,
            "team_answers": team_accurate_answers,
            "winning_team": self.winner_team
        }
    
    def _compute_prince_performance(self) -> Dict[str, Any]:
        """
        Compute metrics related to Prince's performance.
        
        Returns:
            Dict[str, Any]: Prince-related metrics
        """
        if not self.final_guess:
            return {"success": False}
            
        # Question strategy diversity
        question_diversity = len(set([q["question"] for q in self.questions])) / max(1, len(self.questions))
        
        # Player coverage (did Prince question all players?)
        questioned_players = set([q["target"] for q in self.questions])
        player_coverage = len(questioned_players) / max(1, len(self.player_roles) - 1)  # -1 for Prince
        
        return {
            "success": self.final_guess.get("correct", False),
            "guessed_player": self.final_guess.get("guessed_player"),
            "actual_role": self.final_guess.get("actual_role"),
            "question_diversity": question_diversity,
            "player_coverage": player_coverage,
            "questioning_balance": self._compute_questioning_balance()
        }
    
    def _compute_questioning_balance(self) -> float:
        """
        Compute how balanced the Prince's questioning was.
        
        Returns:
            float: A score from 0 to 1, where 1 means equal questions to all players
        """
        if not self.questions:
            return 0.0
            
        questions_per_player = {}
        for q in self.questions:
            target = q["target"]
            if target not in questions_per_player:
                questions_per_player[target] = 0
            questions_per_player[target] += 1
            
        # If no questions were asked, return 0
        if not questions_per_player:
            return 0.0
            
        # Calculate standard deviation of question counts
        counts = list(questions_per_player.values())
        mean = sum(counts) / len(counts)
        std_dev = (sum((x - mean) ** 2 for x in counts) / len(counts)) ** 0.5
        
        # Calculate coefficient of variation (normalized std dev)
        # A smaller CV means more balanced questioning
        if mean == 0:
            return 0.0
            
        cv = std_dev / mean
        
        # Convert to a 0-1 score where 1 is perfectly balanced
        # When CV=0, balance=1; when CV=1, balance≈0.37; when CV=2, balance≈0.14
        balance = np.exp(-cv)
        
        return balance
    
    def _compute_game_outcome(self) -> Dict[str, Any]:
        """
        Compute metrics related to the game outcome.
        
        Returns:
            Dict[str, Any]: Outcome-related metrics
        """
        return {
            "winner_team": self.winner_team,
            "princess_found": self.final_guess.get("correct", False) if self.final_guess else False,
            "teams": {
                team: {"members": [p for p, t in self.team_assignments.items() if t == team]} 
                for team in ["Princess", "Queen", "Neutral"]
            }
        } 