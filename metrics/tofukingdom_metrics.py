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
    
    EVENT_QUESTION = "question"
    EVENT_ANSWER = "answer"
    EVENT_FINAL_GUESS = "final_guess"
    EVENT_PLAYER_ASSIGNED = "player_assigned"
    
    TRUTH_BEHAVIORS = {
        "Princess": "truth",
        "Chef": "truth",
        "Queen": "lie",
        "Minister": "lie",
        "Guard": "lie",
        "Maid": "choice",
        "Spy": "choice"
    }
    
    TEAM_ROLES = {
        "Princess": ["Princess", "Chef"],
        "Queen": ["Queen", "Minister", "Guard"],
        "Neutral": ["Maid", "Spy"]
    }
    
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
        self.questions = []  
        self.answers = []  
        self.player_roles = {}  
        self.team_assignments = {}  
        self.final_guess = None  
        self.winner_team = None  
        self.question_targets = {}  
        self.game_summary = []  
        
    def set_player_role(self, player_name: str, role: str) -> None:
        """
        Record a player's assigned role.
        
        Args:
            player_name (str): Name of the player
            role (str): Role assigned to the player
        """
        self.player_roles[player_name] = role
        
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
        
        self.game_summary.append(f"Prince asks {target_player}: {question}")
        
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
        if question_idx >= len(self.questions):
            question = "Unknown question"
            target_player = player
        else:
            question = self.questions[question_idx]["question"]
            target_player = self.question_targets.get(question_idx, player)
        
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
        
        self.game_summary.append(f"{player} answers: {answer}")
        
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
        
        self.game_summary.append(f"Prince's final guess: {guessed_player} is the Princess")
        self.game_summary.append(f"Result: {guessed_player} is actually the {actual_role}")
        if correct:
            self.game_summary.append("The guess was CORRECT!")
        else:
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
        
        game_summary_text = "\n".join(self.game_summary)
        
        roles_text = ", ".join([f"{player}: {role}" for player, role in self.player_roles.items()])
        
        context = {
            "roles": roles_text,
            "winner_team": self.winner_team or "Unknown",
            "guess": (f"{self.final_guess['guessed_player']} "
                     f"(Actual role: {self.final_guess['actual_role']})") if self.final_guess else "No guess made",
            "game_summary": game_summary_text
        }
            
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

    def calculate_metrics(self, results_dir: str) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics from game result files.
        
        Args:
            results_dir: Directory containing game result files
            
        Returns:
            Dict containing all calculated metrics
        """
        game_logs = self._load_game_logs(results_dir)
        
        if not game_logs:
            return {"error": "No valid game logs found"}
        
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "games_analyzed": len(game_logs),
            "model_performance": self._calculate_model_inference_metrics(game_logs),
            "strategic_metrics": self._calculate_strategic_metrics(game_logs),
            "questioning_analysis": self._calculate_questioning_analysis(game_logs),
            "role_performance": self._calculate_role_performance_metrics(game_logs),
            "team_dynamics": self._calculate_team_dynamics_metrics(game_logs),
            "success_patterns": self._calculate_success_patterns(game_logs),
            "llm_judge_evaluation": self._calculate_llm_judge_metrics(game_logs) if hasattr(self, 'llm_model') else None
        }
        
        return self._convert_numpy_types(metrics_data)

    def _load_game_logs(self, results_dir: str) -> List[Dict[str, Any]]:
        """Load and parse TofuKingdom game log files."""
        game_logs = []
        
        if not os.path.exists(results_dir):
            return game_logs
            
        for filename in os.listdir(results_dir):
            if filename.endswith('.json') and 'tofukingdom' in filename.lower():
                try:
                    filepath = os.path.join(results_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if self._is_valid_tofukingdom_log(data):
                            game_logs.append(data)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    
        return game_logs

    def _is_valid_tofukingdom_log(self, data: Dict[str, Any]) -> bool:
        """Check if the loaded data is a valid TofuKingdom game log."""
        required_fields = ['identities', 'qa_history']
        return all(field in data for field in required_fields)

    def _calculate_model_inference_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive model inference metrics."""
        total_inferences = 0
        question_inferences = 0
        answer_inferences = 0
        guess_inferences = 0
        total_errors = 0
        
        quality_scores = []
        decision_consistency_scores = []
        
        for game in game_logs:
            qa_history = game.get('qa_history', [])
            
            # Count inferences from Q&A history
            for qa_pair in qa_history:
                if 'question' in qa_pair:
                    question_inferences += 1
                    total_inferences += 1
                    
                    # Analyze question quality
                    q_quality = self._analyze_question_quality_tofukingdom(qa_pair['question'])
                    quality_scores.append(q_quality)
                    
                if 'answer' in qa_pair:
                    answer_inferences += 1
                    total_inferences += 1
                    
                    # Analyze answer quality
                    a_quality = self._analyze_answer_quality_tofukingdom(qa_pair['answer'], qa_pair.get('player'))
                    quality_scores.append(a_quality)
            
            # Count final guess inference
            if 'prince_guess' in game:
                guess_inferences += 1
                total_inferences += 1
                
                # Analyze guess quality
                guess_quality = self._analyze_guess_quality_tofukingdom(game)
                quality_scores.append(guess_quality)
            
            # Count errors (if any error tracking is available)
            if game.get('error') or game.get('exception'):
                total_errors += 1
            
            # Analyze decision consistency
            consistency = self._analyze_decision_consistency_tofukingdom(qa_history)
            if consistency is not None:
                decision_consistency_scores.append(consistency)
        
        return {
            "total_inferences": total_inferences,
            "question_inferences": question_inferences,
            "answer_inferences": answer_inferences,
            "guess_inferences": guess_inferences,
            "average_quality_score": np.mean(quality_scores) if quality_scores else 0.0,
            "decision_consistency": np.mean(decision_consistency_scores) if decision_consistency_scores else 0.0,
            "total_errors": total_errors,
            "error_rate": total_errors / len(game_logs) if game_logs else 0.0,
            "inference_breakdown": {
                "questions": question_inferences,
                "answers": answer_inferences, 
                "guesses": guess_inferences
            }
        }

    def _analyze_question_quality_tofukingdom(self, question: str) -> float:
        """Analyze the quality of a question in TofuKingdom context."""
        if not question or len(question.strip()) == 0:
            return 0.0
        
        # Basic quality metrics
        score = 0.5  # Base score
        
        # Strategic question types
        question_lower = question.lower()
        if "princess" in question_lower:
            score += 0.3  # Direct princess questions are high value
        elif "identity" in question_lower or "role" in question_lower:
            score += 0.2  # Identity questions are valuable
        elif any(word in question_lower for word in ["truth", "lie", "honest"]):
            score += 0.15  # Truth/lie probing is moderately valuable
        
        # Question structure
        if question.strip().endswith('?'):
            score += 0.1
        
        # Length appropriateness
        word_count = len(question.split())
        if 5 <= word_count <= 25:
            score += 0.15
        
        return min(score, 1.0)

    def _analyze_answer_quality_tofukingdom(self, answer: str, player: str) -> float:
        """Analyze the quality of an answer in TofuKingdom context."""
        if not answer or len(answer.strip()) == 0:
            return 0.0
        
        # Basic quality metrics
        score = 0.5  # Base score
        
        # Response appropriateness
        word_count = len(answer.split())
        if 3 <= word_count <= 50:
            score += 0.3
        
        # Check for role-appropriate behavior (simplified)
        answer_lower = answer.lower()
        if any(word in answer_lower for word in ['yes', 'no', 'true', 'false']):
            score += 0.1  # Clear responses are good
        
        # Avoid revealing too much information
        if not any(word in answer_lower for word in ['princess', 'queen', 'spy', 'guard', 'minister', 'chef', 'maid']):
            score += 0.1  # Not revealing roles directly
        
        return min(score, 1.0)

    def _analyze_guess_quality_tofukingdom(self, game: Dict[str, Any]) -> float:
        """Analyze the quality of the final guess."""
        prince_guess = game.get('prince_guess', {})
        is_correct = prince_guess.get('correct', False)
        
        if is_correct:
            return 1.0  # Perfect score for correct guess
        else:
            return 0.2  # Low score for incorrect guess

    def _analyze_decision_consistency_tofukingdom(self, qa_history: List[Dict[str, Any]]) -> Optional[float]:
        """Analyze consistency of decisions throughout the game."""
        if len(qa_history) < 2:
            return None
        
        # Analyze if questions become more targeted over time
        princess_focus_scores = []
        for qa in qa_history:
            if 'question' in qa:
                question = qa['question'].lower()
                # Score based on how focused the question is on finding the princess
                focus_score = 0.0
                if "princess" in question:
                    focus_score = 1.0
                elif "identity" in question:
                    focus_score = 0.7
                elif any(word in question for word in ["role", "truth", "lie"]):
                    focus_score = 0.5
                else:
                    focus_score = 0.2
                
                princess_focus_scores.append(focus_score)
        
        if len(princess_focus_scores) < 2:
            return 0.5
        
        # Calculate if there's increasing focus (good strategy)
        trend_score = 0.0
        for i in range(1, len(princess_focus_scores)):
            if princess_focus_scores[i] >= princess_focus_scores[i-1]:
                trend_score += 1.0
        
        return trend_score / (len(princess_focus_scores) - 1)

    def _calculate_strategic_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate strategic performance metrics."""
        success_rate = 0
        princess_team_wins = 0
        queen_team_wins = 0
        neutral_impact = 0
        
        for game in game_logs:
            winner = game.get('winner')
            prince_guess = game.get('prince_guess', {})
            
            if prince_guess.get('correct', False):
                success_rate += 1
                princess_team_wins += 1
            else:
                queen_team_wins += 1
        
        success_rate = success_rate / len(game_logs) if game_logs else 0
        
        return {
            "prince_success_rate": success_rate,
            "princess_team_wins": princess_team_wins,
            "queen_team_wins": queen_team_wins,
            "total_games": len(game_logs),
            "game_balance": abs(0.5 - success_rate),  # How balanced the game is
            "strategic_depth": self._calculate_strategic_depth(game_logs)
        }

    def _calculate_strategic_depth(self, game_logs: List[Dict[str, Any]]) -> float:
        """Calculate how strategically deep the games were."""
        depth_scores = []
        
        for game in game_logs:
            qa_history = game.get('qa_history', [])
            questions = [qa.get('question', '') for qa in qa_history if 'question' in qa]
            
            if not questions:
                depth_scores.append(0.0)
                continue
            
            # Measure strategic depth by question diversity and complexity
            unique_question_ratio = len(set(questions)) / len(questions) if questions else 0
            avg_question_length = np.mean([len(q.split()) for q in questions])
            strategic_words = sum(1 for q in questions if any(word in q.lower() for word in 
                                ['princess', 'identity', 'truth', 'lie', 'role', 'honest']))
            strategic_ratio = strategic_words / len(questions) if questions else 0
            
            depth = (unique_question_ratio * 0.4 + 
                    min(avg_question_length / 15.0, 1.0) * 0.3 + 
                    strategic_ratio * 0.3)
            depth_scores.append(depth)
        
        return np.mean(depth_scores) if depth_scores else 0.0

    def _calculate_questioning_analysis(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate questioning strategy analysis."""
        all_questions = []
        question_targets = []
        question_timing = []
        
        for game in game_logs:
            qa_history = game.get('qa_history', [])
            for i, qa in enumerate(qa_history):
                if 'question' in qa:
                    all_questions.append(qa['question'])
                    question_targets.append(qa.get('player', 'Unknown'))
                    question_timing.append(i + 1)  # Question order
        
        if not all_questions:
            return {"total_questions": 0}
        
        # Question type analysis
        question_types = {
            "direct_princess": sum(1 for q in all_questions if "princess" in q.lower()),
            "identity_probe": sum(1 for q in all_questions if "identity" in q.lower() or "role" in q.lower()),
            "truth_lie_probe": sum(1 for q in all_questions if any(word in q.lower() for word in ["truth", "lie", "honest"])),
            "general_inquiry": 0
        }
        question_types["general_inquiry"] = len(all_questions) - sum(question_types.values())
        
        # Target distribution
        target_distribution = {}
        for target in question_targets:
            target_distribution[target] = target_distribution.get(target, 0) + 1
        
        return {
            "total_questions": len(all_questions),
            "question_types": question_types,
            "target_distribution": target_distribution,
            "average_questions_per_game": len(all_questions) / len(game_logs) if game_logs else 0,
            "question_diversity": len(set(all_questions)) / len(all_questions) if all_questions else 0
        }

    def _calculate_role_performance_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics by role."""
        role_stats = {}
        
        for game in game_logs:
            identities = game.get('identities', {})
            qa_history = game.get('qa_history', [])
            
            # Initialize role stats
            for player, role in identities.items():
                if role not in role_stats:
                    role_stats[role] = {
                        "questions_received": 0,
                        "answers_given": 0,
                        "detection_rate": 0,
                        "performance_score": []
                    }
            
            # Count interactions
            for qa in qa_history:
                if 'player' in qa and 'answer' in qa:
                    player = qa['player']
                    role = identities.get(player, 'Unknown')
                    if role in role_stats:
                        role_stats[role]["answers_given"] += 1
                        
                        # Analyze answer quality for this role
                        answer_quality = self._analyze_answer_quality_tofukingdom(qa['answer'], player)
                        role_stats[role]["performance_score"].append(answer_quality)
                
                if 'question' in qa and 'player' in qa:
                    player = qa['player']
                    role = identities.get(player, 'Unknown')
                    if role in role_stats:
                        role_stats[role]["questions_received"] += 1
        
        # Calculate averages
        for role in role_stats:
            scores = role_stats[role]["performance_score"]
            role_stats[role]["average_performance"] = np.mean(scores) if scores else 0.0
            role_stats[role]["performance_count"] = len(scores)
        
        return role_stats

    def _calculate_team_dynamics_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate team dynamics and cooperation metrics."""
        team_performance = {
            "Princess": {"wins": 0, "games": 0, "avg_survival": 0},
            "Queen": {"wins": 0, "games": 0, "avg_detection": 0},
            "Neutral": {"wins": 0, "games": 0, "influence": 0}
        }
        
        for game in game_logs:
            winner = game.get('winner', 'Unknown')
            prince_guess = game.get('prince_guess', {})
            
            # Count team games
            for team in team_performance:
                team_performance[team]["games"] += 1
            
            # Record wins
            if prince_guess.get('correct', False):
                team_performance["Princess"]["wins"] += 1
            else:
                team_performance["Queen"]["wins"] += 1
        
        # Calculate win rates
        for team in team_performance:
            games = team_performance[team]["games"]
            if games > 0:
                team_performance[team]["win_rate"] = team_performance[team]["wins"] / games
            else:
                team_performance[team]["win_rate"] = 0.0
        
        return team_performance

    def _calculate_success_patterns(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in successful vs failed games."""
        successful_patterns = {"avg_questions": 0, "common_strategies": []}
        failed_patterns = {"avg_questions": 0, "common_errors": []}
        
        successful_games = []
        failed_games = []
        
        for game in game_logs:
            qa_count = len(game.get('qa_history', []))
            prince_guess = game.get('prince_guess', {})
            is_success = prince_guess.get('correct', False)
            
            if is_success:
                successful_games.append(qa_count)
            else:
                failed_games.append(qa_count)
        
        successful_patterns["avg_questions"] = np.mean(successful_games) if successful_games else 0
        failed_patterns["avg_questions"] = np.mean(failed_games) if failed_games else 0
        
        return {
            "successful_patterns": successful_patterns,
            "failed_patterns": failed_patterns,
            "success_vs_failure_analysis": {
                "avg_questions_success": successful_patterns["avg_questions"],
                "avg_questions_failure": failed_patterns["avg_questions"],
                "success_efficiency": (10.0 - successful_patterns["avg_questions"]) / 10.0 if successful_patterns["avg_questions"] > 0 else 0
            }
        }

    def _calculate_llm_judge_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate LLM judge evaluation metrics if available."""
        return {
            "note": "LLM judge evaluation not implemented - requires LLM model configuration"
        }

    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
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
        else:
            return obj

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
        report = "# TofuKingdom Game Analysis Report\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        model_perf = metrics_data.get("model_performance", {})
        strategic = metrics_data.get("strategic_metrics", {})
        
        report += f"**Games Analyzed:** {metrics_data.get('games_analyzed', 0)}\n"
        report += f"**Total Model Inferences:** {model_perf.get('total_inferences', 0)}\n"
        report += f"**Prince Success Rate:** {strategic.get('prince_success_rate', 0):.1%}\n"
        report += f"**Average Quality Score:** {model_perf.get('average_quality_score', 0):.2f}/1.0\n"
        report += f"**Decision Consistency:** {model_perf.get('decision_consistency', 0):.2f}/1.0\n\n"
        
        # Model Performance
        report += "## Model Inference Performance\n\n"
        report += f"- **Total Inferences:** {model_perf.get('total_inferences', 0)}\n"
        
        breakdown = model_perf.get('inference_breakdown', {})
        report += f"  - Questions: {breakdown.get('questions', 0)}\n"
        report += f"  - Answers: {breakdown.get('answers', 0)}\n"
        report += f"  - Guesses: {breakdown.get('guesses', 0)}\n"
        
        report += f"- **Error Rate:** {model_perf.get('error_rate', 0):.1%}\n"
        report += f"- **Average Quality Score:** {model_perf.get('average_quality_score', 0):.2f}/1.0\n\n"
        
        # Strategic Analysis
        report += "## Strategic Performance\n\n"
        report += f"- **Prince Success Rate:** {strategic.get('prince_success_rate', 0):.1%}\n"
        report += f"- **Princess Team Wins:** {strategic.get('princess_team_wins', 0)}\n"
        report += f"- **Queen Team Wins:** {strategic.get('queen_team_wins', 0)}\n"
        report += f"- **Game Balance Score:** {strategic.get('game_balance', 0):.2f} (lower is more balanced)\n"
        report += f"- **Strategic Depth:** {strategic.get('strategic_depth', 0):.2f}/1.0\n\n"
        
        # Questioning Analysis
        report += "## Questioning Strategy Analysis\n\n"
        questioning = metrics_data.get("questioning_analysis", {})
        report += f"- **Total Questions:** {questioning.get('total_questions', 0)}\n"
        report += f"- **Average Questions per Game:** {questioning.get('average_questions_per_game', 0):.1f}\n"
        report += f"- **Question Diversity:** {questioning.get('question_diversity', 0):.2f}\n\n"
        
        question_types = questioning.get("question_types", {})
        if question_types:
            report += "### Question Type Distribution:\n"
            report += f"- **Direct Princess Questions:** {question_types.get('direct_princess', 0)}\n"
            report += f"- **Identity Probes:** {question_types.get('identity_probe', 0)}\n"
            report += f"- **Truth/Lie Probes:** {question_types.get('truth_lie_probe', 0)}\n"
            report += f"- **General Inquiries:** {question_types.get('general_inquiry', 0)}\n\n"
        
        # Role Performance
        report += "## Role Performance Analysis\n\n"
        role_perf = metrics_data.get("role_performance", {})
        for role, stats in role_perf.items():
            report += f"### {role}\n"
            report += f"- **Questions Received:** {stats.get('questions_received', 0)}\n"
            report += f"- **Answers Given:** {stats.get('answers_given', 0)}\n"
            report += f"- **Average Performance:** {stats.get('average_performance', 0):.2f}/1.0\n\n"
        
        # Team Dynamics
        report += "## Team Dynamics\n\n"
        team_dynamics = metrics_data.get("team_dynamics", {})
        for team, stats in team_dynamics.items():
            if team in ["Princess", "Queen"]:
                report += f"### {team} Team\n"
                report += f"- **Win Rate:** {stats.get('win_rate', 0):.1%}\n"
                report += f"- **Games Played:** {stats.get('games', 0)}\n\n"
        
        # Success Patterns
        report += "## Success Patterns\n\n"
        patterns = metrics_data.get("success_patterns", {})
        success_pat = patterns.get("successful_patterns", {})
        failure_pat = patterns.get("failed_patterns", {})
        
        report += f"- **Successful Games - Avg Questions:** {success_pat.get('avg_questions', 0):.1f}\n"
        report += f"- **Failed Games - Avg Questions:** {failure_pat.get('avg_questions', 0):.1f}\n"
        
        analysis = patterns.get("success_vs_failure_analysis", {})
        report += f"- **Success Efficiency:** {analysis.get('success_efficiency', 0):.2f}/1.0\n\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        success_rate = strategic.get('prince_success_rate', 0)
        quality = model_perf.get('average_quality_score', 0)
        depth = strategic.get('strategic_depth', 0)
        
        if success_rate < 0.4:
            report += "- **Improve Success Rate:** Focus on better princess identification strategy\n"
        if quality < 0.6:
            report += "- **Enhance Question Quality:** Develop more strategic and targeted questions\n"
        if depth < 0.5:
            report += "- **Increase Strategic Depth:** Use more diverse and complex questioning approaches\n"
        
        balance = strategic.get('game_balance', 0)
        if balance > 0.3:
            report += "- **Game Balance:** Consider adjusting difficulty or strategy complexity\n"
        
        report += f"\n---\n*Report generated on {metrics_data.get('timestamp', 'Unknown')}*\n"
        
        return report

    def _generate_text_report(self, metrics_data: Dict[str, Any]) -> str:
        """Generate plain text format report."""
        return self._generate_markdown_report(metrics_data).replace('#', '').replace('*', '')