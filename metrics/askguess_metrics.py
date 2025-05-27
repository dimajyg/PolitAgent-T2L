from typing import Dict, List, Any, Optional
import numpy as np
import os
from datetime import datetime

from metrics.base_metrics import BaseMetrics

class AskGuessMetrics(BaseMetrics):
    """
    Metrics collection and computation for the AskGuess game.
    
    This class extends BaseMetrics to handle AskGuess-specific metrics:
    - Question effectiveness and relevance
    - Speed of convergence to correct answer
    - Analysis of question-answering patterns
    - LLM evaluation of game performance
    
    Additionally includes all common metrics from BaseMetrics.
    """
    
    EVENT_QUESTION = "question"
    EVENT_ANSWER = "answer"
    EVENT_GUESS = "guess"
    EVENT_CORRECT_GUESS = "correct_guess"
    EVENT_INCORRECT_GUESS = "incorrect_guess"
    
    ASKGUESS_GAME_EVALUATION_TEMPLATE = """
    Evaluate this AskGuess game based on the provided information:
    
    Game setup:
    - Target word/concept: {target_word}
    - Game mode: {game_mode}
    - Total rounds: {total_rounds}
    
    Game summary:
    {game_summary}
    
    Please provide a detailed analysis of:
    1. Question quality and strategy (score 1-10)
    2. Efficiency of information gathering (score 1-10)
    3. Deduction and reasoning process (score 1-10)
    4. Overall performance (score 1-10)
    5. Key moments that impacted the outcome
    
    If the word was guessed correctly, analyze how efficiently the player reached the answer.
    If the word was not guessed correctly, identify what went wrong in the questioning strategy.
    
    Finally, suggest how the LLM agent could improve its strategy in similar games.
    """
    
    ASKGUESS_QUESTION_EVALUATION_TEMPLATE = """
    Evaluate this question in the AskGuess game:
    
    Game state:
    - Target word/concept: {target_word}
    - Current round: {current_round} of {total_rounds}
    - Previous Q&A: {previous_qa}
    
    Current question: "{question}"
    
    Please analyze:
    1. Relevance of the question (score 1-10)
    2. Information value (how much this question narrows down possibilities) (score 1-10)
    3. Strategic value at this point in the game (score 1-10)
    
    Provide a brief analysis explaining your assessment of this question.
    """
    
    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the AskGuessMetrics collector.
        
        Args:
            metadata (Optional[Dict[str, Any]]): Additional metadata for the game session
        """
        super().__init__("askguess", metadata)
        self.questions = []
        self.answers = []
        self.guesses = []
        self.target_word = None
        self.correct_guess = None
        self.correct_round = None
        self.total_rounds = 0
        self.game_mode = "standard"
        self.guessing_strategy = None
    
    def set_target_word(self, word: str) -> None:
        """
        Set the target word for the game.
        
        Args:
            word (str): The target word to be guessed
        """
        self.target_word = word
        self.add_metadata("target_word", word)
    
    def set_game_mode(self, mode: str) -> None:
        """
        Set the game mode.
        
        Args:
            mode (str): Game mode (e.g., "standard", "hard")
        """
        self.game_mode = mode
        self.add_metadata("game_mode", mode)
    
    def set_total_rounds(self, rounds: int) -> None:
        """
        Set the total number of rounds for the game.
        
        Args:
            rounds (int): Total allowed rounds
        """
        self.total_rounds = rounds
        self.add_metadata("total_rounds", rounds)
    
    def evaluate_game(self) -> Optional[Dict[str, Any]]:
        """
        Request an LLM evaluation of the entire game.
        
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None if LLM evaluation is disabled
        """
        if not self.use_llm_evaluation:
            return None
            
        question_events = [e for e in self.events if e["type"] == self.EVENT_QUESTION]
        answer_events = [e for e in self.events if e["type"] == self.EVENT_ANSWER]
        guess_events = [e for e in self.events if e["type"] in [self.EVENT_GUESS, self.EVENT_CORRECT_GUESS, self.EVENT_INCORRECT_GUESS]]
        
        qa_exchanges = []
        for i, (q_event, a_event) in enumerate(zip(question_events, answer_events)):
            question = q_event["data"].get("question", "No question")
            answer = a_event["data"].get("answer", "No answer")
            qa_exchanges.append(f"Round {i+1}:\nQ: {question}\nA: {answer}")
        
        guess_summary = []
        for event in guess_events:
            round_num = event["data"].get("round", "?")
            guess = event["data"].get("guess", "No guess")
            is_correct = event["type"] == self.EVENT_CORRECT_GUESS
            result = "CORRECT" if is_correct else "INCORRECT"
            guess_summary.append(f"Round {round_num} Guess: {guess} - {result}")
        
        game_summary = "Q&A Exchanges:\n" + "\n\n".join(qa_exchanges) + "\n\n"
        game_summary += "Guesses:\n" + "\n".join(guess_summary)
        
        if self.correct_guess:
            game_summary += f"\n\nOutcome: Successfully guessed the word in round {self.correct_round} out of {self.total_rounds}."
        else:
            game_summary += f"\n\nOutcome: Failed to guess the word within {self.total_rounds} rounds."
        
        context = {
            "target_word": self.target_word,
            "game_mode": self.game_mode,
            "total_rounds": self.total_rounds,
            "game_summary": game_summary
        }
        
        return self.record_llm_evaluation("game", context, self.ASKGUESS_GAME_EVALUATION_TEMPLATE)
    
    def evaluate_question(self, question: str, round_num: int) -> Optional[Dict[str, Any]]:
        """
        Request an LLM evaluation of a specific question.
        
        Args:
            question (str): The question to evaluate
            round_num (int): Current round number
            
        Returns:
            Optional[Dict[str, Any]]: Evaluation results or None if LLM evaluation is disabled
        """
        if not self.use_llm_evaluation:
            return None
            
        previous_qa = []
        for i, (q, a) in enumerate(zip(self.questions, self.answers)):
            if i < round_num - 1:
                previous_qa.append(f"Q{i+1}: {q['question']}\nA{i+1}: {a['answer']}")
        
        previous_qa_text = "\n\n".join(previous_qa)
        
        context = {
            "target_word": self.target_word,
            "current_round": round_num,
            "total_rounds": self.total_rounds,
            "previous_qa": previous_qa_text,
            "question": question
        }
        
        return self.record_llm_evaluation("question", context, self.ASKGUESS_QUESTION_EVALUATION_TEMPLATE)
    
    def record_question(self, question: str, round_num: int, thinking: Optional[str] = None) -> None:
        """
        Record a question asked by the agent.
        
        Args:
            question (str): The question asked
            round_num (int): Round number
            thinking (Optional[str]): Agent's reasoning behind the question
        """
        question_data = {
            "question": question,
            "round": round_num,
            "thinking": thinking
        }
        
        self.questions.append(question_data)
        
        self.record_event(
            self.EVENT_QUESTION,
            question=question,
            round=round_num,
            thinking=thinking
        )
        
        if self.use_llm_evaluation:
            evaluation = self.evaluate_question(question, round_num)
            if evaluation:
                question_data["llm_evaluation"] = evaluation
    
    def record_answer(self, answer: str, round_num: int) -> None:
        """
        Record an answer to a question.
        
        Args:
            answer (str): The answer given
            round_num (int): Round number
        """
        answer_data = {
            "answer": answer,
            "round": round_num
        }
        
        self.answers.append(answer_data)
        
        self.record_event(
            self.EVENT_ANSWER,
            answer=answer,
            round=round_num
        )
    
    def record_guess(self, guess: str, is_correct: bool, round_num: int, thinking: Optional[str] = None) -> None:
        """
        Record a guess made by the agent.
        
        Args:
            guess (str): The guessed word
            is_correct (bool): Whether the guess was correct
            round_num (int): Round number
            thinking (Optional[str]): Agent's reasoning behind the guess
        """
        guess_data = {
            "guess": guess,
            "is_correct": is_correct,
            "round": round_num,
            "thinking": thinking
        }
        
        self.guesses.append(guess_data)
        
        if is_correct:
            self.correct_guess = guess
            self.correct_round = round_num
            
            self.record_event(
                self.EVENT_CORRECT_GUESS,
                guess=guess,
                round=round_num,
                thinking=thinking
            )
        else:
            self.record_event(
                self.EVENT_INCORRECT_GUESS,
                guess=guess,
                round=round_num,
                thinking=thinking
            )
    
    def record_guessing_strategy(self, strategy: str) -> None:
        """
        Record the overall guessing strategy used by the agent.
        
        Args:
            strategy (str): Description of the strategy
        """
        self.guessing_strategy = strategy
        self.add_metadata("guessing_strategy", strategy)
    
    def compute_all(self) -> Dict[str, Any]:
        """
        Compute all AskGuess metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of computed metrics
        """
        base_metrics = super().compute_all()
        
        askguess_metrics = {
            "qa_metrics": self._compute_qa_metrics(),
            "guessing_metrics": self._compute_guessing_metrics(),
            "convergence_metrics": self._compute_convergence_metrics(),
            "game_outcome": {
                "target_word": self.target_word,
                "correct_guess": self.correct_guess,
                "correct_round": self.correct_round,
                "success": self.correct_guess is not None,
                "completion_percentage": 100 if self.correct_guess else (self.correct_round or self.total_rounds) * 100 / self.total_rounds
            }
        }
        
        self.computed_metrics.update(askguess_metrics)
        
        return self.computed_metrics
    
    def _compute_qa_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to questions and answers.
        
        Returns:
            Dict[str, Any]: Q&A-related metrics
        """
        qa_metrics = {
            "questions": self.questions,
            "answers": self.answers,
            "question_count": len(self.questions),
            "unique_question_count": len(set(q["question"] for q in self.questions)),
            "avg_question_length": sum(len(q["question"].split()) for q in self.questions) / len(self.questions) if self.questions else 0,
            "question_similarity": self._compute_question_similarity()
        }
        
        yes_count = sum(1 for a in self.answers if a["answer"].lower() in ["yes", "true", "correct", "right"])
        no_count = sum(1 for a in self.answers if a["answer"].lower() in ["no", "false", "incorrect", "wrong"])
        
        qa_metrics["yes_ratio"] = yes_count / len(self.answers) if self.answers else 0
        qa_metrics["no_ratio"] = no_count / len(self.answers) if self.answers else 0
        qa_metrics["binary_answer_ratio"] = (yes_count + no_count) / len(self.answers) if self.answers else 0
        
        qa_metrics["information_gain"] = self._estimate_information_gain()
        
        return qa_metrics
    
    def _compute_guessing_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to guesses.
        
        Returns:
            Dict[str, Any]: Guessing-related metrics
        """
        guessing_metrics = {
            "guesses": self.guesses,
            "guess_count": len(self.guesses),
            "correct_guess": self.correct_guess,
            "guessing_strategy": self.guessing_strategy
        }
        
        if self.correct_round:
            guessing_metrics["success_round"] = self.correct_round
            guessing_metrics["efficiency"] = 1.0 - ((self.correct_round - 1) / self.total_rounds)
            guessing_metrics["questions_before_success"] = self.correct_round - 1
        else:
            guessing_metrics["success_round"] = None
            guessing_metrics["efficiency"] = 0.0
            guessing_metrics["questions_before_success"] = self.total_rounds
        
        return guessing_metrics
    
    def _compute_convergence_metrics(self) -> Dict[str, Any]:
        """
        Compute metrics related to convergence to the correct answer.
        
        Returns:
            Dict[str, Any]: Convergence-related metrics
        """
        convergence_metrics = {
            "rounds_played": len(self.questions),
            "converged": self.correct_guess is not None,
            "convergence_speed": 1.0 - ((self.correct_round - 1) / self.total_rounds) if self.correct_round else 0.0
        }
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            if self.guesses:
                guess_texts = [g["guess"] for g in self.guesses]
                
                corpus = [self.target_word] + guess_texts
                
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(corpus)
                
                similarities = []
                target_vector = tfidf_matrix[0:1]
                
                for i in range(1, len(corpus)):
                    guess_vector = tfidf_matrix[i:i+1]
                    similarity = (target_vector * guess_vector.T).A[0][0]
                    similarities.append(similarity)
                
                convergence_metrics["guess_similarities"] = similarities
                convergence_metrics["avg_guess_similarity"] = sum(similarities) / len(similarities)
                convergence_metrics["max_guess_similarity"] = max(similarities)
        except:
            pass
        
        return convergence_metrics
    
    def _compute_question_similarity(self) -> Dict[str, float]:
        """
        Compute similarity between consecutive questions.
        
        Returns:
            Dict[str, float]: Question similarity metrics
        """
        if len(self.questions) < 2:
            return {"avg_similarity": 0, "max_similarity": 0, "min_similarity": 0}
            
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            question_texts = [q["question"] for q in self.questions]
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(question_texts)
            
            similarities = []
            for i in range(len(question_texts) - 1):
                q1_vector = tfidf_matrix[i:i+1]
                q2_vector = tfidf_matrix[i+1:i+2]
                similarity = (q1_vector * q2_vector.T).A[0][0]
                similarities.append(similarity)
            
            return {
                "avg_similarity": sum(similarities) / len(similarities),
                "max_similarity": max(similarities),
                "min_similarity": min(similarities)
            }
        except:
            return {"avg_similarity": 0, "max_similarity": 0, "min_similarity": 0}
    
    def _estimate_information_gain(self) -> Dict[str, float]:
        """
        Estimate information gain from questions and answers.
        This is a proxy for how much each question narrows down the search space.
        
        Returns:
            Dict[str, float]: Information gain metrics
        """
        if not self.questions or not self.answers:
            return {"avg_gain": 0, "cumulative_gain": 0}
            
        binary_answers = []
        
        for answer in self.answers:
            text = answer["answer"].lower()
            if text in ["yes", "true", "correct", "right", "no", "false", "incorrect", "wrong"]:
                binary_answers.append(1.0)
            else:
                binary_answers.append(0.5)
        
        gains = []
        for i, is_binary in enumerate(binary_answers):
            position_factor = (i + 1) / len(binary_answers)
            gain = is_binary * (0.5 + 0.5 * position_factor)
            gains.append(gain)
        
        return {
            "avg_gain": sum(gains) / len(gains),
            "cumulative_gain": sum(gains),
            "normalized_gain": sum(gains) / len(gains) / 1.0,
            "by_question": gains
        }

    def calculate_metrics(self, results_dir: str) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics from game result files.
        
        Args:
            results_dir: Directory containing game result files
            
        Returns:
            Dict containing all calculated metrics
        """
        import json
        
        game_logs = self._load_game_logs(results_dir)
        
        if not game_logs:
            return {"error": "No valid game logs found"}
        
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "games_analyzed": len(game_logs),
            "model_performance": self._calculate_model_inference_metrics(game_logs),
            "strategic_metrics": self._calculate_strategic_metrics(game_logs),
            "question_quality": self._calculate_question_quality_metrics(game_logs),
            "convergence_analysis": self._calculate_convergence_analysis(game_logs),
            "success_patterns": self._calculate_success_patterns(game_logs),
            "llm_judge_evaluation": self._calculate_llm_judge_metrics(game_logs) if hasattr(self, 'llm_model') else None
        }
        
        return self._convert_numpy_types(metrics_data)

    def _load_game_logs(self, results_dir: str) -> List[Dict[str, Any]]:
        """Load and parse AskGuess game log files."""
        import json
        
        game_logs = []
        
        if not os.path.exists(results_dir):
            return game_logs
            
        for filename in os.listdir(results_dir):
            if filename.endswith('.json') and 'askguess' in filename.lower():
                try:
                    filepath = os.path.join(results_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if self._is_valid_askguess_log(data):
                            game_logs.append(data)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    
        return game_logs

    def _is_valid_askguess_log(self, data: Dict[str, Any]) -> bool:
        """Check if the loaded data is a valid AskGuess game log."""
        required_fields = ['object', 'qa_history']
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
            
            for qa_pair in qa_history:
                if 'question' in qa_pair:
                    question_inferences += 1
                    total_inferences += 1
                    
                    q_quality = self._analyze_question_quality(qa_pair['question'])
                    quality_scores.append(q_quality)
                    
                if 'answer' in qa_pair:
                    answer_inferences += 1
                    total_inferences += 1
                    
                    a_quality = self._analyze_answer_quality(qa_pair['answer'])
                    quality_scores.append(a_quality)
            
            if game.get('round', -1) >= 0:
                guess_inferences += 1
                total_inferences += 1
                
                guess_quality = self._analyze_guess_quality(game)
                quality_scores.append(guess_quality)
            
            if 'error_type' in game and game['error_type'] != 'SuccessfulTrial':
                total_errors += 1
            
            consistency = self._analyze_decision_consistency_askguess(qa_history)
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

    def _analyze_question_quality(self, question: str) -> float:
        """Analyze the quality of a question."""
        if not question or len(question.strip()) == 0:
            return 0.0
        
        score = 0.5
        
        word_count = len(question.split())
        if 5 <= word_count <= 20:
            score += 0.2
        
        question_words = ['what', 'where', 'when', 'who', 'why', 'how', 'is', 'are', 'can', 'does', 'do']
        if any(word in question.lower() for word in question_words):
            score += 0.2
        
        if question.strip().endswith('?'):
            score += 0.1
        
        return min(score, 1.0)

    def _analyze_answer_quality(self, answer: str) -> float:
        """Analyze the quality of an answer."""
        if not answer or len(answer.strip()) == 0:
            return 0.0
        
        score = 0.5
        
        word_count = len(answer.split())
        if 3 <= word_count <= 30:
            score += 0.3
        
        answer_lower = answer.lower()
        if any(word in answer_lower for word in ['yes', 'no', 'maybe', 'sometimes', 'usually']):
            score += 0.2
        
        return min(score, 1.0)

    def _analyze_guess_quality(self, game: Dict[str, Any]) -> float:
        """Analyze the quality of the final guess."""
        is_correct = game.get('error_type') == 'SuccessfulTrial'
        round_num = game.get('round', -1)
        max_rounds = 10
        
        if is_correct:
            efficiency_bonus = (max_rounds - round_num) / max_rounds if round_num >= 0 else 0
            return 0.7 + 0.3 * efficiency_bonus
        else:
            return 0.3

    def _analyze_decision_consistency_askguess(self, qa_history: List[Dict[str, Any]]) -> Optional[float]:
        """Analyze consistency of decisions throughout the game."""
        if len(qa_history) < 2:
            return None

        question_specificity_scores = []
        for qa in qa_history:
            if 'question' in qa:
                question = qa['question']
                specificity = len(question.split()) / 20.0
                question_specificity_scores.append(min(specificity, 1.0))
        
        if len(question_specificity_scores) < 2:
            return 0.5
        
        trend_score = 0.0
        for i in range(1, len(question_specificity_scores)):
            if question_specificity_scores[i] >= question_specificity_scores[i-1]:
                trend_score += 1.0
        
        return trend_score / (len(question_specificity_scores) - 1)

    def _calculate_strategic_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate strategic performance metrics."""
        success_rate = 0
        avg_rounds_to_success = 0
        successful_games = []
        
        for game in game_logs:
            is_success = game.get('error_type') == 'SuccessfulTrial'
            if is_success:
                success_rate += 1
                round_num = game.get('round', -1)
                if round_num >= 0:
                    successful_games.append(round_num)
        
        success_rate = success_rate / len(game_logs) if game_logs else 0
        avg_rounds_to_success = np.mean(successful_games) if successful_games else 0
        
        return {
            "success_rate": success_rate,
            "average_rounds_to_success": avg_rounds_to_success,
            "successful_games": len(successful_games),
            "failed_games": len(game_logs) - len(successful_games),
            "efficiency_score": (1.0 - avg_rounds_to_success / 10.0) if avg_rounds_to_success > 0 else 0.0
        }

    def _calculate_question_quality_metrics(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate question quality metrics."""
        all_questions = []
        
        for game in game_logs:
            qa_history = game.get('qa_history', [])
            for qa in qa_history:
                if 'question' in qa:
                    all_questions.append(qa['question'])
        
        if not all_questions:
            return {"average_length": 0, "question_diversity": 0, "total_questions": 0}
        
        avg_length = np.mean([len(q.split()) for q in all_questions])
        
        unique_questions = len(set(all_questions))
        diversity = unique_questions / len(all_questions) if all_questions else 0
        
        return {
            "average_length": avg_length,
            "question_diversity": diversity,
            "total_questions": len(all_questions),
            "unique_questions": unique_questions
        }

    def _calculate_convergence_analysis(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well the model converges to the correct answer."""
        convergence_patterns = []
        
        for game in game_logs:
            qa_history = game.get('qa_history', [])
            if len(qa_history) < 2:
                continue
            
            info_score = 0
            for i, qa in enumerate(qa_history):
                if 'question' in qa and 'answer' in qa:
                    answer_detail = len(qa['answer'].split()) / 10.0
                    info_score += min(answer_detail, 1.0)
            
            convergence_patterns.append(info_score / len(qa_history) if qa_history else 0)
        
        return {
            "average_convergence_score": np.mean(convergence_patterns) if convergence_patterns else 0,
            "convergence_patterns": len(convergence_patterns)
        }

    def _calculate_success_patterns(self, game_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in successful vs failed games."""
        successful_patterns = {"avg_questions": 0, "common_strategies": []}
        failed_patterns = {"avg_questions": 0, "common_errors": []}
        
        successful_games = []
        failed_games = []
        
        for game in game_logs:
            qa_count = len(game.get('qa_history', []))
            is_success = game.get('error_type') == 'SuccessfulTrial'
            
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
                "avg_questions_failure": failed_patterns["avg_questions"]
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
        import json
        
        if format_type == "markdown":
            return self._generate_markdown_report(metrics_data)
        elif format_type == "json":
            return json.dumps(metrics_data, indent=2)
        else:
            return self._generate_text_report(metrics_data)

    def _generate_markdown_report(self, metrics_data: Dict[str, Any]) -> str:
        """Generate markdown format report."""
        report = "# AskGuess Game Analysis Report\n\n"
        
        report += "## Executive Summary\n\n"
        model_perf = metrics_data.get("model_performance", {})
        strategic = metrics_data.get("strategic_metrics", {})
        
        report += f"**Games Analyzed:** {metrics_data.get('games_analyzed', 0)}\n"
        report += f"**Total Model Inferences:** {model_perf.get('total_inferences', 0)}\n"
        report += f"**Success Rate:** {strategic.get('success_rate', 0):.1%}\n"
        report += f"**Average Quality Score:** {model_perf.get('average_quality_score', 0):.2f}/1.0\n"
        report += f"**Decision Consistency:** {model_perf.get('decision_consistency', 0):.2f}/1.0\n\n"
        
        report += "## Model Inference Performance\n\n"
        report += f"- **Total Inferences:** {model_perf.get('total_inferences', 0)}\n"
        
        breakdown = model_perf.get('inference_breakdown', {})
        report += f"  - Questions: {breakdown.get('questions', 0)}\n"
        report += f"  - Answers: {breakdown.get('answers', 0)}\n"
        report += f"  - Guesses: {breakdown.get('guesses', 0)}\n"
        
        report += f"- **Error Rate:** {model_perf.get('error_rate', 0):.1%}\n"
        report += f"- **Average Quality Score:** {model_perf.get('average_quality_score', 0):.2f}/1.0\n\n"
        
        report += "## Strategic Performance\n\n"
        report += f"- **Success Rate:** {strategic.get('success_rate', 0):.1%}\n"
        report += f"- **Average Rounds to Success:** {strategic.get('average_rounds_to_success', 0):.1f}\n"
        report += f"- **Efficiency Score:** {strategic.get('efficiency_score', 0):.2f}/1.0\n"
        report += f"- **Successful Games:** {strategic.get('successful_games', 0)}\n"
        report += f"- **Failed Games:** {strategic.get('failed_games', 0)}\n\n"
        
        report += "## Question Quality Analysis\n\n"
        quality = metrics_data.get("question_quality", {})
        report += f"- **Total Questions Asked:** {quality.get('total_questions', 0)}\n"
        report += f"- **Average Question Length:** {quality.get('average_length', 0):.1f} words\n"
        report += f"- **Question Diversity:** {quality.get('question_diversity', 0):.2f}\n"
        report += f"- **Unique Questions:** {quality.get('unique_questions', 0)}\n\n"

        report += "## Convergence Analysis\n\n"
        convergence = metrics_data.get("convergence_analysis", {})
        report += f"- **Average Convergence Score:** {convergence.get('average_convergence_score', 0):.2f}/1.0\n\n"
        
        report += "## Success vs Failure Patterns\n\n"
        patterns = metrics_data.get("success_patterns", {})
        success_pat = patterns.get("successful_patterns", {})
        failure_pat = patterns.get("failed_patterns", {})
        
        report += f"- **Successful Games - Avg Questions:** {success_pat.get('avg_questions', 0):.1f}\n"
        report += f"- **Failed Games - Avg Questions:** {failure_pat.get('avg_questions', 0):.1f}\n\n"
        
        report += "## Recommendations\n\n"
        success_rate = strategic.get('success_rate', 0)
        efficiency = strategic.get('efficiency_score', 0)
        quality = model_perf.get('average_quality_score', 0)
        
        if success_rate < 0.5:
            report += "- **Improve Success Rate:** Focus on better question strategy and information gathering\n"
        if efficiency < 0.5:
            report += "- **Increase Efficiency:** Work on reaching correct conclusions with fewer questions\n"
        if quality < 0.6:
            report += "- **Enhance Question Quality:** Develop more specific and targeted questions\n"
        
        report += f"\n---\n*Report generated on {metrics_data.get('timestamp', 'Unknown')}*\n"
        
        return report

    def _generate_text_report(self, metrics_data: Dict[str, Any]) -> str:
        """Generate plain text format report."""
        return self._generate_markdown_report(metrics_data).replace('#', '').replace('*', '')