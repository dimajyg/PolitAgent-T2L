from typing import Dict, List, Tuple, Union, Any, Set, Optional
import numpy as np
from sklearn.metrics import f1_score
import re
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
    
    # AskGuess specific event types
    EVENT_QUESTION = "question"
    EVENT_ANSWER = "answer"
    EVENT_GUESS = "guess"
    EVENT_CORRECT_GUESS = "correct_guess"
    EVENT_INCORRECT_GUESS = "incorrect_guess"
    
    # AskGuess specific LLM evaluation templates
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
            
        # Build game summary from events
        question_events = [e for e in self.events if e["type"] == self.EVENT_QUESTION]
        answer_events = [e for e in self.events if e["type"] == self.EVENT_ANSWER]
        guess_events = [e for e in self.events if e["type"] in [self.EVENT_GUESS, self.EVENT_CORRECT_GUESS, self.EVENT_INCORRECT_GUESS]]
        
        # Format Q&A exchanges
        qa_exchanges = []
        for i, (q_event, a_event) in enumerate(zip(question_events, answer_events)):
            question = q_event["data"].get("question", "No question")
            answer = a_event["data"].get("answer", "No answer")
            qa_exchanges.append(f"Round {i+1}:\nQ: {question}\nA: {answer}")
        
        # Format guesses
        guess_summary = []
        for event in guess_events:
            round_num = event["data"].get("round", "?")
            guess = event["data"].get("guess", "No guess")
            is_correct = event["type"] == self.EVENT_CORRECT_GUESS
            result = "CORRECT" if is_correct else "INCORRECT"
            guess_summary.append(f"Round {round_num} Guess: {guess} - {result}")
        
        # Build game summary
        game_summary = "Q&A Exchanges:\n" + "\n\n".join(qa_exchanges) + "\n\n"
        game_summary += "Guesses:\n" + "\n".join(guess_summary)
        
        # Final outcome
        if self.correct_guess:
            game_summary += f"\n\nOutcome: Successfully guessed the word in round {self.correct_round} out of {self.total_rounds}."
        else:
            game_summary += f"\n\nOutcome: Failed to guess the word within {self.total_rounds} rounds."
        
        # Context for evaluation
        context = {
            "target_word": self.target_word,
            "game_mode": self.game_mode,
            "total_rounds": self.total_rounds,
            "game_summary": game_summary
        }
        
        # Request evaluation using the AskGuess-specific template
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
            
        # Get previous Q&A exchanges
        previous_qa = []
        for i, (q, a) in enumerate(zip(self.questions, self.answers)):
            if i < round_num - 1:  # Only include previous rounds
                previous_qa.append(f"Q{i+1}: {q['question']}\nA{i+1}: {a['answer']}")
        
        previous_qa_text = "\n\n".join(previous_qa)
        
        # Context for evaluation
        context = {
            "target_word": self.target_word,
            "current_round": round_num,
            "total_rounds": self.total_rounds,
            "previous_qa": previous_qa_text,
            "question": question
        }
        
        # Request evaluation using the AskGuess-specific template
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
        
        # If LLM evaluation is enabled, evaluate this question
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
        # Call the parent method to get common metrics
        base_metrics = super().compute_all()
        
        # Add AskGuess-specific metrics
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
        
        # Merge metrics
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
        
        # Analyze yes/no ratio in answers
        yes_count = sum(1 for a in self.answers if a["answer"].lower() in ["yes", "true", "correct", "right"])
        no_count = sum(1 for a in self.answers if a["answer"].lower() in ["no", "false", "incorrect", "wrong"])
        
        qa_metrics["yes_ratio"] = yes_count / len(self.answers) if self.answers else 0
        qa_metrics["no_ratio"] = no_count / len(self.answers) if self.answers else 0
        qa_metrics["binary_answer_ratio"] = (yes_count + no_count) / len(self.answers) if self.answers else 0
        
        # Question information gain
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
        # Simple convergence metrics
        convergence_metrics = {
            "rounds_played": len(self.questions),
            "converged": self.correct_guess is not None,
            "convergence_speed": 1.0 - ((self.correct_round - 1) / self.total_rounds) if self.correct_round else 0.0
        }
        
        # Add similarity between guesses and target word if we have NLP tools
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            if self.guesses:
                # Get all guesses
                guess_texts = [g["guess"] for g in self.guesses]
                
                # Create a corpus with the target word and guesses
                corpus = [self.target_word] + guess_texts
                
                # Calculate TF-IDF similarity
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(corpus)
                
                # Get similarity of each guess to the target word
                similarities = []
                target_vector = tfidf_matrix[0:1]
                
                for i in range(1, len(corpus)):
                    guess_vector = tfidf_matrix[i:i+1]
                    # Calculate cosine similarity
                    similarity = (target_vector * guess_vector.T).A[0][0]
                    similarities.append(similarity)
                
                convergence_metrics["guess_similarities"] = similarities
                convergence_metrics["avg_guess_similarity"] = sum(similarities) / len(similarities)
                convergence_metrics["max_guess_similarity"] = max(similarities)
        except:
            # If we cannot calculate similarity, skip this metric
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
            
            # Get question texts
            question_texts = [q["question"] for q in self.questions]
            
            # Calculate TF-IDF vectors
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(question_texts)
            
            # Calculate similarity between consecutive questions
            similarities = []
            for i in range(len(question_texts) - 1):
                q1_vector = tfidf_matrix[i:i+1]
                q2_vector = tfidf_matrix[i+1:i+2]
                # Calculate cosine similarity
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
            
        # Simple proxy: yes/no questions provide more information
        binary_answers = []
        
        for answer in self.answers:
            text = answer["answer"].lower()
            if text in ["yes", "true", "correct", "right", "no", "false", "incorrect", "wrong"]:
                # Binary questions typically provide more information
                binary_answers.append(1.0)
            else:
                # Non-binary questions provide less clear information
                binary_answers.append(0.5)
        
        # Estimate information gain for each question
        # We assume later questions are more specific and provide more information
        # if the agent is using what it learned
        gains = []
        for i, is_binary in enumerate(binary_answers):
            # Adjust gain based on question position and answer type
            position_factor = (i + 1) / len(binary_answers)  # Later questions can be more targeted
            gain = is_binary * (0.5 + 0.5 * position_factor)
            gains.append(gain)
        
        return {
            "avg_gain": sum(gains) / len(gains),
            "cumulative_gain": sum(gains),
            "normalized_gain": sum(gains) / len(gains) / 1.0,  # Normalized to [0,1]
            "by_question": gains
        } 