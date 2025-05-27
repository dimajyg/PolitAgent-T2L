from llm.game import Game
from environments.askguess.agents.answer_agent import AnswerAgent
from environments.askguess.agents.question_agent import QuestionAgent
from environments.askguess.utils.utils import create_message

from environments.askguess.utils.prompt import get_host_description_prompt, get_host_qa_prompt

from time import sleep
import time
import os
import re
import json
import logging

# Импортируем класс метрик
from metrics.askguess_metrics import AskGuessMetrics

logger = logging.getLogger(__name__)

class AskGuessGame(Game):
    """
    Main implementation of the AskGuess game, where one agent tries to guess a word
    by asking questions, and another agent answers them.
    
    Args:
        args: Game arguments (mode, settings, etc.)
        model: Language model (LangChain-compatible)
    """
    def __init__(self, args, model):
        super().__init__(args)
        self.model = model
        self.word = None
        self.answer_agent = None
        self.question_agent = None
        self.max_rounds = getattr(args, 'max_rounds', 10)  # Default to 10 rounds for benchmark
        self.qa_history = []
        logger.info(f"Initialized AskGuess game with model: {model.__class__.__name__}")
        
        # Инициализация системы метрик
        self.metrics = AskGuessMetrics(metadata={
            "game_id": f"askguess_{int(time.time())}",
            "model": getattr(model, "__class__.__name__", str(model))
        })
        
        # Если включена LLM-оценка, настраиваем эту функциональность
        self.use_llm_evaluation = getattr(args, "use_llm_evaluation", False)
        if self.use_llm_evaluation:
            evaluator_model = getattr(args, "evaluation_model", None)
            if evaluator_model is None:
                evaluator_model = model
            self.metrics.enable_llm_evaluation(evaluator_model)
            logger.info("LLM game process evaluation enabled")
        
        # Устанавливаем режим игры и количество раундов в метриках
        self.metrics.set_game_mode(getattr(args, 'mode', 'standard'))
        self.metrics.set_total_rounds(self.max_rounds)

    def init_game(self, word):
        """
        Initializes the game with the given word.
        
        Args:
            word: Word to be guessed
        """
        self.word = word.replace("_", " ")
        logger.info(f"Initializing AskGuess game with word: {self.word}")
        
        # Запись начала игры и целевого слова в метрики
        self.metrics.set_target_word(self.word)
        self.metrics.record_event(
            self.metrics.EVENT_GAME_START,
            target_word=self.word
        )
        
        # Создаем агентов с моделями
        self.answer_agent = AnswerAgent(self.model, self.word, self.args)
        self.question_agent = QuestionAgent(self.model, self.word, self.args)
        self.agents = [self.answer_agent, self.question_agent]
        
        # Сбрасываем счетчик раундов
        self.game_round = 0
        self.qa_history = []

    def check_word_mention(self, text):
        """Check whether the answerer directly uses the word as hint"""
        pattern = r'[^\w\s]'
        replaced_text = re.sub(pattern, ' ', text)
        return (" " + self.word + " ") in replaced_text

    def game_loop(self, log_file):
        """
        Main game loop, executing up to max_rounds of questions and answers.
        
        Args:
            log_file: File for logging game events
            
        Returns:
            Dictionary with game results
        """
        if self.args.mode == "easy" and self.game_round == 0:
            result = self.handle_description_stage(log_file)
            if isinstance(result, dict):
                # Запись ошибки в метрики
                self.metrics.record_event(
                    "error",
                    error_type=result.get("error_type", "Unknown"),
                    stage="description"
                )
                
                # Запись конца игры
                self.metrics.record_event(
                    self.metrics.EVENT_GAME_END,
                    success=False,
                    error=result.get("error_type", "Unknown")
                )
                
                # Если включена LLM-оценка, проводим оценку итогов игры
                if self.metrics.use_llm_evaluation:
                    self.metrics.evaluate_game()
                    
                # Добавляем метрики к результату
                result["metrics"] = self.metrics.compute_all()
                result["metrics_file"] = self._save_metrics()
                
                return result
                
            self.game_round += 1
            
        # Запускаем цикл Q&A на несколько раундов
        host_message = create_message("user", get_host_qa_prompt())
        self.answer_agent.update_history(host_message)
        self.question_agent.update_history(host_message)
        
        # Выполняем все раунды последовательно
        for current_round in range(self.game_round, self.max_rounds):
            self.game_round = current_round
            
            # Запись начала раунда в метрики
            self.metrics.record_event(
                self.metrics.EVENT_ROUND_START,
                round_number=self.game_round + 1
            )
            
            # Логируем текущий раунд
            round_msg = f"\n--- Round {self.game_round + 1} ---\n"
            if log_file:
                log_file.write(round_msg)
            logger.info(round_msg)
            
            # Выполняем один раунд Q&A
            result = self.handle_qa_stage(log_file)
            
            # Запись конца раунда в метрики
            self.metrics.record_event(
                self.metrics.EVENT_ROUND_END,
                round_number=self.game_round + 1
            )
            
            # Проверяем результат раунда
            if result is False:
                # Запись ошибки и конца игры в метрики
                self.metrics.record_event(
                    "error",
                    error_type="ChatError",
                    stage="qa"
                )
                
                self.metrics.record_event(
                    self.metrics.EVENT_GAME_END,
                    success=False,
                    error="ChatError"
                )
                
                # Если включена LLM-оценка, проводим оценку итогов игры
                if self.metrics.use_llm_evaluation:
                    self.metrics.evaluate_game()
                
                result_dict = {
                    "object": self.word, 
                    "round": -1, 
                    "qa_history": self.qa_history,
                    "error_type": "ChatError",
                    "metrics": self.metrics.compute_all(),
                    "metrics_file": self._save_metrics()
                }
                
                return result_dict
                
            elif isinstance(result, dict):
                is_success = result.get("error_type") == "SuccessfulTrial"
                
                self.metrics.record_event(
                    self.metrics.EVENT_GAME_END,
                    success=is_success,
                    error=None if is_success else result.get("error_type", "Unknown")
                )
                
                if self.metrics.use_llm_evaluation:
                    self.metrics.evaluate_game()
                
                result["metrics"] = self.metrics.compute_all()
                result["metrics_file"] = self._save_metrics()
                
                return result
                
            # Небольшая пауза между раундами
            sleep(1)
            
        # If we've reached the maximum number of rounds without a result
        logger.info(f"Maximum rounds ({self.max_rounds}) reached without guessing the word")
        
        self.metrics.record_event(
            self.metrics.EVENT_GAME_END,
            success=False,
            error="RoundLimitError"
        )
        
        if self.metrics.use_llm_evaluation:
            self.metrics.evaluate_game()
        
        result_dict = {
            "object": self.word,
            "round": self.max_rounds,
            "qa_history": self.qa_history,
            "error_type": "RoundLimitError",
            "metrics": self.metrics.compute_all(),
            "metrics_file": self._save_metrics()
        }
        
        return result_dict

    def handle_description_stage(self, log_file):
        """
        Handles the description stage (only in easy mode).
        
        Args:
            log_file: File for logging game events
            
        Returns:
            True if successful, or dict with error information if failed
        """
        logger.info("Starting description stage")
        
        host_message = create_message("user", get_host_description_prompt())
        self.answer_agent.update_history(host_message)
        self.question_agent.update_history(host_message)
        
        self.metrics.record_event(
            self.metrics.EVENT_TURN_START,
            agent="answer_agent",
            stage="description"
        )
        
        try:
            start_time = time.time()
            answer_response = self.answer_agent.answer()
            response_time = time.time() - start_time
            
            if self.check_word_mention(answer_response):
                logger.warning(f"Answer agent mentioned the word directly: {answer_response}")
                
                self.metrics.record_event(
                    self.metrics.EVENT_TURN_END,
                    agent="answer_agent",
                    stage="description",
                    success=False,
                    error="word_mentioned"
                )
                
                return {
                    "object": self.word,
                    "round": 0,
                    "qa_history": [],
                    "error_type": "AnswerMentionedError"
                }
            
            self.metrics.record_model_interaction(
                agent_name="answer_agent",
                request="describe",
                response=answer_response,
                model_name=getattr(self.model, "__class__.__name__", "unknown"),
                latency=response_time
            )
            
            self.metrics.record_event(
                self.metrics.EVENT_TURN_END,
                agent="answer_agent",
                stage="description",
                success=True
            )
            
            answer_message = f"Answerer: {answer_response}"
            if log_file:
                log_file.write(answer_message + "\n")
            logger.info(answer_message)
            
            answer_msg_for_answerer = create_message("assistant", answer_response)
            answer_msg_for_questioner = create_message("user", answer_response)
            
            self.answer_agent.update_history(answer_msg_for_answerer)
            self.question_agent.update_history(answer_msg_for_questioner)
            
            self.qa_history.append({
                "role": "answerer",
                "content": answer_response
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error in description stage: {str(e)}")
            
            self.metrics.record_event(
                self.metrics.EVENT_TURN_END,
                agent="answer_agent",
                stage="description",
                success=False,
                error=str(e)
            )
            
            return {
                "object": self.word,
                "round": 0,
                "qa_history": [],
                "error_type": "ChatError"
            }

    def handle_qa_stage(self, log_file):
        """
        Handles a single Q&A round.
        
        Args:
            log_file: File for logging game events
            
        Returns:
            True if round completed successfully,
            False if there was an error,
            dict with results if game ended
        """ 
        logger.debug(f"=== Round {self.game_round + 1} - Current History ===")
        logger.debug(f"Question agent history length: {len(self.question_agent.private_history)}")
        logger.debug(f"Answer agent history length: {len(self.answer_agent.private_history)}")
        
        for i, msg in enumerate(self.question_agent.private_history[-5:]):  # Last 5 messages
            logger.debug(f"Q-Agent msg {i}: {msg['role']}: {msg['content'][:100]}...")
        
        for i, msg in enumerate(self.answer_agent.private_history[-5:]):  # Last 5 messages
            logger.debug(f"A-Agent msg {i}: {msg['role']}: {msg['content'][:100]}...")
        
        self.metrics.record_event(
            self.metrics.EVENT_TURN_START,
            agent="question_agent",
            stage="question",
            round_number=self.game_round + 1
        )
        
        try:
            start_time = time.time()
            if hasattr(self.question_agent, "play_with_thinking"):
                question_response, thinking = self.question_agent.play_with_thinking()
            else:
                question_response = self.question_agent.play()
                thinking = None
                
            response_time = time.time() - start_time
            
            logger.debug(f"Question agent response: {question_response}")
            
            self.metrics.record_model_interaction(
                agent_name="question_agent",
                request="question",
                response=question_response,
                model_name=getattr(self.model, "__class__.__name__", "unknown"),
                latency=response_time
            )
            
            self.metrics.record_event(
                self.metrics.EVENT_TURN_END,
                agent="question_agent",
                stage="question",
                round_number=self.game_round + 1,
                success=True
            )
            
            if "my guess is" in question_response.lower() or "i guess" in question_response.lower():
                logger.info(f"Detected guess in question: {question_response}")
                
                guess_match = re.search(r'(?:my guess is|i guess)[:\s]+([^\.!?,;]+)', question_response.lower())
                if guess_match:
                    guessed_word = guess_match.group(1).strip()
                    logger.info(f"Extracted guess: '{guessed_word}'")
                    
                    is_correct = guessed_word.lower() == self.word.lower()
                    
                    self.metrics.record_guess(
                        guess=guessed_word,
                        is_correct=is_correct,
                        round_num=self.game_round + 1
                    )
                    
                    question_message = f"Questioner: {question_response}"
                    if log_file:
                        log_file.write(question_message + "\n")
                    logger.info(question_message)
                    
                    question_msg_for_answerer = create_message("user", question_response)
                    question_msg_for_questioner = create_message("assistant", question_response)
                    
                    self.answer_agent.update_history(question_msg_for_answerer)
                    self.question_agent.update_history(question_msg_for_questioner)
                    
                    self.qa_history.append({
                        "role": "questioner", 
                        "content": question_response
                    })
                    
                    if is_correct:
                        logger.info(f"Correct guess! The word was: {self.word}")
                        return {
                            "object": self.word,
                            "round": self.game_round + 1,
                            "qa_history": self.qa_history,
                            "error_type": "SuccessfulTrial"
                        }
            
            question_message = f"Questioner: {question_response}"
            if log_file:
                log_file.write(question_message + "\n")
            logger.info(question_message)
            
            question_msg_for_answerer = create_message("user", question_response)
            question_msg_for_questioner = create_message("assistant", question_response)
            
            self.answer_agent.update_history(question_msg_for_answerer)
            self.question_agent.update_history(question_msg_for_questioner)
            
            self.qa_history.append({
                "role": "questioner", 
                "content": question_response
            })
            
        except Exception as e:
            logger.error(f"Error in question stage: {str(e)}")
            return False
            
        self.metrics.record_event(
            self.metrics.EVENT_TURN_START,
            agent="answer_agent",
            stage="answer",
            round_number=self.game_round + 1
        )
        
        try:
            start_time = time.time()
            answer_response = self.answer_agent.answer()
            response_time = time.time() - start_time
            
            logger.debug(f"Answer agent response: {answer_response}")
            
            if "gameover" in answer_response.lower() or "game over" in answer_response.lower():
                logger.info(f"Answer agent signaled game over: {answer_response}")
                
                if any(word in answer_response.lower() for word in ["correct", "right", "yes", "guessed"]):
                    return {
                        "object": self.word,
                        "round": self.game_round + 1,
                        "qa_history": self.qa_history,
                        "error_type": "SuccessfulTrial"
                    }
                else:
                    return {
                        "object": self.word,
                        "round": self.game_round + 1,
                        "qa_history": self.qa_history,
                        "error_type": "EndingError"
                    }
            
            if self.check_word_mention(answer_response):
                logger.warning(f"Answer agent mentioned the word directly: {answer_response}")
                
                self.metrics.record_event(
                    self.metrics.EVENT_TURN_END,
                    agent="answer_agent",
                    stage="answer",
                    round_number=self.game_round + 1,
                    success=False,
                    error="word_mentioned"
                )
                
                return {
                    "object": self.word,
                    "round": self.game_round + 1,
                    "qa_history": self.qa_history,
                    "error_type": "AnswerMentionedError"
                }
            
            answer_lower = answer_response.lower().strip()
            valid_responses = ["yes", "no", "maybe", "sometimes", "often", "rarely", "never", "usually", "occasionally"]
            
            if not any(valid_word in answer_lower for valid_word in valid_responses):
                logger.warning(f"Answer agent gave invalid response: {answer_response}")
                
            self.metrics.record_model_interaction(
                agent_name="answer_agent",
                request="answer",
                response=answer_response,
                model_name=getattr(self.model, "__class__.__name__", "unknown"),
                latency=response_time
            )
            
            self.metrics.record_event(
                self.metrics.EVENT_TURN_END,
                agent="answer_agent",
                stage="answer",
                round_number=self.game_round + 1,
                success=True
            )
            
            answer_message = f"Answerer: {answer_response}"
            if log_file:
                log_file.write(answer_message + "\n")
            logger.info(answer_message)
            
            answer_msg_for_answerer = create_message("assistant", answer_response)
            answer_msg_for_questioner = create_message("user", answer_response)
            
            self.answer_agent.update_history(answer_msg_for_answerer)
            self.question_agent.update_history(answer_msg_for_questioner)
            
            self.qa_history.append({
                "role": "answerer",
                "content": answer_response
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error in answer stage: {str(e)}")
            return False

    def _save_metrics(self) -> str:
        """
        Saves metrics to a file and returns the filename.
        
        Returns:
            str: Filename where metrics were saved
        """
        timestamp = int(time.time())
        metrics_filename = f"askguess_metrics_{timestamp}.json"
        
        results_dir = os.environ.get("BENCHMARK_RESULTS_DIR", "benchmark_results")
        metrics_path = os.path.join(results_dir, metrics_filename)
        
        self.metrics.save(metrics_path)
        
        return metrics_filename