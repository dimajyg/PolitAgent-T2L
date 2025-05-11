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
    Основная реализация игры AskGuess, где один агент пытается угадать слово,
    задавая вопросы, а другой отвечает на них.
    
    Args:
        args: Аргументы игры (режим, настройки и т.д.)
        model: Языковая модель (LangChain-совместимая)
    """
    def __init__(self, args, model):
        super().__init__(args)
        self.model = model
        self.word = None
        self.answer_agent = None
        self.question_agent = None
        self.max_rounds = getattr(args, 'max_rounds', 10)  # Default to 10 rounds for benchmark
        self.qa_history = []
        logger.info(f"Инициализирована игра AskGuess с моделью: {model.__class__.__name__}")
        
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
            logger.info("LLM-оценка игрового процесса включена")
        
        # Устанавливаем режим игры и количество раундов в метриках
        self.metrics.set_game_mode(getattr(args, 'mode', 'standard'))
        self.metrics.set_total_rounds(self.max_rounds)

    def init_game(self, word):
        """
        Инициализирует игру с заданным словом.
        
        Args:
            word: Слово, которое нужно угадать
        """
        self.word = word.replace("_", " ")
        logger.info(f"Инициализация игры AskGuess со словом: {self.word}")
        
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
        Основной игровой цикл, выполняющий до max_rounds раундов вопросов и ответов.
        
        Args:
            log_file: Файл для записи логов игры
            
        Returns:
            Словарь с результатами игры
        """
        # Describing Stage (только для режима easy)
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
                # Обнаружен конец игры (успех или ошибка)
                is_success = result.get("error_type") == "SuccessfulTrial"
                
                # Запись конца игры в метрики
                self.metrics.record_event(
                    self.metrics.EVENT_GAME_END,
                    success=is_success,
                    error=None if is_success else result.get("error_type", "Unknown")
                )
                
                # Если включена LLM-оценка, проводим оценку итогов игры
                if self.metrics.use_llm_evaluation:
                    self.metrics.evaluate_game()
                
                # Добавляем метрики к результату
                result["metrics"] = self.metrics.compute_all()
                result["metrics_file"] = self._save_metrics()
                
                return result
                
            # Небольшая пауза между раундами
            sleep(1)
            
        # Игра завершена по достижению максимального числа раундов
        # Запись конца игры в метрики
        self.metrics.record_event(
            self.metrics.EVENT_GAME_END,
            success=True,
            reason="RoundLimitSuccess"
        )
        
        # Если включена LLM-оценка, проводим оценку итогов игры
        if self.metrics.use_llm_evaluation:
            self.metrics.evaluate_game()
        
        result_dict = {
            "object": self.word, 
            "round": self.game_round + 1,  # +1 так как индексация с 0
            "qa_history": self.qa_history,
            "error_type": "RoundLimitSuccess",
            "metrics": self.metrics.compute_all(),
            "metrics_file": self._save_metrics()
        }
        
        return result_dict

    def handle_description_stage(self, log_file):
        sleep(2)
        
        # Запись начала хода
        self.metrics.record_event(
            self.metrics.EVENT_TURN_START,
            stage="description",
            agent="answer_agent"
        )
        
        # Запись взаимодействия с моделью
        start_time = time.time()
        description = self.answer_agent.play()
        response_time = time.time() - start_time
        
        if description is None:
            # Запись ошибки
            self.metrics.record_model_interaction(
                agent_name="answer_agent",
                request="get_description",
                response="ERROR: Failed to get description",
                model_name=getattr(self.model, "__class__.__name__", "unknown"),
                latency=response_time
            )
            
            # Запись конца хода
            self.metrics.record_event(
                self.metrics.EVENT_TURN_END,
                stage="description",
                agent="answer_agent",
                success=False
            )
            
            return {"object": self.word, "round": -1, "error_type": "ChatError"}

        # Запись успешного взаимодействия с моделью
        self.metrics.record_model_interaction(
            agent_name="answer_agent",
            request="get_description",
            response=description,
            model_name=getattr(self.model, "__class__.__name__", "unknown"),
            latency=response_time
        )

        if self.check_word_mention(description.lower()):
            # Запись конца хода с ошибкой
            self.metrics.record_event(
                self.metrics.EVENT_TURN_END,
                stage="description",
                agent="answer_agent",
                success=False,
                error="AnswerMentionedError"
            )
            
            return {"object": self.word, "round": -1, "error_type": "AnswerMentionedError"}

        self.log_message(log_file, f"description: {description}")
        questioner_message = create_message("user", description)
        answerer_message = create_message("assistant", description)
        self.answer_agent.update_history(answerer_message)
        self.question_agent.update_history(questioner_message)
        
        # Запись конца хода
        self.metrics.record_event(
            self.metrics.EVENT_TURN_END,
            stage="description",
            agent="answer_agent",
            success=True
        )
        
        return True

    def handle_qa_stage(self, log_file):
        # Get question
        
        # Запись начала хода для вопроса
        self.metrics.record_event(
            self.metrics.EVENT_TURN_START,
            stage="question",
            agent="question_agent",
            round=self.game_round + 1
        )
        
        sleep(2)
        
        # Замер времени для вопроса
        start_time = time.time()
        question, thinking = self.question_agent.play_with_thinking()
        response_time = time.time() - start_time
        
        if question is None:
            # Запись ошибки взаимодействия с моделью
            self.metrics.record_model_interaction(
                agent_name="question_agent",
                request="get_question",
                response="ERROR: Failed to get question",
                model_name=getattr(self.model, "__class__.__name__", "unknown"),
                latency=response_time
            )
            
            # Запись конца хода с ошибкой
            self.metrics.record_event(
                self.metrics.EVENT_TURN_END,
                stage="question",
                agent="question_agent",
                round=self.game_round + 1,
                success=False
            )
            
            return False

        # Запись успешного взаимодействия с моделью для вопроса
        self.metrics.record_model_interaction(
            agent_name="question_agent",
            request="get_question",
            response=question,
            model_name=getattr(self.model, "__class__.__name__", "unknown"),
            latency=response_time
        )
        
        # Запись вопроса в метрики
        self.metrics.record_question(
            question=question,
            round_num=self.game_round + 1,
            thinking=thinking
        )

        self.log_message(log_file, f"question: {question}")
        questioner_message = create_message("assistant", question)
        answerer_message = create_message("user", question)
        self.answer_agent.update_history(answerer_message)
        self.question_agent.update_history(questioner_message)
        
        # Запись конца хода для вопроса
        self.metrics.record_event(
            self.metrics.EVENT_TURN_END,
            stage="question",
            agent="question_agent",
            round=self.game_round + 1,
            success=True
        )

        # Get answer
        
        # Запись начала хода для ответа
        self.metrics.record_event(
            self.metrics.EVENT_TURN_START,
            stage="answer",
            agent="answer_agent",
            round=self.game_round + 1
        )
        
        sleep(2)
        
        # Замер времени для ответа
        start_time = time.time()
        answer = self.answer_agent.play()
        response_time = time.time() - start_time
        
        if answer is None:
            # Запись ошибки взаимодействия с моделью
            self.metrics.record_model_interaction(
                agent_name="answer_agent",
                request="get_answer",
                response="ERROR: Failed to get answer",
                model_name=getattr(self.model, "__class__.__name__", "unknown"),
                latency=response_time
            )
            
            # Запись конца хода с ошибкой
            self.metrics.record_event(
                self.metrics.EVENT_TURN_END,
                stage="answer",
                agent="answer_agent",
                round=self.game_round + 1,
                success=False
            )
            
            return False

        # Запись успешного взаимодействия с моделью для ответа
        self.metrics.record_model_interaction(
            agent_name="answer_agent",
            request="get_answer",
            response=answer,
            model_name=getattr(self.model, "__class__.__name__", "unknown"),
            latency=response_time
        )
        
        # Запись ответа в метрики
        self.metrics.record_answer(
            answer=answer,
            round_num=self.game_round + 1
        )

        self.log_message(log_file, f"answer: {answer}")
        questioner_message = create_message("user", answer)
        answerer_message = create_message("assistant", answer)
        self.answer_agent.update_history(answerer_message)
        self.question_agent.update_history(questioner_message)
        
        # Запись конца хода для ответа
        self.metrics.record_event(
            self.metrics.EVENT_TURN_END,
            stage="answer",
            agent="answer_agent",
            round=self.game_round + 1,
            success=True
        )
        
        # Store QA in history
        self.qa_history.append({"question": question, "answer": answer})

        # Check game end conditions
        if "gameover" in answer.lower() or "game over" in answer.lower():
            is_correct = self.word.lower() in question.lower()
            
            # Запись догадки в метрики
            self.metrics.record_guess(
                guess=question,
                is_correct=is_correct,
                round_num=self.game_round + 1,
                thinking=thinking if 'thinking' in locals() else None
            )
            
            if is_correct:
                return {
                    "object": self.word, 
                    "round": self.game_round + 1,  # +1 так как индексация с 0
                    "qa_history": self.qa_history,
                    "error_type": "SuccessfulTrial"
                }
            else:
                return {
                    "object": self.word, 
                    "round": -1, 
                    "qa_history": self.qa_history,
                    "error_type": "EndingError"
                }

        if self.check_word_mention(answer.lower()):
            # Запись ошибки в метрики
            self.metrics.record_event(
                "error",
                error_type="AnswerMentionedError",
                stage="answer",
                round=self.game_round + 1
            )
            
            return {
                "object": self.word, 
                "round": -1, 
                "qa_history": self.qa_history,
                "error_type": "AnswerMentionedError"
            }

        return True
        
    def _save_metrics(self) -> str:
        """
        Сохраняет метрики в файл и возвращает имя файла.
        
        Returns:
            str: Имя сохраненного файла метрик
        """
        # Создаем имя файла метрик с временной меткой
        timestamp = int(time.time())
        metrics_filename = f"askguess_metrics_{timestamp}.json"
        
        # Получаем путь к директории текущих результатов из переменной окружения или используем значение по умолчанию
        results_dir = os.environ.get("BENCHMARK_RESULTS_DIR", "benchmark_results")
        metrics_path = os.path.join(results_dir, metrics_filename)
        
        # Вычисляем метрики
        computed_metrics = self.metrics.compute_all()
        
        # Сохраняем метрики в файл
        self.metrics.save(metrics_path)
        
        return metrics_filename