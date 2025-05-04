from llm.game import Game
from environments.askguess.agents.answer_agent import AnswerAgent
from environments.askguess.agents.question_agent import QuestionAgent
from environments.askguess.utils.utils import create_message

from environments.askguess.utils.prompt import get_host_description_prompt, get_host_qa_prompt

from time import sleep

import re
import json
import logging

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

    def init_game(self, word):
        """
        Инициализирует игру с заданным словом.
        
        Args:
            word: Слово, которое нужно угадать
        """
        self.word = word.replace("_", " ")
        logger.info(f"Инициализация игры AskGuess со словом: {self.word}")
        
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
                return result
            self.game_round += 1
            
        # Запускаем цикл Q&A на несколько раундов
        host_message = create_message("user", get_host_qa_prompt())
        self.answer_agent.update_history(host_message)
        self.question_agent.update_history(host_message)
        
        # Выполняем все раунды последовательно
        for current_round in range(self.game_round, self.max_rounds):
            self.game_round = current_round
            
            # Логируем текущий раунд
            round_msg = f"\n--- Round {self.game_round + 1} ---\n"
            if log_file:
                log_file.write(round_msg)
            logger.info(round_msg)
            
            # Выполняем один раунд Q&A
            result = self.handle_qa_stage(log_file)
            
            # Проверяем результат раунда
            if result is False:
                return {
                    "object": self.word, 
                    "round": -1, 
                    "qa_history": self.qa_history,
                    "error_type": "ChatError"
                }
            elif isinstance(result, dict):
                # Обнаружен конец игры (успех или ошибка)
                return result
                
            # Небольшая пауза между раундами
            sleep(1)
            
        # Игра завершена по достижению максимального числа раундов
        return {
            "object": self.word, 
            "round": self.game_round + 1,  # +1 так как индексация с 0
            "qa_history": self.qa_history,
            "error_type": "RoundLimitSuccess"
        }

    def handle_description_stage(self, log_file):
        sleep(2)
        description = self.answer_agent.play()
        if description is None:
            return {"object": self.word, "round": -1, "error_type": "ChatError"}

        if self.check_word_mention(description.lower()):
            return {"object": self.word, "round": -1, "error_type": "AnswerMentionedError"}

        self.log_message(log_file, f"description: {description}")
        questioner_message = create_message("user", description)
        answerer_message = create_message("assistant", description)
        self.answer_agent.update_history(answerer_message)
        self.question_agent.update_history(questioner_message)
        return True

    def handle_qa_stage(self, log_file):
        # Get question
        sleep(2)
        question = self.question_agent.play()
        if question is None:
            return False

        self.log_message(log_file, f"question: {question}")
        questioner_message = create_message("assistant", question)
        answerer_message = create_message("user", question)
        self.answer_agent.update_history(answerer_message)
        self.question_agent.update_history(questioner_message)

        # Get answer
        sleep(2)
        answer = self.answer_agent.play()
        if answer is None:
            return False

        self.log_message(log_file, f"answer: {answer}")
        questioner_message = create_message("user", answer)
        answerer_message = create_message("assistant", answer)
        self.answer_agent.update_history(answerer_message)
        self.question_agent.update_history(questioner_message)
        
        # Store QA in history
        self.qa_history.append({"question": question, "answer": answer})

        # Check game end conditions
        if "gameover" in answer.lower() or "game over" in answer.lower():
            if self.word.lower() in question.lower():
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
            return {
                "object": self.word, 
                "round": -1, 
                "qa_history": self.qa_history,
                "error_type": "AnswerMentionedError"
            }

        return True