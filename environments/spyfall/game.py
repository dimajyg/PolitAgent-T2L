from llm.game import Game
from environments.spyfall.agents.base_agent import BaseAgent
from environments.spyfall.utils.utils import create_message
import random
import copy
import json
import logging
import time
from time import sleep
from typing import List, Dict, Any, Optional
import os

# Импортируем новый класс метрик из metrics модуля вместо старого
from metrics.spyfall_metrics import SpyfallMetrics

logger = logging.getLogger(__name__)

class SpyfallGame(Game):
    """
    Основной игровой цикл Spyfall с поддержкой LangChain-агентов.
    """

    def __init__(self, args, spy_model, villager_model) -> None:
        super().__init__(args)
        self.spy_model = spy_model
        self.villager_model = villager_model
        self.players: List[str] = ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward"]
        self.living_players: List[str] = []
        self.spy_name: Optional[str] = None
        self.spy_index: Optional[int] = None
        
        # Инициализация системы метрик
        self.metrics = SpyfallMetrics(metadata={
            "game_id": f"spyfall_{int(time.time())}",
            "spy_model": getattr(spy_model, "__class__.__name__", str(spy_model)),
            "villager_model": getattr(villager_model, "__class__.__name__", str(villager_model))
        })
        
        # Если включена LLM-оценка, настраиваем эту функциональность
        # Используем отдельную модель для оценки, если задана, иначе используем модель шпиона
        self.use_llm_evaluation = getattr(args, "use_llm_evaluation", False)
        if self.use_llm_evaluation:
            evaluator_model = getattr(args, "evaluation_model", None)
            if evaluator_model is None:
                evaluator_model = spy_model
            self.metrics.enable_llm_evaluation(evaluator_model)
            logger.info("LLM-оценка игрового процесса включена")

    def init_game(self, phrase_pair: List[str]) -> str:
        # Запись начала игры
        self.metrics.record_event(
            self.metrics.EVENT_GAME_START,
            phrase_pair=phrase_pair
        )
        
        self.living_players = copy.deepcopy(self.players)
        spy_word, villager_word = phrase_pair
        random.shuffle(self.players)
        self.spy_index = random.randint(1, len(self.players))
        self.spy_name = self.players[self.spy_index - 1]
        
        # Запись информации о распределении ролей
        self.metrics.record_role_assignment(self.players, self.spy_index, self.spy_name)
        
        # Добавляем слова в метрики для использования в LLM-оценке
        self.metrics.add_game_words(spy_word, villager_word)
        
        for i, player_name in enumerate(self.players):
            is_spy = (i + 1 == self.spy_index)
            phrase = spy_word if is_spy else villager_word
            model = self.spy_model if is_spy else self.villager_model
            agent = BaseAgent(model, player_name, self.players, phrase, is_spy)
            self.agents.append(agent)
            self.name2agent[player_name] = agent

        settings = f"The spy word is: {spy_word};\n The villager word is {villager_word}.\n"
        for agent in self.agents:
            role = "Spy" if agent.is_spy else "Villager"
            settings += f"Player: {agent.player_name}; Role: {role}; Assigned Word: {agent.phrase} \n"
            
        # Добавляем метаданные об игре
        self.metrics.add_metadata("spy_word", spy_word)
        self.metrics.add_metadata("villager_word", villager_word)
        
        return settings

    def get_voted_name(self, name_list: List[str]) -> Optional[str]:
        counts = {}
        for name in name_list:
            counts[name] = counts.get(name, 0) + 1
        max_count = max(counts.values()) if counts else 0
        for name, count in counts.items():
            if count == max_count:
                return name
        return None

    def game_loop(self, log_file) -> Dict[str, Any]:
        self._announce("Host: The game now start.", log_file)
        self._inform_roles(log_file)
        self._announce(f"Host: The living players are: {json.dumps(self.living_players)}", log_file)

        while True:
            # Запись начала раунда
            self.metrics.record_event(
                self.metrics.EVENT_ROUND_START,
                round_number=self.game_round + 1,
                living_players=self.living_players
            )
            
            self.game_round += 1
            # Обновляем текущий раунд в метриках
            self.metrics.current_round = self.game_round

            if self.spy_name not in self.living_players:
                self._announce(f"Host: The spy {self.spy_name} has been eliminated! Villagers win!", log_file)
                # Запись конца раунда
                self.metrics.record_event(
                    self.metrics.EVENT_ROUND_END,
                    round_number=self.game_round
                )
                # Запись конца игры
                self.metrics.record_game_end("villager", True)
                
                # Если включена LLM-оценка, запрашиваем оценку для раунда и всей игры
                if self.metrics.use_llm_evaluation:
                    self.metrics.evaluate_round(self.game_round)
                    self.metrics.evaluate_game()
                
                return self._collect_results(winner="villager", spy_caught=True)

            if not self.handle_describing_stage(log_file):
                # Запись конца раунда при ошибке
                self.metrics.record_event(
                    self.metrics.EVENT_ROUND_END,
                    round_number=self.game_round,
                    error=True
                )
                # Запись конца игры с ошибкой
                self.metrics.record_game_end("error", False)
                
                # Если включена LLM-оценка, запрашиваем оценку раунда и игры
                if self.metrics.use_llm_evaluation:
                    self.metrics.evaluate_round(self.game_round)
                    self.metrics.evaluate_game()
                
                # Возвращаем результаты с ошибкой, но с включенными метриками
                error_results = self._collect_results(winner="error", spy_caught=False)
                error_results["error"] = "Description stage failed"
                return error_results

            if self._handle_voting(log_file):
                # Запись конца раунда
                self.metrics.record_event(
                    self.metrics.EVENT_ROUND_END,
                    round_number=self.game_round
                )
                # Запись конца игры
                self.metrics.record_game_end("villager", True)
                
                # Если включена LLM-оценка, запрашиваем оценку для раунда и всей игры
                if self.metrics.use_llm_evaluation:
                    self.metrics.evaluate_round(self.game_round)
                    self.metrics.evaluate_game()
                
                return self._collect_results(winner="villager", spy_caught=True)

            if len(self.living_players) < 3:
                self._announce(f"Host: Less than 3 players remain and the spy still lives! Spy {self.spy_name} wins!", log_file)
                # Запись конца раунда
                self.metrics.record_event(
                    self.metrics.EVENT_ROUND_END,
                    round_number=self.game_round
                )
                # Запись конца игры
                self.metrics.record_game_end("spy", False)
                
                # Если включена LLM-оценка, запрашиваем оценку для раунда и всей игры
                if self.metrics.use_llm_evaluation:
                    self.metrics.evaluate_round(self.game_round)
                    self.metrics.evaluate_game()
                
                return self._collect_results(winner="spy", spy_caught=False)
                
            # Конец раунда без определения победителя
            self.metrics.record_event(
                self.metrics.EVENT_ROUND_END,
                round_number=self.game_round
            )
            
            # Если включена LLM-оценка, запрашиваем оценку для завершенного раунда
            if self.metrics.use_llm_evaluation:
                self.metrics.evaluate_round(self.game_round)

    def _announce(self, message: str, log_file) -> None:
        msg = create_message("user", message)
        self.update_history(msg, "host")
        self.log_message(log_file, message)
        logger.info(message)

    def _inform_roles(self, log_file) -> None:
        for agent in self.agents:
            role_info = "spy" if agent.is_spy else "villager"
            host_speech = f"Host: {agent.player_name}, you are a {role_info}."
            role_message = create_message("user", host_speech)
            agent.private_history.append(role_message)
            self.log_message(log_file, f"[PRIVATE] {host_speech}")

    def _handle_voting(self, log_file) -> bool:
        max_vote_attempts = 3
        for attempt in range(max_vote_attempts):
            voted_name = self.handle_voting_stage(log_file)
            if voted_name == self.spy_name:
                self._announce(f"Host: {voted_name} was the spy! Villagers win!", log_file)
                return True
            if voted_name in self.living_players:
                self.living_players.remove(voted_name)
                self._announce(f"Host: the voting result is {voted_name}, he is not the spy. The spy still lives. {voted_name} left the game.", log_file)
                return False
            self._announce(f"Host: {voted_name} is not in the list of living players or has already left the game. Starting a revote.", log_file)
        self._announce("Host: After multiple attempts, no valid vote could be made. Moving to the next round.", log_file)
        return False

    def handle_describing_stage(self, log_file) -> bool:
        self._announce("Host: Now it's the describing stage, players have to say something about the received word.", log_file)
        self.descriptions = []
        self.player_descriptions = {}

        for agent in self.agents:
            if agent.player_name not in self.living_players:
                continue
                
            # Запись начала хода
            self.metrics.record_event(
                self.metrics.EVENT_TURN_START,
                agent=agent.player_name,
                is_spy=agent.is_spy,
                stage="description"
            )
            
            self._announce(f"Host: {agent.player_name}, it's your turn.", log_file)
            sleep(2)
            
            # Замер времени для ответа модели
            start_time = time.time()
            description, cot = agent.describe()
            response_time = time.time() - start_time
            
            if description is None:
                logger.error(f"Agent {agent.player_name} failed to describe.")
                # Запись ошибки
                self.metrics.record_event(
                    "error",
                    agent=agent.player_name,
                    error_type="description_failed"
                )
                return False
                
            # Записываем метрики описания
            self.metrics.record_description(
                agent.player_name, 
                description, 
                agent.is_spy
            )
            
            # Запись взаимодействия с моделью
            self.metrics.record_model_interaction(
                agent_name=agent.player_name,
                request="describe",
                response=description,
                model_name=getattr(agent.chatbot, "__class__.__name__", "unknown"),
                latency=response_time
            )
            
            self._announce(f"{agent.player_name}: {description}", log_file)
            agent.private_history.append(create_message("assistant", json.dumps(cot)))
            self.descriptions.append(description)
            self.player_descriptions[agent.player_name] = description
            
            # Запись конца хода
            self.metrics.record_event(
                self.metrics.EVENT_TURN_END,
                agent=agent.player_name,
                is_spy=agent.is_spy,
                stage="description"
            )
            
        return True

    def handle_voting_stage(self, log_file) -> Optional[str]:
        self._announce(f"Host: The living players are: {json.dumps(self.living_players)}", log_file)
        self._announce("Host: Now the voting start, please vote for the player you think is the spy. You must vote ONLY for a player from the list of living players.", log_file)
        name_list = []
        self.vote_sequence = []
        self.vote_confidences = {}
        
        for agent in self.agents:
            if agent.player_name not in self.living_players:
                continue
                
            # Запись начала голосования
            self.metrics.record_event(
                self.metrics.EVENT_TURN_START,
                agent=agent.player_name,
                is_spy=agent.is_spy,
                stage="voting"
            )
            
            self._announce(f"Host: {agent.player_name}, it's your turn to vote. Remember to choose from the living players: {json.dumps(self.living_players)}", log_file)
            sleep(2)
            
            # Замер времени для голосования
            start_time = time.time()
            name, speak, cot = agent.vote()
            response_time = time.time() - start_time
            
            agent.last_vote = name
            player_idx = self.players.index(agent.player_name) + 1
            self.vote_sequence.append((player_idx, name))
            confidence_data = cot.get("confidence", {}) if isinstance(cot, dict) else {}
            self.vote_confidences[player_idx] = confidence_data
            
            if not hasattr(self, "cot_data"):
                self.cot_data = {}
                
            reasoning = ""
            if isinstance(cot, dict):
                self.cot_data[player_idx] = {"thought": cot.get("thought", ""), "speak": speak}
                reasoning = cot.get("thought", "")
                
            # Запись взаимодействия с моделью
            self.metrics.record_model_interaction(
                agent_name=agent.player_name,
                request="vote",
                response=f"{speak} - voted for {name}",
                model_name=getattr(agent.chatbot, "__class__.__name__", "unknown"),
                latency=response_time
            )
            
            # Запись голоса в метрики с включением обоснования
            self.metrics.record_vote(
                agent.player_name, 
                name, 
                agent.is_spy,
                reasoning
            )
            
            if name not in self.living_players:
                self._announce(f"Host: {agent.player_name}, you voted for {name} who is not in the list of living players. Your vote will be ignored.", log_file)
                
                # Запись ошибки голосования
                self.metrics.record_event(
                    "error",
                    agent=agent.player_name,
                    error_type="invalid_vote",
                    vote=name
                )
                
                # Запись конца голосования
                self.metrics.record_event(
                    self.metrics.EVENT_TURN_END,
                    agent=agent.player_name,
                    is_spy=agent.is_spy,
                    stage="voting",
                    success=False
                )
                
                continue
                
            self._announce(f"{agent.player_name}: {speak}, i will vote {name} as the spy.", log_file)
            agent.private_history.append(create_message("assistant", json.dumps(cot)))
            name_list.append(name)
            
            # Запись конца голосования
            self.metrics.record_event(
                self.metrics.EVENT_TURN_END,
                agent=agent.player_name,
                is_spy=agent.is_spy,
                stage="voting",
                success=True
            )
            
        if not name_list and self.living_players:
            voted_name = random.choice(self.living_players)
            self._announce("Host: No valid votes were cast. Selecting a random player.", log_file)
            
            # Запись случайного выбора
            self.metrics.record_event(
                "random_selection",
                selected_player=voted_name
            )
            
            return voted_name
            
        return self.get_voted_name(name_list)

    def _collect_results(self, winner: str, spy_caught: bool) -> Dict[str, Any]:
        votes = {str(i+1): getattr(agent, 'last_vote', None) for i, agent in enumerate(self.agents) if hasattr(agent, 'last_vote')}
        
        # Вычисляем все метрики
        computed_metrics = self.metrics.compute_all()
        
        # Создаем имя файла метрик с временной меткой
        timestamp = int(time.time())
        metrics_filename = f"spyfall_metrics_{timestamp}.json"
        
        # Получаем путь к директории текущих результатов из переменной окружения или используем значение по умолчанию
        results_dir = os.environ.get("BENCHMARK_RESULTS_DIR", "benchmark_results")
        metrics_path = os.path.join(results_dir, metrics_filename)
        
        # Сохраняем метрики в файл
        self.metrics.save(metrics_path)
        
        # Базовые результаты
        results = {
            "winner": winner,
            "players": self.players,
            "spy_index": self.spy_index,
            "spy_id": self.spy_index,
            "spy_caught": spy_caught,
            "votes": votes,
            "vote_sequence": getattr(self, "vote_sequence", []),
            "vote_confidences": getattr(self, "vote_confidences", {}),
            "cot_data": getattr(self, "cot_data", {}),
            "round": self.game_round,
            "log": f"{self.game_round}.log",
            "descriptions": getattr(self, "player_descriptions", {}),
            "metrics_file": metrics_filename,
            "metrics": computed_metrics
        }
        
        return results