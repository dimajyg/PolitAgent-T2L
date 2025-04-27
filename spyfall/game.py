from base.game import Game
from spyfall.agents.base_agent import BaseAgent
from spyfall.utils.utils import create_message
import random
import copy
import json

from time import sleep
from spyfall.metrics.evaluation import SpyfallMetrics

class SpyfallGame(Game):
    def __init__(self, args, spy_model, villager_model):
        super().__init__(args)
        self.spy_model = spy_model
        self.villager_model = villager_model
        self.players = ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward"]
        self.living_players = []
        self.spy_name = None
        self.spy_index = None

    def init_game(self, phrase_pair):
        self.living_players = copy.deepcopy(self.players)
        spy_word, villager_word = phrase_pair
        random.shuffle(self.players)
        self.spy_index = random.randint(1, len(self.players))
        self.spy_name = self.players[self.spy_index - 1]

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
        return settings

    def get_voted_name(self, name_list):
        counts = {}
        for name in name_list:
            counts[name] = counts.get(name, 0) + 1
        max_count = max(counts.values()) if counts else 0
        for name, count in counts.items():
            if count == max_count:
                return name, sorted(counts.values())
        return None, []

    def game_loop(self, log_file):
        host_speech = "Host: The game now start."
        start_message = create_message("user", host_speech)
        self.update_history(start_message, "host")
        self.log_message(log_file, host_speech)

        # Сообщаем всем игрокам их роли
        for agent in self.agents:
            role_info = "spy" if agent.is_spy else "villager"
            host_speech = f"Host: {agent.player_name}, you are a {role_info}."
            role_message = create_message("user", host_speech)
            # Отправляем сообщение только этому агенту
            agent.private_history.append(role_message)
            self.log_message(log_file, f"[PRIVATE] {host_speech}")

        host_speech = f"Host: The living players are: {json.dumps(self.living_players)}"
        living_player_message = create_message("user", host_speech)
        self.update_history(living_player_message, "host")
        self.log_message(log_file, host_speech)

        while True:
            self.game_round += 1
            
            # Проверяем, есть ли шпион в списке живых игроков
            if self.spy_name not in self.living_players:
                host_speech = f"Host: The spy {self.spy_name} has been eliminated! Villagers win!"
                host_message = create_message("user", host_speech)
                self.update_history(host_message, "host")
                self.log_message(log_file, host_speech)
                
                # Collect votes data
                votes = {}
                for i, agent in enumerate(self.agents):
                    if hasattr(agent, 'last_vote') and agent.last_vote:
                        votes[str(i+1)] = agent.last_vote
                        
                return {
                    "winner": "villager", 
                    "players": self.players,
                    "spy_index": self.spy_index, 
                    "spy_id": self.spy_index,
                    "spy_caught": True,
                    "votes": votes,
                    "vote_sequence": self.vote_sequence if hasattr(self, "vote_sequence") else [],
                    "vote_confidences": self.vote_confidences if hasattr(self, "vote_confidences") else {},
                    "cot_data": self.cot_data if hasattr(self, "cot_data") else {},
                    "round": self.game_round,
                    "log": f"{self.game_round}.log",
                    "descriptions": self.player_descriptions if hasattr(self, "player_descriptions") else {},
                    "description_metrics": {
                        player: {
                            "specificity": SpyfallMetrics.description_specificity(desc),
                            "perplexity": SpyfallMetrics.calculate_perplexity(desc)
                        } for player, desc in (self.player_descriptions.items() if hasattr(self, "player_descriptions") else {})
                    },
                    "vagueness_scores": SpyfallMetrics.vagueness_score(list(self.player_descriptions.values()))
                    if hasattr(self, "player_descriptions") and self.player_descriptions else {}
                }
                        
            # Describing stage
            if not self.handle_describing_stage(log_file):
                return {"error": "Description stage failed"}

            # Voting stage
            max_vote_attempts = 3  # Ограничиваем количество попыток переголосования
            vote_attempt = 0
            
            while vote_attempt < max_vote_attempts:
                vote_attempt += 1
                voted_name = self.handle_voting_stage(log_file)
                
                if voted_name == self.spy_name:
                    host_speech = f"Host: {voted_name} was the spy! Villagers win!"
                    host_message = create_message("user", host_speech)
                    self.update_history(host_message, "host")
                    self.log_message(log_file, host_speech)
                    
                    # Collect votes data
                    votes = {}
                    for i, agent in enumerate(self.agents):
                        if hasattr(agent, 'last_vote') and agent.last_vote:
                            votes[str(i+1)] = agent.last_vote
                            
                    return {
                        "winner": "villager", 
                        "players": self.players,
                        "spy_index": self.spy_index, 
                        "spy_id": self.spy_index,
                        "spy_caught": True,
                        "votes": votes,
                        "vote_sequence": self.vote_sequence if hasattr(self, "vote_sequence") else [],
                        "vote_confidences": self.vote_confidences if hasattr(self, "vote_confidences") else {},
                        "cot_data": self.cot_data if hasattr(self, "cot_data") else {},
                        "round": self.game_round,
                        "log": f"{self.game_round}.log",
                        "descriptions": self.player_descriptions if hasattr(self, "player_descriptions") else {},
                        "description_metrics": {
                            player: {
                                "specificity": SpyfallMetrics.description_specificity(desc),
                                "perplexity": SpyfallMetrics.calculate_perplexity(desc)
                            } for player, desc in (self.player_descriptions.items() if hasattr(self, "player_descriptions") else {})
                        },
                        "vagueness_scores": SpyfallMetrics.vagueness_score(list(self.player_descriptions.values()))
                        if hasattr(self, "player_descriptions") and self.player_descriptions else {}
                    }

                if voted_name in self.living_players:
                    self.living_players.remove(voted_name)
                    host_speech = f"Host: the voting result is {voted_name}, he is not the spy. The spy still lives. {voted_name} left the game."
                    host_message = create_message("user", host_speech)
                    self.update_history(host_message, "host")
                    self.log_message(log_file, host_speech)
                    break  # Выходим из цикла переголосовок если успешно
                else:
                    host_speech = f"Host: {voted_name} is not in the list of living players or has already left the game. Starting a revote."
                    host_message = create_message("user", host_speech)
                    self.update_history(host_message, "host")
                    self.log_message(log_file, host_speech)
                    # Продолжаем цикл для новой попытки голосования
                    continue

            # Если переголосовка не удалась после максимального числа попыток
            if vote_attempt >= max_vote_attempts:
                host_speech = f"Host: After multiple attempts, no valid vote could be made. Moving to the next round."
                host_message = create_message("user", host_speech)
                self.update_history(host_message, "host")
                self.log_message(log_file, host_speech)

            # Проверяем условие победы шпиона (строго меньше 3 игроков)
            if len(self.living_players) < 3:
                host_speech = f"Host: Less than 3 players remain and the spy still lives! Spy {self.spy_name} wins!"
                host_message = create_message("user", host_speech)
                self.update_history(host_message, "host")
                self.log_message(log_file, host_speech)
                
                # Collect votes data
                votes = {}
                for i, agent in enumerate(self.agents):
                    if hasattr(agent, 'last_vote') and agent.last_vote:
                        votes[str(i+1)] = agent.last_vote
                        
                return {
                    "winner": "spy", 
                    "players": self.players,
                    "spy_index": self.spy_index, 
                    "spy_id": self.spy_index,
                    "spy_caught": False,
                    "votes": votes,
                    "vote_sequence": self.vote_sequence if hasattr(self, "vote_sequence") else [],
                    "vote_confidences": self.vote_confidences if hasattr(self, "vote_confidences") else {},
                    "cot_data": self.cot_data if hasattr(self, "cot_data") else {},
                    "round": self.game_round,
                    "log": f"{self.game_round}.log",
                    "descriptions": self.player_descriptions if hasattr(self, "player_descriptions") else {},
                    "description_metrics": {
                        player: {
                            "specificity": SpyfallMetrics.description_specificity(desc),
                            "perplexity": SpyfallMetrics.calculate_perplexity(desc)
                        } for player, desc in (self.player_descriptions.items() if hasattr(self, "player_descriptions") else {})
                    },
                    "vagueness_scores": SpyfallMetrics.vagueness_score(list(self.player_descriptions.values()))
                    if hasattr(self, "player_descriptions") and self.player_descriptions else {}
                }

    def handle_describing_stage(self, log_file):
        host_speech = "Host: Now it's the describing stage, players have to say something about the received word."
        host_message = create_message("user", host_speech)
        self.update_history(host_message, "host")
        self.log_message(log_file, host_speech)

        # Track descriptions for metrics
        self.descriptions = []
        self.player_descriptions = {}
        
        for agent in self.agents:
            if agent.player_name not in self.living_players:
                continue

            host_speech = f"Host: {agent.player_name}, it's your turn."
            host_message = create_message("user", host_speech)
            self.update_history(host_message, "host")
            self.log_message(log_file, host_speech)

            sleep(2)
            description, cot = agent.describe()
            print(description)
            if description is None:
                return False

            temp = f"{agent.player_name}: {description}"
            public_message = create_message("user", temp)
            self.update_history(public_message, agent.player_name)
            private_message = create_message("assistant", json.dumps(cot))
            agent.private_history.append(private_message)
            self.log_message(log_file, temp, cot)

            self.descriptions.append(description)
            self.player_descriptions[agent.player_name] = description

        return True

    def handle_voting_stage(self, log_file):
        # Важно: обновляем информацию о живых игроках перед началом голосования
        host_speech = f"Host: The living players are: {json.dumps(self.living_players)}"
        living_player_message = create_message("user", host_speech)
        self.update_history(living_player_message, "host")
        self.log_message(log_file, host_speech)
        
        host_speech = "Host: Now the voting start, please vote for the player you think is the spy. You must vote ONLY for a player from the list of living players."
        host_message = create_message("user", host_speech)
        self.update_history(host_message, "host")
        self.log_message(log_file, host_speech)

        name_list = []
        vote_confidences = {}  # Store confidence values for Brier score
        vote_sequence = []     # Store voting sequence for influence index
        
        for agent in self.agents:
            if agent.player_name not in self.living_players:
                continue

            host_speech = f"Host: {agent.player_name}, it's your turn to vote. Remember to choose from the living players: {json.dumps(self.living_players)}"
            host_message = create_message("user", host_speech)
            self.update_history(host_message, "host")
            self.log_message(log_file, host_speech)

            sleep(2)
            name, speak, cot = agent.vote()
            # Store vote for metrics
            agent.last_vote = name
            
            # Track voting sequence (player index, vote)
            player_idx = self.players.index(agent.player_name) + 1
            vote_sequence.append((player_idx, name))
            
            # Extract confidence from cot if available (example format)
            # Expected CoT format: {"thought": "...", "confidence": {"Player1": 20, "Player2": 80, ...}}
            confidence_data = {}
            if isinstance(cot, dict) and "confidence" in cot:
                confidence_data = cot["confidence"]
            vote_confidences[player_idx] = confidence_data
            
            # Store CoT data for coherence metric
            if not hasattr(self, "cot_data"):
                self.cot_data = {}
            
            if isinstance(cot, dict):
                self.cot_data[player_idx] = {
                    "thought": cot.get("thought", ""),
                    "speak": speak
                }
            
            # Валидация голоса - проверяем, что голос отдан за живого игрока
            if name not in self.living_players:
                host_speech = f"Host: {agent.player_name}, you voted for {name} who is not in the list of living players. Your vote will be ignored."
                host_message = create_message("user", host_speech)
                self.update_history(host_message, "host")
                self.log_message(log_file, host_speech)
                # Пропускаем добавление невалидного голоса
                continue
                
            temp = f"{agent.player_name}: {speak}, i will vote {name} as the spy."
            public_message = create_message("user", temp)
            self.update_history(public_message, agent.player_name)
            private_message = create_message("assistant", json.dumps(cot))
            agent.private_history.append(private_message)
            self.log_message(log_file, temp, cot)

            name_list.append(name)

        if not name_list:
            # Если нет валидных голосов, выбираем случайного живого игрока
            host_speech = "Host: No valid votes were cast. Selecting a random player."
            host_message = create_message("user", host_speech)
            self.update_history(host_message, "host")
            self.log_message(log_file, host_speech)
            
            if self.living_players:
                voted_name = random.choice(self.living_players)
                return voted_name
            else:
                # Это маловероятный случай, но нужно обработать ситуацию, когда нет живых игроков
                return None

        voted_name, _ = self.get_voted_name(name_list)

        # Store vote sequence and confidences for metrics
        self.vote_sequence = vote_sequence
        self.vote_confidences = vote_confidences
        
        return voted_name