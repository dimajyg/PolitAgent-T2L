from llm.game import Game
from environments.spyfall.agents.base_agent import BaseAgent
from environments.spyfall.utils.utils import create_message
from environments.spyfall.spyfall_metrics import SpyfallMetrics
import random
import copy
import json
import logging
from time import sleep
from typing import List, Dict, Any, Optional

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

    def init_game(self, phrase_pair: List[str]) -> str:
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
            self.game_round += 1

            if self.spy_name not in self.living_players:
                self._announce(f"Host: The spy {self.spy_name} has been eliminated! Villagers win!", log_file)
                return self._collect_results(winner="villager", spy_caught=True)

            if not self.handle_describing_stage(log_file):
                return {"error": "Description stage failed"}

            if self._handle_voting(log_file):
                return self._collect_results(winner="villager", spy_caught=True)

            if len(self.living_players) < 3:
                self._announce(f"Host: Less than 3 players remain and the spy still lives! Spy {self.spy_name} wins!", log_file)
                return self._collect_results(winner="spy", spy_caught=False)

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
            self._announce(f"Host: {agent.player_name}, it's your turn.", log_file)
            sleep(2)
            description, cot = agent.describe()
            if description is None:
                logger.error(f"Agent {agent.player_name} failed to describe.")
                return False
            self._announce(f"{agent.player_name}: {description}", log_file)
            agent.private_history.append(create_message("assistant", json.dumps(cot)))
            self.descriptions.append(description)
            self.player_descriptions[agent.player_name] = description
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
            self._announce(f"Host: {agent.player_name}, it's your turn to vote. Remember to choose from the living players: {json.dumps(self.living_players)}", log_file)
            sleep(2)
            name, speak, cot = agent.vote()
            agent.last_vote = name
            player_idx = self.players.index(agent.player_name) + 1
            self.vote_sequence.append((player_idx, name))
            confidence_data = cot.get("confidence", {}) if isinstance(cot, dict) else {}
            self.vote_confidences[player_idx] = confidence_data
            if not hasattr(self, "cot_data"):
                self.cot_data = {}
            if isinstance(cot, dict):
                self.cot_data[player_idx] = {"thought": cot.get("thought", ""), "speak": speak}
            if name not in self.living_players:
                self._announce(f"Host: {agent.player_name}, you voted for {name} who is not in the list of living players. Your vote will be ignored.", log_file)
                continue
            self._announce(f"{agent.player_name}: {speak}, i will vote {name} as the spy.", log_file)
            agent.private_history.append(create_message("assistant", json.dumps(cot)))
            name_list.append(name)
        if not name_list and self.living_players:
            voted_name = random.choice(self.living_players)
            self._announce("Host: No valid votes were cast. Selecting a random player.", log_file)
            return voted_name
        return self.get_voted_name(name_list)

    def _collect_results(self, winner: str, spy_caught: bool) -> Dict[str, Any]:
        votes = {str(i+1): getattr(agent, 'last_vote', None) for i, agent in enumerate(self.agents) if hasattr(agent, 'last_vote')}
        return {
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
            "description_metrics": {
                player: {
                    "specificity": SpyfallMetrics.description_specificity(desc),
                    "perplexity": SpyfallMetrics.calculate_perplexity(desc)
                } for player, desc in getattr(self, "player_descriptions", {}).items()
            },
            "vagueness_scores": SpyfallMetrics.vagueness_score(list(getattr(self, "player_descriptions", {}).values()))
            if getattr(self, "player_descriptions", None) else {}
        }