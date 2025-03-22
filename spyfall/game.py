from base.game import Game
from spyfall.agents.base_agent import BaseAgent
from spyfall.utils.utils import create_message
import random
import copy
import json

from time import sleep

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
            agent = BaseAgent(model, player_name, self.players, phrase)
            self.agents.append(agent)
            self.name2agent[player_name] = agent

        settings = f"The spy word is: {spy_word};\n The villager word is {villager_word}.\n"
        for agent in self.agents:
            settings += f"Player: {agent.player_name}; Assigned Word: {agent.phrase} \n"
        return settings

    def get_voted_name(self, name_list):
        counts = {}
        for name in name_list:
            counts[name] = counts.get(name, 0) + 1
        max_count = max(counts.values())
        for name, count in counts.items():
            if count == max_count:
                return name, sorted(counts.values())
        return None, []

    def game_loop(self, log_file):
        host_speech = "Host: The game now start."
        start_message = create_message("user", host_speech)
        self.update_history(start_message, "host")
        self.log_message(log_file, host_speech)

        host_speech = f"Host: The living players are:{json.dumps(self.living_players)}"
        living_player_message = create_message("user", host_speech)
        self.update_history(living_player_message, "host")
        self.log_message(log_file, host_speech)

        while True:
            self.game_round += 1
            # Describing stage
            if not self.handle_describing_stage(log_file):
                return {"error": "Description stage failed"}

            # Voting stage
            while True:  # Добавляем внутренний цикл для переголосовок
                voted_name = self.handle_voting_stage(log_file)
                
                if voted_name == self.spy_name:
                    return {"winner": "villager", "players": self.players,
                            "spy_index": self.spy_index, "round": self.game_round,
                            "log": f"{self.game_round}.log"}

                try:
                    self.living_players.remove(voted_name)
                    host_speech = f"Host: the voting result is {voted_name}, he is not the spy. The spy still lives. {voted_name} left the game."
                    host_message = create_message("user", host_speech)
                    self.update_history(host_message, "host")
                    self.log_message(log_file, host_speech)
                    break  # Выходим из цикла переголосовок если успешно
                except Exception as e:
                    print(e)
                    host_speech = f"Host: {voted_name} has already left the game. Starting a revote."
                    host_message = create_message("user", host_speech)
                    self.update_history(host_message, "host")
                    self.log_message(log_file, host_speech)
                    # Продолжаем цикл для новой попытки голосования
                    continue

            if len(self.living_players) <= 3:
                return {"winner": "spy", "players": self.players,
                        "spy_index": self.spy_index, "round": self.game_round,
                        "log": f"{self.game_round}.log"}

    def handle_describing_stage(self, log_file):
        host_speech = "Host: Now it's the describing stage, players have to say something about the received word."
        host_message = create_message("user", host_speech)
        self.update_history(host_message, "host")
        self.log_message(log_file, host_speech)

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
        return True

    def handle_voting_stage(self, log_file):
        host_speech = "Host: Now the voting start, please vote for the player you think is the spy."
        host_message = create_message("user", host_speech)
        self.update_history(host_message, "host")
        self.log_message(log_file, host_speech)

        name_list = []
        for agent in self.agents:
            if agent.player_name not in self.living_players:
                continue

            host_speech = f"Host: {agent.player_name}, it's your turn."
            host_message = create_message("user", host_speech)
            self.update_history(host_message, "host")
            self.log_message(log_file, host_speech)

            sleep(2)
            name, speak, cot = agent.vote()
            temp = f"{agent.player_name}: {speak}, i will vote {name} as the spy."
            public_message = create_message("user", temp)
            self.update_history(public_message, agent.player_name)
            private_message = create_message("assistant", json.dumps(cot))
            agent.private_history.append(private_message)
            self.log_message(log_file, temp, cot)

            name_list.append(name)

        voted_name, _ = self.get_voted_name(name_list)
        return voted_name