from base.game import Game
from tofukingdom.agents import ChefAgent, SpyAgent, MaidAgent, GuardAgent, QueenAgent, PrinceAgent, PrincessAgent, MinisterAgent
from tofukingdom.utils.utils import create_message
import random
import json

class TofuKingdomGame(Game):
    def __init__(self, args, prince_model, queen_model, spy_model):
        super().__init__(args)
        self.prince_model = prince_model
        self.queen_model = queen_model
        self.spy_model = spy_model
        self.players = ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward", "Robert"]
        self.prince = None

    def init_game(self):
        random.shuffle(self.players)
        self.agents = [
            PrincessAgent(self.prince_model, self.players[0], self.players),
            ChefAgent(self.prince_model, self.players[1], self.players),
            SpyAgent(self.spy_model, self.players[2], self.players),
            MaidAgent(self.spy_model, self.players[3], self.players),
            GuardAgent(self.queen_model, self.players[4], self.players),
            QueenAgent(self.queen_model, self.players[5], self.players),
            MinisterAgent(self.queen_model, self.players[6], self.players)
        ]
        random.shuffle(self.agents)
        for agent in self.agents:
            self.name2agent[agent.player_name] = agent

        self.prince = PrinceAgent(self.prince_model, self.players)
        return self.get_game_settings()

    def get_game_settings(self):
        settings = f"PrinceModel: {self.prince_model.name}\n QueenModel: {self.queen_model.name} \n SpyModel: {self.spy_model.name} \n"
        for agent in self.agents:
            settings += f"Player: {agent.player_name}; LLM: {agent.chatbot.name}; Identity: {agent.role}; \n"
        return settings

    def get_identity_text(self):
        return "".join([f"{agent.player_name} is the {agent.role}. \n" for agent in self.agents])

    def game_loop(self, log_file):
        if self.debug:
            self.log_message(log_file, self.get_game_settings())

        identities = self.get_identity_text()
        host_speech = "Host: The game now start."
        start_message = create_message("user", host_speech)
        self.update_history(start_message, "host")
        self.prince.private_history.append(start_message)
        self.log_message(log_file, host_speech)

        # Question round
        for agent in self.agents:
            if not self.handle_question_round(agent, log_file, identities):
                return {"error": "Question or answer is None"}

        # Extra question round
        if not self.handle_extra_question(log_file, identities):
            return {"error": "Extra question or answer is None"}

        # Final choice
        host_speech = "Host: Who do you think is the true princess?"
        host_message = create_message("user", host_speech)
        self.update_history(host_message, "host")
        self.prince.private_history.append(host_message)
        self.log_message(log_file, host_speech)

        name, cot = self.prince.choose()
        if name is None:
            return {"error": "Final answer is None"}

        if self.debug:
            print(f"The final choice is {name}")
            print(json.dumps(cot))

        return {"winner": self.name2agent[name].role, "log": f"{self.game_round}.log"}

    def handle_question_round(self, agent, log_file, identities):
        host_speech = f"Host: The Prince please ask player {agent.player_name} one question."
        host_message = create_message("user", host_speech)
        self.update_history(host_message, "host")
        self.prince.private_history.append(host_message)
        self.log_message(log_file, host_speech)

        question, cot = self.prince.ask()
        if question is None:
            return False

        temp = f"Prince: {question}"
        temp_message = create_message("user", temp)
        self.update_history(temp_message, "Prince")
        prince_message = create_message("assistant", json.dumps(cot))
        self.prince.private_history.append(prince_message)
        self.log_message(log_file, temp, cot)

        answer, cot = agent.chat(identities)
        if answer is None:
            return False

        temp = f"{agent.player_name}: {answer}"
        temp_message = create_message("user", temp)
        self.update_history(temp_message, agent.player_name)
        private_message = create_message("assistant", json.dumps(cot))
        agent.private_history.append(private_message)
        self.prince.private_history.append(temp_message)
        self.log_message(log_file, temp, cot)
        return True

    def handle_extra_question(self, log_file, identities):
        host_speech = "Host: The Prince please choose a player to ask an extra question."
        host_message = create_message("user", host_speech)
        self.update_history(host_message, "host")
        self.prince.private_history.append(host_message)
        self.log_message(log_file, host_speech)

        name, question, cot = self.prince.ask_choose()
        if name is None:
            return False

        temp = f"Prince: I choose {name}, my quesiton is {question}"
        temp_message = create_message("user", temp)
        self.update_history(temp_message, "Prince")
        prince_message = create_message("assistant", json.dumps(cot))
        self.prince.private_history.append(prince_message)
        self.log_message(log_file, temp, cot)

        agent = self.name2agent[name]
        answer, cot = agent.chat(identities)
        if answer is None:
            return False

        temp = f"{agent.player_name}: {answer}"
        temp_message = create_message("user", temp)
        self.update_history(temp_message, agent.player_name)
        private_message = create_message("assistant", json.dumps(cot))
        agent.private_history.append(private_message)
        self.prince.private_history.append(temp_message)
        self.log_message(log_file, temp, cot)
        return True