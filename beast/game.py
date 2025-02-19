from base.game import Game
from beast.agents.base_agent import BeastAgent
from beast.utils.utils import create_message
from beast.utils.prompt import get_current_wealth_prompt, get_voting_prompt
import random


class BeastGame(Game):
    def __init__(self, args, model):
        super().__init__(args)
        self.model = model
        self.num_players = 10
        self.agents = []
        self.player_names = []
        self.name2agent = {}
        self.eliminated_players = []
        self.game_round = 0
        self.max_rounds = 5  # Game ends after 5 eliminations

    def init_game(self):
        # Initialize players with random wealth
        player_names = [f"Player_{i+1}" for i in range(self.num_players)]
        self.player_names = player_names
        for player_name in player_names:
            wealth = random.randint(0, 200000)
            agent = BeastAgent(self.model, player_name, player_names, wealth)
            self.agents.append(agent)
            self.name2agent[player_name] = agent

        # Log initial game state
        settings = "Initial game settings:\n"
        for agent in self.agents:
            settings += f"{agent.player_name}: {agent.wealth} wealth\n"
        return settings

    def handle_conversation_stage(self, log_file):
        # Update all agents with current wealth status
        current_wealth = {agent.player_name: agent.wealth for agent in self.agents if agent.player_name not in self.eliminated_players}
        wealth_status = get_current_wealth_prompt(current_wealth)
        wealth_message = create_message("user", wealth_status)
        self.update_history(wealth_message, "host")
        self.log_message(log_file, f"\nCurrent wealth status:\n{wealth_status}")

        # Get choises from agents
        conversations = []
        remaining_players = list(set(self.player_names) - set(self.eliminated_players))

        for agent in self.agents:
            if agent.player_name not in self.eliminated_players:
                chosen_opponents = agent.choose_opponents(remaining_players)
                for opponent_name in chosen_opponents:
                    if opponent_name not in self.eliminated_players:
                        conversation_pair = tuple(sorted([agent.player_name, opponent_name]))
                        if conversation_pair not in conversations:
                            conversations.append(conversation_pair)

        print(conversations)

        # Handle conversations and money transfers
        for player1_name, player2_name in conversations:
            player1 = self.name2agent[player1_name]
            player2 = self.name2agent[player2_name]

            messages = []
            
            for _ in range(5):  # Max 5 messages per player
                # Player 1's turn
                response1, offer1 = player1.bargain(player2_name)
                if response1 is None:
                    return False
                messages.append((player1_name, f"{response1} with offer: {offer1}"))
                player2.private_history.append(create_message('assistant', f"{player1_name} send you {messages[-1][1]} with offer {offer1}"))
                
                if offer1 > 0 and offer1 <= player1.wealth:
                    accept = player2.handle_offer(player1_name, offer1)
                    if accept:
                        player1.wealth -= offer1
                        player2.wealth += offer1
                        self.log_message(log_file, f"{player1_name} transferred {offer1} to {player2_name}")
                        break

                # Player 2's turn
                response2, offer2 = player2.bargain(player1_name)
                if response2 is None:
                    return False
                messages.append((player2_name, f"{response2} with offer: {offer2}"))
                player1.private_history.append(create_message('assistant', f"{player2_name} send you {messages[-1][1]} with offer {offer2}"))
                
                if offer2 > 0 and offer2 <= player2.wealth:
                    accept = player1.handle_offer(player2_name, offer2)
                    if accept:
                        player2.wealth -= offer2
                        player1.wealth += offer2
                        self.log_message(log_file, f"{player2_name} transferred {offer2} to {player1_name}")
                        break

            # Log conversation
            self.log_message(log_file, f"Conversation between {player1_name} and {player2_name}:")
            for speaker, message in messages:
                self.log_message(log_file, f"{speaker}: {message}")

        return True

    def handle_voting_stage(self, log_file):
        votes = {}
        for agent in self.agents:
            if agent.player_name not in self.eliminated_players:
                voted_player = agent.vote()
                if voted_player in self.eliminated_players or voted_player == agent.player_name:
                    continue
                votes[voted_player] = votes.get(voted_player, 0) + 1

        # Find most voted player
        if not votes:
            return None

        max_votes = max(votes.values())
        winners = [p for p, v in votes.items() if v == max_votes]
        winner = random.choice(winners)

        # Log voting results
        self.log_message(log_file, "\nVoting results:")
        for player, vote_count in votes.items():
            self.log_message(log_file, f"{player}: {vote_count} votes")

        voting_status = get_voting_prompt(votes)
        wealth_message = create_message("user", voting_status)
        self.update_history(wealth_message, "host")

        return winner

    def game_loop(self, log_file):
        while len(self.eliminated_players) < 5:
            self.game_round += 1
            self.log_message(log_file, f"\nRound {self.game_round} begins")

            # Conversation stage
            if not self.handle_conversation_stage(log_file):
                return {"error": "Conversation stage failed"}

            # Voting stage
            winner = self.handle_voting_stage(log_file)
            if winner is None:
                return {"error": "Voting stage failed"}

            # Update winner's wealth and eliminate them
            self.name2agent[winner].wealth += 250000
            self.eliminated_players.append(winner)
            self.log_message(log_file, f"\n{winner} won the round and is eliminated with {self.name2agent[winner].wealth} wealth")

        # Game over - calculate final results
        results = {
            "eliminated_players": [
                {"name": p, "wealth": self.name2agent[p].wealth}
                for p in self.eliminated_players
            ],
            "remaining_players": [
                {"name": agent.player_name, "wealth": agent.wealth}
                for agent in self.agents
                if agent.player_name not in self.eliminated_players
            ],
            "rounds": self.game_round
        }

        return results