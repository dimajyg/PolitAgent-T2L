from llm.game import BaseGame
from environments.beast.agents.base_agent import BeastAgent
from environments.beast.utils.utils import create_message
from environments.beast.utils.prompt import get_current_wealth_prompt, get_voting_prompt
import random
from typing import Dict, List, Any, Optional, Tuple
import logging
from langchain_core.language_models.base import BaseLanguageModel
from pathlib import Path
import json
import os

class BeastGame(BaseGame):
    """
    Beast strategic wealth game implementation.
    
    Players engage in conversations, bargaining, and voting rounds to accumulate wealth.
    The game ends after 5 players are eliminated through voting.
    
    Args:
        args: Game configuration arguments
        llm: LangChain-compatible language model
    """
    def __init__(self, args: Any, llm: BaseLanguageModel):
        # Initialize with empty agents list since we'll populate it in init_game
        super().__init__([], {})
        
        self.llm = llm
        self.num_players = 10
        self.agents: List[BeastAgent] = []
        self.player_names: List[str] = []
        self.name2agent: Dict[str, BeastAgent] = {}
        self.eliminated_players: List[str] = []
        self.game_round = 0
        self.max_rounds = getattr(args, 'max_rounds', 5)  # Game ends after 5 eliminations by default
        self.output_dir = Path(getattr(args, 'output_dir', "./results/beast"))
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.debug = getattr(args, 'debug', False)

    def log_message(self, log_file, message, cot=None):
        """
        Log a message to the log file and console if debug is enabled.
        
        Args:
            log_file: File object for logging
            message: Message to log
            cot: Optional chain of thought to log as JSON
        """
        if log_file:
            log_file.write(message + "\n")
            if cot:
                log_file.write(json.dumps(cot) + "\n")
        
        if self.debug:
            print(message)
            if cot:
                print(json.dumps(cot))
            print()

    def init_game(self) -> str:
        """
        Initialize the game with random players and wealth distribution.
        
        Returns:
            String describing the initial game setup
        """
        # Initialize players with random wealth
        player_names = [f"Player_{i+1}" for i in range(self.num_players)]
        self.player_names = player_names
        
        for player_name in player_names:
            wealth = random.randint(0, 200000)
            agent = BeastAgent(self.llm, player_name, player_names, wealth)
            self.agents.append(agent)
            self.name2agent[player_name] = agent

        # Log initial game state
        settings = "Initial game settings:\n"
        for agent in self.agents:
            settings += f"{agent.player_name}: {agent.wealth} wealth\n"
        return settings

    def handle_conversation_stage(self, log_file: Any) -> bool:
        """
        Handle the conversation and bargaining stage.
        
        Args:
            log_file: File to log game events
            
        Returns:
            True if successful, False if an error occurred
        """
        # Update all agents with current wealth status
        current_wealth = {agent.player_name: agent.wealth for agent in self.agents if agent.player_name not in self.eliminated_players}
        wealth_status = get_current_wealth_prompt(current_wealth)
        wealth_message = create_message("user", wealth_status)
        self.update_history(wealth_message, "host")
        self.log_message(log_file, f"\nCurrent wealth status:\n{wealth_status}")

        # Get choices from agents
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

        logging.info(f"Conversation pairs: {conversations}")

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
                player2.private_history.append(create_message('user', f"{player1_name} sends you: {response1} with offer {offer1}"))
                
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
                player1.private_history.append(create_message('user', f"{player2_name} sends you: {response2} with offer {offer2}"))
                
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

    def handle_voting_stage(self, log_file: Any) -> Optional[str]:
        """
        Handle the voting stage where players vote for each other.
        
        Args:
            log_file: File to log game events
            
        Returns:
            Name of the player with most votes, or None if voting failed
        """
        votes: Dict[str, int] = {}
        for agent in self.agents:
            if agent.player_name not in self.eliminated_players:
                voted_player = agent.vote()
                if voted_player is None or voted_player in self.eliminated_players or voted_player == agent.player_name:
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

        # Format and share voting results with all agents
        voting_status = get_voting_prompt(votes)
        voting_message = create_message("user", voting_status)
        self.update_history(voting_message, "host")

        return winner

    def game_loop(self, log_file: Any) -> Dict[str, Any]:
        """
        Main game loop that runs until 5 players are eliminated.
        
        Args:
            log_file: File to log game events
            
        Returns:
            Dictionary with game results
        """
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
            
            # Save intermediate game state
            self._save_game_state(f"round_{self.game_round}")

        # Game over - calculate final results
        results = self._calculate_final_results()
        self._save_game_state("final")
        
        return results
    
    def _calculate_final_results(self) -> Dict[str, Any]:
        """Calculate and format the final game results.
        
        Returns:
            Dict[str, Any]: Structured results data
        """
        eliminated_players_data = [
                {"name": p, "wealth": self.name2agent[p].wealth}
                for p in self.eliminated_players
        ]
        
        remaining_players_data = [
                {"name": agent.player_name, "wealth": agent.wealth}
                for agent in self.agents
                if agent.player_name not in self.eliminated_players
        ]
        
        # Sort players by wealth
        eliminated_players_data.sort(key=lambda x: x["wealth"], reverse=True)
        remaining_players_data.sort(key=lambda x: x["wealth"], reverse=True)
        
        return {
            "eliminated_players": eliminated_players_data,
            "remaining_players": remaining_players_data,
            "total_rounds": self.game_round,
            "game": "beast"
        }

    def _save_game_state(self, suffix: str) -> None:
        """Save the current game state to a JSON file.
        
        Args:
            suffix: String suffix to add to the filename
        """
        state = {
            "round": self.game_round,
            "eliminated_players": self.eliminated_players,
            "players": {
                agent.player_name: {
                    "wealth": agent.wealth,
                    "eliminated": agent.player_name in self.eliminated_players
                }
                for agent in self.agents
            }
        }
        
        # Save to JSON file
        try:
            with open(self.output_dir / f"game_state_{suffix}.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save game state: {e}")
    
    def update_history(self, message: Dict[str, str], sender: str) -> None:
        """
        Update the history of all agents with a message.
        
        Args:
            message: Message to add to history
            sender: Sender of the message
        """
        for agent in self.agents:
            if agent.player_name != sender and agent.player_name not in self.eliminated_players:
                agent.private_history.append(message)