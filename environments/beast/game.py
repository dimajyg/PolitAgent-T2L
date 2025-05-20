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
import time
from datetime import datetime

from metrics.beast_metrics import BeastMetrics

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
        # Use the benchmark results directory instead of a separate output directory
        self.output_dir = Path(os.environ.get("BENCHMARK_RESULTS_DIR", "./benchmark_results"))
        self.debug = getattr(args, 'debug', False)
        
        # Initialize metrics
        self.metrics = BeastMetrics(metadata={
            "game_id": f"beast_{int(time.time())}",
            "model": getattr(llm, "__class__.__name__", str(llm)),
            "num_players": self.num_players,
            "max_rounds": self.max_rounds
        })
        
        # Enable LLM evaluation if specified
        self.use_llm_evaluation = getattr(args, "use_llm_evaluation", False)
        if self.use_llm_evaluation:
            evaluator_model = getattr(args, "evaluation_model", None)
            if evaluator_model is None:
                evaluator_model = llm
            self.metrics.enable_llm_evaluation(evaluator_model)
            logging.info("LLM evaluation of game performance enabled")

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
        
        # Record game start event
        self.metrics.record_event(
            self.metrics.EVENT_GAME_START,
            players=player_names
        )
        
        for player_name in player_names:
            wealth = random.randint(0, 200000)
            agent = BeastAgent(self.llm, player_name, player_names, wealth)
            self.agents.append(agent)
            self.name2agent[player_name] = agent
            
            # Record initial wealth in metrics
            self.metrics.record_initial_wealth(player_name, wealth)

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
            transfer_outcome = None
            
            for _ in range(5):  # Max 5 messages per player
                # Player 1's turn
                response1, offer1 = player1.bargain(player2_name)
                if response1 is None:
                    return False
                messages.append({"speaker": player1_name, "message": response1, "offer": offer1})
                player2.private_history.append(create_message('user', f"{player1_name} sends you: {response1} with offer {offer1}"))
                
                if offer1 > 0 and offer1 <= player1.wealth:
                    accept = player2.handle_offer(player1_name, offer1)
                    if accept:
                        player1.wealth -= offer1
                        player2.wealth += offer1
                        transfer_outcome = {
                            "from": player1_name,
                            "to": player2_name,
                            "amount": offer1,
                            "reason": "bargain"
                        }
                        self.log_message(log_file, f"{player1_name} transferred {offer1} to {player2_name}")
                        
                        # Record the wealth transfer in metrics
                        self.metrics.record_wealth_transfer(
                            from_player=player1_name,
                            to_player=player2_name,
                            amount=offer1,
                            round_num=self.game_round,
                            reason="bargain"
                        )
                        break

                # Player 2's turn
                response2, offer2 = player2.bargain(player1_name)
                if response2 is None:
                    return False
                messages.append({"speaker": player2_name, "message": response2, "offer": offer2})
                player1.private_history.append(create_message('user', f"{player2_name} sends you: {response2} with offer {offer2}"))
                
                if offer2 > 0 and offer2 <= player2.wealth:
                    accept = player1.handle_offer(player2_name, offer2)
                    if accept:
                        player2.wealth -= offer2
                        player1.wealth += offer2
                        transfer_outcome = {
                            "from": player2_name,
                            "to": player1_name,
                            "amount": offer2,
                            "reason": "bargain"
                        }
                        self.log_message(log_file, f"{player2_name} transferred {offer2} to {player1_name}")
                        
                        # Record the wealth transfer in metrics
                        self.metrics.record_wealth_transfer(
                            from_player=player2_name,
                            to_player=player1_name,
                            amount=offer2,
                            round_num=self.game_round,
                            reason="bargain"
                        )
                        break

            # Log conversation
            self.log_message(log_file, f"Conversation between {player1_name} and {player2_name}:")
            for msg in messages:
                speaker = msg["speaker"]
                message_text = msg["message"]
                offer = msg.get("offer", 0)
                self.log_message(log_file, f"{speaker}: {message_text} with offer: {offer}")
            
            # Record the conversation in metrics
            self.metrics.record_conversation(
                player1=player1_name,
                player2=player2_name,
                messages=messages,
                round_num=self.game_round,
                transfer_outcome=transfer_outcome
            )

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
                
                # Record vote in metrics
                self.metrics.record_vote(
                    voter=agent.player_name,
                    voted_for=voted_player,
                    round_num=self.game_round
                )

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
        try:
            while len(self.eliminated_players) < 5:
                self.game_round += 1
                self.log_message(log_file, f"\nRound {self.game_round} begins")
                
                # Record round start event
                self.metrics.record_event(
                    self.metrics.EVENT_ROUND_START,
                    round_number=self.game_round
                )

                # Conversation stage
                if not self.handle_conversation_stage(log_file):
                    # Record error in metrics
                    self.metrics.record_event("error", error_type="Conversation stage failed")
                    return {"error": "Conversation stage failed"}

                # Voting stage
                winner = self.handle_voting_stage(log_file)
                if winner is None:
                    # Record error in metrics
                    self.metrics.record_event("error", error_type="Voting stage failed")
                    return {"error": "Voting stage failed"}

                # Update winner's wealth and eliminate them
                previous_wealth = self.name2agent[winner].wealth
                self.name2agent[winner].wealth += 250000
                
                # Record bonus wealth in metrics
                self.metrics.update_player_wealth(
                    player_name=winner,
                    wealth=self.name2agent[winner].wealth,
                    round_num=self.game_round,
                    reason="bonus_on_elimination"
                )
                
                self.eliminated_players.append(winner)
                self.log_message(log_file, f"\n{winner} won the round and is eliminated with {self.name2agent[winner].wealth} wealth")
                
                # Record elimination in metrics
                self.metrics.record_elimination(
                    player=winner,
                    round_num=self.game_round,
                    wealth=self.name2agent[winner].wealth
                )
                
                # Record round end event
                self.metrics.record_event(
                    self.metrics.EVENT_ROUND_END,
                    round_number=self.game_round
                )
                
                # Log current game state instead of saving to file
                self._log_game_state(log_file, f"round_{self.game_round}")

            # Game over - calculate final results
            results = self._calculate_final_results()
            
            # Record game end event
            self.metrics.record_event(
                self.metrics.EVENT_GAME_END,
                success=True,
                remaining_players=len(self.player_names) - len(self.eliminated_players),
                result=results
            )
            
            # Run LLM evaluation if enabled
            if self.use_llm_evaluation:
                game_evaluation = self.metrics.evaluate_game()
                if game_evaluation:
                    results["llm_evaluation"] = game_evaluation
            
            # Compute metrics
            metrics_data = self.metrics.compute_all()
            
            # Save metrics to file
            metrics_file = self._save_metrics()
            self.log_message(log_file, f"\nGame metrics saved to: {metrics_file}")
            
            # Log final game state
            self._log_game_state(log_file, "final")
            
            # Include only essential metrics in results
            results["metrics_file"] = metrics_file
            
            # Add key metrics summary
            results["metrics_summary"] = {
                "wealth_inequality": metrics_data.get("wealth_metrics", {}).get("inequality", {}),
                "conversation_success_rate": metrics_data.get("conversation_metrics", {}).get("success_rate", 0),
                "total_wealth_transferred": metrics_data.get("transfer_metrics", {}).get("total_transferred", 0),
                "rounds_played": self.game_round
            }
            
            return results
            
        except Exception as e:
            logging.exception("Error in game loop")
            
            # Record error in metrics
            self.metrics.record_event(
                "error",
                error_type=str(e)
            )
            
            # Record game end with error
            self.metrics.record_event(
                self.metrics.EVENT_GAME_END,
                success=False,
                error=str(e)
            )
            
            # Compute metrics despite error
            metrics_data = self.metrics.compute_all()
            metrics_file = self._save_metrics()
            
            error_result = {
                "error": str(e), 
                "metrics_file": metrics_file,
                "rounds_played": self.game_round
            }
            return error_result
    
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
        
        # Calculate winner (remaining player with most wealth)
        winner = None
        max_wealth = -1
        for player in remaining_players_data:
            if player["wealth"] > max_wealth:
                max_wealth = player["wealth"]
                winner = player["name"]
        
        return {
            "eliminated_players": eliminated_players_data,
            "remaining_players": remaining_players_data,
            "rounds_played": self.game_round,
            "winner": winner,
            "winner_wealth": max_wealth if winner else 0
        }
    
    def _log_game_state(self, log_file: Any, stage: str) -> None:
        """Log the current game state in a detailed format.
        
        Args:
            log_file: File object for logging
            stage (str): Description of the current game stage
        """
        if not log_file:
            return
            
        # Gather current game state
        game_state = {
            "stage": stage,
            "round": self.game_round,
            "player_wealth": {agent.player_name: agent.wealth for agent in self.agents},
            "eliminated_players": self.eliminated_players,
            "remaining_players": [a.player_name for a in self.agents if a.player_name not in self.eliminated_players]
        }
        
        # Log state in a readable format
        self.log_message(log_file, f"\n--- Game State: {stage} ---")
        self.log_message(log_file, f"Round: {self.game_round}")
        
        # Log eliminated players
        if self.eliminated_players:
            self.log_message(log_file, "Eliminated Players:")
            for player in self.eliminated_players:
                self.log_message(log_file, f"  {player}: {self.name2agent[player].wealth}")
        
        # Log remaining players and their wealth
        self.log_message(log_file, "Player Wealth:")
        for player, wealth in game_state["player_wealth"].items():
            if player not in self.eliminated_players:
                self.log_message(log_file, f"  {player}: {wealth}")
        
        # Log wealth statistics when appropriate
        if stage == "final" or stage.startswith("round_"):
            # Calculate wealth statistics
            remaining_wealth = [self.name2agent[p].wealth for p in game_state["remaining_players"]]
            if remaining_wealth:
                total_wealth = sum(remaining_wealth)
                avg_wealth = total_wealth / len(remaining_wealth)
                max_wealth = max(remaining_wealth)
                min_wealth = min(remaining_wealth)
                
                self.log_message(log_file, "Wealth Statistics:")
                self.log_message(log_file, f"  Total Wealth: {total_wealth}")
                self.log_message(log_file, f"  Average Wealth: {avg_wealth:.2f}")
                self.log_message(log_file, f"  Wealth Range: {min_wealth} - {max_wealth}")
        
        # Add separation for readability
        self.log_message(log_file, "--------------------------------")
    
    def _save_metrics(self) -> str:
        """
        Save metrics to a file with improved formatting.
        
        Returns:
            String with the path to the saved metrics file
        """
        # Create metrics filename with human-readable timestamp
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        metrics_filename = f"beast_metrics_{date_str}.json"
        
        # Use the output_dir instead of creating a separate results directory
        metrics_path = self.output_dir / metrics_filename
        
        # Compute metrics if not already computed
        if not self.metrics.computed_metrics:
            self.metrics.compute_all()
            
        # Format the metrics JSON with nice indentation and ensure UTF-8 encoding
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics.computed_metrics, f, indent=4, ensure_ascii=False)
        
        return str(metrics_path)
    
    def update_history(self, message: Dict[str, str], sender: str) -> None:
        """Update the message history for all agents.
        
        Args:
            message (Dict[str, str]): Message to add to history
            sender (str): Sender of the message
        """
        for agent in self.agents:
            if sender == "host" or sender != agent.player_name:
                agent.private_history.append(message)