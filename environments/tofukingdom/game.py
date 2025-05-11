from llm.game import BaseGame
from environments.tofukingdom.agents import GameController
from typing import Dict, List, Any, Optional
import logging
import time
import os
import json

# Import the metrics
from metrics.tofukingdom_metrics import TofuKingdomMetrics

class TofuKingdomGame(BaseGame):
    """
    TofuKingdom game implementation compatible with the PolitAgent benchmark system.
    
    Args:
        args: Game configuration arguments
        prince_llm: LangChain-compatible language model for the Prince
        princess_llm: LangChain-compatible language model for the Princess team (Princess, Chef)
        queen_llm: LangChain-compatible language model for the Queen team (Queen, Minister, Guard)
        neutral_llm: LangChain-compatible language model for the Neutral team (Maid, Spy)
    """
    def __init__(self, args, prince_llm, princess_llm=None, queen_llm=None, neutral_llm=None):
        super().__init__(args)
        
        # For backward compatibility - if only prince_llm is provided, use it for all roles
        if princess_llm is None:
            princess_llm = prince_llm
        if queen_llm is None:
            queen_llm = prince_llm
        if neutral_llm is None:
            neutral_llm = prince_llm
        
        # Set up logging
        self.debug = getattr(args, 'debug', False)
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("TofuKingdomGame")
        
        # Default player names if none provided
        self.players = getattr(args, 'players', ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward", "Robert"])
        
        # Create the game controller
        self.controller = GameController(
            prince_llm=prince_llm,
            team_princess_llm=princess_llm,
            team_queen_llm=queen_llm,
            team_neutral_llm=neutral_llm,
            player_names=self.players,
            debug=self.debug
        )
        
        # Initialize metrics
        self.metrics = TofuKingdomMetrics(metadata={
            "game_id": f"tofukingdom_{int(time.time())}",
            "prince_model": getattr(prince_llm, "__class__.__name__", str(prince_llm)),
            "princess_model": getattr(princess_llm, "__class__.__name__", str(princess_llm)),
            "queen_model": getattr(queen_llm, "__class__.__name__", str(queen_llm)),
            "neutral_model": getattr(neutral_llm, "__class__.__name__", str(neutral_llm))
        })
        
        # Enable LLM evaluation if specified
        self.use_llm_evaluation = getattr(args, "use_llm_evaluation", False)
        if self.use_llm_evaluation:
            evaluator_model = getattr(args, "evaluation_model", None)
            if evaluator_model is None:
                evaluator_model = prince_llm
            self.metrics.enable_llm_evaluation(evaluator_model)
            self.logger.info("LLM evaluation of game performance enabled")
        
    def init_game(self) -> str:
        """
        Initialize the game with random player assignments.
        
        Returns:
            String describing the initial game setup
        """
        # Initialize game through controller
        game_setup = self.controller.initialize_game()
        
        # Record player roles in metrics
        identities = game_setup.get("identities", {})
        for player, role in identities.items():
            self.metrics.set_player_role(player, role)
        
        # Record game start event
        self.metrics.record_event(
            self.metrics.EVENT_GAME_START,
            players=self.players,
            identities=identities
        )
        
        # Format the game setup as a string
        return self._format_game_setup(game_setup)
    
    def _format_game_setup(self, game_setup: Dict[str, Any]) -> str:
        """
        Format game setup information as a string.
        
        Args:
            game_setup: Dictionary with game configuration
            
        Returns:
            Formatted setup information string
        """
        identities = game_setup.get("identities", {})
        
        setup_str = "Game Configuration:\n"
        setup_str += f"Prince: Prince\n"
        setup_str += "Player Assignments:\n"
        
        for player, role in identities.items():
            setup_str += f"Player: {player}; Role: {role}\n"
            
        return setup_str
    
    def get_game_settings(self) -> str:
        """
        Get a description of the game setup.
        
        Returns:
            String with game configuration details
        """
        if not self.controller.game_initialized:
            return "Game not initialized yet."
            
        return self._format_game_setup({
            "identities": self.controller.identities
        })

    def get_identities(self) -> Dict[str, str]:
        """
        Get a mapping of player names to their roles.
        
        Returns:
            Dictionary mapping player names to their roles
        """
        return self.controller.identities

    def get_identity_text(self) -> str:
        """
        Get a text representation of all identities for logging.
        
        Returns:
            String listing all player identities
        """
        return "".join([f"{player} is the {role}. \n" 
                      for player, role in self.controller.identities.items()])

    def _save_metrics(self) -> str:
        """
        Save metrics to a file.
        
        Returns:
            String with the path to the saved metrics file
        """
        # Create metrics filename with timestamp
        timestamp = int(time.time())
        metrics_filename = f"tofukingdom_metrics_{timestamp}.json"
        
        # Get results directory from environment or use default
        results_dir = os.environ.get("BENCHMARK_RESULTS_DIR", "benchmark_results")
        metrics_path = os.path.join(results_dir, metrics_filename)
        
        # Compute and save metrics
        self.metrics.compute_all()
        self.metrics.save(metrics_path)
        
        return metrics_filename

    def game_loop(self, log_file) -> Dict[str, Any]:
        """
        Main game loop where the Prince questions other players.
        
        Args:
            log_file: File to log game events
            
        Returns:
            Dictionary with game results
        """
        try:
            # Log initial game setup
            if self.debug:
                self.logger.debug(self.get_game_settings())
                log_file.write(self.get_game_settings() + "\n")
            
            # Set up monkey patch to intercept questions and answers for metrics
            original_handle_question_round = self.controller.handle_question_round
            original_handle_extra_question = self.controller.handle_extra_question
            
            # Monkey patch the handle_question_round method to capture metrics
            def patched_handle_question_round(log_file=None):
                # Track question index for answer association
                current_question_idx = len(self.metrics.questions)
                
                # Start round in metrics
                self.metrics.record_event(
                    self.metrics.EVENT_ROUND_START,
                    round_number=1
                )
                
                for player_name, agent in self.controller.role_agents.items():
                    # Get Prince's question
                    question = self.controller.prince.ask_question(player_name)
                    if question is None:
                        return False
                    
                    # Record question in metrics with Prince's thinking (not available here)
                    self.metrics.record_question(
                        question=question,
                        prince_player="Prince",
                        target_player=player_name,
                        round_num=1
                    )
                    
                    # Process as usual
                    question_message = f"Prince asks {player_name}: {question}"
                    if log_file:
                        log_file.write(question_message + "\n")
                    
                    # Get agent's answer
                    answer, thought = agent.answer_question(question, self.controller.identities)
                    if answer is None:
                        return False
                    
                    # Record answer in metrics
                    self.metrics.record_answer(
                        answer=answer,
                        player=player_name,
                        question_idx=current_question_idx,
                        thinking=thought
                    )
                    
                    # Increment question index for next answer
                    current_question_idx += 1
                    
                    # Process answer as usual
                    answer_message = f"{player_name}: {answer}"
                    if log_file:
                        log_file.write(answer_message + "\n")
                        if thought and self.debug:
                            log_file.write(json.dumps(thought) + "\n")
                
                # End round in metrics
                self.metrics.record_event(
                    self.metrics.EVENT_ROUND_END,
                    round_number=1
                )
                
                return original_handle_question_round(log_file)
            
            # Monkey patch the handle_extra_question method
            def patched_handle_extra_question(log_file=None):
                # Track question index for answer association
                current_question_idx = len(self.metrics.questions)
                
                # Start round in metrics
                self.metrics.record_event(
                    self.metrics.EVENT_ROUND_START,
                    round_number=2
                )
                
                # Prince chooses a player and question
                target_player, question = self.controller.prince.choose_final_question()
                if target_player is None or question is None:
                    return False
                
                # Record question in metrics
                self.metrics.record_question(
                    question=question,
                    prince_player="Prince",
                    target_player=target_player,
                    round_num=2
                )
                
                # Process as usual
                question_message = f"Prince asks final question to {target_player}: {question}"
                if log_file:
                    log_file.write(question_message + "\n")
                
                # Get agent's answer
                agent = self.controller.role_agents[target_player]
                answer, thought = agent.answer_question(question, self.controller.identities)
                if answer is None:
                    return False
                
                # Record answer in metrics
                self.metrics.record_answer(
                    answer=answer,
                    player=target_player,
                    question_idx=current_question_idx,
                    thinking=thought
                )
                
                # Process answer as usual
                answer_message = f"{target_player}: {answer}"
                if log_file:
                    log_file.write(answer_message + "\n")
                    if thought and self.debug:
                        log_file.write(json.dumps(thought) + "\n")
                
                # End round in metrics
                self.metrics.record_event(
                    self.metrics.EVENT_ROUND_END,
                    round_number=2
                )
                
                return original_handle_extra_question(log_file)
            
            # Apply the patches
            self.controller.handle_question_round = patched_handle_question_round
            self.controller.handle_extra_question = patched_handle_extra_question
            
            # Run the game through the controller
            results = self.controller.run_game(log_file)
            
            # Record final guess and game end
            if "princess_guess" in results and "guessed_role" in results:
                is_correct = results["guessed_role"] == "Princess"
                self.metrics.record_final_guess(
                    prince_player="Prince",
                    guessed_player=results["princess_guess"],
                    actual_role=results["guessed_role"],
                    correct=is_correct
                )
                
                # Set winner team
                if "winner_team" in results:
                    self.metrics.set_winner_team(results["winner_team"])
            
            # Record game end event
            self.metrics.record_event(
                self.metrics.EVENT_GAME_END,
                success=True,
                result=results
            )
            
            # Run LLM evaluation if enabled
            if self.use_llm_evaluation:
                self.metrics.evaluate_game()
            
            # Add metrics to results
            results["metrics"] = self.metrics.compute_all()
            results["metrics_file"] = self._save_metrics()
            
            # Handle possible errors
            if "error" in results:
                self.logger.error(f"Game error: {results['error']}")
            
            return results
            
        except Exception as e:
            self.logger.exception("Error in game loop")
            
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
            error_result = {"error": str(e)}
            error_result["metrics"] = self.metrics.compute_all()
            error_result["metrics_file"] = self._save_metrics()
            
            return error_result 