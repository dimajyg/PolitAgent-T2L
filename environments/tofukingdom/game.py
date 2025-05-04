from llm.game import BaseGame
from environments.tofukingdom.agents import GameController
from typing import Dict, List, Any, Optional
import logging

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
        
    def init_game(self) -> str:
        """
        Initialize the game with random player assignments.
        
        Returns:
            String describing the initial game setup
        """
        # Initialize game through controller
        game_setup = self.controller.initialize_game()
        
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
            
            # Run the game through the controller
            results = self.controller.run_game(log_file)
            
            # Handle possible errors
            if "error" in results:
                self.logger.error(f"Game error: {results['error']}")
            
            return results
            
        except Exception as e:
            self.logger.exception("Error in game loop")
            return {"error": str(e)} 