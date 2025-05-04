from environments.tofukingdom.agents.role_agent import RoleAgent
from environments.tofukingdom.agents.prince_agent import PrinceAgent
from environments.tofukingdom.utils.utils import create_message
from typing import Dict, List, Any, Optional, Tuple, Set
import random
import logging
import json

class GameController:
    """
    Controller for the TofuKingdom game that manages game state and agent interactions.
    
    This controller:
    1. Handles player role assignments
    2. Manages question rounds
    3. Processes game state and determines winners
    4. Provides logging and history tracking
    
    Args:
        prince_llm: LLM for the Prince agent
        team_princess_llm: LLM for the Princess team (Princess, Chef)
        team_queen_llm: LLM for the Queen team (Queen, Minister, Guard)
        team_neutral_llm: LLM for the Neutral team (Maid, Spy)
        player_names: List of player names (excluding Prince)
        debug: Whether to enable debug mode
    """
    def __init__(
        self, 
        prince_llm,
        team_princess_llm,
        team_queen_llm, 
        team_neutral_llm,
        player_names: List[str],
        debug: bool = False
    ):
        if len(player_names) != 7:
            raise ValueError("TofuKingdom requires exactly 7 player names (excluding Prince)")
            
        self.player_names = player_names.copy()
        self.prince_llm = prince_llm
        self.team_llms = {
            "Princess": team_princess_llm,
            "Queen": team_queen_llm,
            "Neutral": team_neutral_llm
        }
        self.debug = debug
        self.prince = None
        self.role_agents = {}
        self.public_messages = []
        self.game_initialized = False
        
        # Game state
        self.eliminated_players = set()
        self.identities = {}
        self.winner_team = None
        
        # Available roles
        self.available_roles = [
            "Princess", "Queen", "Minister", 
            "Chef", "Guard", "Maid", "Spy"
        ]
        
        # Logging
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("TofuKingdom")
    
    def initialize_game(self) -> Dict[str, Any]:
        """
        Initialize the game by assigning roles and creating agents.
        
        Returns:
            Dictionary with game configuration details
        """
        # Shuffle player names
        random.shuffle(self.player_names)
        
        # Shuffle available roles
        roles = self.available_roles.copy()
        random.shuffle(roles)
        
        # Assign roles to players
        role_assignments = dict(zip(self.player_names, roles))
        self.identities = role_assignments
        
        # Create the Prince agent
        self.prince = PrinceAgent(self.prince_llm, "Prince", self.player_names)
        
        # Create role agents
        for player_name, role in role_assignments.items():
            # Determine which LLM to use based on team
            team = RoleAgent.ROLE_TEAMS[role]
            llm = self.team_llms[team]
            
            # Create agent
            self.role_agents[player_name] = RoleAgent.create(
                llm, player_name, self.player_names, role
            )
        
        self.game_initialized = True
        
        # Log game setup
        if self.debug:
            self.logger.debug("Game initialized with roles:")
            for player, role in role_assignments.items():
                self.logger.debug(f"Player: {player}, Role: {role}")
                
        return {
            "identities": self.identities,
            "prince": "Prince"
        }
    
    def broadcast_message(self, message: Dict[str, str], sender: str) -> None:
        """
        Broadcast a message to all agents except the sender.
        
        Args:
            message: Message content
            sender: Name of the sender to exclude
        """
        # Add to public message history
        self.public_messages.append(message)
        
        # Send to Prince if sender is not Prince
        if sender != "Prince":
            self.prince.private_history.append(message)
            
        # Send to other agents
        for player, agent in self.role_agents.items():
            if player != sender:
                agent.private_history.append(message)
    
    def handle_question_round(self, log_file=None) -> bool:
        """
        Handle a round of questions from the Prince to all other players.
        
        Args:
            log_file: Optional file object for logging
            
        Returns:
            True if successful, False if there was an error
        """
        if not self.game_initialized:
            raise RuntimeError("Game not initialized. Call initialize_game() first.")
        
        # Prince questions each player
        for player_name, agent in self.role_agents.items():
            # Prince asks a question to this player
            question = self.prince.ask_question(player_name)
            if question is None:
                self.logger.error(f"Prince failed to generate a question for {player_name}")
                return False
                
            # Log the question
            question_message = f"Prince asks {player_name}: {question}"
            if log_file:
                log_file.write(question_message + "\n")
            self.logger.info(question_message)
            
            # Create a message for the question
            question_msg = create_message("user", question_message)
            self.broadcast_message(question_msg, "Prince")
            
            # Get agent's answer
            answer, thought = agent.answer_question(question, self.identities)
            if answer is None:
                self.logger.error(f"{player_name} failed to generate an answer")
                return False
                
            # Log the answer
            answer_message = f"{player_name}: {answer}"
            if log_file:
                log_file.write(answer_message + "\n")
                if thought and self.debug:
                    log_file.write(json.dumps(thought) + "\n")
            self.logger.info(answer_message)
            
            # Create a message for the answer
            answer_msg = create_message("assistant", answer_message)
            self.broadcast_message(answer_msg, player_name)
            
            # Update Prince's information
            self.prince.update_information(player_name, question, answer)
        
        return True
    
    def handle_extra_question(self, log_file=None) -> bool:
        """
        Handle the extra question round where Prince asks one additional question.
        
        Args:
            log_file: Optional file object for logging
            
        Returns:
            True if successful, False if there was an error
        """
        if not self.game_initialized:
            raise RuntimeError("Game not initialized. Call initialize_game() first.")
        
        # Prince chooses a player and question
        target_player, question = self.prince.choose_final_question()
        if target_player is None or question is None:
            self.logger.error("Prince failed to choose a final question")
            return False
            
        # Log the question
        question_message = f"Prince asks final question to {target_player}: {question}"
        if log_file:
            log_file.write(question_message + "\n")
        self.logger.info(question_message)
        
        # Create a message for the question
        question_msg = create_message("user", question_message)
        self.broadcast_message(question_msg, "Prince")
        
        # Get agent's answer
        agent = self.role_agents[target_player]
        answer, thought = agent.answer_question(question, self.identities)
        if answer is None:
            self.logger.error(f"{target_player} failed to generate an answer")
            return False
            
        # Log the answer
        answer_message = f"{target_player}: {answer}"
        if log_file:
            log_file.write(answer_message + "\n")
            if thought and self.debug:
                log_file.write(json.dumps(thought) + "\n")
        self.logger.info(answer_message)
        
        # Create a message for the answer
        answer_msg = create_message("assistant", answer_message)
        self.broadcast_message(answer_msg, target_player)
        
        # Update Prince's information
        self.prince.update_information(target_player, question, answer)
        
        return True
    
    def finalize_game(self, log_file=None) -> Dict[str, Any]:
        """
        Finalize the game by having the Prince make a guess and determining winners.
        
        Args:
            log_file: Optional file object for logging
            
        Returns:
            Dictionary with game results
        """
        if not self.game_initialized:
            raise RuntimeError("Game not initialized. Call initialize_game() first.")
        
        # Prince makes a final guess
        host_message = "Host: Who do you think is the true Princess?"
        host_msg = create_message("user", host_message)
        self.broadcast_message(host_msg, "Host")
        
        if log_file:
            log_file.write(host_message + "\n")
        self.logger.info(host_message)
        
        # Get Prince's guess
        princess_guess = self.prince.make_guess()
        if princess_guess is None:
            self.logger.error("Prince failed to make a guess")
            return {"error": "Final answer is None"}
            
        # Get the actual role of the guessed player
        guessed_role = self.identities.get(princess_guess)
        if guessed_role is None:
            self.logger.error(f"Invalid guess: {princess_guess} is not a player")
            return {"error": f"Invalid guess: {princess_guess} is not a player"}
            
        # Log the guess
        guess_message = f"Prince: I think {princess_guess} is the Princess."
        if log_file:
            log_file.write(guess_message + "\n")
        self.logger.info(guess_message)
        
        # Determine the winning team
        if guessed_role == "Princess":
            winner_team = "Princess"
            winners = [p for p, r in self.identities.items() 
                     if RoleAgent.ROLE_TEAMS[r] == "Princess"]
        elif guessed_role == "Queen":
            winner_team = "Queen"
            winners = [p for p, r in self.identities.items() 
                     if RoleAgent.ROLE_TEAMS[r] == "Queen"]
        else:
            winner_team = "Neutral"
            winners = [p for p, r in self.identities.items() 
                     if RoleAgent.ROLE_TEAMS[r] == "Neutral"]
        
        # Log results
        result_message = (
            f"Game result: {princess_guess} is the {guessed_role}. "
            f"Team {winner_team} wins! Winners: {', '.join(winners)}"
        )
        if log_file:
            log_file.write(result_message + "\n")
        self.logger.info(result_message)
        
        # Store winner team
        self.winner_team = winner_team
        
        # Return game results
        return {
            "winner_team": winner_team,
            "winners": winners,
            "princess_guess": princess_guess,
            "guessed_role": guessed_role,
            "true_princess": next(p for p, r in self.identities.items() if r == "Princess"),
            "identities": self.identities
        }
    
    def run_game(self, log_file=None) -> Dict[str, Any]:
        """
        Run the complete game from initialization to finale.
        
        Args:
            log_file: Optional file object for logging
            
        Returns:
            Dictionary with game results
        """
        try:
            # Initialize the game
            self.initialize_game()
            
            # Start the game
            host_speech = "Host: The game now starts."
            start_message = create_message("user", host_speech)
            self.broadcast_message(start_message, "Host")
            
            if log_file:
                log_file.write(host_speech + "\n")
            self.logger.info(host_speech)
            
            # Question round
            if not self.handle_question_round(log_file):
                return {"error": "Question round failed"}
                
            # Extra question
            if not self.handle_extra_question(log_file):
                return {"error": "Extra question round failed"}
                
            # Finalize the game
            return self.finalize_game(log_file)
            
        except Exception as e:
            self.logger.exception("Error running game")
            return {"error": str(e)} 