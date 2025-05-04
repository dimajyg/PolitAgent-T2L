from environments.tofukingdom.agents.base_agent import TofuKingdomAgent
from environments.tofukingdom.utils.utils import create_message
from typing import Dict, List, Any, Optional
import random

class PrinceAgent(TofuKingdomAgent):
    """
    Prince agent for the TofuKingdom game.
    
    The Prince needs to guess which player is the Princess by asking questions
    to different characters.
    
    Args:
        llm: LangChain-compatible language model
        player_name: Name of the player
        all_players: List of all players in the game
    """
    def __init__(self, llm, player_name: str, all_players: List[str]) -> None:
        super().__init__(llm, player_name, all_players, "Prince")
        
        # Questions the Prince can ask
        self.questions = [
            "Which player is the Princess?",
            "What is your identity?",
            "What is the identity of {player_name}?"
        ]
        
        # Identity information - will be updated as the game progresses
        self.identity_information = {}
        
        # Questions already asked to each player
        self.asked_questions = {player: [] for player in all_players if player != player_name}
        
    def ask_question(self, target_player: str) -> str:
        """
        Determine which question to ask a target player.
        
        Args:
            target_player: The player to ask
            
        Returns:
            The question to ask
        """
        messages = self.private_history.copy()
        
        # Prepare context
        context = (
            f"You need to ask a question to {target_player}.\n"
            f"You can only choose from these three questions:\n"
            f"1. Which player is the Princess?\n"
            f"2. What is your identity?\n"
            f"3. What is the identity of [player_name]? (You should replace [player_name] with a specific player's name)\n\n"
            f"Current information: {self.identity_information}\n\n"
            f"Please choose which question to ask (1, 2, or 3). If you choose question 3, also specify the player name."
        )
        
        messages.append(create_message("user", context))
        response = super().chat(messages)
        
        # Parse response to determine which question to ask
        question_idx = 0
        try:
            # Check if response contains a choice
            if "1" in response or "first" in response.lower():
                question_idx = 0
            elif "2" in response or "second" in response.lower():
                question_idx = 1
            elif "3" in response or "third" in response.lower():
                question_idx = 2
                # For question 3, extract the player name
                for player in self.all_players:
                    if player in response and player != target_player:
                        target_question = self.questions[question_idx].format(player_name=player)
                        self.asked_questions[target_player].append(target_question)
                        return target_question
                # If no valid player name is found, use a random player
                other_players = [p for p in self.all_players if p != self.player_name and p != target_player]
                if other_players:
                    random_player = random.choice(other_players)
                    target_question = self.questions[question_idx].format(player_name=random_player)
                    self.asked_questions[target_player].append(target_question)
                    return target_question
            
            # For questions 1 and 2
            target_question = self.questions[question_idx]
            self.asked_questions[target_player].append(target_question)
            return target_question
            
        except Exception as e:
            # If parsing fails, use a default question
            question_idx = random.randint(0, 1)  # Avoid question 3 as default
            target_question = self.questions[question_idx]
            self.asked_questions[target_player].append(target_question)
            return target_question
    
    def update_information(self, player: str, question: str, answer: str) -> None:
        """
        Update the Prince's knowledge with new information from a question response.
        
        Args:
            player: The player who answered
            question: The question that was asked
            answer: The answer received
        """
        # Record the Q&A in history
        qa_message = f"{player} was asked: {question}\n{player} answered: {answer}"
        self.private_history.append(create_message("user", qa_message))
        
        # Update identity information (simple format for now)
        if player not in self.identity_information:
            self.identity_information[player] = []
        
        self.identity_information[player].append({"question": question, "answer": answer})
    
    def choose_final_question(self) -> tuple[str, str]:
        """
        Choose a player to ask one final question before making the final guess.
        
        Returns:
            Tuple of (player_name, question)
        """
        messages = self.private_history.copy()
        
        context = (
            f"You're allowed to ask one final question before making your guess about who is the Princess.\n"
            f"Current information: {self.identity_information}\n\n"
            f"Which player would you like to ask a question to, and which question (1-3)?\n"
            f"1. Which player is the Princess?\n"
            f"2. What is your identity?\n"
            f"3. What is the identity of [player_name]? (You should replace [player_name] with a specific player's name)\n\n"
            f"Format your answer as: Player: [player_name], Question: [question_number] [optional player name for Q3]"
        )
        
        messages.append(create_message("user", context))
        response = super().chat(messages)
        
        # Parse response to find player and question
        try:
            target_player = None
            for player in self.all_players:
                if player in response and player != self.player_name:
                    target_player = player
                    break
            
            if not target_player:
                # Choose a random player if parsing fails
                other_players = [p for p in self.all_players if p != self.player_name]
                target_player = random.choice(other_players)
            
            # Determine which question to ask
            question_idx = 0
            if "1" in response or "first" in response.lower():
                question_idx = 0
                target_question = self.questions[question_idx]
            elif "2" in response or "second" in response.lower():
                question_idx = 1
                target_question = self.questions[question_idx]
            elif "3" in response or "third" in response.lower():
                question_idx = 2
                # For question 3, extract the player name
                for player in self.all_players:
                    if player in response and player != target_player and player != self.player_name:
                        target_question = self.questions[question_idx].format(player_name=player)
                        return target_player, target_question
                # If no valid player name is found, use a random player
                other_players = [p for p in self.all_players if p != self.player_name and p != target_player]
                if other_players:
                    random_player = random.choice(other_players)
                    target_question = self.questions[question_idx].format(player_name=random_player)
                    return target_player, target_question
            else:
                # Default to question 1 if parsing fails
                target_question = self.questions[0]
                
            return target_player, target_question
            
        except Exception as e:
            # If parsing completely fails, return defaults
            other_players = [p for p in self.all_players if p != self.player_name]
            target_player = random.choice(other_players)
            return target_player, self.questions[0]
    
    def make_guess(self) -> str:
        """
        Make the final guess about who is the Princess.
        
        Returns:
            The name of the player guessed to be the Princess
        """
        messages = self.private_history.copy()
        
        context = (
            f"Based on all the information you've gathered, it's time to make your final guess.\n"
            f"Current information: {self.identity_information}\n\n"
            f"Who do you think is the Princess? Please answer with just the player's name."
        )
        
        messages.append(create_message("user", context))
        response = super().chat(messages)
        
        # Parse response to find the guessed Princess
        try:
            for player in self.all_players:
                if player in response and player != self.player_name:
                    return player
                    
            # If no valid player is found, return a random player
            other_players = [p for p in self.all_players if p != self.player_name]
            return random.choice(other_players)
            
        except Exception as e:
            # If parsing fails, return a random player
            other_players = [p for p in self.all_players if p != self.player_name]
            return random.choice(other_players)
