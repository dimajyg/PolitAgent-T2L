from llm.agent import BaseAgent
from environments.beast.utils.utils import create_message
from environments.beast.utils.prompt import (
    get_role_prompt_template, 
    get_choose_conv_prompt_template, 
    get_conv_prompt_template,
    format_prompt
)
import json
from typing import Dict, List, Any, Optional, Tuple, cast
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

class BargainResponse(BaseModel):
    """Response for bargaining with another player."""
    message: str = Field(description="The message to send to the other player")
    offer: int = Field(description="The amount of wealth to offer (0 if no offer)")

class VoteResponse(BaseModel):
    """Response for voting."""
    player: str = Field(description="The name of the player to vote for")

class BeastAgent(BaseAgent):
    """
    Beast game agent implementation using the unified BaseAgent architecture.
    
    Args:
        llm: LangChain-compatible language model
        player_name: Name of the player
        players: List of all players in the game
        wealth: Initial wealth of the player
    """
    def __init__(
        self, 
        llm: BaseLanguageModel, 
        player_name: str, 
        players: List[str], 
        wealth: int
    ) -> None:
        self.player_name = player_name
        self.players = players
        self.wealth = wealth
        
        # Get role prompt template and format it
        role_prompt = format_prompt(
            get_role_prompt_template(),
            player_name=self.player_name,
            wealth=self.wealth
        )
        
        # Initialize the BaseAgent with the formatted prompt
        super().__init__(player_name, llm, role_prompt)
        
        # Initialize the message history
        self.private_history: List[Dict[str, str]] = []
        
        # Add the system role prompt to the history
        self.private_history.append(create_message("system", role_prompt))
        
    def get_role_description(self) -> str:
        """Returns a description of the agent's role.
        
        Returns:
            str: A human-readable description of the agent's role
        """
        return f"A player in the Beast game with {self.wealth} wealth"
    
    def choose_opponents(self, players_remaining: List[str]) -> List[str]:
        """
        Choose opponents to have conversations with.
        
        Args:
            players_remaining: List of available players
            
        Returns:
            List of chosen opponent names
        """
        # Format the prompt with the available players
        choose_prompt = format_prompt(
            get_choose_conv_prompt_template(),
            players_remaining=players_remaining
        )
        
        messages = self.private_history.copy()
        messages.append(create_message("user", choose_prompt))
        messages.append(create_message("user", "Remember, you must reply as list of string oponnent names as required. Also you couldn't choose your own name"))
        
        # Use the LLM directly
        response = self.chat(messages)
        
        try:
            # Extract valid player names from the response
            valid_players = [p for p in players_remaining if p != self.player_name and p in response]
            return valid_players[:3]  # Limit to at most 3 players
        except Exception as e:
            print(f"Error choosing opponents: {e}")

        return []
    
    def chat_with_context(self, context: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a chat message and return response with chain of thought.
        
        Args:
            context: The message context
            
        Returns:
            Tuple of response text and thought process
        """
        messages = self.private_history.copy()
        messages.append({"role": "user", "content": context})
        response = self.chat(messages)
        return response, {"thought": "Processed chat message and generated response"}
    
    def bargain(self, opponent_name: str) -> Tuple[Optional[str], int]:
        """
        Negotiate with an opponent.
        
        Args:
            opponent_name: Name of the opponent
            
        Returns:
            Tuple of message text and offer amount
        """
        # Format the prompt with the opponent name
        conv_prompt = format_prompt(
            get_conv_prompt_template(),
            opponent_name=opponent_name
        )

        messages = self.private_history.copy()
        messages.append(create_message("user", conv_prompt))
        
        # Use structured output for bargaining
        structured_llm = self.llm.with_structured_output(BargainResponse)
        
        try:
            # Call the model with structured output
            bargain_response = structured_llm.invoke(messages)
            
            # Get the message and offer from the structured response
            message = bargain_response.message
            offer = bargain_response.offer if bargain_response.offer is not None else 0
            
            # Record the conversation
            self.private_history.append(
                create_message(
                    "assistant", 
                    f"{message} with offer: {offer}"
                )
            )
            
            return message, offer
        except Exception as e:
            print(f"Error in bargain with structured output: {e}")
            # Fallback to regular chat
            try:
                messages.append(create_message("user", 'Remember, you must reply a json string in form of {"message":{your reply},"offer":{your offer(if you want to make one, should always be integer)}}, FOLLOW THIS RULE STRICTLY!!! ONLY JSON!!!'))
                res = self.chat(messages)
                
                # Clean and parse response for message and offer
                res = self._extract_json(res)
                
                # Parse the JSON response
                response_data = json.loads(res)
                message = response_data.get("message", "")
                offer = int(response_data.get("offer", 0) or 0)
                
                # Record the conversation
                self.private_history.append(
                    create_message(
                        "assistant", 
                        f"{message} with offer: {offer}"
                    )
                )
                
                return message, offer
            except Exception as e:
                print(f"Error in bargain fallback: {e}")
                return None, 0
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from a possibly markdown-formatted text.
        
        Args:
            text: Text potentially containing JSON, possibly in markdown code blocks
            
        Returns:
            str: Cleaned JSON string
        """
        # Remove markdown code blocks if present
        if "```json" in text:
            parts = text.split("```json")
            if len(parts) > 1:
                text = parts[1]
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                text = parts[0]
        
        # Find JSON-like content (between curly braces)
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return text.strip()
    
    def handle_offer(self, opponent_name: str, amount: int) -> bool:
        """
        Handle an offer from an opponent.
        
        Args:
            opponent_name: Name of the opponent
            amount: Amount offered
            
        Returns:
            True if the offer is accepted, False otherwise
        """
        messages = self.private_history.copy()

        context = (
            f"{opponent_name} has offered you {amount} wealth.\n"
            f"Your current wealth is {self.wealth}.\n"
            f"Decide whether to accept the offer (true/false).\n"
        )
        
        messages.append(create_message("user", context))
        
        response = self.chat(messages)
        try:
            # Record the decision in history
            accept = 'true' in response.lower()
            self.private_history.append(
                create_message(
                    "assistant", 
                    f"{'Accepted' if accept else 'Rejected'} offer of {amount} from {opponent_name}"
                )
            )
            return accept
        except Exception as e:
            print(f"Error handling offer: {e}")
            return False
    
    def vote(self) -> Optional[str]:
        """
        Vote for a player to receive bonus wealth.
        
        Returns:
            Name of the player to vote for, or None
        """
        messages = self.private_history.copy()

        context = "It's time to vote. Choose an opponent who you want to receive 250,000 wealth this round. Choose only one player and ensure they are not yourself."

        messages.append(create_message("user", context))
        
        # Use structured output for voting
        structured_llm = self.llm.with_structured_output(VoteResponse)
        
        try:
            # Call the model with structured output
            vote_response = structured_llm.invoke(messages)
            
            # Get the player to vote for from the structured response
            player = vote_response.player
            
            # Validate the vote
            if player in self.players and player != self.player_name:
                # Record the vote in history
                self.private_history.append(
                    create_message(
                        "assistant", 
                        f"Voted for {player}"
                    )
                )
                return player
            else:
                print(f"Invalid vote: {player} - falling back to regular chat")
                raise ValueError("Invalid player name")
                
        except Exception as e:
            print(f"Error in vote with structured output: {e}")
            # Fallback to regular chat
            try:
                messages = self.private_history.copy()
                context = "It's time to vote. Choose an opponent who you want to receive 250,000 wealth this round. As answer just write the name of your chosen opponent and nothing else.\n\n"
                messages.append(create_message("user", context))
                
                response = self.chat(messages)
                # Find a valid player name in the response
                for player in self.players:
                    if player in response and player != self.player_name:
                        # Record the vote in history
                        self.private_history.append(
                            create_message(
                                "assistant", 
                                f"Voted for {player}"
                            )
                        )
                        return player
                
                return None
            except Exception as e:
                print(f"Error in vote fallback: {e}")
                return None