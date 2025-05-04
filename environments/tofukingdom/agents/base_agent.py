from llm.agent import BaseAgent
from environments.tofukingdom.utils.prompt import get_game_prompt_template, get_role_prompt_template, format_prompt
from environments.tofukingdom.utils.utils import create_message, format_conversation_history
import json
from typing import Dict, List, Any, Optional, Tuple

class TofuKingdomAgent(BaseAgent):
    """
    Base agent for the TofuKingdom game using the unified BaseAgent architecture.
    
    Args:
        llm: LangChain-compatible language model
        player_name: Name of the player
        all_players: List of all players in the game
        role: Role of the agent (Prince, Princess, etc.)
    """
    def __init__(self, llm, player_name: str, all_players: List[str], role: str) -> None:
        self.player_name = player_name
        self.all_players = all_players
        self.role = role
        
        # Get game prompt template
        self.game_prompt_template = get_game_prompt_template()
        
        # Get role prompt template
        self.role_prompt_template = get_role_prompt_template(role)
        
        # Format the role prompt with player name
        role_prompt = format_prompt(self.role_prompt_template, player_name=player_name)
        
        # Format the game prompt
        game_prompt = format_prompt(self.game_prompt_template)
        
        # Combine the prompts
        prompt = f"{game_prompt}\n\nNow, you are player {player_name}\n{role_prompt}"
        
        # Initialize the BaseAgent with the formatted prompt
        super().__init__(player_name, llm, prompt)
        
        # Initialize the message history
        self.private_history = []

    def get_role_description(self) -> str:
        """Returns a description of the agent's role."""
        return f"A {self.role} in the Tofu Kingdom game"

    def process_question(self, identities: Dict[str, str]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Process a question and return an answer with thought process.
        
        Args:
            identities: Dictionary mapping player names to their roles
            
        Returns:
            Tuple of answer text and complete response with thought
        """
        # Prepare the prompt
        game_prompt = format_prompt(self.game_prompt_template)
        role_prompt = format_prompt(self.role_prompt_template, player_name=self.player_name)
        
        system_prompt = (
            f"{game_prompt}\n"
            f"Now, you are player {self.player_name} "
            f"{role_prompt}\n"
            f"This is the identity information of other players: {identities}\n"
        )
        
        last_prompt = (
            f"Your reply must be a JSON string in the following format:\n"
            f'{{"thought":{{"your thought"}},"answer":"your answer"}}\n'
            f"'thought' represent your thought of how to answer the question according to the rule and your goal. "
            f"'answer' represent your reply to the Prince."
        )
        
        messages = []
        first_message = create_message("system", system_prompt)
        messages.append(first_message)
        messages += self.private_history
        last_message = create_message("system", last_prompt)
        messages.append(last_message)

        cnt = 0
        while True:
            try:
                # Use the chat method from BaseAgent
                res = super().chat(messages)
                
                # Remove markdown code block markers if present
                res = res.strip()
                if res.startswith('```json') and res.endswith('```'):
                    res = res[7:-3].strip()
                if res.startswith('```') and res.endswith('```'):
                    res = res[3:-3].strip()
                
                res = json.loads(res)
                break
            except Exception as e:
                cnt += 1
            if cnt >= 3:
                return None, None
            
        answer = res["answer"]
        return answer, res
        
    def convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert a list of messages to a formatted prompt string.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += "system: "
                prompt += message["content"]
                prompt += "\n"
            elif message["role"] == "assistant":
                prompt += f"{self.player_name}: "
                prompt += message["content"]
                prompt += "\n"
            else:
                prompt += message["content"]
                prompt += "\n"
        prompt += f"{self.player_name}: "
        return prompt
