from llm.agent import BaseAgent
from environments.askguess.utils.prompt import get_questioner_prompt_template
from environments.askguess.utils.utils import create_message
from typing import Dict, List, Any, Optional, Tuple

class QuestionAgent(BaseAgent):
    """
    Question agent for the AskGuess game.
    
    Args:
        llm: LangChain-compatible model
        object_name: Word to be guessed
        args: Game arguments
    """
    def __init__(self, llm, object_name, args) -> None:
        self.object_name = object_name
        self.mode = args.mode
        self.prompt_template = get_questioner_prompt_template(self.mode)
        
        template_str = self.prompt_template.template
        super().__init__("Questioner", llm, template_str)
        
        self.private_history = []
        
        system_prompt = self.prompt_template.format(word=self.object_name)
        role_message = create_message("system", system_prompt)
        self.private_history.append(role_message)
    
    def get_role_description(self) -> str:
        """Returns the agent's role description."""
        return "The questioner in the Ask & Guess game"

    def chat(self, context: str) -> Tuple[str, Dict[str, Any]]:
        """Sends user message and gets response."""
        messages = self.private_history + [create_message("user", context)]
        response = super().chat(messages)
        return response, {"answer": response}

    def play(self) -> str:
        """Generates agent's move based on current history."""
        messages = self.private_history.copy()
        response = super().chat(messages)
        return response
    
    def play_with_thinking(self) -> Tuple[str, Optional[str]]:
        """
        Generates agent's move based on current history and returns
        both the question itself and the agent's reasoning.
        
        Returns:
            Tuple[str, Optional[str]]: Question and reasoning (if available)
        """
        messages = self.private_history.copy()
        thinking_message = create_message("user", "Before you ask your next question, please think through your strategy. What information do you already have? What would be most useful to know next? (This thinking won't be shared with the answerer)")
        thinking_response = super().chat(messages + [thinking_message])
        
        question_message = create_message("user", "Now, ask your next question based on your reasoning.")
        question_response = super().chat(messages + [thinking_message, create_message("assistant", thinking_response), question_message])
        
        return question_response, thinking_response
        
    def update_history(self, message: Dict[str, str]) -> None:
        """Adds message to dialogue history."""
        self.private_history.append(message)



