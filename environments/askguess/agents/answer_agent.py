from llm.agent import BaseAgent
from environments.askguess.utils.prompt import get_answerer_prompt_template
from environments.askguess.utils.utils import create_message
from typing import Dict, List, Any, Optional

class AnswerAgent(BaseAgent):
    """
    Answer agent for the AskGuess game.
    
    Args:
        llm: LangChain-compatible model
        object_name: Word to be guessed
        args: Game arguments
    """
    def __init__(self, llm, object_name, args) -> None:
        self.object_name = object_name
        self.mode = args.mode
        self.prompt_template = get_answerer_prompt_template(self.mode)
        
        template_str = self.prompt_template.template
        super().__init__("Answerer", llm, template_str)
        
        self.private_history = []
        
        system_prompt = self.prompt_template.format(word=self.object_name)
        role_message = create_message("system", system_prompt)
        self.private_history.append(role_message)
 
    def get_role_description(self) -> str:
        """Returns the agent's role description."""
        return "The answerer in the Ask & Guess game"

    def chat(self, context: str) -> tuple:
        """Sends user message and gets response."""
        messages = self.private_history + [create_message("user", context)]
        response = super().chat(messages)
        return response, {"answer": response}

    def play(self) -> str:
        """Generates agent's move based on current history."""
        messages = self.private_history.copy()
        response = super().chat(messages)
        return response
    
    def answer(self) -> str:
        """Generates agent's answer based on current history."""
        messages = self.private_history.copy()
        response = super().chat(messages)
        return response

    def update_history(self, message: Dict[str, str]) -> None:
        """Adds message to dialogue history."""
        self.private_history.append(message)

    def get_answer_prompt(self) -> str:
        """Forms prompt for answer."""
        prompt = "##system##"
        if hasattr(self, 'answer_role'):
            prompt += self.answer_role + "\n"
        else:
            prompt += "You are the answerer in the Ask & Guess game.\n"
            
        flag = (len(self.private_history) + 1) % 2
        for i in range(len(self.private_history)):
            if i % 2 == flag:
                prompt += "##questioner##: "
            else: 
                prompt += "##answerer##:"
            prompt += self.private_history[i]["content"] + "\n\n"
        prompt +=  "##answerer##:"

        return prompt
    
    def get_describe_prompt(self) -> str:
        """Forms prompt for description."""
        prompt = "##system##"
        if hasattr(self, 'describe_role'):
            prompt += self.describe_role + "\n"
        else:
            prompt += "You are the answerer in the Ask & Guess game.\n"
            
        flag = (len(self.private_history) + 1) % 2
        for i in range(len(self.private_history)):
            if i % 2 == flag:
                prompt += "##questioner##: "
            else: 
                prompt += "##answerer##:"
            prompt += self.private_history[i]["content"] + "\n\n"
        prompt += "##answerer##:"

        return prompt


